//! The T1-vs-FL teacher.
//!
//! ```text
//! value(t1) = E_t2deal [ pi_T2( . ) -> E_t3deal [ pi_T3( . ) ->
//!               E_t4deal [ max over T4 actions of  E_fl[ score ] ] ] ]
//! ```
//!
//! T2 and T3 are **continued, not searched**: `pi_T2` and `pi_T3` are the
//! distilled vs-Fantasyland rankers, which pick one action instead of the
//! teacher maximizing over all of them. T4 stays exhaustive and the terminal
//! stays the adaptive Fantasyland frontier.
//!
//! # Why this is a defensible teacher rather than a shortcut
//!
//! Searching T2 and T3 exhaustively from T1 multiplies the tree by roughly
//! `27 x 27` over the T2 teacher, which is not a scheduling problem. The
//! compromise is only worth making because the continuation quality is
//! *measured*: 0.155 and 0.096 mean regret against the exact teacher at their
//! own streets, against a learned chain that leaves 0.335 and 0.485. The
//! resulting label is the value of playing this T1 action and then playing on
//! the way the models actually play -- which is also the regime the model
//! trained on this label will be deployed in.
//!
//! Collapsing the two maxima is also what makes T1 cheap: it removes about a
//! factor of thirteen from the terminal count, so the shared frontier build
//! dominates and a T1 root costs only a little more than a T2 root did.
//!
//! # Common random numbers
//!
//! Every T1 candidate sees the same sampled T2/T3/T4 deals and the same
//! opponent samples. The deals are drawn from the ROOT's unseen set, which
//! already excludes the three dealt cards, so they are disjoint from every
//! candidate's discard and the draw schedule is candidate-independent -- the
//! same property the T2 teacher relies on.

use crate::behavior::{apply_turn, generate_turn_actions, PartialBoard, TurnAction};
use crate::distribution::sample_and_solve;
use crate::engine_features::NodeFeatures;
use crate::frontier::{best_response, build_frontier, FrontierEntry};
use crate::objective::ObjectiveConfig;
use crate::scoring::BoardScore;
use crate::solver::FlSolver;
use crate::teacher::{external_fingerprint, ExternalRoot, Provenance, TeacherConfig};
use crate::vfl_model::{Scratch, VflModel};
use ofc_hu_m3_engine::infoset::Street;
use serde::{Deserialize, Serialize};

pub const T1_LABEL_SCHEMA: &str = "regular_ofc_t1_vs_fl_label_v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct T1CandidateLabel {
    pub action: TurnAction,
    pub top: Vec<u8>,
    pub middle: Vec<u8>,
    pub bottom: Vec<u8>,
    pub discard: u8,
    pub expected_value: f64,
    pub standard_error: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelledT1Root {
    pub root_index: u64,
    pub schema: String,
    pub top: Vec<u8>,
    pub middle: Vec<u8>,
    pub bottom: Vec<u8>,
    pub dealt: [u8; 3],
    pub discards: Vec<u8>,
    pub seen_mask: u64,
    pub candidates: Vec<T1CandidateLabel>,
    pub best_candidate: usize,
    pub decision_margin: f64,
    pub opponent_samples: usize,
    pub t2_draws: usize,
    pub t3_draws: usize,
    pub t4_draws: usize,
    pub opponent_mode: String,
    pub continuation: ContinuationProvenance,
    pub mean_usable_opponents: f64,
    pub fingerprint: u64,
    pub provenance: Provenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_policy: Option<String>,
    /// How many of the root's legal openings were evaluated, and by whose
    /// ranking the rest were dropped. A label that saw ten of twenty-seven
    /// candidates is not the same label as one that saw all of them, so both
    /// facts travel with it. Absent means the whole fan was evaluated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub narrow_keep: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub narrow_model_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legal_actions: Option<usize>,
}

/// The continuation policies are part of the label's identity: relabelling with
/// a different pair of models produces a different quantity, exactly as a
/// different `opponent_mode` does.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContinuationProvenance {
    pub mode: String,
    pub t2_model_sha256: String,
    pub t3_model_sha256: String,
    pub engine_features_rev: String,
}

/// The continuation models, loaded once and shared by every root in a run.
pub struct Continuation {
    pub t2: VflModel,
    pub t3: VflModel,
    pub t2_sha256: String,
    pub t3_sha256: String,
}

impl Continuation {
    pub fn load_pinned(
        t2_bytes: &[u8],
        t2_sha256: &str,
        t3_bytes: &[u8],
        t3_sha256: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            t2: VflModel::load_pinned(t2_bytes, t2_sha256)?,
            t3: VflModel::load_pinned(t3_bytes, t3_sha256)?,
            t2_sha256: t2_sha256.to_owned(),
            t3_sha256: t3_sha256.to_owned(),
        })
    }

    /// Public for [`crate::t0_teacher`], which records the same continuation.
    pub fn provenance(&self) -> ContinuationProvenance {
        ContinuationProvenance {
            mode: "model_guided_v1".to_owned(),
            t2_model_sha256: self.t2_sha256.clone(),
            t3_model_sha256: self.t3_sha256.clone(),
            engine_features_rev: crate::engine_features::engine_features_rev().to_owned(),
        }
    }
}

/// One sampled T2 deal and the T3/T4 deals nested beneath it, shared across
/// every T1 candidate.
struct T2Draw {
    t2_deal: [u8; 3],
    t3: Vec<T3Draw>,
}

struct T3Draw {
    t3_deal: [u8; 3],
    t4_deals: Vec<[u8; 3]>,
    /// Opponent samples that collide with none of the three hero draws.
    valid: Vec<Vec<u32>>,
    /// Sum of the opponent's static best over the usable samples, which is what
    /// a fouled hero board concedes.
    constant: Vec<f64>,
}

/// Pick the continuation action at one node, or `None` when the fan is empty.
///
/// Public as `pick_continuation` for [`crate::t0_teacher`], which continues one
/// street deeper through the same rankers. Sharing it rather than copying it
/// keeps the tie rule -- strictly greater, so the lower index wins -- identical
/// at every street, which is what makes a T0 label and a T1 label describe the
/// same continuation.
pub fn pick_continuation(
    model: &VflModel,
    scratch: &mut Scratch,
    board: &PartialBoard,
    deal: &[u8; 3],
    thrown: &[u8],
    street: Street,
    actions: &[TurnAction],
) -> Result<usize, String> {
    pick(model, scratch, board, deal, thrown, street, actions)
}

/// Keep the `keep` highest-scoring actions, in the generator's own order.
///
/// T1 is the deepest street a teacher searches here: every one of its 27
/// openings pays for a T2 reply, a T3 reply, an exhaustive T4 and a Fantasyland
/// solve underneath, so the fan is the term that multiplies everything. Cutting
/// it with the street's own distilled ranker buys the draws that the axis
/// ladders say are what actually reduce a label's error.
///
/// The cut is by score, but the survivors are returned in the order the
/// generator emitted them, so a label does not depend on the ranker's ordering
/// among the actions it keeps -- only on which ones it keeps.
///
/// `keep` of zero, or a `keep` at least as large as the fan, is the identity.
pub fn narrow_by_model(
    model: &VflModel,
    scratch: &mut Scratch,
    board: &PartialBoard,
    deal: &[u8; 3],
    thrown: &[u8],
    street: Street,
    actions: &[TurnAction],
    keep: usize,
) -> Result<Vec<TurnAction>, String> {
    if keep == 0 || keep >= actions.len() {
        return Ok(actions.to_vec());
    }
    let mut node = NodeFeatures::new(board, deal, thrown, street)?;
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(actions.len());
    for (index, action) in actions.iter().enumerate() {
        let placed = apply_turn(board, deal, action);
        let row = node.encode(&placed)?;
        scored.push((index, model.predict(&row, scratch)?));
    }
    // Descending by score; ties keep the lower generator index, as `pick` does.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let mut survivors: Vec<usize> = scored[..keep].iter().map(|(i, _)| *i).collect();
    survivors.sort_unstable();
    Ok(survivors.into_iter().map(|i| actions[i]).collect())
}

fn pick(
    model: &VflModel,
    scratch: &mut Scratch,
    board: &PartialBoard,
    deal: &[u8; 3],
    thrown: &[u8],
    street: Street,
    actions: &[TurnAction],
) -> Result<usize, String> {
    if actions.is_empty() {
        return Err("continuation node has no legal action".to_owned());
    }
    let mut node = NodeFeatures::new(board, deal, thrown, street)?;
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (index, action) in actions.iter().enumerate() {
        let placed = apply_turn(board, deal, action);
        let row = node.encode(&placed)?;
        let score = model.predict(&row, scratch)?;
        // Strictly greater, so ties keep the lower index and the label does not
        // depend on the order the action generator happens to emit.
        if score > best_score {
            best_score = score;
            best = index;
        }
    }
    Ok(best)
}

#[allow(clippy::too_many_arguments)]
pub fn label_t1_root(
    root: &ExternalRoot,
    objective: &ObjectiveConfig,
    teacher: &TeacherConfig,
    continuation: &Continuation,
    t2_draws: usize,
    t3_draws: usize,
    t4_draws: usize,
    hero_deal_seed_base: u64,
    root_policy: &str,
    narrow_model: Option<&VflModel>,
    narrow_keep: usize,
    narrow_model_sha256: Option<&str>,
    solver: &mut FlSolver,
) -> Result<LabelledT1Root, String> {
    let board = root.board();
    if board.card_count() != 5 {
        return Err(format!(
            "T1 root {} has {} placed cards, expected 5",
            root.root_index,
            board.card_count()
        ));
    }
    let dealt = root.dealt_three()?;
    let mut actions = Vec::with_capacity(27);
    generate_turn_actions(&board, &mut actions);
    if actions.is_empty() {
        return Err(format!("T1 root {} has no legal action", root.root_index));
    }
    let legal_actions = actions.len();
    if let Some(model) = narrow_model {
        if narrow_keep > 0 && narrow_keep < actions.len() {
            let mut scratch = model.scratch();
            actions = narrow_by_model(
                model,
                &mut scratch,
                &board,
                &dealt,
                &root.discards,
                Street::T1,
                &actions,
                narrow_keep,
            )?;
        }
    }

    let opponents = sample_and_solve(
        root.seen_mask,
        teacher.opponent_samples,
        teacher.opponent_seed_base,
        root.root_index * teacher.opponent_samples as u64,
        solver,
    )?;
    let frontiers: Vec<Vec<FrontierEntry>> = opponents
        .iter()
        .map(|sampled| build_frontier(&sampled.deal, objective.fl_ev_stay))
        .collect();
    let static_max: Vec<f64> = frontiers
        .iter()
        .map(|frontier| {
            frontier
                .iter()
                .map(|entry| entry.static_value)
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    let opponent_masks: Vec<u64> = opponents
        .iter()
        .map(|sampled| crate::cards::mask_of(&sampled.deal))
        .collect();

    let unseen = crate::cards::unseen_from_mask(root.seen_mask);
    let mut usable_total = 0_usize;
    let mut usable_count = 0_usize;

    let mut draws: Vec<T2Draw> = Vec::with_capacity(t2_draws);
    for t2_index in 0..t2_draws as u64 {
        let mut rng = crate::rng::SplitMix64::for_stream(
            hero_deal_seed_base,
            root.root_index * 4096 + t2_index,
        );
        let mut pool = unseen.clone();
        rng.partial_shuffle(&mut pool, 3);
        let mut t2_deal: [u8; 3] = pool[..3].try_into().expect("three cards");
        t2_deal.sort_unstable();
        let t2_mask = crate::cards::mask_of(&t2_deal);
        let after_t2: Vec<u8> = unseen
            .iter()
            .copied()
            .filter(|card| t2_mask & (1_u64 << card) == 0)
            .collect();

        let mut t3_list = Vec::with_capacity(t3_draws);
        for t3_index in 0..t3_draws as u64 {
            let mut rng3 = crate::rng::SplitMix64::for_stream(
                hero_deal_seed_base + 1,
                root.root_index * 262_144 + t2_index * 64 + t3_index,
            );
            let mut pool3 = after_t2.clone();
            rng3.partial_shuffle(&mut pool3, 3);
            let mut t3_deal: [u8; 3] = pool3[..3].try_into().expect("three cards");
            t3_deal.sort_unstable();
            let t3_mask = crate::cards::mask_of(&t3_deal);
            let after_t3: Vec<u8> = after_t2
                .iter()
                .copied()
                .filter(|card| t3_mask & (1_u64 << card) == 0)
                .collect();

            let mut t4_deals = Vec::with_capacity(t4_draws);
            let mut valid = Vec::with_capacity(t4_draws);
            let mut constant = Vec::with_capacity(t4_draws);
            for t4_index in 0..t4_draws as u64 {
                let mut rng4 = crate::rng::SplitMix64::for_stream(
                    hero_deal_seed_base + 2,
                    root.root_index * 16_777_216 + t2_index * 4096 + t3_index * 64 + t4_index,
                );
                let mut pool4 = after_t3.clone();
                rng4.partial_shuffle(&mut pool4, 3);
                let mut t4_deal: [u8; 3] = pool4[..3].try_into().expect("three cards");
                t4_deal.sort_unstable();
                // The opponent's fourteen must miss ALL THREE hero draws.
                let all = t2_mask | t3_mask | crate::cards::mask_of(&t4_deal);
                let usable: Vec<u32> = opponent_masks
                    .iter()
                    .enumerate()
                    .filter(|(_, mask)| *mask & all == 0)
                    .map(|(index, _)| index as u32)
                    .collect();
                usable_total += usable.len();
                usable_count += 1;
                constant.push(usable.iter().map(|s| static_max[*s as usize]).sum());
                valid.push(usable);
                t4_deals.push(t4_deal);
            }
            t3_list.push(T3Draw {
                t3_deal,
                t4_deals,
                valid,
                constant,
            });
        }
        draws.push(T2Draw {
            t2_deal,
            t3: t3_list,
        });
    }

    let mut t2_actions = Vec::with_capacity(27);
    let mut t3_actions = Vec::with_capacity(27);
    let mut t4_actions = Vec::with_capacity(6);
    let mut scratch = continuation.t2.scratch();
    let mut candidates = Vec::with_capacity(actions.len());

    for action in actions.iter() {
        let seven = apply_turn(&board, &dealt, action);
        let t1_discard = dealt[action.discard as usize];
        let thrown_after_t1 = [t1_discard];

        let mut t2_values = Vec::with_capacity(t2_draws);
        for draw in draws.iter() {
            generate_turn_actions(&seven, &mut t2_actions);
            let chosen = pick(
                &continuation.t2,
                &mut scratch,
                &seven,
                &draw.t2_deal,
                &thrown_after_t1,
                Street::T2,
                &t2_actions,
            )?;
            let t2_action = t2_actions[chosen];
            let nine = apply_turn(&seven, &draw.t2_deal, &t2_action);
            let thrown_after_t2 = [t1_discard, draw.t2_deal[t2_action.discard as usize]];

            let mut t3_values = Vec::with_capacity(draw.t3.len());
            for t3 in draw.t3.iter() {
                generate_turn_actions(&nine, &mut t3_actions);
                let chosen3 = pick(
                    &continuation.t3,
                    &mut scratch,
                    &nine,
                    &t3.t3_deal,
                    &thrown_after_t2,
                    Street::T3,
                    &t3_actions,
                )?;
                let t3_action = t3_actions[chosen3];
                let eleven = apply_turn(&nine, &t3.t3_deal, &t3_action);
                generate_turn_actions(&eleven, &mut t4_actions);

                let mut total = 0.0_f64;
                let mut used = 0_usize;
                for (slot, t4_deal) in t3.t4_deals.iter().enumerate() {
                    let usable = &t3.valid[slot];
                    if usable.is_empty() {
                        continue;
                    }
                    let count = usable.len() as f64;
                    let mut best_t4 = f64::NEG_INFINITY;
                    for t4_action in t4_actions.iter() {
                        let complete = apply_turn(&eleven, t4_deal, t4_action);
                        let hero = BoardScore::evaluate(
                            &complete.top(),
                            &complete.middle(),
                            &complete.bottom(),
                        );
                        let mean = if hero.fouled {
                            (-6.0 * count - t3.constant[slot]) / count
                        } else {
                            let hero_side = hero.total_royalty as f64
                                + if hero.enters_fantasyland() {
                                    objective.fl_ev_stay
                                } else {
                                    0.0
                                };
                            let mut sum = 0.0_f64;
                            for index in usable.iter() {
                                let (opponent_best, _) = best_response(
                                    &frontiers[*index as usize],
                                    hero.top_key,
                                    hero.middle_key,
                                    hero.bottom_key,
                                );
                                sum += hero_side - opponent_best;
                            }
                            sum / count
                        };
                        if mean > best_t4 {
                            best_t4 = mean;
                        }
                    }
                    total += best_t4;
                    used += 1;
                }
                if used > 0 {
                    t3_values.push(total / used as f64);
                }
            }
            if !t3_values.is_empty() {
                t2_values.push(t3_values.iter().sum::<f64>() / t3_values.len() as f64);
            }
        }
        if t2_values.is_empty() {
            return Err(format!("T1 root {}: no usable draw", root.root_index));
        }
        let count = t2_values.len() as f64;
        let mean = t2_values.iter().sum::<f64>() / count;
        let variance = t2_values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / count;
        candidates.push(T1CandidateLabel {
            action: *action,
            top: seven.rows[0][..seven.lengths[0] as usize].to_vec(),
            middle: seven.rows[1][..seven.lengths[1] as usize].to_vec(),
            bottom: seven.rows[2][..seven.lengths[2] as usize].to_vec(),
            discard: t1_discard,
            expected_value: mean,
            standard_error: (variance / count).sqrt(),
        });
    }

    let best_candidate = candidates
        .iter()
        .enumerate()
        .max_by(|left, right| {
            left.1
                .expected_value
                .partial_cmp(&right.1.expected_value)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|(index, _)| index)
        .expect("at least one candidate");
    let mut sorted: Vec<f64> = candidates.iter().map(|c| c.expected_value).collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let decision_margin = if sorted.len() > 1 {
        sorted[0] - sorted[1]
    } else {
        0.0
    };

    Ok(LabelledT1Root {
        root_index: root.root_index,
        schema: T1_LABEL_SCHEMA.to_owned(),
        top: root.top.clone(),
        middle: root.middle.clone(),
        bottom: root.bottom.clone(),
        dealt,
        discards: root.discards.clone(),
        seen_mask: root.seen_mask,
        candidates,
        best_candidate,
        decision_margin,
        opponent_samples: opponents.len(),
        t2_draws,
        t3_draws,
        t4_draws,
        opponent_mode: "adaptive_v1".to_owned(),
        continuation: continuation.provenance(),
        mean_usable_opponents: usable_total as f64 / usable_count.max(1) as f64,
        fingerprint: external_fingerprint(root),
        provenance: Provenance {
            solver_version: crate::SOLVER_VERSION.to_owned(),
            objective: objective.identity(),
            fl_ev_config_path: objective.fl_ev_config_path.clone(),
            fl_ev_cards: objective.fl_ev_cards,
            fl_ev_value: objective.fl_ev_stay,
            behavior_policy: root_policy.to_owned(),
            root_seed_base: hero_deal_seed_base,
            opponent_seed_base: teacher.opponent_seed_base,
        },
        root_policy: root.root_policy.clone(),
        narrow_keep: if actions.len() < legal_actions {
            Some(narrow_keep)
        } else {
            None
        },
        narrow_model_sha256: if actions.len() < legal_actions {
            narrow_model_sha256.map(str::to_owned)
        } else {
            None
        },
        legal_actions: Some(legal_actions),
    })
}
