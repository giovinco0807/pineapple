//! The T4-vs-FL teacher label.
//!
//! At T4 the hero has eleven cards placed and three dealt: place two, discard
//! one. The opponent is in Fantasyland, so fourteen cards left the deck unseen
//! at T0 and the hero's information set is exactly the seventeen cards they
//! have touched.
//!
//! A candidate's label is
//!
//! ```text
//! E_opp [ hero_vs_fl_score(hero_final, solved_opponent_fl_board, fl_ev) ]
//! ```
//!
//! over opponent Fantasyland deals sampled from the hero's unseen remainder and
//! solved exactly. The hero's own Fantasyland entry value is inside that score
//! (ordinary QQ+ entry, priced at the same `fl_ev`), as is the opponent's
//! continuation value (the *stay* rule, not entry).
//!
//! Every candidate is scored against the *same* sampled opponent boards. Under
//! common random numbers the sampling error is shared, so candidate
//! differences -- which is all a ranking teacher needs -- converge far faster
//! than the absolute EVs do.

use crate::behavior::{apply_turn, generate_turn_actions, HeroT4Root, PartialBoard, TurnAction};
use crate::distribution::sample_and_solve;
use crate::objective::ObjectiveConfig;
use crate::scoring::{hero_vs_fl_score, BoardScore};
use crate::solver::FlSolver;
use serde::{Deserialize, Serialize};

/// One scored hero candidate at T4.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateLabel {
    pub action: TurnAction,
    /// Resulting hero board, as card indices.
    pub top: Vec<u8>,
    pub middle: Vec<u8>,
    pub bottom: Vec<u8>,
    pub discard: u8,
    pub fouled: bool,
    pub total_royalty: i32,
    pub enters_fantasyland: bool,
    /// Packed hand keys for the three rows. A consumer recovers the category
    /// with `key >> 20` and the leading tie-breaker with `(key >> 16) & 0xF`,
    /// which is all a feature encoder needs and avoids re-ranking the board.
    pub top_key: u32,
    pub middle_key: u32,
    pub bottom_key: u32,
    pub top_royalty: i32,
    pub middle_royalty: i32,
    pub bottom_royalty: i32,
    /// Mean score against the sampled opponent Fantasyland boards.
    pub expected_value: f64,
    /// Standard error of that mean over the same samples.
    pub standard_error: f64,
}

/// A fully labelled T4-vs-FL root.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelledRoot {
    pub root_index: u64,
    pub schema: String,
    pub top: Vec<u8>,
    pub middle: Vec<u8>,
    pub bottom: Vec<u8>,
    pub dealt: [u8; 3],
    pub discards: [u8; 3],
    /// Bitmask of the seventeen cards the hero has seen. Its complement is the
    /// deck the opponent's Fantasyland hand was drawn from, so a model that
    /// wants to reason about the opponent needs it.
    pub seen_mask: u64,
    pub candidates: Vec<CandidateLabel>,
    pub best_candidate: usize,
    /// Gap between the best and second-best candidate EV.
    pub decision_margin: f64,
    pub opponent_samples: usize,
    /// Opponent stay rate observed in this root's sample, a per-root echo of
    /// the distribution-wide statistic.
    pub opponent_stay_rate: f64,
    pub opponent_mean_royalty: f64,
    /// Fingerprint over the root's cards; distinct roots must differ.
    pub fingerprint: u64,
    pub provenance: Provenance,
}

/// What an artifact needs to carry to be reproducible and auditable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Provenance {
    pub solver_version: String,
    pub objective: String,
    pub fl_ev_config_path: String,
    pub fl_ev_cards: u8,
    pub fl_ev_value: f64,
    pub behavior_policy: String,
    pub root_seed_base: u64,
    pub opponent_seed_base: u64,
}

/// Configuration for one labelling run.
#[derive(Clone, Debug)]
pub struct TeacherConfig {
    pub opponent_samples: usize,
    pub root_seed_base: u64,
    pub opponent_seed_base: u64,
}

pub const LABEL_SCHEMA: &str = "regular_ofc_t4_vs_fl_label_v1";

/// Label every legal T4 candidate for one root.
pub fn label_root(
    root: &HeroT4Root,
    objective: &ObjectiveConfig,
    teacher: &TeacherConfig,
    behavior_identity: &str,
    solver: &mut FlSolver,
) -> Result<LabelledRoot, String> {
    let board = root.board();
    let mut actions = Vec::with_capacity(6);
    generate_turn_actions(&board, &mut actions);
    if actions.is_empty() {
        return Err(format!("T4 root {} has no legal action", root.root_index));
    }

    // One opponent sample set per root, shared by every candidate. This is the
    // common-random-numbers coupling.
    let opponents = sample_and_solve(
        root.seen_mask,
        teacher.opponent_samples,
        teacher.opponent_seed_base,
        root.root_index * teacher.opponent_samples as u64,
        solver,
    )?;
    let opponent_boards: Vec<BoardScore> = opponents
        .iter()
        .map(|sampled| {
            BoardScore::from_keys(
                sampled.solution.top_key,
                sampled.solution.middle_key,
                sampled.solution.bottom_key,
            )
        })
        .collect();
    let opponent_stays = opponents
        .iter()
        .filter(|sampled| sampled.solution.stays())
        .count();
    let opponent_royalty_total: i32 = opponents
        .iter()
        .map(|sampled| sampled.solution.total_royalty)
        .sum();

    let sample_count = opponents.len() as f64;
    let mut candidates = Vec::with_capacity(actions.len());
    for action in actions.iter() {
        let next = apply_turn(&board, &root.dealt, action);
        if !next.is_complete() {
            return Err(format!(
                "T4 candidate on root {} did not complete the board",
                root.root_index
            ));
        }
        let hero = BoardScore::evaluate(&next.top(), &next.middle(), &next.bottom());
        let mut total = 0.0_f64;
        let mut total_squares = 0.0_f64;
        for opponent in opponent_boards.iter() {
            let score = hero_vs_fl_score(&hero, opponent, objective.fl_ev_stay);
            total += score;
            total_squares += score * score;
        }
        let mean = total / sample_count;
        let variance = (total_squares / sample_count - mean * mean).max(0.0);
        candidates.push(CandidateLabel {
            action: *action,
            top: next.top().to_vec(),
            middle: next.middle().to_vec(),
            bottom: next.bottom().to_vec(),
            discard: root.dealt[action.discard as usize],
            fouled: hero.fouled,
            total_royalty: hero.total_royalty,
            enters_fantasyland: hero.enters_fantasyland(),
            top_key: hero.top_key,
            middle_key: hero.middle_key,
            bottom_key: hero.bottom_key,
            top_royalty: if hero.fouled {
                0
            } else {
                crate::eval::top_royalty(hero.top_key)
            },
            middle_royalty: if hero.fouled {
                0
            } else {
                crate::eval::middle_royalty(hero.middle_key)
            },
            bottom_royalty: if hero.fouled {
                0
            } else {
                crate::eval::bottom_royalty(hero.bottom_key)
            },
            expected_value: mean,
            standard_error: (variance / sample_count).sqrt(),
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
    let mut sorted: Vec<f64> = candidates
        .iter()
        .map(|candidate| candidate.expected_value)
        .collect();
    sorted.sort_by(|left, right| right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal));
    let decision_margin = if sorted.len() > 1 {
        sorted[0] - sorted[1]
    } else {
        0.0
    };

    Ok(LabelledRoot {
        root_index: root.root_index,
        schema: LABEL_SCHEMA.to_owned(),
        top: root.top.clone(),
        middle: root.middle.clone(),
        bottom: root.bottom.clone(),
        dealt: root.dealt,
        discards: root.discards,
        seen_mask: root.seen_mask,
        candidates,
        best_candidate,
        decision_margin,
        opponent_samples: opponents.len(),
        opponent_stay_rate: opponent_stays as f64 / sample_count,
        opponent_mean_royalty: opponent_royalty_total as f64 / sample_count,
        fingerprint: fingerprint(root),
        provenance: Provenance {
            solver_version: crate::SOLVER_VERSION.to_owned(),
            objective: objective.identity(),
            fl_ev_config_path: objective.fl_ev_config_path.clone(),
            fl_ev_cards: objective.fl_ev_cards,
            fl_ev_value: objective.fl_ev_stay,
            behavior_policy: behavior_identity.to_owned(),
            root_seed_base: teacher.root_seed_base,
            opponent_seed_base: teacher.opponent_seed_base,
        },
    })
}

// ---------------------------------------------------------------------------
// T3-vs-FL
// ---------------------------------------------------------------------------

/// A hero decision point loaded from an externally generated roots file, so
/// roots can come from the learned chain instead of this crate's stand-in
/// behavior policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExternalRoot {
    pub root_index: u64,
    pub top: Vec<u8>,
    pub middle: Vec<u8>,
    pub bottom: Vec<u8>,
    pub dealt: Vec<u8>,
    pub discards: Vec<u8>,
    pub seen_mask: u64,
    /// Free-form provenance carried through from the generator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_policy: Option<String>,
}

impl ExternalRoot {
    pub fn board(&self) -> PartialBoard {
        let mut board = PartialBoard::default();
        for card in &self.top {
            board.push(*card, crate::behavior::ROW_TOP);
        }
        for card in &self.middle {
            board.push(*card, crate::behavior::ROW_MIDDLE);
        }
        for card in &self.bottom {
            board.push(*card, crate::behavior::ROW_BOTTOM);
        }
        board
    }

    pub fn dealt_three(&self) -> Result<[u8; 3], String> {
        self.dealt
            .clone()
            .try_into()
            .map_err(|_| format!("root {} does not deal three cards", self.root_index))
    }
}

/// One scored hero candidate at T3.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct T3CandidateLabel {
    pub action: TurnAction,
    /// The eleven-card board this candidate reaches.
    pub top: Vec<u8>,
    pub middle: Vec<u8>,
    pub bottom: Vec<u8>,
    pub discard: u8,
    /// Mean over hero T4 deals of the best T4 continuation's expected value.
    pub expected_value: f64,
    /// Standard error of that mean across the sampled hero deals.
    pub standard_error: f64,
    /// Mean number of legal T4 actions this candidate leaves.
    pub mean_t4_candidates: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelledT3Root {
    pub root_index: u64,
    pub schema: String,
    pub top: Vec<u8>,
    pub middle: Vec<u8>,
    pub bottom: Vec<u8>,
    pub dealt: [u8; 3],
    pub discards: Vec<u8>,
    pub seen_mask: u64,
    pub candidates: Vec<T3CandidateLabel>,
    pub best_candidate: usize,
    pub decision_margin: f64,
    pub opponent_samples: usize,
    pub hero_deal_samples: usize,
    /// `exhaustive` when every dealable T4 hand was enumerated, `sampled`
    /// otherwise.
    pub deal_mode: String,
    /// `adaptive_v1` when the Fantasyland side best-responds to the hero's
    /// finished board, `static_v1` when it sets blind.
    pub opponent_mode: String,
    /// Mean count of opponent samples actually usable per hero deal, after
    /// dropping the ones whose Fantasyland hand collides with the deal.
    pub mean_usable_opponents: f64,
    pub opponent_stay_rate: f64,
    pub opponent_mean_royalty: f64,
    pub fingerprint: u64,
    pub provenance: Provenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_policy: Option<String>,
}

pub const T3_LABEL_SCHEMA: &str = "regular_ofc_t3_vs_fl_label_v1";

/// Label every legal T3 candidate for one root.
///
/// # The nested expectation
///
/// ```text
/// value(t3_candidate) = E_deal [ max over T4 actions of E_fl [ score ] ]
/// ```
///
/// The inner expectation is over the opponent's Fantasyland board and the
/// maximum sits *outside* it, because the hero picks its T4 action without
/// seeing the opponent's hand. Taking the maximum inside would let the hero
/// peek.
///
/// Both the hero's T4 deal and the opponent's fourteen cards are drawn from the
/// same unseen remainder and must not collide. The opponent hands are sampled
/// and solved once per root -- they are the expensive part -- and each hero
/// deal then uses the subset disjoint from it. That subset is exactly the
/// conditional distribution the hero faces, by rejection, and `mean_usable_opponents`
/// records how much of the pool each deal actually got.
#[allow(clippy::too_many_arguments)]
pub fn label_t3_root(
    root: &ExternalRoot,
    objective: &ObjectiveConfig,
    teacher: &TeacherConfig,
    hero_deal_samples: usize,
    hero_deal_seed_base: u64,
    root_policy: &str,
    adaptive: bool,
    solver: &mut FlSolver,
) -> Result<LabelledT3Root, String> {
    let board = root.board();
    if board.card_count() != 9 {
        return Err(format!(
            "T3 root {} has {} placed cards, expected 9",
            root.root_index,
            board.card_count()
        ));
    }
    let dealt = root.dealt_three()?;
    let mut actions = Vec::with_capacity(27);
    generate_turn_actions(&board, &mut actions);
    if actions.is_empty() {
        return Err(format!("T3 root {} has no legal action", root.root_index));
    }

    // Opponent Fantasyland hands: sampled and solved once, shared by every
    // candidate and every T4 continuation. This is the common-random-numbers
    // coupling and it is also where all the time goes.
    let opponents = sample_and_solve(
        root.seen_mask,
        teacher.opponent_samples,
        teacher.opponent_seed_base,
        root.root_index * teacher.opponent_samples as u64,
        solver,
    )?;
    let opponent_scores: Vec<BoardScore> = opponents
        .iter()
        .map(|sampled| {
            BoardScore::from_keys(
                sampled.solution.top_key,
                sampled.solution.middle_key,
                sampled.solution.bottom_key,
            )
        })
        .collect();
    let opponent_masks: Vec<u64> = opponents
        .iter()
        .map(|sampled| crate::cards::mask_of(&sampled.deal))
        .collect();
    let opponent_stays = opponents
        .iter()
        .filter(|sampled| sampled.solution.stays())
        .count();
    let opponent_royalty_total: i32 = opponents
        .iter()
        .map(|sampled| sampled.solution.total_royalty)
        .sum();

    // The adaptive opponent. Under the real rule the Fantasyland player sets
    // their board only after seeing the hero's completed one, so each sampled
    // hand contributes a best-response frontier rather than a single board.
    // `static_max` is what that same hand would have set blind, which is also
    // what it sets when the hero has fouled and every row is already won.
    let frontiers: Vec<Vec<crate::frontier::FrontierEntry>> = if adaptive {
        opponents
            .iter()
            .map(|sampled| {
                crate::frontier::build_frontier(&sampled.deal, objective.fl_ev_stay)
            })
            .collect()
    } else {
        Vec::new()
    };
    let frontier_static_max: Vec<f64> = frontiers
        .iter()
        .map(|frontier| {
            frontier
                .iter()
                .map(|entry| entry.static_value)
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();

    // Opponent side as parallel arrays, so the inner loop walks contiguous
    // memory instead of chasing a struct per sample.
    let opponent_royalty_plus_fl: Vec<f64> = opponent_scores
        .iter()
        .map(|score| {
            let royalty = if score.fouled { 0 } else { score.total_royalty };
            let fantasyland = if score.stays_in_fantasyland() {
                objective.fl_ev_stay
            } else {
                0.0
            };
            royalty as f64 + fantasyland
        })
        .collect();
    // A solved Fantasyland board never fouls -- the solver only ever returns
    // legal arrangements, and that is property-tested. The fast path below
    // relies on it, so check rather than assume.
    if opponent_scores.iter().any(|score| score.fouled) {
        return Err(format!(
            "T3 root {}: a solved Fantasyland board fouled, which must not happen",
            root.root_index
        ));
    }

    // Hero T4 deals, common across candidates.
    let unseen = crate::cards::unseen_from_mask(root.seen_mask);
    if unseen.len() < 3 {
        return Err(format!("T3 root {} has no cards left to deal", root.root_index));
    }
    // `hero_deal_samples == 0` means enumerate every deal the hero can be
    // dealt. There is no opponent branching between T3 and showdown, so the
    // outer expectation is a finite sum over `C(unseen, 3)` -- about 8.4k for a
    // T3 root -- and taking it exactly removes the sampling error in that
    // dimension entirely.
    let exhaustive = hero_deal_samples == 0;
    let mut deals: Vec<[u8; 3]> = Vec::new();
    if exhaustive {
        for first in 0..unseen.len() - 2 {
            for second in (first + 1)..unseen.len() - 1 {
                for third in (second + 1)..unseen.len() {
                    deals.push([unseen[first], unseen[second], unseen[third]]);
                }
            }
        }
    } else {
        for sample in 0..hero_deal_samples as u64 {
            let mut rng = crate::rng::SplitMix64::for_stream(
                hero_deal_seed_base,
                root.root_index * hero_deal_samples as u64 + sample,
            );
            let mut pool = unseen.clone();
            rng.partial_shuffle(&mut pool, 3);
            let mut deal: [u8; 3] = pool[..3].try_into().expect("three cards");
            deal.sort_unstable();
            deals.push(deal);
        }
    }
    let deal_count = deals.len();
    let mut usable: Vec<Vec<u32>> = Vec::with_capacity(deal_count);
    let mut usable_constant: Vec<f64> = Vec::with_capacity(deal_count);
    for deal in deals.iter() {
        let deal_mask = crate::cards::mask_of(deal);
        let valid: Vec<u32> = opponent_masks
            .iter()
            .enumerate()
            .filter(|(_, mask)| *mask & deal_mask == 0)
            .map(|(index, _)| index as u32)
            .collect();
        // The opponent's own royalty and Fantasyland value enter every hero
        // score with the same sign, so their sum over the usable set is a
        // per-deal constant rather than something to re-add 192 times.
        let constant: f64 = valid
            .iter()
            .map(|slot| opponent_royalty_plus_fl[*slot as usize])
            .sum();
        usable.push(valid);
        usable_constant.push(constant);
    }
    let usable_total: usize = usable.iter().map(Vec::len).sum();
    let mean_usable = usable_total as f64 / deal_count.max(1) as f64;
    if mean_usable < 1.0 {
        return Err(format!(
            "T3 root {}: no opponent sample survives the hero deal collision filter",
            root.root_index
        ));
    }

    let mut candidates = Vec::with_capacity(actions.len());
    let mut t4_actions = Vec::with_capacity(6);
    for action in actions.iter() {
        let eleven = apply_turn(&board, &dealt, action);
        // The legal T4 actions depend only on which rows still have room, not
        // on which cards arrive, so they are fixed for this candidate.
        generate_turn_actions(&eleven, &mut t4_actions);
        let t4_action_count = t4_actions.len();

        let mut deal_values = Vec::with_capacity(deal_count);
        for (index, deal) in deals.iter().enumerate() {
            let valid = &usable[index];
            if valid.is_empty() {
                continue;
            }
            let count = valid.len() as f64;
            let constant = usable_constant[index];
            let mut best = f64::NEG_INFINITY;
            for t4_action in t4_actions.iter() {
                let final_board = apply_turn(&eleven, deal, t4_action);
                debug_assert!(final_board.is_complete());
                let hero = BoardScore::evaluate(
                    &final_board.top(),
                    &final_board.middle(),
                    &final_board.bottom(),
                );
                // No sampled Fantasyland board fouls (checked above), so the
                // two branches collapse to closed forms.
                let mean = if adaptive {
                    // The opponent maximises its own score, which is
                    // `lines + scoop + static`; the hero gets the negative of
                    // that plus its own royalty and entry value. A fouled hero
                    // concedes -6 and the opponent simply takes its static max.
                    let hero_side = if hero.fouled {
                        0.0
                    } else {
                        hero.total_royalty as f64
                            + if hero.enters_fantasyland() {
                                objective.fl_ev_stay
                            } else {
                                0.0
                            }
                    };
                    let mut total = 0.0_f64;
                    for slot in valid.iter() {
                        let index = *slot as usize;
                        if hero.fouled {
                            total += -6.0 - frontier_static_max[index];
                        } else {
                            let (opponent_best, _) = crate::frontier::best_response(
                                &frontiers[index],
                                hero.top_key,
                                hero.middle_key,
                                hero.bottom_key,
                            );
                            total += hero_side - opponent_best;
                        }
                    }
                    total / count
                } else if hero.fouled {
                    (-6.0 * count - constant) / count
                } else {
                    let hero_side = hero.total_royalty as f64
                        + if hero.enters_fantasyland() {
                            objective.fl_ev_stay
                        } else {
                            0.0
                        };
                    let mut lines_total = 0_i32;
                    for slot in valid.iter() {
                        let opponent = &opponent_scores[*slot as usize];
                        let mut lines = 0_i32;
                        lines += match hero.top_key.cmp(&opponent.top_key) {
                            std::cmp::Ordering::Greater => 1,
                            std::cmp::Ordering::Less => -1,
                            std::cmp::Ordering::Equal => 0,
                        };
                        lines += match hero.middle_key.cmp(&opponent.middle_key) {
                            std::cmp::Ordering::Greater => 1,
                            std::cmp::Ordering::Less => -1,
                            std::cmp::Ordering::Equal => 0,
                        };
                        lines += match hero.bottom_key.cmp(&opponent.bottom_key) {
                            std::cmp::Ordering::Greater => 1,
                            std::cmp::Ordering::Less => -1,
                            std::cmp::Ordering::Equal => 0,
                        };
                        lines_total += if lines == 3 {
                            6
                        } else if lines == -3 {
                            -6
                        } else {
                            lines
                        };
                    }
                    (lines_total as f64 + count * hero_side - constant) / count
                };
                if mean > best {
                    best = mean;
                }
            }
            deal_values.push(best);
        }
        let t4_count_total = t4_action_count * deal_values.len();
        let counted_deals = deal_values.len();
        if deal_values.is_empty() {
            return Err(format!(
                "T3 root {}: candidate had no usable hero deal",
                root.root_index
            ));
        }
        let count = deal_values.len() as f64;
        let mean = deal_values.iter().sum::<f64>() / count;
        let variance = deal_values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / count;
        candidates.push(T3CandidateLabel {
            action: *action,
            top: eleven.rows[0][..eleven.lengths[0] as usize].to_vec(),
            middle: eleven.rows[1][..eleven.lengths[1] as usize].to_vec(),
            bottom: eleven.rows[2][..eleven.lengths[2] as usize].to_vec(),
            discard: dealt[action.discard as usize],
            expected_value: mean,
            // Under exhaustive enumeration the outer expectation is a complete
            // sum, so it carries no sampling error; the spread across deals is
            // a property of the game, not of the estimate. Sampling error from
            // the opponent pool is reported by `mean_usable_opponents`.
            standard_error: if exhaustive {
                0.0
            } else {
                (variance / count).sqrt()
            },
            mean_t4_candidates: t4_count_total as f64 / counted_deals.max(1) as f64,
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
    let mut sorted: Vec<f64> = candidates
        .iter()
        .map(|candidate| candidate.expected_value)
        .collect();
    sorted.sort_by(|left, right| right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal));
    let decision_margin = if sorted.len() > 1 {
        sorted[0] - sorted[1]
    } else {
        0.0
    };
    let sample_count = opponents.len() as f64;

    Ok(LabelledT3Root {
        root_index: root.root_index,
        schema: T3_LABEL_SCHEMA.to_owned(),
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
        hero_deal_samples: deal_count,
        deal_mode: if exhaustive {
            "exhaustive".to_owned()
        } else {
            "sampled".to_owned()
        },
        opponent_mode: if adaptive {
            "adaptive_v1".to_owned()
        } else {
            "static_v1".to_owned()
        },
        mean_usable_opponents: mean_usable,
        opponent_stay_rate: opponent_stays as f64 / sample_count,
        opponent_mean_royalty: opponent_royalty_total as f64 / sample_count,
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
    })
}

/// Fingerprint for an externally generated root.
pub fn external_fingerprint(root: &ExternalRoot) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut absorb = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    };
    for (row, cards) in [&root.top, &root.middle, &root.bottom].iter().enumerate() {
        absorb(0xF0 | row as u64);
        for card in cards.iter() {
            absorb(*card as u64 + 1);
        }
    }
    absorb(0xD0);
    for card in root.dealt.iter() {
        absorb(*card as u64 + 1);
    }
    absorb(0xE0);
    for card in root.discards.iter() {
        absorb(*card as u64 + 1);
    }
    hash
}

/// Order-sensitive fingerprint of a root's cards. Two roots that differ in any
/// card, or in which row a card sits, get different fingerprints.
pub fn fingerprint(root: &HeroT4Root) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut absorb = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    };
    for (row, cards) in [&root.top, &root.middle, &root.bottom].iter().enumerate() {
        absorb(0xF0 | row as u64);
        for card in cards.iter() {
            absorb(*card as u64 + 1);
        }
    }
    absorb(0xD0);
    for card in root.dealt.iter() {
        absorb(*card as u64 + 1);
    }
    absorb(0xE0);
    for card in root.discards.iter() {
        absorb(*card as u64 + 1);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::{generate_root, BehaviorConfig};
    use crate::objective::ObjectiveKind;

    fn objective() -> ObjectiveConfig {
        ObjectiveConfig {
            kind: ObjectiveKind::PureV1,
            fl_ev_stay: 9.109,
            fl_ev_config_path: "test".to_owned(),
            fl_ev_cards: 14,
            hero_foul_prob: 0.2482,
            hero_foul_prob_provenance: "test".to_owned(),
            reference: None,
        }
    }

    fn behavior() -> BehaviorConfig {
        BehaviorConfig {
            rollouts: 2,
            fl_ev: 9.109,
            foul_penalty: 11.0,
            foul_penalty_provenance: "test".to_owned(),
        }
    }

    #[test]
    fn labelling_is_deterministic_and_covers_every_legal_candidate() {
        let behavior = behavior();
        let objective = objective();
        let teacher = TeacherConfig {
            opponent_samples: 8,
            root_seed_base: 997_000_000,
            opponent_seed_base: 997_500_000,
        };
        let root = generate_root(997_000_000, 11, &behavior);
        let mut solver = FlSolver::new(&objective).unwrap();
        let first = label_root(&root, &objective, &teacher, "test", &mut solver).unwrap();
        let mut solver = FlSolver::new(&objective).unwrap();
        let again = label_root(&root, &objective, &teacher, "test", &mut solver).unwrap();

        assert_eq!(first.candidates.len(), again.candidates.len());
        assert!((3..=6).contains(&first.candidates.len()));
        for (left, right) in first.candidates.iter().zip(again.candidates.iter()) {
            assert_eq!(left.expected_value, right.expected_value);
            assert_eq!(left.action, right.action);
        }
        assert_eq!(first.best_candidate, again.best_candidate);
        assert!(first.decision_margin >= 0.0);
    }

    #[test]
    fn candidate_boards_use_exactly_the_root_cards_plus_two_dealt() {
        let behavior = behavior();
        let objective = objective();
        let teacher = TeacherConfig {
            opponent_samples: 4,
            root_seed_base: 997_000_000,
            opponent_seed_base: 997_500_000,
        };
        let root = generate_root(997_000_000, 21, &behavior);
        let mut solver = FlSolver::new(&objective).unwrap();
        let labelled = label_root(&root, &objective, &teacher, "test", &mut solver).unwrap();
        for candidate in labelled.candidates.iter() {
            let mut cards: Vec<u8> = candidate
                .top
                .iter()
                .chain(candidate.middle.iter())
                .chain(candidate.bottom.iter())
                .copied()
                .collect();
            assert_eq!(cards.len(), 13);
            cards.sort_unstable();
            cards.dedup();
            assert_eq!(cards.len(), 13, "candidate board repeated a card");
            assert!(!cards.contains(&candidate.discard));
        }
    }

    #[test]
    fn distinct_roots_get_distinct_fingerprints() {
        let behavior = behavior();
        let mut seen = std::collections::HashSet::new();
        for index in 0..64 {
            let root = generate_root(997_000_000, index, &behavior);
            assert!(seen.insert(fingerprint(&root)), "fingerprint collision");
        }
    }
}
