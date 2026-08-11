//! The T0 first-seat-vs-FL teacher.
//!
//! ```text
//! value(t0) = E_t1deal [ pi_T1( . ) -> E_t2deal [ pi_T2( . ) ->
//!               E_t3deal [ pi_T3( . ) ->
//!                 E_t4deal [ max over T4 actions of E_fl[ score ] ] ] ] ]
//! ```
//!
//! One street deeper than [`crate::t1_teacher`] and otherwise its shape: T1, T2
//! and T3 are *continued* by the distilled vs-Fantasyland rankers rather than
//! searched, T4 stays exhaustive, and the terminal stays the adaptive
//! Fantasyland frontier.
//!
//! # Why this street is worth a teacher of its own
//!
//! Measured 2026-08-08 on the normal-table T0 evaluation: five seeds gave five
//! different "best" openings, per-action scores moved 26 points on average, and
//! neither raising the sampled worlds to 256 nor cutting the fan to ten made
//! that converge. An ablation over architecture, width, features and objective
//! then found the shipped T0 model recovers its teacher's own best action only
//! 50-54% of the time whatever it is trained as -- so the ceiling is the
//! labels, and the labels are noisy because the opponent is simulated.
//!
//! Against a Fantasyland opponent the opponent is not simulated. The
//! Fantasyland side is solved exactly by the non-dominated frontier, which was
//! pinned against brute force on 344 pairs, so it contributes no variance at
//! all. What is left is hero's own draw, which is the variance a T0 label
//! should have.
//!
//! # What is assumed, and what is measured
//!
//! At T0 first seat hero sees nothing of the opponent in either game -- both
//! boards are empty and no card has been discarded -- so the decision faces the
//! same information here as on a normal table. That is why this street, and
//! only this street, can stand in for the normal one; from T1 on the normal
//! game shows hero the opponent's board while the Fantasyland game shows
//! nothing, and the two stop being the same decision.
//!
//! It does NOT follow that the optimum is the same. A Fantasyland opponent is
//! far stronger on average, which may shift hero toward royalties and its own
//! Fantasyland entry and away from contesting rows. That is a difference in the
//! objective, not in the information, and it is a thing to measure by comparing
//! the two rankings -- not to assume in either direction.
//!
//! # Common random numbers
//!
//! Every one of the 232 openings sees the same sampled T1/T2/T3/T4 deals and
//! the same opponent samples. The deals are drawn from the root's unseen set,
//! which already excludes the five dealt cards, so the schedule is
//! candidate-independent exactly as it is one street up.

use crate::behavior::{apply_turn, generate_opening_actions, generate_turn_actions, PartialBoard};
use crate::distribution::sample_and_solve;
use std::borrow::Cow;

use crate::frontier::{best_response, build_frontier, FrontierEntry};
use crate::objective::ObjectiveConfig;
use crate::scoring::BoardScore;
use crate::solver::FlSolver;
use crate::t1_teacher::{Continuation, ContinuationProvenance};
use crate::teacher::{external_fingerprint, ExternalRoot, Provenance, TeacherConfig};
use ofc_hu_m3_engine::infoset::Street;
use serde::{Deserialize, Serialize};

pub const T0_LABEL_SCHEMA: &str = "regular_ofc_t0_first_vs_fl_label_v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct T0CandidateLabel {
    /// Row per dealt card, in the order the five were dealt.
    pub assignment: [u8; 5],
    pub top: Vec<u8>,
    pub middle: Vec<u8>,
    pub bottom: Vec<u8>,
    pub expected_value: f64,
    pub standard_error: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelledT0Root {
    pub root_index: u64,
    pub schema: String,
    pub dealt: [u8; 5],
    pub seen_mask: u64,
    pub candidates: Vec<T0CandidateLabel>,
    pub best_candidate: usize,
    pub decision_margin: f64,
    pub opponent_samples: usize,
    pub t1_draws: usize,
    pub t2_draws: usize,
    pub t3_draws: usize,
    pub t4_draws: usize,
    pub opponent_mode: String,
    pub continuation: ContinuationProvenance,
    pub mean_usable_opponents: f64,
    pub fingerprint: u64,
    pub provenance: Provenance,
    pub root_policy: Option<String>,
    /// How many of the root's legal openings were evaluated and by whose
    /// ranking the rest were dropped. Absent means the whole fan was scored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub narrow_keep: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub narrow_model_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legal_openings: Option<usize>,
}

/// One sampled T1 deal and everything nested beneath it, shared by all 232
/// openings.
struct T1Draw<'a> {
    t1_deal: [u8; 3],
    t2: Vec<T2Draw<'a>>,
}

struct T2Draw<'a> {
    t2_deal: [u8; 3],
    t3: Vec<T3Draw<'a>>,
}

struct T3Draw<'a> {
    t3_deal: [u8; 3],
    t4_deals: Vec<[u8; 3]>,
    /// Fantasyland frontiers drawn at this leaf, from the deck hero left.
    /// One vector per T4 deal, because the deck each one leaves is different.
    ///
    /// Borrowed from the pool when there is one, and only owned when the leaf
    /// solved for itself. The whole tree is materialised before any opening is
    /// scored, so copying here multiplies: at 64 draws either side and four
    /// below, copying cost 32 GB and the kernel killed the process with no
    /// message at all -- the same silent shape as the solver-chunk stall,
    /// found again by an empty output file.
    frontiers: Vec<Vec<Cow<'a, [FrontierEntry]>>>,
    /// What a fouled hero board concedes at this leaf.
    constant: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn label_t0_root(
    root: &ExternalRoot,
    objective: &ObjectiveConfig,
    teacher: &TeacherConfig,
    continuation: &Continuation,
    t1_model: &crate::vfl_model::VflModel,
    t1_draws: usize,
    t2_draws: usize,
    t3_draws: usize,
    t4_draws: usize,
    hero_deal_seed_base: u64,
    root_policy: &str,
    narrow_model: Option<&crate::vfl_model::VflModel>,
    narrow_keep: usize,
    narrow_model_sha256: Option<&str>,
    library: Option<&crate::fl_library::FlLibrary>,
    solver: &mut FlSolver,
) -> Result<LabelledT0Root, String> {
    let dealt: [u8; 5] = root
        .dealt
        .as_slice()
        .try_into()
        .map_err(|_| format!("T0 root {} needs five dealt cards", root.root_index))?;
    if !root.top.is_empty() || !root.middle.is_empty() || !root.bottom.is_empty() {
        return Err(format!(
            "T0 first-seat root {} must have an empty board",
            root.root_index
        ));
    }

    let mut openings: Vec<[u8; 5]> = Vec::with_capacity(232);
    generate_opening_actions(&mut openings);
    let legal_openings = openings.len();
    // Optionally cut the fan. This is where T0's cost is -- unlike T1, where
    // the shared draw tree dominates and narrowing to ten of twenty-seven bought
    // only 1.3x, every T0 opening leaves a different board, so its Fantasyland
    // solving is its own and the fan multiplies nearly everything.
    //
    // It stayed unimplemented while the only available ranker was the
    // normal-table T0 model, because cutting with THAT would have prejudged this
    // teacher's whole purpose: a Fantasyland-specific preference could never be
    // observed if every opening the normal model ranks low is gone before this
    // teacher scores it. A vs-Fantasyland T0 ranker, bootstrapped from an
    // unnarrowed sample of this teacher's own labels, carries no such prejudice
    // -- so the model is a parameter and the caller supplies it or does not.
    //
    // Survivors keep the generator's order, so a label depends on which openings
    // were kept and not on how the ranker ordered the ones it kept.
    if let Some(model) = narrow_model {
        if narrow_keep > 0 && narrow_keep < openings.len() {
            let mut scratch = model.scratch();
            let mut node = crate::engine_features::NodeFeatures::new(
                &PartialBoard::default(),
                &dealt[..],
                &[],
                Street::T0,
            )?;
            let mut scored: Vec<(usize, f32)> = Vec::with_capacity(openings.len());
            for (index, assignment) in openings.iter().enumerate() {
                // Built exactly as the evaluation below builds it, so the
                // ranker sees the board the teacher would have scored. An
                // assignment that overfills a row is not legal and is dropped
                // here as it is dropped there.
                let mut five = PartialBoard::default();
                let mut legal = true;
                for (slot, row) in assignment.iter().enumerate() {
                    if five.open_slots(*row) == 0 {
                        legal = false;
                        break;
                    }
                    five.push(dealt[slot], *row);
                }
                if !legal {
                    continue;
                }
                let row = node.encode(&five)?;
                scored.push((index, model.predict(&row, &mut scratch)?));
            }
            if scored.len() <= narrow_keep {
                // Fewer legal openings than the cut asks to keep; nothing to do.
                scored.clear();
            }
            if !scored.is_empty() {
                scored.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });
                let mut survivors: Vec<usize> =
                    scored[..narrow_keep].iter().map(|(i, _)| *i).collect();
                survivors.sort_unstable();
                openings = survivors.into_iter().map(|i| openings[i]).collect();
            }
        }
    }

    // The Fantasyland hand is drawn AT THE LEAF, from what is left once hero's
    // twelve are known -- not once at the root and then filtered.
    //
    // Drawing it at the root means one fourteen-card hand has to avoid every
    // branch of hero's draw tree at once, and a tree of 8x4x2x2 leaves uses far
    // too much of the deck for that: measured on this teacher's first run, 200
    // sampled opponents left a mean of 0.2-1.5 usable per leaf. A teacher that
    // says "200 samples" and averages one is not sampling.
    //
    // Dealing in the other order is the same distribution -- the deal is
    // exchangeable, so "opponent's fourteen, then hero's twelve" and "hero's
    // twelve, then opponent's fourteen" agree -- and it cannot collide by
    // construction, so nothing is discarded. The tree then supplies the
    // opponent variety for free: one hand per leaf is already 128 distinct
    // opponents per root, where the old shape built 200 frontiers to use ~1.5.
    let unseen = crate::cards::unseen_from_mask(root.seen_mask);
    let mut usable_total = 0_usize;
    let mut usable_count = 0_usize;

    // The four nested hero draws, drawn once and shared by every opening.
    let mut draws: Vec<T1Draw> = Vec::with_capacity(t1_draws);
    for t1_index in 0..t1_draws as u64 {
        let mut rng1 =
            crate::rng::SplitMix64::for_stream(hero_deal_seed_base, root.root_index * 4096 + t1_index);
        let mut pool1 = unseen.clone();
        rng1.partial_shuffle(&mut pool1, 3);
        let mut t1_deal: [u8; 3] = pool1[..3].try_into().expect("three cards");
        t1_deal.sort_unstable();
        let t1_mask = crate::cards::mask_of(&t1_deal);
        let after_t1: Vec<u8> = unseen
            .iter()
            .copied()
            .filter(|card| t1_mask & (1_u64 << card) == 0)
            .collect();

        let mut t2_list = Vec::with_capacity(t2_draws);
        for t2_index in 0..t2_draws as u64 {
            let mut rng2 = crate::rng::SplitMix64::for_stream(
                hero_deal_seed_base + 1,
                root.root_index * 262_144 + t1_index * 64 + t2_index,
            );
            let mut pool2 = after_t1.clone();
            rng2.partial_shuffle(&mut pool2, 3);
            let mut t2_deal: [u8; 3] = pool2[..3].try_into().expect("three cards");
            t2_deal.sort_unstable();
            let t2_mask = crate::cards::mask_of(&t2_deal);
            let after_t2: Vec<u8> = after_t1
                .iter()
                .copied()
                .filter(|card| t2_mask & (1_u64 << card) == 0)
                .collect();

            let mut t3_list = Vec::with_capacity(t3_draws);
            for t3_index in 0..t3_draws as u64 {
                let mut rng3 = crate::rng::SplitMix64::for_stream(
                    hero_deal_seed_base + 2,
                    root.root_index * 16_777_216
                        + t1_index * 4096
                        + t2_index * 64
                        + t3_index,
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
                let mut frontiers = Vec::with_capacity(t4_draws);
                let mut constant = Vec::with_capacity(t4_draws);
                for t4_index in 0..t4_draws as u64 {
                    let leaf = root.root_index * 1_073_741_824
                        + t1_index * 262_144
                        + t2_index * 4096
                        + t3_index * 64
                        + t4_index;
                    let mut rng4 =
                        crate::rng::SplitMix64::for_stream(hero_deal_seed_base + 3, leaf);
                    let mut pool4 = after_t3.clone();
                    rng4.partial_shuffle(&mut pool4, 3);
                    let mut t4_deal: [u8; 3] = pool4[..3].try_into().expect("three cards");
                    t4_deal.sort_unstable();

                    // Hero's twelve are now fixed for this leaf, so the deck
                    // the Fantasyland hand comes from is what remains: the
                    // opponent cannot hold a card hero was dealt.
                    let hero_mask = root.seen_mask
                        | t1_mask
                        | t2_mask
                        | t3_mask
                        | crate::cards::mask_of(&t4_deal);
                    // With a pool, the leaf selects; without one it solves.
                    // Selecting is the same distribution -- a uniform hand
                    // conditioned on avoiding hero's seventeen is a uniform
                    // hand from the thirty-five left -- and skips the 5.18 ms
                    // solve and 38.3 ms frontier build that dominate the cost.
                    let leaf_frontiers: Vec<Cow<[FrontierEntry]>> = match library {
                        Some(pool) => {
                            let drawn = pool.draw(
                                hero_mask,
                                teacher.opponent_samples,
                                teacher.opponent_seed_base
                                    .wrapping_add(leaf)
                                    .wrapping_mul(0x2545_F491_4F6C_DD1D),
                            );
                            if drawn.len() < teacher.opponent_samples {
                                return Err(format!(
                                    "the pool supplied {} of {} opponents at a leaf;                                      it is too small for this sample count -- a leaf                                      that averages fewer than the plan says is the                                      silent failure the rejection shape had",
                                    drawn.len(),
                                    teacher.opponent_samples
                                ));
                            }
                            drawn
                                .iter()
                                .map(|entry| Cow::Borrowed(entry.frontier.as_slice()))
                                .collect()
                        }
                        None => sample_and_solve(
                            hero_mask,
                            teacher.opponent_samples,
                            teacher.opponent_seed_base,
                            leaf * teacher.opponent_samples as u64,
                            solver,
                        )?
                        .iter()
                        .map(|s| Cow::Owned(build_frontier(&s.deal, objective.fl_ev_stay)))
                        .collect(),
                    };
                    // What a fouled hero board concedes: the opponent's best
                    // static value, summed over the samples at this leaf.
                    let conceded: f64 = leaf_frontiers
                        .iter()
                        .map(|frontier| {
                            frontier
                                .iter()
                                .map(|entry| entry.static_value)
                                .fold(f64::NEG_INFINITY, f64::max)
                        })
                        .sum();
                    usable_total += leaf_frontiers.len();
                    usable_count += 1;
                    constant.push(conceded);
                    frontiers.push(leaf_frontiers);
                    t4_deals.push(t4_deal);
                }
                t3_list.push(T3Draw {
                    t3_deal,
                    t4_deals,
                    frontiers,
                    constant,
                });
            }
            t2_list.push(T2Draw { t2_deal, t3: t3_list });
        }
        draws.push(T1Draw { t1_deal, t2: t2_list });
    }

    let mut t1_actions = Vec::with_capacity(27);
    let mut t2_actions = Vec::with_capacity(27);
    let mut t3_actions = Vec::with_capacity(27);
    let mut t4_actions = Vec::with_capacity(6);
    let mut scratch = continuation.t2.scratch();
    let mut candidates = Vec::with_capacity(openings.len());

    for assignment in openings.iter() {
        let mut five = PartialBoard::default();
        let mut legal = true;
        for (slot, row) in assignment.iter().enumerate() {
            if five.open_slots(*row) == 0 {
                legal = false;
                break;
            }
            five.push(dealt[slot], *row);
        }
        if !legal {
            continue;
        }

        let mut t1_values = Vec::with_capacity(t1_draws);
        for d1 in draws.iter() {
            generate_turn_actions(&five, &mut t1_actions);
            let chosen1 = crate::t1_teacher::pick_continuation(
                t1_model,
                &mut scratch,
                &five,
                &d1.t1_deal,
                &[],
                Street::T1,
                &t1_actions,
            )?;
            let t1_action = t1_actions[chosen1];
            let seven = apply_turn(&five, &d1.t1_deal, &t1_action);
            let thrown_after_t1 = [d1.t1_deal[t1_action.discard as usize]];

            let mut t2_values = Vec::with_capacity(d1.t2.len());
            for d2 in d1.t2.iter() {
                generate_turn_actions(&seven, &mut t2_actions);
                let chosen2 = crate::t1_teacher::pick_continuation(
                    &continuation.t2,
                    &mut scratch,
                    &seven,
                    &d2.t2_deal,
                    &thrown_after_t1,
                    Street::T2,
                    &t2_actions,
                )?;
                let t2_action = t2_actions[chosen2];
                let nine = apply_turn(&seven, &d2.t2_deal, &t2_action);
                let thrown_after_t2 = [
                    thrown_after_t1[0],
                    d2.t2_deal[t2_action.discard as usize],
                ];

                let mut t3_values = Vec::with_capacity(d2.t3.len());
                for t3 in d2.t3.iter() {
                    generate_turn_actions(&nine, &mut t3_actions);
                    let chosen3 = crate::t1_teacher::pick_continuation(
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
                        let leaf_frontiers = &t3.frontiers[slot];
                        if leaf_frontiers.is_empty() {
                            continue;
                        }
                        let count = leaf_frontiers.len() as f64;
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
                                for frontier in leaf_frontiers.iter() {
                                    let (opponent_best, _) = best_response(
                                        frontier,
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
            if !t2_values.is_empty() {
                t1_values.push(t2_values.iter().sum::<f64>() / t2_values.len() as f64);
            }
        }
        if t1_values.is_empty() {
            return Err(format!("T0 root {}: no usable draw", root.root_index));
        }
        let count = t1_values.len() as f64;
        let mean = t1_values.iter().sum::<f64>() / count;
        let variance = t1_values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / count;
        candidates.push(T0CandidateLabel {
            assignment: *assignment,
            top: five.rows[0][..five.lengths[0] as usize].to_vec(),
            middle: five.rows[1][..five.lengths[1] as usize].to_vec(),
            bottom: five.rows[2][..five.lengths[2] as usize].to_vec(),
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

    Ok(LabelledT0Root {
        root_index: root.root_index,
        schema: T0_LABEL_SCHEMA.to_owned(),
        dealt,
        seen_mask: root.seen_mask,
        candidates,
        best_candidate,
        decision_margin,
        // Per leaf now, not per root -- the total drawn for this root is this
        // times the number of leaves, and every one of them is used.
        opponent_samples: teacher.opponent_samples,
        t1_draws,
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
        narrow_keep: if openings.len() < legal_openings {
            Some(narrow_keep)
        } else {
            None
        },
        narrow_model_sha256: if openings.len() < legal_openings {
            narrow_model_sha256.map(str::to_owned)
        } else {
            None
        },
        legal_openings: Some(legal_openings),
    })
}
