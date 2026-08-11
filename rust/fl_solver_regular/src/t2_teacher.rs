//! The T2-vs-FL teacher.
//!
//! ```text
//! value(t2) = E_t3deal [ max over T3 actions of
//!               E_t4deal [ max over T4 actions of  E_fl[ score ] ] ]
//! ```
//!
//! Candidates are enumerated exhaustively at T3 and T4 -- those fans are about
//! 27 and 5 -- while the two *draws* are sampled, because `C(unseen, 3)` at
//! both levels is what makes the tree explode. Every maximum sits outside the
//! Fantasyland expectation, because the hero never sees the opponent hand.
//!
//! # Frontier caching
//!
//! The opponent frontiers are built once per root and shared by every terminal
//! in it. This is the single most important cost decision in the file: a
//! frontier costs about 47 ms and a root has tens of thousands of terminals, so
//! rebuilding one per terminal would dominate the run by four orders of
//! magnitude. Sharing is sound because a frontier depends only on the
//! opponent's fourteen cards, which are fixed per sample and independent of
//! which hero line reached the terminal.

use crate::behavior::{apply_turn, generate_turn_actions, TurnAction};
use crate::distribution::sample_and_solve;
use crate::frontier::{best_response, build_frontier, FrontierEntry};
use crate::objective::ObjectiveConfig;
use crate::scoring::BoardScore;
use crate::solver::FlSolver;
use crate::teacher::{external_fingerprint, ExternalRoot, Provenance, TeacherConfig};
use serde::{Deserialize, Serialize};

pub const T2_LABEL_SCHEMA: &str = "regular_ofc_t2_vs_fl_label_v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct T2CandidateLabel {
    pub action: TurnAction,
    pub top: Vec<u8>,
    pub middle: Vec<u8>,
    pub bottom: Vec<u8>,
    pub discard: u8,
    pub expected_value: f64,
    pub standard_error: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelledT2Root {
    pub root_index: u64,
    pub schema: String,
    pub top: Vec<u8>,
    pub middle: Vec<u8>,
    pub bottom: Vec<u8>,
    pub dealt: [u8; 3],
    pub discards: Vec<u8>,
    pub seen_mask: u64,
    pub candidates: Vec<T2CandidateLabel>,
    pub best_candidate: usize,
    pub decision_margin: f64,
    pub opponent_samples: usize,
    pub t3_draws: usize,
    pub t4_draws: usize,
    pub opponent_mode: String,
    pub mean_usable_opponents: f64,
    pub fingerprint: u64,
    pub provenance: Provenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_policy: Option<String>,
}

/// One sampled (T3 deal, T4 deals) bundle, shared across every T2 candidate so
/// candidate differences are not contaminated by sampling noise.
struct Draw {
    t3_deal: [u8; 3],
    t4_deals: Vec<[u8; 3]>,
    valid: Vec<Vec<u32>>,
    constant: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn label_t2_root(
    root: &ExternalRoot,
    objective: &ObjectiveConfig,
    teacher: &TeacherConfig,
    t3_draws: usize,
    t4_draws: usize,
    hero_deal_seed_base: u64,
    root_policy: &str,
    solver: &mut FlSolver,
) -> Result<LabelledT2Root, String> {
    let board = root.board();
    if board.card_count() != 7 {
        return Err(format!(
            "T2 root {} has {} placed cards, expected 7",
            root.root_index,
            board.card_count()
        ));
    }
    let dealt = root.dealt_three()?;
    let mut actions = Vec::with_capacity(27);
    generate_turn_actions(&board, &mut actions);
    if actions.is_empty() {
        return Err(format!("T2 root {} has no legal action", root.root_index));
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
    let mut draws: Vec<Draw> = Vec::with_capacity(t3_draws);
    for t3_index in 0..t3_draws as u64 {
        let mut rng = crate::rng::SplitMix64::for_stream(
            hero_deal_seed_base,
            root.root_index * 4096 + t3_index,
        );
        let mut pool = unseen.clone();
        rng.partial_shuffle(&mut pool, 3);
        let mut t3_deal: [u8; 3] = pool[..3].try_into().expect("three cards");
        t3_deal.sort_unstable();
        let t3_mask = crate::cards::mask_of(&t3_deal);
        let after_t3: Vec<u8> = unseen
            .iter()
            .copied()
            .filter(|card| t3_mask & (1_u64 << card) == 0)
            .collect();

        let mut t4_deals = Vec::with_capacity(t4_draws);
        let mut valid = Vec::with_capacity(t4_draws);
        let mut constant = Vec::with_capacity(t4_draws);
        for t4_index in 0..t4_draws as u64 {
            let mut rng4 = crate::rng::SplitMix64::for_stream(
                hero_deal_seed_base + 1,
                root.root_index * 262_144 + t3_index * 64 + t4_index,
            );
            let mut pool4 = after_t3.clone();
            rng4.partial_shuffle(&mut pool4, 3);
            let mut t4_deal: [u8; 3] = pool4[..3].try_into().expect("three cards");
            t4_deal.sort_unstable();
            // The opponent's fourteen must miss BOTH hero draws.
            let both = t3_mask | crate::cards::mask_of(&t4_deal);
            let usable: Vec<u32> = opponent_masks
                .iter()
                .enumerate()
                .filter(|(_, mask)| *mask & both == 0)
                .map(|(index, _)| index as u32)
                .collect();
            usable_total += usable.len();
            usable_count += 1;
            constant.push(usable.iter().map(|s| static_max[*s as usize]).sum());
            valid.push(usable);
            t4_deals.push(t4_deal);
        }
        draws.push(Draw {
            t3_deal,
            t4_deals,
            valid,
            constant,
        });
    }

    let mut t3_actions = Vec::with_capacity(27);
    let mut t4_actions = Vec::with_capacity(6);
    let mut candidates = Vec::with_capacity(actions.len());
    for action in actions.iter() {
        let nine = apply_turn(&board, &dealt, action);
        generate_turn_actions(&nine, &mut t3_actions);
        let t3_action_list = t3_actions.clone();

        let mut t3_values = Vec::with_capacity(t3_draws);
        for draw in draws.iter() {
            let mut best_t3 = f64::NEG_INFINITY;
            for t3_action in t3_action_list.iter() {
                let eleven = apply_turn(&nine, &draw.t3_deal, t3_action);
                generate_turn_actions(&eleven, &mut t4_actions);
                let mut total = 0.0_f64;
                let mut used = 0_usize;
                for (slot, t4_deal) in draw.t4_deals.iter().enumerate() {
                    let usable = &draw.valid[slot];
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
                            (-6.0 * count - draw.constant[slot]) / count
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
                if used == 0 {
                    continue;
                }
                let value = total / used as f64;
                if value > best_t3 {
                    best_t3 = value;
                }
            }
            if best_t3.is_finite() {
                t3_values.push(best_t3);
            }
        }
        if t3_values.is_empty() {
            return Err(format!("T2 root {}: no usable draw", root.root_index));
        }
        let count = t3_values.len() as f64;
        let mean = t3_values.iter().sum::<f64>() / count;
        let variance = t3_values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / count;
        candidates.push(T2CandidateLabel {
            action: *action,
            top: nine.rows[0][..nine.lengths[0] as usize].to_vec(),
            middle: nine.rows[1][..nine.lengths[1] as usize].to_vec(),
            bottom: nine.rows[2][..nine.lengths[2] as usize].to_vec(),
            discard: dealt[action.discard as usize],
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
    let decision_margin = if sorted.len() > 1 { sorted[0] - sorted[1] } else { 0.0 };

    Ok(LabelledT2Root {
        root_index: root.root_index,
        schema: T2_LABEL_SCHEMA.to_owned(),
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
        t3_draws,
        t4_draws,
        opponent_mode: "adaptive_v1".to_owned(),
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
    })
}
