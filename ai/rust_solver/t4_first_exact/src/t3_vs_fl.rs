//! T3 decision for the normal player facing a Fantasyland opponent, in Rust.
//!
//! Mirrors `ai/tutor/t3_vs_fl.py`: a T3 action's value is the mean over the
//! hero's own T4 draw of the T4-vs-FL evaluator's node value.  The evaluator
//! marginalizes the hidden FL hand given the hero's seen cards, so sampling
//! draws and taking the per-draw max over completions integrates over
//! (draw, FL hand) with the right joint.  The feature layout is pinned to
//! `t4_vs_fl.encode_action` (hero block 42 + FL context 12 = 54) and the
//! parity harness drives both implementations over identical draw lists.

use anyhow::{bail, Result};
use ofc_core::Card;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::evaluator;
use super::{all_cards, apply, legal_actions, to_core_card, BoardStr, CoreBoard, FlEv};

#[derive(Deserialize)]
pub struct T3VsFlRequest {
    pub id: String,
    /// Hero board, 9 cards, four open slots.
    pub board: BoardStr,
    /// Hero's two prior discards.
    pub dead: Vec<String>,
    /// Hero's three drawn cards.
    pub draw: Vec<String>,
    /// Opponent's Fantasyland card count (14..17), public information.
    pub opp_count: u8,
    /// T4 draws sampled per action; 0 enumerates all C(n,3).
    #[serde(default)]
    pub draw_sample: usize,
    /// Explicit draw override for cross-language parity tests.
    #[serde(default)]
    pub draws: Option<Vec<Vec<String>>>,
}

#[derive(Serialize)]
pub struct T3VsFlActionValue {
    pub action_key: String,
    pub value: f64,
    pub draws: usize,
}

#[derive(Serialize)]
pub struct T3VsFlResponse {
    pub id: String,
    pub schema: &'static str,
    pub leaf: &'static str,
    pub actions: Vec<T3VsFlActionValue>,
}

/// FL context block (12 dims), pinned to `t4_vs_fl.encode_action`.
pub(crate) fn fl_context(
    pool: &[Card],
    opp_count: u8,
    fl_ev: &FlEv,
    out: &mut Vec<f32>,
) {
    let mut jokers = 0u32;
    let (mut aces, mut kings, mut queens) = (0u32, 0u32, 0u32);
    for card in pool {
        if card.is_joker() {
            jokers += 1;
        } else {
            match card.rank {
                14 => aces += 1,
                13 => kings += 1,
                12 => queens += 1,
                _ => {}
            }
        }
    }
    let mut one_hot = [0.0f32; 4];
    one_hot[(opp_count - 14) as usize] = 1.0;
    out.extend_from_slice(&one_hot);
    out.push(jokers as f32 / 2.0);
    out.push(aces as f32 / 4.0);
    out.push(kings as f32 / 4.0);
    out.push(queens as f32 / 4.0);
    out.push(pool.len() as f32 / 54.0);
    out.push(fl_ev.value(opp_count) as f32 / 63.5);
    out.push((aces + kings + queens) as f32 / pool.len().max(1) as f32);
    out.push((2 - jokers.min(2)) as f32 / 2.0);
}

pub(crate) fn sampled_draws(pool_len: usize, want: usize, seed: &str) -> Vec<[usize; 3]> {
    let mut all: Vec<[usize; 3]> = Vec::new();
    for a in 0..pool_len {
        for b in (a + 1)..pool_len {
            for c in (b + 1)..pool_len {
                all.push([a, b, c]);
            }
        }
    }
    if want == 0 || want >= all.len() {
        return all;
    }
    let mut counter: u64 = 0;
    for index in (1..all.len()).rev() {
        let mut hasher = Sha256::new();
        hasher.update(seed.as_bytes());
        hasher.update(counter.to_le_bytes());
        counter += 1;
        let digest = hasher.finalize();
        let value = u64::from_le_bytes(digest[..8].try_into().unwrap());
        all.swap(index, (value % (index as u64 + 1)) as usize);
    }
    all.truncate(want);
    all
}

/// Draw selection shared with the library-leaf module: explicit override for
/// parity, hash-sampled otherwise.
pub(crate) fn draws_for(
    request: &T3VsFlRequest,
    unseen: &[Card],
    seed: &str,
) -> Result<Vec<[usize; 3]>> {
    if let Some(explicit) = &request.draws {
        let index_of = |name: &str| -> Result<usize> {
            let target = super::to_core_card(name)?;
            unseen
                .iter()
                .position(|card| *card == target)
                .ok_or_else(|| anyhow::anyhow!("parity draw card not unseen"))
        };
        return explicit
            .iter()
            .map(|cards| {
                Ok([
                    index_of(&cards[0])?,
                    index_of(&cards[1])?,
                    index_of(&cards[2])?,
                ])
            })
            .collect();
    }
    Ok(sampled_draws(unseen.len(), request.draw_sample, seed))
}

pub fn solve(
    request: &T3VsFlRequest,
    fl_ev: &FlEv,
    model: &evaluator::Model,
    fl_table: &evaluator::FlTable,
) -> Result<T3VsFlResponse> {
    let base = CoreBoard::from_str_board(&request.board)?;
    if base.card_count() != 9 {
        bail!("T3-vs-FL needs a 9-card hero board");
    }
    if request.draw.len() != 3 || request.dead.len() != 2 {
        bail!("T3-vs-FL needs a 3-card draw and two prior discards");
    }
    if !(14..=17).contains(&request.opp_count) {
        bail!("opp_count must be 14..17");
    }

    let actions = legal_actions(&base, &request.draw);
    let values: Result<Vec<T3VsFlActionValue>> = actions
        .par_iter()
        .map(|action| {
            let after = apply(&base, action)?;
            let mut seen: std::collections::BTreeSet<String> =
                std::collections::BTreeSet::new();
            for card in request
                .board
                .top
                .iter()
                .chain(&request.board.middle)
                .chain(&request.board.bottom)
                .chain(&request.draw)
                .chain(&request.dead)
            {
                seen.insert(card.clone());
            }
            let unseen: Vec<Card> = all_cards()
                .into_iter()
                .filter(|card| !seen.contains(card))
                .map(|card| to_core_card(&card))
                .collect::<Result<Vec<_>>>()?;

            let draw_sets: Vec<[usize; 3]> = if let Some(explicit) = &request.draws {
                let index_of = |name: &str| -> Result<usize> {
                    let target = to_core_card(name)?;
                    unseen
                        .iter()
                        .position(|card| *card == target)
                        .ok_or_else(|| anyhow::anyhow!("parity draw card not unseen"))
                };
                explicit
                    .iter()
                    .map(|cards| {
                        Ok([
                            index_of(&cards[0])?,
                            index_of(&cards[1])?,
                            index_of(&cards[2])?,
                        ])
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                sampled_draws(
                    unseen.len(),
                    request.draw_sample,
                    &format!("{}/{}", request.id, action.key()),
                )
            };

            let patterns = super::t3_second::placement_patterns(&after);
            let mut total = 0.0f64;
            let mut features: Vec<f32> = Vec::with_capacity(evaluator::FEATURE_SIZE);
            let mut scratch: Vec<f32> = Vec::new();
            for draw in &draw_sets {
                let draw_cards = [unseen[draw[0]], unseen[draw[1]], unseen[draw[2]]];
                let pool: Vec<Card> = unseen
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !draw.contains(index))
                    .map(|(_, card)| *card)
                    .collect();
                let mut best = f64::NEG_INFINITY;
                for pattern in &patterns {
                    let mut final_board = after.clone();
                    final_board.rows[pattern.rows[0]].push(draw_cards[pattern.cards[0]]);
                    final_board.rows[pattern.rows[1]].push(draw_cards[pattern.cards[1]]);
                    let hero = evaluator::hero_eval(&final_board.rows);
                    features.clear();
                    evaluator::hero_block(&final_board.rows, &hero, fl_table, &mut features);
                    fl_context(&pool, request.opp_count, fl_ev, &mut features);
                    if features.len() != evaluator::HERO_SIZE + 12 {
                        bail!("t3-vs-fl feature width drifted");
                    }
                    let predicted = model.predict(&features, &mut scratch) as f64;
                    if predicted > best {
                        best = predicted;
                    }
                }
                total += best;
            }
            Ok(T3VsFlActionValue {
                action_key: action.key(),
                value: total / draw_sets.len().max(1) as f64,
                draws: draw_sets.len(),
            })
        })
        .collect();

    let mut actions_out = values?;
    actions_out.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    Ok(T3VsFlResponse {
        id: request.id.clone(),
        schema: "ofc_t3_vs_fl_value/v1",
        leaf: "learned_t4_vs_fl_evaluator",
        actions: actions_out,
    })
}
