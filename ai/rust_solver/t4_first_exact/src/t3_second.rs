//! T3 second-seat (BTN) value with the learned T4 first-seat evaluator as leaf.
//!
//! BTN places two of three cards, then BB faces a T4 first-seat decision whose
//! node value the evaluator predicts.  BTN's action value is the negated mean
//! of that value over BB's draw.
//!
//! The whole node block -- the opponent per-row outlook, the joint completion
//! block and the context block -- is computed once per BTN action and shared
//! across BB's draws.  That sharing is what makes the evaluator worth having:
//! the joint block alone costs about as much as solving the node exactly, so
//! recomputing it per draw would buy nothing.

use anyhow::{anyhow, bail, Result};
use ofc_core::Card;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::evaluator;
use super::{
    all_cards, apply, legal_actions, opponent_self_value, terminal_of, BoardStr, CoreBoard,
    FlEv, Terminal,
};

#[derive(Deserialize)]
pub struct T3Request {
    pub id: String,
    /// BB board, 11 cards: BB already acted at T3.
    pub bb: BoardStr,
    /// BTN board, 9 cards: BTN is about to act.
    pub btn: BoardStr,
    /// BTN's own T1/T2 discards.
    pub btn_dead: Vec<String>,
    /// BTN's three drawn cards.
    pub draw: Vec<String>,
    /// BB draws sampled per BTN action; 0 enumerates every draw.
    #[serde(default)]
    pub draw_sample: usize,
}

#[derive(Serialize)]
pub struct T3ActionValue {
    pub action_key: String,
    pub value: f64,
    pub draws: usize,
}

#[derive(Serialize)]
pub struct T3Response {
    pub id: String,
    pub schema: &'static str,
    pub leaf: &'static str,
    pub belief: &'static str,
    pub node_block_shared: bool,
    pub fl_ev_config_sha256: String,
    pub actions: Vec<T3ActionValue>,
}

/// Distinct (which two of three cards, into which rows) shapes for a board.
pub(crate) struct Pattern {
    pub(crate) cards: [usize; 2],
    pub(crate) rows: [usize; 2],
}

pub(crate) fn placement_patterns(board: &CoreBoard) -> Vec<Pattern> {
    let open = board.open_slots();
    let mut seen: std::collections::BTreeSet<(usize, usize, usize, usize)> =
        std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for discard in 0..3usize {
        let kept: Vec<usize> = (0..3).filter(|index| *index != discard).collect();
        for row_a in 0..3usize {
            for row_b in 0..3usize {
                let mut need = [0usize; 3];
                need[row_a] += 1;
                need[row_b] += 1;
                if (0..3).any(|row| need[row] > open[row]) {
                    continue;
                }
                let key = if row_a == row_b {
                    (kept[0].min(kept[1]), kept[0].max(kept[1]), row_a, row_b)
                } else {
                    (kept[0], kept[1], row_a, row_b)
                };
                if seen.insert(key) {
                    out.push(Pattern {
                        cards: [kept[0], kept[1]],
                        rows: [row_a, row_b],
                    });
                }
            }
        }
    }
    out
}

/// The opponent's joint completion block over a pool, from cached pair
/// terminals.  Mirrors `opponent_joint_block` in the Python encoder.
pub(crate) fn joint_block(board: &CoreBoard, pool: &[Card], fl_ev: &FlEv) -> Result<[f64; 8]> {
    let open = board.open_slots();
    if open.iter().sum::<usize>() != 2 {
        bail!("joint block needs an opponent board with two open slots");
    }
    let mut assignments: Vec<[usize; 2]> = Vec::new();
    for row_a in 0..3usize {
        for row_b in 0..3usize {
            let mut need = [0usize; 3];
            need[row_a] += 1;
            need[row_b] += 1;
            if (0..3).any(|row| need[row] > open[row]) {
                continue;
            }
            if row_a == row_b {
                if !assignments.iter().any(|rows| rows == &[row_a, row_b]) {
                    assignments.push([row_a, row_b]);
                }
            } else {
                assignments.push([row_a, row_b]);
            }
        }
    }

    let pool_len = pool.len();
    let mut table: Vec<Option<Vec<Terminal>>> = vec![None; pool_len * pool_len];
    // Identity-keyed memo: the pair table completes the same board with
    // heavily repeating (row, card) additions, so row evaluations are shared
    // across the whole C(pool,2) sweep instead of recomputed per pair.
    let mut memo = super::row_memo::TerminalMemo::new(board);
    for i in 0..pool_len {
        for j in (i + 1)..pool_len {
            let mut local = Vec::with_capacity(assignments.len());
            for rows in &assignments {
                local.push(memo.terminal(&[(rows[0], pool[i]), (rows[1], pool[j])]));
            }
            table[i * pool_len + j] = Some(local);
        }
    }

    let mut best_self: Vec<f64> = Vec::new();
    let (mut fouls, mut survivors) = (0usize, 0usize);
    let (mut survive_royalty, mut survive_fl) = (0.0f64, 0.0f64);
    for a in 0..pool_len {
        for b in (a + 1)..pool_len {
            for c in (b + 1)..pool_len {
                let mut best = f64::NEG_INFINITY;
                let mut parts = (0.0f64, 0.0f64);
                for (first, second) in [(a, b), (a, c), (b, c)] {
                    let terminals = table[first * pool_len + second]
                        .as_ref()
                        .ok_or_else(|| anyhow!("pair terminal missing"))?;
                    for terminal in terminals {
                        let value = opponent_self_value(terminal, fl_ev);
                        if value > best {
                            best = value;
                            parts = if terminal.busted {
                                (0.0, 0.0)
                            } else {
                                (
                                    terminal.royalty as f64,
                                    fl_ev.value(terminal.fl_card_count),
                                )
                            };
                        }
                    }
                }
                if best <= -6.0 {
                    fouls += 1;
                } else {
                    survivors += 1;
                    survive_royalty += parts.0;
                    survive_fl += parts.1;
                }
                best_self.push(best);
            }
        }
    }
    let count = best_self.len() as f64;
    let mean = best_self.iter().sum::<f64>() / count;
    let variance = best_self.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / count;
    let denominator = survivors.max(1) as f64;
    const MAX_ROYALTY: f64 = 25.0;
    const MAX_FL_EV: f64 = 63.5;
    Ok([
        fouls as f64 / count,
        mean / MAX_ROYALTY,
        (survive_royalty / denominator) / MAX_ROYALTY,
        (survive_fl / denominator) / MAX_FL_EV,
        variance.sqrt() / MAX_ROYALTY,
        best_self.iter().filter(|v| **v >= 6.0).count() as f64 / count,
        best_self.iter().filter(|v| **v >= 15.0).count() as f64 / count,
        (best_self.iter().filter(|v| **v > -6.0).sum::<f64>() / denominator) / MAX_ROYALTY,
    ])
}

/// Deterministic draw sampling: a hash-driven shuffle, so a given root and
/// action always see the same draws regardless of thread scheduling.
fn sampled_draws(pool_len: usize, want: usize, seed: &str) -> Vec<[usize; 3]> {
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

pub fn solve(
    request: &T3Request,
    fl_ev: &FlEv,
    model: &evaluator::Model,
    fl_table: &evaluator::FlTable,
) -> Result<T3Response> {
    let bb_base = CoreBoard::from_str_board(&request.bb)?;
    let btn_base = CoreBoard::from_str_board(&request.btn)?;
    if bb_base.card_count() != 11 || btn_base.card_count() != 9 {
        bail!("T3 second seat needs an 11-card BB board and a 9-card BTN board");
    }
    if request.draw.len() != 3 || request.btn_dead.len() != 2 {
        bail!("T3 second seat needs a 3-card draw and two prior BTN discards");
    }

    let bb_patterns = placement_patterns(&bb_base);
    let btn_actions = legal_actions(&btn_base, &request.draw);
    let values: Result<Vec<T3ActionValue>> = btn_actions
        .iter()
        .map(|btn_action| {
            let btn_after = apply(&btn_base, btn_action)?;
            let mut known: std::collections::BTreeSet<String> =
                std::collections::BTreeSet::new();
            for card in request
                .bb
                .top
                .iter()
                .chain(&request.bb.middle)
                .chain(&request.bb.bottom)
                .chain(&request.btn.top)
                .chain(&request.btn.middle)
                .chain(&request.btn.bottom)
                .chain(&request.draw)
                .chain(&request.btn_dead)
            {
                known.insert(card.clone());
            }
            // BB's own earlier discards stay in the pool: BTN cannot see them.
            let pool: Vec<Card> = all_cards()
                .into_iter()
                .filter(|card| !known.contains(card))
                .map(|card| super::to_core_card(&card))
                .collect::<Result<Vec<_>>>()?;

            let mut shared: Vec<f32> = Vec::with_capacity(evaluator::OPPONENT_SIZE);
            let categories = evaluator::opponent_rowwise_block(
                &btn_after.rows,
                &pool,
                fl_table,
                &mut shared,
            );
            for value in joint_block(&btn_after, &pool, fl_ev)? {
                shared.push(value as f32);
            }
            let mut context: Vec<f32> = Vec::with_capacity(evaluator::CONTEXT_SIZE);
            evaluator::context_block(&pool, 3, &mut context);

            let draws = sampled_draws(
                pool.len(),
                request.draw_sample,
                &format!("{}/{}", request.id, btn_action.key()),
            );
            let mut total = 0.0f64;
            let mut features: Vec<f32> = Vec::with_capacity(evaluator::FEATURE_SIZE);
            let mut scratch: Vec<f32> = Vec::new();
            for draw in &draws {
                let bb_draw = [pool[draw[0]], pool[draw[1]], pool[draw[2]]];
                let mut best = f64::NEG_INFINITY;
                for pattern in &bb_patterns {
                    let mut final_bb = bb_base.clone();
                    final_bb.rows[pattern.rows[0]].push(bb_draw[pattern.cards[0]]);
                    final_bb.rows[pattern.rows[1]].push(bb_draw[pattern.cards[1]]);
                    let hero = evaluator::hero_eval(&final_bb.rows);
                    features.clear();
                    evaluator::hero_block(&final_bb.rows, &hero, fl_table, &mut features);
                    features.extend_from_slice(&shared);
                    evaluator::joint_block(
                        &hero,
                        &btn_after.rows,
                        &categories,
                        &mut features,
                    );
                    features.extend_from_slice(&context);
                    if features.len() != evaluator::FEATURE_SIZE {
                        bail!("feature width drifted");
                    }
                    let predicted = model.predict(&features, &mut scratch) as f64;
                    if predicted > best {
                        best = predicted;
                    }
                }
                total += best;
            }
            // Zero sum: BTN's value is the negated BB continuation.
            Ok(T3ActionValue {
                action_key: btn_action.key(),
                value: -total / draws.len() as f64,
                draws: draws.len(),
            })
        })
        .collect();

    let mut actions = values?;
    actions.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    Ok(T3Response {
        id: request.id.clone(),
        schema: "ofc_t3_second_value/v1",
        leaf: "learned_t4_first_evaluator",
        belief: "uniform_exchangeable_restart_v1",
        node_block_shared: true,
        fl_ev_config_sha256: fl_ev.config_sha256.clone(),
        actions,
    })
}
