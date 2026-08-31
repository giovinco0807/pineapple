//! T3-first (BB) teacher labels for the HU phase.
//!
//! Hero acts first in the T3 street, so after hero's placement the opponent
//! still places its own T3 -- seeing hero's fresh eleven-card board -- and
//! only then does the T4 street start, with hero acting first again.  A
//! label is therefore half a street of opponent expansion deep:
//!
//! ```text
//!   label(action) = E[opp T3 draw] ( opp chooser places -> V4_first(hero) )
//! ```
//!
//! `V4_first` applies directly, no zero-sum flip: hero is the T4-street
//! first actor.
//!
//! The opponent's move inside the expansion is a chooser, not a value: the
//! generation-3 own-hand ranker picks its placement, and the measured
//! tolerance for choosers (their errors ride common-mode across hero's
//! actions under common random numbers) is what licenses using an
//! opponent-blind model there until an HU-native chooser exists.  The
//! opponent's draw is sampled with common random numbers across hero's
//! actions -- hero's own cards are the same whatever hero placed, so the
//! same drawn triples serve every action and cancel in the differences a
//! teacher is read for.

use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::v4_first::{self, V4FirstRequest};
use super::{evaluator, playout, to_core_card, BoardStr, CoreBoard, FlEv};

const ROW_CAPACITY: [usize; 3] = [3, 5, 5];

#[derive(Deserialize)]
pub struct T3FirstHuRequest {
    pub id: String,
    /// Hero's nine placed cards.
    pub board: [Vec<String>; 3],
    /// Hero's two prior discards.
    pub dead: Vec<String>,
    /// Hero's three drawn cards.
    pub draw: Vec<String>,
    /// Opponent's board after its T2: nine cards (hero acts first at T3).
    pub opp_board: [Vec<String>; 3],
    /// Opponent T3 draws sampled per action (common across actions).
    #[serde(default = "default_opp_draws")]
    pub opp_draws: usize,
    /// The opponent's actual traced T3 after-board (eleven cards).  Present,
    /// the half-street expansion collapses to this one unbiased sample and
    /// every action is a single exact V4 state read -- the same lap-one
    /// convention the streets below use.  Absent, the sampled expansion runs.
    #[serde(default)]
    pub opp_after: Option<[Vec<String>; 3]>,
    /// Shifts the opponent-draw stream for floor passes.
    #[serde(default)]
    pub stream_offset: u64,
}

fn default_opp_draws() -> usize {
    16
}

#[derive(Serialize)]
pub struct T3FirstHuAction {
    pub action_key: String,
    pub value: f64,
}

#[derive(Serialize)]
pub struct T3FirstHuResponse {
    pub id: String,
    pub schema: &'static str,
    pub opp_draws: usize,
    pub actions: Vec<T3FirstHuAction>,
}

fn rows_key(rows: &[Vec<String>; 3], discard: &str) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(4);
    for row in rows {
        let mut sorted = row.clone();
        sorted.sort();
        parts.push(sorted.join(","));
    }
    parts.push(discard.to_string());
    parts.join("|")
}

/// Two-of-three placements into open slots, generic over the board shape.
fn placements(board: &[Vec<String>; 3], draw: &[String]) -> Vec<([Vec<String>; 3], String)> {
    let mut out = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for discard in 0..3usize {
        let kept: Vec<usize> = (0..3).filter(|k| *k != discard).collect();
        for row_a in 0..3usize {
            for row_b in 0..3usize {
                let mut need = [0usize; 3];
                need[row_a] += 1;
                need[row_b] += 1;
                if (0..3).any(|r| board[r].len() + need[r] > ROW_CAPACITY[r]) {
                    continue;
                }
                let mut after = board.clone();
                after[row_a].push(draw[kept[0]].clone());
                after[row_b].push(draw[kept[1]].clone());
                let key = rows_key(&after, &draw[discard]);
                if seen.insert(key) {
                    out.push((after, draw[discard].clone()));
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn solve(
    request: &T3FirstHuRequest,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    opp_chooser: &evaluator::Model,
    // Let each V4 read collapse its state sweep over rank classes when both
    // boards are flush-dead; see rank_collapse.rs.  The opponent's sampled
    // draws are untouched -- the collapse sits strictly inside one exact
    // state solve, never across the sampling.
    rank_collapse: bool,
) -> Result<T3FirstHuResponse> {
    if request.board.iter().map(Vec::len).sum::<usize>() != 9
        || request.opp_board.iter().map(Vec::len).sum::<usize>() != 9
    {
        bail!("{}: T3-first wants two nine-card boards", request.id);
    }
    if request.dead.len() != 2 || request.draw.len() != 3 {
        bail!("{}: T3 wants two dead and three drawn", request.id);
    }

    // Hero's unseen: everything but own nine, own dead, own draw and the
    // opponent's visible nine.  Jokers counted, never name-matched.
    let mut seen_naturals: std::collections::BTreeSet<&String> =
        std::collections::BTreeSet::new();
    let mut seen_jokers = 0usize;
    for name in request
        .dead
        .iter()
        .chain(&request.draw)
        .chain(request.board.iter().flatten())
        .chain(request.opp_board.iter().flatten())
    {
        if name.starts_with('X') {
            seen_jokers += 1;
        } else if !seen_naturals.insert(name) {
            bail!("{}: card {name} appears twice", request.id);
        }
    }
    let mut unseen: Vec<String> = super::all_cards()
        .into_iter()
        .filter(|n| !n.starts_with('X') && !seen_naturals.contains(n))
        .collect();
    for slot in 0..(2usize.saturating_sub(seen_jokers)) {
        unseen.push(format!("X{}", slot + 1));
    }
    if unseen.len() != 31 {
        bail!("{}: unseen is {}, wanted 31", request.id, unseen.len());
    }

    if let Some(opp_after) = &request.opp_after {
        let hero_actions = placements(&request.board, &request.draw);
        let values: Result<Vec<T3FirstHuAction>> = hero_actions
            .par_iter()
            .map(|(after, discard)| {
                let mut dead = request.dead.clone();
                dead.push(discard.clone());
                let v4 = V4FirstRequest {
                    id: format!("{}/{}", request.id, discard),
                    board: BoardStr {
                        top: after[0].clone(),
                        middle: after[1].clone(),
                        bottom: after[2].clone(),
                    },
                    dead,
                    draw: None,
                    opp_dead: Vec::new(),
                    opp_board: BoardStr {
                        top: opp_after[0].clone(),
                        middle: opp_after[1].clone(),
                        bottom: opp_after[2].clone(),
                    },
                };
                let value = v4_first::solve_maybe_collapsed(
                    &v4,
                    &[
                        fl_ev.value(14),
                        fl_ev.value(15),
                        fl_ev.value(16),
                        fl_ev.value(17),
                    ],
                    None,
                    rank_collapse,
                )?
                .value;
                Ok(T3FirstHuAction {
                    action_key: rows_key(after, discard),
                    value,
                })
            })
            .collect();
        let mut actions = values?;
        actions.sort_by(|a, b| a.action_key.cmp(&b.action_key));
        return Ok(T3FirstHuResponse {
            id: request.id.clone(),
            schema: "ofc_hu_t3_first_traced/v1",
            opp_draws: 1,
            actions,
        });
    }

    // The opponent's sampled draws, common across hero's actions.
    let stream = playout::root_stream(&request.id) ^ request.stream_offset;
    let triples = fl_solver::t2_labels::sampled_t3_draws(unseen.len(), request.opp_draws, stream);

    // The opponent's placement per draw is action-independent in *choice
    // inputs* except for hero's visible board, which does change with the
    // action.  So the chooser runs inside the action loop, but the draw
    // list stays shared.
    let hero_actions = placements(&request.board, &request.draw);

    let values: Result<Vec<T3FirstHuAction>> = hero_actions
        .par_iter()
        .map(|(after, discard)| {
            let mut dead = request.dead.clone();
            dead.push(discard.clone());
            let mut total = 0.0f64;
            let mut features: Vec<f32> = Vec::new();
            let mut scratch: Vec<f32> = Vec::new();
            let rowwise_memo: playout::RowwiseMemo =
                std::sync::Mutex::new(std::collections::HashMap::new());
            for (tick, triple) in triples.iter().enumerate() {
                let opp_draw: Vec<String> =
                    triple.iter().map(|k| unseen[*k].clone()).collect();
                // The opponent's chooser ranks its ~21 placements with the
                // generation-3 encoder; its pool approximates its own view.
                let opp_options = placements(&request.opp_board, &opp_draw);
                let opp_pool: Vec<super::Card> = unseen
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| !triple.contains(k))
                    .map(|(_, n)| to_core_card(n))
                    .collect::<Result<Vec<_>>>()?;
                let mut best_score = f32::NEG_INFINITY;
                let mut chosen: Option<&[Vec<String>; 3]> = None;
                for (candidate, _) in &opp_options {
                    let core = CoreBoard::from_str_board(&BoardStr {
                        top: candidate[0].clone(),
                        middle: candidate[1].clone(),
                        bottom: candidate[2].clone(),
                    })?;
                    playout::encode_for(
                        opp_chooser,
                        &core,
                        &opp_pool,
                        14,
                        fl_ev,
                        fl_table,
                        &rowwise_memo,
                        tick as u64,
                        &format!("t3f/{}/{}", request.id, tick),
                        &mut features,
                        None,
                    )?;
                    let score = opp_chooser.predict(&features, &mut scratch);
                    if score > best_score {
                        best_score = score;
                        chosen = Some(candidate);
                    }
                }
                let opp_after = chosen.expect("an opponent placement always exists");
                let v4 = V4FirstRequest {
                    id: format!("{}/{}", request.id, tick),
                    board: BoardStr {
                        top: after[0].clone(),
                        middle: after[1].clone(),
                        bottom: after[2].clone(),
                    },
                    dead: dead.clone(),
                    draw: None,
                    opp_dead: Vec::new(),
                    opp_board: BoardStr {
                        top: opp_after[0].clone(),
                        middle: opp_after[1].clone(),
                        bottom: opp_after[2].clone(),
                    },
                };
                total += v4_first::solve_maybe_collapsed(
                    &v4,
                    &[
                        fl_ev.value(14),
                        fl_ev.value(15),
                        fl_ev.value(16),
                        fl_ev.value(17),
                    ],
                    None,
                    rank_collapse,
                )?
                .value;
            }
            Ok(T3FirstHuAction {
                action_key: rows_key(after, discard),
                value: total / triples.len() as f64,
            })
        })
        .collect();

    let mut actions = values?;
    actions.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    Ok(T3FirstHuResponse {
        id: request.id.clone(),
        schema: "ofc_hu_t3_first/v1",
        opp_draws: triples.len(),
        actions,
    })
}
