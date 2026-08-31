//! T2 teacher labels against a Fantasyland opponent: chain playouts with the
//! learned T3 policy choosing moves and the FL board library scoring the end.
//!
//! A T2 action's value is the mean over sampled T3 draws of the value of the
//! T3 action the learned 109-dim evaluator would choose, where that chosen
//! action is then priced exactly like the T3 teacher labels: mean over T4
//! draws of the best completion's direct library score.  The model appears
//! only as a move chooser -- every number that enters the label comes from
//! the exact library scorer, which is the error-containment structure the
//! T3-vs-FL v1 failure taught.
//!
//! Playout T3 nodes are enumerated at the Card level (no string round-trip):
//! the two jokers are interchangeable in evaluation, so candidate placements
//! deduplicate on card identity, which also halves symmetric joker work.

use anyhow::{bail, Result};
use ofc_core::Card;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::evaluator;
use super::row_memo::TerminalMemo;
use super::t3_second;
use super::t3_vs_fl::sampled_draws;
use super::t3_vs_fl_lib::{
    card_bit, score_mean, terminal_key, FlLibrary, LibrarySet, MatchedRow, TerminalKey,
};
use super::{all_cards, apply, legal_actions, to_core_card, BoardStr, CoreBoard, FlEv};

#[derive(Deserialize)]
pub struct T2VsFlRequest {
    pub id: String,
    /// Hero board, 7 cards, six open slots.
    pub board: BoardStr,
    /// Hero's one prior discard (T1).
    pub dead: Vec<String>,
    /// Hero's three drawn cards.
    pub draw: Vec<String>,
    /// Opponent's Fantasyland card count (14..17), public information.
    pub opp_count: u8,
    /// T3 draws sampled per action.
    #[serde(default = "default_t3_samples")]
    pub t3_samples: usize,
    /// T4 draws sampled per playout; 0 enumerates all C(n,3).
    #[serde(default = "default_t4_draw_sample")]
    pub t4_draw_sample: usize,
}

fn default_t3_samples() -> usize {
    100
}

fn default_t4_draw_sample() -> usize {
    100
}

#[derive(Serialize)]
pub struct T2VsFlActionValue {
    pub action_key: String,
    pub value: f64,
    pub t3_samples: usize,
    pub mean_fl_samples: f64,
    /// Per-row completion outlook of the 9-card after-board over the unseen
    /// pool -- the teacher encoder's rowwise block (41 dims).  The light-lap
    /// T2 encoder is actor + rowwise + FL context; the four-open-slot joint
    /// block is deliberately deferred.
    pub own_rowwise_block: Vec<f32>,
}

#[derive(Serialize)]
pub struct T2VsFlResponse {
    pub id: String,
    pub schema: &'static str,
    pub leaf: &'static str,
    pub actions: Vec<T2VsFlActionValue>,
}

/// One candidate T3 placement at the Card level.
struct T3Candidate {
    placements: [(usize, Card); 2],
    discard_index: usize,
}

fn t3_candidates(board: &CoreBoard, draw: &[Card; 3]) -> Vec<T3Candidate> {
    let open = board.open_slots();
    let mut out: Vec<T3Candidate> = Vec::new();
    let mut seen: std::collections::BTreeSet<(u32, u32, u32)> = std::collections::BTreeSet::new();
    let id = |card: &Card| -> u32 {
        if card.is_joker() {
            52
        } else {
            card.suit as u32 * 13 + card.rank as u32 - 2
        }
    };
    for discard_index in 0..3usize {
        let kept: Vec<usize> = (0..3).filter(|index| *index != discard_index).collect();
        for row_a in 0..3usize {
            for row_b in 0..3usize {
                let mut need = [0usize; 3];
                need[row_a] += 1;
                need[row_b] += 1;
                if (0..3).any(|row| need[row] > open[row]) {
                    continue;
                }
                let mut pair = [
                    (row_a as u32) << 8 | id(&draw[kept[0]]),
                    (row_b as u32) << 8 | id(&draw[kept[1]]),
                ];
                pair.sort_unstable();
                if seen.insert((pair[0], pair[1], id(&draw[discard_index]))) {
                    out.push(T3Candidate {
                        placements: [(row_a, draw[kept[0]]), (row_b, draw[kept[1]])],
                        discard_index,
                    });
                }
            }
        }
    }
    out
}

/// Exact T4 continuation value of an 11-card board: mean over T4 draws of the
/// best completion's library score.  Mirrors the T3 teacher's per-action
/// pricing (t3_vs_fl_lib), on an already-filtered shortlist.
#[allow(clippy::too_many_arguments)]
fn t4_library_value(
    board: &CoreBoard,
    unseen: &[Card],
    shortlist: &[u32],
    library: &FlLibrary,
    opp_count: u8,
    fl_ev: &FlEv,
    draw_sample: usize,
    seed: &str,
) -> Result<(f64, f64)> {
    let draw_sets = sampled_draws(unseen.len(), draw_sample, seed);
    let patterns = t3_second::placement_patterns(board);
    let mut terminal_memo = TerminalMemo::new(board);
    let mut memo: std::collections::HashMap<TerminalKey, f64> =
        std::collections::HashMap::with_capacity(patterns.len());
    let mut matched: Vec<MatchedRow> = Vec::with_capacity(shortlist.len());
    let mut total = 0.0f64;
    let mut sample_total = 0usize;
    for draw in &draw_sets {
        let draw_cards = [unseen[draw[0]], unseen[draw[1]], unseen[draw[2]]];
        let mut draw_mask = 0u64;
        for card in &draw_cards {
            draw_mask |= card_bit(card)?;
        }
        matched.clear();
        for &index in shortlist {
            let index = index as usize;
            if library.masks[index] & draw_mask == 0 {
                matched.push(MatchedRow {
                    values: library.values[index],
                    royalty: library.royalty[index],
                    stay: library.stay[index],
                    busted: library.busted[index],
                });
            }
        }
        if matched.is_empty() {
            continue;
        }
        sample_total += matched.len();
        let mut best = f64::NEG_INFINITY;
        memo.clear();
        for pattern in &patterns {
            let terminal = terminal_memo.terminal(&[
                (pattern.rows[0], draw_cards[pattern.cards[0]]),
                (pattern.rows[1], draw_cards[pattern.cards[1]]),
            ]);
            let key = terminal_key(&terminal);
            let value = match memo.get(&key) {
                Some(cached) => *cached,
                None => {
                    let computed = score_mean(&terminal, &matched, opp_count, fl_ev);
                    memo.insert(key, computed);
                    computed
                }
            };
            if value > best {
                best = value;
            }
        }
        total += best;
    }
    let draws = draw_sets.len().max(1);
    Ok((total / draws as f64, sample_total as f64 / draws as f64))
}

pub fn solve(
    request: &T2VsFlRequest,
    fl_ev: &FlEv,
    libraries: &LibrarySet,
    fl_table: &evaluator::FlTable,
    t3_model: &evaluator::Model,
) -> Result<T2VsFlResponse> {
    let library = libraries.for_count(request.opp_count);
    let base = CoreBoard::from_str_board(&request.board)?;
    if base.card_count() != 7 {
        bail!("T2-vs-FL needs a 7-card hero board");
    }
    if request.draw.len() != 3 || request.dead.len() != 1 {
        bail!("T2-vs-FL needs a 3-card draw and one prior discard");
    }
    if !(14..=17).contains(&request.opp_count) {
        bail!("opp_count must be 14..17");
    }

    // Seen set is action-independent: board + draw + dead.
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
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
    let unseen_t2: Vec<Card> = all_cards()
        .into_iter()
        .filter(|card| !seen.contains(card))
        .map(|card| to_core_card(&card))
        .collect::<Result<Vec<_>>>()?;
    let mut seen_mask = 0u64;
    for name in &seen {
        seen_mask |= card_bit(&to_core_card(name)?)?;
    }
    let hero_jokers = seen
        .iter()
        .filter(|name| *name == "X1" || *name == "X2")
        .count();
    if hero_jokers == 1 {
        // One joker seen: the other stays available (see t3_vs_fl_lib).
        seen_mask &= !(1u64 << 53);
    }

    let actions = legal_actions(&base, &request.draw);
    let values: Result<Vec<T2VsFlActionValue>> = actions
        .par_iter()
        .map(|action| {
            let after = apply(&base, action)?;
            // Common random numbers: the unseen pool is action-independent,
            // so every action prices the SAME sampled T3 draws.  Action-value
            // differences then share the draw noise, which is what the
            // teacher's argmax and regret actually consume.
            let t3_triples = sampled_draws(
                unseen_t2.len(),
                request.t3_samples,
                &format!("t2/{}", request.id),
            );

            let mut features: Vec<f32> = Vec::with_capacity(109);
            let mut scratch: Vec<f32> = Vec::new();
            let mut total = 0.0f64;
            let mut fl_samples = 0.0f64;
            let mut playouts = 0usize;
            for (triple_index, triple) in t3_triples.iter().enumerate() {
                let t3_draw = [
                    unseen_t2[triple[0]],
                    unseen_t2[triple[1]],
                    unseen_t2[triple[2]],
                ];
                let mut t3_mask = 0u64;
                for card in &t3_draw {
                    t3_mask |= card_bit(card)?;
                }
                let unseen_t3: Vec<Card> = unseen_t2
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !triple.contains(index))
                    .map(|(_, card)| *card)
                    .collect();

                // The learned T3 policy chooses among candidate placements.
                // The model's own input width selects the encoder: 109 is
                // the full evaluator (actor + rowwise + joint + context),
                // 60 is the light actor+context policy whose only job is
                // move choice -- the distillation experiment's fast path.
                let mut best_score = f32::NEG_INFINITY;
                let mut chosen: Option<CoreBoard> = None;
                for candidate in t3_candidates(&after, &t3_draw) {
                    let mut t3_after = after.clone();
                    t3_after.rows[candidate.placements[0].0].push(candidate.placements[0].1);
                    t3_after.rows[candidate.placements[1].0].push(candidate.placements[1].1);
                    let _ = candidate.discard_index;
                    features.clear();
                    evaluator::actor_block(&t3_after.rows, &mut features);
                    if t3_model.input_dim == 109 {
                        let _categories = evaluator::opponent_rowwise_block(
                            &t3_after.rows,
                            &unseen_t3,
                            fl_table,
                            &mut features,
                        );
                        for value in t3_second::joint_block(&t3_after, &unseen_t3, fl_ev)? {
                            features.push(value as f32);
                        }
                    }
                    super::t3_vs_fl::fl_context(
                        &unseen_t3,
                        request.opp_count,
                        fl_ev,
                        &mut features,
                    );
                    if features.len() != t3_model.input_dim {
                        bail!("t2 playout feature width drifted: {}", features.len());
                    }
                    let predicted = t3_model.predict(&features, &mut scratch);
                    if predicted > best_score {
                        best_score = predicted;
                        chosen = Some(t3_after);
                    }
                }
                let Some(t3_board) = chosen else { continue };

                // Price the chosen action exactly, like the T3 teacher does.
                let mut playout_seen = seen_mask | t3_mask;
                let drawn_jokers = t3_draw.iter().filter(|card| card.is_joker()).count();
                if hero_jokers + drawn_jokers == 1 {
                    // Exactly one joker is seen: the other stays available
                    // (same single-joker approximation as t3_vs_fl_lib).
                    playout_seen &= !(1u64 << 53);
                }
                let shortlist: Vec<u32> = (0..library.len() as u32)
                    .filter(|&index| library.masks[index as usize] & playout_seen == 0)
                    .collect();
                let (value, samples) = t4_library_value(
                    &t3_board,
                    &unseen_t3,
                    &shortlist,
                    library,
                    request.opp_count,
                    fl_ev,
                    request.t4_draw_sample,
                    &format!("t2/{}/{}/{}", request.id, action.key(), triple_index),
                )?;
                total += value;
                fl_samples += samples;
                playouts += 1;
            }
            let effective = playouts.max(1);
            let mut rowwise: Vec<f32> = Vec::with_capacity(evaluator::OPPONENT_SIZE);
            let _categories =
                evaluator::opponent_rowwise_block(&after.rows, &unseen_t2, fl_table, &mut rowwise);
            Ok(T2VsFlActionValue {
                action_key: action.key(),
                value: total / effective as f64,
                t3_samples: playouts,
                mean_fl_samples: fl_samples / effective as f64,
                own_rowwise_block: rowwise,
            })
        })
        .collect();

    let mut actions_out = values?;
    actions_out.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    Ok(T2VsFlResponse {
        id: request.id.clone(),
        schema: "ofc_t2_vs_fl_value_playout/v1",
        leaf: "t3_model_moves_library_scoring",
        actions: actions_out,
    })
}
