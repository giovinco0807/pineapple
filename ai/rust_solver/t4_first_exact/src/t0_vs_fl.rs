//! T0 teacher labels against a Fantasyland opponent: the initial five-card
//! placement, the street the legacy serving policy is weakest at.
//!
//! A T0 action assigns all five dealt cards to rows (no discard, no draw);
//! everything after that is the shared chain playout -- T1/T2 evaluators and
//! the light T3 policy choose moves, and the 11-card terminal is priced
//! exactly against whatever opponents the run was handed.
//!
//! Candidate enumeration dedupes on card identity like every other street,
//! which also collapses the interchangeable jokers.  Placements that
//! guarantee a foul are NOT pruned here: the playout prices them honestly,
//! and the teacher needs those values to teach the evaluator why they lose.

use anyhow::{bail, Result};
use ofc_core::Card;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::evaluator;
use super::playout;
use super::t3_vs_fl_lib::card_bit;
use super::{to_core_card, CoreBoard, FlEv};

#[derive(Deserialize)]
pub struct T0VsFlRequest {
    pub id: String,
    /// The five dealt cards; the board starts empty.
    pub cards: Vec<String>,
    /// Opponent's Fantasyland card count (14..17), public information.
    pub opp_count: u8,
    #[serde(default = "default_t1_samples")]
    pub t1_samples: usize,
    #[serde(default = "default_t2_samples")]
    pub t2_samples: usize,
    #[serde(default = "default_t3_samples")]
    pub t3_samples: usize,
    /// T4 draws sampled per terminal; 0 enumerates all C(n,3).
    #[serde(default = "default_t4_draw_sample")]
    pub t4_draw_sample: usize,
    /// Depth at which the chooser's value stands in for the line.
    /// Absent plays every line out; see `playout::Context`.
    #[serde(default)]
    pub truncate_depth: Option<usize>,
    /// Opponents drawn from the pool for this root, shared by every action.
    /// Ignored on the library path, which filters the whole shelf per leaf
    /// instead of sampling it.
    #[serde(default = "default_pool_opponents")]
    pub pool_opponents: usize,
}

fn default_t1_samples() -> usize {
    8
}

fn default_t2_samples() -> usize {
    6
}

fn default_t3_samples() -> usize {
    4
}

fn default_t4_draw_sample() -> usize {
    40
}

/// Attrition is worse here than at T1 -- see the note on the T1 default.  A T0
/// root conditions the draw on five cards and the playout reveals twelve more
/// before the terminal, so about 1% of the drawn entries reach the leaf (4.1 of
/// 400, measured).  60 prices a T0 leaf against well under one opponent.
fn default_pool_opponents() -> usize {
    60
}

#[derive(Serialize)]
pub struct T0VsFlActionValue {
    /// Rows as sorted card names, top/middle/bottom -- the action identity.
    pub action_key: String,
    pub value: f64,
    pub lines: usize,
    pub mean_fl_samples: f64,
    /// Per-row completion outlook of the 5-card placement (41 dims).
    pub own_rowwise_block: Vec<f32>,
}

#[derive(Serialize)]
pub struct T0VsFlResponse {
    pub id: String,
    pub schema: &'static str,
    pub leaf: &'static str,
    /// Which opponents the pool draw landed on, so a label says what it was
    /// priced against rather than leaving it to be re-derived.  Absent on the
    /// library path, which does not draw.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opponent_stream: Option<u64>,
    pub actions: Vec<T0VsFlActionValue>,
}

fn card_id(card: &Card) -> u64 {
    if card.is_joker() {
        52
    } else {
        card.suit as u64 * 13 + card.rank as u64 - 2
    }
}

/// Distinct assignments of the five cards to rows within capacity.
/// Identity-keyed: two assignments that differ only by swapping equal cards
/// (the jokers) collapse to one.
fn t0_candidates(cards: &[Card; 5]) -> Vec<[usize; 5]> {
    let mut out: Vec<[usize; 5]> = Vec::new();
    let mut seen: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for code in 0..3usize.pow(5) {
        let mut rows = [0usize; 5];
        let mut value = code;
        let mut counts = [0usize; 3];
        for slot in 0..5 {
            rows[slot] = value % 3;
            value /= 3;
            counts[rows[slot]] += 1;
        }
        if counts[0] > 3 {
            continue;
        }
        // Sorted (row, card-id) pairs form the action identity.
        let mut pairs: Vec<u64> = (0..5)
            .map(|slot| (rows[slot] as u64) << 6 | card_id(&cards[slot]))
            .collect();
        pairs.sort_unstable();
        let key = pairs.iter().fold(0u64, |accum, pair| accum * 199 + pair + 1);
        if seen.insert(key) {
            out.push(rows);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn solve(
    request: &T0VsFlRequest,
    fl_ev: &FlEv,
    source: &playout::OpponentSource<'_>,
    fl_table: &evaluator::FlTable,
    t1_model: &evaluator::Model,
    t2_model: &evaluator::Model,
    t3_model: &evaluator::Model,
) -> Result<T0VsFlResponse> {
    if request.cards.len() != 5 {
        bail!("T0-vs-FL needs exactly five dealt cards");
    }
    if !(14..=17).contains(&request.opp_count) {
        bail!("opp_count must be 14..17");
    }
    let mut dealt = [Card { rank: 0, suit: 0 }; 5];
    for (slot, name) in request.cards.iter().enumerate() {
        dealt[slot] = to_core_card(name)?;
    }

    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for card in &request.cards {
        seen.insert(card.clone());
    }
    let unseen_root: Vec<Card> = super::all_cards()
        .into_iter()
        .filter(|card| !seen.contains(card))
        .map(|card| to_core_card(&card))
        .collect::<Result<Vec<_>>>()?;
    let mut root_mask = 0u64;
    for card in &dealt {
        root_mask |= card_bit(card)?;
    }
    let root_jokers = dealt.iter().filter(|card| card.is_joker()).count();

    let stream = playout::root_stream(&request.id);
    let drawn = match source {
        playout::OpponentSource::Libraries(_) => Vec::new(),
        playout::OpponentSource::Pool(pool) => {
            if u32::from(request.opp_count) != pool.width {
                bail!(
                    "request faces a {}-card Fantasyland opponent but the pool \
                     solved width {}",
                    request.opp_count,
                    pool.width
                );
            }
            playout::root_opponents(pool, &dealt, request.pool_opponents, stream)?
        }
    };
    let (opponents, leaf) = match source {
        playout::OpponentSource::Libraries(libraries) => (
            playout::Opponents::Library(libraries.for_count(request.opp_count)),
            "t1_t2_t3_models_move_library_scoring",
        ),
        playout::OpponentSource::Pool(_) => (
            playout::Opponents::Pool(&drawn),
            "t1_t2_t3_models_move_pool_best_response",
        ),
    };

    let models = [t1_model, t2_model, t3_model];
    let samples = [request.t1_samples, request.t2_samples, request.t3_samples];
    let context = playout::Context {
        fl_ev,
        opponents,
        fl_table,
        models: &models,
        samples: &samples,
        opp_count: request.opp_count,
        t4_draw_sample: request.t4_draw_sample,
        rowwise_memo: std::sync::Mutex::new(std::collections::HashMap::new()),
        truncate_depth: request.truncate_depth,
    };

    let candidates = t0_candidates(&dealt);
    let values: Result<Vec<T0VsFlActionValue>> = candidates
        .par_iter()
        .map(|assignment| {
            let mut board = CoreBoard {
                rows: [Vec::new(), Vec::new(), Vec::new()],
            };
            for slot in 0..5 {
                board.rows[assignment[slot]].push(dealt[slot]);
            }
            let mut features: Vec<f32> = Vec::new();
            let mut scratch: Vec<f32> = Vec::new();
            // Common random numbers: the seed omits the action, so every
            // placement prices the same sampled continuations.
            let (value, fl_samples) = playout::descend(
                &board,
                &unseen_root,
                root_mask,
                root_jokers,
                &context,
                0,
                &format!("t0/{}", request.id),
                &mut features,
                &mut scratch,
            )?;
            let mut rowwise: Vec<f32> = Vec::with_capacity(evaluator::OPPONENT_SIZE);
            let _categories = evaluator::opponent_rowwise_block(
                &board.rows,
                &unseen_root,
                fl_table,
                &mut rowwise,
            );
            let mut key_rows: Vec<String> = Vec::with_capacity(3);
            for row in 0..3 {
                let mut names: Vec<&String> = (0..5)
                    .filter(|slot| assignment[*slot] == row)
                    .map(|slot| &request.cards[slot])
                    .collect();
                names.sort();
                key_rows.push(
                    names.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(","),
                );
            }
            Ok(T0VsFlActionValue {
                action_key: key_rows.join("|"),
                value,
                lines: request.t1_samples,
                mean_fl_samples: fl_samples,
                own_rowwise_block: rowwise,
            })
        })
        .collect();

    let mut actions_out = values?;
    actions_out.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    Ok(T0VsFlResponse {
        id: request.id.clone(),
        schema: "ofc_t0_vs_fl_value_playout/v1",
        leaf,
        opponent_stream: match source {
            playout::OpponentSource::Libraries(_) => None,
            playout::OpponentSource::Pool(_) => Some(stream),
        },
        actions: actions_out,
    })
}
