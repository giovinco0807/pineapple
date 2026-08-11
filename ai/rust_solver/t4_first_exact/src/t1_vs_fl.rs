//! T1 teacher labels against a Fantasyland opponent.
//!
//! Identical in structure to the T2 labeler, one street deeper: the hero acts
//! on a 5-card board, the learned T2 evaluator chooses the next street's
//! placement, the T3 policy the one after, and the T4 terminal is priced
//! exactly against the FL board library.  All of that lives in `playout`; the
//! only work here is the request shape, the root's seen set, and the encoder
//! block the T1 teacher stores alongside each action value.

use anyhow::{bail, Result};
use ofc_core::Card;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::evaluator;
use super::playout;
use super::t3_vs_fl::sampled_draws;
use super::t3_vs_fl_lib::{card_bit, LibrarySet};
use super::{all_cards, apply, legal_actions, to_core_card, BoardStr, CoreBoard, FlEv};

#[derive(Deserialize)]
pub struct T1VsFlRequest {
    pub id: String,
    /// Hero board, 5 cards, eight open slots.
    pub board: BoardStr,
    /// Hero's prior discards (none at T1; kept for shape symmetry).
    #[serde(default)]
    pub dead: Vec<String>,
    /// Hero's three drawn cards.
    pub draw: Vec<String>,
    /// Opponent's Fantasyland card count (14..17), public information.
    pub opp_count: u8,
    /// T2 draws sampled per action.
    #[serde(default = "default_t2_samples")]
    pub t2_samples: usize,
    /// T3 draws sampled per T2 line.
    #[serde(default = "default_t3_samples")]
    pub t3_samples: usize,
    /// T4 draws sampled per terminal; 0 enumerates all C(n,3).
    #[serde(default = "default_t4_draw_sample")]
    pub t4_draw_sample: usize,
}

fn default_t2_samples() -> usize {
    20
}

fn default_t3_samples() -> usize {
    10
}

fn default_t4_draw_sample() -> usize {
    60
}

#[derive(Serialize)]
pub struct T1VsFlActionValue {
    pub action_key: String,
    pub value: f64,
    pub lines: usize,
    pub mean_fl_samples: f64,
    /// Per-row completion outlook of the 7-card after-board (41 dims), the
    /// rowwise half of the light-lap T1 encoder.
    pub own_rowwise_block: Vec<f32>,
}

#[derive(Serialize)]
pub struct T1VsFlResponse {
    pub id: String,
    pub schema: &'static str,
    pub leaf: &'static str,
    pub actions: Vec<T1VsFlActionValue>,
}

pub fn solve(
    request: &T1VsFlRequest,
    fl_ev: &FlEv,
    libraries: &LibrarySet,
    fl_table: &evaluator::FlTable,
    t2_model: &evaluator::Model,
    t3_model: &evaluator::Model,
) -> Result<T1VsFlResponse> {
    let library = libraries.for_count(request.opp_count);
    let base = CoreBoard::from_str_board(&request.board)?;
    if base.card_count() != 5 {
        bail!("T1-vs-FL needs a 5-card hero board");
    }
    if request.draw.len() != 3 {
        bail!("T1-vs-FL needs a 3-card draw");
    }
    if !(14..=17).contains(&request.opp_count) {
        bail!("opp_count must be 14..17");
    }

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
    let unseen_root: Vec<Card> = all_cards()
        .into_iter()
        .filter(|card| !seen.contains(card))
        .map(|card| to_core_card(&card))
        .collect::<Result<Vec<_>>>()?;
    let mut root_mask = 0u64;
    for name in &seen {
        root_mask |= card_bit(&to_core_card(name)?)?;
    }
    let root_jokers = seen
        .iter()
        .filter(|name| *name == "X1" || *name == "X2")
        .count();

    let models = [t2_model, t3_model];
    let samples = [request.t2_samples, request.t3_samples];
    let context = playout::Context {
        fl_ev,
        opponents: playout::Opponents::Library(library),
        fl_table,
        models: &models,
        samples: &samples,
        opp_count: request.opp_count,
        t4_draw_sample: request.t4_draw_sample,
        rowwise_memo: std::sync::Mutex::new(std::collections::HashMap::new()),
    };

    let actions = legal_actions(&base, &request.draw);
    let values: Result<Vec<T1VsFlActionValue>> = actions
        .par_iter()
        .map(|action| {
            let after = apply(&base, action)?;
            let mut features: Vec<f32> = Vec::new();
            let mut scratch: Vec<f32> = Vec::new();
            // Common random numbers across actions: the seed omits the action
            // key, so every action prices the same sampled continuations and
            // the differences the teacher consumes share their draw noise.
            let (value, fl_samples) = playout::descend(
                &after,
                &unseen_root,
                root_mask,
                root_jokers,
                &context,
                0,
                &format!("t1/{}", request.id),
                &mut features,
                &mut scratch,
            )?;
            let mut rowwise: Vec<f32> = Vec::with_capacity(evaluator::OPPONENT_SIZE);
            let _categories = evaluator::opponent_rowwise_block(
                &after.rows,
                &unseen_root,
                fl_table,
                &mut rowwise,
            );
            Ok(T1VsFlActionValue {
                action_key: action.key(),
                value,
                lines: sampled_draws(
                    unseen_root.len(),
                    request.t2_samples,
                    &format!("t1/{}", request.id),
                )
                .len(),
                mean_fl_samples: fl_samples,
                own_rowwise_block: rowwise,
            })
        })
        .collect();

    let mut actions_out = values?;
    actions_out.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    Ok(T1VsFlResponse {
        id: request.id.clone(),
        schema: "ofc_t1_vs_fl_value_playout/v1",
        leaf: "t2_t3_models_move_library_scoring",
        actions: actions_out,
    })
}
