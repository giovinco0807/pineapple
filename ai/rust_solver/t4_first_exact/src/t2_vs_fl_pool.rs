//! T2 teacher labels against a **best-responding** Fantasyland opponent, with
//! the T3 street handled by a learned model instead of enumerated.
//!
//! This exists to be compared against `fl_solver teach-t2`, which enumerates
//! every T3 placement and prices each one, and it deliberately offers both
//! ways of using the model so the comparison can say which is which:
//!
//! * `truncate_depth: Some(0)` — the T3 evaluator's own best score **is** the
//!   line's value.  Cheap, and it puts a model's number inside the label.
//!   This is the shape that failed in T3-vs-FL v1, where a learned T4 leaf of
//!   MAE 0.645 poisoned T3 labels because its per-board errors were correlated
//!   enough across a root's draws to survive a 300-draw average.
//! * `truncate_depth: None` — the model only *chooses* the T3 placement and
//!   the chosen line is played to an eleven-card board and priced against the
//!   drawn pool entries, so every number entering the label is exact.
//!
//! Neither is asserted to be right here.  `t2_vs_fl.rs` is the sibling that
//! scores against the static FL library; that opponent model was invalidated
//! by the 2026-08-05 rule correction, which is why this module exists rather
//! than a flag on that one.

use anyhow::{bail, Result};
use ofc_core::Card;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::evaluator;
use super::playout;
use super::t3_vs_fl::sampled_draws;
use super::t3_vs_fl_lib::card_bit;
use super::{all_cards, apply, legal_actions, to_core_card, BoardStr, CoreBoard, FlEv};

#[derive(Deserialize)]
pub struct T2VsFlPoolRequest {
    pub id: String,
    /// Hero board, 7 cards, six open slots.
    pub board: BoardStr,
    /// Hero's prior discard.
    #[serde(default)]
    pub dead: Vec<String>,
    /// Hero's three drawn cards.
    pub draw: Vec<String>,
    /// Opponent's Fantasyland card count (14..17), public information.
    pub opp_count: u8,
    /// T3 draws sampled per action.
    #[serde(default = "default_t3_samples")]
    pub t3_samples: usize,
    /// T4 draws sampled per terminal; 0 enumerates all C(n,3).
    #[serde(default = "default_t4_draw_sample")]
    pub t4_draw_sample: usize,
    /// See the module note: `Some(0)` reads the T3 model's value, absent plays
    /// the model's chosen line out and prices it.
    #[serde(default)]
    pub truncate_depth: Option<usize>,
    /// Price terminals by hero's own worth; see `playout::Context::own_only`.
    #[serde(default)]
    pub own_only: bool,
    /// Opponents drawn from the pool for this root, shared by every action.
    #[serde(default = "default_pool_opponents")]
    pub pool_opponents: usize,
}

fn default_t3_samples() -> usize {
    48
}

fn default_t4_draw_sample() -> usize {
    24
}

fn default_pool_opponents() -> usize {
    240
}

#[derive(Serialize)]
pub struct T2VsFlPoolActionValue {
    pub action_key: String,
    pub value: f64,
    pub lines: usize,
    pub mean_fl_samples: f64,
}

#[derive(Serialize)]
pub struct T2VsFlPoolResponse {
    pub id: String,
    pub schema: &'static str,
    pub leaf: &'static str,
    pub opponents: usize,
    pub t3_draws: usize,
    pub opponent_stream: u64,
    pub actions: Vec<T2VsFlPoolActionValue>,
}

pub fn solve(
    request: &T2VsFlPoolRequest,
    fl_ev: &FlEv,
    pool: &fl_solver::pool::Pool,
    fl_table: &evaluator::FlTable,
    t3_model: &evaluator::Model,
) -> Result<T2VsFlPoolResponse> {
    let base = CoreBoard::from_str_board(&request.board)?;
    if base.card_count() != 7 {
        bail!("T2-vs-FL needs a 7-card hero board");
    }
    if request.draw.len() != 3 {
        bail!("T2-vs-FL needs a 3-card draw");
    }
    if !(14..=17).contains(&request.opp_count) {
        bail!("opp_count must be 14..17");
    }
    if u32::from(request.opp_count) != pool.width {
        bail!(
            "request faces a {}-card Fantasyland opponent but the pool solved \
             width {}",
            request.opp_count,
            pool.width
        );
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
    let seen_root: Vec<Card> = seen
        .iter()
        .map(|name| to_core_card(name))
        .collect::<Result<Vec<_>>>()?;
    let mut root_mask = 0u64;
    for card in &seen_root {
        root_mask |= card_bit(card)?;
    }
    let root_jokers = seen
        .iter()
        .filter(|name| *name == "X1" || *name == "X2")
        .count();

    // The same derivation `fl_solver teach-t2` uses for its own root stream,
    // so a comparison of the two teachers can be told which part of a
    // difference is the opponents.
    let stream = playout::root_stream(&request.id);
    let drawn = playout::root_opponents(pool, &seen_root, request.pool_opponents, stream)?;

    // One model: from a T2 root the only street below that chooses is T3.
    let models = [t3_model];
    let samples = [request.t3_samples];
    let context = playout::Context {
        fl_ev,
        opponents: playout::Opponents::Pool(&drawn),
        fl_table,
        models: &models,
        samples: &samples,
        opp_count: request.opp_count,
        t4_draw_sample: request.t4_draw_sample,
        rowwise_memo: std::sync::Mutex::new(std::collections::HashMap::new()),
        truncate_depth: request.truncate_depth,
        own_only: request.own_only,
        t2_fence: None,
    };

    let actions = legal_actions(&base, &request.draw);
    let values: Result<Vec<T2VsFlPoolActionValue>> = actions
        .par_iter()
        .map(|action| {
            let after = apply(&base, action)?;
            let mut features: Vec<f32> = Vec::new();
            let mut scratch: Vec<f32> = Vec::new();
            // Common random numbers across actions: the seed omits the action
            // key, so every action prices the same sampled continuations.
            let (value, fl_samples) = playout::descend(
                &after,
                &unseen_root,
                root_mask,
                root_jokers,
                &context,
                0,
                &format!("t2/{}", request.id),
                &mut features,
                &mut scratch,
            )?;
            Ok(T2VsFlPoolActionValue {
                action_key: action.key(),
                value,
                lines: sampled_draws(
                    unseen_root.len(),
                    request.t3_samples,
                    &format!("t2/{}", request.id),
                )
                .len(),
                mean_fl_samples: fl_samples,
            })
        })
        .collect();

    let mut actions_out = values?;
    actions_out.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    Ok(T2VsFlPoolResponse {
        id: request.id.clone(),
        schema: "ofc_t2_vs_fl_pool_playout/v1",
        leaf: match request.truncate_depth {
            Some(_) => "t3_model_value",
            None => "t3_model_move_pool_best_response",
        },
        opponents: drawn.len(),
        t3_draws: request.t3_samples,
        opponent_stream: stream,
        actions: actions_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const POOL: &str = "D:/ofc_data/fl_pools/fl14_v1.jfl1";
    const T3_MODEL: &str = "D:/ofc_data/lap3_t3_teacher/lap3_t3_model_110_wide/evaluator.bin";
    const CONFIG: &str = "../../config/fl_ev.json";

    fn request(truncate_depth: Option<usize>) -> T2VsFlPoolRequest {
        T2VsFlPoolRequest {
            id: "t2-pool-plumbing".to_string(),
            board: BoardStr {
                top: vec!["Ks".into()],
                middle: vec!["Qs".into(), "Qh".into()],
                bottom: vec!["As".into(), "Ah".into(), "2s".into(), "3s".into()],
            },
            dead: vec!["4d".into()],
            draw: vec!["7d".into(), "3c".into(), "9c".into()],
            opp_count: 14,
            t3_samples: 3,
            t4_draw_sample: 8,
            truncate_depth,
            own_only: false,
            pool_opponents: 60,
        }
    }

    /// **The two leaves are different quantities.**
    ///
    /// Reading the T3 model's value and playing its choice out to a priced
    /// terminal must not agree: if they did, either the descent ignored
    /// `truncate_depth` or the pool never reached the leaf, and both failures
    /// are silent.  `mean_fl_samples` separates them -- the truncated arm
    /// never reaches an opponent, so its own report of how many it averaged
    /// over is zero.
    #[test]
    fn the_truncated_leaf_and_the_priced_leaf_disagree() {
        for path in [POOL, T3_MODEL] {
            if !Path::new(path).exists() {
                eprintln!("skipping: {path} is not on this box");
                return;
            }
        }
        let fl_ev = FlEv::load(Path::new(CONFIG)).expect("fl_ev config");
        let table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let bytes = std::fs::read(POOL).expect("pool");
        let pool = fl_solver::pool::deserialize(
            &bytes,
            14,
            [
                fl_ev.value(14),
                fl_ev.value(15),
                fl_ev.value(16),
                fl_ev.value(17),
            ],
        )
        .expect("pool loads");
        let image = std::fs::read(T3_MODEL).expect("model");
        let model = evaluator::Model::load(&image).expect("model loads");

        let truncated = solve(&request(Some(0)), &fl_ev, &pool, &table, &model).expect("truncated");
        let priced = solve(&request(None), &fl_ev, &pool, &table, &model).expect("priced");

        assert_eq!(truncated.actions.len(), priced.actions.len());
        assert!(
            truncated.actions.iter().all(|a| a.mean_fl_samples == 0.0),
            "a truncated line reported opponents it never reached"
        );
        assert!(
            priced.actions.iter().any(|a| a.mean_fl_samples > 0.0),
            "the priced leaf never reached an opponent"
        );
        assert!(
            truncated
                .actions
                .iter()
                .zip(&priced.actions)
                .any(|(a, b)| (a.value - b.value).abs() > 1e-9),
            "the model's value and the priced continuation agreed exactly"
        );
    }
}
