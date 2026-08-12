//! T1 teacher labels against a Fantasyland opponent.
//!
//! Identical in structure to the T2 labeler, one street deeper: the hero acts
//! on a 5-card board, the learned T2 evaluator chooses the next street's
//! placement, the T3 policy the one after, and the T4 terminal is priced
//! exactly against whatever opponents the run was handed.  All of that lives
//! in `playout`; the only work here is the request shape, the root's seen set,
//! the one pool draw the whole fan-out shares, and the encoder block the T1
//! teacher stores alongside each action value.

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

fn default_t2_samples() -> usize {
    20
}

fn default_t3_samples() -> usize {
    10
}

fn default_t4_draw_sample() -> usize {
    60
}

/// **60 is not enough at T1.**  The draw conditions on the eight cards hero
/// sees at the root, but the playout reveals nine more before the terminal and
/// three more per T4 draw, and the leaf drops every opponent that collides:
/// measured 2.9% of the drawn entries survive to be scored (1.0 of 60, 12.2 of
/// 400, 36.7 of 1200, 117 of 4000 on one root).  At 60 a leaf is a one-opponent
/// average and the value is not converged -- that root moved from -4.65 at 60
/// to -11.42 at 400 and then stayed put.  The default matches the sibling
/// teachers' `--opponents`; a T1 run that means it passes several hundred.
fn default_pool_opponents() -> usize {
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
    /// Which opponents the pool draw landed on, so a label says what it was
    /// priced against rather than leaving it to be re-derived.  Absent on the
    /// library path, which does not draw.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opponent_stream: Option<u64>,
    pub actions: Vec<T1VsFlActionValue>,
}

pub fn solve(
    request: &T1VsFlRequest,
    fl_ev: &FlEv,
    source: &playout::OpponentSource<'_>,
    fl_table: &evaluator::FlTable,
    t2_model: &evaluator::Model,
    t3_model: &evaluator::Model,
) -> Result<T1VsFlResponse> {
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
            playout::root_opponents(pool, &seen_root, request.pool_opponents, stream)?
        }
    };
    let (opponents, leaf) = match source {
        playout::OpponentSource::Libraries(libraries) => (
            playout::Opponents::Library(libraries.for_count(request.opp_count)),
            "t2_t3_models_move_library_scoring",
        ),
        playout::OpponentSource::Pool(_) => (
            playout::Opponents::Pool(&drawn),
            "t2_t3_models_move_pool_best_response",
        ),
    };

    let models = [t2_model, t3_model];
    let samples = [request.t2_samples, request.t3_samples];
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
        leaf,
        opponent_stream: match source {
            playout::OpponentSource::Libraries(_) => None,
            playout::OpponentSource::Pool(_) => Some(stream),
        },
        actions: actions_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::t3_vs_fl_lib::LibrarySet;
    use std::path::{Path, PathBuf};

    /// The shipped artefacts.  Nothing here is synthesised: the point of this
    /// test is that the labeler reaches the real 200,000-entry pool, and a
    /// fixture pool would prove the leaf's arithmetic all over again (which
    /// `playout::tests` already does) rather than the wiring.
    const POOL: &str = "D:/ofc_data/fl_pools/fl14_v1.jfl1";
    const LIBRARY: &str = "D:/ofc_data/fl_library_14";
    const T2_MODEL: &str = "D:/ofc_data/t2_vs_fl_model_v1/evaluator.bin";
    const T3_MODEL: &str = "D:/ofc_data/t3_vs_fl_model_v2/evaluator.bin";
    const CONFIG: &str = "../../config/fl_ev.json";

    fn request() -> T1VsFlRequest {
        T1VsFlRequest {
            id: "pool-plumbing".to_string(),
            board: BoardStr {
                top: vec!["Ks".into()],
                middle: vec!["Qs".into(), "Qh".into()],
                bottom: vec!["As".into(), "Ah".into()],
            },
            dead: Vec::new(),
            draw: vec!["7d".into(), "3c".into(), "9c".into()],
            opp_count: 14,
            truncate_depth: None,
            t2_samples: 4,
            t3_samples: 3,
            t4_draw_sample: 8,
            // Enough that the leaf averages over roughly a dozen survivors
            // rather than one; see `default_pool_opponents`.
            pool_opponents: 400,
        }
    }

    /// **The pool reaches the labeler, and it is a different opponent.**
    ///
    /// A pooled opponent sets its thirteen after seeing hero's finished board
    /// and a library board did not, so the same request scored both ways must
    /// disagree.  Two runs that agreed would mean the pool never reached the
    /// leaf, which is exactly the failure this wiring can have.
    #[test]
    fn a_t1_root_prices_against_the_pool_and_disagrees_with_the_library() {
        for path in [POOL, LIBRARY, T2_MODEL, T3_MODEL] {
            if !Path::new(path).exists() {
                eprintln!("skipping: {path} is not on this box");
                return;
            }
        }
        let fl_ev = FlEv::load(Path::new(CONFIG)).expect("fl_ev config");
        let table = [
            fl_ev.value(14),
            fl_ev.value(15),
            fl_ev.value(16),
            fl_ev.value(17),
        ];
        let fl_table: evaluator::FlTable = [
            table[0] as f32,
            table[1] as f32,
            table[2] as f32,
            table[3] as f32,
        ];
        let bytes = std::fs::read(POOL).expect("read pool");
        let pool = fl_solver::pool::deserialize(&bytes, 14, table).expect("load pool");
        assert_eq!(pool.width, 14);
        assert!(
            pool.entries.len() > 1000,
            "a {}-entry pool cannot support a T1 draw",
            pool.entries.len()
        );
        let libraries =
            LibrarySet::load(&[PathBuf::from(LIBRARY)], [None, None, None]).expect("library");
        let t2_model =
            evaluator::Model::load(&std::fs::read(T2_MODEL).expect("t2 image")).expect("t2 model");
        let t3_model =
            evaluator::Model::load(&std::fs::read(T3_MODEL).expect("t3 image")).expect("t3 model");

        let request = request();
        let pooled = solve(
            &request,
            &fl_ev,
            &playout::OpponentSource::Pool(&pool),
            &fl_table,
            &t2_model,
            &t3_model,
        )
        .expect("pool solve");
        assert_eq!(pooled.leaf, "t2_t3_models_move_pool_best_response");
        assert_eq!(pooled.opponent_stream, Some(playout::root_stream(&request.id)));
        assert!(!pooled.actions.is_empty());

        // One draw per root: hero's seen cards do not depend on the action, so
        // every action must reach the terminal with the same opponents behind
        // it.  A per-action draw would show up here as a spread.
        let counts: std::collections::BTreeSet<u64> = pooled
            .actions
            .iter()
            .map(|action| action.mean_fl_samples.to_bits())
            .collect();
        assert_eq!(
            counts.len(),
            1,
            "actions saw {} different opponent counts; the draw is not shared",
            counts.len()
        );
        assert!(
            pooled.actions[0].mean_fl_samples > 0.0,
            "no opponent survived to any leaf"
        );

        let again = solve(
            &request,
            &fl_ev,
            &playout::OpponentSource::Pool(&pool),
            &fl_table,
            &t2_model,
            &t3_model,
        )
        .expect("pool solve, again");
        for (first, second) in pooled.actions.iter().zip(&again.actions) {
            assert_eq!(
                first.value.to_bits(),
                second.value.to_bits(),
                "{} moved between runs; the stream is not a function of the request",
                first.action_key
            );
        }

        let shelved = solve(
            &request,
            &fl_ev,
            &playout::OpponentSource::Libraries(&libraries),
            &fl_table,
            &t2_model,
            &t3_model,
        )
        .expect("library solve");
        assert_eq!(shelved.leaf, "t2_t3_models_move_library_scoring");
        assert_eq!(shelved.opponent_stream, None);
        assert_eq!(shelved.actions.len(), pooled.actions.len());

        let mut worst = 0.0f64;
        for (pooled, shelved) in pooled.actions.iter().zip(&shelved.actions) {
            assert_eq!(pooled.action_key, shelved.action_key);
            worst = worst.max((pooled.value - shelved.value).abs());
        }
        assert!(
            worst > 0.5,
            "pool and library agreed to within {worst}; the pool did not reach the leaf"
        );
    }
}
