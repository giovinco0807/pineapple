//! T3-second (BTN) teacher labels for the HU phase: every placement priced by
//! the exact V4 with roles swapped.
//!
//! Hero acts last in the T3 street, so hero's after-board lands exactly on
//! the T4-street boundary and each action's value is one V4 read:
//!
//! ```text
//!   label(action) = -V4_first( me := opponent, them := hero after-board )
//! ```
//!
//! The negation is the zero-sum flip; `compose` is antisymmetric by
//! construction, so the flip is exact.
//!
//! One approximation lives here and is auditable: V4_first models the
//! opponent's decisions against a pool that excludes **hero's** discards
//! (which the real opponent cannot see) instead of the opponent's own
//! (which hero cannot see).  Some three cards must come out to keep the
//! pool's arithmetic honest, and hero's are the ones hero actually knows.
//! Passing `opp_dead` (from omniscient traces) prices the true world
//! instead, and the difference between the two on the same roots is the
//! measured cost of the approximation -- run it before trusting the teacher.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use super::discard_model::DiscardModel;
use super::v4_first::{self, DeadSide, V4FirstRequest};
use super::{BoardStr, FlEv};

const ROW_CAPACITY: [usize; 3] = [3, 5, 5];

#[derive(Deserialize)]
pub struct T3SecondHuRequest {
    pub id: String,
    /// Hero's nine placed cards, by row (trace order: top, middle, bottom).
    pub board: [Vec<String>; 3],
    /// Hero's two prior discards.
    pub dead: Vec<String>,
    /// Hero's three drawn cards.
    pub draw: Vec<String>,
    /// Opponent's board after its own T3 placement: eleven cards.
    pub opp_board: [Vec<String>; 3],
    /// Omniscient audit only; see the module note.
    #[serde(default)]
    pub opp_dead: Vec<String>,
}

#[derive(Serialize)]
pub struct T3SecondHuAction {
    /// The after-board plus the discard: `top|middle|bottom|discard`, the
    /// same shape every teacher in this tree writes.
    pub action_key: String,
    pub value: f64,
}

#[derive(Serialize)]
pub struct T3SecondHuResponse {
    pub id: String,
    pub schema: &'static str,
    pub belief: &'static str,
    pub actions: Vec<T3SecondHuAction>,
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

pub fn solve(
    request: &T3SecondHuRequest,
    fl_ev: &FlEv,
    discard: Option<&DiscardModel>,
    // Let the swapped V4 collapse its state sweep over rank classes when
    // both boards are flush-dead; see rank_collapse.rs.  Inert unless the
    // caller passed --rank-collapse.
    rank_collapse: bool,
) -> Result<T3SecondHuResponse> {
    if request.board.iter().map(Vec::len).sum::<usize>() != 9 {
        bail!("{}: a T3 board is nine cards", request.id);
    }
    if request.dead.len() != 2 || request.draw.len() != 3 {
        bail!("{}: T3 wants two dead and three drawn", request.id);
    }
    let table = [
        fl_ev.value(14),
        fl_ev.value(15),
        fl_ev.value(16),
        fl_ev.value(17),
    ];

    let opp_board = BoardStr {
        top: request.opp_board[0].clone(),
        middle: request.opp_board[1].clone(),
        bottom: request.opp_board[2].clone(),
    };

    let mut actions: Vec<T3SecondHuAction> = Vec::new();
    let mut seen_keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for toss in 0..3usize {
        let kept: Vec<usize> = (0..3).filter(|k| *k != toss).collect();
        for row_a in 0..3usize {
            for row_b in 0..3usize {
                let mut need = [0usize; 3];
                need[row_a] += 1;
                need[row_b] += 1;
                if (0..3).any(|r| request.board[r].len() + need[r] > ROW_CAPACITY[r]) {
                    continue;
                }
                let mut after = request.board.clone();
                after[row_a].push(request.draw[kept[0]].clone());
                after[row_b].push(request.draw[kept[1]].clone());
                let key = rows_key(&after, &request.draw[toss]);
                if !seen_keys.insert(key.clone()) {
                    continue;
                }
                let mut dead = request.dead.clone();
                dead.push(request.draw[toss].clone());
                // Roles swapped: the opponent is the T4-street first actor
                // and hero's after-board is what it sees.
                let swapped = V4FirstRequest {
                    id: format!("{}/{}", request.id, key),
                    board: opp_board.clone(),
                    dead,
                    draw: None,
                    opp_dead: request.opp_dead.clone(),
                    opp_board: BoardStr {
                        top: after[0].clone(),
                        middle: after[1].clone(),
                        bottom: after[2].clone(),
                    },
                };
                // In the swapped frame the hidden discards belong to the
                // swapped `board` owner -- the real opponent.
                let value = -v4_first::solve_maybe_collapsed(
                    &swapped,
                    &table,
                    discard.map(|model| (model, DeadSide::Board)),
                    rank_collapse,
                )?
                .value;
                actions.push(T3SecondHuAction {
                    action_key: key,
                    value,
                });
            }
        }
    }
    actions.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    Ok(T3SecondHuResponse {
        id: request.id.clone(),
        schema: "ofc_hu_t3_second/v1",
        belief: if !request.opp_dead.is_empty() {
            "omniscient_audit"
        } else if discard.is_some() {
            "hero_information_discard_weighted"
        } else {
            "hero_information_exchangeable"
        },
        actions,
    })
}
