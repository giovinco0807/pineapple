//! What a T3-boundary state is really worth, measured by playing it.
//!
//! `V3s` is trained on labels its own teacher produced, so checking it
//! against a better-sampled run of that same teacher only shows the teacher
//! agrees with itself.  This is the independent check: from the boundary
//! state, deal the street, let both seats play it the way the served chain
//! plays it, and price the eleven-card pair with the **exact** V4.  No
//! teacher, no boundary net, no learned value anywhere in the number --
//! only the chain's own decisions and an exact terminal.
//!
//! It answers a different question from calibration, and the difference is
//! the point: this is the value of the state *under the policy that will
//! actually play it*.  If the teacher's assumptions (an own-hand ranker
//! standing in for the opponent, one traced placement standing in for the
//! opponent's half-street) are wrong, the gap shows up here and nowhere
//! else.
//!
//! Both seats' draws are sampled `--probe-draws` times with common random
//! numbers across nothing -- there is only one action here, the state -- and
//! the mean over draws is the estimate.  Its standard error is reported, so
//! a disagreement with the net can be read against the noise it was measured
//! through.

use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::evaluator;
use super::hu_encode;
use super::playout;
use super::v4_first::{self, V4FirstRequest};
use super::{all_cards, to_core_card, BoardStr, Card, CoreBoard, FlEv};

const ROW_CAPACITY: [usize; 3] = [3, 5, 5];

#[derive(Deserialize)]
pub struct V3sProbeRequest {
    pub id: String,
    /// The first actor's (BB's) nine cards.
    pub board: [Vec<String>; 3],
    /// BB's two discards.
    pub dead: Vec<String>,
    /// The second actor's (BTN's) nine cards.
    pub opp_board: [Vec<String>; 3],
    /// BTN's two discards, if the trace knows them.  They only shrink the
    /// deck; leaving them out is the belief the server itself has.
    #[serde(default)]
    pub opp_dead: Vec<String>,
    #[serde(default = "default_draws")]
    pub draws: usize,
}

fn default_draws() -> usize {
    24
}

#[derive(Serialize)]
pub struct V3sProbeResponse {
    pub id: String,
    pub schema: &'static str,
    /// Mean over sampled street deals of the exact V4 at the T4 boundary,
    /// from BB's side -- the same quantity V3s is trained to predict.
    pub value: f64,
    pub stderr: f64,
    pub draws: usize,
}

fn rows_key(rows: &[Vec<String>; 3]) -> String {
    rows.iter()
        .map(|row| {
            let mut sorted = row.clone();
            sorted.sort();
            sorted.join(",")
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn placements(board: &[Vec<String>; 3], draw: &[String]) -> Vec<([Vec<String>; 3], String)> {
    let mut out = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for toss in 0..3usize {
        let kept: Vec<usize> = (0..3).filter(|k| *k != toss).collect();
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
                let key = format!("{}|{}", rows_key(&after), draw[toss]);
                if seen.insert(key) {
                    out.push((after, draw[toss].clone()));
                }
            }
        }
    }
    out
}

fn board_of(rows: &[Vec<String>; 3]) -> Result<CoreBoard> {
    CoreBoard::from_str_board(&BoardStr {
        top: rows[0].clone(),
        middle: rows[1].clone(),
        bottom: rows[2].clone(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn solve(
    request: &V3sProbeRequest,
    fl_ev: &FlEv,
    fl_table: &evaluator::FlTable,
    bb_model: &evaluator::Model,
) -> Result<V3sProbeResponse> {
    let table = [
        fl_ev.value(14),
        fl_ev.value(15),
        fl_ev.value(16),
        fl_ev.value(17),
    ];
    if request.board.iter().map(Vec::len).sum::<usize>() != 9
        || request.opp_board.iter().map(Vec::len).sum::<usize>() != 9
    {
        bail!("{}: a T3 boundary is two nine-card boards", request.id);
    }

    // The deck the street is dealt from: everything neither board nor either
    // seat's known discards hold.  Jokers counted, never name-matched.
    let mut naturals: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    let mut jokers = 0usize;
    for name in request
        .board
        .iter()
        .flatten()
        .chain(request.opp_board.iter().flatten())
        .chain(&request.dead)
        .chain(&request.opp_dead)
    {
        if name.starts_with('X') {
            jokers += 1;
        } else if !naturals.insert(name) {
            bail!("{}: card {name} appears twice", request.id);
        }
    }
    let mut deck: Vec<String> = all_cards()
        .into_iter()
        .filter(|n| !n.starts_with('X') && !naturals.contains(n.as_str()))
        .collect();
    for slot in 0..(2usize.saturating_sub(jokers)) {
        deck.push(format!("X{}", slot + 1));
    }
    if deck.len() < 6 {
        bail!("{}: deck too small", request.id);
    }

    let outcomes: Result<Vec<f64>> = (0..request.draws)
        .into_par_iter()
        .map(|tick| {
            // Six cards off the deck: three to each seat.
            let picks = fl_solver::t2_labels::sampled_t3_draws(
                deck.len(),
                2,
                playout::root_stream(&format!("{}/{tick}", request.id)),
            );
            let mut taken: Vec<usize> = picks.iter().flatten().copied().collect();
            taken.sort_unstable();
            taken.dedup();
            if taken.len() < 6 {
                // The sampler returned overlapping triples; skip this tick
                // rather than deal a card twice.
                return Ok(f64::NAN);
            }
            let bb_draw: Vec<String> = taken[..3].iter().map(|k| deck[*k].clone()).collect();
            let btn_draw: Vec<String> = taken[3..6].iter().map(|k| deck[*k].clone()).collect();

            // --- BB acts first, by the served model -----------------------
            let mut seen: Vec<String> = request
                .board
                .iter()
                .flatten()
                .chain(request.opp_board.iter().flatten())
                .chain(&request.dead)
                .chain(&bb_draw)
                .cloned()
                .collect();
            seen.sort();
            let pool: Vec<Card> = all_cards()
                .into_iter()
                .filter(|n| !seen.contains(n))
                .map(|n| to_core_card(&n))
                .collect::<Result<Vec<_>>>()?;
            let opp_board = board_of(&request.opp_board)?;
            let memo: playout::RowwiseMemo =
                std::sync::Mutex::new(std::collections::HashMap::new());
            let mut features: Vec<f32> = Vec::new();
            let mut scratch: Vec<f32> = Vec::new();
            let node_seed = format!("{}/{tick}/bb", request.id);
            let mut best = f32::NEG_INFINITY;
            let mut bb_after: Option<([Vec<String>; 3], String)> = None;
            for (after, toss) in placements(&request.board, &bb_draw) {
                let own = board_of(&after)?;
                hu_encode::encode_pair(
                    &own,
                    &opp_board,
                    &pool,
                    fl_ev,
                    fl_table,
                    &memo,
                    0,
                    &node_seed,
                    &mut features,
                )?;
                let score = bb_model.predict(&features, &mut scratch);
                if score > best {
                    best = score;
                    bb_after = Some((after, toss));
                }
            }
            let (bb_board, bb_toss) =
                bb_after.ok_or_else(|| anyhow!("{}: no BB placement", request.id))?;
            let mut bb_dead = request.dead.clone();
            bb_dead.push(bb_toss);

            // --- BTN acts second, exactly ---------------------------------
            let mut btn_best = f64::NEG_INFINITY;
            let mut btn_board: Option<[Vec<String>; 3]> = None;
            for (after, toss) in placements(&request.opp_board, &btn_draw) {
                let mut dead = request.opp_dead.clone();
                dead.push(toss);
                while dead.len() < 3 {
                    // The probe may not know BTN's earlier discards; V4 wants
                    // three, and any three unseen cards leave the arithmetic
                    // right because the pool is what matters.
                    dead.push(
                        deck.iter()
                            .find(|n| !dead.contains(n) && !taken.iter().any(|k| deck[*k] == **n))
                            .cloned()
                            .ok_or_else(|| anyhow!("no filler card"))?,
                    );
                }
                let request_v4 = V4FirstRequest {
                    id: format!("{}/{tick}/btn", request.id),
                    board: BoardStr {
                        top: bb_board[0].clone(),
                        middle: bb_board[1].clone(),
                        bottom: bb_board[2].clone(),
                    },
                    dead: bb_dead.clone(),
                    draw: None,
                    opp_dead: Vec::new(),
                    opp_board: BoardStr {
                        top: after[0].clone(),
                        middle: after[1].clone(),
                        bottom: after[2].clone(),
                    },
                };
                // BTN maximises its own value, i.e. minimises BB's.
                let bb_value = v4_first::solve(&request_v4, &table)?.value;
                if -bb_value > btn_best {
                    btn_best = -bb_value;
                    btn_board = Some(after);
                }
            }
            btn_board.ok_or_else(|| anyhow!("{}: no BTN placement", request.id))?;
            // The value of the state, from BB's side, is what BTN left it at.
            Ok(-btn_best)
        })
        .collect();

    let values: Vec<f64> = outcomes?.into_iter().filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        bail!("{}: every sampled deal was rejected", request.id);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n.max(2.0);
    Ok(V3sProbeResponse {
        id: request.id.clone(),
        schema: "ofc_hu_v3s_probe/v1",
        value: mean,
        stderr: (variance / n).sqrt(),
        draws: values.len(),
    })
}
