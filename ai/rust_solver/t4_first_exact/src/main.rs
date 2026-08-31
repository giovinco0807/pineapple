//! Exact T4 first-seat (BB) solver under the declared uniform restart belief.
//!
//! Semantics are pinned to the Python reference
//! `ai/tutor/t4_bb_exact_resolver.py`, which is the parity oracle for this
//! crate.  For every legal hero action (place two of the three drawn cards,
//! discard the third) the solver enumerates every remaining `C(n,3)` opponent
//! draw, lets the opponent play its exact best terminal response, and averages
//! the negated opponent score.
//!
//! The legacy `t4_exact_solver` crate is NOT the basis for this one: it treats
//! the Fantasyland card count (14/15/16/17) as the Fantasyland EV and scores in
//! `i32`, which cannot represent the canonical table at all.  That divergence
//! is why `ai/reports/late_hu_status_20260712/README.md` already records the
//! legacy crate as unused.  Here the Fantasyland EV is read from
//! `ai/config/fl_ev.json` and its SHA-256 is emitted with every result.

mod evaluator;
mod joint_outlook;
mod play_roots;
mod playout;
mod rank_collapse;
mod row_memo;
mod t0_policy;
mod t0_vs_fl;
mod t1_vs_fl;
mod t2_vs_fl;
mod hu_encode;
mod hu_match;
mod hu_traces;
mod self_play;
mod fl_sim;
mod fl_t0_deep;
mod fl_t0_teach;
mod discard_model;
mod t3_first_hu;
mod t3_second_hu;
mod v3s_probe;
mod v4_first;
mod t2_vs_fl_pool;
mod t3_second;
mod t3_vs_fl;
mod t3_vs_fl_lib;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use ofc_core::{
    check_fl_entry, compare_3_hands, compare_5_hands, evaluate_board_with_joker_constraint,
    evaluate_hand_value, get_bottom_royalty, get_middle_royalty, get_top_royalty, Card,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const ROWS: [&str; 3] = ["top", "middle", "bottom"];
const ROW_CAPACITY: [usize; 3] = [3, 5, 5];

// ------------------------------------------------------------------
// Cards: strings keep X1/X2 physically distinct for deck accounting;
// evaluation maps both to the wild `ofc_core` joker, which is correct
// because the two jokers are interchangeable for hand strength.
// ------------------------------------------------------------------

pub(crate) fn all_cards() -> Vec<String> {
    let mut cards = Vec::with_capacity(54);
    for suit in ["s", "h", "d", "c"] {
        for rank in [
            "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A",
        ] {
            cards.push(format!("{rank}{suit}"));
        }
    }
    cards.push("X1".to_string());
    cards.push("X2".to_string());
    cards
}

fn is_joker(card: &str) -> bool {
    card == "X1" || card == "X2" || card == "JK"
}

pub(crate) fn to_core_card(card: &str) -> Result<Card> {
    if is_joker(card) {
        return Ok(Card { rank: 0, suit: 4 });
    }
    let bytes = card.as_bytes();
    if bytes.len() != 2 {
        bail!("invalid card token: {card}");
    }
    let rank = match bytes[0] as char {
        '2'..='9' => (bytes[0] - b'0') as u8,
        'T' => 10,
        'J' => 11,
        'Q' => 12,
        'K' => 13,
        'A' => 14,
        other => bail!("invalid rank {other} in {card}"),
    };
    let suit = match bytes[1] as char {
        's' => 0,
        'h' => 1,
        'd' => 2,
        'c' => 3,
        other => bail!("invalid suit {other} in {card}"),
    };
    Ok(Card { rank, suit })
}

// ------------------------------------------------------------------
// Fantasyland EV table (canonical source of truth)
// ------------------------------------------------------------------

pub(crate) struct FlEv {
    by_card_count: BTreeMap<u8, f64>,
    config_sha256: String,
}

impl FlEv {
    fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read(path)
            .with_context(|| format!("cannot read FL EV config {}", path.display()))?;
        let config_sha256 = format!("{:x}", Sha256::digest(&raw));
        let parsed: serde_json::Value = serde_json::from_slice(&raw)
            .with_context(|| format!("FL EV config is not JSON: {}", path.display()))?;
        let table = parsed
            .get("fl_ev")
            .and_then(|value| value.as_object())
            .ok_or_else(|| anyhow!("FL EV config has no object field `fl_ev`"))?;
        let mut by_card_count = BTreeMap::new();
        for (key, value) in table {
            let count: u8 = key
                .parse()
                .with_context(|| format!("FL EV key {key} is not an integer"))?;
            let ev = value
                .as_f64()
                .ok_or_else(|| anyhow!("FL EV value for {key} is not a number"))?;
            by_card_count.insert(count, ev);
        }
        for required in [14u8, 15, 16, 17] {
            if !by_card_count.contains_key(&required) {
                bail!("FL EV config is missing card count {required}");
            }
        }
        Ok(Self {
            by_card_count,
            config_sha256,
        })
    }

    fn value(&self, card_count: u8) -> f64 {
        self.by_card_count.get(&card_count).copied().unwrap_or(0.0)
    }
}

// ------------------------------------------------------------------
// Board handling
// ------------------------------------------------------------------

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct BoardStr {
    top: Vec<String>,
    middle: Vec<String>,
    bottom: Vec<String>,
}

#[derive(Clone)]
pub(crate) struct CoreBoard {
    rows: [Vec<Card>; 3],
}

impl CoreBoard {
    fn from_str_board(board: &BoardStr) -> Result<Self> {
        let mut rows = [Vec::new(), Vec::new(), Vec::new()];
        for (index, cards) in [&board.top, &board.middle, &board.bottom]
            .iter()
            .enumerate()
        {
            for card in cards.iter() {
                rows[index].push(to_core_card(card)?);
            }
            if rows[index].len() > ROW_CAPACITY[index] {
                bail!("row {} exceeds capacity", ROWS[index]);
            }
        }
        Ok(Self { rows })
    }

    fn card_count(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }

    fn open_slots(&self) -> [usize; 3] {
        [
            ROW_CAPACITY[0] - self.rows[0].len(),
            ROW_CAPACITY[1] - self.rows[1].len(),
            ROW_CAPACITY[2] - self.rows[2].len(),
        ]
    }
}

/// Terminal facts of one complete board, after canonical joker constraint.
///
/// Row hand values are kept as the encoded comparison keys rather than the
/// cards, because scoring only ever compares them.  That keeps the struct
/// copyable and lets the opponent's terminals be computed once per root and
/// reused across every hero action.
#[derive(Clone, Copy)]
pub(crate) struct Terminal {
    busted: bool,
    royalty: i32,
    fl_card_count: u8,
    values: [u32; 3],
}

pub(crate) fn terminal_of(board: &CoreBoard) -> Terminal {
    let eval = evaluate_board_with_joker_constraint(&board.rows[0], &board.rows[1], &board.rows[2]);
    let busted = eval.busted;
    let (royalty, fl_card_count) = if busted {
        (0, 0)
    } else {
        let royalty = get_top_royalty(&eval.top)
            + get_middle_royalty(&eval.mid)
            + get_bottom_royalty(&eval.bot);
        let (qualified, count) = check_fl_entry(&eval.top);
        (royalty, if qualified { count } else { 0 })
    };
    Terminal {
        busted,
        royalty,
        fl_card_count,
        values: [
            evaluate_hand_value(&eval.top, 3),
            evaluate_hand_value(&eval.mid, 5),
            evaluate_hand_value(&eval.bot, 5),
        ],
    }
}

/// Canonical heads-up score from the hero's perspective, Fantasyland EV
/// included.  Mirrors `exact_late._score_against_complete_opponent`.
fn hero_score(hero: &Terminal, opponent: &Terminal, fl_ev: &FlEv) -> f64 {
    let base = if hero.busted && opponent.busted {
        0.0
    } else if hero.busted {
        -6.0 - opponent.royalty as f64
    } else if opponent.busted {
        6.0 + hero.royalty as f64
    } else {
        let lines: i32 = (0..3)
            .map(|row| {
                let (a, b) = (hero.values[row], opponent.values[row]);
                (a > b) as i32 - (a < b) as i32
            })
            .sum();
        let scoop = if lines == 3 {
            3
        } else if lines == -3 {
            -3
        } else {
            0
        };
        (lines + scoop + hero.royalty - opponent.royalty) as f64
    };
    base + fl_ev.value(hero.fl_card_count) - fl_ev.value(opponent.fl_card_count)
}

// ------------------------------------------------------------------
// Actions
// ------------------------------------------------------------------

/// One legal placement of two drawn cards, with the third discarded.
pub(crate) struct Action {
    /// Row index per placed card, aligned with `cards`.
    rows: [usize; 2],
    cards: [String; 2],
    discard: String,
}

impl Action {
    /// Canonical key identical to `exact_late.action_key`:
    /// keys sorted (`discard` then `placements`), placements ordered by row
    /// (top, middle, bottom) and by card within a row.
    fn key(&self) -> String {
        let mut by_row: [Vec<&str>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for slot in 0..2 {
            by_row[self.rows[slot]].push(self.cards[slot].as_str());
        }
        let mut parts = Vec::new();
        for row in 0..3 {
            by_row[row].sort_unstable();
            for card in &by_row[row] {
                parts.push(format!("[\"{}\",\"{}\"]", card, ROWS[row]));
            }
        }
        format!(
            "{{\"discard\":\"{}\",\"placements\":[{}]}}",
            self.discard,
            parts.join(",")
        )
    }
}

pub(crate) fn legal_actions(board: &CoreBoard, draw: &[String]) -> Vec<Action> {
    let open = board.open_slots();
    let mut actions = Vec::new();
    // Choose which card is discarded, then place the remaining two in order.
    for discard_index in 0..3 {
        let kept: Vec<usize> = (0..3).filter(|index| *index != discard_index).collect();
        for row_a in 0..3 {
            for row_b in 0..3 {
                let mut need = [0usize; 3];
                need[row_a] += 1;
                need[row_b] += 1;
                if (0..3).any(|row| need[row] > open[row]) {
                    continue;
                }
                actions.push(Action {
                    rows: [row_a, row_b],
                    cards: [draw[kept[0]].clone(), draw[kept[1]].clone()],
                    discard: draw[discard_index].clone(),
                });
            }
        }
    }
    // Distinct placements only: two cards into the same row are order-free.
    let mut seen = std::collections::BTreeSet::new();
    actions.retain(|action| seen.insert(action.key()));
    actions
}

pub(crate) fn apply(board: &CoreBoard, action: &Action) -> Result<CoreBoard> {
    let mut next = board.clone();
    for slot in 0..2 {
        next.rows[action.rows[slot]].push(to_core_card(&action.cards[slot])?);
    }
    Ok(next)
}

// ------------------------------------------------------------------
// Solve
// ------------------------------------------------------------------

#[derive(Deserialize)]
struct Request {
    id: String,
    /// Hero (first seat) board, 11 cards.
    bb: BoardStr,
    /// Opponent (second seat) board, 11 cards.
    btn: BoardStr,
    /// Hero's three drawn cards.
    draw: Vec<String>,
    /// Hero's own hidden discards, excluded from the opponent draw pool.
    #[serde(default)]
    dead: Vec<String>,
}

#[derive(Serialize)]
pub(crate) struct ActionResult {
    action_key: String,
    ev: f64,
    discard: String,
    placements: Vec<(String, String)>,
}

#[derive(Serialize)]
struct Response {
    id: String,
    schema: &'static str,
    method: &'static str,
    belief_model: &'static str,
    fl_ev_config_sha256: String,
    remaining_deck_size: usize,
    enumerated_draws: usize,
    actions: Vec<ActionResult>,
    /// Action-independent joint outlook of the opponent's two-card completion.
    /// Order matches `ai/tutor/t4_first_features.opponent_joint_block`.
    opponent_joint_block: [f64; 8],
}

/// The opponent's own value of a finished board: royalty plus Fantasyland,
/// with a foul worth -6.  Line wins against the hero are excluded on purpose --
/// they are action-dependent, and this block must stay shared across the node.
pub(crate) fn opponent_self_value(terminal: &Terminal, fl_ev: &FlEv) -> f64 {
    if terminal.busted {
        -6.0
    } else {
        terminal.royalty as f64 + fl_ev.value(terminal.fl_card_count)
    }
}

fn solve(request: &Request, fl_ev: &FlEv) -> Result<Response> {
    let hero_base = CoreBoard::from_str_board(&request.bb)?;
    let opponent_base = CoreBoard::from_str_board(&request.btn)?;
    if hero_base.card_count() != 11 || opponent_base.card_count() != 11 {
        bail!("T4 first-seat solve requires two 11-card boards");
    }
    if request.draw.len() != 3 {
        bail!("T4 first-seat solve requires a 3-card draw");
    }

    let mut used: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
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
        .chain(&request.dead)
    {
        if !used.insert(card.clone()) {
            bail!("duplicate card in request: {card}");
        }
    }

    let actions = legal_actions(&hero_base, &request.draw);
    if actions.is_empty() {
        bail!("T4 first-seat root has no legal action");
    }

    let opponent_open = opponent_base.open_slots();
    let opponent_open_total: usize = opponent_open.iter().sum();
    if opponent_open_total != 2 {
        bail!("opponent board must have exactly two open slots");
    }

    // The unknown pool is action-independent: two of the three drawn cards land
    // on the hero board and the third is discarded, so all three leave the pool
    // either way.  Computing it once also lets the inner loop stay allocation
    // free, which is what makes teacher-scale generation affordable.
    let pool: Vec<Card> = all_cards()
        .into_iter()
        .filter(|card| !used.contains(card))
        .map(|card| to_core_card(&card))
        .collect::<Result<Vec<_>>>()?;

    // The opponent keeps two of its three cards, so its final board is fixed by
    // an unordered PAIR of cards plus a row assignment.  Enumerating the row
    // assignments once, and then the terminal of each (pair, assignment), costs
    // `C(n,2) x assignments` evaluations instead of `C(n,3) x replies` -- the
    // same board is otherwise re-evaluated once for every draw that contains
    // the pair.
    let mut row_assignments: Vec<[usize; 2]> = Vec::new();
    for row_a in 0..3usize {
        for row_b in 0..3usize {
            let mut need = [0usize; 3];
            need[row_a] += 1;
            need[row_b] += 1;
            if (0..3).any(|row| need[row] > opponent_open[row]) {
                continue;
            }
            if row_a == row_b && !row_assignments.iter().any(|rows| rows == &[row_a, row_b]) {
                row_assignments.push([row_a, row_b]);
            } else if row_a != row_b {
                row_assignments.push([row_a, row_b]);
            }
        }
    }
    if row_assignments.is_empty() {
        bail!("opponent has no legal T4 reply shape");
    }

    // The opponent's terminal facts do not depend on the hero's action, so all
    // `C(n,3) x replies` boards are evaluated once per root and reused.  This
    // is the difference between one and `len(actions)` passes over the most
    // expensive part of the solve.
    let assignment_count = row_assignments.len();
    let pool_len = pool.len();

    // Index unordered pairs as `pair_index(i, j)` for i < j.
    let pair_index = |i: usize, j: usize| -> usize { i * pool_len + j };
    let mut pair_slots: Vec<(usize, usize)> = Vec::new();
    for i in 0..pool_len {
        for j in (i + 1)..pool_len {
            pair_slots.push((i, j));
        }
    }
    // This precompute dominates the solve, so it carries the root's
    // parallelism; the per-action pass below is pure arithmetic.
    let pair_terminals: Vec<((usize, usize), Vec<Terminal>)> = pair_slots
        .par_iter()
        .map(|(i, j)| {
            let mut scratch = opponent_base.clone();
            let mut local = Vec::with_capacity(assignment_count);
            for rows in &row_assignments {
                scratch.rows[rows[0]].push(pool[*i]);
                scratch.rows[rows[1]].push(pool[*j]);
                local.push(terminal_of(&scratch));
                scratch.rows[rows[1]].pop();
                scratch.rows[rows[0]].pop();
            }
            ((*i, *j), local)
        })
        .collect();
    let mut terminal_table: Vec<Option<Vec<Terminal>>> = vec![None; pool_len * pool_len];
    for ((i, j), terminals) in pair_terminals {
        terminal_table[pair_index(i, j)] = Some(terminals);
    }

    let mut draws: Vec<[usize; 3]> = Vec::new();
    for a in 0..pool_len {
        for b in (a + 1)..pool_len {
            for c in (b + 1)..pool_len {
                draws.push([a, b, c]);
            }
        }
    }
    if draws.is_empty() {
        bail!("opponent draw enumeration is empty");
    }
    let draw_count = draws.len();

    let results: Result<Vec<ActionResult>> = actions
        .par_iter()
        .map(|action| {
            let hero_final = apply(&hero_base, action)?;
            let hero_terminal = terminal_of(&hero_final);

            let mut total = 0.0f64;
            for draw in &draws {
                // The opponent keeps two of the three dealt cards, so its reply
                // is one of three pairs, each already evaluated above.
                let mut best = f64::NEG_INFINITY;
                for (first, second) in [(draw[0], draw[1]), (draw[0], draw[2]), (draw[1], draw[2])]
                {
                    let terminals = terminal_table[pair_index(first, second)]
                        .as_ref()
                        .ok_or_else(|| anyhow!("opponent pair terminal is missing"))?;
                    for terminal in terminals {
                        // Opponent maximizes its own score; ties do not change
                        // the hero EV, so only the max is needed.
                        let score = hero_score(terminal, &hero_terminal, fl_ev);
                        if score > best {
                            best = score;
                        }
                    }
                }
                total += -best;
            }
            let draws = draw_count;
            let mut placements: Vec<(String, String)> = (0..2)
                .map(|slot| {
                    (
                        action.cards[slot].clone(),
                        ROWS[action.rows[slot]].to_string(),
                    )
                })
                .collect();
            placements.sort();
            Ok(ActionResult {
                action_key: action.key(),
                ev: total / draws as f64,
                discard: action.discard.clone(),
                placements,
            })
        })
        .collect();

    // The joint block reuses the pair terminals already computed above, so it
    // costs one pass over the draws rather than a second enumeration.
    let mut best_self: Vec<f64> = Vec::with_capacity(draw_count);
    let mut fouls = 0usize;
    let mut survivors = 0usize;
    let mut survive_royalty = 0.0f64;
    let mut survive_fl = 0.0f64;
    for draw in &draws {
        let mut best = f64::NEG_INFINITY;
        let mut best_parts = (0.0f64, 0.0f64);
        for (first, second) in [(draw[0], draw[1]), (draw[0], draw[2]), (draw[1], draw[2])] {
            let terminals = terminal_table[pair_index(first, second)]
                .as_ref()
                .ok_or_else(|| anyhow!("opponent pair terminal is missing"))?;
            for terminal in terminals {
                let value = opponent_self_value(terminal, fl_ev);
                if value > best {
                    best = value;
                    best_parts = if terminal.busted {
                        (0.0, 0.0)
                    } else {
                        (terminal.royalty as f64, fl_ev.value(terminal.fl_card_count))
                    };
                }
            }
        }
        if best <= -6.0 {
            fouls += 1;
        } else {
            survivors += 1;
            survive_royalty += best_parts.0;
            survive_fl += best_parts.1;
        }
        best_self.push(best);
    }
    let count = best_self.len() as f64;
    let mean = best_self.iter().sum::<f64>() / count;
    let variance = best_self
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        / count;
    let survive_denominator = survivors.max(1) as f64;
    const MAX_ROYALTY: f64 = 25.0;
    const MAX_FL_EV: f64 = 63.5;
    let joint_block = [
        fouls as f64 / count,
        mean / MAX_ROYALTY,
        (survive_royalty / survive_denominator) / MAX_ROYALTY,
        (survive_fl / survive_denominator) / MAX_FL_EV,
        variance.sqrt() / MAX_ROYALTY,
        best_self.iter().filter(|v| **v >= 6.0).count() as f64 / count,
        best_self.iter().filter(|v| **v >= 15.0).count() as f64 / count,
        (best_self.iter().filter(|v| **v > -6.0).sum::<f64>() / survive_denominator) / MAX_ROYALTY,
    ];

    let mut actions_out = results?;
    actions_out.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    let remaining = 54 - used.len();
    let enumerated = remaining * (remaining.saturating_sub(1)) * (remaining.saturating_sub(2)) / 6;
    Ok(Response {
        id: request.id.clone(),
        schema: "ofc_t4_first_exact/v1",
        method: "t4_bb_exact_uniform_deal_response_v1",
        belief_model: "uniform_exchangeable_restart_v1",
        fl_ev_config_sha256: fl_ev.config_sha256.clone(),
        remaining_deck_size: remaining,
        enumerated_draws: enumerated,
        actions: actions_out,
        opponent_joint_block: joint_block,
    })
}

#[derive(Deserialize)]
struct JointRequest {
    id: String,
    /// Opponent board with exactly two open slots.
    btn: BoardStr,
    /// Cards unseen from the acting seat's information set.
    pool: Vec<String>,
}

#[derive(Serialize)]
struct JointResponse {
    id: String,
    schema: &'static str,
    fl_ev_config_sha256: String,
    pool_size: usize,
    opponent_joint_block: [f64; 8],
}

/// Joint block only: the opponent pair terminals without the per-action pass.
/// Used when a caller needs the shared node block but not the exact action EVs.
fn solve_joint_only(request: &JointRequest, fl_ev: &FlEv) -> Result<JointResponse> {
    let opponent_base = CoreBoard::from_str_board(&request.btn)?;
    if opponent_base.card_count() != 11 {
        bail!("joint-only solve requires an 11-card opponent board");
    }
    let opponent_open = opponent_base.open_slots();
    if opponent_open.iter().sum::<usize>() != 2 {
        bail!("opponent board must have exactly two open slots");
    }
    let pool: Vec<Card> = request
        .pool
        .iter()
        .map(|card| to_core_card(card))
        .collect::<Result<Vec<_>>>()?;
    if pool.len() < 3 {
        bail!("pool must hold at least three cards");
    }

    let mut row_assignments: Vec<[usize; 2]> = Vec::new();
    for row_a in 0..3usize {
        for row_b in 0..3usize {
            let mut need = [0usize; 3];
            need[row_a] += 1;
            need[row_b] += 1;
            if (0..3).any(|row| need[row] > opponent_open[row]) {
                continue;
            }
            if row_a == row_b {
                if !row_assignments.iter().any(|rows| rows == &[row_a, row_b]) {
                    row_assignments.push([row_a, row_b]);
                }
            } else {
                row_assignments.push([row_a, row_b]);
            }
        }
    }
    let pool_len = pool.len();
    let mut pair_slots: Vec<(usize, usize)> = Vec::new();
    for i in 0..pool_len {
        for j in (i + 1)..pool_len {
            pair_slots.push((i, j));
        }
    }
    let pair_terminals: Vec<((usize, usize), Vec<Terminal>)> = pair_slots
        .par_iter()
        .map(|(i, j)| {
            let mut scratch = opponent_base.clone();
            let mut local = Vec::with_capacity(row_assignments.len());
            for rows in &row_assignments {
                scratch.rows[rows[0]].push(pool[*i]);
                scratch.rows[rows[1]].push(pool[*j]);
                local.push(terminal_of(&scratch));
                scratch.rows[rows[1]].pop();
                scratch.rows[rows[0]].pop();
            }
            ((*i, *j), local)
        })
        .collect();
    let mut table: Vec<Option<Vec<Terminal>>> = vec![None; pool_len * pool_len];
    for ((i, j), terminals) in pair_terminals {
        table[i * pool_len + j] = Some(terminals);
    }

    let mut best_self: Vec<f64> = Vec::new();
    let mut fouls = 0usize;
    let mut survivors = 0usize;
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
                                (terminal.royalty as f64, fl_ev.value(terminal.fl_card_count))
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
    let variance = best_self
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        / count;
    let denominator = survivors.max(1) as f64;
    const MAX_ROYALTY: f64 = 25.0;
    const MAX_FL_EV: f64 = 63.5;
    Ok(JointResponse {
        id: request.id.clone(),
        schema: "ofc_t4_first_joint_block/v1",
        fl_ev_config_sha256: fl_ev.config_sha256.clone(),
        pool_size: pool_len,
        opponent_joint_block: [
            fouls as f64 / count,
            mean / MAX_ROYALTY,
            (survive_royalty / denominator) / MAX_ROYALTY,
            (survive_fl / denominator) / MAX_FL_EV,
            variance.sqrt() / MAX_ROYALTY,
            best_self.iter().filter(|v| **v >= 6.0).count() as f64 / count,
            best_self.iter().filter(|v| **v >= 15.0).count() as f64 / count,
            (best_self.iter().filter(|v| **v > -6.0).sum::<f64>() / denominator) / MAX_ROYALTY,
        ],
    })
}

#[derive(Parser)]
#[command(about = "Exact T4 first-seat solver (canonical FL EV, f64 scoring)")]
struct Cli {
    /// JSONL input, one request per line.  Every mode except --self-play
    /// requires it.
    #[arg(long)]
    input: Option<PathBuf>,
    /// JSONL output, one response per line.
    #[arg(long)]
    output: PathBuf,
    /// Canonical Fantasyland EV config.
    #[arg(long, default_value = "ai/config/fl_ev.json")]
    fl_ev_config: PathBuf,
    /// Roots solved per parallel batch before flushing output.
    #[arg(long, default_value_t = 256)]
    chunk_size: usize,
    /// Read {id, btn, pool} lines and emit only the shared joint block.
    #[arg(long, default_value_t = false)]
    joint_only: bool,
    /// Read T3 second-seat requests and evaluate them with the learned model.
    #[arg(long)]
    t3_second_model: Option<PathBuf>,
    /// Read T3-vs-FL requests and evaluate with the T4-vs-FL model image.
    #[arg(long)]
    t3_vs_fl_model: Option<PathBuf>,
    /// T3-vs-FL with direct FL-library scoring (no learned model in labels).
    #[arg(long, num_args = 1..)]
    t3_vs_fl_library: Option<Vec<PathBuf>>,
    /// T2-vs-FL playout labels: FL library dirs (used with --t2-t3-model).
    #[arg(long, num_args = 1..)]
    t2_vs_fl_library: Option<Vec<PathBuf>>,
    /// The exported 109-dim T3-vs-FL evaluator that chooses playout moves.
    #[arg(long)]
    t2_t3_model: Option<PathBuf>,
    /// T1-vs-FL playout labels: FL library dirs (with --t1-t2-model and
    /// --t2-t3-model supplying the two playout movers).  Pass the flag with
    /// no directories to take the opponents from --fl-pool instead.
    #[arg(long, num_args = 0..)]
    t1_vs_fl_library: Option<Vec<PathBuf>>,
    /// T2 labels against the best-responding pool, with --t2-t3-model handling
    /// the T3 street.  Each request's `truncate_depth` decides whether the
    /// model's value is the label or only its choice is; see
    /// `t2_vs_fl_pool.rs`.  Built to be compared against `fl_solver teach-t2`,
    /// which enumerates T3 instead.
    #[arg(long, default_value_t = false)]
    t2_vs_fl_pool: bool,
    /// The exported T2-vs-FL evaluator that chooses the T2 playout move.
    #[arg(long)]
    t1_t2_model: Option<PathBuf>,
    /// T0-vs-FL playout labels: FL library dirs (with --t0-t1-model,
    /// --t1-t2-model and --t2-t3-model supplying the three playout movers).
    /// Pass the flag with no directories to take the opponents from
    /// --fl-pool instead.
    #[arg(long, num_args = 0..)]
    t0_vs_fl_library: Option<Vec<PathBuf>>,
    /// The exported T1-vs-FL evaluator that chooses the T1 playout move.
    #[arg(long)]
    t0_t1_model: Option<PathBuf>,
    /// Count-specific FL libraries; counts without one fall back to the
    /// base (14-card) library, which is the pre-fix behavior.
    #[arg(long)]
    fl_library_15: Option<PathBuf>,
    #[arg(long)]
    fl_library_16: Option<PathBuf>,
    #[arg(long)]
    fl_library_17: Option<PathBuf>,
    /// A `.jfl1` pool of pre-solved Fantasyland best-response frontiers, used
    /// by --t0-vs-fl-library / --t1-vs-fl-library in place of a library shelf.
    /// The two are different games: a pooled opponent sets its thirteen after
    /// seeing hero's finished board, a library board was solved against
    /// nobody.
    #[arg(long)]
    fl_pool: Option<PathBuf>,
    /// Emit sampled joint-outlook blocks for {id, board, pool} requests.
    #[arg(long, default_value_t = false)]
    joint_outlook: bool,
    /// Play T0/T1/T2 with the three trained choosers over {id, cards[14]}
    /// lines and emit the T3 root each deal reaches, in the shape
    /// `fl_solver teach --roots-file` consumes.  Nothing is scored, so this
    /// wants no opponents and no --fl-pool.
    #[arg(long, default_value_t = false)]
    play_roots: bool,
    /// The chooser at each played street.  Named by the street that acts, not
    /// by the street that asked, because here no street is asking: the three
    /// models are the players, not a search's continuation.
    #[arg(long)]
    play_t0_model: Option<PathBuf>,
    #[arg(long)]
    play_t1_model: Option<PathBuf>,
    #[arg(long)]
    play_t2_model: Option<PathBuf>,
    /// Optional reached-position outputs used to refresh earlier-street
    /// teachers.  The normal --output remains the T3 roots file.
    #[arg(long)]
    play_t1_output: Option<PathBuf>,
    #[arg(long)]
    play_t2_output: Option<PathBuf>,
    /// Mirror self-play under the frozen session rules; writes one JSONL
    /// line per hand to --output.  Needs the three chooser models and no
    /// --input.  See `self_play.rs`.
    #[arg(long, default_value_t = false)]
    self_play: bool,
    #[arg(long, default_value_t = 100_000)]
    hands: usize,
    #[arg(long, default_value_t = 16)]
    workers: usize,
    #[arg(long, default_value_t = 0xC0FFEE51)]
    self_play_seed: u64,
    /// Weight the unseen pool by this trained discard model (JSON from
    /// ai/tutor/train_discard_model.py); absent, the pool is exchangeable.
    #[arg(long)]
    discard_model: Option<PathBuf>,
    /// T3-first (BB) HU teacher: half a street of opponent expansion, then
    /// the exact V4; see t3_first_hu.rs.  Needs --t2-t3-model as the
    /// opponent's chooser.
    #[arg(long, default_value_t = false)]
    t3_first_hu: bool,
    /// T3-second (BTN) HU teacher: every placement priced by the swapped
    /// exact V4; see t3_second_hu.rs.
    #[arg(long, default_value_t = false)]
    t3_second_hu: bool,
    /// Exact T4-first values/actions with both boards visible; see
    /// v4_first.rs.  Input rows: {id, board, dead, draw?, opp_board}.
    #[arg(long, default_value_t = false)]
    v4_first: bool,
    /// Collapse the exact V4 *state* sweep over rank classes when both
    /// boards are flush-dead; see rank_collapse.rs.  Read only by
    /// --t3-first-hu, --t3-second-hu and --v4-first, and inert everywhere
    /// else by construction -- no serve path and no encoder is wired to it.
    /// Off by default: turning it on is a labelling decision, and the fire
    /// counter on stderr says which way a shard ran.
    #[arg(long, default_value_t = false)]
    rank_collapse: bool,
    /// What a T3-boundary state is worth when it is played out: the
    /// independent check on V3s.  Needs --t2-t3-model as BB's T3 chooser.
    #[arg(long, default_value_t = false)]
    v3s_probe: bool,
    /// Head-up match: two chains playing the interleaved game, settled in
    /// real points.  Arm A is --hu-a-models (a comma-separated list of eight
    /// street/seat evaluators, T0bb,T0btn,T1bb,T1btn,T2bb,T2btn,T3bb,T3btn --
    /// T3btn is ignored, that seat is exact) or --arm-a-own for the
    /// generation-3 own-hand chain.  Same for B.
    #[arg(long)]
    hu_a_models: Option<String>,
    #[arg(long)]
    hu_b_models: Option<String>,
    /// Shortlist rankers, eight slots like the models; "none" allowed.
    /// Rankers are 207-dim no-joint retrains served on the joint-zeroed
    /// encoding, or 623-dim hybrids; either way no sampling is paid.
    #[arg(long)]
    hu_a_rankers: Option<String>,
    #[arg(long)]
    hu_b_rankers: Option<String>,
    /// Candidates the shortlist keeps for the full evaluator; 0 disables.
    #[arg(long, default_value_t = 0)]
    hu_topk: usize,
    /// The T0-BB policy net (54 -> 243 logits, T4F1 image from
    /// `ai/tutor/export_t0_policy.py`), per arm.  Given one, the T0
    /// first-actor shortlist is the policy's top --hu-t0-policy-topk and the
    /// T0-BB ranker is not called; every other node is untouched, and an arm
    /// without the flag behaves exactly as before.
    #[arg(long)]
    hu_a_t0_policy: Option<PathBuf>,
    #[arg(long)]
    hu_b_t0_policy: Option<PathBuf>,
    /// Openings the policy hands to the evaluator.
    #[arg(long, default_value_t = 8)]
    hu_t0_policy_topk: usize,
    /// Arm B override; topk 1 serves the policy argmax directly.
    #[arg(long)]
    hu_b_t0_policy_topk: Option<usize>,
    #[arg(long)]
    arm_a_own: Option<String>,
    #[arg(long)]
    arm_b_own: Option<String>,
    #[arg(long, default_value_t = false)]
    hu_match: bool,
    /// Play this many normal-versus-normal hands with arm A in both seats
    /// and write every placement, so the chain can be read rather than
    /// scored.  Uses the same --hu-a-models / --arm-a-own wiring.
    #[arg(long, default_value_t = 0)]
    hu_trace: usize,
    /// Emit the 623-dim hybrid vector instead of the 207-dim pair vector.
    #[arg(long, default_value_t = false)]
    hu_encode_hybrid: bool,
    /// Emit the 207-dim HU pair vector for {id, board, opp_board, dead}
    /// rows -- the parity harness against ai/tutor/encode_hu_teacher.py.
    #[arg(long, default_value_t = false)]
    hu_encode: bool,
    /// Deep-evaluate one decision from a traced hand: replay the history,
    /// force each candidate, and play the rest out against a re-dealt future.
    /// Needs --input (a trace jsonl), --replay-hand, --replay-street,
    /// --replay-seat, --rollouts.
    #[arg(long, default_value_t = false)]
    hu_deep_replay: bool,
    #[arg(long, default_value_t = 0)]
    replay_hand: u64,
    #[arg(long, default_value_t = 0)]
    replay_street: usize,
    #[arg(long, default_value_t = 0)]
    replay_seat: usize,
    /// Hands per trace checkpoint; each chunk is appended and flushed so a
    /// preempted shard keeps what it already played.
    #[arg(long, default_value_t = 500)]
    trace_chunk: usize,
    /// Serve-time joint samples for arm A (0 = the trained 400).  The whole
    /// cost of serving is these completions; lowering it is only sound if a
    /// match says the cheaper features cost no strength.
    #[arg(long, default_value_t = 0)]
    serve_joint_samples: usize,
    /// The same for arm B, so two counts can be seated against each other.
    #[arg(long, default_value_t = 0)]
    serve_joint_samples_b: usize,
    /// Mirror match: every deal played twice with the seats traded, so deal
    /// luck cancels in the pair.  Writes one row per deal; see
    /// hu_match::mirror_hands.  Uses --hands and --self-play-seed.
    #[arg(long, default_value_t = false)]
    hu_mirror: bool,
    /// Deep T0 evaluation for a specified first-actor (BB) hand: each
    /// candidate opening is forced and the hand rolls out to the end under
    /// both arms' serving, with common random numbers across candidates.
    /// One invocation is one batch; see hu_match::t0_deep_eval.
    #[arg(long, default_value_t = false)]
    hu_t0_deep: bool,
    /// The same audit for the other half of the serving surface: hero's T0
    /// against an opponent already in Fantasyland.  Hero's chain is the
    /// shipped own-hand one (--arm-a-own) and the score is hero's own
    /// finished board -- royalties, entry, foul -- with the opponent ignored
    /// (owner ruling 2026-08-30).  --rollouts 0 reports the chooser's own
    /// ranking of every opening instead of playing any.  See fl_t0_deep.rs.
    #[arg(long, default_value_t = false)]
    fl_t0_deep: bool,
    /// Gen-2 teacher: value each T0 opening by RACING T1 rather than letting
    /// the shipped chooser pick it, and harvest the race as T1 labels.  See
    /// fl_t0_teach.rs.
    #[arg(long, default_value_t = false)]
    fl_t0_teach: bool,
    /// The root's index in whatever ordering the caller mines; the T1 draws
    /// and every future are keyed on it, so two runs of the same root agree.
    #[arg(long, default_value_t = 0)]
    teach_root_index: u64,
    #[arg(long, default_value = "t0c-00000")]
    teach_root_id: String,
    /// T0 candidates raced, by the shipped width-120 net's own order.
    #[arg(long, default_value_t = 12)]
    teach_k: usize,
    /// T1 moves raced per node, by the T1 chooser's order.
    #[arg(long, default_value_t = 6)]
    teach_t1_top: usize,
    /// T1 draws shared across every T0 candidate.
    #[arg(long, default_value_t = 16)]
    teach_n1: usize,
    /// Cumulative particles per race round, comma-separated.
    #[arg(long, default_value = "8,24,64")]
    teach_schedule: String,
    #[arg(long, default_value_t = 64)]
    teach_winner_extra: usize,
    #[arg(long, default_value_t = 3.0)]
    teach_sigma: f64,
    #[arg(long, default_value_t = 8.0)]
    teach_floor: f64,
    /// Cumulative particles before the flat floor may retire a candidate.
    /// 0 means "the first round's count", which is what makes the floor
    /// reachable at teacher budgets.
    #[arg(long, default_value_t = 0)]
    teach_floor_min_n: usize,
    /// Where the harvested T1 rankings go; absent, they are dropped.
    #[arg(long)]
    teach_t1_output: Option<PathBuf>,
    /// Cascade the own chain's T2 street: this model (the cheap width-120
    /// chooser) pre-ranks every candidate and the serving evaluator argmaxes
    /// only the top --own-t2-fence-topk of them.  Absent, T2 is the
    /// full-field argmax it has always been.
    ///
    /// Offline on sharp labels the cascade is free at K=12 -- regret 0.1237
    /// against 0.1245 for the full field -- at half the encodes, and costs
    /// +0.003 at K=8 for 2.8x.  Any width `evaluator::Model::load` accepts is
    /// loadable here; 120 is the one the measurement was taken with.
    #[arg(long)]
    own_t2_fence: Option<PathBuf>,
    /// Candidates the fence lets through to the serving evaluator.
    #[arg(long, default_value_t = 12)]
    own_t2_fence_topk: usize,
    /// Accepted and ignored.  The Fantasyland opponent's width mattered while
    /// this mode settled hero against a dealt opponent; under the own-hand
    /// objective no opponent is dealt.  Kept so saved command lines and fleet
    /// metadata from before the ruling still run.
    #[arg(long, default_value_t = 14)]
    fl_opp_width: u8,
    /// The five dealt cards, comma-separated (e.g. "As,Kd,7c,7h,2s").
    #[arg(long)]
    t0_cards: Option<String>,
    /// Optional file of candidate keys (one per line, "top|middle|bottom"
    /// with cards sorted inside each row); absent, all openings run.
    #[arg(long)]
    t0_keys: Option<PathBuf>,
    /// The same keys inline, ';'-separated -- fleet workers get their
    /// arguments through metadata and have no keys file to read.
    #[arg(long)]
    t0_keys_inline: Option<String>,
    /// Rollouts per candidate in this batch.
    #[arg(long, default_value_t = 200)]
    rollouts: usize,
    /// Two-seat synchronized traces for the HU phase; see hu_traces.rs.
    #[arg(long, default_value_t = false)]
    hu_traces: bool,
    /// Player B's choosers for an A/B match; absent, B mirrors A.
    #[arg(long)]
    arm_b_t0_model: Option<PathBuf>,
    #[arg(long)]
    arm_b_t1_model: Option<PathBuf>,
    #[arg(long)]
    arm_b_t2_model: Option<PathBuf>,
    /// Emit `playout::encode_for`'s vector for {id, board, pool} requests,
    /// under the model given by --t1-t2-model.  The parity harness
    /// `ai/tutor/fl14_encoder_parity.py` drives the shipped encoder through
    /// this rather than a copy of it: a copy would agree with Python while the
    /// playout disagreed, which is the failure the harness exists to catch.
    #[arg(long, default_value_t = false)]
    encode_features: bool,
    /// Fantasyland-only simulator: hero is pinned in Fantasyland at a fixed
    /// width, the opponent runs its natural state machine, chains of stays are
    /// measured episodically.  One invocation = one pass under one fl_ev table;
    /// the fixed-point iteration lives in ai/tutor/fl_sim_driver.py.  Reuses
    /// --play-t0/t1/t2-model (the opponent's own-hand chain), --self-play-seed,
    /// --fl-ev-config and --output; wants no HU serving flags and no --input.
    /// Ignores --workers: blocks run on the global rayon pool, so thread count
    /// is set with RAYON_NUM_THREADS and never changes the output bytes.
    #[arg(long, default_value_t = false)]
    fl_sim: bool,
    /// The entry width hero re-enters at, 14..=17.
    #[arg(long, default_value_t = 0)]
    fl_sim_width: u8,
    /// Independent blocks; block b is a deterministic function of (seed, b).
    #[arg(long, default_value_t = 500)]
    fl_sim_blocks: u64,
    /// Complete chains per block.  Chain 0 of every block starts the opponent
    /// at normal and is flagged burn_in; the headline uses chains 1..K-1.
    #[arg(long, default_value_t = 8)]
    fl_sim_chains_per_block: u32,
}

/// One board and the pool it is judged against, for --encode-features.
#[derive(Deserialize)]
struct EncodeRequest {
    id: String,
    board: BoardStr,
    pool: Vec<String>,
    /// Seed for the sampled joint block; defaults to `id`, matching how
    /// `--joint-outlook` treats an absent seed.
    #[serde(default)]
    seed: Option<String>,
    /// Only read by widths whose tail carries the opponent's card count.
    #[serde(default = "default_encode_opp_count")]
    opp_count: u8,
}

fn default_encode_opp_count() -> u8 {
    14
}

#[derive(Serialize)]
struct EncodeResponse {
    id: String,
    input_dim: usize,
    features: Vec<f32>,
}

/// The width the shipped pool was built for.  Requests naming any other
/// opponent card count are refused per root rather than scored against a
/// Fantasyland hand of the wrong size.
const POOL_WIDTH: u32 = 14;

/// The opponents a `--tN-vs-fl-library` run prices its leaves against.
///
/// Refuses both sources at once rather than preferring one: a run handed a
/// library shelf and a pool has not said which game it meant, and the
/// difference is the whole rule correction, not a precision setting.
fn opponents_for<'a>(
    flag: &str,
    dirs: &[PathBuf],
    extra: [Option<&PathBuf>; 3],
    pool: Option<&'a fl_solver::pool::Pool>,
    slot: &'a mut Option<t3_vs_fl_lib::LibrarySet>,
) -> Result<playout::OpponentSource<'a>> {
    match (pool, dirs.is_empty()) {
        (Some(pool), true) => Ok(playout::OpponentSource::Pool(pool)),
        (Some(_), false) => bail!(
            "{flag} was given library directories and --fl-pool was given too; \
             pass one or the other"
        ),
        (None, true) => bail!(
            "{flag} needs library directories unless --fl-pool supplies the \
             opponents"
        ),
        (None, false) => Ok(playout::OpponentSource::Libraries(
            slot.insert(t3_vs_fl_lib::LibrarySet::load(dirs, extra)?),
        )),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let fl_ev = FlEv::load(&cli.fl_ev_config)?;
    // The T2 fence, loaded once ahead of the modes because it is a serving
    // option rather than a mode of its own.  Absent the flag nothing is read,
    // and a mode that does not thread it simply never looks.  It is announced
    // when present: a cascaded chain is a different chain, and a run that
    // changed what it served without saying so is the expensive kind of quiet.
    let own_t2_fence_model: Option<evaluator::Model> = match &cli.own_t2_fence {
        Some(path) => {
            if cli.own_t2_fence_topk == 0 {
                bail!("--own-t2-fence-topk 0 would leave the evaluator no candidate to choose");
            }
            let image = std::fs::read(path)?;
            let model = evaluator::Model::load(&image)
                .map_err(|e| anyhow!("{}: {e}", path.display()))?;
            eprintln!(
                "own-t2-fence: {} dims, top-{} -> the T2 evaluator",
                model.input_dim, cli.own_t2_fence_topk
            );
            Some(model)
        }
        None => None,
    };
    let own_t2_fence: Option<(&evaluator::Model, usize)> = own_t2_fence_model
        .as_ref()
        .map(|model| (model, cli.own_t2_fence_topk));
    if cli.fl_t0_teach {
        let spec = cli
            .arm_a_own
            .as_deref()
            .ok_or_else(|| anyhow!("--fl-t0-teach requires --arm-a-own t0.bin,t1.bin,t2.bin"))?;
        let paths: Vec<&str> = spec.split(',').map(str::trim).collect();
        if paths.len() != 3 {
            bail!("--arm-a-own names three models (T0,T1,T2), got {}", paths.len());
        }
        let loaded: Vec<evaluator::Model> = paths
            .iter()
            .map(|path| {
                let image = std::fs::read(path)?;
                evaluator::Model::load(&image).map_err(|e| anyhow!("{path}: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let models = [&loaded[0], &loaded[1], &loaded[2]];
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let cards: Vec<String> = cli
            .t0_cards
            .as_deref()
            .ok_or_else(|| anyhow!("--fl-t0-teach requires --t0-cards"))?
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let config = fl_t0_teach::Config {
            top_k: cli.teach_k,
            t1_top: cli.teach_t1_top,
            n1: cli.teach_n1,
            schedule: cli
                .teach_schedule
                .split(',')
                .map(|p| p.trim().parse::<usize>())
                .collect::<std::result::Result<Vec<_>, _>>()?,
            winner_extra: cli.teach_winner_extra,
            sigma: cli.teach_sigma,
            floor: cli.teach_floor,
            floor_min_n: if cli.teach_floor_min_n > 0 {
                cli.teach_floor_min_n
            } else {
                cli.teach_schedule
                    .split(',')
                    .next()
                    .and_then(|p| p.trim().parse::<usize>().ok())
                    .unwrap_or(8)
            },
            root_index: cli.teach_root_index,
        };
        let started = std::time::Instant::now();
        let (t0_labels, t1_labels, hands) = fl_t0_teach::teach_root(
            &cli.teach_root_id, &cards, &config, &fl_ev, &fl_table, models, own_t2_fence,
        )?;
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        for row in &t0_labels {
            writeln!(writer, "{}", serde_json::to_string(row)?)?;
        }
        writer.flush()?;
        if let Some(path) = &cli.teach_t1_output {
            let mut writer = BufWriter::new(File::create(path)?);
            for row in &t1_labels {
                writeln!(writer, "{}", serde_json::to_string(row)?)?;
            }
            writer.flush()?;
        }
        for row in t0_labels.iter().take(6) {
            eprintln!(
                "teach {:+8.3} +-{:.3} n={} rank{:<3} {}",
                row.mean, row.se, row.n, row.t0_rank, row.key
            );
        }
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "fl-t0-teach: {} K={} N1={} t1top={} sched={} extra={} | {} hands,              {:.1} s ({:.2} min/root, {:.1} ms/hand) | {} T1 labels -> {}",
            cli.teach_root_id, config.top_k, config.n1, config.t1_top,
            cli.teach_schedule, config.winner_extra, hands, elapsed, elapsed / 60.0,
            elapsed * 1000.0 / hands.max(1) as f64, t1_labels.len(), cli.output.display()
        );
        return Ok(());
    }
    if cli.fl_t0_deep {
        // No HU arm here: against a face-down Fantasyland board the pair
        // encoders have nothing to read, so the own-hand chain is the whole
        // serving path and --arm-a-own names all three of its models.
        let spec = cli
            .arm_a_own
            .as_deref()
            .ok_or_else(|| anyhow!("--fl-t0-deep requires --arm-a-own t0.bin,t1.bin,t2.bin"))?;
        let paths: Vec<&str> = spec.split(',').map(str::trim).collect();
        if paths.len() != 3 {
            bail!("--arm-a-own names three models (T0,T1,T2), got {}", paths.len());
        }
        let loaded: Vec<evaluator::Model> = paths
            .iter()
            .map(|path| {
                let image = std::fs::read(path)?;
                evaluator::Model::load(&image).map_err(|e| anyhow!("{path}: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let models = [&loaded[0], &loaded[1], &loaded[2]];
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let cards: Vec<String> = cli
            .t0_cards
            .as_deref()
            .ok_or_else(|| anyhow!("--fl-t0-deep requires --t0-cards"))?
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        if cli.rollouts == 0 {
            // The served opening is this argmax outright -- there is no ranker
            // or policy fence on the own-hand path -- so own_rank 1 is the
            // decision the rollout mode is asked to referee.
            let ranked = fl_t0_deep::rank(&cards, &fl_ev, &fl_table, models[0])?;
            for row in &ranked {
                writeln!(writer, "{}", serde_json::to_string(row)?)?;
            }
            writer.flush()?;
            eprintln!(
                "fl-t0-rank: {} openings, chooser {} dims, top {} {:+.4} -> {}",
                ranked.len(),
                models[0].input_dim,
                ranked.first().map(|r| r.key.as_str()).unwrap_or("-"),
                ranked.first().map(|r| r.score).unwrap_or(0.0),
                cli.output.display()
            );
            return Ok(());
        }
        let wanted: Option<Vec<String>> = match (&cli.t0_keys, &cli.t0_keys_inline) {
            (Some(_), Some(_)) => bail!("--t0-keys and --t0-keys-inline are one flag too many"),
            (Some(path), None) => Some(
                std::fs::read_to_string(path)?
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect(),
            ),
            (None, Some(inline)) => Some(
                inline
                    .split(';')
                    .map(|k| k.trim().to_string())
                    .filter(|k| !k.is_empty())
                    .collect(),
            ),
            (None, None) => None,
        };
        let started = std::time::Instant::now();
        let rows = fl_t0_deep::deep_eval(
            &cards,
            wanted.as_deref(),
            cli.rollouts,
            cli.self_play_seed,
            &fl_ev,
            &fl_table,
            models,
            own_t2_fence,
        )?;
        for row in &rows {
            writeln!(writer, "{}", serde_json::to_string(row)?)?;
        }
        writer.flush()?;
        for row in rows.iter().take(10) {
            eprintln!("fl-t0-deep {:+8.3}  n={}  {}", row.mean, row.n, row.key);
        }
        // Two rates because a batch is priced two ways: a rollout is one deal
        // across the whole field (what a seed buys), a hand is one candidate
        // in one rollout (what the field costs).
        let played = (rows.len() * cli.rollouts).max(1);
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "fl-t0-deep: {} candidates x {} rollouts, own-worth objective, \
             seed {:#x}, {:.1} s wall ({:.3} s/rollout, {:.1} ms/hand) -> {}",
            rows.len(),
            cli.rollouts,
            cli.self_play_seed,
            elapsed,
            elapsed / cli.rollouts.max(1) as f64,
            elapsed * 1000.0 / played as f64,
            cli.output.display()
        );
        return Ok(());
    }
    if cli.joint_outlook {
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("this mode requires --input"))?)?);
        let mut requests: Vec<joint_outlook::JointOutlookRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let solved: Result<Vec<String>> = chunk
                .par_iter()
                .map(|request| {
                    Ok(serde_json::to_string(&joint_outlook::solve(
                        request, &fl_ev,
                    )?)?)
                })
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
        }
        return Ok(());
    }

    if cli.t3_first_hu {
        let chooser_path = cli
            .t2_t3_model
            .as_ref()
            .ok_or_else(|| anyhow!("--t3-first-hu requires --t2-t3-model as the opponent chooser"))?;
        let image = std::fs::read(chooser_path)?;
        let opp_chooser = evaluator::Model::load(&image).map_err(|e| anyhow!("{e}"))?;
        eprintln!("t3-first-hu: opponent chooser {} dims", opp_chooser.input_dim);
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("--t3-first-hu requires --input"))?)?);
        let mut requests: Vec<t3_first_hu::T3FirstHuRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        let started = std::time::Instant::now();
        let mut done = 0usize;
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let solved: Result<Vec<String>> = chunk
                .par_iter()
                .map(|request| {
                    Ok(serde_json::to_string(&t3_first_hu::solve(
                        request,
                        &fl_ev,
                        &fl_table,
                        &opp_chooser,
                        cli.rank_collapse,
                    )?)?)
                })
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
            done += chunk.len();
            eprintln!(
                "[{done}/{}] {:.0} ms/root",
                requests.len(),
                started.elapsed().as_secs_f64() * 1000.0 / done as f64
            );
        }
        rank_collapse::report(cli.rank_collapse);
        return Ok(());
    }

    if cli.t3_second_hu {
        let discard = cli
            .discard_model
            .as_ref()
            .map(|path| discard_model::DiscardModel::load(path))
            .transpose()?;
        if discard.is_some() {
            eprintln!("t3-second-hu: discard-weighted pool");
        }
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("--t3-second-hu requires --input"))?)?);
        let mut requests: Vec<t3_second_hu::T3SecondHuRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        let started = std::time::Instant::now();
        let mut done = 0usize;
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let solved: Result<Vec<String>> = chunk
                .par_iter()
                .map(|request| {
                    Ok(serde_json::to_string(&t3_second_hu::solve(
                        request,
                        &fl_ev,
                        discard.as_ref(),
                        cli.rank_collapse,
                    )?)?)
                })
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
            done += chunk.len();
            eprintln!(
                "[{done}/{}] {:.0} ms/root",
                requests.len(),
                started.elapsed().as_secs_f64() * 1000.0 / done as f64
            );
        }
        rank_collapse::report(cli.rank_collapse);
        return Ok(());
    }

    if cli.v4_first {
        let table = [
            fl_ev.value(14),
            fl_ev.value(15),
            fl_ev.value(16),
            fl_ev.value(17),
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("--v4-first requires --input"))?)?);
        let mut requests: Vec<v4_first::V4FirstRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        let started = std::time::Instant::now();
        let mut done = 0usize;
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let solved: Result<Vec<String>> = chunk
                .par_iter()
                .map(|request| {
                    Ok(serde_json::to_string(&v4_first::solve_maybe_collapsed(
                        request,
                        &table,
                        None,
                        cli.rank_collapse,
                    )?)?)
                })
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
            done += chunk.len();
            eprintln!(
                "[{done}/{}] {:.1} ms/state",
                requests.len(),
                started.elapsed().as_secs_f64() * 1000.0 / done as f64
            );
        }
        rank_collapse::report(cli.rank_collapse);
        return Ok(());
    }

    if cli.v3s_probe {
        let model_path = cli
            .t2_t3_model
            .as_ref()
            .ok_or_else(|| anyhow!("--v3s-probe requires --t2-t3-model (BB's T3 evaluator)"))?;
        let image = std::fs::read(model_path)?;
        let bb_model = evaluator::Model::load(&image).map_err(|e| anyhow!("{e}"))?;
        eprintln!("v3s-probe: BB chooser {} dims", bb_model.input_dim);
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("--v3s-probe requires --input"))?)?);
        let mut requests: Vec<v3s_probe::V3sProbeRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        let started = std::time::Instant::now();
        let mut done = 0usize;
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let solved: Result<Vec<String>> = chunk
                .iter()
                .map(|request| {
                    Ok(serde_json::to_string(&v3s_probe::solve(
                        request, &fl_ev, &fl_table, &bb_model,
                    )?)?)
                })
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
            done += chunk.len();
            eprintln!(
                "[{done}/{}] {:.0} ms/state",
                requests.len(),
                started.elapsed().as_secs_f64() * 1000.0 / done as f64
            );
        }
        return Ok(());
    }

    if cli.hu_match {
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let load = |path: &str| -> Result<evaluator::Model> {
            let image = std::fs::read(path)?;
            evaluator::Model::load(&image).map_err(|e| anyhow!("{path}: {e}"))
        };
        // Own-hand arms name three models, HU arms name eight.
        let mut owned: Vec<Vec<evaluator::Model>> = Vec::new();
        for spec in [
            &cli.arm_a_own, &cli.arm_b_own, &cli.hu_a_models, &cli.hu_b_models,
            &cli.hu_a_rankers, &cli.hu_b_rankers,
        ] {
            owned.push(match spec {
                Some(list) => list
                    .split(',')
                    .map(|p| p.trim())
                    .filter(|p| *p != "none")
                    .map(load)
                    .collect::<Result<Vec<_>>>()?,
                None => Vec::new(),
            });
        }
        // The T0-BB policy nets, if either arm carries one.  Loaded through
        // `load_wide` because their last layer is 243 logits rather than the
        // one score every other image here ends in.
        let load_policy = |path: Option<&PathBuf>| -> Result<Option<evaluator::Model>> {
            match path {
                None => Ok(None),
                Some(path) => {
                    let image = std::fs::read(path)?;
                    let model = evaluator::Model::load_wide(&image)
                        .map_err(|e| anyhow!("{}: {e}", path.display()))?;
                    eprintln!(
                        "t0-policy: {} ({} -> {}), K={}",
                        path.display(), model.input_dim, model.output_dim,
                        cli.hu_t0_policy_topk
                    );
                    Ok(Some(model))
                }
            }
        };
        let t0_policy_a = load_policy(cli.hu_a_t0_policy.as_ref())?;
        let t0_policy_b = load_policy(cli.hu_b_t0_policy.as_ref())?;
        // Every arm needs own-hand choosers: they play the hands where the
        // opponent is in Fantasyland and its board is face down.
        let eight_slots = |slot: usize, spec: Option<&str>| -> Result<Vec<Option<&evaluator::Model>>> {
            let m = &owned[slot];
            if m.is_empty() {
                return Ok(Vec::new());
            }
            let names: Vec<&str> = spec.unwrap_or("").split(',').collect();
            if names.len() != 8 {
                bail!("eight slots (T0bb..T3btn) expected, \"none\" allowed");
            }
            let mut cursor = 0usize;
            let mut out: Vec<Option<&evaluator::Model>> = Vec::with_capacity(8);
            for name in &names {
                if name.trim() == "none" {
                    out.push(None);
                } else {
                    out.push(Some(&m[cursor]));
                    cursor += 1;
                }
            }
            Ok(out)
        };
        fn pair_up<'m>(
            per_slot: Vec<Option<&'m evaluator::Model>>,
        ) -> Vec<Option<[&'m evaluator::Model; 2]>> {
            if per_slot.is_empty() {
                return Vec::new();
            }
            (0..4)
                .map(|street| match (per_slot[street * 2], per_slot[street * 2 + 1]) {
                    (Some(bb), Some(btn)) => Some([bb, btn]),
                    _ => None,
                })
                .collect()
        }
        let build = |own_slot: usize, hu_slot: usize| -> Result<hu_match::Arm<'_>> {
            let o = &owned[own_slot];
            if o.len() != 3 {
                bail!("every arm names three own-hand models (T0,T1,T2)");
            }
            // Eight slots, T0bb..T3btn; the literal "none" leaves a street
            // to the own-hand chooser, which is what a half-built ladder
            // needs.
            let hu = if owned[hu_slot].is_empty() {
                Vec::new()
            } else {
                let m = &owned[hu_slot];
                let names: Vec<&str> = match hu_slot {
                    2 => cli.hu_a_models.as_deref().unwrap_or("").split(',').collect(),
                    _ => cli.hu_b_models.as_deref().unwrap_or("").split(',').collect(),
                };
                if names.len() != 8 {
                    bail!("an HU arm names eight slots (T0bb..T3btn), \"none\" allowed");
                }
                let mut cursor = 0usize;
                let mut per_slot: Vec<Option<&evaluator::Model>> = Vec::with_capacity(8);
                for name in &names {
                    if name.trim() == "none" {
                        per_slot.push(None);
                    } else {
                        per_slot.push(Some(&m[cursor]));
                        cursor += 1;
                    }
                }
                (0..4)
                    .map(|street| match (per_slot[street * 2], per_slot[street * 2 + 1]) {
                        (Some(bb), Some(btn)) => Some([bb, btn]),
                        _ => None,
                    })
                    .collect()
            };
            let ranker_spec = match hu_slot {
                2 => (4usize, cli.hu_a_rankers.as_deref()),
                _ => (5usize, cli.hu_b_rankers.as_deref()),
            };
            let rankers = pair_up(eight_slots(ranker_spec.0, ranker_spec.1)?);
            Ok(hu_match::Arm {
                hu,
                own: [&o[0], &o[1], &o[2]],
                rankers,
                topk: cli.hu_topk,
                t0_policy: if hu_slot == 2 {
                    t0_policy_a.as_ref()
                } else {
                    t0_policy_b.as_ref()
                },
                policy_topk: if hu_slot == 2 {
                    cli.hu_t0_policy_topk
                } else {
                    cli.hu_b_t0_policy_topk.unwrap_or(cli.hu_t0_policy_topk)
                },
                joint_samples: if hu_slot == 2 {
                    cli.serve_joint_samples
                } else {
                    cli.serve_joint_samples_b
                },
            })
        };
        let arm_a = build(0, 2)?;
        let arm_b = build(1, 3)?;
        if cli.hu_deep_replay {
            let path = cli.input.as_ref()
                .ok_or_else(|| anyhow!("--hu-deep-replay requires --input (a trace)"))?;
            let mut steps: Vec<(usize, usize, Vec<String>, [Vec<String>; 3])> = Vec::new();
            for line in BufReader::new(File::open(path)?).lines() {
                let line = line?;
                if line.trim().is_empty() { continue; }
                let hand: serde_json::Value = serde_json::from_str(&line)?;
                if hand["hand"].as_u64() != Some(cli.replay_hand) { continue; }
                for step in hand["steps"].as_array()
                    .ok_or_else(|| anyhow!("trace has no steps"))? {
                    let draw: Vec<String> = serde_json::from_value(step["draw"].clone())?;
                    let board: [Vec<String>; 3] = serde_json::from_value(step["board"].clone())?;
                    steps.push((
                        step["street"].as_u64().unwrap() as usize,
                        step["seat"].as_u64().unwrap() as usize,
                        draw, board,
                    ));
                }
            }
            if steps.is_empty() {
                bail!("hand {} not found in {}", cli.replay_hand, path.display());
            }
            let wanted: Option<Vec<String>> = cli.t0_keys_inline.as_ref().map(|inline| {
                inline.split(';').map(|k| k.trim().to_string())
                    .filter(|k| !k.is_empty()).collect()
            });
            let rows = hu_match::deep_replay(
                &steps, cli.replay_street, cli.replay_seat, wanted.as_deref(),
                cli.rollouts, cli.self_play_seed, &[arm_a, arm_b], &fl_ev, &fl_table,
            )?;
            let mut writer = BufWriter::new(File::create(&cli.output)?);
            for row in &rows {
                writeln!(writer, "{}", serde_json::to_string(row)?)?;
            }
            writer.flush()?;
            for row in rows.iter().take(12) {
                eprintln!("replay {:+8.3}  n={}  {}", row.mean, row.n, row.key);
            }
            eprintln!(
                "hu-deep-replay: hand {} T{} seat {} -- {} candidates x {} rollouts -> {}",
                cli.replay_hand, cli.replay_street, cli.replay_seat,
                rows.len(), cli.rollouts, cli.output.display()
            );
            return Ok(());
        }
        if cli.hu_mirror {
            let rows = hu_match::mirror_hands(
                cli.hands, cli.self_play_seed, &[arm_a, arm_b], &fl_ev, &fl_table,
            )?;
            let mut writer = BufWriter::new(File::create(&cli.output)?);
            for row in &rows {
                writeln!(writer, "{}", serde_json::to_string(row)?)?;
            }
            writer.flush()?;
            let settled: i64 = rows.iter().map(|r| (r.settle[0] + r.settle[1]) as i64).sum();
            eprintln!(
                "hu-mirror: {} deals x2, mean line settlement {:+.4}/hand, seed {:#x} -> {}",
                rows.len(),
                settled as f64 / (2.0 * rows.len().max(1) as f64),
                cli.self_play_seed,
                cli.output.display()
            );
            return Ok(());
        }
        if cli.hu_t0_deep {
            let cards: Vec<String> = cli
                .t0_cards
                .as_deref()
                .ok_or_else(|| anyhow!("--hu-t0-deep requires --t0-cards"))?
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if cli.rollouts == 0 {
                // The model's own ranking, for auditing a model-first sieve.
                // `rank`/`score` are the evaluator's; the ranker columns say
                // which openings serving would have let it see at all.
                let scored = hu_match::t0_model_scores(&cards, &arm_a, &fl_ev, &fl_table)?;
                let mut writer = BufWriter::new(File::create(&cli.output)?);
                for (rank, row) in scored.iter().enumerate() {
                    writeln!(
                        writer,
                        "{}",
                        serde_json::json!({
                            "rank": rank + 1,
                            "key": &row.key,
                            "score": row.score,
                            "ranker_score": row.ranker_score,
                            "ranker_rank": row.ranker_rank,
                            "policy_score": row.policy_score,
                            "policy_rank": row.policy_rank,
                        })
                    )?;
                }
                writer.flush()?;
                // The served opening: the shortlist keeps K, the evaluator
                // picks among those.  With no shortlist every row passes the
                // filter and this is the evaluator's own first row.  A policy
                // arm shortlists by `policy_rank` and never by the ranker.
                let ranked = scored.iter().filter(|row| row.ranker_rank.is_some()).count();
                let by_policy = scored.iter().filter(|row| row.policy_rank.is_some()).count();
                let served = scored
                    .iter()
                    .filter(|row| match row.policy_rank {
                        Some(rank) => rank <= cli.hu_t0_policy_topk,
                        None => row.ranker_rank.is_none_or(|rank| rank <= cli.hu_topk),
                    })
                    .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
                eprintln!(
                    "t0-model-rank: {} openings, {} ranker-scored (K={}), \
                     {} policy-scored (K={}) -> {}",
                    scored.len(), ranked, cli.hu_topk,
                    by_policy, cli.hu_t0_policy_topk, cli.output.display()
                );
                if let (Some(top), Some(served)) = (scored.first(), served) {
                    eprintln!(
                        "t0-model-rank: evaluator {} {:+.4} | served {} {:+.4}",
                        top.key, top.score, served.key, served.score
                    );
                }
                return Ok(());
            }
            let wanted: Option<Vec<String>> = match (&cli.t0_keys, &cli.t0_keys_inline) {
                (Some(_), Some(_)) => {
                    bail!("--t0-keys and --t0-keys-inline are one flag too many")
                }
                (Some(path), None) => Some(
                    std::fs::read_to_string(path)?
                        .lines()
                        .map(|l| l.trim().to_string())
                        .filter(|l| !l.is_empty())
                        .collect(),
                ),
                (None, Some(inline)) => Some(
                    inline
                        .split(';')
                        .map(|k| k.trim().to_string())
                        .filter(|k| !k.is_empty())
                        .collect(),
                ),
                (None, None) => None,
            };
            let rows = hu_match::t0_deep_eval(
                &cards,
                wanted.as_deref(),
                cli.rollouts,
                cli.self_play_seed,
                &[arm_a, arm_b],
                &fl_ev,
                &fl_table,
            )?;
            let mut writer = BufWriter::new(File::create(&cli.output)?);
            for row in &rows {
                writeln!(writer, "{}", serde_json::to_string(row)?)?;
            }
            writer.flush()?;
            for row in rows.iter().take(10) {
                eprintln!("t0-deep {:+8.3}  n={}  {}", row.mean, row.n, row.key);
            }
            eprintln!(
                "hu-t0-deep: {} candidates x {} rollouts, seed {:#x} -> {}",
                rows.len(),
                cli.rollouts,
                cli.self_play_seed,
                cli.output.display()
            );
            return Ok(());
        }
        if cli.hu_trace > 0 {
            // Checkpointed: each chunk is appended and flushed, and the line
            // count is announced so the shard script can ship the partial.
            // A trace used to write only at the end, so a preemption at hour
            // two threw away every hand of the shard.
            let chunk = cli.trace_chunk.max(1);
            let arms = [arm_a, arm_b];
            let mut writer = BufWriter::new(File::create(&cli.output)?);
            let mut done = 0usize;
            while done < cli.hu_trace {
                let take = chunk.min(cli.hu_trace - done);
                let traced = hu_match::trace_range(
                    done as u64, take, cli.self_play_seed, &arms,
                    &fl_ev, &fl_table,
                )?;
                for hand in traced {
                    writeln!(writer, "{}", serde_json::to_string(&hand)?)?;
                }
                writer.flush()?;
                done += take;
                eprintln!("hu-trace: {done}/{} hands written", cli.hu_trace);
            }
            eprintln!("hu-trace: {} hands -> {}", cli.hu_trace, cli.output.display());
            return Ok(());
        }
        let config = hu_match::MatchConfig {
            hands: cli.hands,
            workers: cli.workers,
            seed: cli.self_play_seed,
            stack: 200,
            gap: 40,
        };
        hu_match::run(&config, &[arm_a, arm_b], &fl_ev, &fl_table, &cli.output)?;
        return Ok(());
    }

    if cli.hu_encode {
        #[derive(Deserialize)]
        struct HuEncodeRequest {
            id: String,
            board: BoardStr,
            opp_board: BoardStr,
            #[serde(default)]
            dead: Vec<String>,
            #[serde(default)]
            seed: Option<String>,
        }
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("--hu-encode requires --input"))?)?);
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let request: HuEncodeRequest = serde_json::from_str(&line)?;
            let own = CoreBoard::from_str_board(&request.board)?;
            let opp = CoreBoard::from_str_board(&request.opp_board)?;
            let mut seen: std::collections::BTreeSet<String> = request.dead.iter().cloned().collect();
            for name in request
                .board
                .top
                .iter()
                .chain(&request.board.middle)
                .chain(&request.board.bottom)
                .chain(&request.opp_board.top)
                .chain(&request.opp_board.middle)
                .chain(&request.opp_board.bottom)
            {
                seen.insert(name.clone());
            }
            let pool_names: Vec<String> = all_cards()
                .into_iter()
                .filter(|n| !seen.contains(n))
                .collect();
            let pool: Vec<Card> = pool_names
                .iter()
                .map(|n| to_core_card(n))
                .collect::<Result<Vec<_>>>()?;
            let memo: playout::RowwiseMemo =
                std::sync::Mutex::new(std::collections::HashMap::new());
            let mut features: Vec<f32> = Vec::new();
            if cli.hu_encode_hybrid {
                let own_rows = [
                    request.board.top.clone(),
                    request.board.middle.clone(),
                    request.board.bottom.clone(),
                ];
                let opp_rows = [
                    request.opp_board.top.clone(),
                    request.opp_board.middle.clone(),
                    request.opp_board.bottom.clone(),
                ];
                hu_encode::encode_hybrid(
                    &own_rows, &opp_rows, &request.dead, &pool_names,
                    &own, &opp, &pool, &fl_ev, &fl_table, &memo, 0, &mut features,
                );
                writeln!(
                    writer,
                    "{}",
                    serde_json::json!({"id": request.id, "vector": features})
                )?;
                continue;
            }
            hu_encode::encode_pair(
                &own,
                &opp,
                &pool,
                &fl_ev,
                &fl_table,
                &memo,
                0,
                request.seed.as_deref().unwrap_or(&request.id),
                &mut features,
            )?;
            writeln!(
                writer,
                "{}",
                serde_json::json!({"id": request.id, "features": features})
            )?;
        }
        writer.flush()?;
        return Ok(());
    }

    if cli.hu_traces {
        let model_of = |flag: &str, path: &Option<PathBuf>| -> Result<evaluator::Model> {
            let path = path
                .as_ref()
                .ok_or_else(|| anyhow!("--hu-traces requires {flag}"))?;
            let image = std::fs::read(path)?;
            evaluator::Model::load(&image).map_err(|e| anyhow!("{} : {e}", path.display()))
        };
        let t0 = model_of("--play-t0-model", &cli.play_t0_model)?;
        let t1 = model_of("--play-t1-model", &cli.play_t1_model)?;
        let t2 = model_of("--play-t2-model", &cli.play_t2_model)?;
        eprintln!(
            "hu-traces: choosers {} / {} / {} dims, {} hands, seed {:#x}",
            t0.input_dim, t1.input_dim, t2.input_dim, cli.hands, cli.self_play_seed
        );
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        hu_traces::run(
            cli.hands,
            cli.self_play_seed,
            cli.workers,
            &fl_ev,
            &fl_table,
            &t0,
            &t1,
            &t2,
            &cli.output,
        )?;
        return Ok(());
    }

    if cli.fl_sim {
        let model_of = |flag: &str, path: &Option<PathBuf>| -> Result<evaluator::Model> {
            let path = path
                .as_ref()
                .ok_or_else(|| anyhow!("--fl-sim requires {flag}"))?;
            let image = std::fs::read(path)?;
            evaluator::Model::load(&image).map_err(|e| anyhow!("{} : {e}", path.display()))
        };
        let t0 = model_of("--play-t0-model", &cli.play_t0_model)?;
        let t1 = model_of("--play-t1-model", &cli.play_t1_model)?;
        let t2 = model_of("--play-t2-model", &cli.play_t2_model)?;
        // The paths ride along for the metadata line only: a run whose models
        // cannot be named afterwards cannot be compared with another run.
        let named = |path: &Option<PathBuf>| -> String {
            path.as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        };
        let config = fl_sim::Config {
            width: cli.fl_sim_width,
            blocks: cli.fl_sim_blocks,
            chains_per_block: cli.fl_sim_chains_per_block,
            seed: cli.self_play_seed,
            model_paths: [
                named(&cli.play_t0_model),
                named(&cli.play_t1_model),
                named(&cli.play_t2_model),
            ],
        };
        fl_sim::run(&config, &fl_ev, &t0, &t1, &t2, &cli.output)?;
        return Ok(());
    }

    if cli.self_play {
        let model_of = |flag: &str, path: &Option<PathBuf>| -> Result<evaluator::Model> {
            let path = path
                .as_ref()
                .ok_or_else(|| anyhow!("--self-play requires {flag}"))?;
            let image = std::fs::read(path)?;
            evaluator::Model::load(&image).map_err(|e| anyhow!("{} : {e}", path.display()))
        };
        let t0 = model_of("--play-t0-model", &cli.play_t0_model)?;
        let t1 = model_of("--play-t1-model", &cli.play_t1_model)?;
        let t2 = model_of("--play-t2-model", &cli.play_t2_model)?;
        eprintln!(
            "self-play: choosers {} / {} / {} dims, {} hands, seed {:#x}",
            t0.input_dim, t1.input_dim, t2.input_dim, cli.hands, cli.self_play_seed
        );
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let config = self_play::Config {
            hands: cli.hands,
            workers: cli.workers,
            seed: cli.self_play_seed,
            stack: 200,
            gap: 40,
        };
        let b_models: Option<(evaluator::Model, evaluator::Model, evaluator::Model)> =
            match (&cli.arm_b_t0_model, &cli.arm_b_t1_model, &cli.arm_b_t2_model) {
                (None, None, None) => None,
                (Some(_), Some(_), Some(_)) => Some((
                    model_of("--arm-b-t0-model", &cli.arm_b_t0_model)?,
                    model_of("--arm-b-t1-model", &cli.arm_b_t1_model)?,
                    model_of("--arm-b-t2-model", &cli.arm_b_t2_model)?,
                )),
                _ => bail!("an A/B match names all three of B's models or none"),
            };
        let arm_a = self_play::Arm { t0: &t0, t1: &t1, t2: &t2 };
        let arm_b = match &b_models {
            Some((b0, b1, b2)) => {
                eprintln!(
                    "self-play A/B: arm B choosers {} / {} / {} dims",
                    b0.input_dim, b1.input_dim, b2.input_dim
                );
                self_play::Arm { t0: b0, t1: b1, t2: b2 }
            }
            None => self_play::Arm { t0: &t0, t1: &t1, t2: &t2 },
        };
        self_play::run(&config, &fl_ev, &fl_table, &[arm_a, arm_b], &cli.output)?;
        return Ok(());
    }

    if cli.play_roots {
        let model_of = |flag: &str, path: &Option<PathBuf>| -> Result<evaluator::Model> {
            let path = path
                .as_ref()
                .ok_or_else(|| anyhow!("--play-roots requires {flag}"))?;
            let image =
                std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
            evaluator::Model::load(&image).map_err(|e| anyhow!("{} : {e}", path.display()))
        };
        let t0_model = model_of("--play-t0-model", &cli.play_t0_model)?;
        let t1_model = model_of("--play-t1-model", &cli.play_t1_model)?;
        let t2_model = model_of("--play-t2-model", &cli.play_t2_model)?;
        eprintln!(
            "play-roots: choosers {} / {} / {} dims",
            t0_model.input_dim, t1_model.input_dim, t2_model.input_dim
        );
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("this mode requires --input"))?)?);
        let mut requests: Vec<play_roots::PlayRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        let mut t1_writer = cli
            .play_t1_output
            .as_ref()
            .map(File::create)
            .transpose()?
            .map(BufWriter::new);
        let mut t2_writer = cli
            .play_t2_output
            .as_ref()
            .map(File::create)
            .transpose()?
            .map(BufWriter::new);
        let started = std::time::Instant::now();
        let mut streets = play_roots::StreetSeconds::default();
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let played: Result<Vec<(String, String, String, play_roots::StreetSeconds)>> = chunk
                .par_iter()
                .map(|request| {
                    let (trace, timing) = play_roots::play_trace(
                        request, &fl_ev, &fl_table, &t0_model, &t1_model, &t2_model,
                    )?;
                    Ok((
                        serde_json::to_string(&trace.t1)?,
                        serde_json::to_string(&trace.t2)?,
                        serde_json::to_string(&trace.t3)?,
                        timing,
                    ))
                })
                .collect();
            for (t1_line, t2_line, t3_line, timing) in played? {
                if let Some(target) = t1_writer.as_mut() {
                    writeln!(target, "{t1_line}")?;
                }
                if let Some(target) = t2_writer.as_mut() {
                    writeln!(target, "{t2_line}")?;
                }
                writeln!(writer, "{t3_line}")?;
                streets.add(&timing);
            }
            writer.flush()?;
            if let Some(target) = t1_writer.as_mut() {
                target.flush()?;
            }
            if let Some(target) = t2_writer.as_mut() {
                target.flush()?;
            }
        }
        let deals = requests.len().max(1) as f64;
        // Wall time is what a run costs; the per-street figures are summed
        // across threads, so they answer "where did the work go" and not
        // "how long did it take".
        eprintln!(
            "play-roots: {} deals, {:.1} s wall ({:.3} s/deal); thread-seconds \
             per deal t0 {:.4} t1 {:.4} t2 {:.4}",
            requests.len(),
            started.elapsed().as_secs_f64(),
            started.elapsed().as_secs_f64() / deals,
            streets.t0 / deals,
            streets.t1 / deals,
            streets.t2 / deals,
        );
        return Ok(());
    }

    if cli.encode_features {
        let model_path = cli
            .t1_t2_model
            .as_ref()
            .ok_or_else(|| anyhow!("--encode-features needs --t1-t2-model to fix the width"))?;
        let image = std::fs::read(model_path)?;
        let model = evaluator::Model::load(&image).map_err(|e| anyhow!("{e}"))?;
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("this mode requires --input"))?)?);
        let mut requests: Vec<EncodeRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let encoded: Result<Vec<String>> = chunk
                .par_iter()
                .map(|request| {
                    let board = CoreBoard::from_str_board(&request.board)?;
                    let pool: Vec<Card> = request
                        .pool
                        .iter()
                        .map(|name| to_core_card(name))
                        .collect::<Result<Vec<_>>>()?;
                    let memo: playout::RowwiseMemo =
                        std::sync::Mutex::new(std::collections::HashMap::new());
                    let mut features: Vec<f32> = Vec::new();
                    playout::encode_for(
                        &model,
                        &board,
                        &pool,
                        request.opp_count,
                        &fl_ev,
                        &fl_table,
                        &memo,
                        0,
                        request.seed.as_deref().unwrap_or(&request.id),
                        &mut features,
                        // One board per request: nothing to share it with.
                        None,
                    )?;
                    Ok(serde_json::to_string(&EncodeResponse {
                        id: request.id.clone(),
                        input_dim: model.input_dim,
                        features,
                    })?)
                })
                .collect();
            for line in encoded? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
        }
        return Ok(());
    }

    if cli.fl_pool.is_some()
        && cli.t0_vs_fl_library.is_none()
        && cli.t1_vs_fl_library.is_none()
        && !cli.t2_vs_fl_pool
    {
        bail!("--fl-pool only feeds --t0-vs-fl-library, --t1-vs-fl-library and --t2-vs-fl-pool");
    }
    // The header pins the whole fl_ev table and refuses to load under any
    // other, so handing it this run's own config makes a config that has
    // drifted from the pool a load error rather than a leaf scoring entries
    // against numbers they were not solved for.
    let pool = match &cli.fl_pool {
        Some(path) => {
            let bytes = std::fs::read(path)
                .with_context(|| format!("cannot read FL pool {}", path.display()))?;
            let table = [
                fl_ev.value(14),
                fl_ev.value(15),
                fl_ev.value(16),
                fl_ev.value(17),
            ];
            let loaded = fl_solver::pool::deserialize(&bytes, POOL_WIDTH, table)
                .map_err(|error| anyhow!("{} : {error}", path.display()))?;
            eprintln!(
                "fl pool: {} entries, width {}, seed {:#x}",
                loaded.entries.len(),
                loaded.width,
                loaded.seed
            );
            Some(loaded)
        }
        None => None,
    };

    if let Some(library_dirs) = &cli.t0_vs_fl_library {
        let t1_path = cli
            .t0_t1_model
            .as_ref()
            .ok_or_else(|| anyhow!("--t0-vs-fl-library requires --t0-t1-model"))?;
        let t2_path = cli
            .t1_t2_model
            .as_ref()
            .ok_or_else(|| anyhow!("--t0-vs-fl-library requires --t1-t2-model"))?;
        let t3_path = cli
            .t2_t3_model
            .as_ref()
            .ok_or_else(|| anyhow!("--t0-vs-fl-library requires --t2-t3-model"))?;
        let t1_image = std::fs::read(t1_path)?;
        let t2_image = std::fs::read(t2_path)?;
        let t3_image = std::fs::read(t3_path)?;
        let t1_model = evaluator::Model::load(&t1_image).map_err(|e| anyhow!("{e}"))?;
        let t2_model = evaluator::Model::load(&t2_image).map_err(|e| anyhow!("{e}"))?;
        let t3_model = evaluator::Model::load(&t3_image).map_err(|e| anyhow!("{e}"))?;
        let mut library_slot = None;
        let source = opponents_for(
            "--t0-vs-fl-library",
            library_dirs,
            [
                cli.fl_library_15.as_ref(),
                cli.fl_library_16.as_ref(),
                cli.fl_library_17.as_ref(),
            ],
            pool.as_ref(),
            &mut library_slot,
        )?;
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("this mode requires --input"))?)?);
        let mut requests: Vec<t0_vs_fl::T0VsFlRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let solved: Result<Vec<String>> = chunk
                .iter()
                .map(|request| {
                    Ok(serde_json::to_string(&t0_vs_fl::solve(
                        request, &fl_ev, &source, &fl_table, &t1_model, &t2_model, &t3_model,
                    )?)?)
                })
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
        }
        return Ok(());
    }

    if let Some(library_dirs) = &cli.t1_vs_fl_library {
        let t2_path = cli
            .t1_t2_model
            .as_ref()
            .ok_or_else(|| anyhow!("--t1-vs-fl-library requires --t1-t2-model"))?;
        let t3_path = cli
            .t2_t3_model
            .as_ref()
            .ok_or_else(|| anyhow!("--t1-vs-fl-library requires --t2-t3-model"))?;
        let t2_image = std::fs::read(t2_path)?;
        let t3_image = std::fs::read(t3_path)?;
        let t2_model = evaluator::Model::load(&t2_image).map_err(|e| anyhow!("{e}"))?;
        let t3_model = evaluator::Model::load(&t3_image).map_err(|e| anyhow!("{e}"))?;
        let mut library_slot = None;
        let source = opponents_for(
            "--t1-vs-fl-library",
            library_dirs,
            [
                cli.fl_library_15.as_ref(),
                cli.fl_library_16.as_ref(),
                cli.fl_library_17.as_ref(),
            ],
            pool.as_ref(),
            &mut library_slot,
        )?;
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("this mode requires --input"))?)?);
        let mut requests: Vec<t1_vs_fl::T1VsFlRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let solved: Result<Vec<String>> = chunk
                .iter()
                .map(|request| {
                    Ok(serde_json::to_string(&t1_vs_fl::solve(
                        request, &fl_ev, &source, &fl_table, &t2_model, &t3_model,
                    )?)?)
                })
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
        }
        return Ok(());
    }

    if let Some(library_dirs) = &cli.t2_vs_fl_library {
        let model_path = cli
            .t2_t3_model
            .as_ref()
            .ok_or_else(|| anyhow!("--t2-vs-fl-library requires --t2-t3-model"))?;
        let image = std::fs::read(model_path)?;
        let t3_model = evaluator::Model::load(&image).map_err(|e| anyhow!("{e}"))?;
        let library = t3_vs_fl_lib::LibrarySet::load(
            library_dirs,
            [
                cli.fl_library_15.as_ref(),
                cli.fl_library_16.as_ref(),
                cli.fl_library_17.as_ref(),
            ],
        )?;
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("this mode requires --input"))?)?);
        let mut requests: Vec<t2_vs_fl::T2VsFlRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let solved: Result<Vec<String>> = chunk
                .iter()
                .map(|request| {
                    Ok(serde_json::to_string(&t2_vs_fl::solve(
                        request, &fl_ev, &library, &fl_table, &t3_model,
                    )?)?)
                })
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
        }
        return Ok(());
    }

    if cli.t2_vs_fl_pool {
        let pool = pool
            .as_ref()
            .ok_or_else(|| anyhow!("--t2-vs-fl-pool requires --fl-pool"))?;
        let model_path = cli
            .t2_t3_model
            .as_ref()
            .ok_or_else(|| anyhow!("--t2-vs-fl-pool requires --t2-t3-model"))?;
        let image = std::fs::read(model_path)?;
        let t3_model = evaluator::Model::load(&image).map_err(|e| anyhow!("{e}"))?;
        eprintln!("t2-vs-fl-pool: T3 chooser {} dims", t3_model.input_dim);
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("this mode requires --input"))?)?);
        let mut requests: Vec<t2_vs_fl_pool::T2VsFlPoolRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        let started = std::time::Instant::now();
        let mut done = 0usize;
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let solved: Result<Vec<String>> = chunk
                .iter()
                .map(|request| {
                    Ok(serde_json::to_string(&t2_vs_fl_pool::solve(
                        request, &fl_ev, pool, &fl_table, &t3_model,
                    )?)?)
                })
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
            done += chunk.len();
            eprintln!(
                "[{done}/{}] {:.3} s/root",
                requests.len(),
                started.elapsed().as_secs_f64() / done as f64
            );
        }
        return Ok(());
    }

    if let Some(library_dirs) = &cli.t3_vs_fl_library {
        let library = t3_vs_fl_lib::LibrarySet::load(
            library_dirs,
            [
                cli.fl_library_15.as_ref(),
                cli.fl_library_16.as_ref(),
                cli.fl_library_17.as_ref(),
            ],
        )?;
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("this mode requires --input"))?)?);
        let mut requests: Vec<t3_vs_fl::T3VsFlRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let solved: Result<Vec<String>> = chunk
                .iter()
                .map(|request| {
                    Ok(serde_json::to_string(&t3_vs_fl_lib::solve(
                        request, &fl_ev, &library, &fl_table,
                    )?)?)
                })
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
        }
        return Ok(());
    }

    if let Some(model_path) = &cli.t3_vs_fl_model {
        let image = std::fs::read(model_path)?;
        let model = evaluator::Model::load(&image).map_err(|e| anyhow!("{e}"))?;
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("this mode requires --input"))?)?);
        let mut requests: Vec<t3_vs_fl::T3VsFlRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let solved: Result<Vec<String>> = chunk
                .iter()
                .map(|request| {
                    Ok(serde_json::to_string(&t3_vs_fl::solve(
                        request, &fl_ev, &model, &fl_table,
                    )?)?)
                })
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
        }
        return Ok(());
    }

    if let Some(model_path) = &cli.t3_second_model {
        let image = std::fs::read(model_path)?;
        let model = evaluator::Model::load(&image).map_err(|e| anyhow!("{e}"))?;
        let fl_table: evaluator::FlTable = [
            fl_ev.value(14) as f32,
            fl_ev.value(15) as f32,
            fl_ev.value(16) as f32,
            fl_ev.value(17) as f32,
        ];
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("this mode requires --input"))?)?);
        let mut requests: Vec<t3_second::T3Request> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            // Roots carry the parallelism here; the per-action work inside is
            // then sequential, which avoids nested rayon contention.
            let solved: Result<Vec<String>> = chunk
                .par_iter()
                .map(|request| {
                    Ok(serde_json::to_string(&t3_second::solve(
                        request, &fl_ev, &model, &fl_table,
                    )?)?)
                })
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
        }
        return Ok(());
    }

    if cli.joint_only {
        let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("this mode requires --input"))?)?);
        let mut requests: Vec<JointRequest> = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                requests.push(serde_json::from_str(&line)?);
            }
        }
        let mut writer = BufWriter::new(File::create(&cli.output)?);
        for chunk in requests.chunks(cli.chunk_size.max(1)) {
            let solved: Result<Vec<String>> = chunk
                .par_iter()
                .map(|request| Ok(serde_json::to_string(&solve_joint_only(request, &fl_ev)?)?))
                .collect();
            for line in solved? {
                writeln!(writer, "{line}")?;
            }
            writer.flush()?;
        }
        return Ok(());
    }

    let reader = BufReader::new(File::open(cli.input.as_ref().ok_or_else(|| anyhow!("this mode requires --input"))?)?);
    let mut requests: Vec<Request> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        requests.push(serde_json::from_str(&line)?);
    }

    // Root-level parallelism: teacher generation is many independent roots, and
    // the per-action rayon inside `solve` cannot saturate the machine on roots
    // with only three legal actions.
    let mut writer = BufWriter::new(File::create(&cli.output)?);
    for chunk in requests.chunks(cli.chunk_size.max(1)) {
        let solved: Result<Vec<String>> = chunk
            .par_iter()
            .map(|request| {
                let response = solve(request, &fl_ev)?;
                Ok(serde_json::to_string(&response)?)
            })
            .collect();
        for line in solved? {
            writeln!(writer, "{line}")?;
        }
        writer.flush()?;
    }
    Ok(())
}
