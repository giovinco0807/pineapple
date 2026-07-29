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

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use ofc_core::{
    evaluate_hand_value,
    check_fl_entry, compare_3_hands, compare_5_hands, evaluate_board_with_joker_constraint,
    get_bottom_royalty, get_middle_royalty, get_top_royalty, Card,
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

fn all_cards() -> Vec<String> {
    let mut cards = Vec::with_capacity(54);
    for suit in ["s", "h", "d", "c"] {
        for rank in ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"] {
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

fn to_core_card(card: &str) -> Result<Card> {
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

struct FlEv {
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
struct BoardStr {
    top: Vec<String>,
    middle: Vec<String>,
    bottom: Vec<String>,
}

#[derive(Clone)]
struct CoreBoard {
    rows: [Vec<Card>; 3],
}

impl CoreBoard {
    fn from_str_board(board: &BoardStr) -> Result<Self> {
        let mut rows = [Vec::new(), Vec::new(), Vec::new()];
        for (index, cards) in [&board.top, &board.middle, &board.bottom].iter().enumerate() {
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
struct Terminal {
    busted: bool,
    royalty: i32,
    fl_card_count: u8,
    values: [u32; 3],
}

fn terminal_of(board: &CoreBoard) -> Terminal {
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
struct Action {
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

fn legal_actions(board: &CoreBoard, draw: &[String]) -> Vec<Action> {
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

fn apply(board: &CoreBoard, action: &Action) -> Result<CoreBoard> {
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
struct ActionResult {
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
fn opponent_self_value(terminal: &Terminal, fl_ev: &FlEv) -> f64 {
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
                for (first, second) in [
                    (draw[0], draw[1]),
                    (draw[0], draw[2]),
                    (draw[1], draw[2]),
                ] {
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
        for (first, second) in [
            (draw[0], draw[1]),
            (draw[0], draw[2]),
            (draw[1], draw[2]),
        ] {
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
            survive_royalty += best_parts.0;
            survive_fl += best_parts.1;
        }
        best_self.push(best);
    }
    let count = best_self.len() as f64;
    let mean = best_self.iter().sum::<f64>() / count;
    let variance =
        best_self.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / count;
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
        (best_self.iter().filter(|v| **v > -6.0).sum::<f64>() / survive_denominator)
            / MAX_ROYALTY,
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

#[derive(Parser)]
#[command(about = "Exact T4 first-seat solver (canonical FL EV, f64 scoring)")]
struct Cli {
    /// JSONL input, one request per line.
    #[arg(long)]
    input: PathBuf,
    /// JSONL output, one response per line.
    #[arg(long)]
    output: PathBuf,
    /// Canonical Fantasyland EV config.
    #[arg(long, default_value = "ai/config/fl_ev.json")]
    fl_ev_config: PathBuf,
    /// Roots solved per parallel batch before flushing output.
    #[arg(long, default_value_t = 256)]
    chunk_size: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let fl_ev = FlEv::load(&cli.fl_ev_config)?;
    let reader = BufReader::new(File::open(&cli.input)?);
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
