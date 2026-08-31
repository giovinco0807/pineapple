//! Feature encoder and dense-net inference for the T4 first-seat evaluator.
//!
//! Semantics are pinned to `ai/tutor/t4_first_features.py` and
//! `ai/tutor/train_t4_first_evaluator.py`; the parity test drives this module
//! against those.  Only the hero and joint blocks are computed here per action
//! -- the opponent and context blocks are node-shared and supplied by the
//! caller, which is what keeps the per-action cost to a few microseconds.

use anyhow::{anyhow, bail, Result};
use ofc_core::{
    check_fl_entry, evaluate_board_with_joker_constraint, evaluate_hand_value, get_bottom_royalty,
    get_middle_royalty, get_top_royalty, Card,
};

pub const CATEGORIES: usize = 9;
pub const HERO_SIZE: usize = 42;
pub const OPPONENT_SIZE: usize = 49;
/// The T3 acting seat's own-board block (t3_second_features.actor_block).
pub const ACTOR_SIZE: usize = 48;
/// FL14 v2's suffix: two made-rank tiebreaks per row.
pub const ALLOCATION_RANK_SIZE: usize = 6;
pub const JOINT_SIZE: usize = 12;
pub const CONTEXT_SIZE: usize = 6;
/// The FL14 teachers' deck block (encode_fl14_teacher.context_block).
pub const FL14_CONTEXT_SIZE: usize = 7;
pub const FEATURE_SIZE: usize = HERO_SIZE + OPPONENT_SIZE + JOINT_SIZE + CONTEXT_SIZE;

const ROW_CAPACITY: [usize; 3] = [3, 5, 5];
const MAX_ROYALTY: f32 = 25.0;
const MAX_FL_EV: f32 = 63.5;
const B: u32 = 15;

/// Category index of an encoded hand value, matching `hand_category`.
fn category_of(value: u32) -> usize {
    (value / B.pow(5)) as usize
}

fn row_royalty(row: usize, cards: &[Card]) -> i32 {
    match row {
        0 => get_top_royalty(cards),
        1 => get_middle_royalty(cards),
        _ => get_bottom_royalty(cards),
    }
}

/// Category one-hot plus the two leading rank tiebreaks.
fn spread(value: u32, out: &mut Vec<f32>) {
    let category = category_of(value).min(CATEGORIES - 1);
    let start = out.len();
    out.resize(start + CATEGORIES + 2, 0.0);
    out[start + category] = 1.0;
    out[start + CATEGORIES] = ((value / B.pow(4)) % B) as f32 / 14.0;
    out[start + CATEGORIES + 1] = ((value / B.pow(3)) % B) as f32 / 14.0;
}

/// Category of an incomplete row by rank multiplicity, jokers counted wild.
/// An incomplete row cannot hold a straight or flush yet, so this is exact
/// rather than approximate, and a sound lower bound on the finished row.
pub fn partial_category(cards: &[Card], capacity: usize) -> usize {
    if cards.is_empty() {
        return 0;
    }
    if cards.len() == capacity {
        return category_of(evaluate_hand_value(cards, capacity)).min(CATEGORIES - 1);
    }
    let mut counts = [0u8; 15];
    let mut jokers = 0u8;
    for card in cards {
        if card.is_joker() {
            jokers += 1;
        } else {
            counts[card.rank as usize] += 1;
        }
    }
    let best = counts.iter().copied().max().unwrap_or(0) + jokers;
    let pairs = counts.iter().filter(|count| **count >= 2).count();
    if best >= 4 {
        7
    } else if best == 3 && pairs >= 2 {
        6
    } else if best >= 3 {
        3
    } else if pairs >= 2 {
        2
    } else if best == 2 {
        1
    } else {
        0
    }
}

fn rank_with_at_least(counts: &[u8; 15], copies: u8, exclude: Option<usize>) -> usize {
    (2usize..=14)
        .rev()
        .find(|rank| Some(*rank) != exclude && counts[*rank] >= copies)
        .unwrap_or(0)
}

/// Two category-aware leading ranks for a placed, possibly-open row.
///
/// Complete rows reuse the exact encoded hand tiebreaks.  On an open row,
/// jokers join the largest natural rank group (highest rank breaks a tie),
/// after which pair/trips/two-pair ranks precede kickers.  This mirrors
/// `fl14_allocation_features.partial_tiebreaks` and deliberately leaves the
/// historical 48-dim actor block unchanged.
fn partial_tiebreaks(cards: &[Card], capacity: usize) -> (f32, f32) {
    if cards.is_empty() {
        return (0.0, 0.0);
    }
    if cards.len() == capacity {
        let value = evaluate_hand_value(cards, capacity);
        return (
            ((value / B.pow(4)) % B) as f32 / 14.0,
            ((value / B.pow(3)) % B) as f32 / 14.0,
        );
    }

    let category = partial_category(cards, capacity);
    let mut counts = [0u8; 15];
    let mut jokers = 0u8;
    for card in cards {
        if card.is_joker() {
            jokers += 1;
        } else {
            counts[card.rank as usize] += 1;
        }
    }
    if let Some(anchor) = (2usize..=14)
        .max_by_key(|rank| (counts[*rank], *rank))
        .filter(|rank| counts[*rank] > 0)
    {
        counts[anchor] += jokers;
    } else if jokers > 0 {
        counts[14] = jokers;
    }

    let (first, second) = match category {
        0 => {
            let first = rank_with_at_least(&counts, 1, None);
            (first, rank_with_at_least(&counts, 1, Some(first)))
        }
        1 => {
            let first = rank_with_at_least(&counts, 2, None);
            (first, rank_with_at_least(&counts, 1, Some(first)))
        }
        2 => {
            let first = rank_with_at_least(&counts, 2, None);
            (first, rank_with_at_least(&counts, 2, Some(first)))
        }
        3 => {
            let first = rank_with_at_least(&counts, 3, None);
            (first, rank_with_at_least(&counts, 1, Some(first)))
        }
        6 => {
            let first = rank_with_at_least(&counts, 3, None);
            (first, rank_with_at_least(&counts, 2, Some(first)))
        }
        7 => {
            let first = rank_with_at_least(&counts, 4, None);
            (first, rank_with_at_least(&counts, 1, Some(first)))
        }
        _ => {
            let first = rank_with_at_least(&counts, 1, None);
            (first, rank_with_at_least(&counts, 1, Some(first)))
        }
    };
    (first as f32 / 14.0, second as f32 / 14.0)
}

/// Rank-aware FL14 v2 suffix, appended after the unchanged 104 v1 columns.
pub fn allocation_rank_block(rows: &[Vec<Card>; 3], out: &mut Vec<f32>) {
    for row in 0..3 {
        let tiebreaks = partial_tiebreaks(&rows[row], ROW_CAPACITY[row]);
        out.push(tiebreaks.0);
        out.push(tiebreaks.1);
    }
}

pub const CHEAP_DRAW_SIZE: usize = 16;

/// The five-rank runs a straight can occupy, wheel first.
const STRAIGHT_WINDOWS: [[u8; 5]; 10] = [
    [14, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
    [5, 6, 7, 8, 9],
    [6, 7, 8, 9, 10],
    [7, 8, 9, 10, 11],
    [8, 9, 10, 11, 12],
    [9, 10, 11, 12, 13],
    [10, 11, 12, 13, 14],
];

/// How full a row already is toward one five-rank window, as a fraction of
/// five, taking the best window.
///
/// A window scores zero unless every placed natural in the row lies inside it
/// with a distinct rank: a row holding a card outside the run, or a pair,
/// cannot become that straight at all, and scoring it on the cards that do fit
/// would report progress toward a hand the row can no longer make.  Jokers
/// count toward every window, having no rank to conflict.  With `single_suit`
/// the naturals must also share a suit, which turns the same walk into
/// straight-flush (and, over one window, royal) progress.
fn window_occupancy(cards: &[Card], jokers: u32, single_suit: bool, windows: &[[u8; 5]]) -> f32 {
    let naturals: Vec<&Card> = cards.iter().filter(|card| !card.is_joker()).collect();
    if single_suit {
        if let Some(first) = naturals.first() {
            if naturals.iter().any(|card| card.suit != first.suit) {
                return 0.0;
            }
        }
    }
    let mut best = 0.0f32;
    for window in windows {
        let mut filled = [false; 5];
        let mut inside = 0u32;
        let mut reachable = true;
        for card in &naturals {
            match window.iter().position(|rank| *rank == card.rank) {
                Some(slot) if !filled[slot] => {
                    filled[slot] = true;
                    inside += 1;
                }
                // A duplicate rank or a card outside the run kills the window.
                _ => {
                    reachable = false;
                    break;
                }
            }
        }
        if reachable {
            best = best.max((inside + jokers) as f32 / 5.0);
        }
    }
    best.min(1.0)
}

/// Suit counts of a row's naturals, and the suit it is closest to a flush in.
///
/// Ties break on the lowest suit index -- arbitrary, but it has to be written
/// down and obeyed identically on both sides, because the liveness dim reads
/// the deck through whichever suit this returns.
fn best_suit(cards: &[Card]) -> (u32, Option<usize>) {
    let mut counts = [0u32; 4];
    for card in cards {
        if !card.is_joker() {
            counts[card.suit as usize] += 1;
        }
    }
    let mut best = None;
    for suit in 0..4 {
        if counts[suit] > 0 && (best.is_none() || counts[suit] > counts[best.unwrap()]) {
            best = Some(suit);
        }
    }
    (best.map_or(0, |suit| counts[suit]), best)
}

pub const CHEAP_DRAW_V2_SIZE: usize = 24;

/// Best straight window for a row, as (fill fraction, outs filling the gaps).
///
/// Ties on fill break toward the window with more outs and then on window
/// order, so the pair is a function of the board rather than of iteration
/// luck -- the same rule the Python twin applies by comparing the tuple.
fn best_window(cards: &[Card], jokers: u32, unseen_rank: &[u32; 15]) -> (f64, f64) {
    let naturals: Vec<&Card> = cards.iter().filter(|card| !card.is_joker()).collect();
    let mut best = (-1.0f64, -1.0f64);
    for window in STRAIGHT_WINDOWS.iter() {
        let mut filled = [false; 5];
        let mut count = 0u32;
        let mut reachable = true;
        for card in &naturals {
            match window.iter().position(|rank| *rank == card.rank) {
                Some(slot) if !filled[slot] => {
                    filled[slot] = true;
                    count += 1;
                }
                _ => {
                    reachable = false;
                    break;
                }
            }
        }
        if !reachable {
            continue;
        }
        let outs: u32 = window
            .iter()
            .enumerate()
            .filter(|(slot, _)| !filled[*slot])
            .map(|(_, rank)| unseen_rank[*rank as usize])
            .sum();
        let cand = (
            ((count + jokers) as f64 / 5.0).min(1.0),
            outs as f64 / 20.0,
        );
        if cand > best {
            best = cand;
        }
    }
    if best.0 < 0.0 {
        (0.0, 0.0)
    } else {
        best
    }
}

/// Deterministic draw descriptors, second design: continuous and per-rank.
///
/// v1 (`cheap_draw_block`) reached the right serving cost but quantised
/// everything to multiples of 1/5, 1/2, 1/3 and 1/8, so distinct openings
/// collided: on the audit holdout, 14 of 97 roots had the referee's best
/// placement carrying a vector identical to a strictly worse one, which no
/// amount of training can separate.  The sampled 110-dim block collided on
/// none -- not because it sampled, but because it emitted continuous
/// high-entropy values.
///
/// So this keeps the determinism and drops the buckets: raw sums of ranks and
/// squared ranks per row (which nearly determine which cards went where), a
/// suit signature, raw out-counts for flushes, straights and pairs, and the
/// top row's Fantasyland material.  Measured on the same holdout it collides
/// on zero roots and zero candidates.
///
/// Arithmetic is f64 throughout and narrowed only on the way out: the
/// divisors here (70, 980, 13, 20, 9) are not exactly representable, so
/// computing in f32 would round differently from the Python twin's doubles
/// and break parity in the last ulp.
///
/// Mirrored in `ai/tutor/cheap_draw_features.py::cheap_draw_block_v2`.
pub fn cheap_draw_block_v2(rows: &[Vec<Card>; 3], unseen: &[Card], out: &mut Vec<f32>) {
    let mut unseen_suit = [0u32; 4];
    let mut unseen_rank = [0u32; 15];
    for card in unseen {
        if !card.is_joker() {
            unseen_suit[card.suit as usize] += 1;
            unseen_rank[card.rank as usize] += 1;
        }
    }
    let mut jokers = [0u32; 3];
    for row in 0..3 {
        jokers[row] = rows[row].iter().filter(|card| card.is_joker()).count() as u32;
    }
    let natural = |row: usize| -> Vec<&Card> {
        rows[row].iter().filter(|card| !card.is_joker()).collect()
    };

    // Rank and suit content per row.  Two openings that differ only by which
    // card went where differ here, which is exactly what v1 could not see.
    for row in 0..3 {
        let nat = natural(row);
        let sum: u32 = nat.iter().map(|c| c.rank as u32).sum();
        let squares: u32 = nat.iter().map(|c| (c.rank as u32) * (c.rank as u32)).sum();
        let top = nat.iter().map(|c| c.rank).max();
        let suits: u32 = nat.iter().map(|c| c.suit as u32 + 1).sum();
        out.push((sum as f64 / 70.0) as f32);
        out.push((squares as f64 / 980.0) as f32);
        out.push(top.map_or(0.0, |rank| (rank as f64 / 14.0) as f32));
        out.push((suits as f64 / 20.0) as f32);
    }
    // Flush outs for middle and bottom: the raw count, and it weighted by how
    // far the row already is.
    for row in 1..3 {
        let (count, suit) = best_suit(&rows[row]);
        let outs = suit.map_or(0, |suit| unseen_suit[suit]);
        out.push((outs as f64 / 13.0) as f32);
        out.push((((count + jokers[row]) as f64 / 5.0) * (outs as f64 / 13.0)) as f32);
    }
    // Straight fill and the outs that would complete it.
    for row in 1..3 {
        let (fill, outs) = best_window(&rows[row], jokers[row], &unseen_rank);
        out.push(fill as f32);
        out.push(outs as f32);
    }
    // Pairing outs per row.
    for row in 0..3 {
        let outs: u32 = natural(row)
            .iter()
            .map(|card| unseen_rank[card.rank as usize])
            .sum();
        out.push((outs as f64 / 9.0) as f32);
    }
    // The top row's Fantasyland material: copies still live of the queens-up
    // ranks it already holds.
    let top = natural(0);
    let queens_up: u32 = (12..=14u8)
        .filter(|rank| top.iter().any(|card| card.rank == *rank))
        .map(|rank| unseen_rank[rank as usize])
        .sum();
    out.push((queens_up as f64 / 9.0) as f32);
}

/// Deterministic draw descriptors: what each row is building toward, and how
/// much of the deck still supports it.
///
/// Sixteen dims, every one a count over the board and the unseen pool.  This
/// exists because the 110-dim encoder bought its accuracy with a
/// 400-completion sampled joint block: about 50 ms per candidate, which the
/// no-thinking-time-at-serve rule forbids (owner, 2026-08-30).  Most of what
/// that block summarised -- can these rows still reach a flush, a straight, a
/// Fantasyland top, and does the deck still hold the cards for it -- is
/// reachable by counting, and counting is free.
///
/// Mirrored dim for dim in `ai/tutor/cheap_draw_features.py`.  The two are
/// checked against each other on every audited vector, because a drift
/// between them serves a model a vector it never trained on, and nothing
/// downstream would report it.
pub fn cheap_draw_block(rows: &[Vec<Card>; 3], unseen: &[Card], out: &mut Vec<f32>) {
    let mut unseen_suit = [0u32; 4];
    let mut unseen_rank = [0u32; 15];
    for card in unseen {
        if !card.is_joker() {
            unseen_suit[card.suit as usize] += 1;
            unseen_rank[card.rank as usize] += 1;
        }
    }
    let mut jokers = [0u32; 3];
    for row in 0..3 {
        jokers[row] = rows[row].iter().filter(|card| card.is_joker()).count() as u32;
    }

    // 0-1 flush progress, 2-3 flush liveness, for middle and bottom.  The top
    // row holds three cards and cannot make a flush, so it is not asked.
    let mut suited = [(0u32, None); 3];
    for row in 1..3 {
        suited[row] = best_suit(&rows[row]);
    }
    for row in 1..3 {
        out.push((suited[row].0 + jokers[row]) as f32 / 5.0);
    }
    for row in 1..3 {
        out.push(match suited[row].1 {
            Some(suit) => (unseen_suit[suit] as f32 / 8.0).min(1.0),
            // No natural in the row, so no suit is established yet.
            None => 0.0,
        });
    }
    // 4-5 straight, 6-7 straight flush, for middle and bottom.
    for row in 1..3 {
        out.push(window_occupancy(&rows[row], jokers[row], false, &STRAIGHT_WINDOWS));
    }
    for row in 1..3 {
        out.push(window_occupancy(&rows[row], jokers[row], true, &STRAIGHT_WINDOWS));
    }
    // 8 the royal run, bottom only: the one window worth its own dim.
    out.push(window_occupancy(&rows[2], jokers[2], true, &STRAIGHT_WINDOWS[9..10]));
    // 9-11 jokers per row.
    for row in 0..3 {
        out.push(jokers[row] as f32 / 2.0);
    }
    // 12-14 the top row, which is what Fantasyland entry is decided on.
    let top_max = rows[0]
        .iter()
        .filter(|card| !card.is_joker())
        .map(|card| card.rank)
        .max();
    out.push(top_max.map_or(0.0, |rank| rank as f32 / 14.0));
    let mut queens_up = 0u32;
    for rank in 12..=14u8 {
        let count = rows[0].iter().filter(|card| card.rank == rank).count() as u32;
        queens_up = queens_up.max(count);
    }
    out.push((queens_up + jokers[0]).min(2) as f32 / 2.0);
    out.push(top_max.map_or(0.0, |rank| unseen_rank[rank as usize] as f32 / 3.0));
    // 15 jokers anywhere on the board.
    out.push(jokers.iter().sum::<u32>() as f32 / 2.0);
}

/// The acting seat's own 11-card board at T3: per-row made value / royalty /
/// room, ordering slack, bottom-suit concentration, FL entry facts, jokers.
/// Byte-for-byte port of `t3_second_features.actor_block`.
pub fn actor_block(rows: &[Vec<Card>; 3], out: &mut Vec<f32>) {
    let capacities = [3usize, 5, 5];
    let mut categories = [0usize; 3];
    let mut rooms = [0usize; 3];
    for row in 0..3 {
        let cards = &rows[row];
        rooms[row] = capacities[row] - cards.len();
        let category = partial_category(cards, capacities[row]);
        categories[row] = category;
        if cards.len() == capacities[row] {
            let value = evaluate_hand_value(cards, capacities[row]);
            spread(value, out);
            out.push(row_royalty(row, cards) as f32 / MAX_ROYALTY);
        } else {
            let start = out.len();
            out.resize(start + CATEGORIES + 2, 0.0);
            out[start + category.min(CATEGORIES - 1)] = 1.0;
            out.push(0.0);
        }
        out.push(rooms[row] as f32 / 5.0);
    }
    // Ordering slack: how far each adjacent pair is from a foul, with rows
    // still open to fix it.
    out.push((categories[1] as f32 - categories[2] as f32) / 8.0);
    out.push((categories[0] as f32 - categories[1] as f32) / 8.0);
    out.push(if categories[0] > categories[1] {
        1.0
    } else {
        0.0
    });
    out.push(if categories[1] > categories[2] {
        1.0
    } else {
        0.0
    });
    let mut suits = [0u8; 4];
    for card in &rows[2] {
        if !card.is_joker() {
            suits[card.suit as usize] += 1;
        }
    }
    out.push(suits.iter().copied().max().unwrap_or(0) as f32 / 5.0);
    let (fl_qualified, fl_count) = if rows[0].len() == 3 {
        check_fl_entry(&rows[0])
    } else {
        (false, 0)
    };
    out.push(if fl_qualified { 1.0 } else { 0.0 });
    out.push(fl_count as f32 / 17.0);
    let jokers = rows
        .iter()
        .flat_map(|row| row.iter())
        .filter(|card| card.is_joker())
        .count();
    out.push(jokers as f32 / 2.0);
    out.push(rooms.iter().sum::<usize>() as f32 / 5.0);
}

/// Fantasyland EV by card count 14..17, from ai/config/fl_ev.json.
pub type FlTable = [f32; 4];

fn fl_ev_for(table: &FlTable, card_count: u8) -> f32 {
    match card_count {
        14..=17 => table[(card_count - 14) as usize],
        _ => 0.0,
    }
}

/// One constrained evaluation of the hero board, reused by both per-action
/// blocks.  Computing it twice was pure duplicated work: the constrained
/// evaluation is the most expensive thing in the per-draw loop.
pub struct HeroEval {
    pub busted: bool,
    pub rows: [Vec<Card>; 3],
    pub values: [u32; 3],
}

pub fn hero_eval(rows: &[Vec<Card>; 3]) -> HeroEval {
    let eval = evaluate_board_with_joker_constraint(&rows[0], &rows[1], &rows[2]);
    let final_rows = [eval.top, eval.mid, eval.bot];
    let values = [
        evaluate_hand_value(&final_rows[0], 3),
        evaluate_hand_value(&final_rows[1], 5),
        evaluate_hand_value(&final_rows[2], 5),
    ];
    HeroEval {
        busted: eval.busted,
        rows: final_rows,
        values,
    }
}

/// Exact terminal facts of the completed hero board (42 dims).
pub fn hero_block(rows: &[Vec<Card>; 3], eval: &HeroEval, fl_table: &FlTable, out: &mut Vec<f32>) {
    let busted = eval.busted;
    let final_rows = &eval.rows;
    let royalties: [i32; 3] = if busted {
        [0, 0, 0]
    } else {
        [
            row_royalty(0, &final_rows[0]),
            row_royalty(1, &final_rows[1]),
            row_royalty(2, &final_rows[2]),
        ]
    };
    let (fl_qualified, fl_count) = if busted {
        (false, 0u8)
    } else {
        check_fl_entry(&final_rows[0])
    };
    let fl_ev = if fl_qualified {
        fl_ev_for(fl_table, fl_count)
    } else {
        0.0
    };
    out.push(if busted { 1.0 } else { 0.0 });
    out.push(royalties.iter().sum::<i32>() as f32 / MAX_ROYALTY);
    for index in 0..3 {
        out.push(royalties[index] as f32 / MAX_ROYALTY);
    }
    out.push(if fl_qualified { 1.0 } else { 0.0 });
    out.push(fl_count as f32 / 17.0);
    out.push(fl_ev / MAX_FL_EV);
    for index in 0..3 {
        spread(eval.values[index], out);
    }
    let jokers = rows.iter().flatten().filter(|card| card.is_joker()).count();
    out.push(jokers as f32 / 2.0);
}

/// Facts the per-row histograms cannot express (12 dims).
pub fn joint_block(
    hero: &HeroEval,
    opponent_rows: &[Vec<Card>; 3],
    opponent_categories: &[usize; 3],
    out: &mut Vec<f32>,
) {
    let rooms: [usize; 3] = [
        ROW_CAPACITY[0] - opponent_rows[0].len(),
        ROW_CAPACITY[1] - opponent_rows[1].len(),
        ROW_CAPACITY[2] - opponent_rows[2].len(),
    ];
    let hero_values = hero.values;

    let locked_middle = rooms[2] == 0 && opponent_categories[1] > opponent_categories[2];
    let locked_top = rooms[1] == 0 && opponent_categories[0] > opponent_categories[1];
    out.push(if locked_middle { 1.0 } else { 0.0 });
    out.push(if locked_top { 1.0 } else { 0.0 });
    out.push(if locked_middle || locked_top {
        1.0
    } else {
        0.0
    });
    out.push((opponent_categories[1] as f32 - opponent_categories[2] as f32) / 8.0);
    out.push((opponent_categories[0] as f32 - opponent_categories[1] as f32) / 8.0);

    let mut wins = 0;
    for index in 0..3 {
        let sign = if rooms[index] == 0 {
            let opponent_value = evaluate_hand_value(&opponent_rows[index], ROW_CAPACITY[index]);
            (hero_values[index] > opponent_value) as i32
                - (hero_values[index] < opponent_value) as i32
        } else {
            let hero_category = category_of(hero_values[index]);
            (hero_category > opponent_categories[index]) as i32
                - (hero_category < opponent_categories[index]) as i32
        };
        out.push(sign as f32);
        if sign > 0 {
            wins += 1;
        }
    }
    out.push(wins as f32 / 3.0);
    out.push(if wins == 3 { 1.0 } else { 0.0 });
    out.push(if hero.busted { 1.0 } else { 0.0 });
    out.push(rooms.iter().sum::<usize>() as f32 / 5.0);
}

// ------------------------------------------------------------------
// Dense net
// ------------------------------------------------------------------

struct Layer {
    inputs: usize,
    outputs: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

pub struct Model {
    pub input_dim: usize,
    /// Width of the last layer.  Every evaluator in this crate is 1; the T0
    /// policy net is 243.  Kept explicit so `predict` can keep meaning "the
    /// scalar this model scores with" and refuse to average a vector.
    pub output_dim: usize,
    mean: Vec<f32>,
    inverse_std: Vec<f32>,
    layers: Vec<Layer>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32> {
        if self.offset + 4 > self.bytes.len() {
            bail!("model image is truncated");
        }
        let value = u32::from_le_bytes(
            self.bytes[self.offset..self.offset + 4]
                .try_into()
                .map_err(|_| anyhow!("bad u32"))?,
        );
        self.offset += 4;
        Ok(value)
    }

    fn floats(&mut self, count: usize) -> Result<Vec<f32>> {
        if self.offset + count * 4 > self.bytes.len() {
            bail!("model image is truncated");
        }
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let start = self.offset + index * 4;
            out.push(f32::from_le_bytes(
                self.bytes[start..start + 4]
                    .try_into()
                    .map_err(|_| anyhow!("bad f32"))?,
            ));
        }
        self.offset += count * 4;
        Ok(out)
    }
}

impl Model {
    /// The scalar evaluators: anything wider is a loading mistake, and a
    /// mistake that would otherwise be served as a silently different net.
    pub fn load(bytes: &[u8]) -> Result<Self> {
        let model = Self::load_wide(bytes)?;
        if model.output_dim != 1 {
            bail!("model image must end in a single output");
        }
        Ok(model)
    }

    /// The same image with any output width; used by the T0 policy net, whose
    /// last layer is one logit per action.
    pub fn load_wide(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 || &bytes[..4] != b"T4F1" {
            bail!("model image does not start with the expected magic");
        }
        let mut reader = Reader { bytes, offset: 4 };
        if reader.u32()? != 1 {
            bail!("unsupported model image version");
        }
        let layer_count = reader.u32()? as usize;
        let input_dim = reader.u32()? as usize;
        let mean = reader.floats(input_dim)?;
        let std = reader.floats(input_dim)?;
        if std.iter().any(|value| !value.is_finite() || *value == 0.0) {
            bail!("model image has a zero or non-finite standard deviation");
        }
        let inverse_std = std.iter().map(|value| 1.0 / value).collect();
        let mut layers = Vec::with_capacity(layer_count);
        let mut expected = input_dim;
        for index in 0..layer_count {
            let inputs = reader.u32()? as usize;
            let outputs = reader.u32()? as usize;
            if inputs != expected {
                bail!("layer {index} expects {inputs} inputs, previous stage gives {expected}");
            }
            let weight = reader.floats(inputs * outputs)?;
            let bias = reader.floats(outputs)?;
            layers.push(Layer {
                inputs,
                outputs,
                weight,
                bias,
            });
            expected = outputs;
        }
        if expected == 0 {
            bail!("model image has no outputs");
        }
        Ok(Self {
            input_dim,
            output_dim: expected,
            mean,
            inverse_std,
            layers,
        })
    }

    /// Standardize then run the stack; ReLU on every layer but the last.
    pub fn predict(&self, features: &[f32], scratch: &mut Vec<f32>) -> f32 {
        scratch.clear();
        for index in 0..self.input_dim {
            scratch.push((features[index] - self.mean[index]) * self.inverse_std[index]);
        }
        let mut current = std::mem::take(scratch);
        let mut next: Vec<f32> = Vec::new();
        for (position, layer) in self.layers.iter().enumerate() {
            next.clear();
            next.reserve(layer.outputs);
            for output in 0..layer.outputs {
                let row = &layer.weight[output * layer.inputs..(output + 1) * layer.inputs];
                let mut sum = layer.bias[output];
                for input in 0..layer.inputs {
                    sum += row[input] * current[input];
                }
                if position + 1 < self.layers.len() {
                    sum = sum.max(0.0);
                }
                next.push(sum);
            }
            std::mem::swap(&mut current, &mut next);
        }
        let value = current[0];
        *scratch = current;
        value
    }

    /// The whole last layer rather than its first element: the policy net's
    /// 243 logits.  Same arithmetic and same order as `predict`, so a scalar
    /// model read through here gives exactly what `predict` gives.
    pub fn predict_all(&self, features: &[f32], out: &mut Vec<f32>) {
        let mut scratch: Vec<f32> = Vec::new();
        let value = self.predict(features, &mut scratch);
        debug_assert_eq!(scratch.len(), self.output_dim);
        debug_assert_eq!(scratch[0], value);
        out.clear();
        out.extend_from_slice(&scratch);
    }
}

// ------------------------------------------------------------------
// Node-shared blocks (opponent per-row outlook and context)
// ------------------------------------------------------------------

/// Per-row completion histograms, royalty, FL and room (41 dims), plus the
/// opponent categories the joint block needs.  Shared across the node.
pub fn opponent_rowwise_block(
    opponent_rows: &[Vec<Card>; 3],
    pool: &[Card],
    fl_table: &FlTable,
    out: &mut Vec<f32>,
) -> [usize; 3] {
    let mut categories = [0usize; 3];
    for row in 0..3 {
        let capacity = ROW_CAPACITY[row];
        let room = capacity - opponent_rows[row].len();
        categories[row] = partial_category(&opponent_rows[row], capacity);
        let mut histogram = [0.0f32; CATEGORIES];
        let mut royalty_total = 0.0f32;
        let mut fl_total = 0.0f32;
        let mut samples = 0usize;
        let mut filled = opponent_rows[row].clone();
        if room == 0 {
            let value = evaluate_hand_value(&filled, capacity);
            histogram[category_of(value).min(CATEGORIES - 1)] = 1.0;
            royalty_total = row_royalty(row, &filled) as f32;
            if row == 0 {
                let (qualified, count) = check_fl_entry(&filled);
                if qualified {
                    fl_total = fl_ev_for(fl_table, count);
                }
            }
            samples = 1;
        } else if room == 1 {
            for card in pool {
                filled.push(*card);
                let value = evaluate_hand_value(&filled, capacity);
                histogram[category_of(value).min(CATEGORIES - 1)] += 1.0;
                royalty_total += row_royalty(row, &filled) as f32;
                if row == 0 {
                    let (qualified, count) = check_fl_entry(&filled);
                    if qualified {
                        fl_total += fl_ev_for(fl_table, count);
                    }
                }
                samples += 1;
                filled.pop();
            }
        } else {
            for i in 0..pool.len() {
                for j in (i + 1)..pool.len() {
                    filled.push(pool[i]);
                    filled.push(pool[j]);
                    let value = evaluate_hand_value(&filled, capacity);
                    histogram[category_of(value).min(CATEGORIES - 1)] += 1.0;
                    royalty_total += row_royalty(row, &filled) as f32;
                    if row == 0 {
                        let (qualified, count) = check_fl_entry(&filled);
                        if qualified {
                            fl_total += fl_ev_for(fl_table, count);
                        }
                    }
                    samples += 1;
                    filled.pop();
                    filled.pop();
                }
            }
        }
        let denominator = samples.max(1) as f32;
        for bin in histogram {
            out.push(bin / denominator);
        }
        out.push(royalty_total / denominator / MAX_ROYALTY);
        out.push(fl_total / denominator / MAX_FL_EV);
        out.push(room as f32 / 5.0);
    }
    for row in 0..3 {
        let mut suits = [0u8; 4];
        for card in &opponent_rows[row] {
            if !card.is_joker() {
                suits[card.suit as usize] += 1;
            }
        }
        out.push(*suits.iter().max().unwrap_or(&0) as f32 / 5.0);
    }
    out.push(
        opponent_rows
            .iter()
            .flatten()
            .filter(|card| card.is_joker())
            .count() as f32
            / 2.0,
    );
    out.push(pool.iter().filter(|card| card.is_joker()).count() as f32 / 2.0);
    categories
}

/// Identity of one row's cards, order-insensitive, for the rowwise memo.
fn row_cards_key(row: usize, cards: &[Card]) -> u64 {
    // Ids are offset by one so an empty slot (0) is distinct from the 2s.
    let mut ids: Vec<u64> = cards
        .iter()
        .map(|card| {
            1 + if card.is_joker() {
                52
            } else {
                card.suit as u64 * 13 + card.rank as u64 - 2
            }
        })
        .collect();
    ids.sort_unstable();
    let mut key = (row as u64) << 60;
    for (slot, id) in ids.iter().enumerate() {
        key |= id << (slot * 6);
    }
    key
}

/// `opponent_rowwise_block` with the per-row 12-dim slice memoized on the
/// row's cards.  Playout move choosers call the block once per candidate,
/// but a candidate changes at most two rows, so most of each call is a
/// repeat of the previous one -- and the pool is fixed across a node's
/// candidates, which is what makes the row key sufficient.  Byte-identical
/// to the uncached block: this is pure memoization of the same computation.
pub fn opponent_rowwise_block_shared(
    opponent_rows: &[Vec<Card>; 3],
    pool: &[Card],
    fl_table: &FlTable,
    memo: &std::sync::Mutex<std::collections::HashMap<(u64, u64), ([f32; 12], usize)>>,
    pool_key: u64,
    out: &mut Vec<f32>,
) -> [usize; 3] {
    let mut categories = [0usize; 3];
    for row in 0..3 {
        let key = (pool_key, row_cards_key(row, &opponent_rows[row]));
        if let Some((slice, category)) = memo.lock().unwrap().get(&key) {
            out.extend_from_slice(slice);
            categories[row] = *category;
            continue;
        }
        // Computed outside the lock: a miss costs milliseconds and other
        // threads should not wait on it.  A racing duplicate insert is
        // harmless -- both compute the same numbers.
        let mut scratch: Vec<f32> = Vec::with_capacity(12);
        let category = rowwise_single_row(row, &opponent_rows[row], pool, fl_table, &mut scratch);
        let mut slice = [0.0f32; 12];
        slice.copy_from_slice(&scratch);
        memo.lock().unwrap().insert(key, (slice, category));
        out.extend_from_slice(&slice);
        categories[row] = category;
    }
    // Tail: suit concentration per row, jokers on board, jokers in pool --
    // cheap, computed directly (matches the uncached block's tail).
    for row in 0..3 {
        let mut suits = [0u8; 4];
        for card in &opponent_rows[row] {
            if !card.is_joker() {
                suits[card.suit as usize] += 1;
            }
        }
        out.push(*suits.iter().max().unwrap_or(&0) as f32 / 5.0);
    }
    out.push(
        opponent_rows
            .iter()
            .flatten()
            .filter(|card| card.is_joker())
            .count() as f32
            / 2.0,
    );
    out.push(pool.iter().filter(|card| card.is_joker()).count() as f32 / 2.0);
    categories
}

/// One row of the rowwise block (12 dims), extracted from the block above so
/// the cached variant can compute exactly the same numbers per row.
fn rowwise_single_row(
    row: usize,
    cards: &[Card],
    pool: &[Card],
    fl_table: &FlTable,
    out: &mut Vec<f32>,
) -> usize {
    let capacity = ROW_CAPACITY[row];
    let room = capacity - cards.len();
    let category = partial_category(cards, capacity);
    let mut histogram = [0.0f32; CATEGORIES];
    let mut royalty_total = 0.0f32;
    let mut fl_total = 0.0f32;
    let mut samples = 0usize;
    let mut filled = cards.to_vec();
    if room == 0 {
        let value = evaluate_hand_value(&filled, capacity);
        histogram[category_of(value).min(CATEGORIES - 1)] = 1.0;
        royalty_total = row_royalty(row, &filled) as f32;
        if row == 0 {
            let (qualified, count) = check_fl_entry(&filled);
            if qualified {
                fl_total = fl_ev_for(fl_table, count);
            }
        }
        samples = 1;
    } else if room == 1 {
        for card in pool {
            filled.push(*card);
            let value = evaluate_hand_value(&filled, capacity);
            histogram[category_of(value).min(CATEGORIES - 1)] += 1.0;
            royalty_total += row_royalty(row, &filled) as f32;
            if row == 0 {
                let (qualified, count) = check_fl_entry(&filled);
                if qualified {
                    fl_total += fl_ev_for(fl_table, count);
                }
            }
            samples += 1;
            filled.pop();
        }
    } else {
        for i in 0..pool.len() {
            for j in (i + 1)..pool.len() {
                filled.push(pool[i]);
                filled.push(pool[j]);
                let value = evaluate_hand_value(&filled, capacity);
                histogram[category_of(value).min(CATEGORIES - 1)] += 1.0;
                royalty_total += row_royalty(row, &filled) as f32;
                if row == 0 {
                    let (qualified, count) = check_fl_entry(&filled);
                    if qualified {
                        fl_total += fl_ev_for(fl_table, count);
                    }
                }
                samples += 1;
                filled.pop();
                filled.pop();
            }
        }
    }
    let denominator = samples.max(1) as f32;
    for bin in histogram {
        out.push(bin / denominator);
    }
    out.push(royalty_total / denominator / MAX_ROYALTY);
    out.push(fl_total / denominator / MAX_FL_EV);
    out.push(room as f32 / 5.0);
    category
}

/// Deck composition for the FL14 teachers (7 dims).  Port of
/// `context_block` in `ai/tutor/encode_fl14_teacher.py`.
///
/// Neither `context_block` below nor `t3_vs_fl::fl_context` with entries
/// dropped: at a fixed Fantasyland width the opponent's card-count one-hot and
/// `fl_ev[width]` are constants, and a constant column is a zero-variance
/// column for the trainer to divide by.  They come back when widths 15-17 get
/// pools.
pub fn fl14_context_block(pool: &[Card], out: &mut Vec<f32>) {
    let mut jokers = 0u32;
    let (mut aces, mut kings, mut queens) = (0u32, 0u32, 0u32);
    for card in pool {
        if card.is_joker() {
            jokers += 1;
        } else {
            match card.rank {
                14 => aces += 1,
                13 => kings += 1,
                12 => queens += 1,
                _ => {}
            }
        }
    }
    out.push(jokers as f32 / 2.0);
    out.push(aces as f32 / 4.0);
    out.push(kings as f32 / 4.0);
    out.push(queens as f32 / 4.0);
    out.push(pool.len() as f32 / 54.0);
    out.push((aces + kings + queens) as f32 / pool.len().max(1) as f32);
    // Redundant -- it is exactly 1 - dim 0 -- but the trained weights were
    // fitted with it, so dropping it would move every prediction.
    out.push((2 - jokers.min(2)) as f32 / 2.0);
}

/// Remaining-deck summary (6 dims).
pub fn context_block(pool: &[Card], dead_count: usize, out: &mut Vec<f32>) {
    let mut ranks = [0u32; 15];
    let mut jokers = 0u32;
    for card in pool {
        if card.is_joker() {
            jokers += 1;
        } else {
            ranks[card.rank as usize] += 1;
        }
    }
    let high: u32 = ranks[12] + ranks[13] + ranks[14];
    let distinct = ranks.iter().filter(|count| **count > 0).count();
    out.push(pool.len() as f32 / 54.0);
    out.push(dead_count as f32 / 6.0);
    out.push(high as f32 / pool.len().max(1) as f32);
    out.push(jokers as f32 / 2.0);
    out.push(*ranks.iter().max().unwrap_or(&0) as f32 / 4.0);
    out.push(distinct as f32 / 13.0);
}

#[cfg(test)]
mod allocation_rank_tests {
    use super::*;

    fn card(rank: u8) -> Card {
        Card { rank, suit: 0 }
    }

    fn joker() -> Card {
        Card { rank: 0, suit: 4 }
    }

    #[test]
    fn open_pair_and_kicker_are_visible() {
        let pair_aces = vec![card(14), card(14)];
        assert_eq!(partial_tiebreaks(&pair_aces, 3), (1.0, 0.0));

        let middle = vec![card(3), card(3), card(8), card(12)];
        let got = partial_tiebreaks(&middle, 5);
        assert!((got.0 - 3.0 / 14.0).abs() < 1e-7);
        assert!((got.1 - 12.0 / 14.0).abs() < 1e-7);
    }

    #[test]
    fn joker_joins_the_highest_tied_group() {
        let cards = vec![card(14), joker()];
        assert_eq!(partial_tiebreaks(&cards, 3), (1.0, 0.0));
    }

    #[test]
    fn suffix_is_six_columns_in_row_order() {
        let rows = [
            vec![card(14)],
            vec![card(3), card(3), card(8), card(12)],
            vec![card(6), card(6), card(12), card(12)],
        ];
        let mut out = Vec::new();
        allocation_rank_block(&rows, &mut out);
        assert_eq!(out.len(), ALLOCATION_RANK_SIZE);
        let expected = [1.0, 0.0, 3.0 / 14.0, 12.0 / 14.0, 12.0 / 14.0, 6.0 / 14.0];
        for (actual, expected) in out.iter().zip(expected) {
            assert!((*actual - expected).abs() < 1e-7);
        }
    }

    #[test]
    fn complete_wheel_and_joker_row_use_exact_tiebreaks() {
        let wheel = vec![
            Card { rank: 14, suit: 0 },
            Card { rank: 2, suit: 1 },
            Card { rank: 3, suit: 2 },
            Card { rank: 4, suit: 3 },
            Card { rank: 5, suit: 0 },
        ];
        let wheel_value = evaluate_hand_value(&wheel, 5);
        let wheel_expected = (
            ((wheel_value / B.pow(4)) % B) as f32 / 14.0,
            ((wheel_value / B.pow(3)) % B) as f32 / 14.0,
        );
        assert_eq!(partial_tiebreaks(&wheel, 5), wheel_expected);

        let joker_top = vec![card(14), card(14), joker()];
        let joker_value = evaluate_hand_value(&joker_top, 3);
        let joker_expected = (
            ((joker_value / B.pow(4)) % B) as f32 / 14.0,
            ((joker_value / B.pow(3)) % B) as f32 / 14.0,
        );
        assert_eq!(partial_tiebreaks(&joker_top, 3), joker_expected);
    }
}
