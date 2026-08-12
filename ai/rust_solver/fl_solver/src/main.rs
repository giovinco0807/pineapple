//! FL Solver - High-performance Fantasyland solver in Rust
//!
//! Standalone executable that communicates via JSON stdin/stdout

mod frontier;
mod pool;
mod row_memo;
mod t2_labels;
mod t3_labels;
mod t4_labels;
mod vs_fl;

use rayon::prelude::*;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};

/// Card representation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Card {
    pub rank: u8,     // 2-14 (2-A), 0 for joker
    pub suit: u8,     // 0-3 (spades, hearts, diamonds, clubs), 4 for joker
}

impl Card {
    pub fn is_joker(&self) -> bool {
        self.rank == 0
    }
}

/// Hand rank for 5-card hands
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HandRank {
    HighCard = 0,
    OnePair = 1,
    TwoPair = 2,
    Trips = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    Quads = 7,
    StraightFlush = 8,
    RoyalFlush = 9,
}

/// Hand rank for 3-card hands (Top)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HandRank3 {
    HighCard = 0,
    OnePair = 1,
    Trips = 2,
}

/// Placement result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Placement {
    pub top: Vec<Card>,
    pub middle: Vec<Card>,
    pub bottom: Vec<Card>,
    pub discards: Vec<Card>,
    pub top_royalty: i32,
    pub middle_royalty: i32,
    pub bottom_royalty: i32,
    pub total_royalty: i32,
    pub can_stay: bool,
    pub is_bust: bool,
    pub score: f64,
}

#[derive(Clone, Debug)]
struct CanonicalRows {
    busted: bool,
    top: Vec<Card>,
    middle: Vec<Card>,
    bottom: Vec<Card>,
    top_royalty: i32,
    middle_royalty: i32,
    bottom_royalty: i32,
    can_stay: bool,
}

fn to_core_cards(cards: &[Card]) -> Vec<ofc_core::Card> {
    cards
        .iter()
        .map(|card| ofc_core::Card { rank: card.rank, suit: card.suit })
        .collect()
}

fn from_core_cards(cards: &[ofc_core::Card]) -> Vec<Card> {
    cards
        .iter()
        .map(|card| Card { rank: card.rank, suit: card.suit })
        .collect()
}

/// Apply the single canonical Joker rule before bust, royalty, FL, or line scoring.
fn canonical_rows(top: &[Card], middle: &[Card], bottom: &[Card]) -> CanonicalRows {
    let eval = ofc_core::evaluate_board_with_joker_constraint(
        &to_core_cards(top),
        &to_core_cards(middle),
        &to_core_cards(bottom),
    );
    let (top_royalty, middle_royalty, bottom_royalty, can_stay) = if eval.busted {
        (0, 0, 0, false)
    } else {
        (
            ofc_core::get_top_royalty(&eval.top),
            ofc_core::get_middle_royalty(&eval.mid),
            ofc_core::get_bottom_royalty(&eval.bot),
            ofc_core::check_fl_stay(&eval.top, &eval.mid, &eval.bot),
        )
    };
    CanonicalRows {
        busted: eval.busted,
        top: from_core_cards(&eval.top),
        middle: from_core_cards(&eval.mid),
        bottom: from_core_cards(&eval.bot),
        top_royalty,
        middle_royalty,
        bottom_royalty,
        can_stay,
    }
}

// ============================================================
//  Hand Evaluation
// ============================================================

fn count_ranks(cards: &[Card]) -> [u8; 15] {
    let mut counts = [0u8; 15];
    for c in cards {
        if !c.is_joker() {
            counts[c.rank as usize] += 1;
        }
    }
    counts
}

fn count_suits(cards: &[Card]) -> [u8; 4] {
    let mut counts = [0u8; 4];
    for c in cards {
        if !c.is_joker() && c.suit < 4 {
            counts[c.suit as usize] += 1;
        }
    }
    counts
}

fn count_jokers(cards: &[Card]) -> u8 {
    cards.iter().filter(|c| c.is_joker()).count() as u8
}

fn is_straight_possible(rank_counts: &[u8; 15], jokers: u8) -> (bool, bool) {
    let straights: [[u8; 5]; 10] = [
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
    
    let mut is_straight = false;
    let mut is_broadway = false;
    
    for (i, s) in straights.iter().enumerate() {
        let missing: u8 = s.iter()
            .filter(|&&r| rank_counts[r as usize] == 0)
            .count() as u8;
        if missing <= jokers {
            is_straight = true;
            if i == 9 {
                is_broadway = true;
            }
        }
    }
    (is_straight, is_broadway)
}

/// Get the high card of a straight. For wheel (A2345), returns 5.
fn get_straight_high_card(rank_counts: &[u8; 15], jokers: u8) -> u8 {
    let straights: [[u8; 5]; 10] = [
        [14, 2, 3, 4, 5],   // high = 5 (wheel)
        [2, 3, 4, 5, 6],    // high = 6
        [3, 4, 5, 6, 7],    // high = 7
        [4, 5, 6, 7, 8],    // high = 8
        [5, 6, 7, 8, 9],    // high = 9
        [6, 7, 8, 9, 10],   // high = 10
        [7, 8, 9, 10, 11],  // high = 11
        [8, 9, 10, 11, 12], // high = 12
        [9, 10, 11, 12, 13],// high = 13
        [10, 11, 12, 13, 14],// high = 14
    ];
    let high_cards: [u8; 10] = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
    
    let mut best_high = 0u8;
    for (i, s) in straights.iter().enumerate() {
        let missing: u8 = s.iter()
            .filter(|&&r| rank_counts[r as usize] == 0)
            .count() as u8;
        if missing <= jokers {
            best_high = best_high.max(high_cards[i]);
        }
    }
    best_high
}

fn evaluate_5_card(cards: &[Card]) -> (HandRank, u32) {
    let (rank, strength) = ofc_core::evaluate_5_card(&to_core_cards(cards));
    let rank = match rank {
        ofc_core::HandRank::HighCard => HandRank::HighCard,
        ofc_core::HandRank::OnePair => HandRank::OnePair,
        ofc_core::HandRank::TwoPair => HandRank::TwoPair,
        ofc_core::HandRank::Trips => HandRank::Trips,
        ofc_core::HandRank::Straight => HandRank::Straight,
        ofc_core::HandRank::Flush => HandRank::Flush,
        ofc_core::HandRank::FullHouse => HandRank::FullHouse,
        ofc_core::HandRank::Quads => HandRank::Quads,
        ofc_core::HandRank::StraightFlush => HandRank::StraightFlush,
        ofc_core::HandRank::RoyalFlush => HandRank::RoyalFlush,
    };
    (rank, strength)
}

fn evaluate_3_card(cards: &[Card]) -> (HandRank3, u32) {
    let (rank, strength) = ofc_core::evaluate_3_card(&to_core_cards(cards));
    let rank = match rank {
        ofc_core::HandRank3::HighCard => HandRank3::HighCard,
        ofc_core::HandRank3::OnePair => HandRank3::OnePair,
        ofc_core::HandRank3::Trips => HandRank3::Trips,
    };
    (rank, strength)
}

fn calculate_strength(rank_counts: &[u8; 15]) -> u32 {
    let mut strength = 0u32;
    for (i, &count) in rank_counts.iter().enumerate().rev() {
        if count > 0 {
            strength = strength * 15 + i as u32;
        }
    }
    strength
}

// ============================================================
//  Royalty Calculation
// ============================================================

fn get_top_royalty(cards: &[Card]) -> i32 {
    ofc_core::get_top_royalty(&to_core_cards(cards))
}

fn get_middle_royalty(cards: &[Card]) -> i32 {
    ofc_core::get_middle_royalty(&to_core_cards(cards))
}

fn get_bottom_royalty(cards: &[Card]) -> i32 {
    ofc_core::get_bottom_royalty(&to_core_cards(cards))
}

fn check_fl_stay(top: &[Card], middle: &[Card], bottom: &[Card]) -> bool {
    ofc_core::check_fl_stay(
        &to_core_cards(top),
        &to_core_cards(middle),
        &to_core_cards(bottom),
    )
}

// ============================================================
//  Bust Check
// ============================================================

/// Get comparable strength for 5-card hand
#[allow(dead_code)]
fn get_5card_strength(cards: &[Card]) -> (u8, u32) {
    let (rank, strength) = evaluate_5_card(cards);
    (rank as u8, strength)
}

/// Get comparable strength for 3-card hand (mapped to 5-card scale)
#[allow(dead_code)]
fn get_3card_strength(cards: &[Card]) -> (u8, u32) {
    let (rank, strength) = evaluate_3_card(cards);
    // Map 3-card ranks:
    // HighCard=0, OnePair=1, Trips=2
    // In 5-card: HighCard=0, OnePair=1, TwoPair=2, Trips=3
    // 3-card Trips should beat 5-card TwoPair but lose to 5-card Trips
    // So we map it to 2.5 conceptually, but since we use u8, we use special logic
    let mapped_rank = match rank {
        HandRank3::HighCard => 0,
        HandRank3::OnePair => 1,
        HandRank3::Trips => 3, // Same as 5-card trips for comparison
    };
    (mapped_rank, strength)
}

/// Upper bound on the canonical score of a candidate arrangement.
///
/// Each row is evaluated with its jokers at full strength and the royalties
/// read off that maximum.  The canonical bottom-up constraint can only weaken
/// rows, and row royalties and the stay bonus are both monotone in row
/// strength, so no canonical score can exceed this bound.  Candidates whose
/// bound cannot beat the current best are skipped before the expensive
/// canonical evaluation; because this is a true upper bound and the running
/// best uses strict improvement, the pruned search selects exactly the same
/// placement as the unpruned one.
fn raw_score_bound(top: &[Card], middle: &[Card], bottom: &[Card]) -> f64 {
    let royalty = get_top_royalty(top) + get_middle_royalty(middle) + get_bottom_royalty(bottom);
    let (top_rank, _) = evaluate_3_card(top);
    let (bot_rank, _) = evaluate_5_card(bottom);
    let stay_possible = top_rank == HandRank3::Trips
        || matches!(bot_rank, HandRank::Quads | HandRank::StraightFlush | HandRank::RoyalFlush);
    royalty as f64 + if stay_possible { 100.0 } else { 0.0 }
}

/// Compare two 5-card hands. Returns -1 if a < b, 0 if equal, 1 if a > b
fn compare_5_hands(a: &[Card], b: &[Card]) -> i32 {
    ofc_core::compare_5_hands(&to_core_cards(a), &to_core_cards(b))
}

/// Get the rank of the pair in a hand (returns highest paired rank, or 0 if no pair)
fn get_pair_rank(cards: &[Card]) -> u8 {
    let rank_counts = count_ranks(cards);
    let jokers = count_jokers(cards);
    
    // Find highest rank that forms a pair (with joker help)
    for r in (2..=14).rev() {
        if rank_counts[r] >= 2 || (rank_counts[r] >= 1 && jokers >= 1) {
            return r as u8;
        }
    }
    0
}

/// Get the rank of trips in a hand (returns trips rank, or 0 if no trips)
fn get_trips_rank(cards: &[Card]) -> u8 {
    let rank_counts = count_ranks(cards);
    let jokers = count_jokers(cards);
    
    for r in (2..=14).rev() {
        if rank_counts[r] + jokers >= 3 && rank_counts[r] >= 1 {
            return r as u8;
        }
    }
    0
}

/// Get the ranks of both pairs in a two-pair hand (high_pair, low_pair)
fn get_two_pair_ranks(cards: &[Card]) -> (u8, u8) {
    let rank_counts = count_ranks(cards);
    let mut pairs = Vec::new();
    
    for r in (2..=14).rev() {
        if rank_counts[r] >= 2 {
            pairs.push(r as u8);
            if pairs.len() == 2 {
                break;
            }
        }
    }
    
    if pairs.len() >= 2 {
        (pairs[0], pairs[1])
    } else if pairs.len() == 1 {
        (pairs[0], 0)
    } else {
        (0, 0)
    }
}

/// Get the rank of quads in a hand
fn get_quads_rank(cards: &[Card]) -> u8 {
    let rank_counts = count_ranks(cards);
    let jokers = count_jokers(cards);
    
    for r in (2..=14).rev() {
        if rank_counts[r] + jokers >= 4 && rank_counts[r] >= 1 {
            return r as u8;
        }
    }
    0
}


fn is_valid_placement(top: &[Card], middle: &[Card], bottom: &[Card]) -> bool {
    // Bottom must be >= Middle
    if compare_5_hands(bottom, middle) < 0 {
        return false;
    }
    
    // Middle must be >= Top (comparing 5-card to 3-card)
    let (top_rank, _) = evaluate_3_card(top);
    let (mid_rank, _) = evaluate_5_card(middle);
    
    // Map 3-card rank to comparable value
    // 3-card: HighCard=0, OnePair=1, Trips=2
    // 5-card: HighCard=0, OnePair=1, TwoPair=2, Trips=3, ...
    // 3-card Trips (2.5) beats 5-card TwoPair (2) but loses to 5-card Trips (3)
    let top_rank_f: f64 = match top_rank {
        HandRank3::HighCard => 0.0,
        HandRank3::OnePair => 1.0,
        HandRank3::Trips => 2.5,
    };
    let mid_rank_f = mid_rank as u8 as f64;
    
    // If top rank category > mid rank category, bust
    if top_rank_f > mid_rank_f {
        return false;
    }
    
    // Special case: 3-card Trips vs 5-card Trips - compare trips ranks
    if top_rank == HandRank3::Trips && mid_rank == HandRank::Trips {
        let top_trips = get_trips_rank(top);
        let mid_trips = get_trips_rank(middle);
        if top_trips > mid_trips {
            return false;  // J-Trips > T-Trips = bust
        }
    }
    
    // If same rank category, compare within that category
    if (top_rank_f - mid_rank_f).abs() < 0.01 {
        match top_rank {
            HandRank3::HighCard => {
                // Compare high cards
                let top_high = top.iter().filter(|c| !c.is_joker()).map(|c| c.rank).max().unwrap_or(0);
                let mid_high = middle.iter().filter(|c| !c.is_joker()).map(|c| c.rank).max().unwrap_or(0);
                if top_high > mid_high {
                    return false;
                }
            }
            HandRank3::OnePair => {
                // Compare pair ranks
                let top_pair = get_pair_rank(top);
                let mid_pair = get_pair_rank(middle);
                if top_pair > mid_pair {
                    return false;
                }
                // If same pair rank, compare kickers (simplified: just check if top is not stronger)
                if top_pair == mid_pair {
                    let top_kicker = top.iter().filter(|c| !c.is_joker() && c.rank != top_pair).map(|c| c.rank).max().unwrap_or(0);
                    let mid_kickers: Vec<u8> = middle.iter().filter(|c| !c.is_joker() && c.rank != mid_pair).map(|c| c.rank).collect();
                    let mid_kicker = mid_kickers.iter().max().copied().unwrap_or(0);
                    if top_kicker > mid_kicker {
                        return false;
                    }
                }
            }
            HandRank3::Trips => {
                // Already handled above in special case
            }
        }
    }
    
    true
}

// ============================================================
//  Solver
// ============================================================

pub fn solve_fantasyland(cards: &[Card]) -> Option<Placement> {
    let n = cards.len();
    if n < 13 || n > 17 { return None; }
    
    let indices: Vec<usize> = (0..n).collect();
    
    let best = indices.iter().copied()
        .combinations(5)
        .collect::<Vec<_>>()
        .into_par_iter()
        .filter_map(|bot_idx| {
            let bottom: Vec<Card> = bot_idx.iter().map(|&i| cards[i]).collect();
            let remaining: Vec<usize> = indices.iter()
                .copied()
                .filter(|i| !bot_idx.contains(i))
                .collect();
            find_best_for_bottom(cards, &bottom, &remaining)
        })
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    
    best
}

fn find_best_for_bottom(cards: &[Card], bottom: &[Card], remaining: &[usize]) -> Option<Placement> {
    let mut best: Option<Placement> = None;
    let mut best_score = f64::NEG_INFINITY;
    
    for mid_idx in remaining.iter().copied().combinations(5) {
        let middle: Vec<Card> = mid_idx.iter().map(|&i| cards[i]).collect();
        let after_mid: Vec<usize> = remaining.iter()
            .copied()
            .filter(|i| !mid_idx.contains(i))
            .collect();
        
        if after_mid.len() < 3 { continue; }
        
        for top_idx in after_mid.iter().copied().combinations(3) {
            let top: Vec<Card> = top_idx.iter().map(|&i| cards[i]).collect();
            let discards: Vec<Card> = after_mid.iter()
                .copied()
                .filter(|i| !top_idx.contains(i))
                .map(|i| cards[i])
                .collect();
            
            if discards.iter().any(|c| c.is_joker()) { continue; }
            if raw_score_bound(&top, &middle, bottom) <= best_score { continue; }
            let canonical = canonical_rows(&top, &middle, bottom);
            if canonical.busted { continue; }

            let top_roy = canonical.top_royalty;
            let mid_roy = canonical.middle_royalty;
            let bot_roy = canonical.bottom_royalty;
            let can_stay = canonical.can_stay;
            let total = top_roy + mid_roy + bot_roy;
            let stay_bonus = if can_stay { 100.0 } else { 0.0 };
            let score = total as f64 + stay_bonus;
            
            if score > best_score {
                best_score = score;
                best = Some(Placement {
                    top: top.clone(),
                    middle: middle.clone(),
                    bottom: bottom.to_vec(),
                    discards: discards.clone(),
                    top_royalty: top_roy,
                    middle_royalty: mid_roy,
                    bottom_royalty: bot_roy,
                    total_royalty: total,
                    can_stay,
                    is_bust: false,
                    score,
                });
            }
        }
    }
    best
}

// ============================================================
//  Table-Driven Exact Solver v3
// ============================================================
//
// The v1 exhaustive search re-evaluates every row of every candidate from
// scratch: at 17 cards that is C(17,5)*C(12,5)*C(7,3) = 171M candidates
// times several hand evaluations each (measured 37-180s per hand).  Here
// every 5-card and 3-card subset is evaluated exactly once into a table
// (6,188 + 680 rows at 17 cards), and the search walks tables sorted by
// royalty-plus-stay upper bound with branch-and-bound cuts.  The bound is
// the same sound raw bound v1 uses (jokers at full strength, stay from raw
// trips-top / quads-bottom), so pruning never changes the argmax; canonical
// evaluation runs only on candidates whose bound beats the incumbent.
// Unlike v2's role phases, no candidate shape is ever assumed: this is the
// full exact argmax, at production latency.

/// One evaluated 5-card subset: royalty read at raw (joker-max) strength.
struct Sub5 {
    mask: u32,
    roy_mid: i32,
    roy_bot: i32,
    /// Raw rank reaches quads or better: bottom-row stay is possible.
    stay_bot: bool,
    /// Canonical hand value on ofc_core's cross-row scale; exact when the
    /// subset has no joker, joker-max otherwise.
    value: u32,
    has_joker: bool,
}

/// One evaluated 3-card subset.
struct Sub3 {
    mask: u32,
    roy_top: i32,
    /// Raw rank is trips: top-row stay is possible.
    trips: bool,
    value: u32,
    has_joker: bool,
}

fn cards_of_mask(cards: &[Card], mask: u32) -> Vec<Card> {
    (0..cards.len())
        .filter(|index| mask & (1 << index) != 0)
        .map(|index| cards[index])
        .collect()
}

fn subset_masks(n: usize, k: usize) -> Vec<u32> {
    let mut out = Vec::new();
    let mut indices: Vec<usize> = (0..k).collect();
    loop {
        out.push(indices.iter().fold(0u32, |mask, i| mask | 1 << i));
        // Next combination in lexicographic order.
        let mut position = k;
        loop {
            if position == 0 {
                return out;
            }
            position -= 1;
            if indices[position] != position + n - k {
                indices[position] += 1;
                for later in (position + 1)..k {
                    indices[later] = indices[later - 1] + 1;
                }
                break;
            }
        }
    }
}

pub fn solve_fantasyland_v3(cards: &[Card]) -> Option<Placement> {
    let n = cards.len();
    if n < 13 || n > 17 {
        return None;
    }
    let jokers_mask: u32 = (0..n)
        .filter(|index| cards[*index].is_joker())
        .fold(0u32, |mask, index| mask | 1 << index);

    let fives: Vec<Sub5> = subset_masks(n, 5)
        .into_iter()
        .map(|mask| {
            let row = cards_of_mask(cards, mask);
            let (rank, _) = evaluate_5_card(&row);
            Sub5 {
                mask,
                roy_mid: get_middle_royalty(&row),
                roy_bot: get_bottom_royalty(&row),
                stay_bot: matches!(
                    rank,
                    HandRank::Quads | HandRank::StraightFlush | HandRank::RoyalFlush
                ),
                value: ofc_core::evaluate_hand_value(&to_core_cards(&row), 5),
                has_joker: row.iter().any(|card| card.is_joker()),
            }
        })
        .collect();
    let mut threes: Vec<Sub3> = subset_masks(n, 3)
        .into_iter()
        .map(|mask| {
            let row = cards_of_mask(cards, mask);
            let (rank, _) = evaluate_3_card(&row);
            Sub3 {
                mask,
                roy_top: get_top_royalty(&row),
                trips: rank == HandRank3::Trips,
                value: ofc_core::evaluate_hand_value(&to_core_cards(&row), 3),
                has_joker: row.iter().any(|card| card.is_joker()),
            }
        })
        .collect();

    // Bottoms by royalty + own stay grant; mids by royalty; tops by royalty
    // + own stay grant.  Descending, so incumbents form fast and the sorted
    // prefix bounds justify loop breaks.
    let mut bots: Vec<&Sub5> = fives.iter().collect();
    bots.sort_by_key(|sub| -(sub.roy_bot + if sub.stay_bot { 100 } else { 0 }));
    let mut mids: Vec<&Sub5> = fives.iter().collect();
    mids.sort_by_key(|sub| -sub.roy_mid);
    threes.sort_by_key(|sub| -(sub.roy_top + if sub.trips { 100 } else { 0 }));

    let key_bot = |sub: &Sub5| sub.roy_bot + if sub.stay_bot { 100 } else { 0 };
    let key_top = |sub: &Sub3| sub.roy_top + if sub.trips { 100 } else { 0 };
    let max_roy_mid = mids.first().map(|sub| sub.roy_mid).unwrap_or(0);
    let max_key_top = threes.iter().map(&key_top).max().unwrap_or(0);
    // Weakest trips on the canonical cross-row scale (2-2-2).  A joker top
    // can only canonically reach trips -- and therefore stay -- if the
    // middle admits a value at least this strong above it.
    let trips_min = ofc_core::evaluate_hand_value(
        &to_core_cards(&[
            Card { rank: 2, suit: 0 },
            Card { rank: 2, suit: 1 },
            Card { rank: 2, suit: 2 },
        ]),
        3,
    );

    // Shared incumbent: scores are integers (royalty sums plus the 100 stay
    // grant), so an AtomicI32 carries them exactly across rayon threads and
    // every thread prunes against the best score any thread has found.
    use std::sync::atomic::{AtomicI32, Ordering as AtomicOrdering};
    let incumbent = AtomicI32::new(i32::MIN);

    let best = bots
        .par_iter()
        .enumerate()
        .filter_map(|(bot_rank, bot)| {
            let mut local_best: Option<Placement> = None;
            // Sorted bots: everything from here on has a bound no better
            // than this one, but with rayon the ranks run out of order, so
            // this is a skip rather than a break.
            let bound_bot = key_bot(bot) + max_roy_mid + max_key_top;
            if bound_bot <= incumbent.load(AtomicOrdering::Relaxed) {
                return None;
            }
            let _ = bot_rank;
            for mid in &mids {
                if mid.mask & bot.mask != 0 {
                    continue;
                }
                if key_bot(bot) + mid.roy_mid + max_key_top
                    <= incumbent.load(AtomicOrdering::Relaxed)
                {
                    break;
                }
                // Joker-free rows carry exact canonical values, so an
                // ordering violation is a guaranteed bust -- skip before
                // any canonical work.
                if !bot.has_joker && !mid.has_joker && mid.value > bot.value {
                    continue;
                }
                let used_bm = bot.mask | mid.mask;
                // Top-row stay requires canonical trips, which the ordering
                // constraint caps at the middle's value; a joker-free middle
                // below the weakest trips rules it out for the whole pair.
                let top_stay_open = mid.has_joker || mid.value >= trips_min;
                for top in &threes {
                    if key_bot(bot) + mid.roy_mid + key_top(top)
                        <= incumbent.load(AtomicOrdering::Relaxed)
                    {
                        break;
                    }
                    if top.mask & used_bm != 0 {
                        continue;
                    }
                    if !top.has_joker && !mid.has_joker && top.value > mid.value {
                        continue;
                    }
                    // Per-candidate effective bound (v1 had this; its absence
                    // let joker-trips tops flood canonical evaluation): stay
                    // only counts when actually reachable for this pair.
                    let stay_reachable =
                        bot.stay_bot || (top.trips && top_stay_open);
                    let bound = bot.roy_bot
                        + mid.roy_mid
                        + top.roy_top
                        + if stay_reachable { 100 } else { 0 };
                    if bound <= incumbent.load(AtomicOrdering::Relaxed) {
                        continue;
                    }
                    let used = used_bm | top.mask;
                    // Jokers may never be discarded (owner rule: strictly
                    // dominant to keep them).
                    if jokers_mask & used != jokers_mask {
                        continue;
                    }
                    let top_cards = cards_of_mask(cards, top.mask);
                    let mid_cards = cards_of_mask(cards, mid.mask);
                    let bot_cards = cards_of_mask(cards, bot.mask);
                    let canonical = canonical_rows(&top_cards, &mid_cards, &bot_cards);
                    if canonical.busted {
                        continue;
                    }
                    let total = canonical.top_royalty
                        + canonical.middle_royalty
                        + canonical.bottom_royalty;
                    let stay = canonical.can_stay;
                    let score_int = total + if stay { 100 } else { 0 };
                    let previous = incumbent.fetch_max(score_int, AtomicOrdering::Relaxed);
                    let improved_locally = local_best
                        .as_ref()
                        .map(|placement| (score_int as f64) > placement.score)
                        .unwrap_or(true);
                    if score_int > previous || improved_locally {
                        let discards = (0..n)
                            .filter(|index| used & (1 << index) == 0)
                            .map(|index| cards[index])
                            .collect();
                        local_best = Some(Placement {
                            top: top_cards.clone(),
                            middle: mid_cards.clone(),
                            bottom: bot_cards.clone(),
                            discards,
                            top_royalty: canonical.top_royalty,
                            middle_royalty: canonical.middle_royalty,
                            bottom_royalty: canonical.bottom_royalty,
                            total_royalty: total,
                            can_stay: stay,
                            is_bust: false,
                            score: score_int as f64,
                        });
                    }
                }
            }
            local_best
        })
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    best
}

// ============================================================
//  Role-Based Solver v2 (Optimized)
// ============================================================

/// Remove used cards from the set
fn remove_cards(cards: &[Card], used: &[Card]) -> Vec<Card> {
    let mut result = cards.to_vec();
    for u in used {
        if let Some(pos) = result.iter().position(|c| c == u) {
            result.remove(pos);
        }
    }
    result
}

/// Find all Royal Flush combinations (5 cards)
fn find_royal_flushes(cards: &[Card]) -> Vec<Vec<Card>> {
    let jokers: Vec<Card> = cards.iter().filter(|c| c.is_joker()).copied().collect();
    let num_jokers = jokers.len();
    let mut results = Vec::new();
    
    // For each suit, check if AKQJT is present (with jokers)
    for suit in 0..4u8 {
        let royals: Vec<Card> = cards.iter()
            .filter(|c| !c.is_joker() && c.suit == suit && c.rank >= 10)
            .copied()
            .collect();
        
        let needed = 5 - royals.len();
        if needed <= num_jokers {
            let mut hand = royals.clone();
            hand.extend(jokers.iter().take(needed));
            if hand.len() == 5 {
                results.push(hand);
            }
        }
    }
    results
}

/// Find all Straight Flush combinations (5 cards)
fn find_straight_flushes(cards: &[Card]) -> Vec<Vec<Card>> {
    let jokers: Vec<Card> = cards.iter().filter(|c| c.is_joker()).copied().collect();
    let num_jokers = jokers.len();
    let mut results = Vec::new();
    
    // Check each suit and each starting rank
    let straights: [[u8; 5]; 10] = [
        [14, 2, 3, 4, 5],   // A2345 (wheel)
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
        [6, 7, 8, 9, 10],
        [7, 8, 9, 10, 11],
        [8, 9, 10, 11, 12],
        [9, 10, 11, 12, 13],
        [10, 11, 12, 13, 14], // Broadway (skip, handled by Royal)
    ];
    
    for suit in 0..4u8 {
        for straight in straights.iter().take(9) { // Skip Broadway
            let present: Vec<Card> = straight.iter()
                .filter_map(|&r| cards.iter().find(|c| !c.is_joker() && c.suit == suit && c.rank == r))
                .copied()
                .collect();
            
            let needed = 5 - present.len();
            if needed <= num_jokers {
                let mut hand = present.clone();
                hand.extend(jokers.iter().take(needed));
                if hand.len() == 5 {
                    results.push(hand);
                }
            }
        }
    }
    results
}

/// Find all Quads combinations (5 cards = 4 of a kind + kicker)
fn find_quads(cards: &[Card]) -> Vec<Vec<Card>> {
    let jokers: Vec<Card> = cards.iter().filter(|c| c.is_joker()).copied().collect();
    let num_jokers = jokers.len();
    let rank_counts = count_ranks(cards);
    let mut results = Vec::new();
    
    for rank in (2..=14).rev() {
        let count = rank_counts[rank as usize];
        let needed = 4 - count as usize;
        if needed <= num_jokers {
            // Build the quads
            let quad_cards: Vec<Card> = cards.iter()
                .filter(|c| !c.is_joker() && c.rank == rank)
                .copied()
                .collect();
            let mut hand = quad_cards.clone();
            hand.extend(jokers.iter().take(needed));
            
            // Find best kicker
            let remaining = remove_cards(cards, &hand);
            for kicker in remaining.iter().filter(|c| !c.is_joker()) {
                let mut full_hand = hand.clone();
                full_hand.push(*kicker);
                results.push(full_hand);
            }
            // Also try using a joker as kicker if we have extras
            if hand.len() == 4 && num_jokers > needed {
                hand.push(jokers[needed]);
                results.push(hand);
            }
        }
    }
    results
}

/// Find all Trips combinations (3 cards for Top)
fn find_trips_3(cards: &[Card]) -> Vec<Vec<Card>> {
    let jokers: Vec<Card> = cards.iter().filter(|c| c.is_joker()).copied().collect();
    let num_jokers = jokers.len();
    let rank_counts = count_ranks(cards);
    let mut results = Vec::new();
    
    for rank in (2..=14).rev() {
        let count = rank_counts[rank as usize];
        let needed = 3 - count as usize;
        if count >= 1 && needed <= num_jokers {
            let trip_cards: Vec<Card> = cards.iter()
                .filter(|c| !c.is_joker() && c.rank == rank)
                .take(3)
                .copied()
                .collect();
            let mut hand = trip_cards.clone();
            hand.extend(jokers.iter().take(needed));
            if hand.len() == 3 {
                results.push(hand);
            }
        }
    }
    results
}

/// Find all Full House combinations (5 cards)
fn find_full_houses(cards: &[Card]) -> Vec<Vec<Card>> {
    let jokers: Vec<Card> = cards.iter().filter(|c| c.is_joker()).copied().collect();
    let num_jokers = jokers.len();
    let rank_counts = count_ranks(cards);
    let mut results = Vec::new();
    
    // For each possible trips rank and pair rank
    for trips_rank in (2..=14).rev() {
        for pair_rank in (2..=14).rev() {
            if trips_rank == pair_rank { continue; }
            
            let trips_count = rank_counts[trips_rank as usize];
            let pair_count = rank_counts[pair_rank as usize];
            
            let trips_needed = if trips_count >= 3 { 0 } else { 3 - trips_count as usize };
            let pair_needed = if pair_count >= 2 { 0 } else { 2 - pair_count as usize };
            
            if trips_needed + pair_needed <= num_jokers {
                // Build the hand
                let trips_cards: Vec<Card> = cards.iter()
                    .filter(|c| !c.is_joker() && c.rank == trips_rank)
                    .take(3)
                    .copied()
                    .collect();
                let pair_cards: Vec<Card> = cards.iter()
                    .filter(|c| !c.is_joker() && c.rank == pair_rank)
                    .take(2)
                    .copied()
                    .collect();
                
                let mut hand = trips_cards;
                hand.extend(pair_cards);
                // Add jokers to complete
                let jokers_used = trips_needed + pair_needed;
                hand.extend(jokers.iter().take(jokers_used));
                
                if hand.len() == 5 {
                    results.push(hand);
                }
            }
        }
    }
    results
}

/// Find best placement for given bottom hand
fn find_best_top_middle(cards: &[Card], bottom: &[Card]) -> Option<Placement> {
    let remaining = remove_cards(cards, bottom);
    if remaining.len() < 8 { return None; }
    
    let mut best: Option<Placement> = None;
    let mut best_score = f64::NEG_INFINITY;
    
    // Try all middle combinations
    for mid_combo in remaining.iter().copied().combinations(5) {
        let middle: Vec<Card> = mid_combo;
        let after_mid = remove_cards(&remaining, &middle);
        
        if after_mid.len() < 3 { continue; }
        
        // Try all top combinations
        for top_combo in after_mid.iter().copied().combinations(3) {
            let top: Vec<Card> = top_combo;
            let discards = remove_cards(&after_mid, &top);
            
            // Don't discard jokers
            if discards.iter().any(|c| c.is_joker()) { continue; }
            if raw_score_bound(&top, &middle, bottom) <= best_score { continue; }
            
            // Check valid placement
            let canonical = canonical_rows(&top, &middle, bottom);
            if canonical.busted { continue; }

            let top_roy = canonical.top_royalty;
            let mid_roy = canonical.middle_royalty;
            let bot_roy = canonical.bottom_royalty;
            let can_stay = canonical.can_stay;
            let total = top_roy + mid_roy + bot_roy;
            let stay_bonus = if can_stay { 100.0 } else { 0.0 };
            let score = total as f64 + stay_bonus;
            
            if score > best_score {
                best_score = score;
                best = Some(Placement {
                    top: top.clone(),
                    middle: middle.clone(),
                    bottom: bottom.to_vec(),
                    discards: discards.clone(),
                    top_royalty: top_roy,
                    middle_royalty: mid_roy,
                    bottom_royalty: bot_roy,
                    total_royalty: total,
                    can_stay,
                    is_bust: false,
                    score,
                });
            }
        }
    }
    best
}

/// Phase A: Bottom FL Stay (Quads, SF, RF)
fn phase_a_bottom_fl_stay(cards: &[Card]) -> Option<Placement> {
    let mut best: Option<Placement> = None;
    
    // A1: Royal Flush
    for rf in find_royal_flushes(cards) {
        if let Some(p) = find_best_top_middle(cards, &rf) {
            if best.as_ref().map(|b| p.score > b.score).unwrap_or(true) {
                best = Some(p);
            }
        }
    }
    
    // A2: Straight Flush
    for sf in find_straight_flushes(cards) {
        if let Some(p) = find_best_top_middle(cards, &sf) {
            if best.as_ref().map(|b| p.score > b.score).unwrap_or(true) {
                best = Some(p);
            }
        }
    }
    
    // A3: Quads
    for quads in find_quads(cards) {
        if let Some(p) = find_best_top_middle(cards, &quads) {
            if best.as_ref().map(|b| p.score > b.score).unwrap_or(true) {
                best = Some(p);
            }
        }
    }
    
    best
}

/// Phase B: Top FL Stay (Trips)
fn phase_b_top_fl_stay(cards: &[Card]) -> Option<Placement> {
    let mut best: Option<Placement> = None;
    
    // Try each possible Trips for Top
    for trips in find_trips_3(cards) {
        let remaining = remove_cards(cards, &trips);
        
        // Try all bottom combinations from remaining
        for bot_combo in remaining.iter().copied().combinations(5) {
            let bottom: Vec<Card> = bot_combo;
            let after_bot = remove_cards(&remaining, &bottom);
            
            if after_bot.len() < 5 { continue; }
            
            // Try all middle combinations
            for mid_combo in after_bot.iter().copied().combinations(5) {
                let middle: Vec<Card> = mid_combo;
                let discards = remove_cards(&after_bot, &middle);
                
                if discards.iter().any(|c| c.is_joker()) { continue; }
                if best
                    .as_ref()
                    .map(|b| raw_score_bound(&trips, &middle, &bottom) <= b.score)
                    .unwrap_or(false)
                {
                    continue;
                }
                let canonical = canonical_rows(&trips, &middle, &bottom);
                if canonical.busted { continue; }

                let top_roy = canonical.top_royalty;
                let mid_roy = canonical.middle_royalty;
                let bot_roy = canonical.bottom_royalty;
                let can_stay = canonical.can_stay;
                let total = top_roy + mid_roy + bot_roy;
                let stay_bonus = if can_stay { 100.0 } else { 0.0 };
                let score = total as f64 + stay_bonus;
                
                if best.as_ref().map(|b| score > b.score).unwrap_or(true) {
                    best = Some(Placement {
                        top: trips.clone(),
                        middle: middle.clone(),
                        bottom: bottom.clone(),
                        discards: discards.clone(),
                        top_royalty: top_roy,
                        middle_royalty: mid_roy,
                        bottom_royalty: bot_roy,
                        total_royalty: total,
                        can_stay,
                        is_bust: false,
                        score,
                    });
                }
            }
        }
    }
    best
}

/// Find all Flush combinations (5 cards of same suit)
fn find_flushes(cards: &[Card]) -> Vec<Vec<Card>> {
    let jokers: Vec<Card> = cards.iter().filter(|c| c.is_joker()).copied().collect();
    let num_jokers = jokers.len();
    let mut results = Vec::new();
    
    for suit in 0..4u8 {
        let suited: Vec<Card> = cards.iter()
            .filter(|c| !c.is_joker() && c.suit == suit)
            .copied()
            .collect();
        
        if suited.len() + num_jokers >= 5 {
            // Generate all combinations of suited cards + jokers
            let needed = 5 - suited.len().min(5);
            if needed <= num_jokers {
                for combo in suited.iter().copied().combinations(5.min(suited.len())) {
                    let mut hand = combo;
                    hand.extend(jokers.iter().take(5 - hand.len()));
                    if hand.len() == 5 {
                        results.push(hand);
                    }
                }
            }
        }
    }
    results
}

/// Find all Straight combinations (5 consecutive ranks)
fn find_straights(cards: &[Card]) -> Vec<Vec<Card>> {
    let jokers: Vec<Card> = cards.iter().filter(|c| c.is_joker()).copied().collect();
    let num_jokers = jokers.len();
    let rank_counts = count_ranks(cards);
    let mut results = Vec::new();
    
    let straights: [[u8; 5]; 10] = [
        [14, 2, 3, 4, 5],   // Wheel
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
    
    for straight in straights.iter() {
        let missing: usize = straight.iter()
            .filter(|&&r| rank_counts[r as usize] == 0)
            .count();
        
        if missing <= num_jokers {
            // Build a straight using one card of each rank
            let mut hand: Vec<Card> = Vec::new();
            for &r in straight {
                if let Some(c) = cards.iter().find(|c| !c.is_joker() && c.rank == r) {
                    hand.push(*c);
                }
            }
            // Fill with jokers
            hand.extend(jokers.iter().take(5 - hand.len()));
            if hand.len() == 5 {
                results.push(hand);
            }
        }
    }
    results
}

/// Phase C: No FL Stay (maximize royalties)
fn phase_c_no_fl_stay(cards: &[Card]) -> Option<Placement> {
    let mut best: Option<Placement> = None;
    
    // C1: Full House in Bottom (max 27 points: FH6 + FH12 + AA9)
    for fh in find_full_houses(cards) {
        if let Some(p) = find_best_top_middle(cards, &fh) {
            if best.as_ref().map(|b| p.score > b.score).unwrap_or(true) {
                best = Some(p);
            }
        }
    }
    
    // Early exit if we found a great solution
    if best.as_ref().map(|p| p.score >= 22.0).unwrap_or(false) {
        return best;
    }
    
    // C2: Flush in Bottom (max 21 points: Fl4 + Fl8 + AA9)
    for fl in find_flushes(cards) {
        if let Some(p) = find_best_top_middle(cards, &fl) {
            if best.as_ref().map(|b| p.score > b.score).unwrap_or(true) {
                best = Some(p);
            }
        }
    }
    
    // Early exit
    if best.as_ref().map(|p| p.score >= 15.0).unwrap_or(false) {
        return best;
    }
    
    // C3: Straight in Bottom (max 15 points: St2 + St4 + AA9)
    for st in find_straights(cards) {
        if let Some(p) = find_best_top_middle(cards, &st) {
            if best.as_ref().map(|b| p.score > b.score).unwrap_or(true) {
                best = Some(p);
            }
        }
    }
    
    best
}

/// Choose best placement from two Options
fn max_placement(a: Option<Placement>, b: Option<Placement>) -> Option<Placement> {
    match (a, b) {
        (Some(pa), Some(pb)) => Some(if pa.score >= pb.score { pa } else { pb }),
        (Some(p), None) | (None, Some(p)) => Some(p),
        (None, None) => None,
    }
}

/// Role-based solver v2 - much faster than exhaustive
pub fn solve_fantasyland_v2(cards: &[Card]) -> Option<Placement> {
    let n = cards.len();
    if n < 13 || n > 17 { return None; }
    
    // Phase A: Bottom FL Stay (RF, SF, Quads)
    let a = phase_a_bottom_fl_stay(cards);
    
    // Early exit: if score >= 41, can't do better with Phase B
    if a.as_ref().map(|p| p.score >= 41.0).unwrap_or(false) {
        return a;
    }
    
    // Phase B: Top FL Stay (Trips)
    let b = phase_b_top_fl_stay(cards);
    let best_ab = max_placement(a, b);
    
    // Early exit: if score >= 41, done
    if best_ab.as_ref().map(|p| p.score >= 41.0).unwrap_or(false) {
        return best_ab;
    }
    
    // Phase C: No FL Stay (Full House for royalties)
    let c = phase_c_no_fl_stay(cards);
    
    // Fall back to exhaustive if no role-based solution found
    let best_abc = max_placement(best_ab, c);
    if best_abc.is_some() {
        return best_abc;
    }
    
    // Fallback to exhaustive search
    solve_fantasyland(cards)
}

// ============================================================
//  Main - JSON stdin/stdout interface
// ============================================================

#[derive(Deserialize)]
struct OpponentBoard {
    top: Vec<Card>,
    middle: Vec<Card>,
    bottom: Vec<Card>,
}

#[derive(Deserialize)]
struct Request {
    cards: Vec<Card>,
    #[serde(default)]
    version: u8,  // 0 or 1 = v1 (exhaustive), 2 = v2 (role-based)
    #[serde(default)]
    opponent: Option<OpponentBoard>,
}

#[derive(Serialize)]
struct Response {
    success: bool,
    placement: Option<Placement>,
    error: Option<String>,
    /// The non-dominated arrangements, when the request asked for version 4.
    ///
    /// Stored WITHOUT the Fantasyland constant applied. The frontier is the
    /// same set for every `fl_ev >= 0` -- royalty and the stay flag are both
    /// monotone in the row values, so the fourth dominance component adds
    /// nothing -- which was checked by building at 0 and at 63.5 and getting
    /// identical sets. So a library of these survives a re-derived `fl_ev`
    /// table: the constant is applied when scoring, not when solving.
    #[serde(skip_serializing_if = "Option::is_none")]
    frontier: Option<Vec<FrontierRow>>,
}

/// One frontier row on the wire: three canonical row values, the royalty they
/// pay, and whether the arrangement keeps Fantasyland.
#[derive(Serialize)]
struct FrontierRow {
    v: [u32; 3],
    r: i32,
    s: bool,
}

// ============================================================
//  FL vs Normal opponent scoring
// ============================================================

fn compare_3_hands(a: &[Card], b: &[Card]) -> i32 {
    ofc_core::compare_3_hands(&to_core_cards(a), &to_core_cards(b))
}

fn score_vs_opponent(
    fl_top: &[Card], fl_mid: &[Card], fl_bot: &[Card],
    opp: &OpponentBoard,
    fl_top_roy: i32, fl_mid_roy: i32, fl_bot_roy: i32,
    can_stay: bool, current_fl_cards: u8,
) -> f64 {
    let opponent = canonical_rows(&opp.top, &opp.middle, &opp.bottom);
    let own_royalty = fl_top_roy + fl_mid_roy + fl_bot_roy;
    let mut score = if opponent.busted {
        (6 + own_royalty) as f64
    } else {
        let top_win = compare_3_hands(fl_top, &opponent.top);
        let mid_win = compare_5_hands(fl_mid, &opponent.middle);
        let bot_win = compare_5_hands(fl_bot, &opponent.bottom);
        let line_score = top_win + mid_win + bot_win;
        let scoop = if line_score == 3 { 3 } else if line_score == -3 { -3 } else { 0 };
        let opponent_royalty = opponent.top_royalty
            + opponent.middle_royalty
            + opponent.bottom_royalty;
        (line_score + scoop + own_royalty - opponent_royalty) as f64
    };

    // FL stay chain EV
    if can_stay {
        let fl_ev = match current_fl_cards {
            14 => 14.0,
            15 => 27.9,
            16 => 52.4,
            17 => 104.5,
            _ => 0.0,
        };
        score += fl_ev;
    }

    score
}

/// Solve FL placement against a known opponent board.
/// 1. Find max royalty among all valid placements
/// 2. Collect candidates within 4 royalty of max
/// 3. Score each vs opponent (lines + scoop + royalty diff + FL stay EV)
/// 4. Return the best
fn solve_fl_vs_normal(cards: &[Card], opp: &OpponentBoard) -> Option<Placement> {
    let n = cards.len();
    let current_fl_cards = n as u8;

    // Pass 1: find max royalty
    let mut max_royalty = i32::MIN;
    let bot_combos: Vec<Vec<usize>> = (0..n).combinations(5).collect();

    for bot_idx in &bot_combos {
        let remaining: Vec<usize> = (0..n).filter(|i| !bot_idx.contains(i)).collect();
        let bot_c: Vec<Card> = bot_idx.iter().map(|&i| cards[i]).collect();

        for mid_idx in remaining.iter().copied().combinations(5) {
            let mid_c: Vec<Card> = mid_idx.iter().map(|&i| cards[i]).collect();
            let mid_has_joker = mid_c.iter().any(|c| c.is_joker());
            if !mid_has_joker && compare_5_hands(&bot_c, &mid_c) < 0 { continue; }

            let after_mid: Vec<usize> = remaining.iter().copied()
                .filter(|i| !mid_idx.contains(i)).collect();

            for top_idx in after_mid.iter().copied().combinations(3) {
                let disc_idx: Vec<usize> = after_mid.iter().copied()
                    .filter(|i| !top_idx.contains(i)).collect();
                if disc_idx.iter().any(|&i| cards[i].is_joker()) { continue; }

                let top_c: Vec<Card> = top_idx.iter().map(|&i| cards[i]).collect();
                let canonical = canonical_rows(&top_c, &mid_c, &bot_c);
                if canonical.busted { continue; }

                let r = canonical.top_royalty
                    + canonical.middle_royalty
                    + canonical.bottom_royalty;
                if r > max_royalty { max_royalty = r; }
            }
        }
    }

    if max_royalty == i32::MIN { return None; }
    let threshold = max_royalty - 4;

    // Pass 2: collect candidates within threshold, score vs opponent
    let mut best: Option<Placement> = None;
    let mut best_score = f64::NEG_INFINITY;

    for bot_idx in &bot_combos {
        let remaining: Vec<usize> = (0..n).filter(|i| !bot_idx.contains(i)).collect();
        let bot_c: Vec<Card> = bot_idx.iter().map(|&i| cards[i]).collect();

        for mid_idx in remaining.iter().copied().combinations(5) {
            let mid_c: Vec<Card> = mid_idx.iter().map(|&i| cards[i]).collect();
            let mid_has_joker = mid_c.iter().any(|c| c.is_joker());
            if !mid_has_joker && compare_5_hands(&bot_c, &mid_c) < 0 { continue; }

            let after_mid: Vec<usize> = remaining.iter().copied()
                .filter(|i| !mid_idx.contains(i)).collect();

            for top_idx in after_mid.iter().copied().combinations(3) {
                let disc_idx: Vec<usize> = after_mid.iter().copied()
                    .filter(|i| !top_idx.contains(i)).collect();
                if disc_idx.iter().any(|&i| cards[i].is_joker()) { continue; }

                let top_c: Vec<Card> = top_idx.iter().map(|&i| cards[i]).collect();
                let canonical = canonical_rows(&top_c, &mid_c, &bot_c);
                if canonical.busted { continue; }

                let top_roy = canonical.top_royalty;
                let mid_roy = canonical.middle_royalty;
                let bot_roy = canonical.bottom_royalty;
                let total_roy = top_roy + mid_roy + bot_roy;

                if total_roy < threshold { continue; }

                let can_stay = canonical.can_stay;
                let score = score_vs_opponent(
                    &canonical.top, &canonical.middle, &canonical.bottom, opp,
                    top_roy, mid_roy, bot_roy,
                    can_stay, current_fl_cards,
                );

                if score > best_score {
                    best_score = score;
                    best = Some(Placement {
                        top: top_c,
                        middle: mid_c.clone(),
                        bottom: bot_c.clone(),
                        discards: disc_idx.iter().map(|&i| cards[i]).collect(),
                        top_royalty: top_roy,
                        middle_royalty: mid_roy,
                        bottom_royalty: bot_roy,
                        total_royalty: total_roy,
                        can_stay,
                        is_bust: false,
                        score,
                    });
                }
            }
        }
    }

    best
}

// ============================================================
//  Data Generation
// ============================================================

use rand::prelude::*;
use std::fs::File;

const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
const SUIT_CHARS: [char; 4] = ['s', 'h', 'd', 'c'];

fn rank_to_char(rank: u8) -> char {
    if rank >= 2 && rank <= 14 {
        RANK_CHARS[(rank - 2) as usize]
    } else if rank == 0 {
        'X'  // Joker
    } else {
        '?'
    }
}

fn suit_to_char(suit: u8) -> char {
    if suit < 4 {
        SUIT_CHARS[suit as usize]
    } else {
        'J'  // Joker
    }
}

fn create_deck(include_jokers: bool) -> Vec<Card> {
    let mut deck = Vec::new();
    for suit in 0..4u8 {
        for rank in 2..=14u8 {
            deck.push(Card { rank, suit });
        }
    }
    if include_jokers {
        deck.push(Card { rank: 0, suit: 4 }); // Joker 1
        deck.push(Card { rank: 0, suit: 4 }); // Joker 2
    }
    deck
}

fn deal_hand(num_cards: usize, include_jokers: bool, rng: &mut impl Rng) -> Vec<Card> {
    let mut deck = create_deck(include_jokers);
    deck.shuffle(rng);
    deck.into_iter().take(num_cards).collect()
}

fn card_to_string(card: &Card) -> String {
    if card.is_joker() {
        "JK".to_string()
    } else {
        format!("{}{}", rank_to_char(card.rank), suit_to_char(card.suit))
    }
}

#[derive(Serialize)]
struct DataSample {
    sample_id: usize,
    num_cards: usize,
    joker_count: usize,
    hand: Vec<String>,
    solution: SolutionStrings,
    reward: f64,
    royalties: i32,
    can_stay: bool,
}

#[derive(Serialize)]
struct SolutionStrings {
    top: Vec<String>,
    middle: Vec<String>,
    bottom: Vec<String>,
}

fn generate_data(samples: usize, num_cards: usize, fixed_jokers: Option<usize>, output_path: &str) {
    let mut rng = thread_rng();
    let mut results = Vec::new();
    
    let include_jokers = fixed_jokers.map(|j| j > 0).unwrap_or(true);
    
    let start = std::time::Instant::now();
    
    for i in 0..samples {
        // Deal hand (retry if joker count doesn't match)
        let hand = loop {
            let h = deal_hand(num_cards, include_jokers, &mut rng);
            let joker_count = h.iter().filter(|c| c.is_joker()).count();
            
            if let Some(target) = fixed_jokers {
                if joker_count == target {
                    break h;
                }
            } else {
                break h;
            }
        };
        
        let joker_count = hand.iter().filter(|c| c.is_joker()).count();
        
        // Solve using v2 (role-based, ~8x faster)
        if let Some(placement) = solve_fantasyland_v2(&hand) {
            let sample = DataSample {
                sample_id: i,
                num_cards,
                joker_count,
                hand: hand.iter().map(card_to_string).collect(),
                solution: SolutionStrings {
                    top: placement.top.iter().map(card_to_string).collect(),
                    middle: placement.middle.iter().map(card_to_string).collect(),
                    bottom: placement.bottom.iter().map(card_to_string).collect(),
                },
                reward: placement.score,
                royalties: placement.total_royalty,
                can_stay: placement.can_stay,
            };
            results.push(sample);
        }
        
        if (i + 1) % 100 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            let eta = (samples - i - 1) as f64 / rate;
            eprintln!("  {}/{} ({:.1}/s, ETA: {:.0}s)", i + 1, samples, rate, eta);
        }
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    
    // Statistics
    let stay_count = results.iter().filter(|r| r.can_stay).count();
    let total_royalties: i32 = results.iter().map(|r| r.royalties).sum();
    
    eprintln!("\n=== Statistics ===");
    eprintln!("FL Stay Rate: {}/{} ({:.1}%)", stay_count, results.len(), 
        100.0 * stay_count as f64 / results.len() as f64);
    eprintln!("Avg Royalties: {:.2}", total_royalties as f64 / results.len() as f64);
    
    // Write to file
    let file = File::create(output_path).expect("Failed to create output file");
    let mut writer = std::io::BufWriter::new(file);
    for sample in &results {
        writeln!(writer, "{}", serde_json::to_string(sample).unwrap()).unwrap();
    }
    
    eprintln!("\nDone! {} samples in {:.1}s ({:.1}/s)", results.len(), elapsed, 
        results.len() as f64 / elapsed);
    eprintln!("Output: {}", output_path);
}

fn run_stdin_mode() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                let resp = Response {
                    success: false,
                    placement: None,
                    error: Some(format!("Read error: {}", e)),
                    frontier: None,
                };
                writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap()).unwrap();
                continue;
            }
        };
        
        if line.trim().is_empty() { continue; }
        
        let request: Result<Request, _> = serde_json::from_str(&line);
        
        let resp = match request {
            Ok(req) => {
                let start = std::time::Instant::now();
                if req.version == 4 {
                    let rows = frontier::build_frontier(&req.cards, 0.0)
                        .into_iter()
                        .map(|entry| FrontierRow {
                            v: [entry.top, entry.mid, entry.bot],
                            r: entry.royalty,
                            s: entry.stays,
                        })
                        .collect();
                    writeln!(
                        stdout,
                        "{}",
                        serde_json::to_string(&Response {
                            success: true,
                            placement: None,
                            error: None,
                            frontier: Some(rows),
                        })
                        .unwrap()
                    )
                    .unwrap();
                    stdout.flush().unwrap();
                    continue;
                }
                let placement = if let Some(ref opp) = req.opponent {
                    eprintln!("Solving FL vs opponent board...");
                    solve_fl_vs_normal(&req.cards, opp)
                } else if req.version == 3 {
                    solve_fantasyland_v3(&req.cards)
                } else if req.version == 2 {
                    solve_fantasyland_v2(&req.cards)
                } else {
                    solve_fantasyland(&req.cards)
                };
                let elapsed = start.elapsed().as_secs_f64();
                let mode = if req.opponent.is_some() { "vs_opp" } else if req.version == 3 { "v3" } else if req.version == 2 { "v2" } else { "v1" };
                eprintln!("Solved {} in {:.3}s", mode, elapsed);
                
                Response {
                    success: true,
                    placement,
                    error: None,
                    frontier: None,
                }
            }
            Err(e) => Response {
                success: false,
                placement: None,
                error: Some(format!("Parse error: {}", e)),
                frontier: None,
            },
        };
        
        writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap()).unwrap();
        stdout.flush().unwrap();
    }
}

/// Build a pre-solved frontier pool: `fl_solver pool --entries N --out FILE`.
///
/// Parallel over entries, sequential within one -- each frontier's residual row
/// order is enumeration order, and a shared-frontier parallel sweep would make
/// the pool's bytes depend on the thread count.
fn build_pool(entries: usize, width: usize, seed: u64, table: [f64; 4], out_path: &str) {
    use rayon::prelude::*;
    let fl_ev = table[width.saturating_sub(14).min(3)];
    let started = std::time::Instant::now();
    let built: Vec<pool::PoolEntry> = (0..entries as u64)
        .into_par_iter()
        .map(|index| pool::build_entry(seed, index, width, fl_ev))
        .collect();
    let elapsed = started.elapsed().as_secs_f64();
    let rows: usize = built.iter().map(|entry| entry.rows.len()).sum();
    let image = pool::serialize(&pool::Pool {
        width: width as u32,
        fl_ev: table,
        seed,
        entries: built,
    });
    std::fs::write(out_path, &image).expect("write pool");
    eprintln!(
        "pool: {entries} entries, {rows} rows (mean {:.1}), {:.1} MB, {:.1}s ({:.1} ms/entry)",
        rows as f64 / entries as f64,
        image.len() as f64 / 1_048_576.0,
        elapsed,
        elapsed / entries as f64 * 1000.0,
    );
    // Read it back before claiming success: a pool that cannot be loaded under
    // the table it was written with is worse than no pool.
    let reloaded = pool::deserialize(&image, width as u32, table).expect("reload own pool");
    assert_eq!(reloaded.entries.len(), entries);
    eprintln!("pool: reload verified");
}

/// One T3 root read from `teach --roots-file`, as
/// `t4_first_exact --play-roots` writes it.
#[derive(Deserialize)]
struct RootRecord {
    id: String,
    rows: Vec<Vec<String>>,
    dead: Vec<String>,
    draw: Vec<String>,
}

/// A T3 root the file supplied, checked once so a malformed line fails the run
/// rather than one root of it.
struct FileRoot {
    id: String,
    rows: [Vec<Card>; 3],
    dead: Vec<Card>,
    draw: [Card; 3],
}

/// A card name to a `Card`.
///
/// Both joker spellings are accepted: the player keeps X1 and X2 apart because
/// they are two deck slots, and a board only ever needs to know that a joker is
/// a joker.
fn card_of_name(name: &str) -> Card {
    if name == "X1" || name == "X2" || name == "JK" || name == "X" {
        return Card { rank: 0, suit: 4 };
    }
    let bytes = name.as_bytes();
    assert_eq!(bytes.len(), 2, "not a card name: {name}");
    let rank = match bytes[0] as char {
        '2'..='9' => bytes[0] - b'0',
        'T' => 10,
        'J' => 11,
        'Q' => 12,
        'K' => 13,
        'A' => 14,
        other => panic!("invalid rank {other} in {name}"),
    };
    let suit = match bytes[1] as char {
        's' => 0,
        'h' => 1,
        'd' => 2,
        'c' => 3,
        other => panic!("invalid suit {other} in {name}"),
    };
    Card { rank, suit }
}

/// The roots a `--roots-file` supplies, in file order.
///
/// The dealt path derives a root's position from the seed and can afford to;
/// a played root cannot be re-derived from anything, so the checks that the
/// dealt path gets for free are made here instead.
/// `placed` is how many cards a root of this street has already put down:
/// nine at T3, seven at T2.  Passed rather than assumed, because reading a
/// T2 file with T3's expectation is a mistake the assertion should name.
fn read_roots_file(path: &str, placed_expected: usize) -> Vec<FileRoot> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read roots file {path}: {error}"));
    let mut out = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: RootRecord = serde_json::from_str(line)
            .unwrap_or_else(|error| panic!("{path}:{}: {error}", line_number + 1));
        // The label writer builds its JSON by hand, so an id carrying a quote
        // or a backslash would emit a record no reader can parse.
        assert!(
            !record.id.contains(['"', '\\']),
            "{path}:{}: id {:?} cannot be written into a JSON string",
            line_number + 1,
            record.id
        );
        assert_eq!(
            record.rows.len(),
            3,
            "{path}:{}: {} rows",
            line_number + 1,
            record.rows.len()
        );
        let rows = [
            record.rows[0].iter().map(|n| card_of_name(n)).collect::<Vec<Card>>(),
            record.rows[1].iter().map(|n| card_of_name(n)).collect::<Vec<Card>>(),
            record.rows[2].iter().map(|n| card_of_name(n)).collect::<Vec<Card>>(),
        ];
        for (row, capacity) in [3usize, 5, 5].iter().enumerate() {
            assert!(
                rows[row].len() <= *capacity,
                "{path}:{}: row {row} holds {} of {capacity}",
                line_number + 1,
                rows[row].len()
            );
        }
        let placed: usize = rows.iter().map(Vec::len).sum();
        assert_eq!(
            placed, placed_expected,
            "{path}:{}: {placed} cards placed, expected {placed_expected}",
            line_number + 1
        );
        assert_eq!(record.draw.len(), 3, "{path}:{}", line_number + 1);
        let draw: Vec<Card> = record.draw.iter().map(|n| card_of_name(n)).collect();
        out.push(FileRoot {
            id: record.id,
            rows,
            dead: record.dead.iter().map(|n| card_of_name(n)).collect(),
            draw: [draw[0], draw[1], draw[2]],
        });
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "teach" {
        use rayon::prelude::*;
        let mut pool_path = String::from("D:/ofc_data/fl_pools/fl14_v1.jfl1");
        let mut roots = 1000usize;
        let mut opponents = 240usize;
        let mut seed = 0xD00D_0001u64;
        let mut stream_offset = 0u64;
        let mut own_only = false;
        let mut t4_per_root = 2usize;
        let mut out_dir = String::from(".");
        let mut roots_file: Option<String> = None;
        let mut index = 2;
        while index < args.len() {
            match args[index].as_str() {
                "--pool" => { index += 1; pool_path = args[index].clone(); }
                "--roots" => { index += 1; roots = args[index].parse().expect("roots"); }
                "--opponents" => { index += 1; opponents = args[index].parse().expect("opponents"); }
                "--seed" => { index += 1; seed = args[index].parse().expect("seed"); }
                "--t4-per-root" => { index += 1; t4_per_root = args[index].parse().expect("t4"); }
                "--out-dir" => { index += 1; out_dir = args[index].clone(); }
                // Roots played by the trained chain instead of assigned from
                // the deal.  The dealt path fills the rows in deal order,
                // which is a legal T3 position and not one anybody reaches;
                // labels off it teach the model a distribution it never meets.
                // Everything downstream of the root is unchanged, so the two
                // teachers can be run over the same fourteen cards and
                // compared.
                "--roots-file" => { index += 1; roots_file = Some(args[index].clone()); }
                // Shifts the opponent-draw stream WITHOUT shifting the deal,
                // so a second pass labels the same positions against a fresh
                // set of opponents.  The opponent draw is the only sampled
                // quantity left in this teacher, so the spread between two
                // such passes is the floor a trained model is measured
                // against -- a model cannot be asked to beat the teacher's
                // own disagreement with itself.
                "--stream-offset" => { index += 1; stream_offset = args[index].parse().expect("stream-offset"); }
                // Prices leaves by hero's own worth instead of the score
                // against a best response.  A diagnostic pass, not a teacher.
                "--own-only" => { own_only = true; }
                _ => {}
            }
            index += 1;
        }
        let table = [0.0f64, 10.7, 29.9, 63.5];
        let bytes = std::fs::read(&pool_path).expect("read pool");
        let loaded = pool::deserialize(&bytes, 14, table).expect("load pool");
        eprintln!("pool: {} entries, width {}", loaded.entries.len(), loaded.width);
        let file_roots = roots_file.as_deref().map(|path| read_roots_file(path, 9));
        if let Some(supplied) = &file_roots {
            // The file decides how many roots there are; honouring --roots on
            // top of it would only be a way to label a prefix by accident.
            roots = supplied.len();
            eprintln!("roots: {roots} played, from {}", roots_file.as_deref().unwrap_or(""));
        }
        std::fs::create_dir_all(&out_dir).expect("out dir");
        let started = std::time::Instant::now();
        let done = std::sync::atomic::AtomicUsize::new(0);
        let short = std::sync::atomic::AtomicUsize::new(0);

        let produced: Vec<(String, String)> = (0..roots as u64)
            .into_par_iter()
            .map(|root| {
                let request = match &file_roots {
                    Some(supplied) => {
                        let played = &supplied[root as usize];
                        t3_labels::T3Request {
                            id: played.id.clone(),
                            rows: played.rows.clone(),
                            dead: played.dead.clone(),
                            draw: played.draw,
                            opponents,
                            t4_draws: 0,
                        }
                    }
                    None => {
                        let cards = pool::deal(seed, root, 14);
                        t3_labels::T3Request {
                            id: format!("{}", seed.wrapping_add(root)),
                            rows: [cards[0..2].to_vec(), cards[2..6].to_vec(), cards[6..9].to_vec()],
                            dead: cards[9..11].to_vec(),
                            draw: [cards[11], cards[12], cards[13]],
                            opponents,
                            t4_draws: 0,
                        }
                    }
                };
                let mut t3_line = String::new();
                let mut t4_lines = String::new();
                let stream = pool::stream_of(root, stream_offset);
                match t3_labels::solve_harvesting(&request, &loaded, &table, stream, root, t4_per_root, own_only) {
                    Ok((values, decisions)) => {
                        let actions: Vec<String> = values
                            .iter()
                            .map(|v| format!(
                                "{{\"action_key\":\"{}\",\"value\":{},\"t4_draws\":{}}}",
                                v.action_key, v.value, v.t4_draws))
                            .collect();
                        // The file carries the position it labels.  A teacher
                        // whose consumer has to re-derive the deal is one
                        // stream-function change away from being silently
                        // mislabelled.
                        t3_line = format!(
                            "{{\"id\":\"{}\",\"root\":{},\"stream\":{},\"opponents\":{},\"board\":\"{}\",                             \"dead\":\"{}\",\"draw\":\"{}\",\"actions\":[{}]}}
",
                            request.id, root, stream, opponents,
                            t3_labels::rows_key(&request.rows),
                            t3_labels::cards_key(&request.dead),
                            t3_labels::cards_key(&request.draw),
                            actions.join(","));
                        for decision in &decisions {
                            let acts: Vec<String> = decision.actions.iter()
                                .map(|(key, value)| format!("{{\"a\":\"{key}\",\"v\":{value}}}"))
                                .collect();
                            t4_lines.push_str(&format!(
                                "{{\"id\":\"{}\",\"root\":{},\"board\":\"{}\",                                 \"dead\":\"{}\",\"draw\":\"{}\",\"actions\":[{}]}}
",
                                request.id, root, decision.board_key,
                                t3_labels::cards_key(&request.dead),
                                t3_labels::cards_key(&decision.draw),
                                acts.join(",")));
                        }
                    }
                    Err(_) => { short.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                }
                let count = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if count % 500 == 0 {
                    let elapsed = started.elapsed().as_secs_f64();
                    eprintln!("[{count}/{roots}] {:.2} s/root, eta {:.0} min",
                        elapsed / count as f64,
                        (roots - count) as f64 * elapsed / count as f64 / 60.0);
                }
                (t3_line, t4_lines)
            })
            .collect();

        let mut t3_out = String::new();
        let mut t4_out = String::new();
        for (t3, t4) in &produced {
            t3_out.push_str(t3);
            t4_out.push_str(t4);
        }
        std::fs::write(format!("{out_dir}/t3_labels.jsonl"), &t3_out).expect("write t3");
        std::fs::write(format!("{out_dir}/t4_labels.jsonl"), &t4_out).expect("write t4");
        eprintln!(
            "teach: {} T3 roots, {} T4 decisions, {} short draws, {:.1} min total",
            t3_out.lines().count(),
            t4_out.lines().count(),
            short.load(std::sync::atomic::Ordering::Relaxed),
            started.elapsed().as_secs_f64() / 60.0,
        );
        return;
    }

    if args.len() > 1 && args[1] == "teach-t2" {
        let mut pool_path = String::from("D:/ofc_data/fl_pools/fl14_v1.jfl1");
        let mut roots = 100usize;
        let mut opponents = 60usize;
        let mut t3_draws = 24usize;
        let mut t4_draws = 0usize;
        let mut seed = 0xD00E_0001u64;
        let mut stream_offset = 0u64;
        let mut out_dir = String::from("D:/ofc_data/fl14_t2_teacher_v1");
        let mut roots_file: Option<String> = None;
        let mut index = 2;
        while index < args.len() {
            match args[index].as_str() {
                "--pool" => { index += 1; pool_path = args[index].clone(); }
                "--roots" => { index += 1; roots = args[index].parse().expect("roots"); }
                "--opponents" => { index += 1; opponents = args[index].parse().expect("opponents"); }
                "--t3-draws" => { index += 1; t3_draws = args[index].parse().expect("t3-draws"); }
                "--t4-draws" => { index += 1; t4_draws = args[index].parse().expect("t4-draws"); }
                "--seed" => { index += 1; seed = args[index].parse().expect("seed"); }
                "--stream-offset" => { index += 1; stream_offset = args[index].parse().expect("stream-offset"); }
                "--out-dir" => { index += 1; out_dir = args[index].clone(); }
                // Positions from a file instead of dealt ones.  The dealt shape
                // is `cards[0..2]/[2..5]/[5..7]`, i.e. (2,3,2) for every root --
                // one arrangement out of the many a played hand reaches, which
                // is the same defect the T3 teacher had.
                "--roots-file" => { index += 1; roots_file = Some(args[index].clone()); }
                _ => {}
            }
            index += 1;
        }
        let file_roots = roots_file.as_ref().map(|path| read_roots_file(path, 7));
        if let Some(list) = &file_roots {
            roots = list.len();
            eprintln!("teach-t2: {} roots from file", roots);
        }
        let table = [0.0f64, 10.7, 29.9, 63.5];
        let bytes = std::fs::read(&pool_path).expect("read pool");
        let loaded = pool::deserialize(&bytes, 14, table).expect("load pool");
        eprintln!("pool: {} entries, width {}", loaded.entries.len(), loaded.width);
        std::fs::create_dir_all(&out_dir).expect("out dir");
        let started = std::time::Instant::now();
        let done = std::sync::atomic::AtomicUsize::new(0);
        let short = std::sync::atomic::AtomicUsize::new(0);

        // A T2 root is expensive enough that the roots run one at a time and
        // the parallelism lives inside `solve`, across (action, T3 draw)
        // pairs -- the opposite of `teach`, where a root is small and the
        // roots themselves are the work units.
        // Written as they are produced, not collected and dumped at the end.
        // A T2 teacher is hours long; a run that publishes nothing until it
        // finishes is a run whose whole cost is lost to one interruption.
        use std::io::Write;
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(format!("{out_dir}/t2_labels.jsonl")).expect("t2 out"),
        );
        let mut written = 0usize;
        for root in 0..roots as u64 {
            let request = match &file_roots {
                Some(list) => {
                    let entry = &list[root as usize];
                    t2_labels::T2Request {
                        id: entry.id.clone(),
                        rows: entry.rows.clone(),
                        dead: entry.dead.clone(),
                        draw: entry.draw,
                        opponents,
                        t3_draws,
                        t4_draws,
                    }
                }
                None => {
                    let cards = pool::deal(seed, root, 11);
                    t2_labels::T2Request {
                        id: format!("{}", seed.wrapping_add(root)),
                        rows: [cards[0..2].to_vec(), cards[2..5].to_vec(), cards[5..7].to_vec()],
                        dead: cards[7..8].to_vec(),
                        draw: [cards[8], cards[9], cards[10]],
                        opponents,
                        t3_draws,
                        t4_draws,
                    }
                }
            };
            let stream = pool::stream_of(root, stream_offset);
            match t2_labels::solve(&request, &loaded, &table, stream) {
                Ok(values) => {
                    let actions: Vec<String> = values
                        .iter()
                        .map(|v| format!(
                            "{{\"action_key\":\"{}\",\"value\":{},\"t3_draws\":{}}}",
                            v.action_key, v.value, v.t3_draws))
                        .collect();
                    written += 1;
                    let line = format!(
                        "{{\"id\":\"{}\",\"root\":{},\"stream\":{},\"opponents\":{},\"board\":\"{}\",\"dead\":\"{}\",\"draw\":\"{}\",\"actions\":[{}]}}\n",
                        request.id, root, stream, opponents,
                        t3_labels::rows_key(&request.rows),
                        t3_labels::cards_key(&request.dead),
                        t3_labels::cards_key(&request.draw),
                        actions.join(","));
                    file.write_all(line.as_bytes()).expect("write");
                }
                Err(e) => {
                    short.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("root {root}: short draw {}/{}", e.found, e.wanted);
                }
            }
            let seen = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if seen % 10 == 0 || seen == roots {
                file.flush().expect("flush");
                let rate = started.elapsed().as_secs_f64() / seen as f64;
                eprintln!("[{seen}/{roots}] {:.2} s/root, eta {:.0} min",
                    rate, rate * (roots - seen) as f64 / 60.0);
            }
        }
        file.flush().expect("flush");
        eprintln!(
            "teach-t2: {} roots, {} short draws, {} opponents, {} T3 draws, \
             T4 draws {}, {:.1} min total",
            written, short.load(std::sync::atomic::Ordering::Relaxed),
            opponents, t3_draws,
            if t4_draws == 0 { "all".to_string() } else { t4_draws.to_string() },
            started.elapsed().as_secs_f64() / 60.0);
        return;
    }

    if args.len() > 1 && args[1] == "t3-bench" {
        let mut pool_path = String::from("D:/ofc_data/fl_pools/fl14_v1.jfl1");
        let mut roots = 5usize;
        let mut opponents = 64usize;
        let mut index = 2;
        while index < args.len() {
            match args[index].as_str() {
                "--pool" => { index += 1; pool_path = args[index].clone(); }
                "--roots" => { index += 1; roots = args[index].parse().expect("roots"); }
                "--opponents" => { index += 1; opponents = args[index].parse().expect("opponents"); }
                _ => {}
            }
            index += 1;
        }
        let table = [0.0f64, 10.7, 29.9, 63.5];
        let bytes = std::fs::read(&pool_path).expect("read pool");
        let loaded = pool::deserialize(&bytes, 14, table).expect("load pool");
        eprintln!("pool: {} entries", loaded.entries.len());
        let started = std::time::Instant::now();
        let mut actions = 0usize;
        let mut leaves_total = 0usize;
        let mut spreads = Vec::new();
        for root in 0..roots as u64 {
            let cards = pool::deal(0xC0C0_0000 + root, 0, 14);
            let request = t3_labels::T3Request {
                id: root.to_string(),
                rows: [cards[0..2].to_vec(), cards[2..6].to_vec(), cards[6..9].to_vec()],
                dead: cards[9..11].to_vec(),
                draw: [cards[11], cards[12], cards[13]],
                opponents,
                t4_draws: 0,
            };
            match t3_labels::solve_with_t4(&request, &loaded, &table, root, true, false) {
                Ok((values, leaves)) => {
                    actions += values.len();
                    leaves_total += leaves.len();
                    let best = values.iter().map(|v| v.value).fold(f64::MIN, f64::max);
                    let worst = values.iter().map(|v| v.value).fold(f64::MAX, f64::min);
                    spreads.push(best - worst);
                    if root == 0 {
                        eprintln!("root 0: {} actions, {} T4 leaves harvested, {} draws/action",
                            values.len(), leaves.len(), values[0].t4_draws);
                    }
                }
                Err(e) => eprintln!("root {root}: short draw {}/{}", e.found, e.wanted),
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        spreads.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "t3: {roots} roots, {actions} actions, {leaves_total} T4 leaves, {:.2} s/root",
            elapsed / roots as f64,
        );
        if !spreads.is_empty() {
            eprintln!("best-worst spread: median {:.3}, max {:.3}",
                spreads[spreads.len() / 2], spreads[spreads.len() - 1]);
        }
        return;
    }

    if args.len() > 1 && args[1] == "t4-bench" {
        let mut pool_path = String::from("D:/ofc_data/fl_pools/fl14_v1.jfl1");
        let mut roots = 20usize;
        let mut opponents = 200usize;
        let mut index = 2;
        while index < args.len() {
            match args[index].as_str() {
                "--pool" => { index += 1; pool_path = args[index].clone(); }
                "--roots" => { index += 1; roots = args[index].parse().expect("roots"); }
                "--opponents" => { index += 1; opponents = args[index].parse().expect("opponents"); }
                _ => {}
            }
            index += 1;
        }
        let table = [0.0f64, 10.7, 29.9, 63.5];
        let loading = std::time::Instant::now();
        let bytes = std::fs::read(&pool_path).expect("read pool");
        let loaded = pool::deserialize(&bytes, 14, table).expect("load pool");
        eprintln!(
            "pool: {} entries loaded in {:.1}s",
            loaded.entries.len(),
            loading.elapsed().as_secs_f64()
        );
        let started = std::time::Instant::now();
        let mut short = 0usize;
        let mut actions = 0usize;
        let mut spreads = Vec::new();
        for root in 0..roots as u64 {
            let cards = pool::deal(0xB0B0_0000 + root, 0, 15);
            let request = t4_labels::T4Request {
                id: root.to_string(),
                rows: [cards[0..2].to_vec(), cards[2..7].to_vec(), cards[7..11].to_vec()],
                dead: cards[11..12].to_vec(),
                draw: [cards[12], cards[13], cards[14]],
                opponents,
            };
            match t4_labels::solve(&request, &loaded, &table, root) {
                Ok(values) => {
                    actions += values.len();
                    let best = values.iter().map(|v| v.value).fold(f64::MIN, f64::max);
                    let worst = values.iter().map(|v| v.value).fold(f64::MAX, f64::min);
                    spreads.push(best - worst);
                }
                Err(e) => { short += 1; eprintln!("root {root}: short draw {}/{}", e.found, e.wanted); }
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        spreads.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "t4: {roots} roots, {actions} actions, {short} short draws, {:.1} ms/root ({:.2} ms/action)",
            elapsed / roots as f64 * 1000.0,
            elapsed / actions.max(1) as f64 * 1000.0,
        );
        if !spreads.is_empty() {
            eprintln!(
                "best-worst action spread: median {:.2}, max {:.2}",
                spreads[spreads.len() / 2], spreads[spreads.len() - 1]
            );
        }
        return;
    }

    if args.len() > 1 && args[1] == "pool" {
        let mut entries = 1000usize;
        let mut width = 14usize;
        let mut seed = 0x2026_0811u64;
        let mut out_path = String::from("fl_pool_14.jfl1");
        let table = [0.0f64, 10.7, 29.9, 63.5];
        let mut index = 2;
        while index < args.len() {
            match args[index].as_str() {
                "--entries" => { index += 1; entries = args[index].parse().expect("entries"); }
                "--width" => { index += 1; width = args[index].parse().expect("width"); }
                "--seed" => { index += 1; seed = args[index].parse().expect("seed"); }
                "--out" => { index += 1; out_path = args[index].clone(); }
                _ => {}
            }
            index += 1;
        }
        build_pool(entries, width, seed, table, &out_path);
        return;
    }
    
    if args.len() > 1 && args[1] == "frontier" {
        let mut hands = 20usize;
        let mut cards = 14usize;
        let mut fl_ev = 0.0f64;
        let mut boards = 8usize;
        let mut check = false;
        let mut index = 2;
        while index < args.len() {
            match args[index].as_str() {
                "--hands" => { index += 1; hands = args[index].parse().expect("hands"); }
                "--cards" => { index += 1; cards = args[index].parse().expect("cards"); }
                "--fl-ev" => { index += 1; fl_ev = args[index].parse().expect("fl-ev"); }
                "--boards" => { index += 1; boards = args[index].parse().expect("boards"); }
                "--check" => { check = true; }
                _ => {}
            }
            index += 1;
        }
        frontier_bench(hands, cards, fl_ev, boards, check);
        return;
    }

    if args.len() > 1 && args[1] == "generate" {
        // Data generation mode
        let mut samples = 1000;
        let mut num_cards = 14;
        let mut fixed_jokers: Option<usize> = None;
        let mut output = String::new();
        
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--samples" | "-n" => {
                    i += 1;
                    samples = args[i].parse().expect("Invalid samples");
                }
                "--cards" | "-c" => {
                    i += 1;
                    num_cards = args[i].parse().expect("Invalid cards");
                }
                "--jokers" | "-j" => {
                    i += 1;
                    fixed_jokers = Some(args[i].parse().expect("Invalid jokers"));
                }
                "--output" | "-o" => {
                    i += 1;
                    output = args[i].clone();
                }
                _ => {}
            }
            i += 1;
        }
        
        if output.is_empty() {
            if let Some(j) = fixed_jokers {
                output = format!("fl_rust_{}cards_joker{}.jsonl", num_cards, j);
            } else {
                output = format!("fl_rust_{}cards_random.jsonl", num_cards);
            }
        }
        
        eprintln!("Generating {} samples with {} cards", samples, num_cards);
        if let Some(j) = fixed_jokers {
            eprintln!("Fixed jokers: {}", j);
        }
        
        generate_data(samples, num_cards, fixed_jokers, &output);
    } else {
        // Stdin/stdout mode (default)
        run_stdin_mode();
    }
}

/// Measure the frontier, and prove it against brute force.
///
/// The parity arm is the point: a frontier that is merely fast is a different
/// opponent model, not a cheaper one. Hero boards are drawn from the cards the
/// Fantasyland hand did not take, so the comparison is against boards that
/// actually contest the rows rather than ones trivially beaten.
fn frontier_bench(hands: usize, cards: usize, fl_ev: f64, boards: usize, check: bool) {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(910_020_003);
    let mut rows_total = 0usize;
    let mut rows_max = 0usize;
    let mut build_seconds = 0.0f64;
    let mut tabulate_seconds = 0.0f64;
    let mut generate_seconds = 0.0f64;
    let mut sweep_seconds = 0.0f64;
    let mut checked = 0usize;
    let mut mismatches = 0usize;
    let mut bucket_checked = 0usize;
    let mut bucket_mismatches = 0usize;

    for hand_index in 0..hands {
        let mut deck: Vec<Card> = Vec::with_capacity(54);
        for suit in 0..4u8 {
            for rank in 2..=14u8 {
                deck.push(Card { rank, suit });
            }
        }
        deck.push(Card { rank: 0, suit: 4 });
        deck.push(Card { rank: 0, suit: 4 });
        deck.shuffle(&mut rng);
        let hand: Vec<Card> = deck[..cards].to_vec();
        let jokers = hand.iter().filter(|card| card.is_joker()).count();

        let started = std::time::Instant::now();
        let (frontier, tab, gen, swp) = frontier::build_frontier_timed(&hand, fl_ev);
        build_seconds += started.elapsed().as_secs_f64();
        tabulate_seconds += tab;
        generate_seconds += gen;
        sweep_seconds += swp;
        rows_total += frontier.len();
        rows_max = rows_max.max(frontier.len());

        if check {
            let bucketed = frontier::frontier_key(&frontier);
            let global = frontier::frontier_key(
                &frontier::build_frontier_unbucketed(&hand, fl_ev));
            bucket_checked += 1;
            if bucketed != global {
                bucket_mismatches += 1;
                eprintln!(
                    "BUCKET MISMATCH hand {hand_index}: {} rows bucketed, {} global",
                    bucketed.len(),
                    global.len());
            }
            for board in 0..boards {
                let offset = cards + board * 13;
                if offset + 13 > deck.len() {
                    break;
                }
                let hero = &deck[offset..offset + 13];
                let hero_top = ofc_core::evaluate_hand_value(&to_core_cards(&hero[..3]), 3);
                let hero_mid = ofc_core::evaluate_hand_value(&to_core_cards(&hero[3..8]), 5);
                let hero_bot = ofc_core::evaluate_hand_value(&to_core_cards(&hero[8..13]), 5);
                let fast = frontier::best_response(&frontier, hero_top, hero_mid, hero_bot);
                let slow = frontier::best_response_brute_force(
                    &hand, fl_ev, hero_top, hero_mid, hero_bot);
                checked += 1;
                if fast.to_bits() != slow.to_bits() {
                    mismatches += 1;
                    eprintln!(
                        "MISMATCH hand {hand_index} board {board}: frontier {fast} brute {slow}");
                }
            }
        }
        let _ = jokers;
    }

    println!("{}", serde_json::json!({
        "hands": hands,
        "cards": cards,
        "fl_ev": fl_ev,
        "mean_frontier_rows": rows_total as f64 / hands as f64,
        "max_frontier_rows": rows_max,
        "mean_build_ms": 1000.0 * build_seconds / hands as f64,
        "mean_tabulate_ms": 1000.0 * tabulate_seconds / hands as f64,
        "mean_generate_ms": 1000.0 * generate_seconds / hands as f64,
        "mean_sweep_ms": 1000.0 * sweep_seconds / hands as f64,
        "parity_checked": checked,
        "parity_mismatches": mismatches,
        "bucket_checked": bucket_checked,
        "bucket_mismatches": bucket_mismatches,
    }));
    if mismatches > 0 || bucket_mismatches > 0 {
        std::process::exit(1);
    }
}
