//! FL Solver - High-performance Fantasyland solver in Rust
//!
//! Standalone executable that communicates via JSON stdin/stdout

// The best-response machinery, shared as a library so the playout labelers in
// `t4_first_exact` can price a leaf against a real Fantasyland best response
// instead of a statically solved board.  The binary declares the same modules
// against its own copy of `Card`; they are small and the duplication is
// cheaper than untangling main.rs from lib.rs while a six-hour teacher runs.
/// Every k-subset of n as a bitmask, lexicographic.  Lives here as well as
/// in the binary because the frontier needs it and the two roots are
/// separate crates.
pub fn subset_masks(n: usize, k: usize) -> Vec<u32> {
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

pub mod frontier;
pub mod pool;
pub mod t2_labels;
pub mod t3_labels;
pub mod t4_labels;
pub mod vs_fl;

use rayon::prelude::*;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::io;

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

pub fn evaluate_5_card(cards: &[Card]) -> (HandRank, u32) {
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

pub fn evaluate_3_card(cards: &[Card]) -> (HandRank3, u32) {
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

pub fn get_top_royalty(cards: &[Card]) -> i32 {
    ofc_core::get_top_royalty(&to_core_cards(cards))
}

pub fn get_middle_royalty(cards: &[Card]) -> i32 {
    ofc_core::get_middle_royalty(&to_core_cards(cards))
}

pub fn get_bottom_royalty(cards: &[Card]) -> i32 {
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
fn get_5card_strength(cards: &[Card]) -> (u8, u32) {
    let (rank, strength) = evaluate_5_card(cards);
    (rank as u8, strength)
}

/// Get comparable strength for 3-card hand (mapped to 5-card scale)
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
pub fn compare_5_hands(a: &[Card], b: &[Card]) -> i32 {
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

/// Fast role-based solver that avoids exhaustive search fallback
pub fn solve_fantasyland_v2_fast(cards: &[Card]) -> Option<Placement> {
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
    
    max_placement(best_ab, c)
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
}

// ============================================================
//  FL vs Normal opponent scoring
// ============================================================

pub fn compare_3_hands(a: &[Card], b: &[Card]) -> i32 {
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

pub fn create_deck(include_jokers: bool) -> Vec<Card> {
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

pub fn deal_hand(num_cards: usize, include_jokers: bool, rng: &mut impl Rng) -> Vec<Card> {
    let mut deck = create_deck(include_jokers);
    deck.shuffle(rng);
    deck.into_iter().take(num_cards).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(rank: u8, suit: u8) -> Card {
        Card { rank, suit }
    }

    #[test]
    fn bridge_uses_canonical_top_joker_downgrade() {
        let top = [card(12, 0), card(12, 1), card(0, 4)];
        let middle = [card(13, 0), card(13, 1), card(9, 2), card(8, 3), card(7, 0)];
        let bottom = [card(14, 0), card(14, 1), card(14, 2), card(5, 3), card(4, 2)];

        let rows = canonical_rows(&top, &middle, &bottom);
        assert!(!rows.busted);
        assert_eq!(rows.top_royalty, 7);
    }
}


