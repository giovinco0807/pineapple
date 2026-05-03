//! FL Solver - High-performance Fantasyland solver in Rust
//!
//! Standalone executable that communicates via JSON stdin/stdout

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
    let jokers = count_jokers(cards);
    let rank_counts = count_ranks(cards);
    let suit_counts = count_suits(cards);
    
    let mut counts: Vec<u8> = rank_counts.iter()
        .filter(|&&c| c > 0)
        .copied()
        .collect();
    counts.sort_by(|a, b| b.cmp(a));
    
    let is_flush = suit_counts.iter().any(|&c| c + jokers >= 5);
    let (is_straight, is_broadway) = is_straight_possible(&rank_counts, jokers);
    
    let rank = if is_flush && is_straight {
        if is_broadway { HandRank::RoyalFlush } else { HandRank::StraightFlush }
    } else if !counts.is_empty() && counts[0] + jokers >= 4 {
        HandRank::Quads
    } else if counts.len() >= 2 && counts[0] >= 2 && counts[0] + counts[1] + jokers >= 5 {
        HandRank::FullHouse
    } else if is_flush {
        HandRank::Flush
    } else if is_straight {
        HandRank::Straight
    } else if !counts.is_empty() && counts[0] + jokers >= 3 {
        HandRank::Trips
    } else if counts.len() >= 2 && counts[0] >= 2 && counts[1] + jokers >= 2 {
        HandRank::TwoPair
    } else if !counts.is_empty() && counts[0] + jokers >= 2 {
        HandRank::OnePair
    } else {
        HandRank::HighCard
    };
    
    let strength = calculate_strength(&rank_counts);
    (rank, strength)
}

fn evaluate_3_card(cards: &[Card]) -> (HandRank3, u32) {
    let jokers = count_jokers(cards);
    let rank_counts = count_ranks(cards);
    
    let mut counts: Vec<u8> = rank_counts.iter()
        .filter(|&&c| c > 0)
        .copied()
        .collect();
    counts.sort_by(|a, b| b.cmp(a));
    
    let rank = if !counts.is_empty() && counts[0] + jokers >= 3 {
        HandRank3::Trips
    } else if !counts.is_empty() && counts[0] + jokers >= 2 {
        HandRank3::OnePair
    } else {
        HandRank3::HighCard
    };
    
    let strength = calculate_strength(&rank_counts);
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
    let (rank, _) = evaluate_3_card(cards);
    let rank_counts = count_ranks(cards);
    let jokers = count_jokers(cards);
    
    match rank {
        HandRank3::Trips => {
            for r in (2..=14).rev() {
                if rank_counts[r] + jokers >= 3 && rank_counts[r] >= 1 {
                    return 10 + (r as i32 - 2);
                }
            }
            0
        }
        HandRank3::OnePair => {
            for r in (6..=14).rev() {
                if rank_counts[r] + jokers >= 2 && rank_counts[r] >= 1 {
                    return r as i32 - 5;
                }
            }
            0
        }
        HandRank3::HighCard => 0,
    }
}

fn get_middle_royalty(cards: &[Card]) -> i32 {
    let (rank, _) = evaluate_5_card(cards);
    match rank {
        HandRank::Trips => 2,
        HandRank::Straight => 4,
        HandRank::Flush => 8,
        HandRank::FullHouse => 12,
        HandRank::Quads => 20,
        HandRank::StraightFlush => 30,
        HandRank::RoyalFlush => 50,
        _ => 0,
    }
}

fn get_bottom_royalty(cards: &[Card]) -> i32 {
    let (rank, _) = evaluate_5_card(cards);
    match rank {
        HandRank::Straight => 2,
        HandRank::Flush => 4,
        HandRank::FullHouse => 6,
        HandRank::Quads => 10,
        HandRank::StraightFlush => 15,
        HandRank::RoyalFlush => 25,
        _ => 0,
    }
}

fn check_fl_stay(top: &[Card], _middle: &[Card], bottom: &[Card]) -> bool {
    let (top_rank, _) = evaluate_3_card(top);
    let (bottom_rank, _) = evaluate_5_card(bottom);
    
    top_rank == HandRank3::Trips ||
    bottom_rank == HandRank::Quads ||
    bottom_rank == HandRank::StraightFlush ||
    bottom_rank == HandRank::RoyalFlush
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

/// Compare two 5-card hands. Returns -1 if a < b, 0 if equal, 1 if a > b
fn compare_5_hands(a: &[Card], b: &[Card]) -> i32 {
    let (rank_a, str_a) = evaluate_5_card(a);
    let (rank_b, str_b) = evaluate_5_card(b);
    
    if (rank_a as u8) != (rank_b as u8) {
        return if (rank_a as u8) > (rank_b as u8) { 1 } else { -1 };
    }
    
    // Same hand rank - compare based on the specific rank
    match rank_a {
        HandRank::OnePair => {
            // Compare pair ranks first
            let pair_a = get_pair_rank(a);
            let pair_b = get_pair_rank(b);
            if pair_a != pair_b {
                return if pair_a > pair_b { 1 } else { -1 };
            }
            // Same pair rank - compare kickers
            if str_a > str_b { 1 } else if str_a < str_b { -1 } else { 0 }
        }
        HandRank::Trips | HandRank::FullHouse => {
            // Compare trips ranks first
            let trips_a = get_trips_rank(a);
            let trips_b = get_trips_rank(b);
            if trips_a != trips_b {
                return if trips_a > trips_b { 1 } else { -1 };
            }
            if str_a > str_b { 1 } else if str_a < str_b { -1 } else { 0 }
        }
        HandRank::TwoPair => {
            // Compare high pair, then low pair, then kicker
            let pairs_a = get_two_pair_ranks(a);
            let pairs_b = get_two_pair_ranks(b);
            if pairs_a.0 != pairs_b.0 {
                return if pairs_a.0 > pairs_b.0 { 1 } else { -1 };
            }
            if pairs_a.1 != pairs_b.1 {
                return if pairs_a.1 > pairs_b.1 { 1 } else { -1 };
            }
            if str_a > str_b { 1 } else if str_a < str_b { -1 } else { 0 }
        }
        HandRank::Quads => {
            // Compare quads rank
            let quads_a = get_quads_rank(a);
            let quads_b = get_quads_rank(b);
            if quads_a != quads_b {
                return if quads_a > quads_b { 1 } else { -1 };
            }
            if str_a > str_b { 1 } else if str_a < str_b { -1 } else { 0 }
        }
        HandRank::Straight | HandRank::StraightFlush | HandRank::RoyalFlush => {
            // For straights, compare by high card (wheel=5, not Ace=14)
            let rc_a = count_ranks(a);
            let rc_b = count_ranks(b);
            let j_a = count_jokers(a);
            let j_b = count_jokers(b);
            let high_a = get_straight_high_card(&rc_a, j_a);
            let high_b = get_straight_high_card(&rc_b, j_b);
            if high_a > high_b { 1 } else if high_a < high_b { -1 } else { 0 }
        }
        _ => {
            // For high card, flushes - compare kickers
            if str_a > str_b { 1 } else if str_a < str_b { -1 } else { 0 }
        }
    }
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
            if !is_valid_placement(&top, &middle, bottom) { continue; }
            
            let top_roy = get_top_royalty(&top);
            let mid_roy = get_middle_royalty(&middle);
            let bot_roy = get_bottom_royalty(bottom);
            let can_stay = check_fl_stay(&top, &middle, bottom);
            let total = top_roy + mid_roy + bot_roy;
            let stay_bonus = if can_stay { 30.0 } else { 0.0 };
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
            
            // Check valid placement
            if !is_valid_placement(&top, &middle, bottom) { continue; }
            
            let top_roy = get_top_royalty(&top);
            let mid_roy = get_middle_royalty(&middle);
            let bot_roy = get_bottom_royalty(bottom);
            let can_stay = check_fl_stay(&top, &middle, bottom);
            let total = top_roy + mid_roy + bot_roy;
            let stay_bonus = if can_stay { 30.0 } else { 0.0 };
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
                if !is_valid_placement(&trips, &middle, &bottom) { continue; }
                
                let top_roy = get_top_royalty(&trips);
                let mid_roy = get_middle_royalty(&middle);
                let bot_roy = get_bottom_royalty(&bottom);
                let can_stay = check_fl_stay(&trips, &middle, &bottom);
                let total = top_roy + mid_roy + bot_roy;
                let stay_bonus = if can_stay { 30.0 } else { 0.0 };
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
struct Request {
    cards: Vec<Card>,
    #[serde(default)]
    version: u8,  // 0 or 1 = v1 (exhaustive), 2 = v2 (role-based)
}

#[derive(Serialize)]
struct Response {
    success: bool,
    placement: Option<Placement>,
    error: Option<String>,
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
        let mut hand = loop {
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
                let placement = if req.version == 2 {
                    solve_fantasyland_v2(&req.cards)
                } else {
                    solve_fantasyland(&req.cards)
                };
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!("Solved v{} in {:.3}s", if req.version == 2 { 2 } else { 1 }, elapsed);
                
                Response {
                    success: true,
                    placement,
                    error: None,
                }
            }
            Err(e) => Response {
                success: false,
                placement: None,
                error: Some(format!("Parse error: {}", e)),
            },
        };
        
        writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap()).unwrap();
        stdout.flush().unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
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

