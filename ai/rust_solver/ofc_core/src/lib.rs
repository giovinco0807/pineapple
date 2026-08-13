//! OFC Core - Shared card, hand evaluation, royalty, and bust logic
//!
//! Extracted from fl_solver main.rs for reuse across FL solver and backward induction.

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;

// ============================================================
//  Card & Hand Rank Types
// ============================================================

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HandRank3 {
    HighCard = 0,
    OnePair = 1,
    Trips = 2,
}

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
//  Card Counting
// ============================================================

pub fn count_ranks(cards: &[Card]) -> [u8; 15] {
    let mut counts = [0u8; 15];
    for c in cards {
        if !c.is_joker() {
            counts[c.rank as usize] += 1;
        }
    }
    counts
}

pub fn count_suits(cards: &[Card]) -> [u8; 4] {
    let mut counts = [0u8; 4];
    for c in cards {
        if !c.is_joker() && c.suit < 4 {
            counts[c.suit as usize] += 1;
        }
    }
    counts
}

pub fn count_jokers(cards: &[Card]) -> u8 {
    cards.iter().filter(|c| c.is_joker()).count() as u8
}

// ============================================================
//  Hand Evaluation
// ============================================================

pub fn is_straight_possible(rank_counts: &[u8; 15], jokers: u8) -> (bool, bool) {
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
    let mut is_wheel = false;

    for (i, straight) in straights.iter().enumerate() {
        let missing: u8 = straight.iter()
            .filter(|&&r| rank_counts[r as usize] == 0)
            .count() as u8;

        if missing <= jokers {
            is_straight = true;
            if i == 0 { is_wheel = true; }
        }
    }
    (is_straight, is_wheel)
}

pub fn get_straight_high_card(rank_counts: &[u8; 15], jokers: u8) -> u8 {
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

    let high_cards: [u8; 10] = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
    let mut best_high = 0u8;

    for (i, straight) in straights.iter().enumerate() {
        let missing: u8 = straight.iter()
            .filter(|&&r| rank_counts[r as usize] == 0)
            .count() as u8;
        if missing <= jokers {
            best_high = high_cards[i];
        }
    }
    best_high
}

const HAND_VALUE_BASE: u32 = 15;
const HAND_VALUE_CATEGORY_BASE: u32 = 15u32.pow(5);

/// Canonical base-15 hand value shared with `ai.engine.game_engine.evaluate_hand`.
///
/// Keeping this value identical to Python is important: a three-card top row and
/// a five-card middle row can then be compared directly, including trips vs
/// trips and all kickers.  The old Rust evaluator returned only the set of
/// distinct ranks, which lost pair/trips structure and made those comparisons
/// disagree with Python.
fn encode_hand_value(category: u8, ranks: &[u8]) -> u32 {
    let mut value = category as u32;
    for index in 0..5 {
        value = value * HAND_VALUE_BASE + ranks.get(index).copied().unwrap_or(0) as u32;
    }
    value
}

fn category_from_value(value: u32) -> u8 {
    (value / HAND_VALUE_CATEGORY_BASE) as u8
}

fn primary_rank_from_value(value: u32) -> u8 {
    ((value / HAND_VALUE_BASE.pow(4)) % HAND_VALUE_BASE) as u8
}

/// Evaluate a row using the canonical Python base-15 representation.
///
/// Jokers are resolved by exhaustively trying distinct natural-card
/// substitutions and taking the maximum natural hand. This is also what makes
/// flush kickers and a second Joker used as a quads kicker exact.
thread_local! {
    /// Joker hands are evaluated by exhaustive natural substitution, which is
    /// two to three orders of magnitude more work than a natural evaluation
    /// and recurs constantly: a 14-card FL solve revisits the same few
    /// thousand distinct rows across a million arrangements.  The cache key
    /// is the sorted multiset of cards (both jokers are interchangeable for
    /// strength) plus the row size, so a hit is exact, and the map is cleared
    /// if it ever grows past a bound so long-running teacher generation
    /// cannot leak.
    static JOKER_EVAL_CACHE: RefCell<HashMap<u64, u32>> = RefCell::new(HashMap::new());
}

/// A leak guard that cannot bind: the rows this cache can hold are the jokered
/// ones, and there are C(52,4) + C(52,3) five-card and C(52,2) + 52 three-card
/// ones -- about 294,000 in total, a few megabytes a thread.
const JOKER_EVAL_CACHE_LIMIT: usize = 4_000_000;

fn joker_eval_cache_key(cards: &[Card], expected_count: usize) -> u64 {
    // 6 bits per card: naturals are rank*4+suit in 8..=59, any joker is 60.
    let mut codes: [u8; 8] = [63; 8];
    for (index, card) in cards.iter().enumerate() {
        codes[index] = if card.is_joker() {
            60
        } else {
            card.rank * 4 + card.suit
        };
    }
    codes[..cards.len()].sort_unstable();
    let mut key: u64 = expected_count as u64;
    for code in &codes[..cards.len()] {
        key = (key << 6) | *code as u64;
    }
    key
}

pub fn evaluate_hand_value(cards: &[Card], expected_count: usize) -> u32 {
    if cards.len() != expected_count {
        return 0;
    }

    let joker_count = count_jokers(cards) as usize;
    if joker_count == 0 {
        return evaluate_natural_hand_value(cards, expected_count);
    }

    let key = joker_eval_cache_key(cards, expected_count);
    if let Some(value) = JOKER_EVAL_CACHE.with(|cache| cache.borrow().get(&key).copied()) {
        return value;
    }

    let non_jokers: Vec<Card> = cards
        .iter()
        .filter(|card| !card.is_joker())
        .copied()
        .collect();
    let substitutions = available_subs(cards);
    let mut best_value = 0;

    if joker_count == 1 {
        for substitution in substitutions {
            let mut natural = non_jokers.clone();
            natural.push(substitution);
            best_value = best_value.max(evaluate_natural_hand_value(&natural, expected_count));
        }
    } else if joker_count == 2 {
        for first in 0..substitutions.len() {
            for second in (first + 1)..substitutions.len() {
                let mut natural = non_jokers.clone();
                natural.push(substitutions[first]);
                natural.push(substitutions[second]);
                best_value = best_value.max(evaluate_natural_hand_value(&natural, expected_count));
            }
        }
    }

    // The lookup above has been here since the cache was written; the insert
    // had not, so the map stayed empty and every joker row was re-substituted
    // from scratch forever.  `JOKER_EVAL_CACHE_LIMIT` being dead code was the
    // only sign of it.  It costs more than it sounds: with the constrained-row
    // cache already in, a T3 root whose middle holds two jokers ran 26.3 s on
    // one thread against 3.6 s with this insert restored, because a middle that
    // cannot be legally substituted comes back still holding its jokers and is
    // then evaluated again by `is_valid_placement`, by `hero_terminal` and by
    // the royalty, at 990 natural evaluations each time.
    JOKER_EVAL_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() >= JOKER_EVAL_CACHE_LIMIT {
            cache.clear();
        }
        cache.insert(key, best_value);
    });
    best_value
}

fn evaluate_natural_hand_value(cards: &[Card], expected_count: usize) -> u32 {
    if cards.len() != expected_count {
        return 0;
    }

    let rank_counts = count_ranks(cards);
    let suit_counts = count_suits(cards);
    let jokers = count_jokers(cards);
    let mut ranks: Vec<u8> = cards
        .iter()
        .filter(|card| !card.is_joker())
        .map(|card| card.rank)
        .collect();
    ranks.sort_unstable_by(|a, b| b.cmp(a));

    let best = rank_counts.iter().copied().max().unwrap_or(0);

    if expected_count == 3 {
        if best + jokers >= 3 {
            let trip_rank = if best >= 3 {
                (2..=14).rev().find(|&rank| rank_counts[rank] >= 3).unwrap_or(0) as u8
            } else if best >= 2 {
                (2..=14).rev().find(|&rank| rank_counts[rank] >= 2).unwrap_or(0) as u8
            } else {
                ranks.first().copied().unwrap_or(14)
            };
            return encode_hand_value(3, &[trip_rank]);
        }

        if best + jokers >= 2 {
            let (pair_rank, kicker) = if best >= 2 {
                let pair_rank = (2..=14)
                    .rev()
                    .find(|&rank| rank_counts[rank] >= 2)
                    .unwrap_or(0) as u8;
                let kicker = ranks
                    .iter()
                    .copied()
                    .find(|&rank| rank != pair_rank)
                    .unwrap_or(0);
                (pair_rank, kicker)
            } else {
                (
                    ranks.first().copied().unwrap_or(0),
                    ranks.get(1).copied().unwrap_or(0),
                )
            };
            return encode_hand_value(1, &[pair_rank, kicker]);
        }

        return encode_hand_value(0, &ranks);
    }

    let pairs: Vec<u8> = (2..=14)
        .rev()
        .filter(|&rank| rank_counts[rank] >= 2)
        .map(|rank| rank as u8)
        .collect();
    let non_joker_count = cards.len() as u8 - jokers;
    let flush_suits = suit_counts.iter().filter(|&&count| count > 0).count();
    let is_flush = flush_suits == 1 && non_joker_count + jokers == expected_count as u8;
    let (is_straight, _) = is_straight_possible(&rank_counts, jokers);

    if is_flush && is_straight {
        return encode_hand_value(8, &[get_straight_high_card(&rank_counts, jokers)]);
    }

    if best + jokers >= 4 {
        let quad_rank = if best >= 4 {
            (2..=14).rev().find(|&rank| rank_counts[rank] >= 4).unwrap_or(0) as u8
        } else if best >= 3 {
            (2..=14).rev().find(|&rank| rank_counts[rank] >= 3).unwrap_or(0) as u8
        } else {
            pairs.first().copied().or_else(|| ranks.first().copied()).unwrap_or(0)
        };
        let kicker = ranks
            .iter()
            .copied()
            .filter(|&rank| rank != quad_rank)
            .max()
            .unwrap_or(0);
        return encode_hand_value(7, &[quad_rank, kicker]);
    }

    if best >= 3 {
        let trip_rank = (2..=14)
            .rev()
            .find(|&rank| rank_counts[rank] >= 3)
            .unwrap_or(0) as u8;
        if let Some(pair_rank) = (2..=14)
            .rev()
            .find(|&rank| rank as u8 != trip_rank && rank_counts[rank] >= 2)
        {
            return encode_hand_value(6, &[trip_rank, pair_rank as u8]);
        }
    }
    if jokers >= 1 && pairs.len() >= 2 {
        return encode_hand_value(6, &[pairs[0], pairs[1]]);
    }

    if is_flush {
        return encode_hand_value(5, &ranks);
    }
    if is_straight {
        return encode_hand_value(4, &[get_straight_high_card(&rank_counts, jokers)]);
    }

    if best + jokers >= 3 {
        let trip_rank = if best >= 3 {
            (2..=14).rev().find(|&rank| rank_counts[rank] >= 3).unwrap_or(0) as u8
        } else if best >= 2 {
            (2..=14).rev().find(|&rank| rank_counts[rank] >= 2).unwrap_or(0) as u8
        } else {
            ranks.first().copied().unwrap_or(0)
        };
        let kickers: Vec<u8> = ranks
            .iter()
            .copied()
            .filter(|&rank| rank != trip_rank)
            .collect();
        return encode_hand_value(
            3,
            &[
                trip_rank,
                kickers.first().copied().unwrap_or(0),
                kickers.get(1).copied().unwrap_or(0),
            ],
        );
    }

    if pairs.len() >= 2 {
        let kicker = ranks
            .iter()
            .copied()
            .filter(|rank| !pairs[..2].contains(rank))
            .max()
            .unwrap_or(0);
        return encode_hand_value(2, &[pairs[0], pairs[1], kicker]);
    }

    if best >= 2 || jokers >= 1 {
        let pair_rank = pairs.first().copied().or_else(|| ranks.first().copied()).unwrap_or(0);
        let mut kickers: Vec<u8> = if pairs.is_empty() {
            ranks.iter().copied().skip(1).collect()
        } else {
            ranks.iter().copied().filter(|&rank| rank != pair_rank).collect()
        };
        kickers.resize(3, 0);
        return encode_hand_value(1, &[pair_rank, kickers[0], kickers[1], kickers[2]]);
    }

    encode_hand_value(0, &ranks)
}

pub fn evaluate_5_card(cards: &[Card]) -> (HandRank, u32) {
    let value = evaluate_hand_value(cards, 5);
    let category = category_from_value(value);
    let rank = match category {
        8 if primary_rank_from_value(value) == 14 => HandRank::RoyalFlush,
        8 => HandRank::StraightFlush,
        7 => HandRank::Quads,
        6 => HandRank::FullHouse,
        5 => HandRank::Flush,
        4 => HandRank::Straight,
        3 => HandRank::Trips,
        2 => HandRank::TwoPair,
        1 => HandRank::OnePair,
        _ => HandRank::HighCard,
    };
    (rank, value % HAND_VALUE_CATEGORY_BASE)
}

pub fn evaluate_3_card(cards: &[Card]) -> (HandRank3, u32) {
    let value = evaluate_hand_value(cards, 3);
    let rank = match category_from_value(value) {
        3 => HandRank3::Trips,
        1 => HandRank3::OnePair,
        _ => HandRank3::HighCard,
    };
    (rank, value % HAND_VALUE_CATEGORY_BASE)
}

pub fn calculate_strength(rank_counts: &[u8; 15]) -> u32 {
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

pub fn get_middle_royalty(cards: &[Card]) -> i32 {
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

pub fn get_bottom_royalty(cards: &[Card]) -> i32 {
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

pub fn check_fl_stay(top: &[Card], _middle: &[Card], bottom: &[Card]) -> bool {
    let (top_rank, _) = evaluate_3_card(top);
    let (bottom_rank, _) = evaluate_5_card(bottom);

    top_rank == HandRank3::Trips ||
    bottom_rank == HandRank::Quads ||
    bottom_rank == HandRank::StraightFlush ||
    bottom_rank == HandRank::RoyalFlush
}

/// Check FL entry from top row. Returns (qualifies, card_count).
pub fn check_fl_entry(top: &[Card]) -> (bool, u8) {
    let rank_counts = count_ranks(top);
    let jokers = count_jokers(top);

    for r in (2..=14).rev() {
        let count = rank_counts[r];
        if count + jokers >= 3 && count >= 1 {
            return (true, 17); // Trips
        }
        if count + jokers >= 2 && count >= 1 {
            return match r {
                14 => (true, 16), // AA
                13 => (true, 15), // KK
                12 => (true, 14), // QQ
                _ => (false, 0),
            };
        }
    }
    (false, 0)
}

// ============================================================
//  Bust Check & Hand Comparison
// ============================================================

pub fn compare_5_hands(a: &[Card], b: &[Card]) -> i32 {
    let value_a = evaluate_hand_value(a, 5);
    let value_b = evaluate_hand_value(b, 5);
    match value_a.cmp(&value_b) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

pub fn get_pair_rank(cards: &[Card]) -> u8 {
    let rank_counts = count_ranks(cards);
    let jokers = count_jokers(cards);
    for r in (2..=14).rev() {
        if rank_counts[r] >= 2 || (rank_counts[r] >= 1 && jokers >= 1) {
            return r as u8;
        }
    }
    0
}

pub fn get_trips_rank(cards: &[Card]) -> u8 {
    let rank_counts = count_ranks(cards);
    let jokers = count_jokers(cards);
    for r in (2..=14).rev() {
        if rank_counts[r] + jokers >= 3 && rank_counts[r] >= 1 {
            return r as u8;
        }
    }
    0
}

pub fn get_two_pair_ranks(cards: &[Card]) -> (u8, u8) {
    let rank_counts = count_ranks(cards);
    let mut pairs = Vec::new();
    for r in (2..=14).rev() {
        if rank_counts[r] >= 2 {
            pairs.push(r as u8);
            if pairs.len() == 2 { break; }
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

pub fn get_quads_rank(cards: &[Card]) -> u8 {
    let rank_counts = count_ranks(cards);
    let jokers = count_jokers(cards);
    for r in (2..=14).rev() {
        if rank_counts[r] + jokers >= 4 && rank_counts[r] >= 1 {
            return r as u8;
        }
    }
    0
}

pub fn is_valid_placement(top: &[Card], middle: &[Card], bottom: &[Card]) -> bool {
    let top_value = evaluate_hand_value(top, 3);
    let middle_value = evaluate_hand_value(middle, 5);
    let bottom_value = evaluate_hand_value(bottom, 5);
    top_value <= middle_value && middle_value <= bottom_value
}

// ============================================================
//  Deck Utilities
// ============================================================

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

pub const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
pub const SUIT_CHARS: [char; 4] = ['s', 'h', 'd', 'c'];

pub fn rank_to_char(rank: u8) -> char {
    if rank >= 2 && rank <= 14 {
        RANK_CHARS[(rank - 2) as usize]
    } else if rank == 0 {
        'X'
    } else {
        '?'
    }
}

pub fn suit_to_char(suit: u8) -> char {
    if suit < 4 {
        SUIT_CHARS[suit as usize]
    } else {
        'J'
    }
}

pub fn card_to_string(card: &Card) -> String {
    if card.is_joker() {
        "JK".to_string()
    } else {
        format!("{}{}", rank_to_char(card.rank), suit_to_char(card.suit))
    }
}

// ============================================================
//  Joker Bust-Prevention (Constrained Evaluation)
// ============================================================

/// Result of constrained board evaluation.
/// Cards may have jokers substituted to avoid busting.
pub struct BoardEval {
    pub busted: bool,
    pub top: Vec<Card>,
    pub mid: Vec<Card>,
    pub bot: Vec<Card>,
}

/// Evaluate board with joker bust-prevention.
/// Bottom-up: bot (max strength), mid (max ≤ bot), top (max ≤ mid).
/// Jokers pick the strongest hand that doesn't violate Top ≤ Mid ≤ Bot.
pub fn evaluate_board_with_joker_constraint(
    top: &[Card], mid: &[Card], bot: &[Card],
) -> BoardEval {
    let top_has_joker = top.iter().any(|c| c.is_joker());
    let mid_has_joker = mid.iter().any(|c| c.is_joker());

    // Fast path: no jokers in top or mid → standard check
    // (bot jokers are unconstrained, existing evaluate handles them)
    if !top_has_joker && !mid_has_joker {
        return BoardEval {
            busted: !is_valid_placement(top, mid, bot),
            top: top.to_vec(),
            mid: mid.to_vec(),
            bot: bot.to_vec(),
        };
    }

    // Mid: constrain to ≤ bot
    let mid_final = if mid_has_joker {
        constrain_5_vs_5(mid, bot)
    } else {
        mid.to_vec()
    };

    // Top: constrain to ≤ mid_final
    let top_final = if top_has_joker {
        constrain_3_vs_5(top, &mid_final)
    } else {
        top.to_vec()
    };

    let busted = !is_valid_placement(&top_final, &mid_final, bot);

    BoardEval {
        busted,
        top: top_final,
        mid: mid_final,
        bot: bot.to_vec(),
    }
}

/// Compare two 3-card hands (for finding best substitution)
pub fn compare_3_hands(a: &[Card], b: &[Card]) -> i32 {
    let value_a = evaluate_hand_value(a, 3);
    let value_b = evaluate_hand_value(b, 3);
    match value_a.cmp(&value_b) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// Generate all candidate substitution cards (not already in the hand)
fn available_subs(cards: &[Card]) -> Vec<Card> {
    let mut used = std::collections::HashSet::new();
    for c in cards {
        if !c.is_joker() {
            used.insert((c.rank, c.suit));
        }
    }
    let mut subs = Vec::new();
    for rank in 2..=14u8 {
        for suit in 0..4u8 {
            if !used.contains(&(rank, suit)) {
                subs.push(Card { rank, suit });
            }
        }
    }
    subs
}

thread_local! {
    /// Constrained substitutions, keyed by the constrained row and the
    /// reference's VALUE rather than the reference's cards.
    ///
    /// That key is legitimate because neither `constrain_5_vs_5` nor
    /// `constrain_3_vs_5` ever looked at the reference row itself: it appeared
    /// only inside `compare_5_hands(.., ref_cards)` and a
    /// `top_value <= mid_value` test, and each reduced it to
    /// `evaluate_hand_value(ref, 5)` and nothing else.  `available_subs`
    /// depends on the constrained row alone.  So both are pure functions of
    /// `(cards, reference value)`.  The property is pinned by
    /// `the_reference_row_is_read_only_through_its_value` rather than left to
    /// this comment.
    ///
    /// Worth caching because the two-joker branch walks C(~45, 2) = 990
    /// candidate pairs, and `fl_solver`'s T3 labeler asks the same question for
    /// every completion of a base board that does not move.  One thread, 60
    /// opponents, a T3 root whose middle holds two jokers: 97 s, against 0.6 s
    /// for a joker-free one.  This cache took it to 25 s and the joker
    /// evaluation cache above -- which was not storing anything -- took it the
    /// rest of the way to 2 s.  When the placement leaves the middle alone the
    /// whole sweep collapses to one solve per distinct bottom value.
    ///
    /// Keyed on card ORDER, not on the multiset: the answer is a `Vec` whose
    /// order follows the input's, so two orderings of one hand are two
    /// different answers even though they are the same hand.
    static CONSTRAIN_CACHE: RefCell<HashMap<u64, Vec<Card>>> = RefCell::new(HashMap::new());
}

/// Bounded so a long teacher run cannot leak.  Held far below
/// `JOKER_EVAL_CACHE_LIMIT` because an entry here is a heap `Vec` rather than a
/// `u32`, and this runs on every core at once; a T3 action's whole working set
/// is a few thousand keys, so the bound is never the thing that binds.
const CONSTRAIN_CACHE_LIMIT: usize = 100_000;

/// 6 bits a card IN THE ORDER GIVEN, under the row width, over 24 bits of
/// reference value.  The width leads so a three-card row and a five-card one
/// cannot collide.
fn constrain_cache_key(cards: &[Card], reference: u32) -> u64 {
    debug_assert!(reference < 1 << 24, "a hand value no longer fits the key");
    let mut key: u64 = cards.len() as u64;
    for card in cards {
        let code = if card.is_joker() {
            60
        } else {
            card.rank as u64 * 4 + card.suit as u64
        };
        key = (key << 6) | code;
    }
    (key << 24) | reference as u64
}

fn constrain_cached(
    cards: &[Card],
    reference: u32,
    solve: impl FnOnce() -> Vec<Card>,
) -> Vec<Card> {
    // Six bits a card runs out past five, and a key that silently drops its
    // high bits answers one row with another's hand.  No caller builds a longer
    // row -- the searches below assume five and three -- so this is a guard
    // rather than a path.
    if cards.len() > 5 {
        return solve();
    }
    let key = constrain_cache_key(cards, reference);
    if let Some(hit) = CONSTRAIN_CACHE.with(|cache| cache.borrow().get(&key).cloned()) {
        return hit;
    }
    let solved = solve();
    CONSTRAIN_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() >= CONSTRAIN_CACHE_LIMIT {
            cache.clear();
        }
        cache.insert(key, solved.clone());
    });
    solved
}

/// Find best 5-card joker substitution constrained to ≤ ref_cards (5-card)
fn constrain_5_vs_5(cards: &[Card], ref_cards: &[Card]) -> Vec<Card> {
    let reference = evaluate_hand_value(ref_cards, 5);
    constrain_cached(cards, reference, || solve_5_vs_5(cards, reference))
}

/// [`constrain_5_vs_5`] with the reference already reduced to its value, and
/// without the cache in front of it.  Split out so the memo has something to
/// wrap and the tests have something unmemoized to check it against.
fn solve_5_vs_5(cards: &[Card], reference: u32) -> Vec<Card> {
    // If max eval already ≤ ref, keep original.  Also the only guard the
    // candidate buffer below needs: a row that is not five cards long
    // evaluates to 0, which is ≤ every reference, so it returns here.
    if evaluate_hand_value(cards, 5) <= reference {
        return cards.to_vec();
    }

    // The candidate is assembled in place.  It used to be a fresh `Vec` per
    // pair, which on the two-joker branch is 990 allocations for one call and
    // tens of millions across a root.
    let mut candidate = [Card { rank: 0, suit: 0 }; 5];
    let mut naturals = 0usize;
    for card in cards {
        if !card.is_joker() {
            candidate[naturals] = *card;
            naturals += 1;
        }
    }
    let n_jokers = cards.len() - naturals;
    let subs = available_subs(cards);

    // Carried alongside the hand because the loop used to re-derive it, and the
    // incumbent's value cannot change: `compare_5_hands(&test, best)` evaluated
    // BOTH sides afresh on every one of the 990 pairs.
    let mut best: Option<(u32, Vec<Card>)> = None;

    if n_jokers == 1 {
        for sub in &subs {
            candidate[naturals] = *sub;
            let value = evaluate_hand_value(&candidate[..naturals + 1], 5);
            if value <= reference && best.as_ref().map_or(true, |(seen, _)| value > *seen) {
                best = Some((value, candidate[..naturals + 1].to_vec()));
            }
        }
    } else if n_jokers == 2 {
        for i in 0..subs.len() {
            candidate[naturals] = subs[i];
            for j in (i + 1)..subs.len() {
                candidate[naturals + 1] = subs[j];
                let value = evaluate_hand_value(&candidate[..naturals + 2], 5);
                if value <= reference && best.as_ref().map_or(true, |(seen, _)| value > *seen) {
                    best = Some((value, candidate[..naturals + 2].to_vec()));
                }
            }
        }
    }

    // If no valid sub found → genuinely busted, return original
    best.map(|(_, hand)| hand).unwrap_or_else(|| cards.to_vec())
}

/// Find best 3-card joker substitution constrained to ≤ mid (5-card)
fn constrain_3_vs_5(cards: &[Card], mid: &[Card]) -> Vec<Card> {
    let reference = evaluate_hand_value(mid, 5);
    constrain_cached(cards, reference, || solve_3_vs_5(cards, reference))
}

/// [`constrain_3_vs_5`] against the middle's value alone, unmemoized.
fn solve_3_vs_5(cards: &[Card], reference: u32) -> Vec<Card> {
    // If max eval already ≤ mid, keep original
    if evaluate_hand_value(cards, 3) <= reference {
        return cards.to_vec();
    }

    let mut candidate = [Card { rank: 0, suit: 0 }; 3];
    let mut naturals = 0usize;
    for card in cards {
        if !card.is_joker() {
            candidate[naturals] = *card;
            naturals += 1;
        }
    }
    let n_jokers = cards.len() - naturals;
    let subs = available_subs(cards);

    let mut best: Option<(u32, Vec<Card>)> = None;

    if n_jokers == 1 {
        for sub in &subs {
            candidate[naturals] = *sub;
            let value = evaluate_hand_value(&candidate[..naturals + 1], 3);
            if value <= reference && best.as_ref().map_or(true, |(seen, _)| value > *seen) {
                best = Some((value, candidate[..naturals + 1].to_vec()));
            }
        }
    } else if n_jokers == 2 {
        for i in 0..subs.len() {
            candidate[naturals] = subs[i];
            for j in (i + 1)..subs.len() {
                candidate[naturals + 1] = subs[j];
                let value = evaluate_hand_value(&candidate[..naturals + 2], 3);
                if value <= reference && best.as_ref().map_or(true, |(seen, _)| value > *seen) {
                    best = Some((value, candidate[..naturals + 2].to_vec()));
                }
            }
        }
    }

    best.map(|(_, hand)| hand).unwrap_or_else(|| cards.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(rank: u8, suit: u8) -> Card {
        Card { rank, suit }
    }

    fn joker() -> Card {
        card(0, 4)
    }

    #[test]
    fn two_jokers_choose_quads_over_full_house() {
        let cards = [card(14, 0), card(2, 0), card(2, 1), joker(), joker()];
        let (rank, _) = evaluate_5_card(&cards);
        assert_eq!(rank, HandRank::Quads);
        assert_eq!(evaluate_hand_value(&cards, 5), 5_464_125);
        assert_eq!(get_bottom_royalty(&cards), 10);
    }

    #[test]
    fn joker_straight_flush_value_matches_python_base15_encoding() {
        let cards = [card(14, 0), card(13, 0), card(12, 0), joker(), joker()];
        assert_eq!(evaluate_hand_value(&cards, 5), 6_783_750);
        assert_eq!(evaluate_5_card(&cards).0, HandRank::RoyalFlush);
    }

    #[test]
    fn joker_completes_highest_flush_kicker_exactly() {
        let cards = [card(13, 1), card(12, 1), card(9, 1), card(3, 1), joker()];
        assert_eq!(evaluate_hand_value(&cards, 5), 4_552_338);
        assert_eq!(evaluate_5_card(&cards).0, HandRank::Flush);
    }

    #[test]
    fn second_joker_becomes_best_quads_kicker() {
        let cards = [card(12, 0), card(12, 1), card(12, 2), joker(), joker()];
        assert_eq!(evaluate_hand_value(&cards, 5), 5_970_375);
        assert_eq!(evaluate_5_card(&cards).0, HandRank::Quads);
    }

    #[test]
    fn top_trips_rank_is_compared_against_middle_trips() {
        let top = [card(14, 0), card(14, 1), card(14, 2)];
        let middle = [card(2, 0), card(2, 1), card(2, 2), card(13, 0), card(12, 0)];
        let bottom = [card(3, 0), card(3, 1), card(3, 2), card(3, 3), card(4, 0)];
        assert!(!is_valid_placement(&top, &middle, &bottom));
    }

    #[test]
    fn top_joker_downgrades_to_best_non_busting_pair() {
        let top = [card(12, 0), card(12, 1), joker()];
        let middle = [card(13, 0), card(13, 1), card(14, 2), card(11, 2), card(9, 2)];
        let bottom = [card(14, 0), card(14, 1), card(14, 3), card(8, 0), card(7, 0)];

        let eval = evaluate_board_with_joker_constraint(&top, &middle, &bottom);
        assert!(!eval.busted);
        assert_eq!(category_from_value(evaluate_hand_value(&eval.top, 3)), 1);
        assert_eq!(primary_rank_from_value(evaluate_hand_value(&eval.top, 3)), 12);
        assert_eq!(get_top_royalty(&eval.top), 7);
        assert_eq!(check_fl_entry(&eval.top), (true, 14));
    }

    /// Deterministic, and the same shape the exactness suite uses.
    fn lcg(state: &mut u64) -> usize {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as usize
    }

    fn shuffled_deck(state: &mut u64, with_jokers: bool) -> Vec<Card> {
        let mut deck = create_deck(with_jokers);
        for index in (1..deck.len()).rev() {
            let pick = lcg(state) % (index + 1);
            deck.swap(index, pick);
        }
        deck
    }

    /// A row of `size` cards whose first `jokers` slots are jokers.
    fn jokered_row(state: &mut u64, size: usize, jokers: usize) -> Vec<Card> {
        let mut row: Vec<Card> = shuffled_deck(state, false)[..size].to_vec();
        for slot in 0..jokers {
            row[slot] = joker();
        }
        row
    }

    /// `constrain_5_vs_5` as it stood before the reference was reduced to a
    /// value -- taking the reference's CARDS and comparing against them.
    ///
    /// Kept verbatim because the cached version cannot be checked against
    /// itself: a key that reads the reference wrongly would agree with every
    /// call that shares the key.  This is the thing it has to agree with.
    fn mirror_5_vs_5(cards: &[Card], ref_cards: &[Card]) -> Vec<Card> {
        if compare_5_hands(cards, ref_cards) <= 0 {
            return cards.to_vec();
        }
        let non_jokers: Vec<Card> = cards.iter().filter(|c| !c.is_joker()).cloned().collect();
        let n_jokers = cards.len() - non_jokers.len();
        let subs = available_subs(cards);
        let mut best: Option<Vec<Card>> = None;
        if n_jokers == 1 {
            for sub in &subs {
                let mut test = non_jokers.clone();
                test.push(*sub);
                if compare_5_hands(&test, ref_cards) <= 0
                    && (best.is_none() || compare_5_hands(&test, best.as_ref().unwrap()) > 0)
                {
                    best = Some(test);
                }
            }
        } else if n_jokers == 2 {
            for i in 0..subs.len() {
                for j in (i + 1)..subs.len() {
                    let mut test = non_jokers.clone();
                    test.push(subs[i]);
                    test.push(subs[j]);
                    if compare_5_hands(&test, ref_cards) <= 0
                        && (best.is_none() || compare_5_hands(&test, best.as_ref().unwrap()) > 0)
                    {
                        best = Some(test);
                    }
                }
            }
        }
        best.unwrap_or_else(|| cards.to_vec())
    }

    /// `constrain_3_vs_5` likewise.  The `<=` was `is_top_le_mid`, which had no
    /// other caller and went with the rewrite.
    fn mirror_3_vs_5(cards: &[Card], mid: &[Card]) -> Vec<Card> {
        let le_mid = |top: &[Card]| evaluate_hand_value(top, 3) <= evaluate_hand_value(mid, 5);
        if le_mid(cards) {
            return cards.to_vec();
        }
        let non_jokers: Vec<Card> = cards.iter().filter(|c| !c.is_joker()).cloned().collect();
        let n_jokers = cards.len() - non_jokers.len();
        let subs = available_subs(cards);
        let mut best: Option<Vec<Card>> = None;
        if n_jokers == 1 {
            for sub in &subs {
                let mut test = non_jokers.clone();
                test.push(*sub);
                if le_mid(&test)
                    && (best.is_none() || compare_3_hands(&test, best.as_ref().unwrap()) > 0)
                {
                    best = Some(test);
                }
            }
        } else if n_jokers == 2 {
            for i in 0..subs.len() {
                for j in (i + 1)..subs.len() {
                    let mut test = non_jokers.clone();
                    test.push(subs[i]);
                    test.push(subs[j]);
                    if le_mid(&test)
                        && (best.is_none() || compare_3_hands(&test, best.as_ref().unwrap()) > 0)
                    {
                        best = Some(test);
                    }
                }
            }
        }
        best.unwrap_or_else(|| cards.to_vec())
    }

    /// Distinct five-card rows that share a hand value.
    ///
    /// Jokered rows are drawn too: a bottom row holding one is ordinary, and
    /// its value comes from the exhaustive substitution rather than from its
    /// own cards -- which is the case a value-keyed cache has the most to lose
    /// on.
    fn equal_value_reference_pairs(state: &mut u64, wanted: usize) -> Vec<(Vec<Card>, Vec<Card>)> {
        let mut by_value: HashMap<u32, Vec<Vec<Card>>> = HashMap::new();
        let mut out = Vec::new();
        for round in 0..60_000usize {
            let hand: Vec<Card> = shuffled_deck(state, round % 4 == 0)[..5].to_vec();
            let bucket = by_value.entry(evaluate_hand_value(&hand, 5)).or_default();
            if bucket.iter().any(|seen| *seen == hand) {
                continue;
            }
            if let Some(other) = bucket.first() {
                out.push((other.clone(), hand.clone()));
                if out.len() >= wanted {
                    return out;
                }
            }
            bucket.push(hand);
        }
        out
    }

    /// `JOKER_EVAL_CACHE` keys on the sorted multiset, with both jokers on one
    /// code, so a hit is only exact if the value ignores card order and cannot
    /// tell one joker from the other.  Now that the cache actually stores
    /// anything, that is load-bearing.
    #[test]
    fn a_hand_value_depends_on_the_multiset_and_not_the_order() {
        let mut state: u64 = 0x2026_0813_0003;
        for round in 0..4_000usize {
            let width = if round % 2 == 0 { 5 } else { 3 };
            let jokers = round % 3;
            let row = jokered_row(&mut state, width, jokers.min(width));
            let value = evaluate_hand_value(&row, width);
            assert_eq!(
                joker_eval_cache_key(&row, width),
                joker_eval_cache_key(&row, width),
                "the key is not a function of its arguments"
            );
            let mut shuffled = row.clone();
            for index in (1..shuffled.len()).rev() {
                let pick = lcg(&mut state) % (index + 1);
                shuffled.swap(index, pick);
            }
            assert_eq!(
                joker_eval_cache_key(&shuffled, width),
                joker_eval_cache_key(&row, width),
                "reordering {row:?} to {shuffled:?} moved the key"
            );
            assert_eq!(
                evaluate_hand_value(&shuffled, width),
                value,
                "reordering {row:?} to {shuffled:?} moved its value"
            );
        }
    }

    /// The premise the constrained-row cache rests on: `constrain_*` reads its
    /// reference row through `evaluate_hand_value(ref, 5)` and through nothing
    /// else, so two references of equal value are one question.
    ///
    /// Asserted on the UNMEMOIZED mirror, which is the only way to say
    /// anything: put the same question to the cached function twice and the
    /// second answer is the first one back, whatever the premise is worth.  The
    /// cached function is then checked against the mirror in the same loop, so
    /// a wrong key shows up as a disagreement rather than as agreement with
    /// itself.
    #[test]
    fn the_reference_row_is_read_only_through_its_value() {
        let mut state: u64 = 0x2026_0813_0001;
        let pairs = equal_value_reference_pairs(&mut state, 240);
        assert!(pairs.len() >= 200, "only {} equal-value reference pairs", pairs.len());
        let (mut bound_mid, mut bound_top) = (0usize, 0usize);
        for (index, (ref_a, ref_b)) in pairs.iter().enumerate() {
            assert_eq!(evaluate_hand_value(ref_a, 5), evaluate_hand_value(ref_b, 5));
            let jokers = index % 2 + 1;
            let mid = jokered_row(&mut state, 5, jokers);
            let top = jokered_row(&mut state, 3, jokers);

            let mid_a = mirror_5_vs_5(&mid, ref_a);
            assert_eq!(
                mid_a,
                mirror_5_vs_5(&mid, ref_b),
                "two references of value {} constrain {mid:?} differently: {ref_a:?} vs {ref_b:?}",
                evaluate_hand_value(ref_a, 5)
            );
            let top_a = mirror_3_vs_5(&top, ref_a);
            assert_eq!(
                top_a,
                mirror_3_vs_5(&top, ref_b),
                "two references of value {} constrain {top:?} differently: {ref_a:?} vs {ref_b:?}",
                evaluate_hand_value(ref_a, 5)
            );

            // ref_a stores the entry, ref_b reads it back off the shared value.
            assert_eq!(constrain_5_vs_5(&mid, ref_a), mid_a);
            assert_eq!(constrain_5_vs_5(&mid, ref_b), mid_a);
            assert_eq!(constrain_3_vs_5(&top, ref_a), top_a);
            assert_eq!(constrain_3_vs_5(&top, ref_b), top_a);

            if mid_a != mid {
                bound_mid += 1;
            }
            if top_a != top {
                bound_top += 1;
            }
        }
        // Without this the test could pass on nothing but early returns, where
        // the reference is read once and the search never runs.
        assert!(
            bound_mid > 0 && bound_top > 0,
            "no reference bound anything: {bound_mid} middles, {bound_top} tops"
        );
        println!(
            "{} equal-value reference pairs; {bound_mid} bound the middle, {bound_top} the top",
            pairs.len()
        );
    }

    /// And the rewrite moved nothing: the memoized search agrees with the
    /// pre-change one on fresh inputs and on repeats, which is where the cache
    /// is the thing answering.
    #[test]
    fn the_constrained_search_agrees_with_the_unmemoized_one() {
        let mut state: u64 = 0x2026_0813_0002;
        let mut history: Vec<(Vec<Card>, Vec<Card>, Vec<Card>)> = Vec::new();
        for round in 0..600usize {
            let jokers = round % 3;
            let mid = jokered_row(&mut state, 5, jokers);
            let top = jokered_row(&mut state, 3, jokers);
            // A jokered bottom every fifth round: the reference's own value is
            // then the exhaustive one, and the key carries it.
            let bot = jokered_row(&mut state, 5, usize::from(round % 5 == 0));
            assert_eq!(
                constrain_5_vs_5(&mid, &bot),
                mirror_5_vs_5(&mid, &bot),
                "middle {mid:?} against bottom {bot:?}"
            );
            assert_eq!(
                constrain_3_vs_5(&top, &bot),
                mirror_3_vs_5(&top, &bot),
                "top {top:?} against middle {bot:?}"
            );
            history.push((top, mid, bot));
        }
        for (top, mid, bot) in &history {
            assert_eq!(constrain_5_vs_5(mid, bot), mirror_5_vs_5(mid, bot));
            assert_eq!(constrain_3_vs_5(top, bot), mirror_3_vs_5(top, bot));
        }
        println!("{} constrained rows, each checked cold and again on the cache", history.len());
    }

    #[test]
    fn middle_joker_downgrades_before_top_is_checked() {
        let top = [card(11, 0), card(10, 1), card(8, 2)];
        let middle = [card(13, 0), card(13, 1), card(12, 0), card(12, 1), joker()];
        let bottom = [card(14, 0), card(14, 1), card(14, 2), card(9, 0), card(8, 0)];

        let eval = evaluate_board_with_joker_constraint(&top, &middle, &bottom);
        assert!(!eval.busted);
        assert_eq!(category_from_value(evaluate_hand_value(&eval.mid, 5)), 2);
        assert_eq!(get_middle_royalty(&eval.mid), 0);
    }
}
