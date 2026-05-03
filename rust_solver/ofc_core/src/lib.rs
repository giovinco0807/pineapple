//! OFC Core - Shared card, hand evaluation, royalty, and bust logic
//!
//! Extracted from fl_solver main.rs for reuse across FL solver and backward induction.

use serde::{Deserialize, Serialize};

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

pub fn evaluate_5_card(cards: &[Card]) -> (HandRank, u32) {
    let rank_counts = count_ranks(cards);
    let suit_counts = count_suits(cards);
    let jokers = count_jokers(cards);

    let max_suit = suit_counts.iter().max().copied().unwrap_or(0);
    let is_flush = max_suit + jokers >= 5;

    let (is_straight, is_wheel) = is_straight_possible(&rank_counts, jokers);

    if is_flush && is_straight {
        let high = get_straight_high_card(&rank_counts, jokers);
        if high == 14 && !is_wheel {
            return (HandRank::RoyalFlush, calculate_strength(&rank_counts));
        }
        return (HandRank::StraightFlush, calculate_strength(&rank_counts));
    }

    let mut pairs = 0;
    let mut trips = 0;
    let mut quads = 0;
    let mut remaining_jokers = jokers;

    let mut counts_with_jokers: Vec<(u8, u8)> = Vec::new();
    for r in (2..=14).rev() {
        if rank_counts[r] > 0 {
            counts_with_jokers.push((r as u8, rank_counts[r]));
        }
    }

    for &(_, count) in &counts_with_jokers {
        if count + remaining_jokers >= 4 && count >= 1 {
            quads += 1;
            let used = 4 - count;
            remaining_jokers -= used.min(remaining_jokers);
        } else if count + remaining_jokers >= 3 && count >= 1 {
            trips += 1;
            let used = 3 - count;
            remaining_jokers -= used.min(remaining_jokers);
        } else if count >= 2 {
            pairs += 1;
        }
    }

    let rank = if quads >= 1 {
        HandRank::Quads
    } else if trips >= 1 && (pairs >= 1 || trips >= 2) {
        HandRank::FullHouse
    } else if is_flush {
        HandRank::Flush
    } else if is_straight {
        HandRank::Straight
    } else if trips >= 1 {
        HandRank::Trips
    } else if pairs >= 2 {
        HandRank::TwoPair
    } else if pairs >= 1 || (remaining_jokers >= 1 && counts_with_jokers.iter().any(|&(_, c)| c >= 1)) {
        HandRank::OnePair
    } else {
        HandRank::HighCard
    };

    (rank, calculate_strength(&rank_counts))
}

pub fn evaluate_3_card(cards: &[Card]) -> (HandRank3, u32) {
    let rank_counts = count_ranks(cards);
    let jokers = count_jokers(cards);

    let mut has_pair = false;
    let mut has_trips = false;

    for r in (2..=14).rev() {
        let count = rank_counts[r];
        if count + jokers >= 3 && count >= 1 {
            has_trips = true;
            break;
        }
        if count + jokers >= 2 && count >= 1 {
            has_pair = true;
        }
    }

    let rank = if has_trips {
        HandRank3::Trips
    } else if has_pair {
        HandRank3::OnePair
    } else {
        HandRank3::HighCard
    };

    let strength = calculate_strength(&rank_counts);
    (rank, strength)
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
    let (rank_a, str_a) = evaluate_5_card(a);
    let (rank_b, str_b) = evaluate_5_card(b);

    if (rank_a as u8) != (rank_b as u8) {
        return if (rank_a as u8) > (rank_b as u8) { 1 } else { -1 };
    }

    match rank_a {
        HandRank::OnePair => {
            let pair_a = get_pair_rank(a);
            let pair_b = get_pair_rank(b);
            if pair_a != pair_b {
                return if pair_a > pair_b { 1 } else { -1 };
            }
            if str_a > str_b { 1 } else if str_a < str_b { -1 } else { 0 }
        }
        HandRank::Trips | HandRank::FullHouse => {
            let trips_a = get_trips_rank(a);
            let trips_b = get_trips_rank(b);
            if trips_a != trips_b {
                return if trips_a > trips_b { 1 } else { -1 };
            }
            if str_a > str_b { 1 } else if str_a < str_b { -1 } else { 0 }
        }
        HandRank::TwoPair => {
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
            let quads_a = get_quads_rank(a);
            let quads_b = get_quads_rank(b);
            if quads_a != quads_b {
                return if quads_a > quads_b { 1 } else { -1 };
            }
            if str_a > str_b { 1 } else if str_a < str_b { -1 } else { 0 }
        }
        HandRank::Straight | HandRank::StraightFlush | HandRank::RoyalFlush => {
            let rc_a = count_ranks(a);
            let rc_b = count_ranks(b);
            let j_a = count_jokers(a);
            let j_b = count_jokers(b);
            let high_a = get_straight_high_card(&rc_a, j_a);
            let high_b = get_straight_high_card(&rc_b, j_b);
            if high_a > high_b { 1 } else if high_a < high_b { -1 } else { 0 }
        }
        _ => {
            if str_a > str_b { 1 } else if str_a < str_b { -1 } else { 0 }
        }
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
    // Bottom must be >= Middle
    if compare_5_hands(bottom, middle) < 0 {
        return false;
    }

    // Middle must be >= Top
    let (top_rank, _) = evaluate_3_card(top);
    let (mid_rank, _) = evaluate_5_card(middle);

    let top_rank_f: f64 = match top_rank {
        HandRank3::HighCard => 0.0,
        HandRank3::OnePair => 1.0,
        HandRank3::Trips => 2.5,
    };
    let mid_rank_f = mid_rank as u8 as f64;

    if top_rank_f > mid_rank_f {
        return false;
    }

    // Special case: 3-card Trips vs 5-card Trips
    if top_rank == HandRank3::Trips && mid_rank == HandRank::Trips {
        let top_trips = get_trips_rank(top);
        let mid_trips = get_trips_rank(middle);
        if top_trips > mid_trips {
            return false;
        }
    }

    // Same rank category: compare within
    if (top_rank_f - mid_rank_f).abs() < 0.01 {
        match top_rank {
            HandRank3::HighCard => {
                let top_high = top.iter().filter(|c| !c.is_joker()).map(|c| c.rank).max().unwrap_or(0);
                let mid_high = middle.iter().filter(|c| !c.is_joker()).map(|c| c.rank).max().unwrap_or(0);
                if top_high > mid_high {
                    return false;
                }
            }
            HandRank3::OnePair => {
                let top_pair = get_pair_rank(top);
                let mid_pair = get_pair_rank(middle);
                if top_pair > mid_pair {
                    return false;
                }
                if top_pair == mid_pair {
                    let top_kicker = top.iter().filter(|c| !c.is_joker() && c.rank != top_pair).map(|c| c.rank).max().unwrap_or(0);
                    let mid_kicker = middle.iter().filter(|c| !c.is_joker() && c.rank != mid_pair).map(|c| c.rank).max().unwrap_or(0);
                    if top_kicker > mid_kicker {
                        return false;
                    }
                }
            }
            HandRank3::Trips => {
                // Already handled above
            }
        }
    }

    true
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

/// Check if 3-card top ≤ 5-card mid (extracted from is_valid_placement)
fn is_top_le_mid(top: &[Card], mid: &[Card]) -> bool {
    let (top_rank, _) = evaluate_3_card(top);
    let (mid_rank, _) = evaluate_5_card(mid);

    let top_rank_f: f64 = match top_rank {
        HandRank3::HighCard => 0.0,
        HandRank3::OnePair => 1.0,
        HandRank3::Trips => 2.5,
    };
    let mid_rank_f = mid_rank as u8 as f64;

    if top_rank_f < mid_rank_f { return true; }
    if top_rank_f > mid_rank_f { return false; }

    // Same category
    if top_rank == HandRank3::Trips && mid_rank == HandRank::Trips {
        return get_trips_rank(top) <= get_trips_rank(mid);
    }
    match top_rank {
        HandRank3::HighCard => {
            let top_high = top.iter().filter(|c| !c.is_joker()).map(|c| c.rank).max().unwrap_or(0);
            let mid_high = mid.iter().filter(|c| !c.is_joker()).map(|c| c.rank).max().unwrap_or(0);
            top_high <= mid_high
        }
        HandRank3::OnePair => {
            let tp = get_pair_rank(top);
            let mp = get_pair_rank(mid);
            if tp != mp { return tp < mp; }
            let tk = top.iter().filter(|c| !c.is_joker() && c.rank != tp).map(|c| c.rank).max().unwrap_or(0);
            let mk = mid.iter().filter(|c| !c.is_joker() && c.rank != mp).map(|c| c.rank).max().unwrap_or(0);
            tk <= mk
        }
        _ => true,
    }
}

/// Compare two 3-card hands (for finding best substitution)
fn compare_3(a: &[Card], b: &[Card]) -> i32 {
    let (ra, sa) = evaluate_3_card(a);
    let (rb, sb) = evaluate_3_card(b);
    if (ra as u8) != (rb as u8) {
        return if (ra as u8) > (rb as u8) { 1 } else { -1 };
    }
    if sa > sb { 1 } else if sa < sb { -1 } else { 0 }
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

/// Find best 5-card joker substitution constrained to ≤ ref_cards (5-card)
fn constrain_5_vs_5(cards: &[Card], ref_cards: &[Card]) -> Vec<Card> {
    // If max eval already ≤ ref, keep original
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
            if compare_5_hands(&test, ref_cards) <= 0 {
                if best.is_none() || compare_5_hands(&test, best.as_ref().unwrap()) > 0 {
                    best = Some(test);
                }
            }
        }
    } else if n_jokers == 2 {
        for i in 0..subs.len() {
            for j in (i + 1)..subs.len() {
                let mut test = non_jokers.clone();
                test.push(subs[i]);
                test.push(subs[j]);
                if compare_5_hands(&test, ref_cards) <= 0 {
                    if best.is_none() || compare_5_hands(&test, best.as_ref().unwrap()) > 0 {
                        best = Some(test);
                    }
                }
            }
        }
    }

    // If no valid sub found → genuinely busted, return original
    best.unwrap_or_else(|| cards.to_vec())
}

/// Find best 3-card joker substitution constrained to ≤ mid (5-card)
fn constrain_3_vs_5(cards: &[Card], mid: &[Card]) -> Vec<Card> {
    // If max eval already ≤ mid, keep original
    if is_top_le_mid(cards, mid) {
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
            if is_top_le_mid(&test, mid) {
                if best.is_none() || compare_3(&test, best.as_ref().unwrap()) > 0 {
                    best = Some(test);
                }
            }
        }
    } else if n_jokers == 2 {
        for i in 0..subs.len() {
            for j in (i + 1)..subs.len() {
                let mut test = non_jokers.clone();
                test.push(subs[i]);
                test.push(subs[j]);
                if is_top_le_mid(&test, mid) {
                    if best.is_none() || compare_3(&test, best.as_ref().unwrap()) > 0 {
                        best = Some(test);
                    }
                }
            }
        }
    }

    best.unwrap_or_else(|| cards.to_vec())
}