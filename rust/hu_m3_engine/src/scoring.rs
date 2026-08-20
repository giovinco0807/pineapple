//! Exact regular OFC hand ranking, royalties, Fantasyland EV, and HU scoring.

use crate::cards::{validate_cards, Card};
use crate::state::Board;
use serde::{Deserialize, Serialize};

pub const HAND_HIGH: u8 = 0;
pub const HAND_PAIR: u8 = 1;
pub const HAND_TWO_PAIR: u8 = 2;
pub const HAND_TRIPS: u8 = 3;
pub const HAND_STRAIGHT: u8 = 4;
pub const HAND_FLUSH: u8 = 5;
pub const HAND_FULL_HOUSE: u8 = 6;
pub const HAND_QUADS: u8 = 7;
pub const HAND_STRAIGHT_FLUSH: u8 = 8;

/// Mirrors `configs/fl_ev_regular_v4_selfplay.json` (M6 run B, 2026-08-06),
/// superseding 9.109 and, before it, 10.227_020_614_683_454. Reference value
/// only: scoring always takes `fl_ev` from the observation's scoring context.
pub const DEFAULT_FL_EV_14: f64 = 9.6;

/// Lexicographically comparable `(category, tie_breakers)` poker value.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct HandValue(pub u8, pub Vec<u8>);

/// `HandValue`'s ordering as one integer: the category in the top byte and the
/// tie-breakers, zero-padded, below it. Tie-breakers are card ranks (`2..=14`,
/// never zero) and there are at most five, so padding a shorter list with zeros
/// reproduces `Vec`'s "a prefix sorts first" rule exactly. Comparing these is
/// what lets the arrangement loop and the head-to-head block stay out of the
/// heap.
pub(crate) fn compare_key(value: &HandValue) -> u64 {
    debug_assert!(value.1.len() <= 7, "tie-breakers overflow the packed key");
    let mut key = (value.0 as u64) << 56;
    for (slot, rank) in value.1.iter().take(7).enumerate() {
        key |= (*rank as u64) << (48 - 8 * slot);
    }
    key
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FantasylandEntry {
    pub qualifies: bool,
    pub card_count: u8,
    pub entry_type: Option<String>,
}

impl FantasylandEntry {
    fn none() -> Self {
        Self {
            qualifies: false,
            card_count: 0,
            entry_type: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BoardScore {
    pub busted: bool,
    pub top_value: HandValue,
    pub middle_value: HandValue,
    pub bottom_value: HandValue,
    pub top_royalty: i32,
    pub middle_royalty: i32,
    pub bottom_royalty: i32,
    pub total_royalty: i32,
    pub fl_entry: FantasylandEntry,
}

pub fn evaluate_5_card(cards: &[Card]) -> Result<HandValue, String> {
    if cards.len() != 5 {
        return Err(format!(
            "5-card evaluation requires 5 cards, got {}",
            cards.len()
        ));
    }
    validate_cards(cards)?;
    Ok(evaluate_5_card_trusted(cards))
}

pub(crate) fn evaluate_5_card_trusted(cards: &[Card]) -> HandValue {
    let ranks = rank_values(cards);
    let counts = rank_counts(&ranks);
    let groups = rank_groups(&counts);
    let flush = cards
        .iter()
        .all(|card| card.suit_index() == cards[0].suit_index());
    let straight = straight_high(&ranks);

    if let (true, Some(straight_high)) = (flush, straight) {
        return HandValue(HAND_STRAIGHT_FLUSH, vec![straight_high]);
    }
    if groups[0].0 == 4 {
        let quad = groups[0].1;
        let kicker = *ranks.iter().filter(|&&rank| rank != quad).max().unwrap();
        return HandValue(HAND_QUADS, vec![quad, kicker]);
    }
    if groups[0].0 == 3 && groups[1].0 == 2 {
        return HandValue(HAND_FULL_HOUSE, vec![groups[0].1, groups[1].1]);
    }
    if flush {
        return HandValue(HAND_FLUSH, ranks);
    }
    if let Some(high) = straight {
        return HandValue(HAND_STRAIGHT, vec![high]);
    }
    if groups[0].0 == 3 {
        let trips = groups[0].1;
        let mut tie_breakers = vec![trips];
        tie_breakers.extend(ranks.iter().copied().filter(|&rank| rank != trips));
        return HandValue(HAND_TRIPS, tie_breakers);
    }

    let pairs: Vec<u8> = (2..=14)
        .rev()
        .filter(|&rank| counts[rank as usize] == 2)
        .collect();
    if pairs.len() == 2 {
        let kicker = *ranks
            .iter()
            .filter(|rank| !pairs.contains(rank))
            .max()
            .unwrap();
        return HandValue(HAND_TWO_PAIR, vec![pairs[0], pairs[1], kicker]);
    }
    if pairs.len() == 1 {
        let pair = pairs[0];
        let mut tie_breakers = vec![pair];
        tie_breakers.extend(ranks.iter().copied().filter(|&rank| rank != pair));
        return HandValue(HAND_PAIR, tie_breakers);
    }
    HandValue(HAND_HIGH, ranks)
}

pub fn evaluate_3_card(cards: &[Card]) -> Result<HandValue, String> {
    if cards.len() != 3 {
        return Err(format!(
            "3-card evaluation requires 3 cards, got {}",
            cards.len()
        ));
    }
    validate_cards(cards)?;
    Ok(evaluate_3_card_trusted(cards))
}

pub(crate) fn evaluate_3_card_trusted(cards: &[Card]) -> HandValue {
    let ranks = rank_values(cards);
    let counts = rank_counts(&ranks);
    let groups = rank_groups(&counts);
    if groups[0].0 == 3 {
        return HandValue(HAND_TRIPS, vec![groups[0].1]);
    }
    if groups[0].0 == 2 {
        let pair = groups[0].1;
        let kicker = *ranks.iter().filter(|&&rank| rank != pair).max().unwrap();
        return HandValue(HAND_PAIR, vec![pair, kicker]);
    }
    HandValue(HAND_HIGH, ranks)
}

pub fn get_top_royalty(cards: &[Card]) -> Result<i32, String> {
    let value = evaluate_3_card(cards)?;
    Ok(top_royalty_from_value(&value))
}

pub fn get_middle_royalty(cards: &[Card]) -> Result<i32, String> {
    let value = evaluate_5_card(cards)?;
    Ok(middle_royalty_from_value(&value))
}

pub fn get_bottom_royalty(cards: &[Card]) -> Result<i32, String> {
    let value = evaluate_5_card(cards)?;
    Ok(bottom_royalty_from_value(&value))
}

pub(crate) fn top_royalty_from_value(value: &HandValue) -> i32 {
    if value.0 == HAND_TRIPS {
        10 + i32::from(value.1[0] - 2)
    } else if value.0 == HAND_PAIR && value.1[0] >= 6 {
        i32::from(value.1[0] - 5)
    } else {
        0
    }
}

pub(crate) fn middle_royalty_from_value(value: &HandValue) -> i32 {
    match (value.0, value.1.first().copied()) {
        (HAND_STRAIGHT_FLUSH, Some(14)) => 50,
        (HAND_STRAIGHT_FLUSH, _) => 30,
        (HAND_QUADS, _) => 20,
        (HAND_FULL_HOUSE, _) => 12,
        (HAND_FLUSH, _) => 8,
        (HAND_STRAIGHT, _) => 4,
        (HAND_TRIPS, _) => 2,
        _ => 0,
    }
}

pub(crate) fn bottom_royalty_from_value(value: &HandValue) -> i32 {
    match (value.0, value.1.first().copied()) {
        (HAND_STRAIGHT_FLUSH, Some(14)) => 25,
        (HAND_STRAIGHT_FLUSH, _) => 15,
        (HAND_QUADS, _) => 10,
        (HAND_FULL_HOUSE, _) => 6,
        (HAND_FLUSH, _) => 4,
        (HAND_STRAIGHT, _) => 2,
        _ => 0,
    }
}

pub(crate) fn fl_entry_from_top_value(value: &HandValue) -> FantasylandEntry {
    let entry_type = if value.0 == HAND_TRIPS {
        Some("trips")
    } else if value.0 == HAND_PAIR && value.1[0] == 14 {
        Some("aa")
    } else if value.0 == HAND_PAIR && value.1[0] == 13 {
        Some("kk")
    } else if value.0 == HAND_PAIR && value.1[0] == 12 {
        Some("qq")
    } else {
        None
    };
    match entry_type {
        Some(entry_type) => FantasylandEntry {
            qualifies: true,
            card_count: 14,
            entry_type: Some(entry_type.to_string()),
        },
        None => FantasylandEntry::none(),
    }
}

pub fn check_fl_entry(top: &[Card]) -> Result<FantasylandEntry, String> {
    if top.len() != 3 {
        return Ok(FantasylandEntry::none());
    }
    validate_cards(top)?;
    Ok(fl_entry_from_top_value(&evaluate_3_card_trusted(top)))
}

pub fn score_board(board: &Board) -> Result<BoardScore, String> {
    board.validate()?;
    if board.top.len() != 3 || board.middle.len() != 5 || board.bottom.len() != 5 {
        return Err("board must have 3 top, 5 middle, and 5 bottom cards".to_string());
    }
    Ok(score_board_trusted(board))
}

/// Score a complete board assembled from validated native actions. The public
/// scorer retains all schema/card checks; exact search can avoid repeating
/// them millions of times on boards it just constructed.
pub(crate) fn score_board_trusted(board: &Board) -> BoardScore {
    debug_assert_eq!(board.top.len(), 3);
    debug_assert_eq!(board.middle.len(), 5);
    debug_assert_eq!(board.bottom.len(), 5);
    let top_value = evaluate_3_card_trusted(&board.top);
    let middle_value = evaluate_5_card_trusted(&board.middle);
    let bottom_value = evaluate_5_card_trusted(&board.bottom);
    let busted = top_value > middle_value || middle_value > bottom_value;
    // The three hand values above already contain every category and rank
    // needed by royalties and Fantasyland entry.  Reusing them avoids three
    // duplicate poker evaluations (plus their validation/allocation work) for
    // every terminal board scored by exact T4 enumeration.
    let (top_royalty, middle_royalty, bottom_royalty, fl_entry) = if busted {
        (0, 0, 0, FantasylandEntry::none())
    } else {
        (
            top_royalty_from_value(&top_value),
            middle_royalty_from_value(&middle_value),
            bottom_royalty_from_value(&bottom_value),
            fl_entry_from_top_value(&top_value),
        )
    };
    BoardScore {
        busted,
        top_value,
        middle_value,
        bottom_value,
        top_royalty,
        middle_royalty,
        bottom_royalty,
        total_royalty: top_royalty + middle_royalty + bottom_royalty,
        fl_entry,
    }
}

/// Score a complete board, optionally against a complete opponent board.
pub fn terminal_score(
    board: &Board,
    opponent_board: Option<&Board>,
    fl_ev_14: f64,
) -> Result<(f64, BoardScore), String> {
    if !fl_ev_14.is_finite() {
        return Err("Fantasyland EV must be finite".to_string());
    }
    let own = score_board(board)?;
    let Some(opponent_board) = opponent_board else {
        let score = standalone_terminal_score(&own, fl_ev_14);
        return Ok((score, own));
    };
    let own_mask = board
        .all_cards()
        .into_iter()
        .fold(0_u64, |mask, card| mask | card.bit());
    if opponent_board
        .all_cards()
        .iter()
        .any(|card| own_mask & card.bit() != 0)
    {
        return Err("hero and opponent boards overlap".to_string());
    }
    let opponent = score_board(opponent_board)?;
    let score = heads_up_terminal_score(&own, &opponent, fl_ev_14);
    Ok((score, own))
}

pub fn standalone_terminal_score(board: &BoardScore, fl_ev_14: f64) -> f64 {
    if board.busted {
        0.0
    } else {
        f64::from(board.total_royalty) + fantasyland_bonus(board, fl_ev_14)
    }
}

pub fn heads_up_terminal_score(own: &BoardScore, opponent: &BoardScore, fl_ev_14: f64) -> f64 {
    let own_royalty = if own.busted { 0 } else { own.total_royalty };
    let opponent_royalty = if opponent.busted {
        0
    } else {
        opponent.total_royalty
    };
    let own_fl = if own.busted {
        0.0
    } else {
        fantasyland_bonus(own, fl_ev_14)
    };
    let opponent_fl = if opponent.busted {
        0.0
    } else {
        fantasyland_bonus(opponent, fl_ev_14)
    };

    if own.busted && opponent.busted {
        return 0.0;
    }
    if own.busted {
        return -6.0 - f64::from(opponent_royalty) - opponent_fl;
    }
    if opponent.busted {
        return 6.0 + f64::from(own_royalty) + own_fl;
    }

    let mut line_total = 0_i32;
    for (own_value, opponent_value) in [
        (&own.top_value, &opponent.top_value),
        (&own.middle_value, &opponent.middle_value),
        (&own.bottom_value, &opponent.bottom_value),
    ] {
        line_total += if own_value > opponent_value {
            1
        } else if own_value < opponent_value {
            -1
        } else {
            0
        };
    }
    let scoop = if line_total.abs() == 3 {
        3 * line_total.signum()
    } else {
        0
    };
    f64::from(line_total + scoop + own_royalty - opponent_royalty) + own_fl - opponent_fl
}

fn fantasyland_bonus(board: &BoardScore, fl_ev_14: f64) -> f64 {
    if board.fl_entry.card_count == 14 {
        fl_ev_14
    } else {
        0.0
    }
}

fn rank_values(cards: &[Card]) -> Vec<u8> {
    let mut ranks: Vec<u8> = cards.iter().map(|card| card.rank()).collect();
    ranks.sort_unstable_by(|left, right| right.cmp(left));
    ranks
}

fn rank_counts(ranks: &[u8]) -> [u8; 15] {
    let mut counts = [0_u8; 15];
    for &rank in ranks {
        counts[rank as usize] += 1;
    }
    counts
}

fn rank_groups(counts: &[u8; 15]) -> Vec<(u8, u8)> {
    let mut groups: Vec<(u8, u8)> = (2..=14)
        .filter_map(|rank| {
            let count = counts[rank as usize];
            (count > 0).then_some((count, rank))
        })
        .collect();
    groups.sort_unstable_by(|left, right| right.cmp(left));
    groups
}

fn straight_high(ranks: &[u8]) -> Option<u8> {
    let mut present = [false; 15];
    for &rank in ranks {
        present[rank as usize] = true;
    }
    if [14, 5, 4, 3, 2].iter().all(|&rank| present[rank as usize]) {
        return Some(5);
    }
    (6..=14)
        .rev()
        .find(|&high| ((high - 4)..=high).all(|rank| present[rank as usize]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cards(values: &[&str]) -> Vec<Card> {
        values.iter().map(|value| value.parse().unwrap()).collect()
    }

    fn board(top: &[&str], middle: &[&str], bottom: &[&str]) -> Board {
        Board::new(cards(top), cards(middle), cards(bottom)).unwrap()
    }

    #[test]
    fn ranking_handles_wheel_and_normal_category_order() {
        assert_eq!(
            evaluate_5_card(&cards(&["Ah", "2d", "3c", "4s", "5h"])).unwrap(),
            HandValue(HAND_STRAIGHT, vec![5])
        );
        assert!(
            evaluate_5_card(&cards(&["2h", "2d", "2c", "3s", "3h"])).unwrap()
                > evaluate_5_card(&cards(&["Ah", "Kh", "9h", "5h", "3h"])).unwrap()
        );
        assert!(
            evaluate_3_card(&cards(&["2h", "2d", "2c"])).unwrap()
                > evaluate_3_card(&cards(&["Ah", "Ad", "Kc"])).unwrap()
        );
    }

    #[test]
    fn royalties_are_python_golden_values_including_middle_trips_two() {
        assert_eq!(get_top_royalty(&cards(&["6h", "6s", "4d"])).unwrap(), 1);
        assert_eq!(get_top_royalty(&cards(&["Ah", "As", "Ad"])).unwrap(), 22);
        assert_eq!(
            get_middle_royalty(&cards(&["Ah", "Ad", "Ac", "Ks", "Qh"])).unwrap(),
            2
        );
        assert_eq!(
            get_middle_royalty(&cards(&["Ah", "Kh", "Qh", "Jh", "Th"])).unwrap(),
            50
        );
        assert_eq!(
            get_bottom_royalty(&cards(&["Ah", "Kh", "Qh", "Jh", "Th"])).unwrap(),
            25
        );
    }

    #[test]
    fn hu_terminal_score_matches_python_twenty_one_fixture_and_is_zero_sum() {
        let hero = board(
            &["Qh", "Qs", "2d"],
            &["Kh", "Kd", "6c", "8s", "Th"],
            &["9c", "9d", "9s", "Kc", "Ah"],
        );
        let opponent = board(
            &["Jh", "Js", "3d"],
            &["2h", "3h", "4c", "5d", "7s"],
            &["Ac", "Ad", "4h", "4s", "8c"],
        );
        let (score, hero_score) = terminal_score(&hero, Some(&opponent), 8.0).unwrap();
        assert_eq!(score, 21.0);
        assert!(!hero_score.busted);
        assert_eq!(hero_score.fl_entry.entry_type.as_deref(), Some("qq"));
        let reverse = terminal_score(&opponent, Some(&hero), 8.0).unwrap().0;
        assert_eq!(score, -reverse);
    }

    #[test]
    fn bust_fixture_subtracts_opponent_royalty_and_fl() {
        let hero = board(
            &["Ah", "As", "Kd"],
            &["2h", "3h", "4c", "5d", "7s"],
            &["4d", "4s", "8c", "9c", "Td"],
        );
        let opponent = board(
            &["Qh", "Qs", "2d"],
            &["Kh", "Kc", "6c", "8s", "Jh"],
            &["9h", "9s", "9d", "Tc", "Jc"],
        );
        let (score, hero_score) = terminal_score(&hero, Some(&opponent), 8.0).unwrap();
        assert!(hero_score.busted);
        assert_eq!(score, -21.0);
    }

    #[test]
    fn regular_fl_and_total_royalty_fixture() {
        let fixture = board(
            &["Qc", "Qs", "4d"],
            &["2h", "3h", "4h", "5h", "7h"],
            &["Ah", "Kh", "Qh", "Jh", "Th"],
        );
        let scored = score_board(&fixture).unwrap();
        assert!(!scored.busted);
        assert_eq!(scored.total_royalty, 40);
        assert_eq!(scored.fl_entry.card_count, 14);
        assert_eq!(scored.top_royalty, get_top_royalty(&fixture.top).unwrap());
        assert_eq!(
            scored.middle_royalty,
            get_middle_royalty(&fixture.middle).unwrap()
        );
        assert_eq!(
            scored.bottom_royalty,
            get_bottom_royalty(&fixture.bottom).unwrap()
        );
        assert_eq!(scored.fl_entry, check_fl_entry(&fixture.top).unwrap());
    }
}
