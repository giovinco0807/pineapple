//! Allocation-free exact scoring for the internal T4 search hot path.
//!
//! Public scoring keeps the schema-friendly `HandValue(Vec<u8>)` representation.
//! Exact T4 enumeration only needs ordering, royalties, foul state, Fantasyland,
//! and terminal HU score, so it uses the fixed-width representation below.

use crate::cards::Card;
use crate::scoring::{
    HAND_FLUSH, HAND_FULL_HOUSE, HAND_HIGH, HAND_PAIR, HAND_QUADS, HAND_STRAIGHT,
    HAND_STRAIGHT_FLUSH, HAND_TRIPS, HAND_TWO_PAIR,
};
use crate::state::Board;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct CompactHandValue(u32);

impl CompactHandValue {
    fn new(category: u8, tie_breakers: &[u8]) -> Self {
        debug_assert!(tie_breakers.len() <= 5);
        let mut packed = u32::from(category) << 20;
        for (index, &rank) in tie_breakers.iter().enumerate() {
            debug_assert!(rank <= 14);
            packed |= u32::from(rank) << (16 - index * 4);
        }
        Self(packed)
    }

    fn category(self) -> u8 {
        (self.0 >> 20) as u8
    }

    fn primary_rank(self) -> u8 {
        ((self.0 >> 16) & 0x0f) as u8
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompactBoardScore {
    pub(crate) busted: bool,
    pub(crate) top_value: CompactHandValue,
    pub(crate) middle_value: CompactHandValue,
    pub(crate) bottom_value: CompactHandValue,
    pub(crate) total_royalty: i32,
    pub(crate) fl_entry_14: bool,
}

pub(crate) fn score_board_compact_trusted(board: &Board) -> CompactBoardScore {
    debug_assert_eq!(board.top.len(), 3);
    debug_assert_eq!(board.middle.len(), 5);
    debug_assert_eq!(board.bottom.len(), 5);

    let top_value = evaluate_3_card_compact_trusted(&board.top);
    let middle_value = evaluate_5_card_compact_trusted(&board.middle);
    let bottom_value = evaluate_5_card_compact_trusted(&board.bottom);
    let busted = top_value > middle_value || middle_value > bottom_value;
    let (top_royalty, middle_royalty, bottom_royalty, fl_entry_14) = if busted {
        (0, 0, 0, false)
    } else {
        (
            top_royalty_compact(top_value),
            middle_royalty_compact(middle_value),
            bottom_royalty_compact(bottom_value),
            qualifies_fl_14(top_value),
        )
    };

    CompactBoardScore {
        busted,
        top_value,
        middle_value,
        bottom_value,
        total_royalty: top_royalty + middle_royalty + bottom_royalty,
        fl_entry_14,
    }
}

pub(crate) fn heads_up_terminal_score_compact(
    own: &CompactBoardScore,
    opponent: &CompactBoardScore,
    fl_ev_14: f64,
) -> f64 {
    let own_royalty = if own.busted { 0 } else { own.total_royalty };
    let opponent_royalty = if opponent.busted {
        0
    } else {
        opponent.total_royalty
    };
    let own_fl = if own.busted || !own.fl_entry_14 {
        0.0
    } else {
        fl_ev_14
    };
    let opponent_fl = if opponent.busted || !opponent.fl_entry_14 {
        0.0
    } else {
        fl_ev_14
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
        (own.top_value, opponent.top_value),
        (own.middle_value, opponent.middle_value),
        (own.bottom_value, opponent.bottom_value),
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

fn evaluate_3_card_compact_trusted(cards: &[Card]) -> CompactHandValue {
    debug_assert_eq!(cards.len(), 3);
    let ranks = descending_ranks::<3>(cards);
    let counts = rank_counts(&ranks);
    let groups = rank_groups::<3>(&counts);
    if groups[0].0 == 3 {
        return CompactHandValue::new(HAND_TRIPS, &[groups[0].1]);
    }
    if groups[0].0 == 2 {
        let pair = groups[0].1;
        let kicker = ranks
            .iter()
            .copied()
            .find(|&rank| rank != pair)
            .expect("three-card pair must have a kicker");
        return CompactHandValue::new(HAND_PAIR, &[pair, kicker]);
    }
    CompactHandValue::new(HAND_HIGH, &ranks)
}

fn evaluate_5_card_compact_trusted(cards: &[Card]) -> CompactHandValue {
    debug_assert_eq!(cards.len(), 5);
    let ranks = descending_ranks::<5>(cards);
    let counts = rank_counts(&ranks);
    let groups = rank_groups::<5>(&counts);
    let flush = cards
        .iter()
        .all(|card| card.suit_index() == cards[0].suit_index());
    let straight = straight_high(&ranks);

    if let (true, Some(high)) = (flush, straight) {
        return CompactHandValue::new(HAND_STRAIGHT_FLUSH, &[high]);
    }
    if groups[0].0 == 4 {
        let quad = groups[0].1;
        let kicker = ranks
            .iter()
            .copied()
            .find(|&rank| rank != quad)
            .expect("quads must have a kicker");
        return CompactHandValue::new(HAND_QUADS, &[quad, kicker]);
    }
    if groups[0].0 == 3 && groups[1].0 == 2 {
        return CompactHandValue::new(HAND_FULL_HOUSE, &[groups[0].1, groups[1].1]);
    }
    if flush {
        return CompactHandValue::new(HAND_FLUSH, &ranks);
    }
    if let Some(high) = straight {
        return CompactHandValue::new(HAND_STRAIGHT, &[high]);
    }
    if groups[0].0 == 3 {
        let trips = groups[0].1;
        let mut tie_breakers = [0_u8; 3];
        tie_breakers[0] = trips;
        let mut cursor = 1;
        for rank in ranks.iter().copied().filter(|&rank| rank != trips) {
            tie_breakers[cursor] = rank;
            cursor += 1;
        }
        return CompactHandValue::new(HAND_TRIPS, &tie_breakers);
    }
    if groups[0].0 == 2 && groups[1].0 == 2 {
        let high_pair = groups[0].1;
        let low_pair = groups[1].1;
        let kicker = ranks
            .iter()
            .copied()
            .find(|&rank| rank != high_pair && rank != low_pair)
            .expect("two pair must have a kicker");
        return CompactHandValue::new(HAND_TWO_PAIR, &[high_pair, low_pair, kicker]);
    }
    if groups[0].0 == 2 {
        let pair = groups[0].1;
        let mut tie_breakers = [0_u8; 4];
        tie_breakers[0] = pair;
        let mut cursor = 1;
        for rank in ranks.iter().copied().filter(|&rank| rank != pair) {
            tie_breakers[cursor] = rank;
            cursor += 1;
        }
        return CompactHandValue::new(HAND_PAIR, &tie_breakers);
    }
    CompactHandValue::new(HAND_HIGH, &ranks)
}

fn descending_ranks<const N: usize>(cards: &[Card]) -> [u8; N] {
    debug_assert_eq!(cards.len(), N);
    let mut ranks = [0_u8; N];
    for (target, card) in ranks.iter_mut().zip(cards.iter().copied()) {
        *target = card.rank();
    }
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

fn rank_groups<const N: usize>(counts: &[u8; 15]) -> [(u8, u8); N] {
    let mut groups = [(0_u8, 0_u8); N];
    let mut cursor = 0;
    for rank in 2..=14 {
        let count = counts[rank as usize];
        if count > 0 {
            groups[cursor] = (count, rank);
            cursor += 1;
        }
    }
    groups.sort_unstable_by(|left, right| right.cmp(left));
    groups
}

fn straight_high(ranks: &[u8; 5]) -> Option<u8> {
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

fn top_royalty_compact(value: CompactHandValue) -> i32 {
    if value.category() == HAND_TRIPS {
        10 + i32::from(value.primary_rank() - 2)
    } else if value.category() == HAND_PAIR && value.primary_rank() >= 6 {
        i32::from(value.primary_rank() - 5)
    } else {
        0
    }
}

fn middle_royalty_compact(value: CompactHandValue) -> i32 {
    match (value.category(), value.primary_rank()) {
        (HAND_STRAIGHT_FLUSH, 14) => 50,
        (HAND_STRAIGHT_FLUSH, _) => 30,
        (HAND_QUADS, _) => 20,
        (HAND_FULL_HOUSE, _) => 12,
        (HAND_FLUSH, _) => 8,
        (HAND_STRAIGHT, _) => 4,
        (HAND_TRIPS, _) => 2,
        _ => 0,
    }
}

fn bottom_royalty_compact(value: CompactHandValue) -> i32 {
    match (value.category(), value.primary_rank()) {
        (HAND_STRAIGHT_FLUSH, 14) => 25,
        (HAND_STRAIGHT_FLUSH, _) => 15,
        (HAND_QUADS, _) => 10,
        (HAND_FULL_HOUSE, _) => 6,
        (HAND_FLUSH, _) => 4,
        (HAND_STRAIGHT, _) => 2,
        _ => 0,
    }
}

fn qualifies_fl_14(value: CompactHandValue) -> bool {
    value.category() == HAND_TRIPS || (value.category() == HAND_PAIR && value.primary_rank() >= 12)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use crate::cards::ALL_CARDS;
    use crate::scoring::{
        bottom_royalty_from_value, evaluate_3_card_trusted, evaluate_5_card_trusted,
        fl_entry_from_top_value, heads_up_terminal_score, middle_royalty_from_value,
        score_board_trusted, top_royalty_from_value, BoardScore, FantasylandEntry, HandValue,
    };
    use std::collections::BTreeMap;

    fn assert_value_matches(compact: CompactHandValue, legacy: &HandValue) {
        assert_eq!(compact, CompactHandValue::new(legacy.0, &legacy.1));
    }

    fn board(cards: &[Card]) -> Board {
        Board::new(
            cards[0..3].to_vec(),
            cards[3..8].to_vec(),
            cards[8..13].to_vec(),
        )
        .unwrap()
    }

    fn assert_board_matches(board: &Board) {
        let compact = score_board_compact_trusted(board);
        let legacy = score_board_trusted(board);
        assert_eq!(compact.busted, legacy.busted);
        assert_value_matches(compact.top_value, &legacy.top_value);
        assert_value_matches(compact.middle_value, &legacy.middle_value);
        assert_value_matches(compact.bottom_value, &legacy.bottom_value);
        assert_eq!(compact.total_royalty, legacy.total_royalty);
        assert_eq!(compact.fl_entry_14, legacy.fl_entry.card_count == 14);
    }

    #[test]
    #[ignore = "explicit candidate02 exhaustive equivalence gate"]
    fn candidate02_exhaustive_3_card_rank_royalty_and_fl_parity() {
        let mut count = 0_usize;
        for first in 0..50 {
            for second in (first + 1)..51 {
                for third in (second + 1)..52 {
                    let cards = [ALL_CARDS[first], ALL_CARDS[second], ALL_CARDS[third]];
                    let legacy = evaluate_3_card_trusted(&cards);
                    let compact = evaluate_3_card_compact_trusted(&cards);
                    assert_value_matches(compact, &legacy);
                    assert_eq!(
                        top_royalty_compact(compact),
                        top_royalty_from_value(&legacy)
                    );
                    assert_eq!(
                        qualifies_fl_14(compact),
                        fl_entry_from_top_value(&legacy).card_count == 14
                    );
                    count += 1;
                }
            }
        }
        assert_eq!(count, 22_100);
    }

    #[test]
    #[ignore = "explicit candidate02 exhaustive equivalence gate"]
    fn candidate02_exhaustive_5_card_rank_and_royalty_parity() {
        let mut count = 0_usize;
        for first in 0..48 {
            for second in (first + 1)..49 {
                for third in (second + 1)..50 {
                    for fourth in (third + 1)..51 {
                        for fifth in (fourth + 1)..52 {
                            let cards = [
                                ALL_CARDS[first],
                                ALL_CARDS[second],
                                ALL_CARDS[third],
                                ALL_CARDS[fourth],
                                ALL_CARDS[fifth],
                            ];
                            let legacy = evaluate_5_card_trusted(&cards);
                            let compact = evaluate_5_card_compact_trusted(&cards);
                            assert_value_matches(compact, &legacy);
                            assert_eq!(
                                middle_royalty_compact(compact),
                                middle_royalty_from_value(&legacy)
                            );
                            assert_eq!(
                                bottom_royalty_compact(compact),
                                bottom_royalty_from_value(&legacy)
                            );
                            count += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(count, 2_598_960);
    }

    #[test]
    #[ignore = "explicit candidate02 exhaustive equivalence gate"]
    fn candidate02_exhaustive_foul_order_parity() {
        let mut top_values = BTreeMap::<HandValue, CompactHandValue>::new();
        for first in 0..50 {
            for second in (first + 1)..51 {
                for third in (second + 1)..52 {
                    let cards = [ALL_CARDS[first], ALL_CARDS[second], ALL_CARDS[third]];
                    let legacy = evaluate_3_card_trusted(&cards);
                    let compact = evaluate_3_card_compact_trusted(&cards);
                    if let Some(existing) = top_values.insert(legacy, compact) {
                        assert_eq!(existing, compact);
                    }
                }
            }
        }

        let mut five_card_values = BTreeMap::<HandValue, CompactHandValue>::new();
        for first in 0..48 {
            for second in (first + 1)..49 {
                for third in (second + 1)..50 {
                    for fourth in (third + 1)..51 {
                        for fifth in (fourth + 1)..52 {
                            let cards = [
                                ALL_CARDS[first],
                                ALL_CARDS[second],
                                ALL_CARDS[third],
                                ALL_CARDS[fourth],
                                ALL_CARDS[fifth],
                            ];
                            let legacy = evaluate_5_card_trusted(&cards);
                            let compact = evaluate_5_card_compact_trusted(&cards);
                            if let Some(existing) = five_card_values.insert(legacy, compact) {
                                assert_eq!(existing, compact);
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(top_values.len(), 455);
        assert_eq!(five_card_values.len(), 7_462);
        for (top_legacy, top_compact) in &top_values {
            for (middle_legacy, middle_compact) in &five_card_values {
                assert_eq!(
                    top_legacy.cmp(middle_legacy),
                    top_compact.cmp(middle_compact)
                );
            }
        }
        for (middle_legacy, middle_compact) in &five_card_values {
            for (bottom_legacy, bottom_compact) in &five_card_values {
                assert_eq!(
                    middle_legacy.cmp(bottom_legacy),
                    middle_compact.cmp(bottom_compact)
                );
            }
        }
    }

    #[test]
    #[ignore = "explicit candidate02 exhaustive equivalence gate"]
    fn candidate02_exhaustive_hu_score_arithmetic_parity() {
        let line_values = [
            (
                HandValue(HAND_HIGH, vec![2]),
                CompactHandValue::new(HAND_HIGH, &[2]),
            ),
            (
                HandValue(HAND_HIGH, vec![3]),
                CompactHandValue::new(HAND_HIGH, &[3]),
            ),
        ];
        let fl_entry = |qualifies: bool| FantasylandEntry {
            qualifies,
            card_count: if qualifies { 14 } else { 0 },
            entry_type: qualifies.then(|| "qq".to_owned()),
        };
        let line_pair = |outcome: i8| match outcome {
            -1 => (&line_values[0], &line_values[1]),
            0 => (&line_values[0], &line_values[0]),
            1 => (&line_values[1], &line_values[0]),
            _ => unreachable!(),
        };

        let mut count = 0_u64;
        for own_busted in [false, true] {
            for opponent_busted in [false, true] {
                for own_royalty in 0..=97 {
                    for opponent_royalty in 0..=97 {
                        for own_fl in [false, true] {
                            for opponent_fl in [false, true] {
                                for encoded_outcomes in 0..27 {
                                    let mut cursor = encoded_outcomes;
                                    let top = line_pair((cursor % 3) as i8 - 1);
                                    cursor /= 3;
                                    let middle = line_pair((cursor % 3) as i8 - 1);
                                    cursor /= 3;
                                    let bottom = line_pair((cursor % 3) as i8 - 1);

                                    let legacy_own = BoardScore {
                                        busted: own_busted,
                                        top_value: top.0 .0.clone(),
                                        middle_value: middle.0 .0.clone(),
                                        bottom_value: bottom.0 .0.clone(),
                                        top_royalty: 0,
                                        middle_royalty: 0,
                                        bottom_royalty: 0,
                                        total_royalty: own_royalty,
                                        fl_entry: fl_entry(own_fl),
                                    };
                                    let legacy_opponent = BoardScore {
                                        busted: opponent_busted,
                                        top_value: top.1 .0.clone(),
                                        middle_value: middle.1 .0.clone(),
                                        bottom_value: bottom.1 .0.clone(),
                                        top_royalty: 0,
                                        middle_royalty: 0,
                                        bottom_royalty: 0,
                                        total_royalty: opponent_royalty,
                                        fl_entry: fl_entry(opponent_fl),
                                    };
                                    let compact_own = CompactBoardScore {
                                        busted: own_busted,
                                        top_value: top.0 .1,
                                        middle_value: middle.0 .1,
                                        bottom_value: bottom.0 .1,
                                        total_royalty: own_royalty,
                                        fl_entry_14: own_fl,
                                    };
                                    let compact_opponent = CompactBoardScore {
                                        busted: opponent_busted,
                                        top_value: top.1 .1,
                                        middle_value: middle.1 .1,
                                        bottom_value: bottom.1 .1,
                                        total_royalty: opponent_royalty,
                                        fl_entry_14: opponent_fl,
                                    };
                                    let legacy = heads_up_terminal_score(
                                        &legacy_own,
                                        &legacy_opponent,
                                        10.227_020_614_683_454,
                                    );
                                    let compact = heads_up_terminal_score_compact(
                                        &compact_own,
                                        &compact_opponent,
                                        10.227_020_614_683_454,
                                    );
                                    assert_eq!(legacy.to_bits(), compact.to_bits());
                                    count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(count, 4_148_928);
    }

    #[test]
    fn compact_board_foul_fl_royalty_and_hu_score_are_bit_exact() {
        for offset in 0..52 {
            let cards = (0..26)
                .map(|index| ALL_CARDS[(offset + index) % 52])
                .collect::<Vec<_>>();
            let own_board = board(&cards[..13]);
            let opponent_board = board(&cards[13..]);
            assert_board_matches(&own_board);
            assert_board_matches(&opponent_board);

            let own_compact = score_board_compact_trusted(&own_board);
            let opponent_compact = score_board_compact_trusted(&opponent_board);
            let own_legacy = score_board_trusted(&own_board);
            let opponent_legacy = score_board_trusted(&opponent_board);
            let compact = heads_up_terminal_score_compact(
                &own_compact,
                &opponent_compact,
                10.227_020_614_683_454,
            );
            let legacy =
                heads_up_terminal_score(&own_legacy, &opponent_legacy, 10.227_020_614_683_454);
            assert_eq!(compact.to_bits(), legacy.to_bits());
        }

        let fl_board = Board::new(
            ["Qc", "Qs", "4d"]
                .into_iter()
                .map(|card| card.parse().unwrap())
                .collect(),
            ["2h", "3h", "4h", "5h", "7h"]
                .into_iter()
                .map(|card| card.parse().unwrap())
                .collect(),
            ["Ah", "Kh", "Qh", "Jh", "Th"]
                .into_iter()
                .map(|card| card.parse().unwrap())
                .collect(),
        )
        .unwrap();
        assert_board_matches(&fl_board);
        let compact = score_board_compact_trusted(&fl_board);
        assert!(!compact.busted);
        assert!(compact.fl_entry_14);
        assert_eq!(compact.total_royalty, 40);
    }
}
