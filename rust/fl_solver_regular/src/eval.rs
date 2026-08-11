//! Packed hand ranking and the regular royalty tables.
//!
//! `ofc_hu_m3_engine::scoring` is the authority. It returns
//! `HandValue(category, Vec<tie_breakers>)`, which allocates; the FL search
//! evaluates ~2.4k combinations per deal and then compares them across a
//! million arrangements, so a heap allocation per hand value is not viable.
//!
//! This module packs the identical `(category, tie_breakers)` tuple into a
//! `u32` whose numeric order is the engine's lexicographic order:
//!
//! ```text
//! bits 23..20  category (0..=8)
//! bits 19..16  tie breaker 0        (rank 2..=14, 0 when absent)
//! bits 15..12  tie breaker 1
//! bits 11.. 8  tie breaker 2
//! bits  7.. 4  tie breaker 3
//! bits  3.. 0  tie breaker 4
//! ```
//!
//! Zero-padding a short tie-breaker list reproduces `Vec` ordering exactly,
//! because `Vec<u8>` orders a strict prefix below its extension and every
//! engine category emits a fixed-length list. `tests/engine_parity.rs` pins
//! this against the engine over randomized hands, including the mixed
//! 3-card-vs-5-card comparisons the foul check performs.

use crate::cards::{rank_of, suit_of};

pub const HAND_HIGH: u32 = 0;
pub const HAND_PAIR: u32 = 1;
pub const HAND_TWO_PAIR: u32 = 2;
pub const HAND_TRIPS: u32 = 3;
pub const HAND_STRAIGHT: u32 = 4;
pub const HAND_FLUSH: u32 = 5;
pub const HAND_FULL_HOUSE: u32 = 6;
pub const HAND_QUADS: u32 = 7;
pub const HAND_STRAIGHT_FLUSH: u32 = 8;

/// Packed `(category, tie_breakers)` poker value; compare with `<`/`>`.
pub type HandKey = u32;

#[inline(always)]
pub const fn category(key: HandKey) -> u32 {
    key >> 20
}

#[inline(always)]
pub const fn primary_rank(key: HandKey) -> u32 {
    (key >> 16) & 0xF
}

#[inline(always)]
fn pack(cat: u32, tie_breakers: &[u8]) -> HandKey {
    let mut key = cat << 20;
    let mut shift = 16_i32;
    for value in tie_breakers.iter().take(5) {
        key |= (*value as u32) << shift;
        shift -= 4;
    }
    key
}

#[inline(always)]
fn straight_high(present: u16) -> u8 {
    // `present` is a rank-2..14 bitmap in bits 2..14.
    const WHEEL: u16 = (1 << 14) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 2);
    for high in (6..=14_u8).rev() {
        let window: u16 = 0b11111 << (high - 4);
        if present & window == window {
            return high;
        }
    }
    if present & WHEEL == WHEEL {
        return 5;
    }
    0
}

/// Rank the five cards. Order and royalties match `evaluate_5_card`.
pub fn eval5(cards: &[u8; 5]) -> HandKey {
    let mut counts = [0_u8; 15];
    let mut present = 0_u16;
    let mut ranks = [0_u8; 5];
    for (slot, card) in cards.iter().enumerate() {
        let rank = rank_of(*card);
        ranks[slot] = rank;
        counts[rank as usize] += 1;
        present |= 1 << rank;
    }
    ranks.sort_unstable_by(|left, right| right.cmp(left));

    let suit = suit_of(cards[0]);
    let flush = cards.iter().all(|card| suit_of(*card) == suit);
    let straight = straight_high(present);

    if flush && straight > 0 {
        return pack(HAND_STRAIGHT_FLUSH, &[straight]);
    }

    // Descending (count, rank) groups, matching the engine's `rank_groups`.
    let mut groups: [(u8, u8); 5] = [(0, 0); 5];
    let mut group_count = 0_usize;
    for rank in 2..=14_u8 {
        let count = counts[rank as usize];
        if count > 0 {
            groups[group_count] = (count, rank);
            group_count += 1;
        }
    }
    groups[..group_count].sort_unstable_by(|left, right| right.cmp(left));

    if groups[0].0 == 4 {
        let quad = groups[0].1;
        let kicker = ranks.iter().copied().find(|rank| *rank != quad).unwrap_or(0);
        return pack(HAND_QUADS, &[quad, kicker]);
    }
    if groups[0].0 == 3 && group_count > 1 && groups[1].0 == 2 {
        return pack(HAND_FULL_HOUSE, &[groups[0].1, groups[1].1]);
    }
    if flush {
        return pack(HAND_FLUSH, &ranks);
    }
    if straight > 0 {
        return pack(HAND_STRAIGHT, &[straight]);
    }
    if groups[0].0 == 3 {
        let trips = groups[0].1;
        let mut tie = [trips, 0, 0];
        let mut slot = 1;
        for rank in ranks.iter().copied() {
            if rank != trips {
                tie[slot] = rank;
                slot += 1;
            }
        }
        return pack(HAND_TRIPS, &tie);
    }

    let mut pairs = [0_u8; 2];
    let mut pair_count = 0_usize;
    for group in groups[..group_count].iter() {
        if group.0 == 2 && pair_count < 2 {
            pairs[pair_count] = group.1;
            pair_count += 1;
        }
    }
    if pair_count == 2 {
        let kicker = ranks
            .iter()
            .copied()
            .find(|rank| *rank != pairs[0] && *rank != pairs[1])
            .unwrap_or(0);
        return pack(HAND_TWO_PAIR, &[pairs[0], pairs[1], kicker]);
    }
    if pair_count == 1 {
        let pair = pairs[0];
        let mut tie = [pair, 0, 0, 0];
        let mut slot = 1;
        for rank in ranks.iter().copied() {
            if rank != pair {
                tie[slot] = rank;
                slot += 1;
            }
        }
        return pack(HAND_PAIR, &tie);
    }
    pack(HAND_HIGH, &ranks)
}

/// Rank the three cards. Order matches `evaluate_3_card`.
pub fn eval3(cards: &[u8; 3]) -> HandKey {
    let mut ranks = [rank_of(cards[0]), rank_of(cards[1]), rank_of(cards[2])];
    ranks.sort_unstable_by(|left, right| right.cmp(left));

    if ranks[0] == ranks[2] {
        return pack(HAND_TRIPS, &[ranks[0]]);
    }
    if ranks[0] == ranks[1] {
        return pack(HAND_PAIR, &[ranks[0], ranks[2]]);
    }
    if ranks[1] == ranks[2] {
        return pack(HAND_PAIR, &[ranks[1], ranks[0]]);
    }
    pack(HAND_HIGH, &ranks)
}

/// Top-row royalty: trips `10 + (rank - 2)`, pairs `66+` `rank - 5`.
#[inline(always)]
pub fn top_royalty(key: HandKey) -> i32 {
    let cat = category(key);
    let rank = primary_rank(key) as i32;
    if cat == HAND_TRIPS {
        10 + (rank - 2)
    } else if cat == HAND_PAIR && rank >= 6 {
        rank - 5
    } else {
        0
    }
}

#[inline(always)]
pub fn middle_royalty(key: HandKey) -> i32 {
    match category(key) {
        HAND_STRAIGHT_FLUSH if primary_rank(key) == 14 => 50,
        HAND_STRAIGHT_FLUSH => 30,
        HAND_QUADS => 20,
        HAND_FULL_HOUSE => 12,
        HAND_FLUSH => 8,
        HAND_STRAIGHT => 4,
        HAND_TRIPS => 2,
        _ => 0,
    }
}

#[inline(always)]
pub fn bottom_royalty(key: HandKey) -> i32 {
    match category(key) {
        HAND_STRAIGHT_FLUSH if primary_rank(key) == 14 => 25,
        HAND_STRAIGHT_FLUSH => 15,
        HAND_QUADS => 10,
        HAND_FULL_HOUSE => 6,
        HAND_FLUSH => 4,
        HAND_STRAIGHT => 2,
        _ => 0,
    }
}

/// Fantasyland *entry* for a non-fouled board: QQ+ pair or any trips on top.
/// Regular rules always deal 14 for every entry type, so the card count is not
/// a function of the entry type; see `configs/fl_ev_regular_v4_selfplay.json`.
#[inline(always)]
pub fn fl_entry_from_top(top_key: HandKey) -> Option<&'static str> {
    let cat = category(top_key);
    let rank = primary_rank(top_key);
    if cat == HAND_TRIPS {
        Some("trips")
    } else if cat == HAND_PAIR && rank == 14 {
        Some("aa")
    } else if cat == HAND_PAIR && rank == 13 {
        Some("kk")
    } else if cat == HAND_PAIR && rank == 12 {
        Some("qq")
    } else {
        None
    }
}

/// Fantasyland *stay* for a player already in Fantasyland: trips on top or
/// quads-or-better on the bottom. Mirrors `ofc_regular.rules.check_fl_stay`.
#[inline(always)]
pub fn fl_stay(top_key: HandKey, bottom_key: HandKey) -> Option<&'static str> {
    if category(top_key) == HAND_TRIPS {
        Some("stay_top_trips")
    } else if category(bottom_key) >= HAND_QUADS {
        Some("stay_bottom_quads_plus")
    } else {
        None
    }
}

#[inline(always)]
pub fn is_foul(top_key: HandKey, middle_key: HandKey, bottom_key: HandKey) -> bool {
    top_key > middle_key || middle_key > bottom_key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::parse_cards;

    fn five(text: &str) -> [u8; 5] {
        parse_cards(text).unwrap().try_into().unwrap()
    }

    fn three(text: &str) -> [u8; 3] {
        parse_cards(text).unwrap().try_into().unwrap()
    }

    #[test]
    fn wheel_and_category_order_match_the_engine_fixtures() {
        assert_eq!(category(eval5(&five("Ah 2d 3c 4s 5h"))), HAND_STRAIGHT);
        assert_eq!(primary_rank(eval5(&five("Ah 2d 3c 4s 5h"))), 5);
        assert!(eval5(&five("2h 2d 2c 3s 3h")) > eval5(&five("Ah Kh 9h 5h 3h")));
        assert!(eval3(&three("2h 2d 2c")) > eval3(&three("Ah Ad Kc")));
        assert_eq!(
            category(eval5(&five("Ah Kh Qh Jh Th"))),
            HAND_STRAIGHT_FLUSH
        );
    }

    #[test]
    fn royalties_are_the_regular_golden_values() {
        assert_eq!(top_royalty(eval3(&three("6h 6s 4d"))), 1);
        assert_eq!(top_royalty(eval3(&three("Ah As Ad"))), 22);
        assert_eq!(top_royalty(eval3(&three("2h 2s 2d"))), 10);
        assert_eq!(top_royalty(eval3(&three("Ah As 4d"))), 9);
        assert_eq!(top_royalty(eval3(&three("5h 5s 4d"))), 0);
        assert_eq!(middle_royalty(eval5(&five("Ah Ad Ac Ks Qh"))), 2);
        assert_eq!(middle_royalty(eval5(&five("Ah Kh Qh Jh Th"))), 50);
        assert_eq!(bottom_royalty(eval5(&five("Ah Kh Qh Jh Th"))), 25);
        assert_eq!(bottom_royalty(eval5(&five("Ah Ad Ac As Qh"))), 10);
        assert_eq!(bottom_royalty(eval5(&five("Ah Ad Ac Ks Kh"))), 6);
    }

    #[test]
    fn fl_entry_and_stay_conditions_are_the_repo_rules() {
        assert_eq!(fl_entry_from_top(eval3(&three("Qh Qs 4d"))), Some("qq"));
        assert_eq!(fl_entry_from_top(eval3(&three("Kh Ks 4d"))), Some("kk"));
        assert_eq!(fl_entry_from_top(eval3(&three("Ah As 4d"))), Some("aa"));
        assert_eq!(fl_entry_from_top(eval3(&three("2h 2s 2d"))), Some("trips"));
        assert_eq!(fl_entry_from_top(eval3(&three("Jh Js 4d"))), None);

        // Stay: trips on top, or quads+ on the bottom. A QQ top does not stay.
        assert_eq!(
            fl_stay(eval3(&three("2h 2s 2d")), eval5(&five("Ah Kh 9d 5c 3s"))),
            Some("stay_top_trips")
        );
        assert_eq!(
            fl_stay(eval3(&three("Qh Qs 4d")), eval5(&five("Ah Ad Ac As Qc"))),
            Some("stay_bottom_quads_plus")
        );
        assert_eq!(
            fl_stay(eval3(&three("Qh Qs 4d")), eval5(&five("Ah Kh Qh Jh Th"))),
            Some("stay_bottom_quads_plus")
        );
        assert_eq!(
            fl_stay(eval3(&three("Qh Qs 4d")), eval5(&five("Ah Ad Ac Ks Kh"))),
            None
        );
    }

    #[test]
    fn packed_order_treats_a_short_tie_list_as_a_zero_padded_prefix() {
        // Top A-K-Q high vs middle A-K-Q-J-9 high: the engine compares
        // vec![14,13,12] < vec![14,13,12,11,9] because the shorter list is a
        // prefix. The packed keys must agree.
        let top = eval3(&three("Ah Kd Qc"));
        let middle = eval5(&five("As Ks Qs Js 9h"));
        assert_eq!(category(top), HAND_HIGH);
        assert_eq!(category(middle), HAND_HIGH);
        assert!(top < middle);
        assert!(is_foul(middle, top, middle));
    }
}
