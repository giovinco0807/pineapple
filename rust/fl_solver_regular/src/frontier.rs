//! Adaptive (best-response) Fantasyland opponent.
//!
//! # The rule this exists for
//!
//! The Fantasyland player does not commit blind. They watch the normal player's
//! progressive placement and set their 13-from-14 only after the opponent's
//! board is complete. So the Fantasyland side is a *best response* to a concrete
//! hero board, not a draw from a fixed distribution. Scoring a hero candidate
//! against a statically-solved Fantasyland board is therefore optimistic for the
//! hero: it lets the hero beat an opponent who was not allowed to react.
//!
//! # Why a frontier, and why it is exact
//!
//! Best-responding by rescanning all 1,009,008 arrangements per hero board is
//! far too slow. It is also unnecessary. Write the Fantasyland player's score
//! against a fixed hero board `H` as
//!
//! ```text
//! f(A, H) = g(s_top, s_mid, s_bot) + static(A)
//!   where s_row = sign(row_key(A) - row_key(H))  in {-1, 0, +1}
//!         g(s)  = sum(s) + 3 * signum(sum(s)) if |sum(s)| == 3 else sum(s)
//!         static(A) = royalty(A) + fl_ev * stay(A)
//! ```
//!
//! `g` is the 1-6 line term *including the scoop bonus*, and it is monotone
//! non-decreasing in each of its three arguments -- [`scoop_aware_line_is_monotone`]
//! checks that over all 27 sign triples rather than asserting it. Each `s_row`
//! is itself monotone non-decreasing in `row_key(A)` for fixed `H`. Therefore
//! `f(., H)` is monotone non-decreasing in all four of
//! `(top_key, mid_key, bot_key, static)`, **for every `H` simultaneously**.
//!
//! Consequently, if `A'` dominates `A` -- all four components at least as large
//! -- then `f(A', H) >= f(A, H)` for every hero board. A dominated arrangement
//! can never be the unique best response, so the maximum is always attained on
//! the non-dominated frontier. Scanning the frontier is exact, not approximate.
//!
//! `tests/frontier_exactness.rs` pins that against brute force: for random
//! (deal, hero board) pairs it compares the frontier argmax score with the
//! argmax over all 1,009,008 arrangements and requires bit-identical values.

use crate::eval::{
    bottom_royalty, category, eval3, eval5, middle_royalty, top_royalty, HandKey,
    HAND_QUADS, HAND_TRIPS,
};
use serde::{Deserialize, Serialize};

/// One arrangement, reduced to everything a best response can depend on.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrontierEntry {
    pub top_key: HandKey,
    pub middle_key: HandKey,
    pub bottom_key: HandKey,
    /// `royalty + fl_ev * stay`, the part of the score independent of the hero.
    pub static_value: f64,
    pub total_royalty: i32,
    pub stays: bool,
}

/// The 1-6 line term with the scoop bonus, from three row signs.
#[inline(always)]
pub fn scoop_aware_line(top: i32, middle: i32, bottom: i32) -> i32 {
    let sum = top + middle + bottom;
    if sum == 3 {
        6
    } else if sum == -3 {
        -6
    } else {
        sum
    }
}

#[inline(always)]
fn sign_of(own: HandKey, other: HandKey) -> i32 {
    match own.cmp(&other) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    }
}

impl FrontierEntry {
    /// Score of this arrangement against a concrete hero board, from the
    /// Fantasyland side, dropping the hero-only terms that are constant here.
    #[inline(always)]
    pub fn score_against(&self, hero_top: HandKey, hero_middle: HandKey, hero_bottom: HandKey) -> f64 {
        let lines = scoop_aware_line(
            sign_of(self.top_key, hero_top),
            sign_of(self.middle_key, hero_middle),
            sign_of(self.bottom_key, hero_bottom),
        );
        lines as f64 + self.static_value
    }

    #[inline(always)]
    fn dominates(&self, other: &Self) -> bool {
        self.top_key >= other.top_key
            && self.middle_key >= other.middle_key
            && self.bottom_key >= other.bottom_key
            && self.static_value >= other.static_value
    }
}

/// Reduce a candidate set to its non-dominated members.
///
/// Sorted by `static_value` descending first, so when a candidate is examined
/// every already-accepted entry has `static_value >= ` its own and only the
/// three row keys need comparing.
fn sweep(mut candidates: Vec<FrontierEntry>) -> Vec<FrontierEntry> {
    candidates.sort_by(|left, right| {
        right
            .static_value
            .partial_cmp(&left.static_value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.bottom_key.cmp(&left.bottom_key))
            .then_with(|| right.middle_key.cmp(&left.middle_key))
            .then_with(|| right.top_key.cmp(&left.top_key))
    });
    let mut accepted: Vec<FrontierEntry> = Vec::new();
    'outer: for candidate in candidates {
        for kept in accepted.iter() {
            if *kept == candidate {
                continue 'outer; // exact duplicate, already represented
            }
            if kept.dominates(&candidate) {
                continue 'outer;
            }
        }
        accepted.push(candidate);
    }
    accepted
}

/// Build the best-response frontier for a 14-card Fantasyland deal.
///
/// Two-stage to keep the intermediate sets small: a local frontier per bottom
/// row (over that bottom's 126 middles x 4 tops), then one global sweep over
/// the union. The result is identical to sweeping all 1,009,008 arrangements at
/// once -- domination is transitive, so discarding a locally dominated
/// arrangement can never discard a globally non-dominated one.
pub fn build_frontier(hand: &[u8; 14], fl_ev: f64) -> Vec<FrontierEntry> {
    // Rank each 3- and 5-card subset once (2,366 evaluations) and index them by
    // position mask, exactly as the solver's `prepare` does. Evaluating inside
    // the million-leaf loop instead costs three hand rankings per arrangement
    // and dominated the build time.
    let mut five_key = [0_u32; 2002];
    let mut five_middle_royalty = [0_i32; 2002];
    let mut five_bottom_royalty = [0_i32; 2002];
    let mut five_stay = [false; 2002];
    for (index, positions) in FIVE_OF_FOURTEEN.iter().enumerate() {
        let key = eval5(&pick5(hand, positions));
        five_key[index] = key;
        five_middle_royalty[index] = middle_royalty(key);
        five_bottom_royalty[index] = bottom_royalty(key);
        five_stay[index] = category(key) >= HAND_QUADS;
    }
    let mut three_key = [0_u32; 364];
    let mut three_royalty = [0_i32; 364];
    let mut three_trips = [false; 364];
    for (index, positions) in THREE_OF_FOURTEEN.iter().enumerate() {
        let key = eval3(&[
            hand[positions[0] as usize],
            hand[positions[1] as usize],
            hand[positions[2] as usize],
        ]);
        three_key[index] = key;
        three_royalty[index] = top_royalty(key);
        three_trips[index] = category(key) == HAND_TRIPS;
    }

    let mut pooled: Vec<FrontierEntry> = Vec::with_capacity(4096);
    let mut local: Vec<FrontierEntry> = Vec::with_capacity(512);

    for (bottom_index, bottom_positions) in FIVE_OF_FOURTEEN.iter().enumerate() {
        let bottom_key = five_key[bottom_index];
        let bottom_royalty_value = five_bottom_royalty[bottom_index];
        let bottom_stay = five_stay[bottom_index];

        let bottom_mask = bottom_positions
            .iter()
            .fold(0_u16, |mask, position| mask | (1 << position));
        let mut rest = [0_u8; 9];
        let mut slot = 0;
        for position in 0..14_u8 {
            if bottom_mask & (1 << position) == 0 {
                rest[slot] = position;
                slot += 1;
            }
        }

        local.clear();
        for (middle_picks, top_four) in NINE_CHOOSE_FIVE.iter() {
            let middle_mask = middle_picks
                .iter()
                .fold(0_u16, |mask, pick| mask | (1 << rest[*pick as usize]));
            let middle_index = FIVE_INDEX[middle_mask as usize] as usize;
            let middle_key = five_key[middle_index];
            if middle_key > bottom_key {
                continue;
            }
            let middle_royalty_value = five_middle_royalty[middle_index];
            for (top_picks, _discard) in FOUR_CHOOSE_THREE.iter() {
                let top_mask = top_picks.iter().fold(0_u16, |mask, pick| {
                    mask | (1 << rest[top_four[*pick as usize] as usize])
                });
                let top_index = THREE_INDEX[top_mask as usize] as usize;
                let top_key = three_key[top_index];
                if top_key > middle_key {
                    continue;
                }
                let stays = three_trips[top_index] || bottom_stay;
                let royalty =
                    three_royalty[top_index] + middle_royalty_value + bottom_royalty_value;
                local.push(FrontierEntry {
                    top_key,
                    middle_key,
                    bottom_key,
                    static_value: royalty as f64 + if stays { fl_ev } else { 0.0 },
                    total_royalty: royalty,
                    stays,
                });
            }
        }
        if !local.is_empty() {
            pooled.extend(sweep(std::mem::take(&mut local)));
            local = Vec::with_capacity(512);
        }
    }
    sweep(pooled)
}

/// Best response of a Fantasyland frontier to a concrete hero board.
pub fn best_response(
    frontier: &[FrontierEntry],
    hero_top: HandKey,
    hero_middle: HandKey,
    hero_bottom: HandKey,
) -> (f64, usize) {
    let mut best = f64::NEG_INFINITY;
    let mut best_index = 0;
    for (index, entry) in frontier.iter().enumerate() {
        let value = entry.score_against(hero_top, hero_middle, hero_bottom);
        if value > best {
            best = value;
            best_index = index;
        }
    }
    (best, best_index)
}

/// Brute force over every arrangement, for the exactness test only.
pub fn best_response_brute_force(
    hand: &[u8; 14],
    fl_ev: f64,
    hero_top: HandKey,
    hero_middle: HandKey,
    hero_bottom: HandKey,
) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for bottom_positions in FIVE_OF_FOURTEEN.iter() {
        let bottom_cards = pick5(hand, bottom_positions);
        let bottom_key = eval5(&bottom_cards);
        let bottom_royalty_value = bottom_royalty(bottom_key);
        let bottom_stay = category(bottom_key) >= HAND_QUADS;
        let mut rest = [0_u8; 9];
        let mut slot = 0;
        for position in 0..14_u8 {
            if !bottom_positions.contains(&position) {
                rest[slot] = position;
                slot += 1;
            }
        }
        for (middle_picks, top_four) in NINE_CHOOSE_FIVE.iter() {
            let middle_cards = [
                hand[rest[middle_picks[0] as usize] as usize],
                hand[rest[middle_picks[1] as usize] as usize],
                hand[rest[middle_picks[2] as usize] as usize],
                hand[rest[middle_picks[3] as usize] as usize],
                hand[rest[middle_picks[4] as usize] as usize],
            ];
            let middle_key = eval5(&middle_cards);
            if middle_key > bottom_key {
                continue;
            }
            let middle_royalty_value = middle_royalty(middle_key);
            for (top_picks, _discard) in FOUR_CHOOSE_THREE.iter() {
                let top_cards = [
                    hand[rest[top_four[top_picks[0] as usize] as usize] as usize],
                    hand[rest[top_four[top_picks[1] as usize] as usize] as usize],
                    hand[rest[top_four[top_picks[2] as usize] as usize] as usize],
                ];
                let top_key = eval3(&top_cards);
                if top_key > middle_key {
                    continue;
                }
                let stays = category(top_key) == HAND_TRIPS || bottom_stay;
                let royalty =
                    top_royalty(top_key) + middle_royalty_value + bottom_royalty_value;
                let entry = FrontierEntry {
                    top_key,
                    middle_key,
                    bottom_key,
                    static_value: royalty as f64 + if stays { fl_ev } else { 0.0 },
                    total_royalty: royalty,
                    stays,
                };
                let value = entry.score_against(hero_top, hero_middle, hero_bottom);
                if value > best {
                    best = value;
                }
            }
        }
    }
    best
}

#[inline(always)]
fn pick5(hand: &[u8; 14], positions: &[u8; 5]) -> [u8; 5] {
    [
        hand[positions[0] as usize],
        hand[positions[1] as usize],
        hand[positions[2] as usize],
        hand[positions[3] as usize],
        hand[positions[4] as usize],
    ]
}

// Position tables, built once.
static FIVE_OF_FOURTEEN: std::sync::LazyLock<Vec<[u8; 5]>> = std::sync::LazyLock::new(|| {
    let mut out = Vec::with_capacity(2002);
    for a in 0..10_u8 {
        for b in (a + 1)..11 {
            for c in (b + 1)..12 {
                for d in (c + 1)..13 {
                    for e in (d + 1)..14 {
                        out.push([a, b, c, d, e]);
                    }
                }
            }
        }
    }
    out
});

static THREE_OF_FOURTEEN: std::sync::LazyLock<Vec<[u8; 3]>> = std::sync::LazyLock::new(|| {
    let mut out = Vec::with_capacity(364);
    for a in 0..12_u8 {
        for b in (a + 1)..13 {
            for c in (b + 1)..14 {
                out.push([a, b, c]);
            }
        }
    }
    out
});

/// Position-mask to combination index. Depends only on positions, so it is
/// built once for the process rather than once per deal.
static FIVE_INDEX: std::sync::LazyLock<Vec<u16>> = std::sync::LazyLock::new(|| {
    let mut out = vec![u16::MAX; 1 << 14];
    for (index, positions) in FIVE_OF_FOURTEEN.iter().enumerate() {
        let mask = positions
            .iter()
            .fold(0_u16, |mask, position| mask | (1 << position));
        out[mask as usize] = index as u16;
    }
    out
});

static THREE_INDEX: std::sync::LazyLock<Vec<u16>> = std::sync::LazyLock::new(|| {
    let mut out = vec![u16::MAX; 1 << 14];
    for (index, positions) in THREE_OF_FOURTEEN.iter().enumerate() {
        let mask = positions
            .iter()
            .fold(0_u16, |mask, position| mask | (1 << position));
        out[mask as usize] = index as u16;
    }
    out
});

static NINE_CHOOSE_FIVE: std::sync::LazyLock<Vec<([u8; 5], [u8; 4])>> =
    std::sync::LazyLock::new(|| {
        let mut out = Vec::with_capacity(126);
        for a in 0..5_u8 {
            for b in (a + 1)..6 {
                for c in (b + 1)..7 {
                    for d in (c + 1)..8 {
                        for e in (d + 1)..9 {
                            let picked = [a, b, c, d, e];
                            let mut rest = [0_u8; 4];
                            let mut slot = 0;
                            for index in 0..9_u8 {
                                if !picked.contains(&index) {
                                    rest[slot] = index;
                                    slot += 1;
                                }
                            }
                            out.push((picked, rest));
                        }
                    }
                }
            }
        }
        out
    });

static FOUR_CHOOSE_THREE: std::sync::LazyLock<Vec<([u8; 3], u8)>> =
    std::sync::LazyLock::new(|| {
        let mut out = Vec::with_capacity(4);
        for a in 0..2_u8 {
            for b in (a + 1)..3 {
                for c in (b + 1)..4 {
                    let picked = [a, b, c];
                    let left = (0..4_u8).find(|index| !picked.contains(index)).unwrap();
                    out.push((picked, left));
                }
            }
        }
        out
    });

#[cfg(test)]
mod tests {
    use super::*;

    /// The monotonicity the whole frontier argument rests on, checked over
    /// every sign triple rather than asserted. The scoop bonus is inside `g`,
    /// which is exactly why a scoop-aware maximisation still admits the
    /// frontier reduction.
    #[test]
    fn scoop_aware_line_is_monotone() {
        for top in [-1, 0, 1] {
            for middle in [-1, 0, 1] {
                for bottom in [-1, 0, 1] {
                    let base = scoop_aware_line(top, middle, bottom);
                    if top < 1 {
                        assert!(scoop_aware_line(top + 1, middle, bottom) >= base);
                    }
                    if middle < 1 {
                        assert!(scoop_aware_line(top, middle + 1, bottom) >= base);
                    }
                    if bottom < 1 {
                        assert!(scoop_aware_line(top, middle + 1 - 1, bottom + 1) >= base);
                    }
                }
            }
        }
        // The scoop is present, not merely the line sum.
        assert_eq!(scoop_aware_line(1, 1, 1), 6);
        assert_eq!(scoop_aware_line(-1, -1, -1), -6);
        assert_eq!(scoop_aware_line(1, 1, -1), 1);
    }

    #[test]
    fn frontier_members_are_mutually_non_dominated() {
        let hand: [u8; 14] = crate::cards::parse_cards(
            "Ah Kh Qh Jh Th 9h 8h 7h 2c 3d 4s 5c 6d 8s",
        )
        .unwrap()
        .try_into()
        .unwrap();
        let frontier = build_frontier(&hand, 9.109);
        assert!(!frontier.is_empty());
        for (i, a) in frontier.iter().enumerate() {
            for (j, b) in frontier.iter().enumerate() {
                if i != j {
                    assert!(!b.dominates(a) || a == b, "entry {j} dominates {i}");
                }
            }
        }
    }
}
