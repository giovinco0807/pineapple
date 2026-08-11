//! Exact 14-card Fantasyland placement search.
//!
//! The decision is: which single card to discard, and how to split the other
//! thirteen into rows of 3/5/5. Those are the same choice -- picking a top of
//! three, a middle of five and a bottom of five out of fourteen leaves exactly
//! one card over -- so the space is
//! `C(14,3) * C(11,5) * C(6,5) = 364 * 462 * 6 = 1_009_008` arrangements.
//!
//! The search enumerates that space in a different order than the count above
//! suggests: bottom first (`C(14,5) = 2002`), then middle from the nine cards
//! left (`C(9,5) = 126`), then top from the remaining four (`C(4,3) = 4`),
//! which is the same `2002 * 126 * 4 = 1_009_008` leaves but lets the foul
//! checks prune whole subtrees. Every 3- and 5-card subset is ranked once per
//! deal (2366 evaluations) and reused across the leaves that contain it.
//!
//! Exactness is verified against `solve_brute_force`, an independent
//! full-enumeration implementation with no pruning and no shared combination
//! cache. Both call [`arrangement_score`] so the floating-point result is
//! bit-identical and the argmax comparison is exact rather than tolerant.

use crate::cards::mask_of;
use crate::eval::{
    bottom_royalty, category, eval3, eval5, fl_entry_from_top, fl_stay, is_foul, middle_royalty,
    top_royalty, HandKey, HAND_QUADS, HAND_TRIPS,
};
use crate::objective::{ObjectiveConfig, ObjectiveKind, ReferenceSet, REF_WORD_BITS};
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// One solved Fantasyland placement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Solution {
    pub top: [u8; 3],
    pub middle: [u8; 5],
    pub bottom: [u8; 5],
    pub discard: u8,
    pub top_key: HandKey,
    pub middle_key: HandKey,
    pub bottom_key: HandKey,
    pub top_royalty: i32,
    pub middle_royalty: i32,
    pub bottom_royalty: i32,
    pub total_royalty: i32,
    pub stay: Option<String>,
    pub score: f64,
    /// Expected `line + scoop` against the reference set, already scaled by
    /// `1 - hero_foul_prob`. Zero under `pure_v1`.
    pub line_equity: f64,
}

impl Solution {
    pub fn stays(&self) -> bool {
        self.stay.is_some()
    }

    /// Deterministic identity of the arrangement, independent of search order.
    pub fn arrangement_key(&self) -> (u64, u64, u64) {
        (mask_of(&self.top), mask_of(&self.middle), mask_of(&self.bottom))
    }

    /// Fantasyland *entry* type of this board under ordinary QQ+ rules. Note a
    /// board reached from Fantasyland continues on `stay`, not on entry; this
    /// exists so the same board can be scored from a normal hero's viewpoint.
    pub fn fl_entry(&self) -> Option<&'static str> {
        fl_entry_from_top(self.top_key)
    }
}

/// The one place an arrangement's score is computed.
///
/// Both the pruned search and the brute-force reference route through this
/// function so that "exact" can mean bit-identical rather than within-epsilon.
/// `line_sum` is the summed per-reference line total and `scoop_diff` the
/// scooped-minus-scooped-against count, both over the reference set.
#[inline(always)]
pub fn arrangement_score(
    total_royalty: i32,
    stays: bool,
    line_sum: i32,
    scoop_diff: i32,
    fl_ev_stay: f64,
    line_weight: f64,
    inverse_sample_count: f64,
) -> (f64, f64) {
    let mut score = total_royalty as f64;
    if stays {
        score += fl_ev_stay;
    }
    let line_equity = line_weight * ((line_sum + 3 * scoop_diff) as f64) * inverse_sample_count;
    score += line_equity;
    (score, line_equity)
}

/// Strict total order over arrangements: score, then royalty, then the
/// arrangement masks ascending. The mask tiebreak makes the argmax unique, so
/// two independent enumerations of the same space must return the same
/// arrangement rather than an arbitrary member of a tie class.
#[inline(always)]
fn beats(
    score: f64,
    royalty: i32,
    key: (u64, u64, u64),
    best_score: f64,
    best_royalty: i32,
    best_key: (u64, u64, u64),
) -> bool {
    if score != best_score {
        return score > best_score;
    }
    if royalty != best_royalty {
        return royalty > best_royalty;
    }
    key < best_key
}

// ---------------------------------------------------------------------------
// Combination tables. These depend only on positions, never on cards, so they
// are built once for the process rather than once per deal.
// ---------------------------------------------------------------------------

struct Tables {
    /// `C(14,5) = 2002` position quintuples, and mask -> index.
    five: Vec<[u8; 5]>,
    five_index: Vec<u16>,
    /// `C(14,3) = 364` position triples, and mask -> index.
    three: Vec<[u8; 3]>,
    three_index: Vec<u16>,
    /// `C(9,5) = 126` picks out of nine, with the four not picked.
    nine_choose_five: Vec<([u8; 5], [u8; 4])>,
    /// `C(4,3) = 4` picks out of four, with the one not picked.
    four_choose_three: Vec<([u8; 3], u8)>,
}

static TABLES: LazyLock<Tables> = LazyLock::new(|| {
    let mut five = Vec::with_capacity(2002);
    let mut five_index = vec![u16::MAX; 1 << 14];
    for a in 0..10_u8 {
        for b in (a + 1)..11 {
            for c in (b + 1)..12 {
                for d in (c + 1)..13 {
                    for e in (d + 1)..14 {
                        let mask = (1_u16 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e);
                        five_index[mask as usize] = five.len() as u16;
                        five.push([a, b, c, d, e]);
                    }
                }
            }
        }
    }

    let mut three = Vec::with_capacity(364);
    let mut three_index = vec![u16::MAX; 1 << 14];
    for a in 0..12_u8 {
        for b in (a + 1)..13 {
            for c in (b + 1)..14 {
                let mask = (1_u16 << a) | (1 << b) | (1 << c);
                three_index[mask as usize] = three.len() as u16;
                three.push([a, b, c]);
            }
        }
    }

    let mut nine_choose_five = Vec::with_capacity(126);
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
                        nine_choose_five.push((picked, rest));
                    }
                }
            }
        }
    }

    let mut four_choose_three = Vec::with_capacity(4);
    for a in 0..2_u8 {
        for b in (a + 1)..3 {
            for c in (b + 1)..4 {
                let picked = [a, b, c];
                let left = (0..4_u8).find(|index| !picked.contains(index)).unwrap();
                four_choose_three.push((picked, left));
            }
        }
    }

    Tables {
        five,
        five_index,
        three,
        three_index,
        nine_choose_five,
        four_choose_three,
    }
});

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Per-deal scratch space. Reuse one `FlSolver` across many deals to avoid
/// reallocating the combination arrays for every solve.
pub struct FlSolver {
    fl_ev_stay: f64,
    line_weight: f64,
    inverse_sample_count: f64,
    words: usize,
    use_line: bool,
    reference_top: Vec<HandKey>,
    reference_middle: Vec<HandKey>,
    reference_bottom: Vec<HandKey>,

    five_key: Vec<HandKey>,
    five_middle_royalty: Vec<i32>,
    five_bottom_royalty: Vec<i32>,
    five_bottom_stay: Vec<bool>,
    /// Card-index bitmask of each 5-combo for this deal. Precomputed so the
    /// argmax tiebreak never allocates inside the million-leaf loop.
    five_card_mask: Vec<u64>,
    three_key: Vec<HandKey>,
    three_royalty: Vec<i32>,
    three_stay: Vec<bool>,
    three_card_mask: Vec<u64>,

    // Packed per-reference win/loss bitsets, `words` machine words per combo.
    win_middle: Vec<u64>,
    loss_middle: Vec<u64>,
    win_bottom: Vec<u64>,
    loss_bottom: Vec<u64>,
    win_top: Vec<u64>,
    loss_top: Vec<u64>,
    net_middle: Vec<i32>,
    net_bottom: Vec<i32>,
    net_top: Vec<i32>,

    /// Best middle royalty available anywhere in *this deal*. Most deals
    /// cannot reach the game-wide 50-point maximum, so bounding on the deal
    /// prunes far more than bounding on the rules. The top row is bounded
    /// tighter still, from the exact cards left for it.
    deal_max_middle_royalty: i32,

    /// Leaves visited by the most recent solve, for the pruning report.
    pub last_leaves: u64,
    /// Leaves the space contains, for the same report.
    pub last_space: u64,
}

const FIVE_COMBOS: usize = 2002;
const THREE_COMBOS: usize = 364;
/// `line + 3 * scoop` is bounded by `3 + 3`.
const MAX_LINE_UNIT: f64 = 6.0;

impl FlSolver {
    pub fn new(config: &ObjectiveConfig) -> Result<Self, String> {
        config.validate()?;
        let use_line = config.kind == ObjectiveKind::LineEquityV1;
        let (words, sample_count, reference) = match (&config.reference, use_line) {
            (Some(reference), true) => (reference.words(), reference.sample_count, Some(reference)),
            _ => (0, 1, None),
        };
        let empty: Vec<HandKey> = Vec::new();
        Ok(Self {
            fl_ev_stay: config.fl_ev_stay,
            line_weight: if use_line {
                1.0 - config.hero_foul_prob
            } else {
                0.0
            },
            inverse_sample_count: 1.0 / sample_count as f64,
            words,
            use_line,
            reference_top: reference.map(|value| value.top.clone()).unwrap_or(empty.clone()),
            reference_middle: reference.map(|value| value.middle.clone()).unwrap_or(empty.clone()),
            reference_bottom: reference.map(|value| value.bottom.clone()).unwrap_or(empty),

            five_key: vec![0; FIVE_COMBOS],
            five_middle_royalty: vec![0; FIVE_COMBOS],
            five_bottom_royalty: vec![0; FIVE_COMBOS],
            five_bottom_stay: vec![false; FIVE_COMBOS],
            five_card_mask: vec![0; FIVE_COMBOS],
            three_key: vec![0; THREE_COMBOS],
            three_royalty: vec![0; THREE_COMBOS],
            three_stay: vec![false; THREE_COMBOS],
            three_card_mask: vec![0; THREE_COMBOS],

            win_middle: vec![0; FIVE_COMBOS * words],
            loss_middle: vec![0; FIVE_COMBOS * words],
            win_bottom: vec![0; FIVE_COMBOS * words],
            loss_bottom: vec![0; FIVE_COMBOS * words],
            win_top: vec![0; THREE_COMBOS * words],
            loss_top: vec![0; THREE_COMBOS * words],
            net_middle: vec![0; FIVE_COMBOS],
            net_bottom: vec![0; FIVE_COMBOS],
            net_top: vec![0; THREE_COMBOS],

            deal_max_middle_royalty: 0,

            last_leaves: 0,
            last_space: 0,
        })
    }

    fn prepare(&mut self, hand: &[u8; 14]) {
        let tables = &*TABLES;
        self.deal_max_middle_royalty = 0;

        for (index, positions) in tables.five.iter().enumerate() {
            let cards = [
                hand[positions[0] as usize],
                hand[positions[1] as usize],
                hand[positions[2] as usize],
                hand[positions[3] as usize],
                hand[positions[4] as usize],
            ];
            let key = eval5(&cards);
            self.five_key[index] = key;
            self.five_middle_royalty[index] = middle_royalty(key);
            self.five_bottom_royalty[index] = bottom_royalty(key);
            self.five_bottom_stay[index] = category(key) >= HAND_QUADS;
            self.five_card_mask[index] = mask_of(&cards);
            self.deal_max_middle_royalty =
                self.deal_max_middle_royalty.max(self.five_middle_royalty[index]);
        }
        for (index, positions) in tables.three.iter().enumerate() {
            let cards = [
                hand[positions[0] as usize],
                hand[positions[1] as usize],
                hand[positions[2] as usize],
            ];
            let key = eval3(&cards);
            self.three_key[index] = key;
            self.three_royalty[index] = top_royalty(key);
            self.three_stay[index] = category(key) == HAND_TRIPS;
            self.three_card_mask[index] = mask_of(&cards);
        }

        if !self.use_line {
            return;
        }
        let words = self.words;
        self.win_middle.fill(0);
        self.loss_middle.fill(0);
        self.win_bottom.fill(0);
        self.loss_bottom.fill(0);
        self.win_top.fill(0);
        self.loss_top.fill(0);
        for index in 0..FIVE_COMBOS {
            let key = self.five_key[index];
            let mut net_middle = 0_i32;
            let mut net_bottom = 0_i32;
            for (sample, reference) in self.reference_middle.iter().enumerate() {
                if key > *reference {
                    self.win_middle[index * words + sample / REF_WORD_BITS] |=
                        1_u64 << (sample % REF_WORD_BITS);
                    net_middle += 1;
                } else if key < *reference {
                    self.loss_middle[index * words + sample / REF_WORD_BITS] |=
                        1_u64 << (sample % REF_WORD_BITS);
                    net_middle -= 1;
                }
            }
            for (sample, reference) in self.reference_bottom.iter().enumerate() {
                if key > *reference {
                    self.win_bottom[index * words + sample / REF_WORD_BITS] |=
                        1_u64 << (sample % REF_WORD_BITS);
                    net_bottom += 1;
                } else if key < *reference {
                    self.loss_bottom[index * words + sample / REF_WORD_BITS] |=
                        1_u64 << (sample % REF_WORD_BITS);
                    net_bottom -= 1;
                }
            }
            self.net_middle[index] = net_middle;
            self.net_bottom[index] = net_bottom;
        }
        for index in 0..THREE_COMBOS {
            let key = self.three_key[index];
            let mut net_top = 0_i32;
            for (sample, reference) in self.reference_top.iter().enumerate() {
                if key > *reference {
                    self.win_top[index * words + sample / REF_WORD_BITS] |=
                        1_u64 << (sample % REF_WORD_BITS);
                    net_top += 1;
                } else if key < *reference {
                    self.loss_top[index * words + sample / REF_WORD_BITS] |=
                        1_u64 << (sample % REF_WORD_BITS);
                    net_top -= 1;
                }
            }
            self.net_top[index] = net_top;
        }
    }

    #[inline(always)]
    fn scoop_difference(&self, top: usize, middle: usize, bottom: usize) -> i32 {
        let words = self.words;
        let mut scooped = 0_i32;
        let mut scooped_against = 0_i32;
        for word in 0..words {
            let win = self.win_top[top * words + word]
                & self.win_middle[middle * words + word]
                & self.win_bottom[bottom * words + word];
            let loss = self.loss_top[top * words + word]
                & self.loss_middle[middle * words + word]
                & self.loss_bottom[bottom * words + word];
            scooped += win.count_ones() as i32;
            scooped_against += loss.count_ones() as i32;
        }
        scooped - scooped_against
    }

    /// Exact search. Returns `None` only if the hand admits no legal 3/5/5
    /// arrangement, which cannot happen with fourteen cards.
    pub fn solve(&mut self, hand: &[u8; 14]) -> Option<Solution> {
        self.prepare(hand);
        let tables = &*TABLES;

        let mut best_score = f64::NEG_INFINITY;
        let mut best_royalty = i32::MIN;
        let mut best_key = (u64::MAX, u64::MAX, u64::MAX);
        let mut best: Option<(usize, usize, usize, u8, i32, i32)> = None;
        let mut leaves = 0_u64;

        // Strongest-looking bottoms first: a good incumbent early is what makes
        // the branch bound bite. Ranked by what the bottom itself is worth,
        // royalty plus the stay bonus it can unlock on its own.
        let stay_bonus_bound = self.fl_ev_stay.max(0.0);
        let bottom_rank = |index: usize| -> f64 {
            self.five_bottom_royalty[index] as f64
                + if self.five_bottom_stay[index] {
                    stay_bonus_bound
                } else {
                    0.0
                }
        };
        let mut bottom_order: Vec<u16> = (0..FIVE_COMBOS as u16).collect();
        bottom_order.sort_unstable_by(|left, right| {
            bottom_rank(*right as usize)
                .partial_cmp(&bottom_rank(*left as usize))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let line_bound = self.line_weight * MAX_LINE_UNIT;

        for bottom_index in bottom_order.iter().map(|value| *value as usize) {
            let bottom_positions = &tables.five[bottom_index];
            let bottom_key = self.five_key[bottom_index];
            let bottom_royalty_value = self.five_bottom_royalty[bottom_index];
            let bottom_stay = self.five_bottom_stay[bottom_index];

            // Positions not used by the bottom, ascending.
            let bottom_mask = bottom_positions
                .iter()
                .fold(0_u16, |mask, position| mask | (1 << position));
            let mut rest_nine = [0_u8; 9];
            let mut rest_nine_ranks = [0_u8; 9];
            let mut slot = 0;
            for position in 0..14_u8 {
                if bottom_mask & (1 << position) == 0 {
                    rest_nine[slot] = position;
                    rest_nine_ranks[slot] = crate::cards::rank_of(hand[position as usize]);
                    slot += 1;
                }
            }

            // Bound before choosing the middle. Strictly-less, not
            // less-or-equal: a subtree whose bound equals the incumbent can
            // still hold an arrangement that ties on score and wins the mask
            // tiebreak, and the argmax has to be the same one brute force
            // finds.
            let (rest_top_royalty, rest_trips) = best_top_from_ranks(&rest_nine_ranks);
            if (bottom_royalty_value + self.deal_max_middle_royalty + rest_top_royalty) as f64
                + if bottom_stay || rest_trips {
                    stay_bonus_bound
                } else {
                    0.0
                }
                + line_bound
                < best_score
            {
                continue;
            }

            for (middle_picks, top_four) in tables.nine_choose_five.iter() {
                let middle_mask = middle_picks.iter().fold(0_u16, |mask, pick| {
                    mask | (1 << rest_nine[*pick as usize])
                });
                let middle_index = tables.five_index[middle_mask as usize] as usize;
                let middle_key = self.five_key[middle_index];
                if middle_key > bottom_key {
                    continue;
                }
                let middle_royalty_value = self.five_middle_royalty[middle_index];

                // The top can only come from the four cards the middle left, so
                // bound it on exactly those four ranks. This is what makes the
                // search prune: the deal-wide maximum is nearly always
                // unreachable here.
                let four_ranks = [
                    rest_nine_ranks[top_four[0] as usize],
                    rest_nine_ranks[top_four[1] as usize],
                    rest_nine_ranks[top_four[2] as usize],
                    rest_nine_ranks[top_four[3] as usize],
                ];
                let (best_top_royalty, top_trips_possible) = best_top_from_ranks(&four_ranks);
                let prefix_bound = (bottom_royalty_value
                    + middle_royalty_value
                    + best_top_royalty) as f64
                    + if bottom_stay || top_trips_possible {
                        stay_bonus_bound
                    } else {
                        0.0
                    }
                    + line_bound;
                if prefix_bound < best_score {
                    continue;
                }

                for (top_picks, discard_pick) in tables.four_choose_three.iter() {
                    let top_mask = top_picks.iter().fold(0_u16, |mask, pick| {
                        mask | (1 << rest_nine[top_four[*pick as usize] as usize])
                    });
                    let top_index = tables.three_index[top_mask as usize] as usize;
                    let top_key = self.three_key[top_index];
                    leaves += 1;
                    if top_key > middle_key {
                        continue;
                    }

                    let total_royalty = self.three_royalty[top_index]
                        + middle_royalty_value
                        + bottom_royalty_value;
                    let stays = self.three_stay[top_index] || bottom_stay;
                    let (line_sum, scoop_diff) = if self.use_line {
                        (
                            self.net_top[top_index]
                                + self.net_middle[middle_index]
                                + self.net_bottom[bottom_index],
                            self.scoop_difference(top_index, middle_index, bottom_index),
                        )
                    } else {
                        (0, 0)
                    };
                    let (score, _) = arrangement_score(
                        total_royalty,
                        stays,
                        line_sum,
                        scoop_diff,
                        self.fl_ev_stay,
                        self.line_weight,
                        self.inverse_sample_count,
                    );

                    let discard_position = rest_nine[top_four[*discard_pick as usize] as usize];
                    let key = (
                        self.three_card_mask[top_index],
                        self.five_card_mask[middle_index],
                        self.five_card_mask[bottom_index],
                    );
                    if beats(
                        score,
                        total_royalty,
                        key,
                        best_score,
                        best_royalty,
                        best_key,
                    ) {
                        best_score = score;
                        best_royalty = total_royalty;
                        best_key = key;
                        best = Some((
                            top_index,
                            middle_index,
                            bottom_index,
                            discard_position,
                            line_sum,
                            scoop_diff,
                        ));
                    }
                }
            }
        }

        self.last_leaves = leaves;
        self.last_space = (FIVE_COMBOS * 126 * 4) as u64;

        let (top_index, middle_index, bottom_index, discard_position, line_sum, scoop_diff) = best?;
        Some(self.build_solution(
            hand,
            top_index,
            middle_index,
            bottom_index,
            discard_position,
            line_sum,
            scoop_diff,
        ))
    }

    fn build_solution(
        &self,
        hand: &[u8; 14],
        top_index: usize,
        middle_index: usize,
        bottom_index: usize,
        discard_position: u8,
        line_sum: i32,
        scoop_diff: i32,
    ) -> Solution {
        let tables = &*TABLES;
        let top_positions = tables.three[top_index];
        let middle_positions = tables.five[middle_index];
        let bottom_positions = tables.five[bottom_index];
        let top = [
            hand[top_positions[0] as usize],
            hand[top_positions[1] as usize],
            hand[top_positions[2] as usize],
        ];
        let middle = [
            hand[middle_positions[0] as usize],
            hand[middle_positions[1] as usize],
            hand[middle_positions[2] as usize],
            hand[middle_positions[3] as usize],
            hand[middle_positions[4] as usize],
        ];
        let bottom = [
            hand[bottom_positions[0] as usize],
            hand[bottom_positions[1] as usize],
            hand[bottom_positions[2] as usize],
            hand[bottom_positions[3] as usize],
            hand[bottom_positions[4] as usize],
        ];
        let top_key = self.three_key[top_index];
        let middle_key = self.five_key[middle_index];
        let bottom_key = self.five_key[bottom_index];
        let top_royalty_value = self.three_royalty[top_index];
        let middle_royalty_value = self.five_middle_royalty[middle_index];
        let bottom_royalty_value = self.five_bottom_royalty[bottom_index];
        let total_royalty = top_royalty_value + middle_royalty_value + bottom_royalty_value;
        let stay = fl_stay(top_key, bottom_key);
        let (score, line_equity) = arrangement_score(
            total_royalty,
            stay.is_some(),
            line_sum,
            scoop_diff,
            self.fl_ev_stay,
            self.line_weight,
            self.inverse_sample_count,
        );
        Solution {
            top,
            middle,
            bottom,
            discard: hand[discard_position as usize],
            top_key,
            middle_key,
            bottom_key,
            top_royalty: top_royalty_value,
            middle_royalty: middle_royalty_value,
            bottom_royalty: bottom_royalty_value,
            total_royalty,
            stay: stay.map(str::to_owned),
            score,
            line_equity,
        }
    }
}

/// Best top-row royalty obtainable from any three of these ranks, and whether
/// trips is among them. Top royalty depends only on ranks, so this needs no
/// hand evaluation, and trips always outpays any pair, so the two cases do not
/// have to be compared.
#[inline(always)]
fn best_top_from_ranks(ranks: &[u8]) -> (i32, bool) {
    let mut counts = [0_u8; 15];
    for rank in ranks {
        counts[*rank as usize] += 1;
    }
    let mut trips_rank = 0_u8;
    let mut pair_rank = 0_u8;
    for rank in 2..=14_u8 {
        let count = counts[rank as usize];
        if count >= 3 {
            trips_rank = rank;
        }
        if count >= 2 {
            pair_rank = rank;
        }
    }
    if trips_rank > 0 {
        (10 + i32::from(trips_rank) - 2, true)
    } else if pair_rank >= 6 {
        (i32::from(pair_rank) - 5, false)
    } else {
        (0, false)
    }
}

// ---------------------------------------------------------------------------
// Brute-force reference
// ---------------------------------------------------------------------------

/// Independent full enumeration with no pruning, no combination cache and no
/// incumbent-driven skipping. Every one of the 1,009,008 arrangements is built
/// and ranked from raw cards. This exists purely as the exactness oracle for
/// [`FlSolver::solve`]; it is roughly two orders of magnitude slower.
pub fn solve_brute_force(hand: &[u8; 14], config: &ObjectiveConfig) -> Option<Solution> {
    config.validate().ok()?;
    let use_line = config.kind == ObjectiveKind::LineEquityV1;
    let reference = config.reference.as_ref();
    let sample_count = reference.map(|value| value.sample_count).unwrap_or(1);
    let inverse_sample_count = 1.0 / sample_count as f64;
    let line_weight = if use_line {
        1.0 - config.hero_foul_prob
    } else {
        0.0
    };

    let mut best: Option<Solution> = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_royalty = i32::MIN;
    let mut best_key = (u64::MAX, u64::MAX, u64::MAX);

    for top_a in 0..12_usize {
        for top_b in (top_a + 1)..13 {
            for top_c in (top_b + 1)..14 {
                let top_positions = [top_a, top_b, top_c];
                let top_cards = [hand[top_a], hand[top_b], hand[top_c]];
                let top_key = eval3(&top_cards);
                let remaining: Vec<usize> =
                    (0..14).filter(|slot| !top_positions.contains(slot)).collect();
                for middle_a in 0..7_usize {
                    for middle_b in (middle_a + 1)..8 {
                        for middle_c in (middle_b + 1)..9 {
                            for middle_d in (middle_c + 1)..10 {
                                for middle_e in (middle_d + 1)..11 {
                                    let middle_picks =
                                        [middle_a, middle_b, middle_c, middle_d, middle_e];
                                    let middle_cards = [
                                        hand[remaining[middle_a]],
                                        hand[remaining[middle_b]],
                                        hand[remaining[middle_c]],
                                        hand[remaining[middle_d]],
                                        hand[remaining[middle_e]],
                                    ];
                                    let middle_key = eval5(&middle_cards);
                                    let after_middle: Vec<usize> = (0..11)
                                        .filter(|slot| !middle_picks.contains(slot))
                                        .map(|slot| remaining[slot])
                                        .collect();
                                    for skipped in 0..6_usize {
                                        let mut bottom_cards = [0_u8; 5];
                                        let mut slot = 0;
                                        for (index, position) in after_middle.iter().enumerate() {
                                            if index != skipped {
                                                bottom_cards[slot] = hand[*position];
                                                slot += 1;
                                            }
                                        }
                                        let discard = hand[after_middle[skipped]];
                                        let bottom_key = eval5(&bottom_cards);
                                        if is_foul(top_key, middle_key, bottom_key) {
                                            continue;
                                        }
                                        let top_royalty_value = top_royalty(top_key);
                                        let middle_royalty_value = middle_royalty(middle_key);
                                        let bottom_royalty_value = bottom_royalty(bottom_key);
                                        let total_royalty = top_royalty_value
                                            + middle_royalty_value
                                            + bottom_royalty_value;
                                        let stay = fl_stay(top_key, bottom_key);
                                        let (line_sum, scoop_diff) = match (use_line, reference) {
                                            (true, Some(reference)) => reference_line_terms(
                                                reference, top_key, middle_key, bottom_key,
                                            ),
                                            _ => (0, 0),
                                        };
                                        let (score, line_equity) = arrangement_score(
                                            total_royalty,
                                            stay.is_some(),
                                            line_sum,
                                            scoop_diff,
                                            config.fl_ev_stay,
                                            line_weight,
                                            inverse_sample_count,
                                        );
                                        let key = (
                                            mask_of(&top_cards),
                                            mask_of(&middle_cards),
                                            mask_of(&bottom_cards),
                                        );
                                        if beats(
                                            score,
                                            total_royalty,
                                            key,
                                            best_score,
                                            best_royalty,
                                            best_key,
                                        ) {
                                            best_score = score;
                                            best_royalty = total_royalty;
                                            best_key = key;
                                            best = Some(Solution {
                                                top: top_cards,
                                                middle: middle_cards,
                                                bottom: bottom_cards,
                                                discard,
                                                top_key,
                                                middle_key,
                                                bottom_key,
                                                top_royalty: top_royalty_value,
                                                middle_royalty: middle_royalty_value,
                                                bottom_royalty: bottom_royalty_value,
                                                total_royalty,
                                                stay: stay.map(str::to_owned),
                                                score,
                                                line_equity,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    best
}

fn reference_line_terms(
    reference: &ReferenceSet,
    top_key: HandKey,
    middle_key: HandKey,
    bottom_key: HandKey,
) -> (i32, i32) {
    let mut line_sum = 0_i32;
    let mut scoop_diff = 0_i32;
    for sample in 0..reference.sample_count {
        let top = compare(top_key, reference.top[sample]);
        let middle = compare(middle_key, reference.middle[sample]);
        let bottom = compare(bottom_key, reference.bottom[sample]);
        line_sum += top + middle + bottom;
        if top == 1 && middle == 1 && bottom == 1 {
            scoop_diff += 1;
        } else if top == -1 && middle == -1 && bottom == -1 {
            scoop_diff -= 1;
        }
    }
    (line_sum, scoop_diff)
}

#[inline(always)]
fn compare(own: HandKey, other: HandKey) -> i32 {
    match own.cmp(&other) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::parse_cards;
    use crate::objective::ObjectiveConfig;

    fn pure_config(fl_ev_stay: f64) -> ObjectiveConfig {
        ObjectiveConfig {
            kind: ObjectiveKind::PureV1,
            fl_ev_stay,
            fl_ev_config_path: "test".to_owned(),
            fl_ev_cards: 14,
            hero_foul_prob: 0.0,
            hero_foul_prob_provenance: "test".to_owned(),
            reference: None,
        }
    }

    fn hand_of(text: &str) -> [u8; 14] {
        parse_cards(text).unwrap().try_into().unwrap()
    }

    #[test]
    fn solution_is_a_legal_thirteen_card_board_plus_one_discard() {
        let hand = hand_of("Ah Kh Qh Jh Th 9h 8h 7h 6h 5h 4h 3h 2h As");
        let mut solver = FlSolver::new(&pure_config(9.109)).unwrap();
        let solution = solver.solve(&hand).unwrap();
        let mut used: Vec<u8> = solution
            .top
            .iter()
            .chain(solution.middle.iter())
            .chain(solution.bottom.iter())
            .copied()
            .collect();
        used.push(solution.discard);
        used.sort_unstable();
        let mut expected = hand.to_vec();
        expected.sort_unstable();
        assert_eq!(used, expected);
        assert!(!is_foul(
            solution.top_key,
            solution.middle_key,
            solution.bottom_key
        ));
    }

    #[test]
    fn a_royal_flush_bottom_is_found_and_stays() {
        // Nine high cards in one suit plus quad aces: the solver should reach a
        // straight flush bottom, which is a quads-or-better stay.
        let hand = hand_of("Ah Kh Qh Jh Th 9h 8h 7h 2c 3d 4s 5c 6d 8s");
        let mut solver = FlSolver::new(&pure_config(9.109)).unwrap();
        let solution = solver.solve(&hand).unwrap();
        assert!(solution.stays(), "solution: {solution:?}");
        assert_eq!(solution.stay.as_deref(), Some("stay_bottom_quads_plus"));
        assert_eq!(solution.bottom_royalty, 25);
    }

    #[test]
    fn a_large_stay_bonus_buys_royalties_away() {
        // Trips on top stay; with no stay bonus the solver prefers to break
        // them up if royalties elsewhere pay more, and with a big bonus it does
        // not. This is the knob the FL EV config controls.
        let hand = hand_of("7h 7d 7c 2h 3h 4h 5h 6h 9s Ts Js Qs Ks 2c");
        let mut low = FlSolver::new(&pure_config(0.0)).unwrap();
        let mut high = FlSolver::new(&pure_config(100.0)).unwrap();
        let cheap = low.solve(&hand).unwrap();
        let dear = high.solve(&hand).unwrap();
        assert!(dear.stays());
        assert!(dear.total_royalty <= cheap.total_royalty);
    }

    #[test]
    fn pruned_search_matches_brute_force_on_a_fixed_hand() {
        let hand = hand_of("Ah Kd Qc Js Th 9d 8c 7s 6h 5d 4c 3s 2h Ad");
        let config = pure_config(9.109);
        let mut solver = FlSolver::new(&config).unwrap();
        let fast = solver.solve(&hand).unwrap();
        let slow = solve_brute_force(&hand, &config).unwrap();
        assert_eq!(fast, slow);
    }
}
