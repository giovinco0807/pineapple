//! Feature encoding for a learned T3 first-seat evaluator.
//!
//! Acting first the opponent has nine cards and four open slots, and one
//! property survives from the second seat while another fails. Row counts are
//! still forced -- every board finishes at thirteen and row capacities are
//! fixed, so each row receives exactly its open slots -- which keeps the
//! per-row category histograms exact. What fails is the joint condition: with
//! three open rows the opponent chooses which cards go where, so fouling is a
//! fact about their options rather than about the deal, and enumerating it
//! exactly does not fit a feature budget.
//!
//! The joint block is therefore a deterministic stride over the completions,
//! with the opponent taking the legal arrangement that maximizes royalties plus
//! Fantasy Land. No RNG anywhere: the one nondeterminism bug in this line of
//! work was only ever caught because a second implementation was required to
//! reproduce the numbers, and this module is that second implementation -- it
//! must match `t3first_outlook.py` / `t1_outlook_ref.py` value for value,
//! including iteration order and tie-breaks.
//!
//! How much of the board is still open is a parameter, not a constant: four
//! slots for a board that has played T2, six for one that has played T1, eight
//! for one that has played neither. Nothing about the method changes with the
//! number -- row counts stay forced, so the histograms stay exact, and only the
//! joint block samples -- but the sizes do, and every bound below is stated for
//! the largest of the three so a wider board cannot quietly overrun one.

use crate::cards::Card;
use crate::infoset::ActorObservation;
use crate::scoring::{
    bottom_royalty_from_value, compare_key, fl_entry_from_top_value, middle_royalty_from_value,
    top_royalty_from_value, HandValue,
};
use crate::state::{Board, Row};
use crate::t3_features::{
    encode_structural, head_to_head, side_outlook, Finish, HEAD_TO_HEAD_SIZE,
    SIDE_OUTLOOK_SIZE, STRUCTURAL_SIZE,
};
use crate::t4_features::partial_value;

const ROW_CAPACITY: [usize; 3] = [3, 5, 5];

/// Open-slot totals the free outlook is validated for, widest last.
///
/// Eight, six and four are the ordinary streets: a board that has played
/// neither T1 nor T2, one that has played T1, one that has played both. Two is
/// the vs-Fantasyland tail. Against an opponent who took their hand face down
/// the hero still plays all five streets, and its board after acting on T3
/// carries eleven cards -- two open slots -- where in an ordinary hand the T3
/// decision is the last one an outlook is ever asked about at nine.
///
/// Two is the EASIEST of the four, not a stretch of the method. Every bound in
/// this module is stated for eight; two draws two cards instead of eight,
/// enumerates at most `C(2, 1) = 2` part sets per row against a bound of
/// [`MAX_PART_SETS`], and produces at most three arrangements. The reason it
/// was excluded was that no street reached it, not that the arithmetic
/// struggles -- which is a claim the bounds test below checks over every room
/// split rather than one this comment can make.
///
/// A list rather than a range: the odd totals in between are not reachable by
/// any legal board, since every street places two cards, and a board presenting
/// one would be malformed rather than unsupported.
const SUPPORTED_ROOM: [usize; 4] = [2, 4, 6, 8];
const CATEGORIES: usize = 9;
const HISTOGRAM_CAP: usize = 4000;
const JOINT_DRAWS: usize = 512;

/// Mirrors the second seat's outlook width so the assembled vector keeps the
/// same 168 dimensions and block layout.
pub const FIRST_OPP_SIZE: usize = SIDE_OUTLOOK_SIZE;
pub const FEATURE_SIZE: usize =
    STRUCTURAL_SIZE + SIDE_OUTLOOK_SIZE + FIRST_OPP_SIZE + HEAD_TO_HEAD_SIZE;

fn royalty_of(row: usize, value: &HandValue) -> f32 {
    match row {
        0 => top_royalty_from_value(value) as f32,
        1 => middle_royalty_from_value(value) as f32,
        _ => bottom_royalty_from_value(value) as f32,
    }
}

/// Evenly spaced subset by the same float arithmetic as the Python reference.
fn strided_indices(total: usize, cap: usize) -> Vec<usize> {
    if total <= cap {
        return (0..total).collect();
    }
    let step = total as f64 / cap as f64;
    (0..cap).map(|index| (index as f64 * step) as usize).collect()
}

/// All r-card combinations of `unknown`, in lexicographic index order --
/// itertools.combinations order, which the stride depends on.
///
/// Only the unranking test walks the whole sequence now: at eight open slots
/// the sequence is `C(39, 8)`, sixty-one million combinations, and every caller
/// wants at most a few thousand of them.
#[cfg(test)]
fn combinations(unknown: &[Card], r: usize) -> Vec<Vec<Card>> {
    let mut out = Vec::new();
    let mut indices: Vec<usize> = (0..r).collect();
    if r == 0 {
        out.push(Vec::new());
        return out;
    }
    if r > unknown.len() {
        return out;
    }
    loop {
        out.push(indices.iter().map(|&i| unknown[i]).collect());
        // Advance to the next combination.
        let mut position = r;
        loop {
            if position == 0 {
                return out;
            }
            position -= 1;
            if indices[position] != position + unknown.len() - r {
                break;
            }
        }
        indices[position] += 1;
        for later in position + 1..r {
            indices[later] = indices[later - 1] + 1;
        }
    }
}

/// One more than the largest `n` the Pascal table below answers for: a deck is
/// fifty-two cards, so no unknown set this module is handed is wider.
const PASCAL_N: usize = 53;
/// One more than the largest `r`: [`MAX_DRAW`] open slots, and the unranker asks
/// only for `r` and smaller.
const PASCAL_R: usize = MAX_DRAW + 1;

/// `C(n, r)` for every size this module reaches, built by Pascal's recurrence.
///
/// The addition is exact and needs no division, which is the point: the
/// unranker asks for a binomial coefficient once per candidate index it steps
/// over, and the multiplicative form spent an integer division per factor at
/// every one of those. The widest entry is `C(52, 8) = 752_538_150`, which is
/// comfortably inside `u32` and ten orders of magnitude short of `u64`, so
/// nothing here can overflow or round.
static PASCAL: [[u64; PASCAL_R]; PASCAL_N] = pascal_table();

const fn pascal_table() -> [[u64; PASCAL_R]; PASCAL_N] {
    let mut table = [[0u64; PASCAL_R]; PASCAL_N];
    table[0][0] = 1;
    let mut n = 1;
    while n < PASCAL_N {
        table[n][0] = 1;
        let mut r = 1;
        while r < PASCAL_R {
            // `C(n, r) = C(n-1, r-1) + C(n-1, r)`, which is already zero for
            // `r > n` because every term feeding it is.
            table[n][r] = table[n - 1][r - 1] + table[n - 1][r];
            r += 1;
        }
        n += 1;
    }
    table
}

/// `C(n, r)` for the sizes this module reaches (`n <= 52`, `r <= 8`).
///
/// A table lookup within those bounds and the multiplicative walk outside them,
/// which no caller in this module reaches but the arithmetic still has to be
/// defined for. Each step of that walk is exact in integers -- `C(n, k) *
/// (n - k)` is always divisible by `k + 1` -- and the widest intermediate at
/// `n = 52, r = 8` is about `6e9`, which is a `usize` here and nowhere near
/// overflowing one. The table reproduces it entry for entry, which
/// `the_pascal_table_is_the_multiplicative_walk` pins.
#[inline]
fn count_combinations(n: usize, r: usize) -> usize {
    if r > n {
        return 0;
    }
    if n < PASCAL_N && r < PASCAL_R {
        return PASCAL[n][r] as usize;
    }
    let mut out = 1usize;
    for step in 0..r {
        out = out * (n - step) / (step + 1);
    }
    out
}

/// The largest draw this module enumerates -- eight open slots.
const MAX_DRAW: usize = 8;
/// `C(8, 4)`, the most distinct index subsets one row can be dealt out of an
/// eight-card draw, and hence the width of the per-draw completion cache. A row
/// holds at most five open slots (three up top), and `C(8, r)` peaks at `r = 4`
/// with rooms like `(0, 4, 4)`, which an eight-slot board does reach.
const MAX_PART_SETS: usize = 70;

/// The `rank`-th r-subset of `0..n` in `combinations`' lexicographic order.
///
/// Unranking rather than walking: the joint block wants 512 draws out of tens
/// of thousands, and the walk costs the whole sequence. The order is the one
/// `combinations` emits, which `unranking_agrees_with_the_walk` pins.
fn unrank_combination(n: usize, r: usize, rank: usize, out: &mut [usize; MAX_DRAW]) {
    let mut remaining = rank;
    let mut candidate = 0usize;
    for slot in 0..r {
        loop {
            let block = count_combinations(n - 1 - candidate, r - 1 - slot);
            if remaining < block {
                break;
            }
            remaining -= block;
            candidate += 1;
        }
        out[slot] = candidate;
        candidate += 1;
    }
}

/// The combinations `combinations(unknown, r)` would put at `wanted`, flattened
/// into one buffer of stride `r` and without materialising the ones in between.
fn combinations_at(unknown: &[Card], r: usize, wanted: &[usize]) -> Vec<Card> {
    let mut out = Vec::with_capacity(wanted.len() * r);
    if wanted.is_empty() || r == 0 || r > unknown.len() || r > MAX_DRAW {
        return out;
    }
    let mut indices = [0usize; MAX_DRAW];
    for &rank in wanted {
        unrank_combination(unknown.len(), r, rank, &mut indices);
        out.extend(indices[..r].iter().map(|&i| unknown[i]));
    }
    out
}

/// One evaluated row completion: the row's own cards plus a chosen subset of
/// the unknowns, reduced to the packed ordering key. Royalty and Fantasy Land
/// are stored beside it so the joint loop reads all three from the same place
/// it would have computed them.
struct RowCompletion {
    key: u64,
    royalty: f32,
    fantasyland: bool,
}

/// The part of a `RowCompletion` the arrangement loop reads, flat and `Copy`.
#[derive(Clone, Copy)]
struct CompletionView {
    key: u64,
    royalty: f32,
    fantasyland: bool,
}

fn mix64(key: u64) -> u64 {
    let mut z = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z ^= z >> 29;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^ (z >> 32)
}

/// Memo from "which unknown cards complete this row" -- a 52-bit card mask --
/// to the evaluated completion.
///
/// The joint block asks for the same completions over and over: a row only ever
/// sees `room[row]` cards drawn from `unknown`, so there are at most
/// `C(unknown.len(), room[row])` distinct questions, while the loop asks
/// `draws * arrangements` of them. Open addressing with a fixed table sized
/// from that exact bound, so no rehash ever moves an entry.
struct RowMemo {
    keys: Vec<u64>,
    slots: Vec<u32>,
    mask: usize,
    entries: Vec<RowCompletion>,
}

impl RowMemo {
    fn new(row: usize, row_cards: &[Card], room: usize, distinct_bound: usize) -> Self {
        let mut size = 16usize;
        while size < distinct_bound.saturating_mul(3).max(16) {
            size <<= 1;
        }
        let mut memo = Self {
            keys: vec![0; size],
            slots: vec![0; size],
            mask: size - 1,
            entries: Vec::with_capacity(distinct_bound.max(1)),
        };
        if room == 0 {
            // The one completion a closed row has; the mask for it is zero, and
            // zero is the empty-bucket marker, so it is seeded rather than
            // looked up.
            memo.entries.push(evaluate_completion(row, row_cards));
        }
        memo
    }

    /// The completion `row_cards + {draw[k] for k in part}`, evaluating it on
    /// first sight. `key` is the OR of those cards' bits.
    fn resolve(
        &mut self,
        row: usize,
        row_cards: &[Card],
        key: u64,
        draw: &[Card],
        part: &[usize],
        scratch: &mut Vec<Card>,
    ) -> CompletionView {
        if key == 0 {
            return self.view(0);
        }
        let mut probe = mix64(key) as usize & self.mask;
        loop {
            let stored = self.keys[probe];
            if stored == key {
                return self.view(self.slots[probe]);
            }
            if stored == 0 {
                break;
            }
            probe = (probe + 1) & self.mask;
        }
        scratch.clear();
        scratch.extend_from_slice(row_cards);
        for &k in part {
            scratch.push(draw[k]);
        }
        let slot = self.entries.len() as u32;
        self.entries.push(evaluate_completion(row, scratch));
        self.keys[probe] = key;
        self.slots[probe] = slot;
        self.view(slot)
    }

    fn view(&self, slot: u32) -> CompletionView {
        let completion = &self.entries[slot as usize];
        CompletionView {
            key: completion.key,
            royalty: completion.royalty,
            fantasyland: completion.fantasyland,
        }
    }
}

fn evaluate_completion(row: usize, cards: &[Card]) -> RowCompletion {
    let value = partial_value(cards, ROW_CAPACITY[row]);
    RowCompletion {
        key: compare_key(&value),
        royalty: royalty_of(row, &value),
        fantasyland: row == 0 && fl_entry_from_top_value(&value).qualifies,
    }
}

/// Position of `part` in `sets`, which holds every subset of that size.
///
/// The arrangement loop used to intern parts in the order it met them, which
/// made the numbering depend on the whole room triple. Numbering them by their
/// place in the full lexicographic list instead is what lets one row's
/// completions be shared between boards whose other rows differ, and the
/// arrangement sequence is unaffected: only the labels inside each triple
/// change, not the order the triples are visited.
fn locate_part(sets: &[Vec<usize>], part: &[usize]) -> u8 {
    sets.iter()
        .position(|held| held.as_slice() == part)
        .expect("every part is a subset of its own size") as u8
}

/// The bits of `cards`, which identify a row's contents independently of the
/// order they were placed in.
///
/// Every value the outlook derives from a row goes through `partial_value`,
/// which reads rank counts, sorted ranks and a single-suit test -- all
/// order-blind -- so two boards that reached the same row contents by different
/// placement orders have the same completions, and this is a sound cache key.
fn row_mask(cards: &[Card]) -> u64 {
    cards.iter().fold(0u64, |mask, card| mask | card.bit())
}

/// `histogram_cap` is how many completions the stride keeps. It is a parameter
/// rather than the constant it reads like because a second, deliberately
/// coarser encoder shares this body at a smaller cap; every caller on the
/// original path passes [`HISTOGRAM_CAP`] and is unaffected.
fn row_histogram(
    cards: &[Card],
    row: usize,
    unknown: &[Card],
    histogram_cap: usize,
) -> ([f64; CATEGORIES], u8, u8) {
    let capacity = ROW_CAPACITY[row];
    let room = capacity - cards.len();
    let mut counts = [0.0f64; CATEGORIES];
    if room == 0 {
        let value = partial_value(cards, capacity);
        counts[(value.0 as usize).min(CATEGORIES - 1)] = 1.0;
        return (counts, value.0, value.0);
    }
    // Strided by rank rather than over a materialized list: an open row can
    // want five of thirty-nine unknowns, and building all `C(39, 5)` of them to
    // keep four thousand costs more than the histogram it feeds.
    let picks = strided_indices(count_combinations(unknown.len(), room), histogram_cap);
    let combos = combinations_at(unknown, room, &picks);
    let mut buffer = Vec::with_capacity(capacity);
    let (mut lowest, mut highest) = (8u8, 0u8);
    for combo in combos.chunks_exact(room) {
        buffer.clear();
        buffer.extend_from_slice(cards);
        buffer.extend_from_slice(combo);
        let value = partial_value(&buffer, capacity);
        counts[(value.0 as usize).min(CATEGORIES - 1)] += 1.0;
        lowest = lowest.min(value.0);
        highest = highest.max(value.0);
    }
    let total: f64 = counts.iter().sum();
    for count in counts.iter_mut() {
        *count /= total.max(1.0);
    }
    (counts, lowest, highest)
}

/// Exact-histogram, sampled-joint outlook for a board with four, six or eight
/// open slots.
pub fn opponent_outlook_first(
    board: &Board,
    unknown: &[Card],
) -> Result<([f32; FIRST_OPP_SIZE], Vec<Finish>), String> {
    FreeOutlookCache::new().outlook(board, unknown)
}

/// The 512 draws one `(unknown, total_room)` pair produces, flattened.
struct DrawSet {
    total_room: usize,
    /// Flat, stride `total_room`.
    cards: Vec<Card>,
    /// `picks.len()`, the denominator the foul rate divides by.
    picks: usize,
    /// How many whole draws `cards` actually holds, which is `picks` unless the
    /// unknown set is too small to draw from at all.
    draws: usize,
}

/// The arrangement sequence one room triple produces, with each row's part
/// numbered by its place in that row's full subset list.
struct Plan {
    room: [usize; 3],
    total_room: usize,
    part_lists: [Vec<Vec<usize>>; 3],
    arrangements: Vec<[u8; 3]>,
}

/// One row's completions against every draw: `views[draw * parts + part]`.
struct RowTable {
    row: usize,
    total_room: usize,
    mask: u64,
    parts: usize,
    views: Vec<CompletionView>,
}

struct RowHistogram {
    row: usize,
    mask: u64,
    counts: [f64; CATEGORIES],
    lowest: u8,
    highest: u8,
}

/// Work one decision's candidate boards share, held for the decision's duration.
///
/// A T2 decision asks for the outlook once per legal action -- twenty-odd nine
/// card boards built from one seven-card board and two of three dealt cards --
/// and always against the same unknown set, because all three dealt cards are
/// known to the actor however they were placed. Three quarters of the outlook's
/// cost is therefore recomputation: the draw list depends only on the unknown
/// set and the number of open slots, and both the per-row histogram and the
/// per-row completions depend only on that row, so rows the action did not touch
/// are answered identically for every candidate.
///
/// Nothing here changes a value. The same completions are evaluated and the
/// arrangements are visited in the same order, so the first-strict-maximum
/// tie-break sees the same sequence; only the number of times the work happens
/// changes. Entries are keyed on the unknown set as well, and a call with a
/// different one empties the cache rather than answering from it.
pub(crate) struct FreeOutlookCache {
    unknown: Vec<Card>,
    /// Completions the per-row histogram stride keeps.
    histogram_cap: usize,
    /// Completions the joint block strides over.
    joint_draws: usize,
    draw_sets: Vec<DrawSet>,
    plans: Vec<Plan>,
    histograms: Vec<RowHistogram>,
    tables: Vec<RowTable>,
}

impl FreeOutlookCache {
    /// The outlook this module is named for, at the caps the parity fixtures
    /// pin.
    pub(crate) fn new() -> Self {
        Self::with_caps(HISTOGRAM_CAP, JOINT_DRAWS)
    }

    /// The same method at caller-chosen caps.
    ///
    /// The two numbers are the whole of the accuracy/cost dial: how many row
    /// completions the histogram stride keeps and how many draws the joint
    /// block visits. Nothing else about the method depends on them -- row
    /// counts are still forced, so the histograms are still exact over what
    /// they sample, and the arrangement loop still takes the first strict
    /// maximum in the same order. A coarser cache is therefore a different
    /// function of the board rather than a worse implementation of this one,
    /// which is why the fast encoder that uses it carries no parity claim
    /// against the values [`new`](Self::new) produces.
    pub(crate) fn with_caps(histogram_cap: usize, joint_draws: usize) -> Self {
        Self {
            unknown: Vec::new(),
            histogram_cap,
            joint_draws,
            draw_sets: Vec::new(),
            plans: Vec::new(),
            histograms: Vec::new(),
            tables: Vec::new(),
        }
    }

    fn retarget(&mut self, unknown: &[Card]) {
        if self.unknown == unknown {
            return;
        }
        self.unknown.clear();
        self.unknown.extend_from_slice(unknown);
        self.draw_sets.clear();
        self.plans.clear();
        self.histograms.clear();
        self.tables.clear();
    }

    fn draw_slot(&mut self, total_room: usize) -> usize {
        if let Some(found) = self
            .draw_sets
            .iter()
            .position(|held| held.total_room == total_room)
        {
            return found;
        }
        let picks = strided_indices(
            count_combinations(self.unknown.len(), total_room),
            self.joint_draws,
        );
        let cards = combinations_at(&self.unknown, total_room, &picks);
        let draws = cards.len() / total_room;
        self.draw_sets.push(DrawSet {
            total_room,
            cards,
            picks: picks.len(),
            draws,
        });
        self.draw_sets.len() - 1
    }

    fn plan_slot(&mut self, room: [usize; 3], total_room: usize) -> usize {
        if let Some(found) = self
            .plans
            .iter()
            .position(|held| held.room == room && held.total_room == total_room)
        {
            return found;
        }
        let part_lists = [
            combinations_of_indices(total_room, room[0]),
            combinations_of_indices(total_room, room[1]),
            combinations_of_indices(total_room, room[2]),
        ];
        debug_assert!(part_lists.iter().all(|sets| sets.len() <= MAX_PART_SETS));
        // The same nested walk the joint block always did: top splits in
        // `combinations` order, middle within the remainder, bottom whatever is
        // left. Only the labels are different.
        let mut arrangements: Vec<[u8; 3]> = Vec::new();
        for top_part in &part_lists[0] {
            let rest: Vec<usize> = (0..total_room).filter(|k| !top_part.contains(k)).collect();
            for mid_part in combinations_of_indices(rest.len(), room[1]) {
                let mid_actual: Vec<usize> = mid_part.iter().map(|&j| rest[j]).collect();
                let bottom_part: Vec<usize> = rest
                    .iter()
                    .copied()
                    .filter(|k| !mid_actual.contains(k))
                    .collect();
                arrangements.push([
                    locate_part(&part_lists[0], top_part),
                    locate_part(&part_lists[1], &mid_actual),
                    locate_part(&part_lists[2], &bottom_part),
                ]);
            }
        }
        self.plans.push(Plan {
            room,
            total_room,
            part_lists,
            arrangements,
        });
        self.plans.len() - 1
    }

    fn table_slot(
        &mut self,
        row: usize,
        row_cards: &[Card],
        room: usize,
        draw_slot: usize,
        plan_slot: usize,
    ) -> usize {
        let total_room = self.draw_sets[draw_slot].total_room;
        let mask = row_mask(row_cards);
        if let Some(found) = self.tables.iter().position(|held| {
            held.row == row && held.total_room == total_room && held.mask == mask
        }) {
            return found;
        }
        let draw_set = &self.draw_sets[draw_slot];
        let parts = &self.plans[plan_slot].part_lists[row];
        let bound = count_combinations(self.unknown.len(), room).min(
            draw_set
                .picks
                .saturating_mul(count_combinations(total_room, room)),
        );
        let mut memo = RowMemo::new(row, row_cards, room, bound);
        let mut views = Vec::with_capacity(draw_set.draws * parts.len());
        let mut scratch = Vec::with_capacity(5);
        for draw in draw_set.cards.chunks_exact(total_room) {
            for part in parts {
                let mut key = 0u64;
                for &k in part {
                    key |= draw[k].bit();
                }
                views.push(memo.resolve(row, row_cards, key, draw, part, &mut scratch));
            }
        }
        self.tables.push(RowTable {
            row,
            total_room,
            mask,
            parts: parts.len(),
            views,
        });
        self.tables.len() - 1
    }

    fn histogram(&mut self, row: usize, row_cards: &[Card]) -> ([f64; CATEGORIES], u8, u8) {
        let mask = row_mask(row_cards);
        if let Some(held) = self
            .histograms
            .iter()
            .find(|held| held.row == row && held.mask == mask)
        {
            return (held.counts, held.lowest, held.highest);
        }
        let (counts, lowest, highest) =
            row_histogram(row_cards, row, &self.unknown, self.histogram_cap);
        self.histograms.push(RowHistogram {
            row,
            mask,
            counts,
            lowest,
            highest,
        });
        (counts, lowest, highest)
    }

    pub(crate) fn outlook(
        &mut self,
        board: &Board,
        unknown: &[Card],
    ) -> Result<([f32; FIRST_OPP_SIZE], Vec<Finish>), String> {
        self.outlook_impl(board, unknown, true)
    }

    /// The block alone, for callers that discard the finishes.
    ///
    /// The hidden-opponent encoder zeroes its head-to-head columns, so the
    /// finishes -- three cloned `HandValue`s per surviving draw -- are built
    /// and immediately dropped there. Skipping their construction is the only
    /// difference: every count the block reads, including the survivor count
    /// that is `finishes.len()` on the collecting path, is kept by the same
    /// arithmetic, so the block is the same bytes either way.
    pub(crate) fn outlook_block_only(
        &mut self,
        board: &Board,
        unknown: &[Card],
    ) -> Result<[f32; FIRST_OPP_SIZE], String> {
        Ok(self.outlook_impl(board, unknown, false)?.0)
    }

    fn outlook_impl(
        &mut self,
        board: &Board,
        unknown: &[Card],
        collect_finishes: bool,
    ) -> Result<([f32; FIRST_OPP_SIZE], Vec<Finish>), String> {
        let rows = [
            board.cards(Row::Top),
            board.cards(Row::Middle),
            board.cards(Row::Bottom),
        ];
        let room: Vec<usize> = (0..3).map(|i| ROW_CAPACITY[i] - rows[i].len()).collect();
        let total_room = room.iter().sum::<usize>();
        if !SUPPORTED_ROOM.contains(&total_room) {
            return Err(format!(
                "free outlook expects a board with two, four, six or eight open \
                 slots, got {room:?}"
            ));
        }
        self.retarget(unknown);

        let mut out = [0.0f32; FIRST_OPP_SIZE];
        let mut lows = [0u8; 3];
        let mut highs = [0u8; 3];
        for row in 0..3 {
            let (histogram, lowest, highest) = self.histogram(row, rows[row]);
            for (index, value) in histogram.iter().enumerate() {
                out[row * CATEGORIES + index] = *value as f32;
            }
            lows[row] = lowest;
            highs[row] = highest;
        }

        let mut base = 3 * CATEGORIES;
        for row in 0..3 {
            out[base + row] = room[row] as f32 / 5.0;
        }
        base += 3;

        let draw_slot = self.draw_slot(total_room);
        let plan_slot = self.plan_slot([room[0], room[1], room[2]], total_room);
        let mut table_slots = [0usize; 3];
        for row in 0..3 {
            table_slots[row] = self.table_slot(row, rows[row], room[row], draw_slot, plan_slot);
        }

        let arrangements = &self.plans[plan_slot].arrangements;
        let tables = [
            &self.tables[table_slots[0]],
            &self.tables[table_slots[1]],
            &self.tables[table_slots[2]],
        ];
        let draw_set = &self.draw_sets[draw_slot];

        let mut finishes: Vec<Finish> =
            Vec::with_capacity(if collect_finishes { draw_set.picks } else { 0 });
        let mut survivors = 0usize;
        let mut fouls = 0usize;
        let mut fl_count = 0usize;
        let mut royalty_sum = 0.0f64;

        for index in 0..draw_set.draws {
            let bases = [
                index * tables[0].parts,
                index * tables[1].parts,
                index * tables[2].parts,
            ];
            // (score, row keys, royalty, fl); first strict maximum wins,
            // matching the Python `>` comparison. The keys the legality check
            // reads are the same packed values the finish carries, so the
            // winner needs no lookup back into the tables.
            let mut best: Option<(f32, [u64; 3], f32, bool)> = None;
            for arrangement in arrangements {
                let top = tables[0].views[bases[0] + arrangement[0] as usize];
                let middle = tables[1].views[bases[1] + arrangement[1] as usize];
                let bottom = tables[2].views[bases[2] + arrangement[2] as usize];
                if !(top.key <= middle.key && middle.key <= bottom.key) {
                    continue;
                }
                let royalty = top.royalty + middle.royalty + bottom.royalty;
                let fl = top.fantasyland;
                let score = royalty + if fl { 10.0 } else { 0.0 };
                if best.as_ref().map_or(true, |(held, _, _, _)| score > *held) {
                    best = Some((score, [top.key, middle.key, bottom.key], royalty, fl));
                }
            }
            match best {
                None => fouls += 1,
                Some((_score, keys, royalty, fl)) => {
                    survivors += 1;
                    royalty_sum += royalty as f64;
                    if fl {
                        fl_count += 1;
                    }
                    if collect_finishes {
                        finishes.push(Finish::new(keys, royalty, fl));
                    }
                }
            }
        }

        let total = draw_set.picks;
        out[base] = (fouls as f64 / total.max(1) as f64) as f32;
        out[base + 1] = if total > 0 && fouls == total { 1.0 } else { 0.0 };
        base += 2;
        out[base] = (lows[1] as f32 - highs[2] as f32) / 8.0;
        out[base + 1] = (lows[0] as f32 - highs[1] as f32) / 8.0;
        base += 2;
        let survived = survivors.max(1) as f32;
        out[base] = fl_count as f32 / survived;
        out[base + 1] = (royalty_sum as f32 / survived) / 10.0;
        Ok((out, finishes))
    }
}

/// itertools.combinations(range(n), r) order.
fn combinations_of_indices(n: usize, r: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if r > n {
        return out;
    }
    if r == 0 {
        out.push(Vec::new());
        return out;
    }
    let mut indices: Vec<usize> = (0..r).collect();
    loop {
        out.push(indices.clone());
        let mut position = r;
        loop {
            if position == 0 {
                return out;
            }
            position -= 1;
            if indices[position] != position + n - r {
                break;
            }
        }
        indices[position] += 1;
        for later in position + 1..r {
            indices[later] = indices[later - 1] + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::ALL_CARDS;

    /// The table replaced a multiplicative walk with a division per factor, so
    /// the two have to agree on every `(n, r)` the module can reach -- not only
    /// on the ones a fixture happens to exercise.
    ///
    /// The walk is transcribed here rather than called, because `count_combinations`
    /// now answers from the table for exactly the range under test and calling
    /// it would compare the table with itself.
    #[test]
    fn the_pascal_table_is_the_multiplicative_walk() {
        fn walked(n: usize, r: usize) -> usize {
            if r > n {
                return 0;
            }
            let mut out = 1usize;
            for step in 0..r {
                out = out * (n - step) / (step + 1);
            }
            out
        }
        for n in 0..PASCAL_N {
            for r in 0..PASCAL_R {
                assert_eq!(count_combinations(n, r), walked(n, r), "C({n}, {r})");
            }
        }
        // The widest entry the module can ask for, stated rather than derived,
        // so a table that silently narrowed would fail here too.
        assert_eq!(count_combinations(52, 8), 752_538_150);
        assert_eq!(PASCAL[52][8], 752_538_150u64);
        assert!(PASCAL[52][8] < u64::MAX);
    }

    /// The joint block picks its draws by index into `combinations`' output, so
    /// unranking has to land on exactly the combination the walk would have.
    #[test]
    fn unranking_agrees_with_the_walk() {
        for n in 0..14usize {
            for r in 1..=MAX_DRAW.min(n) {
                let walked = combinations(&ALL_CARDS[..n], r);
                assert_eq!(walked.len(), count_combinations(n, r));
                let wanted: Vec<usize> = (0..walked.len()).collect();
                let flat = combinations_at(&ALL_CARDS[..n], r, &wanted);
                for (index, expected) in walked.iter().enumerate() {
                    assert_eq!(&flat[index * r..(index + 1) * r], &expected[..],
                               "n {n} r {r} rank {index}");
                }
            }
        }
    }

    /// `views` is a fixed `MAX_PART_SETS`-wide array, and the joint block
    /// indexes it by interned part set. In release the `debug_assert` guarding
    /// that is compiled out, so the bound is checked here instead, over every
    /// room split a four-, six- or eight-slot board can present.
    #[test]
    fn no_reachable_room_split_overruns_the_completion_cache() {
        let mut widest = 0usize;
        for top in 0..=3usize {
            for middle in 0..=5usize {
                for bottom in 0..=5usize {
                    let total = top + middle + bottom;
                    if !SUPPORTED_ROOM.contains(&total) {
                        continue;
                    }
                    for room in [top, middle, bottom] {
                        let sets = count_combinations(total, room);
                        assert!(
                            sets <= MAX_PART_SETS,
                            "rooms ({top}, {middle}, {bottom}) want {sets} part sets"
                        );
                        widest = widest.max(sets);
                    }
                    assert!(total <= MAX_DRAW);
                }
            }
        }
        assert_eq!(widest, MAX_PART_SETS, "the bound is no longer tight");
    }

    /// The two-slot arm produces a finite, correctly shaped block on every room
    /// split it can present, and its joint block is not degenerate.
    ///
    /// Widening a guard is the cheapest change in this module to make and the
    /// easiest to make wrongly: the machinery is parameterised by open-slot
    /// count, so a total nobody validated still runs, still returns an array of
    /// the right width, and can still be quietly wrong in the joint block --
    /// which is the half that samples. So this checks the parts a width change
    /// can actually break.
    ///
    /// The three splits are every one a two-slot board can have with a legal
    /// row shape: an eleven-card board is missing two cards, and they are
    /// missing from one row or from two.
    #[test]
    fn the_two_slot_arm_is_shaped_and_finite_on_every_room_split() {
        let splits: [[usize; 3]; 3] = [
            [3, 4, 4], // one open in middle, one in bottom
            [2, 5, 4], // two open in top... and one in bottom
            [1, 5, 5], // both open slots in the top row
        ];
        for counts in splits {
            let mut next = 0usize;
            let mut take = |n: usize| -> Vec<Card> {
                let out: Vec<Card> = (0..n).map(|i| ALL_CARDS[next + i]).collect();
                next += n;
                out
            };
            let board = Board::new(take(counts[0]), take(counts[1]), take(counts[2]))
                .expect("legal board");
            assert_eq!(board.card_count(), 11, "split {counts:?} is not a T3 board");
            let used = board.all_cards();
            let unknown: Vec<Card> = ALL_CARDS
                .iter()
                .copied()
                .filter(|card| !used.contains(card))
                .collect();

            let (block, finishes) =
                FreeOutlookCache::new().outlook(&board, &unknown).expect("two-slot outlook");
            assert_eq!(block.len(), FIRST_OPP_SIZE);
            for (slot, value) in block.iter().enumerate() {
                assert!(value.is_finite(), "split {counts:?} slot {slot} is {value}");
            }
            // The joint block draws two cards and picks the best legal
            // arrangement of them. An eleven-card board that fouls on every
            // draw would be possible in principle, but not on these: the rows
            // are dealt in deck order and stay ordered. A block whose joint
            // half never produced a finish would be all-zero in the tail, which
            // is what this catches.
            assert!(
                !finishes.is_empty(),
                "split {counts:?} produced no surviving arrangement at all"
            );
            let histogram_sum: f32 = block[..3 * CATEGORIES].iter().sum();
            assert!(
                (histogram_sum - 3.0).abs() < 1e-4,
                "split {counts:?}: three row histograms should each sum to one, \
                 got {histogram_sum}"
            );
        }
    }

    /// Sharing a cache across two-slot boards answers what a fresh one does.
    ///
    /// The cache keys its draw set on the open-slot total, so a new total is a
    /// new key -- and the one way a widened guard could corrupt the ESTABLISHED
    /// widths is by colliding with one of their keys. Interleaving a two-slot
    /// board with a four-slot one in a single cache is the shape of that
    /// mistake.
    #[test]
    fn a_two_slot_board_does_not_disturb_a_four_slot_one_sharing_its_cache() {
        let mut next = 0usize;
        let mut take = |n: usize| -> Vec<Card> {
            let out: Vec<Card> = (0..n).map(|i| ALL_CARDS[next + i]).collect();
            next += n;
            out
        };
        let nine = Board::new(take(2), take(4), take(3)).expect("nine-card board");
        let used = nine.all_cards();
        let unknown: Vec<Card> = ALL_CARDS
            .iter()
            .copied()
            .filter(|card| !used.contains(card))
            .collect();
        // The same nine-card board plus two more cards, so both widths are read
        // against one unknown set the way a real decision would.
        let eleven = Board::new(
            nine.cards(Row::Top).to_vec(),
            nine.cards(Row::Middle).to_vec(),
            {
                let mut bottom = nine.cards(Row::Bottom).to_vec();
                bottom.push(unknown[0]);
                bottom.push(unknown[1]);
                bottom
            },
        )
        .expect("eleven-card board");
        let unknown_after: Vec<Card> =
            unknown.iter().copied().filter(|card| !eleven.all_cards().contains(card)).collect();

        let (alone_four, _) =
            FreeOutlookCache::new().outlook(&nine, &unknown).expect("four-slot alone");
        let (alone_two, _) = FreeOutlookCache::new()
            .outlook(&eleven, &unknown_after)
            .expect("two-slot alone");

        let mut shared = FreeOutlookCache::new();
        let (shared_four, _) = shared.outlook(&nine, &unknown).expect("four-slot shared");
        let (shared_two, _) = shared
            .outlook(&eleven, &unknown_after)
            .expect("two-slot shared");
        let (again_four, _) = shared.outlook(&nine, &unknown).expect("four-slot again");

        assert_eq!(alone_four, shared_four, "the four-slot block moved");
        assert_eq!(alone_two, shared_two, "the two-slot block moved");
        assert_eq!(
            alone_four, again_four,
            "a two-slot board in between changed a four-slot answer"
        );
    }

    /// The parity fixtures pin the outlook one board at a time, which is the
    /// path `opponent_outlook_first` takes. The T2 teacher takes the other one:
    /// one cache across a decision's candidate boards, so a row the action left
    /// alone is answered from the previous board's work. That is only sound if
    /// it is invisible, and invisible is not something the fixtures can see, so
    /// it is checked here against the fixtures' own positions -- every feature
    /// and the head-to-head the finishes feed, bit for bit, plus the hit count,
    /// so a cache that silently stopped sharing would fail rather than pass.
    #[test]
    fn sharing_one_cache_across_candidate_boards_changes_nothing() {
        use crate::t3_features::{head_to_head, unknown_cards};

        let mut compared = 0usize;
        for name in ["t2first_features_parity.json", "t1first_features_parity.json"] {
            let raw = std::fs::read_to_string(format!(
                "{}/tests/fixtures/{name}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .expect("fixture is present");
            let parsed: serde_json::Value = serde_json::from_str(&raw).expect("fixture parses");
            let cases = parsed["cases"].as_array().expect("cases");
            assert!(!cases.is_empty());
            for case in cases {
                let observation: ActorObservation =
                    serde_json::from_value(case["observation"].clone()).expect("observation");
                let unknown = unknown_cards(&observation);
                let boards: Vec<Board> = case["actions"]
                    .as_array()
                    .expect("actions")
                    .iter()
                    .map(|action| {
                        let placements: Vec<(Card, Row)> = action["placements"]
                            .as_array()
                            .expect("placements")
                            .iter()
                            .map(|pair| {
                                let card: Card =
                                    pair[0].as_str().expect("card").parse().expect("card parses");
                                let row = match pair[1].as_str().expect("row") {
                                    "top" => Row::Top,
                                    "middle" => Row::Middle,
                                    "bottom" => Row::Bottom,
                                    other => panic!("unknown row {other}"),
                                };
                                (card, row)
                            })
                            .collect();
                        observation
                            .hero_board
                            .place(&placements)
                            .expect("legal placement")
                    })
                    .collect();

                let mut shared = FreeOutlookCache::new();
                let (shared_block, shared_finishes) = shared
                    .outlook(&observation.opponent_public_board, &unknown)
                    .expect("shared opponent outlook");
                let (fresh_block, fresh_finishes) =
                    opponent_outlook_first(&observation.opponent_public_board, &unknown)
                        .expect("fresh opponent outlook");
                assert_eq!(shared_block, fresh_block, "{name}: opponent block moved");
                assert_eq!(shared_finishes.len(), fresh_finishes.len());

                for (index, board) in boards.iter().enumerate() {
                    let (cached, cached_finishes) =
                        shared.outlook(board, &unknown).expect("shared hero outlook");
                    let (plain, plain_finishes) =
                        opponent_outlook_first(board, &unknown).expect("fresh hero outlook");
                    assert_eq!(cached, plain, "{name} board {index}: outlook moved");
                    assert_eq!(
                        head_to_head(&cached_finishes, &shared_finishes),
                        head_to_head(&plain_finishes, &fresh_finishes),
                        "{name} board {index}: head-to-head moved"
                    );
                    compared += 1;
                }

                // Three rows per board plus the opponent's three; sharing must
                // have collapsed most of them or the cache is not working.
                let built = shared.tables.len();
                let asked = 3 * (boards.len() + 1);
                assert!(
                    built * 2 < asked,
                    "{name}: {built} row tables for {asked} row requests is not sharing"
                );
            }
        }
        assert!(compared >= 400, "only {compared} boards compared");
    }

    /// The packed key has to order completions exactly as `HandValue` does,
    /// including the "a prefix sorts first" rule for shorter tie-breakers.
    #[test]
    fn the_packed_key_orders_hand_values_identically() {
        let mut values: Vec<HandValue> = Vec::new();
        for category in 0..9u8 {
            for first in 2..15u8 {
                values.push(HandValue(category, vec![first]));
                values.push(HandValue(category, vec![first, 2]));
                values.push(HandValue(category, vec![first, 14, 3]));
                values.push(HandValue(category, vec![14, 13, 12, 11, first]));
            }
        }
        for left in &values {
            for right in &values {
                assert_eq!(
                    left.cmp(right),
                    compare_key(left).cmp(&compare_key(right)),
                    "{left:?} versus {right:?}"
                );
            }
        }
    }
}

/// The full 168-dimension vector for one first-seat hero action.
pub fn encode_first(
    observation: &ActorObservation,
    hero_board: &Board,
    unknown: &[Card],
    opponent_block: &[f32; FIRST_OPP_SIZE],
    opponent_finishes: &[Finish],
) -> [f32; FEATURE_SIZE] {
    let mut out = [0.0f32; FEATURE_SIZE];
    out[..STRUCTURAL_SIZE].copy_from_slice(&encode_structural(observation, hero_board));
    let (hero_outlook, hero_finishes) = side_outlook(hero_board, unknown);
    let mut base = STRUCTURAL_SIZE;
    out[base..base + SIDE_OUTLOOK_SIZE].copy_from_slice(&hero_outlook);
    base += SIDE_OUTLOOK_SIZE;
    out[base..base + FIRST_OPP_SIZE].copy_from_slice(opponent_block);
    base += FIRST_OPP_SIZE;
    out[base..].copy_from_slice(&head_to_head(&hero_finishes, opponent_finishes));
    out
}
