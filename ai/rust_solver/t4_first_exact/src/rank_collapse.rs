//! Live-suit bucket collapse for the exact V4 state sweep.
//!
//! A row can still reach a flush in **at most one** suit: the suit of the
//! natural cards already sitting in it.  Two different natural suits in a row
//! and neither can get to five; the only row with more than one live suit is a
//! five-slot row holding no natural at all (empty, or nothing but jokers), and
//! then all four are live.  The three-card top scores no flush and is never
//! live.  Call the union of the two boards' live suits `L`.  Then two pool
//! cards of the same rank whose suits are both outside `L` are
//! interchangeable in every completion this sweep enumerates: no row they can
//! land in cares which of the two it got.  The equivalence class of a card is
//! therefore `(rank, bucket)`, where the bucket is the card's own suit when
//! that suit is live and a single shared "other" otherwise.
//!
//! `ai/docs/suit_collapse_20260824.md` §5.5 states the generalisation and
//! `scratchpad/suit_parity_l1.py` measures it: on six real `L = 1` boards,
//! permuting the non-live suits left the exact joint block bit-identical
//! through three variants each, while four controls that moved a *live* suit
//! all diverged.  §5.4 of the same document is the `L = 0` special case this
//! file first shipped, where every natural falls into "other" and the class is
//! the rank alone.
//!
//! So the sweep iterates over classes with multiplicity weights instead of
//! over cards.  A 29-card pool holds about fourteen classes at `L = 0` (the
//! opponent's pair table falls from C(29,2)=406 to ~87 and the three-card
//! sweep from C(27,3)=2925 to ~399), about 27 at `L = 1` and about 40 at
//! `L = 2`.
//!
//! # Why the union, and why it stops at one
//!
//! Hero's candidates and the opponent's completions eat the same 29-card
//! pool, so one side's suit consumption moves the other side's outs.  Taking
//! the **union** of the two boards' live suits repairs that: a suit either
//! side can still use is its own class, so whichever side draws it, the
//! bookkeeping is right for both.
//!
//! The collapse is *correct* at every `|L|`; `FIRE_MAX_LIVE` is about whether
//! it is worth doing.  A collapsed term costs more than a card term -- the
//! multiset has to be decomposed, its weight built from binomials and its
//! per-class share accumulated -- so the loop has to shrink by more than that
//! inflation before anything is won.  Measured, on real V4 states:
//!
//! | `|L|` | classes | triples shrink by | wall clock |
//! |---|---|---|---|
//! | 0 | 13.3 | 8.72x | **5.22x faster** |
//! | 1 | 18.7 | 3.55x | **1.58x faster** |
//! | 2 | 24.8 | 1.58x | **0.49x -- slower** |
//! | 3+ | 29.0 | 1.00x | no shrink at all |
//!
//! At `|L| = 2` the alphabet is barely coarser than the deck and the collapse
//! is a **two-fold loss**, so the threshold is one.  `|L| >= 2` takes the old
//! path, which also buys a sharp gate: the flag has to be byte-inert there.
//! The histogram still counts every bin -- the distribution is worth watching
//! even where nothing fires.
//!
//! `|L| <= 1` additionally makes the root's `L` sound for the whole subtree.
//! One live suit leaves no room for a five-slot row with no natural in it (it
//! would contribute all four), so every unfinished five-slot row holds a
//! single natural suit, and placing a card either closes the row, mixes its
//! suits (killing it), or keeps its own suit live.  `L` is therefore monotone
//! non-increasing through every completion the sweep enumerates -- computed
//! once at the root, no propagation machinery.
//!
//! # Why Phase 0 had to land first
//!
//! Collapsing replaces a concrete card pair `(pool[i], pool[j])` with a
//! representative pair drawn from the two classes, and nothing keeps the
//! representative in the same *order* as the original -- pool order is
//! suit-major, class order is rank order.  That is only harmless because
//! `placement_patterns` is now the ordered pair set: swapping which card goes
//! first is undone by swapping the pattern, and the pattern list is closed
//! under that swap, so the minimum over patterns is the same f64 either way.
//! Under the old unordered enumeration it was not, and the collapse would
//! have silently answered a different question.
//!
//! # What is bit-identical and what is not
//!
//! Each three-card term is: `V_T` for a rank multiset equals the old sweep's
//! `draw_min` for any card triple of that shape, **to the bit**, because the
//! `compose` calls and the pattern order are untouched and only the loop
//! bounds moved.  The *aggregate* cannot be: adding `v` k times and adding
//! `k*v` once are different summation trees.  With |V| <= ~140 and at most a
//! few thousand terms the difference lands around 1e-11, and the accepted
//! bound is 1e-9 absolute.  The strong check is therefore on the terms (a bit
//! multiset), not on the total; `tests` below does both.

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use super::row_memo::TerminalMemo;
use super::v4_first::{compose, placement_patterns};
use super::{Card, CoreBoard, Terminal};

const ROW_CAPACITY: [usize; 3] = [3, 5, 5];

/// The most live suits `solve_maybe_collapsed` will fire on.
///
/// One, not two: at `|L| = 2` the class alphabet shrinks the triple sweep by
/// only 1.58x, which is less than a collapsed term costs over a card term, and
/// the measured wall clock is **0.49x** -- twice as slow.  This is the same
/// judgement that already excluded `|L| >= 3` (1.00x shrink), applied to a
/// measurement rather than to arithmetic.  See
/// `ai/docs/live_suit_bucket_20260826.md` §4.8.
///
/// `collapse_state` itself is exact at any `|L|` and the unit checks still
/// exercise it at two; this constant governs the production path only.
pub(crate) const FIRE_MAX_LIVE: u32 = 1;

// ------------------------------------------------------------------
// Fire counters.  Reported to stderr, never to stdout or --output.
// ------------------------------------------------------------------

static V4_CALLS: AtomicU64 = AtomicU64::new(0);
static V4_FIRED: AtomicU64 = AtomicU64::new(0);
/// How many V4 state calls saw a live union of size 0, 1, 2 and 3-or-more.
/// Only counted with the flag on, where the union is computed anyway; with the
/// flag off nothing is measured and nothing is printed.
static V4_LIVE: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

pub(crate) fn note_call() {
    V4_CALLS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn note_fired() {
    V4_FIRED.fetch_add(1, Ordering::Relaxed);
}

/// Called once per eligible-shaped call, right after the union is known and
/// before the threshold decides anything -- so the histogram describes the
/// traffic, not the subset that fired.
pub(crate) fn note_live(union: u8) {
    let bin = (union.count_ones() as usize).min(3);
    V4_LIVE[bin].fetch_add(1, Ordering::Relaxed);
}

/// One line at the end of a run.  Printed with the flag off as well, so that
/// a shard launched without `--rank-collapse` says so instead of looking like
/// a shard where nothing was eligible.  The `[off]` line keeps its old bytes:
/// the histogram is only reported where it was actually measured.
pub(crate) fn report(enabled: bool) {
    let calls = V4_CALLS.load(Ordering::Relaxed);
    let fired = V4_FIRED.load(Ordering::Relaxed);
    let share = if calls == 0 {
        0.0
    } else {
        100.0 * fired as f64 / calls as f64
    };
    let histogram = if enabled {
        let bins: Vec<u64> = V4_LIVE.iter().map(|b| b.load(Ordering::Relaxed)).collect();
        format!(
            " L0 {} L1 {} L2 {} L3+ {}",
            bins[0], bins[1], bins[2], bins[3]
        )
    } else {
        String::new()
    };
    eprintln!(
        "rank-collapse: fired {fired}/{calls} ({share:.1}%){histogram} [{}]",
        if enabled { "on" } else { "off" }
    );
}

// ------------------------------------------------------------------
// Live suits
// ------------------------------------------------------------------

/// The suits a row could still reach a flush in, as a bitmask over
/// `Card::suit` (`s = 0`, `h = 1`, `d = 2`, `c = 3`).
///
/// - **No open slots: zero, made flush or not.**  "Live" here means the pool
///   can still feed the row's flush, and that question is meaningless for a
///   row nobody will add to.  This is a deliberate extension of the canonical
///   rule in `ai/docs/suit_collapse_20260824.md` §1, which counts a completed
///   flush as alive; it was worth +16.5 points of fire rate at `L = 0`
///   (58.8% -> 75.3% on real T3-BTN roots) and it is checked against a naive
///   card-level enumeration by `completed_flush_row_collapses` below.
/// - **No natural card** (empty, or nothing but jokers): all four suits, if
///   the row can still hold five of one.  A three-card top never can, and a
///   five-slot row with `len == jokers` always can -- though the eleven-card
///   boards V4 validates cannot realise one, since the other rows would
///   overflow.  Implemented anyway, on the sound side, and pinned by a
///   row-level test.
/// - **Two or more natural suits: zero.**  With `k >= 2` distinct suits among
///   `len - jokers` naturals, the best suit has at most `len - jokers - 1`, so
///   even with every joker and every open slot it reaches `capacity - 1`.
/// - **One natural suit `s`**: `1 << s` when `count_s + jokers + open >= 5`.
///
/// Jokers count toward the arithmetic and never toward the *choice* of suit:
/// a row of two jokers and one club is live in clubs, not in all four.
pub(crate) fn row_live_mask(cards: &[Card], capacity: usize) -> u8 {
    let open = capacity - cards.len();
    if open == 0 {
        return 0;
    }
    let mut by_suit = [0usize; 4];
    let mut jokers = 0usize;
    for card in cards {
        if card.is_joker() {
            jokers += 1;
        } else {
            by_suit[card.suit as usize] += 1;
        }
    }
    let present = (0..4u8).fold(0u8, |mask, suit| {
        mask | if by_suit[suit as usize] > 0 { 1 << suit } else { 0 }
    });
    match present.count_ones() {
        0 => {
            if jokers + open >= 5 {
                0b1111
            } else {
                0
            }
        }
        1 => {
            let suit = present.trailing_zeros() as usize;
            if by_suit[suit] + jokers + open >= 5 {
                present
            } else {
                0
            }
        }
        _ => 0,
    }
}

/// The union over the board's three rows.  Computed from the board alone: a
/// suit whose pool supply happens to be exhausted still counts, which only
/// ever splits classes further than needed and keeps the predicate -- and the
/// Python reference the gate bins with -- a function of the board.
pub(crate) fn board_live_mask(board: &CoreBoard) -> u8 {
    (0..3).fold(0u8, |mask, row| {
        mask | row_live_mask(&board.rows[row], ROW_CAPACITY[row])
    })
}

// ------------------------------------------------------------------
// Rank classes
// ------------------------------------------------------------------

/// The bucket of a card whose suit is not live.  Suits are 0..4, so 4 is free
/// and no natural can collide with it.
const OTHER: u8 = 4;

/// The class of one pool card: the joker class, or `(rank, bucket)` where the
/// bucket is the card's suit when that suit is live and `OTHER` otherwise.
///
/// At `L = 0` every natural buckets to `OTHER`, so `(rank, OTHER)` induces the
/// same partition -- and the same first-appearance order -- as the rank alone.
/// That degeneracy is load-bearing: it is what makes the generalised build
/// byte-identical to the shipped `L = 0` one, which is how the two versions
/// cross-check each other.  Anything that sorts, normalises or hashes these
/// keys breaks it.
#[inline]
fn class_key(card: &Card, live_union: u8) -> (u8, u8) {
    if card.is_joker() {
        // A natural rank is 2..=14, so rank 0 already separates the jokers
        // from every bucket; naming the key makes that explicit.
        (0, OTHER)
    } else if live_union & (1 << card.suit) != 0 {
        (card.rank, card.suit)
    } else {
        (card.rank, OTHER)
    }
}

/// The pool grouped by class.  Class order is first-appearance order in the
/// pool vector, which makes it a deterministic function of the request and
/// keeps no hash iteration anywhere near the numeric path.
///
/// Jokers form **one** class of their own, never merged with a natural rank:
/// on the board they are wild, in the pool they are a thirteenth kind of
/// card.  The two of them share a class because they are interchangeable in
/// evaluation -- the same fact `row_memo` already relies on when it collides
/// both jokers onto one cache id.  Drawing a joker cannot revive a suit
/// either: it fills an open slot and adds a wild, leaving `count + jokers +
/// open` where it was.
///
/// Class sizes are unchanged by the generalisation: a live class is one rank
/// in one suit and holds at most one card, an `OTHER` class at most four (and
/// fewer as `L` grows), the jokers two.  `binomial` still only needs `n <= 4`.
pub(crate) struct RankClasses {
    /// `count[c]` cards of class `c`; the counts sum to the pool size.
    pub(crate) count: Vec<usize>,
    /// `reps[c]` are the pool indices of class `c`, in pool order.  A context
    /// needing two cards of one class takes `reps[c][0]` and `reps[c][1]`,
    /// which are two *physically distinct* pool cards -- the collapse only
    /// ever builds boards that could really be dealt.
    pub(crate) reps: Vec<Vec<usize>>,
}

impl RankClasses {
    pub(crate) fn from_pool(pool: &[Card], live_union: u8) -> Self {
        let mut keys: Vec<(u8, u8)> = Vec::new();
        let mut count: Vec<usize> = Vec::new();
        let mut reps: Vec<Vec<usize>> = Vec::new();
        for (index, card) in pool.iter().enumerate() {
            let key = class_key(card, live_union);
            match keys.iter().position(|k| *k == key) {
                Some(class) => {
                    count[class] += 1;
                    reps[class].push(index);
                }
                None => {
                    keys.push(key);
                    count.push(1);
                    reps.push(vec![index]);
                }
            }
        }
        Self { count, reps }
    }

    pub(crate) fn len(&self) -> usize {
        self.count.len()
    }

    /// The two pool cards standing for an (unordered) class pair.
    fn pair_reps(&self, a: usize, b: usize) -> (usize, usize) {
        if a == b {
            (self.reps[a][0], self.reps[a][1])
        } else {
            (self.reps[a][0], self.reps[b][0])
        }
    }
}

/// `C[n][k]` for the sizes this sweep can produce: a class holds at most four
/// naturals of one rank (or two jokers) and a draw is three cards.
fn binomial(n: usize, k: usize) -> u64 {
    const TABLE: [[u64; 4]; 5] = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 2, 1, 0],
        [1, 3, 3, 1],
        [1, 4, 6, 4],
    ];
    if n >= TABLE.len() || k >= 4 {
        // Unreachable for a legal deck; returning zero would silently drop
        // terms, so say so.
        panic!("binomial({n}, {k}) is outside the deck's range");
    }
    TABLE[n][k]
}

/// Index of the unordered class pair `{a, b}`, `a == b` included.
fn pair_index(m: usize, a: usize, b: usize) -> usize {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    lo * m - lo * (lo.saturating_sub(1)) / 2 + (hi - lo)
}

fn pair_slots(m: usize) -> usize {
    m * (m + 1) / 2
}

// ------------------------------------------------------------------
// The collapsed sweep
// ------------------------------------------------------------------

/// What one candidate's three-card sweep produced.
struct Sweep {
    total: f64,
    /// Per class, the summed weighted value of the triples containing one
    /// *specific* card of that class -- what the discard subtraction needs.
    containing: Vec<f64>,
    /// `(V_T, W_T)` per rank multiset, filled only when recording; this is
    /// the bit multiset the strong unit check compares.
    terms: Vec<(f64, u64)>,
}

/// The output of a collapsed state solve, plus the bookkeeping a test needs
/// to line it up against a card-level enumeration.
pub(crate) struct Collapsed {
    pub(crate) value: f64,
    pub(crate) candidates: usize,
    /// Per candidate `(class a, class b, pattern)`; empty unless recording.
    pub(crate) keys: Vec<(usize, usize, usize)>,
    /// Per candidate, its `(V_T, W_T)` list; empty unless recording.
    pub(crate) terms: Vec<Vec<(f64, u64)>>,
}

/// The distinct classes of a sorted class triple with their multiplicities.
#[inline]
fn triple_parts(triple: [usize; 3]) -> ([usize; 3], [usize; 3], usize) {
    let mut class = [0usize; 3];
    let mut mult = [0usize; 3];
    let mut parts = 0usize;
    for value in triple {
        if parts > 0 && class[parts - 1] == value {
            mult[parts - 1] += 1;
        } else {
            class[parts] = value;
            mult[parts] = 1;
            parts += 1;
        }
    }
    (class, mult, parts)
}

fn candidate_sweep(
    term: &Terminal,
    left: &[usize],
    opp_terms: &[Vec<Terminal>],
    m: usize,
    fl_ev_table: &[f64; 4],
    record: bool,
) -> Sweep {
    // pairval[{c,d}] = hero's score when the opponent holds one card of class
    // c and one of class d and places them at its own best pattern.  Same
    // shape as the card-level table, with the class pair standing in for
    // every card pair of that shape -- which is the whole theorem.
    let mut pairval = vec![f64::INFINITY; pair_slots(m)];
    for c in 0..m {
        for d in c..m {
            let available = if c == d { left[c] >= 2 } else { left[c] >= 1 && left[d] >= 1 };
            if !available {
                continue;
            }
            let mut best = f64::INFINITY;
            for theirs in &opp_terms[pair_index(m, c, d)] {
                let score = compose(term, theirs, fl_ev_table);
                if score < best {
                    best = score;
                }
            }
            pairval[pair_index(m, c, d)] = best;
        }
    }

    let mut sweep = Sweep {
        total: 0.0,
        containing: vec![0.0; m],
        terms: Vec::new(),
    };
    let mut mass = 0u64;
    for c in 0..m {
        for d in c..m {
            for e in d..m {
                let (class, mult, parts) = triple_parts([c, d, e]);
                if (0..parts).any(|t| left[class[t]] < mult[t]) {
                    continue;
                }
                // The three pairs inside the triple, in the same positional
                // order the card-level sweep used.  Repeated classes make two
                // of the three coincide, which is exactly right: the pairs of
                // {c, c, e} are (c1,c2), (c1,e), (c2,e).
                let mut value = pairval[pair_index(m, c, d)];
                let other = pairval[pair_index(m, c, e)];
                if other < value {
                    value = other;
                }
                let other = pairval[pair_index(m, d, e)];
                if other < value {
                    value = other;
                }

                let mut weight = 1u64;
                for t in 0..parts {
                    weight *= binomial(left[class[t]], mult[t]);
                }
                sweep.total += (weight as f64) * value;
                mass += weight;
                for t in 0..parts {
                    // Triples of this shape containing one nominated card of
                    // class `class[t]`.  Built from binomials, never by
                    // dividing the total weight.
                    let mut share = binomial(left[class[t]] - 1, mult[t] - 1);
                    for u in 0..parts {
                        if u != t {
                            share *= binomial(left[class[u]], mult[u]);
                        }
                    }
                    sweep.containing[class[t]] += (share as f64) * value;
                }
                if record {
                    sweep.terms.push((value, weight));
                }
            }
        }
    }
    let members: usize = left.iter().sum();
    assert_eq!(
        mass,
        (members * (members - 1) * (members - 2) / 6) as u64,
        "collapsed triple weights do not sum to C({members}, 3)"
    );
    sweep
}

/// The exact V4 state value on an eligible board pair, swept over classes.
/// Mirrors `v4_first::solve_weighted`'s `draw == None`, uniform-pool branch
/// step for step; only the loop bounds change.
///
/// `live_union` is the caller's `board_live_mask(hero) | board_live_mask(opp)`,
/// passed in rather than recomputed so that the predicate the threshold
/// decided on is the predicate the alphabet is built from.
pub(crate) fn collapse_state(
    fl_ev_table: &[f64; 4],
    my_board: &CoreBoard,
    opp_board: &CoreBoard,
    pool: &[Card],
    live_union: u8,
    record: bool,
) -> Collapsed {
    let n = pool.len();
    let classes = RankClasses::from_pool(pool, live_union);
    let m = classes.len();

    // --- opponent completions, priced once per class pair ----------------
    let opp_patterns = placement_patterns(opp_board);
    let mut opp_memo = TerminalMemo::new(opp_board);
    let mut opp_terms: Vec<Vec<Terminal>> = vec![Vec::new(); pair_slots(m)];
    for c in 0..m {
        for d in c..m {
            if c == d && classes.count[c] < 2 {
                continue;
            }
            let (x, y) = classes.pair_reps(c, d);
            let mut per = Vec::with_capacity(opp_patterns.len());
            for pattern in &opp_patterns {
                per.push(opp_memo.terminal(&[(pattern[0], pool[x]), (pattern[1], pool[y])]));
            }
            opp_terms[pair_index(m, c, d)] = per;
        }
    }

    // --- hero candidates: every (class pair, ordered pattern) board ------
    let my_patterns = placement_patterns(my_board);
    let mut my_memo = TerminalMemo::new(my_board);
    // One slot per (class pair, pattern) so the lookup stays arithmetic;
    // pairs the pool cannot supply carry `None` and are never swept.
    let mut candidates: Vec<Option<Terminal>> = vec![None; pair_slots(m) * my_patterns.len()];
    let mut keys: Vec<(usize, usize, usize)> = vec![(0, 0, 0); candidates.len()];
    for a in 0..m {
        for b in a..m {
            if a == b && classes.count[a] < 2 {
                continue;
            }
            let (x, y) = classes.pair_reps(a, b);
            for (slot, pattern) in my_patterns.iter().enumerate() {
                let at = pair_index(m, a, b) * my_patterns.len() + slot;
                candidates[at] =
                    Some(my_memo.terminal(&[(pattern[0], pool[x]), (pattern[1], pool[y])]));
                keys[at] = (a, b, slot);
            }
        }
    }
    let candidate_at =
        |a: usize, b: usize, pattern: usize| pair_index(m, a, b) * my_patterns.len() + pattern;

    // --- per candidate: the opponent sweep, with per-class containing sums
    let c26_3 = ((n - 3) * (n - 4) * (n - 5) / 6) as f64;
    let swept: Vec<(Vec<f64>, Vec<(f64, u64)>)> = candidates
        .par_iter()
        .enumerate()
        .map(|(at, candidate)| {
            let Some(term) = candidate else {
                return (Vec::new(), Vec::new());
            };
            let (a, b, _) = keys[at];
            let mut left = classes.count.clone();
            left[a] -= 1;
            left[b] -= 1;
            let sweep = candidate_sweep(term, &left, &opp_terms, m, fl_ev_table, record);
            let values = (0..m)
                .map(|toss| {
                    if left[toss] == 0 {
                        f64::NAN
                    } else {
                        (sweep.total - sweep.containing[toss]) / c26_3
                    }
                })
                .collect();
            (values, sweep.terms)
        })
        .collect();

    // --- state value: the weighted mean over every C(n,3) draw -----------
    let mut total = 0.0f64;
    let mut mass = 0u64;
    for c in 0..m {
        for d in c..m {
            for e in d..m {
                let (class, mult, parts) = triple_parts([c, d, e]);
                if (0..parts).any(|t| classes.count[class[t]] < mult[t]) {
                    continue;
                }
                let mut weight = 1u64;
                for t in 0..parts {
                    weight *= binomial(classes.count[class[t]], mult[t]);
                }
                let mut best = f64::NEG_INFINITY;
                for t in 0..parts {
                    // Discard one card of class[t]; the other two are kept,
                    // already in ascending class order.
                    let mut kept = [0usize; 2];
                    let mut filled = 0usize;
                    for u in 0..parts {
                        let take = mult[u] - usize::from(u == t);
                        for _ in 0..take {
                            kept[filled] = class[u];
                            filled += 1;
                        }
                    }
                    for pattern in 0..my_patterns.len() {
                        let value = swept[candidate_at(kept[0], kept[1], pattern)].0[class[t]];
                        if value > best {
                            best = value;
                        }
                    }
                }
                total += (weight as f64) * best;
                mass += weight;
            }
        }
    }
    assert_eq!(
        mass,
        (n * (n - 1) * (n - 2) / 6) as u64,
        "collapsed draw weights do not sum to C({n}, 3)"
    );

    Collapsed {
        value: total / (mass as f64),
        // Deliberately the *collapsed* count, which is not the card-level
        // `candidates` the uncollapsed response reports.  Only `--v4-first`
        // prints it, and only when the flag is on.
        candidates: candidates.iter().filter(|c| c.is_some()).count(),
        keys: if record { keys } else { Vec::new() },
        terms: swept.into_iter().map(|(_, terms)| terms).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4_first::V4FirstRequest;
    use crate::{all_cards, to_core_card, BoardStr};
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Mutex;

    const TABLE: [f64; 4] = [3.17, 16.61, 38.76, 70.07];

    /// The fire counters are process-wide, so the tests that read them run one
    /// at a time.  Poisoning is ignored: a panic in one of them is already a
    /// failure and must not cascade into the others.
    static COUNTERS: Mutex<()> = Mutex::new(());

    fn board(top: &[&str], middle: &[&str], bottom: &[&str]) -> BoardStr {
        BoardStr {
            top: top.iter().map(|s| s.to_string()).collect(),
            middle: middle.iter().map(|s| s.to_string()).collect(),
            bottom: bottom.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn core(top: &[&str], middle: &[&str], bottom: &[&str]) -> CoreBoard {
        CoreBoard::from_str_board(&board(top, middle, bottom)).expect("board")
    }

    fn cards(names: &[&str]) -> Vec<Card> {
        names.iter().map(|n| to_core_card(n).expect("card")).collect()
    }

    /// Suit bits by name, so a fixture states what it expects in the same
    /// alphabet the request is written in.
    fn suits(letters: &str) -> u8 {
        letters.chars().fold(0u8, |mask, letter| {
            mask | 1
                << match letter {
                    's' => 0,
                    'h' => 1,
                    'd' => 2,
                    'c' => 3,
                    other => panic!("no suit {other}"),
                }
        })
    }

    /// A fixture has to be a deal that could really happen: no card twice
    /// across the two boards and the pool, and no more than two jokers.  A
    /// duplicate would let the naive reference and the collapse agree on a
    /// board neither of them should ever see.
    fn assert_dealable(case: &str, my_board: &CoreBoard, opp_board: &CoreBoard, pool: &[Card]) {
        let mut seen: BTreeSet<(u8, u8)> = BTreeSet::new();
        let mut jokers = 0usize;
        let mut note = |card: &Card, place: &str| {
            if card.is_joker() {
                jokers += 1;
            } else {
                assert!(
                    seen.insert((card.rank, card.suit)),
                    "{case}: a card repeats in {place}"
                );
            }
        };
        for row in &my_board.rows {
            for card in row {
                note(card, "hero's board");
            }
        }
        for row in &opp_board.rows {
            for card in row {
                note(card, "the opponent's board");
            }
        }
        for card in pool {
            note(card, "the pool");
        }
        assert!(jokers <= 2, "{case}: the deck holds two jokers, not {jokers}");
    }

    /// The same pool with every live-suit card pushed into a dead suit of its
    /// own rank -- the "move the live suit" control of
    /// `scratchpad/suit_parity_l1.py`, written so any fixture can produce one.
    fn deadened(
        my_board: &CoreBoard,
        opp_board: &CoreBoard,
        pool: &[Card],
        union: u8,
    ) -> Vec<Card> {
        let mut taken: BTreeSet<(u8, u8)> = BTreeSet::new();
        for board in [my_board, opp_board] {
            for row in &board.rows {
                for card in row {
                    if !card.is_joker() {
                        taken.insert((card.rank, card.suit));
                    }
                }
            }
        }
        for card in pool {
            if !card.is_joker() {
                taken.insert((card.rank, card.suit));
            }
        }
        pool.iter()
            .map(|card| {
                if card.is_joker() || union & (1 << card.suit) == 0 {
                    return *card;
                }
                let free = (0..4u8)
                    .find(|suit| {
                        union & (1 << suit) == 0 && !taken.contains(&(card.rank, *suit))
                    })
                    .unwrap_or_else(|| {
                        panic!("rank {} has no free dead suit to move into", card.rank)
                    });
                taken.insert((card.rank, free));
                Card {
                    rank: card.rank,
                    suit: free,
                }
            })
            .collect()
    }

    /// `scratchpad/suit_parity_l1.py::live_suits`, transcribed one row at a
    /// time.  `closed_is_dead` picks the extension `row_live_mask` implements
    /// -- a row with no open slots contributes nothing -- over the canonical
    /// rule of `ai/docs/suit_collapse_20260824.md` §1, under which a *made*
    /// flush still counts as live.  Closed rows are the only place the two
    /// disagree, which `row_live_mask_matches_the_python_reference` asserts.
    fn reference_row_live(names: &[&str], capacity: usize, closed_is_dead: bool) -> u8 {
        if capacity < 5 {
            return 0;
        }
        let open = capacity - names.len();
        if closed_is_dead && open == 0 {
            return 0;
        }
        let mut by_suit = [0usize; 4];
        let mut jokers = 0usize;
        for name in names {
            if name.starts_with('X') {
                jokers += 1;
            } else {
                by_suit[match name.chars().last().unwrap() {
                    's' => 0,
                    'h' => 1,
                    'd' => 2,
                    'c' => 3,
                    other => panic!("no suit {other}"),
                }] += 1;
            }
        }
        let present: Vec<usize> = (0..4).filter(|suit| by_suit[*suit] > 0).collect();
        if present.is_empty() {
            return if open + jokers >= 5 { 0b1111 } else { 0 };
        }
        if present.len() == 1 && by_suit[present[0]] + jokers + open >= 5 {
            1 << present[0]
        } else {
            0
        }
    }

    // --------------------------------------------------------------
    // The independent card-level reference.
    // --------------------------------------------------------------

    /// The uncollapsed sweep, written here as a plain card triple loop that
    /// calls `TerminalMemo::terminal` and `compose` directly.  It shares no
    /// generator with the collapse: no classes, no multiset weights, no
    /// binomials, and no notion of a live suit -- it is exact whether or not
    /// the fixture is eligible, which is what lets it judge the eligibility
    /// rule as well as the sweep.  Returns the state value and, per hero
    /// candidate board (keyed by the pool pair and pattern), the bit multiset
    /// of the three-card values that candidate's sweep produced.
    #[allow(clippy::type_complexity)]
    fn naive_state(
        my_board: &CoreBoard,
        opp_board: &CoreBoard,
        pool: &[Card],
    ) -> (f64, BTreeMap<(usize, usize, usize), BTreeMap<u64, u64>>) {
        let n = pool.len();
        let my_patterns = placement_patterns(my_board);
        let opp_patterns = placement_patterns(opp_board);
        let mut my_memo = TerminalMemo::new(my_board);
        let mut opp_memo = TerminalMemo::new(opp_board);

        // value[(i, j, pattern)][d] and the term multiset per candidate.
        let mut value: BTreeMap<(usize, usize, usize), Vec<f64>> = BTreeMap::new();
        let mut terms: BTreeMap<(usize, usize, usize), BTreeMap<u64, u64>> = BTreeMap::new();
        for i in 0..n {
            for j in (i + 1)..n {
                for (slot, pattern) in my_patterns.iter().enumerate() {
                    let mine =
                        my_memo.terminal(&[(pattern[0], pool[i]), (pattern[1], pool[j])]);
                    let members: Vec<usize> =
                        (0..n).filter(|k| *k != i && *k != j).collect();
                    let mut total = 0.0f64;
                    let mut containing = vec![0.0f64; n];
                    let mut bits: BTreeMap<u64, u64> = BTreeMap::new();
                    for a in 0..members.len() {
                        for b in (a + 1)..members.len() {
                            for c in (b + 1)..members.len() {
                                let (p, q, r) = (members[a], members[b], members[c]);
                                let mut low = f64::INFINITY;
                                for (x, y) in [(p, q), (p, r), (q, r)] {
                                    for op in &opp_patterns {
                                        let theirs = opp_memo
                                            .terminal(&[(op[0], pool[x]), (op[1], pool[y])]);
                                        let score = compose(&mine, &theirs, &TABLE);
                                        if score < low {
                                            low = score;
                                        }
                                    }
                                }
                                total += low;
                                containing[p] += low;
                                containing[q] += low;
                                containing[r] += low;
                                *bits.entry(low.to_bits()).or_insert(0) += 1;
                            }
                        }
                    }
                    let divisor = ((n - 3) * (n - 4) * (n - 5) / 6) as f64;
                    value.insert(
                        (i, j, slot),
                        (0..n)
                            .map(|d| {
                                if d == i || d == j {
                                    f64::NAN
                                } else {
                                    (total - containing[d]) / divisor
                                }
                            })
                            .collect(),
                    );
                    terms.insert((i, j, slot), bits);
                }
            }
        }

        let mut total = 0.0f64;
        let mut draws = 0u64;
        for x in 0..n {
            for y in (x + 1)..n {
                for z in (y + 1)..n {
                    let mut best = f64::NEG_INFINITY;
                    for (i, j, d) in [(x, y, z), (x, z, y), (y, z, x)] {
                        for pattern in 0..my_patterns.len() {
                            let candidate = value[&(i.min(j), i.max(j), pattern)][d];
                            if candidate > best {
                                best = candidate;
                            }
                        }
                    }
                    total += best;
                    draws += 1;
                }
            }
        }
        (total / draws as f64, terms)
    }

    /// Both sides of one fixture, with the strong check in the middle: for
    /// every collapsed candidate, the `(V_T, W_T)` multiset must equal the
    /// bit multiset the card-level sweep produced for the same final board.
    ///
    /// Two guards keep a fixture from passing vacuously.  Its sweep has to
    /// produce a spread of values, not one repeated constant (an all-fouling
    /// board agrees with anything).  And where a suit is live, moving the
    /// pool's live cards into dead suits has to move the card-level value:
    /// otherwise the fixture cannot tell a bucket from a rank and proves
    /// nothing about the alphabet.
    fn check(
        case: &str,
        my_board: &CoreBoard,
        opp_board: &CoreBoard,
        pool: &[Card],
        expect: u8,
    ) {
        assert_dealable(case, my_board, opp_board, pool);
        let union = board_live_mask(my_board) | board_live_mask(opp_board);
        assert_eq!(union, expect, "{case}: live union is {union:#06b}");
        // Deliberately *not* FIRE_MAX_LIVE: `collapse_state` is exact at any
        // `|L|` and the fixtures keep covering two live suits after the
        // production threshold dropped to one.  Two is simply as wide as the
        // fixtures below go.
        assert!(union.count_ones() <= 2, "{case}: fixture is wider than L = 2");
        let collapsed = collapse_state(&TABLE, my_board, opp_board, pool, union, true);
        let (naive_value, naive_terms) = naive_state(my_board, opp_board, pool);
        let classes = RankClasses::from_pool(pool, union);
        let my_patterns = placement_patterns(my_board);

        let mut checked = 0usize;
        for (at, terms) in collapsed.terms.iter().enumerate() {
            if terms.is_empty() {
                continue;
            }
            let (a, b, pattern) = collapsed.keys[at];
            let (x, y) = classes.pair_reps(a, b);
            let key = (x.min(y), x.max(y), pattern);
            let mine: BTreeMap<u64, u64> =
                terms.iter().fold(BTreeMap::new(), |mut acc, (value, weight)| {
                    *acc.entry(value.to_bits()).or_insert(0) += weight;
                    acc
                });
            let theirs = naive_terms
                .get(&key)
                .unwrap_or_else(|| panic!("{case}: no naive sweep for {key:?}"));
            assert_eq!(
                mine, *theirs,
                "{case}: candidate {key:?} triple-value bit multiset differs"
            );
            checked += 1;
        }
        assert_eq!(
            checked,
            collapsed.candidates,
            "{case}: not every collapsed candidate was compared"
        );
        assert!(
            my_patterns.len() >= 1,
            "{case}: fixture has no legal placement"
        );
        // The opponent's minimum flattens a lot, so the spread is small even
        // on a lively board; what carries the information is the *weights* in
        // the multiset above.  This only rules out the degenerate fixture
        // where every completion fouls and one constant would satisfy any
        // alphabet at all.
        let spread: BTreeSet<u64> = naive_terms
            .values()
            .flat_map(|bits| bits.keys().copied())
            .collect();
        assert!(
            spread.len() >= 3,
            "{case}: the sweep produced {} distinct values -- too flat to judge an alphabet",
            spread.len()
        );
        let gap = (collapsed.value - naive_value).abs();
        assert!(
            gap <= 1e-12,
            "{case}: state value {} vs naive {} (gap {gap:.3e})",
            collapsed.value,
            naive_value
        );

        if union != 0 {
            let moved = deadened(my_board, opp_board, pool, union);
            assert_dealable(&format!("{case} (live suit moved)"), my_board, opp_board, &moved);
            let (shifted, _) = naive_state(my_board, opp_board, &moved);
            assert!(
                (naive_value - shifted).abs() > 1e-9,
                "{case}: emptying the live suit left the value at {naive_value} -- \
                 the fixture cannot tell a bucket from a rank"
            );
            assert!(
                classes.len() > RankClasses::from_pool(pool, 0).len(),
                "{case}: the live suit split no rank, so the bucket is untested"
            );
        }
    }

    // --------------------------------------------------------------
    // Shared fixtures.
    // --------------------------------------------------------------

    /// Both boards dead: the `L = 0` shape the first version shipped on.
    fn dead_hero() -> CoreBoard {
        core(&["Ah", "Kd", "Qc"], &["7c", "7d", "2h", "3s"], &["Ts", "Jh", "4d", "9c"])
    }

    fn dead_opponent() -> CoreBoard {
        core(&["Qh", "Qd", "Jc"], &["5c", "5d", "Kc", "8h"], &["Ac", "Ad", "As", "Kh"])
    }

    /// A board whose middle is four clubs in sequence with one slot open: the
    /// live suit decides between a straight flush, a flush and nothing, so the
    /// bucket has real work to do.  The bottom is quads, which outranks any
    /// flush, so every completion is a legal board rather than a foul.
    fn club_live_board() -> CoreBoard {
        core(&["2h", "3s", "5d"], &["7c", "4c", "5c", "6c"], &["Ac", "Ad", "As", "Ah"])
    }

    /// Its dead partner: a pair, two kings and trip tens, none of them one
    /// suit away from anything.
    fn dead_partner() -> CoreBoard {
        core(&["8h", "8d", "2s"], &["Kh", "Kd", "9s", "Jc"], &["Ts", "Td", "Th", "Jh"])
    }

    /// Three ranks that hold a club and a non-club apiece, so the bucket has
    /// to split them, plus ranks living entirely in one bucket.  The 3c makes
    /// hero's middle a straight flush and the Kc makes it a king-high flush.
    fn club_live_pool() -> Vec<Card> {
        cards(&["3c", "3h", "9c", "9h", "Kc", "Qh", "Qs", "7h", "7s", "6d", "6h", "2d"])
    }

    // --------------------------------------------------------------
    // (a)-(e): the strong check at L = 0, unchanged from the first version.
    // --------------------------------------------------------------

    /// (a) No jokers anywhere: the plain case.
    #[test]
    fn jokerless_dead_boards_collapse() {
        check(
            "a/jokerless",
            &dead_hero(),
            &dead_opponent(),
            &cards(&["2c", "2d", "3c", "3d", "4c", "4h", "5s", "6s", "6h", "8s", "9s", "Th"]),
            0,
        );
    }

    /// (b) Two jokers in the pool: one class of count two, never merged with
    /// a natural rank, and both of them reachable as a pair.
    #[test]
    fn pool_jokers_form_their_own_class() {
        let pool = cards(&["2c", "2d", "3c", "3d", "4c", "4h", "5s", "6s", "9s", "Th", "X1", "X2"]);
        let classes = RankClasses::from_pool(&pool, 0);
        assert_eq!(
            classes.count.iter().filter(|c| **c == 2).count(),
            4,
            "expected three natural pairs (2, 3, 4) plus the joker class"
        );
        assert_eq!(*classes.count.last().unwrap(), 2, "jokers share one class");
        check("b/pool jokers", &dead_hero(), &dead_opponent(), &pool, 0);
    }

    /// (c) A joker on the board's bottom row: the final top and middle stay
    /// joker-free, so `row_memo` takes its independent-rows fast path.  The
    /// pool is joker-free so it stays there for every completion.
    #[test]
    fn board_joker_in_bottom_fast_path() {
        check(
            "c/bottom joker",
            &core(&["Ah", "Kd", "Qc"], &["7c", "7d", "2h", "3s"], &["Ts", "Jh", "4d", "X1"]),
            &dead_opponent(),
            &cards(&["2c", "2d", "3c", "3d", "4c", "4h", "5s", "6s", "6h", "8s", "9s", "Th"]),
            0,
        );
    }

    /// (d) A joker on the board's middle row: every terminal goes through the
    /// constrained joint evaluation, `row_memo`'s slow path.
    #[test]
    fn board_joker_in_middle_slow_path() {
        check(
            "d/middle joker",
            &core(&["Ah", "Kd", "Qc"], &["7c", "7d", "2h", "X1"], &["Ts", "Jh", "4d", "9c"]),
            &dead_opponent(),
            &cards(&["2c", "2d", "3c", "3d", "4c", "4h", "5s", "6s", "6h", "8s", "9s", "Th"]),
            0,
        );
    }

    /// (e) The extension: a **completed** made flush on the bottom.  The
    /// canonical rule calls that row alive; it has no open slots, so nothing
    /// the pool does can move it, and the collapse must still be exact.
    #[test]
    fn completed_flush_row_collapses() {
        let my_board = core(&["Ah", "Kd", "Qc"], &["7c", "7d", "2h"], &["Ts", "Js", "9s", "8s", "6s"]);
        assert_eq!(
            reference_row_live(&["Ts", "Js", "9s", "8s", "6s"], 5, false),
            suits("s"),
            "the fixture's bottom row must be a completed flush"
        );
        assert_eq!(
            board_live_mask(&my_board),
            0,
            "the extension must keep a closed row's suit out of L"
        );
        check(
            "e/completed flush",
            &my_board,
            &dead_opponent(),
            &cards(&["2c", "2d", "3c", "3d", "4c", "4h", "5h", "6h", "9c", "Th", "Td", "Jh"]),
            0,
        );
    }

    // --------------------------------------------------------------
    // (h)-(n): the generalisation, one live shape at a time.
    // --------------------------------------------------------------

    /// (h) `L = {c}` from hero's side.  Clubs stand alone, every other suit
    /// shares its rank's bucket.
    #[test]
    fn one_live_suit_on_heros_board() {
        let hero = club_live_board();
        assert_eq!(board_live_mask(&hero), suits("c"));
        assert_eq!(board_live_mask(&dead_partner()), 0);
        let pool = club_live_pool();
        assert_eq!(
            RankClasses::from_pool(&pool, suits("c")).len(),
            9,
            "seven ranks, three of them split by the bucket"
        );
        assert_eq!(RankClasses::from_pool(&pool, 0).len(), 7, "seven ranks");
        check("h/live hero", &hero, &dead_partner(), &pool, suits("c"));
    }

    /// (i) The union's whole point: hero is dead and the **opponent** is the
    /// live one, so the alphabet still has to keep clubs apart.  The first
    /// version refused this board outright.  Same cards as (h) with the seats
    /// exchanged, which is the cleanest way to isolate the union from the
    /// board.
    #[test]
    fn one_live_suit_on_the_opponents_board_only() {
        let hero = dead_partner();
        let opp = club_live_board();
        assert_eq!(board_live_mask(&hero), 0, "hero is dead");
        assert_eq!(board_live_mask(&opp), suits("c"));
        check(
            "i/live opponent only",
            &hero,
            &opp,
            &club_live_pool(),
            suits("c"),
        );
    }

    /// (j) Both boards live in the **same** suit, eating the same pool: the
    /// union is one suit, and the shared club supply has to price right for
    /// whichever side draws it.  Each side's middle is one club from a
    /// straight flush, and they want different clubs.
    #[test]
    fn both_boards_live_in_the_same_suit() {
        let hero = club_live_board();
        let opp = core(&["8h", "8d", "2s"], &["9c", "Tc", "Jc", "Qc"], &["Ks", "Kd", "Kh", "Kc"]);
        assert_eq!(board_live_mask(&hero), suits("c"));
        assert_eq!(board_live_mask(&opp), suits("c"));
        check(
            "j/same suit contested",
            &hero,
            &opp,
            &cards(&["3c", "3h", "8c", "4h", "9h", "9s", "Th", "Ts", "Qh", "Qs", "Jh", "Js"]),
            suits("c"),
        );
    }

    /// (k) `L = {s, c}`: one live suit from each board.  This was the widest
    /// shape the threshold admitted until the wall clock came back at 0.49x
    /// and `FIRE_MAX_LIVE` dropped to one, so **production no longer takes
    /// this path** -- `past_the_threshold_the_flag_changes_nothing` checks
    /// that it does not.  The case stays because the algorithm is still exact
    /// here, and a two-suit alphabet is the only thing that exercises two
    /// live buckets side by side; if the threshold ever moves back, the
    /// evidence that it may is already written down.
    #[test]
    fn two_live_suits_one_from_each_board() {
        let hero = club_live_board();
        let opp = core(&["7h", "7d", "2s"], &["9s", "Ts", "Js", "Qs"], &["Ks", "Kd", "Kh", "Kc"]);
        assert_eq!(board_live_mask(&hero), suits("c"));
        assert_eq!(board_live_mask(&opp), suits("s"));
        check(
            "k/two live suits",
            &hero,
            &opp,
            &cards(&["3c", "3h", "8c", "8s", "4h", "4d", "9h", "9d", "Th", "Td", "Jh", "Jd"]),
            suits("sc"),
        );
    }

    /// (l) A joker sits in the live row.  It counts toward the five the row
    /// needs and contributes no suit of its own: three clubs and a wild is
    /// live in clubs, not in all four.
    #[test]
    fn a_joker_in_the_live_row_adds_no_suit() {
        let hero = core(&["2h", "3s", "5d"], &["4c", "5c", "6c", "X1"], &["Ac", "Ad", "As", "Ah"]);
        assert_eq!(
            row_live_mask(&cards(&["4c", "5c", "6c", "X1"]), 5),
            suits("c"),
            "the wild joins clubs, it does not open the other three"
        );
        assert_eq!(board_live_mask(&hero), suits("c"));
        check(
            "l/joker in the live row",
            &hero,
            &core(&["7h", "7d", "2s"], &["Kh", "Kd", "9s", "Jc"], &["Ts", "Td", "Th", "Jh"]),
            &cards(&["3c", "3h", "8c", "8s", "9c", "9h", "Qh", "Qs", "6d", "6h", "4h", "4d"]),
            suits("c"),
        );
    }

    /// (m) A completed flush next to a live row.  The made flush's suit must
    /// stay out of `L` -- otherwise the extension is gone, and with it the
    /// fire rate it bought.
    #[test]
    fn a_completed_flush_beside_a_live_row() {
        let hero = core(&["2h", "3s", "5d"], &["4c", "5c", "6c"], &["9d", "Td", "Jd", "Qd", "Kd"]);
        assert_eq!(
            reference_row_live(&["9d", "Td", "Jd", "Qd", "Kd"], 5, false),
            suits("d"),
            "the bottom is a made flush under the canonical rule"
        );
        assert_eq!(
            board_live_mask(&hero),
            suits("c"),
            "clubs are live and the completed diamond flush is not"
        );
        check(
            "m/completed flush beside a live row",
            &hero,
            &core(&["7h", "7d", "2s"], &["Kh", "Ks", "9s", "Jc"], &["Ts", "Th", "Tc", "Jh"]),
            &cards(&["3c", "3h", "7c", "8c", "8h", "9c", "Js", "Qh", "Qs", "2d", "4h", "4d"]),
            suits("c"),
        );
    }

    /// (n) Pool jokers under a live suit: still one class of two, while the
    /// naturals of a rank split into its club and its everything-else.
    #[test]
    fn pool_jokers_stay_one_class_under_a_live_suit() {
        let pool = cards(&["3c", "3h", "9c", "9h", "Qh", "Qs", "6d", "6h", "X1", "X2"]);
        let classes = RankClasses::from_pool(&pool, suits("c"));
        assert_eq!(
            classes.count,
            vec![1, 1, 1, 1, 2, 2, 2],
            "3c | 3h | 9c | 9h | QhQs | 6d6h | jokers"
        );
        assert_eq!(classes.reps.last().unwrap(), &vec![8, 9], "and those are the jokers");
        check(
            "n/pool jokers under a live suit",
            &club_live_board(),
            &dead_partner(),
            &pool,
            suits("c"),
        );
    }

    // --------------------------------------------------------------
    // (q) The parity claim, at the level the collapse actually reads.
    // --------------------------------------------------------------

    #[test]
    fn non_live_suits_are_interchangeable_and_live_ones_are_not() {
        let hero = club_live_board();
        let opp = dead_partner();
        let union = board_live_mask(&hero) | board_live_mask(&opp);
        assert_eq!(union, suits("c"));

        let base = club_live_pool();
        // Every non-club reassigned to another non-club, positions kept.  The
        // class partition and its order are untouched, so the collapse must
        // return the same f64 -- to the bit, not to a tolerance.  A
        // representative synthesised from the rank instead of taken from
        // `reps` would land on a club here and diverge.
        let permuted =
            cards(&["3c", "3d", "9c", "9d", "Kc", "Qd", "Qh", "7d", "7h", "6s", "6d", "2d"]);
        assert_dealable("q/permuted", &hero, &opp, &permuted);
        assert_eq!(
            RankClasses::from_pool(&permuted, union).count,
            RankClasses::from_pool(&base, union).count,
            "the permutation must not repartition the pool"
        );
        let before = collapse_state(&TABLE, &hero, &opp, &base, union, false);
        let after = collapse_state(&TABLE, &hero, &opp, &permuted, union, false);
        assert_eq!(
            before.value.to_bits(),
            after.value.to_bits(),
            "permuting the dead suits moved the value ({} vs {})",
            before.value,
            after.value
        );

        // And the check has teeth: empty the pool of clubs and the exact
        // card-level enumeration -- which knows nothing about buckets -- says
        // the board is worth something else.
        let moved = deadened(&hero, &opp, &base, union);
        let (plain, _) = naive_state(&hero, &opp, &base);
        let (shifted, _) = naive_state(&hero, &opp, &moved);
        assert!(
            (plain - shifted).abs() > 1e-9,
            "moving the live suit must change the value ({plain} vs {shifted})"
        );
    }

    /// The same statement one level down, where the first version made it: a
    /// board with a live flush row is suit-dependent and a dead one is not.
    /// Kept because the collapse now *fires* on live boards, so this is the
    /// fact the bucket has to be earning.
    #[test]
    fn a_live_flush_row_is_suit_dependent_and_a_dead_one_is_not() {
        // Middle holds four hearts with a slot open.  The top is deliberately
        // weak and the bottom is quads, so a completed flush is a *legal*
        // board and not a fouled one -- otherwise every completion busts and
        // the control would have no power.
        let live = core(&["2c", "3d", "5c"], &["7h", "2h", "3h", "4h"], &["Ac", "Ad", "As", "Ah"]);
        assert_eq!(board_live_mask(&live), suits("h"), "hearts are live");

        let opp = core(&["Kc", "Kd", "Qs"], &["5d", "5s", "8c", "9d"], &["Tc", "Td", "Ts", "6d"]);
        assert_eq!(board_live_mask(&opp), 0, "the opponent side must be dead");
        let base = cards(&["6h", "8h", "9h", "Jh", "6c", "8s", "9s", "Js", "4c", "4d"]);
        // The same ranks with every heart moved to another suit.
        let moved = cards(&["6s", "8d", "9c", "Jd", "6c", "8s", "9s", "Js", "4c", "4d"]);
        let (before, _) = naive_state(&live, &opp, &base);
        let (after, _) = naive_state(&live, &opp, &moved);
        assert!(
            (before - after).abs() > 1e-9,
            "a live board must be suit-dependent (before {before}, after {after})"
        );

        let dead = core(&["2c", "3d", "5c"], &["7h", "2h", "3s", "4s"], &["Ac", "Ad", "As", "Ah"]);
        assert_eq!(board_live_mask(&dead), 0);
        let (dead_before, _) = naive_state(&dead, &opp, &base);
        let (dead_after, _) = naive_state(&dead, &opp, &moved);
        assert_eq!(
            dead_before.to_bits(),
            dead_after.to_bits(),
            "a dead board must be bit-invariant under suit reassignment"
        );
    }

    // --------------------------------------------------------------
    // (r) The degeneracy the cross-version byte gate rests on.
    // --------------------------------------------------------------

    #[test]
    fn at_no_live_suits_the_key_is_the_rank_alone() {
        /// The first version's key, rewritten here rather than called, so the
        /// two are pinned by comparison instead of by construction.
        fn rank_only(pool: &[Card]) -> (Vec<usize>, Vec<Vec<usize>>) {
            let mut keys: Vec<u8> = Vec::new();
            let mut count: Vec<usize> = Vec::new();
            let mut reps: Vec<Vec<usize>> = Vec::new();
            for (index, card) in pool.iter().enumerate() {
                let key = if card.is_joker() { 0 } else { card.rank };
                match keys.iter().position(|k| *k == key) {
                    Some(class) => {
                        count[class] += 1;
                        reps[class].push(index);
                    }
                    None => {
                        keys.push(key);
                        count.push(1);
                        reps.push(vec![index]);
                    }
                }
            }
            (count, reps)
        }

        for pool in [
            club_live_pool(),
            cards(&["2c", "2d", "3c", "3d", "4c", "4h", "5s", "6s", "9s", "Th", "X1", "X2"]),
            cards(&["As", "Ah", "Ad", "Ac", "X1", "2s", "X2", "2h"]),
        ] {
            let classes = RankClasses::from_pool(&pool, 0);
            let (count, reps) = rank_only(&pool);
            assert_eq!(classes.count, count, "class sizes and their order");
            assert_eq!(classes.reps, reps, "representatives and their order");
        }
    }

    // --------------------------------------------------------------
    // (g), (g'), (p): the predicate itself.
    // --------------------------------------------------------------

    #[test]
    fn row_live_mask_matches_the_python_reference() {
        // (row cards, capacity, expected mask).  The five-slot pairs straddle
        // the boundary: one card of difference decides whether a suit is
        // still reachable.
        let cases: [(&[&str], usize, u8); 16] = [
            // Three-card top: no flush is scored there, ever.
            (&[], 3, 0),
            (&["As", "Ks"], 3, 0),
            (&["As", "Ks", "Qs"], 3, 0),
            // Two open, three of a suit reachable versus four.
            (&["2s", "3s", "4h"], 5, 0),
            (&["2s", "3s", "4s"], 5, suits("s")),
            // One open.
            (&["2s", "3s", "4s", "5h"], 5, 0),
            (&["2s", "3s", "4s", "5s"], 5, suits("s")),
            // A joker in the row counts toward the suit it would join and
            // opens no suit of its own.
            (&["2h", "3s", "X1"], 5, 0),
            (&["2s", "3s", "X1"], 5, suits("s")),
            (&["2s", "3s", "4h", "X1"], 5, 0),
            (&["2s", "3s", "4s", "X1"], 5, suits("s")),
            (&["2c", "X1", "X2"], 5, suits("c")),
            // (p) No natural at all: every suit is still reachable.
            (&[], 5, 0b1111),
            (&["X1"], 5, 0b1111),
            (&["X1", "X2"], 5, 0b1111),
            // Closed and not a flush: dead under both rules.
            (&["2s", "3s", "4h", "5d", "6c"], 5, 0),
        ];
        for (names, capacity, expected) in cases {
            let row = cards(names);
            assert_eq!(
                row_live_mask(&row, capacity),
                expected,
                "row_live_mask({names:?}, {capacity})"
            );
            assert_eq!(
                row_live_mask(&row, capacity),
                reference_row_live(names, capacity, true),
                "row_live_mask disagrees with the python reference on {names:?}"
            );
            // (g') The canonical rule of the parity script agrees everywhere
            // except on closed rows, which is the whole of the extension.
            if capacity - names.len() > 0 {
                assert_eq!(
                    reference_row_live(names, capacity, false),
                    expected,
                    "the extension must only touch closed rows, not {names:?}"
                );
            }
        }

        // The one documented divergence, stated as an equality rather than an
        // inequality: canon calls a made flush live, the collapse does not.
        let made = ["Ts", "Js", "9s", "8s", "6s"];
        assert_eq!(reference_row_live(&made, 5, false), suits("s"));
        assert_eq!(row_live_mask(&cards(&made), 5), 0);
        // A closed row of three clubs and two wilds is a made flush too.
        let wild = ["2c", "5c", "9c", "X1", "X2"];
        assert_eq!(reference_row_live(&wild, 5, false), suits("c"));
        assert_eq!(row_live_mask(&cards(&wild), 5), 0);
    }

    #[test]
    fn board_live_mask_is_the_union_of_its_rows() {
        // Two live rows on one board, which is how `L` reaches two from a
        // single side.
        let both = core(&["Ah", "Kh", "Qh"], &["2c", "5c", "9c", "Tc"], &["3d", "6d", "8d", "Jd"]);
        assert_eq!(row_live_mask(&both.rows[0], 3), 0);
        assert_eq!(row_live_mask(&both.rows[1], 5), suits("c"));
        assert_eq!(row_live_mask(&both.rows[2], 5), suits("d"));
        assert_eq!(board_live_mask(&both), suits("cd"));
    }

    // --------------------------------------------------------------
    // Plumbing: the collapse fires where it is licensed and nowhere else.
    // --------------------------------------------------------------

    fn request(
        id: &str,
        hero: BoardStr,
        opp: BoardStr,
        dead: &[&str],
        draw: Option<&[&str]>,
    ) -> V4FirstRequest {
        V4FirstRequest {
            id: id.into(),
            board: hero,
            dead: dead.iter().map(|s| s.to_string()).collect(),
            draw: draw.map(|d| d.iter().map(|s| s.to_string()).collect()),
            opp_dead: Vec::new(),
            opp_board: opp,
        }
    }

    fn fired() -> u64 {
        V4_FIRED.load(Ordering::Relaxed)
    }

    fn seen_live(bin: usize) -> u64 {
        V4_LIVE[bin].load(Ordering::Relaxed)
    }

    #[test]
    fn decision_mode_stays_on_the_old_path() {
        let _guard = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
        let ask = request(
            "gate",
            board(&["Ah", "Kd", "Qc"], &["7c", "7d", "2h", "3s"], &["Ts", "Jh", "4d", "9c"]),
            board(&["Qh", "Qd", "Jc"], &["5c", "5d", "Kc", "8h"], &["Ac", "Ad", "As", "Kh"]),
            &["2c", "2d", "2s"],
            Some(&["5h", "6h", "8d"]),
        );
        let before = fired();
        let solved =
            crate::v4_first::solve_maybe_collapsed(&ask, &TABLE, None, true).expect("solve");
        assert_eq!(solved.actions.len(), 6, "decision mode still enumerates six");
        assert_eq!(fired(), before, "a draw is present: the collapse must not fire");
    }

    /// (o) Past `FIRE_MAX_LIVE` the flag must do nothing at all, at **both**
    /// shapes that reach there: two live suits (one from each board -- the
    /// same fixture `two_live_suits_one_from_each_board` still checks the
    /// algorithm on) and three (hero's middle clubs, hero's bottom diamonds,
    /// the opponent's middle spades).  Two is the interesting one: it was
    /// inside the threshold until the wall clock came back at 0.49x.
    ///
    /// The histogram must keep counting these calls even though none of them
    /// fires -- the L distribution is worth watching where nothing collapses,
    /// and losing it would hide a live-suit predicate that had started
    /// inventing suits.
    #[test]
    fn past_the_threshold_the_flag_changes_nothing() {
        let _guard = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
        let cases: [(&str, u8, BoardStr, BoardStr, [&str; 3]); 2] = [
            (
                "L2",
                suits("sc"),
                board(&["2h", "3s", "5d"], &["7c", "4c", "5c", "6c"], &["Ac", "Ad", "As", "Ah"]),
                board(&["7h", "7d", "2s"], &["9s", "Ts", "Js", "Qs"], &["Ks", "Kd", "Kh", "Kc"]),
                ["2c", "2d", "3c"],
            ),
            (
                "L3",
                suits("cds"),
                board(&["Ah", "Kh", "Qh"], &["2c", "5c", "9c", "Tc"], &["3d", "6d", "8d", "Jd"]),
                board(&["Qs", "Qd", "Jc"], &["2s", "5s", "8s", "Ks"], &["Ac", "Ad", "As", "7h"]),
                ["2h", "3h", "4h"],
            ),
        ];
        for (name, expect, hero, opp, dead) in cases {
            let union = board_live_mask(&CoreBoard::from_str_board(&hero).unwrap())
                | board_live_mask(&CoreBoard::from_str_board(&opp).unwrap());
            assert_eq!(union, expect, "{name}: fixture is the wrong shape");
            assert!(
                union.count_ones() > FIRE_MAX_LIVE,
                "{name}: fixture must be past the threshold"
            );

            let ask = request(&format!("gate-{name}"), hero, opp, &dead, None);
            let bin = union.count_ones() as usize;
            let (before, counted) = (fired(), seen_live(bin));
            let on = crate::v4_first::solve_maybe_collapsed(&ask, &TABLE, None, true).expect("on");
            assert_eq!(fired(), before, "{name}: must not fire");
            assert_eq!(
                seen_live(bin),
                counted + 1,
                "{name}: the histogram must count what it refuses"
            );
            let off =
                crate::v4_first::solve_maybe_collapsed(&ask, &TABLE, None, false).expect("off");
            assert_eq!(
                serde_json::to_string(&on).unwrap(),
                serde_json::to_string(&off).unwrap(),
                "{name}: past the threshold the flag must change nothing"
            );
        }
    }

    #[test]
    fn eligible_states_fire_and_agree_with_the_old_path() {
        let _guard = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
        // L = 0, the shape the first version fired on, and L = 1 from each
        // seat in turn -- the opponent-only one is what pins the *union* in
        // `solve_maybe_collapsed`, which a hero-only mask would get wrong.
        // All three are full 29-card requests, so the uncollapsed answer is
        // the real one and not a fixture's.
        let cases: [(&str, BoardStr, BoardStr, [&str; 3]); 3] = [
            (
                "L0",
                board(&["Ah", "Kd", "Qc"], &["7c", "7d", "2h", "3s"], &["Ts", "Jh", "4d", "9c"]),
                board(&["Qh", "Qd", "Jc"], &["5c", "5d", "Kc", "8h"], &["Ac", "Ad", "As", "Kh"]),
                ["2c", "2d", "2s"],
            ),
            (
                "L1 hero",
                board(&["2h", "3s", "5d"], &["7c", "4c", "5c", "6c"], &["Ac", "Ad", "As", "Ah"]),
                board(&["8h", "8d", "2s"], &["Kh", "Kd", "9s", "Jc"], &["Ts", "Td", "Th", "Jh"]),
                ["2c", "2d", "3d"],
            ),
            (
                "L1 opponent",
                board(&["8h", "8d", "2s"], &["Kh", "Kd", "9s", "Jc"], &["Ts", "Td", "Th", "Jh"]),
                board(&["2h", "3s", "5d"], &["7c", "4c", "5c", "6c"], &["Ac", "Ad", "As", "Ah"]),
                ["2c", "2d", "3d"],
            ),
        ];
        for (name, hero, opp, dead) in cases {
            let ask = request(&format!("gate-{name}"), hero, opp, &dead, None);
            let before = fired();
            let on = crate::v4_first::solve_maybe_collapsed(&ask, &TABLE, None, true).expect("on");
            assert_eq!(fired(), before + 1, "{name}: an eligible state must fire");
            let off =
                crate::v4_first::solve_maybe_collapsed(&ask, &TABLE, None, false).expect("off");
            assert!(
                (on.value - off.value).abs() <= 1e-9,
                "{name}: collapsed {} vs uncollapsed {}",
                on.value,
                off.value
            );
            assert_eq!(off.pool, 29);
            assert_eq!(on.pool, 29);
        }
    }

    /// The pool the collapse builds is the pool `solve_weighted` builds --
    /// they are written twice, so this pins them together.
    #[test]
    fn collapse_pool_matches_the_solver_pool() {
        let ask = request(
            "pool",
            board(&["Ah", "Kd", "Qc"], &["7c", "7d", "2h", "3s"], &["Ts", "Jh", "4d", "9c"]),
            board(&["Qh", "Qd", "Jc"], &["5c", "5d", "Kc", "8h"], &["Ac", "Ad", "As", "Kh"]),
            &["2c", "2d", "2s"],
            None,
        );
        let pool = crate::v4_first::unseen_pool(&ask).expect("pool");
        let seen: BTreeSet<String> = ask
            .dead
            .iter()
            .chain(&ask.board.top)
            .chain(&ask.board.middle)
            .chain(&ask.board.bottom)
            .chain(&ask.opp_board.top)
            .chain(&ask.opp_board.middle)
            .chain(&ask.opp_board.bottom)
            .cloned()
            .collect();
        let expected: Vec<Card> = all_cards()
            .into_iter()
            .filter(|n| !seen.contains(n))
            .map(|n| to_core_card(&n).unwrap())
            .collect();
        assert_eq!(pool.len(), expected.len());
        assert_eq!(pool.len(), 29);
        for (a, b) in pool.iter().zip(&expected) {
            assert_eq!((a.rank, a.suit), (b.rank, b.suit));
        }
    }
}
