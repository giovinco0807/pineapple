//! Adaptive (best-response) Fantasyland opponent, for the joker deck.
//!
//! # The rule this exists for
//!
//! The Fantasyland player does not commit blind. They watch the normal
//! player's progressive placement and set their 13 only after that board is
//! complete. So the Fantasyland side is a *best response* to a concrete hero
//! board, not a fixed board drawn in advance.
//!
//! What this crate has shipped until now is the fixed board: `solve_fantasyland_v3`
//! returns one placement per hand, chosen to maximise its own royalty, and the
//! board library stores exactly that -- one `values: [u32; 3]` triple per
//! entry. Scoring a hero candidate against it is optimistic for hero, because
//! it lets hero beat an opponent who was not allowed to react.
//!
//! # Why a frontier, and why it is exact
//!
//! Best-responding by rescanning every arrangement per hero board is far too
//! slow -- 1,009,008 of them at 14 cards, and the joker substitutions on top.
//! It is also unnecessary. Write the Fantasyland player's score against a
//! fixed hero board `H` as
//!
//! ```text
//! f(A, H) = g(s_top, s_mid, s_bot) + static(A)
//!   where s_row     = sign(value_row(A) - value_row(H))  in {-1, 0, +1}
//!         g(s)      = sum(s), except +/-6 when all three agree (the scoop)
//!         static(A) = royalty(A) + fl_ev * stays(A)
//! ```
//!
//! `g` is monotone non-decreasing in each of its three arguments -- checked
//! over all 27 sign triples by [`tests::scoop_aware_line_is_monotone`] rather
//! than asserted, because the scoop is a discontinuity and easy to be wrong
//! about. Each `s_row` is monotone non-decreasing in `value_row(A)` for fixed
//! `H`. Therefore `f(., H)` is monotone in all four of
//! `(top, mid, bot, static)` **for every `H` simultaneously**, so an
//! arrangement dominated in all four can never be the unique best response and
//! the maximum always lies on the non-dominated frontier. Scanning it is exact.
//!
//! Crucially the frontier does not depend on `H`: it is built once per
//! Fantasyland hand and answers every hero board that hand will ever face.
//!
//! # The fourth component is load-bearing here
//!
//! `static` is a function of the three row values -- royalty is a function of
//! each row's value, and `stays` is a function of the top and bottom
//! categories. With `fl_ev > 0` it is monotone in them, so dominance on the
//! three values would imply dominance on `static` and the fourth component
//! could be dropped.
//!
//! The joker rules do not allow that. `fl_ev` is the surplus of Fantasyland
//! over a normal hand, and with jokers a normal hand is strong: the shipped
//! table is `{14: 0, 15: 10.7, 16: 29.9, 17: 63.5}`, so at 14 cards the surplus
//! is zero and may yet prove negative. Below zero, raising the top row to trips
//! raises its value and royalty but triggers `stays` and *subtracts*, so
//! `static` stops being monotone. Anyone porting this sweep and "simplifying"
//! it to three components gets something exact on the regular deck and quietly
//! wrong here.
//!
//! # What is assumed about jokers
//!
//! Row values come from [`ofc_core::evaluate_board_with_joker_constraint`],
//! which puts each row's jokers at full strength from the bottom up: the bottom
//! takes its maximum, the middle is constrained to at most the bottom, the top
//! to at most the middle. That is a single substitution per row rather than a
//! choice among them, and it is exact only while a higher row value is never
//! worth less -- which is to say, only while `fl_ev >= 0`.
//!
//! At `fl_ev = 0` (14 cards, the shipped table) it holds: `static` is royalty
//! alone, and royalty is monotone in the row value. **If the 14-card constant
//! is ever measured below zero, this assumption has to be replaced** by keeping
//! the Pareto set of achievable row values per row rather than the maximum.
//! [`build_frontier`] asserts the sign rather than trusting a caller to have
//! read this paragraph.

use crate::{
    evaluate_3_card, evaluate_5_card, get_bottom_royalty, get_middle_royalty, get_top_royalty,
    subset_masks, to_core_cards, Card, HandRank, HandRank3,
};

/// One arrangement, reduced to everything a best response can depend on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierEntry {
    /// Canonical cross-row hand values, as the board library already stores.
    pub top: u32,
    pub mid: u32,
    pub bot: u32,
    pub royalty: i32,
    pub stays: bool,
    /// `royalty + fl_ev * stays`, the part of the score no hero board moves.
    pub static_value: f64,
}

/// The 1-6 line term with the scoop bonus, from three row signs.
#[inline(always)]
pub fn scoop_aware_line(top: i32, mid: i32, bot: i32) -> i32 {
    let sum = top + mid + bot;
    if sum == 3 {
        6
    } else if sum == -3 {
        -6
    } else {
        sum
    }
}

#[inline(always)]
fn sign_of(own: u32, other: u32) -> i32 {
    match own.cmp(&other) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    }
}

impl FrontierEntry {
    /// Score from the Fantasyland side against a concrete hero board, dropping
    /// the hero-only terms that are constant across this choice.
    #[inline(always)]
    pub fn score_against(&self, hero_top: u32, hero_mid: u32, hero_bot: u32) -> f64 {
        let lines = scoop_aware_line(
            sign_of(self.top, hero_top),
            sign_of(self.mid, hero_mid),
            sign_of(self.bot, hero_bot),
        );
        lines as f64 + self.static_value
    }

    #[inline(always)]
    fn dominates(&self, other: &Self) -> bool {
        self.top >= other.top
            && self.mid >= other.mid
            && self.bot >= other.bot
            && self.static_value >= other.static_value
    }
}

/// Best response of a Fantasyland frontier to a concrete hero board.
pub fn best_response(frontier: &[FrontierEntry], hero_top: u32, hero_mid: u32, hero_bot: u32) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for entry in frontier {
        let value = entry.score_against(hero_top, hero_mid, hero_bot);
        if value > best {
            best = value;
        }
    }
    best
}

/// Reduce a candidate set to its non-dominated members.
///
/// Sorted by `static_value` descending first, so when a candidate is examined
/// every already-accepted entry has `static_value >=` its own and only the
/// three row values need comparing.
fn sweep(mut candidates: Vec<FrontierEntry>) -> Vec<FrontierEntry> {
    candidates.sort_by(|left, right| {
        right
            .static_value
            .partial_cmp(&left.static_value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.bot.cmp(&left.bot))
            .then_with(|| right.mid.cmp(&left.mid))
            .then_with(|| right.top.cmp(&left.top))
    });
    let mut accepted: Vec<FrontierEntry> = Vec::new();
    'outer: for candidate in candidates {
        for kept in accepted.iter() {
            if kept.dominates(&candidate) {
                continue 'outer;
            }
        }
        accepted.push(candidate);
    }
    accepted
}

/// One value a row can be made to take, and what it is worth there.
///
/// A joker-free row has exactly one of these. A row with jokers has one per
/// distinct value its substitutions can reach -- royalty is a function of the
/// value (category, and the pair rank for a top row), so two substitutions that
/// reach the same value are worth the same and collapse into one option.
#[derive(Clone, Copy)]
struct Option5 {
    value: u32,
    roy_mid: i32,
    roy_bot: i32,
    stay_bot: bool,
}

#[derive(Clone, Copy)]
struct Option3 {
    value: u32,
    roy_top: i32,
    trips: bool,
}

/// One 5-card subset: every value it can be made to take, ascending.
struct Row5 {
    mask: u32,
    options: Vec<Option5>,
}

/// One 3-card subset: every value it can be made to take, ascending.
struct Row3 {
    mask: u32,
    options: Vec<Option3>,
}

impl Row5 {
    /// The strongest option no stronger than `ceiling`, or `None` if the row
    /// cannot be held that low -- which is a foul, not a choice.
    #[inline(always)]
    fn under(&self, ceiling: u32) -> Option<&Option5> {
        match self.options.binary_search_by(|option| option.value.cmp(&ceiling)) {
            Ok(index) => Some(&self.options[index]),
            Err(0) => None,
            Err(index) => Some(&self.options[index - 1]),
        }
    }

    #[inline(always)]
    fn best(&self) -> &Option5 {
        self.options.last().expect("a row always has one option")
    }
}

impl Row3 {
    #[inline(always)]
    fn under(&self, ceiling: u32) -> Option<&Option3> {
        match self.options.binary_search_by(|option| option.value.cmp(&ceiling)) {
            Ok(index) => Some(&self.options[index]),
            Err(0) => None,
            Err(index) => Some(&self.options[index - 1]),
        }
    }
}

/// The cards a joker in this row may become: any rank/suit the row does not
/// already hold. This mirrors [`ofc_core`]'s own substitution rule exactly --
/// it excludes duplicates within the row and nothing else, so a joker in the
/// middle may become a card sitting in the bottom. That is the shipped
/// semantics and this is not the place to change it.
fn substitutions(row: &[Card]) -> Vec<Card> {
    let mut used = [[false; 4]; 15];
    for card in row {
        if !card.is_joker() {
            used[card.rank as usize][card.suit as usize] = true;
        }
    }
    let mut out = Vec::with_capacity(52);
    for rank in 2..=14u8 {
        for suit in 0..4u8 {
            if !used[rank as usize][suit as usize] {
                out.push(Card { rank, suit });
            }
        }
    }
    out
}

/// Every distinct row this subset can be made into, ascending by value.
///
/// The search runs once per subset rather than once per arrangement. Doing it
/// per arrangement is what made two-joker hands pathological: with two jokers
/// in one row the constraint pass tries ~1,326 substitution pairs, and a
/// million arrangements each paying that is minutes a hand.
/// The callback receives the resolved row twice, in both card types, from
/// buffers this reuses. A two-joker five-card row runs 1,326 substitution
/// pairs and a hand can hold 220 such rows, so allocating a pair of vectors per
/// evaluation is 584,000 allocations a hand.
fn expand(row: &[Card], mut evaluate: impl FnMut(&[Card], &[ofc_core::Card])) {
    let mut cards = [Card { rank: 0, suit: 0 }; 5];
    let mut core = [ofc_core::Card { rank: 0, suit: 0 }; 5];
    let length = row.len();
    let jokers = row.iter().filter(|card| card.is_joker()).count();

    let mut natural = 0usize;
    for card in row {
        if !card.is_joker() {
            cards[natural] = *card;
            natural += 1;
        }
    }
    let mut fire = |cards: &[Card], core: &mut [ofc_core::Card], count: usize| {
        for index in 0..count {
            core[index] = ofc_core::Card {
                rank: cards[index].rank,
                suit: cards[index].suit,
            };
        }
    };

    if jokers == 0 {
        fire(&cards[..length], &mut core, length);
        evaluate(&cards[..length], &core[..length]);
        return;
    }
    let subs = substitutions(row);
    if jokers == 1 {
        for &sub in &subs {
            cards[natural] = sub;
            fire(&cards[..length], &mut core, length);
            evaluate(&cards[..length], &core[..length]);
        }
    } else {
        for first in 0..subs.len() {
            cards[natural] = subs[first];
            for second in (first + 1)..subs.len() {
                cards[natural + 1] = subs[second];
                fire(&cards[..length], &mut core, length);
                evaluate(&cards[..length], &core[..length]);
            }
        }
    }
}

/// The frontier without the per-bottom reduction, for checking the reduction
/// changes nothing.
///
/// Grouping before sweeping is exact because dominance is transitive: an
/// element the global sweep keeps cannot be dominated inside its own group, so
/// it reaches the pool; and an element surviving the pool sweep cannot be
/// dominated by anything a group dropped, because whatever dominated that also
/// dominates this one and is itself in the pool. That is an argument. This
/// function is how the argument gets checked.
pub fn build_frontier_unbucketed(cards: &[Card], fl_ev: f64) -> Vec<FrontierEntry> {
    let mut candidates = Vec::with_capacity(1 << 20);
    arrangements(cards, |top, mid, bot, royalty, stays| {
        candidates.push(FrontierEntry {
            top,
            mid,
            bot,
            royalty,
            stays,
            static_value: royalty as f64 + if stays { fl_ev } else { 0.0 },
        });
    });
    sweep(candidates)
}

/// A frontier as a comparable set. The sweep may keep either of two identical
/// tuples, so identity is the set of tuples rather than the order they arrived.
pub fn frontier_key(frontier: &[FrontierEntry]) -> Vec<(u32, u32, u32, u64)> {
    let mut key: Vec<(u32, u32, u32, u64)> = frontier
        .iter()
        .map(|entry| (entry.top, entry.mid, entry.bot, entry.static_value.to_bits()))
        .collect();
    key.sort_unstable();
    key.dedup();
    key
}

/// Rank every 3- and 5-card subset once, and index them by position mask.
///
/// This is the difference between a frontier that costs 38 ms and one that
/// costs 21 seconds. Evaluating inside the million-arrangement loop instead
/// costs three hand rankings and a joker constraint pass per arrangement, and
/// the first version of this module did exactly that: 21,552 ms a hand,
/// measured, against 38.3 ms for the same sweep on the regular deck. The
/// arrangement count is identical between the two decks, so all 563x of that
/// was the missing precomputation.
fn tabulate(cards: &[Card]) -> (Vec<Row5>, Vec<Row3>) {
    let n = cards.len();
    let fives = subset_masks(n, 5)
        .into_iter()
        .map(|mask| {
            let row = mask_cards(cards, mask);
            let mut options: Vec<Option5> = Vec::new();
            expand(&row, |resolved, core| {
                let (rank, _) = evaluate_5_card(resolved);
                options.push(Option5 {
                    value: ofc_core::evaluate_hand_value(core, 5),
                    roy_mid: get_middle_royalty(resolved),
                    roy_bot: get_bottom_royalty(resolved),
                    stay_bot: matches!(
                        rank,
                        HandRank::Quads | HandRank::StraightFlush | HandRank::RoyalFlush
                    ),
                });
            });
            options.sort_by_key(|option| option.value);
            options.dedup_by_key(|option| option.value);
            Row5 { mask, options }
        })
        .collect();
    let threes = subset_masks(n, 3)
        .into_iter()
        .map(|mask| {
            let row = mask_cards(cards, mask);
            let mut options: Vec<Option3> = Vec::new();
            expand(&row, |resolved, core| {
                let (rank, _) = evaluate_3_card(resolved);
                options.push(Option3 {
                    value: ofc_core::evaluate_hand_value(core, 3),
                    roy_top: get_top_royalty(resolved),
                    trips: rank == HandRank3::Trips,
                });
            });
            options.sort_by_key(|option| option.value);
            options.dedup_by_key(|option| option.value);
            Row3 { mask, options }
        })
        .collect();
    (fives, threes)
}

/// Every legal arrangement of `cards`, as `(top, mid, bot, royalty, stays)`.
///
/// Shared by the frontier and by the brute-force reference, so the two can
/// never disagree about what an arrangement *is* -- only about which of them
/// survive.
///
/// Two paths. When no row holds a joker every value is already exact, so an
/// ordering violation is a foul and the precomputed royalties stand. When a row
/// does, [`ofc_core::evaluate_board_with_joker_constraint`] resolves the
/// substitutions from the bottom up and the row values have to be read back
/// from what it chose -- a joker-max value is an upper bound the constraint may
/// not have been allowed to reach.
fn arrangements(cards: &[Card], mut emit: impl FnMut(u32, u32, u32, i32, bool)) {
    let tables = tabulate(cards);
    arrangements_with(cards, &tables, |_, top, mid, bot, royalty, stays| {
        emit(top, mid, bot, royalty, stays)
    });
}

/// As [`arrangements`], but the callback also receives which bottom row the
/// arrangement belongs to. Consecutive calls share a bottom, which is what lets
/// the frontier reduce a bottom's 504 arrangements before they join the pool.
fn arrangements_with(
    cards: &[Card],
    tables: &(Vec<Row5>, Vec<Row3>),
    mut emit: impl FnMut(usize, u32, u32, u32, i32, bool),
) {
    let n = cards.len();
    let (fives, threes) = tables;

    // Mask -> table index, so a row generated by position can be looked up
    // without searching. 2^14 entries at 14 cards: trivially small.
    let mut index5 = vec![u16::MAX; 1usize << n];
    for (index, row) in fives.iter().enumerate() {
        index5[row.mask as usize] = index as u16;
    }
    let mut index3 = vec![u16::MAX; 1usize << n];
    for (index, row) in threes.iter().enumerate() {
        index3[row.mask as usize] = index as u16;
    }

    // Choose the middle from what the bottom left, and the top from what both
    // left, rather than walking the full tables and discarding the overlaps.
    // The full-table version visits 2002 x 2002 x 364 = 1.46 billion pairs to
    // find the 1,009,008 that are disjoint -- 1,449x the work, and the reason
    // the first measurement came back at 21.6 seconds a hand.
    let middle_patterns = subset_masks(n - 5, 5);
    let top_patterns = subset_masks(n - 10, 3);

    // Bottom at full strength, middle held at or under it, top at or under the
    // middle -- the same bottom-up resolution `ofc_core` performs, but read off
    // precomputed option lists instead of searched per arrangement.
    // Fixed-size scratch rather than a Vec per (bottom, middle): the loop runs
    // 2002 + 2002x126 = 254,254 times a hand, and an allocation apiece is real
    // money at this count.
    let mut after_bot = [0usize; 17];
    let mut after_mid = [0usize; 17];

    for (bot_index, bot_row) in fives.iter().enumerate() {
        let bot = *bot_row.best();
        let mut left = 0usize;
        for position in 0..n {
            if bot_row.mask & (1 << position) == 0 {
                after_bot[left] = position;
                left += 1;
            }
        }
        for &pattern in &middle_patterns {
            let mid_mask = spread(pattern, &after_bot[..left]);
            let mid_row = &fives[index5[mid_mask as usize] as usize];
            let Some(&mid) = mid_row.under(bot.value) else {
                continue;
            };
            let mut remaining = 0usize;
            for &position in &after_bot[..left] {
                if mid_mask & (1 << position) == 0 {
                    after_mid[remaining] = position;
                    remaining += 1;
                }
            }
            for &top_pattern in &top_patterns {
                let top_mask = spread(top_pattern, &after_mid[..remaining]);
                let top_row = &threes[index3[top_mask as usize] as usize];
                let Some(&top) = top_row.under(mid.value) else {
                    continue;
                };
                emit(
                    bot_index,
                    top.value,
                    mid.value,
                    bot.value,
                    top.roy_top + mid.roy_mid + bot.roy_bot,
                    top.trips || bot.stay_bot,
                );
            }
        }
    }
}

/// The arrangements under one bottom row, so a bounded-away bottom costs
/// nothing rather than costing 504 candidates that are then discarded.
fn arrangements_under(
    cards: &[Card],
    tables: &(Vec<Row5>, Vec<Row3>),
    bot_index: usize,
    mut emit: impl FnMut(u32, u32, u32, i32, bool),
) {
    let n = cards.len();
    let (fives, threes) = tables;
    let mut index5 = vec![u16::MAX; 1usize << n];
    for (index, row) in fives.iter().enumerate() {
        index5[row.mask as usize] = index as u16;
    }
    let mut index3 = vec![u16::MAX; 1usize << n];
    for (index, row) in threes.iter().enumerate() {
        index3[row.mask as usize] = index as u16;
    }
    emit_under(
        cards,
        tables,
        &index5,
        &index3,
        &subset_masks(n - 5, 5),
        &subset_masks(n - 10, 3),
        bot_index,
        &mut emit,
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_under(
    cards: &[Card],
    tables: &(Vec<Row5>, Vec<Row3>),
    index5: &[u16],
    index3: &[u16],
    middle_patterns: &[u32],
    top_patterns: &[u32],
    bot_index: usize,
    emit: &mut impl FnMut(u32, u32, u32, i32, bool),
) {
    let n = cards.len();
    let (fives, threes) = tables;
    let bot_row = &fives[bot_index];
    let bot = *bot_row.best();
    let mut after_bot = [0usize; 17];
    let mut after_mid = [0usize; 17];
    let mut left = 0usize;
    for position in 0..n {
        if bot_row.mask & (1 << position) == 0 {
            after_bot[left] = position;
            left += 1;
        }
    }
    for &pattern in middle_patterns {
        let mid_mask = spread(pattern, &after_bot[..left]);
        let mid_row = &fives[index5[mid_mask as usize] as usize];
        let Some(&mid) = mid_row.under(bot.value) else {
            continue;
        };
        let mut remaining = 0usize;
        for &position in &after_bot[..left] {
            if mid_mask & (1 << position) == 0 {
                after_mid[remaining] = position;
                remaining += 1;
            }
        }
        for &top_pattern in top_patterns {
            let top_mask = spread(top_pattern, &after_mid[..remaining]);
            let top_row = &threes[index3[top_mask as usize] as usize];
            let Some(&top) = top_row.under(mid.value) else {
                continue;
            };
            emit(
                top.value,
                mid.value,
                bot.value,
                top.roy_top + mid.roy_mid + bot.roy_bot,
                top.trips || bot.stay_bot,
            );
        }
    }
}

/// Re-express a mask over `slots.len()` positions as a mask over the original
/// card indices: bit `i` of `pattern` means "take `slots[i]`".
#[inline(always)]
fn spread(pattern: u32, slots: &[usize]) -> u32 {
    let mut mask = 0u32;
    let mut bits = pattern;
    while bits != 0 {
        let bit = bits.trailing_zeros() as usize;
        mask |= 1 << slots[bit];
        bits &= bits - 1;
    }
    mask
}

fn mask_cards(cards: &[Card], mask: u32) -> Vec<Card> {
    (0..cards.len())
        .filter(|index| mask & (1 << index) != 0)
        .map(|index| cards[index])
        .collect()
}

fn from_core(cards: &[ofc_core::Card]) -> Vec<Card> {
    cards
        .iter()
        .map(|card| Card { rank: card.rank, suit: card.suit })
        .collect()
}

/// The non-dominated arrangements of a Fantasyland hand.
///
/// Panics on a negative `fl_ev`: the joker substitution this rests on takes
/// each row at full strength, which stops being optimal once a stronger row can
/// cost value through the stay term. See the module docs.
pub fn build_frontier(cards: &[Card], fl_ev: f64) -> Vec<FrontierEntry> {
    build_frontier_timed(cards, fl_ev).0
}

/// The frontier, with the three stages timed separately.
///
/// Which stage costs what is not something to reason about from the shape of
/// the code -- the first version of this module was 302x slower than it needed
/// to be, and adding the obvious optimisation to it changed nothing because the
/// cost was somewhere else entirely. So the stages are measured.
pub fn build_frontier_timed(cards: &[Card], fl_ev: f64) -> (Vec<FrontierEntry>, f64, f64, f64) {
    assert!(
        fl_ev >= 0.0,
        "build_frontier needs fl_ev >= 0; at {fl_ev} the per-row joker maximum \
         is no longer the best substitution and the frontier would be missing \
         arrangements rather than merely mispriced"
    );
    let started = std::time::Instant::now();
    let tables = tabulate(cards);
    let tabulate_seconds = started.elapsed().as_secs_f64();

    // Drop bottoms before generating them, not after.
    //
    // Reducing each bottom's 504 arrangements before pooling them was worth
    // 1.5x; it still builds all 1,009,008. A bottom can instead be bounded: its
    // own row value is fixed, and the best any arrangement under it could reach
    // is the strongest top and middle still available plus their royalties. If
    // one frontier row already dominates that bound, none of the 504 can
    // survive and none of them need to exist.
    //
    // Bottoms run strongest-first so the frontier that does the dominating
    // forms early -- the same reason `solve_fantasyland_v3` sorts before it
    // prunes against its incumbent.
    let started = std::time::Instant::now();
    let (fives, threes) = &tables;
    let ceiling_top = threes
        .iter()
        .map(|row| {
            let best = row.options.last().expect("a row always has one option");
            (best.value, best.roy_top, best.trips)
        })
        .fold((0u32, 0i32, false), |acc, item| {
            (acc.0.max(item.0), acc.1.max(item.1), acc.2 || item.2)
        });
    let ceiling_mid = fives
        .iter()
        .map(|row| {
            let best = row.options.last().expect("a row always has one option");
            (best.value, best.roy_mid)
        })
        .fold((0u32, 0i32), |acc, item| (acc.0.max(item.0), acc.1.max(item.1)));

    let mut order: Vec<usize> = (0..fives.len()).collect();
    order.sort_by_key(|&index| {
        let best = fives[index].best();
        std::cmp::Reverse((best.value, best.roy_bot))
    });

    let n = cards.len();
    let mut index5 = vec![u16::MAX; 1usize << n];
    for (index, row) in fives.iter().enumerate() {
        index5[row.mask as usize] = index as u16;
    }
    let mut index3 = vec![u16::MAX; 1usize << n];
    for (index, row) in threes.iter().enumerate() {
        index3[row.mask as usize] = index as u16;
    }
    let middle_patterns = subset_masks(n - 5, 5);
    let top_patterns = subset_masks(n - 10, 3);

    let mut frontier: Vec<FrontierEntry> = Vec::new();
    let mut bucket: Vec<FrontierEntry> = Vec::with_capacity(512);
    let mut skipped = 0usize;
    let mut sweep_seconds = 0.0f64;
    let mut swept_at = 0usize;
    for &bot_index in &order {
        let bot = *fives[bot_index].best();
        // The ordering constraint is part of the bound: a middle cannot beat
        // the bottom it sits under, and a top cannot beat that middle. Without
        // this the bound asks for the strongest top and middle in the hand at
        // once, which almost nothing dominates -- measured at a 130.9 ms build
        // against 110.1 ms for no pruning at all.
        let mid_ceiling = ceiling_mid.0.min(bot.value);
        let top_ceiling = ceiling_top.0.min(mid_ceiling);
        let bound = FrontierEntry {
            top: top_ceiling,
            mid: mid_ceiling,
            bot: bot.value,
            royalty: ceiling_top.1 + ceiling_mid.1 + bot.roy_bot,
            stays: ceiling_top.2 || bot.stay_bot,
            static_value: (ceiling_top.1 + ceiling_mid.1 + bot.roy_bot) as f64
                + if ceiling_top.2 || bot.stay_bot { fl_ev } else { 0.0 },
        };
        // Every arrangement under this bottom obeys top <= mid <= bot, so all
        // of them sit under (b, b, b). Bottoms run descending, so a frontier
        // row whose TOP already reaches b dominates this bottom and every one
        // left after it -- the whole tail goes at once.
        if frontier.iter().any(|row| row.top >= bot.value) {
            skipped += order.len() - skipped;
            break;
        }
        if frontier.iter().any(|row| row.dominates(&bound)) {
            skipped += 1;
            continue;
        }
        bucket.clear();
        emit_under(
            cards,
            &tables,
            &index5,
            &index3,
            &middle_patterns,
            &top_patterns,
            bot_index,
            &mut |top, mid, bot_value, royalty, stays| {
                bucket.push(FrontierEntry {
                    top,
                    mid,
                    bot: bot_value,
                    royalty,
                    stays,
                    static_value: royalty as f64 + if stays { fl_ev } else { 0.0 },
                });
            },
        );
        // Sweep the bucket every time, the frontier only when it has grown
        // enough to be worth it. A stale frontier cannot prune wrongly -- every
        // row in it is a real arrangement, so if one dominates a bound then that
        // bottom really is dominated, whether or not the row itself survives the
        // final sweep. Re-sweeping on every bottom cost 41 of 108 ms.
        let at = std::time::Instant::now();
        frontier.extend(sweep(std::mem::take(&mut bucket)));
        if frontier.len() > swept_at + 256 {
            frontier = sweep(frontier);
            swept_at = frontier.len();
        }
        sweep_seconds += at.elapsed().as_secs_f64();
        bucket = Vec::with_capacity(512);
    }
    let at = std::time::Instant::now();
    let frontier = sweep(frontier);
    sweep_seconds += at.elapsed().as_secs_f64();
    let generate_seconds = started.elapsed().as_secs_f64() - sweep_seconds;
    let _ = skipped;

    (frontier, tabulate_seconds, generate_seconds, sweep_seconds)
}

/// Brute force over every arrangement, for the exactness test only.
pub fn best_response_brute_force(
    cards: &[Card],
    fl_ev: f64,
    hero_top: u32,
    hero_mid: u32,
    hero_bot: u32,
) -> f64 {
    let mut best = f64::NEG_INFINITY;
    arrangements(cards, |top, mid, bot, royalty, stays| {
        let entry = FrontierEntry {
            top,
            mid,
            bot,
            royalty,
            stays,
            static_value: royalty as f64 + if stays { fl_ev } else { 0.0 },
        };
        let value = entry.score_against(hero_top, hero_mid, hero_bot);
        if value > best {
            best = value;
        }
    });
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoop_aware_line_is_monotone() {
        // The whole dominance argument rests on this, and the scoop makes it a
        // discontinuous function, so it is checked over all 27 sign triples
        // rather than reasoned about.
        for top in -1..=1 {
            for mid in -1..=1 {
                for bot in -1..=1 {
                    let here = scoop_aware_line(top, mid, bot);
                    if top < 1 {
                        assert!(scoop_aware_line(top + 1, mid, bot) >= here);
                    }
                    if mid < 1 {
                        assert!(scoop_aware_line(top, mid + 1, bot) >= here);
                    }
                    if bot < 1 {
                        assert!(scoop_aware_line(top, mid, bot + 1) >= here);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // The frontier's claim, pinned against brute force.
    //
    // Scanning a few dozen non-dominated rows must answer the same question
    // as scanning every arrangement, for every hero board at once.  Both
    // paths build the same f64 the same way -- an integer line term plus a
    // static value that is a sum of integers and at most one `fl_ev` -- so
    // the comparison is bit-identical and a tolerance would only hide a real
    // disagreement.
    //
    // Hero boards are dealt from what the Fantasyland hand left and resolved
    // through the canonical evaluator, because a random (u32, u32, u32) is
    // usually not a legal board and a frontier bug against an unreachable
    // board is not one worth finding.
    // ------------------------------------------------------------------

    fn lcg(seed: u64) -> impl FnMut() -> usize {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        }
    }

    fn shuffled_deck(next: &mut impl FnMut() -> usize) -> Vec<Card> {
        let mut deck = Vec::with_capacity(54);
        for rank in 2..=14u8 {
            for suit in 0..4u8 {
                deck.push(Card { rank, suit });
            }
        }
        deck.push(Card { rank: 0, suit: 4 });
        deck.push(Card { rank: 0, suit: 4 });
        for index in (1..deck.len()).rev() {
            deck.swap(index, next() % (index + 1));
        }
        deck
    }

    /// Hero's finished thirteen as canonical row values, or `None` if fouled.
    fn hero_keys(cards: &[Card]) -> Option<(u32, u32, u32)> {
        let core = to_core_cards(cards);
        let eval = ofc_core::evaluate_board_with_joker_constraint(
            &core[0..3],
            &core[3..8],
            &core[8..13],
        );
        if eval.busted {
            return None;
        }
        Some((
            ofc_core::evaluate_hand_value(&eval.top, 3),
            ofc_core::evaluate_hand_value(&eval.mid, 5),
            ofc_core::evaluate_hand_value(&eval.bot, 5),
        ))
    }

    fn case_count(default: usize) -> usize {
        std::env::var("FL_FRONTIER_CASES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    #[test]
    fn frontier_best_response_matches_brute_force() {
        let mut next = lcg(0x2026_0811_0001);
        let mut compared = 0usize;
        for _ in 0..case_count(6) {
            let deck = shuffled_deck(&mut next);
            let fl_hand: Vec<Card> = deck[0..14].to_vec();
            let remainder: Vec<Card> = deck[14..].to_vec();
            for fl_ev in [0.0f64, 63.5] {
                let frontier = build_frontier(&fl_hand, fl_ev);
                assert!(!frontier.is_empty(), "empty frontier for a 14-card hand");
                let mut offset = 0usize;
                while offset + 13 <= remainder.len() {
                    let hero = &remainder[offset..offset + 13];
                    offset += 13;
                    let Some((top, mid, bot)) = hero_keys(hero) else {
                        continue;
                    };
                    let via_frontier = best_response(&frontier, top, mid, bot);
                    let via_brute =
                        best_response_brute_force(&fl_hand, fl_ev, top, mid, bot);
                    assert_eq!(
                        via_frontier.to_bits(),
                        via_brute.to_bits(),
                        "frontier and brute force disagree at fl_ev={fl_ev} \
                         hero=({top},{mid},{bot}): {via_frontier} vs {via_brute}"
                    );
                    compared += 1;
                }
            }
        }
        assert!(compared > 0, "no (hand, hero board) pairs compared");
        println!("compared {compared} (hand, hero board) pairs against brute force");
    }

    /// Forced sign vectors, which dealt boards will not reach.  The scoop is a
    /// discontinuity and the row that wins one is not the row that maximises
    /// any single component.
    #[test]
    fn frontier_matches_brute_force_at_the_extremes() {
        let mut next = lcg(0x2026_0811_0002);
        for _ in 0..case_count(4) {
            let deck = shuffled_deck(&mut next);
            let fl_hand: Vec<Card> = deck[0..14].to_vec();
            let frontier = build_frontier(&fl_hand, 0.0);
            for hero in [(0u32, 0u32, 0u32), (u32::MAX, u32::MAX, u32::MAX)] {
                let via_frontier = best_response(&frontier, hero.0, hero.1, hero.2);
                let via_brute =
                    best_response_brute_force(&fl_hand, 0.0, hero.0, hero.1, hero.2);
                assert_eq!(
                    via_frontier.to_bits(),
                    via_brute.to_bits(),
                    "extreme hero board {hero:?}: {via_frontier} vs {via_brute}"
                );
            }
        }
    }

    /// Row count and build cost: the two numbers the FL14 plan says to measure
    /// rather than project, because the pool's affordability follows from them.
    /// Run with `--nocapture`.
    #[test]
    fn frontier_shape_and_cost() {
        let mut next = lcg(0x2026_0811_0003);
        let hands = case_count(12);
        let (mut rows_total, mut rows_max) = (0usize, 0usize);
        let (mut tabulate_total, mut sweep_total, mut build_total) = (0.0, 0.0, 0.0);
        let mut by_jokers: [(usize, usize); 3] = [(0, 0); 3];

        for _ in 0..hands {
            let deck = shuffled_deck(&mut next);
            let fl_hand: Vec<Card> = deck[0..14].to_vec();
            let jokers = fl_hand.iter().filter(|card| card.rank == 0).count();
            let (frontier, tabulate_seconds, enumerate_seconds, sweep_seconds) =
                build_frontier_timed(&fl_hand, 0.0);
            rows_total += frontier.len();
            rows_max = rows_max.max(frontier.len());
            by_jokers[jokers].0 += frontier.len();
            by_jokers[jokers].1 += 1;
            tabulate_total += tabulate_seconds;
            sweep_total += sweep_seconds;
            build_total += tabulate_seconds + enumerate_seconds + sweep_seconds;
        }

        println!(
            "frontier rows: mean {:.1}, max {}, over {hands} hands",
            rows_total as f64 / hands as f64,
            rows_max
        );
        for (jokers, (sum, count)) in by_jokers.iter().enumerate() {
            if *count > 0 {
                println!(
                    "  {jokers} joker(s): mean {:.1} rows over {count} hands",
                    *sum as f64 / *count as f64
                );
            }
        }
        println!(
            "build cost per hand: {:.1} ms (tabulate {:.1}, sweep {:.1})",
            build_total / hands as f64 * 1000.0,
            tabulate_total / hands as f64 * 1000.0,
            sweep_total / hands as f64 * 1000.0,
        );
        assert!(rows_total > 0);
    }

    /// The frontier's *membership* does not move with a non-negative `fl_ev`,
    /// which is what lets the pool store `FrontierRow { v, r, s }` unpriced and
    /// apply the constant at load.
    ///
    /// Membership, not `frontier_key`: that key carries `static_value`, which
    /// is the priced quantity and is *supposed* to move.  What must not move is
    /// which arrangements survive -- and they do not, because `static` is a
    /// function of the three keys (pinned by ofc_core's
    /// `joker_constraint_exactness`), so two rows with equal keys have equal
    /// royalty and equal stays and one price cannot separate them.
    ///
    /// All of this stops being true below zero -- see the module header.
    #[test]
    fn frontier_membership_is_invariant_across_nonnegative_fl_ev() {
        fn membership(frontier: &[FrontierEntry]) -> Vec<(u32, u32, u32, i32, bool)> {
            let mut rows: Vec<(u32, u32, u32, i32, bool)> = frontier
                .iter()
                .map(|entry| (entry.top, entry.mid, entry.bot, entry.royalty, entry.stays))
                .collect();
            rows.sort_unstable();
            rows.dedup();
            rows
        }
        let mut next = lcg(0x2026_0811_0004);
        for _ in 0..case_count(6) {
            let deck = shuffled_deck(&mut next);
            let fl_hand: Vec<Card> = deck[0..14].to_vec();
            let base = membership(&build_frontier(&fl_hand, 0.0));
            for fl_ev in [10.7f64, 29.9, 63.5] {
                let other = membership(&build_frontier(&fl_hand, fl_ev));
                assert_eq!(
                    base, other,
                    "frontier membership moved between fl_ev=0 and fl_ev={fl_ev}; \
                     the pool cannot store rows unpriced"
                );
            }
        }
    }

    /// And the priced view *does* move, in exactly one way: a stay row's static
    /// shifts by the constant and a non-stay row's does not.  Asserted so that
    /// a future change which quietly prices non-stay rows is caught here rather
    /// than in a pool that loads without complaint.
    #[test]
    fn only_stay_rows_are_repriced() {
        let mut next = lcg(0x2026_0811_0005);
        for _ in 0..case_count(4) {
            let deck = shuffled_deck(&mut next);
            let fl_hand: Vec<Card> = deck[0..14].to_vec();
            let base = build_frontier(&fl_hand, 0.0);
            let priced = build_frontier(&fl_hand, 63.5);
            assert_eq!(base.len(), priced.len());
            // Matched by key, not by position: the sweep sorts by static_value
            // descending, so a different constant is a different order.
            let unpriced_by_key: std::collections::HashMap<(u32, u32, u32), &FrontierEntry> =
                base.iter().map(|e| ((e.top, e.mid, e.bot), e)).collect();
            for repriced in &priced {
                let unpriced = unpriced_by_key
                    .get(&(repriced.top, repriced.mid, repriced.bot))
                    .expect("repriced frontier has a row the unpriced one lacks");
                let expected = unpriced.royalty as f64 + if repriced.stays { 63.5 } else { 0.0 };
                assert_eq!(
                    repriced.static_value.to_bits(),
                    expected.to_bits(),
                    "row ({},{},{}) stays={} priced to {} not {expected}",
                    repriced.top,
                    repriced.mid,
                    repriced.bot,
                    repriced.stays,
                    repriced.static_value
                );
            }
        }
    }
}
