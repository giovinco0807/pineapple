//! T3 labels with hero's own draw enumerated, not sampled.
//!
//! At T3 hero holds nine, places two of three and has two slots left.  The
//! action's value is
//!
//! ```text
//! value(action) = mean over hero's C(40,3) T4 draws of
//!                 max over hero's T4 placements of
//!                 mean over opponents of hero_score(terminal)
//! ```
//!
//! # Why the enumeration is affordable, which is not obvious
//!
//! C(40,3) is 9,880 draws and each offers up to nine placements, so the naive
//! reading is ~89,000 terminals an action and 2.4 million a root.  But a
//! terminal does not depend on *which draw* delivered the cards -- only on
//! which two cards went into which rows.  There are C(40,2) = 780 such pairs
//! and at most a few placements each, so the distinct terminals number a few
//! thousand, and the 9,880 draws become lookups over a table already built.
//!
//! That is the whole trick: **score the pairs, then enumerate the draws.**
//! The expensive quantity is priced 780 times instead of 89,000, and hero's
//! draw stops being a sampled quantity at all.
//!
//! # What remains sampled
//!
//! Only how many opponents.  Their play is a best response over the whole
//! frontier and their distribution is exact by conditioning, so the count buys
//! variance and no bias -- and because every action at a root shares one drawn
//! set, it cancels in the action-to-action differences a teacher consumes.
//!
//! # T4 labels fall out of this for free
//!
//! Each pair-and-placement entry in the table *is* a T4 decision's value under
//! this action.  A caller that wants T4 teachers as well as T3 ones gets them
//! by keeping the table rather than by running a second pass, which is what
//! [`solve_with_t4`] returns.

use crate::pool::{draw, mask_of, Pool, PoolEntry, ShortDraw};
use crate::vs_fl::{hero_score, HeroTerminal};
use crate::Card;

pub struct T3Request {
    pub id: String,
    /// Hero's nine placed cards, by row.
    pub rows: [Vec<Card>; 3],
    /// Hero's two earlier discards.
    pub dead: Vec<Card>,
    /// The three drawn cards.
    pub draw: [Card; 3],
    pub opponents: usize,
    /// 0 enumerates every C(n,3) T4 draw; a positive value samples that many.
    pub t4_draws: usize,
}

pub struct T3ActionValue {
    pub action_key: String,
    pub value: f64,
    pub opponents: usize,
    pub t4_draws: usize,
}

/// One T4 decision reached from a T3 action: the pair hero kept, where it went,
/// and what that terminal is worth.
pub struct T4Leaf {
    pub action_key: String,
    pub cards: [Card; 2],
    /// Which two of the root's `unseen` cards these are.
    ///
    /// Identity, not value: the two jokers are both `Card { rank: 0, suit: 4 }`,
    /// so a rank-and-suit comparison cannot tell hero's drawn joker from the
    /// one still in the deck, and reassembling a decision by value lets a draw
    /// holding one joker acquire a two-joker placement.  `unseen` is built once
    /// per root and shared by every action, so these indices compare across
    /// actions.
    pub slots: [usize; 2],
    pub rows: [usize; 2],
    pub value: f64,
}

fn card_name(card: &Card) -> String {
    if card.rank == 0 {
        return "X".to_string();
    }
    let rank = "23456789TJQKA".chars().nth(card.rank as usize - 2).unwrap_or('?');
    let suit = "shdc".chars().nth(card.suit as usize).unwrap_or('?');
    format!("{rank}{suit}")
}

/// Rows as sorted card names, `top|mid|bot` -- the position, without an
/// action attached.
pub fn rows_key(rows: &[Vec<Card>; 3]) -> String {
    rows.iter()
        .map(|row| {
            let mut names: Vec<String> = row.iter().map(card_name).collect();
            names.sort();
            names.join(",")
        })
        .collect::<Vec<String>>()
        .join("|")
}

/// A flat card list as sorted names.
pub fn cards_key(cards: &[Card]) -> String {
    let mut names: Vec<String> = cards.iter().map(card_name).collect();
    names.sort();
    names.join(",")
}

fn board_key(rows: &[Vec<Card>; 3], discard: &Card) -> String {
    let mut parts: Vec<String> = rows
        .iter()
        .map(|row| {
            let mut names: Vec<String> = row.iter().map(card_name).collect();
            names.sort();
            names.join(",")
        })
        .collect();
    parts.push(card_name(discard));
    parts.join("|")
}

fn hero_terminal(rows: &[Vec<Card>; 3]) -> HeroTerminal {
    let core: Vec<Vec<ofc_core::Card>> = rows.iter().map(|row| crate::to_core_cards(row)).collect();
    let eval = ofc_core::evaluate_board_with_joker_constraint(&core[0], &core[1], &core[2]);
    if eval.busted {
        return HeroTerminal { busted: true, top: 0, mid: 0, bot: 0, royalty: 0, entry_width: 0 };
    }
    let (qualifies, width) = ofc_core::check_fl_entry(&eval.top);
    HeroTerminal {
        busted: false,
        top: ofc_core::evaluate_hand_value(&eval.top, 3),
        mid: ofc_core::evaluate_hand_value(&eval.mid, 5),
        bot: ofc_core::evaluate_hand_value(&eval.bot, 5),
        royalty: ofc_core::get_top_royalty(&eval.top)
            + ofc_core::get_middle_royalty(&eval.mid)
            + ofc_core::get_bottom_royalty(&eval.bot),
        entry_width: if qualifies { width } else { 0 },
    }
}

/// Where two more cards can go on a board with two open slots.
fn open_patterns(rows: &[Vec<Card>; 3]) -> Vec<[usize; 2]> {
    let capacity = [3usize, 5, 5];
    let open: Vec<usize> = (0..3).map(|row| capacity[row] - rows[row].len()).collect();
    let mut out = Vec::new();
    for first in 0..3usize {
        for second in first..3usize {
            let mut need = [0usize; 3];
            need[first] += 1;
            need[second] += 1;
            if (0..3).all(|row| need[row] <= open[row]) {
                out.push([first, second]);
            }
        }
    }
    out
}

/// T3 action values, and optionally every T4 decision they passed through.
pub fn solve_with_t4(
    request: &T3Request,
    pool: &Pool,
    fl_ev: &[f64; 4],
    stream: u64,
    keep_leaves: bool,
    own_only: bool,
) -> Result<(Vec<T3ActionValue>, Vec<T4Leaf>), ShortDraw> {
    // Hero's seen cards are action-independent, so one draw serves every
    // action and cancels in their differences.
    let mut seen: Vec<Card> = request.dead.clone();
    for row in &request.rows {
        seen.extend_from_slice(row);
    }
    seen.extend_from_slice(&request.draw);
    let (hero_naturals, hero_jokers) = mask_of(&seen);
    let opponents: Vec<&PoolEntry> =
        draw(pool, hero_naturals, hero_jokers, request.opponents, stream)?;

    let seen_set: std::collections::BTreeSet<(u8, u8)> =
        seen.iter().map(|card| (card.rank, card.suit)).collect();
    let mut unseen: Vec<Card> = Vec::new();
    for rank in 2..=14u8 {
        for suit in 0..4u8 {
            if !seen_set.contains(&(rank, suit)) {
                unseen.push(Card { rank, suit });
            }
        }
    }
    // Jokers hero has not seen are still out there, and there are two of them.
    let jokers_left = 2usize.saturating_sub(hero_jokers as usize);
    for _ in 0..jokers_left {
        unseen.push(Card { rank: 0, suit: 4 });
    }

    let mut values = Vec::new();
    let mut leaves = Vec::new();

    // Each T3 action: place two of the three drawn cards.
    let mut action_seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for discard in 0..3usize {
        let kept: Vec<usize> = (0..3).filter(|index| *index != discard).collect();
        for pattern in open_patterns(&request.rows) {
            let mut after = request.rows.clone();
            after[pattern[0]].push(request.draw[kept[0]]);
            after[pattern[1]].push(request.draw[kept[1]]);
            let action_key = board_key(&after, &request.draw[discard]);
            if !action_seen.insert(action_key.clone()) {
                continue;
            }

            // Price every (pair, placement) once -- the expensive step, and the
            // reason the draw enumeration is affordable.
            let patterns = open_patterns(&after);
            let pair_count = unseen.len() * (unseen.len() - 1) / 2;
            let mut table: Vec<f64> = vec![f64::NEG_INFINITY; pair_count * patterns.len()];
            let mut pair_index = 0usize;
            for first in 0..unseen.len() {
                for second in (first + 1)..unseen.len() {
                    for (slot, pattern) in patterns.iter().enumerate() {
                        let mut final_rows = after.clone();
                        final_rows[pattern[0]].push(unseen[first]);
                        final_rows[pattern[1]].push(unseen[second]);
                        let hero = hero_terminal(&final_rows);
                        let value = if own_only {
                            crate::vs_fl::hero_own(&hero, fl_ev)
                        } else {
                            let total: f64 = opponents
                                .iter()
                                .map(|entry| hero_score(&hero, &entry.rows, fl_ev))
                                .sum();
                            total / opponents.len() as f64
                        };
                        table[pair_index * patterns.len() + slot] = value;
                        if keep_leaves {
                            leaves.push(T4Leaf {
                                action_key: action_key.clone(),
                                cards: [unseen[first], unseen[second]],
                                slots: [first, second],
                                rows: *pattern,
                                value,
                            });
                        }
                    }
                    pair_index += 1;
                }
            }

            // Index of the (first, second) pair in the triangular table.
            let pair_at = |first: usize, second: usize| -> usize {
                let (low, high) = if first < second { (first, second) } else { (second, first) };
                low * unseen.len() - low * (low + 1) / 2 + (high - low - 1)
            };

            // Now the draws: for each, hero keeps the best two of three.
            let mut total = 0.0f64;
            let mut draws = 0usize;
            for a in 0..unseen.len() {
                for b in (a + 1)..unseen.len() {
                    for c in (b + 1)..unseen.len() {
                        let mut best = f64::NEG_INFINITY;
                        for (first, second) in [(a, b), (a, c), (b, c)] {
                            let base = pair_at(first, second) * patterns.len();
                            for slot in 0..patterns.len() {
                                let value = table[base + slot];
                                if value > best {
                                    best = value;
                                }
                            }
                        }
                        total += best;
                        draws += 1;
                    }
                }
            }
            values.push(T3ActionValue {
                action_key,
                value: total / draws as f64,
                opponents: opponents.len(),
                t4_draws: draws,
            });
        }
    }
    values.sort_by(|a, b| a.action_key.cmp(&b.action_key));
    Ok((values, leaves))
}

pub fn solve(
    request: &T3Request,
    pool: &Pool,
    fl_ev: &[f64; 4],
    stream: u64,
) -> Result<Vec<T3ActionValue>, ShortDraw> {
    solve_with_t4(request, pool, fl_ev, stream, false, false).map(|(values, _)| values)
}

/// One T4 decision harvested from a T3 root: hero's eleven-card board, the
/// three cards drawn, and what each placement is worth.
pub struct T4Decision {
    /// The T3 action that produced this board.
    pub board_key: String,
    pub draw: [Card; 3],
    /// `(placement key, value)`, the T4 teacher's rows.
    pub actions: Vec<(String, f64)>,
}

/// T3 action values plus a sample of the T4 decisions they contain.
///
/// A T3 root passes through 9,880 T4 decisions per action; writing them all
/// out is 280 million rows at teacher scale, so a caller says how many it
/// wants.  These are not re-solved -- they are read out of the table the T3
/// expectation already used, so a harvested T4 label and the T3 label above it
/// cannot disagree.
///
/// `stream` draws the opponents; `harvest_stream` chooses which decisions to
/// write out.  They are separate arguments so a second labeling pass can face
/// a fresh set of opponents while harvesting the **same** decisions -- without
/// that, two passes share almost no `(draw, placement)` keys and the T4 half
/// of the teacher has no measurable noise floor.
pub fn solve_harvesting(
    request: &T3Request,
    pool: &Pool,
    fl_ev: &[f64; 4],
    stream: u64,
    harvest_stream: u64,
    t4_per_root: usize,
    own_only: bool,
) -> Result<(Vec<T3ActionValue>, Vec<T4Decision>), ShortDraw> {
    let (values, leaves) = solve_with_t4(request, pool, fl_ev, stream, t4_per_root > 0, own_only)?;
    if t4_per_root == 0 || leaves.is_empty() {
        return Ok((values, Vec::new()));
    }
    // Group the harvested leaves by the T3 action that produced them, then
    // reassemble whole T4 decisions: a draw of three, and every placement of
    // two of them.  Deterministic in `stream`, so a rerun harvests the same
    // decisions.
    let mut by_action: std::collections::BTreeMap<&str, Vec<&T4Leaf>> =
        std::collections::BTreeMap::new();
    for leaf in &leaves {
        by_action.entry(leaf.action_key.as_str()).or_default().push(leaf);
    }
    let action_keys: Vec<&str> = by_action.keys().copied().collect();
    let mut out = Vec::new();
    let mut tick = harvest_stream.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for slot in 0..t4_per_root {
        if action_keys.is_empty() {
            break;
        }
        tick = tick
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let action = action_keys[(tick >> 33) as usize % action_keys.len()];
        let mine = &by_action[action];
        // A decision needs three cards; take one leaf's pair and a third card
        // from another leaf under the same action.
        if mine.len() < 2 {
            continue;
        }
        let anchor = mine[((tick >> 17) as usize).wrapping_add(slot) % mine.len()];
        // A third card, found by slot so a joker cannot stand in for its twin.
        let third = mine
            .iter()
            .find(|leaf| {
                let (a, b) = (leaf.slots[0], leaf.slots[1]);
                let (p, q) = (anchor.slots[0], anchor.slots[1]);
                (a == p && b != q) || (a == q && b != p)
            })
            .map(|leaf| (leaf.slots[1], leaf.cards[1]));
        let Some((third_slot, third)) = third else { continue };
        let draw_slots = [anchor.slots[0], anchor.slots[1], third_slot];
        let draw = [anchor.cards[0], anchor.cards[1], third];
        // Every placement of two of these three, valued from the same table.
        let mut actions = Vec::new();
        for leaf in mine.iter() {
            let in_draw = |slot: usize| draw_slots.contains(&slot);
            if in_draw(leaf.slots[0]) && in_draw(leaf.slots[1]) {
                actions.push((
                    format!(
                        "{}+{}@{},{}",
                        card_name(&leaf.cards[0]),
                        card_name(&leaf.cards[1]),
                        leaf.rows[0],
                        leaf.rows[1]
                    ),
                    leaf.value,
                ));
            }
        }
        if actions.len() < 2 {
            continue;
        }
        actions.sort_by(|a, b| a.0.cmp(&b.0));
        out.push(T4Decision {
            board_key: action.to_string(),
            draw,
            actions,
        });
    }
    Ok((values, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::{build_entry, deal};

    const TABLE: [f64; 4] = [0.0, 10.7, 29.9, 63.5];

    fn tiny_pool(entries: usize) -> Pool {
        Pool {
            width: 14,
            fl_ev: TABLE,
            seed: 0x7311_0001,
            entries: (0..entries as u64)
                .map(|index| build_entry(0x7311_0001, index, 14, TABLE[0]))
                .collect(),
        }
    }

    fn request_from(seed: u64, opponents: usize) -> T3Request {
        let cards = deal(seed, 0, 14);
        T3Request {
            id: "t".into(),
            rows: [cards[0..2].to_vec(), cards[2..6].to_vec(), cards[6..9].to_vec()],
            dead: cards[9..11].to_vec(),
            draw: [cards[11], cards[12], cards[13]],
            opponents,
            t4_draws: 0,
        }
    }

    /// Hero's draw is enumerated, not sampled: every C(n,3) appears once.
    #[test]
    fn every_t4_draw_is_enumerated_exactly_once() {
        let pool = tiny_pool(300);
        let request = request_from(0x7311_1000, 4);
        let Ok(values) = solve(&request, &pool, &TABLE, 5) else {
            return;
        };
        // Hero has seen 14 cards, so 40 remain: C(40,3) = 9880.
        for value in &values {
            assert_eq!(
                value.t4_draws, 9880,
                "action {} enumerated {} draws, not C(40,3)",
                value.action_key, value.t4_draws
            );
        }
    }

    /// The triangular index is the arithmetic everything else rests on: if it
    /// is wrong the table is read at the wrong offsets and every label is
    /// quietly a different hand's value.
    #[test]
    fn the_pair_index_is_a_bijection() {
        for size in [4usize, 10, 40] {
            let pair_at = |first: usize, second: usize| -> usize {
                let (low, high) = if first < second { (first, second) } else { (second, first) };
                low * size - low * (low + 1) / 2 + (high - low - 1)
            };
            let mut seen = vec![false; size * (size - 1) / 2];
            for first in 0..size {
                for second in (first + 1)..size {
                    let index = pair_at(first, second);
                    assert!(index < seen.len(), "index {index} out of range for {size}");
                    assert!(!seen[index], "index {index} used twice at ({first},{second})");
                    seen[index] = true;
                    assert_eq!(index, pair_at(second, first), "not symmetric");
                }
            }
            assert!(seen.iter().all(|hit| *hit), "some index was never produced");
        }
    }

    /// The T4 leaves a T3 root passes through are the same values the T3
    /// expectation is taken over -- so harvesting them costs a flag, not a
    /// second pass.
    #[test]
    fn harvested_t4_leaves_match_the_table_the_expectation_used() {
        let pool = tiny_pool(300);
        let request = request_from(0x7311_2000, 3);
        let Ok((values, leaves)) = solve_with_t4(&request, &pool, &TABLE, 7, true, false) else {
            return;
        };
        assert!(!leaves.is_empty());
        // Every T3 action contributed leaves, and each leaf's value is within
        // the range its action's expectation could have come from.
        for value in &values {
            let mine: Vec<&T4Leaf> = leaves
                .iter()
                .filter(|leaf| leaf.action_key == value.action_key)
                .collect();
            assert!(!mine.is_empty(), "action {} harvested no leaves", value.action_key);
            let best = mine.iter().map(|leaf| leaf.value).fold(f64::MIN, f64::max);
            let worst = mine.iter().map(|leaf| leaf.value).fold(f64::MAX, f64::min);
            assert!(
                value.value <= best + 1e-9 && value.value >= worst - 1e-9,
                "action {} has value {} outside its leaves' range [{worst}, {best}]",
                value.action_key,
                value.value
            );
        }
    }

    /// A T3 action is worth at least the worst its best completion could be
    /// and at most the best -- and specifically, the max-over-placements makes
    /// it no worse than always taking the first placement.
    #[test]
    fn taking_the_best_completion_is_never_worse_than_a_fixed_one() {
        let pool = tiny_pool(300);
        let request = request_from(0x7311_3000, 3);
        let Ok((values, leaves)) = solve_with_t4(&request, &pool, &TABLE, 9, true) else {
            return;
        };
        for value in &values {
            let mean_of_all: f64 = {
                let mine: Vec<f64> = leaves
                    .iter()
                    .filter(|leaf| leaf.action_key == value.action_key)
                    .map(|leaf| leaf.value)
                    .collect();
                mine.iter().sum::<f64>() / mine.len() as f64
            };
            assert!(
                value.value >= mean_of_all - 1e-9,
                "choosing the best completion scored {}, below the mean over all \
                 completions {mean_of_all}",
                value.value
            );
        }
    }

    /// A harvested decision may only place cards its own draw contains.
    ///
    /// This is the joker trap: both jokers are `Card { rank: 0, suit: 4 }`, so
    /// a harvest that reassembles decisions by comparing card values lets a
    /// draw holding one joker pick up the two-joker placement.  Measured on
    /// the first 100 harvested roots that was 2.6% of actions and, worse, the
    /// illegal action was the best one in 7 of the 10 roots it touched -- so
    /// this asserts the draw-membership property directly rather than trusting
    /// the comparison to stay identity-based.
    #[test]
    fn harvested_actions_only_place_cards_from_their_draw() {
        let pool = tiny_pool(3000);
        // Hero holding no joker is the case that matters: both jokers are then
        // unseen, so a two-joker placement is a real leaf under every action.
        let mut harvested = 0usize;
        let (mut jokerless, mut solved) = (0usize, 0usize);
        for attempt in 0..8u64 {
            let request = request_from(0x7311_5000 + attempt, 4);
            if request.rows.iter().flatten().chain(request.draw.iter()).any(|c| c.rank == 0) {
                continue;
            }
            jokerless += 1;
            let Ok((_, decisions)) = solve_harvesting(&request, &pool, &TABLE, 7, 7, 6, false) else {
                continue;
            };
            solved += 1;
            for decision in &decisions {
                for (key, _) in &decision.actions {
                    let placed = key.split('@').next().unwrap();
                    let mut left: Vec<String> =
                        decision.draw.iter().map(card_name).collect();
                    for name in placed.split('+') {
                        let found = left.iter().position(|held| held == name);
                        let index = found.unwrap_or_else(|| {
                            panic!(
                                "action {key} places {name}, but the draw is {:?}",
                                decision.draw.iter().map(card_name).collect::<Vec<_>>()
                            )
                        });
                        left.remove(index);
                    }
                    assert_eq!(left.len(), 1, "action {key} did not discard exactly one");
                    harvested += 1;
                }
            }
            // One jokerless root is ~90 decisions' worth of the property and
            // a full minute of solve; more attempts buy little.
            if harvested > 0 {
                break;
            }
        }
        assert!(
            harvested > 0,
            "no harvest to check: {jokerless} jokerless roots, {solved} solved"
        );
    }
}
