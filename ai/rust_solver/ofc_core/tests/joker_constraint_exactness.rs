//! Is the bottom-up joker collapse exact? Pinned against brute force.
//!
//! `evaluate_board_with_joker_constraint` resolves each row's jokers to ONE
//! substitution -- the bottom takes its maximum, the middle is constrained to
//! at most the bottom, the top to at most the middle.  Everything the
//! Fantasyland best-response frontier computes rests on that collapse being
//! the true optimum over all substitutions, so it is pinned here rather than
//! argued.
//!
//! # What brute force enumerates, and the subtlety that makes it work
//!
//! `available_subs` builds a joker's candidate set from **the row alone**, not
//! from the cards already dealt elsewhere.  That reads like an oversight and is
//! load-bearing: it makes the three rows' substitution sets independent, which
//! is exactly what lets a bottom-up greedy be optimal.  A deck-aware version
//! would couple the rows and this test would fail.  The mirror below reproduces
//! it deliberately -- do not "fix" it.
//!
//! Two jokers in one row are unordered, matching the `for second in (first+1)..`
//! in the evaluator.
//!
//! # Stratification
//!
//! By `(jokers in top, jokers in mid, jokers in bot)`, not by "how many jokers
//! in the hand".  The constraint chain binds per row, and random dealing almost
//! never produces two jokers in the same row, which is where the collapse has
//! the most to lose.  Strata are constructed, not sampled.
//!
//! # What the assertions say
//!
//! * A -- the verdict: `busted` iff no legal assignment exists at all.
//! * B -- componentwise key maximality.  Constant-free, and stronger than
//!   "optimal for the current objective": it implies optimality for every
//!   monotone objective at once, so it survives changes to the royalty tables
//!   or the stay rule.
//! * C -- static optimality at each of the four shipped `fl_ev` widths.
//! * D -- the negative-constant break, asserted AS a break.  The collapse is
//!   provably wrong below roughly -4 (see the module note in
//!   `fl_solver/src/frontier.rs`); this test goes red when someone lands the
//!   per-row Pareto set, which is the signal that the thing under test changed.

use ofc_core::{
    check_fl_stay, create_deck, evaluate_board_with_joker_constraint, evaluate_hand_value,
    get_bottom_royalty, get_middle_royalty, get_top_royalty, Card,
};

/// Mirror of the private `available_subs`: every natural card whose (rank,
/// suit) does not already appear IN THIS ROW.  See the module note.
fn row_subs(row: &[Card]) -> Vec<Card> {
    let mut used = std::collections::HashSet::new();
    for card in row {
        if !card.is_joker() {
            used.insert((card.rank, card.suit));
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

/// Every distinct joker resolution of one row.  Jokerless rows yield exactly
/// one realisation, so the caller needs no special case.
fn realisations(row: &[Card]) -> Vec<Vec<Card>> {
    let joker_slots: Vec<usize> = (0..row.len()).filter(|i| row[*i].is_joker()).collect();
    if joker_slots.is_empty() {
        return vec![row.to_vec()];
    }
    let subs = row_subs(row);
    let mut out = Vec::new();
    match joker_slots.len() {
        1 => {
            for card in &subs {
                let mut filled = row.to_vec();
                filled[joker_slots[0]] = *card;
                out.push(filled);
            }
        }
        _ => {
            // Unordered pairs, matching the evaluator's own enumeration.
            for first in 0..subs.len() {
                for second in (first + 1)..subs.len() {
                    let mut filled = row.to_vec();
                    filled[joker_slots[0]] = subs[first];
                    filled[joker_slots[1]] = subs[second];
                    out.push(filled);
                }
            }
        }
    }
    out
}

struct Truth {
    any_legal: bool,
    best_keys: (u32, u32, u32),
    best_static: [f64; FL_EV.len()],
}

const FL_EV: [f64; 5] = [0.0, 10.7, 29.9, 63.5, -4.5];

/// Exhaustive optimum over the product of the three rows' realisations.
fn brute_force(top: &[Card], mid: &[Card], bot: &[Card]) -> Truth {
    let (tops, mids, bots) = (realisations(top), realisations(mid), realisations(bot));
    let mut truth = Truth {
        any_legal: false,
        best_keys: (0, 0, 0),
        best_static: [f64::NEG_INFINITY; FL_EV.len()],
    };
    for bot_row in &bots {
        let bot_key = evaluate_hand_value(bot_row, 5);
        for mid_row in &mids {
            let mid_key = evaluate_hand_value(mid_row, 5);
            if mid_key > bot_key {
                continue;
            }
            for top_row in &tops {
                let top_key = evaluate_hand_value(top_row, 3);
                if top_key > mid_key {
                    continue;
                }
                truth.any_legal = true;
                truth.best_keys.0 = truth.best_keys.0.max(top_key);
                truth.best_keys.1 = truth.best_keys.1.max(mid_key);
                truth.best_keys.2 = truth.best_keys.2.max(bot_key);
                let royalty = (get_top_royalty(top_row)
                    + get_middle_royalty(mid_row)
                    + get_bottom_royalty(bot_row)) as f64;
                let stays = check_fl_stay(top_row, mid_row, bot_row);
                for (slot, fl_ev) in FL_EV.iter().enumerate() {
                    let value = royalty + if stays { *fl_ev } else { 0.0 };
                    if value > truth.best_static[slot] {
                        truth.best_static[slot] = value;
                    }
                }
            }
        }
    }
    truth
}

fn greedy_static(top: &[Card], mid: &[Card], bot: &[Card], fl_ev: f64) -> Option<f64> {
    let eval = evaluate_board_with_joker_constraint(top, mid, bot);
    if eval.busted {
        return None;
    }
    let royalty = (get_top_royalty(&eval.top)
        + get_middle_royalty(&eval.mid)
        + get_bottom_royalty(&eval.bot)) as f64;
    let stays = check_fl_stay(&eval.top, &eval.mid, &eval.bot);
    Some(royalty + if stays { fl_ev } else { 0.0 })
}

fn shuffled_naturals(next: &mut impl FnMut() -> usize) -> Vec<Card> {
    let mut deck: Vec<Card> = create_deck(false);
    for index in (1..deck.len()).rev() {
        deck.swap(index, next() % (index + 1));
    }
    deck
}

fn cases_per_stratum() -> usize {
    std::env::var("OFC_EXACTNESS_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(600)
}

/// A -- verdict, B -- componentwise key maximality, C -- static optimality at
/// the shipped widths.  Run over every (top, mid, bot) joker placement.
#[test]
fn bottom_up_collapse_is_exact_for_nonnegative_fl_ev() {
    let mut state: u64 = 0x2026_0811;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let cases = cases_per_stratum();
    let mut checked = 0usize;

    for jokers_top in 0..=2usize {
        for jokers_mid in 0..=2usize {
            for jokers_bot in 0..=2usize {
                if jokers_top + jokers_mid + jokers_bot > 2 {
                    continue;
                }
                for _ in 0..cases {
                    let deck = shuffled_naturals(&mut next);
                    let joker = Card { rank: 0, suit: 4 };
                    let mut top = deck[0..3].to_vec();
                    let mut mid = deck[3..8].to_vec();
                    let mut bot = deck[8..13].to_vec();
                    for slot in 0..jokers_top {
                        top[slot] = joker;
                    }
                    for slot in 0..jokers_mid {
                        mid[slot] = joker;
                    }
                    for slot in 0..jokers_bot {
                        bot[slot] = joker;
                    }

                    let truth = brute_force(&top, &mid, &bot);
                    let eval = evaluate_board_with_joker_constraint(&top, &mid, &bot);

                    // A: the verdict is the existence question, exactly.
                    assert_eq!(
                        !eval.busted, truth.any_legal,
                        "verdict disagrees at ({jokers_top},{jokers_mid},{jokers_bot}): \
                         busted={} any_legal={}",
                        eval.busted, truth.any_legal
                    );
                    if eval.busted {
                        continue;
                    }

                    // B: componentwise maximal keys -- constant-free.
                    let keys = (
                        evaluate_hand_value(&eval.top, 3),
                        evaluate_hand_value(&eval.mid, 5),
                        evaluate_hand_value(&eval.bot, 5),
                    );
                    assert!(
                        keys.0 >= truth.best_keys.0
                            && keys.1 >= truth.best_keys.1
                            && keys.2 >= truth.best_keys.2,
                        "collapse is not componentwise maximal at \
                         ({jokers_top},{jokers_mid},{jokers_bot}): greedy={keys:?} \
                         brute={:?}",
                        truth.best_keys
                    );

                    // C: and therefore optimal for every shipped width.
                    for (slot, fl_ev) in FL_EV.iter().enumerate().take(4) {
                        let value = greedy_static(&top, &mid, &bot, *fl_ev).unwrap();
                        assert_eq!(
                            value.to_bits(),
                            truth.best_static[slot].to_bits(),
                            "static differs at fl_ev={fl_ev} \
                             ({jokers_top},{jokers_mid},{jokers_bot}): greedy={value} \
                             brute={}",
                            truth.best_static[slot]
                        );
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0, "no legal boards were checked");
}

/// D -- the break, asserted as a break.
///
/// Below about -4 the collapse is provably wrong: a bottom quads pays royalty
/// 10 and triggers the stay, so at fl_ev = -4.5 it scores 5.5 while the best
/// non-stay bottom (a full house, royalty 6) scores 6.  The collapse only ever
/// builds the quads.  When the per-row Pareto set lands this test goes red,
/// which is the point -- it is the signal that the collapse is no longer what
/// is under test.
#[test]
fn collapse_is_wrong_below_the_negative_threshold() {
    let mut state: u64 = 0x2026_0812;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let negative = FL_EV[4];
    let mut mismatches = 0usize;
    // Jokers in the bottom row: quads are reachable there, which is what the
    // negative constant makes unprofitable.
    for _ in 0..(cases_per_stratum() * 4) {
        let deck = shuffled_naturals(&mut next);
        let joker = Card { rank: 0, suit: 4 };
        let top = deck[0..3].to_vec();
        let mid = deck[3..8].to_vec();
        let mut bot = deck[8..13].to_vec();
        bot[0] = joker;
        bot[1] = joker;

        let truth = brute_force(&top, &mid, &bot);
        if !truth.any_legal {
            continue;
        }
        let value = greedy_static(&top, &mid, &bot, negative).unwrap();
        if value.to_bits() != truth.best_static[4].to_bits() {
            mismatches += 1;
        }
    }
    assert!(
        mismatches > 0,
        "expected the collapse to be suboptimal at fl_ev={negative}; it was not. \
         Either the per-row Pareto set has landed (delete this test and pin the \
         new path instead) or the royalty table changed."
    );
}

/// The frontier reads royalty and the stay off collapsed rows and then dedups
/// rows by value alone, which is only sound if royalty and the stay are
/// functions of the key on joker-free rows.  Middle and bottom royalty are
/// written against the evaluated rank; top royalty and the stay are written
/// against rank counts, so the equivalence is pinned rather than assumed.
#[test]
fn royalty_and_stay_are_functions_of_the_key_on_natural_rows() {
    use std::collections::HashMap;
    let mut state: u64 = 0x2026_0813;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut top_seen: HashMap<u32, (i32, bool)> = HashMap::new();
    let mut mid_seen: HashMap<u32, i32> = HashMap::new();
    let mut bot_seen: HashMap<u32, (i32, bool)> = HashMap::new();
    let filler_mid = vec![
        Card { rank: 2, suit: 0 },
        Card { rank: 3, suit: 1 },
        Card { rank: 4, suit: 2 },
        Card { rank: 5, suit: 3 },
        Card { rank: 7, suit: 0 },
    ];

    for _ in 0..(cases_per_stratum() * 20) {
        let deck = shuffled_naturals(&mut next);
        let top = deck[0..3].to_vec();
        let mid = deck[3..8].to_vec();
        let bot = deck[8..13].to_vec();

        let top_key = evaluate_hand_value(&top, 3);
        let top_fact = (
            get_top_royalty(&top),
            check_fl_stay(&top, &filler_mid, &filler_mid),
        );
        if let Some(previous) = top_seen.insert(top_key, top_fact) {
            assert_eq!(
                previous, top_fact,
                "top key {top_key} maps to two different (royalty, stay) facts"
            );
        }
        let mid_key = evaluate_hand_value(&mid, 5);
        if let Some(previous) = mid_seen.insert(mid_key, get_middle_royalty(&mid)) {
            assert_eq!(previous, get_middle_royalty(&mid), "middle key {mid_key} ambiguous");
        }
        let bot_key = evaluate_hand_value(&bot, 5);
        let bot_fact = (
            get_bottom_royalty(&bot),
            check_fl_stay(&filler_mid[0..3], &filler_mid, &bot),
        );
        if let Some(previous) = bot_seen.insert(bot_key, bot_fact) {
            assert_eq!(
                previous, bot_fact,
                "bottom key {bot_key} maps to two different (royalty, stay) facts"
            );
        }
    }
    assert!(top_seen.len() > 100 && bot_seen.len() > 100, "too few distinct keys sampled");
}
