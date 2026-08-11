//! Hero's score against a best-responding Fantasyland opponent.
//!
//! # The sign convention, which is where this goes wrong quietly
//!
//! [`crate::frontier::best_response`] returns **the Fantasyland side's** score
//! with hero-only terms dropped:
//!
//! ```text
//! best_response = max over arrangements of
//!                 lines_from_FL_side + fl_royalty + fl_ev[width] * stays
//! ```
//!
//! Hero's heads-up score is the negation of that plus the terms it dropped:
//!
//! ```text
//! hero = -best_response + hero_royalty + fl_ev[hero_entry_width]
//! ```
//!
//! Every piece of that identity is checked in [`tests`] against a
//! head-to-head computed the long way, because a sign error here would move
//! every label by a plausible-looking amount rather than an obviously wrong
//! one.
//!
//! # What changes versus the static library this replaces
//!
//! * The Fantasyland side's royalty is no longer a term of its own -- it is
//!   inside `static_value`, on the arrangement the opponent actually chose.
//! * The stay cost is no longer subtracted afterwards.  The opponent decides
//!   whether to keep Fantasyland as part of choosing its arrangement, so the
//!   term belongs to the maximisation, not outside it.
//! * The "opponent fouled" branch is gone.  A best-responding Fantasyland
//!   player never fouls when a legal arrangement exists, and one always does
//!   (`ofc_core::tests::joker_constraint_exactness` finds no counterexample in
//!   540,000 boards).  An empty frontier is a solver defect, not a fouled
//!   opponent, and it is asserted rather than scored.
//! * Hero fouling is still hero's problem, and is the one case where the
//!   opponent's choice does not depend on hero at all: with hero's rows dead
//!   the opponent simply takes its highest `static_value`.

use crate::frontier::{best_response, FrontierEntry};
use crate::pool::PoolEntry;

/// Hero's finished board, reduced to what scoring reads.
#[derive(Clone, Copy, Debug)]
pub struct HeroTerminal {
    pub busted: bool,
    pub top: u32,
    pub mid: u32,
    pub bot: u32,
    pub royalty: i32,
    /// Fantasyland width hero enters at, or 0.
    pub entry_width: u8,
}

/// Hero's score against one Fantasyland hand, the opponent best-responding.
/// Hero's terminal worth with the opponent deleted: royalty plus the
/// Fantasyland entry it earns, or the foul.
///
/// This is not a score anyone plays for -- it is the diagnostic half of
/// `hero_score`.  Labelling a street twice, once with this and once with the
/// real thing, says how much of the ranking a teacher teaches is about hero's
/// own board and how much is about what the opponent does to it, which is the
/// question that decides whether an encoder needs opponent-side features.
pub fn hero_own(hero: &HeroTerminal, fl_ev: &[f64; 4]) -> f64 {
    if hero.busted {
        return -6.0;
    }
    let entry_ev = if hero.entry_width >= 14 && hero.entry_width <= 17 {
        fl_ev[(hero.entry_width - 14) as usize]
    } else {
        0.0
    };
    hero.royalty as f64 + entry_ev
}

pub fn hero_score(hero: &HeroTerminal, frontier: &[FrontierEntry], fl_ev: &[f64; 4]) -> f64 {
    debug_assert!(
        !frontier.is_empty(),
        "an empty frontier is a solver defect; a Fantasyland hand always has a \
         legal arrangement"
    );
    if hero.busted {
        // Hero's rows are dead, so the opponent's choice cannot depend on them:
        // it takes its best static, and hero pays the foul plus that royalty.
        let best = frontier
            .iter()
            .max_by(|a, b| a.static_value.partial_cmp(&b.static_value).unwrap())
            .expect("non-empty frontier");
        return -6.0 - best.royalty as f64;
    }
    let entry_ev = if hero.entry_width >= 14 && hero.entry_width <= 17 {
        fl_ev[(hero.entry_width - 14) as usize]
    } else {
        0.0
    };
    -best_response(frontier, hero.top, hero.mid, hero.bot) + hero.royalty as f64 + entry_ev
}

/// Mean of [`hero_score`] over drawn opponents.  The caller owns the draw, so
/// a short draw has already been refused where it can be reported.
pub fn hero_score_mean(
    hero: &HeroTerminal,
    entries: &[&PoolEntry],
    fl_ev: &[f64; 4],
) -> f64 {
    assert!(!entries.is_empty(), "no opponents drawn");
    let total: f64 = entries
        .iter()
        .map(|entry| hero_score(hero, &entry.rows, fl_ev))
        .sum();
    total / entries.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontier::{build_frontier, scoop_aware_line};
    use crate::pool::deal;
    use crate::Card;

    const TABLE: [f64; 4] = [0.0, 10.7, 29.9, 63.5];

    /// Hero's score against ONE concrete Fantasyland arrangement, written the
    /// long way: lines from hero's side, royalties both ways, hero's entry, the
    /// opponent's stay.  Nothing is folded.
    fn head_to_head_long_form(
        hero: &HeroTerminal,
        fl: &FrontierEntry,
        fl_ev: &[f64; 4],
        width: u8,
    ) -> f64 {
        if hero.busted {
            return -6.0 - fl.royalty as f64;
        }
        let sign = |own: u32, other: u32| -> i32 {
            match own.cmp(&other) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
            }
        };
        let lines = scoop_aware_line(
            sign(hero.top, fl.top),
            sign(hero.mid, fl.mid),
            sign(hero.bot, fl.bot),
        );
        let hero_entry = if hero.entry_width >= 14 {
            fl_ev[(hero.entry_width - 14) as usize]
        } else {
            0.0
        };
        let stay_cost = if fl.stays { fl_ev[(width - 14) as usize] } else { 0.0 };
        lines as f64 + hero.royalty as f64 - fl.royalty as f64 + hero_entry - stay_cost
    }

    fn hero_from(cards: &[Card]) -> HeroTerminal {
        let core = crate::to_core_cards(cards);
        let eval = ofc_core::evaluate_board_with_joker_constraint(
            &core[0..3], &core[3..8], &core[8..13],
        );
        if eval.busted {
            return HeroTerminal {
                busted: true,
                top: 0,
                mid: 0,
                bot: 0,
                royalty: 0,
                entry_width: 0,
            };
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

    /// **The sign convention.** `hero_score` must equal the long-form
    /// head-to-head against whichever arrangement the opponent actually picked
    /// -- and that arrangement must be the one that is worst for hero.
    #[test]
    fn hero_score_equals_the_long_form_against_the_chosen_arrangement() {
        let mut checked = 0usize;
        for index in 0..40u64 {
            let fl_hand = deal(0x5157_0001, index, 14);
            let frontier = build_frontier(&fl_hand, TABLE[0]);
            for hero_index in 0..6u64 {
                let hero_cards = deal(0x5157_1000 + hero_index, index, 13);
                let hero = hero_from(&hero_cards);

                // What the opponent picked.
                let chosen = frontier
                    .iter()
                    .max_by(|a, b| {
                        let (x, y) = if hero.busted {
                            (a.static_value, b.static_value)
                        } else {
                            (
                                a.score_against(hero.top, hero.mid, hero.bot),
                                b.score_against(hero.top, hero.mid, hero.bot),
                            )
                        };
                        x.partial_cmp(&y).unwrap()
                    })
                    .expect("non-empty");

                let folded = hero_score(&hero, &frontier, &TABLE);
                let long = head_to_head_long_form(&hero, chosen, &TABLE, 14);
                assert_eq!(
                    folded.to_bits(),
                    long.to_bits(),
                    "sign convention broke: folded={folded} long-form={long}"
                );

                // And it is genuinely the opponent's best: no other row hurts
                // hero more.
                if !hero.busted {
                    for row in &frontier {
                        let alternative = head_to_head_long_form(&hero, row, &TABLE, 14);
                        assert!(
                            alternative >= long - 1e-9,
                            "an arrangement hurts hero more than the chosen one: \
                             {alternative} < {long}"
                        );
                    }
                }
                checked += 1;
            }
        }
        assert!(checked > 0);
        println!("checked {checked} (hero board, FL hand) pairs both ways");
    }

    /// A best-responding opponent is never better for hero than a static one.
    /// This is the whole reason the rule correction matters, and it is asserted
    /// rather than assumed: if it ever came out the other way, the sign
    /// convention would be inverted somewhere.
    #[test]
    fn best_response_never_helps_hero_versus_the_static_arrangement() {
        for index in 0..30u64 {
            let fl_hand = deal(0x5157_0002, index, 14);
            let frontier = build_frontier(&fl_hand, TABLE[0]);
            // The static solve's analogue: the arrangement maximising the
            // opponent's own royalty and stay, ignoring hero.
            let static_pick = frontier
                .iter()
                .max_by(|a, b| a.static_value.partial_cmp(&b.static_value).unwrap())
                .expect("non-empty");
            for hero_index in 0..8u64 {
                let hero = hero_from(&deal(0x5157_2000 + hero_index, index, 13));
                if hero.busted {
                    continue;
                }
                let adaptive = hero_score(&hero, &frontier, &TABLE);
                let against_static = head_to_head_long_form(&hero, static_pick, &TABLE, 14);
                assert!(
                    adaptive <= against_static + 1e-9,
                    "hero scored {adaptive} against a best response but only \
                     {against_static} against the static arrangement"
                );
            }
        }
    }

    /// How much the correction is worth, in points per hand.  Not an
    /// assertion -- a measurement, because "an unmeasured amount" is exactly
    /// what the plan document says the old numbers were optimistic by.
    #[test]
    fn how_optimistic_was_the_static_opponent() {
        let mut total = 0.0f64;
        let mut worst: f64 = 0.0;
        let mut count = 0usize;
        for index in 0..60u64 {
            let fl_hand = deal(0x5157_0003, index, 14);
            let frontier = build_frontier(&fl_hand, TABLE[0]);
            let static_pick = *frontier
                .iter()
                .max_by(|a, b| a.static_value.partial_cmp(&b.static_value).unwrap())
                .expect("non-empty");
            for hero_index in 0..10u64 {
                let hero = hero_from(&deal(0x5157_3000 + hero_index, index, 13));
                if hero.busted {
                    continue;
                }
                let gap = head_to_head_long_form(&hero, &static_pick, &TABLE, 14)
                    - hero_score(&hero, &frontier, &TABLE);
                total += gap;
                worst = worst.max(gap);
                count += 1;
            }
        }
        println!(
            "static opponent flattered hero by {:.3} points per hand on average \
             (worst {:.1}), over {count} pairs",
            total / count as f64,
            worst
        );
        assert!(count > 0);
    }
}
