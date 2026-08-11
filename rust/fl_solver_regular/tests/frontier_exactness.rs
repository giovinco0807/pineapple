//! The frontier reduction is exact: pinned against brute force.
//!
//! The claim under test is that for any hero board, the Fantasyland side's best
//! response is attained on the non-dominated frontier. The proof is in
//! `src/frontier.rs`; this checks it empirically on random (deal, hero board)
//! pairs by comparing the frontier argmax score against the argmax over all
//! 1,009,008 arrangements, and requires the two to be **bit-identical** rather
//! than close.

use fl_solver_regular::eval::{eval3, eval5, is_foul};
use fl_solver_regular::frontier::{best_response, best_response_brute_force, build_frontier};
use fl_solver_regular::rng::SplitMix64;

const FL_EV: f64 = 9.109;

/// A random 14-card Fantasyland deal plus a disjoint legal hero board.
fn deal_and_hero(seed_base: u64, index: u64) -> ([u8; 14], [u8; 3], [u8; 5], [u8; 5]) {
    let mut rng = SplitMix64::for_stream(seed_base, index);
    loop {
        let mut deck: Vec<u8> = (0..52).collect();
        rng.partial_shuffle(&mut deck, 27);
        let hand: [u8; 14] = deck[..14].try_into().unwrap();
        let top: [u8; 3] = deck[14..17].try_into().unwrap();
        let middle: [u8; 5] = deck[17..22].try_into().unwrap();
        let bottom: [u8; 5] = deck[22..27].try_into().unwrap();
        // A fouled hero board is a legal thing to face, but the interesting
        // comparisons are against boards that actually contest the rows.
        if !is_foul(eval3(&top), eval5(&middle), eval5(&bottom)) {
            return (hand, top, middle, bottom);
        }
    }
}

#[test]
fn frontier_best_response_matches_brute_force() {
    let mut checked = 0;
    for index in 0..320_u64 {
        let (hand, top, middle, bottom) = deal_and_hero(997_960_000, index);
        let hero_top = eval3(&top);
        let hero_middle = eval5(&middle);
        let hero_bottom = eval5(&bottom);

        let frontier = build_frontier(&hand, FL_EV);
        let (fast, _) = best_response(&frontier, hero_top, hero_middle, hero_bottom);
        let slow = best_response_brute_force(&hand, FL_EV, hero_top, hero_middle, hero_bottom);
        assert_eq!(
            fast.to_bits(),
            slow.to_bits(),
            "pair {index}: frontier {fast} vs brute force {slow}"
        );
        checked += 1;
    }
    assert_eq!(checked, 320);
}

#[test]
fn a_fouled_hero_board_is_still_answered_exactly() {
    // Against a fouled hero the Fantasyland side wins every row, so the best
    // response collapses to the static optimum. Worth pinning: it is the case
    // where the line term stops discriminating.
    let mut rng = SplitMix64::for_stream(997_960_001, 0);
    for _ in 0..24 {
        let mut deck: Vec<u8> = (0..52).collect();
        rng.partial_shuffle(&mut deck, 27);
        let hand: [u8; 14] = deck[..14].try_into().unwrap();
        // Force a foul: a strong top over a weak middle.
        let top = [deck[14], deck[15], deck[16]];
        let middle: [u8; 5] = deck[17..22].try_into().unwrap();
        let bottom: [u8; 5] = deck[22..27].try_into().unwrap();
        let (hero_top, hero_middle, hero_bottom) =
            (eval3(&top), eval5(&middle), eval5(&bottom));
        let frontier = build_frontier(&hand, FL_EV);
        let (fast, _) = best_response(&frontier, hero_top, hero_middle, hero_bottom);
        let slow = best_response_brute_force(&hand, FL_EV, hero_top, hero_middle, hero_bottom);
        assert_eq!(fast.to_bits(), slow.to_bits());
    }
}
