//! Differential tests against `ofc_hu_m3_engine`.
//!
//! This crate re-implements hand ranking, royalties and heads-up scoring in a
//! packed form because the engine's `HandValue` allocates a `Vec` per hand and
//! the Fantasyland search compares millions of them. A re-implementation is
//! only safe if it is pinned, so every table and every ordering decision below
//! is checked against the engine over randomized inputs rather than against a
//! handful of fixtures.

use fl_solver_regular::eval::{
    bottom_royalty, eval3, eval5, fl_entry_from_top, is_foul, middle_royalty, top_royalty,
};
use fl_solver_regular::rng::SplitMix64;
use fl_solver_regular::scoring::{symmetric_hu_score, BoardScore};
use ofc_hu_m3_engine::cards::Card;
use ofc_hu_m3_engine::scoring as engine;
use ofc_hu_m3_engine::state::Board;

fn engine_cards(cards: &[u8]) -> Vec<Card> {
    cards
        .iter()
        .map(|card| Card::from_index(*card).expect("card index in range"))
        .collect()
}

fn deal(rng: &mut SplitMix64, count: usize) -> Vec<u8> {
    let mut deck: Vec<u8> = (0..52).collect();
    rng.partial_shuffle(&mut deck, count);
    deck[..count].to_vec()
}

#[test]
fn five_card_ranking_order_matches_the_engine() {
    let mut rng = SplitMix64::for_stream(997_000_000, 1);
    for _ in 0..20_000 {
        let cards = deal(&mut rng, 10);
        let left: [u8; 5] = cards[..5].try_into().unwrap();
        let right: [u8; 5] = cards[5..].try_into().unwrap();

        let packed_order = eval5(&left).cmp(&eval5(&right));
        let engine_order = engine::evaluate_5_card(&engine_cards(&left))
            .unwrap()
            .cmp(&engine::evaluate_5_card(&engine_cards(&right)).unwrap());
        assert_eq!(
            packed_order, engine_order,
            "5-card order disagreed for {left:?} vs {right:?}"
        );
    }
}

#[test]
fn three_card_ranking_order_matches_the_engine() {
    let mut rng = SplitMix64::for_stream(997_000_000, 2);
    for _ in 0..20_000 {
        let cards = deal(&mut rng, 6);
        let left: [u8; 3] = cards[..3].try_into().unwrap();
        let right: [u8; 3] = cards[3..].try_into().unwrap();

        let packed_order = eval3(&left).cmp(&eval3(&right));
        let engine_order = engine::evaluate_3_card(&engine_cards(&left))
            .unwrap()
            .cmp(&engine::evaluate_3_card(&engine_cards(&right)).unwrap());
        assert_eq!(
            packed_order, engine_order,
            "3-card order disagreed for {left:?} vs {right:?}"
        );
    }
}

#[test]
fn mixed_three_versus_five_card_order_matches_the_engine() {
    // This is the comparison the foul check makes, and the one where a naive
    // packing goes wrong: the engine's tie-breaker lists have different lengths
    // for a 3-card and a 5-card hand of the same category.
    let mut rng = SplitMix64::for_stream(997_000_000, 3);
    for _ in 0..40_000 {
        let cards = deal(&mut rng, 8);
        let top: [u8; 3] = cards[..3].try_into().unwrap();
        let middle: [u8; 5] = cards[3..].try_into().unwrap();

        let packed_order = eval3(&top).cmp(&eval5(&middle));
        let engine_order = engine::evaluate_3_card(&engine_cards(&top))
            .unwrap()
            .cmp(&engine::evaluate_5_card(&engine_cards(&middle)).unwrap());
        assert_eq!(
            packed_order, engine_order,
            "mixed order disagreed for top {top:?} vs middle {middle:?}"
        );
    }
}

#[test]
fn royalty_tables_match_the_engine() {
    let mut rng = SplitMix64::for_stream(997_000_000, 4);
    for _ in 0..30_000 {
        let cards = deal(&mut rng, 8);
        let top: [u8; 3] = cards[..3].try_into().unwrap();
        let five: [u8; 5] = cards[3..].try_into().unwrap();

        assert_eq!(
            top_royalty(eval3(&top)),
            engine::get_top_royalty(&engine_cards(&top)).unwrap(),
            "top royalty disagreed for {top:?}"
        );
        assert_eq!(
            middle_royalty(eval5(&five)),
            engine::get_middle_royalty(&engine_cards(&five)).unwrap(),
            "middle royalty disagreed for {five:?}"
        );
        assert_eq!(
            bottom_royalty(eval5(&five)),
            engine::get_bottom_royalty(&engine_cards(&five)).unwrap(),
            "bottom royalty disagreed for {five:?}"
        );
    }
}

#[test]
fn fantasyland_entry_matches_the_engine_including_the_flat_fourteen() {
    let mut rng = SplitMix64::for_stream(997_000_000, 5);
    let mut entries = 0_usize;
    for _ in 0..40_000 {
        let cards = deal(&mut rng, 3);
        let top: [u8; 3] = cards[..3].try_into().unwrap();
        let mine = fl_entry_from_top(eval3(&top));
        let theirs = engine::check_fl_entry(&engine_cards(&top)).unwrap();
        assert_eq!(
            mine.is_some(),
            theirs.qualifies,
            "FL entry disagreed for {top:?}"
        );
        assert_eq!(mine, theirs.entry_type.as_deref());
        if theirs.qualifies {
            entries += 1;
            // Regular rules: every entry type deals 14. No 15/16/17 chain.
            assert_eq!(theirs.card_count, 14);
        }
    }
    assert!(entries > 100, "expected some entries in 40k random tops");
}

#[test]
fn foul_detection_matches_the_engine() {
    let mut rng = SplitMix64::for_stream(997_000_000, 6);
    let mut fouled = 0_usize;
    for _ in 0..20_000 {
        let cards = deal(&mut rng, 13);
        let top: [u8; 3] = cards[..3].try_into().unwrap();
        let middle: [u8; 5] = cards[3..8].try_into().unwrap();
        let bottom: [u8; 5] = cards[8..13].try_into().unwrap();

        let mine = is_foul(eval3(&top), eval5(&middle), eval5(&bottom));
        let board = Board::new(
            engine_cards(&top),
            engine_cards(&middle),
            engine_cards(&bottom),
        )
        .unwrap();
        let theirs = engine::score_board(&board).unwrap();
        assert_eq!(mine, theirs.busted, "foul disagreed for {cards:?}");
        if !mine {
            assert_eq!(
                BoardScore::evaluate(&top, &middle, &bottom).total_royalty,
                theirs.total_royalty
            );
        }
        if mine {
            fouled += 1;
        }
    }
    assert!(fouled > 1_000, "random boards should foul often");
}

#[test]
fn symmetric_heads_up_scoring_matches_the_engine() {
    // Both sides arrived normally, so both are paid on ordinary QQ+ entry.
    // This is the case the engine implements; the Fantasyland-asymmetric case
    // is pinned separately against score_fl_vs_normal's definition.
    let mut rng = SplitMix64::for_stream(997_000_000, 7);
    let fl_ev = 9.109_f64;
    for _ in 0..20_000 {
        let cards = deal(&mut rng, 26);
        let hero_top: [u8; 3] = cards[..3].try_into().unwrap();
        let hero_middle: [u8; 5] = cards[3..8].try_into().unwrap();
        let hero_bottom: [u8; 5] = cards[8..13].try_into().unwrap();
        let villain_top: [u8; 3] = cards[13..16].try_into().unwrap();
        let villain_middle: [u8; 5] = cards[16..21].try_into().unwrap();
        let villain_bottom: [u8; 5] = cards[21..26].try_into().unwrap();

        let mine = symmetric_hu_score(
            &BoardScore::evaluate(&hero_top, &hero_middle, &hero_bottom),
            &BoardScore::evaluate(&villain_top, &villain_middle, &villain_bottom),
            fl_ev,
        );
        let hero = Board::new(
            engine_cards(&hero_top),
            engine_cards(&hero_middle),
            engine_cards(&hero_bottom),
        )
        .unwrap();
        let villain = Board::new(
            engine_cards(&villain_top),
            engine_cards(&villain_middle),
            engine_cards(&villain_bottom),
        )
        .unwrap();
        let (theirs, _) = engine::terminal_score(&hero, Some(&villain), fl_ev).unwrap();
        assert_eq!(mine, theirs, "heads-up score disagreed for {cards:?}");
    }
}
