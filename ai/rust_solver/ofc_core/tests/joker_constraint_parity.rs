//! Does the canonical joker constraint agree with the Python reference on the
//! board the FL audit flagged?  Python reports a foul with royalty 0; the FL
//! solver reports no foul with royalty 20, and it delegates the verdict here.

use ofc_core::{
    evaluate_board_with_joker_constraint, get_bottom_royalty, get_middle_royalty,
    get_top_royalty, Card,
};

fn card(text: &str) -> Card {
    if text == "X1" || text == "X2" {
        return Card { rank: 0, suit: 4 };
    }
    let bytes = text.as_bytes();
    let rank = match bytes[0] as char {
        'T' => 10,
        'J' => 11,
        'Q' => 12,
        'K' => 13,
        'A' => 14,
        other => other.to_digit(10).unwrap() as u8,
    };
    let suit = match bytes[1] as char {
        's' => 0,
        'h' => 1,
        'd' => 2,
        'c' => 3,
        _ => panic!("bad suit"),
    };
    Card { rank, suit }
}

fn row(cards: &[&str]) -> Vec<Card> {
    cards.iter().map(|text| card(text)).collect()
}

#[test]
fn fl_audit_board_is_a_foul() {
    // top KK, middle joker + four spades, bottom J-high heart flush.
    // The middle cannot stay a spade flush (it would beat the bottom), so it
    // drops to a queen pair -- and then the top's kings exceed it, with no
    // joker left on top to rescue the ordering.
    let top = row(&["Kd", "2d", "Kc"]);
    let middle = row(&["Qs", "X2", "7s", "9s", "4s"]);
    let bottom = row(&["Jh", "9h", "3h", "6h", "2h"]);

    let eval = evaluate_board_with_joker_constraint(&top, &middle, &bottom);
    let royalty = if eval.busted {
        0
    } else {
        get_top_royalty(&eval.top) + get_middle_royalty(&eval.mid) + get_bottom_royalty(&eval.bot)
    };
    println!("busted={} royalty={}", eval.busted, royalty);
    assert!(eval.busted, "expected a foul, got royalty {royalty}");
}

/// The same targeted top-vs-middle verdicts as tests/test_top_mid_ordering.py,
/// so the two languages are pinned to identical rulings on the exact boundary
/// the frozen rules contract (Q6) defines.
#[test]
fn top_mid_targeted_verdicts_match_python() {
    let royal = ["As", "Ks", "Qs", "Js", "Ts"];
    let sf_hearts = ["5h", "6h", "7h", "8h", "9h"];
    let cases: Vec<(&[&str], &[&str], &[&str], bool)> = vec![
        (&["2c", "2d", "2h"], &["Kc", "Kd", "Qc", "Qd", "4h"], &royal, true),
        (&["2c", "2d", "2h"], &["3c", "3d", "3h", "4d", "5c"], &royal, false),
        (&["Jc", "Jd", "Jh"], &["Tc", "Td", "Th", "4d", "5c"], &royal, true),
        (&["Qc", "Qd", "Qh"], &["2h", "3d", "4c", "5h", "6d"], &royal, false),
        (&["Qc", "Qd", "Ah"], &["Qh", "Qs", "Kc", "3d", "2c"], &sf_hearts, true),
        (&["Qc", "Qd", "Ah"], &["Qh", "Qs", "Ac", "5d", "4c"], &sf_hearts, false),
        (&["Ah", "Kh", "Qh"], &["Ac", "Qc", "Jd", "9d", "2c"], &royal, true),
        (&["Ah", "Kh", "Qh"], &["Ac", "Kd", "Jd", "9d", "2c"], &royal, true),
        (&["Ah", "Kh", "Qh"], &["Ac", "Kd", "Qc", "9d", "2c"], &royal, false),
        (&["6c", "6d", "2h"], &["Ac", "Kd", "Jd", "9d", "3c"], &royal, true),
        (&["Kc", "Kd", "2h"], &["Ac", "Ad", "Jd", "9d", "3c"], &royal, false),
        (&["Ac", "Ad", "2h"], &["Kc", "Kd", "Jd", "9d", "3c"], &royal, true),
        // Joker on top must drop below a jack-pair middle, not foul.
        (&["Qh", "X1", "2c"], &["Jc", "Jd", "8h", "5d", "3c"], &royal, false),
        // Two jokers on top must drop below even a deuce-pair middle.
        (&["X1", "X2", "Ah"], &["2c", "2d", "8h", "5d", "3c"], &royal, false),
    ];
    for (index, (top, mid, bottom, expected)) in cases.iter().enumerate() {
        let eval = evaluate_board_with_joker_constraint(&row(top), &row(mid), &row(bottom));
        assert_eq!(
            eval.busted, *expected,
            "case {index}: top {top:?} mid {mid:?} expected busted={expected}"
        );
    }
}

/// Bulk invariant mirroring the Python test: on non-busted boards the
/// constrained values are ordered, and on natural boards the verdict equals
/// the raw ordering check exactly.
#[test]
fn top_mid_bulk_invariants() {
    use ofc_core::{create_deck, evaluate_hand_value};
    let mut state: u64 = 0x2026_0731;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let (mut naturals, mut jokered) = (0usize, 0usize);
    for _ in 0..5000 {
        let mut deck = create_deck(true);
        for i in (1..deck.len()).rev() {
            deck.swap(i, next() % (i + 1));
        }
        let top = deck[0..3].to_vec();
        let mid = deck[3..8].to_vec();
        let bottom = deck[8..13].to_vec();
        let eval = evaluate_board_with_joker_constraint(&top, &mid, &bottom);
        let has_joker = deck[..13].iter().any(|card| card.is_joker());
        if !eval.busted {
            let tv = evaluate_hand_value(&eval.top, 3);
            let mv = evaluate_hand_value(&eval.mid, 5);
            let bv = evaluate_hand_value(&eval.bot, 5);
            assert!(tv <= mv && mv <= bv, "ordering violated on non-busted board");
        }
        if !has_joker {
            let raw = evaluate_hand_value(&top, 3) > evaluate_hand_value(&mid, 5)
                || evaluate_hand_value(&mid, 5) > evaluate_hand_value(&bottom, 5);
            assert_eq!(eval.busted, raw, "natural verdict differs from raw ordering");
            naturals += 1;
        } else {
            jokered += 1;
        }
    }
    assert!(naturals > 1000 && jokered > 1000);
}
