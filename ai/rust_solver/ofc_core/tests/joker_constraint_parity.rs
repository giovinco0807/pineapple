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
