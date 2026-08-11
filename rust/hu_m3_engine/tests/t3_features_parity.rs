//! The Rust T3 encoder must reproduce the Python one exactly.
//!
//! A model reads whatever description it is handed. If the two encoders drift
//! apart nothing raises an error -- the model simply predicts from a position it
//! was never trained on, and the damage shows up as slightly worse play that is
//! very hard to attribute. So parity is asserted feature by feature against a
//! recorded fixture rather than checked by spot inspection.
//!
//! The enumeration counts are compared first. If the two sides disagree about
//! how many completions are legal, every feature downstream is describing a
//! different position and the per-feature differences would be noise about a
//! cause that is already known.

use ofc_hu_m3_engine::cards::Card;
use ofc_hu_m3_engine::infoset::ActorObservation;
use ofc_hu_m3_engine::state::Row;
use ofc_hu_m3_engine::t3_features::{
    encode, side_outlook, unknown_cards, FEATURE_SIZE,
};
use serde::Deserialize;
use std::str::FromStr;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    observation: ActorObservation,
    unknown_count: usize,
    opponent_legal_finishes: usize,
    actions: Vec<ActionCase>,
}

#[derive(Deserialize)]
struct ActionCase {
    placements: Vec<(String, String)>,
    features: Vec<f64>,
}

fn parse_row(token: &str) -> Row {
    match token {
        "top" => Row::Top,
        "middle" => Row::Middle,
        "bottom" => Row::Bottom,
        other => panic!("unknown row {other}"),
    }
}

fn fixture() -> Fixture {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/t3_features_parity.json"
    ))
    .expect("fixture is present");
    serde_json::from_str(&raw).expect("fixture parses")
}

#[test]
fn t3_feature_encoder_matches_the_python_reference_exactly() {
    let fixture = fixture();
    assert!(!fixture.cases.is_empty());

    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut worst_where = String::new();

    for (case_index, case) in fixture.cases.iter().enumerate() {
        let unknown = unknown_cards(&case.observation);
        assert_eq!(
            unknown.len(),
            case.unknown_count,
            "case {case_index}: the two sides disagree about what is unknown"
        );
        let (opponent_outlook, opponent_finishes) =
            side_outlook(&case.observation.opponent_public_board, &unknown);
        assert_eq!(
            opponent_finishes.len(),
            case.opponent_legal_finishes,
            "case {case_index}: the completion enumeration itself has diverged"
        );

        for (action_index, action) in case.actions.iter().enumerate() {
            let placements: Vec<(Card, Row)> = action
                .placements
                .iter()
                .map(|(card, row)| (Card::from_str(card).expect("card"), parse_row(row)))
                .collect();
            let board = case
                .observation
                .hero_board
                .place(&placements)
                .expect("legal placement");
            let produced = encode(
                &case.observation,
                &board,
                &unknown,
                &opponent_outlook,
                &opponent_finishes,
            );

            assert_eq!(action.features.len(), FEATURE_SIZE);
            for (slot, expected) in action.features.iter().enumerate() {
                let difference = (produced[slot] as f64 - expected).abs();
                if difference > worst {
                    worst = difference;
                    worst_where =
                        format!("case {case_index} action {action_index} feature {slot}");
                }
                assert!(
                    difference <= 1e-6,
                    "case {case_index} action {action_index} feature {slot}: \
                     rust {} vs python {expected}",
                    produced[slot]
                );
            }
            compared += 1;
        }
    }

    assert!(compared >= 900, "expected a meaningful number of rows");
    println!(
        "compared {compared} action rows across {} positions; worst difference {worst:.3e} at {worst_where}",
        fixture.cases.len()
    );
}

/// What the encoding costs, which is the only reason to prefer it to a search.
/// Run with `cargo test --release -- --nocapture`; debug is not representative.
#[test]
fn t3_feature_encoding_cost_is_reported() {
    let fixture = fixture();
    let mut prepared = Vec::new();
    for case in &fixture.cases {
        let unknown = unknown_cards(&case.observation);
        let mut boards = Vec::new();
        for action in &case.actions {
            let placements: Vec<(Card, Row)> = action
                .placements
                .iter()
                .map(|(card, row)| (Card::from_str(card).expect("card"), parse_row(row)))
                .collect();
            boards.push(case.observation.hero_board.place(&placements).expect("legal"));
        }
        prepared.push((unknown, boards));
    }

    let rounds = 5;
    let mut sink = 0.0f64;

    let began = std::time::Instant::now();
    for _ in 0..rounds {
        for (index, case) in fixture.cases.iter().enumerate() {
            let (outlook, _finishes) =
                side_outlook(&case.observation.opponent_public_board, &prepared[index].0);
            sink += outlook[0] as f64;
        }
    }
    let opponent_us =
        began.elapsed().as_secs_f64() * 1e6 / (rounds * fixture.cases.len()) as f64;

    let shared: Vec<_> = fixture
        .cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            side_outlook(&case.observation.opponent_public_board, &prepared[index].0)
        })
        .collect();
    let mut rows = 0usize;
    let began = std::time::Instant::now();
    for _ in 0..rounds {
        for (index, case) in fixture.cases.iter().enumerate() {
            let (unknown, boards) = &prepared[index];
            let (outlook, finishes) = &shared[index];
            for board in boards {
                sink += encode(&case.observation, board, unknown, outlook, finishes)[0] as f64;
                rows += 1;
            }
        }
    }
    let per_action_us = began.elapsed().as_secs_f64() * 1e6 / rows as f64;
    let actions =
        prepared.iter().map(|(_u, b)| b.len()).sum::<usize>() as f64 / fixture.cases.len() as f64;

    println!("  opponent outlook   {opponent_us:9.1} us per position");
    println!("  action encoding    {per_action_us:9.1} us per action");
    println!(
        "  amortised total    {:9.1} us per action ({actions:.1} actions per position)",
        opponent_us / actions + per_action_us
    );
    println!("  (the Python encoder measured about 3400 us per action)");
    assert!(sink.is_finite());
}
