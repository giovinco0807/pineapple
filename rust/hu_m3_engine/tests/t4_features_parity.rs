//! The Rust encoder must reproduce the Python one exactly.
//!
//! A model reads whatever description it is handed. If the two encoders drift
//! apart, nothing raises an error -- the model simply predicts from a position
//! it was never trained on, and the damage shows up as slightly worse play that
//! is very hard to attribute. So parity is asserted feature by feature against
//! a recorded fixture rather than checked by spot inspection.

use ofc_hu_m3_engine::cards::Card;
use ofc_hu_m3_engine::infoset::ActorObservation;
use ofc_hu_m3_engine::state::Row;
use ofc_hu_m3_engine::t4_features::{encode_board, opponent_outlook, FEATURE_SIZE};
use serde::Deserialize;
use std::str::FromStr;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    observation: ActorObservation,
    legal_finishes: usize,
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

#[test]
fn t4_feature_encoder_matches_the_python_reference_exactly() {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/t4_features_parity.json"
    ))
    .expect("fixture is present");
    let fixture: Fixture = serde_json::from_str(&raw).expect("fixture parses");
    assert!(!fixture.cases.is_empty(), "fixture must not be empty");

    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut worst_where = String::new();

    for (case_index, case) in fixture.cases.iter().enumerate() {
        let (outlook, legal) = opponent_outlook(&case.observation);
        assert_eq!(
            legal.len(),
            case.legal_finishes,
            "case {case_index}: number of legal opponent finishes disagrees, \
             which means the completion enumeration itself has diverged"
        );

        for (action_index, action) in case.actions.iter().enumerate() {
            let placements: Vec<(Card, Row)> = action
                .placements
                .iter()
                .map(|(card, row)| {
                    (Card::from_str(card).expect("card parses"), parse_row(row))
                })
                .collect();
            let board = case
                .observation
                .hero_board
                .place(&placements)
                .expect("placement is legal");
            let produced = encode_board(&case.observation, &board, &outlook, &legal);

            assert_eq!(action.features.len(), FEATURE_SIZE);
            for (slot, expected) in action.features.iter().enumerate() {
                let difference = (produced[slot] as f64 - expected).abs();
                if difference > worst {
                    worst = difference;
                    worst_where = format!("case {case_index} action {action_index} feature {slot}");
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

    assert!(compared >= 400, "expected a meaningful number of rows");
    println!(
        "compared {compared} action rows across {} positions; worst difference {worst:.3e} at {worst_where}",
        fixture.cases.len()
    );
}

/// What the encoding costs, which is the only reason to prefer it to an exact
/// solve. Run with `cargo test --release -- --nocapture` for a usable number;
/// the debug profile is not representative.
#[test]
fn t4_feature_encoding_cost_is_reported() {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/t4_features_parity.json"
    ))
    .expect("fixture is present");
    let fixture: Fixture = serde_json::from_str(&raw).expect("fixture parses");

    let mut boards = Vec::new();
    for case in &fixture.cases {
        let mut per_case = Vec::new();
        for action in &case.actions {
            let placements: Vec<(Card, Row)> = action
                .placements
                .iter()
                .map(|(card, row)| (Card::from_str(card).expect("card"), parse_row(row)))
                .collect();
            per_case.push(
                case.observation
                    .hero_board
                    .place(&placements)
                    .expect("legal"),
            );
        }
        boards.push(per_case);
    }

    let rounds = 20;
    let mut sink = 0.0f64;

    let began = std::time::Instant::now();
    for _ in 0..rounds {
        for case in &fixture.cases {
            let (outlook, _legal) = opponent_outlook(&case.observation);
            sink += outlook[0] as f64;
        }
    }
    let outlook_us =
        began.elapsed().as_secs_f64() * 1e6 / (rounds * fixture.cases.len()) as f64;

    let prepared: Vec<_> = fixture
        .cases
        .iter()
        .map(|case| opponent_outlook(&case.observation))
        .collect();
    let mut rows = 0usize;
    let began = std::time::Instant::now();
    for _ in 0..rounds {
        for (index, case) in fixture.cases.iter().enumerate() {
            let (outlook, legal) = &prepared[index];
            for board in &boards[index] {
                sink += encode_board(&case.observation, board, outlook, legal)[0] as f64;
                rows += 1;
            }
        }
    }
    let per_action_us = began.elapsed().as_secs_f64() * 1e6 / rows as f64;
    let actions = boards.iter().map(Vec::len).sum::<usize>() as f64
        / fixture.cases.len() as f64;

    println!("  opponent outlook   {outlook_us:8.1} us per position");
    println!("  action encoding    {per_action_us:8.1} us per action");
    println!(
        "  amortised total    {:8.1} us per action ({actions:.2} actions per position)",
        outlook_us / actions + per_action_us
    );
    println!("  (exact T4-first solve is roughly 1100 us inside the search)");
    assert!(sink.is_finite());
}
