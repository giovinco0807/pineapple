//! The Rust pipeline must reproduce what the trained model actually predicts.
//!
//! Encoder parity is established separately, in `t4_features_parity`. This runs
//! the whole path -- Rust features into the Rust forward pass -- and compares it
//! against PyTorch's own output for the same positions. Checking the halves
//! independently would leave the seam between them untested, and a seam that is
//! merely close produces a model that keeps predicting from a description it was
//! never trained on.

use ofc_hu_m3_engine::cards::Card;
use ofc_hu_m3_engine::infoset::ActorObservation;
use ofc_hu_m3_engine::state::Row;
use ofc_hu_m3_engine::t4_features::{encode_board, opponent_outlook, FEATURE_SIZE};
use ofc_hu_m3_engine::t4_model::Model;
use serde::Deserialize;
use std::str::FromStr;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    observation: ActorObservation,
    actions: Vec<ActionCase>,
}

#[derive(Deserialize)]
struct ActionCase {
    placements: Vec<(String, String)>,
}

#[derive(Deserialize)]
struct Expected {
    weights_sha256: String,
    architecture: Vec<usize>,
    predictions: Vec<f64>,
}

fn parse_row(token: &str) -> Row {
    match token {
        "top" => Row::Top,
        "middle" => Row::Middle,
        "bottom" => Row::Bottom,
        other => panic!("unknown row {other}"),
    }
}

fn fixture_path(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn the_rust_pipeline_reproduces_the_trained_models_predictions() {
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(fixture_path("t4_features_parity.json")).expect("fixture"),
    )
    .expect("fixture parses");
    let expected: Expected = serde_json::from_str(
        &std::fs::read_to_string(fixture_path("t4_model_v5_predictions.json"))
            .expect("predictions"),
    )
    .expect("predictions parse");

    let image = std::fs::read(fixture_path("t4_model_v5.bin")).expect("weights");
    // Pinned, so a swapped or truncated image fails here rather than downstream.
    let model = Model::load_pinned(&image, &expected.weights_sha256).expect("weights load");
    assert_eq!(model.input_dim(), FEATURE_SIZE);
    assert_eq!(expected.architecture[0], FEATURE_SIZE);

    let mut scratch = model.scratch();
    let mut index = 0usize;
    let mut worst = 0.0f64;
    let mut worst_where = String::new();

    for (case_index, case) in fixture.cases.iter().enumerate() {
        let (outlook, legal) = opponent_outlook(&case.observation);
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
            let features = encode_board(&case.observation, &board, &outlook, &legal);
            let produced = model
                .predict_with(&features, &mut scratch)
                .expect("prediction") as f64;

            let difference = (produced - expected.predictions[index]).abs();
            if difference > worst {
                worst = difference;
                worst_where = format!("case {case_index} action {action_index}");
            }
            assert!(
                difference <= 1e-3,
                "case {case_index} action {action_index}: rust {produced} vs \
                 pytorch {}",
                expected.predictions[index]
            );
            index += 1;
        }
    }

    assert_eq!(index, expected.predictions.len(), "every row was compared");
    println!(
        "compared {index} predictions end to end; worst difference {worst:.3e} at {worst_where}"
    );
}

#[test]
fn inference_cost_is_reported() {
    let image = std::fs::read(fixture_path("t4_model_v5.bin")).expect("weights");
    let model = Model::load(&image).expect("weights load");
    let mut scratch = model.scratch();
    let features = vec![0.05f32; FEATURE_SIZE];

    model.predict_with(&features, &mut scratch).expect("warm");
    let rounds = 20_000;
    let began = std::time::Instant::now();
    let mut sink = 0.0f64;
    for _ in 0..rounds {
        sink += model.predict_with(&features, &mut scratch).expect("predict") as f64;
    }
    let per_call = began.elapsed().as_secs_f64() * 1e6 / rounds as f64;
    println!("  forward pass       {per_call:8.2} us per action");
    assert!(sink.is_finite());
}
