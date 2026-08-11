//! The Rust T3 first-seat encoder must reproduce the Python one exactly.
//!
//! Beyond the usual reasons, this encoder has two extra ways to be silently
//! wrong: the strided sampling depends on iteration order, and the opponent's
//! best-arrangement selection breaks ties by first strict maximum. Both are
//! pinned here by comparing survivor counts before features, so a divergence
//! in the enumeration is named as such instead of surfacing as a smear of
//! small per-feature differences.

use ofc_hu_m3_engine::cards::Card;
use ofc_hu_m3_engine::infoset::ActorObservation;
use ofc_hu_m3_engine::state::Row;
use ofc_hu_m3_engine::t3_features::unknown_cards;
use ofc_hu_m3_engine::t3first_features::{
    encode_first, opponent_outlook_first, FEATURE_SIZE,
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
    opponent_survivors: usize,
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
fn t3_first_seat_encoder_matches_the_python_reference_exactly() {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/t3first_features_parity.json"
    ))
    .expect("fixture is present");
    let fixture: Fixture = serde_json::from_str(&raw).expect("fixture parses");
    assert!(!fixture.cases.is_empty());

    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut worst_where = String::new();

    for (case_index, case) in fixture.cases.iter().enumerate() {
        let unknown = unknown_cards(&case.observation);
        assert_eq!(unknown.len(), case.unknown_count,
                   "case {case_index}: unknown-card set disagrees");
        let (opponent_block, opponent_finishes) =
            opponent_outlook_first(&case.observation.opponent_public_board, &unknown)
                .expect("opponent outlook");
        assert_eq!(
            opponent_finishes.len(),
            case.opponent_survivors,
            "case {case_index}: the sampled joint enumeration has diverged -- \
             stride order or tie-break, look there before the features"
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
            let produced = encode_first(
                &case.observation,
                &board,
                &unknown,
                &opponent_block,
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
    assert!(compared >= 600);
    println!(
        "compared {compared} first-seat action rows across {} positions; \
         worst difference {worst:.3e} at {worst_where}",
        fixture.cases.len()
    );
}

/// Cost, because the T2 teacher will call this at every nested response.
#[test]
fn t3_first_seat_encoding_cost_is_reported() {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/t3first_features_parity.json"
    ))
    .expect("fixture is present");
    let fixture: Fixture = serde_json::from_str(&raw).expect("fixture parses");

    let prepared: Vec<_> = fixture
        .cases
        .iter()
        .map(|case| {
            let unknown = unknown_cards(&case.observation);
            let boards: Vec<_> = case
                .actions
                .iter()
                .map(|action| {
                    let placements: Vec<(Card, Row)> = action
                        .placements
                        .iter()
                        .map(|(card, row)| {
                            (Card::from_str(card).expect("card"), parse_row(row))
                        })
                        .collect();
                    case.observation.hero_board.place(&placements).expect("legal")
                })
                .collect();
            (unknown, boards)
        })
        .collect();

    let rounds = 5;
    let mut sink = 0.0f64;
    let began = std::time::Instant::now();
    for _ in 0..rounds {
        for (index, case) in fixture.cases.iter().enumerate() {
            let (outlook, _finishes) = opponent_outlook_first(
                &case.observation.opponent_public_board,
                &prepared[index].0,
            )
            .expect("outlook");
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
            opponent_outlook_first(
                &case.observation.opponent_public_board,
                &prepared[index].0,
            )
            .expect("outlook")
        })
        .collect();
    let mut rows = 0usize;
    let began = std::time::Instant::now();
    for _ in 0..rounds {
        for (index, case) in fixture.cases.iter().enumerate() {
            let (unknown, boards) = &prepared[index];
            let (outlook, finishes) = &shared[index];
            for board in boards {
                sink +=
                    encode_first(&case.observation, board, unknown, outlook, finishes)[0]
                        as f64;
                rows += 1;
            }
        }
    }
    let per_action_us = began.elapsed().as_secs_f64() * 1e6 / rows as f64;
    let actions = prepared.iter().map(|(_u, b)| b.len()).sum::<usize>() as f64
        / fixture.cases.len() as f64;

    println!("  opponent outlook   {opponent_us:9.1} us per position");
    println!("  action encoding    {per_action_us:9.1} us per action");
    println!(
        "  amortised total    {:9.1} us per action ({actions:.1} actions per position)",
        opponent_us / actions + per_action_us
    );
    println!("  (the Python encoder measured about 500 ms per position)");
    assert!(sink.is_finite());
}

/// The whole first-seat pipeline against PyTorch's own output, plus the swap
/// of the T2 plumbing stand-in for the real weights: end-to-end parity is what
/// authorizes using this model inside the T2 teacher.
#[test]
fn the_rust_first_seat_pipeline_reproduces_the_trained_models_predictions() {
    use ofc_hu_m3_engine::t4_model::Model;

    #[derive(Deserialize)]
    struct Expected {
        weights_sha256: String,
        predictions: Vec<f64>,
    }

    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/t3first_features_parity.json"
    ))
    .expect("fixture");
    let fixture: Fixture = serde_json::from_str(&raw).expect("fixture parses");
    let expected: Expected = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/t3first_model_v1_predictions.json"
        ))
        .expect("predictions"),
    )
    .expect("predictions parse");
    let image = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/t3first_model_v1.bin"
    ))
    .expect("weights");
    let model = Model::load_pinned(&image, &expected.weights_sha256).expect("load");
    assert_eq!(model.input_dim(), FEATURE_SIZE);

    let mut scratch = model.scratch();
    let mut index = 0usize;
    let mut worst = 0.0f64;
    for case in &fixture.cases {
        let unknown = unknown_cards(&case.observation);
        let (block, finishes) =
            opponent_outlook_first(&case.observation.opponent_public_board, &unknown)
                .expect("outlook");
        for action in &case.actions {
            let placements: Vec<(Card, Row)> = action
                .placements
                .iter()
                .map(|(card, row)| (Card::from_str(card).expect("card"), parse_row(row)))
                .collect();
            let board = case
                .observation
                .hero_board
                .place(&placements)
                .expect("legal");
            let features =
                encode_first(&case.observation, &board, &unknown, &block, &finishes);
            let produced = model
                .predict_with(&features, &mut scratch)
                .expect("prediction") as f64;
            let difference = (produced - expected.predictions[index]).abs();
            worst = worst.max(difference);
            assert!(
                difference <= 1e-3,
                "row {index}: rust {produced} vs pytorch {}",
                expected.predictions[index]
            );
            index += 1;
        }
    }
    assert_eq!(index, expected.predictions.len());
    println!("compared {index} first-seat predictions end to end; worst {worst:.3e}");
}

/// The six-open-slot generalization against its own Python reference.
///
/// T2-first positions: the hero completes to four open slots, the opponent
/// still holds seven cards and six. The draw space grows from C(n,4) to
/// C(n,6) and the arrangement count from 12 to 90, which is 90 new ways for
/// an iteration order to silently diverge -- hence its own fixture.
#[test]
fn six_slot_free_outlook_matches_the_python_reference_exactly() {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/t2first_features_parity.json"
    ))
    .expect("fixture is present");
    let fixture: Fixture = serde_json::from_str(&raw).expect("fixture parses");
    assert!(!fixture.cases.is_empty());

    use ofc_hu_m3_engine::t3_features::{encode_structural, head_to_head};

    let mut compared = 0usize;
    let mut worst = 0.0f64;
    for (case_index, case) in fixture.cases.iter().enumerate() {
        let unknown = unknown_cards(&case.observation);
        assert_eq!(unknown.len(), case.unknown_count);
        let (opponent_block, opponent_finishes) =
            opponent_outlook_first(&case.observation.opponent_public_board, &unknown)
                .expect("six-slot outlook");
        assert_eq!(
            opponent_finishes.len(),
            case.opponent_survivors,
            "case {case_index}: the six-slot joint enumeration has diverged"
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
            let mut produced = [0.0f32; FEATURE_SIZE];
            produced[..86].copy_from_slice(&encode_structural(&case.observation, &board));
            let (hero_block, hero_finishes) =
                opponent_outlook_first(&board, &unknown).expect("hero outlook");
            produced[86..122].copy_from_slice(&hero_block);
            produced[122..158].copy_from_slice(&opponent_block);
            produced[158..]
                .copy_from_slice(&head_to_head(&hero_finishes, &opponent_finishes));
            for (slot, expected) in action.features.iter().enumerate() {
                let difference = (produced[slot] as f64 - expected).abs();
                worst = worst.max(difference);
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
    assert!(compared >= 200);
    println!("compared {compared} six-slot rows; worst difference {worst:.3e}");
}

/// The eight-open-slot generalization against its own Python reference.
///
/// T1-first positions: the hero completes to six open slots, the opponent still
/// holds eight cards and eight slots. This is the widest the encoder is asked
/// for -- the draw space is `C(39, 8)`, sixty-one million completions, which
/// neither side may materialize, and the arrangement count reaches 560. Both
/// the stride and the first-strict-maximum tie-break get their largest test
/// here.
#[test]
fn eight_slot_free_outlook_matches_the_python_reference_exactly() {
    let fixture = load_eight_slot_fixture();
    assert!(!fixture.cases.is_empty());

    use ofc_hu_m3_engine::t3_features::{encode_structural, head_to_head};

    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut worst_where = String::new();
    for (case_index, case) in fixture.cases.iter().enumerate() {
        let unknown = unknown_cards(&case.observation);
        assert_eq!(unknown.len(), case.unknown_count,
                   "case {case_index}: unknown-card set disagrees");
        let (opponent_block, opponent_finishes) =
            opponent_outlook_first(&case.observation.opponent_public_board, &unknown)
                .expect("eight-slot outlook");
        assert_eq!(
            opponent_finishes.len(),
            case.opponent_survivors,
            "case {case_index}: the eight-slot joint enumeration has diverged -- \
             stride order or tie-break, look there before the features"
        );
        for (action_index, action) in case.actions.iter().enumerate() {
            let board = place(case, action);
            let mut produced = [0.0f32; FEATURE_SIZE];
            produced[..86].copy_from_slice(&encode_structural(&case.observation, &board));
            let (hero_block, hero_finishes) =
                opponent_outlook_first(&board, &unknown).expect("hero outlook");
            produced[86..122].copy_from_slice(&hero_block);
            produced[122..158].copy_from_slice(&opponent_block);
            produced[158..]
                .copy_from_slice(&head_to_head(&hero_finishes, &opponent_finishes));
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
    assert!(compared >= 150);
    println!(
        "compared {compared} eight-slot rows across {} positions; \
         worst difference {worst:.3e} at {worst_where}",
        fixture.cases.len()
    );
}

/// What eight slots cost, reported the way the four-slot cost test does.
///
/// The T1 teacher will pay the opponent block once per node and the hero block
/// once per action, and at eight slots the opponent block is the expensive one,
/// so the two are timed apart rather than only in aggregate.
#[test]
fn eight_slot_encoding_cost_is_reported() {
    let fixture = load_eight_slot_fixture();
    let prepared: Vec<_> = fixture
        .cases
        .iter()
        .map(|case| {
            let unknown = unknown_cards(&case.observation);
            let boards: Vec<_> =
                case.actions.iter().map(|action| place(case, action)).collect();
            (unknown, boards)
        })
        .collect();

    let rounds = 3;
    let mut sink = 0.0f64;
    let began = std::time::Instant::now();
    for _ in 0..rounds {
        for (index, case) in fixture.cases.iter().enumerate() {
            let (outlook, _finishes) = opponent_outlook_first(
                &case.observation.opponent_public_board,
                &prepared[index].0,
            )
            .expect("outlook");
            sink += outlook[0] as f64;
        }
    }
    let opponent_us =
        began.elapsed().as_secs_f64() * 1e6 / (rounds * fixture.cases.len()) as f64;

    let mut rows = 0usize;
    let began = std::time::Instant::now();
    for _ in 0..rounds {
        for (unknown, boards) in &prepared {
            for board in boards {
                let (outlook, _finishes) =
                    opponent_outlook_first(board, unknown).expect("hero outlook");
                sink += outlook[0] as f64;
                rows += 1;
            }
        }
    }
    let hero_us = began.elapsed().as_secs_f64() * 1e6 / rows as f64;
    let actions = prepared.iter().map(|(_u, b)| b.len()).sum::<usize>() as f64
        / fixture.cases.len() as f64;

    println!("  eight-slot opponent outlook  {opponent_us:9.1} us per position");
    println!("  six-slot hero outlook        {hero_us:9.1} us per action");
    println!(
        "  amortised total              {:9.1} us per action \
         ({actions:.1} actions per position)",
        opponent_us / actions + hero_us
    );
    println!("  (the Python reference measured 7-29 s per opponent block)");
    assert!(sink.is_finite());
}

fn load_eight_slot_fixture() -> Fixture {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/t1first_features_parity.json"
    ))
    .expect("fixture is present");
    serde_json::from_str(&raw).expect("fixture parses")
}

fn place(case: &Case, action: &ActionCase) -> ofc_hu_m3_engine::state::Board {
    let placements: Vec<(Card, Row)> = action
        .placements
        .iter()
        .map(|(card, row)| (Card::from_str(card).expect("card"), parse_row(row)))
        .collect();
    case.observation
        .hero_board
        .place(&placements)
        .expect("legal placement")
}
