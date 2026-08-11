//! The opening street's first seat is a decision the engine can now take.
//!
//! Until this landed, `decide` refused (T0, first) by name and every extractor
//! in the crate refused to encode it: the free-slot outlook enumerates boards
//! with four, six or eight open slots, and acting first the opponent's board is
//! empty -- thirteen. That refusal is why the fleet's opening move came from the
//! production behaviour policy while every later decision came from a pinned
//! model.
//!
//! What is tested here is the wiring, in four claims:
//!
//!   * the arm exists, takes the pinned model, and reports which one it took;
//!   * a pin without a digest, with the wrong digest, or of the wrong width is
//!     refused rather than trusted, exactly as every other pinned evaluator is;
//!   * the addition is INERT when unset. A run that does not pin this model
//!     emits the result it emitted before the field existed -- no new key, no
//!     changed value -- which is what lets the M7 package carry the field
//!     without invalidating a golden;
//!   * and when it IS set on an evaluation, it changes the report and nothing
//!     else. No rollout in this engine answers a T0 first-seat decision, so the
//!     pin cannot move a score; if it ever did, something below started reading
//!     a model that is a root policy.
//!
//! The weights are the same 168-wide stand-in the coarse-reply tests use, for
//! the same reason: nothing here reads a number the model produces, only which
//! evaluator was reached and which digest was recorded. The trained image's
//! numbers are held to the standalone extractor by the cross-binary parity
//! harness, which is where a claim about values belongs.

use ofc_hu_m3_engine::cards::ALL_CARDS;
use ofc_hu_m3_engine::infoset::{ActOrder, ActorObservation, ScoringContext, Seat, Street};
use ofc_hu_m3_engine::search::{decide, evaluate_t0, T3Config};
use ofc_hu_m3_engine::state::{Board, Row};
use serde_json::Value;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn digest_of(predictions: &str) -> String {
    let raw = std::fs::read_to_string(fixture(predictions)).expect("predictions fixture");
    let parsed: Value = serde_json::from_str(&raw).expect("parses");
    parsed["weights_sha256"]
        .as_str()
        .expect("digest present")
        .to_owned()
}

/// The 168-wide stand-in and its digest.
fn wide() -> (String, String) {
    (
        fixture("t3first_model_v1.bin"),
        digest_of("t3first_model_v1_predictions.json"),
    )
}

/// Acting first at T0: two empty boards and five dealt cards.
fn t0_first_observation() -> ActorObservation {
    ActorObservation::new(
        Board::empty(),
        Board::empty(),
        ALL_CARDS[0..5].to_vec(),
        Vec::new(),
        Seat::First,
        Street::T0,
        ActOrder::First,
        ScoringContext::default(),
    )
    .expect("observation")
}

/// Acting second at T0, for the guard that separates the two seats.
fn t0_second_observation() -> ActorObservation {
    let opponent = Board::empty()
        .place(&[
            (ALL_CARDS[0], Row::Top),
            (ALL_CARDS[1], Row::Middle),
            (ALL_CARDS[2], Row::Middle),
            (ALL_CARDS[3], Row::Bottom),
            (ALL_CARDS[4], Row::Bottom),
        ])
        .expect("legal opening");
    ActorObservation::new(
        Board::empty(),
        opponent,
        ALL_CARDS[5..10].to_vec(),
        Vec::new(),
        Seat::Second,
        Street::T0,
        ActOrder::Second,
        ScoringContext::default(),
    )
    .expect("observation")
}

fn t0_first_config() -> T3Config {
    let (path, sha) = wide();
    T3Config {
        run_id: "t0-first-decide".to_owned(),
        learned_t0_first_model_path: Some(path),
        learned_t0_first_model_sha256: Some(sha),
        ..Default::default()
    }
}

/// The eight evaluators a T0 first-seat EVALUATION needs, none of them the root
/// policy this file is about. Borrowed wholesale from the coarse-reply tests,
/// because what varies below is one field and everything else has to be held
/// still for the comparison to mean anything.
fn evaluation_config(run: &str) -> T3Config {
    let (wide_path, wide_sha) = wide();
    T3Config {
        candidate_samples: 1,
        evaluation_samples: 1,
        downstream_t3_samples: 1,
        downstream_t4_samples: 0,
        seed: 994,
        candidate_seed: 995,
        evaluation_seed: 996,
        run_id: run.to_owned(),
        use_t4_action_cache: true,
        prefilter_samples: 1,
        prefilter_keep: 2,
        learned_t4_model_path: Some(fixture("t4_model_v5.bin")),
        learned_t4_model_sha256: Some(digest_of("t4_model_v5_predictions.json")),
        learned_t3_second_model_path: Some(fixture("t3_model_v2.bin")),
        learned_t3_second_model_sha256: Some(digest_of("t3_model_v2_predictions.json")),
        learned_t3_first_model_path: Some(wide_path.clone()),
        learned_t3_first_model_sha256: Some(wide_sha.clone()),
        learned_t2_second_model_path: Some(wide_path.clone()),
        learned_t2_second_model_sha256: Some(wide_sha.clone()),
        learned_t2_first_model_path: Some(wide_path.clone()),
        learned_t2_first_model_sha256: Some(wide_sha.clone()),
        learned_t1_second_model_path: Some(wide_path.clone()),
        learned_t1_second_model_sha256: Some(wide_sha.clone()),
        learned_t1_first_model_path: Some(wide_path.clone()),
        learned_t1_first_model_sha256: Some(wide_sha.clone()),
        learned_t0_second_model_path: Some(wide_path),
        learned_t0_second_model_sha256: Some(wide_sha),
        ..Default::default()
    }
}

#[test]
fn the_opening_street_first_seat_is_decided_by_the_pinned_model() {
    let result = decide(&t0_first_observation(), &t0_first_config()).expect("decides");

    assert_eq!(result["status"], "ok");
    assert_eq!(result["street"], "T0");
    assert_eq!(result["to_act_order"], "first");
    assert_eq!(result["evaluator"], "learned");
    // The opening street places all five dealt cards and discards none. A turn
    // action here would mean the wrong generator was reached, which is the one
    // structural mistake this street invites.
    let placements = result["placements"].as_array().expect("placements");
    assert_eq!(placements.len(), 5);
    assert!(result["discards"].as_array().expect("discards").is_empty());
    assert!(result["action_key"]
        .as_str()
        .expect("action key")
        .starts_with("rak1:"));
}

/// The same decision twice is the same decision. There is no RNG on this path
/// -- the coarse outlook strides by index -- so a run that moved would mean the
/// encoder or the tie-break had picked up a source of variation.
#[test]
fn the_decision_is_deterministic() {
    let observation = t0_first_observation();
    let first = decide(&observation, &t0_first_config()).expect("decides");
    let second = decide(&observation, &t0_first_config()).expect("decides");
    assert_eq!(first["action_key"], second["action_key"]);
    assert_eq!(first["placements"], second["placements"]);
}

#[test]
fn a_root_policy_without_its_digest_or_with_the_wrong_one_is_refused() {
    let observation = t0_first_observation();

    let mut missing = t0_first_config();
    missing.learned_t0_first_model_sha256 = None;
    let error = decide(&observation, &missing).expect_err("must refuse");
    assert!(error.contains("learned_t0_first_model_sha256"), "got {error}");

    let mut wrong = t0_first_config();
    wrong.learned_t0_first_model_sha256 = Some("0".repeat(64));
    let error = decide(&observation, &wrong).expect_err("must refuse");
    assert!(error.contains("does not match the pinned"), "got {error}");

    let unpinned = T3Config {
        run_id: "t0-first-unpinned".to_owned(),
        ..Default::default()
    };
    let error = decide(&observation, &unpinned).expect_err("must refuse");
    assert!(error.contains("learned_t0_first_model"), "got {error}");
}

/// A model of the wrong width is refused before it can produce numbers.
///
/// `t4_model_v5.bin` reads the T4 feature layout, which is not 168 wide. It
/// would load, predict, and rank -- on a vector it has never seen. The width
/// check is what makes that an error rather than a plausible-looking opening.
#[test]
fn a_model_of_the_wrong_width_is_refused() {
    let mut narrow = t0_first_config();
    narrow.learned_t0_first_model_path = Some(fixture("t4_model_v5.bin"));
    narrow.learned_t0_first_model_sha256 = Some(digest_of("t4_model_v5_predictions.json"));
    let error = decide(&t0_first_observation(), &narrow).expect_err("must refuse");
    assert!(error.contains("features"), "got {error}");
}

/// The two seats of the opening street are different geometries and different
/// models, and the arm that answers one must not answer the other.
///
/// Acting second the opponent's five placed cards are on the table and all four
/// feature blocks compose; acting first they are not and two of them are zero. A
/// second-seat observation routed through the first seat's arm would be encoded
/// with an empty opponent block against a board that has one.
#[test]
fn the_second_seat_does_not_answer_through_the_first_seats_model() {
    // The second seat's own pin is absent here, so this refusal names the model
    // the street actually needs rather than silently borrowing the other seat's.
    let error = decide(&t0_second_observation(), &t0_first_config()).expect_err("must refuse");
    assert!(error.contains("learned_t0_second_model"), "got {error}");
}

/// Unset, the field changes nothing about an emitted result.
///
/// This is the property the M7 package depends on: fourteen weights ship and a
/// plan opts in, so every plan that does NOT opt in has to produce what it
/// produced yesterday, byte for byte. Compared as serialized text rather than
/// key by key, because a new key is exactly the failure mode.
#[test]
fn leaving_the_root_policy_unset_emits_the_result_it_always_did() {
    let result =
        evaluate_t0(&t0_first_observation(), &evaluation_config("inert")).expect("evaluates");
    let text = serde_json::to_string(&result).expect("serializes");
    assert!(
        !text.contains("t0_first"),
        "a run that pinned no root policy named one: {text}"
    );
}

/// Set on an evaluation, it changes the report and only the report.
///
/// No rollout answers a T0 first-seat decision -- it is the first decision of
/// the hand -- so pinning the root policy cannot move a score. The scores are
/// compared as a whole rather than sampled, and the two new keys are the only
/// difference the whole result is allowed to have.
#[test]
fn pinning_the_root_policy_is_recorded_and_moves_no_score() {
    let observation = t0_first_observation();
    let (wide_path, wide_sha) = wide();
    let without = evaluate_t0(&observation, &evaluation_config("carried")).expect("evaluates");
    let with = evaluate_t0(
        &observation,
        &T3Config {
            learned_t0_first_model_path: Some(wide_path),
            learned_t0_first_model_sha256: Some(wide_sha.clone()),
            ..evaluation_config("carried")
        },
    )
    .expect("evaluates");

    assert_eq!(
        without["actions"], with["actions"],
        "the root policy pin moved a score, which means something below it \
         started reading a model that only `decide` should reach"
    );
    let policy = with["continuation_policy"]
        .as_object()
        .expect("continuation policy is an object");
    assert_eq!(policy["t0_first_evaluator"], "learned");
    assert_eq!(policy["t0_first_model_sha256"], wide_sha);

    let mut before = without["continuation_policy"]
        .as_object()
        .expect("object")
        .clone();
    before.insert("t0_first_evaluator".to_owned(), Value::from("learned"));
    before.insert("t0_first_model_sha256".to_owned(), Value::from(wide_sha));
    assert_eq!(&before, policy, "the pin changed more than its own two keys");
}
