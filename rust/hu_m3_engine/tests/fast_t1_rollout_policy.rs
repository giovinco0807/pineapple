//! The coarse T1 reply is opt-in, and a run that took it has to say so.
//!
//! Two claims, and the second is the one that matters. Turning the fast reply
//! on changes what the T0 evaluator's rollouts play, so a result produced with
//! it is not comparable to one produced without it, and the difference has to
//! be legible in the emitted result rather than inferred from the command that
//! produced it. Turning it off has to leave everything exactly as it was.
//!
//! The weights below are stand-ins: `t3first_model_v1.bin` is a 168-wide model
//! trained for a different street, and it is used here for every slot that
//! wants that width. Nothing in these tests reads a number the models produce
//! -- only which evaluator was reached and which digest was recorded -- and the
//! digest is what distinguishes a run, so a stand-in with a real digest tests
//! the plumbing exactly as a trained one would. The trained weights are pinned
//! by the validation driver, which does read the numbers.

use ofc_hu_m3_engine::cards::ALL_CARDS;
use ofc_hu_m3_engine::infoset::{ActOrder, ActorObservation, ScoringContext, Seat, Street};
use ofc_hu_m3_engine::search::{evaluate_t0, T3Config};
use ofc_hu_m3_engine::state::Board;
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

/// The 168-wide stand-in and its digest, used for every slot of that width.
fn wide() -> (String, String) {
    (
        fixture("t3first_model_v1.bin"),
        digest_of("t3first_model_v1_predictions.json"),
    )
}

/// Acting second at T0: the hero has placed nothing, the opponent has placed
/// five, and five cards are dealt.
fn t0_second_observation() -> ActorObservation {
    let opponent = Board::new(
        ALL_CARDS[5..7].to_vec(),
        ALL_CARDS[7..9].to_vec(),
        ALL_CARDS[9..10].to_vec(),
    )
    .expect("opponent board");
    ActorObservation::new(
        Board::empty(),
        opponent,
        ALL_CARDS[0..5].to_vec(),
        Vec::new(),
        Seat::Second,
        Street::T0,
        ActOrder::Second,
        ScoringContext::default(),
    )
    .expect("observation")
}

/// The seven evaluators a T0 second-seat run needs, all pinned, none fast.
///
/// Deliberately the smallest schedule that still reaches a T1 reply: one
/// particle each and a two-of-232 prefilter, because what is being tested is
/// which function the rollout called, not how well it searched.
fn full_config(run: &str) -> T3Config {
    let (wide_path, wide_sha) = wide();
    T3Config {
        candidate_samples: 1,
        evaluation_samples: 1,
        downstream_t3_samples: 1,
        downstream_t4_samples: 0,
        seed: 991,
        candidate_seed: 992,
        evaluation_seed: 993,
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
        learned_t1_first_model_path: Some(wide_path),
        learned_t1_first_model_sha256: Some(wide_sha),
        ..Default::default()
    }
}

/// The same, with both T1 replies pinned to the coarse evaluator.
fn fast_config(run: &str) -> T3Config {
    let (wide_path, wide_sha) = wide();
    T3Config {
        fast_t1_second_model_path: Some(wide_path.clone()),
        fast_t1_second_model_sha256: Some(wide_sha.clone()),
        fast_t1_first_model_path: Some(wide_path),
        fast_t1_first_model_sha256: Some(wide_sha),
        ..full_config(run)
    }
}

fn policy_of(result: &Value) -> &serde_json::Map<String, Value> {
    result["continuation_policy"]
        .as_object()
        .expect("continuation policy is an object")
}

#[test]
fn without_the_fast_models_the_t1_replies_are_reported_as_learned() {
    let result =
        evaluate_t0(&t0_second_observation(), &full_config("full-t1")).expect("evaluates");
    let policy = policy_of(&result);

    assert_eq!(policy["t1_first_evaluator"], "learned");
    assert_eq!(policy["t1_second_evaluator"], "learned");
    // Absence is the claim: a reader that does not know about the fast path
    // must not find a key suggesting one was used.
    assert!(
        !policy.contains_key("fast_t1_first_model_sha256"),
        "a run that pinned no fast model named one"
    );
    assert!(!policy.contains_key("fast_t1_second_model_sha256"));
}

#[test]
fn pinning_the_fast_models_is_recorded_for_both_t1_seats() {
    let result =
        evaluate_t0(&t0_second_observation(), &fast_config("fast-t1")).expect("evaluates");
    let policy = policy_of(&result);
    let (_, wide_sha) = wide();

    assert_eq!(
        policy["t1_first_evaluator"], "learned_fast",
        "a rollout that played the coarse reply must not report the full one"
    );
    assert_eq!(policy["t1_second_evaluator"], "learned_fast");
    assert_eq!(policy["fast_t1_first_model_sha256"], wide_sha);
    assert_eq!(policy["fast_t1_second_model_sha256"], wide_sha);
    // The full-precision pins are still loaded and still named; only the
    // evaluator the rollouts reached has changed.
    assert_eq!(policy["t1_first_model_sha256"], wide_sha);
    assert_eq!(policy["t1_second_model_sha256"], wide_sha);
}

#[test]
fn a_fast_model_without_its_digest_is_refused_rather_than_trusted() {
    let observation = t0_second_observation();

    for field in ["fast_t1_first", "fast_t1_second"] {
        let mut unpinned = fast_config("fast-no-digest");
        match field {
            "fast_t1_first" => unpinned.fast_t1_first_model_sha256 = None,
            _ => unpinned.fast_t1_second_model_sha256 = None,
        }
        let error = evaluate_t0(&observation, &unpinned).expect_err("must refuse");
        assert!(
            error.contains(&format!("{field}_model_sha256")),
            "{field}: got {error}"
        );

        let mut wrong = fast_config("fast-wrong-digest");
        match field {
            "fast_t1_first" => wrong.fast_t1_first_model_sha256 = Some("0".repeat(64)),
            _ => wrong.fast_t1_second_model_sha256 = Some("0".repeat(64)),
        }
        let error = evaluate_t0(&observation, &wrong).expect_err("must refuse");
        assert!(error.contains("does not match the pinned"), "{field}: got {error}");
    }
}

/// Turning the fast reply on has to actually change the search, or the
/// provenance key above would be reporting a distinction without a difference.
///
/// Asserted as an inequality of the emitted result rather than of a chosen
/// action: on a small schedule the two encoders may well agree about the root,
/// and the claim being made is that the rollouts below it ran a different
/// function, which the recorded policy and the scores both witness. The models
/// here are the same weights in both configurations, so anything that does
/// differ came from the encoder.
#[test]
fn the_fast_reply_is_not_the_full_one_under_another_name() {
    let observation = t0_second_observation();
    let full = evaluate_t0(&observation, &full_config("compare")).expect("evaluates");
    let fast = evaluate_t0(&observation, &fast_config("compare")).expect("evaluates");

    assert_ne!(
        policy_of(&full)["t1_first_evaluator"],
        policy_of(&fast)["t1_first_evaluator"]
    );
    assert_ne!(
        full["actions"], fast["actions"],
        "the coarse reply produced the same scores as the full one, which \
         means the rollouts did not reach it"
    );
}
