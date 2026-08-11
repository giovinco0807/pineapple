//! The coarse T2 reply is opt-in, and a run that took it has to say so.
//!
//! The sibling of `fast_t1_rollout_policy` one street later, and the same two
//! claims: turning the fast reply on changes what the rollouts play, so a
//! result produced with it is not comparable to one produced without it and the
//! difference has to be legible in the emitted result rather than inferred from
//! the command that produced it; turning it off has to leave everything exactly
//! as it was.
//!
//! One claim this file makes that neither earlier one could. The coarse T0
//! second-seat reply is reachable from exactly one shape -- a T0 first-seat
//! evaluation -- because acting second the opening it would answer is already
//! on the board. The T2 pair has no such restriction: every rollout that begins
//! at T2 first or any earlier street still has both T2 replies ahead of it. So
//! `a_t1_evaluation_reaches_the_coarse_t2_replies_too` runs the pin through
//! `evaluate_t1`, which never touches a T0 root at all, and holds that the
//! report says the same thing there.
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
use ofc_hu_m3_engine::search::{evaluate_t0, evaluate_t1, T3Config};
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
///
/// A T0 second-seat root rather than a first-seat one, deliberately: it is the
/// cheaper of the two -- seven learned evaluators rather than eight, and no
/// nested opening reply at the head of every rollout -- and both T2 replies are
/// still ahead of it, which is all these tests need.
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

/// Acting second at T1: five cards on the hero's board, seven on the
/// opponent's, three dealt, and nothing discarded yet -- the decision about to
/// be made produces this hand's first discard.
///
/// Used only by the reachability test. A T1 second-seat evaluation needs five
/// learned evaluators, which is the smallest set that still plays both T2
/// replies.
fn t1_second_observation() -> ActorObservation {
    let hero = Board::new(
        ALL_CARDS[3..4].to_vec(),
        ALL_CARDS[4..6].to_vec(),
        ALL_CARDS[6..8].to_vec(),
    )
    .expect("hero board");
    let opponent = Board::new(
        ALL_CARDS[8..10].to_vec(),
        ALL_CARDS[10..12].to_vec(),
        ALL_CARDS[12..15].to_vec(),
    )
    .expect("opponent board");
    ActorObservation::new(
        hero,
        opponent,
        ALL_CARDS[0..3].to_vec(),
        Vec::new(),
        Seat::Second,
        Street::T1,
        ActOrder::Second,
        ScoringContext::default(),
    )
    .expect("observation")
}

/// The seven evaluators a T0 second-seat run needs, all pinned, none fast.
///
/// Deliberately the smallest schedule that still reaches a T2 reply: one
/// particle each and a two-of-232 prefilter, because what is being tested is
/// which function the rollout called, not how well it searched.
fn full_config(run: &str) -> T3Config {
    let (wide_path, wide_sha) = wide();
    T3Config {
        candidate_samples: 1,
        evaluation_samples: 1,
        downstream_t3_samples: 1,
        downstream_t4_samples: 0,
        seed: 981,
        candidate_seed: 982,
        evaluation_seed: 983,
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

/// The same, with both T2 replies pinned to the coarse evaluator.
fn fast_config(run: &str) -> T3Config {
    let (wide_path, wide_sha) = wide();
    T3Config {
        fast_t2_second_model_path: Some(wide_path.clone()),
        fast_t2_second_model_sha256: Some(wide_sha.clone()),
        fast_t2_first_model_path: Some(wide_path),
        fast_t2_first_model_sha256: Some(wide_sha),
        ..full_config(run)
    }
}

fn policy_of(result: &Value) -> &serde_json::Map<String, Value> {
    result["continuation_policy"]
        .as_object()
        .expect("continuation policy is an object")
}

#[test]
fn without_the_fast_models_the_t2_replies_are_reported_as_learned() {
    let result =
        evaluate_t0(&t0_second_observation(), &full_config("full-t2")).expect("evaluates");
    let policy = policy_of(&result);

    assert_eq!(policy["t2_first_evaluator"], "learned");
    assert_eq!(policy["t2_second_evaluator"], "learned");
    // Absence is the claim: a reader that does not know about the fast path
    // must not find a key suggesting one was used.
    assert!(
        !policy.contains_key("fast_t2_first_model_sha256"),
        "a run that pinned no fast model named one"
    );
    assert!(!policy.contains_key("fast_t2_second_model_sha256"));
}

#[test]
fn pinning_the_fast_models_is_recorded_for_both_t2_seats() {
    let result =
        evaluate_t0(&t0_second_observation(), &fast_config("fast-t2")).expect("evaluates");
    let policy = policy_of(&result);
    let (_, wide_sha) = wide();

    assert_eq!(
        policy["t2_first_evaluator"], "learned_fast",
        "a rollout that played the coarse reply must not report the full one"
    );
    assert_eq!(policy["t2_second_evaluator"], "learned_fast");
    assert_eq!(policy["fast_t2_first_model_sha256"], wide_sha);
    assert_eq!(policy["fast_t2_second_model_sha256"], wide_sha);
    // The full-precision pins are still loaded and still named; only the
    // evaluator the rollouts reached has changed.
    assert_eq!(policy["t2_first_model_sha256"], wide_sha);
    assert_eq!(policy["t2_second_model_sha256"], wide_sha);
    // And the T1 replies, which this pin says nothing about, are untouched.
    assert_eq!(policy["t1_first_evaluator"], "learned");
    assert_eq!(policy["t1_second_evaluator"], "learned");
}

#[test]
fn a_fast_model_without_its_digest_is_refused_rather_than_trusted() {
    let observation = t0_second_observation();

    for field in ["fast_t2_first", "fast_t2_second"] {
        let mut unpinned = fast_config("fast-t2-no-digest");
        match field {
            "fast_t2_first" => unpinned.fast_t2_first_model_sha256 = None,
            _ => unpinned.fast_t2_second_model_sha256 = None,
        }
        let error = evaluate_t0(&observation, &unpinned).expect_err("must refuse");
        assert!(
            error.contains(&format!("{field}_model_sha256")),
            "{field}: got {error}"
        );

        let mut wrong = fast_config("fast-t2-wrong-digest");
        match field {
            "fast_t2_first" => wrong.fast_t2_first_model_sha256 = Some("0".repeat(64)),
            _ => wrong.fast_t2_second_model_sha256 = Some("0".repeat(64)),
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
    let full = evaluate_t0(&observation, &full_config("compare-t2")).expect("evaluates");
    let fast = evaluate_t0(&observation, &fast_config("compare-t2")).expect("evaluates");

    assert_ne!(
        policy_of(&full)["t2_first_evaluator"],
        policy_of(&fast)["t2_first_evaluator"]
    );
    assert_ne!(
        full["actions"], fast["actions"],
        "the coarse reply produced the same scores as the full one, which \
         means the rollouts did not reach it"
    );
}

/// The pin bites on a T1 evaluation too, which is what distinguishes this pair
/// from the T0 second-seat one.
///
/// `evaluate_t1` never builds a T0 root and never plays an opening reply, so a
/// run of it is a shape where `fast_t0_second_model_path` would do nothing at
/// all. Both T2 replies are still ahead of every one of its rollouts, so both
/// keys have to move -- and the scores have to move with them, for the same
/// reason as the test above.
#[test]
fn a_t1_evaluation_reaches_the_coarse_t2_replies_too() {
    let observation = t1_second_observation();
    let full = evaluate_t1(&observation, &full_config("t1-full-t2")).expect("evaluates");
    let fast = evaluate_t1(&observation, &fast_config("t1-fast-t2")).expect("evaluates");
    let (_, wide_sha) = wide();

    assert_eq!(policy_of(&full)["t2_first_evaluator"], "learned");
    assert_eq!(policy_of(&full)["t2_second_evaluator"], "learned");
    assert_eq!(policy_of(&fast)["t2_first_evaluator"], "learned_fast");
    assert_eq!(policy_of(&fast)["t2_second_evaluator"], "learned_fast");
    assert_eq!(policy_of(&fast)["fast_t2_first_model_sha256"], wide_sha);
    assert_eq!(policy_of(&fast)["fast_t2_second_model_sha256"], wide_sha);
    assert_ne!(
        full["actions"], fast["actions"],
        "a T1 evaluation played the coarse T2 replies and got the full ones' scores"
    );
}
