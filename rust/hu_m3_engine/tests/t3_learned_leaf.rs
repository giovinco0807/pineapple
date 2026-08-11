//! The learned first-seat leaf is opt-in, and opting out has to cost nothing.
//!
//! The value of this integration is that T3 labels become affordable, but that
//! is worthless if turning it on is hard to distinguish from leaving it off.
//! Two things therefore have to hold and are asserted here rather than assumed.
//!
//! Without configuration the emitted result is what it has always been, key for
//! key, so results and the digests pinned to them are unaffected. With it, the
//! continuation policy says so and names the weights, because a run that does
//! not record which evaluator produced it cannot be compared against one that
//! used the other.

use ofc_hu_m3_engine::cards::{Card, ALL_CARDS};
use ofc_hu_m3_engine::infoset::{ActOrder, ActorObservation, ScoringContext, Seat, Street};
use ofc_hu_m3_engine::search::{evaluate_t3, T3Config};
use ofc_hu_m3_engine::state::Board;
use serde_json::Value;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn model_digest() -> String {
    let raw = std::fs::read_to_string(fixture("t4_model_v5_predictions.json"))
        .expect("predictions fixture");
    let parsed: Value = serde_json::from_str(&raw).expect("parses");
    parsed["weights_sha256"]
        .as_str()
        .expect("digest present")
        .to_owned()
}

fn constrained_board(cards: &[Card]) -> Board {
    let top = cards.len().min(3);
    let middle = (cards.len() - top).min(5);
    Board::new(
        cards[..top].to_vec(),
        cards[top..top + middle].to_vec(),
        cards[top + middle..].to_vec(),
    )
    .unwrap()
}

fn t3_second_observation() -> ActorObservation {
    let hero = constrained_board(&ALL_CARDS[0..9]);
    let opponent = constrained_board(&ALL_CARDS[9..20]);
    ActorObservation::new(
        hero,
        opponent,
        ALL_CARDS[20..23].to_vec(),
        ALL_CARDS[23..25].to_vec(),
        Seat::Second,
        Street::T3,
        ActOrder::Second,
        ScoringContext::default(),
    )
    .unwrap()
}

fn base_config(run: &str) -> T3Config {
    T3Config {
        candidate_samples: 2,
        evaluation_samples: 4,
        downstream_t3_samples: 1,
        downstream_t4_samples: 0,
        seed: 4242,
        candidate_seed: 4243,
        evaluation_seed: 4244,
        run_id: run.to_owned(),
        use_t4_action_cache: true,
        ..Default::default()
    }
}

fn learned_config(run: &str) -> T3Config {
    T3Config {
        learned_t4_model_path: Some(fixture("t4_model_v5.bin")),
        learned_t4_model_sha256: Some(model_digest()),
        ..base_config(run)
    }
}

#[test]
fn leaving_the_learned_leaf_unconfigured_changes_nothing_about_the_result() {
    let observation = t3_second_observation();
    let result = evaluate_t3(&observation, &base_config("exact-mode")).expect("evaluates");
    let policy = result["continuation_policy"]
        .as_object()
        .expect("continuation policy is an object");

    assert_eq!(
        policy["id"], "local_infoset_response_t3_second_t4_v1",
        "the identifier callers already pin must not move when nothing was opted into"
    );
    let mut keys: Vec<&str> = policy.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "downstream_t3_samples",
            "downstream_t4_samples",
            "id",
            "strategy_fusion_guard",
        ],
        "no key may appear that was not there before"
    );
}

#[test]
fn configuring_the_learned_leaf_is_recorded_in_the_result() {
    let observation = t3_second_observation();
    let result =
        evaluate_t3(&observation, &learned_config("learned-mode")).expect("evaluates");
    let policy = result["continuation_policy"]
        .as_object()
        .expect("continuation policy is an object");

    assert_eq!(
        policy["id"], "local_infoset_response_t3_second_t4_learned_v1",
        "a run using the learned leaf must not be mistakable for an exact one"
    );
    assert_eq!(policy["t4_first_evaluator"], "learned");
    assert_eq!(policy["t4_second_evaluator"], "exact_closed_form");
    assert_eq!(policy["t4_first_model_sha256"], model_digest());
}

#[test]
fn weights_that_are_not_the_pinned_ones_are_refused() {
    let observation = t3_second_observation();

    let mut wrong = learned_config("wrong-digest");
    wrong.learned_t4_model_sha256 = Some("0".repeat(64));
    let error = evaluate_t3(&observation, &wrong).expect_err("must refuse");
    assert!(error.contains("does not match the pinned"), "got {error}");

    let mut unpinned = learned_config("no-digest");
    unpinned.learned_t4_model_sha256 = None;
    let error = evaluate_t3(&observation, &unpinned).expect_err("must refuse");
    assert!(
        error.contains("learned_t4_model_sha256"),
        "an unpinned model must be refused rather than trusted, got {error}"
    );

    let mut missing = learned_config("no-file");
    missing.learned_t4_model_path = Some(fixture("does_not_exist.bin"));
    assert!(evaluate_t3(&observation, &missing).is_err());
}

/// Ignored by default, on the grounds the file header of `t2_decision_cost`
/// already states for its own cost reports: this is a wall-clock comparison,
/// and the two sides are not scheduled alike -- the exact path fans its T4
/// children out across rayon while the learned path is one thread, so on a
/// loaded machine the exact side borrows idle cores the learned side cannot.
/// Run in the default parallel suite it therefore reports how busy the binary
/// happened to be rather than what the two evaluators cost, and it flips
/// whenever anything else in the file gets heavier. The measurement is real and
/// worth keeping; it just has to be taken on a quiet machine. Run it with
/// `cargo test --release --test t3_learned_leaf -- --ignored --nocapture`.
#[test]
#[ignore = "wall-clock cost report; run explicitly with --ignored on a quiet machine"]
fn the_learned_leaf_is_the_cheaper_of_the_two() {
    let observation = t3_second_observation();
    let heavy = |mut config: T3Config| {
        config.evaluation_samples = 8;
        config.candidate_samples = 4;
        config
    };

    let began = std::time::Instant::now();
    let exact = evaluate_t3(&observation, &heavy(base_config("cost-exact"))).expect("exact");
    let exact_elapsed = began.elapsed().as_secs_f64();

    let began = std::time::Instant::now();
    let learned =
        evaluate_t3(&observation, &heavy(learned_config("cost-learned"))).expect("learned");
    let learned_elapsed = began.elapsed().as_secs_f64();

    println!(
        "  exact {:.3}s   learned {:.3}s   ratio {:.1}x",
        exact_elapsed,
        learned_elapsed,
        exact_elapsed / learned_elapsed.max(1e-9)
    );
    // Both must still produce a usable decision over the same action set.
    assert_eq!(
        exact["legal_action_count"], learned["legal_action_count"],
        "the two modes must rank the same actions"
    );
    assert!(
        learned_elapsed < exact_elapsed,
        "the learned leaf exists to be cheaper: exact {exact_elapsed:.3}s vs \
         learned {learned_elapsed:.3}s"
    );
}

fn t3_first_observation() -> ActorObservation {
    let hero = constrained_board(&ALL_CARDS[0..9]);
    let opponent = constrained_board(&ALL_CARDS[9..18]);
    ActorObservation::new(
        hero,
        opponent,
        ALL_CARDS[18..21].to_vec(),
        ALL_CARDS[21..23].to_vec(),
        Seat::First,
        Street::T3,
        ActOrder::First,
        ScoringContext::default(),
    )
    .unwrap()
}

fn t3_second_model_digest() -> String {
    let raw = std::fs::read_to_string(fixture("t3_model_v2_predictions.json"))
        .expect("predictions fixture");
    let parsed: Value = serde_json::from_str(&raw).expect("parses");
    parsed["weights_sha256"].as_str().expect("digest").to_owned()
}

/// The first seat is where the nested T3 response actually costs something.
///
/// Acting second there is no nested response to solve, so replacing it saves
/// nothing there. Acting first every rollout plays one, which is most of why
/// that seat costs roughly sixty times what the other does.
///
/// Ignored by default for the same reason as the cost report above: it is a
/// wall-clock comparison between a rayon-parallel sampled path and a
/// single-threaded learned one, which the default parallel suite cannot measure
/// fairly. Its provenance assertions are covered in the default run by
/// `configuring_the_learned_leaf_is_recorded_in_the_result`.
#[test]
#[ignore = "wall-clock cost report; run explicitly with --ignored on a quiet machine"]
fn the_learned_nested_response_is_what_makes_the_first_seat_affordable() {
    let observation = t3_first_observation();
    let heavy = |mut config: T3Config| {
        config.evaluation_samples = 8;
        config.candidate_samples = 4;
        config.downstream_t3_samples = 2;
        config
    };

    let began = std::time::Instant::now();
    let sampled = evaluate_t3(&observation, &heavy(base_config("first-sampled")))
        .expect("sampled nested response");
    let sampled_elapsed = began.elapsed().as_secs_f64();

    let mut learned = heavy(base_config("first-learned"));
    learned.learned_t3_second_model_path = Some(fixture("t3_model_v2.bin"));
    learned.learned_t3_second_model_sha256 = Some(t3_second_model_digest());
    let began = std::time::Instant::now();
    let result = evaluate_t3(&observation, &learned).expect("learned nested response");
    let learned_elapsed = began.elapsed().as_secs_f64();

    println!(
        "  T3 first seat: sampled {:.3}s   learned {:.3}s   ratio {:.1}x",
        sampled_elapsed,
        learned_elapsed,
        sampled_elapsed / learned_elapsed.max(1e-9)
    );

    let policy = result["continuation_policy"]
        .as_object()
        .expect("continuation policy");
    assert_eq!(policy["t3_second_evaluator"], "learned");
    assert_eq!(policy["t3_second_model_sha256"], t3_second_model_digest());
    assert_eq!(
        sampled["legal_action_count"], result["legal_action_count"],
        "both modes must rank the same actions"
    );
    assert!(
        learned_elapsed < sampled_elapsed,
        "the learned nested response exists to be cheaper: sampled \
         {sampled_elapsed:.3}s vs learned {learned_elapsed:.3}s"
    );
}

fn t2_second_observation() -> ActorObservation {
    let hero = constrained_board(&ALL_CARDS[0..7]);
    let opponent = constrained_board(&ALL_CARDS[7..16]);
    ActorObservation::new(
        hero,
        opponent,
        ALL_CARDS[16..19].to_vec(),
        ALL_CARDS[19..20].to_vec(),
        Seat::Second,
        Street::T2,
        ActOrder::Second,
        ScoringContext::default(),
    )
    .unwrap()
}

/// The T2 teacher must not exist in a degraded form.
///
/// Every other evaluator here degrades gracefully to sampling or enumeration;
/// this one cannot, because its fallback costs minutes per decision, so the
/// honest behavior is refusal that names what is missing.
#[test]
fn t2_evaluation_refuses_to_run_without_every_learned_evaluator() {
    use ofc_hu_m3_engine::search::evaluate_t2;

    let observation = t2_second_observation();
    let error = evaluate_t2(&observation, &base_config("t2-bare"))
        .expect_err("must refuse without models");
    assert!(error.contains("requires"), "got {error}");

    let mut partial = learned_config("t2-partial");
    partial.learned_t3_second_model_path = Some(fixture("t3_model_v2.bin"));
    partial.learned_t3_second_model_sha256 = Some(t3_second_model_digest());
    let error = evaluate_t2(&observation, &partial)
        .expect_err("must refuse without the first-seat model");
    assert!(error.contains("learned_t3_first_model"), "got {error}");
}

/// Plumbing only: the first-seat weights do not exist yet, so the second-seat
/// image stands in -- identical 168-dimension architecture, wrong semantics,
/// which the plumbing does not care about. The real weights replace the
/// fixture when M2 finishes, and the provenance fields asserted here are what
/// prevents the stand-in from ever being mistaken for them.
#[test]
fn t2_evaluation_runs_end_to_end_and_records_all_three_evaluators() {
    use ofc_hu_m3_engine::search::evaluate_t2;

    let observation = t2_second_observation();
    let mut config = learned_config("t2-plumbing");
    config.candidate_samples = 2;
    config.evaluation_samples = 3;
    config.learned_t3_second_model_path = Some(fixture("t3_model_v2.bin"));
    config.learned_t3_second_model_sha256 = Some(t3_second_model_digest());
    config.learned_t3_first_model_path = Some(fixture("t3first_model_v1.bin"));
    config.learned_t3_first_model_sha256 = Some(t3_first_model_digest());

    let result = evaluate_t2(&observation, &config).expect("evaluates");
    let policy = result["continuation_policy"]
        .as_object()
        .expect("continuation policy");
    assert_eq!(policy["t4_first_evaluator"], "learned");
    assert_eq!(policy["t3_second_evaluator"], "learned");
    assert_eq!(policy["t3_first_evaluator"], "learned");
    assert_eq!(result["kind"], "t2");
    let rows = result["actions"].as_array().expect("action rows");
    assert!(rows.len() >= 2, "expected several legal actions");
    for row in rows {
        assert!(row["score"].is_f64());
        assert!(row["action_key"].as_str().unwrap().starts_with("rak1:"));
    }
    assert_eq!(
        result["legal_action_count"].as_u64().unwrap() as usize,
        rows.len()
    );
}

/// The in-play T2 first-seat decision has its own model, and its absence must
/// be named rather than papered over with the second seat's weights: the two
/// seats see different opponent geometry, so answering one with the other's
/// evaluator would be a silently wrong policy rather than a missing one.
#[test]
fn t2_first_seat_decide_refuses_without_its_own_model() {
    use ofc_hu_m3_engine::search::decide;

    let observation = t2_first_observation();
    let mut config = learned_config("t2-first-decide-missing");
    config.learned_t3_second_model_path = Some(fixture("t3_model_v2.bin"));
    config.learned_t3_second_model_sha256 = Some(t3_second_model_digest());
    config.learned_t3_first_model_path = Some(fixture("t3first_model_v1.bin"));
    config.learned_t3_first_model_sha256 = Some(t3_first_model_digest());
    config.learned_t2_second_model_path = Some(fixture("t2_model_v1.bin"));
    config.learned_t2_second_model_sha256 = Some(t2_second_model_digest());

    let error = decide(&observation, &config)
        .expect_err("must refuse without the first-seat T2 model");
    assert!(error.contains("learned_t2_first_model"), "got {error}");

    // A path without its digest is refused by the same convention as every
    // other evaluator: unidentified weights are worse than absent ones.
    let mut unpinned = config.clone();
    unpinned.learned_t2_first_model_path = Some(fixture("t2first_model_v1.bin"));
    let error = decide(&observation, &unpinned).expect_err("must refuse unpinned weights");
    assert!(error.contains("learned_t2_first_model_sha256"), "got {error}");
}

/// The assembled policy's T2 first-seat decision, end to end.
///
/// A decision is only usable if it is legal, so what is asserted is the shape
/// the table needs: two of the three dealt cards placed, the third discarded,
/// nothing invented. The continuation policy is checked separately because a
/// run that does not record which evaluator produced it cannot be compared
/// against one that used another.
#[test]
fn t2_first_seat_decide_returns_a_legal_action_and_records_its_evaluator() {
    use ofc_hu_m3_engine::search::{decide, evaluate_t2};

    let observation = t2_first_observation();
    let mut config = learned_config("t2-first-decide");
    config.candidate_samples = 1;
    config.evaluation_samples = 1;
    config.downstream_t3_samples = 1;
    config.downstream_t4_samples = 0;
    config.learned_t3_second_model_path = Some(fixture("t3_model_v2.bin"));
    config.learned_t3_second_model_sha256 = Some(t3_second_model_digest());
    config.learned_t3_first_model_path = Some(fixture("t3first_model_v1.bin"));
    config.learned_t3_first_model_sha256 = Some(t3_first_model_digest());
    config.learned_t2_second_model_path = Some(fixture("t2_model_v1.bin"));
    config.learned_t2_second_model_sha256 = Some(t2_second_model_digest());
    config.learned_t2_first_model_path = Some(fixture("t2first_model_v1.bin"));
    config.learned_t2_first_model_sha256 = Some(t2_first_model_digest());

    let result = decide(&observation, &config).expect("decides");
    assert_eq!(result["kind"], "decide");
    assert_eq!(result["street"], "T2");
    assert_eq!(result["to_act_order"], "first");
    assert_eq!(result["evaluator"], "learned");

    let dealt: Vec<String> = ALL_CARDS[14..17]
        .iter()
        .map(|card| card.as_str().to_owned())
        .collect();
    let placements = result["placements"].as_array().expect("placements");
    let discards = result["discards"].as_array().expect("discards");
    assert_eq!(placements.len(), 2, "T2 places two of the three dealt cards");
    assert_eq!(discards.len(), 1, "and discards the third");

    let mut used: Vec<String> = Vec::new();
    for placement in placements {
        let pair = placement.as_array().expect("[card, row]");
        used.push(pair[0].as_str().expect("card").to_owned());
        assert!(pair[1].is_string(), "a placement names its row");
    }
    used.push(discards[0].as_str().expect("card").to_owned());
    used.sort();
    let mut expected = dealt.clone();
    expected.sort();
    assert_eq!(
        used, expected,
        "every dealt card is accounted for exactly once"
    );

    // Provenance: a run with the first-seat T2 model pinned says so and names
    // the weights, exactly as the other four evaluators do.
    let policy = evaluate_t2(&observation, &config).expect("evaluates")["continuation_policy"]
        .as_object()
        .expect("continuation policy")
        .clone();
    assert_eq!(policy["t2_first_evaluator"], "learned");
    assert_eq!(policy["t2_first_model_sha256"], t2_first_model_digest());
    assert_eq!(policy["t2_second_evaluator"], "learned");
}

/// Acting second at T1 the hero has placed five cards and no discard yet: the
/// first discard of the hand is the one this decision is about. The opponent
/// has already answered T1, so its public board carries seven. That is the
/// geometry `regular_decision_geometry` requires -- (5, 7, 3, 0) -- and an
/// empty hero discard list is part of it rather than an omission.
fn t1_second_observation() -> ActorObservation {
    let hero = constrained_board(&ALL_CARDS[0..5]);
    let opponent = constrained_board(&ALL_CARDS[5..12]);
    ActorObservation::new(
        hero,
        opponent,
        ALL_CARDS[12..15].to_vec(),
        Vec::new(),
        Seat::Second,
        Street::T1,
        ActOrder::Second,
        ScoringContext::default(),
    )
    .unwrap()
}

fn t1_first_observation() -> ActorObservation {
    let hero = constrained_board(&ALL_CARDS[0..5]);
    let opponent = constrained_board(&ALL_CARDS[5..10]);
    ActorObservation::new(
        hero,
        opponent,
        ALL_CARDS[10..13].to_vec(),
        Vec::new(),
        Seat::First,
        Street::T1,
        ActOrder::First,
        ScoringContext::default(),
    )
    .unwrap()
}

fn t1_learned_config(run: &str) -> T3Config {
    let mut config = learned_config(run);
    config.candidate_samples = 1;
    config.evaluation_samples = 1;
    config.downstream_t3_samples = 1;
    config.downstream_t4_samples = 0;
    config.learned_t3_second_model_path = Some(fixture("t3_model_v2.bin"));
    config.learned_t3_second_model_sha256 = Some(t3_second_model_digest());
    config.learned_t3_first_model_path = Some(fixture("t3first_model_v1.bin"));
    config.learned_t3_first_model_sha256 = Some(t3_first_model_digest());
    config.learned_t2_second_model_path = Some(fixture("t2_model_v1.bin"));
    config.learned_t2_second_model_sha256 = Some(t2_second_model_digest());
    config.learned_t2_first_model_path = Some(fixture("t2first_model_v1.bin"));
    config.learned_t2_first_model_sha256 = Some(t2_first_model_digest());
    config
}

/// One street earlier there is one more nested reply again: the opponent's T2
/// first-seat decision, which the T2 evaluator never had to play because at T2
/// the opponent's first-seat turn is already behind it. It has no affordable
/// fallback either, so its absence is named rather than papered over.
#[test]
fn t1_second_seat_refuses_without_the_nested_t2_first_seat_model() {
    use ofc_hu_m3_engine::search::evaluate_t1;

    let observation = t1_second_observation();
    let mut config = t1_learned_config("t1-second-missing");
    config.learned_t2_first_model_path = None;
    config.learned_t2_first_model_sha256 = None;

    let error = evaluate_t1(&observation, &config)
        .expect_err("must refuse without the nested T2 first-seat model");
    assert!(error.contains("learned_t2_first_model"), "got {error}");
}

/// The T1 second seat runs end to end once all five evaluators are pinned.
///
/// Every rollout plays six nested decisions and consumes eighteen particle
/// cards, so the sample counts here are the smallest that still exercise the
/// whole path.
#[test]
fn t1_second_seat_runs_end_to_end_and_records_all_five_evaluators() {
    use ofc_hu_m3_engine::search::evaluate_t1;

    let observation = t1_second_observation();
    let config = t1_learned_config("t1-second-plumbing");

    let result = evaluate_t1(&observation, &config).expect("evaluates");
    let policy = result["continuation_policy"]
        .as_object()
        .expect("continuation policy");
    assert_eq!(policy["t4_first_evaluator"], "learned");
    assert_eq!(policy["t3_second_evaluator"], "learned");
    assert_eq!(policy["t3_first_evaluator"], "learned");
    assert_eq!(policy["t2_second_evaluator"], "learned");
    assert_eq!(policy["t2_first_evaluator"], "learned");
    assert_eq!(policy["t2_first_model_sha256"], t2_first_model_digest());
    assert_eq!(result["kind"], "t1");
    let rows = result["actions"].as_array().expect("action rows");
    assert!(rows.len() >= 2, "expected several legal actions");
    for row in rows {
        assert!(row["score"].is_f64());
        assert!(row["action_key"].as_str().unwrap().starts_with("rak1:"));
    }
    assert_eq!(
        result["legal_action_count"].as_u64().unwrap() as usize,
        rows.len()
    );
}

/// Plumbing only, exactly as at T2: the T1 second-seat weights are still
/// training, so the T2 first-seat image stands in -- identical 168-dimension
/// architecture, wrong semantics, which the plumbing does not care about. The
/// real weights replace the fixture when training finishes, and the provenance
/// fields asserted below are what prevents the stand-in from ever being
/// mistaken for them.
fn t1_second_model_digest() -> String {
    t2_first_model_digest()
}

fn t1_first_learned_config(run: &str) -> T3Config {
    let mut config = t1_learned_config(run);
    config.learned_t1_second_model_path = Some(fixture("t2first_model_v1.bin"));
    config.learned_t1_second_model_sha256 = Some(t1_second_model_digest());
    config
}

/// Acting first at T1 there is one more nested reply than acting second: the
/// opponent's own T1 answer, which the second seat never has to play because by
/// then it is already behind it. It has no affordable fallback either, so the
/// seat that needs it refuses by name rather than answering with the wrong
/// seat's weights.
#[test]
fn t1_first_seat_refuses_without_the_nested_t1_second_seat_model() {
    use ofc_hu_m3_engine::search::evaluate_t1;

    let observation = t1_first_observation();
    let config = t1_learned_config("t1-first-missing");

    let error = evaluate_t1(&observation, &config)
        .expect_err("must refuse without the nested T1 second-seat model");
    assert!(error.contains("learned_t1_second_model"), "got {error}");
}

/// The T1 first seat runs end to end once all six evaluators are pinned.
///
/// Every rollout plays seven nested decisions and consumes twenty-one particle
/// cards -- one decision and one draw more than the second seat -- so the
/// sample counts here are the smallest that still exercise the whole path.
#[test]
fn t1_first_seat_runs_end_to_end_and_records_all_six_evaluators() {
    use ofc_hu_m3_engine::search::evaluate_t1;

    let observation = t1_first_observation();
    let config = t1_first_learned_config("t1-first-plumbing");

    let began = std::time::Instant::now();
    let result = evaluate_t1(&observation, &config).expect("evaluates");
    println!("  T1 first seat end to end: {:.3}s", began.elapsed().as_secs_f64());

    let policy = result["continuation_policy"]
        .as_object()
        .expect("continuation policy");
    assert_eq!(policy["t4_first_evaluator"], "learned");
    assert_eq!(policy["t3_second_evaluator"], "learned");
    assert_eq!(policy["t3_first_evaluator"], "learned");
    assert_eq!(policy["t2_second_evaluator"], "learned");
    assert_eq!(policy["t2_first_evaluator"], "learned");
    assert_eq!(policy["t1_second_evaluator"], "learned");
    assert_eq!(policy["t1_second_model_sha256"], t1_second_model_digest());
    assert_eq!(result["kind"], "t1");
    let rows = result["actions"].as_array().expect("action rows");
    assert!(rows.len() >= 2, "expected several legal actions");
    for row in rows {
        assert!(row["score"].is_f64());
        assert!(row["action_key"].as_str().unwrap().starts_with("rak1:"));
    }
    assert_eq!(
        result["legal_action_count"].as_u64().unwrap() as usize,
        rows.len()
    );
}

/// The assembled policy's T1 second-seat decision, end to end.
///
/// A decision is only usable if it is legal, so what is asserted is the shape
/// the table needs: two of the three dealt cards placed, the third discarded,
/// nothing invented.
#[test]
fn t1_second_seat_decide_returns_a_legal_action_and_records_its_evaluator() {
    use ofc_hu_m3_engine::search::decide;

    let observation = t1_second_observation();
    let config = t1_first_learned_config("t1-second-decide");

    let result = decide(&observation, &config).expect("decides");
    assert_eq!(result["kind"], "decide");
    assert_eq!(result["street"], "T1");
    assert_eq!(result["to_act_order"], "second");
    assert_eq!(result["evaluator"], "learned");

    let dealt: Vec<String> = ALL_CARDS[12..15]
        .iter()
        .map(|card| card.as_str().to_owned())
        .collect();
    let placements = result["placements"].as_array().expect("placements");
    let discards = result["discards"].as_array().expect("discards");
    assert_eq!(placements.len(), 2, "T1 places two of the three dealt cards");
    assert_eq!(discards.len(), 1, "and discards the third");

    let mut used: Vec<String> = Vec::new();
    for placement in placements {
        let pair = placement.as_array().expect("[card, row]");
        used.push(pair[0].as_str().expect("card").to_owned());
        assert!(pair[1].is_string(), "a placement names its row");
    }
    used.push(discards[0].as_str().expect("card").to_owned());
    used.sort();
    let mut expected = dealt.clone();
    expected.sort();
    assert_eq!(
        used, expected,
        "every dealt card is accounted for exactly once"
    );
}

/// Acting second at T0 the hero's board is empty -- the decision is what to put
/// on it -- and the opponent's holds the five cards it has already placed. The
/// opening street deals five and discards none of them, so an empty hero
/// discard list is part of the geometry `regular_decision_geometry` requires,
/// `(0, 5, 5, 0)`, rather than an omission.
fn t0_second_observation() -> ActorObservation {
    let opponent = constrained_board(&ALL_CARDS[0..5]);
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
    .unwrap()
}

/// Acting first at T0 nothing has happened yet: two empty boards, five cards,
/// no discards on either side -- geometry `(0, 0, 5, 0)`.
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
    .unwrap()
}

/// Plumbing only, exactly as at T1 and T2: neither the T1 first-seat nor the T0
/// second-seat weights exist yet, so the T2 first-seat image stands in for both
/// -- identical 168-dimension architecture, wrong semantics, which the plumbing
/// does not care about. The provenance fields asserted below are what prevents
/// a stand-in from ever being mistaken for the real weights.
fn t1_first_model_digest() -> String {
    t2_first_model_digest()
}

fn t0_second_model_digest() -> String {
    t2_first_model_digest()
}

/// The seven evaluators a T0 second-seat root needs, all pinned.
fn t0_second_learned_config(run: &str) -> T3Config {
    let mut config = t1_first_learned_config(run);
    config.learned_t1_first_model_path = Some(fixture("t2first_model_v1.bin"));
    config.learned_t1_first_model_sha256 = Some(t1_first_model_digest());
    config
}

/// The eighth on top of those seven, which only the first seat plays.
fn t0_first_learned_config(run: &str) -> T3Config {
    let mut config = t0_second_learned_config(run);
    config.learned_t0_second_model_path = Some(fixture("t2first_model_v1.bin"));
    config.learned_t0_second_model_sha256 = Some(t0_second_model_digest());
    config
}

fn legal_t0_action_count(observation: &ActorObservation) -> usize {
    use ofc_hu_m3_engine::action::generate_initial_actions;

    generate_initial_actions(&observation.hero_board, &observation.dealt_cards)
        .expect("opening actions")
        .len()
}

/// Test A. One street earlier there is one more nested reply again: the T1
/// first-seat decision, which both T0 seats play -- the hero's own acting first,
/// the opponent's acting second. It has no affordable fallback either, so its
/// absence is named rather than answered with the wrong seat's weights.
#[test]
fn t0_second_seat_refuses_without_the_nested_t1_first_seat_model() {
    use ofc_hu_m3_engine::search::evaluate_t0;

    let observation = t0_second_observation();
    let config = t1_first_learned_config("t0-second-missing");

    let error = evaluate_t0(&observation, &config)
        .expect_err("must refuse without the nested T1 first-seat model");
    assert!(error.contains("learned_t1_first_model"), "got {error}");
}

/// Test B. Acting first at T0 there is an eighth evaluator: the opponent's own
/// opening, which the second seat never has to play because by then it is
/// already behind it.
#[test]
fn t0_first_seat_refuses_without_the_nested_t0_second_seat_model() {
    use ofc_hu_m3_engine::search::evaluate_t0;

    let observation = t0_first_observation();
    let config = t0_second_learned_config("t0-first-missing");

    let error = evaluate_t0(&observation, &config)
        .expect_err("must refuse without the nested T0 second-seat model");
    assert!(error.contains("learned_t0_second_model"), "got {error}");
}

/// Test C. The T0 second seat runs end to end once all seven are pinned.
///
/// Single-stage: no prefilter, so every one of the 232 openings is scored over
/// the full evaluation particle set. This is the exact-comparison path the
/// pruning validation measures the two-stage schedule against, which is why it
/// has to stay available rather than become the schedule's degenerate case.
///
/// Ignored by default, on the same grounds as the nested cost tests in
/// `t2_decision_cost`: 232 openings scored over two particle sets is minutes of
/// single-threaded work, and leaving it in the default parallel run starves the
/// two wall-clock benchmarks in this file -- which race a rayon-parallel exact
/// path against a single-threaded learned one -- into comparing scheduling
/// noise. Run it, and the two-stage test below, with
/// `cargo test --release --test t3_learned_leaf -- --ignored --nocapture`.
#[test]
#[ignore = "minutes of single-threaded work; run explicitly with --ignored"]
fn t0_second_seat_runs_end_to_end_and_records_all_seven_evaluators() {
    use ofc_hu_m3_engine::search::evaluate_t0;

    let observation = t0_second_observation();
    let config = t0_second_learned_config("t0-second-plumbing");
    assert_eq!(config.prefilter_samples, 0, "this is the single-stage path");
    assert_eq!(config.prefilter_keep, 0);

    let began = std::time::Instant::now();
    let result = evaluate_t0(&observation, &config).expect("evaluates");
    println!(
        "  T0 second seat single-stage end to end: {:.3}s",
        began.elapsed().as_secs_f64()
    );

    assert_eq!(result["kind"], "t0");
    assert_eq!(result["street"], "T0");
    assert_eq!(result["root_schedule"], "single_stage");

    let policy = result["continuation_policy"]
        .as_object()
        .expect("continuation policy");
    for evaluator in [
        "t4_first_evaluator",
        "t3_second_evaluator",
        "t3_first_evaluator",
        "t2_second_evaluator",
        "t2_first_evaluator",
        "t1_second_evaluator",
        "t1_first_evaluator",
    ] {
        assert_eq!(policy[evaluator], "learned", "{evaluator} must be recorded");
    }
    assert_eq!(policy["t1_first_model_sha256"], t1_first_model_digest());
    assert!(
        policy.get("t0_second_evaluator").is_none(),
        "the second seat never plays the opponent's opening, so it must not \
         claim to have pinned an evaluator for it"
    );

    let expected_actions = legal_t0_action_count(&observation);
    assert_eq!(
        expected_actions, 232,
        "the opening generator produces the Python 232 semantic actions"
    );
    let rows = result["actions"].as_array().expect("action rows");
    assert_eq!(
        rows.len(),
        expected_actions,
        "every legal opening gets a row"
    );
    assert_eq!(
        result["legal_action_count"].as_u64().unwrap() as usize,
        expected_actions
    );
    for row in rows {
        assert!(row["score"].is_f64());
        assert!(row["action_key"].as_str().unwrap().starts_with("rak1:"));
        assert_eq!(row["placements"].as_array().unwrap().len(), 5);
        assert!(row["discards"].as_array().unwrap().is_empty());
        assert_eq!(
            row["stage"], 2,
            "single-stage scores every action over the full evaluation set"
        );
    }
}

/// Test D. The two-stage schedule prunes without hiding what it pruned.
///
/// Every legal action still has a row, because a label set that silently lost
/// 224 of its 232 actions would be indistinguishable from one where those
/// actions were genuinely worthless. What changes is which particle set
/// produced each row's score, and that is stated per row rather than inferred.
///
/// Ignored by default for the same reason as the single-stage test above.
#[test]
#[ignore = "minutes of single-threaded work; run explicitly with --ignored"]
fn t0_two_stage_prefilter_keeps_every_action_and_says_which_stage_scored_it() {
    use ofc_hu_m3_engine::search::evaluate_t0;

    let observation = t0_second_observation();
    let mut config = t0_second_learned_config("t0-second-prefilter");
    config.prefilter_samples = 1;
    config.prefilter_keep = 8;
    config.evaluation_samples = 2;

    let began = std::time::Instant::now();
    let result = evaluate_t0(&observation, &config).expect("evaluates");
    println!(
        "  T0 second seat two-stage (keep 8 of 232): {:.3}s",
        began.elapsed().as_secs_f64()
    );

    assert_eq!(result["root_schedule"], "two_stage_prefilter");
    assert_eq!(result["prefilter_samples"], 1);
    assert_eq!(result["prefilter_keep"], 8);
    assert_eq!(result["stage_two_action_count"], 8);

    let expected_actions = legal_t0_action_count(&observation);
    let rows = result["actions"].as_array().expect("action rows");
    assert_eq!(
        rows.len(),
        expected_actions,
        "pruning must not remove rows, only change how they were scored"
    );

    let stage_two: Vec<&Value> = rows.iter().filter(|row| row["stage"] == 2).collect();
    assert_eq!(
        stage_two.len(),
        8,
        "exactly the kept actions carry a full-evaluation score"
    );
    assert!(
        rows.iter().filter(|row| row["stage"] == 1).count() == expected_actions - 8,
        "every other action carries its prefilter score"
    );

    let selected = result["selected_action_key"].as_str().expect("selection");
    assert!(
        stage_two
            .iter()
            .any(|row| row["action_key"].as_str() == Some(selected)),
        "the selection must come from an action that was actually re-scored"
    );

    // Provenance is unchanged by the schedule: pruning is a cost decision, not
    // a different continuation policy.
    let policy = result["continuation_policy"]
        .as_object()
        .expect("continuation policy");
    for evaluator in [
        "t4_first_evaluator",
        "t3_second_evaluator",
        "t3_first_evaluator",
        "t2_second_evaluator",
        "t2_first_evaluator",
        "t1_second_evaluator",
        "t1_first_evaluator",
    ] {
        assert_eq!(policy[evaluator], "learned");
    }
    assert_eq!(policy["t1_first_model_sha256"], t1_first_model_digest());
}

/// Test E. The two newly assembled in-play decisions, end to end.
///
/// A decision is only usable if it is legal, and the two streets have different
/// notions of legal: T1 places two of three dealt cards and discards the third,
/// T0 places all five and discards nothing. Both shapes are asserted here
/// because a T0 decision that discarded would be a silently wrong policy rather
/// than a missing one.
#[test]
fn t1_first_and_t0_second_decide_return_legal_actions_for_their_streets() {
    use ofc_hu_m3_engine::search::decide;

    let t1_observation = t1_first_observation();
    let config = t0_first_learned_config("t0-t1-decide");

    let result = decide(&t1_observation, &config).expect("decides T1 first");
    assert_eq!(result["street"], "T1");
    assert_eq!(result["to_act_order"], "first");
    assert_eq!(result["evaluator"], "learned");
    let placements = result["placements"].as_array().expect("placements");
    let discards = result["discards"].as_array().expect("discards");
    assert_eq!(placements.len(), 2, "T1 places two of the three dealt cards");
    assert_eq!(discards.len(), 1, "and discards the third");
    assert_account_for_dealt(&t1_observation, placements, discards);

    let t0_observation = t0_second_observation();
    let result = decide(&t0_observation, &config).expect("decides T0 second");
    assert_eq!(result["street"], "T0");
    assert_eq!(result["to_act_order"], "second");
    assert_eq!(result["evaluator"], "learned");
    let placements = result["placements"].as_array().expect("placements");
    let discards = result["discards"].as_array().expect("discards");
    assert_eq!(placements.len(), 5, "T0 places all five dealt cards");
    assert!(discards.is_empty(), "and discards none of them");
    assert_account_for_dealt(&t0_observation, placements, discards);
}

/// Every dealt card is used exactly once, placed or discarded, and nothing that
/// was not dealt appears.
fn assert_account_for_dealt(
    observation: &ActorObservation,
    placements: &[Value],
    discards: &[Value],
) {
    let mut used: Vec<String> = Vec::new();
    for placement in placements {
        let pair = placement.as_array().expect("[card, row]");
        used.push(pair[0].as_str().expect("card").to_owned());
        assert!(pair[1].is_string(), "a placement names its row");
    }
    for discard in discards {
        used.push(discard.as_str().expect("card").to_owned());
    }
    used.sort();
    let mut expected: Vec<String> = observation
        .dealt_cards
        .iter()
        .map(|card| card.as_str().to_owned())
        .collect();
    expected.sort();
    assert_eq!(used, expected, "every dealt card is accounted for once");
}

fn t3_first_model_digest() -> String {
    let raw = std::fs::read_to_string(fixture("t3first_model_v1_predictions.json"))
        .expect("predictions fixture");
    let parsed: Value = serde_json::from_str(&raw).expect("parses");
    parsed["weights_sha256"].as_str().expect("digest").to_owned()
}

fn t2_second_model_digest() -> String {
    let raw =
        std::fs::read_to_string(fixture("t2_model_v1.sha256.json")).expect("digest fixture");
    let parsed: Value = serde_json::from_str(&raw).expect("parses");
    parsed["weights_sha256"].as_str().expect("digest").to_owned()
}

fn t2_first_model_digest() -> String {
    let raw = std::fs::read_to_string(fixture("t2first_model_v1.sha256.json"))
        .expect("digest fixture");
    let parsed: Value = serde_json::from_str(&raw).expect("parses");
    parsed["weights_sha256"].as_str().expect("digest").to_owned()
}

fn t2_first_observation() -> ActorObservation {
    let hero = constrained_board(&ALL_CARDS[0..7]);
    let opponent = constrained_board(&ALL_CARDS[7..14]);
    ActorObservation::new(
        hero,
        opponent,
        ALL_CARDS[14..17].to_vec(),
        ALL_CARDS[17..18].to_vec(),
        Seat::First,
        Street::T2,
        ActOrder::First,
        ScoringContext::default(),
    )
    .unwrap()
}

/// Acting first there is one more nested reply than acting second: the
/// opponent's own T2 decision. It has no affordable fallback either, so the
/// seat that needs it must refuse by name rather than quietly answer with a
/// search that would take minutes.
#[test]
fn t2_first_seat_refuses_without_the_nested_second_seat_model() {
    use ofc_hu_m3_engine::search::evaluate_t2;

    let observation = t2_first_observation();
    let mut config = learned_config("t2-first-missing");
    config.learned_t3_second_model_path = Some(fixture("t3_model_v2.bin"));
    config.learned_t3_second_model_sha256 = Some(t3_second_model_digest());
    config.learned_t3_first_model_path = Some(fixture("t3first_model_v1.bin"));
    config.learned_t3_first_model_sha256 = Some(t3_first_model_digest());

    let error = evaluate_t2(&observation, &config)
        .expect_err("must refuse without the nested second-seat model");
    assert!(error.contains("learned_t2_second_model"), "got {error}");
}

/// The first seat runs end to end once all four evaluators are pinned.
///
/// Every rollout nests a T2 second-seat model decision on top of the three the
/// second seat already nests, so the sample counts here are the smallest that
/// still exercise the path; the cost is real until a parallel optimization
/// lands.
#[test]
fn t2_first_seat_runs_end_to_end_and_records_all_four_evaluators() {
    use ofc_hu_m3_engine::search::evaluate_t2;

    let observation = t2_first_observation();
    let mut config = learned_config("t2-first-plumbing");
    config.candidate_samples = 1;
    config.evaluation_samples = 1;
    config.downstream_t3_samples = 1;
    config.downstream_t4_samples = 0;
    config.learned_t3_second_model_path = Some(fixture("t3_model_v2.bin"));
    config.learned_t3_second_model_sha256 = Some(t3_second_model_digest());
    config.learned_t3_first_model_path = Some(fixture("t3first_model_v1.bin"));
    config.learned_t3_first_model_sha256 = Some(t3_first_model_digest());
    config.learned_t2_second_model_path = Some(fixture("t2_model_v1.bin"));
    config.learned_t2_second_model_sha256 = Some(t2_second_model_digest());

    let result = evaluate_t2(&observation, &config).expect("evaluates");
    let policy = result["continuation_policy"]
        .as_object()
        .expect("continuation policy");
    assert_eq!(policy["t4_first_evaluator"], "learned");
    assert_eq!(policy["t3_second_evaluator"], "learned");
    assert_eq!(policy["t3_first_evaluator"], "learned");
    assert_eq!(policy["t2_second_evaluator"], "learned");
    assert_eq!(policy["t2_second_model_sha256"], t2_second_model_digest());
    assert_eq!(result["kind"], "t2");
    let rows = result["actions"].as_array().expect("action rows");
    assert!(rows.len() >= 2, "expected several legal actions");
    for row in rows {
        assert!(row["score"].is_f64());
        assert!(row["action_key"].as_str().unwrap().starts_with("rak1:"));
    }
    assert_eq!(
        result["legal_action_count"].as_u64().unwrap() as usize,
        rows.len()
    );
}
