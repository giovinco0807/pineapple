//! The two pruning-safety mechanisms, and the promise that neither costs
//! anything until it is asked for.
//!
//! The T0 prefilter was validated on behavior roots. The distributions it has
//! not been validated on -- a cascade relabel, live play -- are exactly the ones
//! where its failure is silent, because a pruned action that was the best one
//! leaves no trace in the emitted result. `prefilter_margin` narrows the
//! opportunity for that failure by refusing to draw the keep boundary through
//! scores stage one cannot separate; `audit_full_every` removes the silence by
//! making every Nth position pay for the single-stage answer and report the
//! comparison.
//!
//! Both are opt-in, and the first test here is the reason: a plan that sets
//! neither must produce the bytes it produced before either mechanism was
//! written, so that every label already generated stays reproducible from the
//! rebuilt engine.
//!
//! The weights are stand-ins, as in `fast_t1_rollout_policy`:
//! `t3first_model_v1.bin` is a 168-wide model trained for another street, used
//! here for every slot of that width. Nothing below reads a number for its
//! poker meaning -- only which action outranked which, and whether two runs
//! agreed -- so a stand-in with a real digest exercises the schedule exactly as
//! trained weights would, and far more cheaply.

use ofc_hu_m3_engine::cards::ALL_CARDS;
use ofc_hu_m3_engine::infoset::{ActOrder, ActorObservation, ScoringContext, Seat, Street};
use ofc_hu_m3_engine::search::{evaluate_t0, T3Config};
use ofc_hu_m3_engine::state::Board;
use serde_json::{Map, Value};
use std::sync::OnceLock;

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
/// five, and five cards are dealt. 232 legal openings.
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

/// The schedule every test below shares: seven pinned evaluators, one particle
/// per stage, keep four of 232.
///
/// The run id and the three seeds are constants rather than parameters because
/// the particles derive from them. Two runs that differ only in a margin or a
/// cadence have to draw the same staged particles, or "nothing else changed"
/// would be untestable.
fn staged_config() -> T3Config {
    let (wide_path, wide_sha) = wide();
    T3Config {
        candidate_samples: 1,
        evaluation_samples: 1,
        downstream_t3_samples: 1,
        downstream_t4_samples: 0,
        seed: 7_001,
        candidate_seed: 7_002,
        evaluation_seed: 7_003,
        run_id: "t0-pruning-safety".to_owned(),
        use_t4_action_cache: true,
        prefilter_samples: 1,
        prefilter_keep: 4,
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

/// The run with both mechanisms at their defaults, computed once for the whole
/// binary. Five of the tests below compare against it, and each one paying for
/// its own copy of a 232-action root would triple this file's cost for no
/// additional claim.
fn baseline() -> &'static Value {
    static BASELINE: OnceLock<Value> = OnceLock::new();
    BASELINE.get_or_init(|| {
        let config = staged_config();
        assert_eq!(config.prefilter_margin, 0.0, "the baseline is unwidened");
        assert_eq!(config.audit_full_every, 0, "and unaudited");
        evaluate_t0(&t0_second_observation(), &config).expect("evaluates")
    })
}

/// The keep whose stage-one boundary this fixture actually separates.
///
/// The schedule's own keep of four lands inside a run of ties, so the gap there
/// is 0.0 -- and a margin of 0.0 is the mechanism switched off, which cannot
/// distinguish `<` from `<=`.
///
/// Six under the v4 FL EV (9.6), where the ladder steps down by 2.0. It was
/// eight, stepping down by 5.0, under v3's 9.109: the Fantasyland line is worth
/// taking again at the higher constant, so every score on this ladder now
/// carries an FL EV term and the whole shape of the ladder moved with it.
const SEPARATED_KEEP: usize = 6;

/// The unwidened run at `SEPARATED_KEEP`, for the one test that needs a
/// same-keep reference rather than the shipped-keep baseline.
fn unwidened_at_separated_keep() -> &'static Value {
    static RESULT: OnceLock<Value> = OnceLock::new();
    RESULT.get_or_init(|| {
        let mut config = staged_config();
        config.prefilter_keep = SEPARATED_KEEP;
        assert_eq!(config.prefilter_margin, 0.0, "the reference is unwidened");
        assert_eq!(config.audit_full_every, 0, "and unaudited");
        evaluate_t0(&t0_second_observation(), &config).expect("evaluates")
    })
}

fn object(value: &Value) -> &Map<String, Value> {
    value.as_object().expect("the result is an object")
}

/// The result with the named keys removed, for comparing a run that added one
/// against a run that could not have.
fn without(value: &Value, keys: &[&str]) -> Value {
    let mut map = object(value).clone();
    for key in keys {
        map.remove(*key);
    }
    Value::Object(map)
}

/// Stage-one scores in the order stage one ranked them.
fn candidate_scores_in_rank_order(result: &Value) -> Vec<f64> {
    let rows = result["actions"].as_array().expect("action rows");
    let mut ordered = vec![f64::NAN; rows.len()];
    for row in rows {
        let rank = row["sorted_index"].as_u64().expect("sorted_index") as usize;
        ordered[rank] = row["candidate_score"].as_f64().expect("candidate_score");
    }
    ordered
}

/// The stage-one gap across the boundary a keep of `keep` would draw.
fn boundary_gap(result: &Value, keep: usize) -> f64 {
    let ordered = candidate_scores_in_rank_order(result);
    ordered[keep - 1] - ordered[keep]
}

/// Which actions were rescored at stage two, by their stage-one rank.
fn stage_two_ranks(result: &Value) -> Vec<usize> {
    let mut ranks = result["actions"]
        .as_array()
        .expect("action rows")
        .iter()
        .filter(|row| row["stage"] == 2)
        .map(|row| row["sorted_index"].as_u64().expect("sorted_index") as usize)
        .collect::<Vec<_>>();
    ranks.sort_unstable();
    ranks
}

/// The kept set has to be a PREFIX of the stage-one ranking, widened or not.
/// Asserting only the count would be satisfied by a rule that admitted an
/// arbitrary set of the right size.
fn assert_kept_is_the_top(result: &Value, expected: usize) {
    assert_eq!(
        result["stage_two_action_count"].as_u64().expect("count") as usize,
        expected,
        "stage_two_action_count"
    );
    assert_eq!(
        stage_two_ranks(result),
        (0..expected).collect::<Vec<_>>(),
        "the rescored actions must be exactly the top {expected} of the \
         stage-one ranking"
    );
}

/// Test 1. The regression pin.
///
/// The fixture was written by the engine as it stood before either mechanism
/// existed, and the claim is the strongest available one: the serialized result
/// is the same sequence of bytes. Anything weaker -- the same selection, the
/// same scores to some tolerance -- would tolerate a change in what a plan that
/// asked for neither mechanism gets, and every label already generated was
/// generated by a plan that asked for neither.
///
/// It pins this build on this platform. That is the scope of the promise being
/// made: the same source, rebuilt, still answers the same way.
///
/// The fixture is `v3`, recaptured on 2026-08-06 when the FL EV moved from
/// v3-the-config's 9.109 to the self-play fixed point 9.6 that
/// `configs/fl_ev_regular_v4_selfplay.json` now carries. `v1` (June's
/// 10.227020614683454) and `v2` (9.109) both stay on disk, unmodified, as the
/// record of what the engine answered under each earlier economics.
///
/// Each recapture was justified rather than assumed. With only `DEFAULT_FL_EV`
/// and `DEFAULT_FL_EV_14` reverted, this same source reproduces the previous
/// pin byte for byte, so the entire delta is attributable to the constant and
/// to nothing else. Structurally all three captures agree: the same twenty-three
/// top-level keys, the same 232 actions carrying identical action keys,
/// placements, discards and original indices -- in the same order -- the same
/// two stages with four actions promoted, the promoted set still exactly the
/// top four of the stage-one ranking, and the same selected action, still the
/// stage-two argmax.
///
/// What moves is the arithmetic. Every score in all three files is a whole
/// number of points plus a whole number of FL EV terms. v1 carried such a term
/// where v2 carried none -- 9.109 had made the Fantasy Land line no longer worth
/// taking -- and at 9.6 it is worth taking again, so v3's ladder carries the
/// term throughout and its top scores are 38.6/30.6/28.6 where v2's were
/// 14.0/10.0/5.0. That is also why four other tests in this file moved rungs:
/// they are written around this one ladder's gaps.
///
/// To recapture again, point `REGEN_T0_PIN` at a path: the test writes what it
/// produced there and then still asserts, so a regeneration cannot pass a
/// broken result off as a pin. Write it somewhere else and diff before
/// promoting it.
#[test]
fn defaults_reproduce_the_pre_mechanism_result_byte_for_byte() {
    let pinned = std::fs::read_to_string(fixture("t0_two_stage_result_pin_v3.json"))
        .expect("the pinned result fixture");
    let produced = serde_json::to_string(baseline()).expect("serializes");

    if let Ok(path) = std::env::var("REGEN_T0_PIN") {
        std::fs::write(path, &produced).expect("write");
    }

    assert_eq!(
        produced.len(),
        pinned.trim_end().len(),
        "the result changed length; a plan that asked for neither mechanism \
         must get the bytes it got before they existed"
    );
    assert_eq!(produced, pinned.trim_end());

    // And the two new keys are absent rather than present-and-zero: a reader
    // written before this change must not find a key it cannot interpret.
    let map = object(baseline());
    assert!(!map.contains_key("prefilter_margin"));
    assert!(!map.contains_key("audit_full_every"));
    assert!(!map.contains_key("audit"));
}

/// Test 2. A margin the boundary gap does not fit under changes nothing.
///
/// The gap across the keep-eight boundary of this fixture is exactly 5.0, so a
/// margin of exactly 5.0 is the case that separates `<` from `<=`. Under `<` it
/// must not extend, and the result must be the unwidened one key for key apart
/// from the margin's own echo -- not merely the same count, since a rule that
/// extended and then contracted would also produce the same count.
///
/// It is asked at `SEPARATED_KEEP` rather than at the schedule's own keep of
/// four because, under the re-measured FL EV, four falls inside a run of
/// stage-one ties: the gap there is 0.0, and a margin of 0.0 is the mechanism
/// switched off, which is the one setting that cannot tell `<` from `<=`. The
/// comparison is therefore against the unwidened run at the same keep, not
/// against `baseline()`.
#[test]
fn a_margin_equal_to_the_boundary_gap_does_not_extend_the_keep_set() {
    let gap = boundary_gap(baseline(), SEPARATED_KEEP);
    assert_eq!(
        gap, 2.0,
        "this test is written around the fixture's keep-six boundary gap; if \
         the schedule, the weights or the FL EV moved, re-derive the margins \
         below"
    );

    let mut config = staged_config();
    config.prefilter_keep = SEPARATED_KEEP;
    config.prefilter_margin = gap;
    let result = evaluate_t0(&t0_second_observation(), &config).expect("evaluates");

    assert_kept_is_the_top(&result, SEPARATED_KEEP);
    assert_eq!(result["prefilter_margin"], 2.0, "the margin is reported");
    assert_eq!(
        without(&result, &["prefilter_margin"]),
        *unwidened_at_separated_keep(),
        "a margin that admitted nothing must leave the result untouched"
    );
}

/// Test 3. A margin the gap does fit under admits the next action, and only as
/// far as the next gap allows.
///
/// Keep five, where this fixture's stage-one scores tie across the boundary
/// (gap 0.0). A margin of 2.0 admits rank five, re-anchors on it, and then
/// meets a gap of exactly 2.0 -- which does not fit under the margin, so the
/// walk stops. One run therefore witnesses three things at once: that the
/// boundary widens, that it re-anchors on each action it admits rather than
/// measuring everything against the original boundary, and that it stops
/// somewhere short of the cap rather than running to it.
///
/// It was keep seven at margin 5.0 under v3's FL EV. The shape of the claim is
/// unchanged; only the rung of the ladder that exhibits it moved.
#[test]
fn a_margin_the_boundary_gap_fits_under_extends_it_and_then_stops() {
    let ordered = candidate_scores_in_rank_order(baseline());
    assert_eq!(
        ordered[4] - ordered[5],
        0.0,
        "stage one could not separate ranks five and six in this fixture"
    );
    assert_eq!(
        ordered[5] - ordered[6],
        2.0,
        "and could separate ranks six and seven by exactly 2.0"
    );

    let mut config = staged_config();
    config.prefilter_keep = 5;
    config.prefilter_margin = 2.0;
    let result = evaluate_t0(&t0_second_observation(), &config).expect("evaluates");

    assert_kept_is_the_top(&result, 6);
    assert!(
        (result["stage_two_action_count"].as_u64().unwrap() as usize) < 2 * 5,
        "the walk stopped at a gap it could not fit under, not at the cap"
    );
}

/// Test 4. The cap is hard.
///
/// The keep-four boundary of this fixture sits inside a run of stage-one ties,
/// and the largest step anywhere in the whole 232-action ladder is 12.6. So a
/// margin a hair above 12.6 has nothing anywhere that can stop it: an uncapped
/// walk would run to the bottom. What stops it is the cap, and the cap is
/// exactly twice `prefilter_keep` -- the bound that keeps a degenerate position
/// from turning a pruned run into a single-stage one.
///
/// The widest step was 5.0 under v3's FL EV. It is derived from the ladder here
/// rather than written down twice, so the next re-measurement moves one literal.
#[test]
fn the_extension_stops_at_twice_prefilter_keep() {
    let ordered = candidate_scores_in_rank_order(baseline());
    let widest = (1..ordered.len())
        .map(|rank| ordered[rank - 1] - ordered[rank])
        .fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(
        widest, 12.600000000000001,
        "no gap in this fixture can stop a walk carrying this margin, so only \
         the cap can"
    );

    // The next representable f64 above the ladder's largest step: the smallest
    // margin that every gap in this fixture is strictly less than.
    let just_above = f64::from_bits(widest.to_bits() + 1);
    assert!(just_above > widest && just_above < widest + 1e-7);

    let mut config = staged_config();
    config.prefilter_margin = just_above;
    let result = evaluate_t0(&t0_second_observation(), &config).expect("evaluates");

    assert_kept_is_the_top(&result, 8);
    assert_eq!(
        result["stage_two_action_count"], 8,
        "twice prefilter_keep, not the 232 the run of ties would otherwise reach"
    );
    // The margin bought real work: the pruned run rescored four.
    assert_eq!(baseline()["stage_two_action_count"], 4);
}

/// Test 5. An audited position reports the comparison, and the staged answer is
/// not disturbed by having been measured.
///
/// The second half is the load-bearing one. An audit that perturbed the result
/// it audits would make a fleet's audited positions a different population from
/// its unaudited ones, and the whole point of the mechanism is that the audited
/// positions are a fair sample of the rest.
#[test]
fn an_audited_position_reports_the_full_comparison_without_disturbing_the_staged_one() {
    let mut config = staged_config();
    config.audit_full_every = 1;
    let result = evaluate_t0(&t0_second_observation(), &config).expect("evaluates");

    assert_eq!(result["audit_full_every"], 1);
    let audit = result["audit"].as_object().expect("an audit object");
    assert_eq!(audit["audited"], true);

    for field in [
        "audited",
        "audit_seed",
        "audit_stream",
        "evaluation_samples",
        "staged_pick",
        "full_pick",
        "agree",
        "full_best_survived_stage1",
        "regret_of_staged_pick_under_full",
        "full_best_score",
        "staged_pick_full_score",
        "staged_seconds",
        "audit_seconds",
    ] {
        assert!(audit.contains_key(field), "the audit must report {field}");
    }

    // `agree` is checkable against the two picks the audit itself names, so it
    // cannot be a constant that happens to be right.
    let staged = audit["staged_pick"].as_str().expect("staged pick");
    let full = audit["full_pick"].as_str().expect("full pick");
    assert_eq!(
        audit["agree"],
        Value::Bool(staged == full),
        "agree must be the comparison of the two picks it reports"
    );
    assert_eq!(
        staged,
        result["selected_action_key"].as_str().expect("selection"),
        "the staged pick is the run's own selection, not a re-derivation"
    );

    let regret = audit["regret_of_staged_pick_under_full"]
        .as_f64()
        .expect("regret");
    let full_best = audit["full_best_score"].as_f64().expect("full best");
    let staged_under_full = audit["staged_pick_full_score"].as_f64().expect("staged");
    assert_eq!(regret, full_best - staged_under_full);
    assert!(
        regret >= 0.0,
        "the full pass's own argmax cannot score below another action under it"
    );
    if audit["agree"] == Value::Bool(true) {
        assert_eq!(regret, 0.0);
        assert_eq!(audit["full_best_survived_stage1"], true);
    }
    assert!(audit["full_best_survived_stage1"].is_boolean());

    // Its own stream, and a seed nobody configured.
    assert_eq!(
        audit["audit_stream"],
        "t0-pruning-safety:audit_full_evaluation"
    );
    assert_ne!(audit["audit_seed"], 7_003, "the evaluation seed reused");
    assert_ne!(audit["audit_seed"], 7_002);
    assert!(audit["staged_seconds"].as_f64().expect("staged seconds") > 0.0);
    assert!(audit["audit_seconds"].as_f64().expect("audit seconds") > 0.0);

    // And the claim the mechanism rests on: everything else is the unaudited
    // run, byte for byte -- the same rows, the same selection, and the same
    // child information set count, which a shared search context would have
    // moved.
    assert_eq!(
        without(&result, &["audit", "audit_full_every"]),
        *baseline(),
        "the audit perturbed the result it was auditing"
    );
}

/// Test 6. A position the cadence does not select pays nothing and says so by
/// saying nothing.
///
/// This fixture's fingerprint hash is odd, so a cadence of two does not select
/// it. The residue is asserted from the emitted fingerprint rather than assumed,
/// because "no audit key" is also what a mechanism that never fires looks like,
/// and the two have to be told apart.
#[test]
fn a_position_the_cadence_passes_over_carries_no_audit() {
    let fingerprint = baseline()["observation_fingerprint"]
        .as_str()
        .expect("fingerprint");
    let hash = u64::from_str_radix(&fingerprint[..16], 16).expect("hex");
    assert_eq!(hash % 2, 1, "this fixture is not on a cadence of two");

    let mut config = staged_config();
    config.audit_full_every = 2;
    let result = evaluate_t0(&t0_second_observation(), &config).expect("evaluates");

    assert!(
        object(&result).get("audit").is_none(),
        "a position the cadence passed over must carry no audit object"
    );
    // The cadence itself is still reported, so that an unaudited position under
    // a live audit is distinguishable from a run with the audit switched off.
    assert_eq!(result["audit_full_every"], 2);
    assert_eq!(
        without(&result, &["audit_full_every"]),
        *baseline(),
        "a position the cadence passed over must cost nothing and change nothing"
    );
}

/// Test 7. The cadence is arithmetic on the fingerprint, not a special case for
/// one.
///
/// A rule spelled "audit when the cadence is one" would satisfy tests 5 and 6
/// exactly. This fixture's hash is divisible by thirty-one, so a cadence of
/// thirty-one has to select it.
///
/// The divisor has moved with every FL EV re-measurement, because the constant
/// sits inside the scoring context and the scoring context is part of the
/// observation the fingerprint hashes: eleven under June's 10.227, forty-one
/// under v3's 9.109, and thirty-one under v4's 9.6. Unlike the previous two it
/// is no longer the only candidate below two hundred -- the v4 hash is also
/// divisible by three and by ninety-three -- so thirty-one is chosen rather
/// than forced, on the grounds that a cadence of three would be weak evidence
/// that the arithmetic is real when roughly a third of all positions would
/// satisfy it by chance.
#[test]
fn the_cadence_selects_by_the_fingerprint_at_intervals_other_than_one() {
    let fingerprint = baseline()["observation_fingerprint"]
        .as_str()
        .expect("fingerprint");
    let hash = u64::from_str_radix(&fingerprint[..16], 16).expect("hex");
    assert_eq!(hash % 31, 0, "this fixture is on a cadence of thirty-one");

    let mut config = staged_config();
    config.audit_full_every = 31;
    let result = evaluate_t0(&t0_second_observation(), &config).expect("evaluates");

    assert_eq!(result["audit_full_every"], 31);
    assert_eq!(result["audit"]["audited"], true);
    assert_eq!(
        without(&result, &["audit", "audit_full_every"]),
        *baseline()
    );
}

/// Test 8. A margin with no boundary to widen is refused rather than ignored.
///
/// The single-stage path scores everything at the full particle count, so a
/// margin there is not wrong, it is vacuous -- and a plan carrying one would
/// read as having taken a pruning-safety measure it did not take.
#[test]
fn a_margin_without_the_two_stage_schedule_is_refused() {
    let mut config = staged_config();
    config.prefilter_samples = 0;
    config.prefilter_keep = 0;
    config.prefilter_margin = 2.4;

    let error = evaluate_t0(&t0_second_observation(), &config).expect_err("must refuse");
    assert!(error.contains("prefilter_margin"), "got {error}");
    assert!(error.contains("single-stage"), "got {error}");
}

/// Test 9. A margin that is not a score difference is refused.
///
/// NaN is the one that matters. It compares false against every gap, so a NaN
/// margin would silently disable the widening it was set to enable, and the run
/// would report a margin it never applied.
#[test]
fn a_negative_or_non_finite_margin_is_refused() {
    for margin in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut config = staged_config();
        config.prefilter_margin = margin;
        let error = match evaluate_t0(&t0_second_observation(), &config) {
            Ok(_) => panic!("a margin of {margin} was accepted"),
            Err(error) => error,
        };
        assert!(
            error.contains("finite non-negative"),
            "margin {margin}: got {error}"
        );
    }
}
