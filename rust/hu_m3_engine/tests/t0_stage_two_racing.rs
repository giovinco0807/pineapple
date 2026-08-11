//! Racing inside stage two, and the promise that it costs nothing until it is
//! asked for.
//!
//! The prefilter's saving comes from asking a cheap question of all 232 root
//! actions and an expensive one of forty-eight. Stage two then asks the same
//! expensive question of all forty-eight, and that is the remaining waste: most
//! survivors are separated from the leader long before the last particle, and
//! every particle spent after that buys nothing. `race_schedule` spends stage
//! two in instalments and drops a candidate when the leader has beaten it
//! decisively; `race_lcb_z` is how decisive "decisively" has to be.
//!
//! Two properties carry the whole design and most of this file tests them.
//!
//! COMMON RANDOM NUMBERS. Particles accumulate rather than being redrawn, so at
//! any checkpoint every live candidate has consumed the same particles in the
//! same order. Comparisons are therefore paired -- whatever a particle did to
//! both candidates cancels -- which is what lets thirty-two particles decide
//! anything at all about a rollout whose own spread is enormous. The observable
//! signature is that instalments change no score and that everything eliminated
//! at a checkpoint consumed exactly that checkpoint's particles.
//!
//! NON-INTERFERENCE. Unset, the schedule must produce the bytes the engine
//! produced before it existed -- `t0_pruning_safety` pins that against a
//! fixture written by the pre-mechanism engine, and this file adds the sharper
//! version: set but never eliminating, the result must be the uniform result
//! plus the race's own provenance and nothing else. The standing audit is held
//! to the stricter form again, being the reference the schedule is measured
//! against: it is a full independent evaluation of every action, and racing may
//! not reach inside it.
//!
//! The weights are stand-ins, as in `t0_pruning_safety`: `t3first_model_v1.bin`
//! is a 168-wide model trained for another street, used here for every slot of
//! that width. Nothing below reads a number for its poker meaning -- only which
//! action outranked which, whether two runs agreed, and how many particles were
//! spent -- so a stand-in with a real digest exercises the schedule exactly as
//! trained weights would, and far more cheaply.

use ofc_hu_m3_engine::cards::ALL_CARDS;
use ofc_hu_m3_engine::infoset::{ActOrder, ActorObservation, ScoringContext, Seat, Street};
use ofc_hu_m3_engine::search::{evaluate_t0, T3Config};
use ofc_hu_m3_engine::state::Board;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
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

/// The evaluation batch every test here shares, and the schedule that divides
/// it.
///
/// Eight particles rather than the fleet's 256 because the cost of this file is
/// the 232-action stage-one pass, not stage two, and because the properties
/// being tested are structural: a schedule that accumulates correctly over
/// eight accumulates correctly over 256. Eight is still enough for the
/// elimination rule to have a variance to work with, which one particle -- what
/// `t0_pruning_safety` runs on -- would not be.
const EVALUATION_SAMPLES: usize = 8;
const SCHEDULE: [usize; 3] = [2, 4, 8];

/// Large enough that no paired difference in this fixture can clear it, so a
/// run at this z races without ever eliminating. That is the arm the
/// non-interference tests compare against: it exercises every line of the
/// instalment machinery and must still land exactly where the uniform run did.
const NEVER_ELIMINATES: f64 = 1.0e9;

/// The schedule every test below shares: seven pinned evaluators, one particle
/// for stage one, eight for stage two, keep eight of 232.
///
/// The run id and the three seeds are constants rather than parameters because
/// the particles derive from them. Two runs that differ only in a schedule or a
/// z have to draw the same particles, or "nothing else changed" would be
/// untestable.
fn staged_config() -> T3Config {
    let (wide_path, wide_sha) = wide();
    T3Config {
        candidate_samples: 1,
        evaluation_samples: EVALUATION_SAMPLES,
        downstream_t3_samples: 1,
        downstream_t4_samples: 0,
        seed: 7_001,
        candidate_seed: 7_002,
        evaluation_seed: 7_003,
        run_id: "t0-stage-two-racing".to_owned(),
        use_t4_action_cache: true,
        prefilter_samples: 1,
        prefilter_keep: 8,
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

fn raced(schedule: &[usize], z: f64) -> T3Config {
    T3Config {
        race_schedule: schedule.to_vec(),
        race_lcb_z: z,
        ..staged_config()
    }
}

/// The uniform stage two, computed once for the whole binary. Four of the tests
/// below compare against it, and each paying for its own 232-action stage-one
/// pass would multiply this file's cost for no additional claim.
fn uniform() -> &'static Value {
    static UNIFORM: OnceLock<Value> = OnceLock::new();
    UNIFORM.get_or_init(|| {
        let config = staged_config();
        assert!(config.race_schedule.is_empty(), "the reference does not race");
        assert_eq!(config.race_lcb_z, 0.0, "and carries no threshold");
        evaluate_t0(&t0_second_observation(), &config).expect("evaluates")
    })
}

/// The same root raced at a z nothing in this fixture can clear.
fn raced_without_eliminating() -> &'static Value {
    static RESULT: OnceLock<Value> = OnceLock::new();
    RESULT.get_or_init(|| {
        evaluate_t0(
            &t0_second_observation(),
            &raced(&SCHEDULE, NEVER_ELIMINATES),
        )
        .expect("evaluates")
    })
}

/// The z the eliminating tests use. Two standard errors of the paired
/// difference: the leader has to be ahead by twice the noise in the very
/// quantity being measured before anything is dropped.
const WORKING_Z: f64 = 2.0;

fn raced_for_real() -> &'static Value {
    static RESULT: OnceLock<Value> = OnceLock::new();
    RESULT.get_or_init(|| {
        evaluate_t0(&t0_second_observation(), &raced(&SCHEDULE, WORKING_Z))
            .expect("evaluates")
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

/// Every top-level key the race adds. Removing exactly these has to turn a
/// raced result back into the uniform one, which is the sense in which the
/// provenance is additive.
const RACE_KEYS: &[&str] = &[
    "race_schedule",
    "race_lcb_z",
    "race_entrants",
    "race_checkpoint_survivors",
    "race_particle_evaluations",
    "race_uniform_particle_evaluations",
];

/// Every per-row key the race adds.
const RACE_ROW_KEYS: &[&str] = &[
    "race_particles",
    "race_full_samples",
    "race_eliminated_at_checkpoint",
];

fn rows(result: &Value) -> &Vec<Value> {
    result["actions"].as_array().expect("action rows")
}

/// The stage-two rows, keyed by action, with the race's per-row fields stripped
/// so a raced result and a uniform one can be compared directly.
fn stage_two_rows_without_race_fields(result: &Value) -> BTreeMap<String, Value> {
    rows(result)
        .iter()
        .filter(|row| row["stage"] == 2)
        .map(|row| {
            (
                row["action_key"].as_str().expect("action_key").to_owned(),
                without(row, RACE_ROW_KEYS),
            )
        })
        .collect()
}

/// Stage-two scores keyed by action, as raw bits.
///
/// Compared as bits rather than as floats on purpose. The claim racing makes
/// about a candidate that reached the last checkpoint is not that its score is
/// close to the uniform one but that it IS the uniform one: the same rollout
/// outcomes summed in the same order. A tolerance would hide the difference
/// between that and an accumulation that merely converged.
fn stage_two_score_bits(result: &Value) -> BTreeMap<String, u64> {
    rows(result)
        .iter()
        .filter(|row| row["stage"] == 2)
        .map(|row| {
            (
                row["action_key"].as_str().expect("action_key").to_owned(),
                row["score"].as_f64().expect("score").to_bits(),
            )
        })
        .collect()
}

/// Particles consumed, keyed by action, over the stage-two entrants.
fn particles_by_action(result: &Value) -> BTreeMap<String, usize> {
    rows(result)
        .iter()
        .filter(|row| row["stage"] == 2)
        .map(|row| {
            (
                row["action_key"].as_str().expect("action_key").to_owned(),
                row["race_particles"].as_u64().expect("race_particles") as usize,
            )
        })
        .collect()
}

/// Test 1. The unset schedule changes nothing.
///
/// `t0_pruning_safety` already pins the stronger form of this against a fixture
/// written before any of these mechanisms existed, and that test is the reason
/// every label already generated stays reproducible. This one states the same
/// promise locally and in the form this file can check without a fixture: a
/// result from a config that did not ask for a race carries no trace of one.
#[test]
fn an_unraced_run_carries_no_trace_of_the_race() {
    let unraced = object(uniform());
    let leaked = unraced
        .keys()
        .filter(|key| key.starts_with("race"))
        .collect::<Vec<_>>();
    assert!(
        leaked.is_empty(),
        "a run that did not race emitted {leaked:?}"
    );
    for row in rows(uniform()) {
        let row = row.as_object().expect("a row is an object");
        let leaked = row
            .keys()
            .filter(|key| key.starts_with("race"))
            .collect::<Vec<_>>();
        assert!(
            leaked.is_empty(),
            "a row from a run that did not race emitted {leaked:?}"
        );
    }
}

/// Test 2. Racing without eliminating is the uniform run.
///
/// The sharpest available statement of non-interference, and the one that
/// separates "the schedule is off" from "the schedule ran and changed
/// nothing". This arm walks every instalment, accumulates every candidate
/// across three checkpoints, and evaluates the elimination rule at two of them
/// -- it simply never clears the threshold. Strip the provenance the race adds
/// and what is left has to be the uniform result, key for key.
///
/// If instalments perturbed anything -- a re-drawn particle, a re-summed mean,
/// a candidate scored against a different prefix -- this is where it would
/// show, because everything else about the two runs is identical by
/// construction.
#[test]
fn racing_that_eliminates_nothing_reproduces_the_uniform_run() {
    let raced = raced_without_eliminating();
    assert_eq!(
        raced["race_checkpoint_survivors"],
        serde_json::json!([8, 8, 8]),
        "at this z nothing may be eliminated at any checkpoint"
    );
    assert_eq!(
        raced["race_particle_evaluations"], raced["race_uniform_particle_evaluations"],
        "a race that eliminates nothing spends exactly the uniform budget"
    );

    let mut stripped = object(&without(raced, RACE_KEYS)).clone();
    let stripped_rows = rows(raced)
        .iter()
        .map(|row| without(row, RACE_ROW_KEYS))
        .collect::<Vec<_>>();
    stripped.insert("actions".to_owned(), Value::Array(stripped_rows));

    assert_eq!(
        serde_json::to_string(&Value::Object(stripped)).expect("serializes"),
        serde_json::to_string(uniform()).expect("serializes"),
        "a race that eliminated nothing must be the uniform run plus its own \
         provenance and nothing else"
    );
}

/// Test 3. Common random numbers: the instalments are a prefix, not a redraw.
///
/// Two schedules over the same eight particles -- one instalment or three --
/// must give every entrant the same score, bit for bit. That is only true if a
/// candidate carried across a checkpoint keeps the particles it has and draws
/// the NEXT ones, which is the accumulation property, and if the accumulated
/// mean is summed in the batch's own order, which is what makes it equal to the
/// uniform sum rather than merely close to it.
///
/// It is also the pairing property in the only form observable from outside: if
/// candidates at a checkpoint had consumed different particles, splitting the
/// batch differently would move their scores.
#[test]
fn splitting_the_batch_into_instalments_does_not_move_a_score() {
    let one_instalment = evaluate_t0(
        &t0_second_observation(),
        &raced(&[EVALUATION_SAMPLES], NEVER_ELIMINATES),
    )
    .expect("evaluates");

    assert_eq!(
        stage_two_score_bits(&one_instalment),
        stage_two_score_bits(raced_without_eliminating()),
        "one instalment and three must produce identical scores"
    );
    assert_eq!(
        stage_two_score_bits(&one_instalment),
        stage_two_score_bits(uniform()),
        "and both must equal the uniform stage two"
    );
}

/// Test 4. The race actually eliminates something.
///
/// Every other test here would pass on an engine whose elimination rule never
/// fired. This one fails on it, and is why `WORKING_Z` is a real threshold
/// rather than a large one.
#[test]
fn the_working_threshold_eliminates_candidates() {
    let raced = raced_for_real();
    let survivors = raced["race_checkpoint_survivors"]
        .as_array()
        .expect("survivor counts")
        .iter()
        .map(|count| count.as_u64().expect("a count") as usize)
        .collect::<Vec<_>>();
    assert_eq!(survivors.len(), SCHEDULE.len(), "one count per checkpoint");
    let entrants = raced["race_entrants"].as_u64().expect("entrants") as usize;
    assert!(
        survivors[survivors.len() - 1] < entrants,
        "the race was supposed to eliminate someone: {entrants} entrants, \
         survivors {survivors:?}"
    );
    assert!(
        raced["race_particle_evaluations"].as_u64().expect("spent")
            < raced["race_uniform_particle_evaluations"]
                .as_u64()
                .expect("uniform"),
        "eliminating someone has to cost fewer particles than not"
    );
}

/// Test 5. Elimination monotonicity.
///
/// A candidate eliminated at checkpoint k consumed exactly the particles that
/// checkpoint names -- not more, which would mean the race paid for evidence it
/// had already decided against, and not fewer, which would mean it was judged
/// on a different prefix than the leader it lost to. A survivor consumed the
/// whole batch.
///
/// Together with the counts being non-increasing, this is the statement that
/// the schedule is a schedule: the field only shrinks, and every candidate's
/// spend is one of the checkpoints.
#[test]
fn an_eliminated_candidate_stopped_at_exactly_its_checkpoint() {
    let raced = raced_for_real();
    let mut eliminated_seen = 0usize;
    for row in rows(raced).iter().filter(|row| row["stage"] == 2) {
        let particles = row["race_particles"].as_u64().expect("race_particles") as usize;
        let full = row["race_full_samples"].as_bool().expect("race_full_samples");
        match row["race_eliminated_at_checkpoint"].as_u64() {
            Some(checkpoint) => {
                let checkpoint = checkpoint as usize;
                assert!(
                    checkpoint < SCHEDULE.len() - 1,
                    "nothing can be eliminated at the final checkpoint"
                );
                assert_eq!(
                    particles, SCHEDULE[checkpoint],
                    "a candidate eliminated at checkpoint {checkpoint} must have \
                     consumed exactly {} particles",
                    SCHEDULE[checkpoint]
                );
                assert!(!full, "an eliminated candidate did not reach full samples");
                eliminated_seen += 1;
            }
            None => {
                assert_eq!(
                    particles, EVALUATION_SAMPLES,
                    "a survivor must have consumed the whole batch"
                );
                assert!(full, "a survivor reached full samples");
            }
        }
    }
    assert!(eliminated_seen > 0, "the fixture must eliminate someone");

    let survivors = raced["race_checkpoint_survivors"]
        .as_array()
        .expect("survivor counts")
        .iter()
        .map(|count| count.as_u64().expect("a count"))
        .collect::<Vec<_>>();
    assert!(
        survivors.windows(2).all(|pair| pair[1] <= pair[0]),
        "the field may only shrink: {survivors:?}"
    );
}

/// Test 6. Everything still alive at a checkpoint spent the same particles.
///
/// The pairing property stated directly over the emitted rows: group the
/// entrants by where they stopped, and every group is a single particle count.
/// A race that let two candidates reach the same checkpoint on different
/// budgets would be comparing an average over one prefix against an average
/// over another, and its eliminations would carry the difference between the
/// two prefixes as though it were a difference between the candidates.
#[test]
fn candidates_that_stopped_together_consumed_the_same_particles() {
    let raced = raced_for_real();
    let mut by_checkpoint: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for row in rows(raced).iter().filter(|row| row["stage"] == 2) {
        let checkpoint = row["race_eliminated_at_checkpoint"]
            .as_i64()
            .unwrap_or(SCHEDULE.len() as i64);
        by_checkpoint
            .entry(checkpoint)
            .or_default()
            .push(row["race_particles"].as_u64().expect("race_particles") as usize);
    }
    for (checkpoint, spends) in &by_checkpoint {
        let first = spends[0];
        assert!(
            spends.iter().all(|spend| *spend == first),
            "candidates that stopped at checkpoint {checkpoint} spent {spends:?}, \
             which are not the same particles"
        );
    }
}

/// Test 7. The selected action was measured over the whole batch.
///
/// The one guarantee a label consumer needs from racing: whatever else the
/// schedule dropped early, the action it chose was not chosen on a partial
/// measurement. Its score has to be the score the uniform run gave that same
/// action, bit for bit.
#[test]
fn the_selected_action_reached_full_samples() {
    let raced = raced_for_real();
    let selected = raced["selected_action_key"].as_str().expect("selected");
    let row = rows(raced)
        .iter()
        .find(|row| row["action_key"] == *selected)
        .expect("the selected action has a row");
    assert_eq!(row["stage"], 2, "the selection came from stage two");
    assert!(
        row["race_full_samples"].as_bool().expect("race_full_samples"),
        "the selection must have reached full samples"
    );
    assert_eq!(
        row["race_particles"].as_u64().expect("race_particles") as usize,
        EVALUATION_SAMPLES
    );
    assert_eq!(
        stage_two_score_bits(raced)[selected],
        stage_two_score_bits(uniform())[selected],
        "a finalist's score must be the uniform score bit for bit"
    );
    assert_eq!(
        raced["evaluation_sample_best_score"], row["score"],
        "the reported best is the selected finalist's score"
    );
}

/// Test 8. A finalist's score is the uniform score; an eliminated one's is not
/// claimed to be.
///
/// Every candidate that reached the last checkpoint has to match the uniform
/// run exactly, which is what makes the race's answer comparable to a uniform
/// answer at all. An eliminated candidate carries the partial mean it was
/// dropped on -- a real stage-two measurement, just a shorter one -- and its
/// row says how short, which is the whole reason the per-row particle count is
/// emitted.
#[test]
fn finalists_carry_the_uniform_score_and_the_eliminated_say_they_do_not() {
    let raced = raced_for_real();
    let raced_bits = stage_two_score_bits(raced);
    let uniform_bits = stage_two_score_bits(uniform());
    let spends = particles_by_action(raced);
    let mut finalists = 0usize;
    for row in rows(raced).iter().filter(|row| row["stage"] == 2) {
        let key = row["action_key"].as_str().expect("action_key");
        if row["race_full_samples"].as_bool().expect("race_full_samples") {
            assert_eq!(
                raced_bits[key], uniform_bits[key],
                "finalist {key} must carry the uniform score bit for bit"
            );
            finalists += 1;
        } else {
            assert!(
                spends[key] < EVALUATION_SAMPLES,
                "an eliminated candidate's row must show a partial spend"
            );
        }
    }
    assert!(finalists > 0, "somebody has to reach the end");
    // Both runs promoted the same actions into stage two: racing composes with
    // the prefilter by consuming its output, not by changing it.
    assert_eq!(
        stage_two_rows_without_race_fields(raced).keys().collect::<Vec<_>>(),
        stage_two_rows_without_race_fields(uniform()).keys().collect::<Vec<_>>(),
        "the race must not change which actions stage two saw"
    );
    assert_eq!(
        raced["stage_two_action_count"], uniform()["stage_two_action_count"],
        "nor how many"
    );
}

/// Test 9. Racing composes with the adaptive beam.
///
/// The beam widens the keep boundary wherever stage one could not separate the
/// actions across it, and the widened set is what stage two receives. Racing
/// has to see that widened set -- entrants are whatever the prefilter promoted,
/// however many that is -- rather than the fixed keep it would have promoted
/// alone. Otherwise the two mechanisms would fight: the beam would admit an
/// action precisely because it could not be ruled out, and the race would never
/// hear about it.
#[test]
fn the_race_runs_over_the_widened_beam() {
    let mut config = raced(&SCHEDULE, WORKING_Z);
    // Wide enough to pull in the whole run of ties below the boundary, and
    // capped by the engine at twice the keep.
    config.prefilter_margin = 100.0;
    let widened = evaluate_t0(&t0_second_observation(), &config).expect("evaluates");

    let entrants = widened["race_entrants"].as_u64().expect("entrants") as usize;
    let promoted = widened["stage_two_action_count"]
        .as_u64()
        .expect("stage_two_action_count") as usize;
    assert_eq!(
        entrants, promoted,
        "the race's entrants are exactly what the prefilter promoted"
    );
    assert!(
        entrants > config.prefilter_keep,
        "this margin was supposed to widen the beam: {entrants} entrants for a \
         keep of {}",
        config.prefilter_keep
    );
    assert_eq!(
        widened["race_uniform_particle_evaluations"]
            .as_u64()
            .expect("uniform budget") as usize,
        entrants * EVALUATION_SAMPLES,
        "the counterfactual budget is over the widened set too"
    );
    assert_eq!(
        rows(&widened)
            .iter()
            .filter(|row| row["stage"] == 2)
            .count(),
        entrants,
        "every promoted action raced"
    );
}

/// Test 10. The audit is a full evaluation and racing does not reach inside it.
///
/// The audit exists to charge the schedule for what it pruned, which only works
/// while the audit itself is uncompromised: it scores every legal action over
/// its own independent particle batch, drops nothing, and races nothing. If
/// racing leaked into it, the audit would be measuring the schedule against a
/// schedule.
///
/// Checked by running the same audited position with and without a race. The
/// audit's own numbers -- which action the full evaluation preferred and what
/// it scored -- are properties of the audit batch alone, so they must be
/// identical across the two runs even though the staged answers they are
/// charging need not be.
#[test]
fn the_audit_is_a_full_evaluation_that_the_race_does_not_touch() {
    let mut unraced = staged_config();
    unraced.audit_full_every = 1;
    let unraced = evaluate_t0(&t0_second_observation(), &unraced).expect("evaluates");

    let mut with_race = raced(&SCHEDULE, WORKING_Z);
    with_race.audit_full_every = 1;
    let with_race = evaluate_t0(&t0_second_observation(), &with_race).expect("evaluates");

    let left = &unraced["audit"];
    let right = &with_race["audit"];
    assert_eq!(left["audited"], serde_json::json!(true), "the fixture is audited");
    assert_eq!(right["audited"], serde_json::json!(true));
    for key in ["audit_seed", "audit_stream", "evaluation_samples", "full_pick",
                "full_best_score"] {
        assert_eq!(
            left[key], right[key],
            "the audit's {key} is a property of the audit batch and must not \
             move when the staged answer races"
        );
    }
    // The audit reports the race's own failure mode alongside the prefilter's,
    // and only when there is a race to report it for.
    assert!(
        left.get("full_best_survived_race").is_none(),
        "an unraced run has no race for the best action to have survived"
    );
    assert!(
        right.get("full_best_survived_race").is_some(),
        "a raced run must say whether the race kept the full evaluation's pick"
    );
    // The audit's verdict on the staged pick is still computed against the
    // audit's own scores, so it is charged at full resolution whichever arm
    // produced the pick.
    assert!(
        right["regret_of_staged_pick_under_full"]
            .as_f64()
            .expect("regret")
            >= 0.0,
        "the full evaluation cannot prefer the staged pick to its own best"
    );
}

/// Test 11. A raced run is deterministic, and pinned.
///
/// Two runs of the same config agree, which rules out anything order-dependent
/// having crept into the elimination rule -- a hash iteration, a tie broken by
/// generation order. The fixture pins the answer across rebuilds, on the same
/// terms as the two-stage pin one file over: same source, rebuilt, same bytes.
///
/// To recapture, point `REGEN_T0_RACING_PIN` at a path: the test writes what it
/// produced there and then still asserts, so a regeneration cannot pass a
/// broken result off as a pin. Write it somewhere else and diff before
/// promoting it.
#[test]
fn a_raced_run_is_deterministic_and_pinned() {
    let again = evaluate_t0(&t0_second_observation(), &raced(&SCHEDULE, WORKING_Z))
        .expect("evaluates");
    let produced = serde_json::to_string(&again).expect("serializes");
    assert_eq!(
        produced,
        serde_json::to_string(raced_for_real()).expect("serializes"),
        "two runs of the same raced config must agree"
    );

    if let Ok(path) = std::env::var("REGEN_T0_RACING_PIN") {
        std::fs::write(&path, &produced).expect("writes the regenerated pin");
    }
    let pinned = std::fs::read_to_string(fixture("t0_racing_result_pin_v2.json"))
        .expect("the pinned racing result fixture");
    assert_eq!(
        produced, pinned,
        "the raced result must reproduce its pin byte for byte"
    );
}

/// Test 12. A z with no race to threshold is refused.
///
/// The same reasoning as a margin with no boundary to widen: it reads as a
/// tuned elimination rule and does nothing.
#[test]
fn a_threshold_without_a_schedule_is_refused() {
    let mut config = staged_config();
    config.race_lcb_z = 2.0;
    let error = evaluate_t0(&t0_second_observation(), &config).expect_err("refused");
    assert!(
        error.contains("race_lcb_z") && error.contains("race_schedule"),
        "the refusal must name both fields: {error}"
    );
}

/// Test 13. A schedule that does not end at the evaluation batch is refused.
///
/// It would be a cheaper evaluation wearing the full one's name: the selected
/// action would carry a label claiming `evaluation_samples` particles behind a
/// score measured on fewer.
#[test]
fn a_schedule_that_stops_short_of_the_batch_is_refused() {
    let error = evaluate_t0(&t0_second_observation(), &raced(&[2, 4], WORKING_Z))
        .expect_err("refused");
    assert!(
        error.contains("evaluation_samples") && error.contains("8"),
        "the refusal must name the batch it fell short of: {error}"
    );
}

/// Test 14. A schedule that does not advance is refused.
///
/// A repeated checkpoint spends no particles and would evaluate the same
/// elimination twice on identical evidence; a decreasing one has no reading at
/// all.
#[test]
fn a_schedule_that_does_not_strictly_increase_is_refused() {
    for schedule in [vec![2, 2, 8], vec![4, 2, 8]] {
        let error = evaluate_t0(&t0_second_observation(), &raced(&schedule, WORKING_Z))
            .expect_err("refused");
        assert!(
            error.contains("strictly increasing"),
            "the refusal must say why: {error}"
        );
    }
    let error = evaluate_t0(&t0_second_observation(), &raced(&[0, 8], WORKING_Z))
        .expect_err("refused");
    assert!(
        error.contains("positive"),
        "a zero checkpoint must be refused as such: {error}"
    );
}

/// Test 15. A negative or non-finite z is refused.
///
/// NaN in particular would compare false against every deficit and quietly
/// eliminate nothing, which is the failure that looks most like success: a
/// gate measuring a race that never raced.
#[test]
fn a_negative_or_non_finite_threshold_is_refused() {
    for z in [-1.0, f64::NAN, f64::INFINITY] {
        let error = evaluate_t0(&t0_second_observation(), &raced(&SCHEDULE, z))
            .expect_err("refused");
        assert!(
            error.contains("race_lcb_z"),
            "the refusal must name the field: {error}"
        );
    }
}
