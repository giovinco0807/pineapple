//! Information-set-safe T4/T3 search ported from the M2 Python reference.

use crate::action::{
    generate_initial_actions, generate_turn_actions, generate_turn_actions_trusted, Action,
};
use crate::action_key::{
    action_key, canonical_descending_indices, legal_action_set_digest,
    ordered_action_mapping_digest,
};
use crate::belief::{sample_hidden_card_particles, HiddenCardParticle};
use crate::cards::{Card, ALL_CARDS};
use crate::compact_scoring::{
    heads_up_terminal_score_compact, score_board_compact_trusted, CompactBoardScore,
};
use crate::counter_rng::{sha256_hex_json, CounterActor, CounterRngKey, COUNTER_RNG_SCHEMA};
use crate::explicit_support::{evaluate_t3_explicit_support, ExplicitSupportConfig};
use crate::fast_features::{fast_encode, fast_encode_hidden_opponent, fast_outlook, FastOutlookCache};
use crate::infoset::{ActOrder, ActorObservation, Seat, Street};
use crate::scoring::{heads_up_terminal_score, score_board_trusted};
use crate::t4_model::Model;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

pub const ENGINE_VERSION: &str = "ofc_hu_m3_engine/0.1.0";
pub const REQUEST_SCHEMA: &str = "hu_m3_engine_request_v1";
pub const BATCH_REQUEST_SCHEMA: &str = "hu_m3_engine_batch_request_v1";
pub const RESULT_SCHEMA: &str = "hu_m3_engine_result_v1";
pub const BATCH_RESULT_SCHEMA: &str = "hu_m3_engine_batch_result_v1";
pub const T3_ABR_COMPONENT_RESULT_SCHEMA: &str = "hu_m3_t3_abr_component_result_v1";

const RNG_DOMAIN: u64 = 1_u64 << 63;
const ATTEMPT_BITS: u32 = 32;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineRequest {
    pub schema: String,
    pub kind: String,
    pub observation: ActorObservation,
    pub observation_fingerprint: String,
    #[serde(default)]
    pub config: Value,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct T4Config {
    #[serde(default)]
    pub candidate_samples: usize,
    #[serde(default)]
    pub evaluation_samples: usize,
    #[serde(default = "default_seed")]
    pub seed: i64,
    #[serde(default = "default_seed")]
    pub candidate_seed: i64,
    #[serde(default = "default_seed")]
    pub evaluation_seed: i64,
    #[serde(default = "default_t4_run_id")]
    pub run_id: String,
}

impl Default for T4Config {
    fn default() -> Self {
        Self {
            candidate_samples: 0,
            evaluation_samples: 0,
            seed: default_seed(),
            candidate_seed: default_seed(),
            evaluation_seed: default_seed(),
            run_id: default_t4_run_id(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct T3Config {
    #[serde(default = "default_candidate_samples")]
    pub candidate_samples: usize,
    #[serde(default = "default_evaluation_samples")]
    pub evaluation_samples: usize,
    #[serde(default = "default_downstream_t3_samples")]
    pub downstream_t3_samples: usize,
    #[serde(default = "default_downstream_t4_samples")]
    pub downstream_t4_samples: usize,
    #[serde(default = "default_seed")]
    pub seed: i64,
    #[serde(default = "default_seed")]
    pub candidate_seed: i64,
    #[serde(default = "default_seed")]
    pub evaluation_seed: i64,
    #[serde(default = "default_t3_run_id")]
    pub run_id: String,
    #[serde(default = "default_true")]
    pub use_t4_action_cache: bool,
    /// Opt in to the learned first-seat T4 evaluator.
    ///
    /// Absent by default, and absent means the search is exactly what it was:
    /// every T4 child is solved by enumeration and the emitted result is
    /// byte-identical to a run from before this existed. Supplying a path
    /// replaces only the first-seat leaf, which is the one that costs anything;
    /// second seat has no uncertainty left to resolve and stays closed-form.
    #[serde(default)]
    pub learned_t4_model_path: Option<String>,
    /// Required whenever a path is given. A model paired with weights that have
    /// silently changed produces plausible numbers rather than an error, so the
    /// identity of the image is part of the configuration rather than a
    /// convention.
    #[serde(default)]
    pub learned_t4_model_sha256: Option<String>,
    /// Opt in to the learned T3 second-seat evaluator.
    ///
    /// This is the response the first seat's rollouts play against, and solving
    /// it by nested search is the reason the first seat costs roughly sixty
    /// times what the second does. Absent by default; absent means the nested
    /// response is sampled exactly as before.
    #[serde(default)]
    pub learned_t3_second_model_path: Option<String>,
    /// Required whenever a path is given, for the same reason as at T4.
    #[serde(default)]
    pub learned_t3_second_model_sha256: Option<String>,
    /// Opt in to the learned T3 first-seat evaluator. Only the T2 evaluator
    /// consumes it: a T2 rollout must play the opponent's T3 first-seat reply,
    /// and the nested sampled search that would otherwise answer costs minutes
    /// per decision.
    #[serde(default)]
    pub learned_t3_first_model_path: Option<String>,
    #[serde(default)]
    pub learned_t3_first_model_sha256: Option<String>,
    /// Consumed by the `decide` request for T2 second-seat play.
    #[serde(default)]
    pub learned_t2_second_model_path: Option<String>,
    #[serde(default)]
    pub learned_t2_second_model_sha256: Option<String>,
    /// Consumed by the `decide` request for T2 first-seat play. Acting first
    /// the opponent has not answered T2 yet, so the geometry differs from the
    /// second seat's even though the encoding is the same.
    #[serde(default)]
    pub learned_t2_first_model_path: Option<String>,
    #[serde(default)]
    pub learned_t2_first_model_sha256: Option<String>,
    /// Consumed by the `decide` request for T1 second-seat play, and by the T1
    /// first-seat evaluator, whose every rollout opens with the opponent's T1
    /// second-seat reply. Acting second at T1 the opponent's board carries
    /// seven cards where the first seat sees five, so the seat has its own
    /// model rather than sharing the first seat's.
    #[serde(default)]
    pub learned_t1_second_model_path: Option<String>,
    #[serde(default)]
    pub learned_t1_second_model_sha256: Option<String>,
    /// Consumed by the `decide` request for T1 first-seat play, and required by
    /// the T0 evaluator on both seats: acting first at T0 the rollout plays the
    /// hero's own T1 first-seat turn, and acting second it plays the
    /// opponent's. Acting first at T1 both boards carry five cards where the
    /// second seat sees seven, so the seat has its own model.
    #[serde(default)]
    pub learned_t1_first_model_path: Option<String>,
    #[serde(default)]
    pub learned_t1_first_model_sha256: Option<String>,
    /// Consumed by the `decide` request for T0 second-seat play, and by the T0
    /// first-seat evaluator, whose every rollout opens with the opponent's T0
    /// second-seat reply. Acting second at T0 the opponent's five placed cards
    /// are the whole of the public information, which the first seat does not
    /// have, so the seat has its own model.
    #[serde(default)]
    pub learned_t0_second_model_path: Option<String>,
    #[serde(default)]
    pub learned_t0_second_model_sha256: Option<String>,
    /// Consumed by the `decide` request for T0 first-seat play, and by nothing
    /// else in the engine.
    ///
    /// The asymmetry is the street's, not an omission. Every other learned
    /// evaluator is named twice over -- once by the `decide` request that plays
    /// it and once by some rollout that has it still ahead -- because every
    /// other decision sits below at least one street's root. The opening street
    /// acting first sits below nothing: it IS the first decision of the hand,
    /// so no rollout in the engine ever has to answer it, and this model is a
    /// root policy rather than a continuation.
    ///
    /// The image it names is not interchangeable with the seven above. Acting
    /// first the opponent's board is empty -- thirteen open slots -- and the
    /// free-slot outlook enumerates four, six and eight, so the two blocks that
    /// read the opponent are zero and the model was fitted that way. See
    /// [`crate::fast_features::fast_encode_hidden_opponent`]; a full-precision
    /// four-block image supplied here would be read against a vector whose last
    /// 46 columns are always zero.
    #[serde(default)]
    pub learned_t0_first_model_path: Option<String>,
    /// Required whenever the path is given, for the reason every other pinned
    /// evaluator states it: weights that quietly changed still produce a legal
    /// action.
    #[serde(default)]
    pub learned_t0_first_model_sha256: Option<String>,
    /// Opt in to the coarse T2 second-seat reply inside a rollout.
    ///
    /// With the T1 pair and the T0 second-seat opening already answerable
    /// coarsely, T2 is the largest block of full-precision work a T0 first-seat
    /// rollout has left: it plays one reply on each T2 seat, twenty-seven
    /// candidate boards apiece, and does so once per rollout. Supplying this
    /// replaces the reply's encoder with [`crate::fast_features`], which asks
    /// the same question at a tenth of the row completions and an eighth of the
    /// joint draws. Absent by default; absent means the reply is the
    /// full-precision one it always was.
    ///
    /// Reachable from more shapes than the T0 second-seat pin is. Every rollout
    /// that begins at T2 first, T1 on either seat, or T0 on either seat still
    /// has both T2 replies ahead of it, so this pin bites on all of them --
    /// where `fast_t0_second_model_path` bites only on a T0 first-seat run.
    ///
    /// Only the rollouts consult it. The `decide` request builds its T2 answer
    /// from `learned_t2_second_model` whether this is set or not, because a
    /// decision that is played is worth its full cost and a reply that is
    /// sampled hundreds of times over is not.
    #[serde(default)]
    pub fast_t2_second_model_path: Option<String>,
    /// Required whenever the path is given, for the same reason as every other
    /// pinned evaluator: coarse weights that drifted still produce a legal
    /// action.
    #[serde(default)]
    pub fast_t2_second_model_sha256: Option<String>,
    /// Opt in to the coarse T2 first-seat reply inside a rollout. The mirror of
    /// `fast_t2_second_model_path` one seat over, and separate for the same
    /// reason the full-precision pair is separate: acting first the opponent
    /// has answered T1 and no more, so its board carries seven cards where the
    /// second seat sees nine.
    #[serde(default)]
    pub fast_t2_first_model_path: Option<String>,
    #[serde(default)]
    pub fast_t2_first_model_sha256: Option<String>,
    /// Opt in to the coarse T1 second-seat reply inside a rollout.
    ///
    /// The T1 replies are where a T0 evaluation spends nearly all of its time,
    /// and a rollout reads only which action they picked. Supplying this
    /// replaces the reply's encoder with [`crate::fast_features`], which asks
    /// the same question at a tenth of the row completions and an eighth of the
    /// joint draws. Absent by default; absent means the reply is the
    /// full-precision one it always was.
    ///
    /// Only the rollouts consult it. The `decide` request builds its T1 answer
    /// from `learned_t1_second_model` whether this is set or not, because a
    /// decision that is played is worth its full cost and a reply that is
    /// sampled sixty-four times over is not.
    #[serde(default)]
    pub fast_t1_second_model_path: Option<String>,
    /// Required whenever the path is given, for the same reason as every other
    /// pinned evaluator: coarse weights that drifted still produce a legal
    /// action.
    #[serde(default)]
    pub fast_t1_second_model_sha256: Option<String>,
    /// Opt in to the coarse T1 first-seat reply inside a rollout. The mirror of
    /// `fast_t1_second_model_path` one seat over, and separate for the same
    /// reason the full-precision pair is separate: acting first both boards
    /// carry five cards where the second seat sees seven.
    #[serde(default)]
    pub fast_t1_first_model_path: Option<String>,
    #[serde(default)]
    pub fast_t1_first_model_sha256: Option<String>,
    /// Opt in to the coarse T0 second-seat reply inside a rollout.
    ///
    /// The T0 first-seat evaluator's rollouts each open with the opponent's
    /// opening, and that reply is the most expensive nested decision in the
    /// engine: 232 candidate boards against the full-precision outlook, where
    /// every later street faces twenty-seven. Supplying this replaces the
    /// reply's encoder with [`crate::fast_features`], which asks the same
    /// question at a tenth of the row completions and an eighth of the joint
    /// draws. Absent by default; absent means the reply is the full-precision
    /// one it always was.
    ///
    /// Only the rollouts consult it. The `decide` request builds its T0 answer
    /// from `learned_t0_second_model` whether this is set or not, and the T0
    /// second-seat evaluator never reaches this reply at all -- by the time it
    /// acts, the opening it would have answered is already on the board.
    #[serde(default)]
    pub fast_t0_second_model_path: Option<String>,
    /// Required whenever the path is given, for the same reason as every other
    /// pinned evaluator: coarse weights that drifted still produce a legal
    /// action.
    #[serde(default)]
    pub fast_t0_second_model_sha256: Option<String>,
    /// Particles used by the T0 prefilter stage. Zero disables the schedule.
    ///
    /// Every other street faces tens of root actions; T0 faces 232, and paying
    /// the full particle set on all of them is what makes the naive cost
    /// prohibitive. When this and `prefilter_keep` are both nonzero the root is
    /// scored twice: cheaply over everything, then fully over the survivors.
    /// Both zero is single-stage, which is the exact-comparison path the
    /// pruning validation measures the schedule against.
    #[serde(default)]
    pub prefilter_samples: usize,
    /// How many actions survive the prefilter into the full evaluation.
    #[serde(default)]
    pub prefilter_keep: usize,
    /// Widen the keep boundary when stage one cannot separate the actions on
    /// either side of it. Zero, the default, disables the widening entirely and
    /// leaves the schedule the fixed-rank one it has always been.
    ///
    /// A fixed rank is a claim that stage one can order the actions around it,
    /// and over `prefilter_samples` particles it often cannot: the boundary
    /// falls inside the coarse pass's own noise as readily as outside it. This
    /// is that noise expressed as a score difference. While the best excluded
    /// action is within it of the worst included one, the two are not
    /// distinguishable at stage one and the excluded one is pulled in as well.
    ///
    /// The number belongs to the measurement, not to the engine: it is the
    /// measured spread of stage-one scores for the schedule in use, and the
    /// engine only honours whatever the caller measured. The cost is bounded by
    /// a hard cap of twice `prefilter_keep`, so a position where every action
    /// ties cannot silently become a single-stage run.
    ///
    /// Refused without the two-stage schedule, on the same grounds as half a
    /// schedule: a margin with no boundary to widen would read as pruning
    /// safety while doing nothing.
    #[serde(default)]
    pub prefilter_margin: f64,
    /// Keep only this many candidates at T1, T2 and T3, chosen by the street's
    /// LEARNED evaluator, before any particle is spent. Zero, the default,
    /// disables it and leaves the emitted result byte-identical to one from
    /// before it existed.
    ///
    /// Distinct from `prefilter_samples`/`prefilter_keep`, which are T0's and
    /// cut with a coarse sampled pass. This cut is deterministic -- the model
    /// scores the fan once, cached, and no seed can move the boundary -- and it
    /// is informed by the joint-exact labels the model was fitted on rather
    /// than by one or two particles.
    ///
    /// Measured before it was wired in (50 hands at T2, 12 at T1 and T3): the
    /// rollout's own best action survives the model's top ten 91-96 % of the
    /// time, and a miss costs 0.02-2.2 points where the rollout's own spread at
    /// these world counts is 1-6. At top five, survival falls to 84-86 % for a
    /// saving that is 5.4x rather than 2.6x -- both spent on worlds, so the
    /// safer cut was taken. See docs/trainer_ranking_quality_20260808.md.
    ///
    /// Off for label generation unless a plan sets it: narrowing changes what a
    /// label means, from "the best action" to "the best of the ten the model
    /// liked", and that is a decision for whoever emits the plan.
    ///
    /// `serde(default)` so that every request written before this field existed
    /// still deserialises. Without it the field would be mandatory and every
    /// pinned plan in the fleet would stop parsing the moment the engine was
    /// rebuilt -- an opt-in knob that breaks everything that did not opt in.
    #[serde(default)]
    pub learned_prefilter_keep: usize,
    /// Evaluate one position in every `audit_full_every` single-stage as well,
    /// and report what the schedule cost on that position. Zero, the default,
    /// disables the audit and leaves the emitted result byte-identical to one
    /// from before it existed.
    ///
    /// The prefilter was validated on one distribution of roots. A fleet that
    /// relabels a cascade, or that plays live, meets distributions nobody has
    /// measured it on, and the failure it can have there is silent: a pruned
    /// action was the best one and no artefact of the run says so. A standing
    /// audit makes the run measure itself -- every Nth position is scored the
    /// expensive way too, and the two answers are compared in the result.
    ///
    /// Which positions are audited is a deterministic function of the
    /// observation, not a counter, so that a shard is an unbiased 1-in-N sample
    /// of whatever the fleet actually saw rather than the same ordinal position
    /// of every shard.
    #[serde(default)]
    pub audit_full_every: u64,
    /// Cumulative particle checkpoints for racing inside stage two. Empty, the
    /// default, spends the whole evaluation batch on every survivor, which is
    /// the uniform stage two the pruning validation measured.
    ///
    /// The prefilter's saving comes from asking a cheap question of everything
    /// and an expensive one of a few. Stage two then asks the SAME expensive
    /// question of all of those few, which is the remaining waste: most
    /// survivors are separated from the leader long before the last particle,
    /// and the particles spent confirming it buy nothing. A schedule of
    /// `[32, 64, 128, 256]` scores every survivor on the first thirty-two,
    /// drops the ones the leader has already beaten decisively, carries the
    /// rest to sixty-four, and so on.
    ///
    /// Cumulative, and the particles ACCUMULATE: a candidate carried from 32 to
    /// 64 keeps the thirty-two it has and draws thirty-two more, so nothing is
    /// ever simulated twice and a candidate reaching the last checkpoint has
    /// consumed exactly the batch a uniform run would have given it -- in the
    /// same order, so its score is the uniform score bit for bit. This is also
    /// what makes the comparisons paired: at any checkpoint every live
    /// candidate has consumed the same particles, so the difference between two
    /// of them carries no sampling difference at all. Common random numbers,
    /// obtained by construction rather than by arranging for it.
    ///
    /// The last checkpoint must equal `evaluation_samples`. A schedule stopping
    /// short would be a cheaper evaluation wearing the full one's name, and the
    /// selected action would not have been measured at the resolution the label
    /// claims.
    #[serde(default)]
    pub race_schedule: Vec<usize>,
    /// How decisive the leader has to be before a candidate is dropped, in
    /// units of the paired difference's own standard error.
    ///
    /// Elimination is one-sided and variance-aware: a candidate goes only when
    /// the leader's mean advantage over it, measured on the particles they have
    /// both consumed, exceeds `race_lcb_z` standard errors of that same paired
    /// difference. Larger is more conservative; ties and anything the particles
    /// have not separated are carried forward. The rule is deliberately not a
    /// fixed rank -- sequential halving by rank would drop half the field on
    /// every checkpoint whether or not the evidence supported dropping any of
    /// it.
    ///
    /// Refused without a schedule to apply it to, on the same grounds as a
    /// margin with no boundary to widen.
    #[serde(default)]
    pub race_lcb_z: f64,
}

impl Default for T3Config {
    fn default() -> Self {
        Self {
            candidate_samples: default_candidate_samples(),
            evaluation_samples: default_evaluation_samples(),
            downstream_t3_samples: default_downstream_t3_samples(),
            downstream_t4_samples: default_downstream_t4_samples(),
            seed: default_seed(),
            candidate_seed: default_seed(),
            evaluation_seed: default_seed(),
            run_id: default_t3_run_id(),
            use_t4_action_cache: true,
            learned_t4_model_path: None,
            learned_t4_model_sha256: None,
            learned_t3_second_model_path: None,
            learned_t3_second_model_sha256: None,
            learned_t3_first_model_path: None,
            learned_t3_first_model_sha256: None,
            learned_t2_second_model_path: None,
            learned_t2_second_model_sha256: None,
            learned_t2_first_model_path: None,
            learned_t2_first_model_sha256: None,
            learned_t1_second_model_path: None,
            learned_t1_second_model_sha256: None,
            learned_t1_first_model_path: None,
            learned_t1_first_model_sha256: None,
            learned_t0_second_model_path: None,
            learned_t0_second_model_sha256: None,
            learned_t0_first_model_path: None,
            learned_t0_first_model_sha256: None,
            fast_t2_second_model_path: None,
            fast_t2_second_model_sha256: None,
            fast_t2_first_model_path: None,
            fast_t2_first_model_sha256: None,
            fast_t1_second_model_path: None,
            fast_t1_second_model_sha256: None,
            fast_t1_first_model_path: None,
            fast_t1_first_model_sha256: None,
            fast_t0_second_model_path: None,
            fast_t0_second_model_sha256: None,
            prefilter_samples: 0,
            prefilter_keep: 0,
            prefilter_margin: 0.0,
            learned_prefilter_keep: 0,
            audit_full_every: 0,
            race_schedule: Vec::new(),
            race_lcb_z: 0.0,
        }
    }
}

fn default_seed() -> i64 {
    42
}
fn default_candidate_samples() -> usize {
    4
}
fn default_evaluation_samples() -> usize {
    8
}
fn default_downstream_t3_samples() -> usize {
    2
}
fn default_downstream_t4_samples() -> usize {
    8
}
fn default_t4_run_id() -> String {
    "hu-m2-t4".to_owned()
}
fn default_t3_run_id() -> String {
    "hu-m2-t3".to_owned()
}
fn default_true() -> bool {
    true
}

#[derive(Clone)]
struct FuturePlan {
    deals: Vec<[Card; 3]>,
    mode: &'static str,
    stream: String,
    rng_key_digests: Vec<String>,
}

#[derive(Clone)]
struct ScoredAction {
    score: f64,
    board_score: CompactBoardScore,
    future_count: usize,
}

#[derive(Clone)]
struct ScoredOpponentResponse {
    board_score: CompactBoardScore,
}

#[derive(Copy, Clone, Debug)]
struct TerminalOutcome {
    hu_score: f64,
    hero_busted: bool,
    opponent_busted: bool,
    hero_scoop: bool,
    opponent_scoop: bool,
    hero_royalty: i32,
    opponent_royalty: i32,
    hero_fl_value: f64,
    opponent_fl_value: f64,
}

#[derive(Clone, Debug)]
struct TerminalComponentMeans {
    hu_score: f64,
    hero_bust_rate: f64,
    opponent_bust_rate: f64,
    hero_scoop_rate: f64,
    opponent_scoop_rate: f64,
    hero_royalty_mean: f64,
    opponent_royalty_mean: f64,
    hero_fl_value_mean: f64,
    opponent_fl_value_mean: f64,
    future_count: usize,
}

/// Card masks as the cache identity of one child observation.
///
/// Within one search context every child observation shares the root's
/// immutable scoring context and carries the same hardcoded Fantasyland flags,
/// so the fields that vary -- boards, dealt cards, discards, seat, street,
/// order -- are the whole identity, and masks carry them order-invariantly
/// without JSON serialization, repeated validation, or cryptographic hashing
/// at every rollout node. Street is derivable from the mask populations, but
/// carrying it makes the injectivity argument structural rather than counted.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
struct ObservationKey {
    hero_top: u64,
    hero_middle: u64,
    hero_bottom: u64,
    opponent_top: u64,
    opponent_middle: u64,
    opponent_bottom: u64,
    dealt: u64,
    hero_discards: u64,
    seat: Seat,
    street: Street,
    to_act_order: ActOrder,
}

impl ObservationKey {
    fn new(observation: &ActorObservation) -> Self {
        let mask = |cards: &[Card]| cards.iter().fold(0_u64, |value, card| value | card.bit());
        Self {
            hero_top: mask(&observation.hero_board.top),
            hero_middle: mask(&observation.hero_board.middle),
            hero_bottom: mask(&observation.hero_board.bottom),
            opponent_top: mask(&observation.opponent_public_board.top),
            opponent_middle: mask(&observation.opponent_public_board.middle),
            opponent_bottom: mask(&observation.opponent_public_board.bottom),
            dealt: mask(&observation.dealt_cards),
            hero_discards: mask(&observation.hero_private_discards),
            seat: observation.seat,
            street: observation.street,
            to_act_order: observation.to_act_order,
        }
    }
}

struct SearchContext {
    config: T3Config,
    fl_ev_14: f64,
    /// Present only when the caller opted in; `None` keeps every T4 child exact.
    t4_model: Option<Model>,
    /// Present only when the caller opted in; `None` keeps the nested T3 response sampled.
    t3_second_model: Option<Model>,
    /// Required by the T2 evaluator, unused elsewhere.
    t3_first_model: Option<Model>,
    /// Required by the T2 first-seat evaluator, unused elsewhere.
    t2_second_model: Option<Model>,
    /// Only the `decide` path consults it; held here so a run that pins it
    /// says so in its continuation policy like every other evaluator does.
    t2_first_model: Option<Model>,
    /// Required by the T1 first-seat evaluator, whose every rollout opens with
    /// the opponent's T1 second-seat reply; the `decide` path consults it too.
    t1_second_model: Option<Model>,
    /// Required by the T0 evaluator on both seats; the `decide` path uses it too.
    t1_first_model: Option<Model>,
    /// Required by the T0 first-seat evaluator, whose every rollout opens with
    /// the opponent's T0 second-seat reply; the `decide` path uses it too.
    t0_second_model: Option<Model>,
    /// Held so a run that pinned the T0 first-seat root policy says so in its
    /// continuation policy, and read by nothing below. No rollout in this
    /// engine answers a T0 first-seat decision -- it is the first decision of
    /// the hand -- so a search that carries this image never consults it, and
    /// the report is the only place the pin can show. That is the same
    /// treatment `fast_t0_second_model` gets on a kind that cannot reach it,
    /// and for the same reason: a pin carried somewhere it does nothing is
    /// worth being able to see.
    t0_first_model: Option<Model>,
    /// Present only when the caller opted in, and consulted only by the T2
    /// second-seat reply inside a rollout. When it is present that reply is
    /// coarse and `t2_second_model` goes unread on the rollout path; the
    /// `decide` path is unaffected either way, because it never reaches here.
    fast_t2_second_model: Option<Model>,
    /// The same, one seat over.
    fast_t2_first_model: Option<Model>,
    /// Present only when the caller opted in, and consulted only by the T1
    /// second-seat reply inside a rollout. When it is present that reply is
    /// coarse and `t1_second_model` goes unread on the rollout path; the
    /// `decide` path is unaffected either way, because it never reaches here.
    fast_t1_second_model: Option<Model>,
    /// The same, one seat over.
    fast_t1_first_model: Option<Model>,
    /// Present only when the caller opted in, and consulted only by the T0
    /// second-seat reply inside a T0 first-seat rollout. When it is present
    /// that reply is coarse and `t0_second_model` goes unread on the rollout
    /// path; the `decide` path is unaffected either way, because it never
    /// reaches here.
    fast_t0_second_model: Option<Model>,
    t4_action_cache: HashMap<ObservationKey, Action>,
    t3_second_action_cache: HashMap<ObservationKey, Action>,
    t3_first_action_cache: HashMap<ObservationKey, Action>,
    t2_second_action_cache: HashMap<ObservationKey, Action>,
    t2_first_action_cache: HashMap<ObservationKey, Action>,
    t1_second_action_cache: HashMap<ObservationKey, Action>,
    t1_first_action_cache: HashMap<ObservationKey, Action>,
    t0_second_action_cache: HashMap<ObservationKey, Action>,
    /// Replies memoised by the coarse evaluator, kept apart from the
    /// full-precision caches above so a fingerprint answered by one encoder can
    /// never be served by the other.
    fast_t2_second_action_cache: HashMap<ObservationKey, Action>,
    fast_t2_first_action_cache: HashMap<ObservationKey, Action>,
    fast_t1_second_action_cache: HashMap<ObservationKey, Action>,
    fast_t1_first_action_cache: HashMap<ObservationKey, Action>,
    fast_t0_second_action_cache: HashMap<ObservationKey, Action>,
    /// Outlook work the coarse replies share. Held for the search rather than
    /// built per decision so candidate boards drawn against a repeated unknown
    /// set answer from it; a call with a different unknown set empties it, so
    /// holding it changes no value.
    fast_t2_second_outlook: FastOutlookCache,
    fast_t2_first_outlook: FastOutlookCache,
    fast_t1_second_outlook: FastOutlookCache,
    fast_t1_first_outlook: FastOutlookCache,
    fast_t0_second_outlook: FastOutlookCache,
    t3_child_observation_keys: HashSet<ObservationKey>,
    t4_child_observation_keys: HashSet<ObservationKey>,
}


pub fn evaluate_engine_request(request: EngineRequest) -> Result<Value, String> {
    if request.schema != REQUEST_SCHEMA {
        return Err(format!(
            "unsupported M3 request schema: {:?}",
            request.schema
        ));
    }
    request.observation.validate()?;
    let actual_fingerprint = request.observation.fingerprint();
    if request.observation_fingerprint != actual_fingerprint {
        return Err("observation_fingerprint does not match ActorObservation".to_owned());
    }
    match request.kind.as_str() {
        "t4" => {
            let config: T4Config = if request.config.is_null() {
                T4Config::default()
            } else {
                serde_json::from_value(request.config)
                    .map_err(|error| format!("invalid T4 config: {error}"))?
            };
            evaluate_t4(&request.observation, &config)
        }
        "t2" => {
            let config: T3Config = if request.config.is_null() {
                T3Config::default()
            } else {
                serde_json::from_value(request.config)
                    .map_err(|error| format!("invalid T2 config: {error}"))?
            };
            evaluate_t2(&request.observation, &config)
        }
        "t1" => {
            let config: T3Config = if request.config.is_null() {
                T3Config::default()
            } else {
                serde_json::from_value(request.config)
                    .map_err(|error| format!("invalid T1 config: {error}"))?
            };
            evaluate_t1(&request.observation, &config)
        }
        "t0" => {
            let config: T3Config = if request.config.is_null() {
                T3Config::default()
            } else {
                serde_json::from_value(request.config)
                    .map_err(|error| format!("invalid T0 config: {error}"))?
            };
            evaluate_t0(&request.observation, &config)
        }
        "decide" => {
            let config: T3Config = if request.config.is_null() {
                T3Config::default()
            } else {
                serde_json::from_value(request.config)
                    .map_err(|error| format!("invalid decide config: {error}"))?
            };
            decide(&request.observation, &config)
        }
        "t0_features" => {
            let config: T3Config = if request.config.is_null() {
                T3Config::default()
            } else {
                serde_json::from_value(request.config)
                    .map_err(|error| format!("invalid t0_features config: {error}"))?
            };
            t0_features(&request.observation, &config)
        }
        "model_scores" => {
            let config: T3Config = if request.config.is_null() {
                T3Config::default()
            } else {
                serde_json::from_value(request.config)
                    .map_err(|error| format!("invalid model_scores config: {error}"))?
            };
            model_scores(&request.observation, &config)
        }
        "t3" => {
            let config: T3Config = if request.config.is_null() {
                T3Config::default()
            } else {
                serde_json::from_value(request.config)
                    .map_err(|error| format!("invalid T3 config: {error}"))?
            };
            evaluate_t3(&request.observation, &config)
        }
        "t3_abr_components" => {
            let config: T3Config = if request.config.is_null() {
                T3Config::default()
            } else {
                serde_json::from_value(request.config)
                    .map_err(|error| format!("invalid T3 ABR config: {error}"))?
            };
            evaluate_t3_abr_components(&request.observation, &config)
        }
        "t3_explicit_support" => {
            let config: ExplicitSupportConfig = serde_json::from_value(request.config)
                .map_err(|error| format!("invalid T3 explicit-support config: {error}"))?;
            let mut result = evaluate_t3_explicit_support(&request.observation, &config)?;
            let object = result
                .as_object_mut()
                .ok_or_else(|| "T3 explicit-support result must be an object".to_owned())?;
            object.insert("status".to_owned(), json!("ok"));
            object.insert("engine_version".to_owned(), json!(ENGINE_VERSION));
            object.insert("kind".to_owned(), json!("t3_explicit_support"));
            object.insert(
                "teacher_value_status".to_owned(),
                json!("diagnostic_not_match_EV"),
            );
            Ok(result)
        }
        other => Err(format!("unsupported M3 request kind: {other:?}")),
    }
}

pub fn evaluate_t4(observation: &ActorObservation, config: &T4Config) -> Result<Value, String> {
    // These functions are also public Rust APIs, not only FFI dispatch targets.
    // Preserve the fail-closed boundary before entering trusted hot paths.
    observation.validate()?;
    if observation.street != Street::T4 {
        return Err("T4 search requires a T4 ActorObservation".to_owned());
    }
    if config.run_id.is_empty() {
        return Err("T4 run_id must not be empty".to_owned());
    }
    let actions = generate_turn_actions(&observation.hero_board, &observation.dealt_cards)?;
    if actions.is_empty() {
        return Err("T4 observation has no legal actions".to_owned());
    }
    if observation.to_act_order == ActOrder::Second {
        return evaluate_t4_second(observation, &actions);
    }
    evaluate_t4_first(observation, &actions, config)
}

fn evaluate_t4_second(observation: &ActorObservation, actions: &[Action]) -> Result<Value, String> {
    if observation.opponent_public_board.card_count() != 13 {
        return Err("T4 second requires a complete opponent board".to_owned());
    }
    let fl_ev = fl_ev_14(observation)?;
    let opponent_score = score_board_compact_trusted(&observation.opponent_public_board);
    let mut scored = Vec::with_capacity(actions.len());
    for action in actions {
        let board = action.apply_trusted(&observation.hero_board);
        let own = score_board_compact_trusted(&board);
        scored.push(ScoredAction {
            score: heads_up_terminal_score_compact(&own, &opponent_score, fl_ev),
            board_score: own,
            future_count: 1,
        });
    }
    let values = scored.iter().map(|row| row.score).collect::<Vec<_>>();
    t4_result(
        observation,
        actions,
        &scored,
        &values,
        &values,
        None,
        None,
        "not_applicable_no_future_chance",
        "t4_second_terminal_exhaustive_v1",
    )
}

fn evaluate_t4_first(
    observation: &ActorObservation,
    actions: &[Action],
    config: &T4Config,
) -> Result<Value, String> {
    if observation.hero_board.card_count() != 11
        || observation.opponent_public_board.card_count() != 11
    {
        return Err("T4 first requires 11-card hero and opponent boards".to_owned());
    }
    let candidate_plan = build_t4_future_plan(
        observation,
        config.candidate_samples,
        config.candidate_seed,
        &config.run_id,
        "candidate_selection",
    )?;
    let exact_reuse = config.candidate_samples == 0 && config.evaluation_samples == 0;
    let evaluation_plan = if exact_reuse {
        candidate_plan.clone()
    } else {
        build_t4_future_plan(
            observation,
            config.evaluation_samples,
            config.evaluation_seed,
            &config.run_id,
            "locked_evaluation",
        )?
    };
    let candidate_rows = score_t4_first_actions(observation, actions, &candidate_plan.deals)?;
    let evaluation_rows = if exact_reuse {
        candidate_rows.clone()
    } else {
        score_t4_first_actions(observation, actions, &evaluation_plan.deals)?
    };
    let candidate_values = candidate_rows
        .iter()
        .map(|row| row.score)
        .collect::<Vec<_>>();
    let evaluation_values = evaluation_rows
        .iter()
        .map(|row| row.score)
        .collect::<Vec<_>>();
    let independence = if exact_reuse {
        "not_applicable_exact_enumeration"
    } else if candidate_plan.mode != evaluation_plan.mode {
        "exact_and_counter_mc_no_shared_rng"
    } else {
        "disjoint_counter_rng_domains"
    };
    t4_result(
        observation,
        actions,
        &evaluation_rows,
        &candidate_values,
        &evaluation_values,
        Some(&candidate_plan),
        Some(&evaluation_plan),
        independence,
        "rust_t4_exchangeable_expectimax_exact_response_v1",
    )
}

#[allow(clippy::too_many_arguments)]
fn t4_result(
    observation: &ActorObservation,
    actions: &[Action],
    evaluation_rows: &[ScoredAction],
    candidate_values: &[f64],
    evaluation_values: &[f64],
    candidate_plan: Option<&FuturePlan>,
    evaluation_plan: Option<&FuturePlan>,
    independence: &str,
    solver_id: &str,
) -> Result<Value, String> {
    let ranked = canonical_descending_indices(candidate_values, actions)?;
    let selected = ranked[0];
    let second_value = ranked
        .get(1)
        .map(|&index| candidate_values[index])
        .unwrap_or(candidate_values[selected]);
    let evaluation_best = evaluation_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let rows = ranked
        .iter()
        .enumerate()
        .map(|(sorted_index, &original_index)| {
            action_result_row(
                &actions[original_index],
                original_index,
                sorted_index,
                candidate_values[original_index],
                evaluation_values[original_index],
                evaluation_best,
                original_index == selected,
                Some(&evaluation_rows[original_index]),
                Some(candidate_plan.map_or(1, |plan| plan.deals.len())),
                Some(evaluation_plan.map_or(1, |plan| plan.deals.len())),
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    let plan_payload = |plan: &FuturePlan| {
        json!({
            "mode": plan.mode,
            "stream": plan.stream,
            "future_count": plan.deals.len(),
            "root_fingerprint": observation.fingerprint(),
            "rng_key_digests": plan.rng_key_digests,
            "sample_indices": (0..plan.rng_key_digests.len()).collect::<Vec<_>>(),
        })
    };
    Ok(json!({
        "status": "ok",
        "schema": RESULT_SCHEMA,
        "engine_version": ENGINE_VERSION,
        "solver_id": solver_id,
        "kind": "t4",
        "street": "T4",
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "observation_fingerprint": observation.fingerprint(),
        "legal_action_count": actions.len(),
        "legal_action_set_digest": legal_action_set_digest(actions)?,
        "legal_action_order_digest": ordered_action_mapping_digest(actions)?,
        "selected_action_original_index": selected,
        "selected_action_key": action_key(&actions[selected])?.to_token(),
        "selected_action_evaluation_score": evaluation_values[selected],
        "best_score": evaluation_values[selected],
        "selection_score_gap": candidate_values[selected] - second_value,
        "evaluation_sample_best_score": evaluation_best,
        "evaluation_sample_regret_of_locked_selection": evaluation_best - evaluation_values[selected],
        "candidate_plan": candidate_plan.map(plan_payload),
        "evaluation_plan": evaluation_plan.map(plan_payload),
        "sample_independence": independence,
        "actions": rows,
        "counter_rng_schema": COUNTER_RNG_SCHEMA,
        "teacher_value_status": "diagnostic_not_match_EV",
    }))
}

#[allow(clippy::too_many_arguments)]
fn action_result_row(
    action: &Action,
    original_index: usize,
    sorted_index: usize,
    selection_score: f64,
    evaluation_score: f64,
    evaluation_best: f64,
    selected: bool,
    stats: Option<&ScoredAction>,
    selection_future_count: Option<usize>,
    evaluation_future_count: Option<usize>,
) -> Result<Value, String> {
    let mut value = json!({
        "original_index": original_index,
        "sorted_index": sorted_index,
        "action_key": action_key(action)?.to_token(),
        "placements": action.placements,
        "discards": action.discards,
        "score": evaluation_score,
        "joint_ev": evaluation_score,
        "selection_score": selection_score,
        "selected_by_candidate_plan": selected,
        "evaluation_regret_vs_sample_best": evaluation_best - evaluation_score,
    });
    if let Some(stats) = stats {
        let board = &stats.board_score;
        value["future_count"] = json!(stats.future_count);
        value["bust_rate"] = json!(if board.busted { 1.0 } else { 0.0 });
        value["fl_entry_rate"] = json!(if board.fl_entry_14 { 1.0 } else { 0.0 });
        value["royalty_mean"] = json!(board.total_royalty as f64);
    }
    if let Some(count) = selection_future_count {
        value["selection_future_count"] = json!(count);
    }
    if let Some(count) = evaluation_future_count {
        value["evaluation_future_count"] = json!(count);
    }
    Ok(value)
}

fn score_t4_first_actions(
    observation: &ActorObservation,
    actions: &[Action],
    deals: &[[Card; 3]],
) -> Result<Vec<ScoredAction>, String> {
    if deals.is_empty() {
        return Err("T4 future plan must not be empty".to_owned());
    }
    let fl_ev = fl_ev_14(observation)?;
    // Opponent terminal boards depend on the future deal, but not on which
    // hero action is being evaluated. Score them once and share that exact
    // response table across every legal hero action. This preserves the full
    // tree while removing the dominant repeated hand evaluation.
    let opponent_responses =
        precompute_t4_opponent_responses(&observation.opponent_public_board, deals)?;
    actions
        .par_iter()
        .map(|action| {
            let hero_final = action.apply_trusted(&observation.hero_board);
            let hero_score = score_board_compact_trusted(&hero_final);
            let mut sum = 0.0;
            for responses in &opponent_responses {
                sum += best_opponent_response_score(&hero_score, responses, fl_ev)?;
            }
            Ok(ScoredAction {
                score: sum / deals.len() as f64,
                board_score: hero_score,
                future_count: deals.len(),
            })
        })
        .collect()
}

fn precompute_t4_opponent_responses(
    opponent_board: &crate::state::Board,
    deals: &[[Card; 3]],
) -> Result<Vec<Vec<ScoredOpponentResponse>>, String> {
    deals
        .par_iter()
        .map(|deal| {
            let actions = generate_turn_actions_trusted(opponent_board, deal);
            if actions.is_empty() {
                return Err("opponent T4 deal has no legal response".to_owned());
            }
            actions
                .into_iter()
                .map(|action| {
                    let opponent_final = action.apply_trusted(opponent_board);
                    Ok(ScoredOpponentResponse {
                        board_score: score_board_compact_trusted(&opponent_final),
                    })
                })
                .collect()
        })
        .collect()
}

fn best_opponent_response_score(
    hero_score: &CompactBoardScore,
    responses: &[ScoredOpponentResponse],
    fl_ev: f64,
) -> Result<f64, String> {
    let mut best: Option<f64> = None;
    for response in responses {
        let hero_value = heads_up_terminal_score_compact(hero_score, &response.board_score, fl_ev);
        // Only the minimax value is consumed here.  ActionKey tie-breaking is
        // irrelevant when two opponent responses have exactly the same score,
        // so hashing and retaining one key per exact response is pure overhead.
        if best.is_none_or(|best_value| hero_value < best_value) {
            best = Some(hero_value);
        }
    }
    best.ok_or_else(|| "opponent T4 deal has no legal response".to_owned())
}

fn build_t4_future_plan(
    observation: &ActorObservation,
    sample_count: usize,
    seed: i64,
    run_id: &str,
    stream: &str,
) -> Result<FuturePlan, String> {
    let known_mask = observation
        .known_unavailable_cards()
        .iter()
        .fold(0_u64, |mask, card| mask | card.bit());
    let unknown = ALL_CARDS
        .iter()
        .copied()
        .filter(|card| known_mask & card.bit() == 0)
        .collect::<Vec<_>>();
    if unknown.len() != 24 {
        return Err(format!(
            "T4 first uniform belief requires 24 unknown cards, got {}",
            unknown.len()
        ));
    }
    let (deals, rng_key_digests) = if sample_count == 0 {
        (combinations_three(&unknown), Vec::new())
    } else {
        let domain_run_id = format!("{run_id}:{stream}");
        let fingerprint = observation.fingerprint();
        let mut deals = Vec::with_capacity(sample_count);
        let mut digests = Vec::with_capacity(sample_count);
        for sample_index in 0..sample_count {
            deals.push(counter_sample_t4_deal(
                &unknown,
                seed,
                &domain_run_id,
                stream,
                sample_index as u64,
                &fingerprint,
            )?);
            let key = CounterRngKey::new(
                seed,
                &domain_run_id,
                "t4_common_future",
                sample_index as u64,
                CounterActor::Chance,
                "T4",
                stream,
                0,
                &fingerprint,
            )?;
            digests.push(sha256_hex_json(&key.payload()));
        }
        (deals, digests)
    };
    Ok(FuturePlan {
        deals,
        mode: if sample_count == 0 {
            "exact_uniform_marginal"
        } else {
            "counter_mc"
        },
        stream: stream.to_owned(),
        rng_key_digests,
    })
}

fn combinations_three(cards: &[Card]) -> Vec<[Card; 3]> {
    let mut result = Vec::with_capacity(cards.len() * (cards.len() - 1) * (cards.len() - 2) / 6);
    for first in 0..cards.len() - 2 {
        for second in first + 1..cards.len() - 1 {
            for third in second + 1..cards.len() {
                result.push([cards[first], cards[second], cards[third]]);
            }
        }
    }
    result
}

fn counter_sample_t4_deal(
    cards: &[Card],
    seed: i64,
    run_id: &str,
    stream: &str,
    sample_index: u64,
    root_fingerprint: &str,
) -> Result<[Card; 3], String> {
    let mut available = cards.to_vec();
    let mut selected = Vec::with_capacity(3);
    for draw_index in 0..3_u64 {
        let offset = counter_randbelow_t4(
            available.len(),
            seed,
            run_id,
            stream,
            sample_index,
            root_fingerprint,
            draw_index,
        )?;
        selected.push(available.remove(offset));
    }
    selected.sort_unstable_by_key(|card| card.index());
    Ok([selected[0], selected[1], selected[2]])
}

#[allow(clippy::too_many_arguments)]
fn counter_randbelow_t4(
    upper_bound: usize,
    seed: i64,
    run_id: &str,
    stream: &str,
    sample_index: u64,
    root_fingerprint: &str,
    draw_index: u64,
) -> Result<usize, String> {
    let upper = upper_bound as u64;
    let limit = RNG_DOMAIN - (RNG_DOMAIN % upper);
    let mut attempt = 0_u64;
    loop {
        let value = CounterRngKey::new(
            seed,
            run_id,
            "t4_common_future",
            sample_index,
            CounterActor::Chance,
            "T4",
            stream,
            (draw_index << ATTEMPT_BITS) | attempt,
            root_fingerprint,
        )?
        .seed();
        if value < limit {
            return Ok((value % upper) as usize);
        }
        attempt += 1;
    }
}

pub fn evaluate_t3(observation: &ActorObservation, config: &T3Config) -> Result<Value, String> {
    observation.validate()?;
    if observation.street != Street::T3 {
        return Err("T3 search requires a T3 ActorObservation".to_owned());
    }
    if config.candidate_samples == 0
        || config.evaluation_samples == 0
        || config.downstream_t3_samples == 0
    {
        return Err("T3 candidate/evaluation/downstream_t3 samples must be positive".to_owned());
    }
    if config.run_id.is_empty() {
        return Err("T3 run_id must not be empty".to_owned());
    }
    let candidate_batch = sample_hidden_card_particles(
        observation,
        config.candidate_seed,
        &format!("{}:candidate_selection", config.run_id),
        config.candidate_samples,
        0,
    )?;
    let evaluation_batch = sample_hidden_card_particles(
        observation,
        config.evaluation_seed,
        &format!("{}:locked_evaluation", config.run_id),
        config.evaluation_samples,
        0,
    )?;
    let candidate_keys = candidate_batch
        .particles
        .iter()
        .map(|particle| particle.rng_key_digest.as_str())
        .collect::<HashSet<_>>();
    if evaluation_batch
        .particles
        .iter()
        .any(|particle| candidate_keys.contains(particle.rng_key_digest.as_str()))
    {
        return Err("candidate-selection and evaluation particle RNG keys overlap".to_owned());
    }
    let actions = generate_turn_actions(&observation.hero_board, &observation.dealt_cards)?;
    if actions.is_empty() {
        return Err("T3 observation has no legal actions".to_owned());
    }
    let actions = narrow_by_learned_model(observation, config, actions)?;
    let mut context = SearchContext {
        config: config.clone(),
        fl_ev_14: fl_ev_14(observation)?,
        t4_model: load_learned_t4_model(config)?,
        t3_second_model: load_learned_model(
            config.learned_t3_second_model_path.as_deref(),
            config.learned_t3_second_model_sha256.as_deref(),
            "learned_t3_second_model",
            crate::t3_features::FEATURE_SIZE,
        )?,
        t3_first_model: None,
        t2_second_model: None,
        t2_first_model: None,
        t1_second_model: None,
        t1_first_model: None,
        t0_second_model: None,
        t0_first_model: None,
        fast_t2_second_model: None,
        fast_t2_first_model: None,
        fast_t1_second_model: None,
        fast_t1_first_model: None,
        fast_t0_second_model: None,
        t4_action_cache: HashMap::new(),
        t3_second_action_cache: HashMap::new(),
        t3_first_action_cache: HashMap::new(),
        t2_second_action_cache: HashMap::new(),
        t2_first_action_cache: HashMap::new(),
        t1_second_action_cache: HashMap::new(),
        t1_first_action_cache: HashMap::new(),
        t0_second_action_cache: HashMap::new(),
        fast_t2_second_action_cache: HashMap::new(),
        fast_t2_first_action_cache: HashMap::new(),
        fast_t1_second_action_cache: HashMap::new(),
        fast_t1_first_action_cache: HashMap::new(),
        fast_t0_second_action_cache: HashMap::new(),
        fast_t2_second_outlook: FastOutlookCache::new(),
        fast_t2_first_outlook: FastOutlookCache::new(),
        fast_t1_second_outlook: FastOutlookCache::new(),
        fast_t1_first_outlook: FastOutlookCache::new(),
        fast_t0_second_outlook: FastOutlookCache::new(),
        t3_child_observation_keys: HashSet::new(),
        t4_child_observation_keys: HashSet::new(),
    };
    let candidate_values = score_t3_actions(
        observation,
        &actions,
        &candidate_batch.particles,
        &mut context,
    )?;
    let evaluation_values = score_t3_actions(
        observation,
        &actions,
        &evaluation_batch.particles,
        &mut context,
    )?;
    let ranked = canonical_descending_indices(&candidate_values, &actions)?;
    let selected = ranked[0];
    let candidate_second = ranked
        .get(1)
        .map(|&index| candidate_values[index])
        .unwrap_or(candidate_values[selected]);
    let evaluation_best = evaluation_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let rows = ranked
        .iter()
        .enumerate()
        .map(|(sorted_index, &original_index)| {
            action_result_row(
                &actions[original_index],
                original_index,
                sorted_index,
                candidate_values[original_index],
                evaluation_values[original_index],
                evaluation_best,
                original_index == selected,
                None,
                Some(candidate_batch.particles.len()),
                Some(evaluation_batch.particles.len()),
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(json!({
        "status": "ok",
        "schema": RESULT_SCHEMA,
        "engine_version": ENGINE_VERSION,
        "solver_id": "rust_crn_sequential_t3_v1",
        "kind": "t3",
        "street": "T3",
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "observation_fingerprint": observation.fingerprint(),
        "legal_action_count": actions.len(),
        "legal_action_set_digest": legal_action_set_digest(&actions)?,
        "legal_action_order_digest": ordered_action_mapping_digest(&actions)?,
        "selected_action_original_index": selected,
        "best_action_original_index": selected,
        "selected_action_key": action_key(&actions[selected])?.to_token(),
        "selected_action_evaluation_score": evaluation_values[selected],
        "best_score": evaluation_values[selected],
        "selection_score_gap": candidate_values[selected] - candidate_second,
        "score_gap": candidate_values[selected] - candidate_second,
        "evaluation_sample_best_score": evaluation_best,
        "evaluation_sample_regret_of_locked_selection": evaluation_best - evaluation_values[selected],
        "candidate_belief": candidate_batch.to_json(false),
        "evaluation_belief": evaluation_batch.to_json(false),
        "candidate_rng_key_digests": candidate_batch.particles.iter().map(|particle| particle.rng_key_digest.clone()).collect::<Vec<_>>(),
        "evaluation_rng_key_digests": evaluation_batch.particles.iter().map(|particle| particle.rng_key_digest.clone()).collect::<Vec<_>>(),
        "sample_independence": "disjoint_particle_rng_keys",
        "continuation_policy": continuation_policy_report(config, &context),
        "child_information_set_count": context.t3_child_observation_keys.len() + context.t4_child_observation_keys.len(),
        "actions": rows,
        "teacher_value_status": "diagnostic_not_match_EV",
    }))
}

/// Diagnostic-only T3 evaluator for ABR development labels.
///
/// The search tree, chance particles, continuation policy, action ranking, and
/// HU score accumulation are identical to [`evaluate_t3`].  This entrypoint
/// additionally reports linear terminal-reward components required to define
/// auditable foul-pressure and royalty-denial utilities.  It is not consumed
/// by the runtime policy and never exposes sampled cards or opponent discards.
pub fn evaluate_t3_abr_components(
    observation: &ActorObservation,
    config: &T3Config,
) -> Result<Value, String> {
    observation.validate()?;
    if observation.street != Street::T3 {
        return Err("T3 ABR component search requires a T3 ActorObservation".to_owned());
    }
    if config.candidate_samples == 0
        || config.evaluation_samples == 0
        || config.downstream_t3_samples == 0
    {
        return Err(
            "T3 ABR candidate/evaluation/downstream_t3 samples must be positive".to_owned(),
        );
    }
    if config.run_id.is_empty() {
        return Err("T3 ABR run_id must not be empty".to_owned());
    }
    let candidate_batch = sample_hidden_card_particles(
        observation,
        config.candidate_seed,
        &format!("{}:candidate_selection", config.run_id),
        config.candidate_samples,
        0,
    )?;
    let evaluation_batch = sample_hidden_card_particles(
        observation,
        config.evaluation_seed,
        &format!("{}:locked_evaluation", config.run_id),
        config.evaluation_samples,
        0,
    )?;
    let candidate_keys = candidate_batch
        .particles
        .iter()
        .map(|particle| particle.rng_key_digest.as_str())
        .collect::<HashSet<_>>();
    if evaluation_batch
        .particles
        .iter()
        .any(|particle| candidate_keys.contains(particle.rng_key_digest.as_str()))
    {
        return Err("candidate-selection and evaluation particle RNG keys overlap".to_owned());
    }
    let actions = generate_turn_actions(&observation.hero_board, &observation.dealt_cards)?;
    if actions.is_empty() {
        return Err("T3 ABR observation has no legal actions".to_owned());
    }
    let mut context = SearchContext {
        config: config.clone(),
        fl_ev_14: fl_ev_14(observation)?,
        t4_model: load_learned_t4_model(config)?,
        t3_second_model: load_learned_model(
            config.learned_t3_second_model_path.as_deref(),
            config.learned_t3_second_model_sha256.as_deref(),
            "learned_t3_second_model",
            crate::t3_features::FEATURE_SIZE,
        )?,
        t3_first_model: None,
        t2_second_model: None,
        t2_first_model: None,
        t1_second_model: None,
        t1_first_model: None,
        t0_second_model: None,
        t0_first_model: None,
        fast_t2_second_model: None,
        fast_t2_first_model: None,
        fast_t1_second_model: None,
        fast_t1_first_model: None,
        fast_t0_second_model: None,
        t4_action_cache: HashMap::new(),
        t3_second_action_cache: HashMap::new(),
        t3_first_action_cache: HashMap::new(),
        t2_second_action_cache: HashMap::new(),
        t2_first_action_cache: HashMap::new(),
        t1_second_action_cache: HashMap::new(),
        t1_first_action_cache: HashMap::new(),
        t0_second_action_cache: HashMap::new(),
        fast_t2_second_action_cache: HashMap::new(),
        fast_t2_first_action_cache: HashMap::new(),
        fast_t1_second_action_cache: HashMap::new(),
        fast_t1_first_action_cache: HashMap::new(),
        fast_t0_second_action_cache: HashMap::new(),
        fast_t2_second_outlook: FastOutlookCache::new(),
        fast_t2_first_outlook: FastOutlookCache::new(),
        fast_t1_second_outlook: FastOutlookCache::new(),
        fast_t1_first_outlook: FastOutlookCache::new(),
        fast_t0_second_outlook: FastOutlookCache::new(),
        t3_child_observation_keys: HashSet::new(),
        t4_child_observation_keys: HashSet::new(),
    };
    let candidate_components = score_t3_action_components(
        observation,
        &actions,
        &candidate_batch.particles,
        &mut context,
    )?;
    let evaluation_components = score_t3_action_components(
        observation,
        &actions,
        &evaluation_batch.particles,
        &mut context,
    )?;
    let candidate_values = candidate_components
        .iter()
        .map(|row| row.hu_score)
        .collect::<Vec<_>>();
    let evaluation_values = evaluation_components
        .iter()
        .map(|row| row.hu_score)
        .collect::<Vec<_>>();
    let ranked = canonical_descending_indices(&candidate_values, &actions)?;
    let selected = ranked[0];
    let candidate_second = ranked
        .get(1)
        .map(|&index| candidate_values[index])
        .unwrap_or(candidate_values[selected]);
    let evaluation_best = evaluation_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let rows = ranked
        .iter()
        .enumerate()
        .map(|(sorted_index, &original_index)| {
            let mut row = action_result_row(
                &actions[original_index],
                original_index,
                sorted_index,
                candidate_values[original_index],
                evaluation_values[original_index],
                evaluation_best,
                original_index == selected,
                None,
                Some(candidate_batch.particles.len()),
                Some(evaluation_batch.particles.len()),
            )?;
            row["terminal_components"] =
                terminal_component_payload(&evaluation_components[original_index]);
            Ok(row)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(json!({
        "status": "ok",
        "schema": T3_ABR_COMPONENT_RESULT_SCHEMA,
        "engine_version": ENGINE_VERSION,
        "solver_id": "rust_crn_sequential_t3_abr_components_v1",
        "legacy_solver_id": "rust_crn_sequential_t3_v1",
        "kind": "t3_abr_components",
        "street": "T3",
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "observation_fingerprint": observation.fingerprint(),
        "legal_action_count": actions.len(),
        "legal_action_set_digest": legal_action_set_digest(&actions)?,
        "legal_action_order_digest": ordered_action_mapping_digest(&actions)?,
        "selected_action_original_index": selected,
        "best_action_original_index": selected,
        "selected_action_key": action_key(&actions[selected])?.to_token(),
        "selected_action_evaluation_score": evaluation_values[selected],
        "best_score": evaluation_values[selected],
        "selection_score_gap": candidate_values[selected] - candidate_second,
        "score_gap": candidate_values[selected] - candidate_second,
        "evaluation_sample_best_score": evaluation_best,
        "evaluation_sample_regret_of_locked_selection":
            evaluation_best - evaluation_values[selected],
        "candidate_belief": candidate_batch.to_json(false),
        "evaluation_belief": evaluation_batch.to_json(false),
        "candidate_rng_key_digests": candidate_batch.particles.iter().map(|particle| particle.rng_key_digest.clone()).collect::<Vec<_>>(),
        "evaluation_rng_key_digests": evaluation_batch.particles.iter().map(|particle| particle.rng_key_digest.clone()).collect::<Vec<_>>(),
        "sample_independence": "disjoint_particle_rng_keys",
        "continuation_policy": continuation_policy_report(config, &context),
        "child_information_set_count": context.t3_child_observation_keys.len() + context.t4_child_observation_keys.len(),
        "terminal_component_schema": "hu_m3_t3_terminal_component_means_v1",
        "terminal_component_visibility":
            "aggregate_only_no_sampled_cards_no_opponent_private_discards",
        "actions": rows,
        "teacher_value_status": "diagnostic_not_match_EV",
    }))
}

fn score_t3_action_components(
    observation: &ActorObservation,
    actions: &[Action],
    particles: &[HiddenCardParticle],
    context: &mut SearchContext,
) -> Result<Vec<TerminalComponentMeans>, String> {
    if particles.is_empty() {
        return Err("T3 ABR component particle batch must not be empty".to_owned());
    }
    let mut values = Vec::with_capacity(actions.len());
    for action in actions {
        let mut hu_score = 0.0;
        let mut hero_busted = 0_usize;
        let mut opponent_busted = 0_usize;
        let mut hero_scoop = 0_usize;
        let mut opponent_scoop = 0_usize;
        let mut hero_royalty = 0_i64;
        let mut opponent_royalty = 0_i64;
        let mut hero_fl_value = 0.0;
        let mut opponent_fl_value = 0.0;
        for particle in particles {
            let outcome = if observation.to_act_order == ActOrder::First {
                rollout_t3_first(observation, action, particle, context)?
            } else {
                rollout_t3_second(observation, action, particle, context)?
            };
            hu_score += outcome.hu_score;
            hero_busted += usize::from(outcome.hero_busted);
            opponent_busted += usize::from(outcome.opponent_busted);
            hero_scoop += usize::from(outcome.hero_scoop);
            opponent_scoop += usize::from(outcome.opponent_scoop);
            hero_royalty += i64::from(outcome.hero_royalty);
            opponent_royalty += i64::from(outcome.opponent_royalty);
            hero_fl_value += outcome.hero_fl_value;
            opponent_fl_value += outcome.opponent_fl_value;
        }
        let count = particles.len();
        let denominator = count as f64;
        values.push(TerminalComponentMeans {
            hu_score: hu_score / denominator,
            hero_bust_rate: hero_busted as f64 / denominator,
            opponent_bust_rate: opponent_busted as f64 / denominator,
            hero_scoop_rate: hero_scoop as f64 / denominator,
            opponent_scoop_rate: opponent_scoop as f64 / denominator,
            hero_royalty_mean: hero_royalty as f64 / denominator,
            opponent_royalty_mean: opponent_royalty as f64 / denominator,
            hero_fl_value_mean: hero_fl_value / denominator,
            opponent_fl_value_mean: opponent_fl_value / denominator,
            future_count: count,
        });
    }
    Ok(values)
}

fn terminal_component_payload(value: &TerminalComponentMeans) -> Value {
    json!({
        "hu_score_mean": value.hu_score,
        "hero_bust_rate": value.hero_bust_rate,
        "opponent_bust_rate": value.opponent_bust_rate,
        "hero_scoop_rate": value.hero_scoop_rate,
        "opponent_scoop_rate": value.opponent_scoop_rate,
        "hero_royalty_mean": value.hero_royalty_mean,
        "opponent_royalty_mean": value.opponent_royalty_mean,
        "hero_fl_value_mean": value.hero_fl_value_mean,
        "opponent_fl_value_mean": value.opponent_fl_value_mean,
        "future_count": value.future_count,
    })
}

fn score_t3_actions(
    observation: &ActorObservation,
    actions: &[Action],
    particles: &[HiddenCardParticle],
    context: &mut SearchContext,
) -> Result<Vec<f64>, String> {
    let mut values = Vec::with_capacity(actions.len());
    for action in actions {
        let mut sum = 0.0;
        for particle in particles {
            let outcome = if observation.to_act_order == ActOrder::First {
                rollout_t3_first(observation, action, particle, context)?
            } else {
                rollout_t3_second(observation, action, particle, context)?
            };
            sum += outcome.hu_score;
        }
        values.push(sum / particles.len() as f64);
    }
    Ok(values)
}

fn rollout_t3_second(
    observation: &ActorObservation,
    root_action: &Action,
    particle: &HiddenCardParticle,
    context: &mut SearchContext,
) -> Result<TerminalOutcome, String> {
    let after_root = root_action.apply_trusted(&observation.hero_board);
    let opponent_t4_observation = ActorObservation::new(
        observation.opponent_public_board.clone(),
        after_root.clone(),
        particle.draw(3, 0)?.to_vec(),
        particle.opponent_private_discards.clone(),
        Seat::First,
        Street::T4,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let opponent_t4_action = locked_t4_action(&opponent_t4_observation, context)?;
    let opponent_final = opponent_t4_action.apply_trusted(&observation.opponent_public_board);
    let mut hero_discards = observation.hero_private_discards.clone();
    hero_discards.extend(root_action.discards.iter().copied());
    let hero_t4_observation = ActorObservation::new(
        after_root.clone(),
        opponent_final.clone(),
        particle.draw(3, 3)?.to_vec(),
        hero_discards,
        Seat::Second,
        Street::T4,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let hero_t4_action = locked_t4_action(&hero_t4_observation, context)?;
    let hero_final = hero_t4_action.apply_trusted(&after_root);
    terminal_outcome(&hero_final, &opponent_final, context.fl_ev_14)
}

fn rollout_t3_first(
    observation: &ActorObservation,
    root_action: &Action,
    particle: &HiddenCardParticle,
    context: &mut SearchContext,
) -> Result<TerminalOutcome, String> {
    let after_root = root_action.apply_trusted(&observation.hero_board);
    let opponent_t3_observation = ActorObservation::new(
        observation.opponent_public_board.clone(),
        after_root.clone(),
        particle.draw(3, 0)?.to_vec(),
        particle.opponent_private_discards.clone(),
        Seat::Second,
        Street::T3,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t3_action = locked_t3_second_action(&opponent_t3_observation, context)?;
    let opponent_after_t3 = opponent_t3_action.apply_trusted(&observation.opponent_public_board);
    let mut hero_discards = observation.hero_private_discards.clone();
    hero_discards.extend(root_action.discards.iter().copied());
    let hero_t4_observation = ActorObservation::new(
        after_root.clone(),
        opponent_after_t3.clone(),
        particle.draw(3, 3)?.to_vec(),
        hero_discards,
        Seat::First,
        Street::T4,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let hero_t4_action = locked_t4_action(&hero_t4_observation, context)?;
    let hero_final = hero_t4_action.apply_trusted(&after_root);
    let mut opponent_discards = particle.opponent_private_discards.clone();
    opponent_discards.extend(opponent_t3_action.discards.iter().copied());
    let opponent_t4_observation = ActorObservation::new(
        opponent_after_t3.clone(),
        hero_final.clone(),
        particle.draw(3, 6)?.to_vec(),
        opponent_discards,
        Seat::Second,
        Street::T4,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t4_action = locked_t4_action(&opponent_t4_observation, context)?;
    let opponent_final = opponent_t4_action.apply_trusted(&opponent_after_t3);
    terminal_outcome(&hero_final, &opponent_final, context.fl_ev_14)
}

fn locked_t4_action(
    observation: &ActorObservation,
    context: &mut SearchContext,
) -> Result<Action, String> {
    // All child T4 observations share one immutable scoring context inside a
    // root solve. Card masks therefore provide the same order-invariant cache
    // identity as the public SHA-256 fingerprint without JSON serialization,
    // repeated validation, or cryptographic hashing at every rollout node.
    let key = ObservationKey::new(observation);
    context.t4_child_observation_keys.insert(key);
    if context.config.use_t4_action_cache {
        if let Some(action) = context.t4_action_cache.get(&key) {
            return Ok(action.clone());
        }
    }
    // Only the first seat is ever replaced. Acting second the opponent board is
    // already complete, so the exact answer is a closed-form comparison with
    // nothing left to approximate and nothing to gain by approximating it.
    let selected = if context.t4_model.is_some() && observation.to_act_order == ActOrder::First {
        let model = context
            .t4_model
            .as_ref()
            .expect("presence checked immediately above");
        learned_t4_first_action(observation, model)?
    } else {
        let config = T4Config {
            candidate_samples: context.config.downstream_t4_samples,
            evaluation_samples: context.config.downstream_t4_samples,
            seed: context.config.seed,
            candidate_seed: context.config.seed,
            evaluation_seed: context.config.seed,
            run_id: format!("{}:child-t4", context.config.run_id),
        };
        select_t4_action_without_result(observation, &config)?
    };
    if context.config.use_t4_action_cache {
        context.t4_action_cache.insert(key, selected.clone());
    }
    Ok(selected)
}

/// Rank the legal T3 first-seat actions with the learned evaluator.
///
/// The opponent block is independent of the hero's action and is computed once
/// per call; the fingerprint cache above this absorbs repeats across rollouts.
fn learned_t3_first_action(
    observation: &ActorObservation,
    model: &Model,
) -> Result<Action, String> {
    if observation.hero_board.card_count() != 9
        || observation.opponent_public_board.card_count() != 9
    {
        return Err(
            "learned T3 first-seat evaluator requires 9-card hero and opponent boards"
                .to_owned(),
        );
    }
    let actions = generate_turn_actions_trusted(&observation.hero_board, &observation.dealt_cards);
    if actions.is_empty() {
        return Err("T3 observation has no legal actions".to_owned());
    }
    let unknown = crate::t3_features::unknown_cards(observation);
    let (opponent_block, opponent_finishes) = crate::t3first_features::opponent_outlook_first(
        &observation.opponent_public_board,
        &unknown,
    )?;
    let mut scratch = model.scratch();
    let mut values = Vec::with_capacity(actions.len());
    for action in &actions {
        let board = action.apply_trusted(&observation.hero_board);
        let features = crate::t3first_features::encode_first(
            observation,
            &board,
            &unknown,
            &opponent_block,
            &opponent_finishes,
        );
        values.push(model.predict_with(&features, &mut scratch)? as f64);
    }
    let selected = canonical_descending_indices(&values, &actions)?[0];
    Ok(actions[selected].clone())
}

/// The nested first-seat reply inside a T2 rollout, cached by fingerprint.
///
/// No sampled fallback exists on purpose: the search that would answer costs
/// minutes per T2 decision, so this street's teacher requires the model.
fn locked_t3_first_action(
    observation: &ActorObservation,
    context: &mut SearchContext,
) -> Result<Action, String> {
    let Some(model) = context.t3_first_model.as_ref() else {
        return Err(
            "T2 evaluation requires learned_t3_first_model; a nested sampled T3 \
             first-seat search would cost minutes per decision"
                .to_owned(),
        );
    };
    let key = ObservationKey::new(observation);
    context.t3_child_observation_keys.insert(key);
    if let Some(action) = context.t3_first_action_cache.get(&key) {
        return Ok(action.clone());
    }
    let selected = learned_t3_first_action(observation, model)?;
    context
        .t3_first_action_cache
        .insert(key, selected.clone());
    Ok(selected)
}

/// Rank the legal T2 second-seat actions with the learned evaluator.
///
/// Shares the first-seat T3 feature layout: structural block, the free outlook
/// of the hero board the action produces, the free outlook of the opponent
/// board, and the head-to-head comparison of the two finish distributions. The
/// opponent block does not depend on the hero's action and is computed once.
fn learned_t2_second_action(
    observation: &ActorObservation,
    model: &Model,
) -> Result<Action, String> {
    if observation.hero_board.card_count() != 7
        || observation.opponent_public_board.card_count() != 9
    {
        return Err(
            "learned T2 second-seat evaluator requires a 7-card hero board and a \
             9-card opponent board"
                .to_owned(),
        );
    }
    learned_t2_action(observation, model)
}

/// Rank the legal T2 first-seat actions with the learned evaluator.
///
/// Same encoding as the second seat, one street of opponent information less:
/// acting first the opponent has answered T1 and no more, so its public board
/// carries seven cards and six open slots rather than nine and four. The free
/// outlook covers both shapes, so the composition below is shared rather than
/// duplicated.
fn learned_t2_first_action(
    observation: &ActorObservation,
    model: &Model,
) -> Result<Action, String> {
    if observation.hero_board.card_count() != 7
        || observation.opponent_public_board.card_count() != 7
    {
        return Err(
            "learned T2 first-seat evaluator requires a 7-card hero board and a \
             7-card opponent board"
                .to_owned(),
        );
    }
    learned_t2_action(observation, model)
}

/// Rank the legal T1 second-seat actions with the learned evaluator.
///
/// One street earlier than the T2 pair and otherwise the same encoding. Acting
/// second at T1 the hero holds five cards and the opponent seven, so after the
/// action both sides show seven cards with six open slots -- a shape the free
/// outlook already describes, which is why the composition below is the shared
/// one rather than a fourth copy of it.
fn learned_t1_second_action(
    observation: &ActorObservation,
    model: &Model,
) -> Result<Action, String> {
    if observation.hero_board.card_count() != 5
        || observation.opponent_public_board.card_count() != 7
    {
        return Err(
            "learned T1 second-seat evaluator requires a 5-card hero board and a \
             7-card opponent board"
                .to_owned(),
        );
    }
    learned_t2_action(observation, model)
}

/// The composition both T2 seats share, once the geometry is known to be one
/// the free outlook can describe.
///
/// The T1 second seat shares it too: after its action both boards show seven
/// cards, which is a shape the free outlook already covers. The name is the one
/// the T2 cost tests refer to and is kept for that reason; nothing in the body
/// is specific to T2.
fn learned_t2_action(observation: &ActorObservation, model: &Model) -> Result<Action, String> {
    let actions = generate_turn_actions_trusted(&observation.hero_board, &observation.dealt_cards);
    if actions.is_empty() {
        return Err("T2 observation has no legal actions".to_owned());
    }
    learned_action_over(observation, model, &actions)
}

/// Rank the legal T1 first-seat actions with the learned evaluator.
///
/// One street earlier than [`learned_t1_second_action`] and otherwise the same
/// encoding. Acting first at T1 neither side has answered, so both boards hold
/// five cards; after the action the hero shows seven with six open slots and
/// the opponent still five with eight. Both are shapes the free outlook
/// describes, which is why this composes the shared body rather than adding a
/// fifth copy of it.
fn learned_t1_first_action(
    observation: &ActorObservation,
    model: &Model,
) -> Result<Action, String> {
    if observation.hero_board.card_count() != 5
        || observation.opponent_public_board.card_count() != 5
    {
        return Err(
            "learned T1 first-seat evaluator requires 5-card hero and opponent boards".to_owned(),
        );
    }
    let actions = generate_turn_actions_trusted(&observation.hero_board, &observation.dealt_cards);
    if actions.is_empty() {
        return Err("T1 observation has no legal actions".to_owned());
    }
    learned_action_over(observation, model, &actions)
}

/// Rank the legal T0 second-seat actions with the learned evaluator.
///
/// The opening street is the one place the action set comes from a different
/// generator: five cards are dealt, all five are placed, and nothing is
/// discarded, so the legal set is the 232 openings rather than the two-of-three
/// turn actions. The encoding is unchanged. A T0 action produces a five-card
/// hero board with eight open slots, and acting second the opponent's board is
/// also five with eight open -- exactly the width the free outlook already
/// covers, so the same structural, hero-outlook, opponent-outlook and
/// head-to-head composition describes an opening candidate board without a new
/// feature layout.
fn learned_t0_second_action(
    observation: &ActorObservation,
    model: &Model,
) -> Result<Action, String> {
    if observation.hero_board.card_count() != 0
        || observation.opponent_public_board.card_count() != 5
    {
        return Err(
            "learned T0 second-seat evaluator requires an empty hero board and a \
             5-card opponent board"
                .to_owned(),
        );
    }
    let actions = generate_initial_actions(&observation.hero_board, &observation.dealt_cards)?;
    if actions.is_empty() {
        return Err("T0 observation has no legal actions".to_owned());
    }
    learned_action_over(observation, model, &actions)
}

/// Score a supplied legal set with the 168-wide evaluator and pick the best.
///
/// Split out from [`learned_t2_action`] when T0 arrived, because T0 is the one
/// street whose legal set comes from the opening generator rather than the turn
/// generator. Nothing below depends on how the set was produced or on which
/// street it belongs to -- only that every candidate board it yields is a width
/// the free outlook can describe, which each caller checks before arriving.
/// The learned evaluator's value for every legal action, in the order given.
///
/// Split out of [`learned_action_over`], which now calls it and keeps taking
/// the first canonical element. Nothing about the chosen action changes: the
/// values, their order and the tie-break are the ones that were always
/// computed here, and only the return type is new.
///
/// It exists because a narrowing stage needs the ranking rather than the pick.
/// The learned evaluator already scores the whole fan in one cached pass, so
/// the scores are free where a sampled prefilter has to buy them, and they do
/// not move with a seed.
fn learned_values_over(
    observation: &ActorObservation,
    model: &Model,
    actions: &[Action],
) -> Result<Vec<f64>, String> {
    let unknown = crate::t3_features::unknown_cards(observation);
    // Every candidate board differs from the last in one or two rows, and all of
    // them are read against this one unknown set, so the outlook's per-row and
    // per-draw work is shared rather than repeated. Values are unaffected; see
    // `FreeOutlookCache`.
    let mut outlook = crate::t3first_features::FreeOutlookCache::new();
    let (opponent_block, opponent_finishes) =
        outlook.outlook(&observation.opponent_public_board, &unknown)?;
    let mut scratch = model.scratch();
    let mut values = Vec::with_capacity(actions.len());
    for action in actions {
        let board = action.apply_trusted(&observation.hero_board);
        let mut features = [0.0f32; crate::t3first_features::FEATURE_SIZE];
        features[..86].copy_from_slice(&crate::t3_features::encode_structural(observation, &board));
        let (hero_block, hero_finishes) = outlook.outlook(&board, &unknown)?;
        features[86..122].copy_from_slice(&hero_block);
        features[122..158].copy_from_slice(&opponent_block);
        features[158..].copy_from_slice(&crate::t3_features::head_to_head(
            &hero_finishes,
            &opponent_finishes,
        ));
        values.push(model.predict_with(&features, &mut scratch)? as f64);
    }
    Ok(values)
}

fn learned_action_over(
    observation: &ActorObservation,
    model: &Model,
    actions: &[Action],
) -> Result<Action, String> {
    let values = learned_values_over(observation, model, actions)?;
    let selected = canonical_descending_indices(&values, actions)?[0];
    Ok(actions[selected].clone())
}

/// Cut a T1/T2/T3 fan to the learned evaluator's best `keep`, before particles.
///
/// Returns the actions unchanged when the cut is disabled, when the model for
/// this street is not pinned, or when the fan is already no wider than `keep` --
/// so a caller that does not ask for narrowing gets byte-identical work.
///
/// The order of the survivors is the CANONICAL one, not the model's ranking:
/// downstream code takes `[0]` on its own canonical ordering and compares
/// action keys, and reordering the fan here would change which of two tied
/// actions a later stage picks.
fn narrow_by_learned_model(
    observation: &ActorObservation,
    config: &T3Config,
    actions: Vec<Action>,
) -> Result<Vec<Action>, String> {
    let keep = config.learned_prefilter_keep;
    if keep == 0 || actions.len() <= keep {
        return Ok(actions);
    }
    let (path, sha, field) = match (observation.street, observation.to_act_order) {
        (Street::T3, ActOrder::Second) => (
            &config.learned_t3_second_model_path,
            &config.learned_t3_second_model_sha256,
            "learned_t3_second_model",
        ),
        (Street::T3, ActOrder::First) => (
            &config.learned_t3_first_model_path,
            &config.learned_t3_first_model_sha256,
            "learned_t3_first_model",
        ),
        (Street::T2, ActOrder::Second) => (
            &config.learned_t2_second_model_path,
            &config.learned_t2_second_model_sha256,
            "learned_t2_second_model",
        ),
        (Street::T2, ActOrder::First) => (
            &config.learned_t2_first_model_path,
            &config.learned_t2_first_model_sha256,
            "learned_t2_first_model",
        ),
        (Street::T1, ActOrder::Second) => (
            &config.learned_t1_second_model_path,
            &config.learned_t1_second_model_sha256,
            "learned_t1_second_model",
        ),
        (Street::T1, ActOrder::First) => (
            &config.learned_t1_first_model_path,
            &config.learned_t1_first_model_sha256,
            "learned_t1_first_model",
        ),
        _ => return Ok(actions),
    };
    let model = match load_learned_model(
        path.as_deref(),
        sha.as_deref(),
        field,
        crate::t3first_features::FEATURE_SIZE,
    )? {
        Some(model) => model,
        // No model pinned for this street: the cut cannot be made, and refusing
        // here would turn an optional speed-up into a hard requirement.
        None => return Ok(actions),
    };
    let values = learned_values_over(observation, &model, &actions)?;
    let order = canonical_descending_indices(&values, &actions)?;
    let mut survivors: Vec<usize> = order[..keep].to_vec();
    survivors.sort_unstable();
    Ok(survivors.into_iter().map(|i| actions[i].clone()).collect())
}

/// Rank the legal T2 actions with the coarse evaluator, on either seat.
///
/// The same shape as [`learned_action_over`] -- same legal set from the same
/// generator, same 168-wide vector in the same block order, same canonical
/// descending order and same first element taken -- with
/// [`crate::fast_features`] supplying the outlook instead of
/// [`crate::t3first_features`]. What changes is only how finely the two outlook
/// blocks are sampled, and the model that reads them was fitted to the coarse
/// numbers rather than the fine ones, so this is not the full evaluator run
/// cheaply; it is a different evaluator that answers the same question.
///
/// Both T2 seats share it, the way both T1 seats share [`fast_t1_action`].
/// Acting first the opponent has answered T1 and no more, so its board carries
/// seven cards; acting second it has answered T2 and carries nine. Those are
/// six and four open slots, and the hero's own board is seven cards either way,
/// which becomes nine -- four open -- once the candidate places. Every one of
/// those widths is one the free-slot outlook covers, so the seats differ in the
/// weights they arrive with rather than in anything below. The guard names the
/// pair it accepts rather than the seat, for that reason.
fn fast_t2_action(
    observation: &ActorObservation,
    model: &Model,
    cache: &mut FastOutlookCache,
) -> Result<Action, String> {
    let hero = observation.hero_board.card_count();
    let opponent = observation.opponent_public_board.card_count();
    if hero != 7 || (opponent != 7 && opponent != 9) {
        return Err(format!(
            "fast T2 evaluator requires a 7-card hero board and a 7- or 9-card \
             opponent board, got {hero} and {opponent}"
        ));
    }
    let actions = generate_turn_actions_trusted(&observation.hero_board, &observation.dealt_cards);
    if actions.is_empty() {
        return Err("T2 observation has no legal actions".to_owned());
    }
    let unknown = crate::t3_features::unknown_cards(observation);
    // Computed once and read by every candidate: the opponent's board does not
    // move when the hero places.
    let (opponent_block, opponent_finishes) =
        fast_outlook(&observation.opponent_public_board, &unknown, cache)?;
    let mut scratch = model.scratch();
    let mut values = Vec::with_capacity(actions.len());
    for action in &actions {
        let board = action.apply_trusted(&observation.hero_board);
        let features = fast_encode(
            observation,
            &board,
            &unknown,
            &opponent_block,
            &opponent_finishes,
            cache,
        )?;
        values.push(model.predict_with(&features, &mut scratch)? as f64);
    }
    let selected = canonical_descending_indices(&values, &actions)?[0];
    Ok(actions[selected].clone())
}

/// Rank the legal T1 actions with the coarse evaluator, on either seat.
///
/// The same shape as [`learned_action_over`] -- same legal set from the same
/// generator, same 168-wide vector in the same block order, same canonical
/// descending order and same first element taken -- with
/// [`crate::fast_features`] supplying the outlook instead of
/// [`crate::t3first_features`]. What changes is only how finely the two outlook
/// blocks are sampled, and the model that reads them was fitted to the coarse
/// numbers rather than the fine ones, so this is not the full evaluator run
/// cheaply; it is a different evaluator that answers the same question.
///
/// Both T1 seats share it. Acting first the opponent has not answered, so both
/// boards carry five cards; acting second the opponent's carries seven. Those
/// are eight and six open slots respectively, and the free-slot outlook covers
/// both, so the seats differ in the weights they arrive with rather than in
/// anything below. The guard names the pair it accepts rather than the seat,
/// for that reason.
fn fast_t1_action(
    observation: &ActorObservation,
    model: &Model,
    cache: &mut FastOutlookCache,
) -> Result<Action, String> {
    let hero = observation.hero_board.card_count();
    let opponent = observation.opponent_public_board.card_count();
    if hero != 5 || (opponent != 5 && opponent != 7) {
        return Err(format!(
            "fast T1 evaluator requires a 5-card hero board and a 5- or 7-card \
             opponent board, got {hero} and {opponent}"
        ));
    }
    let actions = generate_turn_actions_trusted(&observation.hero_board, &observation.dealt_cards);
    if actions.is_empty() {
        return Err("T1 observation has no legal actions".to_owned());
    }
    let unknown = crate::t3_features::unknown_cards(observation);
    // Computed once and read by every candidate: the opponent's board does not
    // move when the hero places.
    let (opponent_block, opponent_finishes) =
        fast_outlook(&observation.opponent_public_board, &unknown, cache)?;
    let mut scratch = model.scratch();
    let mut values = Vec::with_capacity(actions.len());
    for action in &actions {
        let board = action.apply_trusted(&observation.hero_board);
        let features = fast_encode(
            observation,
            &board,
            &unknown,
            &opponent_block,
            &opponent_finishes,
            cache,
        )?;
        values.push(model.predict_with(&features, &mut scratch)? as f64);
    }
    let selected = canonical_descending_indices(&values, &actions)?[0];
    Ok(actions[selected].clone())
}

/// Rank the legal T0 second-seat openings with the coarse evaluator.
///
/// The opening street's twin of [`fast_t1_action`], and separate from it for
/// the reason the street is always separate: five cards are dealt, all five are
/// placed and nothing is discarded, so the legal set is the 232 openings from
/// [`generate_initial_actions`] rather than the twenty-seven two-of-three turn
/// actions. Everything below that is the same -- same 168-wide vector in the
/// same block order, same canonical descending order, same first element taken
/// -- with [`crate::fast_features`] supplying both outlook blocks.
///
/// The width ratio is why this reply is worth coarsening more than any other.
/// A turn reply pays the outlook twenty-seven times; this one pays it 232
/// times, against boards with eight open slots rather than six or four, and a
/// T0 first-seat evaluation plays one of these at the head of every rollout.
///
/// Both boards here carry the geometry the free-slot outlook already covers:
/// the candidate board is the hero's five placed cards with eight slots open,
/// and the opponent's board is the five-card opening being answered, also with
/// eight. The guard names that pair rather than the street.
fn fast_t0_second_action(
    observation: &ActorObservation,
    model: &Model,
    cache: &mut FastOutlookCache,
) -> Result<Action, String> {
    let hero = observation.hero_board.card_count();
    let opponent = observation.opponent_public_board.card_count();
    if hero != 0 || opponent != 5 {
        return Err(format!(
            "fast T0 second-seat evaluator requires an empty hero board and a \
             5-card opponent board, got {hero} and {opponent}"
        ));
    }
    let actions = generate_initial_actions(&observation.hero_board, &observation.dealt_cards)?;
    if actions.is_empty() {
        return Err("T0 observation has no legal actions".to_owned());
    }
    let unknown = crate::t3_features::unknown_cards(observation);
    // Computed once and read by every candidate: the opponent's board does not
    // move when the hero places. Across 232 candidates that saving is larger
    // here than anywhere else in the engine.
    let (opponent_block, opponent_finishes) =
        fast_outlook(&observation.opponent_public_board, &unknown, cache)?;
    let mut scratch = model.scratch();
    let mut values = Vec::with_capacity(actions.len());
    for action in &actions {
        let board = action.apply_trusted(&observation.hero_board);
        let features = fast_encode(
            observation,
            &board,
            &unknown,
            &opponent_block,
            &opponent_finishes,
            cache,
        )?;
        values.push(model.predict_with(&features, &mut scratch)? as f64);
    }
    let selected = canonical_descending_indices(&values, &actions)?[0];
    Ok(actions[selected].clone())
}

/// Rank the legal T0 **first**-seat openings with the coarse evaluator.
///
/// The last street the engine could not encode, and the only one whose model is
/// a root policy rather than a continuation: acting first on the opening street
/// nothing has happened yet, so nothing in the engine ever has to answer this
/// decision on someone else's behalf. Only `decide` reaches here.
///
/// What separates it from [`fast_t0_second_action`] is one board. Acting second
/// the opponent's five placed cards are on the table, both boards have eight
/// open slots, and all four blocks compose. Acting first the opponent's board
/// is empty -- thirteen open -- and the free-slot outlook refuses that width by
/// name, so the two blocks that read it are zero. That is not this function
/// choosing to skip work: it is the geometry the trained image was fitted
/// under, and [`fast_encode_hidden_opponent`] states the consequences.
///
/// Everything else is the opening street as the second seat already had it:
/// 232 candidates from [`generate_initial_actions`], the same 168-wide vector
/// in the same block order, the same canonical descending order, the same first
/// element taken. The guard names the board pair rather than the street,
/// because empty-against-empty is the whole discriminator.
fn fast_t0_first_action(
    observation: &ActorObservation,
    model: &Model,
    cache: &mut FastOutlookCache,
) -> Result<Action, String> {
    let hero = observation.hero_board.card_count();
    let opponent = observation.opponent_public_board.card_count();
    if hero != 0 || opponent != 0 {
        return Err(format!(
            "T0 first-seat evaluator requires two empty boards, got {hero} and \
             {opponent}"
        ));
    }
    let actions = generate_initial_actions(&observation.hero_board, &observation.dealt_cards)?;
    if actions.is_empty() {
        return Err("T0 observation has no legal actions".to_owned());
    }
    let unknown = crate::t3_features::unknown_cards(observation);
    // No opponent block to compute once and share: there is no opponent board
    // to compute it from. The per-row sharing across the 232 candidates is the
    // whole of what the cache buys here, and it is the larger half.
    let mut scratch = model.scratch();
    let mut values = Vec::with_capacity(actions.len());
    for action in &actions {
        let board = action.apply_trusted(&observation.hero_board);
        let features = fast_encode_hidden_opponent(observation, &board, &unknown, cache)?;
        values.push(model.predict_with(&features, &mut scratch)? as f64);
    }
    let selected = canonical_descending_indices(&values, &actions)?[0];
    Ok(actions[selected].clone())
}

/// The nested second-seat reply inside a T2 first-seat rollout, cached by
/// fingerprint.
///
/// No sampled fallback exists, for the same reason as the nested T3 first-seat
/// reply: the search that would answer costs minutes per T2 decision.
fn locked_t2_second_action(
    observation: &ActorObservation,
    context: &mut SearchContext,
) -> Result<Action, String> {
    // The coarse reply when one was pinned. Nothing above this function knows
    // the difference: it returns a legal action chosen by the same canonical
    // rule, and only the encoder behind the choice is cheaper.
    if let Some(model) = context.fast_t2_second_model.as_ref() {
        let key = ObservationKey::new(observation);
        context.t3_child_observation_keys.insert(key);
        if let Some(action) = context.fast_t2_second_action_cache.get(&key) {
            return Ok(action.clone());
        }
        let selected = fast_t2_action(observation, model, &mut context.fast_t2_second_outlook)?;
        context
            .fast_t2_second_action_cache
            .insert(key, selected.clone());
        return Ok(selected);
    }
    let Some(model) = context.t2_second_model.as_ref() else {
        return Err(
            "T2 first-seat evaluation requires learned_t2_second_model; a nested \
             sampled T2 second-seat search would cost minutes per decision"
                .to_owned(),
        );
    };
    let key = ObservationKey::new(observation);
    context.t3_child_observation_keys.insert(key);
    if let Some(action) = context.t2_second_action_cache.get(&key) {
        return Ok(action.clone());
    }
    let selected = learned_t2_second_action(observation, model)?;
    context
        .t2_second_action_cache
        .insert(key, selected.clone());
    Ok(selected)
}

/// The nested first-seat reply inside a T1 second-seat rollout, cached by
/// fingerprint.
///
/// The mirror of [`locked_t2_second_action`] one seat over, and it exists for
/// the same reason: a T1 rollout must play the opponent's T2 first-seat reply,
/// and the nested sampled search that would otherwise answer costs minutes per
/// decision. So there is no fallback here either, only refusal by name.
fn locked_t2_first_action(
    observation: &ActorObservation,
    context: &mut SearchContext,
) -> Result<Action, String> {
    // The coarse reply when one was pinned; see [`locked_t2_second_action`].
    if let Some(model) = context.fast_t2_first_model.as_ref() {
        let key = ObservationKey::new(observation);
        context.t3_child_observation_keys.insert(key);
        if let Some(action) = context.fast_t2_first_action_cache.get(&key) {
            return Ok(action.clone());
        }
        let selected = fast_t2_action(observation, model, &mut context.fast_t2_first_outlook)?;
        context
            .fast_t2_first_action_cache
            .insert(key, selected.clone());
        return Ok(selected);
    }
    let Some(model) = context.t2_first_model.as_ref() else {
        return Err(
            "T1 evaluation requires learned_t2_first_model; a nested sampled T2 \
             first-seat search would cost minutes per decision"
                .to_owned(),
        );
    };
    let key = ObservationKey::new(observation);
    context.t3_child_observation_keys.insert(key);
    if let Some(action) = context.t2_first_action_cache.get(&key) {
        return Ok(action.clone());
    }
    let selected = learned_t2_first_action(observation, model)?;
    context
        .t2_first_action_cache
        .insert(key, selected.clone());
    Ok(selected)
}

/// The nested second-seat reply inside a T1 first-seat rollout, cached by
/// fingerprint.
///
/// The mirror of [`locked_t2_first_action`] one street over, and it exists for
/// the same reason: a T1 first-seat rollout opens with the opponent's T1
/// second-seat reply, and the nested sampled search that would otherwise answer
/// costs minutes per decision. So there is no fallback here either, only
/// refusal by name.
fn locked_t1_second_action(
    observation: &ActorObservation,
    context: &mut SearchContext,
) -> Result<Action, String> {
    // The coarse reply when one was pinned. Nothing above this function knows
    // the difference: it returns a legal action chosen by the same canonical
    // rule, and only the encoder behind the choice is cheaper.
    if let Some(model) = context.fast_t1_second_model.as_ref() {
        let key = ObservationKey::new(observation);
        context.t3_child_observation_keys.insert(key);
        if let Some(action) = context.fast_t1_second_action_cache.get(&key) {
            return Ok(action.clone());
        }
        let selected = fast_t1_action(observation, model, &mut context.fast_t1_second_outlook)?;
        context
            .fast_t1_second_action_cache
            .insert(key, selected.clone());
        return Ok(selected);
    }
    let Some(model) = context.t1_second_model.as_ref() else {
        return Err(
            "T1 first-seat evaluation requires learned_t1_second_model; a nested \
             sampled T1 second-seat search would cost minutes per decision"
                .to_owned(),
        );
    };
    let key = ObservationKey::new(observation);
    context.t3_child_observation_keys.insert(key);
    if let Some(action) = context.t1_second_action_cache.get(&key) {
        return Ok(action.clone());
    }
    let selected = learned_t1_second_action(observation, model)?;
    context
        .t1_second_action_cache
        .insert(key, selected.clone());
    Ok(selected)
}

/// The nested T1 first-seat reply inside a T0 rollout, cached by fingerprint.
///
/// Both T0 seats need it: acting first the rollout plays the hero's own T1
/// first-seat turn, acting second it plays the opponent's. It has no affordable
/// fallback either, so its absence is named rather than papered over.
fn locked_t1_first_action(
    observation: &ActorObservation,
    context: &mut SearchContext,
) -> Result<Action, String> {
    // The coarse reply when one was pinned; see [`locked_t1_second_action`].
    if let Some(model) = context.fast_t1_first_model.as_ref() {
        let key = ObservationKey::new(observation);
        context.t3_child_observation_keys.insert(key);
        if let Some(action) = context.fast_t1_first_action_cache.get(&key) {
            return Ok(action.clone());
        }
        let selected = fast_t1_action(observation, model, &mut context.fast_t1_first_outlook)?;
        context
            .fast_t1_first_action_cache
            .insert(key, selected.clone());
        return Ok(selected);
    }
    let Some(model) = context.t1_first_model.as_ref() else {
        return Err(
            "T0 evaluation requires learned_t1_first_model; a nested sampled T1 \
             first-seat search would cost minutes per decision"
                .to_owned(),
        );
    };
    let key = ObservationKey::new(observation);
    context.t3_child_observation_keys.insert(key);
    if let Some(action) = context.t1_first_action_cache.get(&key) {
        return Ok(action.clone());
    }
    let selected = learned_t1_first_action(observation, model)?;
    context
        .t1_first_action_cache
        .insert(key, selected.clone());
    Ok(selected)
}

/// The nested second-seat reply inside a T0 first-seat rollout, cached by
/// fingerprint.
///
/// The last of the nested deciders and the mirror of every one above it: a T0
/// first-seat rollout opens with the opponent's T0 second-seat reply, and the
/// sampled search that would otherwise answer costs minutes per decision. So
/// there is no fallback here either, only refusal by name.
fn locked_t0_second_action(
    observation: &ActorObservation,
    context: &mut SearchContext,
) -> Result<Action, String> {
    // The coarse reply when one was pinned; see [`locked_t1_second_action`].
    // Nothing above this function knows the difference: it returns a legal
    // action chosen by the same canonical rule, and only the encoder behind the
    // choice is cheaper.
    if let Some(model) = context.fast_t0_second_model.as_ref() {
        let key = ObservationKey::new(observation);
        context.t3_child_observation_keys.insert(key);
        if let Some(action) = context.fast_t0_second_action_cache.get(&key) {
            return Ok(action.clone());
        }
        let selected =
            fast_t0_second_action(observation, model, &mut context.fast_t0_second_outlook)?;
        context
            .fast_t0_second_action_cache
            .insert(key, selected.clone());
        return Ok(selected);
    }
    let Some(model) = context.t0_second_model.as_ref() else {
        return Err(
            "T0 first-seat evaluation requires learned_t0_second_model; a nested \
             sampled T0 second-seat search would cost minutes per decision"
                .to_owned(),
        );
    };
    let key = ObservationKey::new(observation);
    context.t3_child_observation_keys.insert(key);
    if let Some(action) = context.t0_second_action_cache.get(&key) {
        return Ok(action.clone());
    }
    let selected = learned_t0_second_action(observation, model)?;
    context
        .t0_second_action_cache
        .insert(key, selected.clone());
    Ok(selected)
}

/// One T2 first-seat rollout: the opponent answers T2 second, the hero answers
/// T3 first, the opponent answers T3 second, and the final street resolves
/// through the same locked T4 path every other rollout uses.
fn rollout_t2_first(
    observation: &ActorObservation,
    root_action: &Action,
    particle: &HiddenCardParticle,
    context: &mut SearchContext,
) -> Result<TerminalOutcome, String> {
    let after_root = root_action.apply_trusted(&observation.hero_board);
    let mut hero_discards = observation.hero_private_discards.clone();
    hero_discards.extend(root_action.discards.iter().copied());

    let opponent_t2_observation = ActorObservation::new(
        observation.opponent_public_board.clone(),
        after_root.clone(),
        particle.draw(3, 0)?.to_vec(),
        particle.opponent_private_discards.clone(),
        Seat::Second,
        Street::T2,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t2_action = locked_t2_second_action(&opponent_t2_observation, context)?;
    let opponent_after_t2 = opponent_t2_action.apply_trusted(&observation.opponent_public_board);
    let mut opponent_discards = particle.opponent_private_discards.clone();
    opponent_discards.extend(opponent_t2_action.discards.iter().copied());

    let hero_t3_observation = ActorObservation::new(
        after_root.clone(),
        opponent_after_t2.clone(),
        particle.draw(3, 3)?.to_vec(),
        hero_discards.clone(),
        Seat::First,
        Street::T3,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let hero_t3_action = locked_t3_first_action(&hero_t3_observation, context)?;
    let hero_after_t3 = hero_t3_action.apply_trusted(&after_root);
    hero_discards.extend(hero_t3_action.discards.iter().copied());

    let opponent_t3_observation = ActorObservation::new(
        opponent_after_t2.clone(),
        hero_after_t3.clone(),
        particle.draw(3, 6)?.to_vec(),
        opponent_discards.clone(),
        Seat::Second,
        Street::T3,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t3_action = locked_t3_second_action(&opponent_t3_observation, context)?;
    let opponent_after_t3 = opponent_t3_action.apply_trusted(&opponent_after_t2);
    opponent_discards.extend(opponent_t3_action.discards.iter().copied());

    let hero_t4_observation = ActorObservation::new(
        hero_after_t3.clone(),
        opponent_after_t3.clone(),
        particle.draw(3, 9)?.to_vec(),
        hero_discards,
        Seat::First,
        Street::T4,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let hero_t4_action = locked_t4_action(&hero_t4_observation, context)?;
    let hero_final = hero_t4_action.apply_trusted(&hero_after_t3);

    let opponent_t4_observation = ActorObservation::new(
        opponent_after_t3.clone(),
        hero_final.clone(),
        particle.draw(3, 12)?.to_vec(),
        opponent_discards,
        Seat::Second,
        Street::T4,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t4_action = locked_t4_action(&opponent_t4_observation, context)?;
    let opponent_final = opponent_t4_action.apply_trusted(&opponent_after_t3);
    terminal_outcome(&hero_final, &opponent_final, context.fl_ev_14)
}

/// One T2 second-seat rollout: the opponent answers T3 first, the hero answers
/// T3 second, and the final street resolves through the same locked T4 path
/// the T3 rollouts use.
fn rollout_t2_second(
    observation: &ActorObservation,
    root_action: &Action,
    particle: &HiddenCardParticle,
    context: &mut SearchContext,
) -> Result<TerminalOutcome, String> {
    let after_root = root_action.apply_trusted(&observation.hero_board);
    let mut hero_discards = observation.hero_private_discards.clone();
    hero_discards.extend(root_action.discards.iter().copied());

    let opponent_t3_observation = ActorObservation::new(
        observation.opponent_public_board.clone(),
        after_root.clone(),
        particle.draw(3, 0)?.to_vec(),
        particle.opponent_private_discards.clone(),
        Seat::First,
        Street::T3,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let opponent_t3_action = locked_t3_first_action(&opponent_t3_observation, context)?;
    let opponent_after_t3 = opponent_t3_action.apply_trusted(&observation.opponent_public_board);
    let mut opponent_discards = particle.opponent_private_discards.clone();
    opponent_discards.extend(opponent_t3_action.discards.iter().copied());

    let hero_t3_observation = ActorObservation::new(
        after_root.clone(),
        opponent_after_t3.clone(),
        particle.draw(3, 3)?.to_vec(),
        hero_discards.clone(),
        Seat::Second,
        Street::T3,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let hero_t3_action = locked_t3_second_action(&hero_t3_observation, context)?;
    let hero_after_t3 = hero_t3_action.apply_trusted(&after_root);
    hero_discards.extend(hero_t3_action.discards.iter().copied());

    let opponent_t4_observation = ActorObservation::new(
        opponent_after_t3.clone(),
        hero_after_t3.clone(),
        particle.draw(3, 6)?.to_vec(),
        opponent_discards.clone(),
        Seat::First,
        Street::T4,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let opponent_t4_action = locked_t4_action(&opponent_t4_observation, context)?;
    let opponent_final = opponent_t4_action.apply_trusted(&opponent_after_t3);

    let hero_t4_observation = ActorObservation::new(
        hero_after_t3.clone(),
        opponent_final.clone(),
        particle.draw(3, 9)?.to_vec(),
        hero_discards,
        Seat::Second,
        Street::T4,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let hero_t4_action = locked_t4_action(&hero_t4_observation, context)?;
    let hero_final = hero_t4_action.apply_trusted(&hero_after_t3);
    terminal_outcome(&hero_final, &opponent_final, context.fl_ev_14)
}

/// One T1 second-seat rollout: every remaining decision in the hand, played by
/// the locked evaluators.
///
/// Acting second at T1 the hero's board holds five cards and the opponent's
/// seven, because the opponent has already answered T1. From the root action
/// onward the two seats alternate for three more streets, so this rollout plays
/// six nested decisions and consumes eighteen particle cards -- the same shape
/// as [`rollout_t2_first`], one street earlier.
fn rollout_t1_second(
    observation: &ActorObservation,
    root_action: &Action,
    particle: &HiddenCardParticle,
    context: &mut SearchContext,
) -> Result<TerminalOutcome, String> {
    let after_root = root_action.apply_trusted(&observation.hero_board);
    let mut hero_discards = observation.hero_private_discards.clone();
    hero_discards.extend(root_action.discards.iter().copied());

    // The opponent answers T2 first: its board is seven cards and so is the
    // hero's after the root action, which is T2 first-seat geometry.
    let opponent_t2_observation = ActorObservation::new(
        observation.opponent_public_board.clone(),
        after_root.clone(),
        particle.draw(3, 0)?.to_vec(),
        particle.opponent_private_discards.clone(),
        Seat::First,
        Street::T2,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let opponent_t2_action = locked_t2_first_action(&opponent_t2_observation, context)?;
    let opponent_after_t2 = opponent_t2_action.apply_trusted(&observation.opponent_public_board);
    let mut opponent_discards = particle.opponent_private_discards.clone();
    opponent_discards.extend(opponent_t2_action.discards.iter().copied());

    let hero_t2_observation = ActorObservation::new(
        after_root.clone(),
        opponent_after_t2.clone(),
        particle.draw(3, 3)?.to_vec(),
        hero_discards.clone(),
        Seat::Second,
        Street::T2,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let hero_t2_action = locked_t2_second_action(&hero_t2_observation, context)?;
    let hero_after_t2 = hero_t2_action.apply_trusted(&after_root);
    hero_discards.extend(hero_t2_action.discards.iter().copied());

    let opponent_t3_observation = ActorObservation::new(
        opponent_after_t2.clone(),
        hero_after_t2.clone(),
        particle.draw(3, 6)?.to_vec(),
        opponent_discards.clone(),
        Seat::First,
        Street::T3,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let opponent_t3_action = locked_t3_first_action(&opponent_t3_observation, context)?;
    let opponent_after_t3 = opponent_t3_action.apply_trusted(&opponent_after_t2);
    opponent_discards.extend(opponent_t3_action.discards.iter().copied());

    let hero_t3_observation = ActorObservation::new(
        hero_after_t2.clone(),
        opponent_after_t3.clone(),
        particle.draw(3, 9)?.to_vec(),
        hero_discards.clone(),
        Seat::Second,
        Street::T3,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let hero_t3_action = locked_t3_second_action(&hero_t3_observation, context)?;
    let hero_after_t3 = hero_t3_action.apply_trusted(&hero_after_t2);
    hero_discards.extend(hero_t3_action.discards.iter().copied());

    let opponent_t4_observation = ActorObservation::new(
        opponent_after_t3.clone(),
        hero_after_t3.clone(),
        particle.draw(3, 12)?.to_vec(),
        opponent_discards,
        Seat::First,
        Street::T4,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let opponent_t4_action = locked_t4_action(&opponent_t4_observation, context)?;
    let opponent_final = opponent_t4_action.apply_trusted(&opponent_after_t3);

    let hero_t4_observation = ActorObservation::new(
        hero_after_t3.clone(),
        opponent_final.clone(),
        particle.draw(3, 15)?.to_vec(),
        hero_discards,
        Seat::Second,
        Street::T4,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let hero_t4_action = locked_t4_action(&hero_t4_observation, context)?;
    let hero_final = hero_t4_action.apply_trusted(&hero_after_t3);
    terminal_outcome(&hero_final, &opponent_final, context.fl_ev_14)
}

/// One T1 first-seat rollout: every decision in the hand after the hero's own,
/// played by the locked evaluators.
///
/// Acting first at T1 both boards hold five cards, because the opponent has not
/// answered yet -- so the rollout opens with that answer and then alternates for
/// three more streets. That is seven nested decisions and twenty-one particle
/// cards, one decision and one draw more than [`rollout_t1_second`]. The
/// particle carries the whole hidden deck, which at a T1 first-seat root is
/// `52 - 13` visible `- 0` opponent discards `= 39` cards, so the last draw at
/// offset eighteen is well inside it.
fn rollout_t1_first(
    observation: &ActorObservation,
    root_action: &Action,
    particle: &HiddenCardParticle,
    context: &mut SearchContext,
) -> Result<TerminalOutcome, String> {
    let after_root = root_action.apply_trusted(&observation.hero_board);
    let mut hero_discards = observation.hero_private_discards.clone();
    hero_discards.extend(root_action.discards.iter().copied());

    // The opponent answers T1 second: its board still holds five cards and the
    // hero's holds seven after the root action, which is T1 second-seat
    // geometry. Its discard list is empty here and can only be -- the decision
    // it is about to make produces the first discard of its hand.
    let opponent_t1_observation = ActorObservation::new(
        observation.opponent_public_board.clone(),
        after_root.clone(),
        particle.draw(3, 0)?.to_vec(),
        particle.opponent_private_discards.clone(),
        Seat::Second,
        Street::T1,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t1_action = locked_t1_second_action(&opponent_t1_observation, context)?;
    let opponent_after_t1 = opponent_t1_action.apply_trusted(&observation.opponent_public_board);
    let mut opponent_discards = particle.opponent_private_discards.clone();
    opponent_discards.extend(opponent_t1_action.discards.iter().copied());

    let hero_t2_observation = ActorObservation::new(
        after_root.clone(),
        opponent_after_t1.clone(),
        particle.draw(3, 3)?.to_vec(),
        hero_discards.clone(),
        Seat::First,
        Street::T2,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let hero_t2_action = locked_t2_first_action(&hero_t2_observation, context)?;
    let hero_after_t2 = hero_t2_action.apply_trusted(&after_root);
    hero_discards.extend(hero_t2_action.discards.iter().copied());

    let opponent_t2_observation = ActorObservation::new(
        opponent_after_t1.clone(),
        hero_after_t2.clone(),
        particle.draw(3, 6)?.to_vec(),
        opponent_discards.clone(),
        Seat::Second,
        Street::T2,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t2_action = locked_t2_second_action(&opponent_t2_observation, context)?;
    let opponent_after_t2 = opponent_t2_action.apply_trusted(&opponent_after_t1);
    opponent_discards.extend(opponent_t2_action.discards.iter().copied());

    let hero_t3_observation = ActorObservation::new(
        hero_after_t2.clone(),
        opponent_after_t2.clone(),
        particle.draw(3, 9)?.to_vec(),
        hero_discards.clone(),
        Seat::First,
        Street::T3,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let hero_t3_action = locked_t3_first_action(&hero_t3_observation, context)?;
    let hero_after_t3 = hero_t3_action.apply_trusted(&hero_after_t2);
    hero_discards.extend(hero_t3_action.discards.iter().copied());

    let opponent_t3_observation = ActorObservation::new(
        opponent_after_t2.clone(),
        hero_after_t3.clone(),
        particle.draw(3, 12)?.to_vec(),
        opponent_discards.clone(),
        Seat::Second,
        Street::T3,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t3_action = locked_t3_second_action(&opponent_t3_observation, context)?;
    let opponent_after_t3 = opponent_t3_action.apply_trusted(&opponent_after_t2);
    opponent_discards.extend(opponent_t3_action.discards.iter().copied());

    let hero_t4_observation = ActorObservation::new(
        hero_after_t3.clone(),
        opponent_after_t3.clone(),
        particle.draw(3, 15)?.to_vec(),
        hero_discards,
        Seat::First,
        Street::T4,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let hero_t4_action = locked_t4_action(&hero_t4_observation, context)?;
    let hero_final = hero_t4_action.apply_trusted(&hero_after_t3);

    let opponent_t4_observation = ActorObservation::new(
        opponent_after_t3.clone(),
        hero_final.clone(),
        particle.draw(3, 18)?.to_vec(),
        opponent_discards,
        Seat::Second,
        Street::T4,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t4_action = locked_t4_action(&opponent_t4_observation, context)?;
    let opponent_final = opponent_t4_action.apply_trusted(&opponent_after_t3);
    terminal_outcome(&hero_final, &opponent_final, context.fl_ev_14)
}

/// One T0 second-seat rollout: every decision in the hand, played by the locked
/// evaluators.
///
/// Acting second at T0 the hero's board is empty and the opponent's holds the
/// five cards it has already placed. The root action places five more, and from
/// there the two seats alternate for four full streets -- eight nested
/// decisions and twenty-four particle cards. Neither T0 action discards
/// anything, so both discard lists are still empty when T1 begins and the first
/// discard of the hand is the one T1 produces.
fn rollout_t0_second(
    observation: &ActorObservation,
    root_action: &Action,
    particle: &HiddenCardParticle,
    context: &mut SearchContext,
) -> Result<TerminalOutcome, String> {
    let after_root = root_action.apply_trusted(&observation.hero_board);
    let mut hero_discards = observation.hero_private_discards.clone();
    hero_discards.extend(root_action.discards.iter().copied());

    // The opponent answers T1 first: its board is five cards and so is the
    // hero's after the root action, which is T1 first-seat geometry.
    let opponent_t1_observation = ActorObservation::new(
        observation.opponent_public_board.clone(),
        after_root.clone(),
        particle.draw(3, 0)?.to_vec(),
        particle.opponent_private_discards.clone(),
        Seat::First,
        Street::T1,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let opponent_t1_action = locked_t1_first_action(&opponent_t1_observation, context)?;
    let opponent_after_t1 = opponent_t1_action.apply_trusted(&observation.opponent_public_board);
    let mut opponent_discards = particle.opponent_private_discards.clone();
    opponent_discards.extend(opponent_t1_action.discards.iter().copied());

    let hero_t1_observation = ActorObservation::new(
        after_root.clone(),
        opponent_after_t1.clone(),
        particle.draw(3, 3)?.to_vec(),
        hero_discards.clone(),
        Seat::Second,
        Street::T1,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let hero_t1_action = locked_t1_second_action(&hero_t1_observation, context)?;
    let hero_after_t1 = hero_t1_action.apply_trusted(&after_root);
    hero_discards.extend(hero_t1_action.discards.iter().copied());

    let opponent_t2_observation = ActorObservation::new(
        opponent_after_t1.clone(),
        hero_after_t1.clone(),
        particle.draw(3, 6)?.to_vec(),
        opponent_discards.clone(),
        Seat::First,
        Street::T2,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let opponent_t2_action = locked_t2_first_action(&opponent_t2_observation, context)?;
    let opponent_after_t2 = opponent_t2_action.apply_trusted(&opponent_after_t1);
    opponent_discards.extend(opponent_t2_action.discards.iter().copied());

    let hero_t2_observation = ActorObservation::new(
        hero_after_t1.clone(),
        opponent_after_t2.clone(),
        particle.draw(3, 9)?.to_vec(),
        hero_discards.clone(),
        Seat::Second,
        Street::T2,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let hero_t2_action = locked_t2_second_action(&hero_t2_observation, context)?;
    let hero_after_t2 = hero_t2_action.apply_trusted(&hero_after_t1);
    hero_discards.extend(hero_t2_action.discards.iter().copied());

    let opponent_t3_observation = ActorObservation::new(
        opponent_after_t2.clone(),
        hero_after_t2.clone(),
        particle.draw(3, 12)?.to_vec(),
        opponent_discards.clone(),
        Seat::First,
        Street::T3,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let opponent_t3_action = locked_t3_first_action(&opponent_t3_observation, context)?;
    let opponent_after_t3 = opponent_t3_action.apply_trusted(&opponent_after_t2);
    opponent_discards.extend(opponent_t3_action.discards.iter().copied());

    let hero_t3_observation = ActorObservation::new(
        hero_after_t2.clone(),
        opponent_after_t3.clone(),
        particle.draw(3, 15)?.to_vec(),
        hero_discards.clone(),
        Seat::Second,
        Street::T3,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let hero_t3_action = locked_t3_second_action(&hero_t3_observation, context)?;
    let hero_after_t3 = hero_t3_action.apply_trusted(&hero_after_t2);
    hero_discards.extend(hero_t3_action.discards.iter().copied());

    let opponent_t4_observation = ActorObservation::new(
        opponent_after_t3.clone(),
        hero_after_t3.clone(),
        particle.draw(3, 18)?.to_vec(),
        opponent_discards,
        Seat::First,
        Street::T4,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let opponent_t4_action = locked_t4_action(&opponent_t4_observation, context)?;
    let opponent_final = opponent_t4_action.apply_trusted(&opponent_after_t3);

    let hero_t4_observation = ActorObservation::new(
        hero_after_t3.clone(),
        opponent_final.clone(),
        particle.draw(3, 21)?.to_vec(),
        hero_discards,
        Seat::Second,
        Street::T4,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let hero_t4_action = locked_t4_action(&hero_t4_observation, context)?;
    let hero_final = hero_t4_action.apply_trusted(&hero_after_t3);
    terminal_outcome(&hero_final, &opponent_final, context.fl_ev_14)
}

/// One T0 first-seat rollout: the whole hand after the hero's opening, played
/// by the locked evaluators.
///
/// Acting first at T0 both boards are empty, so the rollout opens with the
/// opponent's own opening and then alternates for four more streets. That is
/// nine nested decisions and twenty-nine particle cards -- the deepest rollout
/// in the engine, and one decision and five draws more than
/// [`rollout_t0_second`], because the opening deals five where every later
/// street deals three. The particle carries the whole hidden deck, which at a
/// T0 first-seat root is `52 - 5` visible `= 47` cards, so the last draw at
/// offset twenty-six is well inside it.
fn rollout_t0_first(
    observation: &ActorObservation,
    root_action: &Action,
    particle: &HiddenCardParticle,
    context: &mut SearchContext,
) -> Result<TerminalOutcome, String> {
    let after_root = root_action.apply_trusted(&observation.hero_board);
    let mut hero_discards = observation.hero_private_discards.clone();
    hero_discards.extend(root_action.discards.iter().copied());

    // The opponent answers T0 second: its board is still empty and the hero's
    // holds five after the root action, which is T0 second-seat geometry. Both
    // discard lists are empty here and can only be -- the opening street places
    // all five dealt cards and discards none of them.
    let opponent_t0_observation = ActorObservation::new(
        observation.opponent_public_board.clone(),
        after_root.clone(),
        particle.draw(5, 0)?.to_vec(),
        particle.opponent_private_discards.clone(),
        Seat::Second,
        Street::T0,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t0_action = locked_t0_second_action(&opponent_t0_observation, context)?;
    let opponent_after_t0 = opponent_t0_action.apply_trusted(&observation.opponent_public_board);
    let mut opponent_discards = particle.opponent_private_discards.clone();
    opponent_discards.extend(opponent_t0_action.discards.iter().copied());

    let hero_t1_observation = ActorObservation::new(
        after_root.clone(),
        opponent_after_t0.clone(),
        particle.draw(3, 5)?.to_vec(),
        hero_discards.clone(),
        Seat::First,
        Street::T1,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let hero_t1_action = locked_t1_first_action(&hero_t1_observation, context)?;
    let hero_after_t1 = hero_t1_action.apply_trusted(&after_root);
    hero_discards.extend(hero_t1_action.discards.iter().copied());

    let opponent_t1_observation = ActorObservation::new(
        opponent_after_t0.clone(),
        hero_after_t1.clone(),
        particle.draw(3, 8)?.to_vec(),
        opponent_discards.clone(),
        Seat::Second,
        Street::T1,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t1_action = locked_t1_second_action(&opponent_t1_observation, context)?;
    let opponent_after_t1 = opponent_t1_action.apply_trusted(&opponent_after_t0);
    opponent_discards.extend(opponent_t1_action.discards.iter().copied());

    let hero_t2_observation = ActorObservation::new(
        hero_after_t1.clone(),
        opponent_after_t1.clone(),
        particle.draw(3, 11)?.to_vec(),
        hero_discards.clone(),
        Seat::First,
        Street::T2,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let hero_t2_action = locked_t2_first_action(&hero_t2_observation, context)?;
    let hero_after_t2 = hero_t2_action.apply_trusted(&hero_after_t1);
    hero_discards.extend(hero_t2_action.discards.iter().copied());

    let opponent_t2_observation = ActorObservation::new(
        opponent_after_t1.clone(),
        hero_after_t2.clone(),
        particle.draw(3, 14)?.to_vec(),
        opponent_discards.clone(),
        Seat::Second,
        Street::T2,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t2_action = locked_t2_second_action(&opponent_t2_observation, context)?;
    let opponent_after_t2 = opponent_t2_action.apply_trusted(&opponent_after_t1);
    opponent_discards.extend(opponent_t2_action.discards.iter().copied());

    let hero_t3_observation = ActorObservation::new(
        hero_after_t2.clone(),
        opponent_after_t2.clone(),
        particle.draw(3, 17)?.to_vec(),
        hero_discards.clone(),
        Seat::First,
        Street::T3,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let hero_t3_action = locked_t3_first_action(&hero_t3_observation, context)?;
    let hero_after_t3 = hero_t3_action.apply_trusted(&hero_after_t2);
    hero_discards.extend(hero_t3_action.discards.iter().copied());

    let opponent_t3_observation = ActorObservation::new(
        opponent_after_t2.clone(),
        hero_after_t3.clone(),
        particle.draw(3, 20)?.to_vec(),
        opponent_discards.clone(),
        Seat::Second,
        Street::T3,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t3_action = locked_t3_second_action(&opponent_t3_observation, context)?;
    let opponent_after_t3 = opponent_t3_action.apply_trusted(&opponent_after_t2);
    opponent_discards.extend(opponent_t3_action.discards.iter().copied());

    let hero_t4_observation = ActorObservation::new(
        hero_after_t3.clone(),
        opponent_after_t3.clone(),
        particle.draw(3, 23)?.to_vec(),
        hero_discards,
        Seat::First,
        Street::T4,
        ActOrder::First,
        observation.scoring.clone(),
    )?;
    let hero_t4_action = locked_t4_action(&hero_t4_observation, context)?;
    let hero_final = hero_t4_action.apply_trusted(&hero_after_t3);

    let opponent_t4_observation = ActorObservation::new(
        opponent_after_t3.clone(),
        hero_final.clone(),
        particle.draw(3, 26)?.to_vec(),
        opponent_discards,
        Seat::Second,
        Street::T4,
        ActOrder::Second,
        observation.scoring.clone(),
    )?;
    let opponent_t4_action = locked_t4_action(&opponent_t4_observation, context)?;
    let opponent_final = opponent_t4_action.apply_trusted(&opponent_after_t3);
    terminal_outcome(&hero_final, &opponent_final, context.fl_ev_14)
}

/// The mean HU score of one T0 root action over one particle set.
///
/// Scored one action at a time rather than as a batch, because the two-stage
/// schedule evaluates different actions over different particle sets and only
/// the survivors are ever seen by the second one.
fn score_t0_action(
    observation: &ActorObservation,
    action: &Action,
    particles: &[HiddenCardParticle],
    context: &mut SearchContext,
) -> Result<f64, String> {
    if particles.is_empty() {
        return Err("T0 particle batch must not be empty".to_owned());
    }
    let mut sum = 0.0;
    for particle in particles {
        let outcome = if observation.to_act_order == ActOrder::First {
            rollout_t0_first(observation, action, particle, context)?
        } else {
            rollout_t0_second(observation, action, particle, context)?
        };
        sum += outcome.hu_score;
    }
    Ok(sum / particles.len() as f64)
}

/// The same rollouts as [`score_t0_action`], kept per particle instead of
/// summed.
///
/// Racing needs the individual outcomes for two reasons the mean cannot serve.
/// The elimination rule is a PAIRED test, so it needs particle i's outcome for
/// two candidates side by side rather than two averages. And a candidate's
/// particles arrive in instalments, so the running score has to be
/// reconstructible from what it has consumed so far.
///
/// Reconstructed with [`mean_in_order`], which sums left to right exactly as
/// the loop in `score_t0_action` does. That is what makes a candidate that
/// reaches the final checkpoint carry the uniform score bit for bit rather than
/// to within rounding: same outcomes, same order, same additions.
fn score_t0_action_per_particle(
    observation: &ActorObservation,
    action: &Action,
    particles: &[HiddenCardParticle],
    context: &mut SearchContext,
) -> Result<Vec<f64>, String> {
    if particles.is_empty() {
        return Err("T0 particle batch must not be empty".to_owned());
    }
    let mut outcomes = Vec::with_capacity(particles.len());
    for particle in particles {
        let outcome = if observation.to_act_order == ActOrder::First {
            rollout_t0_first(observation, action, particle, context)?
        } else {
            rollout_t0_second(observation, action, particle, context)?
        };
        outcomes.push(outcome.hu_score);
    }
    Ok(outcomes)
}

/// The mean of `values`, summed in the order they are given.
///
/// Deliberately the naive left-to-right sum rather than a compensated one: the
/// claim being preserved is equality with `score_t0_action`, which sums this
/// way, and a more accurate mean here would break that equality in exactly the
/// cases where accuracy mattered.
fn mean_in_order(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    for value in values {
        sum += *value;
    }
    sum / values.len() as f64
}

/// Whether `leader`'s advantage over `challenger` is decisive at `z` standard
/// errors, on the particles they have both consumed.
///
/// Both slices are the per-particle outcomes in particle order, so element i of
/// each is the same particle and `leader[i] - challenger[i]` is a paired
/// difference: whatever that particle did to both candidates cancels. Pairing
/// is the whole reason this can decide anything on thirty-two particles -- the
/// unpaired difference of two T0 means carries the full spread of the rollout,
/// which is enormous next to the gap between two good openings.
///
/// One-sided, and it eliminates only on evidence: the mean advantage has to
/// exceed `z` standard errors of its own paired distribution. Anything the
/// particles have not separated -- a tie, a lead inside the noise, a difference
/// too few particles to have a variance at all -- is carried forward.
///
/// Zero variance eliminates nothing, which is worth being explicit about
/// because the arithmetic says otherwise: with a standard error of zero,
/// `mean > z * se` is `mean > 0` for every finite z, and a threshold set to
/// disable elimination would not disable it. A checkpoint at which the paired
/// difference never varied is not evidence that it never would -- it is a small
/// sample that happened not to move, and the first checkpoint is where samples
/// are smallest and that happens most. The cost of being wrong in one direction
/// is a lost optimum and in the other is particles, so this spends the
/// particles.
fn race_elimination_is_decisive(leader: &[f64], challenger: &[f64], z: f64) -> bool {
    debug_assert_eq!(leader.len(), challenger.len());
    let count = leader.len();
    // A variance needs two observations. With fewer, nothing is decisive.
    if count < 2 {
        return false;
    }
    let differences = leader
        .iter()
        .zip(challenger.iter())
        .map(|(lead, chase)| lead - chase)
        .collect::<Vec<_>>();
    let mean = mean_in_order(&differences);
    if !(mean > 0.0) {
        return false;
    }
    let mut sum_squares = 0.0;
    for difference in &differences {
        let centred = difference - mean;
        sum_squares += centred * centred;
    }
    let variance = sum_squares / (count - 1) as f64;
    let standard_error = (variance / count as f64).sqrt();
    if !standard_error.is_finite() || standard_error <= 0.0 {
        return false;
    }
    mean > z * standard_error
}

fn score_t1_actions(
    observation: &ActorObservation,
    actions: &[Action],
    particles: &[HiddenCardParticle],
    context: &mut SearchContext,
) -> Result<Vec<f64>, String> {
    let mut values = Vec::with_capacity(actions.len());
    for action in actions {
        let mut sum = 0.0;
        for particle in particles {
            let outcome = if observation.to_act_order == ActOrder::First {
                rollout_t1_first(observation, action, particle, context)?
            } else {
                rollout_t1_second(observation, action, particle, context)?
            };
            sum += outcome.hu_score;
        }
        values.push(sum / particles.len() as f64);
    }
    Ok(values)
}

fn score_t2_actions(
    observation: &ActorObservation,
    actions: &[Action],
    particles: &[HiddenCardParticle],
    context: &mut SearchContext,
) -> Result<Vec<f64>, String> {
    let mut values = Vec::with_capacity(actions.len());
    for action in actions {
        let mut sum = 0.0;
        for particle in particles {
            let outcome = if observation.to_act_order == ActOrder::First {
                rollout_t2_first(observation, action, particle, context)?
            } else {
                rollout_t2_second(observation, action, particle, context)?
            };
            sum += outcome.hu_score;
        }
        values.push(sum / particles.len() as f64);
    }
    Ok(values)
}

/// Teacher-only T2 evaluator.
///
/// Exists only with every learned evaluator its seat needs pinned, and says so
/// in its output: there is no exact or sampled fallback for the nested replies,
/// because those searches cost minutes per T2 decision. The second seat needs
/// the three downstream evaluators; the first seat additionally plays the
/// opponent's T2 second-seat reply and so needs a fourth. Everything else
/// mirrors [`evaluate_t3`] -- disjoint candidate and evaluation particle
/// batches, canonical ordering, provenance in the continuation policy.
pub fn evaluate_t2(observation: &ActorObservation, config: &T3Config) -> Result<Value, String> {
    observation.validate()?;
    if observation.street != Street::T2 {
        return Err("T2 evaluation requires a T2 ActorObservation".to_owned());
    }
    if config.candidate_samples == 0 || config.evaluation_samples == 0 {
        return Err("T2 candidate/evaluation samples must be positive".to_owned());
    }
    if config.run_id.is_empty() {
        return Err("T2 run_id must not be empty".to_owned());
    }
    let candidate_batch = sample_hidden_card_particles(
        observation,
        config.candidate_seed,
        &format!("{}:candidate_selection", config.run_id),
        config.candidate_samples,
        0,
    )?;
    let evaluation_batch = sample_hidden_card_particles(
        observation,
        config.evaluation_seed,
        &format!("{}:locked_evaluation", config.run_id),
        config.evaluation_samples,
        0,
    )?;
    let candidate_keys = candidate_batch
        .particles
        .iter()
        .map(|particle| particle.rng_key_digest.as_str())
        .collect::<HashSet<_>>();
    if evaluation_batch
        .particles
        .iter()
        .any(|particle| candidate_keys.contains(particle.rng_key_digest.as_str()))
    {
        return Err("candidate-selection and evaluation particle RNG keys overlap".to_owned());
    }
    let actions = generate_turn_actions(&observation.hero_board, &observation.dealt_cards)?;
    if actions.is_empty() {
        return Err("T2 observation has no legal actions".to_owned());
    }
    let actions = narrow_by_learned_model(observation, config, actions)?;
    let mut context = SearchContext {
        config: config.clone(),
        fl_ev_14: fl_ev_14(observation)?,
        t4_model: load_learned_t4_model(config)?,
        t3_second_model: load_learned_model(
            config.learned_t3_second_model_path.as_deref(),
            config.learned_t3_second_model_sha256.as_deref(),
            "learned_t3_second_model",
            crate::t3_features::FEATURE_SIZE,
        )?,
        t3_first_model: load_learned_model(
            config.learned_t3_first_model_path.as_deref(),
            config.learned_t3_first_model_sha256.as_deref(),
            "learned_t3_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t2_second_model: load_learned_model(
            config.learned_t2_second_model_path.as_deref(),
            config.learned_t2_second_model_sha256.as_deref(),
            "learned_t2_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t2_first_model: load_learned_model(
            config.learned_t2_first_model_path.as_deref(),
            config.learned_t2_first_model_sha256.as_deref(),
            "learned_t2_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t1_second_model: load_learned_model(
            config.learned_t1_second_model_path.as_deref(),
            config.learned_t1_second_model_sha256.as_deref(),
            "learned_t1_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t1_first_model: load_learned_model(
            config.learned_t1_first_model_path.as_deref(),
            config.learned_t1_first_model_sha256.as_deref(),
            "learned_t1_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t0_second_model: load_learned_model(
            config.learned_t0_second_model_path.as_deref(),
            config.learned_t0_second_model_sha256.as_deref(),
            "learned_t0_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t0_first_model: load_learned_model(
            config.learned_t0_first_model_path.as_deref(),
            config.learned_t0_first_model_sha256.as_deref(),
            "learned_t0_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t2_second_model: load_learned_model(
            config.fast_t2_second_model_path.as_deref(),
            config.fast_t2_second_model_sha256.as_deref(),
            "fast_t2_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t2_first_model: load_learned_model(
            config.fast_t2_first_model_path.as_deref(),
            config.fast_t2_first_model_sha256.as_deref(),
            "fast_t2_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t1_second_model: load_learned_model(
            config.fast_t1_second_model_path.as_deref(),
            config.fast_t1_second_model_sha256.as_deref(),
            "fast_t1_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t1_first_model: load_learned_model(
            config.fast_t1_first_model_path.as_deref(),
            config.fast_t1_first_model_sha256.as_deref(),
            "fast_t1_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t0_second_model: load_learned_model(
            config.fast_t0_second_model_path.as_deref(),
            config.fast_t0_second_model_sha256.as_deref(),
            "fast_t0_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t4_action_cache: HashMap::new(),
        t3_second_action_cache: HashMap::new(),
        t3_first_action_cache: HashMap::new(),
        t2_second_action_cache: HashMap::new(),
        t2_first_action_cache: HashMap::new(),
        t1_second_action_cache: HashMap::new(),
        t1_first_action_cache: HashMap::new(),
        t0_second_action_cache: HashMap::new(),
        fast_t2_second_action_cache: HashMap::new(),
        fast_t2_first_action_cache: HashMap::new(),
        fast_t1_second_action_cache: HashMap::new(),
        fast_t1_first_action_cache: HashMap::new(),
        fast_t0_second_action_cache: HashMap::new(),
        fast_t2_second_outlook: FastOutlookCache::new(),
        fast_t2_first_outlook: FastOutlookCache::new(),
        fast_t1_second_outlook: FastOutlookCache::new(),
        fast_t1_first_outlook: FastOutlookCache::new(),
        fast_t0_second_outlook: FastOutlookCache::new(),
        t3_child_observation_keys: HashSet::new(),
        t4_child_observation_keys: HashSet::new(),
    };
    let mut required = vec![
        ("learned_t4_model", context.t4_model.is_some()),
        ("learned_t3_second_model", context.t3_second_model.is_some()),
        ("learned_t3_first_model", context.t3_first_model.is_some()),
    ];
    // Acting first there is one more nested reply to play: the opponent's own
    // T2 second-seat decision, which has no affordable fallback either.
    if observation.to_act_order == ActOrder::First {
        required.push(("learned_t2_second_model", context.t2_second_model.is_some()));
    }
    for (field, present) in required {
        if !present {
            return Err(format!(
                "T2 evaluation requires {field}; this street's teacher exists \
                 only with every learned evaluator pinned"
            ));
        }
    }

    let candidate_values =
        score_t2_actions(observation, &actions, &candidate_batch.particles, &mut context)?;
    let evaluation_values =
        score_t2_actions(observation, &actions, &evaluation_batch.particles, &mut context)?;
    let ranked = canonical_descending_indices(&candidate_values, &actions)?;
    let selected = ranked[0];
    let evaluation_order = canonical_descending_indices(&evaluation_values, &actions)?;
    let evaluation_best = evaluation_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut sorted_position = vec![0usize; actions.len()];
    for (position, &index) in evaluation_order.iter().enumerate() {
        sorted_position[index] = position;
    }
    let mut rows: Vec<Value> = Vec::with_capacity(actions.len());
    for (index, action) in actions.iter().enumerate() {
        rows.push(json!({
            "action_key": action_key(action)?.to_token(),
            "placements": action
                .placements
                .iter()
                .map(|(card, row)| json!([card, row]))
                .collect::<Vec<_>>(),
            "discards": action.discards,
            "score": evaluation_values[index],
            "candidate_score": candidate_values[index],
            "original_index": index,
            "sorted_index": sorted_position[index],
        }));
    }

    Ok(json!({
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "kind": "t2",
        "schema": "hu_m3_t2_result_v1",
        "street": "T2",
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "observation_fingerprint": observation.fingerprint(),
        "legal_action_count": actions.len(),
        "legal_action_set_digest": legal_action_set_digest(&actions)?,
        "selected_action_key": action_key(&actions[selected])?.to_token(),
        "selected_action_original_index": selected,
        "evaluation_sample_best_score": evaluation_best,
        "evaluation_sample_regret_of_locked_selection":
            evaluation_best - evaluation_values[selected],
        "sample_independence": "disjoint_particle_rng_keys",
        "continuation_policy": continuation_policy_report(config, &context),
        "child_information_set_count": context.t3_child_observation_keys.len()
            + context.t4_child_observation_keys.len(),
        "actions": rows,
        "teacher_value_status": "diagnostic_not_match_EV",
    }))
}

/// Teacher-only T1 evaluator.
///
/// The same contract as [`evaluate_t2`] one street earlier, and the same
/// reason for it: every nested reply below a T1 root is answered by a pinned
/// evaluator, because the searches that would otherwise answer them cost
/// minutes per decision. Acting second the rollout plays all five learned
/// evaluators -- the opponent's T2 first-seat reply on top of the four the T2
/// first seat already needs -- so all five must be pinned and refusal names
/// whichever is missing. The first seat plays the opponent's T1 second-seat
/// reply on top of those five and so needs a sixth.
pub fn evaluate_t1(observation: &ActorObservation, config: &T3Config) -> Result<Value, String> {
    observation.validate()?;
    if observation.street != Street::T1 {
        return Err("T1 evaluation requires a T1 ActorObservation".to_owned());
    }
    if config.candidate_samples == 0 || config.evaluation_samples == 0 {
        return Err("T1 candidate/evaluation samples must be positive".to_owned());
    }
    if config.run_id.is_empty() {
        return Err("T1 run_id must not be empty".to_owned());
    }
    let candidate_batch = sample_hidden_card_particles(
        observation,
        config.candidate_seed,
        &format!("{}:candidate_selection", config.run_id),
        config.candidate_samples,
        0,
    )?;
    let evaluation_batch = sample_hidden_card_particles(
        observation,
        config.evaluation_seed,
        &format!("{}:locked_evaluation", config.run_id),
        config.evaluation_samples,
        0,
    )?;
    let candidate_keys = candidate_batch
        .particles
        .iter()
        .map(|particle| particle.rng_key_digest.as_str())
        .collect::<HashSet<_>>();
    if evaluation_batch
        .particles
        .iter()
        .any(|particle| candidate_keys.contains(particle.rng_key_digest.as_str()))
    {
        return Err("candidate-selection and evaluation particle RNG keys overlap".to_owned());
    }
    let actions = generate_turn_actions(&observation.hero_board, &observation.dealt_cards)?;
    if actions.is_empty() {
        return Err("T1 observation has no legal actions".to_owned());
    }
    let actions = narrow_by_learned_model(observation, config, actions)?;
    let mut context = SearchContext {
        config: config.clone(),
        fl_ev_14: fl_ev_14(observation)?,
        t4_model: load_learned_t4_model(config)?,
        t3_second_model: load_learned_model(
            config.learned_t3_second_model_path.as_deref(),
            config.learned_t3_second_model_sha256.as_deref(),
            "learned_t3_second_model",
            crate::t3_features::FEATURE_SIZE,
        )?,
        t3_first_model: load_learned_model(
            config.learned_t3_first_model_path.as_deref(),
            config.learned_t3_first_model_sha256.as_deref(),
            "learned_t3_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t2_second_model: load_learned_model(
            config.learned_t2_second_model_path.as_deref(),
            config.learned_t2_second_model_sha256.as_deref(),
            "learned_t2_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t2_first_model: load_learned_model(
            config.learned_t2_first_model_path.as_deref(),
            config.learned_t2_first_model_sha256.as_deref(),
            "learned_t2_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t1_second_model: load_learned_model(
            config.learned_t1_second_model_path.as_deref(),
            config.learned_t1_second_model_sha256.as_deref(),
            "learned_t1_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t1_first_model: load_learned_model(
            config.learned_t1_first_model_path.as_deref(),
            config.learned_t1_first_model_sha256.as_deref(),
            "learned_t1_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t0_second_model: load_learned_model(
            config.learned_t0_second_model_path.as_deref(),
            config.learned_t0_second_model_sha256.as_deref(),
            "learned_t0_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t0_first_model: load_learned_model(
            config.learned_t0_first_model_path.as_deref(),
            config.learned_t0_first_model_sha256.as_deref(),
            "learned_t0_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t2_second_model: load_learned_model(
            config.fast_t2_second_model_path.as_deref(),
            config.fast_t2_second_model_sha256.as_deref(),
            "fast_t2_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t2_first_model: load_learned_model(
            config.fast_t2_first_model_path.as_deref(),
            config.fast_t2_first_model_sha256.as_deref(),
            "fast_t2_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t1_second_model: load_learned_model(
            config.fast_t1_second_model_path.as_deref(),
            config.fast_t1_second_model_sha256.as_deref(),
            "fast_t1_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t1_first_model: load_learned_model(
            config.fast_t1_first_model_path.as_deref(),
            config.fast_t1_first_model_sha256.as_deref(),
            "fast_t1_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t0_second_model: load_learned_model(
            config.fast_t0_second_model_path.as_deref(),
            config.fast_t0_second_model_sha256.as_deref(),
            "fast_t0_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t4_action_cache: HashMap::new(),
        t3_second_action_cache: HashMap::new(),
        t3_first_action_cache: HashMap::new(),
        t2_second_action_cache: HashMap::new(),
        t2_first_action_cache: HashMap::new(),
        t1_second_action_cache: HashMap::new(),
        t1_first_action_cache: HashMap::new(),
        t0_second_action_cache: HashMap::new(),
        fast_t2_second_action_cache: HashMap::new(),
        fast_t2_first_action_cache: HashMap::new(),
        fast_t1_second_action_cache: HashMap::new(),
        fast_t1_first_action_cache: HashMap::new(),
        fast_t0_second_action_cache: HashMap::new(),
        fast_t2_second_outlook: FastOutlookCache::new(),
        fast_t2_first_outlook: FastOutlookCache::new(),
        fast_t1_second_outlook: FastOutlookCache::new(),
        fast_t1_first_outlook: FastOutlookCache::new(),
        fast_t0_second_outlook: FastOutlookCache::new(),
        t3_child_observation_keys: HashSet::new(),
        t4_child_observation_keys: HashSet::new(),
    };
    // Acting second every one of the five is on the path: the opponent's T2
    // first-seat reply, the hero's own T2 second-seat reply, both T3 seats, and
    // the final street. None of them has an affordable fallback.
    let mut required = vec![
        ("learned_t4_model", context.t4_model.is_some()),
        ("learned_t3_second_model", context.t3_second_model.is_some()),
        ("learned_t3_first_model", context.t3_first_model.is_some()),
        ("learned_t2_second_model", context.t2_second_model.is_some()),
        ("learned_t2_first_model", context.t2_first_model.is_some()),
    ];
    // Acting first there is a sixth: the opponent's T1 second-seat reply, which
    // opens every rollout and which the second seat never has to play because
    // by then the opponent's T1 turn is already behind it.
    if observation.to_act_order == ActOrder::First {
        required.push(("learned_t1_second_model", context.t1_second_model.is_some()));
    }
    for (field, present) in required {
        if !present {
            return Err(format!(
                "T1 evaluation requires {field}; this street's teacher exists \
                 only with every learned evaluator pinned"
            ));
        }
    }

    let candidate_values =
        score_t1_actions(observation, &actions, &candidate_batch.particles, &mut context)?;
    let evaluation_values =
        score_t1_actions(observation, &actions, &evaluation_batch.particles, &mut context)?;
    let ranked = canonical_descending_indices(&candidate_values, &actions)?;
    let selected = ranked[0];
    let evaluation_order = canonical_descending_indices(&evaluation_values, &actions)?;
    let evaluation_best = evaluation_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut sorted_position = vec![0usize; actions.len()];
    for (position, &index) in evaluation_order.iter().enumerate() {
        sorted_position[index] = position;
    }
    let mut rows: Vec<Value> = Vec::with_capacity(actions.len());
    for (index, action) in actions.iter().enumerate() {
        rows.push(json!({
            "action_key": action_key(action)?.to_token(),
            "placements": action
                .placements
                .iter()
                .map(|(card, row)| json!([card, row]))
                .collect::<Vec<_>>(),
            "discards": action.discards,
            "score": evaluation_values[index],
            "candidate_score": candidate_values[index],
            "original_index": index,
            "sorted_index": sorted_position[index],
        }));
    }

    Ok(json!({
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "kind": "t1",
        "schema": "hu_m3_t1_result_v1",
        "street": "T1",
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "observation_fingerprint": observation.fingerprint(),
        "legal_action_count": actions.len(),
        "legal_action_set_digest": legal_action_set_digest(&actions)?,
        "selected_action_key": action_key(&actions[selected])?.to_token(),
        "selected_action_original_index": selected,
        "evaluation_sample_best_score": evaluation_best,
        "evaluation_sample_regret_of_locked_selection":
            evaluation_best - evaluation_values[selected],
        "sample_independence": "disjoint_particle_rng_keys",
        "continuation_policy": continuation_policy_report(config, &context),
        "child_information_set_count": context.t3_child_observation_keys.len()
            + context.t4_child_observation_keys.len(),
        "actions": rows,
        "teacher_value_status": "diagnostic_not_match_EV",
    }))
}

/// Load the evaluator set a T0 root reaches, with every cache empty.
///
/// Built through a function rather than inline because the standing audit runs
/// its own full pass and has to run it on its own context. Sharing one would
/// not change a score -- every cache here is keyed by the observation it
/// answers, so a hit and a miss return the same action -- but the context also
/// accumulates the child information sets the result reports, and those would
/// then depend on whether the position happened to be audited. A number in the
/// emitted result that moves because a diagnostic ran is not a diagnostic.
fn build_t0_context(
    observation: &ActorObservation,
    config: &T3Config,
) -> Result<SearchContext, String> {
    Ok(SearchContext {
        config: config.clone(),
        fl_ev_14: fl_ev_14(observation)?,
        t4_model: load_learned_t4_model(config)?,
        t3_second_model: load_learned_model(
            config.learned_t3_second_model_path.as_deref(),
            config.learned_t3_second_model_sha256.as_deref(),
            "learned_t3_second_model",
            crate::t3_features::FEATURE_SIZE,
        )?,
        t3_first_model: load_learned_model(
            config.learned_t3_first_model_path.as_deref(),
            config.learned_t3_first_model_sha256.as_deref(),
            "learned_t3_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t2_second_model: load_learned_model(
            config.learned_t2_second_model_path.as_deref(),
            config.learned_t2_second_model_sha256.as_deref(),
            "learned_t2_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t2_first_model: load_learned_model(
            config.learned_t2_first_model_path.as_deref(),
            config.learned_t2_first_model_sha256.as_deref(),
            "learned_t2_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t1_second_model: load_learned_model(
            config.learned_t1_second_model_path.as_deref(),
            config.learned_t1_second_model_sha256.as_deref(),
            "learned_t1_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t1_first_model: load_learned_model(
            config.learned_t1_first_model_path.as_deref(),
            config.learned_t1_first_model_sha256.as_deref(),
            "learned_t1_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t0_second_model: load_learned_model(
            config.learned_t0_second_model_path.as_deref(),
            config.learned_t0_second_model_sha256.as_deref(),
            "learned_t0_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t0_first_model: load_learned_model(
            config.learned_t0_first_model_path.as_deref(),
            config.learned_t0_first_model_sha256.as_deref(),
            "learned_t0_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t2_second_model: load_learned_model(
            config.fast_t2_second_model_path.as_deref(),
            config.fast_t2_second_model_sha256.as_deref(),
            "fast_t2_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t2_first_model: load_learned_model(
            config.fast_t2_first_model_path.as_deref(),
            config.fast_t2_first_model_sha256.as_deref(),
            "fast_t2_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t1_second_model: load_learned_model(
            config.fast_t1_second_model_path.as_deref(),
            config.fast_t1_second_model_sha256.as_deref(),
            "fast_t1_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t1_first_model: load_learned_model(
            config.fast_t1_first_model_path.as_deref(),
            config.fast_t1_first_model_sha256.as_deref(),
            "fast_t1_first_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        fast_t0_second_model: load_learned_model(
            config.fast_t0_second_model_path.as_deref(),
            config.fast_t0_second_model_sha256.as_deref(),
            "fast_t0_second_model",
            crate::t3first_features::FEATURE_SIZE,
        )?,
        t4_action_cache: HashMap::new(),
        t3_second_action_cache: HashMap::new(),
        t3_first_action_cache: HashMap::new(),
        t2_second_action_cache: HashMap::new(),
        t2_first_action_cache: HashMap::new(),
        t1_second_action_cache: HashMap::new(),
        t1_first_action_cache: HashMap::new(),
        t0_second_action_cache: HashMap::new(),
        fast_t2_second_action_cache: HashMap::new(),
        fast_t2_first_action_cache: HashMap::new(),
        fast_t1_second_action_cache: HashMap::new(),
        fast_t1_first_action_cache: HashMap::new(),
        fast_t0_second_action_cache: HashMap::new(),
        fast_t2_second_outlook: FastOutlookCache::new(),
        fast_t2_first_outlook: FastOutlookCache::new(),
        fast_t1_second_outlook: FastOutlookCache::new(),
        fast_t1_first_outlook: FastOutlookCache::new(),
        fast_t0_second_outlook: FastOutlookCache::new(),
        t3_child_observation_keys: HashSet::new(),
        t4_child_observation_keys: HashSet::new(),
    })
}

/// A deterministic 64-bit reading of an observation fingerprint.
///
/// The fingerprint is already the SHA-256 of the canonical observation, so
/// there is nothing left to mix: its leading 64 bits are as uniform as anything
/// further hashing could produce, and reading them directly means anyone
/// holding a position file can recompute which positions a run should have
/// audited without the engine.
fn observation_audit_hash(fingerprint: &str) -> Result<u64, String> {
    let head = fingerprint.get(..16).ok_or_else(|| {
        format!("observation fingerprint is too short to sample an audit cadence from: {fingerprint}")
    })?;
    u64::from_str_radix(head, 16)
        .map_err(|error| format!("observation fingerprint is not hexadecimal: {error}"))
}

/// The audit's seed, derived from the run rather than chosen by it.
///
/// A configurable audit seed would be one more field a plan could set to the
/// evaluation seed by accident, and an audit drawn from the staged evaluation's
/// own particles measures the schedule against itself. Deriving it from the
/// evaluation seed, the run id and the observation keeps it reproducible
/// without being settable, and gives two positions of one run two different
/// audit draws.
fn derived_audit_seed(config: &T3Config, fingerprint: &str) -> Result<i64, String> {
    let digest = sha256_hex_json(&json!({
        "domain": "hu_m3_t0_standing_audit_seed_v1",
        "evaluation_seed": config.evaluation_seed,
        "run_id": config.run_id,
        "observation_fingerprint": fingerprint,
    }));
    let head = digest
        .get(..16)
        .ok_or_else(|| "audit seed digest is too short".to_owned())?;
    let value = u64::from_str_radix(head, 16)
        .map_err(|error| format!("audit seed digest is not hexadecimal: {error}"))?;
    // One bit dropped rather than a cast that would wrap half the range into
    // the negatives. A negative seed is legal everywhere else in this engine,
    // but a derivation whose sign depends on how the cast is spelled is not a
    // derivation.
    Ok((value >> 1) as i64)
}

/// Teacher-only T0 evaluator, the last street of the backward curriculum.
///
/// The same contract as [`evaluate_t1`] one street earlier -- every nested
/// reply answered by a pinned evaluator, refusal by name when one is missing --
/// with one addition the earlier streets did not need. T0 offers 232 legal
/// openings where T1 offers tens, so paying the full particle set on every one
/// of them is what makes the naive cost prohibitive. When `prefilter_samples`
/// and `prefilter_keep` are both nonzero the root is therefore scored in two
/// stages: cheaply over every action, then fully over the best `prefilter_keep`
/// of them. Every legal action still gets a row either way; a row says which
/// stage produced its score, because a stage-1 score over one particle and a
/// stage-2 score over the full set are not the same measurement and must not be
/// compared as though they were.
///
/// Both fields zero runs single-stage like every other street. That is the path
/// the pruning validation compares the schedule against, so it has to remain
/// available rather than become the schedule's degenerate case.
///
/// Two further fields make the schedule safer to run on distributions it was
/// not validated against, and both are off unless asked for. `prefilter_margin`
/// widens the keep boundary wherever stage one cannot separate the actions
/// across it. `audit_full_every` makes every Nth position pay for the
/// single-stage answer as well and reports the comparison, so that a fleet
/// carries its own running measurement of what pruning cost it instead of an
/// argument that it probably cost nothing. With both at their defaults this
/// function emits exactly the bytes it emitted before either existed.
pub fn evaluate_t0(observation: &ActorObservation, config: &T3Config) -> Result<Value, String> {
    observation.validate()?;
    if observation.street != Street::T0 {
        return Err("T0 evaluation requires a T0 ActorObservation".to_owned());
    }
    if config.candidate_samples == 0 || config.evaluation_samples == 0 {
        return Err("T0 candidate/evaluation samples must be positive".to_owned());
    }
    if config.run_id.is_empty() {
        return Err("T0 run_id must not be empty".to_owned());
    }
    // Half a schedule is not a schedule: one field without the other would
    // silently run single-stage while the caller believed it had pruned.
    if (config.prefilter_samples == 0) != (config.prefilter_keep == 0) {
        return Err(
            "prefilter_samples and prefilter_keep must be set together; leave \
             both zero for the single-stage path"
                .to_owned(),
        );
    }
    let two_stage = config.prefilter_samples > 0 && config.prefilter_keep > 0;
    // A margin is a score difference, so a negative or non-finite one is not a
    // narrower rule -- it is a rule with no meaning, and NaN in particular would
    // compare false against every gap and quietly disable the widening it was
    // set to enable.
    if !config.prefilter_margin.is_finite() || config.prefilter_margin < 0.0 {
        return Err(
            "prefilter_margin must be a finite non-negative score difference; \
             leave it at zero for the fixed-rank keep boundary"
                .to_owned(),
        );
    }
    // The same reasoning as half a schedule: a margin with no boundary to widen
    // reads as pruning safety and does nothing.
    if config.prefilter_margin > 0.0 && !two_stage {
        return Err(
            "prefilter_margin widens the two-stage prefilter's keep boundary \
             and has no boundary to widen on the single-stage path; set \
             prefilter_samples and prefilter_keep, or leave the margin at zero"
                .to_owned(),
        );
    }

    // The racing schedule, on the same terms as everything else optional here:
    // an empty one is the uniform stage two, and every rejection below is a
    // schedule that would have looked like it raced while measuring something
    // other than what it claimed.
    let racing = !config.race_schedule.is_empty();
    if racing {
        if config.race_schedule[0] == 0 {
            return Err(
                "race_schedule checkpoints are cumulative particle counts and \
                 must be positive"
                    .to_owned(),
            );
        }
        // A checkpoint that did not advance spends no particles and would
        // eliminate twice on identical evidence; one that went backwards has no
        // reading at all.
        if config
            .race_schedule
            .windows(2)
            .any(|pair| pair[1] <= pair[0])
        {
            return Err(
                "race_schedule must be strictly increasing cumulative particle \
                 counts"
                    .to_owned(),
            );
        }
        // The selected action has to have been measured at the resolution the
        // label claims, and a schedule ending short of the batch would mean it
        // was not.
        let last = *config
            .race_schedule
            .last()
            .expect("the schedule is non-empty");
        if last != config.evaluation_samples {
            return Err(format!(
                "the last race_schedule checkpoint must equal \
                 evaluation_samples ({}); the schedule ends at {last}",
                config.evaluation_samples
            ));
        }
    }
    // A z is a count of standard errors, so a negative or non-finite one is not
    // a narrower rule but a rule with no meaning -- and NaN in particular would
    // compare false against every deficit and quietly eliminate nothing.
    if !config.race_lcb_z.is_finite() || config.race_lcb_z < 0.0 {
        return Err(
            "race_lcb_z must be a finite non-negative number of standard \
             errors; leave it at zero for the uniform stage two"
                .to_owned(),
        );
    }
    // The same reasoning as a margin with no boundary to widen: a z with no
    // race to threshold reads as a tuned elimination rule and does nothing.
    if config.race_lcb_z > 0.0 && !racing {
        return Err(
            "race_lcb_z is the elimination threshold for the stage-two race \
             and has no race to threshold; set race_schedule, or leave the z \
             at zero"
                .to_owned(),
        );
    }

    // Stage one draws its own particles from its own domain, exactly as
    // candidate selection and locked evaluation already draw from theirs. No
    // new RNG kind is involved -- only a third run_id domain, which is what
    // makes the two sets disjoint.
    let selection_stream = if two_stage {
        "prefilter"
    } else {
        "candidate_selection"
    };
    let selection_count = if two_stage {
        config.prefilter_samples
    } else {
        config.candidate_samples
    };
    let selection_batch = sample_hidden_card_particles(
        observation,
        config.candidate_seed,
        &format!("{}:{selection_stream}", config.run_id),
        selection_count,
        0,
    )?;
    let evaluation_batch = sample_hidden_card_particles(
        observation,
        config.evaluation_seed,
        &format!("{}:locked_evaluation", config.run_id),
        config.evaluation_samples,
        0,
    )?;
    let selection_keys = selection_batch
        .particles
        .iter()
        .map(|particle| particle.rng_key_digest.as_str())
        .collect::<HashSet<_>>();
    if evaluation_batch
        .particles
        .iter()
        .any(|particle| selection_keys.contains(particle.rng_key_digest.as_str()))
    {
        return Err("candidate-selection and evaluation particle RNG keys overlap".to_owned());
    }

    let actions = generate_initial_actions(&observation.hero_board, &observation.dealt_cards)?;
    if actions.is_empty() {
        return Err("T0 observation has no legal actions".to_owned());
    }
    let mut context = build_t0_context(observation, config)?;
    // Acting second every street below the root is on the path once: both T1
    // seats, both T2 seats, both T3 seats, and the final street.
    let mut required = vec![
        ("learned_t4_model", context.t4_model.is_some()),
        ("learned_t3_second_model", context.t3_second_model.is_some()),
        ("learned_t3_first_model", context.t3_first_model.is_some()),
        ("learned_t2_second_model", context.t2_second_model.is_some()),
        ("learned_t2_first_model", context.t2_first_model.is_some()),
        ("learned_t1_second_model", context.t1_second_model.is_some()),
        ("learned_t1_first_model", context.t1_first_model.is_some()),
    ];
    // Acting first there is an eighth: the opponent's T0 second-seat reply,
    // which opens every rollout and which the second seat never has to play
    // because by then the opponent's opening is already behind it.
    if observation.to_act_order == ActOrder::First {
        required.push(("learned_t0_second_model", context.t0_second_model.is_some()));
    }
    for (field, present) in required {
        if !present {
            return Err(format!(
                "T0 evaluation requires {field}; this street's teacher exists \
                 only with every learned evaluator pinned"
            ));
        }
    }

    // Stage one: every legal action, over the selection particle set.
    let staged_began = std::time::Instant::now();
    let mut selection_values = Vec::with_capacity(actions.len());
    for action in &actions {
        selection_values.push(score_t0_action(
            observation,
            action,
            &selection_batch.particles,
            &mut context,
        )?);
    }
    let selection_order = canonical_descending_indices(&selection_values, &actions)?;

    // Stage two: the survivors, over the full evaluation particle set. Without
    // the schedule every action survives, which is the single-stage path.
    let fixed_keep = if two_stage {
        config.prefilter_keep.min(actions.len())
    } else {
        actions.len()
    };
    // The adaptive beam. A keep boundary drawn at a fixed rank is a claim that
    // stage one could order the two actions it falls between, and over
    // `prefilter_samples` particles it frequently could not. While the gap from
    // the last kept action down to the next one is SMALLER than the margin, the
    // two are inside stage one's own noise of each other and the next one is
    // kept as well.
    //
    // The comparison is strict: a gap of exactly the margin does not extend.
    // That is what makes a margin of zero mean "disabled" by arithmetic as well
    // as by the branch above it -- with `<` no gap can ever be under zero.
    //
    // Each step re-anchors on the action it just admitted, so a chain of
    // actions each within the margin of its predecessor is admitted as a chain.
    // Deliberately: stage one did not order any of them, and keeping the first
    // while dropping the rest would be an ordering claim it never made. The cap
    // of twice `prefilter_keep` is what bounds the chain, so a position where
    // everything ties costs twice the schedule rather than the whole root fan.
    //
    // Determinism is inherited rather than argued: `selection_order` is the
    // canonical descending order of the stage-one scores, tie-broken by action
    // key, and this reads only that order and those scores.
    let survivor_count = if two_stage && config.prefilter_margin > 0.0 {
        let cap = config.prefilter_keep.saturating_mul(2).min(actions.len());
        let mut extended = fixed_keep;
        while extended >= 1
            && extended < cap
            && selection_values[selection_order[extended - 1]]
                - selection_values[selection_order[extended]]
                < config.prefilter_margin
        {
            extended += 1;
        }
        extended
    } else {
        fixed_keep
    };
    let survivors = &selection_order[..survivor_count];
    let mut evaluation_values = selection_values.clone();
    let mut stage_two = vec![false; actions.len()];
    // Particles each stage-two entrant actually consumed, and how far it got.
    // Zero and `None` for everything the prefilter never promoted, which is
    // what keeps these reportable per row without claiming the non-entrants
    // were in a race they never entered.
    let mut race_particles = vec![0usize; actions.len()];
    let mut race_eliminated_at = vec![None::<usize>; actions.len()];
    // Survivors of each checkpoint, in schedule order. Empty when not racing.
    let mut checkpoint_survivors: Vec<usize> = Vec::new();
    // The candidates that reached the last checkpoint. Without racing that is
    // every survivor, which is the uniform path.
    let finalists: Vec<usize>;

    if racing {
        // Per-particle outcomes for each survivor, indexed by its position in
        // `survivors` rather than by action index, so the paired arithmetic
        // below reads two dense vectors instead of chasing the action fan.
        let mut outcomes: Vec<Vec<f64>> = vec![Vec::new(); survivor_count];
        // Positions into `survivors`, in stage-one rank order. Filtering
        // preserves that order, so the schedule inherits its determinism from
        // the same canonical ranking the keep boundary used.
        let mut alive: Vec<usize> = (0..survivor_count).collect();
        let mut consumed = 0usize;
        let last_checkpoint = config.race_schedule.len() - 1;
        for (checkpoint_index, &checkpoint) in config.race_schedule.iter().enumerate() {
            // Draw every live candidate up to this checkpoint. Only the new
            // particles are rolled out -- `consumed..checkpoint` -- which is
            // the whole saving, and the reason a survivor's score is an
            // accumulation rather than a re-measurement.
            for &slot in &alive {
                let index = survivors[slot];
                let drawn = score_t0_action_per_particle(
                    observation,
                    &actions[index],
                    &evaluation_batch.particles[consumed..checkpoint],
                    &mut context,
                )?;
                outcomes[slot].extend_from_slice(&drawn);
            }
            consumed = checkpoint;

            if checkpoint_index == last_checkpoint {
                checkpoint_survivors.push(alive.len());
                break;
            }

            // The leader by mean over what everyone has consumed, tie-broken by
            // the same canonical action ordering the rest of this function
            // uses. A tie decided any other way would make the eliminations
            // depend on the order the actions happened to be generated in.
            let live_values = alive
                .iter()
                .map(|&slot| mean_in_order(&outcomes[slot]))
                .collect::<Vec<_>>();
            let live_actions = alive
                .iter()
                .map(|&slot| actions[survivors[slot]].clone())
                .collect::<Vec<_>>();
            let leader_rank = canonical_descending_indices(&live_values, &live_actions)?[0];
            let leader = alive[leader_rank];

            let mut carried = Vec::with_capacity(alive.len());
            for &slot in &alive {
                // The leader cannot be eliminated by itself, and is the one
                // candidate guaranteed to reach the final checkpoint. Without
                // this the field could in principle empty.
                if slot == leader
                    || !race_elimination_is_decisive(
                        &outcomes[leader],
                        &outcomes[slot],
                        config.race_lcb_z,
                    )
                {
                    carried.push(slot);
                    continue;
                }
                race_eliminated_at[survivors[slot]] = Some(checkpoint_index);
            }
            alive = carried;
            checkpoint_survivors.push(alive.len());
        }

        for slot in 0..survivor_count {
            let index = survivors[slot];
            stage_two[index] = true;
            race_particles[index] = outcomes[slot].len();
            // An eliminated candidate carries the partial mean it was
            // eliminated on, not its stage-one score: it is a stage-two
            // measurement, just a shorter one, and the row says how short.
            evaluation_values[index] = mean_in_order(&outcomes[slot]);
        }
        finalists = alive.iter().map(|&slot| survivors[slot]).collect();
    } else {
        for &index in survivors {
            evaluation_values[index] = score_t0_action(
                observation,
                &actions[index],
                &evaluation_batch.particles,
                &mut context,
            )?;
            stage_two[index] = true;
        }
        finalists = survivors.to_vec();
    }

    // The selection is the best FINALIST by its stage-two score, which is the
    // only score measured over the full particle set. Without racing every
    // survivor is a finalist and this is the rule it always was; with racing it
    // is what keeps a candidate that looked good on thirty-two particles from
    // being selected on thirty-two particles.
    let finalist_actions = finalists
        .iter()
        .map(|&index| actions[index].clone())
        .collect::<Vec<_>>();
    let finalist_values = finalists
        .iter()
        .map(|&index| evaluation_values[index])
        .collect::<Vec<_>>();
    let selected = finalists[canonical_descending_indices(&finalist_values, &finalist_actions)?[0]];
    let evaluation_best = finalist_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let staged_seconds = staged_began.elapsed().as_secs_f64();

    // The standing audit. Everything the emitted result reports about the
    // staged answer is settled above this line, which is what makes the claim
    // that the audit cannot perturb it structural rather than a matter of care:
    // there is nothing left for it to perturb.
    let fingerprint = observation.fingerprint();
    let audited = config.audit_full_every > 0
        && observation_audit_hash(&fingerprint)? % config.audit_full_every == 0;
    let audit = if audited {
        // Its own seed and its own run_id domain, so the audit's particles are
        // a second opinion rather than a second look at the first one.
        //
        // Worth being precise about what that costs the reading. Scored on the
        // staged evaluation's own particles, the survivors' audit scores would
        // be their staged scores exactly, and the regret below would isolate
        // pruning loss and nothing else. Scored on an independent draw it also
        // carries the difference between two samples, so a single position's
        // regret is an OUT-OF-SAMPLE charge against the staged pick rather than
        // a pure measure of what the prefilter discarded. That is the stricter
        // of the two questions -- an action that looked best only because of
        // the particles that chose it is penalised here and would not be
        // otherwise -- and over a fleet's worth of audited positions the
        // sampling half averages out while a real pruning loss does not.
        // `full_best_survived_stage1` is the statistic with no such caveat.
        let audit_seed = derived_audit_seed(config, &fingerprint)?;
        let audit_stream = format!("{}:audit_full_evaluation", config.run_id);
        let audit_batch = sample_hidden_card_particles(
            observation,
            audit_seed,
            &audit_stream,
            config.evaluation_samples,
            0,
        )?;
        // Checked rather than assumed, exactly as the two staged batches check
        // each other: a derivation that happened to collide would make the
        // audit agree with the schedule for a reason that has nothing to do
        // with the schedule being right.
        let staged_keys = selection_batch
            .particles
            .iter()
            .chain(evaluation_batch.particles.iter())
            .map(|particle| particle.rng_key_digest.as_str())
            .collect::<HashSet<_>>();
        if audit_batch
            .particles
            .iter()
            .any(|particle| staged_keys.contains(particle.rng_key_digest.as_str()))
        {
            return Err("audit and staged particle RNG keys overlap".to_owned());
        }
        // Its own context, so the audit's rollouts cannot reach the counts the
        // staged result reports. See `build_t0_context`.
        let mut audit_context = build_t0_context(observation, config)?;
        let audit_began = std::time::Instant::now();
        let mut full_values = Vec::with_capacity(actions.len());
        for action in &actions {
            full_values.push(score_t0_action(
                observation,
                action,
                &audit_batch.particles,
                &mut audit_context,
            )?);
        }
        let audit_seconds = audit_began.elapsed().as_secs_f64();
        let full_pick = canonical_descending_indices(&full_values, &actions)?[0];
        // The audit is the uncompromised reference and stays one: every action
        // above was scored over the whole audit batch by the same
        // `score_t0_action` the single-stage path uses, with no schedule
        // consulted and no candidate dropped early. Racing is a property of the
        // staged answer being audited, not of the audit -- an audit that raced
        // would be marking its own homework.
        //
        // What racing does add is a second thing worth reporting: the prefilter
        // can throw the best action away, and now so can the race, and those
        // are different failures with different fixes. `survived_stage1` is
        // still the prefilter's; `survived_race` is the race's.
        let mut report = json!({
            "audited": true,
            "audit_seed": audit_seed,
            "audit_stream": audit_stream,
            "evaluation_samples": config.evaluation_samples,
            "staged_pick": action_key(&actions[selected])?.to_token(),
            "full_pick": action_key(&actions[full_pick])?.to_token(),
            "agree": full_pick == selected,
            // The narrower of the two failures, and the one the schedule is
            // actually responsible for: agreement can fail because stage two
            // ranked the survivors differently, but survival failing means the
            // prefilter threw the answer away before stage two ever saw it.
            "full_best_survived_stage1": survivors.contains(&full_pick),
            "regret_of_staged_pick_under_full": full_values[full_pick] - full_values[selected],
            "full_best_score": full_values[full_pick],
            "staged_pick_full_score": full_values[selected],
            "staged_seconds": staged_seconds,
            "audit_seconds": audit_seconds,
        });
        if racing {
            report
                .as_object_mut()
                .expect("json! built an object")
                .insert(
                    "full_best_survived_race".to_owned(),
                    json!(finalists.contains(&full_pick)),
                );
        }
        Some(report)
    } else {
        None
    };

    let mut sorted_position = vec![0usize; actions.len()];
    for (position, &index) in selection_order.iter().enumerate() {
        sorted_position[index] = position;
    }
    let mut rows: Vec<Value> = Vec::with_capacity(actions.len());
    for (index, action) in actions.iter().enumerate() {
        let mut row = json!({
            "action_key": action_key(action)?.to_token(),
            "placements": action
                .placements
                .iter()
                .map(|(card, row)| json!([card, row]))
                .collect::<Vec<_>>(),
            "discards": action.discards,
            "score": evaluation_values[index],
            "candidate_score": selection_values[index],
            "original_index": index,
            "sorted_index": sorted_position[index],
            "stage": if stage_two[index] { 2 } else { 1 },
        });
        // Written only under a race, so a row from a uniform run is the row it
        // has always been. Under one, `stage` alone stops being enough to read
        // a score by: two stage-two rows can now be measurements of different
        // lengths, and a reader comparing them without knowing that would be
        // comparing thirty-two particles against two hundred and fifty-six.
        if racing && stage_two[index] {
            let map = row.as_object_mut().expect("json! built an object");
            map.insert("race_particles".to_owned(), json!(race_particles[index]));
            map.insert(
                "race_full_samples".to_owned(),
                json!(race_eliminated_at[index].is_none()),
            );
            map.insert(
                "race_eliminated_at_checkpoint".to_owned(),
                match race_eliminated_at[index] {
                    Some(checkpoint) => json!(checkpoint),
                    None => Value::Null,
                },
            );
        }
        rows.push(row);
    }

    let mut result = json!({
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "kind": "t0",
        "schema": "hu_m3_t0_result_v1",
        "street": "T0",
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "observation_fingerprint": fingerprint,
        "legal_action_count": actions.len(),
        "legal_action_set_digest": legal_action_set_digest(&actions)?,
        "selected_action_key": action_key(&actions[selected])?.to_token(),
        "selected_action_original_index": selected,
        "evaluation_sample_best_score": evaluation_best,
        "evaluation_sample_regret_of_locked_selection":
            evaluation_best - evaluation_values[selected],
        "sample_independence": "disjoint_particle_rng_keys",
        "root_schedule": if two_stage { "two_stage_prefilter" } else { "single_stage" },
        "prefilter_samples": config.prefilter_samples,
        "prefilter_keep": config.prefilter_keep,
        "stage_two_action_count": survivor_count,
        "continuation_policy": continuation_policy_report(config, &context),
        "child_information_set_count": context.t3_child_observation_keys.len()
            + context.t4_child_observation_keys.len(),
        "actions": rows,
        "teacher_value_status": "diagnostic_not_match_EV",
    });
    // Written only when asked for, so a run that took neither mechanism emits
    // the object it emitted before either existed, key for key and byte for
    // byte. `stage_two_action_count` above already exceeds `prefilter_keep`
    // whenever the beam widened; the margin is named here so a reader can tell
    // a widened boundary from a differently configured one.
    {
        let map = result.as_object_mut().expect("json! built an object");
        if config.prefilter_margin > 0.0 {
            map.insert(
                "prefilter_margin".to_owned(),
                json!(config.prefilter_margin),
            );
        }
        // The cadence is reported whenever it is running, including on the
        // positions it did not select. Otherwise an unaudited position under a
        // live audit would be indistinguishable from one produced with the
        // audit switched off, and "no audit key" would mean two different
        // things.
        if config.audit_full_every > 0 {
            map.insert(
                "audit_full_every".to_owned(),
                json!(config.audit_full_every),
            );
        }
        // The race, reported so that a label can be audited later by someone
        // holding only the result. The schedule and the z say what rule was
        // applied; the survivor counts say what it did, which is the part no
        // reader could reconstruct -- two runs under the same schedule can
        // eliminate wildly different numbers, and a position where the race
        // eliminated nothing is a position where it cost nothing and saved
        // nothing. The two particle totals are the cost side of the same
        // ledger: what stage two actually spent, against what a uniform stage
        // two over the same entrants would have.
        if racing {
            map.insert("race_schedule".to_owned(), json!(config.race_schedule));
            map.insert("race_lcb_z".to_owned(), json!(config.race_lcb_z));
            map.insert("race_entrants".to_owned(), json!(survivor_count));
            map.insert(
                "race_checkpoint_survivors".to_owned(),
                json!(checkpoint_survivors),
            );
            map.insert(
                "race_particle_evaluations".to_owned(),
                json!(race_particles.iter().sum::<usize>()),
            );
            map.insert(
                "race_uniform_particle_evaluations".to_owned(),
                json!(survivor_count * config.evaluation_samples),
            );
        }
        if let Some(audit) = audit {
            map.insert("audit".to_owned(), audit);
        }
    }
    Ok(result)
}

/// One in-play decision through the learned evaluators.
///
/// This is the assembled policy's decision function: the final street is
/// solved exactly because that costs single-digit milliseconds at the table,
/// and every earlier supported street answers through its pinned model. It
/// exists for whole-game benchmarking; refusal names the missing model.
/// The learned evaluator's score for every legal action at this decision.
///
/// The ranking behind `decide`, published instead of thrown away. `decide`
/// computes exactly these numbers and returns only the argmax; a caller that
/// wants to narrow a candidate fan before spending particles on it needs the
/// rest of them.
///
/// Why this rather than a sampled prefilter: the scores are deterministic, so
/// the cut carries no seed dependence of its own, and they come from a model
/// fitted on joint-exact labels rather than from one or two particles.
///
/// T4 is absent on purpose. It decides by exact enumeration, so there is no
/// learned ranking to publish and nothing there needs narrowing.
pub fn model_scores(
    observation: &ActorObservation,
    config: &T3Config,
) -> Result<Value, String> {
    observation.validate()?;
    let require = |path: &Option<String>, sha: &Option<String>, field: &str, dim: usize| {
        load_learned_model(path.as_deref(), sha.as_deref(), field, dim)?
            .ok_or_else(|| format!("model_scores requires {field} for this street"))
    };
    let (actions, values) = match (observation.street, observation.to_act_order) {
        (Street::T4, _) => {
            return Err(
                "model_scores does not cover T4: it decides by exact \
                 enumeration and has no learned ranking"
                    .to_owned(),
            )
        }
        (Street::T0, ActOrder::First) => {
            let model = require(
                &config.learned_t0_first_model_path,
                &config.learned_t0_first_model_sha256,
                "learned_t0_first_model",
                crate::t3first_features::FEATURE_SIZE,
            )?;
            let actions =
                generate_initial_actions(&observation.hero_board, &observation.dealt_cards)?;
            let unknown = crate::t3_features::unknown_cards(observation);
            let mut cache = FastOutlookCache::new();
            let mut scratch = model.scratch();
            let mut values = Vec::with_capacity(actions.len());
            for action in &actions {
                let board = action.apply_trusted(&observation.hero_board);
                let features =
                    fast_encode_hidden_opponent(observation, &board, &unknown, &mut cache)?;
                values.push(model.predict_with(&features, &mut scratch)? as f64);
            }
            (actions, values)
        }
        (street, order) => {
            let (field, path, sha) = match (street, order) {
                (Street::T3, ActOrder::Second) => (
                    "learned_t3_second_model",
                    &config.learned_t3_second_model_path,
                    &config.learned_t3_second_model_sha256,
                ),
                (Street::T3, ActOrder::First) => (
                    "learned_t3_first_model",
                    &config.learned_t3_first_model_path,
                    &config.learned_t3_first_model_sha256,
                ),
                (Street::T2, ActOrder::Second) => (
                    "learned_t2_second_model",
                    &config.learned_t2_second_model_path,
                    &config.learned_t2_second_model_sha256,
                ),
                (Street::T2, ActOrder::First) => (
                    "learned_t2_first_model",
                    &config.learned_t2_first_model_path,
                    &config.learned_t2_first_model_sha256,
                ),
                (Street::T1, ActOrder::Second) => (
                    "learned_t1_second_model",
                    &config.learned_t1_second_model_path,
                    &config.learned_t1_second_model_sha256,
                ),
                (Street::T1, ActOrder::First) => (
                    "learned_t1_first_model",
                    &config.learned_t1_first_model_path,
                    &config.learned_t1_first_model_sha256,
                ),
                (Street::T0, ActOrder::Second) => (
                    "learned_t0_second_model",
                    &config.learned_t0_second_model_path,
                    &config.learned_t0_second_model_sha256,
                ),
                _ => unreachable!("T4 and T0 first seat handled above"),
            };
            let model = require(path, sha, field, crate::t3first_features::FEATURE_SIZE)?;
            let actions =
                generate_turn_actions_trusted(&observation.hero_board, &observation.dealt_cards);
            if actions.is_empty() {
                return Err(format!("{field}: observation has no legal actions"));
            }
            let values = learned_values_over(observation, &model, &actions)?;
            (actions, values)
        }
    };

    // Canonical descending order, the same rule `decide` takes its first
    // element from, so `rows[0]` here is that decision by construction.
    let order = canonical_descending_indices(&values, &actions)?;
    let rows: Vec<Value> = order
        .iter()
        .map(|&index| -> Result<Value, String> {
            Ok(json!({
                "action_key": action_key(&actions[index])?.to_token(),
                "score": values[index],
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(json!({
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "kind": "model_scores",
        "street": observation.street,
        "to_act_order": observation.to_act_order,
        "evaluator": "learned",
        "actions": rows,
    }))
}

pub fn decide(observation: &ActorObservation, config: &T3Config) -> Result<Value, String> {
    observation.validate()?;
    let require = |path: &Option<String>, sha: &Option<String>, field: &str, dim: usize| {
        load_learned_model(path.as_deref(), sha.as_deref(), field, dim)?
            .ok_or_else(|| format!("decide requires {field} for this street"))
    };
    let (action, evaluator) = match (observation.street, observation.to_act_order) {
        (Street::T4, _) => {
            let t4_config = T4Config {
                candidate_samples: 0,
                evaluation_samples: 0,
                seed: config.seed,
                candidate_seed: config.seed,
                evaluation_seed: config.seed,
                run_id: format!("{}:decide-t4", config.run_id),
            };
            (
                select_t4_action_without_result(observation, &t4_config)?,
                "exact_enumeration",
            )
        }
        (Street::T3, ActOrder::Second) => {
            let model = require(
                &config.learned_t3_second_model_path,
                &config.learned_t3_second_model_sha256,
                "learned_t3_second_model",
                crate::t3_features::FEATURE_SIZE,
            )?;
            (learned_t3_second_action(observation, &model)?, "learned")
        }
        (Street::T3, ActOrder::First) => {
            let model = require(
                &config.learned_t3_first_model_path,
                &config.learned_t3_first_model_sha256,
                "learned_t3_first_model",
                crate::t3first_features::FEATURE_SIZE,
            )?;
            (learned_t3_first_action(observation, &model)?, "learned")
        }
        (Street::T2, ActOrder::Second) => {
            let model = require(
                &config.learned_t2_second_model_path,
                &config.learned_t2_second_model_sha256,
                "learned_t2_second_model",
                crate::t3first_features::FEATURE_SIZE,
            )?;
            (learned_t2_second_action(observation, &model)?, "learned")
        }
        (Street::T2, ActOrder::First) => {
            let model = require(
                &config.learned_t2_first_model_path,
                &config.learned_t2_first_model_sha256,
                "learned_t2_first_model",
                crate::t3first_features::FEATURE_SIZE,
            )?;
            (learned_t2_first_action(observation, &model)?, "learned")
        }
        (Street::T1, ActOrder::Second) => {
            let model = require(
                &config.learned_t1_second_model_path,
                &config.learned_t1_second_model_sha256,
                "learned_t1_second_model",
                crate::t3first_features::FEATURE_SIZE,
            )?;
            (learned_t1_second_action(observation, &model)?, "learned")
        }
        (Street::T1, ActOrder::First) => {
            let model = require(
                &config.learned_t1_first_model_path,
                &config.learned_t1_first_model_sha256,
                "learned_t1_first_model",
                crate::t3first_features::FEATURE_SIZE,
            )?;
            (learned_t1_first_action(observation, &model)?, "learned")
        }
        (Street::T0, ActOrder::Second) => {
            let model = require(
                &config.learned_t0_second_model_path,
                &config.learned_t0_second_model_sha256,
                "learned_t0_second_model",
                crate::t3first_features::FEATURE_SIZE,
            )?;
            (learned_t0_second_action(observation, &model)?, "learned")
        }
        (Street::T0, ActOrder::First) => {
            let model = require(
                &config.learned_t0_first_model_path,
                &config.learned_t0_first_model_sha256,
                "learned_t0_first_model",
                crate::t3first_features::FEATURE_SIZE,
            )?;
            // A fresh cache per decision rather than one held across calls:
            // `decide` builds no search context, and the sharing that matters
            // is within the 232-candidate fan, which one call is.
            let mut cache = FastOutlookCache::new();
            (
                fast_t0_first_action(observation, &model, &mut cache)?,
                "learned",
            )
        }
        // No catch-all any more: with the opening street's first seat added
        // this match covers every decision point a hand has, and the compiler
        // says so. The arm that used to sit here told the caller that earlier
        // streets still answered through the production policy. None do.
    };
    Ok(json!({
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "kind": "decide",
        "street": observation.street,
        "to_act_order": observation.to_act_order,
        "evaluator": evaluator,
        "action_key": action_key(&action)?.to_token(),
        "placements": action
            .placements
            .iter()
            .map(|(card, row)| json!([card, row]))
            .collect::<Vec<_>>(),
        "discards": action.discards,
    }))
}

/// Score two completed boards exactly as the rollouts do.
///
/// Reached from the raw request path rather than through `EngineRequest`: a
/// finished 13-card board is not a decision point, so the observation
/// machinery rightly refuses to represent it.
pub fn score_completed_boards(
    hero: &crate::state::Board,
    opponent: &crate::state::Board,
    scoring: &crate::infoset::ScoringContext,
) -> Result<Value, String> {
    if hero.card_count() != 13 || opponent.card_count() != 13 {
        return Err("score_final requires two complete 13-card boards".to_owned());
    }
    scoring.validate()?;
    let fl_ev = scoring
        .fl_ev
        .get(&14)
        .copied()
        .ok_or_else(|| "scoring context must define FL EV for 14 cards".to_owned())?;
    let outcome = terminal_outcome(hero, opponent, fl_ev)?;
    Ok(json!({
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "kind": "score_final",
        "hu_score": outcome.hu_score,
    }))
}

/// Describe the continuation policy the run actually used.
///
/// With no learned evaluator this is the object it has always been, key for
/// key, so results and anything pinned to their digests are unaffected. When
/// the learned leaf is active the identifier changes and the weights are named,
/// because a result that does not say which evaluator produced it cannot be
/// compared against one that used the other.
fn continuation_policy_report(config: &T3Config, context: &SearchContext) -> Value {
    let mut report = json!({
        "id": "local_infoset_response_t3_second_t4_v1",
        "downstream_t3_samples": config.downstream_t3_samples,
        "downstream_t4_samples": config.downstream_t4_samples,
        "strategy_fusion_guard": "child_actions_keyed_only_by_actor_observation",
    });
    if context.t4_model.is_some() || context.t3_second_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert(
            "id".to_owned(),
            json!("local_infoset_response_t3_second_t4_learned_v1"),
        );
    }
    if context.t4_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t4_first_evaluator".to_owned(), json!("learned"));
        map.insert("t4_second_evaluator".to_owned(), json!("exact_closed_form"));
        map.insert(
            "t4_first_model_sha256".to_owned(),
            json!(config.learned_t4_model_sha256),
        );
    }
    if context.t3_second_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t3_second_evaluator".to_owned(), json!("learned"));
        map.insert(
            "t3_second_model_sha256".to_owned(),
            json!(config.learned_t3_second_model_sha256),
        );
    }
    if context.t3_first_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t3_first_evaluator".to_owned(), json!("learned"));
        map.insert(
            "t3_first_model_sha256".to_owned(),
            json!(config.learned_t3_first_model_sha256),
        );
    }
    if context.t2_second_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t2_second_evaluator".to_owned(), json!("learned"));
        map.insert(
            "t2_second_model_sha256".to_owned(),
            json!(config.learned_t2_second_model_sha256),
        );
    }
    if context.t2_first_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t2_first_evaluator".to_owned(), json!("learned"));
        map.insert(
            "t2_first_model_sha256".to_owned(),
            json!(config.learned_t2_first_model_sha256),
        );
    }
    if context.t1_second_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t1_second_evaluator".to_owned(), json!("learned"));
        map.insert(
            "t1_second_model_sha256".to_owned(),
            json!(config.learned_t1_second_model_sha256),
        );
    }
    if context.t1_first_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t1_first_evaluator".to_owned(), json!("learned"));
        map.insert(
            "t1_first_model_sha256".to_owned(),
            json!(config.learned_t1_first_model_sha256),
        );
    }
    // Written after the full-precision pair rather than beside it, because a
    // run with both pinned reaches its T2 replies through the coarse one and
    // the report should say what the search did rather than what it loaded.
    //
    // Unlike the T0 second-seat key below, these two say nothing about which
    // seat the run itself took: every rollout that starts at T2 first or
    // earlier still has both T2 replies ahead of it, so both keys are
    // meaningful on every kind of run that reaches a rollout at all.
    if context.fast_t2_second_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t2_second_evaluator".to_owned(), json!("learned_fast"));
        map.insert(
            "fast_t2_second_model_sha256".to_owned(),
            json!(config.fast_t2_second_model_sha256),
        );
    }
    if context.fast_t2_first_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t2_first_evaluator".to_owned(), json!("learned_fast"));
        map.insert(
            "fast_t2_first_model_sha256".to_owned(),
            json!(config.fast_t2_first_model_sha256),
        );
    }
    // Written after the full-precision pair rather than beside it, because a
    // run with both pinned reaches its T1 replies through the coarse one and
    // the report should say what the search did rather than what it loaded.
    if context.fast_t1_second_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t1_second_evaluator".to_owned(), json!("learned_fast"));
        map.insert(
            "fast_t1_second_model_sha256".to_owned(),
            json!(config.fast_t1_second_model_sha256),
        );
    }
    if context.fast_t1_first_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t1_first_evaluator".to_owned(), json!("learned_fast"));
        map.insert(
            "fast_t1_first_model_sha256".to_owned(),
            json!(config.fast_t1_first_model_sha256),
        );
    }
    if context.t0_second_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t0_second_evaluator".to_owned(), json!("learned"));
        map.insert(
            "t0_second_model_sha256".to_owned(),
            json!(config.learned_t0_second_model_sha256),
        );
    }
    // Written after the full-precision pin rather than beside it, for the same
    // reason as the coarse T1 pair above: a run with both pinned reaches the
    // reply through the coarse one, and the report should say what the search
    // did rather than what it loaded.
    //
    // The reply itself is reachable only from a T0 first-seat evaluation --
    // acting second the opening being answered is already on the board -- so
    // this key appearing on any other kind of run means the pin was carried
    // somewhere it does nothing, which is worth being able to see.
    if context.fast_t0_second_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t0_second_evaluator".to_owned(), json!("learned_fast"));
        map.insert(
            "fast_t0_second_model_sha256".to_owned(),
            json!(config.fast_t0_second_model_sha256),
        );
    }
    // The T0 first-seat root policy. Unlike every key above it, this one never
    // describes work the search did: no rollout answers a T0 first-seat
    // decision, because it is the first decision of the hand. It is here so
    // that a run which pinned the root policy says which weights it pinned --
    // the same reason the coarse T0 key is worth reading on a kind that cannot
    // reach it.
    if context.t0_first_model.is_some() {
        let map = report.as_object_mut().expect("json! built an object");
        map.insert("t0_first_evaluator".to_owned(), json!("learned"));
        map.insert(
            "t0_first_model_sha256".to_owned(),
            json!(config.learned_t0_first_model_sha256),
        );
    }
    report
}

/// Load the learned first-seat evaluator, or nothing if none was configured.
///
/// A configured path without a digest is refused rather than trusted. Weights
/// that quietly changed would still produce numbers, and a search that is
/// subtly evaluating a different function is far harder to notice than one that
/// failed to start.
fn load_learned_t4_model(config: &T3Config) -> Result<Option<Model>, String> {
    load_learned_model(
        config.learned_t4_model_path.as_deref(),
        config.learned_t4_model_sha256.as_deref(),
        "learned_t4_model",
        crate::t4_features::FEATURE_SIZE,
    )
}

/// Load one pinned evaluator, or nothing if none was configured.
fn load_learned_model(
    path: Option<&str>,
    expected: Option<&str>,
    field: &str,
    feature_size: usize,
) -> Result<Option<Model>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    let Some(expected) = expected else {
        return Err(format!(
            "{field}_path requires {field}_sha256 so the weights in use are \
             identified rather than assumed"
        ));
    };
    let bytes = std::fs::read(path)
        .map_err(|error| format!("cannot read {field} at {path}: {error}"))?;
    let model = Model::load_pinned(&bytes, expected)?;
    if model.input_dim() != feature_size {
        return Err(format!(
            "{field} expects {} features but its encoder produces {feature_size}",
            model.input_dim()
        ));
    }
    Ok(Some(model))
}

/// Rank the legal T3 second-seat actions with the learned evaluator.
///
/// The opponent block does not depend on the hero's action, so it is computed
/// once and shared. Ordering goes through the same canonical tie-break as the
/// sampled path so equal values resolve identically in both modes.
fn learned_t3_second_action(
    observation: &ActorObservation,
    model: &Model,
) -> Result<Action, String> {
    // The encoder completes both boards on the assumption that each has exactly
    // two slots left, which is what makes the assignment forced and the
    // enumeration exact. Anything else would silently describe a different
    // position.
    if observation.hero_board.card_count() != 9
        || observation.opponent_public_board.card_count() != 11
    {
        return Err(
            "learned T3 second-seat evaluator requires a 9-card hero board and an \
             11-card opponent board"
                .to_owned(),
        );
    }
    let actions = generate_turn_actions_trusted(&observation.hero_board, &observation.dealt_cards);
    if actions.is_empty() {
        return Err("T3 observation has no legal actions".to_owned());
    }
    let unknown = crate::t3_features::unknown_cards(observation);
    let (opponent_outlook, opponent_finishes) =
        crate::t3_features::side_outlook(&observation.opponent_public_board, &unknown);
    let mut scratch = model.scratch();
    let mut values = Vec::with_capacity(actions.len());
    for action in &actions {
        let board = action.apply_trusted(&observation.hero_board);
        let features = crate::t3_features::encode(
            observation,
            &board,
            &unknown,
            &opponent_outlook,
            &opponent_finishes,
        );
        values.push(model.predict_with(&features, &mut scratch)? as f64);
    }
    let selected = canonical_descending_indices(&values, &actions)?[0];
    Ok(actions[selected].clone())
}

/// Rank the legal first-seat T4 actions with the learned evaluator.
///
/// The opponent block does not depend on which action the hero takes, so it is
/// computed once and shared, which is what makes this cheaper than the solve it
/// stands in for. Ordering goes through the same canonical tie-break as the
/// exact path so that equal values resolve identically in both modes.
fn learned_t4_first_action(
    observation: &ActorObservation,
    model: &Model,
) -> Result<Action, String> {
    // Same precondition as the exact path. The outlook completes the opponent's
    // board on the assumption that exactly two slots remain, which is what makes
    // the assignment forced and the enumeration exact; anything else would be
    // silently describing a different position.
    if observation.hero_board.card_count() != 11
        || observation.opponent_public_board.card_count() != 11
    {
        return Err("T4 first requires 11-card hero and opponent boards".to_owned());
    }
    let actions = generate_turn_actions_trusted(&observation.hero_board, &observation.dealt_cards);
    if actions.is_empty() {
        return Err("T4 observation has no legal actions".to_owned());
    }
    let (outlook, legal) = crate::t4_features::opponent_outlook(observation);
    let mut scratch = model.scratch();
    let mut values = Vec::with_capacity(actions.len());
    for action in &actions {
        let board = action.apply_trusted(&observation.hero_board);
        let score = score_board_trusted(&board);
        let features = crate::t4_features::encode(observation, &score, &outlook, &legal);
        values.push(model.predict_with(&features, &mut scratch)? as f64);
    }
    let selected = canonical_descending_indices(&values, &actions)?[0];
    Ok(actions[selected].clone())
}

/// Select the same T4 action as the public evaluator without materializing its
/// JSON certificate.  Nested T3 search only consumes the action; serializing
/// every exact T4 action row and then regenerating the legal set to recover the
/// selected token was a dominant avoidable cost in first-seat T3 trees.
fn select_t4_action_without_result(
    observation: &ActorObservation,
    config: &T4Config,
) -> Result<Action, String> {
    if observation.street != Street::T4 {
        return Err("T4 search requires a T4 ActorObservation".to_owned());
    }
    if config.run_id.is_empty() {
        return Err("T4 run_id must not be empty".to_owned());
    }
    let actions = generate_turn_actions_trusted(&observation.hero_board, &observation.dealt_cards);
    if actions.is_empty() {
        return Err("T4 observation has no legal actions".to_owned());
    }
    let values = if observation.to_act_order == ActOrder::Second {
        if observation.opponent_public_board.card_count() != 13 {
            return Err("T4 second requires a complete opponent board".to_owned());
        }
        let fl_ev = fl_ev_14(observation)?;
        let opponent_score = score_board_compact_trusted(&observation.opponent_public_board);
        actions
            .iter()
            .map(|action| {
                let board = action.apply_trusted(&observation.hero_board);
                let own = score_board_compact_trusted(&board);
                Ok(heads_up_terminal_score_compact(
                    &own,
                    &opponent_score,
                    fl_ev,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?
    } else {
        if observation.hero_board.card_count() != 11
            || observation.opponent_public_board.card_count() != 11
        {
            return Err("T4 first requires 11-card hero and opponent boards".to_owned());
        }
        let plan = build_t4_future_plan(
            observation,
            config.candidate_samples,
            config.candidate_seed,
            &config.run_id,
            "candidate_selection",
        )?;
        score_t4_first_actions(observation, &actions, &plan.deals)?
            .into_iter()
            .map(|row| row.score)
            .collect()
    };
    let selected = canonical_descending_indices(&values, &actions)?[0];
    Ok(actions[selected].clone())
}

fn locked_t3_second_action(
    observation: &ActorObservation,
    context: &mut SearchContext,
) -> Result<Action, String> {
    if observation.street != Street::T3 || observation.to_act_order != ActOrder::Second {
        return Err("nested T3 response must be a T3 second-seat observation".to_owned());
    }
    let key = ObservationKey::new(observation);
    context.t3_child_observation_keys.insert(key);
    if let Some(action) = context.t3_second_action_cache.get(&key) {
        return Ok(action.clone());
    }
    if let Some(model) = context.t3_second_model.as_ref() {
        let selected = learned_t3_second_action(observation, model)?;
        context
            .t3_second_action_cache
            .insert(key, selected.clone());
        return Ok(selected);
    }
    let actions = generate_turn_actions_trusted(&observation.hero_board, &observation.dealt_cards);
    // The sampled fallback addresses its particles by the public fingerprint,
    // so that string survives the cache keys moving to card masks.
    let fingerprint = observation.fingerprint();
    let batch = sample_hidden_card_particles(
        observation,
        context.config.seed,
        &format!("{}:child-t3:{fingerprint}", context.config.run_id),
        context.config.downstream_t3_samples,
        0,
    )?;
    let values = score_t3_actions(observation, &actions, &batch.particles, context)?;
    let selected = actions[canonical_descending_indices(&values, &actions)?[0]].clone();
    context
        .t3_second_action_cache
        .insert(key, selected.clone());
    Ok(selected)
}

fn terminal_outcome(
    hero: &crate::state::Board,
    opponent: &crate::state::Board,
    fl_ev: f64,
) -> Result<TerminalOutcome, String> {
    let hero_score = score_board_trusted(hero);
    let opponent_score = score_board_trusted(opponent);
    let hero_royalty = if hero_score.busted {
        0
    } else {
        hero_score.total_royalty
    };
    let opponent_royalty = if opponent_score.busted {
        0
    } else {
        opponent_score.total_royalty
    };
    let hero_fl_value = if !hero_score.busted && hero_score.fl_entry.card_count == 14 {
        fl_ev
    } else {
        0.0
    };
    let opponent_fl_value = if !opponent_score.busted && opponent_score.fl_entry.card_count == 14 {
        fl_ev
    } else {
        0.0
    };
    let line_total = if hero_score.busted || opponent_score.busted {
        0
    } else {
        [
            (&hero_score.top_value, &opponent_score.top_value),
            (&hero_score.middle_value, &opponent_score.middle_value),
            (&hero_score.bottom_value, &opponent_score.bottom_value),
        ]
        .into_iter()
        .map(|(hero_value, opponent_value)| {
            if hero_value > opponent_value {
                1_i32
            } else if hero_value < opponent_value {
                -1_i32
            } else {
                0_i32
            }
        })
        .sum()
    };
    Ok(TerminalOutcome {
        hu_score: heads_up_terminal_score(&hero_score, &opponent_score, fl_ev),
        hero_busted: hero_score.busted,
        opponent_busted: opponent_score.busted,
        hero_scoop: line_total == 3,
        opponent_scoop: line_total == -3,
        hero_royalty,
        opponent_royalty,
        hero_fl_value,
        opponent_fl_value,
    })
}

fn fl_ev_14(observation: &ActorObservation) -> Result<f64, String> {
    observation
        .scoring
        .fl_ev
        .get(&14)
        .copied()
        .ok_or_else(|| "scoring context must define FL EV for 14 cards".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_t4_config() -> T4Config {
        T4Config {
            candidate_samples: 0,
            evaluation_samples: 0,
            seed: 42,
            candidate_seed: 43,
            evaluation_seed: 44,
            run_id: "child-fast-path-parity".to_owned(),
        }
    }

    fn t4_observation(first: bool) -> ActorObservation {
        let value = if first {
            json!({
                "schema":"regular_ofc_actor_observation_v1",
                "hero_board":{"top":["Kh"],"middle":["7d","Jh","Ac","6s","3c"],"bottom":["Qs","Qc","Ks","As","Js"]},
                "opponent_public_board":{"top":["Ad"],"middle":["3s","8c","4c","7c","6d"],"bottom":["Td","3d","9d","8d","6h"]},
                "dealt_cards":["4h","2c","Ah"],
                "hero_private_discards":["Tc","7h","Th"],
                "seat":"first","street":"T4","to_act_order":"first",
                "scoring":{"schema":"regular_ofc_scoring_context_v1","fl_ev":{"14":10.227020614683454},"middle_trips_royalty":2,"hu_line_points":true,"scoop_bonus":3,"foul_enabled":true,"fantasyland_cards":14}
            })
        } else {
            json!({
                "schema":"regular_ofc_actor_observation_v1",
                "hero_board":{"top":["Qh","Kc"],"middle":["Ah","Ac","4s","5s"],"bottom":["Td","7h","7s","7c","Th"]},
                "opponent_public_board":{"top":["2h","3h","4h"],"middle":["5c","2s","6h","4c","8c"],"bottom":["7d","Jd","9d","Qd","Kd"]},
                "dealt_cards":["2c","3c","4d"],
                "hero_private_discards":["6c","8h","9c"],
                "seat":"second","street":"T4","to_act_order":"second",
                "scoring":{"schema":"regular_ofc_scoring_context_v1","fl_ev":{"14":10.227020614683454},"middle_trips_royalty":2,"hu_line_points":true,"scoop_bonus":3,"foul_enabled":true,"fantasyland_cards":14}
            })
        };
        serde_json::from_value(value).unwrap()
    }

    fn t3_second_observation() -> ActorObservation {
        serde_json::from_str(
            r#"{"schema":"regular_ofc_actor_observation_v1","hero_board":{"top":[],"middle":["Qc","7d","Th","2d"],"bottom":["2c","2s","7h","8c","7s"]},"opponent_public_board":{"top":["4c"],"middle":["4d","3c","3h","5h","9c"],"bottom":["3d","Ah","6s","5c","6h"]},"dealt_cards":["8d","8s","As"],"hero_private_discards":["Js","6d"],"seat":"second","street":"T3","to_act_order":"second","scoring":{"schema":"regular_ofc_scoring_context_v1","fl_ev":{"14":10.227020614683454},"middle_trips_royalty":2,"hu_line_points":true,"scoop_bonus":3,"foul_enabled":true,"fantasyland_cards":14},"hero_in_fantasyland":false,"opponent_in_fantasyland":false,"opponent_discard_count":3}"#,
        )
        .unwrap()
    }

    fn t3_first_observation() -> ActorObservation {
        serde_json::from_str(
            r#"{"schema":"regular_ofc_actor_observation_v1","hero_board":{"top":[],"middle":["2c","Td","7c","2s"],"bottom":["9h","Js","Ad","Jc","Kc"]},"opponent_public_board":{"top":[],"middle":["Qc","Tc","2h","9s"],"bottom":["2d","Th","3h","4s","5c"]},"dealt_cards":["Ts","8h","4h"],"hero_private_discards":["9c","3s"],"seat":"first","street":"T3","to_act_order":"first","scoring":{"schema":"regular_ofc_scoring_context_v1","fl_ev":{"14":10.227020614683454},"middle_trips_royalty":2,"hu_line_points":true,"scoop_bonus":3,"foul_enabled":true,"fantasyland_cards":14},"hero_in_fantasyland":false,"opponent_in_fantasyland":false,"opponent_discard_count":2}"#,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // TEMPORARY rollout profile. Removed with the instrument it reads.
    // Run with:
    //   cargo test --release --lib -- --ignored --nocapture rollout_t0_second
    // -----------------------------------------------------------------------
    fn bench_fixture(name: &str) -> String {
        format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    fn bench_digest(name: &str) -> String {
        let parsed: Value =
            serde_json::from_str(&std::fs::read_to_string(bench_fixture(name)).expect("digest"))
                .expect("digest parses");
        parsed["weights_sha256"]
            .as_str()
            .expect("digest present")
            .to_owned()
    }

    /// A T0 second-seat root: the hero has placed nothing, the opponent five.
    fn bench_t0_second_observation() -> ActorObservation {
        let opponent = crate::state::Board::new(
            ALL_CARDS[43..45].to_vec(),
            ALL_CARDS[20..22].to_vec(),
            ALL_CARDS[7..8].to_vec(),
        )
        .expect("opponent board");
        ActorObservation::new(
            crate::state::Board::empty(),
            opponent,
            vec![
                ALL_CARDS[0], ALL_CARDS[13], ALL_CARDS[26], ALL_CARDS[39], ALL_CARDS[5],
            ],
            Vec::new(),
            Seat::Second,
            Street::T0,
            ActOrder::Second,
            crate::infoset::ScoringContext::default(),
        )
        .expect("T0 second-seat observation")
    }

    /// The gate driver's evaluator set, with the distilled T1 replies pinned.
    fn bench_config(fast: bool) -> T3Config {
        let wide = bench_fixture("t3first_model_v1.bin");
        let wide_sha = bench_digest("t3first_model_v1_predictions.json");
        T3Config {
            candidate_samples: 8,
            evaluation_samples: 128,
            downstream_t3_samples: 4,
            downstream_t4_samples: 0,
            seed: 991,
            candidate_seed: 992,
            evaluation_seed: 993,
            run_id: "rollout-bench".to_owned(),
            learned_t4_model_path: Some(bench_fixture("t4_model_v5.bin")),
            learned_t4_model_sha256: Some(bench_digest("t4_model_v5_predictions.json")),
            learned_t3_second_model_path: Some(bench_fixture("t3_model_v2.bin")),
            learned_t3_second_model_sha256: Some(bench_digest("t3_model_v2_predictions.json")),
            learned_t3_first_model_path: Some(wide.clone()),
            learned_t3_first_model_sha256: Some(wide_sha.clone()),
            learned_t2_second_model_path: Some(bench_fixture("t2_model_v1.bin")),
            learned_t2_second_model_sha256: Some(bench_digest("t2_model_v1.sha256.json")),
            learned_t2_first_model_path: Some(bench_fixture("t2first_model_v1.bin")),
            learned_t2_first_model_sha256: Some(bench_digest("t2first_model_v1.sha256.json")),
            learned_t1_second_model_path: Some(wide.clone()),
            learned_t1_second_model_sha256: Some(wide_sha.clone()),
            learned_t1_first_model_path: Some(wide.clone()),
            learned_t1_first_model_sha256: Some(wide_sha.clone()),
            fast_t1_second_model_path: fast.then(|| wide.clone()),
            fast_t1_second_model_sha256: fast.then(|| wide_sha.clone()),
            fast_t1_first_model_path: fast.then(|| wide.clone()),
            fast_t1_first_model_sha256: fast.then_some(wide_sha),
            ..Default::default()
        }
    }

    fn bench_context(config: &T3Config, observation: &ActorObservation) -> SearchContext {
        SearchContext {
            config: config.clone(),
            fl_ev_14: fl_ev_14(observation).expect("fl ev"),
            t4_model: load_learned_t4_model(config).expect("t4"),
            t3_second_model: load_learned_model(
                config.learned_t3_second_model_path.as_deref(),
                config.learned_t3_second_model_sha256.as_deref(),
                "learned_t3_second_model",
                crate::t3_features::FEATURE_SIZE,
            )
            .expect("t3 second"),
            t3_first_model: load_learned_model(
                config.learned_t3_first_model_path.as_deref(),
                config.learned_t3_first_model_sha256.as_deref(),
                "learned_t3_first_model",
                crate::t3first_features::FEATURE_SIZE,
            )
            .expect("t3 first"),
            t2_second_model: load_learned_model(
                config.learned_t2_second_model_path.as_deref(),
                config.learned_t2_second_model_sha256.as_deref(),
                "learned_t2_second_model",
                crate::t3first_features::FEATURE_SIZE,
            )
            .expect("t2 second"),
            t2_first_model: load_learned_model(
                config.learned_t2_first_model_path.as_deref(),
                config.learned_t2_first_model_sha256.as_deref(),
                "learned_t2_first_model",
                crate::t3first_features::FEATURE_SIZE,
            )
            .expect("t2 first"),
            t1_second_model: load_learned_model(
                config.learned_t1_second_model_path.as_deref(),
                config.learned_t1_second_model_sha256.as_deref(),
                "learned_t1_second_model",
                crate::t3first_features::FEATURE_SIZE,
            )
            .expect("t1 second"),
            t1_first_model: load_learned_model(
                config.learned_t1_first_model_path.as_deref(),
                config.learned_t1_first_model_sha256.as_deref(),
                "learned_t1_first_model",
                crate::t3first_features::FEATURE_SIZE,
            )
            .expect("t1 first"),
            t0_second_model: None,
            t0_first_model: None,
            fast_t2_second_model: load_learned_model(
                config.fast_t2_second_model_path.as_deref(),
                config.fast_t2_second_model_sha256.as_deref(),
                "fast_t2_second_model",
                crate::t3first_features::FEATURE_SIZE,
            )
            .expect("fast t2 second"),
            fast_t2_first_model: load_learned_model(
                config.fast_t2_first_model_path.as_deref(),
                config.fast_t2_first_model_sha256.as_deref(),
                "fast_t2_first_model",
                crate::t3first_features::FEATURE_SIZE,
            )
            .expect("fast t2 first"),
            fast_t1_second_model: load_learned_model(
                config.fast_t1_second_model_path.as_deref(),
                config.fast_t1_second_model_sha256.as_deref(),
                "fast_t1_second_model",
                crate::t3first_features::FEATURE_SIZE,
            )
            .expect("fast t1 second"),
            fast_t1_first_model: load_learned_model(
                config.fast_t1_first_model_path.as_deref(),
                config.fast_t1_first_model_sha256.as_deref(),
                "fast_t1_first_model",
                crate::t3first_features::FEATURE_SIZE,
            )
            .expect("fast t1 first"),
            fast_t0_second_model: load_learned_model(
                config.fast_t0_second_model_path.as_deref(),
                config.fast_t0_second_model_sha256.as_deref(),
                "fast_t0_second_model",
                crate::t3first_features::FEATURE_SIZE,
            )
            .expect("fast t0 second"),
            t4_action_cache: HashMap::new(),
            t3_second_action_cache: HashMap::new(),
            t3_first_action_cache: HashMap::new(),
            t2_second_action_cache: HashMap::new(),
            t2_first_action_cache: HashMap::new(),
            t1_second_action_cache: HashMap::new(),
            t1_first_action_cache: HashMap::new(),
            t0_second_action_cache: HashMap::new(),
            fast_t2_second_action_cache: HashMap::new(),
            fast_t2_first_action_cache: HashMap::new(),
            fast_t1_second_action_cache: HashMap::new(),
            fast_t1_first_action_cache: HashMap::new(),
            fast_t0_second_action_cache: HashMap::new(),
            fast_t2_second_outlook: FastOutlookCache::new(),
            fast_t2_first_outlook: FastOutlookCache::new(),
            fast_t1_second_outlook: FastOutlookCache::new(),
            fast_t1_first_outlook: FastOutlookCache::new(),
            fast_t0_second_outlook: FastOutlookCache::new(),
            t3_child_observation_keys: HashSet::new(),
            t4_child_observation_keys: HashSet::new(),
        }
    }

    /// What one teacher rollout costs, on both T1 replies.
    ///
    /// A T0 evaluation is nothing but rollouts -- six thousand of them at gate
    /// settings -- so this is the number every other cost in the engine is
    /// measured against, and the one a change to the rollout path has to move.
    ///
    /// Each rollout below is handed a particle no other rollout in the loop was
    /// handed, which is the condition production runs under: every particle
    /// deals different cards, so every nested observation is new and the
    /// fingerprint caches above the nested decisions never hit. Reusing one
    /// particle would measure the caches instead of the work.
    ///
    /// The rollout's eight nested decisions are dominated by the free outlook,
    /// twice over: once per decision for the opponent block and once per
    /// candidate action for the hero's. The fingerprints, the caches around
    /// them, the observation construction and the legal-set generation together
    /// are under one percent, which is why none of them is worth bypassing.
    #[test]
    #[ignore]
    fn rollout_t0_second_cost_is_reported() {
        let observation = bench_t0_second_observation();
        let actions =
            generate_initial_actions(&observation.hero_board, &observation.dealt_cards).unwrap();
        let root = actions[0].clone();

        println!("one rollout_t0_second");
        for fast in [false, true] {
            let config = bench_config(fast);
            let mut context = bench_context(&config, &observation);
            let particles = sample_hidden_card_particles(
                &observation,
                config.evaluation_seed,
                "rollout-bench:locked_evaluation",
                40,
                0,
            )
            .unwrap()
            .particles;

            // Warm the allocator and the weight files; not measured.
            rollout_t0_second(&observation, &root, &particles[0], &mut context).unwrap();

            let began = std::time::Instant::now();
            for particle in &particles[1..] {
                rollout_t0_second(&observation, &root, particle, &mut context).unwrap();
            }
            let rollouts = (particles.len() - 1) as f64;
            println!(
                "  {:<9} T1 reply  {:8.3} ms  ({rollouts:.0} distinct particles)",
                if fast { "distilled" } else { "full" },
                began.elapsed().as_secs_f64() * 1e3 / rollouts,
            );
        }
    }

    #[test]
    fn combinations_of_twenty_four_are_2024() {
        assert_eq!(combinations_three(&ALL_CARDS[..24]).len(), 2024);
    }

    #[test]
    fn nested_t4_fast_path_selects_the_public_evaluator_action() {
        let config = exact_t4_config();
        for observation in [t4_observation(true), t4_observation(false)] {
            let public = evaluate_t4(&observation, &config).unwrap();
            let direct = select_t4_action_without_result(&observation, &config).unwrap();
            assert_eq!(
                action_key(&direct).unwrap().to_token(),
                public["selected_action_key"].as_str().unwrap()
            );
        }
    }

    #[test]
    fn request_schema_rejects_unknown_kind_without_fallback() {
        let request = EngineRequest {
            schema: REQUEST_SCHEMA.to_owned(),
            kind: "raw_world".to_owned(),
            observation: serde_json::from_value(json!({
                "schema":"regular_ofc_actor_observation_v1",
                "hero_board":{"top":["2h","3h","4h"],"middle":["5h","6h","7h","8h"],"bottom":["9h","Th","Jh","Qh"]},
                "opponent_public_board":{"top":["Kh","Ah","2d"],"middle":["3d","4d","5d","6d"],"bottom":["7d","8d","9d","Td"]},
                "dealt_cards":["Jd","Qd","Kd"],
                "hero_private_discards":["Ad","2c","3c"],
                "seat":"first","street":"T4","to_act_order":"first",
                "scoring":{"schema":"regular_ofc_scoring_context_v1","fl_ev":{"14":10.227020614683454},"middle_trips_royalty":2,"hu_line_points":true,"scoop_bonus":3,"foul_enabled":true,"fantasyland_cards":14}
            })).unwrap(),
            observation_fingerprint: String::new(),
            config: Value::Null,
        };
        let mut request = request;
        request.observation_fingerprint = request.observation.fingerprint();
        assert!(evaluate_engine_request(request)
            .unwrap_err()
            .contains("unsupported M3 request kind"));
    }

    #[test]
    fn abr_terminal_components_preserve_every_legacy_q_bit_exactly() {
        let config = T3Config {
            candidate_samples: 1,
            evaluation_samples: 2,
            downstream_t3_samples: 1,
            downstream_t4_samples: 0,
            seed: 7001,
            candidate_seed: 7002,
            evaluation_seed: 7003,
            run_id: "abr-component-parity".to_owned(),
            use_t4_action_cache: true,
            ..Default::default()
        };
        for observation in [t3_first_observation(), t3_second_observation()] {
            let legacy = evaluate_t3(&observation, &config).unwrap();
            let diagnostic = evaluate_t3_abr_components(&observation, &config).unwrap();
            assert_eq!(
                legacy["selected_action_key"],
                diagnostic["selected_action_key"]
            );
            assert_eq!(
                legacy["selected_action_evaluation_score"],
                diagnostic["selected_action_evaluation_score"]
            );
            let legacy_rows = legacy["actions"].as_array().unwrap();
            let diagnostic_rows = diagnostic["actions"].as_array().unwrap();
            assert_eq!(legacy_rows.len(), diagnostic_rows.len());
            for (legacy_row, diagnostic_row) in legacy_rows.iter().zip(diagnostic_rows) {
                assert_eq!(legacy_row["action_key"], diagnostic_row["action_key"]);
                assert_eq!(
                    legacy_row["selection_score"],
                    diagnostic_row["selection_score"]
                );
                assert_eq!(legacy_row["score"], diagnostic_row["score"]);
                assert_eq!(
                    diagnostic_row["terminal_components"]["hu_score_mean"],
                    legacy_row["score"]
                );
                assert_eq!(
                    diagnostic_row["terminal_components"]["future_count"],
                    config.evaluation_samples
                );
            }
            let candidate = diagnostic["candidate_rng_key_digests"]
                .as_array()
                .unwrap()
                .iter()
                .map(Value::to_string)
                .collect::<HashSet<_>>();
            let evaluation = diagnostic["evaluation_rng_key_digests"].as_array().unwrap();
            assert!(evaluation
                .iter()
                .map(Value::to_string)
                .all(|key| !candidate.contains(&key)));
            let encoded = diagnostic.to_string();
            assert!(!encoded.contains("\"opponent_private_discards\":"));
            assert!(!encoded.contains("\"future_cards\":"));
        }
    }
}

/// The 168-wide row the T0 first-seat model is given, per candidate opening.
///
/// A diagnostic, and deliberately the same call the scoring path makes rather
/// than a reimplementation: a second encoder here could agree with the corpus
/// while the served one still disagreed, which would be worse than no answer.
fn t0_features(observation: &ActorObservation, _config: &T3Config) -> Result<Value, String> {
    if observation.street != Street::T0 || observation.to_act_order != ActOrder::First {
        return Err("t0_features covers T0 first seat only".to_owned());
    }
    let actions =
        generate_initial_actions(&observation.hero_board, &observation.dealt_cards)?;
    let unknown = crate::t3_features::unknown_cards(observation);
    let mut cache = FastOutlookCache::new();
    let mut rows: Vec<Value> = Vec::with_capacity(actions.len());
    for action in &actions {
        let board = action.apply_trusted(&observation.hero_board);
        let features = fast_encode_hidden_opponent(observation, &board, &unknown, &mut cache)?;
        rows.push(json!({
            "action_key": action_key(action)?.to_token(),
            "features": features.to_vec(),
        }));
    }
    Ok(json!({
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "kind": "t0_features",
        "feature_size": crate::t3first_features::FEATURE_SIZE,
        "unknown_count": unknown.len(),
        "actions": rows,
    }))
}
