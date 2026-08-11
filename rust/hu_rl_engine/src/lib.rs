//! Scalar, deterministic full-hand HU Regular OFC environment.
//!
//! The first contract is deliberately narrow: normal T0--T4 hands, an explicit
//! 52-card deck, semantic [`ActionKey`] actions, and no policy-visible simulator
//! truth. The batch layer preserves those scalar semantics exactly.  An
//! optional PyO3 boundary exposes that deterministic explicit-deck batch for
//! correctness work and a Rust-owned paired seed/reset path for production
//! replay. Search labels and explicit Fantasyland transitions belong to later
//! layers.

pub mod batch;
pub mod history;
pub mod observation;
pub mod seeded_deck;
pub mod snapshot;
pub mod trace;
pub mod transition;
mod world;

#[cfg(feature = "python")]
mod python;

use std::{error::Error, fmt};

pub use batch::{
    BatchExecutionConfig, BatchHuRlEnv, BatchStepOutcomeV1, BatchWorldSnapshot,
    BATCH_STEP_OUTCOME_SCHEMA, MAX_BATCH_LANES, MAX_BATCH_THREADS,
};
pub use history::{PublicPlacement, PUBLIC_PLACEMENT_SCHEMA};
pub use observation::{
    HuRlActorViewV1, LegalActionMappingV1, HU_RL_ACTOR_VIEW_SCHEMA, LEGAL_ACTION_MAPPING_SCHEMA,
    MAX_LEGAL_ACTIONS,
};
pub use ofc_hu_m3_engine::action_key::ActionKey;
pub use ofc_hu_m3_engine::cards::{Card, ALL_CARDS};
pub use ofc_hu_m3_engine::infoset::{ActOrder, ScoringContext, Seat, Street};
pub use ofc_hu_m3_engine::state::{Board, Row};
pub use seeded_deck::{
    cpython_oracle_deck, paired_seeded_decks, PairedSeedRange, MAX_PAIRED_HANDS_PER_BATCH,
    MAX_PAIRED_SEED,
};
pub use snapshot::WorldSnapshot;
pub use trace::{
    evaluate_scalar_trace_request, run_scalar_trace_json, HU_RL_SCALAR_TRACE_ARTIFACT_ROLE,
    HU_RL_SCALAR_TRACE_REQUEST_SCHEMA, HU_RL_SCALAR_TRACE_RESULT_SCHEMA,
    MAX_SCALAR_TRACE_REQUEST_BYTES,
};
pub use transition::{
    DecisionSpec, ScalarHuRlEnv, StepResult, DECISION_COUNT, DECISION_SCHEDULE,
    RAW_STEP_RESULT_ARTIFACT_ROLE,
};

/// Fail-closed scalar environment error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuRlError(String);

impl HuRlError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for HuRlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for HuRlError {}

impl From<String> for HuRlError {
    fn from(value: String) -> Self {
        Self(value)
    }
}

pub type HuRlResult<T> = Result<T, HuRlError>;
