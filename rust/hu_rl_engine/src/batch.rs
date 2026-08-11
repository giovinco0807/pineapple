//! Deterministic, fail-closed batching over the scalar normal-hand engine.
//!
//! Lane index is the only public batch identity in V1.  Every operation
//! consumes and emits lanes in that exact order.  Parallel chunks run in a
//! private fixed-width Rayon pool and are reassembled by chunk index, so chunk
//! width and thread count are non-semantic execution hints.

use crate::{
    ActionKey, Card, HuRlActorViewV1, HuRlError, HuRlResult, LegalActionMappingV1, PublicPlacement,
    ScalarHuRlEnv, StepResult, Street, WorldSnapshot,
};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::{
    fmt,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

pub const MAX_BATCH_LANES: usize = 4096;
pub const MAX_BATCH_THREADS: usize = 64;
pub const BATCH_STEP_OUTCOME_SCHEMA: &str = "regular_ofc_hu_rl_batch_step_outcome_v1";

static NEXT_BATCH_ID: AtomicU64 = AtomicU64::new(1);

/// Non-semantic execution hints for ordered parallel batch traversal.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BatchExecutionConfig {
    chunk_width: usize,
    thread_count: usize,
}

impl BatchExecutionConfig {
    pub fn new(chunk_width: usize, thread_count: usize) -> HuRlResult<Self> {
        if chunk_width == 0 {
            return Err(HuRlError::new(
                "batch chunk_width must be greater than zero",
            ));
        }
        if thread_count == 0 {
            return Err(HuRlError::new(
                "batch thread_count must be greater than zero",
            ));
        }
        if thread_count > MAX_BATCH_THREADS {
            return Err(HuRlError::new(format!(
                "batch thread_count exceeds maximum {MAX_BATCH_THREADS}"
            )));
        }
        Ok(Self {
            chunk_width,
            thread_count,
        })
    }

    pub const fn chunk_width(self) -> usize {
        self.chunk_width
    }

    pub const fn thread_count(self) -> usize {
        self.thread_count
    }
}

impl Default for BatchExecutionConfig {
    fn default() -> Self {
        Self {
            chunk_width: 64,
            thread_count: 1,
        }
    }
}

/// Actor-safe result returned by the public batch transition boundary.
///
/// Unlike raw [`StepResult`], this type never contains the semantic
/// [`ActionKey`] because that key carries private discard identity after T0.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchStepOutcomeV1 {
    pub actor: usize,
    pub street: Street,
    pub public_placement: PublicPlacement,
    pub done: bool,
    pub rewards: [f64; 2],
}

impl From<StepResult> for BatchStepOutcomeV1 {
    fn from(result: StepResult) -> Self {
        Self {
            actor: result.actor,
            street: result.street,
            public_placement: result.public_placement,
            done: result.done,
            rewards: result.rewards,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct BatchLineage {
    batch_id: u64,
    generation: u64,
}

impl BatchLineage {
    fn fresh() -> HuRlResult<Self> {
        let batch_id = NEXT_BATCH_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| HuRlError::new("batch lineage space is exhausted"))?;
        Ok(Self {
            batch_id,
            generation: 0,
        })
    }

    fn next_generation(self) -> HuRlResult<Self> {
        Ok(Self {
            batch_id: self.batch_id,
            generation: self
                .generation
                .checked_add(1)
                .ok_or_else(|| HuRlError::new("batch reset generation is exhausted"))?,
        })
    }
}

/// Opaque, lineage-bound, lane-ordered restore point for an entire batch.
#[derive(Clone, PartialEq)]
pub struct BatchWorldSnapshot {
    lineage: BatchLineage,
    lanes: Vec<WorldSnapshot>,
}

impl BatchWorldSnapshot {
    pub fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    pub fn decision_counts(&self) -> Vec<usize> {
        self.lanes
            .iter()
            .map(WorldSnapshot::decision_count)
            .collect()
    }

    pub fn all_done(&self) -> bool {
        self.lanes.iter().all(WorldSnapshot::done)
    }
}

impl fmt::Debug for BatchWorldSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BatchWorldSnapshot")
            .field("lane_count", &self.lane_count())
            .field("decision_counts", &self.decision_counts())
            .field("hidden_state", &"<redacted>")
            .finish()
    }
}

/// A fixed-width, lane-ordered batch of independent scalar environments.
pub struct BatchHuRlEnv {
    lanes: Vec<ScalarHuRlEnv>,
    execution: BatchExecutionConfig,
    pool: Arc<ThreadPool>,
    lineage: BatchLineage,
}

impl fmt::Debug for BatchHuRlEnv {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BatchHuRlEnv")
            .field("lane_count", &self.lane_count())
            .field("decision_counts", &self.decision_counts())
            .field("execution", &self.execution)
            .field("hidden_state", &"<redacted>")
            .finish()
    }
}

impl BatchHuRlEnv {
    pub fn from_explicit_decks(explicit_decks: &[Vec<Card>]) -> HuRlResult<Self> {
        Self::with_execution_config(explicit_decks, BatchExecutionConfig::default())
    }

    pub fn with_execution_config(
        explicit_decks: &[Vec<Card>],
        execution: BatchExecutionConfig,
    ) -> HuRlResult<Self> {
        // The width cap is checked before lane construction, thread-pool
        // construction, or any internal clone/allocation proportional to input.
        validate_batch_width(explicit_decks.len())?;
        let lanes = build_lanes(explicit_decks)?;
        let pool = build_thread_pool(execution)?;
        let lineage = BatchLineage::fresh()?;
        Ok(Self {
            lanes,
            execution,
            pool,
            lineage,
        })
    }

    /// Construct paired AB/BA lanes from a validated Rust-owned seed range.
    pub fn from_paired_seed_range(range: crate::PairedSeedRange) -> HuRlResult<Self> {
        Self::with_execution_config_from_paired_seed_range(range, BatchExecutionConfig::default())
    }

    /// Construct paired AB/BA lanes with explicit bounded scheduling hints.
    pub fn with_execution_config_from_paired_seed_range(
        range: crate::PairedSeedRange,
        execution: BatchExecutionConfig,
    ) -> HuRlResult<Self> {
        let decks = crate::paired_seeded_decks(range);
        Self::with_execution_config(&decks, execution)
    }

    pub fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    pub fn execution_config(&self) -> BatchExecutionConfig {
        self.execution
    }

    /// Atomically changes scheduling hints without changing any lane state.
    pub fn set_execution_config(&mut self, execution: BatchExecutionConfig) -> HuRlResult<()> {
        let pool = build_thread_pool(execution)?;
        self.execution = execution;
        self.pool = pool;
        Ok(())
    }

    pub fn decision_counts(&self) -> Vec<usize> {
        self.lanes
            .iter()
            .map(ScalarHuRlEnv::decision_count)
            .collect()
    }

    pub fn all_done(&self) -> bool {
        self.lanes.iter().all(ScalarHuRlEnv::done)
    }

    /// Atomically resets every existing lane from an ordered explicit deck.
    ///
    /// A successful reset advances the opaque lineage generation, invalidating
    /// stale snapshots from the previous episode.  Batch width is fixed for the
    /// lifetime of the object.
    pub fn reset_from_explicit_decks(
        &mut self,
        explicit_decks: &[Vec<Card>],
    ) -> HuRlResult<Vec<HuRlActorViewV1>> {
        validate_batch_width(explicit_decks.len())?;
        self.require_lane_count(explicit_decks.len(), "reset")?;
        let next_lineage = self.lineage.next_generation()?;
        let mut next = self.lanes.clone();
        let width = self.execution.chunk_width;
        let chunk_results = self.pool.install(|| {
            next.par_chunks_mut(width)
                .enumerate()
                .map(|(chunk_index, lane_chunk)| {
                    let start = chunk_index * width;
                    let result = lane_chunk
                        .iter_mut()
                        .enumerate()
                        .map(|(offset, lane)| {
                            let lane_index = start + offset;
                            lane.reset(&explicit_decks[lane_index])
                                .map_err(|error| lane_error("reset", lane_index, error))
                        })
                        .collect();
                    (chunk_index, result)
                })
                .collect()
        });
        let views = flatten_ordered_chunks(chunk_results, next.len())?;
        self.lanes = next;
        self.lineage = next_lineage;
        Ok(views)
    }

    /// Atomically reset all lanes from a paired Rust-owned seed range.
    ///
    /// The range is validated and every deck is generated before the existing
    /// explicit reset transaction is entered. A rejected range or width cannot
    /// change lane state or snapshot lineage.
    pub fn reset_from_paired_seed_range(
        &mut self,
        range: crate::PairedSeedRange,
    ) -> HuRlResult<Vec<HuRlActorViewV1>> {
        self.require_lane_count(range.lane_count(), "paired seed reset")?;
        let decks = crate::paired_seeded_decks(range);
        self.reset_from_explicit_decks(&decks)
    }

    pub fn observe_batch(&self) -> HuRlResult<Vec<HuRlActorViewV1>> {
        let width = self.execution.chunk_width;
        let chunk_results = self.pool.install(|| {
            self.lanes
                .par_chunks(width)
                .enumerate()
                .map(|(chunk_index, lane_chunk)| {
                    let start = chunk_index * width;
                    let result = lane_chunk
                        .iter()
                        .enumerate()
                        .map(|(offset, lane)| {
                            lane.observe()
                                .map_err(|error| lane_error("observe", start + offset, error))
                        })
                        .collect();
                    (chunk_index, result)
                })
                .collect()
        });
        flatten_ordered_chunks(chunk_results, self.lanes.len())
    }

    pub fn legal_mappings_batch(&self) -> HuRlResult<Vec<LegalActionMappingV1>> {
        let width = self.execution.chunk_width;
        let chunk_results = self.pool.install(|| {
            self.lanes
                .par_chunks(width)
                .enumerate()
                .map(|(chunk_index, lane_chunk)| {
                    let start = chunk_index * width;
                    let result = lane_chunk
                        .iter()
                        .enumerate()
                        .map(|(offset, lane)| {
                            lane.legal_mapping()
                                .map_err(|error| lane_error("legal mapping", start + offset, error))
                        })
                        .collect();
                    (chunk_index, result)
                })
                .collect()
        });
        flatten_ordered_chunks(chunk_results, self.lanes.len())
    }

    /// Applies one semantic [`ActionKey`] per lane as one atomic transaction.
    ///
    /// The input key is needed to select the private action but is intentionally
    /// absent from the returned actor-safe outcome.
    pub fn step_batch(&mut self, selected: &[ActionKey]) -> HuRlResult<Vec<BatchStepOutcomeV1>> {
        validate_batch_width(selected.len())?;
        self.require_lane_count(selected.len(), "step")?;

        // Every scalar successor is constructed from an immutable lane.  The
        // live batch is replaced only after every parallel lane succeeds, so
        // a failure cannot partially mutate state.  This avoids both a
        // redundant pre-validation pass and an eager clone of the entire
        // batch before scalar transition construction.
        let width = self.execution.chunk_width;
        let chunk_results = self.pool.install(|| {
            self.lanes
                .par_chunks(width)
                .enumerate()
                .map(|(chunk_index, lane_chunk)| {
                    let start = chunk_index * width;
                    let result = lane_chunk
                        .iter()
                        .enumerate()
                        .map(|(offset, lane)| {
                            let lane_index = start + offset;
                            lane.preview_step(selected[lane_index])
                                .map(|(next, result)| (next, BatchStepOutcomeV1::from(result)))
                                .map_err(|error| lane_error("step", lane_index, error))
                        })
                        .collect();
                    (chunk_index, result)
                })
                .collect()
        });
        let successors = flatten_ordered_chunks(chunk_results, self.lanes.len())?;
        let (next, results): (Vec<_>, Vec<_>) = successors.into_iter().unzip();
        self.lanes = next;
        Ok(results)
    }

    pub fn terminal_rewards_batch(&self) -> HuRlResult<Vec<[f64; 2]>> {
        let width = self.execution.chunk_width;
        let chunk_results = self.pool.install(|| {
            self.lanes
                .par_chunks(width)
                .enumerate()
                .map(|(chunk_index, lane_chunk)| {
                    let start = chunk_index * width;
                    let result = lane_chunk
                        .iter()
                        .enumerate()
                        .map(|(offset, lane)| {
                            lane.terminal_rewards().map_err(|error| {
                                lane_error("terminal rewards", start + offset, error)
                            })
                        })
                        .collect();
                    (chunk_index, result)
                })
                .collect()
        });
        flatten_ordered_chunks(chunk_results, self.lanes.len())
    }

    pub fn snapshot(&self) -> BatchWorldSnapshot {
        BatchWorldSnapshot {
            lineage: self.lineage,
            lanes: self.lanes.iter().map(ScalarHuRlEnv::snapshot).collect(),
        }
    }

    /// Restores all ordered lanes atomically; execution hints are unchanged.
    pub fn restore(&mut self, snapshot: &BatchWorldSnapshot) -> HuRlResult<()> {
        if snapshot.lineage != self.lineage {
            return Err(HuRlError::new(
                "batch snapshot lineage does not match this batch",
            ));
        }
        validate_batch_width(snapshot.lane_count())?;
        self.require_lane_count(snapshot.lane_count(), "restore")?;
        let mut next = self.lanes.clone();
        let width = self.execution.chunk_width;
        let chunk_results = self.pool.install(|| {
            next.par_chunks_mut(width)
                .enumerate()
                .map(|(chunk_index, lane_chunk)| {
                    let start = chunk_index * width;
                    let result = lane_chunk.iter_mut().enumerate().try_for_each(
                        |(offset, lane)| -> HuRlResult<()> {
                            let lane_index = start + offset;
                            lane.restore(&snapshot.lanes[lane_index])
                                .map_err(|error| lane_error("restore", lane_index, error))
                        },
                    );
                    (chunk_index, result)
                })
                .collect::<Vec<_>>()
        });
        validate_ordered_chunks(chunk_results)?;
        self.lanes = next;
        Ok(())
    }

    fn require_lane_count(&self, actual: usize, operation: &str) -> HuRlResult<()> {
        if actual != self.lanes.len() {
            return Err(HuRlError::new(format!(
                "batch {operation} requires exactly {} lanes, got {actual}",
                self.lanes.len()
            )));
        }
        Ok(())
    }

    #[cfg(test)]
    fn actual_thread_count(&self) -> usize {
        self.pool.current_num_threads()
    }
}

fn validate_batch_width(lane_count: usize) -> HuRlResult<()> {
    if lane_count == 0 {
        return Err(HuRlError::new("batch requires at least one lane"));
    }
    if lane_count > MAX_BATCH_LANES {
        return Err(HuRlError::new(format!(
            "batch lane count exceeds maximum {MAX_BATCH_LANES}"
        )));
    }
    Ok(())
}

fn build_lanes(explicit_decks: &[Vec<Card>]) -> HuRlResult<Vec<ScalarHuRlEnv>> {
    explicit_decks
        .iter()
        .enumerate()
        .map(|(lane_index, deck)| {
            ScalarHuRlEnv::new(deck).map_err(|error| lane_error("construction", lane_index, error))
        })
        .collect()
}

fn build_thread_pool(execution: BatchExecutionConfig) -> HuRlResult<Arc<ThreadPool>> {
    ThreadPoolBuilder::new()
        .num_threads(execution.thread_count)
        .thread_name(|index| format!("ofc-hu-rl-batch-{index}"))
        .build()
        .map(Arc::new)
        .map_err(|_| HuRlError::new("failed to construct bounded batch thread pool"))
}

fn flatten_ordered_chunks<T>(
    mut chunks: Vec<(usize, HuRlResult<Vec<T>>)>,
    total_capacity: usize,
) -> HuRlResult<Vec<T>> {
    chunks.sort_unstable_by_key(|(chunk_index, _)| *chunk_index);
    let mut flattened = Vec::with_capacity(total_capacity);
    for (expected_index, (actual_index, chunk)) in chunks.into_iter().enumerate() {
        if actual_index != expected_index {
            return Err(HuRlError::new("parallel batch chunk index changed"));
        }
        flattened.extend(chunk?);
    }
    if flattened.len() != total_capacity {
        return Err(HuRlError::new("parallel batch lane count changed"));
    }
    Ok(flattened)
}

fn validate_ordered_chunks(mut chunks: Vec<(usize, HuRlResult<()>)>) -> HuRlResult<()> {
    chunks.sort_unstable_by_key(|(chunk_index, _)| *chunk_index);
    for (expected_index, (actual_index, result)) in chunks.into_iter().enumerate() {
        if actual_index != expected_index {
            return Err(HuRlError::new("parallel batch chunk index changed"));
        }
        result?;
    }
    Ok(())
}

fn lane_error(operation: &str, lane_index: usize, error: HuRlError) -> HuRlError {
    HuRlError::new(format!(
        "batch {operation} failed at lane {lane_index}: {}",
        error.message()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ALL_CARDS;

    #[test]
    fn private_pool_honors_the_requested_thread_count() {
        let decks = vec![ALL_CARDS.to_vec(); 8];
        let batch =
            BatchHuRlEnv::with_execution_config(&decks, BatchExecutionConfig::new(1, 3).unwrap())
                .unwrap();
        assert_eq!(batch.actual_thread_count(), 3);
    }
}
