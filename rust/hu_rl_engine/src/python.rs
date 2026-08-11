//! Optional, fail-closed PyO3 boundary for the deterministic batch engine.
//!
//! This module is a coordinator boundary, not a replay serializer.  Actor
//! observations and legal actions are safe only for the lane's current actor.
//! Public step results deliberately erase the selected ActionKey because its
//! discard mask identifies a private discard.  Snapshots remain opaque native
//! objects and cannot be serialized through this API.

use crate::{
    ActOrder, ActionKey, BatchExecutionConfig, BatchHuRlEnv, BatchStepOutcomeV1,
    BatchWorldSnapshot, Card, HuRlActorViewV1, LegalActionMappingV1, PairedSeedRange,
    ScoringContext, Seat, Street, BATCH_STEP_OUTCOME_SCHEMA, MAX_BATCH_LANES, MAX_BATCH_THREADS,
    MAX_LEGAL_ACTIONS,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBytes, PyInt, PyList, PyModule, PyString, PyType},
};
use serde_json::json;

const ACTION_KEY_TOKEN_LENGTH: usize = 60;

/// Opt-in packed V1 is deliberately separate from the canonical JSON V1 API.
/// Every integer and IEEE-754 value is little-endian.  Padding bytes are zero.
const PACKED_SCHEMA: &str = "regular_ofc_hu_rl_packed_boundary_v1";
const PACKED_ENDIANNESS: &str = "little";
const PACKED_SCORING_IDENTITY: &str = "regular_ofc_hu_standard_score_v1";
const PACKED_ACTION_U64S: usize = 4;
const PACKED_ACTION_BYTES: usize = PACKED_ACTION_U64S * size_of::<u64>();
const PACKED_ACTION_COUNT_BYTES: usize = 1;
const PACKED_DIGEST_BYTES: usize = 32;
const PACKED_HISTORY_SLOTS: usize = 9;
const PACKED_HISTORY_RECORD_BYTES: usize = 32;
const PACKED_OBSERVATION_PREFIX_BYTES: usize = 104;
const PACKED_OBSERVATION_RECORD_BYTES: usize =
    PACKED_OBSERVATION_PREFIX_BYTES + PACKED_HISTORY_SLOTS * PACKED_HISTORY_RECORD_BYTES;
const PACKED_STEP_RECORD_BYTES: usize = 48;

#[pyclass(name = "BatchSnapshot", module = "_ofc_hu_rl_engine", frozen)]
struct PyBatchSnapshot {
    snapshot: BatchWorldSnapshot,
}

#[pymethods]
impl PyBatchSnapshot {
    #[getter]
    fn lane_count(&self) -> usize {
        self.snapshot.lane_count()
    }

    #[getter]
    fn decision_counts(&self) -> Vec<usize> {
        self.snapshot.decision_counts()
    }

    #[getter]
    fn all_done(&self) -> bool {
        self.snapshot.all_done()
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchSnapshot(lane_count={}, decision_counts={:?}, hidden_state='<redacted>')",
            self.snapshot.lane_count(),
            self.snapshot.decision_counts()
        )
    }
}

/// Python owner for one fixed-width, lane-ordered deterministic batch.
#[pyclass(name = "BatchHuRlEnv", module = "_ofc_hu_rl_engine")]
struct PyBatchHuRlEnv {
    batch: BatchHuRlEnv,
}

#[pymethods]
impl PyBatchHuRlEnv {
    #[new]
    #[pyo3(signature = (explicit_decks, *, chunk_width=64, thread_count=1))]
    fn new(
        explicit_decks: &Bound<'_, PyAny>,
        chunk_width: usize,
        thread_count: usize,
    ) -> PyResult<Self> {
        let decks = parse_explicit_decks(explicit_decks, "construction")?;
        let execution = BatchExecutionConfig::new(chunk_width, thread_count)
            .map_err(|_| rejected_value("construction"))?;
        let batch = BatchHuRlEnv::with_execution_config(&decks, execution)
            .map_err(|_| rejected_value("construction"))?;
        Ok(Self { batch })
    }

    /// Construct `[AB, BA]` lane pairs without materializing a deck in Python.
    #[classmethod]
    #[pyo3(signature = (seed_base, global_pair_start, pair_count, seed_stride, *, chunk_width=64, thread_count=1))]
    fn from_paired_seed_range(
        _class: &Bound<'_, PyType>,
        seed_base: &Bound<'_, PyAny>,
        global_pair_start: &Bound<'_, PyAny>,
        pair_count: &Bound<'_, PyAny>,
        seed_stride: &Bound<'_, PyAny>,
        chunk_width: usize,
        thread_count: usize,
    ) -> PyResult<Self> {
        let range = parse_paired_seed_range(
            seed_base,
            global_pair_start,
            pair_count,
            seed_stride,
            "from_paired_seed_range",
        )?;
        let execution = BatchExecutionConfig::new(chunk_width, thread_count)
            .map_err(|_| rejected_value("from_paired_seed_range"))?;
        let batch = BatchHuRlEnv::with_execution_config_from_paired_seed_range(range, execution)
            .map_err(|_| rejected_value("from_paired_seed_range"))?;
        Ok(Self { batch })
    }

    #[getter]
    fn lane_count(&self) -> usize {
        self.batch.lane_count()
    }

    #[getter]
    fn decision_counts(&self) -> Vec<usize> {
        self.batch.decision_counts()
    }

    #[getter]
    fn all_done(&self) -> bool {
        self.batch.all_done()
    }

    /// Atomically reset all existing lanes from lane-ordered explicit decks.
    fn reset_batch(&mut self, explicit_decks: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
        let decks = parse_explicit_decks(explicit_decks, "reset_batch")?;
        if decks.len() != self.batch.lane_count() {
            return Err(rejected_value("reset_batch"));
        }
        let views = self
            .batch
            .reset_from_explicit_decks(&decks)
            .map_err(|_| rejected_value("reset_batch"))?;
        Ok(views
            .into_iter()
            .map(|view| view.canonical_json())
            .collect())
    }

    /// Atomically reset fixed-width `[AB, BA]` lanes from a Rust-owned range.
    fn reset_from_paired_seed_range(
        &mut self,
        seed_base: &Bound<'_, PyAny>,
        global_pair_start: &Bound<'_, PyAny>,
        pair_count: &Bound<'_, PyAny>,
        seed_stride: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<String>> {
        let range = parse_paired_seed_range(
            seed_base,
            global_pair_start,
            pair_count,
            seed_stride,
            "reset_from_paired_seed_range",
        )?;
        if range.lane_count() != self.batch.lane_count() {
            return Err(rejected_value("reset_from_paired_seed_range"));
        }
        let views = self
            .batch
            .reset_from_paired_seed_range(range)
            .map_err(|_| rejected_value("reset_from_paired_seed_range"))?;
        Ok(views
            .into_iter()
            .map(|view| view.canonical_json())
            .collect())
    }

    /// Return one canonical actor-view JSON document per lane, in lane order.
    fn observe_batch(&self) -> PyResult<Vec<String>> {
        let views = self
            .batch
            .observe_batch()
            .map_err(|_| rejected_runtime("observe_batch"))?;
        Ok(views
            .into_iter()
            .map(|view| view.canonical_json())
            .collect())
    }

    /// Return actor-safe observations as fixed-width little-endian records.
    ///
    /// This opt-in path contains exactly the policy-visible fields listed by
    /// `PACKED_SCHEMA`.  In particular, there is no world/deck-tail field and
    /// public history stores only placement masks plus discard counts.
    fn observe_batch_packed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let encoded = py.detach(|| {
            let views = self
                .batch
                .observe_batch()
                .map_err(|_| rejected_runtime("observe_batch_packed"))?;
            encode_actor_views_packed(&views)
        })?;
        Ok(PyBytes::new(py, &encoded))
    }

    /// Fixed-width `[batch, 232]` semantic keys and masks plus mapping proofs.
    ///
    /// Padded action entries are `None` and always have a false mask.  The
    /// tuple fields are action_keys, mask, action_counts, set digests, order
    /// digests, in that order.
    #[allow(clippy::type_complexity)]
    fn legal_actions_batch(
        &self,
    ) -> PyResult<(
        Vec<Vec<Option<String>>>,
        Vec<Vec<bool>>,
        Vec<usize>,
        Vec<String>,
        Vec<String>,
    )> {
        let mappings = self
            .batch
            .legal_mappings_batch()
            .map_err(|_| rejected_runtime("legal_actions_batch"))?;
        Ok(encode_legal_mappings(&mappings))
    }

    /// Return fixed-width little-endian ActionKeys, masks, counts and raw
    /// SHA-256 mapping digests.  Each padded ActionKey is 32 zero bytes.
    #[allow(clippy::type_complexity)]
    fn legal_actions_batch_packed<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
    )> {
        let encoded = py.detach(|| {
            let mappings = self
                .batch
                .legal_mappings_batch()
                .map_err(|_| rejected_runtime("legal_actions_batch_packed"))?;
            encode_legal_mappings_packed(mappings.iter(), "legal_actions_batch_packed")
        })?;
        Ok((
            PyBytes::new(py, &encoded.action_keys),
            PyBytes::new(py, &encoded.mask),
            PyBytes::new(py, &encoded.counts),
            PyBytes::new(py, &encoded.set_digests),
            PyBytes::new(py, &encoded.order_digests),
        ))
    }

    /// Return one actor-decision boundary from exactly one scalar observation
    /// traversal per lane.
    ///
    /// `HuRlActorViewV1` already owns the canonical legal mapping constructed
    /// by `observe_batch()`.  Encoding that mapping here avoids a second
    /// `legal_mappings_batch()` traversal and a second legal-action generation.
    /// The tuple is observations, ActionKeys, mask, uint8 counts, set digests,
    /// and order digests; every element is immutable Python `bytes`.
    #[allow(clippy::type_complexity)]
    fn actor_decision_batch_packed<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
    )> {
        let (observations, legal) = py.detach(|| {
            let views = self
                .batch
                .observe_batch()
                .map_err(|_| rejected_runtime("actor_decision_batch_packed"))?;
            let observations = encode_actor_views_packed(&views)?;
            let legal = encode_legal_mappings_packed(
                views.iter().map(HuRlActorViewV1::legal_action_mapping),
                "actor_decision_batch_packed",
            )?;
            Ok::<_, PyErr>((observations, legal))
        })?;
        Ok((
            PyBytes::new(py, &observations),
            PyBytes::new(py, &legal.action_keys),
            PyBytes::new(py, &legal.mask),
            PyBytes::new(py, &legal.counts),
            PyBytes::new(py, &legal.set_digests),
            PyBytes::new(py, &legal.order_digests),
        ))
    }

    /// Atomically step every lane and return actor-safe public results only.
    ///
    /// The selected ActionKey is intentionally absent because its discard mask
    /// is private.  Coordinators must retain their own submitted keys in a
    /// privileged audit stream if exact action reconstruction is required.
    fn step_batch(&mut self, selected_action_keys: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
        let selected =
            parse_action_keys(selected_action_keys, self.batch.lane_count(), "step_batch")?;
        let results = self
            .batch
            .step_batch(&selected)
            .map_err(|_| rejected_value("step_batch"))?;
        results
            .iter()
            .map(encode_public_step_result)
            .collect::<PyResult<Vec<_>>>()
    }

    /// Atomically consume one packed 4-u64 ActionKey per lane and return only
    /// actor-safe public fixed-width step records.
    ///
    /// Input must be an exact Python `bytes` object.  Length, 52-card-domain,
    /// disjointness and per-lane legality are all checked before the live batch
    /// can commit.  Error text never echoes a private discard mask.
    fn step_batch_packed<'py>(
        &mut self,
        py: Python<'py>,
        selected_action_keys: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let selected = parse_action_keys_packed(
            selected_action_keys.as_bytes(),
            self.batch.lane_count(),
            "step_batch_packed",
        )?;
        let encoded = py.detach(|| {
            let results = self
                .batch
                .step_batch(&selected)
                .map_err(|_| rejected_value("step_batch_packed"))?;
            encode_public_step_results_packed(&results)
        })?;
        Ok(PyBytes::new(py, &encoded))
    }

    /// Capture an opaque, non-serializable native restore point.
    fn snapshot_batch(&self) -> PyBatchSnapshot {
        PyBatchSnapshot {
            snapshot: self.batch.snapshot(),
        }
    }

    /// Atomically restore every lane; lane-count mismatch fails closed.
    fn restore_batch(&mut self, snapshot: PyRef<'_, PyBatchSnapshot>) -> PyResult<()> {
        self.batch
            .restore(&snapshot.snapshot)
            .map_err(|_| rejected_value("restore_batch"))
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchHuRlEnv(lane_count={}, decision_counts={:?}, hidden_state='<redacted>')",
            self.batch.lane_count(),
            self.batch.decision_counts()
        )
    }
}

fn parse_paired_seed_range(
    seed_base: &Bound<'_, PyAny>,
    global_pair_start: &Bound<'_, PyAny>,
    pair_count: &Bound<'_, PyAny>,
    seed_stride: &Bound<'_, PyAny>,
    operation: &str,
) -> PyResult<PairedSeedRange> {
    fn exact_u64(value: &Bound<'_, PyAny>, operation: &str) -> PyResult<u64> {
        if !value.is_exact_instance_of::<PyInt>() {
            return Err(rejected_type(operation));
        }
        value
            .extract::<u64>()
            .map_err(|_| rejected_value(operation))
    }

    let pair_count = usize::try_from(exact_u64(pair_count, operation)?)
        .map_err(|_| rejected_value(operation))?;
    PairedSeedRange::new(
        exact_u64(seed_base, operation)?,
        exact_u64(global_pair_start, operation)?,
        pair_count,
        exact_u64(seed_stride, operation)?,
    )
    .map_err(|_| rejected_value(operation))
}

fn parse_explicit_decks(value: &Bound<'_, PyAny>, operation: &str) -> PyResult<Vec<Vec<Card>>> {
    let outer = value
        .cast::<PyList>()
        .map_err(|_| rejected_type(operation))?;
    let lane_count = outer.len();
    if lane_count == 0 || lane_count > MAX_BATCH_LANES {
        return Err(rejected_value(operation));
    }

    // Check all lane geometry before allocating any Rust lane or parsing card
    // tokens.  This bounds native allocation and prevents partial validation.
    for lane in outer.iter() {
        let cards = lane
            .cast::<PyList>()
            .map_err(|_| rejected_type(operation))?;
        if cards.len() != 52 {
            return Err(rejected_value(operation));
        }
    }

    let mut decks = Vec::with_capacity(lane_count);
    for lane in outer.iter() {
        let cards = lane
            .cast::<PyList>()
            .map_err(|_| rejected_type(operation))?;
        let mut deck = Vec::with_capacity(52);
        for token in cards.iter() {
            let token = token
                .cast::<PyString>()
                .map_err(|_| rejected_type(operation))?;
            let token = token.to_str().map_err(|_| rejected_type(operation))?;
            if token.len() != 2 {
                return Err(rejected_value(operation));
            }
            let card = token
                .parse::<Card>()
                .map_err(|_| rejected_value(operation))?;
            deck.push(card);
        }
        decks.push(deck);
    }
    Ok(decks)
}

fn parse_action_keys(
    value: &Bound<'_, PyAny>,
    expected_count: usize,
    operation: &str,
) -> PyResult<Vec<ActionKey>> {
    let values = value
        .cast::<PyList>()
        .map_err(|_| rejected_type(operation))?;
    if values.len() != expected_count {
        return Err(rejected_value(operation));
    }
    let mut selected = Vec::with_capacity(expected_count);
    for value in values.iter() {
        let token = value
            .cast::<PyString>()
            .map_err(|_| rejected_type(operation))?;
        let token = token.to_str().map_err(|_| rejected_type(operation))?;
        if token.len() != ACTION_KEY_TOKEN_LENGTH {
            return Err(rejected_value(operation));
        }
        selected.push(ActionKey::from_token(token).map_err(|_| rejected_value(operation))?);
    }
    Ok(selected)
}

fn parse_action_keys_packed(
    value: &[u8],
    expected_count: usize,
    operation: &str,
) -> PyResult<Vec<ActionKey>> {
    let expected_bytes = expected_count
        .checked_mul(PACKED_ACTION_BYTES)
        .ok_or_else(|| rejected_value(operation))?;
    if value.len() != expected_bytes {
        return Err(rejected_value(operation));
    }
    let mut selected = Vec::with_capacity(expected_count);
    for record in value.chunks_exact(PACKED_ACTION_BYTES) {
        let mut masks = [0_u64; PACKED_ACTION_U64S];
        for (index, encoded) in record.chunks_exact(size_of::<u64>()).enumerate() {
            masks[index] =
                u64::from_le_bytes(encoded.try_into().map_err(|_| rejected_value(operation))?);
        }
        selected.push(
            ActionKey::new(masks[0], masks[1], masks[2], masks[3])
                .map_err(|_| rejected_value(operation))?,
        );
    }
    if selected.len() != expected_count {
        return Err(rejected_value(operation));
    }
    Ok(selected)
}

fn encode_actor_views_packed(views: &[HuRlActorViewV1]) -> PyResult<Vec<u8>> {
    let mut output = vec![0_u8; views.len() * PACKED_OBSERVATION_RECORD_BYTES];
    for (lane, view) in views.iter().enumerate() {
        let observation = view.observation();
        if observation.scoring != ScoringContext::default()
            || observation.hero_in_fantasyland
            || observation.opponent_in_fantasyland
            || view.public_history().len() > PACKED_HISTORY_SLOTS
        {
            return Err(rejected_runtime("observe_batch_packed"));
        }
        let record = &mut output
            [lane * PACKED_OBSERVATION_RECORD_BYTES..(lane + 1) * PACKED_OBSERVATION_RECORD_BYTES];
        let board_masks = [
            cards_mask(&observation.hero_board.top),
            cards_mask(&observation.hero_board.middle),
            cards_mask(&observation.hero_board.bottom),
            cards_mask(&observation.opponent_public_board.top),
            cards_mask(&observation.opponent_public_board.middle),
            cards_mask(&observation.opponent_public_board.bottom),
            cards_mask(&observation.hero_private_discards),
            cards_mask(&observation.dealt_cards),
        ];
        for (index, mask) in board_masks.into_iter().enumerate() {
            write_u64(record, index * size_of::<u64>(), mask);
        }
        record[64] = seat_code(observation.seat);
        record[65] = street_code(observation.street);
        record[66] = order_code(observation.to_act_order);
        record[67] = u8::from(observation.hero_in_fantasyland);
        record[68] = u8::from(observation.opponent_in_fantasyland);
        record[69] = u8::try_from(observation.opponent_discard_count())
            .map_err(|_| rejected_runtime("observe_batch_packed"))?;
        record[70] = u8::try_from(view.public_history().len())
            .map_err(|_| rejected_runtime("observe_batch_packed"))?;
        record[72..PACKED_OBSERVATION_PREFIX_BYTES]
            .copy_from_slice(PACKED_SCORING_IDENTITY.as_bytes());

        for (event_index, event) in view.public_history().iter().enumerate() {
            let offset =
                PACKED_OBSERVATION_PREFIX_BYTES + event_index * PACKED_HISTORY_RECORD_BYTES;
            let event_record = &mut record[offset..offset + PACKED_HISTORY_RECORD_BYTES];
            for (row, mask) in event.placement_masks().into_iter().enumerate() {
                write_u64(event_record, row * size_of::<u64>(), mask);
            }
            event_record[24] = street_code(event.street());
            event_record[25] = seat_code(event.acting_seat());
            event_record[26] = event.discard_count();
            event_record[27] = 1;
        }
    }
    Ok(output)
}

struct PackedLegalMappings {
    action_keys: Vec<u8>,
    mask: Vec<u8>,
    counts: Vec<u8>,
    set_digests: Vec<u8>,
    order_digests: Vec<u8>,
}

fn encode_legal_mappings_packed<'a>(
    mappings: impl ExactSizeIterator<Item = &'a LegalActionMappingV1>,
    operation: &str,
) -> PyResult<PackedLegalMappings> {
    let mapping_count = mappings.len();
    let key_lane_bytes = MAX_LEGAL_ACTIONS * PACKED_ACTION_BYTES;
    let mut output = PackedLegalMappings {
        action_keys: vec![0_u8; mapping_count * key_lane_bytes],
        mask: vec![0_u8; mapping_count * MAX_LEGAL_ACTIONS],
        counts: Vec::with_capacity(mapping_count * PACKED_ACTION_COUNT_BYTES),
        set_digests: Vec::with_capacity(mapping_count * PACKED_DIGEST_BYTES),
        order_digests: Vec::with_capacity(mapping_count * PACKED_DIGEST_BYTES),
    };
    for (lane, mapping) in mappings.enumerate() {
        let count =
            u8::try_from(mapping.action_count()).map_err(|_| rejected_runtime(operation))?;
        output.counts.push(count);
        let mask_offset = lane * MAX_LEGAL_ACTIONS;
        output.mask[mask_offset..mask_offset + mapping.action_count()].fill(1);
        let key_lane_offset = lane * key_lane_bytes;
        for (action_index, key) in mapping.action_keys().iter().enumerate() {
            let key_offset = key_lane_offset + action_index * PACKED_ACTION_BYTES;
            for (mask_index, mask) in key.masks().into_iter().enumerate() {
                write_u64(
                    &mut output.action_keys,
                    key_offset + mask_index * size_of::<u64>(),
                    mask,
                );
            }
        }
        output
            .set_digests
            .extend_from_slice(&decode_sha256_hex(mapping.action_set_digest(), operation)?);
        output.order_digests.extend_from_slice(&decode_sha256_hex(
            mapping.action_order_digest(),
            operation,
        )?);
    }
    Ok(output)
}

fn encode_public_step_results_packed(results: &[BatchStepOutcomeV1]) -> PyResult<Vec<u8>> {
    let mut output = vec![0_u8; results.len() * PACKED_STEP_RECORD_BYTES];
    for (lane, result) in results.iter().enumerate() {
        if !result.rewards.iter().all(|reward| reward.is_finite()) {
            return Err(rejected_runtime("step_batch_packed"));
        }
        let record =
            &mut output[lane * PACKED_STEP_RECORD_BYTES..(lane + 1) * PACKED_STEP_RECORD_BYTES];
        record[0] =
            u8::try_from(result.actor).map_err(|_| rejected_runtime("step_batch_packed"))?;
        record[1] = street_code(result.street);
        record[2] = u8::from(result.done);
        record[3] = result.public_placement.discard_count();
        for (row, mask) in result
            .public_placement
            .placement_masks()
            .into_iter()
            .enumerate()
        {
            write_u64(record, 8 + row * size_of::<u64>(), mask);
        }
        write_f64(record, 32, result.rewards[0]);
        write_f64(record, 40, result.rewards[1]);
    }
    Ok(output)
}

fn cards_mask(cards: &[Card]) -> u64 {
    cards.iter().fold(0_u64, |mask, card| mask | card.bit())
}

fn write_u64(output: &mut [u8], offset: usize, value: u64) {
    output[offset..offset + size_of::<u64>()].copy_from_slice(&value.to_le_bytes());
}

fn write_f64(output: &mut [u8], offset: usize, value: f64) {
    output[offset..offset + size_of::<f64>()].copy_from_slice(&value.to_le_bytes());
}

fn decode_sha256_hex(value: &str, operation: &str) -> PyResult<[u8; PACKED_DIGEST_BYTES]> {
    if value.len() != PACKED_DIGEST_BYTES * 2 || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(rejected_runtime(operation));
    }
    let mut output = [0_u8; PACKED_DIGEST_BYTES];
    for (index, encoded) in value.as_bytes().chunks_exact(2).enumerate() {
        let encoded = std::str::from_utf8(encoded).map_err(|_| rejected_runtime(operation))?;
        output[index] = u8::from_str_radix(encoded, 16).map_err(|_| rejected_runtime(operation))?;
    }
    Ok(output)
}

const fn seat_code(seat: Seat) -> u8 {
    match seat {
        Seat::First => 0,
        Seat::Second => 1,
    }
}

const fn street_code(street: Street) -> u8 {
    match street {
        Street::T0 => 0,
        Street::T1 => 1,
        Street::T2 => 2,
        Street::T3 => 3,
        Street::T4 => 4,
    }
}

const fn order_code(order: ActOrder) -> u8 {
    match order {
        ActOrder::First => 0,
        ActOrder::Second => 1,
    }
}

#[allow(clippy::type_complexity)]
fn encode_legal_mappings(
    mappings: &[LegalActionMappingV1],
) -> (
    Vec<Vec<Option<String>>>,
    Vec<Vec<bool>>,
    Vec<usize>,
    Vec<String>,
    Vec<String>,
) {
    let mut action_keys = Vec::with_capacity(mappings.len());
    let mut masks = Vec::with_capacity(mappings.len());
    let mut action_counts = Vec::with_capacity(mappings.len());
    let mut set_digests = Vec::with_capacity(mappings.len());
    let mut order_digests = Vec::with_capacity(mappings.len());
    for mapping in mappings {
        let mut keys = mapping
            .action_keys()
            .iter()
            .map(|key| Some(key.to_token()))
            .collect::<Vec<_>>();
        keys.resize(MAX_LEGAL_ACTIONS, None);
        action_keys.push(keys);
        masks.push(mapping.mask().to_vec());
        action_counts.push(mapping.action_count());
        set_digests.push(mapping.action_set_digest().to_owned());
        order_digests.push(mapping.action_order_digest().to_owned());
    }
    (
        action_keys,
        masks,
        action_counts,
        set_digests,
        order_digests,
    )
}

fn encode_public_step_result(result: &BatchStepOutcomeV1) -> PyResult<String> {
    serde_json::to_string(&json!({
        "schema": BATCH_STEP_OUTCOME_SCHEMA,
        "actor": result.actor,
        "street": result.street,
        "public_placement": result.public_placement.to_json(),
        "done": result.done,
        "rewards": result.rewards,
    }))
    .map_err(|_| rejected_runtime("step_batch"))
}

fn rejected_type(operation: &str) -> PyErr {
    PyTypeError::new_err(format!("hu_rl_engine {operation} rejected input type"))
}

fn rejected_value(operation: &str) -> PyErr {
    PyValueError::new_err(format!("hu_rl_engine {operation} rejected input"))
}

fn rejected_runtime(operation: &str) -> PyErr {
    PyRuntimeError::new_err(format!("hu_rl_engine {operation} failed closed"))
}

#[pymodule]
fn _ofc_hu_rl_engine(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBatchHuRlEnv>()?;
    module.add_class::<PyBatchSnapshot>()?;
    module.add("MAX_BATCH_LANES", MAX_BATCH_LANES)?;
    module.add("MAX_BATCH_THREADS", MAX_BATCH_THREADS)?;
    module.add(
        "MAX_PAIRED_HANDS_PER_BATCH",
        crate::MAX_PAIRED_HANDS_PER_BATCH,
    )?;
    module.add("MAX_PAIRED_SEED", crate::MAX_PAIRED_SEED)?;
    module.add("MAX_LEGAL_ACTIONS", MAX_LEGAL_ACTIONS)?;
    module.add("BATCH_STEP_OUTCOME_SCHEMA", BATCH_STEP_OUTCOME_SCHEMA)?;
    module.add("PACKED_SCHEMA", PACKED_SCHEMA)?;
    module.add("PACKED_ENDIANNESS", PACKED_ENDIANNESS)?;
    module.add("PACKED_SCORING_IDENTITY", PACKED_SCORING_IDENTITY)?;
    module.add("PACKED_ACTION_U64S", PACKED_ACTION_U64S)?;
    module.add("PACKED_ACTION_BYTES", PACKED_ACTION_BYTES)?;
    module.add("PACKED_ACTION_COUNT_BYTES", PACKED_ACTION_COUNT_BYTES)?;
    module.add("PACKED_DIGEST_BYTES", PACKED_DIGEST_BYTES)?;
    module.add("PACKED_HISTORY_SLOTS", PACKED_HISTORY_SLOTS)?;
    module.add("PACKED_HISTORY_RECORD_BYTES", PACKED_HISTORY_RECORD_BYTES)?;
    module.add(
        "PACKED_OBSERVATION_PREFIX_BYTES",
        PACKED_OBSERVATION_PREFIX_BYTES,
    )?;
    module.add(
        "PACKED_OBSERVATION_RECORD_BYTES",
        PACKED_OBSERVATION_RECORD_BYTES,
    )?;
    module.add("PACKED_STEP_RECORD_BYTES", PACKED_STEP_RECORD_BYTES)?;
    Ok(())
}
