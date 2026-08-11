from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_rl_contract import MAX_LEGAL_ACTIONS
from ofc_regular.hu_rl_native import (
    PACKED_ACTION_BYTES,
    PACKED_DIGEST_BYTES,
    PACKED_OBSERVATION_RECORD_BYTES,
    PACKED_STEP_RECORD_BYTES,
)
from ofc_regular.hu_rl_native_benchmark import (
    BENCHMARK_BOUNDARY_MODES,
    BENCHMARK_LANE_COUNTS,
    DECISIONS_PER_HAND,
    HuRlNativeBenchmarkError,
    ProcessMemorySample,
    canonical_benchmark_json,
    generate_benchmark_decks,
    read_process_memory,
    run_native_batch_mechanics_benchmark,
    validate_native_batch_benchmark_document,
)


class _FakeObservationBatch:
    def __init__(self, lane_count: int, decision: int) -> None:
        self.lane_count = lane_count
        self.payload = bytes([decision]) * (
            lane_count * PACKED_OBSERVATION_RECORD_BYTES
        )


class _FakeLegalBatch:
    def __init__(self, lane_count: int, decision: int) -> None:
        self.lane_count = lane_count
        key_lane = bytes(PACKED_ACTION_BYTES) + bytes(
            (MAX_LEGAL_ACTIONS - 1) * PACKED_ACTION_BYTES
        )
        self.action_keys = key_lane * lane_count
        self.mask = (b"\x01" + bytes(MAX_LEGAL_ACTIONS - 1)) * lane_count
        self.action_counts = b"\x01" * lane_count
        self.action_set_digests = bytes([decision + 1]) * (
            lane_count * PACKED_DIGEST_BYTES
        )
        self.action_order_digests = bytes([decision + 2]) * (
            lane_count * PACKED_DIGEST_BYTES
        )

    def action_count(self, lane: int) -> int:
        assert 0 <= lane < self.lane_count
        return 1

    def select(self, indices: Sequence[int]) -> bytes:
        assert tuple(indices) == (0,) * self.lane_count
        return bytes(self.lane_count * PACKED_ACTION_BYTES)


class _FakeActorDecisionBatch:
    def __init__(self, lane_count: int, decision: int) -> None:
        self.observations = _FakeObservationBatch(lane_count, decision)
        self.legal_actions = _FakeLegalBatch(lane_count, decision)


class _FakeStepBatch:
    def __init__(self, lane_count: int, decision: int, *, wrong_actor: bool) -> None:
        self.lane_count = lane_count
        self._decision = decision
        self._wrong_actor = wrong_actor
        self.payload = bytes([decision + 1]) * (lane_count * PACKED_STEP_RECORD_BYTES)

    def lane(self, lane: int) -> SimpleNamespace:
        assert 0 <= lane < self.lane_count
        street, actor = divmod(self._decision, 2)
        if self._wrong_actor:
            actor = 1 - actor
        done = self._decision == DECISIONS_PER_HAND - 1
        return SimpleNamespace(
            actor=actor,
            street=f"T{street}",
            done=done,
            rewards=(-3.0, 3.0) if done else (0.0, 0.0),
        )


class _FakeBatchEnv:
    wrong_actor_at: int | None = None

    def __init__(
        self,
        explicit_decks: Sequence[Sequence[str]],
        *,
        chunk_width: int,
        thread_count: int,
    ) -> None:
        assert chunk_width > 0
        assert thread_count > 0
        self._lane_count = len(explicit_decks)
        self._decision = 0

    @property
    def lane_count(self) -> int:
        return self._lane_count

    @property
    def decision_counts(self) -> tuple[int, ...]:
        return (self._decision,) * self._lane_count

    @property
    def all_done(self) -> bool:
        return self._decision == DECISIONS_PER_HAND

    def reset_batch(self, explicit_decks: Sequence[Sequence[str]]) -> tuple[dict[str, Any], ...]:
        assert len(explicit_decks) == self._lane_count
        self._decision = 0
        return ({},) * self._lane_count

    def observe_batch_packed(self) -> _FakeObservationBatch:
        return _FakeObservationBatch(self._lane_count, self._decision)

    def legal_actions_batch_packed(self) -> _FakeLegalBatch:
        return _FakeLegalBatch(self._lane_count, self._decision)

    def actor_decision_batch_packed(self) -> _FakeActorDecisionBatch:
        return _FakeActorDecisionBatch(self._lane_count, self._decision)

    def step_batch_packed(self, selected_action_keys: bytes) -> _FakeStepBatch:
        assert len(selected_action_keys) == self._lane_count * PACKED_ACTION_BYTES
        result = _FakeStepBatch(
            self._lane_count,
            self._decision,
            wrong_actor=self.wrong_actor_at == self._decision,
        )
        self._decision += 1
        return result


class _WrongActorBatchEnv(_FakeBatchEnv):
    wrong_actor_at = 3


class _IncreasingMemoryReader:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> ProcessMemorySample:
        self.calls += 1
        rss = 32_000_000 + (self.calls * 4096)
        return ProcessMemorySample(
            rss_bytes=rss,
            peak_rss_bytes=rss,
            source="test_memory_reader",
        )


def _test_double_document() -> dict[str, Any]:
    return run_native_batch_mechanics_benchmark(
        _env_factory=_FakeBatchEnv,
        _memory_reader=_IncreasingMemoryReader(),
        thread_count=4,
    )


def test_fixed_128_512_local_diagnostic_contract() -> None:
    document = _test_double_document()
    validate_native_batch_benchmark_document(document, require_native_engine=False)

    assert document["lane_counts"] == list(BENCHMARK_LANE_COUNTS)
    assert document["engine"] == "injected_test_double"
    assert document["scope"] == "strictly_local_no_cloud"
    assert document["cloud_executable"] is False
    assert document["launch_authorized"] is False
    assert document["performance_gate_evaluated"] is False
    assert document["performance_gate_pass"] is None
    assert document["separate_packed_byte_exact_by_lane"] == {
        "128": True,
        "512": True,
    }
    assert len(document["receipt_sha256"]) == 64
    assert set(document["provenance"]["source_sha256"]) == {
        "rust/hu_rl_engine/Cargo.toml",
        "rust/hu_rl_engine/src/batch.rs",
        "rust/hu_rl_engine/src/history.rs",
        "rust/hu_rl_engine/src/lib.rs",
        "rust/hu_rl_engine/src/observation.rs",
        "rust/hu_rl_engine/src/python.rs",
        "rust/hu_rl_engine/src/transition.rs",
        "rust/hu_rl_engine/src/world.rs",
        "scripts/benchmark_hu_rl_native_batch.py",
        "src/ofc_regular/ai_profiles.py",
        "src/ofc_regular/hu_rl_native.py",
        "src/ofc_regular/hu_rl_native_benchmark.py",
    }
    assert document["diagnostic_disposition"] in {
        "below_rate_target_no_go",
        "rate_target_met_but_gate_not_evaluated",
    }
    expected_pairs = tuple(
        (lane_count, boundary_mode)
        for lane_count in BENCHMARK_LANE_COUNTS
        for boundary_mode in BENCHMARK_BOUNDARY_MODES
    )
    for (lane_count, boundary_mode), result in zip(
        expected_pairs, document["results"], strict=True
    ):
        assert result["lane_count"] == lane_count
        assert result["boundary_mode"] == boundary_mode
        assert result["actor_decisions"] == lane_count * DECISIONS_PER_HAND
        assert result["separate_packed_byte_exact"] is True
        assert result["actor_decisions_per_second"] > 0.0
        assert result["end_to_end_actor_decisions_per_second"] > 0.0
        assert result["all_done"] is True
        assert result["memory"]["sample_count"] == 24
        assert result["memory"]["peak_rss_bytes"] >= result["memory"]["final_rss_bytes"]

    encoded = json.dumps(document, sort_keys=True)
    lowered = encoded.lower()
    assert "deck_tail" not in lowered
    assert "opponent_private_discard" not in lowered
    assert "opponent_private_discards" not in lowered


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda value: value.__setitem__("performance_gate_evaluated", True),
            "performance_gate_evaluated",
        ),
        (
            lambda value: value["results"][0].__setitem__(
                "actor_decisions_per_second", float("nan")
            ),
            "throughput",
        ),
        (
            lambda value: value["results"][1].__setitem__(
                "decision_counts_digest", "0" * 64
            ),
            "decision-count digest",
        ),
        (
            lambda value: value["results"][0]["memory"].__setitem__(
                "peak_growth_bytes", 1
            ),
            "growth",
        ),
        (
            lambda value: value["separate_packed_byte_exact_by_lane"].__setitem__(
                "512", False
            ),
            "byte equivalence",
        ),
        (
            lambda value: value["provenance"].__setitem__(
                "wheel_sha256", "not-a-hash"
            ),
            "wheel_sha256",
        ),
        (
            lambda value: value.__setitem__("receipt_sha256", "0" * 64),
            "receipt digest",
        ),
    ],
)
def test_receipt_validation_fails_closed_on_malformed_results(mutator, match: str) -> None:
    document = _test_double_document()
    mutator(document)
    with pytest.raises(HuRlNativeBenchmarkError, match=match):
        validate_native_batch_benchmark_document(
            document, require_native_engine=False
        )


def test_test_double_receipt_cannot_be_canonicalized_for_persistence() -> None:
    with pytest.raises(HuRlNativeBenchmarkError, match="requires the native engine"):
        canonical_benchmark_json(_test_double_document())


def test_runtime_aborts_on_wrong_actor_sequence() -> None:
    with pytest.raises(HuRlNativeBenchmarkError, match="identity changed"):
        run_native_batch_mechanics_benchmark(
            _env_factory=_WrongActorBatchEnv,
            _memory_reader=_IncreasingMemoryReader(),
        )


def test_seeded_decks_are_complete_reproducible_and_lane_independent() -> None:
    first = generate_benchmark_decks(lane_count=128, seed=7)
    second = generate_benchmark_decks(lane_count=128, seed=7)
    assert first == second
    assert first[0] != first[1]
    assert all(len(deck) == 52 and set(deck) == set(ALL_CARDS) for deck in first)


def test_dependency_free_process_memory_reader_is_positive() -> None:
    sample = read_process_memory()
    assert isinstance(sample, ProcessMemorySample)
    assert sample.rss_bytes > 0
    assert sample.peak_rss_bytes >= sample.rss_bytes
    assert sample.source
