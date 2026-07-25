"""Frozen scientific gate for the Candidate02 performance-lock v4.

The caller must first create ``generic`` with
``merge_hu_m31_t3_step6d_performance_v2.merge_performance_v2`` from immutable
candidate/reference DONE inputs.  This gate then binds that replay to the v4
run contract and independently recomputes coverage, profile assignment,
portable parity, candidate latency percentiles, and paired speedup from the
per-root rows.  Peak RSS remains source-replay owned by the generic merger.

This module is deliberately not the final authority.  Even a metrics pass
keeps quality closed until the production bridge proves that the sealed roots
were package-pinned before launch and that the accepted candidate/reference
jobs came from one source-isolated one-shot lifecycle.  Rerun and reseed are
never opened here.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

from . import merge_hu_m31_t3_step6d_performance_v2 as performance_v2
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from .hu_m31_t3_behavior_roots import behavior_profile_for_index


GATE_SCHEMA = "hu_m31_t3_step6d_performance_lock_v4_gate_v2"
GATE_MODE = "one_shot_performance_lock_v4_source_row_recomputed_gate"

FIRST_P95_SECONDS_MAX = 150.0
FIRST_P99_SECONDS_MAX = 240.0
FIRST_MAX_SECONDS_MAX = 240.0
SECOND_P95_SECONDS_MAX = 5.0
PEAK_RSS_BYTES_MAX = 858_993_459
FIRST_GEOMETRIC_MEAN_SPEEDUP_MIN = 1.55

PASS_DECISION = "performance_lock_v4_metrics_pass_transport_lineage_required"
NO_GO_DECISION = (
    "performance_lock_v4_metrics_no_go_transport_lineage_required"
)

_PARITY_TRUE_KEYS = frozenset(
    {
        "action_keys_exact",
        "child_information_set_count_exact",
        "evaluation_q_exact",
        "portable_payload_exact",
        "rng_exact",
        "selected_action_exact",
        "selection_q_exact",
    }
)
_PARITY_KEYS = _PARITY_TRUE_KEYS | frozenset(
    {"schema", "candidate_portable_sha256", "reference_portable_sha256"}
)
_GATE_KEYS = frozenset(
    {
        "exactly_100_paired_hands",
        "exactly_200_roots",
        "exactly_100_first_and_100_second",
        "exactly_20_hands_per_profile",
        "missing_or_censored_roots_zero",
        "portable_semantic_parity_fraction_one",
        "first_p95_within_150_seconds",
        "first_p99_within_240_seconds",
        "first_max_within_240_seconds",
        "second_p95_within_5_seconds",
        "peak_rss_within_858993459_bytes",
        "first_geometric_mean_speedup_at_least_1_55",
    }
)
_THRESHOLD_KEYS = frozenset(
    {
        "portable_semantic_parity_fraction",
        "first_p95_seconds_max",
        "first_p99_seconds_max",
        "first_max_seconds_max",
        "second_p95_seconds_max",
        "peak_rss_bytes_max",
        "first_geometric_mean_speedup_min",
    }
)
_OBSERVED_KEYS = frozenset(
    {
        "portable_semantic_parity_fraction",
        "first_p95_seconds",
        "first_p99_seconds",
        "first_max_seconds",
        "second_p95_seconds",
        "peak_rss_bytes",
        "first_geometric_mean_speedup",
        "first_speedup_count",
    }
)
_RESULT_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "mode",
        "contract_variant",
        "run_contract_digest",
        "generic_merge_sha256",
        "thresholds",
        "observed",
        "gates",
        "all_gates_passed",
        "scientific_performance_gate_passed",
        "transport_lineage_required",
        "transport_lineage_validated",
        "performance_lock_finalized",
        "performance_lock_qualified",
        "candidate_finalized_no_go",
        "one_shot_lock_consumed",
        "rerun_authorized",
        "reseed_authorized",
        "quality_pilot_authorized",
        "artifact_fanout_authorized",
        "training_eligible",
        "training_authorized",
        "promotion_evidence",
        "promotion_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    }
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_contract_and_generic(
    generic: Mapping[str, Any], run_contract: Mapping[str, Any]
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    contract = runner.validate_run_contract(run_contract)
    if runner.contract_variant(contract) != runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT:
        raise ValueError("performance-lock gate requires the v4 run contract")
    aggregate = _mapping(generic, "generic performance aggregate")
    performance = _mapping(aggregate.get("performance"), "generic performance")
    integrity = _mapping(aggregate.get("integrity"), "generic integrity")
    expected_digest = runner.canonical_sha256(contract)
    source_inputs = _mapping(aggregate.get("source_done_inputs"), "source DONE inputs")
    candidate_inputs = _array(source_inputs.get("candidate"), "candidate DONE inputs")
    reference_inputs = _array(source_inputs.get("reference"), "reference DONE inputs")
    if (
        aggregate.get("schema") != performance_v2.MERGE_SCHEMA
        or aggregate.get("scope") != performance_v2.FULL_PERFORMANCE_SCOPE
        or aggregate.get("run_contract") != contract
        or aggregate.get("run_contract_digest") != expected_digest
        or aggregate.get("budget") != contract["budget"]
        or aggregate.get("allocation") != contract["allocation"]
        or aggregate.get("candidate_library_sha256")
        != contract["candidate_library_sha256"]
        or aggregate.get("reference_library_sha256")
        != contract["reference_library_sha256"]
        or len(candidate_inputs) != 10
        or len(reference_inputs) != 10
    ):
        raise ValueError("generic merge is not bound to the v4 source replay")
    return contract, performance, integrity


def _source_row_replay(
    generic: Mapping[str, Any], run_contract: Mapping[str, Any]
) -> dict[str, Any]:
    contract, performance, integrity = _validate_contract_and_generic(
        generic, run_contract
    )
    paired = _array(generic.get("paired_artifacts"), "paired artifacts")
    if len(paired) != 100:
        raise ValueError("performance-lock v4 requires exactly 100 paired artifacts")

    first_candidate: list[float] = []
    second_candidate: list[float] = []
    first_speedups: list[float] = []
    fingerprints: set[str] = set()
    hand_indices: list[int] = []
    profile_counts: dict[str, int] = {}
    portable_count = 0
    for expected_hand, raw_hand in enumerate(paired):
        hand = _mapping(raw_hand, "paired artifact")
        hand_index = _integer(hand.get("hand_index"), "paired hand index")
        if hand_index != expected_hand:
            raise ValueError("paired hand indices must be ordered unique 0..99")
        expected_profile = behavior_profile_for_index(hand_index)
        if hand.get("profile") != expected_profile:
            raise ValueError("paired hand profile differs from preregistration")
        profile_counts[expected_profile] = profile_counts.get(expected_profile, 0) + 1
        rows = _array(hand.get("rows"), "paired artifact rows")
        if len(rows) != 2:
            raise ValueError("each paired hand must contain first and second rows")
        for offset, raw_row in enumerate(rows):
            row = _mapping(raw_row, "paired artifact row")
            seat = "first" if offset == 0 else "second"
            root_index = _integer(row.get("root_index"), "paired root index")
            if row.get("seat") != seat or root_index != 2 * hand_index + offset:
                raise ValueError("paired root seat/index ordering changed")
            fingerprint = _sha256(
                row.get("observation_fingerprint"), "observation fingerprint"
            )
            if fingerprint in fingerprints:
                raise ValueError("paired observation fingerprint is duplicated")
            fingerprints.add(fingerprint)
            _sha256(row.get("observation_sha256"), "observation SHA-256")
            portable_sha = _sha256(
                row.get("portable_decision_sha256"), "portable decision SHA-256"
            )
            parity = _mapping(row.get("portable_parity"), "portable parity")
            if (
                set(parity) != _PARITY_KEYS
                or parity.get("schema") != "hu_m31_t3_step6d_portable_parity_v1"
                or any(parity.get(key) is not True for key in _PARITY_TRUE_KEYS)
                or parity.get("candidate_portable_sha256") != portable_sha
                or parity.get("reference_portable_sha256") != portable_sha
            ):
                raise ValueError("paired portable semantic parity changed")
            portable_count += 1
            reference_seconds = _finite(
                row.get("reference_solve_wall_seconds"), "reference solve seconds"
            )
            candidate_seconds = _finite(
                row.get("candidate_solve_wall_seconds"), "candidate solve seconds"
            )
            if reference_seconds <= 0.0 or candidate_seconds <= 0.0:
                raise ValueError("paired solve times must be positive")
            speedup = reference_seconds / candidate_seconds
            if _finite(row.get("paired_speedup"), "stored paired speedup") != speedup:
                raise ValueError("stored paired speedup differs from source times")
            if seat == "first":
                first_candidate.append(candidate_seconds)
                first_speedups.append(speedup)
            else:
                second_candidate.append(candidate_seconds)
        hand_indices.append(hand_index)

    if (
        hand_indices != list(runner.CONTRACT_HAND_INDICES)
        or len(fingerprints) != 200
        or len(first_candidate) != 100
        or len(second_candidate) != 100
        or len(first_speedups) != 100
    ):
        raise ValueError("performance-lock v4 paired row coverage changed")

    candidate_by_seat = _mapping(
        performance.get("candidate_by_seat"), "candidate seat performance"
    )
    stored_first = _mapping(candidate_by_seat.get("first"), "stored candidate first")
    stored_second = _mapping(candidate_by_seat.get("second"), "stored candidate second")
    recomputed_first = {
        "count": len(first_candidate),
        "p95_seconds": performance_v2.nearest_rank_percentile(first_candidate, 0.95),
        "p99_seconds": performance_v2.nearest_rank_percentile(first_candidate, 0.99),
        "max_seconds": max(first_candidate),
    }
    recomputed_second = {
        "count": len(second_candidate),
        "p95_seconds": performance_v2.nearest_rank_percentile(second_candidate, 0.95),
    }
    if any(stored_first.get(key) != value for key, value in recomputed_first.items()) or any(
        stored_second.get(key) != value for key, value in recomputed_second.items()
    ):
        raise ValueError("stored candidate latency differs from source rows")
    speedup = performance_v2.geometric_mean(first_speedups)
    if (
        performance.get("first_paired_speedups") != first_speedups
        or _finite(
            performance.get("first_geometric_mean_speedup_diagnostic"),
            "stored first geometric-mean speedup",
        )
        != speedup
    ):
        raise ValueError("stored first speedup differs from source rows")

    expected_integrity = {
        "candidate_hand_count": 100,
        "reference_hand_count": 100,
        "paired_hand_count": 100,
        "paired_root_count": 200,
        "unique_observation_fingerprint_count": 200,
        "paired_hand_parity_count": 100,
        "paired_root_parity_count": portable_count,
        "missing_hand_indices": [],
        "duplicate_hand_indices": [],
        "out_of_contract_hand_indices": [],
    }
    if any(integrity.get(key) != value for key, value in expected_integrity.items()):
        raise ValueError("stored integrity differs from source-row replay")
    if generic.get("hand_indices") != hand_indices:
        raise ValueError("generic hand coverage differs from source rows")

    observed = {
        "portable_semantic_parity_fraction": portable_count / 200,
        "first_p95_seconds": recomputed_first["p95_seconds"],
        "first_p99_seconds": recomputed_first["p99_seconds"],
        "first_max_seconds": recomputed_first["max_seconds"],
        "second_p95_seconds": recomputed_second["p95_seconds"],
        "peak_rss_bytes": _integer(
            performance.get("peak_source_process_rss_bytes"), "peak source-process RSS"
        ),
        "first_geometric_mean_speedup": speedup,
        "first_speedup_count": len(first_speedups),
    }
    return {
        "contract": contract,
        "observed": observed,
        "profile_counts": profile_counts,
        "portable_count": portable_count,
    }


def _build_value(
    generic: Mapping[str, Any], run_contract: Mapping[str, Any]
) -> dict[str, Any]:
    replay = _source_row_replay(generic, run_contract)
    contract = replay["contract"]
    observed = replay["observed"]
    profile_counts = replay["profile_counts"]
    gates = {
        "exactly_100_paired_hands": generic.get("paired_hand_count") == 100,
        "exactly_200_roots": generic.get("root_count") == 200,
        "exactly_100_first_and_100_second": observed["first_speedup_count"] == 100,
        "exactly_20_hands_per_profile": set(profile_counts.values()) == {20},
        "missing_or_censored_roots_zero": True,
        "portable_semantic_parity_fraction_one": (
            observed["portable_semantic_parity_fraction"] == 1.0
        ),
        "first_p95_within_150_seconds": observed["first_p95_seconds"] <= FIRST_P95_SECONDS_MAX,
        "first_p99_within_240_seconds": observed["first_p99_seconds"] <= FIRST_P99_SECONDS_MAX,
        "first_max_within_240_seconds": observed["first_max_seconds"] <= FIRST_MAX_SECONDS_MAX,
        "second_p95_within_5_seconds": observed["second_p95_seconds"] <= SECOND_P95_SECONDS_MAX,
        "peak_rss_within_858993459_bytes": observed["peak_rss_bytes"] <= PEAK_RSS_BYTES_MAX,
        "first_geometric_mean_speedup_at_least_1_55": (
            observed["first_geometric_mean_speedup"] >= FIRST_GEOMETRIC_MEAN_SPEEDUP_MIN
        ),
    }
    if set(observed) != _OBSERVED_KEYS or set(gates) != _GATE_KEYS:
        raise AssertionError("performance-lock v4 gate schema implementation changed")
    thresholds = {
        "portable_semantic_parity_fraction": 1.0,
        "first_p95_seconds_max": FIRST_P95_SECONDS_MAX,
        "first_p99_seconds_max": FIRST_P99_SECONDS_MAX,
        "first_max_seconds_max": FIRST_MAX_SECONDS_MAX,
        "second_p95_seconds_max": SECOND_P95_SECONDS_MAX,
        "peak_rss_bytes_max": PEAK_RSS_BYTES_MAX,
        "first_geometric_mean_speedup_min": FIRST_GEOMETRIC_MEAN_SPEEDUP_MIN,
    }
    passed = all(gates.values())
    value = {
        "schema": GATE_SCHEMA,
        "status": "pass" if passed else "no_go",
        "decision": PASS_DECISION if passed else NO_GO_DECISION,
        "mode": GATE_MODE,
        "contract_variant": runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT,
        "run_contract_digest": runner.canonical_sha256(contract),
        "generic_merge_sha256": runner.canonical_sha256(generic),
        "thresholds": thresholds,
        "observed": observed,
        "gates": gates,
        "all_gates_passed": passed,
        "scientific_performance_gate_passed": passed,
        "transport_lineage_required": True,
        "transport_lineage_validated": False,
        "performance_lock_finalized": False,
        "performance_lock_qualified": False,
        "candidate_finalized_no_go": False,
        "one_shot_lock_consumed": False,
        "rerun_authorized": False,
        "reseed_authorized": False,
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_eligible": False,
        "training_authorized": False,
        "promotion_evidence": False,
        "promotion_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    if set(value) != _RESULT_KEYS:
        raise AssertionError("performance-lock v4 result schema implementation changed")
    return value


def build_performance_lock_v4_gate(
    generic: Mapping[str, Any], *, run_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the one-shot decision from a freshly source-replayed aggregate."""

    value = _build_value(generic, run_contract)
    return validate_performance_lock_v4_gate_value(
        value, generic=generic, run_contract=run_contract
    )


def _validate_stored_types(value: Mapping[str, Any]) -> None:
    thresholds = _mapping(value.get("thresholds"), "v4 gate thresholds")
    observed = _mapping(value.get("observed"), "v4 gate observed values")
    gates = _mapping(value.get("gates"), "v4 gate decisions")
    if (
        set(thresholds) != _THRESHOLD_KEYS
        or set(observed) != _OBSERVED_KEYS
        or set(gates) != _GATE_KEYS
        or any(type(gate) is not bool for gate in gates.values())
        or type(value.get("all_gates_passed")) is not bool
    ):
        raise ValueError("performance-lock v4 stored gate types changed")
    boundary_booleans = (
        "scientific_performance_gate_passed",
        "transport_lineage_required",
        "transport_lineage_validated",
        "performance_lock_finalized",
        "performance_lock_qualified",
        "candidate_finalized_no_go",
        "one_shot_lock_consumed",
        "rerun_authorized",
        "reseed_authorized",
        "quality_pilot_authorized",
        "artifact_fanout_authorized",
        "training_eligible",
        "training_authorized",
        "promotion_evidence",
        "promotion_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    )
    if any(type(value.get(field)) is not bool for field in boundary_booleans):
        raise ValueError("performance-lock v4 boundary flags must be booleans")
    for key, item in thresholds.items():
        if key == "peak_rss_bytes_max":
            _integer(item, f"threshold {key}")
        elif type(item) is not float or not math.isfinite(item):
            raise ValueError(f"threshold {key} must be a finite float")
    for key, item in observed.items():
        if key in {"peak_rss_bytes", "first_speedup_count"}:
            _integer(item, f"observed {key}")
        elif type(item) is not float or not math.isfinite(item):
            raise ValueError(f"observed {key} must be a finite float")


def validate_performance_lock_v4_gate_value(
    value: Mapping[str, Any], *, generic: Mapping[str, Any], run_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject a stored decision that differs from fresh source-row replay."""

    if not isinstance(value, Mapping):
        raise ValueError("performance-lock v4 gate must be an object")
    result = deepcopy(dict(value))
    if set(result) != _RESULT_KEYS:
        raise ValueError("performance-lock v4 gate fields changed")
    _validate_stored_types(result)
    expected = _build_value(generic, run_contract)
    if result != expected:
        raise ValueError("performance-lock v4 gate differs from source replay")
    return result


__all__ = [
    "FIRST_GEOMETRIC_MEAN_SPEEDUP_MIN",
    "FIRST_MAX_SECONDS_MAX",
    "FIRST_P95_SECONDS_MAX",
    "FIRST_P99_SECONDS_MAX",
    "GATE_MODE",
    "GATE_SCHEMA",
    "NO_GO_DECISION",
    "PASS_DECISION",
    "PEAK_RSS_BYTES_MAX",
    "SECOND_P95_SECONDS_MAX",
    "build_performance_lock_v4_gate",
    "validate_performance_lock_v4_gate_value",
]
