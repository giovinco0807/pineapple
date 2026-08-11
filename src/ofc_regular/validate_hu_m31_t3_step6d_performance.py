"""Independently validate M3.1 Step 6d performance-development artifacts.

The runner's aggregate fields and booleans are not trusted.  This validator
loads every canonical root and hand artifact, checks their hash/provenance,
reconstructs the legal ActionKey/index mapping, compares every candidate and
reference Q value, and recomputes the aggregate with nearest-rank percentiles.

Only the complete, frozen 100-paired-hand development set receives a gate.
Any smaller subset is an integrity smoke with a ``not_applicable`` result.
Neither outcome authorizes quality labels, training, profile activation, or a
change to ``current``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    action_key_from_payload,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import generate_turn_actions
from .hu_belief import HIDDEN_CARD_PRIOR
from .hu_infoset import ActorObservation
from .hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    behavior_profile_for_index,
)
from .hu_m31_t3_runtime import (
    HU_M31_T3_ENGINE_VERSION,
    HU_M31_T3_RUNTIME_ID,
    HU_M31_T3_RUNTIME_SCHEMA,
    HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA,
)
from .hu_m31_t3_step6d_contract import (
    EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
    SEED_STRIDE,
    STEP6D_RUN_ID,
    schedule_by_name,
)


STEP6D_PERFORMANCE_ROOT_SCHEMA = "hu_m31_t3_step6d_performance_root_v1"
STEP6D_PERFORMANCE_HAND_SCHEMA = "hu_m31_t3_step6d_performance_hand_v1"
STEP6D_PERFORMANCE_SUMMARY_SCHEMA = "hu_m31_t3_step6d_performance_summary_v1"
STEP6D_PERFORMANCE_RUN_CONTRACT_SCHEMA = (
    "hu_m31_t3_step6d_performance_run_contract_v1"
)
STEP6D_PERFORMANCE_VALIDATION_SCHEMA = (
    "hu_m31_t3_step6d_performance_validation_v1"
)
STEP6D_PORTABLE_PARITY_SCHEMA = "hu_m31_t3_step6d_portable_parity_v1"

PERFORMANCE_DEVELOPMENT_SCHEDULE = "performance_development"
FULL_SCOPE = "full_performance_development"
SUBSET_SMOKE_SCOPE = "subset_smoke"
PERCENTILE_METHOD = "nearest_rank_ceiling_n_times_p"

FULL_HAND_COUNT = 100
FULL_ROOT_COUNT = 200
FULL_SEAT_ROOT_COUNT = 100
FULL_HANDS_PER_PROFILE = 20
MAX_FIRST_P95_SECONDS = 150.0
MAX_FIRST_P99_SECONDS = 240.0
MAX_FIRST_SECONDS = 240.0
MAX_SECOND_P95_SECONDS = 5.0
MAX_PEAK_RSS_BYTES = 858_993_459

REFERENCE_NATIVE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
SEARCH_BUDGET = {
    "candidate_samples": 8,
    "evaluation_samples": 32,
    "downstream_t3_samples": 4,
    "downstream_t4_samples": 0,
}
ALLOWED_ALLOCATIONS = {(1, 16), (2, 8), (4, 4)}

_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "status",
        "contract_canonical_sha256",
        "schedule",
        "scope",
        "run_contract",
        "run_contract_digest",
        "subset_smoke",
        "hand_indices",
        "paired_hand_count",
        "root_count",
        "profile_hand_counts",
        "budget",
        "allocation",
        "reference_library_sha256",
        "candidate_library_sha256",
        "portable_parity",
        "performance",
        "geometry",
        "resumed_hand_count",
        "integrity_gates",
        "candidate_freeze_gates_applied",
        "candidate_freeze_gates",
        "all_gates_passed",
        "candidate_freeze_ready",
        "missing_or_censored_root_indices",
        "hand_manifest",
        "teacher_value_status",
        "training_eligible",
        "threshold_eligible",
        "quality_evidence",
        "promotion_evidence",
        "performance_lock_opened",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "cloud_started",
    }
)
_RUN_CONTRACT_KEYS = frozenset(
    {
        "schema",
        "step6d_run_id",
        "contract_canonical_sha256",
        "schedule",
        "hand_indices",
        "budget",
        "allocation",
        "reference_library_sha256",
        "candidate_library_sha256",
        "execution_mode",
        "source_order",
        "training_eligible",
    }
)
_MANIFEST_KEYS = frozenset({"hand_index", "path", "sha256", "bytes"})
_ROOT_KEYS = frozenset(
    {
        "schema",
        "contract_canonical_sha256",
        "schedule",
        "schedule_row_sha256",
        "hand_index",
        "root_indices",
        "profile",
        "seeds",
        "budget",
        "observations",
        "current_profile_resolved",
        "opponent_private_discards_used",
        "training_eligible",
    }
)
_ROOT_OBSERVATION_KEYS = frozenset(
    {"root_index", "seat", "observation_fingerprint", "observation"}
)
_HAND_KEYS = frozenset(
    {
        "schema",
        "contract_canonical_sha256",
        "schedule",
        "hand_index",
        "profile",
        "seeds",
        "budget",
        "run_contract_digest",
        "root_artifact_sha256",
        "allocation",
        "reference_library_sha256",
        "candidate_library_sha256",
        "queue_seconds",
        "worker_wall_seconds",
        "process_id",
        "memory",
        "rows",
        "portable_parity_exact",
        "teacher_value_status",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "cloud_started",
    }
)
_ROOT_RESULT_KEYS = frozenset(
    {
        "root_index",
        "seat",
        "observation_fingerprint",
        "solve_order",
        "geometry",
        "reference",
        "candidate",
        "parity",
        "wall_seconds",
    }
)
_SOURCE_RESULT_KEYS = frozenset(
    {
        "source",
        "native_library_sha256",
        "solve_wall_seconds",
        "native_seconds",
        "validation_seconds",
        "runtime_total_seconds",
        "rss_after",
        "decision",
    }
)
_PARITY_KEYS = frozenset(
    {
        "schema",
        "reference_portable_sha256",
        "candidate_portable_sha256",
        "action_keys_exact",
        "selection_q_exact",
        "evaluation_q_exact",
        "selected_action_exact",
        "rng_exact",
        "child_information_set_count_exact",
        "portable_payload_exact",
    }
)
_GEOMETRY_KEYS = frozenset(
    {
        "hero_open_slots",
        "opponent_open_slots",
        "dealt_card_count",
        "legal_action_count",
        "child_information_set_count",
    }
)
_OPEN_SLOT_KEYS = frozenset({"top", "middle", "bottom"})
_HAND_MEMORY_KEYS = frozenset(
    {"before_solver_load", "after_solver_load", "after_hand", "peak_rss_bytes"}
)
_MEMORY_KEYS = frozenset(
    {"supported", "source", "rss_bytes", "peak_rss_bytes", "private_bytes"}
)
_DECISION_KEYS = frozenset(
    {
        "schema",
        "runtime_id",
        "seat",
        "value_scope",
        "observation_fingerprint",
        "selected_action_key",
        "selected_selection_ev",
        "selected_evaluation_ev",
        "selection_gap",
        "evaluation_sample_regret",
        "selected_action",
        "action_key_schema",
        "legal_action_set_digest",
        "legal_action_order_digest",
        "action_values",
        "belief_prior",
        "candidate_belief_digest",
        "evaluation_belief_digest",
        "candidate_rng_digest",
        "evaluation_rng_digest",
        "candidate_samples",
        "evaluation_samples",
        "downstream_t3_samples",
        "downstream_t4_samples",
        "run_id",
        "continuation_seed",
        "candidate_seed",
        "evaluation_seed",
        "use_t4_action_cache",
        "continuation_policy_id",
        "strategy_fusion_guard",
        "search_contract_digest",
        "downstream_t4_native_semantics_id",
        "downstream_t4_native_anchor",
        "downstream_t4_mode",
        "child_information_set_count",
        "solver_id",
        "engine_version",
        "native_library_sha256",
        "teacher_value_status",
        "native_latency_ms",
        "validation_latency_ms",
        "total_latency_ms",
        "execution_mode",
        "batch_size",
        "semantic_result_digest_schema",
        "semantic_result_digest_scope",
        "semantic_result_digest",
        "result_digest_scope",
        "result_digest",
    }
)
_ACTION_VALUE_KEYS = frozenset(
    {
        "original_index",
        "rank",
        "action_key",
        "selection_ev",
        "evaluation_ev",
        "evaluation_regret",
        "placements",
        "discards",
    }
)
_SELECTED_ACTION_KEYS = frozenset({"placements", "discards"})
_GATE_KEYS = (
    "exactly_100_paired_hands",
    "exactly_200_roots",
    "exactly_100_first_and_100_second",
    "exactly_20_hands_per_profile",
    "portable_semantic_parity_fraction_one",
    "missing_or_censored_roots_zero",
    "first_p95_within_150_seconds",
    "first_p99_within_240_seconds",
    "first_max_within_240_seconds",
    "second_p95_within_5_seconds",
    "peak_rss_within_858993459_bytes",
)


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} fields changed: missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
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


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} must be finite and in range")
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _artifact_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _artifact_digest(value: Any) -> str:
    return hashlib.sha256(_artifact_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or raw != _artifact_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def expected_profile(hand_index: int) -> str:
    _checked_hand_index(hand_index)
    return behavior_profile_for_index(hand_index)


def expected_seeds(hand_index: int) -> dict[str, int]:
    _checked_hand_index(hand_index)
    schedule = schedule_by_name(PERFORMANCE_DEVELOPMENT_SCHEDULE)
    if schedule.index_count != FULL_HAND_COUNT:
        raise ValueError("Step 6d performance-development seed schedule changed")
    return {
        key: base + SEED_STRIDE * hand_index
        for key, base in zip(
            schedule.namespace_keys, schedule.namespace_bases, strict=True
        )
    }


def expected_schedule_row(hand_index: int) -> dict[str, Any]:
    return {
        "schedule": PERFORMANCE_DEVELOPMENT_SCHEDULE,
        "hand_index": _checked_hand_index(hand_index),
        "root_indices": [hand_index * 2, hand_index * 2 + 1],
        "profile": expected_profile(hand_index),
        "seeds": expected_seeds(hand_index),
        "budget": dict(SEARCH_BUDGET),
        "training_eligible": False,
    }


def _checked_hand_index(value: Any) -> int:
    index = _integer(value, "Step 6d hand index")
    if index >= FULL_HAND_COUNT:
        raise ValueError("Step 6d hand index is outside 0..99")
    return index


def _validate_sorted_indices(
    value: Any, label: str, *, maximum: int
) -> list[int]:
    rows = _array(value, label)
    result = [_integer(item, label) for item in rows]
    if any(item > maximum for item in result):
        raise ValueError(f"{label} contains an out-of-range index")
    if result != sorted(result) or len(result) != len(set(result)):
        raise ValueError(f"{label} must be sorted and duplicate-free")
    return result


def _validate_run_contract(
    value: Mapping[str, Any], *, hand_indices: Sequence[int]
) -> tuple[int, int, str]:
    _require_exact_keys(value, _RUN_CONTRACT_KEYS, "Step 6d run contract")
    allocation = _mapping(value.get("allocation"), "Step 6d run allocation")
    _require_exact_keys(
        allocation,
        frozenset({"workers", "rayon_threads_per_worker"}),
        "Step 6d run allocation",
    )
    workers = _integer(allocation.get("workers"), "Step 6d workers", minimum=1)
    threads = _integer(
        allocation.get("rayon_threads_per_worker"),
        "Step 6d Rayon threads",
        minimum=1,
    )
    candidate_sha256 = _require_sha256(
        value.get("candidate_library_sha256"), "candidate library"
    )
    if (
        (workers, threads) not in ALLOWED_ALLOCATIONS
        or value.get("schema") != STEP6D_PERFORMANCE_RUN_CONTRACT_SCHEMA
        or value.get("step6d_run_id") != STEP6D_RUN_ID
        or value.get("contract_canonical_sha256")
        != EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
        or value.get("schedule") != PERFORMANCE_DEVELOPMENT_SCHEDULE
        or value.get("hand_indices") != list(hand_indices)
        or value.get("budget") != SEARCH_BUDGET
        or value.get("reference_library_sha256")
        != REFERENCE_NATIVE_LIBRARY_SHA256
        or value.get("execution_mode")
        != "two_independent_scalar_solves_per_root"
        or value.get("source_order")
        != "reference_first_on_first_root_candidate_first_on_second_root"
        or value.get("training_eligible") is not False
    ):
        raise ValueError("Step 6d run contract changed")
    return workers, threads, candidate_sha256


def _validate_memory_snapshot(value: Any, label: str) -> int:
    snapshot = _mapping(value, label)
    _require_exact_keys(snapshot, _MEMORY_KEYS, label)
    if snapshot.get("supported") is not True:
        raise ValueError(f"{label} must support the RSS gate")
    if not isinstance(snapshot.get("source"), str) or not snapshot["source"]:
        raise ValueError(f"{label} source changed")
    for key in ("rss_bytes", "private_bytes"):
        raw = snapshot.get(key)
        if raw is not None:
            _integer(raw, f"{label} {key}")
    return _integer(snapshot.get("peak_rss_bytes"), f"{label} peak RSS")


def _validate_root(
    value: Mapping[str, Any], *, hand_index: int
) -> tuple[ActorObservation, ActorObservation]:
    _require_exact_keys(value, _ROOT_KEYS, "Step 6d root artifact")
    schedule_row = expected_schedule_row(hand_index)
    rows = _array(value.get("observations"), "Step 6d root observations")
    if (
        value.get("schema") != STEP6D_PERFORMANCE_ROOT_SCHEMA
        or value.get("contract_canonical_sha256")
        != EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
        or value.get("schedule") != PERFORMANCE_DEVELOPMENT_SCHEDULE
        or value.get("schedule_row_sha256") != _artifact_digest(schedule_row)
        or value.get("hand_index") != hand_index
        or value.get("root_indices") != schedule_row["root_indices"]
        or value.get("profile") != schedule_row["profile"]
        or value.get("seeds") != schedule_row["seeds"]
        or value.get("budget") != SEARCH_BUDGET
        or value.get("current_profile_resolved") is not False
        or value.get("opponent_private_discards_used") is not False
        or value.get("training_eligible") is not False
        or len(rows) != 2
    ):
        raise ValueError(f"Step 6d root seed/profile/index contract changed: {hand_index}")
    observations: list[ActorObservation] = []
    for offset, raw in enumerate(rows):
        row = _mapping(raw, "Step 6d root observation row")
        _require_exact_keys(
            row, _ROOT_OBSERVATION_KEYS, "Step 6d root observation row"
        )
        root_index = hand_index * 2 + offset
        seat = "first" if offset == 0 else "second"
        observation = ActorObservation.from_dict(
            _mapping(row.get("observation"), "Step 6d observation")
        )
        if (
            row.get("root_index") != root_index
            or row.get("seat") != seat
            or observation.seat != seat
            or observation.to_act_order != seat
            or observation.street != "T3"
            or row.get("observation_fingerprint") != observation.fingerprint()
        ):
            raise ValueError("Step 6d root observation/index/fingerprint mismatch")
        observations.append(observation)
    return observations[0], observations[1]


def _validate_action_payload(
    payload: Mapping[str, Any], token: Any, label: str
) -> None:
    _require_exact_keys(payload, _SELECTED_ACTION_KEYS, label)
    if not isinstance(token, str):
        raise ValueError(f"{label} ActionKey is missing")
    try:
        expected = ActionKey.from_token(token)
        observed = action_key_from_payload(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} action payload is invalid") from error
    if expected != observed:
        raise ValueError(f"{label} ActionKey/payload mismatch")


def _validate_decision(
    value: Mapping[str, Any],
    *,
    label: str,
    observation: ActorObservation,
    seeds: Mapping[str, int],
    native_sha256: str,
) -> list[Mapping[str, Any]]:
    _require_exact_keys(value, _DECISION_KEYS, label)
    static = {
        "schema": HU_M31_T3_RUNTIME_SCHEMA,
        "runtime_id": HU_M31_T3_RUNTIME_ID,
        "seat": observation.seat,
        "observation_fingerprint": observation.fingerprint(),
        "action_key_schema": ACTION_KEY_SCHEMA,
        "belief_prior": HIDDEN_CARD_PRIOR,
        "candidate_samples": SEARCH_BUDGET["candidate_samples"],
        "evaluation_samples": SEARCH_BUDGET["evaluation_samples"],
        "downstream_t3_samples": SEARCH_BUDGET["downstream_t3_samples"],
        "downstream_t4_samples": SEARCH_BUDGET["downstream_t4_samples"],
        "run_id": STEP6D_RUN_ID,
        "continuation_seed": seeds["child"],
        "candidate_seed": seeds["candidate"],
        "evaluation_seed": seeds["evaluation"],
        "use_t4_action_cache": True,
        "continuation_policy_id": "local_infoset_response_t3_second_t4_v1",
        "strategy_fusion_guard": "child_actions_keyed_only_by_actor_observation",
        "downstream_t4_native_semantics_id": "m30_exact_t4_native_kernel_semantics_v1",
        "downstream_t4_native_anchor": "same_pinned_m30_native_engine",
        "downstream_t4_mode": "exact",
        "solver_id": "rust_crn_sequential_t3_v1",
        "engine_version": HU_M31_T3_ENGINE_VERSION,
        "native_library_sha256": native_sha256,
        "teacher_value_status": "diagnostic_not_match_EV",
        "execution_mode": "scalar",
        "batch_size": 1,
        "semantic_result_digest_schema": HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA,
        "semantic_result_digest_scope": "dealt_order_independent_action_value_result",
        "result_digest_scope": "ordered_action_mapping_bound",
    }
    if any(value.get(key) != expected for key, expected in static.items()):
        raise ValueError(f"{label} frozen decision contract mismatch")
    for key in (
        "legal_action_set_digest",
        "legal_action_order_digest",
        "candidate_belief_digest",
        "evaluation_belief_digest",
        "candidate_rng_digest",
        "evaluation_rng_digest",
        "search_contract_digest",
        "semantic_result_digest",
        "result_digest",
    ):
        _require_sha256(value.get(key), f"{label} {key}")
    for key in (
        "selected_selection_ev",
        "selected_evaluation_ev",
        "selection_gap",
        "evaluation_sample_regret",
    ):
        _finite(value.get(key), f"{label} {key}")
    for key in ("native_latency_ms", "validation_latency_ms", "total_latency_ms"):
        _finite(value.get(key), f"{label} {key}", minimum=0.0)
    _integer(
        value.get("child_information_set_count"),
        f"{label} child information-set count",
    )
    if not isinstance(value.get("value_scope"), str) or not value["value_scope"]:
        raise ValueError(f"{label} value scope changed")

    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    expected_by_key = {
        action_key(action).to_token(): index for index, action in enumerate(legal)
    }
    raw_actions = _array(value.get("action_values"), f"{label} action values")
    if len(raw_actions) != len(legal):
        raise ValueError(f"{label} legal action/Q coverage mismatch")
    actions: list[Mapping[str, Any]] = []
    for raw in raw_actions:
        row = _mapping(raw, f"{label} action row")
        _require_exact_keys(row, _ACTION_VALUE_KEYS, f"{label} action row")
        token = row.get("action_key")
        _validate_action_payload(
            {"placements": row["placements"], "discards": row["discards"]},
            token,
            f"{label} action row",
        )
        original_index = _integer(
            row.get("original_index"), f"{label} original index"
        )
        rank = _integer(row.get("rank"), f"{label} rank")
        for key in ("selection_ev", "evaluation_ev", "evaluation_regret"):
            _finite(row.get(key), f"{label} {key}")
        if expected_by_key.get(str(token)) != original_index:
            raise ValueError(f"{label} action index mismatch")
        if rank >= len(legal):
            raise ValueError(f"{label} action rank mismatch")
        actions.append(row)
    tokens = [str(row["action_key"]) for row in actions]
    ranks = [int(row["rank"]) for row in actions]
    if (
        len(set(tokens)) != len(legal)
        or set(ranks) != set(range(len(legal)))
        or tokens
        != sorted(tokens, key=lambda token: ActionKey.from_token(token).sort_key())
        or value.get("legal_action_set_digest") != legal_action_set_digest(legal)
        or value.get("legal_action_order_digest")
        != ordered_action_mapping_digest(legal)
    ):
        raise ValueError(f"{label} action index/order mapping mismatch")
    selected_token = value.get("selected_action_key")
    _validate_action_payload(
        _mapping(value.get("selected_action"), f"{label} selected action"),
        selected_token,
        f"{label} selected action",
    )
    selected = [row for row in actions if row["rank"] == 0]
    if (
        len(selected) != 1
        or selected[0]["action_key"] != selected_token
        or selected[0]["selection_ev"] != value.get("selected_selection_ev")
        or selected[0]["evaluation_ev"] != value.get("selected_evaluation_ev")
    ):
        raise ValueError(f"{label} selected action/index/Q mismatch")
    return actions


def _runner_portable_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "action_key": row["action_key"],
            "rank": row["rank"],
            "selection_q": float(row["selection_ev"]),
            "evaluation_q": float(row["evaluation_ev"]),
        }
        for row in value["action_values"]
    ]
    rows.sort(key=lambda row: ActionKey.from_token(row["action_key"]).sort_key())
    selected = str(value["selected_action_key"])
    by_key = {row["action_key"]: row for row in rows}
    key = ActionKey.from_token(selected)
    return {
        "schema": STEP6D_PORTABLE_PARITY_SCHEMA,
        "seat": value["seat"],
        "observation_fingerprint": value["observation_fingerprint"],
        "action_key_schema": value["action_key_schema"],
        "legal_action_set_digest": value["legal_action_set_digest"],
        "action_values": rows,
        "selected_action_key": selected,
        "selected_action": {
            "action_key": selected,
            "top": list(key.cards("top")),
            "middle": list(key.cards("middle")),
            "bottom": list(key.cards("bottom")),
            "discards": list(key.cards("discards")),
        },
        "selected_selection_q": by_key[selected]["selection_q"],
        "selected_evaluation_q": by_key[selected]["evaluation_q"],
        "rng": {
            "run_id": value["run_id"],
            "continuation_seed": value["continuation_seed"],
            "candidate_seed": value["candidate_seed"],
            "evaluation_seed": value["evaluation_seed"],
            "candidate_belief_digest": value["candidate_belief_digest"],
            "evaluation_belief_digest": value["evaluation_belief_digest"],
            "candidate_rng_digest": value["candidate_rng_digest"],
            "evaluation_rng_digest": value["evaluation_rng_digest"],
            "search_contract_digest": value["search_contract_digest"],
        },
        "budget": dict(SEARCH_BUDGET),
        "child_information_set_count": value["child_information_set_count"],
    }


def _portable_pair(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    reference_rows = reference["action_values"]
    candidate_rows = candidate["action_values"]
    reference_mapping = [
        (row["action_key"], row["original_index"], row["rank"])
        for row in reference_rows
    ]
    candidate_mapping = [
        (row["action_key"], row["original_index"], row["rank"])
        for row in candidate_rows
    ]
    if reference_mapping != candidate_mapping:
        raise ValueError("Step 6d portable action index mismatch")
    reference_q = [
        (
            row["action_key"],
            row["selection_ev"],
            row["evaluation_ev"],
            row["evaluation_regret"],
        )
        for row in reference_rows
    ]
    candidate_q = [
        (
            row["action_key"],
            row["selection_ev"],
            row["evaluation_ev"],
            row["evaluation_regret"],
        )
        for row in candidate_rows
    ]
    if reference_q != candidate_q:
        raise ValueError("Step 6d portable action Q mismatch")
    portable_reference = deepcopy(dict(reference))
    portable_candidate = deepcopy(dict(candidate))
    for payload in (portable_reference, portable_candidate):
        for key in (
            "native_library_sha256",
            "native_latency_ms",
            "validation_latency_ms",
            "total_latency_ms",
            "semantic_result_digest",
            "result_digest",
        ):
            payload.pop(key, None)
    if portable_reference != portable_candidate:
        raise ValueError("Step 6d portable semantic decision mismatch")

    reference_payload = _runner_portable_payload(reference)
    candidate_payload = _runner_portable_payload(candidate)
    exact = reference_payload == candidate_payload
    return {
        "schema": STEP6D_PORTABLE_PARITY_SCHEMA,
        "reference_portable_sha256": _artifact_digest(reference_payload),
        "candidate_portable_sha256": _artifact_digest(candidate_payload),
        "action_keys_exact": True,
        "selection_q_exact": True,
        "evaluation_q_exact": True,
        "selected_action_exact": (
            reference_payload["selected_action"]
            == candidate_payload["selected_action"]
        ),
        "rng_exact": reference_payload["rng"] == candidate_payload["rng"],
        "child_information_set_count_exact": (
            reference_payload["child_information_set_count"]
            == candidate_payload["child_information_set_count"]
        ),
        "portable_payload_exact": exact,
    }


def _validate_source(
    value: Mapping[str, Any],
    *,
    source: str,
    observation: ActorObservation,
    seeds: Mapping[str, int],
    native_sha256: str,
) -> Mapping[str, Any]:
    _require_exact_keys(value, _SOURCE_RESULT_KEYS, f"Step 6d {source} result")
    if (
        value.get("source") != source
        or value.get("native_library_sha256") != native_sha256
    ):
        raise ValueError(f"Step 6d {source} source/native mismatch")
    for field in (
        "solve_wall_seconds",
        "native_seconds",
        "validation_seconds",
        "runtime_total_seconds",
    ):
        _finite(value.get(field), f"Step 6d {source} {field}", minimum=0.0)
    _validate_memory_snapshot(value.get("rss_after"), f"Step 6d {source} RSS")
    decision = _mapping(value.get("decision"), f"Step 6d {source} decision")
    _validate_decision(
        decision,
        label=f"Step 6d {source} decision",
        observation=observation,
        seeds=seeds,
        native_sha256=native_sha256,
    )
    if (
        value.get("native_seconds") != float(decision["native_latency_ms"]) / 1000.0
        or value.get("validation_seconds")
        != float(decision["validation_latency_ms"]) / 1000.0
        or value.get("runtime_total_seconds")
        != float(decision["total_latency_ms"]) / 1000.0
    ):
        raise ValueError(f"Step 6d {source} timing/decision mismatch")
    return decision


def _expected_geometry(
    observation: ActorObservation, decision: Mapping[str, Any]
) -> dict[str, Any]:
    def slots(board: Any) -> dict[str, int]:
        return {
            "top": 3 - len(board.top),
            "middle": 5 - len(board.middle),
            "bottom": 5 - len(board.bottom),
        }

    return {
        "hero_open_slots": slots(observation.hero_board),
        "opponent_open_slots": slots(observation.opponent_public_board),
        "dealt_card_count": len(observation.dealt_cards),
        "legal_action_count": len(decision["action_values"]),
        "child_information_set_count": decision["child_information_set_count"],
    }


def _validate_geometry(value: Any, expected: Mapping[str, Any]) -> None:
    geometry = _mapping(value, "Step 6d geometry")
    _require_exact_keys(geometry, _GEOMETRY_KEYS, "Step 6d geometry")
    for key in ("hero_open_slots", "opponent_open_slots"):
        slots = _mapping(geometry.get(key), f"Step 6d {key}")
        _require_exact_keys(slots, _OPEN_SLOT_KEYS, f"Step 6d {key}")
        for count in slots.values():
            _integer(count, f"Step 6d {key} count")
    if geometry != expected:
        raise ValueError("Step 6d board/action geometry mismatch")


def _validate_hand(
    value: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
    observations: Sequence[ActorObservation],
    hand_index: int,
    run_contract_digest: str,
    workers: int,
    threads: int,
    candidate_sha256: str,
) -> list[dict[str, Any]]:
    _require_exact_keys(value, _HAND_KEYS, "Step 6d hand result")
    schedule_row = expected_schedule_row(hand_index)
    memory = _mapping(value.get("memory"), "Step 6d hand memory")
    _require_exact_keys(memory, _HAND_MEMORY_KEYS, "Step 6d hand memory")
    memory_peaks = [
        _validate_memory_snapshot(memory.get(key), f"Step 6d hand memory {key}")
        for key in ("before_solver_load", "after_solver_load", "after_hand")
    ]
    hand_peak = _integer(memory.get("peak_rss_bytes"), "Step 6d hand peak RSS")
    rows = _array(value.get("rows"), "Step 6d hand rows")
    if (
        hand_peak != max(memory_peaks)
        or value.get("schema") != STEP6D_PERFORMANCE_HAND_SCHEMA
        or value.get("contract_canonical_sha256")
        != EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
        or value.get("schedule") != PERFORMANCE_DEVELOPMENT_SCHEDULE
        or value.get("hand_index") != hand_index
        or value.get("profile") != schedule_row["profile"]
        or value.get("seeds") != schedule_row["seeds"]
        or value.get("budget") != SEARCH_BUDGET
        or value.get("run_contract_digest") != run_contract_digest
        or value.get("root_artifact_sha256") != _artifact_digest(root)
        or value.get("allocation")
        != {"workers": workers, "rayon_threads_per_worker": threads}
        or value.get("reference_library_sha256")
        != REFERENCE_NATIVE_LIBRARY_SHA256
        or value.get("candidate_library_sha256") != candidate_sha256
        or value.get("teacher_value_status") != "diagnostic_not_match_EV"
        or any(
            value.get(key) is not False
            for key in (
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
                "cloud_started",
            )
        )
        or len(rows) != 2
    ):
        raise ValueError(f"Step 6d hand seed/profile/provenance changed: {hand_index}")
    _integer(value.get("process_id"), "Step 6d process id", minimum=1)
    _finite(value.get("queue_seconds"), "Step 6d queue seconds", minimum=0.0)
    _finite(
        value.get("worker_wall_seconds"),
        "Step 6d worker wall seconds",
        minimum=0.0,
    )

    aggregate_rows: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for offset, raw in enumerate(rows):
        row = _mapping(raw, "Step 6d root result")
        _require_exact_keys(row, _ROOT_RESULT_KEYS, "Step 6d root result")
        root_index = hand_index * 2 + offset
        observation = observations[offset]
        expected_order = (
            ["reference", "candidate"]
            if offset == 0
            else ["candidate", "reference"]
        )
        fingerprint = observation.fingerprint()
        if (
            row.get("root_index") != root_index
            or row.get("seat") != observation.seat
            or row.get("observation_fingerprint") != fingerprint
            or row.get("solve_order") != expected_order
            or fingerprint in fingerprints
        ):
            raise ValueError("Step 6d result root/seat/index mismatch")
        fingerprints.add(fingerprint)
        _finite(row.get("wall_seconds"), "Step 6d root wall seconds", minimum=0.0)
        reference = _validate_source(
            _mapping(row.get("reference"), "Step 6d reference result"),
            source="reference",
            observation=observation,
            seeds=schedule_row["seeds"],
            native_sha256=REFERENCE_NATIVE_LIBRARY_SHA256,
        )
        candidate_source = _mapping(row.get("candidate"), "Step 6d candidate result")
        candidate = _validate_source(
            candidate_source,
            source="candidate",
            observation=observation,
            seeds=schedule_row["seeds"],
            native_sha256=candidate_sha256,
        )
        expected_parity = _portable_pair(reference, candidate)
        parity = _mapping(row.get("parity"), "Step 6d parity")
        _require_exact_keys(parity, _PARITY_KEYS, "Step 6d parity")
        if parity != expected_parity or expected_parity["portable_payload_exact"] is not True:
            raise ValueError("Step 6d portable parity evidence mismatch")
        _validate_geometry(
            row.get("geometry"), _expected_geometry(observation, candidate)
        )
        aggregate_rows.append(
            {
                "root_index": root_index,
                "hand_index": hand_index,
                "seat": observation.seat,
                "profile": schedule_row["profile"],
                "observation_fingerprint": fingerprint,
                "geometry": dict(row["geometry"]),
                "reference": dict(row["reference"]),
                "candidate": dict(candidate_source),
                "peak_rss_bytes": hand_peak,
            }
        )
    if value.get("portable_parity_exact") is not True:
        raise ValueError("Step 6d hand portable parity status mismatch")
    return aggregate_rows


def nearest_rank_percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("nearest-rank percentile requires values")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("nearest-rank fraction must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def _runner_percentiles(values: Sequence[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    return {
        "count": len(rows),
        "mean_seconds": statistics.fmean(rows),
        "p50_seconds": nearest_rank_percentile(rows, 0.50),
        "p95_seconds": nearest_rank_percentile(rows, 0.95),
        "p99_seconds": nearest_rank_percentile(rows, 0.99),
        "max_seconds": max(rows),
    }


def _validation_latency(values: Sequence[float]) -> dict[str, Any]:
    return {**_runner_percentiles(values), "percentile_method": PERCENTILE_METHOD}


def _source_performance(
    reports: Sequence[Mapping[str, Any]], *, source: str
) -> dict[str, Any]:
    return {
        seat: {
            metric: _runner_percentiles(
                [
                    float(row[source][metric])
                    for report in reports
                    for row in report["rows"]
                    if row["seat"] == seat
                ]
            )
            for metric in (
                "solve_wall_seconds",
                "native_seconds",
                "validation_seconds",
                "runtime_total_seconds",
            )
        }
        for seat in ("first", "second")
    }


def _safe_manifest_hand(
    summary_dir: Path, entry: Mapping[str, Any], expected_index: int
) -> tuple[Path, dict[str, Any]]:
    _require_exact_keys(entry, _MANIFEST_KEYS, "Step 6d hand manifest entry")
    expected_relative = f"hands/hand_{expected_index:03d}.json"
    if entry.get("hand_index") != expected_index or entry.get("path") != expected_relative:
        raise ValueError("Step 6d hand manifest index/path mismatch")
    expected_sha = _require_sha256(entry.get("sha256"), "Step 6d hand manifest")
    expected_bytes = _integer(entry.get("bytes"), "Step 6d hand bytes", minimum=1)
    path = summary_dir / "hands" / f"hand_{expected_index:03d}.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Step 6d hand manifest path is missing or unsafe")
    if _sha256(path) != expected_sha or path.stat().st_size != expected_bytes:
        raise ValueError("Step 6d hand artifact hash/size mismatch")
    return path, _read_canonical(path, f"Step 6d hand {expected_index}")


def _expected_runner_summary(
    *,
    summary_dir: Path,
    hand_indices: Sequence[int],
    reports: Sequence[Mapping[str, Any]],
    workers: int,
    threads: int,
    candidate_sha256: str,
    run_contract: Mapping[str, Any],
    resumed_hand_count: int,
) -> dict[str, Any]:
    full = list(hand_indices) == list(range(FULL_HAND_COUNT))
    rows = [row for report in reports for row in report["rows"]]
    reference_performance = _source_performance(reports, source="reference")
    candidate_performance = _source_performance(reports, source="candidate")
    queue = _runner_percentiles([float(report["queue_seconds"]) for report in reports])
    worker_wall = _runner_percentiles(
        [float(report["worker_wall_seconds"]) for report in reports]
    )
    peak_rss = max(int(report["memory"]["peak_rss_bytes"]) for report in reports)
    profile_counts = {
        profile: sum(report["profile"] == profile for report in reports)
        for profile in M31_T3_BEHAVIOR_PROFILES
    }
    geometry_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["geometry"]["legal_action_count"])
        geometry_counts[key] = geometry_counts.get(key, 0) + 1
    parity_count = len(rows)
    parity_fraction = parity_count / len(rows)
    integrity_gates = {
        "exact_hand_indices": [report["hand_index"] for report in reports]
        == list(hand_indices),
        "exact_root_count": len(rows) == len(hand_indices) * 2,
        "exact_seat_balance": (
            sum(row["seat"] == "first" for row in rows)
            == sum(row["seat"] == "second" for row in rows)
            == len(hand_indices)
        ),
        "profile_cycle_exact": all(
            report["profile"] == expected_profile(int(report["hand_index"]))
            for report in reports
        ),
        "full_profile_quota_if_applicable": (
            not full
            or profile_counts
            == {profile: FULL_HANDS_PER_PROFILE for profile in M31_T3_BEHAVIOR_PROFILES}
        ),
        "portable_semantic_parity_fraction_1": parity_fraction == 1.0,
        "no_training_quality_promotion_activation_or_cloud": True,
    }
    first = candidate_performance["first"]["solve_wall_seconds"]
    second = candidate_performance["second"]["solve_wall_seconds"]
    candidate_gates = {
        "portable_semantic_parity_fraction": parity_fraction == 1.0,
        "first_p95_seconds_max": first["p95_seconds"] <= MAX_FIRST_P95_SECONDS,
        "first_p99_and_max_seconds_max": (
            first["p99_seconds"] <= MAX_FIRST_P99_SECONDS
            and first["max_seconds"] <= MAX_FIRST_SECONDS
        ),
        "second_p95_seconds_max": second["p95_seconds"] <= MAX_SECOND_P95_SECONDS,
        "peak_rss_bytes_max": peak_rss <= MAX_PEAK_RSS_BYTES,
    }
    all_gates = all(integrity_gates.values()) and (
        all(candidate_gates.values()) if full else True
    )
    hand_dir = summary_dir / "hands"
    return {
        "schema": STEP6D_PERFORMANCE_SUMMARY_SCHEMA,
        "status": "pass" if all_gates else "no_go",
        "contract_canonical_sha256": EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
        "schedule": PERFORMANCE_DEVELOPMENT_SCHEDULE,
        "scope": FULL_SCOPE if full else SUBSET_SMOKE_SCOPE,
        "run_contract": dict(run_contract),
        "run_contract_digest": _artifact_digest(run_contract),
        "subset_smoke": not full,
        "hand_indices": list(hand_indices),
        "paired_hand_count": len(hand_indices),
        "root_count": len(rows),
        "profile_hand_counts": profile_counts,
        "budget": dict(SEARCH_BUDGET),
        "allocation": {
            "workers": workers,
            "rayon_threads_per_worker": threads,
            "allocation_id": f"{workers}x{threads}",
        },
        "reference_library_sha256": REFERENCE_NATIVE_LIBRARY_SHA256,
        "candidate_library_sha256": candidate_sha256,
        "portable_parity": {
            "matching_roots": parity_count,
            "root_count": len(rows),
            "fraction": parity_fraction,
        },
        "performance": {
            "queue_seconds_per_hand": queue,
            "worker_wall_seconds_per_hand": worker_wall,
            "reference_by_seat": reference_performance,
            "candidate_by_seat": candidate_performance,
            "peak_process_rss_bytes": peak_rss,
        },
        "geometry": {"legal_action_count_histogram": geometry_counts},
        "resumed_hand_count": resumed_hand_count,
        "integrity_gates": integrity_gates,
        "candidate_freeze_gates_applied": full,
        "candidate_freeze_gates": candidate_gates,
        "all_gates_passed": all_gates,
        "candidate_freeze_ready": full and all_gates,
        "missing_or_censored_root_indices": [],
        "hand_manifest": [
            {
                "hand_index": report["hand_index"],
                "path": f"hands/hand_{report['hand_index']:03d}.json",
                "sha256": _sha256(
                    hand_dir / f"hand_{report['hand_index']:03d}.json"
                ),
                "bytes": (
                    hand_dir / f"hand_{report['hand_index']:03d}.json"
                ).stat().st_size,
            }
            for report in reports
        ],
        "teacher_value_status": "diagnostic_not_match_EV",
        "training_eligible": False,
        "threshold_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "performance_lock_opened": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "cloud_started": False,
    }


def validate_performance(
    *, summary_path: Path, output_path: Path | None = None
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    summary = _read_canonical(summary_path, "Step 6d performance summary")
    _require_exact_keys(summary, _SUMMARY_KEYS, "Step 6d performance summary")
    hand_indices = _validate_sorted_indices(
        summary.get("hand_indices"),
        "Step 6d summary hand indices",
        maximum=FULL_HAND_COUNT - 1,
    )
    if not hand_indices:
        raise ValueError("Step 6d summary hand set is empty")
    scope = summary.get("scope")
    if scope == FULL_SCOPE:
        if hand_indices != list(range(FULL_HAND_COUNT)):
            raise ValueError("full Step 6d gate requires hands 0..99")
    elif scope == SUBSET_SMOKE_SCOPE:
        if len(hand_indices) >= FULL_HAND_COUNT:
            raise ValueError("full Step 6d set cannot be labeled subset smoke")
    else:
        raise ValueError("Step 6d summary scope changed")
    run_contract = _mapping(summary.get("run_contract"), "Step 6d run contract")
    workers, threads, candidate_sha256 = _validate_run_contract(
        run_contract, hand_indices=hand_indices
    )
    run_contract_digest = _artifact_digest(run_contract)
    if summary.get("run_contract_digest") != run_contract_digest:
        raise ValueError("Step 6d summary run-contract digest mismatch")
    manifest = _array(summary.get("hand_manifest"), "Step 6d hand manifest")
    if len(manifest) != len(hand_indices):
        raise ValueError("Step 6d hand manifest count mismatch")

    reports: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for position, hand_index in enumerate(hand_indices):
        root_path = summary_path.parent / "roots" / f"hand_{hand_index:03d}.json"
        root = _read_canonical(root_path, f"Step 6d root {hand_index}")
        observations = _validate_root(root, hand_index=hand_index)
        _hand_path, hand = _safe_manifest_hand(
            summary_path.parent,
            _mapping(manifest[position], "Step 6d hand manifest entry"),
            hand_index,
        )
        rows = _validate_hand(
            hand,
            root=root,
            observations=observations,
            hand_index=hand_index,
            run_contract_digest=run_contract_digest,
            workers=workers,
            threads=threads,
            candidate_sha256=candidate_sha256,
        )
        for row in rows:
            if row["observation_fingerprint"] in fingerprints:
                raise ValueError("Step 6d duplicate observation fingerprint")
            fingerprints.add(str(row["observation_fingerprint"]))
        aggregate_rows.extend(rows)
        reports.append(dict(hand))

    resumed = _integer(
        summary.get("resumed_hand_count"), "Step 6d resumed hand count"
    )
    if resumed > len(hand_indices):
        raise ValueError("Step 6d resumed hand count is out of range")
    expected_summary = _expected_runner_summary(
        summary_dir=summary_path.parent,
        hand_indices=hand_indices,
        reports=reports,
        workers=workers,
        threads=threads,
        candidate_sha256=candidate_sha256,
        run_contract=run_contract,
        resumed_hand_count=resumed,
    )
    if summary != expected_summary:
        raise ValueError("Step 6d summary aggregate/tamper mismatch")

    first_values = [
        float(row["candidate"]["solve_wall_seconds"])
        for row in aggregate_rows
        if row["seat"] == "first"
    ]
    second_values = [
        float(row["candidate"]["solve_wall_seconds"])
        for row in aggregate_rows
        if row["seat"] == "second"
    ]
    first = _validation_latency(first_values)
    second = _validation_latency(second_values)
    seat_counts = {"first": len(first_values), "second": len(second_values)}
    profile_counts = {
        profile: sum(expected_profile(index) == profile for index in hand_indices)
        for profile in M31_T3_BEHAVIOR_PROFILES
    }
    peak_rss = max(int(row["peak_rss_bytes"]) for row in aggregate_rows)
    gate_applicable = scope == FULL_SCOPE
    if gate_applicable:
        gates: dict[str, bool | None] = {
            "exactly_100_paired_hands": len(hand_indices) == FULL_HAND_COUNT,
            "exactly_200_roots": len(aggregate_rows) == FULL_ROOT_COUNT,
            "exactly_100_first_and_100_second": seat_counts
            == {"first": FULL_SEAT_ROOT_COUNT, "second": FULL_SEAT_ROOT_COUNT},
            "exactly_20_hands_per_profile": profile_counts
            == {
                profile: FULL_HANDS_PER_PROFILE
                for profile in M31_T3_BEHAVIOR_PROFILES
            },
            "portable_semantic_parity_fraction_one": True,
            "missing_or_censored_roots_zero": summary[
                "missing_or_censored_root_indices"
            ]
            == [],
            "first_p95_within_150_seconds": first["p95_seconds"]
            <= MAX_FIRST_P95_SECONDS,
            "first_p99_within_240_seconds": first["p99_seconds"]
            <= MAX_FIRST_P99_SECONDS,
            "first_max_within_240_seconds": first["max_seconds"]
            <= MAX_FIRST_SECONDS,
            "second_p95_within_5_seconds": second["p95_seconds"]
            <= MAX_SECOND_P95_SECONDS,
            "peak_rss_within_858993459_bytes": peak_rss <= MAX_PEAK_RSS_BYTES,
        }
    else:
        gates = {name: None for name in _GATE_KEYS}
    all_gates_passed = gate_applicable and all(value is True for value in gates.values())
    status = (
        "pass"
        if all_gates_passed
        else ("no_go" if gate_applicable else "not_applicable")
    )
    decision = {
        "pass": "performance_development_candidate_freeze_pass_open_performance_lock_only",
        "no_go": "performance_development_no_go_continue_engineering_on_same_set",
        "not_applicable": "subset_smoke_integrity_only_performance_gate_not_applied",
    }[status]
    report = {
        "schema": STEP6D_PERFORMANCE_VALIDATION_SCHEMA,
        "status": status,
        "decision": decision,
        "scope": scope,
        "schedule": PERFORMANCE_DEVELOPMENT_SCHEDULE,
        "run_id": STEP6D_RUN_ID,
        "contract_canonical_sha256": EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
        "run_contract_digest": run_contract_digest,
        "gate_applicable": gate_applicable,
        "integrity": {
            "paired_hand_count": len(hand_indices),
            "root_count": len(aggregate_rows),
            "seat_root_counts": seat_counts,
            "profile_hand_counts": profile_counts,
            "unique_observation_fingerprint_count": len(fingerprints),
            "portable_semantic_parity_count": len(aggregate_rows),
            "portable_semantic_parity_fraction": 1.0,
            "missing_or_censored_root_count": len(
                summary["missing_or_censored_root_indices"]
            ),
        },
        "performance": {
            "candidate_latency_by_seat": {"first": first, "second": second},
            "peak_rss_bytes": peak_rss,
            "limits": {
                "first_p95_seconds": MAX_FIRST_P95_SECONDS,
                "first_p99_seconds": MAX_FIRST_P99_SECONDS,
                "first_max_seconds": MAX_FIRST_SECONDS,
                "second_p95_seconds": MAX_SECOND_P95_SECONDS,
                "peak_rss_bytes": MAX_PEAK_RSS_BYTES,
            },
        },
        "gates": gates,
        "all_gates_passed": all_gates_passed,
        "performance_candidate_frozen": all_gates_passed,
        "performance_lock_authorized": all_gates_passed,
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    if output_path is not None:
        _write_once(Path(output_path).resolve(), report)
    return report


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite Step 6d performance validation: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_artifact_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_performance(summary_path=args.summary, output_path=args.output)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 1 if report["status"] == "no_go" else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_ALLOCATIONS",
    "FULL_HAND_COUNT",
    "FULL_ROOT_COUNT",
    "FULL_SCOPE",
    "MAX_FIRST_P95_SECONDS",
    "MAX_FIRST_P99_SECONDS",
    "MAX_FIRST_SECONDS",
    "MAX_PEAK_RSS_BYTES",
    "MAX_SECOND_P95_SECONDS",
    "PERFORMANCE_DEVELOPMENT_SCHEDULE",
    "PERCENTILE_METHOD",
    "REFERENCE_NATIVE_LIBRARY_SHA256",
    "SEARCH_BUDGET",
    "STEP6D_PERFORMANCE_HAND_SCHEMA",
    "STEP6D_PERFORMANCE_ROOT_SCHEMA",
    "STEP6D_PERFORMANCE_RUN_CONTRACT_SCHEMA",
    "STEP6D_PERFORMANCE_SUMMARY_SCHEMA",
    "STEP6D_PERFORMANCE_VALIDATION_SCHEMA",
    "SUBSET_SMOKE_SCOPE",
    "expected_profile",
    "expected_schedule_row",
    "expected_seeds",
    "main",
    "nearest_rank_percentile",
    "validate_performance",
]
