"""Run one restart-safe M3.1 Step 6c production-label quality shard.

Step 6c is a bounded quality pilot, not production teacher fanout.  Each shard
contains 25 paired hands (50 T3 roots).  Primary decisions use the frozen
8/32/4/0 budget.  Five precommitted paired hands are additionally rerun with
the same candidate and child-policy domains but an independent 128-particle
evaluation domain.  The primary ActionKey remains locked when confirmation
regret is computed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
from .ai_profiles import ModelPaths, load_model_bundle
from .hu_belief import HIDDEN_CARD_PRIOR, sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    generate_behavior_t3_roots,
)
from .hu_m31_t3_runtime import HuM31T3RuntimeConfig, HuM31T3SearchSolver
from .hu_m31_t3_step6c_contract import (
    CONFIRMATION_BUDGET,
    CONFIRMATION_HAND_INDICES,
    EXPECTED_STEP6C_CONTRACT_BYTE_SHA256,
    EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256,
    PILOT_HANDS_PER_SHARD,
    PILOT_HAND_INDICES,
    PILOT_ROOTS_PER_SHARD,
    PILOT_SHARD_COUNT,
    PRODUCTION_LABEL_BUDGET,
    STEP6C_BEHAVIOR_SCHEDULE_SCHEMA,
    STEP6C_RUN_ID,
    behavior_profile_for_train_index,
    canonical_sha256,
    schedule_rows,
    train_seed_values,
)
from .run_hu_m31_t3_step6a_shard import (
    MAX_FIRST_P95_SECONDS,
    MAX_PEAK_RSS_BYTES,
    MAX_SECOND_P95_SECONDS,
    RAYON_THREADS,
    STEP5_CONTRACT_CANONICAL_SHA256,
    WORKERS,
    _verify_package_files,
    portable_decision_sha256,
)
from .run_hu_m31_t3_step6b_shard import EXPECTED_MODELS
from .validate_hu_m31_t3_profile import process_memory_snapshot


STEP6C_PACKAGE_SCHEMA = "hu_m31_t3_step6c_spot_package_v1"
STEP6C_SHARD_SCHEMA = "hu_m31_t3_step6c_shard_v1"
STEP6C_ROOT_TASK_SCHEMA = "hu_m31_t3_step6c_root_task_v1"
STEP6C_SEARCH_TASK_SCHEMA = "hu_m31_t3_step6c_search_task_v1"
STEP6C_CONFIRMATION_SCHEMA = "hu_m31_t3_step6c_confirmation_v1"
STEP6C_HEARTBEAT_SCHEMA = "hu_m31_t3_step6c_heartbeat_v1"
STEP6C_PARITY_SCHEMA = "hu_m31_t3_step6c_linux_parity_v1"
STEP6C_SUMMARY_SCHEMA = "hu_m31_t3_step6c_summary_v1"

AUTHORIZED_SHARDS = tuple(range(PILOT_SHARD_COUNT))
EXPECTED_NATIVE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
EXPECTED_FEATURE_ENCODER_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)

_PRIMARY_BUDGET = {
    "candidate_samples": 8,
    "evaluation_samples": 32,
    "downstream_t3_samples": 4,
    "downstream_t4_samples": 0,
}
_CONFIRMATION_BUDGET = {
    "candidate_samples": 8,
    "evaluation_samples": 128,
    "downstream_t3_samples": 4,
    "downstream_t4_samples": 0,
}
_ACTION_VALUE_KEYS = frozenset(
    {
        "action_key",
        "original_index",
        "rank",
        "placements",
        "discards",
        "selection_ev",
        "evaluation_ev",
        "evaluation_regret",
    }
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
_ROOT_TASK_KEYS = frozenset(
    {
        "schema",
        "contract_digest",
        "global_hand_index",
        "profile",
        "seeds",
        "confirmation_required",
        "observations",
        "current_profile_resolved",
        "opponent_private_discards_used",
    }
)
_ROOT_OBSERVATION_KEYS = frozenset({"seat", "observation_fingerprint", "observation"})
_SEARCH_TASK_KEYS = frozenset(
    {
        "schema",
        "contract_digest",
        "global_hand_index",
        "profile",
        "seeds",
        "confirmation_required",
        "process_id",
        "rayon_threads",
        "wall_seconds",
        "memory",
        "engine",
        "rows",
        "all_gates_passed",
        "teacher_value_status",
        "training_eligible",
        "current_profile_changed",
        "production_fanout_authorized",
    }
)
_SEARCH_ROW_KEYS = frozenset(
    {
        "root_index",
        "global_hand_index",
        "seat",
        "observation_fingerprint",
        "legal_action_count",
        "primary_wall_seconds",
        "confirmation_wall_seconds",
        "wall_seconds",
        "decision",
        "confirmation",
        "integrity",
    }
)
_CONFIRMATION_KEYS = frozenset(
    {
        "schema",
        "locked_selected_action_key",
        "solver_selected_action_key",
        "best_confirmation_action_key",
        "best_confirmation_ev",
        "locked_selected_confirmation_ev",
        "selected_regret",
        "decision",
    }
)
_ROW_INTEGRITY_KEYS = frozenset(
    {
        "primary_decision_valid",
        "confirmation_presence",
        "confirmation_decision_valid",
        "candidate_belief_exact",
        "candidate_rng_exact",
        "selection_values_exact",
        "selected_action_key_exact",
        "confirmation_action_set_exact",
        "confirmation_regret_consistent",
        "primary_action_locked",
        "hidden_information_safe",
    }
)
_PARITY_GATE_KEYS = frozenset(
    {
        "both_seats",
        "portable_action_values_exact",
        "linux_native_hash_bound",
        "manifest_source_and_golden_bound",
        "first_production_budget_speed_smoke_within_180_seconds",
        "second_production_budget_speed_smoke_within_6_seconds",
        "no_profile_or_current_resolution",
    }
)
STEP6C_SHARD_GATE_KEYS = frozenset(
    {
        "exact_shard_hand_indices",
        "exactly_50_roots",
        "exactly_25_each_seat",
        "exact_confirmation_root_count",
        "unique_observation_fingerprints",
        "candidate_rng_unique",
        "evaluation_rng_unique",
        "confirmation_rng_unique",
        "candidate_evaluation_confirmation_rng_disjoint",
        "resume_drill_recovered_task",
        "first_p95_within_180_seconds",
        "second_p95_within_6_seconds",
        "peak_rss_within_1_gib",
        "geometry_diagnostic_only",
        "quality_thresholds_deferred_to_merged_validator",
        "no_training_current_profile_or_production_fanout",
    }
)
_FORBIDDEN_HIDDEN_KEYS = frozenset(
    {
        "opponent_private_discards",
        "true_dead_cards",
        "remaining_deck",
        "world_state",
        "replay_truth",
        "draw_pile",
        "future_cards",
        "deck_tail",
        "true_opponent_private_discards",
    }
)


def _budget_dict(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        raw = dict(value)
    elif hasattr(value, "to_dict"):
        raw = dict(value.to_dict())
    else:
        raw = {
            name: getattr(value, name)
            for name in (
                "candidate_samples",
                "evaluation_samples",
                "downstream_t3_samples",
            )
        }
    return {
        "candidate_samples": int(raw["candidate_samples"]),
        "evaluation_samples": int(raw["evaluation_samples"]),
        "downstream_t3_samples": int(raw["downstream_t3_samples"]),
        "downstream_t4_samples": int(raw.get("downstream_t4_samples", 0)),
    }


def _assert_frozen_contract() -> None:
    if _budget_dict(PRODUCTION_LABEL_BUDGET) != _PRIMARY_BUDGET:
        raise ValueError("Step 6c production-label budget changed")
    if _budget_dict(CONFIRMATION_BUDGET) != _CONFIRMATION_BUDGET:
        raise ValueError("Step 6c confirmation budget changed")
    if tuple(PILOT_HAND_INDICES) != tuple(range(50)):
        raise ValueError("Step 6c pilot hand indices changed")
    if tuple(CONFIRMATION_HAND_INDICES) != (5, 16, 29, 39, 45):
        raise ValueError("Step 6c confirmation hand indices changed")
    if (
        PILOT_SHARD_COUNT != 2
        or PILOT_HANDS_PER_SHARD != 25
        or PILOT_ROOTS_PER_SHARD != 50
    ):
        raise ValueError("Step 6c shard geometry changed")


@dataclass(frozen=True)
class ShardSpec:
    run_name: str
    shard: int
    hand_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run_name, str) or not self.run_name:
            raise ValueError("Step 6c run name must be non-empty")
        if (
            isinstance(self.shard, bool)
            or not isinstance(self.shard, int)
            or self.shard not in AUTHORIZED_SHARDS
        ):
            raise ValueError("Step 6c execution authorizes shards 0 and 1 only")
        expected = tuple(
            range(
                self.shard * PILOT_HANDS_PER_SHARD,
                (self.shard + 1) * PILOT_HANDS_PER_SHARD,
            )
        )
        if self.hand_indices != expected:
            raise ValueError("Step 6c shard schedule changed")

    @property
    def hand_count(self) -> int:
        return len(self.hand_indices)

    @property
    def root_count(self) -> int:
        return self.hand_count * 2

    @property
    def global_hand_start(self) -> int:
        return self.hand_indices[0]


@dataclass(frozen=True)
class DecisionEvidence:
    candidate_keys: frozenset[str]
    evaluation_keys: frozenset[str]
    selection_by_key: Mapping[str, float]
    evaluation_by_key: Mapping[str, float]


@dataclass(frozen=True)
class RowEvidence:
    candidate_keys: frozenset[str]
    evaluation_keys: frozenset[str]
    confirmation_keys: frozenset[str]
    confirmation_regret: float | None
    primary_wall_seconds: float
    confirmation_wall_seconds: float | None
    legal_action_count: int
    seat: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Step 6c JSON artifact must be an object: {path}")
    return value


def _write_once(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Step 6c artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _commit_or_validate_final_summary(
    path: Path, summary: Mapping[str, Any], existing: Mapping[str, Any] | None
) -> dict[str, Any]:
    frozen = dict(summary)
    if existing is None:
        _write_once(path, frozen)
        return frozen
    if dict(existing) != frozen:
        raise ValueError("existing Step 6c final summary content changed")
    return dict(existing)


def _write_heartbeat(path: Path, **values: Any) -> None:
    payload = {
        "schema": STEP6C_HEARTBEAT_SCHEMA,
        **values,
        "process_id": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "current_profile_changed": False,
        "production_fanout_authorized": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(payload))
    os.replace(temporary, path)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} is outside the accepted range")
    return result


def _reject_hidden(value: Any, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_HIDDEN_KEYS:
                raise ValueError(f"forbidden hidden-information field at {path}.{key}")
            _reject_hidden(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_hidden(child, f"{path}[{index}]")


def _validate_frozen_runtime_manifest(manifest: Mapping[str, Any]) -> None:
    native = manifest.get("native_library")
    feature = manifest.get("feature_encoder_library")
    if (
        not isinstance(native, Mapping)
        or native.get("path") != "native/release/libofc_hu_m3_engine.so"
        or native.get("sha256") != EXPECTED_NATIVE_LIBRARY_SHA256
        or native.get("engine_version") != "ofc_hu_m3_engine/0.1.0"
        or not isinstance(feature, Mapping)
        or feature.get("path") != "target/release/libofc_stage3_feature_encoder.so"
        or feature.get("sha256") != EXPECTED_FEATURE_ENCODER_SHA256
        or manifest.get("models") != EXPECTED_MODELS
    ):
        raise ValueError("Step 6c frozen native/model identity changed")


def _load_manifest_and_spec(
    *,
    manifest_path: Path,
    schedule_path: Path,
    source_package_sha256: str,
    shard: int,
) -> tuple[dict[str, Any], ShardSpec, str]:
    _assert_frozen_contract()
    manifest = _read_json(manifest_path)
    manifest_sha = _sha256(manifest_path)
    expected_schedule = list(schedule_rows(indices=PILOT_HAND_INDICES))
    try:
        observed_schedule = [
            json.loads(line)
            for line in schedule_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Step 6c behavior schedule is unreadable") from exc
    required_hash_fields = (
        "step6b_status_sha256",
        "step6b_validation_sha256",
        "step6b_receive_receipt_sha256",
        "step6c_contract_byte_sha256",
        "step6c_contract_canonical_sha256",
        "step6c_validation_sha256",
    )
    if (
        manifest.get("schema") != STEP6C_PACKAGE_SCHEMA
        or manifest.get("status") != "packaged_local_no_gcloud"
        or manifest.get("source_sha256") != source_package_sha256
        or manifest.get("schedule_sha256") != _sha256(schedule_path)
        or manifest.get("total_shards") != PILOT_SHARD_COUNT
        or manifest.get("authorized_shards") != list(AUTHORIZED_SHARDS)
        or manifest.get("paired_hands_per_shard") != PILOT_HANDS_PER_SHARD
        or manifest.get("roots_per_shard") != PILOT_ROOTS_PER_SHARD
        or manifest.get("step5_contract_canonical_sha256")
        != STEP5_CONTRACT_CANONICAL_SHA256
        or manifest.get("step6c_run_id") != STEP6C_RUN_ID
        or manifest.get("production_label_budget") != _PRIMARY_BUDGET
        or manifest.get("confirmation_budget") != _CONFIRMATION_BUDGET
        or manifest.get("pilot_hand_indices") != list(PILOT_HAND_INDICES)
        or manifest.get("confirmation_hand_indices") != list(CONFIRMATION_HAND_INDICES)
        or manifest.get("behavior_schedule_schema") != STEP6C_BEHAVIOR_SCHEDULE_SCHEMA
        or manifest.get("teacher_schedule_canonical_sha256")
        != canonical_sha256(expected_schedule)
        or manifest.get("quality_pilot_authorized") is not True
        or manifest.get("pilot_rows_training_eligible") is not False
        or manifest.get("production_fanout_authorized") is not False
        or manifest.get("current_profile_changed") is not False
        or manifest.get("named_profile_added") is not False
        or manifest.get("m31_complete") is not False
        or not _is_sha256(manifest.get("parity_golden_sha256"))
        or any(not _is_sha256(manifest.get(name)) for name in required_hash_fields)
        or manifest.get("step6c_contract_byte_sha256")
        != EXPECTED_STEP6C_CONTRACT_BYTE_SHA256
        or manifest.get("step6c_contract_canonical_sha256")
        != EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256
        or observed_schedule != expected_schedule
    ):
        raise ValueError("Step 6c package manifest or schedule boundary changed")
    _validate_frozen_runtime_manifest(manifest)
    if shard not in AUTHORIZED_SHARDS:
        raise ValueError("Step 6c execution authorizes shards 0 and 1 only")
    start = shard * PILOT_HANDS_PER_SHARD
    indices = tuple(PILOT_HAND_INDICES[start : start + PILOT_HANDS_PER_SHARD])
    return manifest, ShardSpec(str(manifest["run_name"]), shard, indices), manifest_sha


def _validate_parity_report(
    value: Mapping[str, Any],
    *,
    source_package_sha256: str,
    manifest_sha256: str,
    parity_golden_sha256: str,
    native_library_sha256: str,
) -> None:
    rows = value.get("rows")
    gates = value.get("gates")
    if (
        value.get("schema") != STEP6C_PARITY_SCHEMA
        or value.get("status") != "pass"
        or value.get("source_package_sha256") != source_package_sha256
        or value.get("manifest_sha256") != manifest_sha256
        or value.get("parity_golden_sha256") != parity_golden_sha256
        or value.get("native_library_sha256") != native_library_sha256
        or value.get("all_gates_passed") is not True
        or value.get("current_profile_changed") is not False
        or value.get("teacher_generation_started") is not False
        or value.get("production_fanout_authorized") is not False
        or not isinstance(rows, list)
        or len(rows) != 2
        or not isinstance(gates, Mapping)
        or set(gates) != _PARITY_GATE_KEYS
        or any(result is not True for result in gates.values())
    ):
        raise ValueError("Step 6c Linux parity provenance changed")
    if [row.get("seat") for row in rows if isinstance(row, Mapping)] != [
        "first",
        "second",
    ]:
        raise ValueError("Step 6c Linux parity seats changed")
    for row in rows:
        if not isinstance(row, Mapping) or row.get("match") is not True:
            raise ValueError("Step 6c Linux parity row changed")
        limit = (
            MAX_FIRST_P95_SECONDS
            if row.get("seat") == "first"
            else MAX_SECOND_P95_SECONDS
        )
        if (
            not _is_sha256(row.get("observation_fingerprint"))
            or not _is_sha256(row.get("expected_portable_sha256"))
            or row.get("observed_portable_sha256")
            != row.get("expected_portable_sha256")
            or not math.isfinite(float(row.get("wall_seconds", math.nan)))
            or float(row["wall_seconds"]) < 0.0
            or float(row["wall_seconds"]) > limit
        ):
            raise ValueError("Step 6c Linux parity row changed")


def run_linux_parity(
    *,
    manifest: Mapping[str, Any],
    source_package_sha256: str,
    manifest_sha256: str,
    repository_root: Path,
    parity_golden_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    golden_sha = _sha256(parity_golden_path)
    native = manifest["native_library"]
    expected = {
        "source_package_sha256": source_package_sha256,
        "manifest_sha256": manifest_sha256,
        "parity_golden_sha256": golden_sha,
        "native_library_sha256": str(native["sha256"]),
    }
    if golden_sha != manifest.get("parity_golden_sha256"):
        raise ValueError("Step 6c parity golden changed")
    if output_path.exists():
        existing = _read_json(output_path)
        _validate_parity_report(existing, **expected)
        return existing
    golden = _read_json(parity_golden_path)
    config = golden.get("runtime_config")
    rows = golden.get("rows")
    if not isinstance(config, Mapping) or not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("Step 6c parity golden changed")
    solver = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=str(native["sha256"]),
            library_path=repository_root / str(native["path"]),
            run_id=str(config["run_id"]),
            candidate_samples=int(config["candidate_samples"]),
            evaluation_samples=int(config["evaluation_samples"]),
            downstream_t3_samples=int(config["downstream_t3_samples"]),
            seed=int(config["continuation_seed"]),
            candidate_seed=int(config["candidate_seed"]),
            evaluation_seed=int(config["evaluation_seed"]),
        )
    )
    observed = []
    for row in rows:
        observation = ActorObservation.from_dict(row["observation"])
        started = time.perf_counter()
        digest = portable_decision_sha256(solver.solve(observation).to_dict())
        wall_seconds = time.perf_counter() - started
        observed.append(
            {
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "expected_portable_sha256": row["portable_decision_sha256"],
                "observed_portable_sha256": digest,
                "wall_seconds": wall_seconds,
                "match": digest == row["portable_decision_sha256"],
            }
        )
    gates = {
        "both_seats": [row["seat"] for row in observed] == ["first", "second"],
        "portable_action_values_exact": all(row["match"] for row in observed),
        "linux_native_hash_bound": solver.library_sha256 == native["sha256"],
        "manifest_source_and_golden_bound": True,
        "first_production_budget_speed_smoke_within_180_seconds": (
            float(observed[0]["wall_seconds"]) <= MAX_FIRST_P95_SECONDS
        ),
        "second_production_budget_speed_smoke_within_6_seconds": (
            float(observed[1]["wall_seconds"]) <= MAX_SECOND_P95_SECONDS
        ),
        "no_profile_or_current_resolution": True,
    }
    report = {
        "schema": STEP6C_PARITY_SCHEMA,
        "status": "pass" if all(gates.values()) else "no_go",
        **expected,
        "rows": observed,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "current_profile_changed": False,
        "teacher_generation_started": False,
        "production_fanout_authorized": False,
    }
    _write_once(output_path, report)
    if not report["all_gates_passed"]:
        raise RuntimeError("Step 6c Linux portable parity failed")
    return report


def _root_contract_digest(
    *, manifest_sha256: str, spec: ShardSpec, global_hand_index: int
) -> str:
    return canonical_sha256(
        {
            "manifest_sha256": manifest_sha256,
            "run_name": spec.run_name,
            "shard": spec.shard,
            "global_hand_index": global_hand_index,
            "seeds": train_seed_values(global_hand_index),
            "profile": behavior_profile_for_train_index(global_hand_index),
            "confirmation_required": global_hand_index in CONFIRMATION_HAND_INDICES,
        }
    )


def _validate_root_task(
    value: Mapping[str, Any],
    *,
    digest: str,
    global_hand_index: int,
    expected_observations: Sequence[ActorObservation] | None = None,
) -> None:
    rows = value.get("observations")
    if (
        set(value) != _ROOT_TASK_KEYS
        or value.get("schema") != STEP6C_ROOT_TASK_SCHEMA
        or value.get("contract_digest") != digest
        or value.get("global_hand_index") != global_hand_index
        or value.get("profile") != behavior_profile_for_train_index(global_hand_index)
        or value.get("seeds") != train_seed_values(global_hand_index)
        or value.get("confirmation_required")
        is not (global_hand_index in CONFIRMATION_HAND_INDICES)
        or value.get("current_profile_resolved") is not False
        or value.get("opponent_private_discards_used") is not False
        or not isinstance(rows, list)
        or len(rows) != 2
        or [row.get("seat") for row in rows if isinstance(row, Mapping)]
        != ["first", "second"]
    ):
        raise ValueError(f"Step 6c root task changed: {global_hand_index}")
    _reject_hidden(value, "root")
    for offset, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _ROOT_OBSERVATION_KEYS:
            raise ValueError(f"Step 6c root observation changed: {global_hand_index}")
        observation = ActorObservation.from_dict(row["observation"])
        if observation.fingerprint() != row.get("observation_fingerprint") or (
            expected_observations is not None
            and (
                offset >= len(expected_observations)
                or observation.to_dict() != expected_observations[offset].to_dict()
                or observation.fingerprint()
                != expected_observations[offset].fingerprint()
            )
        ):
            raise ValueError(f"Step 6c root observation changed: {global_hand_index}")
    if expected_observations is not None and len(expected_observations) != len(rows):
        raise ValueError(f"Step 6c root generation changed: {global_hand_index}")


def _materialize_roots(
    *,
    manifest_sha256: str,
    spec: ShardSpec,
    root_dir: Path,
    heartbeat_path: Path,
) -> list[dict[str, Any]]:
    root_dir.mkdir(parents=True, exist_ok=True)
    expected_profiles = set(M31_T3_BEHAVIOR_PROFILES) - {"random_exact_final"}
    bundle = None
    tasks: list[dict[str, Any]] = []
    for global_index in spec.hand_indices:
        path = root_dir / f"hand_{global_index:03d}.json"
        digest = _root_contract_digest(
            manifest_sha256=manifest_sha256,
            spec=spec,
            global_hand_index=global_index,
        )
        if bundle is None:
            _write_heartbeat(
                heartbeat_path,
                status="loading_behavior_models",
                total_tasks=spec.hand_count,
                completed_tasks=len(tasks),
                pending_tasks=spec.hand_count - len(tasks),
                resumed_task_count=0,
            )
            bundle = load_model_bundle(ModelPaths(), profiles=expected_profiles)
        seeds = train_seed_values(global_index)
        profile = behavior_profile_for_train_index(global_index)
        observations = generate_behavior_t3_roots(
            hand_seed=seeds["hand"],
            behavior_seed=seeds["behavior"],
            profile=profile,
            bundle=bundle,
        )
        if path.exists():
            value = _read_json(path)
            _validate_root_task(
                value,
                digest=digest,
                global_hand_index=global_index,
                expected_observations=observations,
            )
            tasks.append(value)
            continue
        value = {
            "schema": STEP6C_ROOT_TASK_SCHEMA,
            "contract_digest": digest,
            "global_hand_index": global_index,
            "profile": profile,
            "seeds": seeds,
            "confirmation_required": global_index in CONFIRMATION_HAND_INDICES,
            "observations": [
                {
                    "seat": observation.seat,
                    "observation_fingerprint": observation.fingerprint(),
                    "observation": observation.to_dict(),
                }
                for observation in observations
            ],
            "current_profile_resolved": False,
            "opponent_private_discards_used": False,
        }
        _write_once(path, value)
        tasks.append(value)
        _write_heartbeat(
            heartbeat_path,
            status="materializing_behavior_roots",
            total_tasks=spec.hand_count,
            completed_tasks=len(tasks),
            pending_tasks=spec.hand_count - len(tasks),
            resumed_task_count=0,
        )
    return tasks


def _particle_evidence(
    observation: ActorObservation,
    *,
    base_seed: int,
    role: str,
    sample_count: int,
) -> tuple[frozenset[str], str, str]:
    batch = sample_hidden_card_particles(
        observation,
        base_seed=base_seed,
        run_id=f"{STEP6C_RUN_ID}:{role}",
        sample_count=sample_count,
    )
    keys = tuple(particle.rng_key_digest for particle in batch.particles)
    return frozenset(keys), batch.digest(), _runtime_digest(keys)


def _canonical_digest_float(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _validate_decision_certificates(
    decision: Mapping[str, Any],
    *,
    observation: ActorObservation,
    raw_values: Sequence[Mapping[str, Any]],
    native_library_sha256: str,
) -> None:
    child_count = decision.get("child_information_set_count")
    if (
        isinstance(child_count, bool)
        or not isinstance(child_count, int)
        or child_count < 0
    ):
        raise ValueError("Step 6c child information-set count changed")
    for field in (
        "native_latency_ms",
        "validation_latency_ms",
        "total_latency_ms",
    ):
        _finite(decision.get(field), field, minimum=0.0)
    expected_strings = {
        "runtime_id": "hu_m31_t3_crn_exact_t4_v1",
        "value_scope": "q_pi_uniform_exchangeable_t3_crn_with_exact_t4_children",
        "continuation_policy_id": "local_infoset_response_t3_second_t4_v1",
        "strategy_fusion_guard": "child_actions_keyed_only_by_actor_observation",
        "downstream_t4_native_semantics_id": "m30_exact_t4_native_kernel_semantics_v1",
        "downstream_t4_native_anchor": "same_pinned_m30_native_engine",
        "solver_id": "rust_crn_sequential_t3_v1",
        "engine_version": "ofc_hu_m3_engine/0.1.0",
        "semantic_result_digest_schema": "hu_m31_t3_semantic_result_digest_v1",
        "semantic_result_digest_scope": "dealt_order_independent_action_value_result",
        "result_digest_scope": "ordered_action_mapping_bound",
    }
    if any(decision.get(field) != value for field, value in expected_strings.items()):
        raise ValueError("Step 6c decision semantic identity changed")
    digest_fields = (
        "search_contract_digest",
        "semantic_result_digest",
        "result_digest",
    )
    if any(not _is_sha256(decision.get(field)) for field in digest_fields):
        raise ValueError("Step 6c decision digest schema changed")
    search_contract = {
        "run_id": decision["run_id"],
        "continuation_seed": decision["continuation_seed"],
        "candidate_seed": decision["candidate_seed"],
        "evaluation_seed": decision["evaluation_seed"],
        "candidate_samples": decision["candidate_samples"],
        "evaluation_samples": decision["evaluation_samples"],
        "downstream_t3_samples": decision["downstream_t3_samples"],
        "downstream_t4_samples": 0,
        "use_t4_action_cache": True,
        "continuation_policy_id": decision["continuation_policy_id"],
        "strategy_fusion_guard": decision["strategy_fusion_guard"],
        "downstream_t4_native_semantics_id": decision[
            "downstream_t4_native_semantics_id"
        ],
    }
    search_digest = _runtime_digest(search_contract)
    if decision["search_contract_digest"] != search_digest:
        raise ValueError("Step 6c search-contract digest changed")
    mapping_payload = {
        "runtime_id": decision["runtime_id"],
        "request_schema": "hu_m3_engine_request_v1",
        "observation_fingerprint": observation.fingerprint(),
        "selected_action_key": decision["selected_action_key"],
        "selected_selection_ev": decision["selected_selection_ev"],
        "selected_evaluation_ev": decision["selected_evaluation_ev"],
        "selection_gap": decision["selection_gap"],
        "evaluation_sample_regret": decision["evaluation_sample_regret"],
        "legal_action_set_digest": decision["legal_action_set_digest"],
        "legal_action_order_digest": decision["legal_action_order_digest"],
        "action_values": [
            [
                row["action_key"],
                row["original_index"],
                row["rank"],
                row["selection_ev"],
                row["evaluation_ev"],
            ]
            for row in raw_values
        ],
        "candidate_belief_digest": decision["candidate_belief_digest"],
        "evaluation_belief_digest": decision["evaluation_belief_digest"],
        "candidate_rng_digest": decision["candidate_rng_digest"],
        "evaluation_rng_digest": decision["evaluation_rng_digest"],
        "search_contract": search_contract,
        "search_contract_digest": search_digest,
        "downstream_t4_native_semantics_id": decision[
            "downstream_t4_native_semantics_id"
        ],
        "child_information_set_count": child_count,
        "solver_id": decision["solver_id"],
        "engine_version": decision["engine_version"],
        "native_library_sha256": native_library_sha256,
    }
    semantic_payload = {
        "schema": decision["semantic_result_digest_schema"],
        "runtime_id": decision["runtime_id"],
        "request_schema": "hu_m3_engine_request_v1",
        "action_key_schema": ACTION_KEY_SCHEMA,
        "seat": observation.seat,
        "value_scope": decision["value_scope"],
        "observation_fingerprint": observation.fingerprint(),
        "selected_action_key": decision["selected_action_key"],
        "selected_selection_ev": _canonical_digest_float(
            float(decision["selected_selection_ev"])
        ),
        "selected_evaluation_ev": _canonical_digest_float(
            float(decision["selected_evaluation_ev"])
        ),
        "selection_gap": _canonical_digest_float(float(decision["selection_gap"])),
        "evaluation_sample_regret": _canonical_digest_float(
            float(decision["evaluation_sample_regret"])
        ),
        "legal_action_set_digest": decision["legal_action_set_digest"],
        "action_values": [
            [
                row["action_key"],
                row["rank"],
                _canonical_digest_float(float(row["selection_ev"])),
                _canonical_digest_float(float(row["evaluation_ev"])),
                _canonical_digest_float(float(row["evaluation_regret"])),
            ]
            for row in raw_values
        ],
        "candidate_belief_digest": decision["candidate_belief_digest"],
        "evaluation_belief_digest": decision["evaluation_belief_digest"],
        "candidate_rng_digest": decision["candidate_rng_digest"],
        "evaluation_rng_digest": decision["evaluation_rng_digest"],
        "search_contract": search_contract,
        "search_contract_digest": search_digest,
        "sample_independence": "disjoint_particle_rng_keys",
        "downstream_t4_native_semantics_id": decision[
            "downstream_t4_native_semantics_id"
        ],
        "teacher_value_status": "diagnostic_not_match_EV",
        "child_information_set_count": child_count,
        "solver_id": decision["solver_id"],
        "engine_version": decision["engine_version"],
        "native_library_sha256": native_library_sha256,
    }
    if decision["result_digest"] != _runtime_digest(mapping_payload) or decision[
        "semantic_result_digest"
    ] != _runtime_digest(semantic_payload):
        raise ValueError("Step 6c decision result certificate changed")


def _validate_decision_payload(
    decision: Any,
    *,
    observation: ActorObservation,
    seeds: Mapping[str, int],
    budget: Mapping[str, int],
    evaluation_seed_key: str,
    native_library_sha256: str,
) -> DecisionEvidence:
    if not isinstance(decision, Mapping) or set(decision) != _DECISION_KEYS:
        raise ValueError("Step 6c decision is missing")
    _reject_hidden(decision, "decision")
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    expected_by_index = {
        index: action_key(action).to_token() for index, action in enumerate(legal)
    }
    raw_values = decision.get("action_values")
    if not isinstance(raw_values, list) or len(raw_values) != len(legal):
        raise ValueError("Step 6c decision does not cover every legal action")
    indices: list[int] = []
    ranks: list[int] = []
    tokens_in_output_order: list[str] = []
    selection_by_key: dict[str, float] = {}
    evaluation_by_key: dict[str, float] = {}
    regret_by_key: dict[str, float] = {}
    rank_by_key: dict[str, int] = {}
    for raw in raw_values:
        if not isinstance(raw, Mapping) or set(raw) != _ACTION_VALUE_KEYS:
            raise ValueError("Step 6c action-value schema changed")
        index = raw.get("original_index")
        rank = raw.get("rank")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(legal)
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 0 <= rank < len(legal)
            or raw.get("action_key") != expected_by_index[index]
            or action_key_from_payload(raw).to_token() != expected_by_index[index]
        ):
            raise ValueError("Step 6c action/index mapping changed")
        token = str(raw["action_key"])
        if token in selection_by_key:
            raise ValueError("Step 6c action keys are duplicated")
        selection_by_key[token] = _finite(raw.get("selection_ev"), "selection EV")
        evaluation_by_key[token] = _finite(raw.get("evaluation_ev"), "evaluation EV")
        regret_by_key[token] = _finite(
            raw.get("evaluation_regret"), "evaluation regret", minimum=0.0
        )
        rank_by_key[token] = rank
        indices.append(index)
        ranks.append(rank)
        tokens_in_output_order.append(token)
    if sorted(indices) != list(range(len(legal))) or sorted(ranks) != list(
        range(len(legal))
    ):
        raise ValueError("Step 6c action index/rank coverage changed")
    canonical_tokens = sorted(
        expected_by_index.values(),
        key=lambda token: ActionKey.from_token(token).sort_key(),
    )
    if tokens_in_output_order != canonical_tokens:
        raise ValueError("Step 6c action values are not in canonical ActionKey order")
    expected_ranking = sorted(
        selection_by_key,
        key=lambda token: (
            -selection_by_key[token],
            ActionKey.from_token(token).sort_key(),
        ),
    )
    if any(rank_by_key[token] != rank for rank, token in enumerate(expected_ranking)):
        raise ValueError("Step 6c candidate rank/tie-break mapping changed")
    best_evaluation = max(evaluation_by_key.values())
    for token, value in evaluation_by_key.items():
        if not math.isclose(
            regret_by_key[token], best_evaluation - value, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("Step 6c per-action regret is inconsistent")
    selected_key = decision.get("selected_action_key")
    selected_payload = decision.get("selected_action")
    second_selection_ev = (
        selection_by_key[expected_ranking[1]]
        if len(expected_ranking) > 1
        else selection_by_key[expected_ranking[0]]
    )
    expected_selection_gap = selection_by_key[expected_ranking[0]] - second_selection_ev
    if (
        selected_key not in evaluation_by_key
        or selected_key != expected_ranking[0]
        or not isinstance(selected_payload, Mapping)
        or set(selected_payload) != {"placements", "discards"}
        or action_key_from_payload(selected_payload).to_token() != selected_key
        or not math.isclose(
            _finite(decision.get("selected_selection_ev"), "selected selection EV"),
            selection_by_key[str(selected_key)],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite(decision.get("selected_evaluation_ev"), "selected evaluation EV"),
            evaluation_by_key[str(selected_key)],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite(decision.get("evaluation_sample_regret"), "selected regret"),
            best_evaluation - evaluation_by_key[str(selected_key)],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _finite(decision.get("selection_gap"), "selection gap"),
            expected_selection_gap,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Step 6c selected-action mapping changed")
    expected_budget = _budget_dict(budget)
    if (
        decision.get("schema") != "hu_m31_t3_runtime_decision_v2"
        or decision.get("seat") != observation.seat
        or decision.get("observation_fingerprint") != observation.fingerprint()
        or decision.get("run_id") != STEP6C_RUN_ID
        or decision.get("continuation_seed") != seeds["child"]
        or decision.get("candidate_seed") != seeds["candidate"]
        or decision.get("evaluation_seed") != seeds[evaluation_seed_key]
        or decision.get("candidate_samples") != expected_budget["candidate_samples"]
        or decision.get("evaluation_samples") != expected_budget["evaluation_samples"]
        or decision.get("downstream_t3_samples")
        != expected_budget["downstream_t3_samples"]
        or decision.get("downstream_t4_samples") != 0
        or decision.get("downstream_t4_mode") != "exact"
        or decision.get("use_t4_action_cache") is not True
        or decision.get("native_library_sha256") != native_library_sha256
        or decision.get("execution_mode") != "scalar"
        or decision.get("batch_size") != 1
        or decision.get("teacher_value_status") != "diagnostic_not_match_EV"
        or decision.get("belief_prior") != HIDDEN_CARD_PRIOR
        or decision.get("action_key_schema") != ACTION_KEY_SCHEMA
        or decision.get("legal_action_set_digest") != legal_action_set_digest(legal)
        or decision.get("legal_action_order_digest")
        != ordered_action_mapping_digest(legal)
    ):
        raise ValueError("Step 6c decision contract changed")
    _validate_decision_certificates(
        decision,
        observation=observation,
        raw_values=raw_values,
        native_library_sha256=native_library_sha256,
    )
    candidate_keys, candidate_belief, candidate_rng = _particle_evidence(
        observation,
        base_seed=seeds["candidate"],
        role="candidate_selection",
        sample_count=expected_budget["candidate_samples"],
    )
    evaluation_keys, evaluation_belief, evaluation_rng = _particle_evidence(
        observation,
        base_seed=seeds[evaluation_seed_key],
        role="locked_evaluation",
        sample_count=expected_budget["evaluation_samples"],
    )
    if (
        decision.get("candidate_belief_digest") != candidate_belief
        or decision.get("evaluation_belief_digest") != evaluation_belief
        or decision.get("candidate_rng_digest") != candidate_rng
        or decision.get("evaluation_rng_digest") != evaluation_rng
        or candidate_keys & evaluation_keys
    ):
        raise ValueError("Step 6c belief/RNG evidence changed")
    return DecisionEvidence(
        candidate_keys=candidate_keys,
        evaluation_keys=evaluation_keys,
        selection_by_key=selection_by_key,
        evaluation_by_key=evaluation_by_key,
    )


def _confirmation_payload(
    primary: Mapping[str, Any], confirmation: Mapping[str, Any]
) -> dict[str, Any]:
    primary_key = str(primary["selected_action_key"])
    confirmation_values = {
        str(row["action_key"]): float(row["evaluation_ev"])
        for row in confirmation["action_values"]
    }
    best_ev = max(confirmation_values.values())
    best_key = min(
        (key for key, value in confirmation_values.items() if value == best_ev),
        key=lambda key: ActionKey.from_token(key).sort_key(),
    )
    selected_ev = confirmation_values[primary_key]
    return {
        "schema": STEP6C_CONFIRMATION_SCHEMA,
        "locked_selected_action_key": primary_key,
        "solver_selected_action_key": confirmation["selected_action_key"],
        "best_confirmation_action_key": best_key,
        "best_confirmation_ev": best_ev,
        "locked_selected_confirmation_ev": selected_ev,
        "selected_regret": best_ev - selected_ev,
        "decision": dict(confirmation),
    }


def _validate_confirmation_pair(
    primary: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    if (
        set(payload) != _CONFIRMATION_KEYS
        or payload.get("schema") != STEP6C_CONFIRMATION_SCHEMA
    ):
        raise ValueError("Step 6c confirmation schema changed")
    primary_values = {
        str(row["action_key"]): float(row["selection_ev"])
        for row in primary["action_values"]
    }
    confirmation_values = {
        str(row["action_key"]): float(row["selection_ev"])
        for row in confirmation["action_values"]
    }
    evaluation_values = {
        str(row["action_key"]): float(row["evaluation_ev"])
        for row in confirmation["action_values"]
    }
    locked = str(primary["selected_action_key"])
    best_ev = max(evaluation_values.values())
    best_key = min(
        (key for key, value in evaluation_values.items() if value == best_ev),
        key=lambda key: ActionKey.from_token(key).sort_key(),
    )
    expected_regret = best_ev - evaluation_values[locked]
    if (
        primary.get("candidate_seed") != confirmation.get("candidate_seed")
        or primary.get("continuation_seed") != confirmation.get("continuation_seed")
        or primary.get("run_id") != confirmation.get("run_id")
        or primary.get("candidate_belief_digest")
        != confirmation.get("candidate_belief_digest")
        or primary.get("candidate_rng_digest")
        != confirmation.get("candidate_rng_digest")
        or primary.get("legal_action_set_digest")
        != confirmation.get("legal_action_set_digest")
        or primary_values != confirmation_values
        or confirmation.get("selected_action_key") != locked
        or payload.get("locked_selected_action_key") != locked
        or payload.get("solver_selected_action_key") != locked
        or payload.get("best_confirmation_action_key") != best_key
        or payload.get("decision") != confirmation
        or not math.isclose(
            _finite(payload.get("best_confirmation_ev"), "confirmation best EV"),
            best_ev,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite(
                payload.get("locked_selected_confirmation_ev"),
                "locked confirmation EV",
            ),
            evaluation_values[locked],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite(payload.get("selected_regret"), "confirmation regret", minimum=0),
            expected_regret,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Step 6c confirmation changed primary selection semantics")


def _search_worker(payload: Mapping[str, Any]) -> str:
    task_path = Path(str(payload["task_path"]))
    if task_path.exists():
        raise FileExistsError(f"Step 6c search task already exists: {task_path}")
    root_task = dict(payload["root_task"])
    seeds = dict(root_task["seeds"])
    global_index = int(root_task["global_hand_index"])
    library = Path(str(payload["library_path"]))
    library_sha = str(payload["library_sha256"])
    primary_solver = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=library_sha,
            library_path=library,
            run_id=STEP6C_RUN_ID,
            candidate_samples=8,
            evaluation_samples=32,
            downstream_t3_samples=4,
            seed=seeds["child"],
            candidate_seed=seeds["candidate"],
            evaluation_seed=seeds["evaluation"],
        )
    )
    confirmation_solver = None
    if root_task["confirmation_required"]:
        confirmation_solver = HuM31T3SearchSolver(
            HuM31T3RuntimeConfig(
                expected_library_sha256=library_sha,
                library_path=library,
                run_id=STEP6C_RUN_ID,
                candidate_samples=8,
                evaluation_samples=128,
                downstream_t3_samples=4,
                seed=seeds["child"],
                candidate_seed=seeds["candidate"],
                evaluation_seed=seeds["confirmation"],
            )
        )
    task_started = time.perf_counter()
    rows = []
    for seat_offset, raw in enumerate(root_task["observations"]):
        observation = ActorObservation.from_dict(raw["observation"])
        primary_started = time.perf_counter()
        primary = primary_solver.solve(observation).to_dict()
        primary_seconds = time.perf_counter() - primary_started
        _validate_decision_payload(
            primary,
            observation=observation,
            seeds=seeds,
            budget=_PRIMARY_BUDGET,
            evaluation_seed_key="evaluation",
            native_library_sha256=library_sha,
        )
        confirmation_payload = None
        confirmation_seconds = None
        if confirmation_solver is not None:
            confirmation_started = time.perf_counter()
            confirmation = confirmation_solver.solve(observation).to_dict()
            confirmation_seconds = time.perf_counter() - confirmation_started
            _validate_decision_payload(
                confirmation,
                observation=observation,
                seeds=seeds,
                budget=_CONFIRMATION_BUDGET,
                evaluation_seed_key="confirmation",
                native_library_sha256=library_sha,
            )
            confirmation_payload = _confirmation_payload(primary, confirmation)
            _validate_confirmation_pair(primary, confirmation, confirmation_payload)
        rows.append(
            {
                "root_index": global_index * 2 + seat_offset,
                "global_hand_index": global_index,
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "legal_action_count": len(primary["action_values"]),
                "primary_wall_seconds": primary_seconds,
                "confirmation_wall_seconds": confirmation_seconds,
                "wall_seconds": primary_seconds + (confirmation_seconds or 0.0),
                "decision": primary,
                "confirmation": confirmation_payload,
                "integrity": {key: True for key in _ROW_INTEGRITY_KEYS},
            }
        )
    report = {
        "schema": STEP6C_SEARCH_TASK_SCHEMA,
        "contract_digest": root_task["contract_digest"],
        "global_hand_index": global_index,
        "profile": root_task["profile"],
        "seeds": seeds,
        "confirmation_required": root_task["confirmation_required"],
        "process_id": os.getpid(),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
        "wall_seconds": time.perf_counter() - task_started,
        "memory": process_memory_snapshot(),
        "engine": {
            "version": primary_solver.engine_version,
            "library_sha256": primary_solver.library_sha256,
            "build_or_fallback": False,
        },
        "rows": rows,
        "all_gates_passed": True,
        "teacher_value_status": "diagnostic_not_match_EV",
        "training_eligible": False,
        "current_profile_changed": False,
        "production_fanout_authorized": False,
    }
    _write_once(task_path, report)
    return task_path.name


def _validate_search_task(
    value: Mapping[str, Any],
    *,
    root_task: Mapping[str, Any],
    library_sha256: str,
) -> tuple[RowEvidence, RowEvidence]:
    global_index = int(root_task["global_hand_index"])
    _reject_hidden(value, "task")
    if (
        set(value) != _SEARCH_TASK_KEYS
        or value.get("schema") != STEP6C_SEARCH_TASK_SCHEMA
        or value.get("contract_digest") != root_task.get("contract_digest")
        or value.get("global_hand_index") != global_index
        or value.get("profile") != root_task.get("profile")
        or value.get("seeds") != root_task.get("seeds")
        or value.get("confirmation_required")
        is not root_task.get("confirmation_required")
        or value.get("rayon_threads") != str(RAYON_THREADS)
        or value.get("engine", {}).get("library_sha256") != library_sha256
        or value.get("all_gates_passed") is not True
        or value.get("teacher_value_status") != "diagnostic_not_match_EV"
        or value.get("training_eligible") is not False
        or value.get("current_profile_changed") is not False
        or value.get("production_fanout_authorized") is not False
        or not isinstance(value.get("rows"), list)
        or len(value["rows"]) != 2
    ):
        raise ValueError(f"Step 6c search task changed: {global_index}")
    memory = value.get("memory")
    if (
        not isinstance(memory, Mapping)
        or isinstance(memory.get("peak_rss_bytes"), bool)
        or not isinstance(memory.get("peak_rss_bytes"), int)
        or memory["peak_rss_bytes"] <= 0
    ):
        raise ValueError(f"Step 6c task memory changed: {global_index}")
    seeds = root_task["seeds"]
    evidence_rows: list[RowEvidence] = []
    for offset, (raw_root, row) in enumerate(
        zip(root_task["observations"], value["rows"], strict=True)
    ):
        if (
            not isinstance(raw_root, Mapping)
            or not isinstance(row, Mapping)
            or set(row) != _SEARCH_ROW_KEYS
        ):
            raise ValueError("Step 6c task row changed")
        observation = ActorObservation.from_dict(raw_root["observation"])
        fingerprint = observation.fingerprint()
        confirmation_required = bool(root_task["confirmation_required"])
        primary_seconds = _finite(
            row.get("primary_wall_seconds"), "primary wall seconds", minimum=0.0
        )
        confirmation_seconds = row.get("confirmation_wall_seconds")
        if confirmation_required:
            confirmation_seconds = _finite(
                confirmation_seconds, "confirmation wall seconds", minimum=0.0
            )
        elif confirmation_seconds is not None:
            raise ValueError("Step 6c unexpected confirmation latency")
        expected_total = primary_seconds + float(confirmation_seconds or 0.0)
        if (
            row.get("root_index") != global_index * 2 + offset
            or row.get("global_hand_index") != global_index
            or row.get("seat") != observation.seat
            or row.get("observation_fingerprint") != fingerprint
            or raw_root.get("seat") != observation.seat
            or raw_root.get("observation_fingerprint") != fingerprint
            or row.get("legal_action_count")
            != len(
                generate_turn_actions(observation.hero_board, observation.dealt_cards)
            )
            or not math.isclose(
                _finite(row.get("wall_seconds"), "row wall seconds", minimum=0.0),
                expected_total,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not isinstance(row.get("integrity"), Mapping)
            or set(row["integrity"]) != _ROW_INTEGRITY_KEYS
            or any(result is not True for result in row["integrity"].values())
        ):
            raise ValueError("Step 6c task row identity/integrity changed")
        primary = row.get("decision")
        primary_evidence = _validate_decision_payload(
            primary,
            observation=observation,
            seeds=seeds,
            budget=_PRIMARY_BUDGET,
            evaluation_seed_key="evaluation",
            native_library_sha256=library_sha256,
        )
        confirmation_payload = row.get("confirmation")
        confirmation_keys: frozenset[str] = frozenset()
        confirmation_regret = None
        if confirmation_required:
            if not isinstance(confirmation_payload, Mapping):
                raise ValueError("Step 6c required confirmation is missing")
            confirmation = confirmation_payload.get("decision")
            confirmation_evidence = _validate_decision_payload(
                confirmation,
                observation=observation,
                seeds=seeds,
                budget=_CONFIRMATION_BUDGET,
                evaluation_seed_key="confirmation",
                native_library_sha256=library_sha256,
            )
            _validate_confirmation_pair(primary, confirmation, confirmation_payload)
            if (
                confirmation_evidence.candidate_keys != primary_evidence.candidate_keys
                or confirmation_evidence.selection_by_key
                != primary_evidence.selection_by_key
            ):
                raise ValueError("Step 6c confirmation candidate evidence changed")
            confirmation_keys = confirmation_evidence.evaluation_keys
            confirmation_regret = _finite(
                confirmation_payload.get("selected_regret"),
                "confirmation selected regret",
                minimum=0.0,
            )
        elif confirmation_payload is not None:
            raise ValueError("Step 6c unexpected confirmation payload")
        if (
            primary_evidence.candidate_keys & primary_evidence.evaluation_keys
            or primary_evidence.candidate_keys & confirmation_keys
            or primary_evidence.evaluation_keys & confirmation_keys
        ):
            raise ValueError("Step 6c candidate/evaluation/confirmation RNG overlap")
        evidence_rows.append(
            RowEvidence(
                candidate_keys=primary_evidence.candidate_keys,
                evaluation_keys=primary_evidence.evaluation_keys,
                confirmation_keys=confirmation_keys,
                confirmation_regret=confirmation_regret,
                primary_wall_seconds=primary_seconds,
                confirmation_wall_seconds=(
                    float(confirmation_seconds)
                    if confirmation_seconds is not None
                    else None
                ),
                legal_action_count=int(row["legal_action_count"]),
                seat=observation.seat,
            )
        )
    return evidence_rows[0], evidence_rows[1]


def _latency(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "mean_seconds": 0.0,
            "p50_seconds": 0.0,
            "p95_seconds": 0.0,
            "p99_seconds": 0.0,
            "max_seconds": 0.0,
        }

    def percentile(fraction: float) -> float:
        index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "mean_seconds": statistics.fmean(ordered),
        "p50_seconds": percentile(0.50),
        "p95_seconds": percentile(0.95),
        "p99_seconds": percentile(0.99),
        "max_seconds": max(ordered),
    }


def _ordered_missing_tasks(
    tasks: Sequence[Mapping[str, Any]], *, resume_drill: bool
) -> list[Mapping[str, Any]]:
    """Keep production confirmation-first, but make the resume smoke bounded."""

    return sorted(
        tasks,
        key=lambda task: (
            (
                int(task["global_hand_index"]) in CONFIRMATION_HAND_INDICES
                if resume_drill
                else int(task["global_hand_index"]) not in CONFIRMATION_HAND_INDICES
            ),
            int(task["global_hand_index"]),
        ),
    )


def run_shard(
    *,
    manifest_path: Path,
    schedule_path: Path,
    source_package_sha256: str,
    repository_root: Path,
    shard: int,
    output_dir: Path,
    parity_report_path: Path,
    stop_after_tasks: int | None = None,
) -> dict[str, Any]:
    manifest, spec, manifest_sha = _load_manifest_and_spec(
        manifest_path=manifest_path,
        schedule_path=schedule_path,
        source_package_sha256=source_package_sha256,
        shard=shard,
    )
    _verify_package_files(manifest, repository_root)
    parity = _read_json(parity_report_path)
    _validate_parity_report(
        parity,
        source_package_sha256=source_package_sha256,
        manifest_sha256=manifest_sha,
        parity_golden_sha256=str(manifest["parity_golden_sha256"]),
        native_library_sha256=str(manifest["native_library"]["sha256"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output_dir / "heartbeat.json"
    root_tasks = _materialize_roots(
        manifest_sha256=manifest_sha,
        spec=spec,
        root_dir=output_dir / "roots",
        heartbeat_path=heartbeat_path,
    )
    task_dir = output_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[int, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    native = manifest["native_library"]
    for root_task in root_tasks:
        index = int(root_task["global_hand_index"])
        path = task_dir / f"hand_{index:03d}.json"
        if path.exists():
            task = _read_json(path)
            _validate_search_task(
                task, root_task=root_task, library_sha256=str(native["sha256"])
            )
            existing[index] = task
        else:
            missing.append(root_task)
    resumed_task_count = len(existing)
    missing = _ordered_missing_tasks(missing, resume_drill=False)
    selected = missing
    if stop_after_tasks is not None:
        if isinstance(stop_after_tasks, bool) or stop_after_tasks <= 0:
            raise ValueError("stop_after_tasks must be positive")
        selected = _ordered_missing_tasks(missing, resume_drill=True)[:stop_after_tasks]
    _write_heartbeat(
        heartbeat_path,
        status="running_search",
        total_tasks=spec.hand_count,
        completed_tasks=len(existing),
        pending_tasks=len(missing),
        resumed_task_count=resumed_task_count,
    )
    if selected:
        os.environ["RAYON_NUM_THREADS"] = str(RAYON_THREADS)
        max_workers = 1 if stop_after_tasks is not None else WORKERS
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {
                pool.submit(
                    _search_worker,
                    {
                        "root_task": root_task,
                        "task_path": str(
                            task_dir
                            / f"hand_{int(root_task['global_hand_index']):03d}.json"
                        ),
                        "library_path": str(repository_root / str(native["path"])),
                        "library_sha256": str(native["sha256"]),
                    },
                ): root_task
                for root_task in selected
            }
            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=5.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    future.result()
                completed = len(list(task_dir.glob("hand_*.json")))
                _write_heartbeat(
                    heartbeat_path,
                    status="running_search",
                    total_tasks=spec.hand_count,
                    completed_tasks=completed,
                    pending_tasks=spec.hand_count - completed,
                    resumed_task_count=resumed_task_count,
                )
    remaining = [
        root_task
        for root_task in root_tasks
        if not (
            task_dir / f"hand_{int(root_task['global_hand_index']):03d}.json"
        ).exists()
    ]
    if remaining:
        report = {
            "schema": STEP6C_SUMMARY_SCHEMA,
            "status": "interrupted_for_resume_drill",
            "run_name": spec.run_name,
            "shard": spec.shard,
            "hand_indices": list(spec.hand_indices),
            "source_package_sha256": source_package_sha256,
            "manifest_sha256": manifest_sha,
            "parity_report_sha256": _sha256(parity_report_path),
            "completed_tasks": spec.hand_count - len(remaining),
            "pending_tasks": len(remaining),
            "resumed_task_count": resumed_task_count,
            "quality_pilot_authorized": True,
            "training_eligible": False,
            "current_profile_changed": False,
            "production_fanout_authorized": False,
            "m31_complete": False,
        }
        _write_heartbeat(
            heartbeat_path,
            status=report["status"],
            total_tasks=spec.hand_count,
            completed_tasks=report["completed_tasks"],
            pending_tasks=report["pending_tasks"],
            resumed_task_count=resumed_task_count,
        )
        return report
    reports: list[dict[str, Any]] = []
    evidence: list[RowEvidence] = []
    for root_task in root_tasks:
        path = task_dir / f"hand_{int(root_task['global_hand_index']):03d}.json"
        task = _read_json(path)
        evidence.extend(
            _validate_search_task(
                task, root_task=root_task, library_sha256=str(native["sha256"])
            )
        )
        reports.append(task)
    candidate_keys = set().union(*(row.candidate_keys for row in evidence))
    evaluation_keys = set().union(*(row.evaluation_keys for row in evidence))
    confirmation_keys = set().union(*(row.confirmation_keys for row in evidence))
    confirmation_regrets = [
        row.confirmation_regret
        for row in evidence
        if row.confirmation_regret is not None
    ]
    primary_latency = {
        seat: _latency(
            [row.primary_wall_seconds for row in evidence if row.seat == seat]
        )
        for seat in ("first", "second")
    }
    confirmation_latency = {
        seat: _latency(
            [
                float(row.confirmation_wall_seconds)
                for row in evidence
                if row.seat == seat and row.confirmation_wall_seconds is not None
            ]
        )
        for seat in ("first", "second")
    }
    summary_path = output_dir / "summary.json"
    existing_summary = _read_json(summary_path) if summary_path.exists() else None
    summary_resumed_task_count = resumed_task_count
    if existing_summary is not None:
        existing_resumed = existing_summary.get("resumed_task_count")
        if (
            existing_summary.get("schema") != STEP6C_SUMMARY_SCHEMA
            or existing_summary.get("run_name") != spec.run_name
            or existing_summary.get("shard") != spec.shard
            or existing_summary.get("hand_indices") != list(spec.hand_indices)
            or existing_summary.get("source_package_sha256") != source_package_sha256
            or existing_summary.get("manifest_sha256") != manifest_sha
            or isinstance(existing_resumed, bool)
            or not isinstance(existing_resumed, int)
            or not 0 <= existing_resumed <= spec.hand_count
        ):
            raise ValueError("existing Step 6c final summary provenance changed")
        summary_resumed_task_count = existing_resumed
    peak_rss = max(int(report["memory"]["peak_rss_bytes"]) for report in reports)
    expected_confirmation_roots = 2 * sum(
        index in CONFIRMATION_HAND_INDICES for index in spec.hand_indices
    )
    profile_counts = {
        profile: sum(report["profile"] == profile for report in reports)
        for profile in M31_T3_BEHAVIOR_PROFILES
    }
    geometry_counts = {
        str(count): sum(row.legal_action_count == count for row in evidence)
        for count in sorted({row.legal_action_count for row in evidence})
    }
    gates = {
        "exact_shard_hand_indices": [
            int(report["global_hand_index"]) for report in reports
        ]
        == list(spec.hand_indices),
        "exactly_50_roots": len(evidence) == spec.root_count,
        "exactly_25_each_seat": (
            sum(row.seat == "first" for row in evidence)
            == sum(row.seat == "second" for row in evidence)
            == spec.hand_count
        ),
        "exact_confirmation_root_count": len(confirmation_regrets)
        == expected_confirmation_roots,
        "unique_observation_fingerprints": len(
            {
                row["observation_fingerprint"]
                for report in reports
                for row in report["rows"]
            }
        )
        == len(evidence),
        "candidate_rng_unique": len(candidate_keys) == len(evidence) * 8,
        "evaluation_rng_unique": len(evaluation_keys) == len(evidence) * 32,
        "confirmation_rng_unique": len(confirmation_keys)
        == expected_confirmation_roots * 128,
        "candidate_evaluation_confirmation_rng_disjoint": not (
            candidate_keys & evaluation_keys
            or candidate_keys & confirmation_keys
            or evaluation_keys & confirmation_keys
        ),
        "resume_drill_recovered_task": summary_resumed_task_count >= 1,
        "first_p95_within_180_seconds": primary_latency["first"]["p95_seconds"]
        <= MAX_FIRST_P95_SECONDS,
        "second_p95_within_6_seconds": primary_latency["second"]["p95_seconds"]
        <= MAX_SECOND_P95_SECONDS,
        "peak_rss_within_1_gib": peak_rss <= MAX_PEAK_RSS_BYTES,
        "geometry_diagnostic_only": True,
        "quality_thresholds_deferred_to_merged_validator": True,
        "no_training_current_profile_or_production_fanout": True,
    }
    if set(gates) != STEP6C_SHARD_GATE_KEYS:
        raise RuntimeError("Step 6c shard gate schema changed")
    summary = {
        "schema": STEP6C_SUMMARY_SCHEMA,
        "status": "pass" if all(gates.values()) else "no_go",
        "run_name": spec.run_name,
        "shard": spec.shard,
        "hand_indices": list(spec.hand_indices),
        "source_package_sha256": source_package_sha256,
        "manifest_sha256": manifest_sha,
        "parity_report_sha256": _sha256(parity_report_path),
        "native_library_sha256": native["sha256"],
        "contract": {
            "paired_hands": spec.hand_count,
            "roots": spec.root_count,
            "workers": WORKERS,
            "rayon_threads_per_worker": RAYON_THREADS,
            "primary_budget": _PRIMARY_BUDGET,
            "confirmation_budget": _CONFIRMATION_BUDGET,
            "run_id": STEP6C_RUN_ID,
        },
        "integrity": {
            "candidate_rng_keys": len(candidate_keys),
            "evaluation_rng_keys": len(evaluation_keys),
            "confirmation_rng_keys": len(confirmation_keys),
            "candidate_evaluation_overlap": len(candidate_keys & evaluation_keys),
            "candidate_confirmation_overlap": len(candidate_keys & confirmation_keys),
            "evaluation_confirmation_overlap": len(evaluation_keys & confirmation_keys),
            "profile_hand_counts": profile_counts,
            "legal_action_geometry_counts": geometry_counts,
        },
        "confirmation": {
            "root_count": len(confirmation_regrets),
            "selected_regrets": confirmation_regrets,
            "quality_thresholds_applied": False,
        },
        "performance": {
            "primary_latency_by_seat": primary_latency,
            "confirmation_latency_by_seat": confirmation_latency,
            "peak_process_rss_bytes": peak_rss,
        },
        "resumed_task_count": summary_resumed_task_count,
        "completed_tasks": spec.hand_count,
        "pending_tasks": 0,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "task_manifest": [
            {
                "global_hand_index": report["global_hand_index"],
                "sha256": _sha256(
                    task_dir / f"hand_{int(report['global_hand_index']):03d}.json"
                ),
            }
            for report in reports
        ],
        "teacher_value_status": "diagnostic_not_match_EV",
        "quality_pilot_authorized": True,
        "training_eligible": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "production_fanout_authorized": False,
        "m31_complete": False,
    }
    summary = _commit_or_validate_final_summary(summary_path, summary, existing_summary)
    _write_heartbeat(
        heartbeat_path,
        status=summary["status"],
        total_tasks=spec.hand_count,
        completed_tasks=spec.hand_count,
        pending_tasks=0,
        resumed_task_count=summary_resumed_task_count,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--source-package-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parity-golden", type=Path, required=True)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--stop-after-tasks", type=int)
    parser.add_argument("--report-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest, _spec, manifest_sha = _load_manifest_and_spec(
        manifest_path=args.manifest,
        schedule_path=args.schedule,
        source_package_sha256=args.source_package_sha256,
        shard=args.shard,
    )
    _verify_package_files(manifest, args.repository_root)
    parity_path = args.output_dir / "parity.json"
    parity = run_linux_parity(
        manifest=manifest,
        source_package_sha256=args.source_package_sha256,
        manifest_sha256=manifest_sha,
        repository_root=args.repository_root,
        parity_golden_path=args.parity_golden,
        output_path=parity_path,
    )
    if args.parity_only:
        if args.report_output is not None:
            _write_once(args.report_output, parity)
        print(json.dumps(parity, sort_keys=True))
        return 0
    result = run_shard(
        manifest_path=args.manifest,
        schedule_path=args.schedule,
        source_package_sha256=args.source_package_sha256,
        repository_root=args.repository_root,
        shard=args.shard,
        output_dir=args.output_dir,
        parity_report_path=parity_path,
        stop_after_tasks=args.stop_after_tasks,
    )
    if args.report_output is not None:
        _write_once(args.report_output, result)
    print(json.dumps(result, sort_keys=True))
    return 75 if result["status"] == "interrupted_for_resume_drill" else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZED_SHARDS",
    "EXPECTED_FEATURE_ENCODER_SHA256",
    "EXPECTED_NATIVE_LIBRARY_SHA256",
    "STEP6C_CONFIRMATION_SCHEMA",
    "STEP6C_HEARTBEAT_SCHEMA",
    "STEP6C_PACKAGE_SCHEMA",
    "STEP6C_PARITY_SCHEMA",
    "STEP6C_ROOT_TASK_SCHEMA",
    "STEP6C_SEARCH_TASK_SCHEMA",
    "STEP6C_SHARD_GATE_KEYS",
    "STEP6C_SHARD_SCHEMA",
    "STEP6C_SUMMARY_SCHEMA",
    "ShardSpec",
    "_PRIMARY_BUDGET",
    "_CONFIRMATION_BUDGET",
    "_load_manifest_and_spec",
    "_validate_confirmation_pair",
    "_validate_decision_payload",
    "_validate_parity_report",
    "_validate_search_task",
    "run_linux_parity",
    "run_shard",
]
