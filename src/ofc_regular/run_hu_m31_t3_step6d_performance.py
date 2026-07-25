"""Run the restart-safe M3.1 Step 6d performance-development comparison.

This runner is deliberately limited to the repeatable ``performance_development``
set frozen by the Step 6d contract.  It compares an explicitly supplied reference
native library with an explicitly supplied candidate native library.  It does not
read Step 6c labels, resolve ``current``, train a model, activate a profile, or
start cloud work.

Every native decision is validated by :class:`HuM31T3SearchSolver` before it
reaches this module.  Cross-binary parity is then recomputed from a portable,
ActionKey-ordered payload that excludes binary identity, measured latency, and
binary-bound result certificates.  All ActionKeys, candidate-selection Q values,
locked-evaluation Q values, the selected action, RNG evidence, and child
information-set count must match exactly.
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
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    action_key_from_payload,
    legal_action_set_digest,
)
from .action_space import generate_turn_actions
from .ai_profiles import ModelPaths, load_model_bundle
from .hu_infoset import ActorObservation
from .hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    behavior_profile_for_index,
    generate_behavior_t3_roots,
)
from .hu_m31_t3_runtime import HuM31T3RuntimeConfig, HuM31T3SearchSolver
from .hu_m31_t3_step6d_contract import (
    EXPECTED_STEP6D_CONTRACT_BYTE_SHA256,
    EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
    SEED_STRIDE,
    STEP6D_RUN_ID,
    canonical_sha256,
    schedule_by_name,
    validate_seed_schedule,
)
from .validate_hu_m31_t3_profile import process_memory_snapshot


STEP6D_PERFORMANCE_ROOT_SCHEMA = "hu_m31_t3_step6d_performance_root_v1"
STEP6D_PERFORMANCE_HAND_SCHEMA = "hu_m31_t3_step6d_performance_hand_v1"
STEP6D_PERFORMANCE_SUMMARY_SCHEMA = "hu_m31_t3_step6d_performance_summary_v1"
STEP6D_PORTABLE_PARITY_SCHEMA = "hu_m31_t3_step6d_portable_parity_v1"
STEP6D_PERFORMANCE_SCHEDULE = "performance_development"
REFERENCE_NATIVE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)

PERFORMANCE_HAND_INDICES = tuple(range(100))
PERFORMANCE_BUDGET = {
    "candidate_samples": 8,
    "evaluation_samples": 32,
    "downstream_t3_samples": 4,
    "downstream_t4_samples": 0,
}
ALLOWED_ALLOCATIONS = frozenset({(1, 16), (2, 8), (4, 4)})
PERFORMANCE_GATE_LIMITS = {
    "first_p95_seconds_max": 150.0,
    "first_p99_and_max_seconds_max": 240.0,
    "second_p95_seconds_max": 5.0,
    "peak_rss_bytes_max": 858_993_459,
}

_CONTRACT_RELATIVE_PATH = Path("configs/hu_joint_policy_m31_t3_step6d_contract.json")
_FORBIDDEN_FIELDS = frozenset(
    {
        "opponent_private_discards",
        "true_dead_cards",
        "remaining_deck",
        "world_state",
        "replay_truth",
        "draw_pile",
        "future_cards",
    }
)
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


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _memory_peak(value: Mapping[str, Any], label: str) -> int | None:
    if not isinstance(value.get("supported"), bool) or not isinstance(
        value.get("source"), str
    ):
        raise ValueError(f"{label} memory snapshot identity changed")
    peak = value.get("peak_rss_bytes")
    if peak is not None:
        _integer(peak, f"{label} peak RSS")
    for field in ("rss_bytes", "private_bytes"):
        item = value.get(field)
        if item is not None:
            _integer(item, f"{label} {field}")
    return peak


def _validate_memory(value: Mapping[str, Any]) -> int | None:
    if set(value) != {
        "before_solver_load",
        "after_solver_load",
        "after_hand",
        "peak_rss_bytes",
    }:
        raise ValueError("Step 6d hand memory schema changed")
    peaks: list[int] = []
    for field in ("before_solver_load", "after_solver_load", "after_hand"):
        snapshot = value.get(field)
        if not isinstance(snapshot, Mapping):
            raise ValueError("Step 6d hand memory snapshot changed")
        peak = _memory_peak(snapshot, field)
        if peak is not None:
            peaks.append(peak)
    expected = max(peaks) if peaks else None
    if value.get("peak_rss_bytes") != expected:
        raise ValueError("Step 6d hand peak RSS changed")
    return expected


def _read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    if raw != _canonical_bytes(payload):
        raise ValueError(f"JSON artifact is not canonical and immutable: {path}")
    return payload


def _write_once(path: Path, value: Any) -> None:
    """Atomically publish one immutable artifact, refusing every overwrite."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite Step 6d artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _reject_hidden(value: Any, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            token = str(key).casefold()
            if token in _FORBIDDEN_FIELDS:
                raise ValueError(f"forbidden hidden-information field at {path}.{key}")
            _reject_hidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_hidden(child, f"{path}[{index}]")


def normalize_hand_indices(indices: Iterable[int] | None = None) -> tuple[int, ...]:
    if indices is None:
        return PERFORMANCE_HAND_INDICES
    checked: list[int] = []
    for value in indices:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Step 6d performance hand indices must be integers")
        if value not in PERFORMANCE_HAND_INDICES:
            raise ValueError("Step 6d performance hand index must be in 0..99")
        checked.append(value)
    if not checked:
        raise ValueError("Step 6d performance subset must not be empty")
    if len(set(checked)) != len(checked):
        raise ValueError("Step 6d performance hand indices must be unique")
    return tuple(sorted(checked))


def performance_seed_values(index: int) -> dict[str, int]:
    checked = normalize_hand_indices((index,))[0]
    schedule = schedule_by_name(STEP6D_PERFORMANCE_SCHEDULE)
    return {
        key: base + SEED_STRIDE * checked
        for key, base in zip(
            schedule.namespace_keys, schedule.namespace_bases, strict=True
        )
    }


def performance_schedule_row(index: int) -> dict[str, Any]:
    checked = normalize_hand_indices((index,))[0]
    return {
        "schedule": STEP6D_PERFORMANCE_SCHEDULE,
        "hand_index": checked,
        "root_indices": [checked * 2, checked * 2 + 1],
        "profile": behavior_profile_for_index(checked),
        "seeds": performance_seed_values(checked),
        "budget": dict(PERFORMANCE_BUDGET),
        "training_eligible": False,
    }


def validate_performance_contract(contract_path: Path) -> dict[str, Any]:
    """Validate only the frozen Step 6d contract; no historical labels are read."""

    contract_path = Path(contract_path).resolve()
    raw = contract_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_STEP6D_CONTRACT_BYTE_SHA256:
        raise ValueError("Step 6d contract byte SHA-256 changed")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("Step 6d contract must be an object")
    if canonical_sha256(payload) != EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256:
        raise ValueError("Step 6d canonical contract changed")
    schedule = schedule_by_name(STEP6D_PERFORMANCE_SCHEDULE)
    seed_rows = payload.get("seed_contract", {}).get("schedules")
    substep = payload.get("substeps", {}).get(STEP6D_PERFORMANCE_SCHEDULE)
    reuse = payload.get("reuse_matrix", {}).get(STEP6D_PERFORMANCE_SCHEDULE)
    expected_substep = {
        "schedule": STEP6D_PERFORMANCE_SCHEDULE,
        "paired_hands": 100,
        "roots": 200,
        "hands_per_behavior_profile": 20,
        "first_roots": 100,
        "second_roots": 100,
        "search_budget": dict(PERFORMANCE_BUDGET),
        "confirmation_executed": False,
        "repeatable": True,
        "allowed_allocations": ["1x16", "2x8", "4x4"],
        "candidate_freeze_gates": {
            "portable_semantic_parity_fraction": 1.0,
            **PERFORMANCE_GATE_LIMITS,
        },
        "training_eligible": False,
        "failure_action": (
            "continue_engineering_on_same_development_set_without_opening_"
            "performance_lock"
        ),
    }
    if (
        not isinstance(seed_rows, list)
        or [row for row in seed_rows if row.get("name") == schedule.name]
        != [schedule.to_dict()]
        or substep != expected_substep
        or reuse
        != {
            "allowed_use": "repeatable_performance_engineering_only",
            "training_eligible": False,
            "threshold_eligible": False,
            "quality_evidence": False,
            "promotion_evidence": False,
        }
    ):
        raise ValueError("Step 6d performance-development contract changed")
    seed_audit = validate_seed_schedule()
    rows = [performance_schedule_row(index) for index in PERFORMANCE_HAND_INDICES]
    profile_counts = {
        profile: sum(row["profile"] == profile for row in rows)
        for profile in M31_T3_BEHAVIOR_PROFILES
    }
    seeds = [seed for row in rows for seed in row["seeds"].values()]
    if (
        profile_counts != {profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES}
        or len(seeds) != 600
        or len(set(seeds)) != 600
    ):
        raise ValueError("Step 6d performance schedule quota changed")
    return {
        "contract_byte_sha256": EXPECTED_STEP6D_CONTRACT_BYTE_SHA256,
        "contract_canonical_sha256": EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
        "schedule": schedule.to_dict(),
        "profile_counts": profile_counts,
        "performance_seed_count": len(seeds),
        "planned_seed_set_sha256": seed_audit["planned_seed_set_sha256"],
    }


def _absolute_model_paths(repository_root: Path) -> ModelPaths:
    defaults = ModelPaths()
    values: dict[str, Any] = {}
    for field in fields(ModelPaths):
        value = getattr(defaults, field.name)
        if isinstance(value, Path):
            values[field.name] = (
                value if value.is_absolute() else repository_root / value
            )
        elif isinstance(value, tuple) and all(isinstance(item, Path) for item in value):
            values[field.name] = tuple(
                item if item.is_absolute() else repository_root / item for item in value
            )
        else:
            values[field.name] = value
    return ModelPaths(**values)


def _validate_root_artifact(
    value: Mapping[str, Any], *, expected_row: Mapping[str, Any]
) -> tuple[ActorObservation, ActorObservation]:
    if (
        set(value) != _ROOT_KEYS
        or value.get("schema") != STEP6D_PERFORMANCE_ROOT_SCHEMA
    ):
        raise ValueError("Step 6d performance root schema changed")
    observations = value.get("observations")
    expected_digest = _digest(expected_row)
    if (
        value.get("contract_canonical_sha256")
        != EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
        or value.get("schedule") != STEP6D_PERFORMANCE_SCHEDULE
        or value.get("schedule_row_sha256") != expected_digest
        or value.get("hand_index") != expected_row["hand_index"]
        or value.get("root_indices") != expected_row["root_indices"]
        or value.get("profile") != expected_row["profile"]
        or value.get("seeds") != expected_row["seeds"]
        or value.get("budget") != PERFORMANCE_BUDGET
        or value.get("current_profile_resolved") is not False
        or value.get("opponent_private_discards_used") is not False
        or value.get("training_eligible") is not False
        or not isinstance(observations, list)
        or len(observations) != 2
    ):
        raise ValueError("Step 6d performance root provenance changed")
    parsed: list[ActorObservation] = []
    for offset, raw in enumerate(observations):
        if not isinstance(raw, Mapping) or set(raw) != _ROOT_OBSERVATION_KEYS:
            raise ValueError("Step 6d performance root observation schema changed")
        observation = ActorObservation.from_dict(raw["observation"])
        expected_seat = "first" if offset == 0 else "second"
        if (
            raw.get("root_index") != expected_row["root_indices"][offset]
            or raw.get("seat") != expected_seat
            or observation.seat != expected_seat
            or observation.to_act_order != expected_seat
            or observation.street != "T3"
            or raw.get("observation_fingerprint") != observation.fingerprint()
        ):
            raise ValueError("Step 6d performance root observation changed")
        _reject_hidden(raw["observation"], "root.observation")
        parsed.append(observation)
    return parsed[0], parsed[1]


def _materialize_roots(
    *, repository_root: Path, output_dir: Path, indices: Sequence[int]
) -> list[dict[str, Any]]:
    root_dir = output_dir / "roots"
    root_dir.mkdir(parents=True, exist_ok=True)
    bundle = None
    roots: list[dict[str, Any]] = []
    model_profiles = set(M31_T3_BEHAVIOR_PROFILES) - {"random_exact_final"}
    for index in indices:
        schedule_row = performance_schedule_row(index)
        path = root_dir / f"hand_{index:03d}.json"
        if path.exists():
            value = _read_json(path)
            _validate_root_artifact(value, expected_row=schedule_row)
            roots.append(value)
            continue
        if bundle is None:
            bundle = load_model_bundle(
                _absolute_model_paths(repository_root), profiles=model_profiles
            )
        observations = generate_behavior_t3_roots(
            hand_seed=schedule_row["seeds"]["hand"],
            behavior_seed=schedule_row["seeds"]["behavior"],
            profile=schedule_row["profile"],
            bundle=bundle,
        )
        value = {
            "schema": STEP6D_PERFORMANCE_ROOT_SCHEMA,
            "contract_canonical_sha256": EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
            "schedule": STEP6D_PERFORMANCE_SCHEDULE,
            "schedule_row_sha256": _digest(schedule_row),
            "hand_index": index,
            "root_indices": schedule_row["root_indices"],
            "profile": schedule_row["profile"],
            "seeds": schedule_row["seeds"],
            "budget": dict(PERFORMANCE_BUDGET),
            "observations": [
                {
                    "root_index": schedule_row["root_indices"][offset],
                    "seat": observation.seat,
                    "observation_fingerprint": observation.fingerprint(),
                    "observation": observation.to_dict(),
                }
                for offset, observation in enumerate(observations)
            ],
            "current_profile_resolved": False,
            "opponent_private_discards_used": False,
            "training_eligible": False,
        }
        _validate_root_artifact(value, expected_row=schedule_row)
        _write_once(path, value)
        roots.append(value)
    return roots


def _action_set_digest(tokens: Sequence[str]) -> str:
    ordered = sorted(tokens, key=lambda token: ActionKey.from_token(token).sort_key())
    return hashlib.sha256("\n".join(ordered).encode("ascii")).hexdigest()


def portable_parity_payload(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Build a binary-, latency-, and enumeration-order-independent payload."""

    raw_rows = decision.get("action_values")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("Step 6d portable decision requires all action values")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    ranks: set[int] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != _ACTION_VALUE_KEYS:
            raise ValueError("Step 6d portable action-value schema changed")
        token = raw.get("action_key")
        if not isinstance(token, str):
            raise ValueError("Step 6d portable action lacks ActionKey")
        key = ActionKey.from_token(token)
        if token in seen or action_key_from_payload(raw) != key:
            raise ValueError("Step 6d portable ActionKey mapping changed")
        rank = _integer(raw.get("rank"), "action rank")
        selection_q = _finite(raw.get("selection_ev"), "selection Q")
        evaluation_q = _finite(raw.get("evaluation_ev"), "evaluation Q")
        seen.add(token)
        ranks.add(rank)
        rows.append(
            {
                "action_key": token,
                "rank": rank,
                "selection_q": selection_q,
                "evaluation_q": evaluation_q,
            }
        )
    if ranks != set(range(len(rows))):
        raise ValueError("Step 6d portable action rank coverage changed")
    rows.sort(key=lambda row: ActionKey.from_token(row["action_key"]).sort_key())
    tokens = [row["action_key"] for row in rows]
    selected_key = decision.get("selected_action_key")
    selected = decision.get("selected_action")
    by_key = {row["action_key"]: row for row in rows}
    if (
        selected_key not in by_key
        or not isinstance(selected, Mapping)
        or set(selected) != {"placements", "discards"}
        or action_key_from_payload(selected).to_token() != selected_key
        or _finite(decision.get("selected_selection_ev"), "selected selection Q")
        != by_key[selected_key]["selection_q"]
        or _finite(decision.get("selected_evaluation_ev"), "selected evaluation Q")
        != by_key[selected_key]["evaluation_q"]
    ):
        raise ValueError("Step 6d portable selected action changed")
    legal_digest = decision.get("legal_action_set_digest")
    if legal_digest != _action_set_digest(tokens):
        raise ValueError("Step 6d portable legal ActionKey set changed")
    digest_fields = (
        "candidate_belief_digest",
        "evaluation_belief_digest",
        "candidate_rng_digest",
        "evaluation_rng_digest",
        "search_contract_digest",
    )
    if any(not _is_sha256(decision.get(field)) for field in digest_fields):
        raise ValueError("Step 6d portable RNG/search digest changed")
    child_count = _integer(
        decision.get("child_information_set_count"), "child information-set count"
    )
    budget = {
        "candidate_samples": _integer(
            decision.get("candidate_samples"), "candidate samples", minimum=1
        ),
        "evaluation_samples": _integer(
            decision.get("evaluation_samples"), "evaluation samples", minimum=1
        ),
        "downstream_t3_samples": _integer(
            decision.get("downstream_t3_samples"),
            "downstream T3 samples",
            minimum=1,
        ),
        "downstream_t4_samples": _integer(
            decision.get("downstream_t4_samples"), "downstream T4 samples"
        ),
    }
    return {
        "schema": STEP6D_PORTABLE_PARITY_SCHEMA,
        "seat": decision.get("seat"),
        "observation_fingerprint": decision.get("observation_fingerprint"),
        "action_key_schema": decision.get("action_key_schema"),
        "legal_action_set_digest": legal_digest,
        "action_values": rows,
        "selected_action_key": selected_key,
        "selected_action": {
            "action_key": selected_key,
            "top": list(ActionKey.from_token(selected_key).cards("top")),
            "middle": list(ActionKey.from_token(selected_key).cards("middle")),
            "bottom": list(ActionKey.from_token(selected_key).cards("bottom")),
            "discards": list(ActionKey.from_token(selected_key).cards("discards")),
        },
        "selected_selection_q": by_key[selected_key]["selection_q"],
        "selected_evaluation_q": by_key[selected_key]["evaluation_q"],
        "rng": {
            "run_id": decision.get("run_id"),
            "continuation_seed": _integer(
                decision.get("continuation_seed"), "continuation seed"
            ),
            "candidate_seed": _integer(
                decision.get("candidate_seed"), "candidate seed"
            ),
            "evaluation_seed": _integer(
                decision.get("evaluation_seed"), "evaluation seed"
            ),
            "candidate_belief_digest": decision["candidate_belief_digest"],
            "evaluation_belief_digest": decision["evaluation_belief_digest"],
            "candidate_rng_digest": decision["candidate_rng_digest"],
            "evaluation_rng_digest": decision["evaluation_rng_digest"],
            "search_contract_digest": decision["search_contract_digest"],
        },
        "budget": budget,
        "child_information_set_count": child_count,
    }


def compare_portable_decisions(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    reference_payload = portable_parity_payload(reference)
    candidate_payload = portable_parity_payload(candidate)
    reference_rows = reference_payload["action_values"]
    candidate_rows = candidate_payload["action_values"]
    action_keys_exact = [row["action_key"] for row in reference_rows] == [
        row["action_key"] for row in candidate_rows
    ]
    selection_q_exact = action_keys_exact and [
        (row["action_key"], row["selection_q"]) for row in reference_rows
    ] == [(row["action_key"], row["selection_q"]) for row in candidate_rows]
    evaluation_q_exact = action_keys_exact and [
        (row["action_key"], row["evaluation_q"]) for row in reference_rows
    ] == [(row["action_key"], row["evaluation_q"]) for row in candidate_rows]
    selected_action_exact = all(
        reference_payload[field] == candidate_payload[field]
        for field in (
            "selected_action_key",
            "selected_action",
            "selected_selection_q",
            "selected_evaluation_q",
        )
    )
    rng_exact = (
        reference_payload["rng"] == candidate_payload["rng"]
        and reference_payload["budget"] == candidate_payload["budget"]
    )
    child_exact = (
        reference_payload["child_information_set_count"]
        == candidate_payload["child_information_set_count"]
    )
    payload_exact = reference_payload == candidate_payload
    return {
        "schema": STEP6D_PORTABLE_PARITY_SCHEMA,
        "reference_portable_sha256": _digest(reference_payload),
        "candidate_portable_sha256": _digest(candidate_payload),
        "action_keys_exact": action_keys_exact,
        "selection_q_exact": selection_q_exact,
        "evaluation_q_exact": evaluation_q_exact,
        "selected_action_exact": selected_action_exact,
        "rng_exact": rng_exact,
        "child_information_set_count_exact": child_exact,
        "portable_payload_exact": payload_exact,
    }


def _validate_decision(
    decision: Mapping[str, Any],
    *,
    observation: ActorObservation,
    seeds: Mapping[str, int],
    library_sha256: str,
) -> None:
    if set(decision) != _DECISION_KEYS:
        raise ValueError("Step 6d runtime decision schema changed")
    portable = portable_parity_payload(decision)
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    expected_tokens = sorted(
        (action_key(action).to_token() for action in legal),
        key=lambda token: ActionKey.from_token(token).sort_key(),
    )
    observed_tokens = [row["action_key"] for row in portable["action_values"]]
    for field in ("native_latency_ms", "validation_latency_ms", "total_latency_ms"):
        _finite(decision.get(field), field, minimum=0.0)
    if (
        decision.get("schema") != "hu_m31_t3_runtime_decision_v2"
        or decision.get("seat") != observation.seat
        or decision.get("observation_fingerprint") != observation.fingerprint()
        or decision.get("action_key_schema") != ACTION_KEY_SCHEMA
        or observed_tokens != expected_tokens
        or decision.get("legal_action_set_digest") != legal_action_set_digest(legal)
        or decision.get("run_id") != STEP6D_RUN_ID
        or decision.get("continuation_seed") != seeds["child"]
        or decision.get("candidate_seed") != seeds["candidate"]
        or decision.get("evaluation_seed") != seeds["evaluation"]
        or portable["budget"] != PERFORMANCE_BUDGET
        or decision.get("downstream_t4_mode") != "exact"
        or decision.get("use_t4_action_cache") is not True
        or decision.get("native_library_sha256") != library_sha256
        or decision.get("execution_mode") != "scalar"
        or decision.get("batch_size") != 1
        or decision.get("teacher_value_status") != "diagnostic_not_match_EV"
    ):
        raise ValueError("Step 6d runtime decision contract changed")
    _reject_hidden(decision, "decision")


def _geometry(
    observation: ActorObservation, decision: Mapping[str, Any]
) -> dict[str, Any]:
    def open_slots(board: Any) -> dict[str, int]:
        return {
            "top": 3 - len(board.top),
            "middle": 5 - len(board.middle),
            "bottom": 5 - len(board.bottom),
        }

    return {
        "hero_open_slots": open_slots(observation.hero_board),
        "opponent_open_slots": open_slots(observation.opponent_public_board),
        "dealt_card_count": len(observation.dealt_cards),
        "legal_action_count": len(decision["action_values"]),
        "child_information_set_count": decision["child_information_set_count"],
    }


def _solver(
    *, library_path: Path, library_sha256: str, seeds: Mapping[str, int]
) -> HuM31T3SearchSolver:
    return HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=library_sha256,
            library_path=library_path,
            run_id=STEP6D_RUN_ID,
            candidate_samples=8,
            evaluation_samples=32,
            downstream_t3_samples=4,
            seed=seeds["child"],
            candidate_seed=seeds["candidate"],
            evaluation_seed=seeds["evaluation"],
        )
    )


def _solve_source(
    *,
    source: str,
    solver: HuM31T3SearchSolver,
    observation: ActorObservation,
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    started = time.perf_counter()
    decision = solver.solve(observation).to_dict()
    wall_seconds = time.perf_counter() - started
    _validate_decision(
        decision,
        observation=observation,
        seeds=seeds,
        library_sha256=solver.library_sha256,
    )
    return {
        "source": source,
        "native_library_sha256": solver.library_sha256,
        "solve_wall_seconds": wall_seconds,
        "native_seconds": float(decision["native_latency_ms"]) / 1000.0,
        "validation_seconds": float(decision["validation_latency_ms"]) / 1000.0,
        "runtime_total_seconds": float(decision["total_latency_ms"]) / 1000.0,
        "rss_after": process_memory_snapshot(),
        "decision": decision,
    }


def _run_hand_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    queued_ns = _integer(payload.get("queued_at_ns"), "queued_at_ns")
    root = payload.get("root")
    if not isinstance(root, Mapping):
        raise ValueError("Step 6d worker root is missing")
    expected_row = performance_schedule_row(
        _integer(root.get("hand_index"), "hand index")
    )
    observations = _validate_root_artifact(root, expected_row=expected_row)
    seeds = expected_row["seeds"]
    rayon_threads = _integer(payload.get("rayon_threads"), "Rayon threads", minimum=1)
    if os.environ.get("RAYON_NUM_THREADS") != str(rayon_threads):
        raise RuntimeError("Step 6d worker RAYON_NUM_THREADS was not explicit")
    before = process_memory_snapshot()
    reference_solver = _solver(
        library_path=Path(str(payload["reference_library"])),
        library_sha256=str(payload["reference_sha256"]),
        seeds=seeds,
    )
    candidate_solver = _solver(
        library_path=Path(str(payload["candidate_library"])),
        library_sha256=str(payload["candidate_sha256"]),
        seeds=seeds,
    )
    after_solver_load = process_memory_snapshot()
    if reference_solver.engine_version != candidate_solver.engine_version:
        raise ValueError("Step 6d reference/candidate engine versions differ")
    rows: list[dict[str, Any]] = []
    for offset, observation in enumerate(observations):
        root_started = time.perf_counter()
        root_index = expected_row["root_indices"][offset]
        # Each paired hand contains one solve in each source order.  This prevents
        # first-call warming from being assigned to only one binary.
        order = (
            ("reference", "candidate")
            if root_index % 2 == 0
            else ("candidate", "reference")
        )
        solvers = {"reference": reference_solver, "candidate": candidate_solver}
        results: dict[str, dict[str, Any]] = {}
        for source in order:
            results[source] = _solve_source(
                source=source,
                solver=solvers[source],
                observation=observation,
                seeds=seeds,
            )
        parity = compare_portable_decisions(
            results["reference"]["decision"], results["candidate"]["decision"]
        )
        rows.append(
            {
                "root_index": root_index,
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "solve_order": list(order),
                "geometry": _geometry(observation, results["candidate"]["decision"]),
                "reference": results["reference"],
                "candidate": results["candidate"],
                "parity": parity,
                "wall_seconds": time.perf_counter() - root_started,
            }
        )
    memory = {
        "before_solver_load": before,
        "after_solver_load": after_solver_load,
        "after_hand": process_memory_snapshot(),
    }
    peaks = [
        snapshot.get("peak_rss_bytes")
        for snapshot in memory.values()
        if isinstance(snapshot, Mapping)
        and isinstance(snapshot.get("peak_rss_bytes"), int)
        and not isinstance(snapshot.get("peak_rss_bytes"), bool)
    ]
    memory["peak_rss_bytes"] = max(peaks) if peaks else None
    report = {
        "schema": STEP6D_PERFORMANCE_HAND_SCHEMA,
        "contract_canonical_sha256": EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
        "schedule": STEP6D_PERFORMANCE_SCHEDULE,
        "hand_index": expected_row["hand_index"],
        "profile": expected_row["profile"],
        "seeds": seeds,
        "budget": dict(PERFORMANCE_BUDGET),
        "run_contract_digest": str(payload["run_contract_digest"]),
        "root_artifact_sha256": _digest(root),
        "allocation": {
            "workers": _integer(payload.get("workers"), "workers", minimum=1),
            "rayon_threads_per_worker": rayon_threads,
        },
        "reference_library_sha256": reference_solver.library_sha256,
        "candidate_library_sha256": candidate_solver.library_sha256,
        "queue_seconds": max(0.0, (started_ns - queued_ns) / 1_000_000_000.0),
        "worker_wall_seconds": (time.perf_counter_ns() - started_ns) / 1_000_000_000.0,
        "process_id": os.getpid(),
        "memory": memory,
        "rows": rows,
        "portable_parity_exact": all(
            row["parity"]["portable_payload_exact"] for row in rows
        ),
        "teacher_value_status": "diagnostic_not_match_EV",
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "cloud_started": False,
    }
    return report


def _validate_source_result(
    value: Mapping[str, Any],
    *,
    source: str,
    observation: ActorObservation,
    seeds: Mapping[str, int],
    library_sha256: str,
) -> None:
    if (
        set(value) != _SOURCE_RESULT_KEYS
        or value.get("source") != source
        or value.get("native_library_sha256") != library_sha256
        or not isinstance(value.get("decision"), Mapping)
        or not isinstance(value.get("rss_after"), Mapping)
    ):
        raise ValueError("Step 6d source result schema changed")
    for field in (
        "solve_wall_seconds",
        "native_seconds",
        "validation_seconds",
        "runtime_total_seconds",
    ):
        _finite(value.get(field), field, minimum=0.0)
    _memory_peak(value["rss_after"], f"{source} rss_after")
    _validate_decision(
        value["decision"],
        observation=observation,
        seeds=seeds,
        library_sha256=library_sha256,
    )


def _validate_hand_artifact(
    value: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
    reference_sha256: str,
    candidate_sha256: str,
    workers: int,
    rayon_threads: int,
    run_contract_digest: str,
) -> dict[str, Any]:
    expected_row = performance_schedule_row(int(root["hand_index"]))
    observations = _validate_root_artifact(root, expected_row=expected_row)
    if (
        set(value) != _HAND_KEYS
        or value.get("schema") != STEP6D_PERFORMANCE_HAND_SCHEMA
        or value.get("contract_canonical_sha256")
        != EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
        or value.get("schedule") != STEP6D_PERFORMANCE_SCHEDULE
        or value.get("hand_index") != expected_row["hand_index"]
        or value.get("profile") != expected_row["profile"]
        or value.get("seeds") != expected_row["seeds"]
        or value.get("budget") != PERFORMANCE_BUDGET
        or value.get("run_contract_digest") != run_contract_digest
        or value.get("root_artifact_sha256") != _digest(root)
        or value.get("allocation")
        != {"workers": workers, "rayon_threads_per_worker": rayon_threads}
        or value.get("reference_library_sha256") != reference_sha256
        or value.get("candidate_library_sha256") != candidate_sha256
        or not isinstance(value.get("rows"), list)
        or len(value["rows"]) != 2
        or not isinstance(value.get("memory"), Mapping)
        or value.get("teacher_value_status") != "diagnostic_not_match_EV"
        or any(
            value.get(field) is not False
            for field in (
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
                "cloud_started",
            )
        )
    ):
        raise ValueError("Step 6d performance hand provenance changed")
    _integer(value.get("process_id"), "process id", minimum=1)
    _finite(value.get("queue_seconds"), "queue seconds", minimum=0.0)
    _finite(value.get("worker_wall_seconds"), "worker wall seconds", minimum=0.0)
    _validate_memory(value["memory"])
    exact: list[bool] = []
    for offset, raw in enumerate(value["rows"]):
        if not isinstance(raw, Mapping) or set(raw) != _ROOT_RESULT_KEYS:
            raise ValueError("Step 6d root-result schema changed")
        observation = observations[offset]
        expected_order = (
            ["reference", "candidate"]
            if expected_row["root_indices"][offset] % 2 == 0
            else ["candidate", "reference"]
        )
        if (
            raw.get("root_index") != expected_row["root_indices"][offset]
            or raw.get("seat") != observation.seat
            or raw.get("observation_fingerprint") != observation.fingerprint()
            or raw.get("solve_order") != expected_order
            or not isinstance(raw.get("reference"), Mapping)
            or not isinstance(raw.get("candidate"), Mapping)
            or not isinstance(raw.get("parity"), Mapping)
        ):
            raise ValueError("Step 6d root-result provenance changed")
        _finite(raw.get("wall_seconds"), "root wall seconds", minimum=0.0)
        _validate_source_result(
            raw["reference"],
            source="reference",
            observation=observation,
            seeds=expected_row["seeds"],
            library_sha256=reference_sha256,
        )
        _validate_source_result(
            raw["candidate"],
            source="candidate",
            observation=observation,
            seeds=expected_row["seeds"],
            library_sha256=candidate_sha256,
        )
        expected_parity = compare_portable_decisions(
            raw["reference"]["decision"], raw["candidate"]["decision"]
        )
        if set(raw["parity"]) != _PARITY_KEYS or raw["parity"] != expected_parity:
            raise ValueError("Step 6d portable parity evidence changed")
        expected_geometry = _geometry(observation, raw["candidate"]["decision"])
        if raw.get("geometry") != expected_geometry:
            raise ValueError("Step 6d root geometry changed")
        exact.append(bool(expected_parity["portable_payload_exact"]))
    if value.get("portable_parity_exact") != all(exact):
        raise ValueError("Step 6d hand parity status changed")
    _reject_hidden(value, "hand")
    return dict(value)


def _percentiles(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Step 6d performance metric is empty")

    def nearest(fraction: float) -> float:
        index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "mean_seconds": statistics.fmean(ordered),
        "p50_seconds": nearest(0.50),
        "p95_seconds": nearest(0.95),
        "p99_seconds": nearest(0.99),
        "max_seconds": ordered[-1],
    }


def _performance_summary(
    reports: Sequence[Mapping[str, Any]], *, source: str
) -> dict[str, Any]:
    return {
        seat: {
            metric: _percentiles(
                [
                    row[source][metric]
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


def _build_summary(
    *,
    reports: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    output_dir: Path,
    reference_sha256: str,
    candidate_sha256: str,
    workers: int,
    rayon_threads: int,
    resumed_hand_count: int,
) -> dict[str, Any]:
    full_run = tuple(indices) == PERFORMANCE_HAND_INDICES
    roots = [row for report in reports for row in report["rows"]]
    reference_performance = _performance_summary(reports, source="reference")
    candidate_performance = _performance_summary(reports, source="candidate")
    queue = _percentiles([report["queue_seconds"] for report in reports])
    worker_wall = _percentiles([report["worker_wall_seconds"] for report in reports])
    peaks = [
        report["memory"].get("peak_rss_bytes")
        for report in reports
        if isinstance(report["memory"].get("peak_rss_bytes"), int)
        and not isinstance(report["memory"].get("peak_rss_bytes"), bool)
    ]
    peak_rss = max(peaks) if peaks else None
    profile_counts = {
        profile: sum(report["profile"] == profile for report in reports)
        for profile in M31_T3_BEHAVIOR_PROFILES
    }
    geometry_counts: dict[str, int] = {}
    for row in roots:
        key = str(row["geometry"]["legal_action_count"])
        geometry_counts[key] = geometry_counts.get(key, 0) + 1
    parity_count = sum(row["parity"]["portable_payload_exact"] for row in roots)
    parity_fraction = parity_count / len(roots)
    integrity_gates = {
        "exact_hand_indices": [report["hand_index"] for report in reports]
        == list(indices),
        "exact_root_count": len(roots) == len(indices) * 2,
        "exact_seat_balance": (
            sum(row["seat"] == "first" for row in roots)
            == sum(row["seat"] == "second" for row in roots)
            == len(indices)
        ),
        "profile_cycle_exact": all(
            report["profile"] == behavior_profile_for_index(report["hand_index"])
            for report in reports
        ),
        "full_profile_quota_if_applicable": (
            not full_run
            or profile_counts == {profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES}
        ),
        "portable_semantic_parity_fraction_1": parity_fraction == 1.0,
        "no_training_quality_promotion_activation_or_cloud": True,
    }
    candidate_first = candidate_performance["first"]["solve_wall_seconds"]
    candidate_second = candidate_performance["second"]["solve_wall_seconds"]
    candidate_freeze_gates = {
        "portable_semantic_parity_fraction": parity_fraction == 1.0,
        "first_p95_seconds_max": candidate_first["p95_seconds"]
        <= PERFORMANCE_GATE_LIMITS["first_p95_seconds_max"],
        "first_p99_and_max_seconds_max": (
            candidate_first["p99_seconds"]
            <= PERFORMANCE_GATE_LIMITS["first_p99_and_max_seconds_max"]
            and candidate_first["max_seconds"]
            <= PERFORMANCE_GATE_LIMITS["first_p99_and_max_seconds_max"]
        ),
        "second_p95_seconds_max": candidate_second["p95_seconds"]
        <= PERFORMANCE_GATE_LIMITS["second_p95_seconds_max"],
        "peak_rss_bytes_max": (
            peak_rss is not None
            and peak_rss <= PERFORMANCE_GATE_LIMITS["peak_rss_bytes_max"]
        ),
    }
    all_gates = all(integrity_gates.values()) and (
        all(candidate_freeze_gates.values()) if full_run else True
    )
    hand_dir = output_dir / "hands"
    run_contract = _run_contract(
        indices=indices,
        reference_sha256=reference_sha256,
        candidate_sha256=candidate_sha256,
        workers=workers,
        rayon_threads=rayon_threads,
    )
    return {
        "schema": STEP6D_PERFORMANCE_SUMMARY_SCHEMA,
        "status": "pass" if all_gates else "no_go",
        "contract_canonical_sha256": EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
        "schedule": STEP6D_PERFORMANCE_SCHEDULE,
        "scope": "full_performance_development" if full_run else "subset_smoke",
        "run_contract": run_contract,
        "run_contract_digest": _digest(run_contract),
        "subset_smoke": not full_run,
        "hand_indices": list(indices),
        "paired_hand_count": len(indices),
        "root_count": len(roots),
        "profile_hand_counts": profile_counts,
        "budget": dict(PERFORMANCE_BUDGET),
        "allocation": {
            "workers": workers,
            "rayon_threads_per_worker": rayon_threads,
            "allocation_id": f"{workers}x{rayon_threads}",
        },
        "reference_library_sha256": reference_sha256,
        "candidate_library_sha256": candidate_sha256,
        "portable_parity": {
            "matching_roots": parity_count,
            "root_count": len(roots),
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
        "candidate_freeze_gates_applied": full_run,
        "candidate_freeze_gates": candidate_freeze_gates,
        "all_gates_passed": all_gates,
        "candidate_freeze_ready": full_run and all_gates,
        "missing_or_censored_root_indices": [],
        "hand_manifest": [
            {
                "hand_index": report["hand_index"],
                "path": f"hands/hand_{report['hand_index']:03d}.json",
                "sha256": _sha256(hand_dir / f"hand_{report['hand_index']:03d}.json"),
                "bytes": (hand_dir / f"hand_{report['hand_index']:03d}.json")
                .stat()
                .st_size,
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


def _validate_binary(
    path: Path, expected_sha256: str, *, label: str
) -> tuple[Path, str]:
    resolved = Path(path).resolve()
    expected = str(expected_sha256).casefold()
    if not _is_sha256(expected):
        raise ValueError(f"{label} SHA-256 must be explicit")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} native library is missing: {resolved}")
    actual = _sha256(resolved)
    if actual != expected:
        raise ValueError(f"{label} native library SHA-256 mismatch")
    return resolved, actual


def _validate_allocation(workers: int, rayon_threads: int) -> None:
    if (workers, rayon_threads) not in ALLOWED_ALLOCATIONS:
        raise ValueError("Step 6d allocation must be one of 1x16, 2x8, or 4x4")


def _run_contract(
    *,
    indices: Sequence[int],
    reference_sha256: str,
    candidate_sha256: str,
    workers: int,
    rayon_threads: int,
) -> dict[str, Any]:
    return {
        "schema": "hu_m31_t3_step6d_performance_run_contract_v1",
        "step6d_run_id": STEP6D_RUN_ID,
        "contract_canonical_sha256": EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
        "schedule": STEP6D_PERFORMANCE_SCHEDULE,
        "hand_indices": list(indices),
        "budget": dict(PERFORMANCE_BUDGET),
        "allocation": {
            "workers": workers,
            "rayon_threads_per_worker": rayon_threads,
        },
        "reference_library_sha256": reference_sha256,
        "candidate_library_sha256": candidate_sha256,
        "execution_mode": "two_independent_scalar_solves_per_root",
        "source_order": "reference_first_on_first_root_candidate_first_on_second_root",
        "training_eligible": False,
    }


def run_performance_development(
    *,
    repository_root: Path,
    output_dir: Path,
    reference_library: Path,
    reference_sha256: str,
    candidate_library: Path,
    candidate_sha256: str,
    workers: int,
    rayon_threads: int,
    indices: Iterable[int] | None = None,
    stop_after_hands: int | None = None,
) -> dict[str, Any]:
    """Run or resume one explicitly allocated performance-development comparison."""

    repository_root = Path(repository_root).resolve()
    output_dir = Path(output_dir).resolve()
    validate_performance_contract(repository_root / _CONTRACT_RELATIVE_PATH)
    selected_indices = normalize_hand_indices(indices)
    _validate_allocation(workers, rayon_threads)
    reference_path, reference_digest = _validate_binary(
        reference_library, reference_sha256, label="reference"
    )
    if reference_digest != REFERENCE_NATIVE_LIBRARY_SHA256:
        raise ValueError("reference native library is not the frozen accepted binary")
    candidate_path, candidate_digest = _validate_binary(
        candidate_library, candidate_sha256, label="candidate"
    )
    if reference_path == candidate_path:
        raise ValueError("reference and candidate must be separate library paths")
    if stop_after_hands is not None:
        _integer(stop_after_hands, "stop_after_hands", minimum=1)
    os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)
    # Scalar solve is mandatory.  Pinning the batch setting prevents a future
    # accidental solve_many conversion from silently multiplying thread pools.
    os.environ["OFC_HU_M3_BATCH_THREADS"] = "1"
    run_contract = _run_contract(
        indices=selected_indices,
        reference_sha256=reference_digest,
        candidate_sha256=candidate_digest,
        workers=workers,
        rayon_threads=rayon_threads,
    )
    run_contract_digest = _digest(run_contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = _materialize_roots(
        repository_root=repository_root,
        output_dir=output_dir,
        indices=selected_indices,
    )
    root_by_index = {int(root["hand_index"]): root for root in roots}
    hand_dir = output_dir / "hands"
    hand_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for index in selected_indices:
        path = hand_dir / f"hand_{index:03d}.json"
        if path.exists():
            existing[index] = _validate_hand_artifact(
                _read_json(path),
                root=root_by_index[index],
                reference_sha256=reference_digest,
                candidate_sha256=candidate_digest,
                workers=workers,
                rayon_threads=rayon_threads,
                run_contract_digest=run_contract_digest,
            )
        else:
            missing.append(index)
    resumed_hand_count = len(existing)
    selected_missing = missing[:stop_after_hands] if stop_after_hands else missing

    def payload(index: int) -> dict[str, Any]:
        return {
            "root": root_by_index[index],
            "reference_library": str(reference_path),
            "reference_sha256": reference_digest,
            "candidate_library": str(candidate_path),
            "candidate_sha256": candidate_digest,
            "workers": workers,
            "rayon_threads": rayon_threads,
            "run_contract_digest": run_contract_digest,
            "queued_at_ns": time.perf_counter_ns(),
        }

    if workers == 1:
        for index in selected_missing:
            report = _run_hand_worker(payload(index))
            validated = _validate_hand_artifact(
                report,
                root=root_by_index[index],
                reference_sha256=reference_digest,
                candidate_sha256=candidate_digest,
                workers=workers,
                rayon_threads=rayon_threads,
                run_contract_digest=run_contract_digest,
            )
            _write_once(hand_dir / f"hand_{index:03d}.json", validated)
            existing[index] = validated
    elif selected_missing:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {
                pool.submit(_run_hand_worker, payload(index)): index
                for index in selected_missing
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                report = future.result()
                validated = _validate_hand_artifact(
                    report,
                    root=root_by_index[index],
                    reference_sha256=reference_digest,
                    candidate_sha256=candidate_digest,
                    workers=workers,
                    rayon_threads=rayon_threads,
                    run_contract_digest=run_contract_digest,
                )
                _write_once(hand_dir / f"hand_{index:03d}.json", validated)
                existing[index] = validated

    remaining = [index for index in selected_indices if index not in existing]
    if remaining:
        missing_roots = [
            root_index
            for index in remaining
            for root_index in (index * 2, index * 2 + 1)
        ]
        return {
            "schema": STEP6D_PERFORMANCE_SUMMARY_SCHEMA,
            "status": "interrupted_for_resume",
            "subset_smoke": selected_indices != PERFORMANCE_HAND_INDICES,
            "hand_indices": list(selected_indices),
            "completed_hand_count": len(existing),
            "pending_hand_count": len(remaining),
            "missing_or_censored_root_indices": missing_roots,
            "resumed_hand_count": resumed_hand_count,
            "training_eligible": False,
            "current_profile_changed": False,
            "cloud_started": False,
        }
    reports = [existing[index] for index in selected_indices]
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        old = _read_json(summary_path)
        stored_resumed = _integer(old.get("resumed_hand_count"), "resumed hand count")
        expected = _build_summary(
            reports=reports,
            indices=selected_indices,
            output_dir=output_dir,
            reference_sha256=reference_digest,
            candidate_sha256=candidate_digest,
            workers=workers,
            rayon_threads=rayon_threads,
            resumed_hand_count=stored_resumed,
        )
        if old != expected:
            raise ValueError("existing Step 6d performance summary changed")
        return old
    summary = _build_summary(
        reports=reports,
        indices=selected_indices,
        output_dir=output_dir,
        reference_sha256=reference_digest,
        candidate_sha256=candidate_digest,
        workers=workers,
        rayon_threads=rayon_threads,
        resumed_hand_count=resumed_hand_count,
    )
    _write_once(summary_path, summary)
    return summary


def _parse_indices(
    csv: str | None, repeated: Sequence[int] | None
) -> tuple[int, ...] | None:
    if csv is not None and repeated:
        raise ValueError("use either --indices or --hand-index, not both")
    if csv is not None:
        try:
            values = [int(token.strip()) for token in csv.split(",") if token.strip()]
        except ValueError as exc:
            raise ValueError("--indices must be comma-separated integers") from exc
        return normalize_hand_indices(values)
    return normalize_hand_indices(repeated) if repeated else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-library", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--rayon-threads", type=int, required=True)
    parser.add_argument("--indices", help="comma-separated 0..99 subset smoke")
    parser.add_argument("--hand-index", type=int, action="append")
    parser.add_argument("--stop-after-hands", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_performance_development(
        repository_root=args.repository_root,
        output_dir=args.output_dir,
        reference_library=args.reference_library,
        reference_sha256=args.reference_sha256,
        candidate_library=args.candidate_library,
        candidate_sha256=args.candidate_sha256,
        workers=args.workers,
        rayon_threads=args.rayon_threads,
        indices=_parse_indices(args.indices, args.hand_index),
        stop_after_hands=args.stop_after_hands,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALLOWED_ALLOCATIONS",
    "PERFORMANCE_BUDGET",
    "PERFORMANCE_GATE_LIMITS",
    "PERFORMANCE_HAND_INDICES",
    "REFERENCE_NATIVE_LIBRARY_SHA256",
    "STEP6D_PERFORMANCE_HAND_SCHEMA",
    "STEP6D_PERFORMANCE_ROOT_SCHEMA",
    "STEP6D_PERFORMANCE_SCHEDULE",
    "STEP6D_PERFORMANCE_SUMMARY_SCHEMA",
    "STEP6D_PORTABLE_PARITY_SCHEMA",
    "compare_portable_decisions",
    "main",
    "normalize_hand_indices",
    "performance_schedule_row",
    "performance_seed_values",
    "portable_parity_payload",
    "run_performance_development",
    "validate_performance_contract",
]
