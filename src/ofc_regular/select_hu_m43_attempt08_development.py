"""One-shot, fail-closed Attempt08 development200 Go/No-Go selector.

There is one frozen search architecture and no arm or threshold selection.
The selector validates exactly 200 deterministic development rows, computes
quality metrics only from the locked final action's disjoint A256 raw paired
deltas, and applies the frozen gates once.  A Go authorizes only a separate
immutable search freeze; it does not authorize audit50, fit, thresholding,
runtime activation, or a ``current`` profile change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .action_key import ActionKey, resolve_action_key
from .action_space import generate_turn_actions
from .hu_infoset import ActorObservation
from .hu_m43_attempt08_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT08_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT08_PLAN_SHA256,
    M43_ATTEMPT08_PROFILES,
    enumerate_attempt08_seed_schedules,
    load_and_validate_attempt08_plan,
    validate_attempt08_plan,
)
from .hu_m43_attempt08_teacher import (
    Attempt08TeacherConfig,
    validate_attempt08_teacher_output,
)
from .hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_GCP_IMAGE_ID,
    ATTEMPT08_GCP_IMAGE_NAME,
)
from .run_hu_m43_attempt08_development import (
    ATTEMPT08_DEVELOPMENT_ROOTS,
    ATTEMPT08_SHARD_ROW_SCHEMA,
    _canonical_sha256 as _runner_canonical_sha256,
    _fixed_contract as _runner_fixed_contract,
    _root_input_sha256 as _runner_root_input_sha256,
    _validate_provenance as _validate_runner_provenance,
    load_attempt08_development_open_bindings,
)


ATTEMPT08_DEVELOPMENT_DECISION_SCHEMA = (
    "hu_m43_attempt08_development_go_no_go_v1"
)
_ROOT_COUNT = 200
_SEED_DOMAINS = (
    "hand",
    "rerank",
    "veto",
    "stress",
    "assessment",
    "child",
)
_ROW_KEYS = frozenset(
    {
        "schema",
        "root_index",
        "hand_seed",
        "root_profile",
        "policy_observation",
        "baseline_action_key",
        "provenance",
        "teacher",
    }
)
_HEX = frozenset("0123456789abcdef")
_EXPECTED_GATES = {
    "fires_total_min": 40,
    "fires_each_profile_min": 3,
    "mean_delta_per_state_strictly_greater_than": 0.0,
    "mean_delta_per_fire_strictly_greater_than": 0.0,
    "false_positive_rate_per_fire_max": 0.4,
    "override_loss_p95_max": 25.0,
    "override_loss_p99_max": 40.0,
    "override_loss_max": 50.0,
    "action_mapping_violation_count_max": 0,
    "rng_domain_violation_count_max": 0,
    "hidden_information_violation_count_max": 0,
    "risk_reserve_contract_violation_count_max": 0,
    "locked_action_change_violation_count_max": 0,
    "nonfire_exact_baseline_action_fallback_required": True,
    "nonfire_complete_trajectory_acceptance_deferred": True,
    "teacher_values_reported_as_realized_match_ev": False,
}
_SELECTOR_WRITE_LIFECYCLE_TOKEN = object()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_jsonl_bytes(raw: bytes) -> list[Mapping[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Attempt08 development input must be UTF-8") from exc
    if not text.endswith("\n") or "\r" in text:
        raise ValueError("Attempt08 development input must use canonical LF JSONL")
    lines = text.splitlines()
    if len(lines) != _ROOT_COUNT or any(not line for line in lines):
        raise ValueError("Attempt08 development input requires exactly 200 rows")
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            row = _mapping(json.loads(line), f"row[{index}]")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Attempt08 row {index} is invalid JSON") from exc
        rows.append(row)
    if raw != b"".join(_canonical_json_bytes(row) for row in rows):
        raise ValueError("Attempt08 development input is not canonical JSONL")
    return rows


def _reject_hidden_fields(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "opponent_private_discards":
                raise ValueError(
                    f"Attempt08 hidden-information violation at {path}.{key}"
                )
            _reject_hidden_fields(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_hidden_fields(child, path=f"{path}[{index}]")


def _flatten_digests(value: Any, *, label: str) -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in sorted(value):
            result.extend(_flatten_digests(value[key], label=f"{label}.{key}"))
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        result = []
        for index, child in enumerate(value):
            result.extend(_flatten_digests(child, label=f"{label}[{index}]"))
        return result
    if not _is_sha256(value):
        raise ValueError(f"{label} contains a non-SHA-256 digest")
    return [str(value)]


def _assessment_metrics(
    raw_value: Any, *, root_index: int, fired: bool
) -> dict[str, Any]:
    raw = _sequence(raw_value, f"root[{root_index}].assessment A256")
    if not fired:
        if raw:
            raise ValueError(
                f"Attempt08 nonfire root {root_index} opened the A256 namespace"
            )
        return {
            "raw": [],
            "mean": 0.0,
            "loss": {"p95": 0.0, "p99": 0.0, "max": 0.0},
        }
    if len(raw) != 256:
        raise ValueError(f"Attempt08 root {root_index} A256 count changed")
    values = np.asarray(
        [
            _finite(value, f"root[{root_index}].assessment[{index}]")
            for index, value in enumerate(raw)
        ],
        dtype=np.float64,
    )
    mean = float(np.mean(values))
    p05 = float(np.quantile(values, 0.05, method="linear"))
    p01 = float(np.quantile(values, 0.01, method="linear"))
    minimum = float(np.min(values))
    return {
        "raw": [float(value) for value in values],
        "mean": mean,
        "loss": {
            "p95": max(0.0, -p05),
            "p99": max(0.0, -p01),
            "max": max(0.0, -minimum),
        },
    }


def _teacher_config(
    *,
    seeds: Mapping[str, int],
    run_id: str,
    root_index: int,
    observation: ActorObservation,
    batch_child_selectors: bool,
) -> Attempt08TeacherConfig:
    return Attempt08TeacherConfig(
        frozen_model_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256,
        hand_seed=int(seeds["hand"]),
        rerank_seed=int(seeds["rerank"]),
        veto_seed=int(seeds["veto"]),
        stress_seed=int(seeds["stress"]),
        assessment_seed=int(seeds["assessment"]),
        child_policy_seed=int(seeds["child"]),
        run_id=(
            f"{run_id}:root={root_index}:seed={seeds['hand']}:"
            f"obs={observation.fingerprint()}"
        ),
        batch_child_selectors=batch_child_selectors,
    )


def _validate_row(
    row: Mapping[str, Any],
    *,
    root_index: int,
    expected_seeds: Mapping[str, int],
    authorization_bindings: Mapping[str, str],
    expected_run_id: str,
) -> dict[str, Any]:
    if set(row) != _ROW_KEYS:
        raise ValueError(f"Attempt08 row fields changed at root {root_index}")
    if row.get("schema") != ATTEMPT08_SHARD_ROW_SCHEMA:
        raise ValueError(f"Attempt08 row schema changed at root {root_index}")
    if _strict_int(row.get("root_index"), "root_index") != root_index:
        raise ValueError(f"Attempt08 root index changed at root {root_index}")
    profile = str(row.get("root_profile"))
    expected_profile = M43_ATTEMPT08_PROFILES[
        root_index % len(M43_ATTEMPT08_PROFILES)
    ]
    if profile != expected_profile:
        raise ValueError(f"Attempt08 profile assignment changed at root {root_index}")
    if _strict_int(row.get("hand_seed"), "hand_seed") != expected_seeds["hand"]:
        raise ValueError(f"Attempt08 hand seed changed at root {root_index}")
    _reject_hidden_fields(row, path=f"row[{root_index}]")

    observation_payload = _mapping(
        row.get("policy_observation"), f"root[{root_index}].policy_observation"
    )
    observation = ActorObservation.from_dict(observation_payload)
    if (
        observation.to_dict() != observation_payload
        or observation.street != "T1"
        or observation.seat != "second"
        or observation.to_act_order != "second"
    ):
        raise ValueError(f"Attempt08 observation identity changed at root {root_index}")
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    baseline_token = str(row.get("baseline_action_key"))
    baseline_key = ActionKey.from_token(baseline_token)
    resolve_action_key(actions, baseline_key)

    provenance = _mapping(row.get("provenance"), f"root[{root_index}].provenance")
    if provenance.get("seeds") != dict(expected_seeds):
        raise ValueError(f"Attempt08 provenance seeds changed at root {root_index}")
    run_id = provenance.get("run_id")
    if not isinstance(run_id, str) or run_id != expected_run_id:
        raise ValueError(f"Attempt08 provenance run_id changed at root {root_index}")
    fixed_contract = _runner_fixed_contract(
        root_index=root_index,
        root_profile=profile,
        seeds=expected_seeds,
        run_id=run_id,
        plan_sha256=M43_ATTEMPT08_PLAN_SHA256,
        ai_profiles_sha256=AI_PROFILES_SHA256,
        model_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256,
        authorization_bindings=authorization_bindings,
        batch_child_selectors=True,
        native_batch_threads=4,
    )
    config_sha256 = _runner_canonical_sha256(fixed_contract)
    root_input_sha256 = _runner_root_input_sha256(
        root_index=root_index,
        hand_seed=expected_seeds["hand"],
        root_profile=profile,
        observation=observation,
        baseline_action_key=baseline_token,
    )
    _validate_runner_provenance(
        provenance,
        fixed_contract=fixed_contract,
        config_sha256=config_sha256,
        root_input_sha256=root_input_sha256,
    )

    config = _teacher_config(
        seeds=expected_seeds,
        run_id=run_id,
        root_index=root_index,
        observation=observation,
        batch_child_selectors=True,
    )
    teacher = _mapping(row.get("teacher"), f"root[{root_index}].teacher")
    normalized = _mapping(
        validate_attempt08_teacher_output(
            observation,
            baseline_action_key=baseline_token,
            payload=teacher,
            config=config,
        ),
        f"root[{root_index}].normalized_teacher",
    )
    expected_normalized_keys = {
        "selected_action_key",
        "override_fired",
        "exact_baseline_fallback",
        "assessment_raw_paired_deltas_vs_baseline",
        "rng_key_digests",
        "belief_digests",
    }
    if set(normalized) != expected_normalized_keys:
        raise ValueError(
            f"Attempt08 normalized teacher fields changed at root {root_index}"
        )
    selected_token = str(normalized["selected_action_key"])
    resolve_action_key(actions, ActionKey.from_token(selected_token))
    fired = normalized["override_fired"]
    fallback = normalized["exact_baseline_fallback"]
    if type(fired) is not bool or type(fallback) is not bool:
        raise ValueError(f"Attempt08 decision flags changed at root {root_index}")
    if fired:
        if selected_token == baseline_token or fallback:
            raise ValueError(f"Attempt08 fired action is baseline at root {root_index}")
    elif selected_token != baseline_token or not fallback:
        raise ValueError(
            f"Attempt08 nonfire lacks exact baseline fallback at root {root_index}"
        )
    assessment = _assessment_metrics(
        normalized["assessment_raw_paired_deltas_vs_baseline"],
        root_index=root_index,
        fired=fired,
    )
    rng_digests = _flatten_digests(
        normalized["rng_key_digests"], label=f"root[{root_index}].rng"
    )
    belief_digests = _flatten_digests(
        normalized["belief_digests"], label=f"root[{root_index}].belief"
    )
    if not rng_digests or len(rng_digests) != len(set(rng_digests)):
        raise ValueError(f"Attempt08 RNG digests overlap at root {root_index}")
    if not belief_digests or len(belief_digests) != len(set(belief_digests)):
        raise ValueError(f"Attempt08 belief digests overlap at root {root_index}")
    return {
        "root_index": root_index,
        "profile": profile,
        "hand_seed": expected_seeds["hand"],
        "observation_fingerprint": observation.fingerprint(),
        "baseline_action_key": baseline_token,
        "selected_action_key": selected_token,
        "config_sha256": config_sha256,
        "root_input_sha256": root_input_sha256,
        "fired": fired,
        "assessment_mean": assessment["mean"],
        "assessment_loss": assessment["loss"],
        "rng_digests": rng_digests,
        "belief_digests": belief_digests,
    }


def _profile_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    states = len(rows)
    fired = [row for row in rows if row["fired"]]
    fired_means = [float(row["assessment_mean"]) for row in fired]
    false_positive_fires = sum(value <= 0.0 for value in fired_means)
    return {
        "states": states,
        "fires": len(fired),
        "mean_delta_per_state": (
            sum(float(row["assessment_mean"]) for row in rows) / states
            if states
            else None
        ),
        "mean_delta_per_fire": (
            sum(fired_means) / len(fired_means) if fired_means else None
        ),
        "false_positive_fires": false_positive_fires,
        "false_positive_rate_per_fire": (
            false_positive_fires / len(fired_means) if fired_means else None
        ),
        "maximum_per_fired_root_loss": {
            name: max(float(row["assessment_loss"][name]) for row in fired)
            if fired
            else None
            for name in ("p95", "p99", "max")
        },
    }


def _gate(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def _validate_frozen_gates(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    gates = _mapping(plan.get("development_go_no_go"), "development gates")
    for key, expected in _EXPECTED_GATES.items():
        if gates.get(key) != expected or type(gates.get(key)) is not type(expected):
            raise ValueError(f"Attempt08 frozen development gate changed: {key}")
    if (
        gates.get("all_gates_required") is not True
        or gates.get("assessment_source")
        != "disjoint_A256_locked_final_nonbaseline_output_vs_explicit_baseline"
        or gates.get("quantile_method") != "numpy_linear"
    ):
        raise ValueError("Attempt08 frozen development gate semantics changed")
    return gates


def aggregate_attempt08_development_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_input_sha256: str,
    source_plan_sha256: str,
    authorization_bindings: Mapping[str, str],
    run_name: str,
) -> dict[str, Any]:
    """Validate exactly 200 rows and evaluate the frozen gates once."""

    validate_attempt08_plan(plan)
    gates_plan = _validate_frozen_gates(plan)
    if source_plan_sha256 != M43_ATTEMPT08_PLAN_SHA256:
        raise ValueError("Attempt08 source plan SHA-256 changed")
    if not _is_sha256(source_input_sha256):
        raise ValueError("Attempt08 source input SHA-256 is invalid")
    if not isinstance(run_name, str) or not run_name:
        raise ValueError("Attempt08 run_name is invalid")
    expected_authorization_keys = {
        "development_open_authorization_sha256",
        "preflight_plan_sha256",
        "preflight_result_sha256",
        "preflight_proof_evidence_sha256",
        "preflight_proof_gates_sha256",
        "preflight_operational_gates_sha256",
        "preflight_execution_evidence_sha256",
        "runtime_semantic_anchor_sha256",
        "runtime_source_closure_sha256",
        "runtime_fingerprint_sha256",
        "runtime_requirements_sha256",
        "gcp_image_name",
        "gcp_image_id",
    }
    sha_binding_keys = expected_authorization_keys - {"gcp_image_name", "gcp_image_id"}
    if (
        set(authorization_bindings) != expected_authorization_keys
        or not all(_is_sha256(authorization_bindings[key]) for key in sha_binding_keys)
        or authorization_bindings.get("gcp_image_name") != ATTEMPT08_GCP_IMAGE_NAME
        or authorization_bindings.get("gcp_image_id") != ATTEMPT08_GCP_IMAGE_ID
    ):
        raise ValueError("Attempt08 authorization bindings changed")
    if len(rows) != _ROOT_COUNT:
        raise ValueError("Attempt08 development requires exactly 200 rows")
    by_index: dict[int, Mapping[str, Any]] = {}
    for raw_row in rows:
        row = _mapping(raw_row, "Attempt08 row")
        root_index = _strict_int(row.get("root_index"), "root_index")
        if root_index in by_index:
            raise ValueError(f"Attempt08 duplicate root_index: {root_index}")
        by_index[root_index] = row
    if set(by_index) != set(range(_ROOT_COUNT)):
        raise ValueError("Attempt08 root_index set must be exactly 0..199")

    schedules = enumerate_attempt08_seed_schedules(plan, population="development")
    if set(schedules) != set(_SEED_DOMAINS):
        raise ValueError("Attempt08 development seed domains changed")
    validated: list[dict[str, Any]] = []
    for root_index in range(_ROOT_COUNT):
        expected_seeds = {
            domain: int(schedules[domain][root_index]) for domain in _SEED_DOMAINS
        }
        if len(set(expected_seeds.values())) != len(expected_seeds):
            raise ValueError(f"Attempt08 seed overlap at root {root_index}")
        validated.append(
            _validate_row(
                by_index[root_index],
                root_index=root_index,
                expected_seeds=expected_seeds,
                authorization_bindings=authorization_bindings,
                expected_run_id=f"{run_name}:shard={root_index}",
            )
        )

    all_schedule_seeds = [
        int(schedules[domain][root_index])
        for domain in _SEED_DOMAINS
        for root_index in range(_ROOT_COUNT)
    ]
    if len(all_schedule_seeds) != len(set(all_schedule_seeds)):
        raise ValueError("Attempt08 development namespace schedules overlap")
    profiles = Counter(str(row["profile"]) for row in validated)
    expected_profiles = Counter({profile: 40 for profile in M43_ATTEMPT08_PROFILES})
    if profiles != expected_profiles:
        raise ValueError("Attempt08 development population is not 40/profile")
    fingerprints = [str(row["observation_fingerprint"]) for row in validated]
    if len(set(fingerprints)) != _ROOT_COUNT:
        raise ValueError("Attempt08 development observation fingerprints repeat")
    hand_seeds = [int(row["hand_seed"]) for row in validated]
    if len(set(hand_seeds)) != _ROOT_COUNT:
        raise ValueError("Attempt08 development hand seeds repeat")
    all_rng = [digest for row in validated for digest in row["rng_digests"]]
    if len(all_rng) != len(set(all_rng)):
        raise ValueError("Attempt08 particle RNG keys are not globally unique")
    all_beliefs = [
        digest for row in validated for digest in row["belief_digests"]
    ]
    if len(all_beliefs) != len(set(all_beliefs)):
        raise ValueError("Attempt08 belief digests are not globally unique")

    overall = _profile_metrics(validated)
    by_profile = {
        profile: _profile_metrics(
            [row for row in validated if row["profile"] == profile]
        )
        for profile in M43_ATTEMPT08_PROFILES
    }
    profile_fires = {
        profile: int(by_profile[profile]["fires"])
        for profile in M43_ATTEMPT08_PROFILES
    }
    tails = _mapping(
        overall["maximum_per_fired_root_loss"], "maximum fired-root losses"
    )
    gates = [
        _gate(
            "fires_total",
            int(overall["fires"]) >= int(gates_plan["fires_total_min"]),
            overall["fires"],
            f">= {gates_plan['fires_total_min']}",
        ),
        _gate(
            "fires_each_profile",
            all(
                count >= int(gates_plan["fires_each_profile_min"])
                for count in profile_fires.values()
            ),
            profile_fires,
            f"each >= {gates_plan['fires_each_profile_min']}",
        ),
        _gate(
            "mean_delta_per_state",
            float(overall["mean_delta_per_state"])
            > float(gates_plan["mean_delta_per_state_strictly_greater_than"]),
            overall["mean_delta_per_state"],
            f"> {gates_plan['mean_delta_per_state_strictly_greater_than']}",
        ),
        _gate(
            "mean_delta_per_fire",
            overall["mean_delta_per_fire"] is not None
            and float(overall["mean_delta_per_fire"])
            > float(gates_plan["mean_delta_per_fire_strictly_greater_than"]),
            overall["mean_delta_per_fire"],
            f"> {gates_plan['mean_delta_per_fire_strictly_greater_than']}",
        ),
        _gate(
            "false_positive_rate_per_fire",
            overall["false_positive_rate_per_fire"] is not None
            and float(overall["false_positive_rate_per_fire"])
            <= float(gates_plan["false_positive_rate_per_fire_max"]),
            overall["false_positive_rate_per_fire"],
            f"<= {gates_plan['false_positive_rate_per_fire_max']}",
        ),
        _gate(
            "maximum_per_fired_root_p95_loss",
            tails["p95"] is not None
            and float(tails["p95"]) <= float(gates_plan["override_loss_p95_max"]),
            tails["p95"],
            f"<= {gates_plan['override_loss_p95_max']}",
        ),
        _gate(
            "maximum_per_fired_root_p99_loss",
            tails["p99"] is not None
            and float(tails["p99"]) <= float(gates_plan["override_loss_p99_max"]),
            tails["p99"],
            f"<= {gates_plan['override_loss_p99_max']}",
        ),
        _gate(
            "maximum_per_fired_root_max_loss",
            tails["max"] is not None
            and float(tails["max"]) <= float(gates_plan["override_loss_max"]),
            tails["max"],
            f"<= {gates_plan['override_loss_max']}",
        ),
        _gate("action_mapping_violation_count", True, 0, "= 0"),
        _gate("rng_domain_violation_count", True, 0, "= 0"),
        _gate("hidden_information_violation_count", True, 0, "= 0"),
        _gate("risk_reserve_contract_violation_count", True, 0, "= 0"),
        _gate("locked_action_change_violation_count", True, 0, "= 0"),
        _gate(
            "nonfire_exact_baseline_action_fallback",
            True,
            True,
            "required for every nonfire",
        ),
    ]
    passed = all(bool(gate["passed"]) for gate in gates)
    root_identities = [
        {
            "root_index": row["root_index"],
            "profile": row["profile"],
            "hand_seed": row["hand_seed"],
            "observation_fingerprint": row["observation_fingerprint"],
            "baseline_action_key": row["baseline_action_key"],
            "config_sha256": row["config_sha256"],
            "root_input_sha256": row["root_input_sha256"],
        }
        for row in validated
    ]
    return {
        "schema": ATTEMPT08_DEVELOPMENT_DECISION_SCHEMA,
        "status": (
            "go_write_separate_search_freeze_only"
            if passed
            else "no_go_close_attempt08_development"
        ),
        "decision": "go" if passed else "no_go",
        "decision_scope": (
            "write_separate_immutable_search_freeze_before_future_audit_authorization"
            if passed
            else "close_attempt08_development_without_audit_fit_threshold_or_runtime"
        ),
        "search_freeze_authorized": passed,
        "selected_arm": None,
        "selected_threshold": None,
        "source": {
            "input_jsonl_sha256": source_input_sha256,
            "run_name": run_name,
            "plan_sha256": source_plan_sha256,
            **dict(authorization_bindings),
            "selector_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "root_identity_sha256": _runner_canonical_sha256(
                {"roots": root_identities}
            ),
        },
        "development_population": {
            "roots": _ROOT_COUNT,
            "profiles": list(M43_ATTEMPT08_PROFILES),
            "profile_counts": dict(profiles),
            "unique_hand_seeds": len(set(hand_seeds)),
            "unique_observation_fingerprints": len(set(fingerprints)),
        },
        "metrics": {
            "overall": overall,
            "by_profile": by_profile,
            "fired_root_diagnostics": [
                {
                    "root_index": row["root_index"],
                    "profile": row["profile"],
                    "selected_action_key": row["selected_action_key"],
                    "a256_paired_delta_mean": row["assessment_mean"],
                    "a256_per_root_loss": row["assessment_loss"],
                    "false_positive": row["assessment_mean"] <= 0.0,
                }
                for row in validated
                if row["fired"]
            ],
        },
        "gates": gates,
        "decision_contract": {
            "single_frozen_search_architecture": True,
            "arm_selection_performed": False,
            "threshold_selection_performed": False,
            "gate_evaluation_count": 1,
            "all_gates_required": True,
        },
        "integrity": {
            "action_mapping_violation_count": 0,
            "rng_domain_violation_count": 0,
            "hidden_information_violation_count": 0,
            "risk_reserve_contract_violation_count": 0,
            "locked_action_change_violation_count": 0,
            "nonfire_exact_baseline_action_fallback_verified": True,
            "nonfire_complete_trajectory_cancellation_verified": False,
            "nonfire_complete_trajectory_acceptance_deferred": True,
        },
        "science_boundary": {
            "assessment_source": gates_plan["assessment_source"],
            "teacher_values_are_realized_match_ev": False,
            "arm_selection_performed": False,
            "threshold_selected": False,
            "fit_performed": False,
            "future_audit_directly_authorized": False,
            "future_audit_opened": False,
            "runtime_trajectory_cancellation_claimed": False,
            "runtime_policy_activated": False,
            "current_profile_mutated": False,
            "full_replacement_enabled": False,
        },
    }


def select_attempt08_development(
    *,
    input_path: str | Path,
    plan_path: str | Path,
    preflight_plan_path: str | Path,
    development_open_authorization_path: str | Path,
    run_name: str,
) -> dict[str, Any]:
    """Read, hash, validate, and decide one canonical development200 input."""

    plan_bytes = Path(plan_path).read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != M43_ATTEMPT08_PLAN_SHA256:
        raise ValueError("Attempt08 plan byte SHA-256 changed")
    plan = load_and_validate_attempt08_plan(plan_path)
    authorization_bindings = load_attempt08_development_open_bindings(
        development_open_authorization_path,
        plan=plan_path,
        preflight_plan=preflight_plan_path,
    )
    input_bytes = Path(input_path).read_bytes()
    rows = _read_jsonl_bytes(input_bytes)
    return aggregate_attempt08_development_rows(
        rows,
        plan=plan,
        source_input_sha256=hashlib.sha256(input_bytes).hexdigest(),
        source_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        authorization_bindings=authorization_bindings,
        run_name=run_name,
    )


def write_attempt08_development_decision(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    _lifecycle_token: object | None = None,
) -> None:
    """Create one immutable decision without replacing any prior output."""

    if _lifecycle_token is not _SELECTOR_WRITE_LIFECYCLE_TOKEN:
        raise RuntimeError(
            "direct Attempt08 decision writes are disabled; use the run-global "
            "Spot selector lifecycle"
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite Attempt08 development decision: {destination}"
        )
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite Attempt08 development decision: {destination}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the frozen Attempt08 development200 gates exactly once"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--preflight-plan", required=True, type=Path)
    parser.add_argument(
        "--development-open-authorization", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _build_parser().parse_args(argv)
    raise RuntimeError(
        "direct Attempt08 selector CLI is disabled; use "
        "ofc_regular.hu_m43_attempt08_spot selector-claim/select-once"
    )


if __name__ == "__main__":
    raise SystemExit(main())
