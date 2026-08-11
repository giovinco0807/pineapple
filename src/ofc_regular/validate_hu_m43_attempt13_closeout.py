"""Validate a terminal Attempt13 closeout without changing runtime state.

The closeout is deliberately a separate receipt.  It never authorizes cloud
work, mutates ``current``, or turns an accepted artifact on.  Development and
Audit50 terminal decisions are re-opened through their immutable hash chains;
population results are recomputed through the existing Attempt13 acceptance
validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m43_attempt13_spot as spot
from . import select_hu_m43_attempt13_audit50 as audit_selector
from . import select_hu_m43_attempt13_development as development_selector
from . import validate_hu_m43_attempt13_acceptance as acceptance
from .freeze_hu_m43_attempt13_development import (
    ATTEMPT13_DEVELOPMENT_GO_FREEZE_SCHEMA,
    ATTEMPT13_DEVELOPMENT_GO_FREEZE_STATUS,
)
from .hu_m43_attempt13_contract import (
    M43_ATTEMPT13_PLAN_SHA256,
    M43_ATTEMPT13_PROFILES,
    load_and_validate_attempt13_plan,
)
from .select_hu_m43_attempt13_audit50 import (
    ATTEMPT13_AUDIT50_DECISION_SCHEMA,
    ATTEMPT13_AUDIT50_RECEIPT_SCHEMA,
)
from .select_hu_m43_attempt13_development import (
    ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA,
    ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA,
)
from .train_hu_m43_attempt13_distilled import (
    ATTEMPT13_DISTILLATION_CONFIG_SCHEMA,
    ATTEMPT13_DISTILLATION_CONFIG_SHA256,
)


ATTEMPT13_CLOSEOUT_SCHEMA = "hu_m43_attempt13_closeout_v1"
ATTEMPT13_CLOSEOUT_VALIDATION_SCHEMA = "hu_m43_attempt13_closeout_validation_v1"
ATTEMPT13_CLOSEOUT_PATH = "configs/hu_joint_policy_m43_attempt13_closeout.json"

_AI_PROFILES_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
_PREFLIGHT_SHA256 = (
    "aa9e49800abdd21cde01e69fc03de1d60fc9562315177b12e0a78ecd3eb720a5"
)
_PREFLIGHT_PATH = (
    "outputs/hu_joint_policy/m43_attempt13_preflight/"
    "regular-hu-m43-attempt13-preflight-20260715-212950/preflight_result.json"
)
_FROZEN_CONTRACTS = {
    "search_plan": {
        "path": "configs/hu_joint_policy_m43_attempt13.json",
        "sha256": M43_ATTEMPT13_PLAN_SHA256,
    },
    "distillation_plan": {
        "path": "configs/hu_joint_policy_m43_attempt13_distillation.json",
        "sha256": ATTEMPT13_DISTILLATION_CONFIG_SHA256,
    },
    "population_plan": {
        "path": "configs/hu_joint_policy_m43_attempt13_population.json",
        "sha256": acceptance.ATTEMPT13_POPULATION_PLAN_SHA256,
    },
    "ai_profiles": {
        "path": "src/ofc_regular/ai_profiles.py",
        "sha256": _AI_PROFILES_SHA256,
    },
    "correctness_preflight": {
        "path": _PREFLIGHT_PATH,
        "sha256": _PREFLIGHT_SHA256,
    },
}

_RUN_ARTIFACT_SUFFIXES = frozenset(
    {
        "manifest",
        "authorization",
        "launch_authorization",
        "schedule",
        "source_archive",
        "merged_teacher",
        "receive_receipt",
        "decision",
        "decision_receipt",
        "selector_source",
    }
)
_DEVELOPMENT_ARTIFACTS = frozenset(
    f"development_{suffix}" for suffix in _RUN_ARTIFACT_SUFFIXES
)
_AUDIT_ARTIFACTS = frozenset(f"audit_{suffix}" for suffix in _RUN_ARTIFACT_SUFFIXES)
_FIT_ARTIFACTS = frozenset(
    {
        "training_model",
        "training_manifest",
        "runtime_source_archive",
        "runtime_source_manifest",
    }
)
_POPULATION_ARTIFACTS = frozenset(
    {
        "runtime_model",
        "runtime_freeze",
        "population_preflight",
        "population_run_manifest",
        "population_source_archive",
        "population_evaluation",
        "population_records",
        "population_merge_manifest",
        "population_receipt",
        "population_acceptance_status",
    }
)
_POPULATION_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "run_manifest_sha256",
        "population_plan_sha256",
        "model_sha256",
        "population_source_archive_sha256",
        "runtime_source_archive_sha256",
        "runtime_source_manifest_sha256",
        "runtime_source_closure_sha256",
        "runtime_semantic_closure_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "runtime_dependency_closure_sha256",
        "seed_registry_sha256",
        "development_decision_sha256",
        "development_selector_receipt_sha256",
        "development_pass_freeze_sha256",
        "audit50_decision_sha256",
        "audit50_selector_receipt_sha256",
        "development200_full_fit_bound",
        "audit50_one_shot_go_bound",
        "model_schema",
        "artifact_schema",
        "feature_schema",
        "head_schema",
        "action_score_mode",
        "profile_id",
        "baseline_profile",
        "evaluation_path",
        "evaluation_sha256",
        "records_path",
        "records_sha256",
        "merge_manifest_sha256",
        "paired_seeds_per_opponent",
        "valid_overrides",
        "invalid_counterfactuals",
        "nonfire_cancellation_mismatches",
        "shards",
        "teacher_calibration_locked_content_received",
        "current_profile_mutated",
        "no_runtime_activation",
        "received_at",
        "acceptance_status_sha256",
        "acceptance_decision",
    }
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "milestone",
        "status_date",
        "status",
        "terminal_stage",
        "frozen_contracts",
        "immutable_artifacts",
        "terminal_gate_snapshot",
        "opened_scope",
        "science_boundary",
        "activation",
        "population_recompute",
    }
)
_GATE_SNAPSHOT_KEYS = frozenset(
    {
        "source_artifact",
        "source_status",
        "decision",
        "passed_gates",
        "total_gates",
        "failed_gates",
        "gate_evaluation_count",
        "recomputed_from_records",
    }
)
_SCOPE_KEYS = frozenset(
    {
        "preflight_passed",
        "development_completed",
        "development_decision",
        "future_audit_started",
        "future_audit_decision",
        "fit_state",
        "audit50_fit_rows",
        "fit_artifact_runtime_eligible",
        "runtime_freeze_written",
        "population_started",
        "population_decision",
    }
)
_SCIENCE_KEYS = frozenset(
    {
        "public_information_set_only",
        "opponent_private_discard_used",
        "opponent_profile_runtime_feature_used",
        "teacher_values_are_realized_match_ev",
        "teacher_ev_or_lcb_runtime_gate",
        "threshold_reselection_performed",
        "alternate_seed_retry_performed",
        "top1_accuracy_is_acceptance_gate",
        "candidate_selection_and_evaluation_rng_independent",
        "realized_population_evidence",
        "nonfire_full_trajectory_cancellation_verified",
        "first_seat_candidate_is_exact_baseline_delegate",
        "unseen_population_robustness_guaranteed",
        "nash_equilibrium_claim",
        "exploitability_bound_claim",
        "mathematically_proven_optimal",
        "evaluation_scope",
    }
)
_ACTIVATION_KEYS = frozenset(
    {
        "profile_id",
        "runtime_artifact_frozen",
        "runtime_binding_verified",
        "promotion_eligible",
        "explicit_opt_in_authorized",
        "automatic_activation_authorized",
        "current_profile_mutated",
        "runtime_policy_activated",
        "full_replacement_enabled",
    }
)
_INTEGRITY_KEYS = frozenset(
    {
        "action_mapping_violation_count",
        "rng_domain_violation_count",
        "hidden_information_violation_count",
        "risk_reserve_contract_violation_count",
        "retained_order_violation_count",
        "phase_filter_violation_count",
        "pooled_phase_violation_count",
        "locked_action_change_violation_count",
        "extreme_tail_statistic_violation_count",
        "nonfire_exact_baseline_action_fallback_verified",
    }
)
_RECEIVE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "mode",
        "roots",
        "root_indices",
        "profiles",
        "manifest_sha256",
        "schedule_sha256",
        "source_sha256",
        "authorization_sha256",
        "merged_sha256",
        "audit_sha256",
        "batch_boundary_validation_count",
        "per_shard_boundary_revalidation_count",
        "selector_executed",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_QUALITY_GATE_NAMES = (
    "fires_total",
    "fires_each_profile",
    "mean_delta_per_state",
    "mean_delta_per_fire",
    "false_positive_rate_per_fire",
    "maximum_per_fired_root_p95_loss",
    "maximum_per_fired_root_p99_loss",
    "maximum_per_fired_root_max_loss",
)
_INTEGRITY_GATE_NAMES = tuple(
    key.removesuffix("_violation_count") + "_violation_count"
    for key in (
        "action_mapping_violation_count",
        "rng_domain_violation_count",
        "hidden_information_violation_count",
        "risk_reserve_contract_violation_count",
        "retained_order_violation_count",
        "phase_filter_violation_count",
        "pooled_phase_violation_count",
        "locked_action_change_violation_count",
        "extreme_tail_statistic_violation_count",
    )
)
_SELECTOR_GATE_NAMES = (
    *_QUALITY_GATE_NAMES,
    *_INTEGRITY_GATE_NAMES,
    "nonfire_exact_baseline_action_fallback",
)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HEX = frozenset("0123456789abcdef")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an array")
    return value


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{label} changed: expected {expected!r}, got {actual!r}")


def _keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{label} keys changed")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    return dict(_mapping(value, label))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be repo-relative")
    relative = Path(value)
    if (
        relative.is_absolute()
        or bool(relative.drive)
        or bool(relative.root)
        or ".." in relative.parts
    ):
        raise ValueError(f"{label} must be repo-relative")
    return relative


def _reject_links(root: Path, relative: Path, label: str) -> Path:
    lexical = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and (
            current.is_symlink()
            or (hasattr(current, "is_junction") and current.is_junction())
        ):
            raise ValueError(f"{label} traverses a symlink or junction")
    return lexical


def _resolve_file(
    root: Path, identity: Mapping[str, Any], label: str
) -> Path:
    _keys(identity, frozenset({"path", "sha256"}), f"{label} identity")
    relative = _relative_path(identity.get("path"), f"{label} path")
    expected_sha = identity.get("sha256")
    if not _is_sha256(expected_sha):
        raise ValueError(f"{label} SHA-256 is invalid")
    target = _reject_links(root, relative, label).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repo root") from exc
    if not target.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if _sha256(target) != expected_sha:
        raise ValueError(f"{label} SHA-256 changed")
    return target


def _resolve_directory(root: Path, relative: Any, label: str) -> Path:
    relative_path = _relative_path(relative, label)
    target = _reject_links(root, relative_path, label).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repo root") from exc
    if not target.is_dir():
        raise ValueError(f"{label} must be a non-symlink directory")
    return target


def _validate_frozen_contracts(
    root: Path, raw: Any
) -> tuple[dict[str, Path], dict[str, Any]]:
    contracts = _mapping(raw, "frozen contracts")
    _require(contracts, _FROZEN_CONTRACTS, "frozen contracts")
    paths = {
        name: _resolve_file(root, _mapping(identity, name), name)
        for name, identity in contracts.items()
    }
    load_and_validate_attempt13_plan(paths["search_plan"])
    distillation = _load_json(paths["distillation_plan"], "distillation plan")
    _require(
        distillation.get("schema"),
        ATTEMPT13_DISTILLATION_CONFIG_SCHEMA,
        "distillation plan schema",
    )
    population = acceptance.load_and_validate_attempt13_population_plan(
        paths["population_plan"]
    )
    population.pop("_freshness_counts", None)
    population.pop("_freshness_registry", None)
    preflight = _load_json(paths["correctness_preflight"], "correctness preflight")
    _require(
        {
            key: preflight.get(key)
            for key in (
                "schema",
                "status",
                "decision",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        },
        {
            "schema": "hu_m43_attempt13_preflight_result_v1",
            "status": "pass_correctness_preflight",
            "decision": "authorize_development200_package_only",
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
        "correctness preflight",
    )
    return paths, population


def _artifact_paths(
    root: Path, raw: Any, expected: frozenset[str]
) -> tuple[Mapping[str, Any], dict[str, Path]]:
    artifacts = _mapping(raw, "immutable artifacts")
    _keys(artifacts, expected, "immutable artifacts")
    return artifacts, {
        name: _resolve_file(root, _mapping(identity, name), name)
        for name, identity in artifacts.items()
    }


def _selector_expected_status(stage: str, decision: str) -> str:
    if stage == "development":
        return (
            "go_write_separate_search_freeze_only"
            if decision == "go"
            else "no_go_close_attempt13_development"
        )
    return (
        "go_attempt13_audit50_search_quality"
        if decision == "go"
        else "no_go_close_attempt13_audit50"
    )


def _validate_spot_chain(
    *,
    stage: str,
    paths: Mapping[str, Path],
    expected_preceding_gate_sha256: str,
) -> None:
    """Re-open the immutable package, schedule, and launch authorization."""

    prefix = stage
    run_dir = paths[f"{prefix}_manifest"].parent
    expected_layout = {
        f"{prefix}_manifest": "manifest.json",
        f"{prefix}_authorization": "execution_authorization.json",
        f"{prefix}_launch_authorization": "launch_authorization.json",
        f"{prefix}_schedule": spot.SCHEDULE_NAME,
        f"{prefix}_source_archive": spot.SOURCE_NAME,
    }
    for artifact, filename in expected_layout.items():
        expected_path = (run_dir / filename).resolve()
        if paths[artifact] != expected_path:
            raise ValueError(f"{stage} {artifact} is outside its immutable run layout")

    validated_manifest, validated_launch = spot.validate_launch(run_dir)
    recorded_manifest = _load_json(paths[f"{prefix}_manifest"], f"{stage} manifest")
    recorded_launch = _load_json(
        paths[f"{prefix}_launch_authorization"], f"{stage} launch authorization"
    )
    if _canonical(validated_manifest) != _canonical(recorded_manifest):
        raise ValueError(f"{stage} package validator returned a different manifest")
    if _canonical(validated_launch) != _canonical(recorded_launch):
        raise ValueError(f"{stage} launch validator returned a different authorization")
    gate = _mapping(recorded_manifest.get("preceding_gate"), f"{stage} preceding gate")
    _require(
        gate.get("sha256"),
        expected_preceding_gate_sha256,
        f"{stage} preceding gate SHA-256",
    )


def _validate_selector_chain(
    *,
    stage: str,
    expected_decision: str,
    plan_path: Path,
    preceding_gate_sha256: str,
    artifacts: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    if stage not in {"development", "audit"} or expected_decision not in {"go", "no_go"}:
        raise ValueError("invalid selector validation request")
    prefix = stage
    mode = "development" if stage == "development" else "future_audit"
    roots = 200 if stage == "development" else 50
    first = 0 if stage == "development" else 200
    last = first + roots - 1
    _validate_spot_chain(
        stage=stage,
        paths=paths,
        expected_preceding_gate_sha256=preceding_gate_sha256,
    )
    manifest = _load_json(paths[f"{prefix}_manifest"], f"{stage} manifest")
    authorization = _load_json(
        paths[f"{prefix}_authorization"], f"{stage} authorization"
    )
    launch = _load_json(
        paths[f"{prefix}_launch_authorization"], f"{stage} launch authorization"
    )
    receive = _load_json(
        paths[f"{prefix}_receive_receipt"], f"{stage} receive receipt"
    )
    decision = _load_json(paths[f"{prefix}_decision"], f"{stage} decision")
    receipt = _load_json(
        paths[f"{prefix}_decision_receipt"], f"{stage} decision receipt"
    )
    run_name = manifest.get("run_name")
    if not isinstance(run_name, str) or not run_name:
        raise ValueError(f"{stage} run name is invalid")
    manifest_sha = artifacts[f"{prefix}_manifest"]["sha256"]
    authorization_sha = artifacts[f"{prefix}_authorization"]["sha256"]
    schedule_sha = artifacts[f"{prefix}_schedule"]["sha256"]
    source_sha = artifacts[f"{prefix}_source_archive"]["sha256"]
    merged_sha = artifacts[f"{prefix}_merged_teacher"]["sha256"]
    selector_sha = artifacts[f"{prefix}_selector_source"]["sha256"]
    decision_sha = artifacts[f"{prefix}_decision"]["sha256"]

    _require(manifest.get("schema"), "hu_m43_attempt13_spot_package_v1", f"{stage} manifest schema")
    _require(manifest.get("status"), "packaged_without_root_or_gcloud", f"{stage} manifest status")
    _require(manifest.get("mode"), mode, f"{stage} manifest mode")
    _require(manifest.get("plan_sha256"), M43_ATTEMPT13_PLAN_SHA256, f"{stage} manifest plan")
    _require(manifest.get("schedule_sha256"), schedule_sha, f"{stage} manifest schedule")
    _require(manifest.get("source_sha256"), source_sha, f"{stage} manifest source")
    _require(manifest.get("total_shards"), roots, f"{stage} manifest shards")
    _require(manifest.get("ai_profiles_sha256"), _AI_PROFILES_SHA256, f"{stage} manifest profiles")
    for key in ("current_profile_mutated", "runtime_policy_activated"):
        _require(manifest.get(key), False, f"{stage} manifest {key}")

    _require(authorization.get("schema"), "hu_m43_attempt13_execution_authorization_v1", f"{stage} authorization schema")
    _require(authorization.get("status"), "authorized", f"{stage} authorization status")
    _require(authorization.get("mode"), mode, f"{stage} authorization mode")
    _require(authorization.get("plan_sha256"), M43_ATTEMPT13_PLAN_SHA256, f"{stage} authorization plan")
    _require(authorization.get("source_package_sha256"), source_sha, f"{stage} authorization source")
    _require(authorization.get("root_index_first"), first, f"{stage} authorization first root")
    _require(authorization.get("root_index_last"), last, f"{stage} authorization last root")
    for key in ("current_profile_mutated", "runtime_policy_activated"):
        _require(authorization.get(key), False, f"{stage} authorization {key}")

    _require(launch.get("schema"), "hu_m43_attempt13_spot_launch_authorization_v1", f"{stage} launch schema")
    _require(launch.get("status"), "authorized", f"{stage} launch status")
    _require(launch.get("mode"), mode, f"{stage} launch mode")
    _require(launch.get("run_name"), run_name, f"{stage} launch run")
    _require(launch.get("manifest_sha256"), manifest_sha, f"{stage} launch manifest")
    _require(launch.get("schedule_sha256"), schedule_sha, f"{stage} launch schedule")
    _require(launch.get("source_sha256"), source_sha, f"{stage} launch source")
    _require(launch.get("total_shards"), roots, f"{stage} launch shards")
    _require(launch.get("spot_authorized"), True, f"{stage} Spot authorization")
    for key in ("current_profile_mutated", "runtime_policy_activated"):
        _require(launch.get(key), False, f"{stage} launch {key}")

    _require(receive.get("schema"), "hu_m43_attempt13_receive_v1", f"{stage} receive schema")
    _keys(receive, _RECEIVE_KEYS, f"{stage} receive")
    _require(receive.get("status"), "complete", f"{stage} receive status")
    _require(receive.get("run_name"), run_name, f"{stage} receive run")
    _require(receive.get("mode"), mode, f"{stage} receive mode")
    _require(receive.get("roots"), roots, f"{stage} receive roots")
    _require(receive.get("root_indices"), list(range(first, last + 1)), f"{stage} receive root indices")
    _require(
        receive.get("profiles"),
        [M43_ATTEMPT13_PROFILES[index % 5] for index in range(first, last + 1)],
        f"{stage} receive profiles",
    )
    _require(receive.get("manifest_sha256"), manifest_sha, f"{stage} receive manifest")
    _require(receive.get("authorization_sha256"), authorization_sha, f"{stage} receive authorization")
    _require(receive.get("schedule_sha256"), schedule_sha, f"{stage} receive schedule")
    _require(receive.get("source_sha256"), source_sha, f"{stage} receive source")
    _require(receive.get("merged_sha256"), merged_sha, f"{stage} receive merged")
    if not _is_sha256(receive.get("audit_sha256")):
        raise ValueError(f"{stage} receive audit SHA-256 is invalid")
    _require(
        receive.get("batch_boundary_validation_count"),
        1,
        f"{stage} receive batch boundary validation",
    )
    _require(
        receive.get("per_shard_boundary_revalidation_count"),
        0,
        f"{stage} receive per-shard boundary validation",
    )
    _require(receive.get("selector_executed"), False, f"{stage} receive selector")
    for key in ("current_profile_mutated", "runtime_policy_activated"):
        _require(receive.get(key), False, f"{stage} receive {key}")

    population_key = "development_population" if stage == "development" else "audit_population"
    decision_keys = frozenset(
        {
            "schema",
            "status",
            "decision",
            "search_freeze_authorized",
            "selected_arm",
            "selected_threshold",
            "source",
            population_key,
            "metrics",
            "gates",
            "decision_contract",
            "integrity",
            "science_boundary",
        }
    )
    _keys(decision, decision_keys, f"{stage} decision")
    expected_schema = (
        ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA
        if stage == "development"
        else ATTEMPT13_AUDIT50_DECISION_SCHEMA
    )
    _require(decision.get("schema"), expected_schema, f"{stage} decision schema")
    _require(decision.get("status"), _selector_expected_status(stage, expected_decision), f"{stage} decision status")
    _require(decision.get("decision"), expected_decision, f"{stage} decision")
    _require(decision.get("search_freeze_authorized"), expected_decision == "go", f"{stage} search freeze")
    _require(decision.get("selected_arm"), None, f"{stage} selected arm")
    _require(decision.get("selected_threshold"), None, f"{stage} selected threshold")
    population = _mapping(decision.get(population_key), f"{stage} population")
    _require(
        population,
        {
            "roots": roots,
            "profile_counts": {
                profile: roots // len(M43_ATTEMPT13_PROFILES)
                for profile in M43_ATTEMPT13_PROFILES
            },
        },
        f"{stage} population",
    )
    source = _mapping(decision.get("source"), f"{stage} decision source")
    _keys(
        source,
        frozenset(
            {
                "input_jsonl_sha256",
                "plan_sha256",
                "authorization_sha256",
                "source_package_sha256",
                "run_name",
                "selector_source_sha256",
                "root_identity_sha256",
            }
        ),
        f"{stage} decision source",
    )
    _require(source.get("input_jsonl_sha256"), merged_sha, f"{stage} decision input")
    _require(source.get("plan_sha256"), M43_ATTEMPT13_PLAN_SHA256, f"{stage} decision plan")
    _require(source.get("authorization_sha256"), authorization_sha, f"{stage} decision authorization")
    _require(source.get("source_package_sha256"), source_sha, f"{stage} decision package")
    _require(source.get("run_name"), run_name, f"{stage} decision run")
    _require(source.get("selector_source_sha256"), selector_sha, f"{stage} decision selector")
    if not _is_sha256(source.get("root_identity_sha256")):
        raise ValueError(f"{stage} root identity SHA-256 is invalid")

    selector = (
        development_selector.select_attempt13_development
        if stage == "development"
        else audit_selector.select_attempt13_audit50
    )
    recomputed = selector(
        input_path=paths[f"{prefix}_merged_teacher"],
        plan_path=plan_path,
        authorization_path=paths[f"{prefix}_authorization"],
        source_package_sha256=source_sha,
        run_name=run_name,
    )
    if _canonical(recomputed) != _canonical(decision):
        raise ValueError(f"{stage} decision differs from read-only selector recomputation")
    expected_decision_bytes = (_canonical(recomputed) + "\n").encode("utf-8")
    if paths[f"{prefix}_decision"].read_bytes() != expected_decision_bytes:
        raise ValueError(f"{stage} decision is not the canonical selector output")

    gates = [_mapping(gate, f"{stage} gate") for gate in _sequence(decision.get("gates"), f"{stage} gates")]
    if not gates or any(set(gate) != {"name", "passed", "observed", "requirement"} for gate in gates):
        raise ValueError(f"{stage} gate contract changed")
    names = [gate.get("name") for gate in gates]
    _require(names, list(_SELECTOR_GATE_NAMES), f"{stage} gate names")
    requirements = {
        "fires_total": ">= 40" if stage == "development" else ">= 10",
        "fires_each_profile": "each >= 3" if stage == "development" else "each >= 1",
        "mean_delta_per_state": "> 0",
        "mean_delta_per_fire": "> 0",
        "false_positive_rate_per_fire": "<= 0.4",
        "maximum_per_fired_root_p95_loss": "<= 25",
        "maximum_per_fired_root_p99_loss": "<= 40",
        "maximum_per_fired_root_max_loss": "<= 50",
        **{name: "= 0" for name in _INTEGRITY_GATE_NAMES},
        "nonfire_exact_baseline_action_fallback": "required",
    }
    for gate in gates:
        _require(
            gate.get("requirement"),
            requirements[str(gate["name"])],
            f"{stage} {gate['name']} requirement",
        )
    if expected_decision == "go" and not all(gate.get("passed") is True for gate in gates):
        raise ValueError(f"{stage} Go contains a failed gate")
    if expected_decision == "no_go" and not any(gate.get("passed") is False for gate in gates):
        raise ValueError(f"{stage} No-Go has no failed gate")
    if any(type(gate.get("passed")) is not bool for gate in gates):
        raise ValueError(f"{stage} gate passed value changed")
    contract = _mapping(decision.get("decision_contract"), f"{stage} decision contract")
    _keys(
        contract,
        frozenset(
            {
                "single_frozen_search_architecture",
                "arm_selection_performed",
                "threshold_selection_performed",
                "gate_evaluation_count",
                "all_gates_required",
            }
        ),
        f"{stage} decision contract",
    )
    _require(contract.get("single_frozen_search_architecture"), True, f"{stage} single architecture")
    _require(contract.get("arm_selection_performed"), False, f"{stage} arm selection")
    _require(contract.get("threshold_selection_performed"), False, f"{stage} threshold selection")
    _require(contract.get("gate_evaluation_count"), 1, f"{stage} gate count")
    _require(contract.get("all_gates_required"), True, f"{stage} all gates")
    integrity = _mapping(decision.get("integrity"), f"{stage} integrity")
    _keys(integrity, _INTEGRITY_KEYS, f"{stage} integrity")
    for key, value in integrity.items():
        _require(value, True if key.endswith("fallback_verified") else 0, f"{stage} {key}")
    science = _mapping(decision.get("science_boundary"), f"{stage} science boundary")
    _keys(
        science,
        frozenset(
            {
                "assessment_source",
                "teacher_values_are_realized_match_ev",
                "future_audit_authorized",
                "fit_performed",
                "threshold_selected",
                "runtime_policy_activated",
                "current_profile_mutated",
                "full_replacement_enabled",
            }
        ),
        f"{stage} science boundary",
    )
    _require(science.get("assessment_source"), "disjoint_E512_locked_final_nonbaseline_output_vs_explicit_baseline", f"{stage} assessment source")
    for key in (
        "teacher_values_are_realized_match_ev",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "runtime_policy_activated",
        "current_profile_mutated",
        "full_replacement_enabled",
    ):
        _require(science.get(key), False, f"{stage} science {key}")

    expected_receipt_schema = (
        ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA
        if stage == "development"
        else ATTEMPT13_AUDIT50_RECEIPT_SCHEMA
    )
    receipt_keys = {
        "schema",
        "status",
        "run_name",
        "decision_sha256",
        "decision",
        "search_freeze_authorized",
        "gate_evaluation_count",
        "selector_executed",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "runtime_policy_activated",
        "current_profile_mutated",
    }
    if stage == "audit":
        receipt_keys.add("audit_rows_used_for_fit")
    _keys(receipt, frozenset(receipt_keys), f"{stage} receipt")
    _require(receipt.get("schema"), expected_receipt_schema, f"{stage} receipt schema")
    _require(receipt.get("status"), "single_frozen_gate_evaluation_complete", f"{stage} receipt status")
    _require(receipt.get("run_name"), run_name, f"{stage} receipt run")
    _require(receipt.get("decision_sha256"), decision_sha, f"{stage} receipt decision SHA")
    _require(receipt.get("decision"), expected_decision, f"{stage} receipt decision")
    _require(receipt.get("search_freeze_authorized"), expected_decision == "go", f"{stage} receipt search freeze")
    _require(receipt.get("gate_evaluation_count"), 1, f"{stage} receipt gate count")
    _require(receipt.get("selector_executed"), True, f"{stage} receipt selector")
    for key in (
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "runtime_policy_activated",
        "current_profile_mutated",
    ):
        _require(receipt.get(key), False, f"{stage} receipt {key}")
    if stage == "audit":
        _require(receipt.get("audit_rows_used_for_fit"), False, "audit rows used for fit")
    return decision


def _validate_development_freeze(
    artifacts: Mapping[str, Any], paths: Mapping[str, Path], decision: Mapping[str, Any]
) -> dict[str, Any]:
    freeze = _load_json(paths["development_go_freeze"], "development Go freeze")
    _require(freeze.get("schema"), ATTEMPT13_DEVELOPMENT_GO_FREEZE_SCHEMA, "development freeze schema")
    _require(freeze.get("status"), ATTEMPT13_DEVELOPMENT_GO_FREEZE_STATUS, "development freeze status")
    _require(freeze.get("decision"), "go", "development freeze decision")
    _require(freeze.get("run_name"), _mapping(decision.get("source"), "development source").get("run_name"), "development freeze run")
    bindings = _mapping(freeze.get("bindings"), "development freeze bindings")
    expected = {
        "plan_sha256": M43_ATTEMPT13_PLAN_SHA256,
        "manifest_sha256": artifacts["development_manifest"]["sha256"],
        "execution_authorization_sha256": artifacts["development_authorization"]["sha256"],
        "source_package_sha256": artifacts["development_source_archive"]["sha256"],
        "schedule_sha256": artifacts["development_schedule"]["sha256"],
        "startup_sha256": _load_json(paths["development_manifest"], "development manifest")["startup_sha256"],
        "merged_input_sha256": artifacts["development_merged_teacher"]["sha256"],
        "receive_receipt_sha256": artifacts["development_receive_receipt"]["sha256"],
        "selector_decision_sha256": artifacts["development_decision"]["sha256"],
        "selector_receipt_sha256": artifacts["development_decision_receipt"]["sha256"],
        "selector_source_sha256": artifacts["development_selector_source"]["sha256"],
        "root_identity_sha256": _mapping(decision.get("source"), "development source")["root_identity_sha256"],
    }
    _require(bindings, expected, "development freeze bindings")
    authorization = _mapping(freeze.get("authorization_scope"), "development freeze authorization")
    _require(authorization.get("future_audit_package_authorized"), True, "future audit package authorization")
    _require(authorization.get("future_audit_authorization_artifact_authorized"), True, "future audit artifact authorization")
    for key in (
        "future_audit_launch_authorized",
        "future_audit_started",
        "fit_authorized",
        "fit_started",
        "threshold_selection_authorized",
        "threshold_selection_started",
        "runtime_policy_activated",
        "current_profile_mutated",
        "full_replacement_enabled",
    ):
        _require(authorization.get(key), False, f"development freeze {key}")
    for key in (
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "runtime_policy_activated",
        "current_profile_mutated",
        "full_replacement_enabled",
    ):
        _require(freeze.get(key), False, f"development freeze {key}")
    return freeze


def _validate_training_artifacts(
    artifacts: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    manifest = _load_json(paths["training_manifest"], "training manifest")
    acceptance._validate_training_manifest(
        manifest, expected_model_sha256=artifacts["training_model"]["sha256"]
    )
    source = _mapping(manifest.get("source"), "training source")
    expected = {
        "development_jsonl_sha256": artifacts["development_merged_teacher"]["sha256"],
        "development_decision_sha256": artifacts["development_decision"]["sha256"],
        "development_selector_receipt_sha256": artifacts["development_decision_receipt"]["sha256"],
        "development_pass_freeze_sha256": artifacts["development_go_freeze"]["sha256"],
        "development_plan_sha256": M43_ATTEMPT13_PLAN_SHA256,
        "development_package_manifest_sha256": artifacts["development_manifest"]["sha256"],
        "development_source_package_sha256": artifacts["development_source_archive"]["sha256"],
        "runtime_source_archive_sha256": artifacts["runtime_source_archive"]["sha256"],
        "runtime_source_manifest_sha256": artifacts["runtime_source_manifest"]["sha256"],
    }
    for key, value in expected.items():
        _require(source.get(key), value, f"training source {key}")


def _gate_snapshot(source_key: str, payload: Mapping[str, Any], recomputed: bool) -> dict[str, Any]:
    gates = [_mapping(gate, "terminal gate") for gate in _sequence(payload.get("gates"), "terminal gates")]
    failed = [str(gate.get("name")) for gate in gates if gate.get("passed") is False]
    passed = sum(gate.get("passed") is True for gate in gates)
    decision = payload.get("decision", payload.get("status"))
    contract = payload.get("decision_contract")
    count = _mapping(contract, "terminal decision contract").get("gate_evaluation_count") if isinstance(contract, Mapping) else None
    return {
        "source_artifact": source_key,
        "source_status": payload.get("status"),
        "decision": decision,
        "passed_gates": passed,
        "total_gates": len(gates),
        "failed_gates": failed,
        "gate_evaluation_count": count,
        "recomputed_from_records": recomputed,
    }


def _named_gate_passed(payload: Mapping[str, Any], name: str) -> bool:
    gates = [
        _mapping(gate, "terminal gate")
        for gate in _sequence(payload.get("gates"), "terminal gates")
    ]
    matches = [gate for gate in gates if gate.get("name") == name]
    if len(matches) != 1 or type(matches[0].get("passed")) is not bool:
        raise ValueError(f"terminal gate {name} is missing or duplicated")
    return bool(matches[0]["passed"])


def _validate_science(
    raw: Any, *, terminal_stage: str, terminal: Mapping[str, Any]
) -> None:
    science = _mapping(raw, "closeout science boundary")
    _keys(science, _SCIENCE_KEYS, "closeout science boundary")
    always_true = (
        "public_information_set_only",
        "candidate_selection_and_evaluation_rng_independent",
        "first_seat_candidate_is_exact_baseline_delegate",
    )
    always_false = (
        "opponent_private_discard_used",
        "opponent_profile_runtime_feature_used",
        "teacher_values_are_realized_match_ev",
        "teacher_ev_or_lcb_runtime_gate",
        "threshold_reselection_performed",
        "alternate_seed_retry_performed",
        "top1_accuracy_is_acceptance_gate",
        "unseen_population_robustness_guaranteed",
        "nash_equilibrium_claim",
        "exploitability_bound_claim",
        "mathematically_proven_optimal",
    )
    for key in always_true:
        _require(science.get(key), True, f"science {key}")
    for key in always_false:
        _require(science.get(key), False, f"science {key}")
    is_population = terminal_stage == "population"
    _require(science.get("realized_population_evidence"), is_population, "realized population evidence")
    cancellation_verified = is_population and _named_gate_passed(
        terminal, "nonfire_counterfactual_cancellation"
    )
    _require(
        science.get("nonfire_full_trajectory_cancellation_verified"),
        cancellation_verified,
        "full trajectory cancellation",
    )
    expected_scope = {
        "development200": "fresh_development200_search_quality_only",
        "audit50": "fresh_audit50_search_quality_only",
        "population": "frozen_four_opponent_1000_paired_seeds_each_realized_match_ev",
    }[terminal_stage]
    _require(science.get("evaluation_scope"), expected_scope, "science evaluation scope")


def _validate_activation(
    raw: Any,
    *,
    terminal_stage: str,
    status: str,
    terminal: Mapping[str, Any],
) -> None:
    activation = _mapping(raw, "closeout activation")
    _keys(activation, _ACTIVATION_KEYS, "closeout activation")
    _require(activation.get("profile_id"), acceptance.ATTEMPT13_PROFILE_ID, "activation profile")
    for key in (
        "automatic_activation_authorized",
        "current_profile_mutated",
        "runtime_policy_activated",
        "full_replacement_enabled",
    ):
        _require(activation.get(key), False, f"activation {key}")
    is_population = terminal_stage == "population"
    is_go = is_population and status == "complete_go"
    binding_verified = is_population and _named_gate_passed(
        terminal, "runtime_artifact_hash_chain"
    )
    _require(activation.get("runtime_artifact_frozen"), is_population, "runtime artifact frozen")
    _require(activation.get("runtime_binding_verified"), binding_verified, "runtime binding")
    _require(activation.get("promotion_eligible"), is_go, "promotion eligibility")
    _require(activation.get("explicit_opt_in_authorized"), is_go, "explicit opt-in")


def _validate_population(
    *,
    root: Path,
    artifacts: Mapping[str, Any],
    paths: Mapping[str, Path],
    population_plan: Mapping[str, Any],
    recompute_raw: Any,
) -> dict[str, Any]:
    _validate_training_artifacts(artifacts, paths)
    recompute = _mapping(recompute_raw, "population recompute inputs")
    _keys(
        recompute,
        frozenset({"runtime_source_root", "runtime_dependency_root"}),
        "population recompute inputs",
    )
    runtime_source_root = _resolve_directory(
        root, recompute.get("runtime_source_root"), "runtime source root"
    )
    runtime_dependency_root = _resolve_directory(
        root, recompute.get("runtime_dependency_root"), "runtime dependency root"
    )
    preflight = _load_json(paths["population_preflight"], "population preflight")
    recomputed_preflight = acceptance.build_attempt13_population_preflight(
        model_path=paths["runtime_model"],
        training_manifest_path=paths["training_manifest"],
        runtime_freeze_path=paths["runtime_freeze"],
        population_plan_path=root / _FROZEN_CONTRACTS["population_plan"]["path"],
        runtime_source_archive_path=paths["runtime_source_archive"],
        runtime_source_manifest_path=paths["runtime_source_manifest"],
        runtime_source_root=runtime_source_root,
        runtime_dependency_root=runtime_dependency_root,
        development_decision_path=paths["development_decision"],
        development_selector_receipt_path=paths["development_decision_receipt"],
        development_pass_freeze_path=paths["development_go_freeze"],
        audit_decision_path=paths["audit_decision"],
        audit_selector_receipt_path=paths["audit_decision_receipt"],
    )
    if _canonical(preflight) != _canonical(recomputed_preflight):
        raise ValueError("population preflight differs from full provenance recomputation")
    _require(preflight.get("schema"), acceptance.ATTEMPT13_POPULATION_PREFLIGHT_SCHEMA, "population preflight schema")
    _require(preflight.get("status"), "pass", "population preflight status")
    _require(preflight.get("profile_id"), acceptance.ATTEMPT13_PROFILE_ID, "population preflight profile")
    _require(preflight.get("model_sha256"), artifacts["runtime_model"]["sha256"], "population preflight model")
    _require(preflight.get("training_manifest_sha256"), artifacts["training_manifest"]["sha256"], "population preflight training")
    _require(preflight.get("runtime_freeze_sha256"), artifacts["runtime_freeze"]["sha256"], "population preflight freeze")
    _require(preflight.get("runtime_source_archive_sha256"), artifacts["runtime_source_archive"]["sha256"], "population preflight source archive")
    _require(preflight.get("runtime_source_manifest_sha256"), artifacts["runtime_source_manifest"]["sha256"], "population preflight source manifest")
    for field, artifact in (
        ("development_decision_sha256", "development_decision"),
        ("development_selector_receipt_sha256", "development_decision_receipt"),
        ("development_pass_freeze_sha256", "development_go_freeze"),
        ("audit50_decision_sha256", "audit_decision"),
        ("audit50_selector_receipt_sha256", "audit_decision_receipt"),
    ):
        _require(
            preflight.get(field),
            artifacts[artifact]["sha256"],
            f"population preflight {field}",
        )
    _require(preflight.get("current_profile_mutated"), False, "population preflight current")
    _require(preflight.get("no_runtime_activation"), True, "population preflight activation")

    run_manifest = _load_json(paths["population_run_manifest"], "population run manifest")
    _require(run_manifest.get("schema"), "hu_m43_attempt13_population_spot_manifest_v1", "population manifest schema")
    _require(run_manifest.get("population_plan", {}).get("sha256"), acceptance.ATTEMPT13_POPULATION_PLAN_SHA256, "population manifest plan")
    _require(run_manifest.get("source", {}).get("sha256"), artifacts["population_source_archive"]["sha256"], "population manifest source")
    run_runtime = _mapping(run_manifest.get("runtime"), "population manifest runtime")
    _require(run_runtime.get("model_sha256"), artifacts["runtime_model"]["sha256"], "population manifest model")
    _require(run_runtime.get("training_manifest_sha256"), artifacts["training_manifest"]["sha256"], "population manifest training")
    _require(run_runtime.get("runtime_freeze_sha256"), artifacts["runtime_freeze"]["sha256"], "population manifest freeze")
    _require(run_runtime.get("runtime_source_archive_sha256"), artifacts["runtime_source_archive"]["sha256"], "population manifest runtime source archive")
    _require(run_runtime.get("runtime_source_manifest_sha256"), artifacts["runtime_source_manifest"]["sha256"], "population manifest runtime source manifest")
    run_preflight = _mapping(run_manifest.get("launch_preflight"), "population manifest preflight")
    if _canonical(run_preflight) != _canonical(preflight):
        raise ValueError("population manifest does not embed the exact launch preflight")
    for field, artifact in (
        ("development_decision_sha256", "development_decision"),
        ("development_selector_receipt_sha256", "development_decision_receipt"),
        ("development_pass_freeze_sha256", "development_go_freeze"),
        ("audit50_decision_sha256", "audit_decision"),
        ("audit50_selector_receipt_sha256", "audit_decision_receipt"),
    ):
        _require(
            run_preflight.get(field),
            artifacts[artifact]["sha256"],
            f"population manifest {field}",
        )
    _require(run_manifest.get("current_profile_mutated"), False, "population manifest current")
    _require(run_manifest.get("no_runtime_activation"), True, "population manifest activation")

    evaluation = _load_json(paths["population_evaluation"], "population evaluation")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(paths["population_records"].read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank population record at line {number}")
        try:
            records.append(dict(_mapping(json.loads(line), f"population record {number}")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid population record at line {number}") from exc
    merge = _load_json(paths["population_merge_manifest"], "population merge manifest")
    recorded_status = _load_json(paths["population_acceptance_status"], "population acceptance status")
    recomputed = acceptance.validate_attempt13_population_acceptance(
        evaluation=evaluation,
        records=records,
        population_plan=population_plan,
        merge_manifest=merge,
        model_path=paths["runtime_model"],
        training_manifest_path=paths["training_manifest"],
        runtime_freeze_path=paths["runtime_freeze"],
        runtime_source_archive_path=paths["runtime_source_archive"],
        runtime_source_manifest_path=paths["runtime_source_manifest"],
        runtime_source_root=runtime_source_root,
        runtime_dependency_root=runtime_dependency_root,
        source_hashes={
            "population_plan": acceptance.ATTEMPT13_POPULATION_PLAN_SHA256,
            "population_plan_path": str((root / _FROZEN_CONTRACTS["population_plan"]["path"]).resolve()),
            "records": artifacts["population_records"]["sha256"],
            "evaluation": artifacts["population_evaluation"]["sha256"],
            "merge_manifest": artifacts["population_merge_manifest"]["sha256"],
        },
    )
    if _canonical(recomputed) != _canonical(recorded_status):
        raise ValueError("population acceptance status differs from record recomputation")
    receipt = _load_json(paths["population_receipt"], "population receipt")
    _keys(receipt, _POPULATION_RECEIPT_KEYS, "population receipt")
    _require(receipt.get("schema"), "hu_m43_attempt13_population_spot_receipt_v1", "population receipt schema")
    _require(receipt.get("status"), "verified_and_merged", "population receipt status")
    _require(receipt.get("run_name"), run_manifest.get("run_name"), "population receipt run")
    _require(receipt.get("run_manifest_sha256"), artifacts["population_run_manifest"]["sha256"], "population receipt manifest")
    _require(receipt.get("population_plan_sha256"), acceptance.ATTEMPT13_POPULATION_PLAN_SHA256, "population receipt plan")
    _require(receipt.get("model_sha256"), artifacts["runtime_model"]["sha256"], "population receipt model")
    _require(receipt.get("population_source_archive_sha256"), artifacts["population_source_archive"]["sha256"], "population receipt source")
    _require(receipt.get("runtime_source_archive_sha256"), artifacts["runtime_source_archive"]["sha256"], "population receipt runtime source archive")
    _require(receipt.get("runtime_source_manifest_sha256"), artifacts["runtime_source_manifest"]["sha256"], "population receipt runtime source manifest")
    for field in (
        "runtime_source_closure_sha256",
        "runtime_semantic_closure_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "runtime_dependency_closure_sha256",
        "seed_registry_sha256",
        "model_schema",
        "artifact_schema",
        "feature_schema",
        "head_schema",
        "action_score_mode",
        "profile_id",
        "baseline_profile",
    ):
        _require(
            receipt.get(field),
            preflight.get(field),
            f"population receipt {field}",
        )
    for field, artifact in (
        ("development_decision_sha256", "development_decision"),
        ("development_selector_receipt_sha256", "development_decision_receipt"),
        ("development_pass_freeze_sha256", "development_go_freeze"),
        ("audit50_decision_sha256", "audit_decision"),
        ("audit50_selector_receipt_sha256", "audit_decision_receipt"),
    ):
        _require(
            receipt.get(field),
            artifacts[artifact]["sha256"],
            f"population receipt {field}",
        )
    _require(receipt.get("development200_full_fit_bound"), True, "population receipt development fit")
    _require(receipt.get("audit50_one_shot_go_bound"), True, "population receipt audit Go")
    _require(receipt.get("evaluation_sha256"), artifacts["population_evaluation"]["sha256"], "population receipt evaluation")
    _require(receipt.get("records_sha256"), artifacts["population_records"]["sha256"], "population receipt records")
    _require(receipt.get("merge_manifest_sha256"), artifacts["population_merge_manifest"]["sha256"], "population receipt merge")
    _require(receipt.get("acceptance_status_sha256"), artifacts["population_acceptance_status"]["sha256"], "population receipt acceptance")
    _require(receipt.get("acceptance_decision"), recorded_status.get("status"), "population receipt decision")
    _require(
        Path(str(receipt.get("evaluation_path"))).resolve(),
        paths["population_evaluation"],
        "population receipt evaluation path",
    )
    _require(
        Path(str(receipt.get("records_path"))).resolve(),
        paths["population_records"],
        "population receipt records path",
    )
    _require(
        receipt.get("paired_seeds_per_opponent"),
        evaluation.get("paired_seeds_per_opponent"),
        "population receipt paired seeds",
    )
    all_seats = _mapping(
        _mapping(evaluation.get("population"), "population evaluation").get("all_seats"),
        "population evaluation all seats",
    )
    _require(
        receipt.get("valid_overrides"),
        all_seats.get("overrides"),
        "population receipt valid overrides",
    )
    _require(
        receipt.get("invalid_counterfactuals"),
        evaluation.get("invalid_counterfactuals"),
        "population receipt invalid counterfactuals",
    )
    _require(
        receipt.get("nonfire_cancellation_mismatches"),
        evaluation.get("nonfire_cancellation_mismatches"),
        "population receipt cancellation mismatches",
    )
    receipt_shards = [
        _mapping(row, "population receipt shard")
        for row in _sequence(receipt.get("shards"), "population receipt shards")
    ]
    merge_shards = [
        _mapping(row, "population merge shard")
        for row in _sequence(merge.get("shards"), "population merge shards")
    ]
    if len(receipt_shards) != acceptance.ATTEMPT13_POPULATION_SHARDS or len(
        merge_shards
    ) != acceptance.ATTEMPT13_POPULATION_SHARDS:
        raise ValueError("population receipt shard count changed")
    for index, (receipt_shard, merge_shard) in enumerate(
        zip(receipt_shards, merge_shards, strict=True)
    ):
        _keys(
            receipt_shard,
            frozenset(
                {"shard", "done_sha256", "evaluation_sha256", "records_sha256"}
            ),
            f"population receipt shard {index}",
        )
        _require(receipt_shard.get("shard"), index, f"population receipt shard {index}")
        if not _is_sha256(receipt_shard.get("done_sha256")):
            raise ValueError(f"population receipt shard {index} DONE SHA-256 is invalid")
        _require(
            receipt_shard.get("evaluation_sha256"),
            merge_shard.get("evaluation_sha256"),
            f"population receipt shard {index} evaluation",
        )
        _require(
            receipt_shard.get("records_sha256"),
            merge_shard.get("records_sha256"),
            f"population receipt shard {index} records",
        )
    _require(
        receipt.get("teacher_calibration_locked_content_received"),
        False,
        "population receipt teacher content",
    )
    received_at = receipt.get("received_at")
    if not isinstance(received_at, str):
        raise ValueError("population receipt timestamp is invalid")
    try:
        datetime.fromisoformat(received_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("population receipt timestamp is invalid") from exc
    _require(receipt.get("current_profile_mutated"), False, "population receipt current")
    _require(receipt.get("no_runtime_activation"), True, "population receipt activation")
    return recorded_status


def validate_attempt13_closeout(
    repo_root: str | Path,
    *,
    closeout_path: str | Path = ATTEMPT13_CLOSEOUT_PATH,
) -> dict[str, Any]:
    """Validate one terminal Attempt13 branch and return a read-only receipt."""

    root = Path(repo_root).resolve()
    requested_input = Path(closeout_path)
    if requested_input.is_absolute():
        try:
            requested_relative = requested_input.relative_to(root)
        except ValueError as exc:
            raise ValueError("closeout path escapes repo root") from exc
        requested_relative = _relative_path(
            str(requested_relative), "closeout path"
        )
    else:
        requested_relative = _relative_path(str(requested_input), "closeout path")
    requested = _reject_links(root, requested_relative, "closeout").resolve()
    try:
        requested.relative_to(root)
    except ValueError as exc:
        raise ValueError("closeout path escapes repo root") from exc
    if not requested.is_file():
        raise ValueError("closeout must be a regular non-symlink file")
    closeout = _load_json(requested, "Attempt13 closeout")
    _keys(closeout, _TOP_LEVEL_KEYS, "Attempt13 closeout")
    _require(closeout.get("schema"), ATTEMPT13_CLOSEOUT_SCHEMA, "closeout schema")
    _require(closeout.get("milestone"), "M4.3-attempt13", "closeout milestone")
    status_date = closeout.get("status_date")
    if not isinstance(status_date, str) or not _DATE.fullmatch(status_date):
        raise ValueError("closeout status date is invalid")
    try:
        date.fromisoformat(status_date)
    except ValueError as exc:
        raise ValueError("closeout status date is invalid") from exc
    terminal_stage = closeout.get("terminal_stage")
    status = closeout.get("status")
    valid = {
        "development200": {"complete_no_go_development"},
        "audit50": {"complete_no_go_audit50"},
        "population": {"complete_go", "complete_no_go"},
    }
    if terminal_stage not in valid or status not in valid[terminal_stage]:
        raise ValueError("closeout terminal stage/status combination changed")
    contract_paths, population_plan = _validate_frozen_contracts(
        root, closeout.get("frozen_contracts")
    )

    scope = _mapping(closeout.get("opened_scope"), "opened scope")
    _keys(scope, _SCOPE_KEYS, "opened scope")
    _require(scope.get("preflight_passed"), True, "scope preflight")
    _require(scope.get("development_completed"), True, "scope development")
    _require(scope.get("audit50_fit_rows"), 0, "scope audit fit rows")
    _require(scope.get("fit_artifact_runtime_eligible"), False, "scope fit runtime eligibility")
    if terminal_stage == "development200":
        expected_artifacts = _DEVELOPMENT_ARTIFACTS
        _require(closeout.get("population_recompute"), None, "development population recompute")
        _require(
            scope,
            {
                "preflight_passed": True,
                "development_completed": True,
                "development_decision": "no_go",
                "future_audit_started": False,
                "future_audit_decision": None,
                "fit_state": "not_started",
                "audit50_fit_rows": 0,
                "fit_artifact_runtime_eligible": False,
                "runtime_freeze_written": False,
                "population_started": False,
                "population_decision": None,
            },
            "development opened scope",
        )
        artifacts, paths = _artifact_paths(root, closeout.get("immutable_artifacts"), expected_artifacts)
        decision = _validate_selector_chain(
            stage="development",
            expected_decision="no_go",
            plan_path=contract_paths["search_plan"],
            preceding_gate_sha256=_PREFLIGHT_SHA256,
            artifacts=artifacts,
            paths=paths,
        )
        terminal_key = "development_decision"
        terminal = decision
    elif terminal_stage == "audit50":
        fit_state = scope.get("fit_state")
        if fit_state not in {"not_started", "aborted_without_artifact", "completed_development200_only"}:
            raise ValueError("audit fit state changed")
        expected_artifacts = _DEVELOPMENT_ARTIFACTS | _AUDIT_ARTIFACTS | {"development_go_freeze"}
        if fit_state == "completed_development200_only":
            expected_artifacts |= _FIT_ARTIFACTS
        _require(closeout.get("population_recompute"), None, "audit population recompute")
        _require(scope.get("development_decision"), "go", "audit scope development decision")
        _require(scope.get("future_audit_started"), True, "audit scope started")
        _require(scope.get("future_audit_decision"), "no_go", "audit scope decision")
        _require(scope.get("runtime_freeze_written"), False, "audit scope runtime freeze")
        _require(scope.get("population_started"), False, "audit scope population")
        _require(scope.get("population_decision"), None, "audit scope population decision")
        artifacts, paths = _artifact_paths(root, closeout.get("immutable_artifacts"), frozenset(expected_artifacts))
        development = _validate_selector_chain(
            stage="development",
            expected_decision="go",
            plan_path=contract_paths["search_plan"],
            preceding_gate_sha256=_PREFLIGHT_SHA256,
            artifacts=artifacts,
            paths=paths,
        )
        _validate_development_freeze(artifacts, paths, development)
        if fit_state == "completed_development200_only":
            _validate_training_artifacts(artifacts, paths)
        terminal = _validate_selector_chain(
            stage="audit",
            expected_decision="no_go",
            plan_path=contract_paths["search_plan"],
            preceding_gate_sha256=artifacts["development_go_freeze"]["sha256"],
            artifacts=artifacts,
            paths=paths,
        )
        terminal_key = "audit_decision"
    else:
        expected_artifacts = (
            _DEVELOPMENT_ARTIFACTS
            | _AUDIT_ARTIFACTS
            | _FIT_ARTIFACTS
            | _POPULATION_ARTIFACTS
            | {"development_go_freeze"}
        )
        _require(
            scope,
            {
                "preflight_passed": True,
                "development_completed": True,
                "development_decision": "go",
                "future_audit_started": True,
                "future_audit_decision": "go",
                "fit_state": "completed_development200_only",
                "audit50_fit_rows": 0,
                "fit_artifact_runtime_eligible": False,
                "runtime_freeze_written": True,
                "population_started": True,
                "population_decision": status,
            },
            "population opened scope",
        )
        artifacts, paths = _artifact_paths(root, closeout.get("immutable_artifacts"), frozenset(expected_artifacts))
        development = _validate_selector_chain(
            stage="development",
            expected_decision="go",
            plan_path=contract_paths["search_plan"],
            preceding_gate_sha256=_PREFLIGHT_SHA256,
            artifacts=artifacts,
            paths=paths,
        )
        _validate_development_freeze(artifacts, paths, development)
        _validate_selector_chain(
            stage="audit",
            expected_decision="go",
            plan_path=contract_paths["search_plan"],
            preceding_gate_sha256=artifacts["development_go_freeze"]["sha256"],
            artifacts=artifacts,
            paths=paths,
        )
        terminal = _validate_population(
            root=root,
            artifacts=artifacts,
            paths=paths,
            population_plan=population_plan,
            recompute_raw=closeout.get("population_recompute"),
        )
        _require(terminal.get("status"), status, "population terminal status")
        _require(terminal.get("current_profile_mutated"), False, "population status current")
        _require(terminal.get("runtime_policy_activated"), False, "population status activation")
        _require(terminal.get("full_replacement"), False, "population status full replacement")
        _require(terminal.get("automatic_activation_authorized"), False, "population automatic activation")
        _require(terminal.get("explicit_opt_in_authorized"), status == "complete_go", "population explicit opt-in")
        terminal_key = "population_acceptance_status"

    _validate_science(
        closeout.get("science_boundary"),
        terminal_stage=terminal_stage,
        terminal=terminal,
    )
    _validate_activation(
        closeout.get("activation"),
        terminal_stage=terminal_stage,
        status=status,
        terminal=terminal,
    )
    snapshot = _mapping(closeout.get("terminal_gate_snapshot"), "terminal gate snapshot")
    _keys(snapshot, _GATE_SNAPSHOT_KEYS, "terminal gate snapshot")
    _require(snapshot, _gate_snapshot(terminal_key, terminal, terminal_stage == "population"), "terminal gate snapshot")
    activation = _mapping(closeout.get("activation"), "activation")
    return {
        "schema": ATTEMPT13_CLOSEOUT_VALIDATION_SCHEMA,
        "status": f"validated_{status}",
        "terminal_stage": terminal_stage,
        "terminal_artifact": terminal_key,
        "validated_artifacts": len(paths),
        "failed_gates": snapshot["failed_gates"],
        "profile_id": acceptance.ATTEMPT13_PROFILE_ID,
        "explicit_opt_in_authorized": activation["explicit_opt_in_authorized"],
        "automatic_activation_authorized": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement_enabled": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--closeout", type=Path, default=Path(ATTEMPT13_CLOSEOUT_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_attempt13_closeout(args.repo_root, closeout_path=args.closeout)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT13_CLOSEOUT_PATH",
    "ATTEMPT13_CLOSEOUT_SCHEMA",
    "ATTEMPT13_CLOSEOUT_VALIDATION_SCHEMA",
    "validate_attempt13_closeout",
]
