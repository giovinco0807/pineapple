"""Dormant, fail-closed Attempt08 future-audit runner.

This module is packaged in the same immutable source closure as development200,
but it cannot materialize roots 200..249 unless both a GO-only development-pass
freeze and a separate immutable future-audit authorization already exist.  It
uses the same root-generation policy, Lambda model, candidate/search teacher,
and runtime closure as development while emitting a distinct audit row and
provenance schema.  Importing this module never opens the audit population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import action_key
from .ai_profiles import ModelPaths, build_policy, load_model_bundle
from .generate_hu_m4_t1_data import _profile_policy_seed, generate_t1_second_root
from .hu_m43_attempt06_teacher import _require_concrete_stage9f_p2_policies
from .hu_m43_attempt08_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT08_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT08_PLAN_SHA256,
    M43_ATTEMPT08_PROFILES,
    enumerate_attempt08_seed_schedules,
    load_and_validate_attempt08_plan,
)
from .hu_m43_attempt08_runtime_anchor_contract import (
    ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
)
from .hu_m43_attempt08_teacher import (
    ATTEMPT08_TEACHER_SCHEMA,
    Attempt08TeacherConfig,
    FrozenAttempt08LambdaRanker,
    evaluate_attempt08_t1_second,
    validate_attempt08_teacher_output,
)
from .play_ai import _choose_from_observation, _hand_decision_seed
from .run_hu_m43_attempt08_development import (
    ATTEMPT08_BASELINE_PROFILE,
    ATTEMPT08_CONTINUATION_PROFILE,
    ATTEMPT08_NATIVE_BATCH_THREADS,
    _native_batch_threads,
    _process_peak_rss_bytes,
)
from .select_hu_m43_attempt08_development import (
    ATTEMPT08_DEVELOPMENT_DECISION_SCHEMA,
)


ATTEMPT08_DEVELOPMENT_PASS_FREEZE_SCHEMA = (
    "hu_m43_attempt08_development_pass_freeze_v1"
)
ATTEMPT08_FUTURE_AUDIT_OPEN_AUTHORIZATION_SCHEMA = (
    "hu_m43_attempt08_future_audit_open_authorization_v1"
)
ATTEMPT08_FUTURE_AUDIT_ROW_SCHEMA = "hu_m43_attempt08_future_audit_shard_row_v1"
ATTEMPT08_FUTURE_AUDIT_PROVENANCE_SCHEMA = (
    "hu_m43_attempt08_future_audit_provenance_v1"
)
ATTEMPT08_FUTURE_AUDIT_ROOT_FIRST = 200
ATTEMPT08_FUTURE_AUDIT_ROOT_LAST = 249
ATTEMPT08_FUTURE_AUDIT_ROOTS = 50
ATTEMPT08_FUTURE_AUDIT_POPULATION = "future_audit"
ATTEMPT08_FUTURE_AUDIT_CORE = (
    "same_attempt08_t1_second_root_generation_lambda_candidate_and_teacher_core"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_PATH = _REPO_ROOT / "configs/hu_joint_policy_m43_attempt08.json"
DEFAULT_AI_PROFILES_PATH = _REPO_ROOT / "src/ofc_regular/ai_profiles.py"
_SEED_DOMAINS = ("hand", "rerank", "veto", "stress", "assessment", "child")

_FREEZE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "package_manifest_sha256",
        "launch_authorization_sha256",
        "development_decision_sha256",
        "selector_receipt_sha256",
        "development_open_authorization_sha256",
        "plan_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "source_closure_sha256",
        "source_zip_sha256",
        "runtime_semantic_anchor_sha256",
        "search_core",
        "search_freeze_authorized",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "development_pass_freeze_sha256",
        "package_manifest_sha256",
        "launch_authorization_sha256",
        "development_decision_sha256",
        "selector_receipt_sha256",
        "development_open_authorization_sha256",
        "plan_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "source_closure_sha256",
        "source_zip_sha256",
        "runtime_semantic_anchor_sha256",
        "population",
        "root_index_first",
        "root_index_last",
        "total_roots",
        "search_core",
        "development_passed",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_SELECTOR_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "selector_claim_sha256",
        "remote_selector_claim_sha256",
        "execution_marker_sha256",
        "merge_receipt_sha256",
        "merged_sha256",
        "selector_source_sha256",
        "decision_sha256",
        "decision",
        "search_freeze_authorized",
        "gate_evaluation_count",
        "selector_executed",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _load_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {source}") from exc
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise ValueError(f"{label} must be one canonical JSON mapping")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _atomic_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"immutable file concurrently created: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _validate_go_decision(
    decision: Mapping[str, Any], *, run_name: str, decision_sha256: str
) -> None:
    source = decision.get("source")
    contract = decision.get("decision_contract")
    science = decision.get("science_boundary")
    if (
        decision.get("schema") != ATTEMPT08_DEVELOPMENT_DECISION_SCHEMA
        or decision.get("status") != "go_write_separate_search_freeze_only"
        or decision.get("decision") != "go"
        or decision.get("search_freeze_authorized") is not True
        or not isinstance(source, Mapping)
        or source.get("run_name") != run_name
        or source.get("plan_sha256") != M43_ATTEMPT08_PLAN_SHA256
        or not isinstance(contract, Mapping)
        or contract.get("gate_evaluation_count") != 1
        or contract.get("all_gates_required") is not True
        or not isinstance(science, Mapping)
        or science.get("future_audit_directly_authorized") is not False
        or science.get("future_audit_opened") is not False
        or science.get("fit_performed") is not False
        or science.get("threshold_selected") is not False
        or science.get("runtime_policy_activated") is not False
        or science.get("current_profile_mutated") is not False
        or not all(
            isinstance(gate, Mapping) and gate.get("passed") is True
            for gate in decision.get("gates", ())
        )
        or not decision.get("gates")
    ):
        raise ValueError("Attempt08 development decision is not a frozen GO")
    _require_sha256(decision_sha256, "development decision SHA-256")


def _validate_selector_receipt(
    receipt: Mapping[str, Any], *, run_name: str, decision_sha256: str
) -> None:
    if (
        set(receipt) != _SELECTOR_RECEIPT_KEYS
        or receipt.get("schema")
        != "hu_m43_attempt08_development_selector_receipt_v1"
        or receipt.get("status") != "single_frozen_gate_evaluation_complete"
        or receipt.get("run_name") != run_name
        or receipt.get("decision_sha256") != decision_sha256
        or receipt.get("decision") != "go"
        or receipt.get("search_freeze_authorized") is not True
        or receipt.get("gate_evaluation_count") != 1
        or receipt.get("selector_executed") is not True
        or any(
            receipt.get(field) is not False
            for field in (
                "future_audit_authorized",
                "fit_performed",
                "threshold_selected",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Attempt08 selector receipt cannot freeze development GO")
    for field in (
        "selector_claim_sha256",
        "remote_selector_claim_sha256",
        "execution_marker_sha256",
        "merge_receipt_sha256",
        "merged_sha256",
        "selector_source_sha256",
        "decision_sha256",
    ):
        _require_sha256(receipt.get(field), f"selector receipt {field}")


def create_development_pass_freeze(
    *,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
    development_decision_path: str | Path,
    selector_receipt_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Create the immutable GO freeze; this does not authorize audit roots."""

    from .hu_m43_attempt08_spot import (
        validate_launch_authorization,
        validate_package,
    )

    root = Path(run_dir)
    manifest = validate_package(root)
    validate_launch_authorization(launch_authorization_path, run_dir=root)
    decision = _load_canonical(development_decision_path, "Attempt08 decision")
    receipt = _load_canonical(selector_receipt_path, "Attempt08 selector receipt")
    decision_sha = _sha256_file(development_decision_path)
    _validate_go_decision(
        decision, run_name=str(manifest["run_name"]), decision_sha256=decision_sha
    )
    _validate_selector_receipt(
        receipt, run_name=str(manifest["run_name"]), decision_sha256=decision_sha
    )
    if _sha256_file(selector_receipt_path) == decision_sha:
        raise ValueError("Attempt08 decision and selector receipt must be distinct")
    payload = {
        "schema": ATTEMPT08_DEVELOPMENT_PASS_FREEZE_SCHEMA,
        "status": "development_go_frozen_without_future_audit_authorization",
        "run_name": manifest["run_name"],
        "package_manifest_sha256": _sha256_file(root / "manifest.json"),
        "launch_authorization_sha256": _sha256_file(launch_authorization_path),
        "development_decision_sha256": decision_sha,
        "selector_receipt_sha256": _sha256_file(selector_receipt_path),
        "development_open_authorization_sha256": manifest[
            "development_open_authorization_sha256"
        ],
        "plan_sha256": manifest["plan_sha256"],
        "model_sha256": manifest["model_sha256"],
        "ai_profiles_sha256": manifest["ai_profiles_sha256"],
        "source_closure_sha256": manifest["source_closure_sha256"],
        "source_zip_sha256": manifest["source_zip_sha256"],
        "runtime_semantic_anchor_sha256": manifest[
            "runtime_semantic_anchor_sha256"
        ],
        "search_core": ATTEMPT08_FUTURE_AUDIT_CORE,
        "search_freeze_authorized": True,
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if set(payload) != _FREEZE_KEYS:
        raise AssertionError("Attempt08 development-pass freeze schema changed")
    _atomic_create(Path(output), _canonical_json_bytes(payload))
    return payload


def load_and_validate_future_audit_open_authorization(
    authorization_path: str | Path,
    *,
    development_pass_freeze_path: str | Path,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
    development_decision_path: str | Path,
    selector_receipt_path: str | Path,
) -> dict[str, Any]:
    """Validate all audit-open prerequisites before any audit seed or model read."""

    # Authorization and freeze are deliberately opened before the package, plan,
    # seeds, model, output, checkpoint, policy, observation, or teacher.
    authorization = _load_canonical(
        authorization_path, "Attempt08 future-audit authorization"
    )
    freeze = _load_canonical(
        development_pass_freeze_path, "Attempt08 development-pass freeze"
    )
    if set(authorization) != _AUTHORIZATION_KEYS or set(freeze) != _FREEZE_KEYS:
        raise ValueError("Attempt08 future-audit authorization fields changed")
    if (
        authorization.get("schema")
        != ATTEMPT08_FUTURE_AUDIT_OPEN_AUTHORIZATION_SCHEMA
        or authorization.get("status")
        != "separately_authorized_after_immutable_development_pass_freeze"
        or authorization.get("development_pass_freeze_sha256")
        != _sha256_file(development_pass_freeze_path)
        or authorization.get("population") != ATTEMPT08_FUTURE_AUDIT_POPULATION
        or authorization.get("root_index_first")
        != ATTEMPT08_FUTURE_AUDIT_ROOT_FIRST
        or authorization.get("root_index_last") != ATTEMPT08_FUTURE_AUDIT_ROOT_LAST
        or authorization.get("total_roots") != ATTEMPT08_FUTURE_AUDIT_ROOTS
        or authorization.get("search_core") != ATTEMPT08_FUTURE_AUDIT_CORE
        or authorization.get("development_passed") is not True
        or authorization.get("future_audit_authorized") is not True
        or any(
            authorization.get(field) is not False
            for field in (
                "fit_performed",
                "threshold_selected",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
        or freeze.get("schema") != ATTEMPT08_DEVELOPMENT_PASS_FREEZE_SCHEMA
        or freeze.get("status")
        != "development_go_frozen_without_future_audit_authorization"
        or freeze.get("search_freeze_authorized") is not True
        or freeze.get("future_audit_authorized") is not False
        or freeze.get("search_core") != ATTEMPT08_FUTURE_AUDIT_CORE
    ):
        raise ValueError("Attempt08 future audit is not separately authorized")

    from .hu_m43_attempt08_spot import (
        validate_launch_authorization,
        validate_package,
    )

    root = Path(run_dir)
    manifest = validate_package(root)
    validate_launch_authorization(launch_authorization_path, run_dir=root)
    decision = _load_canonical(development_decision_path, "Attempt08 decision")
    receipt = _load_canonical(selector_receipt_path, "Attempt08 selector receipt")
    decision_sha = _sha256_file(development_decision_path)
    _validate_go_decision(
        decision, run_name=str(manifest["run_name"]), decision_sha256=decision_sha
    )
    _validate_selector_receipt(
        receipt, run_name=str(manifest["run_name"]), decision_sha256=decision_sha
    )
    expected_bindings = {
        "run_name": manifest["run_name"],
        "package_manifest_sha256": _sha256_file(root / "manifest.json"),
        "launch_authorization_sha256": _sha256_file(launch_authorization_path),
        "development_decision_sha256": decision_sha,
        "selector_receipt_sha256": _sha256_file(selector_receipt_path),
        "development_open_authorization_sha256": manifest[
            "development_open_authorization_sha256"
        ],
        "plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
        "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "source_closure_sha256": manifest["source_closure_sha256"],
        "source_zip_sha256": manifest["source_zip_sha256"],
        "runtime_semantic_anchor_sha256": ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
    }
    if any(
        authorization.get(field) != value or freeze.get(field) != value
        for field, value in expected_bindings.items()
    ):
        raise ValueError("Attempt08 future-audit authorization closure changed")
    for field in _AUTHORIZATION_KEYS:
        if field.endswith("_sha256"):
            _require_sha256(authorization.get(field), f"audit authorization {field}")
    return authorization


def _future_audit_seeds(plan: Mapping[str, Any], root_index: int) -> dict[str, int]:
    if type(root_index) is not int or not (
        ATTEMPT08_FUTURE_AUDIT_ROOT_FIRST
        <= root_index
        <= ATTEMPT08_FUTURE_AUDIT_ROOT_LAST
    ):
        raise ValueError("Attempt08 future-audit root_index is outside 200..249")
    schedules = enumerate_attempt08_seed_schedules(
        plan, population=ATTEMPT08_FUTURE_AUDIT_POPULATION
    )
    if set(schedules) != set(_SEED_DOMAINS):
        raise ValueError("Attempt08 future-audit seed domains changed")
    local_index = root_index - ATTEMPT08_FUTURE_AUDIT_ROOT_FIRST
    seeds = {domain: int(schedules[domain][local_index]) for domain in _SEED_DOMAINS}
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("Attempt08 future-audit per-root seed domains overlap")
    return seeds


def run_future_audit_shard(
    *,
    root_index: int,
    output: str | Path,
    run_id: str,
    audit_open_authorization: str | Path,
    development_pass_freeze: str | Path,
    run_dir: str | Path,
    launch_authorization: str | Path,
    development_decision: str | Path,
    selector_receipt: str | Path,
    model: str | Path,
    model_sha256: str,
    plan: str | Path = DEFAULT_PLAN_PATH,
    ai_profiles: str | Path = DEFAULT_AI_PROFILES_PATH,
    paths: ModelPaths | None = None,
) -> dict[str, Any]:
    """Generate one audit-only row after the separate authorization chain."""

    authorization = load_and_validate_future_audit_open_authorization(
        audit_open_authorization,
        development_pass_freeze_path=development_pass_freeze,
        run_dir=run_dir,
        launch_authorization_path=launch_authorization,
        development_decision_path=development_decision,
        selector_receipt_path=selector_receipt,
    )
    if type(root_index) is not int or not (
        ATTEMPT08_FUTURE_AUDIT_ROOT_FIRST
        <= root_index
        <= ATTEMPT08_FUTURE_AUDIT_ROOT_LAST
    ):
        raise ValueError("Attempt08 future-audit root_index is outside 200..249")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Attempt08 future-audit run_id must not be empty")
    plan_path = Path(plan)
    ai_profiles_path = Path(ai_profiles)
    model_path = Path(model)
    if _sha256_file(plan_path) != M43_ATTEMPT08_PLAN_SHA256:
        raise ValueError("Attempt08 future-audit plan changed")
    if _sha256_file(ai_profiles_path) != AI_PROFILES_SHA256:
        raise ValueError("Attempt08 future-audit ai_profiles.py changed")
    if model_sha256 != ATTEMPT08_LAMBDA_MODEL_SHA256:
        raise ValueError("Attempt08 future-audit model identity changed")
    if _sha256_file(model_path) != ATTEMPT08_LAMBDA_MODEL_SHA256:
        raise ValueError("Attempt08 future-audit model bytes changed")
    plan_payload = load_and_validate_attempt08_plan(plan_path)
    seeds = _future_audit_seeds(plan_payload, root_index)
    root_profile = M43_ATTEMPT08_PROFILES[root_index % len(M43_ATTEMPT08_PROFILES)]
    explicit_profiles = {
        root_profile,
        ATTEMPT08_BASELINE_PROFILE,
        ATTEMPT08_CONTINUATION_PROFILE,
    }
    if "current" in explicit_profiles:
        raise AssertionError("Attempt08 future audit must never resolve current")
    started = time.perf_counter()
    bundle = load_model_bundle(paths or ModelPaths(), profiles=explicit_profiles)
    root_policies = {
        seat: build_policy(
            root_profile,
            bundle,
            seed=_profile_policy_seed(seeds["hand"], root_profile, seat),
            seat=seat,
            opening_lookahead_samples=0,
        )
        for seat in ("first", "second")
    }
    observation = generate_t1_second_root(seeds["hand"], root_policies=root_policies)
    baseline_policy = build_policy(
        ATTEMPT08_BASELINE_PROFILE,
        bundle,
        seed=_profile_policy_seed(seeds["hand"], ATTEMPT08_BASELINE_PROFILE, "second"),
        seat="second",
        opening_lookahead_samples=0,
    )
    baseline_action = _choose_from_observation(
        baseline_policy,
        observation,
        hand_id=seeds["hand"],
        game_id=seeds["hand"],
        decision_seed=_hand_decision_seed(
            base_seed=seeds["hand"], observation=observation
        ),
    )
    baseline_token = action_key(baseline_action).to_token()
    t2_policies = {
        seat: build_policy(
            ATTEMPT08_CONTINUATION_PROFILE,
            bundle,
            seed=seeds["child"] + (1 if seat == "second" else 0),
            seat=seat,
            opening_lookahead_samples=0,
        )
        for seat in ("first", "second")
    }
    _require_concrete_stage9f_p2_policies(t2_policies)
    ranker = FrozenAttempt08LambdaRanker.load(
        model_path, expected_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256
    )
    teacher_config = Attempt08TeacherConfig(
        frozen_model_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256,
        hand_seed=seeds["hand"],
        rerank_seed=seeds["rerank"],
        veto_seed=seeds["veto"],
        stress_seed=seeds["stress"],
        assessment_seed=seeds["assessment"],
        child_policy_seed=seeds["child"],
        run_id=(
            f"{run_id}:root={root_index}:seed={seeds['hand']}:"
            f"obs={observation.fingerprint()}"
        ),
        batch_child_selectors=True,
    )
    with _native_batch_threads(True, ATTEMPT08_NATIVE_BATCH_THREADS):
        teacher = evaluate_attempt08_t1_second(
            observation,
            baseline_action_key=baseline_token,
            ranker=ranker,
            t2_policies=t2_policies,
            config=teacher_config,
        )
    validate_attempt08_teacher_output(
        observation,
        baseline_action_key=baseline_token,
        payload=teacher,
        config=teacher_config,
    )
    if (
        teacher.get("schema") != ATTEMPT08_TEACHER_SCHEMA
        or teacher.get("status") != "ok"
        or teacher.get("teacher_value_status") != "diagnostic_not_match_EV"
        or teacher.get("runtime_gate_allowed") is not False
        or teacher.get("profile_activation_allowed") is not False
        or teacher.get("current_profile_resolved") is not False
    ):
        raise ValueError("Attempt08 future-audit teacher identity changed")
    provenance = {
        "schema": ATTEMPT08_FUTURE_AUDIT_PROVENANCE_SCHEMA,
        "population": ATTEMPT08_FUTURE_AUDIT_POPULATION,
        "run_id": run_id,
        "root_index": root_index,
        "root_profile": root_profile,
        "seeds": seeds,
        "audit_open_authorization_sha256": _sha256_file(
            audit_open_authorization
        ),
        "development_pass_freeze_sha256": authorization[
            "development_pass_freeze_sha256"
        ],
        "package_manifest_sha256": authorization["package_manifest_sha256"],
        "source_closure_sha256": authorization["source_closure_sha256"],
        "source_zip_sha256": authorization["source_zip_sha256"],
        "runtime_semantic_anchor_sha256": authorization[
            "runtime_semantic_anchor_sha256"
        ],
        "plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
        "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "search_core": ATTEMPT08_FUTURE_AUDIT_CORE,
        "baseline_profile": ATTEMPT08_BASELINE_PROFILE,
        "continuation_profile": ATTEMPT08_CONTINUATION_PROFILE,
        "current_profile_resolved": False,
        "opponent_private_discard_input_allowed": False,
        "teacher_values_are_realized_match_ev": False,
        "future_audit_only": True,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
    }
    row = {
        "schema": ATTEMPT08_FUTURE_AUDIT_ROW_SCHEMA,
        "population": ATTEMPT08_FUTURE_AUDIT_POPULATION,
        "root_index": root_index,
        "audit_local_index": root_index - ATTEMPT08_FUTURE_AUDIT_ROOT_FIRST,
        "hand_seed": seeds["hand"],
        "root_profile": root_profile,
        "policy_observation": observation.to_dict(),
        "baseline_action_key": baseline_token,
        "provenance": provenance,
        "teacher": teacher,
    }
    encoded = _canonical_json_bytes(row)
    _atomic_create(Path(output), encoded)
    return {
        "schema": "hu_m43_attempt08_future_audit_summary_v1",
        "status": "complete",
        "population": ATTEMPT08_FUTURE_AUDIT_POPULATION,
        "root_index": root_index,
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "generator_elapsed_seconds": time.perf_counter() - started,
        "generator_peak_rss_bytes": _process_peak_rss_bytes(),
        "teacher_values_are_realized_match_ev": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-development-pass")
    freeze.add_argument("--run-dir", type=Path, required=True)
    freeze.add_argument("--launch-authorization", type=Path, required=True)
    freeze.add_argument("--development-decision", type=Path, required=True)
    freeze.add_argument("--selector-receipt", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root-index", type=int, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--audit-open-authorization", type=Path, required=True)
    run.add_argument("--development-pass-freeze", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--launch-authorization", type=Path, required=True)
    run.add_argument("--development-decision", type=Path, required=True)
    run.add_argument("--selector-receipt", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--model-sha256", required=True)
    run.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    run.add_argument("--ai-profiles", type=Path, default=DEFAULT_AI_PROFILES_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "freeze-development-pass":
        result = create_development_pass_freeze(
            run_dir=args.run_dir,
            launch_authorization_path=args.launch_authorization,
            development_decision_path=args.development_decision,
            selector_receipt_path=args.selector_receipt,
            output=args.output,
        )
    else:
        result = run_future_audit_shard(
            root_index=args.root_index,
            output=args.output,
            run_id=args.run_id,
            audit_open_authorization=args.audit_open_authorization,
            development_pass_freeze=args.development_pass_freeze,
            run_dir=args.run_dir,
            launch_authorization=args.launch_authorization,
            development_decision=args.development_decision,
            selector_receipt=args.selector_receipt,
            model=args.model,
            model_sha256=args.model_sha256,
            plan=args.plan,
            ai_profiles=args.ai_profiles,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ATTEMPT08_DEVELOPMENT_PASS_FREEZE_SCHEMA",
    "ATTEMPT08_FUTURE_AUDIT_OPEN_AUTHORIZATION_SCHEMA",
    "ATTEMPT08_FUTURE_AUDIT_PROVENANCE_SCHEMA",
    "ATTEMPT08_FUTURE_AUDIT_ROOT_FIRST",
    "ATTEMPT08_FUTURE_AUDIT_ROOT_LAST",
    "ATTEMPT08_FUTURE_AUDIT_ROOTS",
    "ATTEMPT08_FUTURE_AUDIT_ROW_SCHEMA",
    "create_development_pass_freeze",
    "load_and_validate_future_audit_open_authorization",
    "run_future_audit_shard",
]
