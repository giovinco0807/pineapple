"""Run one restart-safe Attempt09 T1-second search root.

The runner is intentionally small.  Search semantics live in
``hu_m43_attempt09_teacher`` and seed semantics live in the frozen Attempt09
plan.  Preflight roots may run without an authorization artifact; development
and future-audit roots require a hash-bound, fail-closed authorization created
after the preceding gate has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import action_key
from .action_space import generate_turn_actions
from .ai_profiles import ModelPaths, build_policy, load_model_bundle
from .generate_hu_m4_t1_data import _profile_policy_seed, generate_t1_second_root
from .hu_infoset import ActorObservation
from .hu_m43_attempt06_teacher import _require_concrete_stage9f_p2_policies
from .hu_m43_attempt09_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT09_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT09_PLAN_SHA256,
    M43_ATTEMPT09_PROFILES,
    enumerate_attempt09_seed_schedules,
    load_and_validate_attempt09_plan,
    validate_attempt09_artifact_bindings,
)
from .hu_m43_attempt09_teacher import (
    ATTEMPT09_TEACHER_SCHEMA,
    Attempt09TeacherConfig,
    FrozenAttempt09LambdaRanker,
    evaluate_attempt09_t1_second,
    validate_attempt09_teacher_output,
)
from .play_ai import _choose_from_observation, _hand_decision_seed
from .run_hu_m43_attempt08_development import (
    _HeartbeatPump,
    _ShardFileLock,
    _atomic_json,
    _native_batch_threads,
    _process_peak_rss_bytes,
    _require_distinct_paths,
    _sha256_file,
)


ATTEMPT09_ROW_SCHEMA = "hu_m43_attempt09_search_root_v1"
ATTEMPT09_CHECKPOINT_SCHEMA = "hu_m43_attempt09_search_checkpoint_v1"
ATTEMPT09_HEARTBEAT_SCHEMA = "hu_m43_attempt09_search_heartbeat_v1"
ATTEMPT09_SUMMARY_SCHEMA = "hu_m43_attempt09_search_summary_v1"
ATTEMPT09_AUTHORIZATION_SCHEMA = "hu_m43_attempt09_execution_authorization_v1"
ATTEMPT09_ROOT_CONTRACT_SCHEMA = "hu_m43_attempt09_root_contract_v1"
ATTEMPT09_BASELINE_PROFILE = "stage18_p1"
ATTEMPT09_CONTINUATION_PROFILE = "stage9f_p2"
ATTEMPT09_NATIVE_BATCH_THREADS = 4

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt09.json"
DEFAULT_AI_PROFILES = _REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
_MODES = ("preflight", "development", "future_audit")
_SEED_DOMAINS = (
    "hand",
    "rerank",
    "veto",
    "stress",
    "confirmation",
    "evaluation",
    "child",
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _load_canonical_mapping(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _authorization(
    path: str | Path | None,
    *,
    mode: str,
    source_package_sha256: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if mode == "preflight":
        if path is not None:
            raise ValueError("Attempt09 preflight must not consume later authorization")
        return None, None
    if path is None:
        raise ValueError(f"Attempt09 {mode} requires explicit authorization")
    target = Path(path)
    payload = _load_canonical_mapping(target, "Attempt09 execution authorization")
    expected_keys = {
        "schema",
        "status",
        "mode",
        "plan_sha256",
        "source_package_sha256",
        "preceding_gate_artifact",
        "preceding_gate_sha256",
        "root_index_first",
        "root_index_last",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
    if set(payload) != expected_keys:
        raise ValueError("Attempt09 authorization fields changed")
    expected_range = (0, 199) if mode == "development" else (200, 249)
    if (
        payload.get("schema") != ATTEMPT09_AUTHORIZATION_SCHEMA
        or payload.get("status") != "authorized"
        or payload.get("mode") != mode
        or payload.get("plan_sha256") != M43_ATTEMPT09_PLAN_SHA256
        or payload.get("root_index_first") != expected_range[0]
        or payload.get("root_index_last") != expected_range[1]
        or payload.get("current_profile_mutated") is not False
        or payload.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt09 authorization boundary changed")
    package_hash = payload.get("source_package_sha256")
    if not _is_sha256(package_hash) or source_package_sha256 != package_hash:
        raise ValueError("Attempt09 source package authorization changed")
    gate_path = Path(str(payload.get("preceding_gate_artifact")))
    gate_hash = payload.get("preceding_gate_sha256")
    if not gate_path.is_file() or not _is_sha256(gate_hash):
        raise ValueError("Attempt09 preceding gate artifact is missing")
    if _sha256_file(gate_path) != gate_hash:
        raise ValueError("Attempt09 preceding gate artifact hash changed")
    return payload, _sha256_file(target)


def _seed_values(
    plan: Mapping[str, Any], *, mode: str, root_index: int
) -> dict[str, int]:
    if type(root_index) is not int:
        raise TypeError("Attempt09 root_index must be an integer")
    if mode == "preflight":
        declared = plan["preflight_seed_contract"]["source_root_indices"]
        if root_index not in declared:
            raise ValueError("Attempt09 preflight root must be one of 0, 1, 2")
        offset = declared.index(root_index)
    elif mode == "development":
        if not 0 <= root_index <= 199:
            raise ValueError("Attempt09 development root must be in 0..199")
        offset = root_index
    elif mode == "future_audit":
        if not 200 <= root_index <= 249:
            raise ValueError("Attempt09 future-audit root must be in 200..249")
        offset = root_index - 200
    else:
        raise ValueError(f"unknown Attempt09 mode: {mode}")
    schedules = enumerate_attempt09_seed_schedules(plan, population=mode)
    if tuple(schedules) != _SEED_DOMAINS:
        raise ValueError("Attempt09 seed-domain order changed")
    result = {name: int(values[offset]) for name, values in schedules.items()}
    if len(set(result.values())) != len(result):
        raise ValueError("Attempt09 per-root seed domains overlap")
    return result


def _reject_hidden(value: Any, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized == "opponent_private_discard_input_allowed":
                if child is not False:
                    raise ValueError(
                        f"hidden opponent discard enabled at {path}.{key}"
                    )
            elif (
                "opponent" in normalized
                and "discard" in normalized
                and ("private" in normalized or "hidden" in normalized)
            ):
                raise ValueError(f"hidden opponent discard leaked at {path}.{key}")
            _reject_hidden(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_hidden(child, f"{path}[{index}]")


def _write_atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"Attempt09 output already exists: {path}")
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    *,
    config_sha256: str,
    root_index: int,
    run_id: str,
) -> dict[str, Any]:
    checkpoint = _load_canonical_mapping(path, "Attempt09 checkpoint")
    common = {
        "schema": ATTEMPT09_CHECKPOINT_SCHEMA,
        "config_sha256": config_sha256,
        "root_index": root_index,
        "run_id": run_id,
    }
    if any(checkpoint.get(key) != value for key, value in common.items()):
        raise ValueError("Attempt09 checkpoint belongs to another root contract")
    updated = checkpoint.get("updated_unix_seconds")
    if (
        isinstance(updated, bool)
        or not isinstance(updated, (int, float))
        or not math.isfinite(float(updated))
        or float(updated) < 0.0
    ):
        raise ValueError("Attempt09 checkpoint update time is invalid")
    status = checkpoint.get("status")
    running_keys = {*common, "status", "updated_unix_seconds"}
    complete_keys = {*running_keys, "output_sha256"}
    if status == "running":
        if set(checkpoint) != running_keys:
            raise ValueError("Attempt09 running checkpoint fields changed")
    elif status == "complete":
        if set(checkpoint) != complete_keys or not _is_sha256(
            checkpoint.get("output_sha256")
        ):
            raise ValueError("Attempt09 complete checkpoint fields changed")
    else:
        raise ValueError("Attempt09 checkpoint status changed")
    return checkpoint


def _validate_completed_row(
    raw: bytes,
    *,
    fixed: Mapping[str, Any],
    config_sha256: str,
    baseline_profile: str,
    continuation_profile: str,
) -> dict[str, Any]:
    rows = raw.splitlines()
    if len(rows) != 1 or raw != rows[0] + b"\n":
        raise ValueError("Attempt09 completed output is not one canonical row")
    try:
        row = json.loads(rows[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt09 completed row is invalid") from exc
    expected_keys = {
        "schema",
        "root_index",
        "hand_seed",
        "root_profile",
        "policy_observation",
        "baseline_action_key",
        "provenance",
        "teacher",
    }
    if (
        not isinstance(row, dict)
        or raw != _canonical_json_bytes(row)
        or set(row) != expected_keys
    ):
        raise ValueError("Attempt09 completed row is not canonical")
    provenance = row.get("provenance")
    teacher = row.get("teacher")
    if not isinstance(provenance, Mapping) or not isinstance(teacher, Mapping):
        raise ValueError("Attempt09 completed row payload changed")
    expected_provenance_keys = {
        *fixed,
        "config_sha256",
        "root_policy_profiles",
        "root_policy_seeds",
        "baseline_profile",
        "baseline_policy_seed",
        "continuation_profile",
        "continuation_policy_seeds",
        "explicit_loaded_profiles",
        "opponent_private_discard_input_allowed",
        "opponent_profile_runtime_feature_allowed",
        "teacher_values_are_realized_match_ev",
        "teacher_ev_or_lcb_runtime_gate_allowed",
        "development_only",
        "fit_allowed",
        "threshold_selection_allowed",
        "runtime_activation_allowed",
        "current_profile_resolved",
        "elapsed_seconds",
        "peak_rss_bytes",
    }
    if set(provenance) != expected_provenance_keys:
        raise ValueError("Attempt09 completed provenance fields changed")
    for key, value in fixed.items():
        if provenance.get(key) != value:
            raise ValueError(f"Attempt09 completed provenance {key} changed")
    observation_payload = row.get("policy_observation")
    if not isinstance(observation_payload, Mapping):
        raise ValueError("Attempt09 completed observation is missing")
    observation = ActorObservation.from_dict(observation_payload)
    if observation.to_dict() != dict(observation_payload):
        raise ValueError("Attempt09 completed observation changed")
    baseline_token = row.get("baseline_action_key")
    legal_baseline_matches = [
        action
        for action in generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )
        if action_key(action).to_token() == baseline_token
    ]
    expected_profiles = sorted(
        {str(fixed["root_profile"]), baseline_profile, continuation_profile}
    )
    if (
        row.get("schema") != ATTEMPT09_ROW_SCHEMA
        or row.get("root_index") != fixed["root_index"]
        or row.get("hand_seed") != fixed["seeds"]["hand"]
        or row.get("root_profile") != fixed["root_profile"]
        or not isinstance(baseline_token, str)
        or len(legal_baseline_matches) != 1
        or provenance.get("config_sha256") != config_sha256
        or provenance.get("root_policy_profiles")
        != {"first": fixed["root_profile"], "second": fixed["root_profile"]}
        or provenance.get("baseline_profile") != baseline_profile
        or provenance.get("continuation_profile") != continuation_profile
        or provenance.get("explicit_loaded_profiles") != expected_profiles
        or "current" in expected_profiles
        or provenance.get("opponent_private_discard_input_allowed") is not False
        or provenance.get("opponent_profile_runtime_feature_allowed") is not False
        or provenance.get("teacher_values_are_realized_match_ev") is not False
        or provenance.get("teacher_ev_or_lcb_runtime_gate_allowed") is not False
        or provenance.get("development_only") is not True
        or provenance.get("fit_allowed") is not False
        or provenance.get("threshold_selection_allowed") is not False
        or provenance.get("runtime_activation_allowed") is not False
        or provenance.get("current_profile_resolved") is not False
        or teacher.get("status") != "ok"
        or teacher.get("schema") != ATTEMPT09_TEACHER_SCHEMA
        or teacher.get("policy_observation") != row.get("policy_observation")
        or teacher.get("baseline_action_key") != row.get("baseline_action_key")
        or teacher.get("observation_fingerprint") != observation.fingerprint()
        or teacher.get("runtime_gate_allowed") is not False
        or teacher.get("profile_activation_allowed") is not False
        or teacher.get("current_profile_resolved") is not False
        or teacher.get("development_only") is not True
    ):
        raise ValueError("Attempt09 completed row identity changed")
    elapsed = provenance.get("elapsed_seconds")
    peak_rss = provenance.get("peak_rss_bytes")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
        or type(peak_rss) is not int
        or peak_rss < 0
    ):
        raise ValueError("Attempt09 completed generator metrics changed")
    _reject_hidden(row)
    return row


def run_attempt09_root(
    *,
    mode: str,
    root_index: int,
    output: str | Path,
    checkpoint: str | Path,
    heartbeat: str | Path,
    model: str | Path,
    model_sha256: str,
    run_id: str,
    plan: str | Path = DEFAULT_PLAN,
    ai_profiles: str | Path = DEFAULT_AI_PROFILES,
    authorization: str | Path | None = None,
    source_package_sha256: str | None = None,
    batch_child_selectors: bool = True,
    native_batch_threads: int = ATTEMPT09_NATIVE_BATCH_THREADS,
    paths: ModelPaths | None = None,
) -> dict[str, Any]:
    """Generate or validate exactly one deterministic Attempt09 row."""

    if mode not in _MODES:
        raise ValueError("Attempt09 mode must be preflight, development, or future_audit")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Attempt09 run_id must not be empty")
    if not isinstance(batch_child_selectors, bool):
        raise TypeError("Attempt09 batch_child_selectors must be a bool")
    if native_batch_threads != 4:
        raise ValueError("Attempt09 roots require exactly 4 native batch threads")
    if mode != "preflight" and batch_child_selectors is not True:
        raise ValueError("Attempt09 development/audit roots require batched child selectors")
    output_path = Path(output)
    checkpoint_path = Path(checkpoint)
    heartbeat_path = Path(heartbeat)
    model_path = Path(model)
    plan_path = Path(plan)
    ai_profiles_path = Path(ai_profiles)
    lock_path = output_path.with_name(output_path.name + ".lock")
    protected_paths = {
        "output": output_path,
        "checkpoint": checkpoint_path,
        "heartbeat": heartbeat_path,
        "model": model_path,
        "plan": plan_path,
        "ai_profiles": ai_profiles_path,
        "lock": lock_path,
    }
    if authorization is not None:
        protected_paths["authorization"] = Path(authorization)
    _require_distinct_paths(protected_paths)

    with _ShardFileLock(lock_path):
        plan_payload = load_and_validate_attempt09_plan(plan_path)
        validate_attempt09_artifact_bindings(
            plan_payload, repository_root=plan_path.resolve().parents[1]
        )
        if _sha256_file(plan_path) != M43_ATTEMPT09_PLAN_SHA256:
            raise ValueError("Attempt09 plan hash changed")
        if _sha256_file(ai_profiles_path) != AI_PROFILES_SHA256:
            raise ValueError("Attempt09 ai_profiles.py hash changed")
        if model_sha256 != ATTEMPT09_LAMBDA_MODEL_SHA256:
            raise ValueError("Attempt09 model declaration changed")
        if _sha256_file(model_path) != ATTEMPT09_LAMBDA_MODEL_SHA256:
            raise ValueError("Attempt09 frozen Lambda artifact changed")
        auth_payload, auth_sha256 = _authorization(
            authorization,
            mode=mode,
            source_package_sha256=source_package_sha256,
        )
        seeds = _seed_values(plan_payload, mode=mode, root_index=root_index)
        root_profile = M43_ATTEMPT09_PROFILES[
            root_index % len(M43_ATTEMPT09_PROFILES)
        ]
        fixed = {
            "schema": ATTEMPT09_ROOT_CONTRACT_SCHEMA,
            "mode": mode,
            "root_index": root_index,
            "root_profile": root_profile,
            "run_id": run_id,
            "seeds": seeds,
            "plan_sha256": M43_ATTEMPT09_PLAN_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "model_sha256": ATTEMPT09_LAMBDA_MODEL_SHA256,
            "authorization_sha256": auth_sha256,
            "source_package_sha256": source_package_sha256,
            "batch_child_selectors": batch_child_selectors,
            "native_batch_threads": 4,
        }
        config_sha256 = _canonical_sha256(fixed)
        prior: dict[str, Any] | None = None
        if checkpoint_path.exists():
            prior = _load_checkpoint(
                checkpoint_path,
                config_sha256=config_sha256,
                root_index=root_index,
                run_id=run_id,
            )
            if prior.get("status") == "complete" and not output_path.is_file():
                raise ValueError("Attempt09 complete checkpoint lost its output")
        if output_path.exists():
            if prior is None or prior.get("status") != "complete":
                raise ValueError("Attempt09 completed output has no complete checkpoint")
            raw = output_path.read_bytes()
            if _sha256_file(output_path) != prior.get("output_sha256"):
                raise ValueError("Attempt09 completed output hash disagrees with checkpoint")
            row = _validate_completed_row(
                raw,
                fixed=fixed,
                config_sha256=config_sha256,
                baseline_profile=ATTEMPT09_BASELINE_PROFILE,
                continuation_profile=ATTEMPT09_CONTINUATION_PROFILE,
            )
            elapsed = float(row["provenance"]["elapsed_seconds"])
            peak_rss = int(row["provenance"]["peak_rss_bytes"])
        else:
            started = time.perf_counter()
            start_record = {
                "schema": ATTEMPT09_CHECKPOINT_SCHEMA,
                "status": "running",
                "config_sha256": config_sha256,
                "root_index": root_index,
                "run_id": run_id,
                "updated_unix_seconds": time.time(),
            }
            _atomic_json(checkpoint_path, start_record)
            heartbeat_pump = _HeartbeatPump(
                heartbeat_path,
                {
                    "schema": ATTEMPT09_HEARTBEAT_SCHEMA,
                    "status": "running",
                    "config_sha256": config_sha256,
                    "root_index": root_index,
                    "run_id": run_id,
                },
            )
            heartbeat_pump.start()
            try:
                explicit_profiles = {
                    root_profile,
                    ATTEMPT09_BASELINE_PROFILE,
                    ATTEMPT09_CONTINUATION_PROFILE,
                }
                if "current" in explicit_profiles:
                    raise AssertionError("Attempt09 must never resolve current")
                bundle = load_model_bundle(paths or ModelPaths(), profiles=explicit_profiles)
                root_policy_seeds = {
                    seat: _profile_policy_seed(seeds["hand"], root_profile, seat)
                    for seat in ("first", "second")
                }
                root_policies = {
                    seat: build_policy(
                        root_profile,
                        bundle,
                        seed=root_policy_seeds[seat],
                        seat=seat,
                        opening_lookahead_samples=0,
                    )
                    for seat in ("first", "second")
                }
                observation = generate_t1_second_root(
                    seeds["hand"], root_policies=root_policies
                )
                baseline_policy_seed = _profile_policy_seed(
                    seeds["hand"], ATTEMPT09_BASELINE_PROFILE, "second"
                )
                baseline_policy = build_policy(
                    ATTEMPT09_BASELINE_PROFILE,
                    bundle,
                    seed=baseline_policy_seed,
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
                continuation_policy_seeds = {
                    "first": seeds["child"],
                    "second": seeds["child"] + 1,
                }
                t2_policies = {
                    seat: build_policy(
                        ATTEMPT09_CONTINUATION_PROFILE,
                        bundle,
                        seed=continuation_policy_seeds[seat],
                        seat=seat,
                        opening_lookahead_samples=0,
                    )
                    for seat in ("first", "second")
                }
                _require_concrete_stage9f_p2_policies(t2_policies)
                ranker = FrozenAttempt09LambdaRanker.load(
                    model_path, expected_sha256=ATTEMPT09_LAMBDA_MODEL_SHA256
                )
                teacher_config = Attempt09TeacherConfig(
                    frozen_model_sha256=ATTEMPT09_LAMBDA_MODEL_SHA256,
                    hand_seed=seeds["hand"],
                    rerank_seed=seeds["rerank"],
                    veto_seed=seeds["veto"],
                    stress_seed=seeds["stress"],
                    confirmation_seed=seeds["confirmation"],
                    evaluation_seed=seeds["evaluation"],
                    child_policy_seed=seeds["child"],
                    run_id=(
                        f"{run_id}:root={root_index}:seed={seeds['hand']}:"
                        f"obs={observation.fingerprint()}"
                    ),
                    batch_child_selectors=batch_child_selectors,
                )
                with _native_batch_threads(True, native_batch_threads):
                    teacher = evaluate_attempt09_t1_second(
                        observation,
                        baseline_action_key=baseline_token,
                        ranker=ranker,
                        t2_policies=t2_policies,
                        config=teacher_config,
                    )
                validate_attempt09_teacher_output(
                    observation,
                    baseline_action_key=baseline_token,
                    payload=teacher,
                    config=teacher_config,
                )
            finally:
                heartbeat_pump.stop()
            elapsed = time.perf_counter() - started
            peak_rss = _process_peak_rss_bytes()
            provenance = {
                **fixed,
                "config_sha256": config_sha256,
                "root_policy_profiles": {
                    "first": root_profile,
                    "second": root_profile,
                },
                "root_policy_seeds": root_policy_seeds,
                "baseline_profile": ATTEMPT09_BASELINE_PROFILE,
                "baseline_policy_seed": baseline_policy_seed,
                "continuation_profile": ATTEMPT09_CONTINUATION_PROFILE,
                "continuation_policy_seeds": continuation_policy_seeds,
                "explicit_loaded_profiles": sorted(explicit_profiles),
                "opponent_private_discard_input_allowed": False,
                "opponent_profile_runtime_feature_allowed": False,
                "teacher_values_are_realized_match_ev": False,
                "teacher_ev_or_lcb_runtime_gate_allowed": False,
                "development_only": True,
                "fit_allowed": False,
                "threshold_selection_allowed": False,
                "runtime_activation_allowed": False,
                "current_profile_resolved": False,
                "elapsed_seconds": elapsed,
                "peak_rss_bytes": peak_rss,
            }
            row = {
                "schema": ATTEMPT09_ROW_SCHEMA,
                "root_index": root_index,
                "hand_seed": seeds["hand"],
                "root_profile": root_profile,
                "policy_observation": observation.to_dict(),
                "baseline_action_key": baseline_token,
                "provenance": provenance,
                "teacher": teacher,
            }
            _reject_hidden(row)
            _write_atomic_bytes(output_path, _canonical_json_bytes(row))
            _atomic_json(
                checkpoint_path,
                {
                    "schema": ATTEMPT09_CHECKPOINT_SCHEMA,
                    "status": "complete",
                    "config_sha256": config_sha256,
                    "root_index": root_index,
                    "run_id": run_id,
                    "output_sha256": _sha256_file(output_path),
                    "updated_unix_seconds": time.time(),
                },
            )
        summary = {
            "schema": ATTEMPT09_SUMMARY_SCHEMA,
            "status": "complete",
            "mode": mode,
            "root_index": root_index,
            "root_profile": root_profile,
            "run_id": run_id,
            "config_sha256": config_sha256,
            "output_sha256": _sha256_file(output_path),
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak_rss,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        _atomic_json(
            heartbeat_path,
            {
                "schema": ATTEMPT09_HEARTBEAT_SCHEMA,
                **summary,
                "updated_unix_seconds": time.time(),
            },
        )
        return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--root-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--ai-profiles", type=Path, default=DEFAULT_AI_PROFILES)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--source-package-sha256")
    parser.add_argument("--batch-child-selectors", action="store_true")
    parser.add_argument("--native-batch-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_attempt09_root(
        mode=args.mode,
        root_index=args.root_index,
        output=args.output,
        checkpoint=args.checkpoint,
        heartbeat=args.heartbeat,
        model=args.model,
        model_sha256=args.model_sha256,
        run_id=args.run_id,
        plan=args.plan,
        ai_profiles=args.ai_profiles,
        authorization=args.authorization,
        source_package_sha256=args.source_package_sha256,
        batch_child_selectors=args.batch_child_selectors,
        native_batch_threads=args.native_batch_threads,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ATTEMPT09_AUTHORIZATION_SCHEMA",
    "ATTEMPT09_CHECKPOINT_SCHEMA",
    "ATTEMPT09_HEARTBEAT_SCHEMA",
    "ATTEMPT09_ROOT_CONTRACT_SCHEMA",
    "ATTEMPT09_ROW_SCHEMA",
    "ATTEMPT09_SUMMARY_SCHEMA",
    "run_attempt09_root",
]
