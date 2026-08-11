"""Validate the native M3 HU search engine before any large-scale run.

This module deliberately stays on the bounded, local correctness path.  It
does not load an AI profile, read the ``current`` selector, or invoke a cloud
API.  Fresh T3 information sets are generated from deterministic deck seeds,
written as a versioned JSONL shard, and evaluated by the restart-safe Rust
runner.  Teacher scores are diagnostics; they are not realized match EV and
must not be used as promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .hu_m3_rust import (
    build_native_engine,
    evaluate_t3,
    load_native_engine,
    repository_root,
    runner_path,
    t3_request,
)
from .hu_turn3_joint_exact_teacher import (
    JointExactConfig,
    evaluate_t3_joint_exact_actions,
)
from .validate_hu_m2_teacher import generate_fresh_t3_roots


M3_VALIDATION_SCHEMA = "hu_m3_engine_validation_summary_v1"
M3_PILOT_INPUT_SCHEMA = "hu_m3_engine_pilot_input_v1"
M3_RUNNER_RESULT_SCHEMA = "hu_m3_result_v1"
M3_ENGINE_RESULT_SCHEMA = "hu_m3_engine_result_v1"
TEACHER_VALUE_STATUS = "diagnostic_not_match_EV"
ROOT_GENERATION_POLICY = "fresh_canonical_seed_stride_t3_v1"
ACTION_KEY_RE = re.compile(r"^rak1:(?:[0-9a-f]{13}:){3}[0-9a-f]{13}$")


@dataclass(frozen=True)
class M3PilotConfig:
    roots_100: int = 100
    roots_1000: int = 1_000
    seed_start: int = 2026071301
    seed_stride: int = 1_000_003
    teacher_seed: int = 20260713
    candidate_seed: int = 2026071301
    evaluation_seed: int = 2026071302
    candidate_samples: int = 1
    evaluation_samples: int = 1
    downstream_t3_samples: int = 1
    downstream_t4_samples: int = 1
    deterministic_subset_roots: int = 4
    parity_roots: int = 4
    minimum_speedup: float = 5.0

    def __post_init__(self) -> None:
        for name in ("roots_100", "roots_1000"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0 or value % 2:
                raise ValueError(f"{name} must be a positive even root count")
        if self.roots_1000 < self.roots_100:
            raise ValueError("roots_1000 must be at least roots_100")
        for name in (
            "seed_start",
            "seed_stride",
            "teacher_seed",
            "candidate_seed",
            "evaluation_seed",
            "candidate_samples",
            "evaluation_samples",
            "downstream_t3_samples",
            "downstream_t4_samples",
            "deterministic_subset_roots",
            "parity_roots",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive")
        for name in (
            "candidate_samples",
            "evaluation_samples",
            "downstream_t3_samples",
            "downstream_t4_samples",
        ):
            if getattr(self, name) != 1:
                raise ValueError(f"{name} must be 1 in the bounded M3 pilot")
        if not 1 <= self.deterministic_subset_roots <= 8:
            raise ValueError("deterministic_subset_roots must be between 1 and 8")
        if not 1 <= self.parity_roots <= 4:
            raise ValueError("parity_roots must be between 1 and 4")
        if not math.isfinite(self.minimum_speedup) or self.minimum_speedup <= 0:
            raise ValueError("minimum_speedup must be finite and positive")

    def hand_seed(self, hand_index: int) -> int:
        if hand_index < 0:
            raise IndexError("hand_index must be non-negative")
        return self.seed_start + hand_index * self.seed_stride

    def to_dict(self) -> dict[str, Any]:
        return {
            "roots_100": self.roots_100,
            "roots_1000": self.roots_1000,
            "seed_start": self.seed_start,
            "seed_stride": self.seed_stride,
            "teacher_seed": self.teacher_seed,
            "candidate_seed": self.candidate_seed,
            "evaluation_seed": self.evaluation_seed,
            "candidate_samples": self.candidate_samples,
            "evaluation_samples": self.evaluation_samples,
            "downstream_t3_samples": self.downstream_t3_samples,
            "downstream_t4_samples": self.downstream_t4_samples,
            "deterministic_subset_roots": self.deterministic_subset_roots,
            "parity_roots": self.parity_roots,
            "minimum_speedup": self.minimum_speedup,
        }


@dataclass(frozen=True)
class PilotRoot:
    root_id: str
    root_index: int
    hand_index: int
    hand_seed: int
    seat: str
    fingerprint: str
    request: dict[str, Any]

    def input_row(self) -> dict[str, Any]:
        return {
            "schema_version": M3_PILOT_INPUT_SCHEMA,
            "root_id": self.root_id,
            "request": self.request,
        }


@dataclass(frozen=True)
class PhasePaths:
    input: Path
    output: Path
    checkpoint: Path
    heartbeat: Path

    @property
    def partial(self) -> Path:
        return self.output.with_name(self.output.name + ".partial")


def generate_pilot_roots(
    root_count: int,
    config: M3PilotConfig,
    *,
    run_id: str,
) -> list[PilotRoot]:
    """Generate an exactly balanced, deterministic first/second root set."""

    if isinstance(root_count, bool) or not isinstance(root_count, int):
        raise TypeError("root_count must be an integer")
    if root_count <= 0 or root_count % 2:
        raise ValueError("root_count must be a positive even integer")
    roots: list[PilotRoot] = []
    for hand_index in range(root_count // 2):
        hand_seed = config.hand_seed(hand_index)
        for observation in generate_fresh_t3_roots(hand_seed):
            root_index = len(roots)
            fingerprint = observation.fingerprint()
            root_id = f"root-{root_index:06d}-{observation.seat}-{fingerprint[:16]}"
            root_run_id = (
                f"{run_id}:hand={hand_index}:seed={hand_seed}:root={fingerprint}"
            )
            teacher_config = JointExactConfig(
                candidate_samples=1,
                evaluation_samples=1,
                downstream_t3_samples=1,
                downstream_t4_samples=1,
                seed=config.teacher_seed,
                candidate_seed=config.candidate_seed,
                evaluation_seed=config.evaluation_seed,
                run_id=root_run_id,
                seat=observation.seat,
                to_act_order=observation.to_act_order,
            )
            roots.append(
                PilotRoot(
                    root_id=root_id,
                    root_index=root_index,
                    hand_index=hand_index,
                    hand_seed=hand_seed,
                    seat=observation.seat,
                    fingerprint=fingerprint,
                    request=t3_request(observation, config=teacher_config),
                )
            )
    return roots


def write_runner_input(
    path: str | Path,
    roots: Sequence[PilotRoot],
    *,
    overwrite: bool = False,
) -> str:
    """Atomically write canonical JSONL and return its SHA-256 digest."""

    if not roots:
        raise ValueError("runner input requires at least one root")
    root_ids = [root.root_id for root in roots]
    if len(root_ids) != len(set(root_ids)):
        raise ValueError("runner input root IDs must be unique")
    serialized = "".join(
        json.dumps(
            root.input_row(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        + "\n"
        for root in roots
    ).encode("ascii")
    output = Path(path)
    if output.exists() and not overwrite:
        existing = output.read_bytes()
        if existing != serialized:
            raise FileExistsError(f"runner input exists with different bytes: {output}")
        return hashlib.sha256(existing).hexdigest()
    _atomic_write_bytes(output, serialized)
    return hashlib.sha256(serialized).hexdigest()


def run_parity_benchmark(
    roots: Sequence[PilotRoot],
    *,
    minimum_speedup: float = 5.0,
    library: Any | None = None,
) -> dict[str, Any]:
    """Compare Python and Rust on at most four identical 1/1/1/1 roots."""

    selected = list(roots[:4])
    if not selected:
        raise ValueError("parity benchmark requires at least one root")
    native = library or load_native_engine(build_if_missing=True, release=True)

    records: list[dict[str, Any]] = []
    python_total = 0.0
    rust_total = 0.0
    for root in selected:
        request = root.request
        observation = _observation_from_request(request)
        config = _joint_config_from_request(request)

        started = time.perf_counter()
        python_result = evaluate_t3_joint_exact_actions(
            observation=observation, config=config
        )
        python_seconds = time.perf_counter() - started

        started = time.perf_counter()
        rust_result = evaluate_t3(observation, config=config, library=native)
        rust_seconds = time.perf_counter() - started
        python_total += python_seconds
        rust_total += rust_seconds

        parity_errors = _parity_errors(python_result, rust_result)
        records.append(
            {
                "root_id": root.root_id,
                "seat": root.seat,
                "fingerprint": root.fingerprint,
                "python_seconds": python_seconds,
                "rust_seconds": rust_seconds,
                "speedup": python_seconds / max(rust_seconds, 1e-12),
                "parity": not parity_errors,
                "parity_errors": parity_errors,
                "selected_action_key": rust_result.get("selected_action_key"),
            }
        )

    aggregate_speedup = python_total / max(rust_total, 1e-12)
    gates = {
        "action_key_value_parity": all(record["parity"] for record in records),
        "aggregate_speedup_at_least_5x": aggregate_speedup >= minimum_speedup,
    }
    return {
        "schema": "hu_m3_python_rust_parity_benchmark_v1",
        "configuration": {
            "root_limit": 4,
            "sample_counts": "1/1/1/1",
            "minimum_speedup": minimum_speedup,
        },
        "root_count": len(records),
        "python_total_seconds": python_total,
        "rust_total_seconds": rust_total,
        "aggregate_speedup": aggregate_speedup,
        "gates": gates,
        "passed": all(gates.values()),
        "roots": records,
    }


def run_m3_validation(
    config: M3PilotConfig,
    output_dir: str | Path,
    *,
    force: bool = False,
    executable: str | Path | None = None,
    parity_benchmark: Callable[..., dict[str, Any]] = run_parity_benchmark,
) -> dict[str, Any]:
    """Run the bounded 100-root gate, then the 1000-root phase only on Go."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "validation_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(f"M3 validation summary already exists: {summary_path}")

    started = time.perf_counter()
    binary = Path(executable) if executable is not None else _ensure_release_runner()
    roots_100 = generate_pilot_roots(
        config.roots_100, config, run_id="hu-m3-pilot-100"
    )
    phase_100 = _run_pilot_phase(
        directory,
        "roots_100",
        roots_100,
        binary,
        deterministic_subset_roots=config.deterministic_subset_roots,
        force=force,
    )
    # Run the shard smoke before paying the Python-reference benchmark cost.
    # The benchmark remains an independent Go/No-Go input and is capped at
    # four roots.
    benchmark = parity_benchmark(
        roots_100[: config.parity_roots], minimum_speedup=config.minimum_speedup
    )
    go_1000 = bool(phase_100["passed"] and benchmark.get("passed") is True)

    phase_1000: dict[str, Any]
    if go_1000:
        roots_1000 = generate_pilot_roots(
            config.roots_1000, config, run_id="hu-m3-pilot-1000"
        )
        phase_1000 = _run_pilot_phase(
            directory,
            "roots_1000",
            roots_1000,
            binary,
            deterministic_subset_roots=config.deterministic_subset_roots,
            force=force,
        )
    else:
        phase_1000 = {
            "status": "not_run",
            "passed": False,
            "reason": "100-root correctness/parity/performance Go gate failed",
            "root_count": 0,
        }

    completed = go_1000 and phase_1000["passed"]
    gates = {
        "python_rust_parity": bool(
            benchmark.get("gates", {}).get("action_key_value_parity")
        ),
        "aggregate_speedup_at_least_5x": bool(
            benchmark.get("gates", {}).get("aggregate_speedup_at_least_5x")
        ),
        "roots_100_passed": bool(phase_100["passed"]),
        "roots_1000_ran_only_after_go": (not go_1000)
        or phase_1000.get("status") == "pass",
        "roots_1000_passed": bool(completed),
        "no_profile_current_or_cloud_access": True,
        "teacher_values_diagnostic_only": True,
    }
    summary = {
        "schema": M3_VALIDATION_SCHEMA,
        "status": "complete_pass" if all(gates.values()) else "no_go",
        "phase": "M3_native_engine_correctness_and_speed_pilot",
        "configuration": config.to_dict(),
        "root_generation": {
            "policy": ROOT_GENERATION_POLICY,
            "seed_formula": "seed_start + hand_index * seed_stride",
            "profile_selection": "none",
            "current_profile_read": False,
            "model_artifacts_loaded": [],
        },
        "compute_scope": {
            "local_bounded_pilot": True,
            "cloud_actions_performed": False,
            "spot_vm_started": False,
            "sample_counts": "1/1/1/1",
        },
        "teacher_values": TEACHER_VALUE_STATUS,
        "match_ev_reported": False,
        "promotion_evidence": False,
        "runner": {
            "path": str(binary.resolve()),
            "release_required_by_default": executable is None,
            "checkpoint_heartbeat_resume": True,
        },
        "parity_benchmark": benchmark,
        "phase_100": phase_100,
        "go_1000": go_1000,
        "phase_1000": phase_1000,
        "gates": gates,
        "total_seconds": time.perf_counter() - started,
    }
    _atomic_write_json(summary_path, summary)
    return summary


def _run_pilot_phase(
    directory: Path,
    name: str,
    roots: Sequence[PilotRoot],
    executable: Path,
    *,
    deterministic_subset_roots: int,
    force: bool,
) -> dict[str, Any]:
    paths = _phase_paths(directory, name)
    if force:
        _clear_known_runner_artifacts(paths)
    input_digest = write_runner_input(paths.input, roots, overwrite=force)
    started = time.perf_counter()
    runner_summary = _invoke_runner(
        executable, paths, run_id=f"hu-m3-{name}", resume=paths.checkpoint.exists()
    )
    runner_seconds = time.perf_counter() - started
    rows = _read_jsonl(paths.output)
    validation = validate_runner_results(roots, rows)
    checkpoint = _read_json_object(paths.checkpoint)
    heartbeat = _read_json_object(paths.heartbeat)

    subset = list(roots[: min(len(roots), deterministic_subset_roots)])
    subset_paths = _phase_paths(directory, name + "_deterministic_rerun")
    if force:
        _clear_known_runner_artifacts(subset_paths)
    write_runner_input(subset_paths.input, subset, overwrite=force)
    subset_summary = _invoke_runner(
        executable,
        subset_paths,
        run_id=f"hu-m3-{name}-deterministic-rerun",
        resume=subset_paths.checkpoint.exists(),
    )
    subset_rows = _read_jsonl(subset_paths.output)
    subset_validation = validate_runner_results(subset, subset_rows)
    original_by_id = {row["root_id"]: row["result"] for row in rows}
    repeated_by_id = {row["root_id"]: row["result"] for row in subset_rows}
    rerun_records = [
        {
            "root_id": root.root_id,
            "first_digest": _digest(original_by_id[root.root_id]),
            "second_digest": _digest(repeated_by_id[root.root_id]),
            "match": original_by_id[root.root_id] == repeated_by_id[root.root_id],
        }
        for root in subset
    ]

    checkpoint_ok = (
        checkpoint.get("complete") is True
        and checkpoint.get("committed_rows") == len(roots)
    )
    heartbeat_ok = (
        heartbeat.get("status") == "complete"
        and heartbeat.get("committed_rows") == len(roots)
    )
    runner_ok = (
        runner_summary.get("output_rows") == len(roots)
        and runner_summary.get("input_rows") == len(roots)
    )
    gates = {
        **validation["gates"],
        "runner_summary_complete": runner_ok,
        "checkpoint_complete": checkpoint_ok,
        "heartbeat_complete": heartbeat_ok,
        "deterministic_subset_well_formed": subset_validation["passed"],
        "deterministic_subset_rerun_match": all(
            record["match"] for record in rerun_records
        ),
        "no_profile_current_or_cloud_access": True,
        "teacher_values_diagnostic_only": True,
    }
    seats = Counter(root.seat for root in roots)
    return {
        "status": "pass" if all(gates.values()) else "fail",
        "passed": all(gates.values()),
        "root_count": len(roots),
        "seat_counts": {"first": seats["first"], "second": seats["second"]},
        "hand_seed_first": min(root.hand_seed for root in roots),
        "hand_seed_last": max(root.hand_seed for root in roots),
        "input_sha256": input_digest,
        "runner_seconds": runner_seconds,
        "runner_summary": runner_summary,
        "output_validation": validation,
        "checkpoint": {
            "schema_version": checkpoint.get("schema_version"),
            "committed_rows": checkpoint.get("committed_rows"),
            "complete": checkpoint.get("complete"),
        },
        "heartbeat": {
            "schema_version": heartbeat.get("schema_version"),
            "status": heartbeat.get("status"),
            "committed_rows": heartbeat.get("committed_rows"),
        },
        "deterministic_rerun": {
            "root_count": len(subset),
            "runner_summary": subset_summary,
            "records": rerun_records,
        },
        "gates": gates,
        "artifacts": {
            "input": str(paths.input.resolve()),
            "output": str(paths.output.resolve()),
            "checkpoint": str(paths.checkpoint.resolve()),
            "heartbeat": str(paths.heartbeat.resolve()),
        },
    }


def validate_runner_results(
    roots: Sequence[PilotRoot], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {root.root_id: root for root in roots}
    errors: list[str] = []
    seen: set[str] = set()
    candidate_digests: set[str] = set()
    evaluation_digests: set[str] = set()
    per_root_disjoint = True
    well_formed = True
    diagnostic = True

    for index, envelope in enumerate(rows):
        prefix = f"row {index}"
        root_id = envelope.get("root_id")
        if not isinstance(root_id, str) or root_id not in expected:
            errors.append(f"{prefix}: unknown root_id")
            well_formed = False
            continue
        if root_id in seen:
            errors.append(f"{prefix}: duplicate root_id {root_id}")
            well_formed = False
            continue
        seen.add(root_id)
        root = expected[root_id]
        if envelope.get("schema_version") != M3_RUNNER_RESULT_SCHEMA:
            errors.append(f"{prefix}: invalid runner result schema")
            well_formed = False
        if envelope.get("request_schema_version") != M3_PILOT_INPUT_SCHEMA:
            errors.append(f"{prefix}: input schema was not preserved")
            well_formed = False
        if envelope.get("input_index") != root.root_index:
            errors.append(f"{prefix}: input index mismatch")
            well_formed = False

        result = envelope.get("result")
        result_errors = _result_errors(root, result)
        if result_errors:
            errors.extend(f"{prefix}: {error}" for error in result_errors)
            well_formed = False
        if not isinstance(result, Mapping):
            continue
        diagnostic &= result.get("teacher_value_status") == TEACHER_VALUE_STATUS
        candidate = _hex_digest_list(result.get("candidate_rng_key_digests"))
        evaluation = _hex_digest_list(result.get("evaluation_rng_key_digests"))
        if not candidate:
            candidate = _belief_rng_digests(result.get("candidate_belief"))
        if not evaluation:
            evaluation = _belief_rng_digests(result.get("evaluation_belief"))
        if not candidate or not evaluation or not candidate.isdisjoint(evaluation):
            per_root_disjoint = False
        candidate_digests.update(candidate)
        evaluation_digests.update(evaluation)

    fingerprints = [root.fingerprint for root in roots]
    seats = Counter(root.seat for root in roots)
    complete = seen == set(expected) and len(rows) == len(roots)
    gates = {
        "balanced_first_second_roots": seats["first"] == seats["second"]
        and seats["first"] + seats["second"] == len(roots),
        "unique_root_fingerprints": len(fingerprints) == len(set(fingerprints)),
        "complete_unique_output_envelopes": complete,
        "selected_results_well_formed": well_formed,
        "candidate_evaluation_rng_digest_disjoint_per_root": per_root_disjoint,
        "candidate_evaluation_rng_digest_disjoint_global": candidate_digests.isdisjoint(
            evaluation_digests
        ),
        "teacher_values_labeled_diagnostic": diagnostic,
    }
    return {
        "passed": all(gates.values()),
        "root_count": len(roots),
        "output_row_count": len(rows),
        "candidate_rng_digest_count": len(candidate_digests),
        "evaluation_rng_digest_count": len(evaluation_digests),
        "errors": errors,
        "gates": gates,
    }


def _result_errors(root: PilotRoot, result: object) -> list[str]:
    if not isinstance(result, Mapping):
        return ["result must be an object"]
    errors: list[str] = []
    if result.get("status") != "ok":
        errors.append("engine status is not ok")
    if result.get("schema") != M3_ENGINE_RESULT_SCHEMA:
        errors.append("engine result schema mismatch")
    if result.get("kind") != "t3" or result.get("street") != "T3":
        errors.append("engine result is not T3")
    if result.get("seat") != root.seat:
        errors.append("seat mismatch")
    if result.get("observation_fingerprint") != root.fingerprint:
        errors.append("observation fingerprint mismatch")
    selected_key = result.get("selected_action_key")
    if not isinstance(selected_key, str) or not ACTION_KEY_RE.fullmatch(selected_key):
        errors.append("selected action key is malformed")
    selected_index = result.get("selected_action_original_index")
    actions = result.get("actions")
    legal_count = result.get("legal_action_count")
    if (
        isinstance(selected_index, bool)
        or not isinstance(selected_index, int)
        or not isinstance(actions, list)
        or not isinstance(legal_count, int)
        or legal_count <= 0
        or legal_count != len(actions)
    ):
        errors.append("legal action mapping is malformed")
        return errors
    action_keys: set[str] = set()
    selected_rows = []
    for action in actions:
        if not isinstance(action, Mapping):
            errors.append("action row must be an object")
            continue
        key = action.get("action_key")
        index = action.get("original_index")
        score = action.get("score")
        selection_score = action.get("selection_score")
        if not isinstance(key, str) or not ACTION_KEY_RE.fullmatch(key):
            errors.append("action row key is malformed")
        else:
            action_keys.add(key)
        if isinstance(index, bool) or not isinstance(index, int):
            errors.append("action original index is malformed")
        if not _finite_number(score) or not _finite_number(selection_score):
            errors.append("action score is not finite")
        if index == selected_index:
            selected_rows.append(action)
    if len(action_keys) != legal_count:
        errors.append("action keys are not unique")
    if len(selected_rows) != 1 or selected_rows[0].get("action_key") != selected_key:
        errors.append("selected key/index mapping is inconsistent")
    if not _finite_number(result.get("selected_action_evaluation_score")):
        errors.append("selected evaluation score is not finite")
    return errors


def _parity_errors(
    python_result: Mapping[str, Any], rust_result: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    for key in (
        "observation_fingerprint",
        "legal_action_count",
        "legal_action_set_digest",
        "legal_action_order_digest",
        "selected_action_key",
        "selected_action_original_index",
    ):
        if python_result.get(key) != rust_result.get(key):
            errors.append(f"{key} mismatch")
    python_actions = {
        row["action_key"]: row
        for row in python_result.get("actions", [])
        if isinstance(row, Mapping) and isinstance(row.get("action_key"), str)
    }
    rust_actions = {
        row["action_key"]: row
        for row in rust_result.get("actions", [])
        if isinstance(row, Mapping) and isinstance(row.get("action_key"), str)
    }
    if python_actions.keys() != rust_actions.keys():
        errors.append("ActionKey set mismatch")
        return errors
    for key in python_actions:
        for field in ("selection_score", "score"):
            left = python_actions[key].get(field)
            right = rust_actions[key].get(field)
            if not _numbers_close(left, right):
                errors.append(f"{key} {field} mismatch: {left!r} != {right!r}")
    return errors


def _belief_rng_digests(value: object) -> set[str]:
    if not isinstance(value, Mapping):
        return set()
    # Newer engine versions may expose the counter-key digests directly.  The
    # v1 engine exposes particle digests; those canonically include the RNG-key
    # digest and therefore remain a deterministic independence witness.
    raw = value.get("rng_key_digests", value.get("particle_digests"))
    return _hex_digest_list(raw)


def _hex_digest_list(raw: object) -> set[str]:
    if not isinstance(raw, list):
        return set()
    return {
        item
        for item in raw
        if isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item)
    }


def _observation_from_request(request: Mapping[str, Any]):
    from .hu_infoset import ActorObservation

    return ActorObservation.from_dict(dict(request["observation"]))


def _joint_config_from_request(request: Mapping[str, Any]) -> JointExactConfig:
    raw = request["config"]
    observation = request["observation"]
    return JointExactConfig(
        candidate_samples=int(raw["candidate_samples"]),
        evaluation_samples=int(raw["evaluation_samples"]),
        downstream_t3_samples=int(raw["downstream_t3_samples"]),
        downstream_t4_samples=int(raw["downstream_t4_samples"]),
        seed=int(raw["seed"]),
        candidate_seed=int(raw["candidate_seed"]),
        evaluation_seed=int(raw["evaluation_seed"]),
        run_id=str(raw["run_id"]),
        seat=str(observation["seat"]),
        to_act_order=str(observation["to_act_order"]),
        use_final_turn_cache=bool(raw.get("use_t4_action_cache", True)),
    )


def _ensure_release_runner() -> Path:
    path = runner_path(release=True)
    if not path.is_file():
        build_native_engine(release=True)
    if not path.is_file():
        raise FileNotFoundError(f"release M3 runner is missing: {path}")
    return path


def _phase_paths(directory: Path, name: str) -> PhasePaths:
    return PhasePaths(
        input=directory / f"{name}.input.jsonl",
        output=directory / f"{name}.output.jsonl",
        checkpoint=directory / f"{name}.checkpoint.json",
        heartbeat=directory / f"{name}.heartbeat.json",
    )


def _clear_known_runner_artifacts(paths: PhasePaths) -> None:
    for path in (
        paths.output,
        paths.checkpoint,
        paths.heartbeat,
        paths.partial,
        paths.checkpoint.with_name(paths.checkpoint.name + ".atomic.tmp"),
        paths.heartbeat.with_name(paths.heartbeat.name + ".atomic.tmp"),
    ):
        if path.exists():
            path.unlink()


def _invoke_runner(
    executable: Path,
    paths: PhasePaths,
    *,
    run_id: str,
    resume: bool,
) -> dict[str, Any]:
    command = [
        str(executable),
        "--input",
        str(paths.input),
        "--output",
        str(paths.output),
        "--checkpoint",
        str(paths.checkpoint),
        "--heartbeat",
        str(paths.heartbeat),
        "--run-id",
        run_id,
        "--checkpoint-every",
        "16",
        "--heartbeat-every",
        "8",
    ]
    if resume:
        command.append("--resume")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("RUST_BACKTRACE", "1")
    completed = subprocess.run(
        command,
        cwd=repository_root(),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "M3 Rust runner failed\n"
            + completed.stdout[-4000:]
            + completed.stderr[-8000:]
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("M3 Rust runner produced no summary")
    try:
        summary = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("M3 Rust runner summary is not JSON") from exc
    if not isinstance(summary, dict):
        raise RuntimeError("M3 Rust runner summary must be an object")
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    ).hexdigest()


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _numbers_close(left: object, right: object) -> bool:
    return _finite_number(left) and _finite_number(right) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roots-100", type=int, default=100)
    parser.add_argument("--roots-1000", type=int, default=1_000)
    parser.add_argument("--seed-start", type=int, default=2026071301)
    parser.add_argument("--seed-stride", type=int, default=1_000_003)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = M3PilotConfig(
        roots_100=args.roots_100,
        roots_1000=args.roots_1000,
        seed_start=args.seed_start,
        seed_stride=args.seed_stride,
    )
    report = run_m3_validation(config, args.output_dir, force=bool(args.force))
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "go_1000": report["go_1000"],
                "output": str(Path(args.output_dir) / "validation_summary.json"),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
