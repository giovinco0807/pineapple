"""Run the restart-safe M3.1 Step 4 local 1,000-root T3 profile gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .action_space import generate_turn_actions
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_m31_t3_runtime import HuM31T3RuntimeConfig, HuM31T3SearchSolver
from .validate_hu_m31_t3_convergence import REFERENCE_BUDGET, _decision_integrity
from .validate_hu_m31_t3_profile import (
    DEFAULT_CANDIDATE_SEED,
    DEFAULT_CONTINUATION_SEED,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_SEED_STRIDE,
    _generate_hand_t3_roots,
    process_memory_snapshot,
)


LOCAL1000_TASK_SCHEMA = "hu_m31_t3_step4_local1000_task_v1"
LOCAL1000_SUMMARY_SCHEMA = "hu_m31_t3_step4_local1000_summary_v1"
LOCAL1000_SMOKE_SCHEMA = "hu_m31_t3_step4_local1000_smoke_v1"
LOCAL1000_HEARTBEAT_SCHEMA = "hu_m31_t3_step4_local1000_heartbeat_v1"
LOCAL1000_RUN_ID = "hu-m31-step4-local1000-v1"
ROOT_COUNT = 1_000
DETERMINISTIC_ROOT_COUNT = 20
SEED_START = 2026074201
WORKER_COUNT = 2
RAYON_THREADS_PER_WORKER = 8
MAX_WALL_SECONDS = 12_960.0
PROJECTION_BUFFER = 1.10
EXPECTED_ACTION_GEOMETRIES = frozenset({3, 9, 12, 21})
_FORBIDDEN_OBSERVATION_FIELDS = frozenset(
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


@dataclass(frozen=True)
class Local1000Config:
    root_count: int = ROOT_COUNT
    deterministic_root_count: int = DETERMINISTIC_ROOT_COUNT
    seed_start: int = SEED_START
    seed_stride: int = DEFAULT_SEED_STRIDE
    run_id: str = LOCAL1000_RUN_ID
    continuation_seed: int = DEFAULT_CONTINUATION_SEED
    candidate_seed: int = DEFAULT_CANDIDATE_SEED
    evaluation_seed: int = DEFAULT_EVALUATION_SEED
    worker_count: int = WORKER_COUNT
    rayon_threads_per_worker: int = RAYON_THREADS_PER_WORKER
    max_wall_seconds: float = MAX_WALL_SECONDS

    def __post_init__(self) -> None:
        for name in (
            "root_count",
            "deterministic_root_count",
            "seed_start",
            "seed_stride",
            "continuation_seed",
            "candidate_seed",
            "evaluation_seed",
            "worker_count",
            "rayon_threads_per_worker",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.root_count != ROOT_COUNT:
            raise ValueError("Step 4 requires exactly 1,000 roots")
        if self.deterministic_root_count != DETERMINISTIC_ROOT_COUNT:
            raise ValueError("Step 4 requires exactly 20 deterministic roots")
        if self.seed_start != SEED_START:
            raise ValueError("Step 4 seed_start is frozen")
        if self.seed_stride != DEFAULT_SEED_STRIDE:
            raise ValueError("Step 4 seed_stride is frozen")
        if self.run_id != LOCAL1000_RUN_ID:
            raise ValueError("Step 4 run_id is frozen")
        if self.candidate_seed == self.evaluation_seed:
            raise ValueError("candidate and evaluation seeds must be distinct")
        if self.worker_count != WORKER_COUNT:
            raise ValueError("Step 4 worker count is frozen at two")
        if self.rayon_threads_per_worker != RAYON_THREADS_PER_WORKER:
            raise ValueError("Step 4 Rayon threads are frozen at eight")
        if (
            not math.isfinite(self.max_wall_seconds)
            or self.max_wall_seconds != MAX_WALL_SECONDS
        ):
            raise ValueError("Step 4 wall-time gate is frozen at 12,960 seconds")

    @property
    def hand_count(self) -> int:
        return self.root_count // 2

    @property
    def deterministic_hand_count(self) -> int:
        return self.deterministic_root_count // 2

    def hand_seed(self, hand_index: int) -> int:
        if not 0 <= hand_index < self.hand_count:
            raise IndexError("hand index outside local1000")
        return self.seed_start + hand_index * self.seed_stride

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_count": self.root_count,
            "hand_count": self.hand_count,
            "deterministic_root_count": self.deterministic_root_count,
            "seed_start": self.seed_start,
            "seed_stride": self.seed_stride,
            "run_id": self.run_id,
            "continuation_seed": self.continuation_seed,
            "candidate_seed": self.candidate_seed,
            "evaluation_seed": self.evaluation_seed,
            "worker_count": self.worker_count,
            "rayon_threads_per_worker": self.rayon_threads_per_worker,
            "max_wall_seconds": self.max_wall_seconds,
            "budget": REFERENCE_BUDGET.to_dict(),
        }


@dataclass(frozen=True)
class TaskSpec:
    kind: str
    hand_index: int

    @property
    def task_id(self) -> str:
        if self.kind not in {"primary", "determinism"}:
            raise ValueError(f"unsupported task kind: {self.kind}")
        return f"{self.kind}_hand_{self.hand_index:03d}"


def build_task_specs(config: Local1000Config) -> list[TaskSpec]:
    return [
        *(TaskSpec("primary", index) for index in range(config.hand_count)),
        *(
            TaskSpec("determinism", index)
            for index in range(config.deterministic_hand_count)
        ),
    ]


def schedule_task_specs(config: Local1000Config) -> list[TaskSpec]:
    """Put expensive public first-seat geometries first to minimize tail."""

    scored: list[tuple[int, int, int, TaskSpec]] = []
    for hand_index in range(config.hand_count):
        observations = _generate_hand_t3_roots(config.hand_seed(hand_index))
        counts = [
            len(
                generate_turn_actions(
                    observation.hero_board,
                    observation.dealt_cards,
                )
            )
            for observation in observations
        ]
        scored.append(
            (counts[0], counts[1], -hand_index, TaskSpec("primary", hand_index))
        )
    primary = [row[-1] for row in sorted(scored, reverse=True)]
    deterministic = [
        TaskSpec("determinism", index)
        for index in range(config.deterministic_hand_count)
    ]
    scheduled = [*primary, *deterministic]
    if {spec.task_id for spec in scheduled} != {
        spec.task_id for spec in build_task_specs(config)
    }:
        raise RuntimeError("Step 4 scheduler did not cover every task")
    return scheduled


def projected_local1000_seconds(step3_summary: dict[str, Any]) -> dict[str, Any]:
    performance = step3_summary["performance"]
    unbuffered = (
        float(performance["primary_scalar_seconds_sum"]) * 10.0
        + float(performance["determinism_seconds_sum"])
    ) / WORKER_COUNT
    buffered = unbuffered * PROJECTION_BUFFER
    return {
        "method": (
            "(step3_primary_scalar_sum * 10 + step3_determinism_sum) "
            "/ 2 workers * 1.10 buffer"
        ),
        "unbuffered_seconds": unbuffered,
        "buffer": PROJECTION_BUFFER,
        "projected_seconds": buffered,
        "projected_hours": buffered / 3600.0,
        "gate_seconds": MAX_WALL_SECONDS,
        "passed": buffered <= MAX_WALL_SECONDS,
    }


def run_smoke(
    *,
    config: Local1000Config,
    library_path: Path,
    expected_library_sha256: str,
    step3_summary_path: Path,
    throughput_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    step3, throughput, projection = _load_authorization(
        config=config,
        expected_library_sha256=expected_library_sha256,
        step3_summary_path=step3_summary_path,
        throughput_report_path=throughput_report_path,
    )
    observations = _generate_hand_t3_roots(config.hand_seed(0))
    solver = _solver(config, library_path, expected_library_sha256)
    started = time.perf_counter()
    first = []
    repeated = []
    timings = []
    for observation in observations:
        root_started = time.perf_counter()
        first.append(solver.solve(observation))
        timings.append(time.perf_counter() - root_started)
    for observation in observations:
        repeated.append(solver.solve(observation))
    rows = []
    for observation, original, rerun, elapsed in zip(
        observations, first, repeated, timings, strict=True
    ):
        integrity = _full_decision_integrity(
            config, observation, original, solver.library_sha256
        )
        rerun_integrity = _full_decision_integrity(
            config, observation, rerun, solver.library_sha256
        )
        rows.append(
            {
                "seat": observation.seat,
                "observation": observation.to_dict(),
                "observation_fingerprint": observation.fingerprint(),
                "wall_seconds": elapsed,
                "legal_action_count": len(original.action_values),
                "semantic_digest_match": (
                    original.semantic_result_digest == rerun.semantic_result_digest
                ),
                "mapping_digest_match": original.result_digest == rerun.result_digest,
                "integrity": integrity,
                "rerun_integrity": rerun_integrity,
            }
        )
    gates = {
        "both_seats": [row["seat"] for row in rows] == ["first", "second"],
        "all_integrity": all(
            all(row[key].values())
            for row in rows
            for key in ("integrity", "rerun_integrity")
        ),
        "deterministic_both_digest_scopes": all(
            row["semantic_digest_match"] and row["mapping_digest_match"]
            for row in rows
        ),
        "projection_within_3p6_hours": projection["passed"],
        "step3_and_throughput_authorized": True,
        "no_profile_current_or_cloud_access": True,
    }
    report = {
        "schema": LOCAL1000_SMOKE_SCHEMA,
        "status": "pass" if all(gates.values()) else "no_go",
        "scope": "local1000_two_root_smoke_not_strength_or_match_ev",
        "config": config.to_dict(),
        "engine": {
            "version": solver.engine_version,
            "library": str(solver.library_path),
            "library_sha256": solver.library_sha256,
            "build_or_fallback": False,
        },
        "authorization_hashes": {
            str(step3_summary_path): _sha256_file(step3_summary_path),
            str(throughput_report_path): _sha256_file(throughput_report_path),
        },
        "projection": projection,
        "wall_seconds": time.perf_counter() - started,
        "rows": rows,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "teacher_value_status": "diagnostic_not_match_EV",
        "current_profile_changed": False,
        "spot_vm_started": False,
    }
    _write_json_atomic(output_path, report)
    return report


def run_local1000(
    *,
    config: Local1000Config,
    library_path: Path,
    expected_library_sha256: str,
    step3_summary_path: Path,
    throughput_report_path: Path,
    smoke_report_path: Path,
    task_dir: Path,
    heartbeat_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    step3, throughput, projection = _load_authorization(
        config=config,
        expected_library_sha256=expected_library_sha256,
        step3_summary_path=step3_summary_path,
        throughput_report_path=throughput_report_path,
    )
    smoke = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    if smoke.get("schema") != LOCAL1000_SMOKE_SCHEMA or smoke.get(
        "all_gates_passed"
    ) is not True:
        raise ValueError("Step 4 smoke did not pass")
    if smoke.get("engine", {}).get("library_sha256") != (
        expected_library_sha256.casefold()
    ):
        raise ValueError("Step 4 smoke native library mismatch")
    source_hash = _sha256_file(Path(__file__))
    contract = {
        "schema": LOCAL1000_SUMMARY_SCHEMA,
        "config": config.to_dict(),
        "library": str(library_path.resolve()),
        "library_sha256": expected_library_sha256.casefold(),
        "step3_summary": str(step3_summary_path),
        "step3_summary_sha256": _sha256_file(step3_summary_path),
        "throughput_report": str(throughput_report_path),
        "throughput_report_sha256": _sha256_file(throughput_report_path),
        "smoke_report": str(smoke_report_path),
        "smoke_report_sha256": _sha256_file(smoke_report_path),
        "runner_source_sha256": source_hash,
        "projection": projection,
    }
    contract_digest = _digest(contract)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema") != LOCAL1000_SUMMARY_SCHEMA
            or existing.get("contract_digest") != contract_digest
            or existing.get("all_gates_passed") is not True
        ):
            raise ValueError("existing Step 4 summary does not match contract")
        return existing

    os.environ["RAYON_NUM_THREADS"] = str(config.rayon_threads_per_worker)
    task_dir.mkdir(parents=True, exist_ok=True)
    scheduled = schedule_task_specs(config)
    completed_ids: set[str] = set()
    missing: list[TaskSpec] = []
    for spec in scheduled:
        path = task_dir / f"{spec.task_id}.json"
        if path.exists():
            _validate_task_artifact(
                json.loads(path.read_text(encoding="utf-8")),
                spec=spec,
                contract_digest=contract_digest,
                expected_library_sha256=expected_library_sha256,
            )
            completed_ids.add(spec.task_id)
        else:
            missing.append(spec)

    started_epoch = time.time()
    _write_heartbeat(
        heartbeat_path,
        status="running",
        contract_digest=contract_digest,
        total=len(scheduled),
        completed=len(completed_ids),
        pending=len(missing),
        failed_task=None,
    )
    if missing:
        common = {
            "config": config,
            "library_path": library_path,
            "expected_library_sha256": expected_library_sha256.casefold(),
            "contract_digest": contract_digest,
            "task_dir": task_dir,
        }
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.worker_count
        ) as executor:
            futures = {
                executor.submit(_run_task, spec, common): spec for spec in missing
            }
            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=5.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    spec = futures[future]
                    try:
                        task_id = future.result()
                    except BaseException:
                        _write_heartbeat(
                            heartbeat_path,
                            status="failed",
                            contract_digest=contract_digest,
                            total=len(scheduled),
                            completed=len(completed_ids),
                            pending=len(pending),
                            failed_task=spec.task_id,
                        )
                        raise
                    if task_id != spec.task_id:
                        raise RuntimeError("Step 4 worker returned wrong task id")
                    completed_ids.add(task_id)
                _write_heartbeat(
                    heartbeat_path,
                    status="running",
                    contract_digest=contract_digest,
                    total=len(scheduled),
                    completed=len(completed_ids),
                    pending=len(pending),
                    failed_task=None,
                )

    wall_seconds = time.time() - started_epoch
    report = aggregate_local1000(
        config=config,
        library_path=library_path,
        expected_library_sha256=expected_library_sha256,
        step3_summary=step3,
        task_dir=task_dir,
        contract=contract,
        contract_digest=contract_digest,
        run_wall_seconds=wall_seconds,
    )
    _write_json_atomic(output_path, report)
    _write_heartbeat(
        heartbeat_path,
        status="complete" if report["all_gates_passed"] else "no_go",
        contract_digest=contract_digest,
        total=len(scheduled),
        completed=len(completed_ids),
        pending=0,
        failed_task=None,
    )
    return report


def _run_task(spec: TaskSpec, common: dict[str, Any]) -> str:
    config: Local1000Config = common["config"]
    task_dir = Path(common["task_dir"])
    output_path = task_dir / f"{spec.task_id}.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Step 4 task: {output_path}")
    observations = _generate_hand_t3_roots(config.hand_seed(spec.hand_index))
    solver = _solver(
        config,
        Path(common["library_path"]),
        str(common["expected_library_sha256"]),
    )
    memory_start = process_memory_snapshot()
    started_epoch = time.time()
    started = time.perf_counter()
    rows = []
    for seat_offset, observation in enumerate(observations):
        root_started = time.perf_counter()
        decision = solver.solve(observation)
        rows.append(
            {
                "root_index": spec.hand_index * 2 + seat_offset,
                "hand_index": spec.hand_index,
                "hand_seed": config.hand_seed(spec.hand_index),
                "seat": observation.seat,
                "observation": observation.to_dict(),
                "observation_fingerprint": observation.fingerprint(),
                "wall_seconds": time.perf_counter() - root_started,
                "legal_action_count": len(decision.action_values),
                "decision": decision.to_dict(),
                "integrity": _full_decision_integrity(
                    config, observation, decision, solver.library_sha256
                ),
            }
        )
    gates = {
        "exactly_two_balanced_roots": (
            len(rows) == 2 and [row["seat"] for row in rows] == ["first", "second"]
        ),
        "all_decision_integrity": all(
            all(row["integrity"].values()) for row in rows
        ),
        "public_information_only": all(
            not (_FORBIDDEN_OBSERVATION_FIELDS & row["observation"].keys())
            for row in rows
        ),
    }
    report = {
        "schema": LOCAL1000_TASK_SCHEMA,
        "task_id": spec.task_id,
        "task_kind": spec.kind,
        "hand_index": spec.hand_index,
        "contract_digest": str(common["contract_digest"]),
        "process_id": os.getpid(),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
        "started_epoch": started_epoch,
        "completed_epoch": time.time(),
        "wall_seconds": time.perf_counter() - started,
        "engine": {
            "version": solver.engine_version,
            "library": str(solver.library_path),
            "library_sha256": solver.library_sha256,
            "build_or_fallback": False,
        },
        "memory": {"start": memory_start, "end": process_memory_snapshot()},
        "rows": rows,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    _write_json_atomic(output_path, report)
    return spec.task_id


def aggregate_local1000(
    *,
    config: Local1000Config,
    library_path: Path,
    expected_library_sha256: str,
    step3_summary: dict[str, Any],
    task_dir: Path,
    contract: dict[str, Any],
    contract_digest: str,
    run_wall_seconds: float,
) -> dict[str, Any]:
    specs = build_task_specs(config)
    reports: dict[str, dict[str, Any]] = {}
    manifest = []
    for spec in specs:
        path = task_dir / f"{spec.task_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing Step 4 task artifact: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        _validate_task_artifact(
            report,
            spec=spec,
            contract_digest=contract_digest,
            expected_library_sha256=expected_library_sha256,
        )
        reports[spec.task_id] = report
        manifest.append(
            {"task_id": spec.task_id, "path": str(path), "sha256": _sha256_file(path)}
        )
    expected_files = {f"{spec.task_id}.json" for spec in specs}
    actual_files = {path.name for path in task_dir.glob("*.json")}
    primary_rows = sorted(
        (
            row
            for index in range(config.hand_count)
            for row in reports[f"primary_hand_{index:03d}"]["rows"]
        ),
        key=lambda row: row["root_index"],
    )
    deterministic_rows = sorted(
        (
            row
            for index in range(config.deterministic_hand_count)
            for row in reports[f"determinism_hand_{index:03d}"]["rows"]
        ),
        key=lambda row: row["root_index"],
    )
    fingerprints = [row["observation_fingerprint"] for row in primary_rows]
    step3_fingerprints = {
        row["observation_fingerprint"] for row in step3_summary["rows"]
    }
    seats = [row["seat"] for row in primary_rows]
    hand_seeds = {row["hand_seed"] for row in primary_rows}
    step3_hand_seeds = {row["hand_seed"] for row in step3_summary["rows"]}
    action_geometries = {row["legal_action_count"] for row in primary_rows}
    candidate_keys: set[str] = set()
    evaluation_keys: set[str] = set()
    for row in primary_rows:
        observation = ActorObservation.from_dict(row["observation"])
        candidate = sample_hidden_card_particles(
            observation,
            base_seed=config.candidate_seed,
            run_id=config.run_id,
            sample_count=REFERENCE_BUDGET.candidate_samples,
        )
        evaluation = sample_hidden_card_particles(
            observation,
            base_seed=config.evaluation_seed,
            run_id=config.run_id,
            sample_count=REFERENCE_BUDGET.evaluation_samples,
        )
        candidate_keys.update(particle.rng_key_digest for particle in candidate.particles)
        evaluation_keys.update(particle.rng_key_digest for particle in evaluation.particles)

    primary_by_root = {row["root_index"]: row for row in primary_rows}
    deterministic_match = all(
        row["decision"]["semantic_result_digest"]
        == primary_by_root[row["root_index"]]["decision"]["semantic_result_digest"]
        and row["decision"]["result_digest"]
        == primary_by_root[row["root_index"]]["decision"]["result_digest"]
        for row in deterministic_rows
    )
    gates = {
        "exactly_1000_primary_roots": (
            len(primary_rows) == config.root_count
            and [row["root_index"] for row in primary_rows]
            == list(range(config.root_count))
        ),
        "exactly_500_first_and_500_second": (
            seats.count("first") == seats.count("second") == config.hand_count
        ),
        "unique_observation_fingerprints": len(set(fingerprints)) == config.root_count,
        "disjoint_from_step3_fingerprints": not (
            set(fingerprints) & step3_fingerprints
        ),
        "unique_nonoverlapping_hand_seeds": (
            len(hand_seeds) == config.hand_count
            and not (hand_seeds & step3_hand_seeds)
        ),
        "frozen_seed_formula": all(
            row["hand_seed"] == config.hand_seed(row["hand_index"])
            for row in primary_rows
        ),
        "all_expected_action_geometries_present": (
            action_geometries == EXPECTED_ACTION_GEOMETRIES
        ),
        "all_task_and_decision_integrity": all(
            all(row["integrity"].values())
            for row in [*primary_rows, *deterministic_rows]
        ),
        "deterministic_first_20_both_digest_scopes_match": (
            len(deterministic_rows) == config.deterministic_root_count
            and deterministic_match
        ),
        "candidate_evaluation_rng_globally_disjoint": not (
            candidate_keys & evaluation_keys
        ),
        "candidate_rng_keys_globally_unique": (
            len(candidate_keys)
            == config.root_count * REFERENCE_BUDGET.candidate_samples
        ),
        "evaluation_rng_keys_globally_unique": (
            len(evaluation_keys)
            == config.root_count * REFERENCE_BUDGET.evaluation_samples
        ),
        "exact_expected_write_once_task_set": actual_files == expected_files,
        "actual_wall_time_within_3p6_hours": run_wall_seconds <= config.max_wall_seconds,
        "teacher_values_diagnostic_only": all(
            row["decision"]["teacher_value_status"] == "diagnostic_not_match_EV"
            for row in primary_rows
        ),
        "no_profile_current_or_cloud_access": True,
    }
    latency = {
        seat: _latency_summary(
            [
                float(row["wall_seconds"])
                for row in primary_rows
                if row["seat"] == seat
            ]
        )
        for seat in ("first", "second")
    }
    geometry_counts = {
        str(count): sum(row["legal_action_count"] == count for row in primary_rows)
        for count in sorted(action_geometries)
    }
    peak_rss = max(
        (
            int(report["memory"]["end"]["peak_rss_bytes"])
            for report in reports.values()
            if report["memory"]["end"].get("peak_rss_bytes") is not None
        ),
        default=None,
    )
    return {
        "schema": LOCAL1000_SUMMARY_SCHEMA,
        "status": "pass" if all(gates.values()) else "no_go",
        "scope": "local1000_profile_gate_not_strength_match_ev_or_promotion",
        "contract_digest": contract_digest,
        "contract": contract,
        "config": config.to_dict(),
        "engine": {
            "version": primary_rows[0]["decision"]["engine_version"],
            "library": str(library_path.resolve()),
            "library_sha256": expected_library_sha256.casefold(),
            "build_or_fallback": False,
        },
        "performance": {
            "wall_seconds": run_wall_seconds,
            "wall_hours": run_wall_seconds / 3600.0,
            "latency_by_seat": latency,
            "peak_process_rss_bytes": peak_rss,
        },
        "integrity": {
            "fingerprint_count": len(fingerprints),
            "unique_fingerprint_count": len(set(fingerprints)),
            "step3_fingerprint_overlap_count": len(
                set(fingerprints) & step3_fingerprints
            ),
            "hand_seed_count": len(hand_seeds),
            "step3_hand_seed_overlap_count": len(hand_seeds & step3_hand_seeds),
            "candidate_rng_key_count": len(candidate_keys),
            "evaluation_rng_key_count": len(evaluation_keys),
            "candidate_evaluation_rng_overlap_count": len(
                candidate_keys & evaluation_keys
            ),
            "legal_action_geometry_counts": geometry_counts,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "task_manifest": manifest,
        "teacher_value_status": "diagnostic_not_match_EV",
        "current_profile_changed": False,
        "policy_or_profile_activated": False,
        "spot_vm_started": False,
        "rows": primary_rows,
    }


def _load_authorization(
    *,
    config: Local1000Config,
    expected_library_sha256: str,
    step3_summary_path: Path,
    throughput_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    step3 = json.loads(step3_summary_path.read_text(encoding="utf-8"))
    if step3.get("schema") != "hu_m31_t3_step3_local100_summary_v1" or step3.get(
        "all_gates_passed"
    ) is not True:
        raise ValueError("Step 3 summary did not pass")
    if step3.get("engine", {}).get("library_sha256") != (
        expected_library_sha256.casefold()
    ):
        raise ValueError("Step 3 native library mismatch")
    if step3.get("config", {}).get("budget") != REFERENCE_BUDGET.to_dict():
        raise ValueError("Step 3 budget mismatch")
    throughput = json.loads(throughput_report_path.read_text(encoding="utf-8"))
    if throughput.get("schema") != "hu_m31_t3_step3_parallel_profile_v1" or throughput.get(
        "all_gates_passed"
    ) is not True:
        raise ValueError("throughput report did not pass")
    if throughput.get("engine", {}).get("library_sha256") != (
        expected_library_sha256.casefold()
    ):
        raise ValueError("throughput native library mismatch")
    if throughput.get("config", {}).get("workers") != config.worker_count:
        raise ValueError("throughput worker count mismatch")
    if throughput.get("config", {}).get("budget") != REFERENCE_BUDGET.to_dict():
        raise ValueError("throughput budget mismatch")
    projection = projected_local1000_seconds(step3)
    if not projection["passed"]:
        raise ValueError("projected local1000 wall time exceeds 3.6 hours")
    return step3, throughput, projection


def _solver(config, library_path, expected_library_sha256):
    return HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=expected_library_sha256,
            library_path=library_path,
            run_id=config.run_id,
            candidate_samples=REFERENCE_BUDGET.candidate_samples,
            evaluation_samples=REFERENCE_BUDGET.evaluation_samples,
            downstream_t3_samples=REFERENCE_BUDGET.downstream_t3_samples,
            seed=config.continuation_seed,
            candidate_seed=config.candidate_seed,
            evaluation_seed=config.evaluation_seed,
        )
    )


def _full_decision_integrity(config, observation, decision, library_sha256):
    checks = _decision_integrity(observation, decision, REFERENCE_BUDGET)
    payload = decision.to_dict()
    checks.update(
        {
            "run_id": decision.run_id == config.run_id,
            "continuation_seed": decision.continuation_seed == config.continuation_seed,
            "candidate_seed": decision.candidate_seed == config.candidate_seed,
            "evaluation_seed": decision.evaluation_seed == config.evaluation_seed,
            "native_library_sha256": decision.native_library_sha256 == library_sha256,
            "execution_mode_scalar": (
                decision.execution_mode == "scalar" and decision.batch_size == 1
            ),
            "downstream_t4_exact": payload["downstream_t4_samples"] == 0,
            "teacher_value_diagnostic": (
                payload["teacher_value_status"] == "diagnostic_not_match_EV"
            ),
        }
    )
    return checks


def _validate_task_artifact(
    report, *, spec, contract_digest, expected_library_sha256
):
    if report.get("schema") != LOCAL1000_TASK_SCHEMA:
        raise ValueError(f"unexpected Step 4 task schema: {spec.task_id}")
    if (
        report.get("task_id") != spec.task_id
        or report.get("task_kind") != spec.kind
        or report.get("hand_index") != spec.hand_index
    ):
        raise ValueError(f"Step 4 task identity mismatch: {spec.task_id}")
    if report.get("contract_digest") != contract_digest:
        raise ValueError(f"Step 4 task contract mismatch: {spec.task_id}")
    if report.get("engine", {}).get("library_sha256") != (
        expected_library_sha256.casefold()
    ):
        raise ValueError(f"Step 4 task library mismatch: {spec.task_id}")
    if report.get("rayon_threads") != str(RAYON_THREADS_PER_WORKER):
        raise ValueError(f"Step 4 task Rayon mismatch: {spec.task_id}")
    if report.get("all_gates_passed") is not True:
        raise ValueError(f"Step 4 task failed: {spec.task_id}")


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean_seconds": statistics.fmean(ordered),
        "p50_seconds": _percentile(ordered, 0.50),
        "p95_seconds": _percentile(ordered, 0.95),
        "p99_seconds": _percentile(ordered, 0.99),
        "max_seconds": max(ordered),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, math.ceil(len(values) * fraction) - 1))
    return float(values[index])


def _write_heartbeat(
    path,
    *,
    status,
    contract_digest,
    total,
    completed,
    pending,
    failed_task,
):
    payload = {
        "schema": LOCAL1000_HEARTBEAT_SCHEMA,
        "status": status,
        "contract_digest": contract_digest,
        "total_tasks": total,
        "completed_tasks": completed,
        "pending_tasks": pending,
        "failed_task": failed_task,
        "process_id": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Step 4 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--step3-summary", type=Path, required=True)
    parser.add_argument("--throughput-report", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path)
    parser.add_argument("--heartbeat", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    config = Local1000Config()
    if args.smoke_only:
        report = run_smoke(
            config=config,
            library_path=args.library,
            expected_library_sha256=args.library_sha256,
            step3_summary_path=args.step3_summary,
            throughput_report_path=args.throughput_report,
            output_path=args.smoke_report,
        )
    else:
        if args.task_dir is None or args.heartbeat is None or args.output is None:
            raise SystemExit("--task-dir, --heartbeat, and --output are required")
        report = run_local1000(
            config=config,
            library_path=args.library,
            expected_library_sha256=args.library_sha256,
            step3_summary_path=args.step3_summary,
            throughput_report_path=args.throughput_report,
            smoke_report_path=args.smoke_report,
            task_dir=args.task_dir,
            heartbeat_path=args.heartbeat,
            output_path=args.output,
        )
    summary = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not report["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
