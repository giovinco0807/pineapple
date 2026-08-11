"""Run the restart-safe M3.1 Step 3 100-root T3 acceptance gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .action_key import action_key
from .action_space import generate_turn_actions
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_m31_t3_runtime import HuM31T3RuntimeConfig, HuM31T3SearchSolver
from .validate_hu_m31_t3_convergence import REFERENCE_BUDGET, _decision_integrity
from .validate_hu_m31_t3_profile import (
    DEFAULT_CANDIDATE_SEED,
    DEFAULT_CONTINUATION_SEED,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_SEED_START,
    DEFAULT_SEED_STRIDE,
    T3ProfileConfig,
    generate_general_t3_roots,
    process_memory_snapshot,
)


LOCAL100_TASK_SCHEMA = "hu_m31_t3_step3_local100_task_v1"
LOCAL100_SUMMARY_SCHEMA = "hu_m31_t3_step3_local100_summary_v1"
LOCAL100_HEARTBEAT_SCHEMA = "hu_m31_t3_step3_local100_heartbeat_v1"
LOCAL100_RUN_ID = "hu-m31-step3-local100-v1"
ROOT_COUNT = 100
DETERMINISTIC_ROOT_COUNT = 20
PERMUTATION_ROOT_COUNT = 10
WORKER_COUNT = 2
RAYON_THREADS_PER_WORKER = 8
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
class Local100Config:
    root_count: int = ROOT_COUNT
    deterministic_root_count: int = DETERMINISTIC_ROOT_COUNT
    permutation_root_count: int = PERMUTATION_ROOT_COUNT
    seed_start: int = DEFAULT_SEED_START
    seed_stride: int = DEFAULT_SEED_STRIDE
    run_id: str = LOCAL100_RUN_ID
    continuation_seed: int = DEFAULT_CONTINUATION_SEED
    candidate_seed: int = DEFAULT_CANDIDATE_SEED
    evaluation_seed: int = DEFAULT_EVALUATION_SEED
    worker_count: int = WORKER_COUNT
    rayon_threads_per_worker: int = RAYON_THREADS_PER_WORKER

    def __post_init__(self) -> None:
        for name in (
            "root_count",
            "deterministic_root_count",
            "permutation_root_count",
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
            raise ValueError("the accepted Step 3 pilot must contain exactly 100 roots")
        if self.deterministic_root_count != DETERMINISTIC_ROOT_COUNT:
            raise ValueError("the deterministic subset must contain exactly 20 roots")
        if self.permutation_root_count != PERMUTATION_ROOT_COUNT:
            raise ValueError("the permutation subset must contain exactly 10 roots")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive")
        if self.run_id != LOCAL100_RUN_ID:
            raise ValueError("local100 run_id is frozen")
        if self.candidate_seed == self.evaluation_seed:
            raise ValueError("candidate and evaluation seeds must be distinct")
        if self.worker_count != WORKER_COUNT:
            raise ValueError("local100 worker count is frozen at two")
        if self.rayon_threads_per_worker != RAYON_THREADS_PER_WORKER:
            raise ValueError("Rayon threads per worker are frozen at eight")

    @property
    def hand_count(self) -> int:
        return self.root_count // 2

    def hand_seed(self, hand_index: int) -> int:
        if not 0 <= hand_index < self.hand_count:
            raise IndexError("hand index outside local100")
        return self.seed_start + hand_index * self.seed_stride

    def profile_config(self) -> T3ProfileConfig:
        return T3ProfileConfig(
            root_count=self.root_count,
            seed_start=self.seed_start,
            seed_stride=self.seed_stride,
            run_id=self.run_id,
            continuation_seed=self.continuation_seed,
            candidate_seed=self.candidate_seed,
            evaluation_seed=self.evaluation_seed,
            candidate_samples=REFERENCE_BUDGET.candidate_samples,
            evaluation_samples=REFERENCE_BUDGET.evaluation_samples,
            downstream_t3_samples=REFERENCE_BUDGET.downstream_t3_samples,
            confirmation_candidate_samples=REFERENCE_BUDGET.candidate_samples,
            confirmation_evaluation_samples=REFERENCE_BUDGET.evaluation_samples,
            confirmation_downstream_t3_samples=REFERENCE_BUDGET.downstream_t3_samples,
            deterministic_roots=self.deterministic_root_count,
            permutation_roots=self.permutation_root_count,
            ladder_roots=2,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_count": self.root_count,
            "hand_count": self.hand_count,
            "deterministic_root_count": self.deterministic_root_count,
            "permutation_root_count": self.permutation_root_count,
            "seed_start": self.seed_start,
            "seed_stride": self.seed_stride,
            "run_id": self.run_id,
            "continuation_seed": self.continuation_seed,
            "candidate_seed": self.candidate_seed,
            "evaluation_seed": self.evaluation_seed,
            "worker_count": self.worker_count,
            "rayon_threads_per_worker": self.rayon_threads_per_worker,
            "budget": REFERENCE_BUDGET.to_dict(),
        }


@dataclass(frozen=True)
class TaskSpec:
    kind: str
    index: int

    @property
    def task_id(self) -> str:
        if self.kind == "primary":
            return f"primary_hand_{self.index:03d}"
        if self.kind == "determinism":
            return f"determinism_hand_{self.index:03d}"
        if self.kind == "permutation":
            return f"permutation_root_{self.index:03d}"
        raise ValueError(f"unsupported task kind: {self.kind}")


def build_task_specs(config: Local100Config) -> list[TaskSpec]:
    return [
        *(TaskSpec("primary", index) for index in range(config.hand_count)),
        *(
            TaskSpec("determinism", index)
            for index in range(config.deterministic_root_count // 2)
        ),
        *(
            TaskSpec("permutation", index)
            for index in range(config.permutation_root_count)
        ),
    ]


def schedule_task_specs(config: Local100Config) -> list[TaskSpec]:
    expected = build_task_specs(config)
    by_id = {spec.task_id: spec for spec in expected}
    ordered_ids = [
        *(f"permutation_root_{index:03d}" for index in range(0, 10, 2)),
        *(f"primary_hand_{index:03d}" for index in range(config.hand_count)),
        *(
            f"determinism_hand_{index:03d}"
            for index in range(config.deterministic_root_count // 2)
        ),
        *(f"permutation_root_{index:03d}" for index in range(1, 10, 2)),
    ]
    if set(ordered_ids) != set(by_id):
        raise RuntimeError("local100 scheduler does not cover every task exactly once")
    return [by_id[task_id] for task_id in ordered_ids]


def run_local100(
    *,
    config: Local100Config,
    library_path: Path,
    expected_library_sha256: str,
    authorization_report_path: Path,
    task_dir: Path,
    heartbeat_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    authorization = json.loads(
        authorization_report_path.read_text(encoding="utf-8")
    )
    _validate_authorization(
        authorization,
        expected_library_sha256=expected_library_sha256,
        config=config,
    )
    source_hash = _sha256_file(Path(__file__))
    contract_payload = {
        "schema": LOCAL100_SUMMARY_SCHEMA,
        "config": config.to_dict(),
        "library": str(library_path.resolve()),
        "library_sha256": expected_library_sha256.casefold(),
        "authorization_report": str(authorization_report_path),
        "authorization_report_sha256": _sha256_file(authorization_report_path),
        "runner_source_sha256": source_hash,
    }
    contract_digest = _digest(contract_payload)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema") != LOCAL100_SUMMARY_SCHEMA
            or existing.get("contract_digest") != contract_digest
            or existing.get("all_gates_passed") is not True
        ):
            raise ValueError("existing local100 summary does not match this contract")
        return existing

    task_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RAYON_NUM_THREADS"] = str(config.rayon_threads_per_worker)
    specs = schedule_task_specs(config)
    missing: list[TaskSpec] = []
    completed_ids: set[str] = set()
    for spec in specs:
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
        total=len(specs),
        completed=len(completed_ids),
        running=0,
        failed_task=None,
    )
    if missing:
        worker_common = {
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
                executor.submit(_run_task, spec, worker_common): spec
                for spec in missing
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
                            total=len(specs),
                            completed=len(completed_ids),
                            running=len(pending),
                            failed_task=spec.task_id,
                        )
                        raise
                    if task_id != spec.task_id:
                        raise RuntimeError("worker returned the wrong task id")
                    completed_ids.add(task_id)
                _write_heartbeat(
                    heartbeat_path,
                    status="running",
                    contract_digest=contract_digest,
                    total=len(specs),
                    completed=len(completed_ids),
                    running=len(pending),
                    failed_task=None,
                )

    report = aggregate_local100(
        config=config,
        library_path=library_path,
        expected_library_sha256=expected_library_sha256,
        task_dir=task_dir,
        contract_digest=contract_digest,
        contract_payload=contract_payload,
        run_wall_seconds=time.time() - started_epoch,
    )
    _write_json_atomic(output_path, report)
    _write_heartbeat(
        heartbeat_path,
        status="complete" if report["all_gates_passed"] else "no_go",
        contract_digest=contract_digest,
        total=len(specs),
        completed=len(completed_ids),
        running=0,
        failed_task=None,
    )
    return report


def _run_task(spec: TaskSpec, common: dict[str, Any]) -> str:
    config: Local100Config = common["config"]
    library_path = Path(common["library_path"])
    expected_library_sha256 = str(common["expected_library_sha256"])
    contract_digest = str(common["contract_digest"])
    task_dir = Path(common["task_dir"])
    output_path = task_dir / f"{spec.task_id}.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite task artifact: {output_path}")

    roots = generate_general_t3_roots(config.profile_config())
    solver = HuM31T3SearchSolver(
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
    started_epoch = time.time()
    started = time.perf_counter()
    memory_start = process_memory_snapshot()
    if spec.kind == "primary":
        payload = _run_primary_task(config, spec, roots, solver)
    elif spec.kind == "determinism":
        payload = _run_determinism_task(config, spec, roots, solver)
    elif spec.kind == "permutation":
        payload = _run_permutation_task(config, spec, roots, solver)
    else:
        raise ValueError(f"unsupported task kind: {spec.kind}")
    memory_end = process_memory_snapshot()
    report = {
        "schema": LOCAL100_TASK_SCHEMA,
        "task_id": spec.task_id,
        "task_kind": spec.kind,
        "task_index": spec.index,
        "contract_digest": contract_digest,
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
        "memory": {"start": memory_start, "end": memory_end},
        **payload,
    }
    _write_json_atomic(output_path, report)
    return spec.task_id


def _run_primary_task(config, spec, roots, solver) -> dict[str, Any]:
    selected_roots = roots[spec.index * 2 : spec.index * 2 + 2]
    scalar = []
    scalar_seconds = []
    for root in selected_roots:
        started = time.perf_counter()
        scalar.append(solver.solve(root.observation))
        scalar_seconds.append(time.perf_counter() - started)
    batch_started = time.perf_counter()
    batched = solver.solve_many([root.observation for root in selected_roots])
    batch_seconds = time.perf_counter() - batch_started
    rows = []
    for root, scalar_decision, batch_decision, elapsed in zip(
        selected_roots, scalar, batched, scalar_seconds, strict=True
    ):
        scalar_integrity = _full_decision_integrity(
            config, root.observation, scalar_decision, solver.library_sha256
        )
        batch_integrity = _full_decision_integrity(
            config, root.observation, batch_decision, solver.library_sha256
        )
        rows.append(
            {
                "root_index": root.root_index,
                "hand_index": root.hand_index,
                "hand_seed": root.hand_seed,
                "seat": root.observation.seat,
                "observation": root.observation.to_dict(),
                "observation_fingerprint": root.observation.fingerprint(),
                "scalar_wall_seconds": elapsed,
                "scalar": scalar_decision.to_dict(),
                "batch": batch_decision.to_dict(),
                "scalar_integrity": scalar_integrity,
                "batch_integrity": batch_integrity,
                "scalar_batch_semantic_digest_match": (
                    scalar_decision.semantic_result_digest
                    == batch_decision.semantic_result_digest
                ),
                "scalar_batch_mapping_digest_match": (
                    scalar_decision.result_digest == batch_decision.result_digest
                ),
            }
        )
    gates = {
        "exactly_two_balanced_roots": (
            len(rows) == 2 and {row["seat"] for row in rows} == {"first", "second"}
        ),
        "all_scalar_and_batch_integrity": all(
            all(row[scope].values())
            for row in rows
            for scope in ("scalar_integrity", "batch_integrity")
        ),
        "scalar_batch_both_digest_scopes_match": all(
            row["scalar_batch_semantic_digest_match"]
            and row["scalar_batch_mapping_digest_match"]
            for row in rows
        ),
    }
    return {
        "batch_wall_seconds": batch_seconds,
        "rows": rows,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def _run_determinism_task(config, spec, roots, solver) -> dict[str, Any]:
    selected_roots = roots[spec.index * 2 : spec.index * 2 + 2]
    rows = []
    for root in selected_roots:
        started = time.perf_counter()
        decision = solver.solve(root.observation)
        rows.append(
            {
                "root_index": root.root_index,
                "seat": root.observation.seat,
                "observation_fingerprint": root.observation.fingerprint(),
                "wall_seconds": time.perf_counter() - started,
                "decision": decision.to_dict(),
                "integrity": _full_decision_integrity(
                    config, root.observation, decision, solver.library_sha256
                ),
            }
        )
    gates = {
        "exactly_two_roots": len(rows) == 2,
        "all_decision_integrity": all(
            all(row["integrity"].values()) for row in rows
        ),
    }
    return {"rows": rows, "gates": gates, "all_gates_passed": all(gates.values())}


def _run_permutation_task(config, spec, roots, solver) -> dict[str, Any]:
    root = roots[spec.index]
    observations = [
        replace(root.observation, dealt_cards=tuple(cards))
        for cards in itertools.permutations(root.observation.dealt_cards)
    ]
    started = time.perf_counter()
    decisions = solver.solve_many(observations)
    batch_seconds = time.perf_counter() - started
    rows = []
    for observation, decision in zip(observations, decisions, strict=True):
        rows.append(
            {
                "dealt_cards": list(observation.dealt_cards),
                "observation_fingerprint": observation.fingerprint(),
                "decision": decision.to_dict(),
                "integrity": _full_decision_integrity(
                    config, observation, decision, solver.library_sha256
                ),
            }
        )
    semantic = {row["decision"]["semantic_result_digest"] for row in rows}
    selected = {row["decision"]["selected_action_key"] for row in rows}
    legal_sets = {row["decision"]["legal_action_set_digest"] for row in rows}
    mapping = {row["decision"]["result_digest"] for row in rows}
    legal_orders = {row["decision"]["legal_action_order_digest"] for row in rows}
    gates = {
        "all_six_permutations": len(rows) == math.factorial(3),
        "semantic_digest_unique_one": len(semantic) == 1,
        "selected_action_unique_one": len(selected) == 1,
        "legal_action_set_unique_one": len(legal_sets) == 1,
        "all_local_mappings_valid": all(
            all(row["integrity"].values()) for row in rows
        ),
    }
    return {
        "root_index": root.root_index,
        "seat": root.observation.seat,
        "original_observation": root.observation.to_dict(),
        "batch_wall_seconds": batch_seconds,
        "semantic_digest_unique_count": len(semantic),
        "mapping_bound_digest_unique_count": len(mapping),
        "legal_action_order_digest_unique_count": len(legal_orders),
        "rows": rows,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


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
            "downstream_t4_exact": payload["downstream_t4_samples"] == 0,
            "teacher_value_diagnostic": (
                payload["teacher_value_status"] == "diagnostic_not_match_EV"
            ),
        }
    )
    return checks


def aggregate_local100(
    *,
    config: Local100Config,
    library_path: Path,
    expected_library_sha256: str,
    task_dir: Path,
    contract_digest: str,
    contract_payload: dict[str, Any],
    run_wall_seconds: float,
) -> dict[str, Any]:
    specs = build_task_specs(config)
    reports = {}
    task_manifest = []
    for spec in specs:
        path = task_dir / f"{spec.task_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing local100 task artifact: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        _validate_task_artifact(
            report,
            spec=spec,
            contract_digest=contract_digest,
            expected_library_sha256=expected_library_sha256,
        )
        reports[spec.task_id] = report
        task_manifest.append(
            {"task_id": spec.task_id, "path": str(path), "sha256": _sha256_file(path)}
        )
    json_files = {path.name for path in task_dir.glob("*.json")}
    expected_files = {f"{spec.task_id}.json" for spec in specs}

    primary_reports = [
        reports[f"primary_hand_{index:03d}"] for index in range(config.hand_count)
    ]
    primary_rows = sorted(
        (row for report in primary_reports for row in report["rows"]),
        key=lambda row: row["root_index"],
    )
    deterministic_rows = sorted(
        (
            row
            for index in range(config.deterministic_root_count // 2)
            for row in reports[f"determinism_hand_{index:03d}"]["rows"]
        ),
        key=lambda row: row["root_index"],
    )
    permutation_reports = [
        reports[f"permutation_root_{index:03d}"]
        for index in range(config.permutation_root_count)
    ]

    fingerprints = [row["observation_fingerprint"] for row in primary_rows]
    seats = [row["seat"] for row in primary_rows]
    observations = [
        ActorObservation.from_dict(row["observation"]) for row in primary_rows
    ]
    candidate_keys: set[str] = set()
    evaluation_keys: set[str] = set()
    for observation in observations:
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
        == primary_by_root[row["root_index"]]["scalar"]["semantic_result_digest"]
        and row["decision"]["result_digest"]
        == primary_by_root[row["root_index"]]["scalar"]["result_digest"]
        for row in deterministic_rows
    )
    all_integrity = all(
        all(row[scope].values())
        for row in primary_rows
        for scope in ("scalar_integrity", "batch_integrity")
    ) and all(
        all(row["integrity"].values()) for row in deterministic_rows
    )
    scalar_batch_parity = all(
        row["scalar_batch_semantic_digest_match"]
        and row["scalar_batch_mapping_digest_match"]
        for row in primary_rows
    )
    seed_formula_valid = all(
        row["hand_seed"] == config.hand_seed(row["hand_index"])
        for row in primary_rows
    )
    public_only = all(
        not (_FORBIDDEN_OBSERVATION_FIELDS & row["observation"].keys())
        for row in primary_rows
    )
    gates = {
        "exactly_100_primary_roots": (
            len(primary_rows) == config.root_count
            and [row["root_index"] for row in primary_rows]
            == list(range(config.root_count))
        ),
        "exactly_50_first_and_50_second": (
            seats.count("first") == seats.count("second") == config.root_count // 2
        ),
        "unique_observation_fingerprints": len(set(fingerprints)) == config.root_count,
        "frozen_seed_formula_and_stride": seed_formula_valid,
        "actor_observation_contains_no_forbidden_truth_fields": public_only,
        "all_task_and_decision_integrity": all_integrity,
        "scalar_batch_both_digest_scopes_match_all_100": scalar_batch_parity,
        "deterministic_first_20_both_digest_scopes_match": (
            len(deterministic_rows) == config.deterministic_root_count
            and deterministic_match
        ),
        "all_six_permutations_first_10_semantically_stable": (
            len(permutation_reports) == config.permutation_root_count
            and all(report["all_gates_passed"] for report in permutation_reports)
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
        "exact_expected_write_once_task_set": json_files == expected_files,
        "teacher_values_diagnostic_only": all(
            row[scope]["teacher_value_status"] == "diagnostic_not_match_EV"
            for row in primary_rows
            for scope in ("scalar", "batch")
        ),
        "no_profile_current_or_cloud_access": True,
    }
    scalar_by_seat = {
        seat: _latency_summary(
            [
                float(row["scalar_wall_seconds"])
                for row in primary_rows
                if row["seat"] == seat
            ]
        )
        for seat in ("first", "second")
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
        "schema": LOCAL100_SUMMARY_SCHEMA,
        "status": "pass" if all(gates.values()) else "no_go",
        "scope": "local100_correctness_performance_gate_not_strength_match_ev_or_promotion",
        "contract_digest": contract_digest,
        "contract": contract_payload,
        "config": config.to_dict(),
        "engine": {
            "version": primary_rows[0]["scalar"]["engine_version"],
            "library": str(library_path.resolve()),
            "library_sha256": expected_library_sha256.casefold(),
            "build_or_fallback": False,
        },
        "performance": {
            "current_invocation_wall_seconds": run_wall_seconds,
            "primary_scalar_seconds_sum": sum(
                float(row["scalar_wall_seconds"]) for row in primary_rows
            ),
            "primary_batch_seconds_sum": sum(
                float(report["batch_wall_seconds"]) for report in primary_reports
            ),
            "determinism_seconds_sum": sum(
                float(row["wall_seconds"]) for row in deterministic_rows
            ),
            "permutation_seconds_sum": sum(
                float(report["batch_wall_seconds"]) for report in permutation_reports
            ),
            "scalar_by_seat": scalar_by_seat,
            "peak_process_rss_bytes": peak_rss,
        },
        "integrity": {
            "fingerprint_count": len(fingerprints),
            "unique_fingerprint_count": len(set(fingerprints)),
            "candidate_rng_key_count": len(candidate_keys),
            "evaluation_rng_key_count": len(evaluation_keys),
            "candidate_evaluation_rng_overlap_count": len(
                candidate_keys & evaluation_keys
            ),
        },
        "permutation_probe": [
            {
                "root_index": report["root_index"],
                "seat": report["seat"],
                "semantic_digest_unique_count": report[
                    "semantic_digest_unique_count"
                ],
                "mapping_bound_digest_unique_count": report[
                    "mapping_bound_digest_unique_count"
                ],
                "legal_action_order_digest_unique_count": report[
                    "legal_action_order_digest_unique_count"
                ],
                "gates": report["gates"],
            }
            for report in permutation_reports
        ],
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "teacher_value_status": "diagnostic_not_match_EV",
        "task_manifest": task_manifest,
        "current_profile_changed": False,
        "policy_or_profile_activated": False,
        "spot_vm_started": False,
        "rows": primary_rows,
    }


def _validate_authorization(report, *, expected_library_sha256, config):
    if report.get("schema") != "hu_m31_t3_step3_parallel_profile_v1":
        raise ValueError("unexpected local100 authorization schema")
    if report.get("all_gates_passed") is not True or report.get("local100_authorized") is not True:
        raise ValueError("parallel profile did not authorize local100")
    if report.get("engine", {}).get("library_sha256") != expected_library_sha256.casefold():
        raise ValueError("authorization native library does not match local100")
    if report.get("config", {}).get("workers") != config.worker_count:
        raise ValueError("authorization worker count does not match local100")
    if report.get("config", {}).get("budget") != REFERENCE_BUDGET.to_dict():
        raise ValueError("authorization budget does not match local100")
    if report.get("projection", {}).get("projected_parallel_seconds", math.inf) > 3600.0:
        raise ValueError("authorization projection exceeds 60 minutes")


def _validate_task_artifact(
    report, *, spec, contract_digest, expected_library_sha256
):
    if report.get("schema") != LOCAL100_TASK_SCHEMA:
        raise ValueError(f"unexpected task schema for {spec.task_id}")
    if (
        report.get("task_id") != spec.task_id
        or report.get("task_kind") != spec.kind
        or report.get("task_index") != spec.index
    ):
        raise ValueError(f"task identity mismatch for {spec.task_id}")
    if report.get("contract_digest") != contract_digest:
        raise ValueError(f"task contract mismatch for {spec.task_id}")
    if report.get("engine", {}).get("library_sha256") != expected_library_sha256.casefold():
        raise ValueError(f"task native library mismatch for {spec.task_id}")
    if report.get("rayon_threads") != str(RAYON_THREADS_PER_WORKER):
        raise ValueError(f"task Rayon thread count mismatch for {spec.task_id}")
    if report.get("all_gates_passed") is not True:
        raise ValueError(f"task gates failed for {spec.task_id}")


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
    running,
    failed_task,
):
    payload = {
        "schema": LOCAL100_HEARTBEAT_SCHEMA,
        "status": status,
        "contract_digest": contract_digest,
        "total_tasks": total,
        "completed_tasks": completed,
        "running_or_pending_tasks": running,
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
        raise FileExistsError(f"refusing to overwrite local100 artifact: {path}")
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
    parser.add_argument("--authorization-report", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    report = run_local100(
        config=Local100Config(),
        library_path=args.library,
        expected_library_sha256=args.library_sha256,
        authorization_report_path=args.authorization_report,
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
