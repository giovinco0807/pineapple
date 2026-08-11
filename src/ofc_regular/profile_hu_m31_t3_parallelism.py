"""Measure local process-level scaling before the M3.1 100-root T3 pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .hu_m31_t3_runtime import HuM31T3RuntimeConfig, HuM31T3SearchSolver
from .validate_hu_m31_t3_convergence import REFERENCE_BUDGET, _decision_integrity
from .validate_hu_m31_t3_profile import (
    DEFAULT_CANDIDATE_SEED,
    DEFAULT_CONTINUATION_SEED,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_SEED_STRIDE,
    T3ProfileConfig,
    generate_general_t3_roots,
    process_memory_snapshot,
)


PARALLEL_PROFILE_SCHEMA = "hu_m31_t3_step3_parallel_profile_v1"
DEFAULT_WORKERS = 8
DEFAULT_HANDS = 8
DEFAULT_SEED_START = 2126073201
DEFAULT_RUN_ID = "hu-m31-step3-parallel-probe-v1"
DEFAULT_MAX_PROJECTED_SECONDS = 3_600.0


@dataclass(frozen=True)
class ParallelProfileConfig:
    workers: int = DEFAULT_WORKERS
    hand_count: int = DEFAULT_HANDS
    seed_start: int = DEFAULT_SEED_START
    seed_stride: int = DEFAULT_SEED_STRIDE
    run_id: str = DEFAULT_RUN_ID
    continuation_seed: int = DEFAULT_CONTINUATION_SEED
    candidate_seed: int = DEFAULT_CANDIDATE_SEED
    evaluation_seed: int = DEFAULT_EVALUATION_SEED
    max_projected_seconds: float = DEFAULT_MAX_PROJECTED_SECONDS

    def __post_init__(self) -> None:
        for name in (
            "workers",
            "hand_count",
            "seed_start",
            "seed_stride",
            "continuation_seed",
            "candidate_seed",
            "evaluation_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.workers <= 0 or self.hand_count <= 0:
            raise ValueError("workers and hand_count must be positive")
        if self.workers > self.hand_count:
            raise ValueError("workers must not exceed hand_count")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive")
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.candidate_seed == self.evaluation_seed:
            raise ValueError("candidate and evaluation seeds must be distinct")
        if (
            not math.isfinite(self.max_projected_seconds)
            or self.max_projected_seconds <= 0.0
        ):
            raise ValueError("max_projected_seconds must be positive")

    def hand_seed(self, hand_index: int) -> int:
        if not 0 <= hand_index < self.hand_count:
            raise IndexError("hand index outside parallel probe")
        return self.seed_start + hand_index * self.seed_stride

    def to_dict(self) -> dict[str, Any]:
        return {
            "workers": self.workers,
            "hand_count": self.hand_count,
            "root_count": self.hand_count * 2,
            "seed_start": self.seed_start,
            "seed_stride": self.seed_stride,
            "run_id": self.run_id,
            "continuation_seed": self.continuation_seed,
            "candidate_seed": self.candidate_seed,
            "evaluation_seed": self.evaluation_seed,
            "max_projected_seconds": self.max_projected_seconds,
            "budget": REFERENCE_BUDGET.to_dict(),
        }


def run_parallel_profile(
    *,
    config: ParallelProfileConfig,
    library_path: Path,
    expected_library_sha256: str,
    local10_report: dict[str, Any],
    convergence_report: dict[str, Any],
) -> dict[str, Any]:
    """Run one both-seat hand per process and project the full local100 gate."""

    _validate_source_reports(
        local10_report,
        convergence_report,
        expected_library_sha256=expected_library_sha256,
    )
    worker_args = [
        {
            "hand_index": index,
            "hand_seed": config.hand_seed(index),
            "run_id": config.run_id,
            "continuation_seed": config.continuation_seed,
            "candidate_seed": config.candidate_seed,
            "evaluation_seed": config.evaluation_seed,
            "library_path": str(library_path),
            "expected_library_sha256": expected_library_sha256,
        }
        for index in range(config.hand_count)
    ]
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=config.workers
    ) as executor:
        worker_rows = list(executor.map(_run_one_hand, worker_args))
    wall_seconds = time.perf_counter() - started
    projection = project_local100_parallel_wall_time(
        local10_report=local10_report,
        convergence_report=convergence_report,
        worker_rows=worker_rows,
        parallel_wall_seconds=wall_seconds,
        gate_seconds=config.max_projected_seconds,
    )
    fingerprints = [
        fingerprint
        for row in worker_rows
        for fingerprint in row["observation_fingerprints"]
    ]
    pids = {int(row["process_id"]) for row in worker_rows}
    worker_engine_rows = {
        (
            row["engine_version"],
            row["library_sha256"],
            str(Path(row["library"]).resolve()),
        )
        for row in worker_rows
    }
    expected_engine_row = (
        worker_rows[0]["engine_version"],
        expected_library_sha256.casefold(),
        str(library_path.resolve()),
    )
    gates = {
        "all_workers_completed": len(worker_rows) == config.hand_count,
        "worker_process_count_matches_config": len(pids) == config.workers,
        "all_worker_integrity": all(row["all_integrity_passed"] for row in worker_rows),
        "unique_probe_observation_fingerprints": (
            len(fingerprints) == len(set(fingerprints)) == config.hand_count * 2
        ),
        "reference_budget_4_8_2_exact_t4": all(
            row["budget"] == REFERENCE_BUDGET.to_dict() for row in worker_rows
        ),
        "all_workers_use_single_requested_native_engine": (
            worker_engine_rows == {expected_engine_row}
        ),
        "projected_full_local100_workflow_within_60_minutes": (
            projection["projected_parallel_seconds"]
            <= config.max_projected_seconds
        ),
        "no_profile_current_or_cloud_access": True,
    }
    return {
        "schema": PARALLEL_PROFILE_SCHEMA,
        "status": "pass" if all(gates.values()) else "no_go",
        "scope": "local_parallel_scaling_probe_not_100_root_strength_or_match_ev",
        "config": config.to_dict(),
        "engine": {
            "version": worker_rows[0]["engine_version"],
            "library": str(library_path.resolve()),
            "library_sha256": expected_library_sha256.casefold(),
            "build_or_fallback": False,
        },
        "parallel_wall_seconds": wall_seconds,
        "worker_rows": worker_rows,
        "projection": projection,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "local100_authorized": all(gates.values()),
        "teacher_value_status": "diagnostic_not_match_EV",
        "current_profile_changed": False,
        "spot_vm_started": False,
    }


def _run_one_hand(args: dict[str, Any]) -> dict[str, Any]:
    hand_index = int(args["hand_index"])
    hand_seed = int(args["hand_seed"])
    root_config = T3ProfileConfig(
        root_count=2,
        seed_start=hand_seed,
        seed_stride=DEFAULT_SEED_STRIDE,
        run_id=str(args["run_id"]),
        continuation_seed=int(args["continuation_seed"]),
        candidate_seed=int(args["candidate_seed"]),
        evaluation_seed=int(args["evaluation_seed"]),
        candidate_samples=REFERENCE_BUDGET.candidate_samples,
        evaluation_samples=REFERENCE_BUDGET.evaluation_samples,
        downstream_t3_samples=REFERENCE_BUDGET.downstream_t3_samples,
        confirmation_candidate_samples=REFERENCE_BUDGET.candidate_samples,
        confirmation_evaluation_samples=REFERENCE_BUDGET.evaluation_samples,
        confirmation_downstream_t3_samples=REFERENCE_BUDGET.downstream_t3_samples,
        deterministic_roots=2,
        permutation_roots=2,
        ladder_roots=2,
    )
    roots = generate_general_t3_roots(root_config)
    solver = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=str(args["expected_library_sha256"]),
            library_path=Path(str(args["library_path"])),
            run_id=str(args["run_id"]),
            candidate_samples=REFERENCE_BUDGET.candidate_samples,
            evaluation_samples=REFERENCE_BUDGET.evaluation_samples,
            downstream_t3_samples=REFERENCE_BUDGET.downstream_t3_samples,
            seed=int(args["continuation_seed"]),
            candidate_seed=int(args["candidate_seed"]),
            evaluation_seed=int(args["evaluation_seed"]),
        )
    )
    memory_start = process_memory_snapshot()
    started = time.perf_counter()
    decisions = solver.solve_many([root.observation for root in roots])
    elapsed = time.perf_counter() - started
    memory_end = process_memory_snapshot()
    integrity = [
        _decision_integrity(root.observation, decision, REFERENCE_BUDGET)
        for root, decision in zip(roots, decisions, strict=True)
    ]
    return {
        "hand_index": hand_index,
        "hand_seed": hand_seed,
        "process_id": os.getpid(),
        "engine_version": solver.engine_version,
        "library": str(solver.library_path),
        "library_sha256": solver.library_sha256,
        "wall_seconds": elapsed,
        "budget": REFERENCE_BUDGET.to_dict(),
        "seats": [root.observation.seat for root in roots],
        "observation_fingerprints": [
            root.observation.fingerprint() for root in roots
        ],
        "legal_action_counts": [len(decision.action_values) for decision in decisions],
        "semantic_result_digests": [
            decision.semantic_result_digest for decision in decisions
        ],
        "mapping_bound_result_digests": [decision.result_digest for decision in decisions],
        "integrity": integrity,
        "all_integrity_passed": all(
            all(checks.values()) for checks in integrity
        ),
        "memory_start": memory_start,
        "memory_end": memory_end,
    }


def project_local100_parallel_wall_time(
    *,
    local10_report: dict[str, Any],
    convergence_report: dict[str, Any],
    worker_rows: Sequence[dict[str, Any]],
    parallel_wall_seconds: float,
    gate_seconds: float,
) -> dict[str, Any]:
    if parallel_wall_seconds <= 0.0 or not worker_rows:
        raise ValueError("parallel timing inputs must be positive and non-empty")
    budget_seconds = {
        row["budget"]["label"]: float(row["batch_wall_seconds"])
        for row in convergence_report["budget_runs"]
    }
    base_seconds = budget_seconds["baseline_1_1_1"]
    reference_seconds = budget_seconds["teacher_default_4_8_2"]
    if base_seconds <= 0.0 or reference_seconds <= 0.0:
        raise ValueError("convergence budget timings must be positive")
    sample_budget_cost_ratio = reference_seconds / base_seconds
    base_serial_projection = float(
        local10_report["local100_projection"]["projected_total_seconds"]
    )
    projected_reference_serial = base_serial_projection * sample_budget_cost_ratio
    worker_seconds_sum = sum(float(row["wall_seconds"]) for row in worker_rows)
    effective_parallelism = worker_seconds_sum / parallel_wall_seconds
    projected_parallel = projected_reference_serial / effective_parallelism
    return {
        "method": (
            "local10_full_workflow_projection * measured_4_8_2_to_1_1_1_cost_ratio "
            "/ measured_process_parallelism"
        ),
        "base_1_1_1_full_workflow_seconds": base_serial_projection,
        "measured_1_1_1_two_root_batch_seconds": base_seconds,
        "measured_4_8_2_two_root_batch_seconds": reference_seconds,
        "sample_budget_cost_ratio": sample_budget_cost_ratio,
        "projected_4_8_2_serial_seconds": projected_reference_serial,
        "parallel_worker_seconds_sum": worker_seconds_sum,
        "parallel_wall_seconds": parallel_wall_seconds,
        "measured_effective_parallelism": effective_parallelism,
        "projected_parallel_seconds": projected_parallel,
        "projected_parallel_minutes": projected_parallel / 60.0,
        "gate_seconds": gate_seconds,
        "projection_is_diagnostic": True,
    }


def _validate_source_reports(
    local10_report: dict[str, Any],
    convergence_report: dict[str, Any],
    *,
    expected_library_sha256: str | None = None,
) -> None:
    if local10_report.get("schema") != "hu_m31_t3_step3_local10_profile_v1":
        raise ValueError("unexpected local10 report schema")
    if local10_report.get("all_gates_passed") is not True:
        raise ValueError("local10 report did not pass")
    if convergence_report.get("schema") != "hu_m31_t3_step3_budget_convergence_v1":
        raise ValueError("unexpected convergence report schema")
    if convergence_report.get("all_gates_passed") is not True:
        raise ValueError("convergence report did not pass")
    authorization = convergence_report.get("local100_authorization", {})
    if authorization.get("authorized") is not True:
        raise ValueError("convergence report did not authorize local100")
    if authorization.get("budget") != REFERENCE_BUDGET.to_dict():
        raise ValueError("convergence report authorized an unexpected budget")
    local10_hash = local10_report.get("engine", {}).get("library_sha256")
    convergence_hash = convergence_report.get("engine", {}).get("library_sha256")
    if not local10_hash or local10_hash != convergence_hash:
        raise ValueError("source reports do not pin the same native library")
    if (
        expected_library_sha256 is not None
        and local10_hash != expected_library_sha256.casefold()
    ):
        raise ValueError("source reports do not match the requested native library")


def _write_json_atomic(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite parallel profile: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    parser.add_argument("--local10-report", type=Path, required=True)
    parser.add_argument("--convergence-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--hands", type=int, default=DEFAULT_HANDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED_START)
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    config = ParallelProfileConfig(
        workers=args.workers,
        hand_count=args.hands,
        seed_start=args.seed,
    )
    local10_report = json.loads(args.local10_report.read_text(encoding="utf-8"))
    convergence_report = json.loads(
        args.convergence_report.read_text(encoding="utf-8")
    )
    report = run_parallel_profile(
        config=config,
        library_path=args.library,
        expected_library_sha256=args.library_sha256,
        local10_report=local10_report,
        convergence_report=convergence_report,
    )
    report["source_hashes"] = {
        "src/ofc_regular/profile_hu_m31_t3_parallelism.py": _sha256_file(
            Path(__file__)
        ),
        "src/ofc_regular/hu_m31_t3_runtime.py": _sha256_file(
            Path(__file__).with_name("hu_m31_t3_runtime.py")
        ),
    }
    report["input_artifact_hashes"] = {
        str(args.local10_report): _sha256_file(args.local10_report),
        str(args.convergence_report): _sha256_file(args.convergence_report),
    }
    _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
