"""Run one restart-safe M3.1 Step 6a infrastructure canary shard."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_space import generate_turn_actions
from .ai_profiles import ModelPaths, load_model_bundle
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    behavior_profile_for_index,
    generate_behavior_t3_roots,
)
from .hu_m31_t3_runtime import (
    HuM31T3RuntimeConfig,
    HuM31T3SearchSolver,
)
from .validate_hu_m31_t3_convergence import REFERENCE_BUDGET
from .validate_hu_m31_t3_local1000 import (
    _full_decision_integrity,
    _percentile,
)
from .validate_hu_m31_t3_profile import process_memory_snapshot


STEP6A_PACKAGE_SCHEMA = "hu_m31_t3_step6a_spot_package_v1"
STEP6A_SHARD_SCHEMA = "hu_m31_t3_step6a_shard_v1"
STEP6A_ROOT_TASK_SCHEMA = "hu_m31_t3_step6a_root_task_v1"
STEP6A_SEARCH_TASK_SCHEMA = "hu_m31_t3_step6a_search_task_v1"
STEP6A_HEARTBEAT_SCHEMA = "hu_m31_t3_step6a_heartbeat_v1"
STEP6A_PARITY_SCHEMA = "hu_m31_t3_step6a_linux_parity_v1"
STEP6A_SUMMARY_SCHEMA = "hu_m31_t3_step6a_summary_v1"
STEP6A_RUN_ID = "hu-m31-step6a-infrastructure-canary-v1"
STEP5_CONTRACT_CANONICAL_SHA256 = (
    "04c4298feaed78f327f7fb6601f70a6821c3a91dcd2996cc34becbc4bb38e0b4"
)
SEED_STRIDE = 1_000_003
HAND_SEED_BASE = 300_108_071_901
BEHAVIOR_SEED_BASE = 301_108_071_901
CANDIDATE_SEED_BASE = 302_108_071_901
EVALUATION_SEED_BASE = 303_108_071_901
CHILD_SEED_BASE = 304_108_071_901
CONFIRMATION_SEED_BASE = 305_108_071_901
PAIRED_HANDS_PER_SHARD = 25
ROOTS_PER_SHARD = 50
WORKERS = 2
RAYON_THREADS = 8
MAX_FIRST_P95_SECONDS = 180.0
MAX_SECOND_P95_SECONDS = 6.0
MAX_PEAK_RSS_BYTES = 1_073_741_824
_PORTABLE_OMIT = frozenset(
    {
        "native_library_sha256",
        "native_latency_ms",
        "validation_latency_ms",
        "total_latency_ms",
        "semantic_result_digest",
        "result_digest",
    }
)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _write_once(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Step 6a artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_heartbeat(path: Path, **values: Any) -> None:
    payload = {
        "schema": STEP6A_HEARTBEAT_SCHEMA,
        **values,
        "process_id": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(payload))
    os.replace(temporary, path)


def portable_decision_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only binary identity and measured latency from a decision."""

    payload = deepcopy(dict(value))
    for key in _PORTABLE_OMIT:
        payload.pop(key, None)
    return payload


def portable_decision_sha256(value: Mapping[str, Any]) -> str:
    return _digest(portable_decision_payload(value))


@dataclass(frozen=True)
class ShardSpec:
    run_name: str
    shard: int
    global_hand_start: int
    hand_count: int = PAIRED_HANDS_PER_SHARD

    def __post_init__(self) -> None:
        if self.shard != 0:
            raise ValueError("Step 6a execution authorizes shard 0 only")
        if self.global_hand_start != 0 or self.hand_count != PAIRED_HANDS_PER_SHARD:
            raise ValueError("Step 6a shard 0 schedule changed")

    @property
    def root_count(self) -> int:
        return self.hand_count * 2


@dataclass(frozen=True)
class _SearchConfig:
    run_id: str
    continuation_seed: int
    candidate_seed: int
    evaluation_seed: int


def seed_values(global_hand_index: int) -> dict[str, int]:
    if not 0 <= global_hand_index < 250:
        raise ValueError("infrastructure canary index must be in 0..249")
    offset = SEED_STRIDE * global_hand_index
    return {
        "hand": HAND_SEED_BASE + offset,
        "behavior": BEHAVIOR_SEED_BASE + offset,
        "candidate": CANDIDATE_SEED_BASE + offset,
        "evaluation": EVALUATION_SEED_BASE + offset,
        "child": CHILD_SEED_BASE + offset,
        "confirmation": CONFIRMATION_SEED_BASE + offset,
    }


def _load_manifest_and_spec(
    *,
    manifest_path: Path,
    schedule_path: Path,
    source_package_sha256: str,
    shard: int,
) -> tuple[dict[str, Any], ShardSpec, str]:
    manifest = _load_json(manifest_path)
    manifest_sha = _sha256(manifest_path)
    if (
        manifest.get("schema") != STEP6A_PACKAGE_SCHEMA
        or manifest.get("source_sha256") != source_package_sha256
        or manifest.get("schedule_sha256") != _sha256(schedule_path)
        or manifest.get("total_shards") != 10
        or manifest.get("authorized_shards") != [0]
        or manifest.get("step5_contract_canonical_sha256")
        != STEP5_CONTRACT_CANONICAL_SHA256
        or manifest.get("current_profile_changed") is not False
        or manifest.get("production_fanout_authorized") is not False
    ):
        raise ValueError("Step 6a package manifest boundary changed")
    lines = schedule_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 10 or not 0 <= shard < len(lines):
        raise ValueError("Step 6a schedule changed")
    row = json.loads(lines[shard])
    if (
        not isinstance(row, dict)
        or row.get("schema") != STEP6A_SHARD_SCHEMA
        or row.get("run_name") != manifest.get("run_name")
        or row.get("shard") != shard
        or row.get("hand_count") != PAIRED_HANDS_PER_SHARD
        or row.get("root_count") != ROOTS_PER_SHARD
    ):
        raise ValueError("Step 6a shard row changed")
    spec = ShardSpec(
        run_name=str(row["run_name"]),
        shard=int(row["shard"]),
        global_hand_start=int(row["global_hand_start"]),
        hand_count=int(row["hand_count"]),
    )
    return manifest, spec, manifest_sha


def _verify_package_files(manifest: Mapping[str, Any], repository_root: Path) -> None:
    native = manifest.get("native_library")
    feature_encoder = manifest.get("feature_encoder_library")
    models = manifest.get("models")
    if (
        not isinstance(native, Mapping)
        or not isinstance(feature_encoder, Mapping)
        or not isinstance(models, Mapping)
    ):
        raise ValueError("Step 6a native/model manifest is missing")
    native_path = repository_root / str(native.get("path"))
    if not native_path.is_file() or _sha256(native_path) != native.get("sha256"):
        raise ValueError("Step 6a Linux native library changed")
    feature_path = repository_root / str(feature_encoder.get("path"))
    if not feature_path.is_file() or _sha256(feature_path) != feature_encoder.get(
        "sha256"
    ):
        raise ValueError("Step 6a Linux Stage3 feature encoder changed")
    for relative, expected in models.items():
        path = repository_root / str(relative)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Step 6a model changed: {relative}")


def run_linux_parity(
    *,
    manifest: Mapping[str, Any],
    repository_root: Path,
    parity_golden_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        existing = _load_json(output_path)
        if (
            existing.get("schema") != STEP6A_PARITY_SCHEMA
            or existing.get("all_gates_passed") is not True
        ):
            raise ValueError("existing Step 6a parity artifact did not pass")
        return existing
    golden = _load_json(parity_golden_path)
    native = manifest["native_library"]
    library = repository_root / str(native["path"])
    config = golden.get("runtime_config")
    rows = golden.get("rows")
    if not isinstance(config, Mapping) or not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("Step 6a parity golden changed")
    solver = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=str(native["sha256"]),
            library_path=library,
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
        decision = solver.solve(observation).to_dict()
        digest = portable_decision_sha256(decision)
        observed.append(
            {
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "expected_portable_sha256": row["portable_decision_sha256"],
                "observed_portable_sha256": digest,
                "match": digest == row["portable_decision_sha256"],
            }
        )
    gates = {
        "both_seats": [row["seat"] for row in observed] == ["first", "second"],
        "portable_action_values_exact": all(row["match"] for row in observed),
        "linux_native_hash_bound": solver.library_sha256 == native["sha256"],
        "no_profile_or_current_resolution": True,
    }
    report = {
        "schema": STEP6A_PARITY_SCHEMA,
        "status": "pass" if all(gates.values()) else "no_go",
        "native_library_sha256": solver.library_sha256,
        "rows": observed,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "current_profile_changed": False,
        "teacher_generation_started": False,
    }
    _write_once(output_path, report)
    if not report["all_gates_passed"]:
        raise RuntimeError("Step 6a Linux portable parity failed")
    return report


def _root_contract_digest(
    *, manifest_sha256: str, spec: ShardSpec, global_hand_index: int
) -> str:
    return _digest(
        {
            "manifest_sha256": manifest_sha256,
            "run_name": spec.run_name,
            "shard": spec.shard,
            "global_hand_index": global_hand_index,
            "seeds": seed_values(global_hand_index),
            "profile": behavior_profile_for_index(global_hand_index),
        }
    )


def _validate_root_task(
    value: Mapping[str, Any], *, digest: str, global_hand_index: int
) -> None:
    rows = value.get("observations")
    if (
        value.get("schema") != STEP6A_ROOT_TASK_SCHEMA
        or value.get("contract_digest") != digest
        or value.get("global_hand_index") != global_hand_index
        or value.get("profile") != behavior_profile_for_index(global_hand_index)
        or value.get("seeds") != seed_values(global_hand_index)
        or not isinstance(rows, list)
        or len(rows) != 2
        or [row.get("seat") for row in rows] != ["first", "second"]
    ):
        raise ValueError(f"Step 6a root task changed: {global_hand_index}")
    for row in rows:
        observation = ActorObservation.from_dict(row["observation"])
        if observation.fingerprint() != row.get(
            "observation_fingerprint"
        ) or _FORBIDDEN_FIELDS & set(row["observation"]):
            raise ValueError(f"Step 6a root observation changed: {global_hand_index}")


def _materialize_roots(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    spec: ShardSpec,
    root_dir: Path,
    heartbeat_path: Path,
) -> list[dict[str, Any]]:
    root_dir.mkdir(parents=True, exist_ok=True)
    expected_profiles = set(M31_T3_BEHAVIOR_PROFILES) - {"random_exact_final"}
    bundle = None
    tasks: list[dict[str, Any]] = []
    for local_index in range(spec.hand_count):
        global_index = spec.global_hand_start + local_index
        path = root_dir / f"hand_{global_index:03d}.json"
        digest = _root_contract_digest(
            manifest_sha256=manifest_sha256,
            spec=spec,
            global_hand_index=global_index,
        )
        if path.exists():
            value = _load_json(path)
            _validate_root_task(value, digest=digest, global_hand_index=global_index)
            tasks.append(value)
            continue
        if bundle is None:
            _write_heartbeat(
                heartbeat_path,
                status="loading_behavior_models",
                total_tasks=spec.hand_count,
                completed_tasks=len(tasks),
                pending_tasks=spec.hand_count - len(tasks),
            )
            bundle = load_model_bundle(ModelPaths(), profiles=expected_profiles)
        seeds = seed_values(global_index)
        profile = behavior_profile_for_index(global_index)
        observations = generate_behavior_t3_roots(
            hand_seed=seeds["hand"],
            behavior_seed=seeds["behavior"],
            profile=profile,
            bundle=bundle,
        )
        value = {
            "schema": STEP6A_ROOT_TASK_SCHEMA,
            "contract_digest": digest,
            "global_hand_index": global_index,
            "profile": profile,
            "seeds": seeds,
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
        )
    return tasks


def _search_worker(payload: Mapping[str, Any]) -> str:
    task_path = Path(str(payload["task_path"]))
    if task_path.exists():
        raise FileExistsError(f"Step 6a search task already exists: {task_path}")
    root_task = dict(payload["root_task"])
    seeds = dict(root_task["seeds"])
    global_index = int(root_task["global_hand_index"])
    library = Path(str(payload["library_path"]))
    library_sha = str(payload["library_sha256"])
    config = _SearchConfig(
        run_id=STEP6A_RUN_ID,
        continuation_seed=int(seeds["child"]),
        candidate_seed=int(seeds["candidate"]),
        evaluation_seed=int(seeds["evaluation"]),
    )
    solver = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=library_sha,
            library_path=library,
            run_id=config.run_id,
            candidate_samples=REFERENCE_BUDGET.candidate_samples,
            evaluation_samples=REFERENCE_BUDGET.evaluation_samples,
            downstream_t3_samples=REFERENCE_BUDGET.downstream_t3_samples,
            seed=config.continuation_seed,
            candidate_seed=config.candidate_seed,
            evaluation_seed=config.evaluation_seed,
        )
    )
    started = time.perf_counter()
    rows = []
    for seat_offset, raw in enumerate(root_task["observations"]):
        observation = ActorObservation.from_dict(raw["observation"])
        root_started = time.perf_counter()
        decision = solver.solve(observation)
        rows.append(
            {
                "root_index": global_index * 2 + seat_offset,
                "global_hand_index": global_index,
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "legal_action_count": len(decision.action_values),
                "wall_seconds": time.perf_counter() - root_started,
                "decision": decision.to_dict(),
                "integrity": _full_decision_integrity(
                    config, observation, decision, solver.library_sha256
                ),
            }
        )
    report = {
        "schema": STEP6A_SEARCH_TASK_SCHEMA,
        "contract_digest": root_task["contract_digest"],
        "global_hand_index": global_index,
        "profile": root_task["profile"],
        "seeds": seeds,
        "process_id": os.getpid(),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
        "wall_seconds": time.perf_counter() - started,
        "memory": process_memory_snapshot(),
        "engine": {
            "version": solver.engine_version,
            "library_sha256": solver.library_sha256,
            "build_or_fallback": False,
        },
        "rows": rows,
        "all_gates_passed": all(all(row["integrity"].values()) for row in rows),
    }
    _write_once(task_path, report)
    return task_path.name


def _validate_search_task(
    value: Mapping[str, Any], *, root_task: Mapping[str, Any], library_sha256: str
) -> None:
    if (
        value.get("schema") != STEP6A_SEARCH_TASK_SCHEMA
        or value.get("contract_digest") != root_task.get("contract_digest")
        or value.get("global_hand_index") != root_task.get("global_hand_index")
        or value.get("profile") != root_task.get("profile")
        or value.get("seeds") != root_task.get("seeds")
        or value.get("rayon_threads") != str(RAYON_THREADS)
        or value.get("engine", {}).get("library_sha256") != library_sha256
        or value.get("all_gates_passed") is not True
        or len(value.get("rows", [])) != 2
    ):
        raise ValueError(
            f"Step 6a search task changed: {root_task.get('global_hand_index')}"
        )


def _latency(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean_seconds": statistics.fmean(ordered),
        "p50_seconds": _percentile(ordered, 0.50),
        "p95_seconds": _percentile(ordered, 0.95),
        "p99_seconds": _percentile(ordered, 0.99),
        "max_seconds": max(ordered),
    }


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
    if shard not in manifest["authorized_shards"]:
        raise ValueError("Step 6a shard is not authorized")
    _verify_package_files(manifest, repository_root)
    parity = _load_json(parity_report_path)
    if (
        parity.get("schema") != STEP6A_PARITY_SCHEMA
        or parity.get("all_gates_passed") is not True
    ):
        raise ValueError("Step 6a Linux parity did not pass")
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output_dir / "heartbeat.json"
    root_tasks = _materialize_roots(
        manifest=manifest,
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
            value = _load_json(path)
            _validate_search_task(
                value, root_task=root_task, library_sha256=str(native["sha256"])
            )
            existing[index] = value
        else:
            missing.append(root_task)
    resumed_task_count = len(existing)
    selected = missing
    if stop_after_tasks is not None:
        if stop_after_tasks <= 0:
            raise ValueError("stop_after_tasks must be positive")
        selected = missing[:stop_after_tasks]
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
        # Linux defaults to ``fork``.  Forking after the parity check has loaded
        # the Rust engine also inherits Rayon's already-initialized global
        # thread-pool state, which can leave the child permanently asleep.
        # ``spawn`` gives every search worker a fresh Rust/Rayon process and is
        # also the same process-start contract on Windows and Linux.
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
            "schema": STEP6A_SUMMARY_SCHEMA,
            "status": "interrupted_for_resume_drill",
            "completed_tasks": spec.hand_count - len(remaining),
            "pending_tasks": len(remaining),
            "resumed_task_count": resumed_task_count,
            "spot_vm_started": True,
            "current_profile_changed": False,
            "production_fanout_authorized": False,
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

    reports = []
    for root_task in root_tasks:
        path = task_dir / f"hand_{int(root_task['global_hand_index']):03d}.json"
        value = _load_json(path)
        _validate_search_task(
            value, root_task=root_task, library_sha256=str(native["sha256"])
        )
        reports.append(value)
    rows = sorted(
        (row for report in reports for row in report["rows"]),
        key=lambda row: row["root_index"],
    )
    fingerprints = [row["observation_fingerprint"] for row in rows]
    profiles = [report["profile"] for report in reports]
    candidate_keys: set[str] = set()
    evaluation_keys: set[str] = set()
    for root_task in root_tasks:
        seeds = root_task["seeds"]
        for raw in root_task["observations"]:
            observation = ActorObservation.from_dict(raw["observation"])
            candidate_keys.update(
                particle.rng_key_digest
                for particle in sample_hidden_card_particles(
                    observation,
                    base_seed=seeds["candidate"],
                    run_id=STEP6A_RUN_ID,
                    sample_count=REFERENCE_BUDGET.candidate_samples,
                ).particles
            )
            evaluation_keys.update(
                particle.rng_key_digest
                for particle in sample_hidden_card_particles(
                    observation,
                    base_seed=seeds["evaluation"],
                    run_id=STEP6A_RUN_ID,
                    sample_count=REFERENCE_BUDGET.evaluation_samples,
                ).particles
            )
    latency = {
        seat: _latency(
            [float(row["wall_seconds"]) for row in rows if row["seat"] == seat]
        )
        for seat in ("first", "second")
    }
    peak_rss = max(int(report["memory"]["peak_rss_bytes"]) for report in reports)
    gates = {
        "exactly_50_roots": len(rows) == spec.root_count,
        "exactly_25_each_seat": (
            sum(row["seat"] == "first" for row in rows)
            == sum(row["seat"] == "second" for row in rows)
            == spec.hand_count
        ),
        "five_profiles_equal_quota": all(
            profiles.count(profile) == 5 for profile in M31_T3_BEHAVIOR_PROFILES
        ),
        "unique_observation_fingerprints": len(set(fingerprints)) == len(rows),
        "all_task_and_decision_integrity": all(
            all(row["integrity"].values()) for row in rows
        ),
        "candidate_evaluation_rng_disjoint": not (candidate_keys & evaluation_keys),
        "candidate_rng_unique": len(candidate_keys)
        == len(rows) * REFERENCE_BUDGET.candidate_samples,
        "evaluation_rng_unique": len(evaluation_keys)
        == len(rows) * REFERENCE_BUDGET.evaluation_samples,
        "linux_portable_parity": parity["all_gates_passed"],
        "resume_drill_recovered_task": resumed_task_count >= 1,
        "first_p95_within_180_seconds": latency["first"]["p95_seconds"]
        <= MAX_FIRST_P95_SECONDS,
        "second_p95_within_6_seconds": latency["second"]["p95_seconds"]
        <= MAX_SECOND_P95_SECONDS,
        "peak_rss_within_1_gib": peak_rss <= MAX_PEAK_RSS_BYTES,
        "canary_rows_not_training_eligible": True,
        "no_current_profile_or_production_fanout": True,
    }
    summary = {
        "schema": STEP6A_SUMMARY_SCHEMA,
        "status": "pass" if all(gates.values()) else "no_go",
        "run_name": spec.run_name,
        "shard": spec.shard,
        "source_package_sha256": source_package_sha256,
        "manifest_sha256": manifest_sha,
        "native_library_sha256": native["sha256"],
        "contract": {
            "paired_hands": spec.hand_count,
            "roots": spec.root_count,
            "workers": WORKERS,
            "rayon_threads_per_worker": RAYON_THREADS,
            "budget": REFERENCE_BUDGET.to_dict(),
            "run_id": STEP6A_RUN_ID,
        },
        "integrity": {
            "fingerprints": len(fingerprints),
            "unique_fingerprints": len(set(fingerprints)),
            "candidate_rng_keys": len(candidate_keys),
            "evaluation_rng_keys": len(evaluation_keys),
            "candidate_evaluation_overlap": len(candidate_keys & evaluation_keys),
            "profile_counts": {
                profile: profiles.count(profile) for profile in M31_T3_BEHAVIOR_PROFILES
            },
        },
        "performance": {
            "latency_by_seat": latency,
            "peak_process_rss_bytes": peak_rss,
        },
        "resumed_task_count": resumed_task_count,
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
        "training_eligible": False,
        "current_profile_changed": False,
        "spot_vm_started": True,
        "production_fanout_authorized": False,
        "m31_complete": False,
    }
    summary_path = output_dir / "summary.json"
    _write_once(summary_path, summary)
    _write_heartbeat(
        heartbeat_path,
        status=summary["status"],
        total_tasks=spec.hand_count,
        completed_tasks=spec.hand_count,
        pending_tasks=0,
        resumed_task_count=resumed_task_count,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest, _spec, _manifest_sha = _load_manifest_and_spec(
        manifest_path=args.manifest,
        schedule_path=args.schedule,
        source_package_sha256=args.source_package_sha256,
        shard=args.shard,
    )
    _verify_package_files(manifest, args.repository_root)
    parity_path = args.output_dir / "parity.json"
    parity = run_linux_parity(
        manifest=manifest,
        repository_root=args.repository_root,
        parity_golden_path=args.parity_golden,
        output_path=parity_path,
    )
    if args.parity_only:
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
    print(json.dumps(result, sort_keys=True))
    return 75 if result["status"] == "interrupted_for_resume_drill" else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PAIRED_HANDS_PER_SHARD",
    "ROOTS_PER_SHARD",
    "STEP6A_PACKAGE_SCHEMA",
    "STEP6A_SHARD_SCHEMA",
    "STEP6A_SUMMARY_SCHEMA",
    "ShardSpec",
    "portable_decision_payload",
    "portable_decision_sha256",
    "run_linux_parity",
    "run_shard",
    "seed_values",
]
