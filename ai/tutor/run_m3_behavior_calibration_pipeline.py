"""Resume the locked local M3 shard evaluation and calibration pipeline.

This runner deliberately starts *after* both preregistered collections are
complete.  Before it writes an evaluation shard it freshly verifies the
canonical gate-config file, requires that config to equal both the current
production default and the config embedded in the preregistration plan, and
checks every collection shard against the plan's exact root-range
commitments.  Evaluation shards are then appended through the existing
manifest-last evaluator and ``calibration.json`` is published only after both
evaluations are complete.

The command is restartable: an authenticated evaluation prefix is resumed,
an already complete evaluation is read-only verified, and an existing
calibration artifact is accepted only after a full raw-shard rebuild.  No
collection process is started or managed by this module.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import ai.tutor.evaluate_hu_behavior_trace_shards as evaluation_shards_module
from ai.tutor.behavior_calibration_contract import (
    canonical_json,
    canonical_snapshot,
)
from ai.tutor.behavior_logit_evaluator_torch import (
    build_known_hu_policy_value_logit_evaluators,
)
from ai.tutor.behavior_temperature_calibration import (
    PreTemperatureLegalLogitEvaluator,
    build_temperature_gate_config,
    verify_temperature_gate_config,
)
from ai.tutor.behavior_temperature_calibration_shards import (
    build_sharded_behavior_temperature_calibration,
    read_sharded_behavior_temperature_calibration,
    write_sharded_behavior_temperature_calibration,
)
from ai.tutor.collect_hu_behavior_trace_shards import (
    ShardedBehaviorTraceCollection,
    read_sharded_behavior_trace_collection,
)
from ai.tutor.evaluate_hu_behavior_trace_shards import (
    ShardedBehaviorTraceEvaluation,
    evaluate_hu_behavior_trace_shards,
    read_sharded_behavior_trace_evaluation,
)
from ai.tutor.plan_m3_behavior_collection import (
    PLAN_SCHEMA,
    verify_behavior_collection_plan,
)


PIPELINE_RESULT_SCHEMA = "ofc_m3_behavior_calibration_pipeline_result/v1"
DEFAULT_PLAN = "ai/reports/m3_behavior_collection_plan_20260713/plan.json"
DEFAULT_GATE_CONFIG = "ai/config/m3_behavior_temperature_gate_v2_20260713.json"
DEFAULT_RUN_DIR = "ai/data/m3_behavior_calibration_production_20260713"
PIPELINE_WRITER_LOCK_NAME = ".m3-behavior-calibration-pipeline.writer.lock"


_LOCAL_WRITER_LOCK_GUARD = threading.Lock()
_LOCAL_WRITER_LOCKS: set[str] = set()


def _initialize_pipeline_writer_lock(lock_path: Path) -> None:
    """Publish the permanent one-byte lock inode without a partial final name."""
    if lock_path.is_symlink():
        raise ValueError("pipeline writer lock must not be a symbolic link")
    if lock_path.exists():
        if not lock_path.is_file() or lock_path.stat().st_size < 1:
            raise ValueError("pipeline writer lock is not a valid lock file")
        return

    stage = lock_path.parent / f".{lock_path.name}.{uuid.uuid4().hex}.complete"
    try:
        with stage.open("xb", buffering=0) as handle:
            handle.write(b"\0")
            os.fsync(handle.fileno())
        try:
            # The hard link is an atomic no-replace publication on the same
            # filesystem as every evaluation manifest protected by this lock.
            os.link(stage, lock_path)
        except FileExistsError:
            pass
    finally:
        stage.unlink(missing_ok=True)

    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("pipeline writer lock is not a regular file")
    if lock_path.stat().st_size < 1:
        raise ValueError("pipeline writer lock is empty")


def _try_lock_pipeline_writer_file(handle: BinaryIO) -> bool:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        raise
    return True


def _unlock_pipeline_writer_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _pipeline_writer_lock(run_root: Path) -> Iterator[Path]:
    """Fail fast unless this process owns the run's sole writer lease.

    The OS advisory byte-range lock is released automatically when a process
    exits, including an abnormal exit.  The process-local registry also
    rejects concurrent threads before they can share the same OS process lock.
    """
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"pipeline run directory does not exist: {root}")
    lock_path = root / PIPELINE_WRITER_LOCK_NAME
    _initialize_pipeline_writer_lock(lock_path)
    lock_key = str(lock_path.resolve())

    with _LOCAL_WRITER_LOCK_GUARD:
        if lock_key in _LOCAL_WRITER_LOCKS:
            raise RuntimeError(
                f"another M3 calibration pipeline writer is active: {lock_path}"
            )
        _LOCAL_WRITER_LOCKS.add(lock_key)

    handle: BinaryIO | None = None
    acquired = False
    try:
        handle = lock_path.open("r+b", buffering=0)
        acquired = _try_lock_pipeline_writer_file(handle)
        if not acquired:
            raise RuntimeError(
                f"another M3 calibration pipeline writer is active: {lock_path}"
            )
        yield lock_path
    finally:
        try:
            if acquired and handle is not None:
                _unlock_pipeline_writer_file(handle)
        finally:
            if handle is not None:
                handle.close()
            with _LOCAL_WRITER_LOCK_GUARD:
                _LOCAL_WRITER_LOCKS.discard(lock_key)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(workspace_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


def _read_canonical_object(path: Path, *, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError(
            f"{label} must be one canonical JSON object with one trailing newline"
        )
    try:
        value = json.loads(raw[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != raw[:-1].decode(
        "utf-8"
    ):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _verify_gate_and_plan(
    *, workspace_root: Path, gate_config_path: Path, plan_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = verify_temperature_gate_config(
        _read_canonical_object(gate_config_path, label="temperature gate config")
    )
    current_default = verify_temperature_gate_config(
        build_temperature_gate_config()
    )
    if canonical_snapshot(gate) != canonical_snapshot(current_default):
        raise ValueError(
            "explicit gate config does not equal the current locked production default"
        )

    raw_plan = _read_canonical_object(plan_path, label="collection plan")
    if raw_plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported collection plan schema")
    plan = verify_behavior_collection_plan(
        raw_plan, gate_config=gate, workspace_root=workspace_root
    )
    embedded = plan.get("temperature_gate", {}).get("config")
    if canonical_snapshot(embedded) != canonical_snapshot(gate):
        raise ValueError("explicit gate config differs from preregistered plan")
    if plan["temperature_gate"].get("gate_config_sha256") != gate[
        "gate_config_sha256"
    ]:
        raise ValueError("plan gate-config SHA-256 differs from explicit gate config")
    return gate, plan


def _require_collection_matches_plan(
    *,
    label: str,
    collection: ShardedBehaviorTraceCollection,
    plan_section: Mapping[str, Any],
    source_contract: Mapping[str, Any],
) -> None:
    manifest = collection.manifest
    expected_config = canonical_snapshot(plan_section["collection_config"])
    expected_range = plan_section["range"]
    if collection.collection_complete is not True:
        raise ValueError(f"{label} collection is not complete and frozen")
    if canonical_snapshot(manifest["collection_config"]) != expected_config:
        raise ValueError(f"{label} collection config differs from preregistration")
    if (
        manifest["requested_total_root_target"] != expected_range["root_count"]
        or manifest["counters"]["root_count"] != expected_range["root_count"]
        or manifest["root_index_start"] != expected_range["root_index_start"]
        or manifest["root_index_stop_exclusive"]
        != expected_range["root_index_stop_exclusive"]
        or manifest["shard_size"] != expected_range["shard_size_roots"]
    ):
        raise ValueError(f"{label} collection range differs from preregistration")

    expected_shards = expected_range["shards"]
    actual_shards = manifest["shards"]
    if (
        expected_range["shard_count"] != len(expected_shards)
        or len(actual_shards) != len(expected_shards)
    ):
        raise ValueError(f"{label} collection shard count differs from preregistration")
    for index, (actual, expected) in enumerate(zip(actual_shards, expected_shards)):
        exact = {
            "index": expected["shard_index"],
            "name": f"shard-{index:06d}",
            "root_index_start": expected["root_index_start"],
            "root_index_stop_exclusive": expected["root_index_stop_exclusive"],
            "root_count": expected["root_count"],
            "root_id_order_sha256": expected["root_id_order_sha256"],
        }
        observed = {
            "index": actual["index"],
            "name": actual["name"],
            "root_index_start": actual["root_index_start"],
            "root_index_stop_exclusive": actual["root_index_stop_exclusive"],
            "root_count": actual["root_count"],
            "root_id_order_sha256": actual["root_id_order_sha256"],
        }
        if observed != exact:
            raise ValueError(
                f"{label} collection shard {index} differs from preregistration"
            )

    source_hashes = manifest["source_hashes"]
    if (
        source_hashes["behavior_contract_source_sha256"]
        != source_contract["calibration_contract_source_sha256"]
        or source_hashes["core_collector_source_sha256"]
        != source_contract["collector_source_sha256"]
    ):
        raise ValueError(f"{label} collection source differs from preregistration")


def _require_complete_collection_marker(path: Path, *, label: str) -> None:
    """Fail quickly on a growing collection before scanning its large shards."""
    manifest = _read_canonical_object(
        path / "manifest.json", label=f"{label} collection top manifest"
    )
    if manifest.get("collection_complete") is not True:
        raise ValueError(f"{label} collection is not complete and frozen")
    counters = manifest.get("counters")
    if not isinstance(counters, Mapping) or counters.get("root_count") != manifest.get(
        "requested_total_root_target"
    ):
        raise ValueError(f"{label} complete marker has inconsistent root counts")


def _read_existing_evaluation(
    *,
    collection_dir: Path,
    evaluation_dir: Path,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
) -> ShardedBehaviorTraceEvaluation | None:
    recovered = _recover_single_published_evaluation_shard(
        collection_dir=collection_dir,
        evaluation_dir=evaluation_dir,
        evaluators=evaluators,
    )
    if recovered is not None:
        return recovered
    top = evaluation_dir / "manifest.json"
    if not top.exists():
        return None
    return read_sharded_behavior_trace_evaluation(
        collection_dir, evaluation_dir, evaluators
    )


def _private_evaluation_staging_directories(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(
        child.name
        for child in output_dir.iterdir()
        if child.is_dir() and child.name.startswith(".evaluation-shard-")
    )


def _recover_single_published_evaluation_shard(
    *,
    collection_dir: Path,
    evaluation_dir: Path,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
) -> ShardedBehaviorTraceEvaluation | None:
    """Adopt exactly one fully verified next shard after a publisher crash.

    The evaluator publishes a final shard directory before republishing its
    top manifest.  A crash in that narrow interval leaves one otherwise valid
    next directory.  The evaluator intentionally rejects it as an orphan; the
    pipeline can recover it only after reconstructing the complete prefix and
    the candidate with the evaluator's own readers, bindings and builders.
    """
    if not evaluation_dir.exists():
        return None
    staging = _private_evaluation_staging_directories(evaluation_dir)
    if staging:
        raise ValueError(
            f"evaluation contains unpublished staging directories: {staging}"
        )
    actual_names = evaluation_shards_module._finalized_evaluation_directories(
        evaluation_dir
    )
    top_path = evaluation_dir / evaluation_shards_module.TOP_MANIFEST_NAME
    if not actual_names:
        return None

    lightweight_top: dict[str, Any] | None = None
    if top_path.exists():
        lightweight_top = evaluation_shards_module._read_canonical_manifest(
            top_path, label="evaluation top manifest"
        )
        raw_entries = lightweight_top.get("shards")
        if isinstance(raw_entries, list) and all(
            isinstance(entry, Mapping) for entry in raw_entries
        ):
            ordered_names = [
                entry.get("name") for entry in raw_entries
            ]
            canonical_names = [
                evaluation_shards_module._shard_name(index)
                for index in range(len(raw_entries))
            ]
            if ordered_names == canonical_names and actual_names == set(
                canonical_names
            ):
                # Normal state: let the public reader authenticate it once.
                return read_sharded_behavior_trace_evaluation(
                    collection_dir, evaluation_dir, evaluators
                )

    (
        collection,
        input_binding,
        evaluator_bindings,
        evaluator_set_sha,
        bindings_by_role,
    ) = evaluation_shards_module._verified_input_and_bindings(
        collection_dir, evaluators
    )
    source_hashes = evaluation_shards_module._source_hashes()
    aggregate = evaluation_shards_module._StreamAggregate.empty()
    entries: list[dict[str, Any]] = []
    original_top: dict[str, Any] | None = None

    if top_path.exists():
        original_top = lightweight_top
        if original_top is None:  # Defensive: top existence was re-observed.
            original_top = evaluation_shards_module._read_canonical_manifest(
                top_path, label="evaluation top manifest"
            )
        evaluation_shards_module._verify_top_envelope(original_top)
        if original_top["source_hashes"] != source_hashes:
            raise ValueError("evaluation source hash binding mismatch")
        if original_top["input_collection"] != input_binding:
            raise ValueError("evaluation input collection binding drift")
        if (
            original_top["evaluator_routes"] != evaluator_bindings
            or original_top["evaluator_set_sha256"] != evaluator_set_sha
        ):
            raise ValueError("evaluation evaluator/checkpoint binding drift")
        if (
            original_top["evaluation_contract"]
            != evaluation_shards_module._evaluation_contract()
        ):
            raise ValueError("evaluation contract drift")
        raw_entries = original_top["shards"]
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError("evaluation top manifest must reference at least one shard")
        if len(raw_entries) > len(collection.manifest["shards"]):
            raise ValueError("evaluation has more shards than its input collection")
        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, Mapping):
                raise TypeError("evaluation shard entry must be an object")
            entry = dict(raw_entry)
            dataset, input_entry = evaluation_shards_module._fresh_input_shard(
                collection, index
            )
            if entry.get("input_shard") != evaluation_shards_module._input_shard_binding(
                input_entry
            ):
                raise ValueError("evaluation shard points to the wrong input shard")
            rebuilt, _local = evaluation_shards_module._verify_one_evaluation_shard(
                shard_dir=evaluation_dir
                / evaluation_shards_module._shard_name(index),
                index=index,
                dataset=dataset,
                input_entry=input_entry,
                input_collection_content_sha256=input_binding[
                    "collection_content_sha256"
                ],
                evaluator_set_sha256=evaluator_set_sha,
                source_hashes=source_hashes,
                bindings_by_role=bindings_by_role,
                global_aggregate=aggregate,
            )
            if rebuilt != entry:
                raise ValueError(
                    "evaluation shard entry does not match immutable artifacts"
                )
            entries.append(entry)
        rebuilt_prefix = evaluation_shards_module._build_top_manifest(
            input_binding=input_binding,
            evaluator_bindings=evaluator_bindings,
            evaluator_set_sha256=evaluator_set_sha,
            source_hashes=source_hashes,
            entries=entries,
            aggregate=aggregate,
        )
        if rebuilt_prefix != original_top:
            raise ValueError(
                "evaluation top manifest does not match verified shard artifacts"
            )

    expected_prefix = {
        evaluation_shards_module._shard_name(index)
        for index in range(len(entries))
    }
    missing = expected_prefix - actual_names
    extras = actual_names - expected_prefix
    if missing:
        raise ValueError(
            f"evaluation recovery prefix has missing finalized shards: {sorted(missing)}"
        )
    if not extras:
        return None
    if len(extras) != 1:
        raise ValueError(
            f"evaluation recovery requires exactly one next orphan; found {sorted(extras)}"
        )
    next_index = len(entries)
    if next_index >= len(collection.manifest["shards"]):
        raise ValueError("complete evaluation cannot adopt another shard")
    next_name = evaluation_shards_module._shard_name(next_index)
    if extras != {next_name}:
        raise ValueError(
            f"evaluation orphan is not the exact next shard: {sorted(extras)}"
        )

    dataset, input_entry = evaluation_shards_module._fresh_input_shard(
        collection, next_index
    )
    recovered_entry, _local = evaluation_shards_module._verify_one_evaluation_shard(
        shard_dir=evaluation_dir / next_name,
        index=next_index,
        dataset=dataset,
        input_entry=input_entry,
        input_collection_content_sha256=input_binding[
            "collection_content_sha256"
        ],
        evaluator_set_sha256=evaluator_set_sha,
        source_hashes=source_hashes,
        bindings_by_role=bindings_by_role,
        global_aggregate=aggregate,
    )
    entries.append(recovered_entry)
    recovered_top = evaluation_shards_module._build_top_manifest(
        input_binding=input_binding,
        evaluator_bindings=evaluator_bindings,
        evaluator_set_sha256=evaluator_set_sha,
        source_hashes=source_hashes,
        entries=entries,
        aggregate=aggregate,
    )

    # Recheck the publication boundary immediately before changing the top.
    if _private_evaluation_staging_directories(evaluation_dir):
        raise ValueError("evaluation staging appeared during orphan recovery")
    if (
        evaluation_shards_module._finalized_evaluation_directories(evaluation_dir)
        != expected_prefix | {next_name}
    ):
        raise ValueError("evaluation shard set changed during orphan recovery")
    if original_top is None:
        if top_path.exists():
            # A concurrent adopter won.  Accept only its fully verified result.
            return read_sharded_behavior_trace_evaluation(
                collection_dir, evaluation_dir, evaluators
            )
    else:
        current_top = evaluation_shards_module._read_canonical_manifest(
            top_path, label="evaluation top manifest"
        )
        if current_top != original_top:
            return read_sharded_behavior_trace_evaluation(
                collection_dir, evaluation_dir, evaluators
            )
    evaluation_shards_module._atomic_write_top(top_path, recovered_top)
    return read_sharded_behavior_trace_evaluation(
        collection_dir, evaluation_dir, evaluators
    )


def _advance_evaluation(
    *,
    collection: ShardedBehaviorTraceCollection,
    collection_dir: Path,
    evaluation_dir: Path,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
    existing: ShardedBehaviorTraceEvaluation | None,
    max_new_shards: int | None,
) -> ShardedBehaviorTraceEvaluation:
    if existing is not None and existing.evaluation_complete:
        return existing
    remaining = collection.shard_count - (
        0 if existing is None else existing.shard_count
    )
    if remaining <= 0:
        raise ValueError("incomplete evaluation has no remaining input shards")
    limit = remaining if max_new_shards is None else min(remaining, max_new_shards)
    return evaluate_hu_behavior_trace_shards(
        collection_dir,
        evaluation_dir,
        evaluators,
        resume=existing is not None,
        max_new_shards=limit,
    )


def _publish_calibration_no_replace(
    *,
    artifact: Mapping[str, Any],
    output: Path,
    natural_collection_dir: Path,
    natural_evaluation_dir: Path,
    challenge_collection_dir: Path,
    challenge_evaluation_dir: Path,
    evaluators: Mapping[tuple[int, str], PreTemperatureLegalLogitEvaluator],
    scratch_dir: Path,
) -> tuple[dict[str, Any], bool]:
    """Publish a complete same-filesystem stage without replacing ``output``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid.uuid4().hex}.complete"
    if stage.exists():  # UUID collision or a hostile precreation: never reuse it.
        raise FileExistsError(f"private calibration stage already exists: {stage}")
    try:
        write_sharded_behavior_temperature_calibration(artifact, stage)
        staged = _read_canonical_object(stage, label="staged calibration artifact")
        if canonical_snapshot(staged) != canonical_snapshot(artifact):
            raise IOError("staged calibration artifact differs from built artifact")
        try:
            # A hard link exposes the already-complete inode atomically and
            # fails if the final name exists; unlike replace it cannot clobber.
            os.link(stage, output)
        except FileExistsError:
            existing = read_sharded_behavior_temperature_calibration(
                output,
                natural_collection_dir,
                natural_evaluation_dir,
                challenge_collection_dir,
                challenge_evaluation_dir,
                evaluators,
                scratch_dir=scratch_dir,
            )
            if canonical_snapshot(existing) != canonical_snapshot(artifact):
                raise FileExistsError(
                    "concurrent calibration artifact differs from built artifact"
                )
            return existing, False
        published = _read_canonical_object(
            output, label="published calibration artifact"
        )
        if canonical_snapshot(published) != canonical_snapshot(artifact):
            raise IOError("no-replace calibration publication changed artifact bytes")
        return dict(artifact), True
    finally:
        stage.unlink(missing_ok=True)


def _collection_summary(
    collection: ShardedBehaviorTraceCollection,
) -> dict[str, Any]:
    manifest = collection.manifest
    return {
        "root_count": collection.root_count,
        "decision_count": collection.decision_count,
        "shard_count": collection.shard_count,
        "collection_content_sha256": manifest["collection_content_sha256"],
        "layout_sha256": manifest["layout_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _evaluation_summary(
    evaluation: ShardedBehaviorTraceEvaluation | None,
) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    manifest = evaluation.manifest
    return {
        "evaluation_complete": evaluation.evaluation_complete,
        "row_count": evaluation.row_count,
        "shard_count": evaluation.shard_count,
        "evaluation_content_sha256": manifest["evaluation_content_sha256"],
        "layout_sha256": manifest["layout_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluator_set_sha256": manifest["evaluator_set_sha256"],
    }


def _base_result(
    *,
    workspace_root: Path,
    run_dir: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    gate_config_path: Path,
    gate: Mapping[str, Any],
    natural_collection: ShardedBehaviorTraceCollection,
    challenge_collection: ShardedBehaviorTraceCollection,
    natural_evaluation: ShardedBehaviorTraceEvaluation | None,
    challenge_evaluation: ShardedBehaviorTraceEvaluation | None,
) -> dict[str, Any]:
    return {
        "schema": PIPELINE_RESULT_SCHEMA,
        "workspace_root": str(workspace_root),
        "run_dir": str(run_dir),
        "plan": {
            "path": str(plan_path),
            "file_sha256": _sha256_file(plan_path),
            "plan_sha256": plan["plan_sha256"],
        },
        "gate_config": {
            "path": str(gate_config_path),
            "file_sha256": _sha256_file(gate_config_path),
            "gate_config_sha256": gate["gate_config_sha256"],
        },
        "collections": {
            "natural": _collection_summary(natural_collection),
            "challenge": _collection_summary(challenge_collection),
        },
        "evaluations": {
            "natural": _evaluation_summary(natural_evaluation),
            "challenge": _evaluation_summary(challenge_evaluation),
        },
    }


def _run_m3_behavior_calibration_pipeline_locked(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
    plan_path: str | Path,
    gate_config_path: str | Path,
    output_path: str | Path | None = None,
    scratch_dir: str | Path | None = None,
    max_new_evaluation_shards: int | None = None,
    evaluators: Mapping[
        tuple[int, str], PreTemperatureLegalLogitEvaluator
    ] | None = None,
) -> dict[str, Any]:
    """Run the pipeline while the caller owns the run-wide writer lock."""
    if max_new_evaluation_shards is not None and (
        isinstance(max_new_evaluation_shards, bool)
        or not isinstance(max_new_evaluation_shards, int)
        or max_new_evaluation_shards <= 0
    ):
        raise ValueError("max_new_evaluation_shards must be a positive integer")

    workspace = Path(workspace_root).resolve()
    run_root = _resolve(workspace, run_dir)
    plan_file = _resolve(workspace, plan_path)
    gate_file = _resolve(workspace, gate_config_path)
    output = (
        run_root / "calibration.json"
        if output_path is None
        else _resolve(workspace, output_path)
    )
    scratch = (
        run_root / "calibration_scratch"
        if scratch_dir is None
        else _resolve(workspace, scratch_dir)
    )
    natural_collection_dir = run_root / "natural"
    natural_evaluation_dir = run_root / "natural_evaluation"
    challenge_collection_dir = run_root / "challenge"
    challenge_evaluation_dir = run_root / "challenge_evaluation"
    artifact_roots = (
        natural_collection_dir.resolve(),
        natural_evaluation_dir.resolve(),
        challenge_collection_dir.resolve(),
        challenge_evaluation_dir.resolve(),
    )
    if len(set(artifact_roots)) != len(artifact_roots):
        raise ValueError("collection/evaluation roots must be distinct")
    if output.resolve() in {plan_file, gate_file}:
        raise ValueError("calibration output must not replace plan or gate config")
    for root in artifact_roots:
        if output.resolve() == root or root in output.resolve().parents:
            raise ValueError("calibration output must stay outside artifact roots")
        if scratch.resolve() == root or root in scratch.resolve().parents:
            raise ValueError("scratch directory must stay outside artifact roots")

    # This is the pre-label lock: no evaluator or output directory is touched
    # before the explicit canonical gate and preregistered plan agree exactly.
    gate, plan = _verify_gate_and_plan(
        workspace_root=workspace,
        gate_config_path=gate_file,
        plan_path=plan_file,
    )
    _require_complete_collection_marker(
        natural_collection_dir, label="natural"
    )
    _require_complete_collection_marker(
        challenge_collection_dir, label="challenge"
    )
    natural_collection = read_sharded_behavior_trace_collection(
        natural_collection_dir
    )
    challenge_collection = read_sharded_behavior_trace_collection(
        challenge_collection_dir
    )
    source_contract = plan["source_contract"]
    _require_collection_matches_plan(
        label="natural",
        collection=natural_collection,
        plan_section=plan["natural"],
        source_contract=source_contract,
    )
    _require_collection_matches_plan(
        label="challenge",
        collection=challenge_collection,
        plan_section=plan["joker_challenge"],
        source_contract=source_contract,
    )
    if natural_collection.manifest["policy"] != challenge_collection.manifest[
        "policy"
    ]:
        raise ValueError("natural and challenge behavior-policy bindings differ")

    route_evaluators = (
        build_known_hu_policy_value_logit_evaluators(workspace)
        if evaluators is None
        else evaluators
    )

    # Freshly authenticate every already-published prefix before appending.
    challenge_evaluation = _read_existing_evaluation(
        collection_dir=challenge_collection_dir,
        evaluation_dir=challenge_evaluation_dir,
        evaluators=route_evaluators,
    )
    natural_evaluation = _read_existing_evaluation(
        collection_dir=natural_collection_dir,
        evaluation_dir=natural_evaluation_dir,
        evaluators=route_evaluators,
    )
    if output.exists() and not (
        challenge_evaluation is not None
        and challenge_evaluation.evaluation_complete
        and natural_evaluation is not None
        and natural_evaluation.evaluation_complete
    ):
        raise ValueError(
            "published calibration cannot coexist with missing/incomplete evaluation"
        )

    challenge_evaluation = _advance_evaluation(
        collection=challenge_collection,
        collection_dir=challenge_collection_dir,
        evaluation_dir=challenge_evaluation_dir,
        evaluators=route_evaluators,
        existing=challenge_evaluation,
        max_new_shards=max_new_evaluation_shards,
    )
    if not challenge_evaluation.evaluation_complete:
        result = _base_result(
            workspace_root=workspace,
            run_dir=run_root,
            plan_path=plan_file,
            plan=plan,
            gate_config_path=gate_file,
            gate=gate,
            natural_collection=natural_collection,
            challenge_collection=challenge_collection,
            natural_evaluation=natural_evaluation,
            challenge_evaluation=challenge_evaluation,
        )
        result.update(
            {
                "stage": "challenge_evaluation_incomplete",
                "pipeline_complete": False,
                "calibration": None,
            }
        )
        return result

    natural_evaluation = _advance_evaluation(
        collection=natural_collection,
        collection_dir=natural_collection_dir,
        evaluation_dir=natural_evaluation_dir,
        evaluators=route_evaluators,
        existing=natural_evaluation,
        max_new_shards=max_new_evaluation_shards,
    )
    if not natural_evaluation.evaluation_complete:
        result = _base_result(
            workspace_root=workspace,
            run_dir=run_root,
            plan_path=plan_file,
            plan=plan,
            gate_config_path=gate_file,
            gate=gate,
            natural_collection=natural_collection,
            challenge_collection=challenge_collection,
            natural_evaluation=natural_evaluation,
            challenge_evaluation=challenge_evaluation,
        )
        result.update(
            {
                "stage": "natural_evaluation_incomplete",
                "pipeline_complete": False,
                "calibration": None,
            }
        )
        return result

    if output.exists():
        artifact = read_sharded_behavior_temperature_calibration(
            output,
            natural_collection_dir,
            natural_evaluation_dir,
            challenge_collection_dir,
            challenge_evaluation_dir,
            route_evaluators,
            scratch_dir=scratch,
        )
        artifact_created = False
        if canonical_snapshot(artifact["gate_config"]) != canonical_snapshot(gate):
            raise ValueError("existing calibration uses a different gate config")
    else:
        artifact = build_sharded_behavior_temperature_calibration(
            natural_collection_dir,
            natural_evaluation_dir,
            challenge_collection_dir,
            challenge_evaluation_dir,
            route_evaluators,
            gate_config=gate,
            scratch_dir=scratch,
        )
        artifact, artifact_created = _publish_calibration_no_replace(
            artifact=artifact,
            output=output,
            natural_collection_dir=natural_collection_dir,
            natural_evaluation_dir=natural_evaluation_dir,
            challenge_collection_dir=challenge_collection_dir,
            challenge_evaluation_dir=challenge_evaluation_dir,
            evaluators=route_evaluators,
            scratch_dir=scratch,
        )

    result = _base_result(
        workspace_root=workspace,
        run_dir=run_root,
        plan_path=plan_file,
        plan=plan,
        gate_config_path=gate_file,
        gate=gate,
        natural_collection=natural_collection,
        challenge_collection=challenge_collection,
        natural_evaluation=natural_evaluation,
        challenge_evaluation=challenge_evaluation,
    )
    result.update(
        {
            "stage": "calibration_complete",
            "pipeline_complete": True,
            "calibration": {
                "path": str(output.resolve()),
                "created": artifact_created,
                "file_sha256": _sha256_file(output),
                "artifact_sha256": artifact["artifact_sha256"],
                "gate_result_sha256": artifact["gate_result"][
                    "gate_result_sha256"
                ],
                "promotion_eligible": artifact["promotion_eligible"],
            },
        }
    )
    return result


def run_m3_behavior_calibration_pipeline(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
    plan_path: str | Path,
    gate_config_path: str | Path,
    output_path: str | Path | None = None,
    scratch_dir: str | Path | None = None,
    max_new_evaluation_shards: int | None = None,
    evaluators: Mapping[
        tuple[int, str], PreTemperatureLegalLogitEvaluator
    ] | None = None,
) -> dict[str, Any]:
    """Run or resume the pipeline under one crash-released writer lease.

    The lease spans prefix authentication, orphan recovery, evaluation shard
    publication, and calibration publication.  Consequently a delayed second
    runner can never replace a newer top manifest with its stale reconstruction.
    """
    workspace = Path(workspace_root).resolve()
    run_root = _resolve(workspace, run_dir)
    with _pipeline_writer_lock(run_root):
        return _run_m3_behavior_calibration_pipeline_locked(
            workspace_root=workspace,
            run_dir=run_root,
            plan_path=plan_path,
            gate_config_path=gate_config_path,
            output_path=output_path,
            scratch_dir=scratch_dir,
            max_new_evaluation_shards=max_new_evaluation_shards,
            evaluators=evaluators,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=Path.cwd(), type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--gate-config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--max-new-evaluation-shards", type=int)
    args = parser.parse_args(argv)
    result = run_m3_behavior_calibration_pipeline(
        workspace_root=args.workspace_root,
        run_dir=args.run_dir,
        plan_path=args.plan,
        gate_config_path=args.gate_config,
        output_path=args.output,
        scratch_dir=args.scratch_dir,
        max_new_evaluation_shards=args.max_new_evaluation_shards,
    )
    print(canonical_json(result))
    return 0


__all__ = [
    "DEFAULT_GATE_CONFIG",
    "DEFAULT_PLAN",
    "DEFAULT_RUN_DIR",
    "PIPELINE_RESULT_SCHEMA",
    "main",
    "run_m3_behavior_calibration_pipeline",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
