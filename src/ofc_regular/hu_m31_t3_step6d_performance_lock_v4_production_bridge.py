"""Fail-closed finalization bridge for the Candidate02 performance lock v4.

The pure v4 merger deliberately leaves transport pending.  This module closes
that boundary only after replaying the immutable wave-v2 accepted inventory,
controller lifecycle chain, merge view, and the source package that contained
the v4 plan, materialization receipt, seal, and all roots before launch.

There is no cloud client here.  The only write is an optional create-only final
receipt.  No AI profile is selected, added, or modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from . import hu_m31_t3_step6d_candidate02_performance_lock_v4_plan as lock_plan
from . import hu_m31_t3_step6d_full100_wave_package_v2 as outer_package
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry
from . import hu_m31_t3_step6d_full100_wave_scientific_bridge_v2 as wave_bridge
from . import hu_m31_t3_step6d_performance_lock_v4_spot_package as v4_package
from . import merge_hu_m31_t3_step6d_candidate02_performance_lock_v4 as pure_v4
from . import run_hu_m31_t3_step6d_performance_v2 as runner


RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_v4_production_final_receipt_v1"
)
QUALIFIED_DECISION = (
    "performance_lock_v4_finalized_qualified_open_quality_pilot_only"
)
NO_GO_DECISION = "performance_lock_v4_candidate_finalized_no_go"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PATH_KEYS = frozenset(
    {
        "pure_merge",
        "performance_lock_plan",
        "materialization_receipt",
        "root_seal",
        "outer_package",
        "merge_view_manifest",
        "current_profile_registry",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "source_paths",
        "source_file_sha256",
        "performance_lock_plan",
        "materialization_receipt",
        "root_seal",
        "pure_v4_merge",
        "wave_plan",
        "attempt_ledger",
        "accepted_results_snapshot",
        "validated_lifecycle_chain",
        "merge_view_manifest",
        "prelaunch_root_chronology",
        "one_shot_attempt_audit",
        "process_isolation_audit",
        "transport_audit",
        "all_gates_passed",
        "performance_lock_finalized",
        "one_shot_lock_consumed",
        "transport_lineage_validated",
        "performance_lock_qualified",
        "candidate_finalized_no_go",
        "quality_pilot_authorized",
        "rerun_authorized",
        "reseed_authorized",
        "artifact_fanout_authorized",
        "training_eligible",
        "training_authorized",
        "promotion_evidence",
        "promotion_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
        "bridge_cloud_network_invoked",
        "receipt_sha256",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return runner.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required plain file is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_file(path: str | Path, label: str) -> Path:
    target = Path(path)
    if not target.is_absolute() or target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} must be an absolute non-symlink file")
    return target.resolve()


def _plain_directory(path: str | Path, label: str) -> Path:
    target = Path(path)
    if not target.is_absolute() or target.is_symlink() or not target.is_dir():
        raise ValueError(f"{label} must be an absolute non-symlink directory")
    return target.resolve()


def _read_canonical(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    target = _plain_file(path, label)
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical LF JSON")
    return value, target


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_source_paths(value: Mapping[str, Any]) -> dict[str, str]:
    paths = dict(value)
    if set(paths) != _SOURCE_PATH_KEYS:
        raise ValueError("production bridge source paths changed")
    result: dict[str, str] = {}
    for key, raw in paths.items():
        path = Path(str(raw))
        if not path.is_absolute():
            raise ValueError(f"{key} source path is not absolute")
        result[key] = str(path.resolve())
    return result


def _load_core_sources(
    source_paths: Mapping[str, Any],
    *,
    expected_profile_sha256: str,
    replay_sources: bool,
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if replay_sources is not True:
        raise PermissionError("final production receipt requires full source replay")
    expected_profile = _require_sha(
        expected_profile_sha256, "expected profile SHA-256"
    )
    paths = _validate_source_paths(source_paths)
    pure_raw, pure_path = _read_canonical(paths["pure_merge"], "pure v4 merge")
    plan_raw, plan_path = _read_canonical(
        paths["performance_lock_plan"], "performance-lock-v4 plan"
    )
    material_raw, material_path = _read_canonical(
        paths["materialization_receipt"], "v4 materialization receipt"
    )
    seal_raw, seal_path = _read_canonical(paths["root_seal"], "v4 root seal")
    merge_raw, merge_path = _read_canonical(
        paths["merge_view_manifest"], "wave-v2 merge-view manifest"
    )
    outer_root = _plain_directory(paths["outer_package"], "outer package")
    profile_path = _plain_file(
        paths["current_profile_registry"], "current profile registry"
    )
    profile_sha = _sha256_file(profile_path)
    if (
        expected_profile != lock_plan.CURRENT_PROFILE_REGISTRY_SHA256
        or profile_sha != expected_profile
    ):
        raise PermissionError("current profile registry differs from the frozen pin")

    plan = lock_plan.validate_performance_lock_v4_plan(plan_raw)
    materialization = lock_plan.validate_materialization_receipt(material_raw)
    seal = lock_plan.validate_root_seal(seal_raw)
    pure = pure_v4.validate_candidate02_performance_lock_v4_value(
        pure_raw, replay_sources=True
    )
    if (
        _sha256_file(plan_path) != lock_plan.PLAN_SHA256
        or pure.get("performance_lock_plan") != plan
        or pure.get("materialization_receipt") != materialization
        or pure.get("root_seal") != seal
        or materialization.get("plan_sha256") != lock_plan.PLAN_SHA256
        or seal.get("plan_sha256") != lock_plan.PLAN_SHA256
        or seal.get("materialization_receipt_sha256")
        != canonical_sha256(materialization)
        or materialization.get("claim_sha256") != seal.get("claim_sha256")
    ):
        raise ValueError("pure merge and real v4 plan/materialization/seal differ")
    hashes = {
        "pure_merge": _sha256_file(pure_path),
        "performance_lock_plan": _sha256_file(plan_path),
        "materialization_receipt": _sha256_file(material_path),
        "root_seal": _sha256_file(seal_path),
        "merge_view_manifest": _sha256_file(merge_path),
        "current_profile_registry": profile_sha,
    }
    normalized_paths = {
        **paths,
        "pure_merge": str(pure_path),
        "performance_lock_plan": str(plan_path),
        "materialization_receipt": str(material_path),
        "root_seal": str(seal_path),
        "outer_package": str(outer_root),
        "merge_view_manifest": str(merge_path),
        "current_profile_registry": str(profile_path),
    }
    return (
        normalized_paths,
        hashes,
        plan,
        materialization,
        seal,
        pure,
    )


def _read_archived_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != v4_package.canonical_bytes(value):
        raise ValueError(f"{label} is not canonical LF JSON")
    return value


def _read_outer_wave_plan(
    path: str | Path,
    *,
    expected_wave_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Read the copied plan with its deliberately LF-terminated serializer."""

    target = outer_package._safe_existing_file(  # type: ignore[attr-defined]
        path, "outer-package wave plan"
    )
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "outer-package wave plan is not canonical LF JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or raw != wave_v2.canonical_bytes(value)
    ):
        raise ValueError(
            "outer-package wave plan is not canonical LF JSON"
        )
    copied = wave_v2.validate_wave_plan(value)
    expected = wave_v2.validate_wave_plan(expected_wave_plan)
    expected_raw = wave_v2.canonical_bytes(expected)
    if copied != expected or raw != expected_raw:
        raise ValueError(
            "outer-package wave plan differs from its frozen source"
        )
    return copied


def _validate_prelaunch_root_chronology(
    *,
    outer_package_root: Path,
    wave_plan: Mapping[str, Any],
    accepted_snapshot: Mapping[str, Any],
    lifecycle_chain: Mapping[str, Any],
    merge_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    materialization: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove sealed root bytes were in immutable content before launch."""

    startup_sha256 = science_registry.resolve_startup_sha256(wave_plan)
    outer = outer_package.validate_outer_package(
        outer_package_root,
        wave_plan,
        expected_startup_sha256=startup_sha256,
    )
    science_manifest_path = (
        outer_package_root / outer_package.SCIENTIFIC_MANIFEST_PATH
    )
    source_archive_path = outer_package_root / outer_package.SOURCE_PATH
    science_manifest_raw, _ = _read_canonical(
        science_manifest_path, "archived v4 package manifest"
    )
    science_manifest = v4_package._validate_manifest_value(
        science_manifest_raw
    )
    source_archive = _plain_file(
        source_archive_path, "archived v4 scientific source"
    )
    archived_payloads = v4_package._validate_archive(
        source_archive, science_manifest
    )
    archived_plan = v4_package._validate_archived_science(
        archived_payloads, science_manifest
    )
    archived_materialization = lock_plan.validate_materialization_receipt(
        _read_archived_json(
            archived_payloads[v4_package.MATERIALIZATION_ARCHIVE_PATH],
            "archived v4 materialization",
        )
    )
    archived_seal = lock_plan.validate_root_seal(
        _read_archived_json(
            archived_payloads[v4_package.SEAL_ARCHIVE_PATH],
            "archived v4 root seal",
        )
    )
    archived_plan_file = lock_plan.validate_performance_lock_v4_plan(
        _read_archived_json(
            archived_payloads[v4_package.PLAN_ARCHIVE_PATH],
            "archived v4 plan",
        )
    )
    copied_wave_plan = _read_outer_wave_plan(
        outer_package_root / outer_package.WAVE_PLAN_PATH,
        expected_wave_plan=wave_plan,
    )
    proofs = lifecycle_chain.get("wave_proofs")
    pair_rows = merge_manifest.get("pair_launch_lineage_audit", {}).get("pairs")
    if not isinstance(proofs, list) or not isinstance(pair_rows, list):
        raise ValueError("prelaunch lifecycle or pair evidence is missing")
    prelaunch_hashes: list[str] = []
    for proof in proofs:
        if (
            not isinstance(proof, Mapping)
            or proof.get("actual_launch_receipt_present") is not True
            or proof.get("actual_launch_receipt_revalidated") is not True
        ):
            raise ValueError("execution lacks a revalidated actual launch receipt")
        prelaunch_hashes.append(
            _require_sha(
                proof.get("prelaunch_authorization_sha256"),
                "prelaunch authorization",
            )
        )
    if (
        archived_plan != plan
        or archived_plan_file != plan
        or archived_materialization != materialization
        or archived_seal != seal
        or copied_wave_plan != wave_plan
        or wave_plan.get("full100_plan") != plan
        or wave_plan.get("full100_plan_sha256") != lock_plan.PLAN_SHA256
        or wave_plan.get("run_contract_digest") != lock_plan.RUN_CONTRACT_DIGEST
        or outer.get("full100_plan_sha256") != lock_plan.PLAN_SHA256
        or outer.get("run_contract_digest") != lock_plan.RUN_CONTRACT_DIGEST
        or outer.get("expected_startup_sha256") != startup_sha256
        or accepted_snapshot.get("expected_startup_sha256") != startup_sha256
        or outer.get("content_payload_sha256")
        != accepted_snapshot.get("content_payload_sha256")
        or outer.get("content_payload_sha256")
        not in {row.get("content_payload_sha256") for row in pair_rows}
        or len({row.get("content_payload_sha256") for row in pair_rows}) != 1
        or outer.get("manifest_sha256")
        != accepted_snapshot.get("outer_manifest_sha256")
        or outer.get("manifest_sha256")
        not in {row.get("outer_manifest_sha256") for row in pair_rows}
        or len({row.get("outer_manifest_sha256") for row in pair_rows}) != 1
        or science_manifest.get("plan_sha256") != lock_plan.PLAN_SHA256
        or science_manifest.get("materialization_receipt_sha256")
        != canonical_sha256(materialization)
        or science_manifest.get("root_seal_sha256") != canonical_sha256(seal)
        or science_manifest.get("root_hash_aggregate_sha256")
        != seal.get("root_hash_aggregate_sha256")
        or science_manifest.get("observation_fingerprint_aggregate_sha256")
        != seal.get("observation_fingerprint_aggregate_sha256")
        or science_manifest.get("seed_set_sha256") != seal.get("seed_set_sha256")
        or outer["scientific_lineage"]["source_sha256"]
        != science_manifest["source_sha256"]
        or _sha256_file(source_archive) != science_manifest["source_sha256"]
        or len(prelaunch_hashes) != len(set(prelaunch_hashes))
        or any(
            value is not False
            for value in (
                materialization.get("cloud_started"),
                seal.get("cloud_started"),
                science_manifest.get("cloud_started"),
                outer.get("cloud_started"),
            )
        )
    ):
        raise ValueError("sealed roots were not fixed in prelaunch content")
    body = {
        "schema": (
            "hu_m31_t3_step6d_performance_lock_v4_prelaunch_root_chronology_v1"
        ),
        "status": "sealed_roots_in_immutable_content_before_every_launch",
        "plan_sha256": lock_plan.PLAN_SHA256,
        "materialization_receipt_sha256": canonical_sha256(materialization),
        "root_seal_sha256": canonical_sha256(seal),
        "root_hash_aggregate_sha256": seal["root_hash_aggregate_sha256"],
        "observation_fingerprint_aggregate_sha256": seal[
            "observation_fingerprint_aggregate_sha256"
        ],
        "seed_set_sha256": seal["seed_set_sha256"],
        "startup_sha256": startup_sha256,
        "scientific_source_sha256": science_manifest["source_sha256"],
        "outer_manifest_sha256": outer["manifest_sha256"],
        "content_payload_sha256": outer["content_payload_sha256"],
        "prelaunch_authorization_sha256": prelaunch_hashes,
        "prelaunch_authorization_count": len(prelaunch_hashes),
        "archived_root_file_count": 100,
        "root_sealed_before_launch": True,
        "outer_content_create_only": True,
        "current_profile_changed": False,
    }
    return {**body, "chronology_sha256": canonical_sha256(body)}


def _validate_one_shot_attempts(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    lifecycle_chain: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    expected_jobs = list(plan["coverage"]["job_ids"])
    transitions = ledger.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("attempt ledger has no transitions")
    history = transitions[-1].get("attempt_history")
    if not isinstance(history, list) or len(history) != 20:
        raise ValueError("final attempt ledger must cover exactly twenty jobs")
    history_by_job: dict[str, Mapping[str, Any]] = {}
    for raw in history:
        if not isinstance(raw, Mapping):
            raise ValueError("attempt history row is not an object")
        attempts = raw.get("attempts")
        job_id = raw.get("job_id")
        if (
            not isinstance(job_id, str)
            or job_id in history_by_job
            or not isinstance(attempts, list)
            or len(attempts) != 1
        ):
            raise ValueError("one-shot attempt history changed")
        attempt = attempts[0]
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("attempt_id") != "a00"
            or attempt.get("terminal_status") != "accepted"
        ):
            raise ValueError("performance lock was retried or not accepted")
        history_by_job[job_id] = raw

    proofs = lifecycle_chain.get("wave_proofs")
    if not isinstance(proofs, list) or not proofs:
        raise ValueError("validated lifecycle execution proofs are missing")
    accepted: dict[str, Mapping[str, Any]] = {}
    job_bindings: dict[str, dict[str, str]] = {}
    prelaunch_authorizations: set[str] = set()
    for proof in proofs:
        if not isinstance(proof, Mapping):
            raise ValueError("lifecycle proof is not an object")
        selected = proof.get("selected_attempts")
        prelaunch = _require_sha(
            proof.get("prelaunch_authorization_sha256"),
            "one-shot prelaunch authorization",
        )
        if (
            not isinstance(selected, list)
            or proof.get("accepted_attempt_count") != len(selected)
            or proof.get("failed_attempt_count") != 0
            or proof.get("actual_launch_receipt_present") is not True
            or proof.get("actual_launch_receipt_revalidated") is not True
            or prelaunch in prelaunch_authorizations
        ):
            raise ValueError("one-shot lifecycle contains a failed attempt")
        prelaunch_authorizations.add(prelaunch)
        for attempt in selected:
            if not isinstance(attempt, Mapping):
                raise ValueError("lifecycle attempt is not an object")
            job_id = attempt.get("job_id")
            principal = attempt.get("worker_principal")
            instance_id = attempt.get("instance_id")
            if (
                not isinstance(job_id, str)
                or job_id in accepted
                or attempt.get("attempt_id") != "a00"
                or attempt.get("terminal_status") != "accepted"
                or attempt.get("exact_instance_created") is not True
                or attempt.get("valid_done_observed") is not True
                or not isinstance(principal, str)
                or not principal
                or not isinstance(instance_id, str)
                or not instance_id
            ):
                raise ValueError("lifecycle is not one accepted exact a00 per job")
            accepted[job_id] = attempt
            job_bindings[job_id] = {
                "instance_id": instance_id,
                "worker_principal": principal,
            }
    if (
        set(history_by_job) != set(expected_jobs)
        or set(accepted) != set(expected_jobs)
        or lifecycle_chain.get("execution_attempt_count") != 20
        or lifecycle_chain.get("accepted_job_count") != 20
        or lifecycle_chain.get("accepted_launch_receipt_count") != 20
    ):
        raise ValueError("one-shot lifecycle coverage changed")
    body = {
        "schema": "hu_m31_t3_step6d_performance_lock_v4_one_shot_audit_v1",
        "status": "twenty_jobs_each_accepted_once_on_a00",
        "accepted_job_count": 20,
        "execution_attempt_count": 20,
        "failed_attempt_count": 0,
        "retry_attempt_count": 0,
        "reseed_count": 0,
        "accepted_attempt_ids": ["a00"],
        "prelaunch_authorization_count": len(prelaunch_authorizations),
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "validated_lifecycle_chain_sha256": lifecycle_chain["chain_sha256"],
        "one_shot_lock_consumed": True,
        "rerun_authorized": False,
        "reseed_authorized": False,
        "current_profile_changed": False,
    }
    return {**body, "audit_sha256": canonical_sha256(body)}, job_bindings


def _validate_accepted_manifest_binding(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    accepted_evidence: Any,
    lifecycle_chain: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[Path, ...], tuple[Path, ...]]:
    manifest, candidate_paths, reference_paths = (
        wave_bridge._validate_merge_view_manifest_files(manifest_value)
    )
    wave_bridge._validate_pair_lineage_lifecycle_binding(
        manifest=manifest, validated_lifecycle_chain=lifecycle_chain
    )
    accepted_by_job = {
        item["meta"]["job_id"]: item for item in accepted_evidence.jobs
    }
    rows = manifest["jobs"]
    if (
        manifest.get("run_name") != plan["run_name"]
        or manifest.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or manifest.get("wave_plan_sha256") != plan["schedule_sha256"]
        or manifest.get("attempt_ledger_sha256") != ledger["ledger_sha256"]
        or manifest.get("accepted_snapshot_sha256")
        != accepted_evidence.snapshot["snapshot_sha256"]
        or manifest.get("accepted_root") != str(accepted_evidence.accepted_root)
        or manifest.get("run_contract_digest") != lock_plan.RUN_CONTRACT_DIGEST
        or manifest.get("pair_launch_lineage_audit")
        != accepted_evidence.pair_launch_lineage_audit
        or len(accepted_by_job) != 20
    ):
        raise ValueError("merge view differs from accepted wave-v2 evidence")
    for row in rows:
        item = accepted_by_job.get(row["job_id"])
        if item is None:
            raise ValueError("merge-view job is absent from accepted evidence")
        record = item["record"]
        transport = item["transport_done"]
        if (
            row["source_role"] != item["meta"]["source_role"]
            or row["work_hand_indices"] != item["meta"]["work_hand_indices"]
            or row["accepted_attempt_id"] != item["attempt_id"]
            or row["accepted_instance_id"] != item["instance_id"]
            or row["accepted_done_path"] != record["done_path"]
            or row["accepted_done_sha256"] != record["done_sha256"]
            or row["runner_done_sha256"] != transport["runner_done_sha256"]
            or row["package_sha256"] != record["package_sha256"]
            or row["image_digest"] != record["image_digest"]
            or row["binary_sha256"] != record["binary_sha256"]
            or row["allocation_digest"] != record["allocation_digest"]
            or row["root_digest"] != record["root_digest"]
        ):
            raise ValueError("merge-view job differs from accepted snapshot")
    return manifest, candidate_paths, reference_paths


def _validate_process_isolation(
    *,
    manifest: Mapping[str, Any],
    job_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    process_by_job: dict[str, dict[str, Any]] = {}
    hands_by_job: dict[str, dict[int, dict[str, Any]]] = {}
    all_process_namespaces: set[str] = set()
    all_process_ids: set[int] = set()
    manifest_jobs = manifest["jobs"]
    row_by_job = {row["job_id"]: row for row in manifest_jobs}
    if (
        len(manifest_jobs) != 20
        or len(row_by_job) != 20
        or set(job_bindings) != set(row_by_job)
    ):
        raise ValueError("process isolation job binding coverage changed")
    for job_id, row in row_by_job.items():
        done_path = _plain_file(row["merge_done_path"], f"{job_id} DONE")
        work_hand_indices = row.get("work_hand_indices")
        if (
            not isinstance(work_hand_indices, list)
            or len(work_hand_indices) != 10
            or len(set(work_hand_indices)) != 10
            or any(
                isinstance(hand_index, bool)
                or not isinstance(hand_index, int)
                or hand_index < 0
                for hand_index in work_hand_indices
            )
        ):
            raise ValueError(
                "job must bind exactly ten unique work hand indices"
            )
        binding = job_bindings[job_id]
        instance_id = binding.get("instance_id")
        principal = binding.get("worker_principal")
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or not isinstance(principal, str)
            or not principal
            or row.get("accepted_instance_id") != instance_id
        ):
            raise ValueError(
                "job did not use exactly one bound instance/principal"
            )
        hand_directory = (
            done_path.parent / "hands" / row["source_role"]
        )
        hand_directory = _plain_directory(
            hand_directory, f"{job_id} hand directory"
        )
        expected_names = {
            f"hand_{hand_index:03d}.json"
            for hand_index in work_hand_indices
        }
        observed_entries = list(hand_directory.iterdir())
        if (
            len(observed_entries) != 10
            or any(
                entry.is_symlink() or not entry.is_file()
                for entry in observed_entries
            )
            or {entry.name for entry in observed_entries} != expected_names
        ):
            raise ValueError(
                "job hand records do not exactly match work_hand_indices"
            )
        process_ids: set[int] = set()
        hand_rows: list[dict[str, Any]] = []
        hand_rows_by_index: dict[int, dict[str, Any]] = {}
        for hand_index in work_hand_indices:
            hand_path = hand_directory / f"hand_{hand_index:03d}.json"
            hand, _ = _read_canonical(hand_path, f"{job_id} hand {hand_index}")
            process_id = _positive_integer(hand.get("process_id"), "process id")
            if (
                hand.get("source_role") != row["source_role"]
                or hand.get("hand_index") != hand_index
                or hand.get("shard_manifest_sha256")
                != row["shard_manifest_sha256"]
            ):
                raise ValueError("source-hand process evidence differs from job")
            process_ids.add(process_id)
            all_process_ids.add(process_id)
            identity = {
                "job_id": job_id,
                "source_role": row["source_role"],
                "hand_index": hand_index,
                "instance_id": instance_id,
                "worker_principal": principal,
                "process_id": process_id,
            }
            namespace = canonical_sha256(identity)
            if namespace in all_process_namespaces:
                raise ValueError(
                    "hand process namespace is not globally unique"
                )
            all_process_namespaces.add(namespace)
            hand_row = {
                **identity,
                "process_namespace_sha256": namespace,
            }
            hand_rows.append(hand_row)
            hand_rows_by_index[hand_index] = hand_row
        if len(process_ids) != 10:
            raise ValueError(
                "job did not use one distinct process id per hand"
            )
        process_by_job[job_id] = {
            "job_id": job_id,
            "source_role": row["source_role"],
            "instance_id": instance_id,
            "worker_principal": principal,
            "work_hand_indices": list(work_hand_indices),
            "hand_count": 10,
            "distinct_process_id_count": 10,
            "hands": hand_rows,
        }
        hands_by_job[job_id] = hand_rows_by_index

    pairs = manifest["pair_launch_lineage_audit"]["pairs"]
    pair_rows: list[dict[str, Any]] = []
    paired_hand_count = 0
    for pair in pairs:
        candidate = process_by_job.get(pair["candidate_job_id"])
        reference = process_by_job.get(pair["reference_job_id"])
        if candidate is None or reference is None:
            raise ValueError("pair process evidence is incomplete")
        if (
            candidate["source_role"] != "candidate"
            or reference["source_role"] != "reference"
            or candidate["work_hand_indices"]
            != reference["work_hand_indices"]
            or candidate["instance_id"] == reference["instance_id"]
            or candidate["worker_principal"] == reference["worker_principal"]
            or pair.get("candidate_instance_id") != candidate["instance_id"]
            or pair.get("reference_instance_id") != reference["instance_id"]
            or pair.get("candidate_worker_principal")
            != candidate["worker_principal"]
            or pair.get("reference_worker_principal")
            != reference["worker_principal"]
        ):
            raise ValueError(
                "candidate/reference instance/process/principal overlapped"
            )
        hand_pairs: list[dict[str, Any]] = []
        candidate_hands = hands_by_job[candidate["job_id"]]
        reference_hands = hands_by_job[reference["job_id"]]
        for hand_index in candidate["work_hand_indices"]:
            candidate_hand = candidate_hands.get(hand_index)
            reference_hand = reference_hands.get(hand_index)
            if (
                candidate_hand is None
                or reference_hand is None
                or candidate_hand["instance_id"]
                == reference_hand["instance_id"]
                or candidate_hand["worker_principal"]
                == reference_hand["worker_principal"]
                or candidate_hand["process_namespace_sha256"]
                == reference_hand["process_namespace_sha256"]
            ):
                raise ValueError(
                    "candidate/reference paired hand process namespace "
                    "overlapped"
                )
            hand_pairs.append(
                {
                    "hand_index": hand_index,
                    "candidate_process_id": candidate_hand["process_id"],
                    "reference_process_id": reference_hand["process_id"],
                    "candidate_process_namespace_sha256": candidate_hand[
                        "process_namespace_sha256"
                    ],
                    "reference_process_namespace_sha256": reference_hand[
                        "process_namespace_sha256"
                    ],
                    "separate_instance": True,
                    "separate_principal": True,
                    "separate_process_namespace": True,
                }
            )
        if len(hand_pairs) != 10:
            raise ValueError("paired hand process coverage changed")
        paired_hand_count += len(hand_pairs)
        pair_rows.append(
            {
                "pair_id": pair["pair_id"],
                "candidate_job_id": candidate["job_id"],
                "reference_job_id": reference["job_id"],
                "candidate_instance_id": candidate["instance_id"],
                "reference_instance_id": reference["instance_id"],
                "candidate_worker_principal": candidate[
                    "worker_principal"
                ],
                "reference_worker_principal": reference[
                    "worker_principal"
                ],
                "hand_pair_count": 10,
                "hand_pairs": hand_pairs,
                "separate_instance": True,
                "separate_principal": True,
                "all_hand_process_namespaces_separate": True,
            }
        )
    instances = {row["instance_id"] for row in process_by_job.values()}
    if (
        len(process_by_job) != 20
        or len(all_process_namespaces) != 200
        or len(instances) != 20
        or len(pair_rows) != 10
        or paired_hand_count != 100
    ):
        raise ValueError("process isolation coverage changed")
    body = {
        "schema": (
            "hu_m31_t3_step6d_performance_lock_v4_process_isolation_audit_v2"
        ),
        "status": (
            "two_hundred_hand_process_namespaces_and_ten_isolated_job_pairs"
        ),
        "jobs": [process_by_job[job] for job in sorted(process_by_job)],
        "pairs": pair_rows,
        "job_count": 20,
        "pair_count": 10,
        "hand_execution_count": 200,
        "paired_hand_count": 100,
        "hand_executions_per_job": 10,
        "distinct_instance_count": 20,
        "distinct_raw_process_id_count": len(all_process_ids),
        "distinct_process_namespace_count": 200,
        "all_jobs_use_ten_distinct_process_ids": True,
        "all_hand_process_namespaces_unique": True,
        "all_paired_hands_separate_instance": True,
        "all_paired_hands_separate_process_namespace": True,
        "all_paired_hands_separate_principal": True,
        "raw_process_id_global_uniqueness_required": False,
        "current_profile_changed": False,
    }
    return {**body, "audit_sha256": canonical_sha256(body)}


def _pure_source_paths(pure: Mapping[str, Any], role: str) -> set[Path]:
    generic = pure.get("generic_merge")
    if not isinstance(generic, Mapping):
        raise ValueError("pure v4 generic merge is missing")
    sources = generic.get("source_done_inputs")
    if not isinstance(sources, Mapping):
        raise ValueError("pure v4 source DONE inputs are missing")
    rows = sources.get(role)
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError(f"pure v4 {role} source coverage changed")
    result: set[Path] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("pure v4 source row is not an object")
        path = _plain_file(str(row.get("path")), f"pure v4 {role} DONE")
        if path in result:
            raise ValueError("pure v4 source DONE path duplicated")
        result.add(path)
    return result


def _compose_receipt(
    *,
    source_paths: Mapping[str, Any],
    expected_profile_sha256: str,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    accepted_results_snapshot: Mapping[str, Any],
    validated_lifecycle_chain: Mapping[str, Any],
    merge_view_manifest: Mapping[str, Any],
    replay_sources: bool,
) -> dict[str, Any]:
    (
        paths,
        source_hashes,
        plan,
        materialization,
        seal,
        pure,
    ) = _load_core_sources(
        source_paths,
        expected_profile_sha256=expected_profile_sha256,
        replay_sources=replay_sources,
    )
    wave = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(wave, attempt_ledger)
    lifecycle = wave_bridge.validate_validated_lifecycle_chain(
        wave_plan=wave,
        attempt_ledger=ledger,
        value=validated_lifecycle_chain,
    )
    accepted = wave_bridge.validate_accepted_results_snapshot(
        wave_plan=wave,
        attempt_ledger=ledger,
        validated_lifecycle_chain=lifecycle,
        value=accepted_results_snapshot,
    )
    manifest, candidate_paths, reference_paths = _validate_accepted_manifest_binding(
        plan=wave,
        ledger=ledger,
        accepted_evidence=accepted,
        lifecycle_chain=lifecycle,
        manifest_value=merge_view_manifest,
    )
    if manifest != _read_canonical(
        paths["merge_view_manifest"], "stored merge-view manifest"
    )[0]:
        raise ValueError("embedded and stored merge-view manifests differ")
    if (
        _pure_source_paths(pure, "candidate") != set(candidate_paths)
        or _pure_source_paths(pure, "reference") != set(reference_paths)
    ):
        raise ValueError("pure v4 merge did not consume the accepted merge view")

    one_shot, job_bindings = _validate_one_shot_attempts(
        plan=wave, ledger=ledger, lifecycle_chain=lifecycle
    )
    processes = _validate_process_isolation(
        manifest=manifest, job_bindings=job_bindings
    )
    chronology = _validate_prelaunch_root_chronology(
        outer_package_root=Path(paths["outer_package"]),
        wave_plan=wave,
        accepted_snapshot=accepted.snapshot,
        lifecycle_chain=lifecycle,
        merge_manifest=manifest,
        plan=plan,
        materialization=materialization,
        seal=seal,
    )
    if (
        accepted.snapshot.get("accepted_job_count") != 20
        or accepted.snapshot.get("accepted_object_count") != 440
        or manifest.get("job_count") != 20
        or manifest.get("pair_launch_lineage_audit", {}).get("pair_count") != 10
        or _sha256_file(Path(paths["current_profile_registry"]))
        != expected_profile_sha256
    ):
        raise ValueError("final transport cardinality or profile pin changed")
    passed = pure.get("all_gates_passed") is True
    transport_body = {
        "schema": "hu_m31_t3_step6d_performance_lock_v4_transport_audit_v1",
        "status": "wave_v2_transport_and_source_replay_validated",
        "accepted_job_count": 20,
        "accepted_object_count": 440,
        "paired_job_count": 10,
        "paired_hand_count": 100,
        "root_count": 200,
        "run_contract_digest": lock_plan.RUN_CONTRACT_DIGEST,
        "accepted_snapshot_sha256": accepted.snapshot["snapshot_sha256"],
        "validated_lifecycle_chain_sha256": lifecycle["chain_sha256"],
        "merge_view_manifest_sha256": manifest["manifest_sha256"],
        "pure_v4_merge_sha256": pure["summary_sha256"],
        "prelaunch_root_chronology_sha256": chronology["chronology_sha256"],
        "one_shot_attempt_audit_sha256": one_shot["audit_sha256"],
        "process_isolation_audit_sha256": processes["audit_sha256"],
        "pure_merge_sources_replayed": True,
        "transport_lineage_validated": True,
        "current_profile_changed": False,
    }
    transport = {
        **transport_body,
        "audit_sha256": canonical_sha256(transport_body),
    }
    body = {
        "schema": RECEIPT_SCHEMA,
        "status": "qualified" if passed else "no_go",
        "decision": QUALIFIED_DECISION if passed else NO_GO_DECISION,
        "source_paths": paths,
        "source_file_sha256": source_hashes,
        "performance_lock_plan": deepcopy(plan),
        "materialization_receipt": deepcopy(materialization),
        "root_seal": deepcopy(seal),
        "pure_v4_merge": deepcopy(pure),
        "wave_plan": deepcopy(wave),
        "attempt_ledger": deepcopy(ledger),
        "accepted_results_snapshot": deepcopy(accepted.snapshot),
        "validated_lifecycle_chain": deepcopy(lifecycle),
        "merge_view_manifest": deepcopy(manifest),
        "prelaunch_root_chronology": chronology,
        "one_shot_attempt_audit": one_shot,
        "process_isolation_audit": processes,
        "transport_audit": transport,
        "all_gates_passed": passed,
        "performance_lock_finalized": True,
        "one_shot_lock_consumed": True,
        "transport_lineage_validated": True,
        "performance_lock_qualified": passed,
        "candidate_finalized_no_go": not passed,
        "quality_pilot_authorized": passed,
        "rerun_authorized": False,
        "reseed_authorized": False,
        "artifact_fanout_authorized": False,
        "training_eligible": False,
        "training_authorized": False,
        "promotion_evidence": False,
        "promotion_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
        "bridge_cloud_network_invoked": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def build_performance_lock_v4_production_receipt(
    *,
    pure_merge_path: str | Path,
    performance_lock_plan_path: str | Path,
    materialization_receipt_path: str | Path,
    root_seal_path: str | Path,
    outer_package_path: str | Path,
    merge_view_manifest_path: str | Path,
    current_profile_registry_path: str | Path,
    expected_profile_sha256: str,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    accepted_results_snapshot: Mapping[str, Any],
    validated_lifecycle_chain: Mapping[str, Any],
    merge_view_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay all local evidence and return a deterministic final receipt."""

    source_paths = {
        "pure_merge": str(Path(pure_merge_path).resolve()),
        "performance_lock_plan": str(
            Path(performance_lock_plan_path).resolve()
        ),
        "materialization_receipt": str(
            Path(materialization_receipt_path).resolve()
        ),
        "root_seal": str(Path(root_seal_path).resolve()),
        "outer_package": str(Path(outer_package_path).resolve()),
        "merge_view_manifest": str(Path(merge_view_manifest_path).resolve()),
        "current_profile_registry": str(
            Path(current_profile_registry_path).resolve()
        ),
    }
    return _compose_receipt(
        source_paths=source_paths,
        expected_profile_sha256=expected_profile_sha256,
        wave_plan=wave_plan,
        attempt_ledger=attempt_ledger,
        accepted_results_snapshot=accepted_results_snapshot,
        validated_lifecycle_chain=validated_lifecycle_chain,
        merge_view_manifest=merge_view_manifest,
        replay_sources=True,
    )


def validate_performance_lock_v4_production_receipt_value(
    value: Mapping[str, Any],
    *,
    expected_profile_sha256: str,
    replay_sources: bool = True,
) -> dict[str, Any]:
    """Recompute a stored receipt from every embedded and path-bound source."""

    if not isinstance(value, Mapping):
        raise ValueError("production receipt must be an object")
    receipt = deepcopy(dict(value))
    if set(receipt) != _RECEIPT_KEYS:
        raise ValueError("production receipt fields changed")
    digest = receipt.pop("receipt_sha256", None)
    if digest != canonical_sha256(receipt):
        raise ValueError("production receipt digest changed")
    receipt["receipt_sha256"] = _require_sha(digest, "production receipt")
    expected = _compose_receipt(
        source_paths=receipt["source_paths"],
        expected_profile_sha256=expected_profile_sha256,
        wave_plan=receipt["wave_plan"],
        attempt_ledger=receipt["attempt_ledger"],
        accepted_results_snapshot=receipt["accepted_results_snapshot"],
        validated_lifecycle_chain=receipt["validated_lifecycle_chain"],
        merge_view_manifest=receipt["merge_view_manifest"],
        replay_sources=replay_sources,
    )
    if receipt != expected:
        raise ValueError("production receipt differs from complete source replay")
    return receipt


def validate_performance_lock_v4_production_receipt(
    receipt_path: str | Path,
    *,
    expected_profile_sha256: str,
    replay_sources: bool = True,
) -> dict[str, Any]:
    value, _ = _read_canonical(receipt_path, "production final receipt")
    return validate_performance_lock_v4_production_receipt_value(
        value,
        expected_profile_sha256=expected_profile_sha256,
        replay_sources=replay_sources,
    )


def _write_once(path: Path, raw: bytes) -> None:
    if not path.is_absolute():
        raise ValueError("production receipt output path must be absolute")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("production receipt parent must already exist and be safe")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    created = True
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def write_performance_lock_v4_production_receipt(
    *,
    output_path: str | Path,
    expected_profile_sha256: str,
    **build_kwargs: Any,
) -> dict[str, Any]:
    """Build, create once, reload, and fully replay the final receipt."""

    output = Path(output_path)
    if not output.is_absolute():
        raise ValueError("production receipt output path must be absolute")
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("production final receipt is write-once")
    protected_files = {
        Path(str(build_kwargs[key])).resolve()
        for key in (
            "pure_merge_path",
            "performance_lock_plan_path",
            "materialization_receipt_path",
            "root_seal_path",
            "merge_view_manifest_path",
            "current_profile_registry_path",
        )
    }
    protected_directories = {
        Path(str(build_kwargs["outer_package_path"])).resolve(),
        Path(str(build_kwargs["merge_view_manifest_path"])).resolve().parent,
    }
    snapshot = build_kwargs.get("accepted_results_snapshot")
    if isinstance(snapshot, Mapping):
        accepted_root = snapshot.get("accepted_root")
        if isinstance(accepted_root, str) and Path(accepted_root).is_absolute():
            protected_directories.add(Path(accepted_root).resolve())
    if output in protected_files or any(
        output == root or root in output.parents or output in root.parents
        for root in protected_directories
    ):
        raise ValueError("production receipt output overlaps immutable evidence")
    receipt = build_performance_lock_v4_production_receipt(
        expected_profile_sha256=expected_profile_sha256,
        **build_kwargs,
    )
    _write_once(output, canonical_bytes(receipt))
    replayed = validate_performance_lock_v4_production_receipt(
        output,
        expected_profile_sha256=expected_profile_sha256,
        replay_sources=True,
    )
    if replayed != receipt:
        raise ValueError("written production receipt differs from source replay")
    return receipt


__all__ = [
    "NO_GO_DECISION",
    "QUALIFIED_DECISION",
    "RECEIPT_SCHEMA",
    "build_performance_lock_v4_production_receipt",
    "canonical_bytes",
    "canonical_sha256",
    "validate_performance_lock_v4_production_receipt",
    "validate_performance_lock_v4_production_receipt_value",
    "write_performance_lock_v4_production_receipt",
]
