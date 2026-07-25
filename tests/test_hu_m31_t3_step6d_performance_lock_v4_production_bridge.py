from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_performance_lock_v4_production_bridge as subject,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.canonical_bytes(value))


def test_outer_wave_plan_uses_its_frozen_lf_serializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {"schema": "fixture-wave-plan", "nested": {"value": 7}}
    monkeypatch.setattr(
        subject.wave_v2,
        "validate_wave_plan",
        lambda value: deepcopy(dict(value)),
    )
    path = tmp_path / "wave_plan.json"
    raw = subject.wave_v2.canonical_bytes(plan)
    path.write_bytes(raw)

    assert raw == subject.outer_package.canonical_bytes(plan) + b"\n"
    assert subject._read_outer_wave_plan(
        path.resolve(),
        expected_wave_plan=plan,
    ) == plan


@pytest.mark.parametrize(
    "tamper",
    ("missing_lf", "crlf", "extra_lf", "changed_content"),
)
def test_outer_wave_plan_rejects_other_bytes_or_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    plan = {"schema": "fixture-wave-plan", "nested": {"value": 7}}
    monkeypatch.setattr(
        subject.wave_v2,
        "validate_wave_plan",
        lambda value: deepcopy(dict(value)),
    )
    if tamper == "missing_lf":
        raw = subject.outer_package.canonical_bytes(plan)
        match = "canonical LF JSON"
    elif tamper == "crlf":
        raw = subject.outer_package.canonical_bytes(plan) + b"\r\n"
        match = "canonical LF JSON"
    elif tamper == "extra_lf":
        raw = subject.wave_v2.canonical_bytes(plan) + b"\n"
        match = "canonical LF JSON"
    else:
        raw = subject.wave_v2.canonical_bytes(
            {"schema": "fixture-wave-plan", "nested": {"value": 8}}
        )
        match = "differs from its frozen source"
    path = tmp_path / "wave_plan.json"
    path.write_bytes(raw)

    with pytest.raises(ValueError, match=match):
        subject._read_outer_wave_plan(
            path.resolve(),
            expected_wave_plan=plan,
        )


@pytest.fixture
def evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    root = tmp_path.resolve()
    accepted_root = root / "accepted"
    merge_root = root / "merge-view"
    outer_root = root / "outer-package"
    accepted_root.mkdir()
    merge_root.mkdir()
    outer_root.mkdir()
    profile = root / "ai_profiles.py"
    profile.write_bytes(b"# unchanged current profile\n")
    profile_sha = hashlib.sha256(profile.read_bytes()).hexdigest()

    job_ids: list[str] = []
    manifest_jobs: list[dict[str, Any]] = []
    accepted_jobs: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    lifecycle_attempts: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    candidate_paths: list[Path] = []
    reference_paths: list[Path] = []
    pure_sources: dict[str, list[dict[str, Any]]] = {
        "candidate": [],
        "reference": [],
    }
    for shard in range(10):
        work = list(range(shard * 10, shard * 10 + 10))
        pair_jobs: dict[str, str] = {}
        pair_instances: dict[str, str] = {}
        pair_principals: dict[str, str] = {}
        for role in ("candidate", "reference"):
            job_id = f"{role}-shard-{shard:02d}"
            instance = f"instance-{role}-{shard:02d}"
            principal = f"{role}-{shard:02d}@fixture.iam.gserviceaccount.com"
            shard_sha = hashlib.sha256(job_id.encode("ascii")).hexdigest()
            done = merge_root / "jobs" / job_id / "DONE.json"
            _write_json(done, {"job_id": job_id})
            for hand_index in work:
                _write_json(
                    done.parent
                    / "hands"
                    / role
                    / f"hand_{hand_index:03d}.json",
                    {
                        "source_role": role,
                        "hand_index": hand_index,
                        "shard_manifest_sha256": shard_sha,
                        # Each VM starts one process per hand.  Raw PIDs may
                        # safely recur on another VM namespace.
                        "process_id": 1000 + hand_index - work[0],
                    },
                )
            (candidate_paths if role == "candidate" else reference_paths).append(
                done
            )
            pure_sources[role].append({"path": str(done)})
            row = {
                "job_id": job_id,
                "source_role": role,
                "shard_index": shard,
                "work_hand_indices": work,
                "accepted_attempt_id": "a00",
                "accepted_instance_id": instance,
                "accepted_done_path": f"accepted/{job_id}/DONE.json",
                "accepted_done_sha256": hashlib.sha256(
                    f"accepted-{job_id}".encode("ascii")
                ).hexdigest(),
                "merge_done_path": str(done),
                "merge_done_sha256": hashlib.sha256(
                    f"runner-{job_id}".encode("ascii")
                ).hexdigest(),
                "runner_done_sha256": hashlib.sha256(
                    f"runner-{job_id}".encode("ascii")
                ).hexdigest(),
                "shard_manifest_sha256": shard_sha,
                "run_contract_digest": subject.lock_plan.RUN_CONTRACT_DIGEST,
                "package_sha256": "2" * 64,
                "image_digest": "sha256:" + "3" * 64,
                "binary_sha256": ("4" if role == "candidate" else "5") * 64,
                "allocation_digest": "6" * 64,
                "root_digest": hashlib.sha256(
                    f"root-{job_id}".encode("ascii")
                ).hexdigest(),
                "candidate_reference_process_isolated": True,
            }
            item = {
                "meta": {
                    "job_id": job_id,
                    "source_role": role,
                    "shard_index": shard,
                    "work_hand_indices": work,
                },
                "record": {
                    "done_path": row["accepted_done_path"],
                    "done_sha256": row["accepted_done_sha256"],
                    "package_sha256": row["package_sha256"],
                    "image_digest": row["image_digest"],
                    "binary_sha256": row["binary_sha256"],
                    "allocation_digest": row["allocation_digest"],
                    "root_digest": row["root_digest"],
                },
                "attempt_id": "a00",
                "instance_id": instance,
                "transport_done": {
                    "runner_done_sha256": row["runner_done_sha256"]
                },
            }
            attempt = {
                "job_id": job_id,
                "source_role": role,
                "attempt_id": "a00",
                "instance_id": instance,
                "worker_principal": principal,
                "terminal_status": "accepted",
                "exact_instance_created": True,
                "valid_done_observed": True,
            }
            job_ids.append(job_id)
            manifest_jobs.append(row)
            accepted_jobs.append(item)
            lifecycle_attempts.append(attempt)
            history.append(
                {
                    "job_id": job_id,
                    "source_role": role,
                    "attempts": [
                        {
                            "attempt_id": "a00",
                            "instance_id": instance,
                            "terminal_status": "accepted",
                        }
                    ],
                }
            )
            pair_jobs[role] = job_id
            pair_instances[role] = instance
            pair_principals[role] = principal
        pairs.append(
            {
                "pair_id": f"paired-shard-{shard:02d}",
                "candidate_job_id": pair_jobs["candidate"],
                "reference_job_id": pair_jobs["reference"],
                "candidate_instance_id": pair_instances["candidate"],
                "reference_instance_id": pair_instances["reference"],
                "candidate_worker_principal": pair_principals["candidate"],
                "reference_worker_principal": pair_principals["reference"],
                "content_payload_sha256": "7" * 64,
                "outer_manifest_sha256": "8" * 64,
            }
        )

    plan = {
        "coverage": {"job_ids": job_ids},
        "run_name": "performance-lock-v4-fixture",
        "execution_identity_sha256": "9" * 64,
        "schedule_sha256": "a" * 64,
        "ledger_sha256": "b" * 64,
        "full100_plan": {"schema": "fixture"},
        "full100_plan_sha256": "c" * 64,
        "run_contract_digest": subject.lock_plan.RUN_CONTRACT_DIGEST,
    }
    ledger = {
        "ledger_sha256": "d" * 64,
        "transitions": [{"attempt_history": history}],
    }
    lifecycle = {
        "chain_sha256": "e" * 64,
        "wave_proofs": [
            {
                "selected_attempts": lifecycle_attempts,
                "accepted_attempt_count": 20,
                "failed_attempt_count": 0,
                "actual_launch_receipt_present": True,
                "actual_launch_receipt_revalidated": True,
                "prelaunch_authorization_sha256": "f" * 64,
            }
        ],
        "execution_attempt_count": 20,
        "accepted_job_count": 20,
        "accepted_launch_receipt_count": 20,
    }
    snapshot = {
        "snapshot_sha256": "0" * 64,
        "accepted_job_count": 20,
        "accepted_object_count": 440,
        "content_payload_sha256": "7" * 64,
        "outer_manifest_sha256": "8" * 64,
    }
    pair_audit = {"pairs": pairs, "pair_count": 10}
    manifest = {
        "manifest_sha256": "1" * 64,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "accepted_snapshot_sha256": snapshot["snapshot_sha256"],
        "accepted_root": str(accepted_root),
        "run_contract_digest": subject.lock_plan.RUN_CONTRACT_DIGEST,
        "jobs": manifest_jobs,
        "job_count": 20,
        "pair_launch_lineage_audit": pair_audit,
    }
    merge_manifest_path = merge_root / "MERGE_VIEW.json"
    _write_json(merge_manifest_path, manifest)
    accepted_evidence = SimpleNamespace(
        snapshot=snapshot,
        accepted_root=accepted_root,
        jobs=tuple(accepted_jobs),
        pair_launch_lineage_audit=pair_audit,
    )

    core_plan = {"plan": "v4"}
    materialization = {"materialization": "v4"}
    seal = {"seal": "v4"}
    pure = {
        "summary_sha256": "2" * 64,
        "all_gates_passed": True,
        "generic_merge": {"source_done_inputs": pure_sources},
    }
    dummy_paths: dict[str, Path] = {}
    for name in (
        "pure_merge",
        "performance_lock_plan",
        "materialization_receipt",
        "root_seal",
    ):
        path = root / f"{name}.json"
        _write_json(path, {name: True})
        dummy_paths[name] = path
    source_paths = {
        **{key: str(path) for key, path in dummy_paths.items()},
        "outer_package": str(outer_root),
        "merge_view_manifest": str(merge_manifest_path),
        "current_profile_registry": str(profile),
    }
    replay_calls: list[bool] = []

    def load_sources(
        raw_paths: dict[str, Any],
        *,
        expected_profile_sha256: str,
        replay_sources: bool,
    ) -> tuple[Any, ...]:
        replay_calls.append(replay_sources)
        if replay_sources is not True:
            raise PermissionError(
                "final production receipt requires full source replay"
            )
        assert expected_profile_sha256 == profile_sha
        normalized = {
            key: str(Path(str(value)).resolve())
            for key, value in raw_paths.items()
        }
        hashes = {
            key: hashlib.sha256(key.encode("ascii")).hexdigest()
            for key in (
                "pure_merge",
                "performance_lock_plan",
                "materialization_receipt",
                "root_seal",
                "merge_view_manifest",
                "current_profile_registry",
            )
        }
        return (
            normalized,
            hashes,
            deepcopy(core_plan),
            deepcopy(materialization),
            deepcopy(seal),
            deepcopy(pure),
        )

    def chronology(**_: Any) -> dict[str, Any]:
        body = {
            "root_sealed_before_launch": True,
            "current_profile_changed": False,
        }
        return {**body, "chronology_sha256": subject.canonical_sha256(body)}

    monkeypatch.setattr(subject, "_load_core_sources", load_sources)
    monkeypatch.setattr(
        subject.wave_v2, "validate_wave_plan", lambda value: deepcopy(dict(value))
    )
    monkeypatch.setattr(
        subject.wave_v2,
        "validate_attempt_ledger",
        lambda _plan, value: deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        subject.wave_bridge,
        "validate_validated_lifecycle_chain",
        lambda **kwargs: deepcopy(dict(kwargs["value"])),
    )
    monkeypatch.setattr(
        subject.wave_bridge,
        "validate_accepted_results_snapshot",
        lambda **kwargs: SimpleNamespace(
            snapshot=deepcopy(dict(kwargs["value"])),
            accepted_root=accepted_evidence.accepted_root,
            jobs=accepted_evidence.jobs,
            pair_launch_lineage_audit=accepted_evidence.pair_launch_lineage_audit,
        ),
    )
    monkeypatch.setattr(
        subject.wave_bridge,
        "_validate_merge_view_manifest_files",
        lambda value: (
            deepcopy(dict(value)),
            tuple(candidate_paths),
            tuple(reference_paths),
        ),
    )
    monkeypatch.setattr(
        subject.wave_bridge,
        "_validate_pair_lineage_lifecycle_binding",
        lambda **_: None,
    )
    monkeypatch.setattr(subject, "_validate_prelaunch_root_chronology", chronology)

    return {
        "root": root,
        "profile_sha": profile_sha,
        "source_paths": source_paths,
        "plan": plan,
        "ledger": ledger,
        "lifecycle": lifecycle,
        "snapshot": snapshot,
        "manifest": manifest,
        "pure": pure,
        "replay_calls": replay_calls,
        "accepted_evidence": accepted_evidence,
        "merge_manifest_path": merge_manifest_path,
    }


def _build_kwargs(evidence: dict[str, Any]) -> dict[str, Any]:
    paths = evidence["source_paths"]
    return {
        "pure_merge_path": paths["pure_merge"],
        "performance_lock_plan_path": paths["performance_lock_plan"],
        "materialization_receipt_path": paths["materialization_receipt"],
        "root_seal_path": paths["root_seal"],
        "outer_package_path": paths["outer_package"],
        "merge_view_manifest_path": paths["merge_view_manifest"],
        "current_profile_registry_path": paths["current_profile_registry"],
        "expected_profile_sha256": evidence["profile_sha"],
        "wave_plan": evidence["plan"],
        "attempt_ledger": evidence["ledger"],
        "accepted_results_snapshot": evidence["snapshot"],
        "validated_lifecycle_chain": evidence["lifecycle"],
        "merge_view_manifest": evidence["manifest"],
    }


def test_qualified_receipt_closes_transport_and_only_opens_quality_pilot(
    evidence: dict[str, Any],
) -> None:
    receipt = subject.build_performance_lock_v4_production_receipt(
        **_build_kwargs(evidence)
    )
    assert receipt["status"] == "qualified"
    assert receipt["performance_lock_finalized"] is True
    assert receipt["one_shot_lock_consumed"] is True
    assert receipt["transport_lineage_validated"] is True
    assert receipt["performance_lock_qualified"] is True
    assert receipt["candidate_finalized_no_go"] is False
    assert receipt["quality_pilot_authorized"] is True
    assert receipt["transport_audit"]["accepted_object_count"] == 440
    process_audit = receipt["process_isolation_audit"]
    assert process_audit["pair_count"] == 10
    assert process_audit["hand_execution_count"] == 200
    assert process_audit["paired_hand_count"] == 100
    assert process_audit["distinct_process_namespace_count"] == 200
    assert process_audit["distinct_raw_process_id_count"] == 10
    assert process_audit["raw_process_id_global_uniqueness_required"] is False
    assert all(
        row["hand_count"] == 10
        and row["distinct_process_id_count"] == 10
        and len(row["hands"]) == 10
        for row in process_audit["jobs"]
    )
    assert all(
        pair["hand_pair_count"] == 10
        and len(pair["hand_pairs"]) == 10
        and all(
            hand_pair["candidate_process_id"]
            == hand_pair["reference_process_id"]
            and hand_pair["candidate_process_namespace_sha256"]
            != hand_pair["reference_process_namespace_sha256"]
            and hand_pair["separate_process_namespace"] is True
            for hand_pair in pair["hand_pairs"]
        )
        for pair in process_audit["pairs"]
    )
    for field in (
        "rerun_authorized",
        "reseed_authorized",
        "training_eligible",
        "training_authorized",
        "promotion_evidence",
        "promotion_authorized",
        "current_profile_changed",
        "runtime_policy_activated",
    ):
        assert receipt[field] is False


def test_metric_no_go_is_final_and_does_not_open_quality(
    evidence: dict[str, Any],
) -> None:
    evidence["pure"]["all_gates_passed"] = False
    receipt = subject.build_performance_lock_v4_production_receipt(
        **_build_kwargs(evidence)
    )
    assert receipt["status"] == "no_go"
    assert receipt["performance_lock_finalized"] is True
    assert receipt["candidate_finalized_no_go"] is True
    assert receipt["performance_lock_qualified"] is False
    assert receipt["quality_pilot_authorized"] is False
    assert receipt["rerun_authorized"] is False
    assert receipt["reseed_authorized"] is False


def test_retry_or_missing_prelaunch_authorization_fails_closed(
    evidence: dict[str, Any],
) -> None:
    retried = deepcopy(evidence["ledger"])
    retried["transitions"][-1]["attempt_history"][0]["attempts"].append(
        {
            "attempt_id": "a01",
            "instance_id": "retry",
            "terminal_status": "accepted",
        }
    )
    kwargs = _build_kwargs(evidence)
    kwargs["attempt_ledger"] = retried
    with pytest.raises(ValueError, match="one-shot attempt history"):
        subject.build_performance_lock_v4_production_receipt(**kwargs)

    missing = deepcopy(evidence["lifecycle"])
    missing["wave_proofs"][0]["prelaunch_authorization_sha256"] = None
    kwargs = _build_kwargs(evidence)
    kwargs["validated_lifecycle_chain"] = missing
    with pytest.raises(ValueError, match="prelaunch authorization"):
        subject.build_performance_lock_v4_production_receipt(**kwargs)


def test_pair_principal_or_process_tamper_fails_closed(
    evidence: dict[str, Any],
) -> None:
    lifecycle = deepcopy(evidence["lifecycle"])
    candidate = lifecycle["wave_proofs"][0]["selected_attempts"][0]
    reference = lifecycle["wave_proofs"][0]["selected_attempts"][1]
    reference["worker_principal"] = candidate["worker_principal"]
    manifest = deepcopy(evidence["manifest"])
    manifest["pair_launch_lineage_audit"]["pairs"][0][
        "reference_worker_principal"
    ] = candidate["worker_principal"]
    evidence["accepted_evidence"].pair_launch_lineage_audit = manifest[
        "pair_launch_lineage_audit"
    ]
    _write_json(evidence["merge_manifest_path"], manifest)
    kwargs = _build_kwargs(evidence)
    kwargs["validated_lifecycle_chain"] = lifecycle
    kwargs["merge_view_manifest"] = manifest
    with pytest.raises(ValueError, match="instance/process/principal"):
        subject.build_performance_lock_v4_production_receipt(**kwargs)

    evidence["accepted_evidence"].pair_launch_lineage_audit = evidence[
        "manifest"
    ]["pair_launch_lineage_audit"]
    _write_json(evidence["merge_manifest_path"], evidence["manifest"])
    row = evidence["manifest"]["jobs"][0]
    hand = (
        Path(row["merge_done_path"]).parent
        / "hands"
        / row["source_role"]
        / f"hand_{row['work_hand_indices'][0]:03d}.json"
    )
    value = {
        "source_role": row["source_role"],
        "hand_index": row["work_hand_indices"][0],
        "shard_manifest_sha256": row["shard_manifest_sha256"],
        "process_id": 0,
    }
    _write_json(hand, value)
    with pytest.raises(ValueError, match="process id"):
        subject.build_performance_lock_v4_production_receipt(
            **_build_kwargs(evidence)
        )


def test_duplicate_pid_inside_one_job_fails_closed(
    evidence: dict[str, Any],
) -> None:
    row = evidence["manifest"]["jobs"][0]
    first_index, second_index = row["work_hand_indices"][:2]
    hand_directory = (
        Path(row["merge_done_path"]).parent
        / "hands"
        / row["source_role"]
    )
    first, _ = subject._read_canonical(
        hand_directory / f"hand_{first_index:03d}.json",
        "first fixture hand",
    )
    second_path = hand_directory / f"hand_{second_index:03d}.json"
    second, _ = subject._read_canonical(
        second_path, "second fixture hand"
    )
    second["process_id"] = first["process_id"]
    _write_json(second_path, second)

    with pytest.raises(ValueError, match="distinct process id per hand"):
        subject.build_performance_lock_v4_production_receipt(
            **_build_kwargs(evidence)
        )


@pytest.mark.parametrize("tamper", ("missing", "extra"))
def test_exact_ten_hand_record_inventory_fails_closed(
    evidence: dict[str, Any],
    tamper: str,
) -> None:
    row = evidence["manifest"]["jobs"][0]
    hand_directory = (
        Path(row["merge_done_path"]).parent
        / "hands"
        / row["source_role"]
    )
    if tamper == "missing":
        first_index = row["work_hand_indices"][0]
        (hand_directory / f"hand_{first_index:03d}.json").unlink()
    else:
        _write_json(
            hand_directory / "hand_999.json",
            {
                "source_role": row["source_role"],
                "hand_index": 999,
                "shard_manifest_sha256": row[
                    "shard_manifest_sha256"
                ],
                "process_id": 9999,
            },
        )

    with pytest.raises(ValueError, match="exactly match work_hand_indices"):
        subject.build_performance_lock_v4_production_receipt(
            **_build_kwargs(evidence)
        )


def test_lifecycle_instance_binding_tamper_fails_closed(
    evidence: dict[str, Any],
) -> None:
    lifecycle = deepcopy(evidence["lifecycle"])
    lifecycle["wave_proofs"][0]["selected_attempts"][0][
        "instance_id"
    ] = "another-instance"
    kwargs = _build_kwargs(evidence)
    kwargs["validated_lifecycle_chain"] = lifecycle

    with pytest.raises(
        ValueError, match="one bound instance/principal"
    ):
        subject.build_performance_lock_v4_production_receipt(**kwargs)


def test_snapshot_and_pure_merge_view_tamper_fail_closed(
    evidence: dict[str, Any],
) -> None:
    snapshot = deepcopy(evidence["snapshot"])
    snapshot["snapshot_sha256"] = "9" * 64
    kwargs = _build_kwargs(evidence)
    kwargs["accepted_results_snapshot"] = snapshot
    with pytest.raises(ValueError, match="accepted wave-v2 evidence"):
        subject.build_performance_lock_v4_production_receipt(**kwargs)

    extra = evidence["root"] / "unaccepted-DONE.json"
    _write_json(extra, {"unexpected": True})
    evidence["pure"]["generic_merge"]["source_done_inputs"]["candidate"][0][
        "path"
    ] = str(extra)
    with pytest.raises(ValueError, match="did not consume"):
        subject.build_performance_lock_v4_production_receipt(
            **_build_kwargs(evidence)
        )


def test_write_once_replays_and_resealed_boundary_tamper_is_rejected(
    evidence: dict[str, Any],
) -> None:
    output = evidence["root"] / "FINAL_RECEIPT.json"
    kwargs = _build_kwargs(evidence)
    profile_sha = kwargs.pop("expected_profile_sha256")
    receipt = subject.write_performance_lock_v4_production_receipt(
        output_path=output,
        expected_profile_sha256=profile_sha,
        **kwargs,
    )
    assert output.read_bytes() == subject.canonical_bytes(receipt)
    assert evidence["replay_calls"] == [True, True]
    with pytest.raises(FileExistsError, match="write-once"):
        subject.write_performance_lock_v4_production_receipt(
            output_path=output,
            expected_profile_sha256=profile_sha,
            **kwargs,
        )

    assert subject.validate_performance_lock_v4_production_receipt(
        output,
        expected_profile_sha256=profile_sha,
        replay_sources=True,
    ) == receipt
    assert evidence["replay_calls"][-1] is True
    with pytest.raises(PermissionError, match="full source replay"):
        subject.validate_performance_lock_v4_production_receipt(
            output,
            expected_profile_sha256=profile_sha,
            replay_sources=False,
        )

    forged = deepcopy(receipt)
    forged["quality_pilot_authorized"] = False
    body = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = subject.canonical_sha256(body)
    with pytest.raises(ValueError, match="complete source replay"):
        subject.validate_performance_lock_v4_production_receipt_value(
            forged,
            expected_profile_sha256=profile_sha,
            replay_sources=True,
        )


def test_prelaunch_chronology_resolves_v4_startup_from_science_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup_sha = "a" * 64
    content_sha = "b" * 64
    outer_sha = "c" * 64
    source = b"v4-source-archive"
    source_sha = hashlib.sha256(source).hexdigest()
    plan = {"schema": "v4-plan"}
    materialization = {
        "schema": "v4-materialization",
        "cloud_started": False,
    }
    seal = {
        "schema": "v4-seal",
        "cloud_started": False,
        "root_hash_aggregate_sha256": "d" * 64,
        "observation_fingerprint_aggregate_sha256": "e" * 64,
        "seed_set_sha256": "f" * 64,
    }
    wave = {
        "full100_plan": plan,
        "full100_plan_sha256": subject.lock_plan.PLAN_SHA256,
        "run_contract_digest": subject.lock_plan.RUN_CONTRACT_DIGEST,
    }
    outer_root = tmp_path.resolve() / "outer"
    wave_path = outer_root / subject.outer_package.WAVE_PLAN_PATH
    wave_path.parent.mkdir(parents=True)
    wave_path.write_bytes(subject.wave_v2.canonical_bytes(wave))
    science_manifest_path = (
        outer_root / subject.outer_package.SCIENTIFIC_MANIFEST_PATH
    )
    source_path = outer_root / subject.outer_package.SOURCE_PATH
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source)
    science_manifest = {
        "plan_sha256": subject.lock_plan.PLAN_SHA256,
        "materialization_receipt_sha256": subject.canonical_sha256(
            materialization
        ),
        "root_seal_sha256": subject.canonical_sha256(seal),
        "root_hash_aggregate_sha256": seal["root_hash_aggregate_sha256"],
        "observation_fingerprint_aggregate_sha256": seal[
            "observation_fingerprint_aggregate_sha256"
        ],
        "seed_set_sha256": seal["seed_set_sha256"],
        "source_sha256": source_sha,
        "cloud_started": False,
    }
    _write_json(science_manifest_path, science_manifest)
    outer = {
        "full100_plan_sha256": subject.lock_plan.PLAN_SHA256,
        "run_contract_digest": subject.lock_plan.RUN_CONTRACT_DIGEST,
        "expected_startup_sha256": startup_sha,
        "content_payload_sha256": content_sha,
        "manifest_sha256": outer_sha,
        "scientific_lineage": {"source_sha256": source_sha},
        "cloud_started": False,
    }
    archived = {
        subject.v4_package.PLAN_ARCHIVE_PATH: subject.canonical_bytes(plan),
        subject.v4_package.MATERIALIZATION_ARCHIVE_PATH: subject.canonical_bytes(
            materialization
        ),
        subject.v4_package.SEAL_ARCHIVE_PATH: subject.canonical_bytes(seal),
    }
    observed_startups: list[str] = []

    monkeypatch.setattr(
        subject.science_registry,
        "resolve_startup_sha256",
        lambda value: startup_sha if value is wave else "",
    )

    def validate_outer(
        root: Path, value: dict[str, Any], *, expected_startup_sha256: str
    ) -> dict[str, Any]:
        assert root == outer_root
        assert value is wave
        observed_startups.append(expected_startup_sha256)
        return outer

    monkeypatch.setattr(
        subject.outer_package, "validate_outer_package", validate_outer
    )
    monkeypatch.setattr(
        subject.v4_package,
        "_validate_manifest_value",
        lambda value: deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        subject.v4_package, "_validate_archive", lambda *_args: archived
    )
    monkeypatch.setattr(
        subject.v4_package,
        "_validate_archived_science",
        lambda *_args: plan,
    )
    monkeypatch.setattr(
        subject.lock_plan,
        "validate_performance_lock_v4_plan",
        lambda value: deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        subject.lock_plan,
        "validate_materialization_receipt",
        lambda value: deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        subject.lock_plan,
        "validate_root_seal",
        lambda value: deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        subject.wave_v2,
        "validate_wave_plan",
        lambda value: deepcopy(dict(value)),
    )
    audit = subject._validate_prelaunch_root_chronology(
        outer_package_root=outer_root,
        wave_plan=wave,
        accepted_snapshot={
            "expected_startup_sha256": startup_sha,
            "content_payload_sha256": content_sha,
            "outer_manifest_sha256": outer_sha,
        },
        lifecycle_chain={
            "wave_proofs": [
                {
                    "actual_launch_receipt_present": True,
                    "actual_launch_receipt_revalidated": True,
                    "prelaunch_authorization_sha256": "1" * 64,
                }
            ]
        },
        merge_manifest={
            "pair_launch_lineage_audit": {
                "pairs": [
                    {
                        "content_payload_sha256": content_sha,
                        "outer_manifest_sha256": outer_sha,
                    }
                ]
            }
        },
        plan=plan,
        materialization=materialization,
        seal=seal,
    )
    assert observed_startups == [startup_sha]
    assert audit["startup_sha256"] == startup_sha
    assert audit["root_sealed_before_launch"] is True


def test_write_rejects_output_inside_immutable_merge_view(
    evidence: dict[str, Any],
) -> None:
    kwargs = _build_kwargs(evidence)
    profile_sha = kwargs.pop("expected_profile_sha256")
    with pytest.raises(ValueError, match="overlaps immutable evidence"):
        subject.write_performance_lock_v4_production_receipt(
            output_path=Path(evidence["source_paths"]["merge_view_manifest"]).parent
            / "FINAL_RECEIPT.json",
            expected_profile_sha256=profile_sha,
            **kwargs,
        )
