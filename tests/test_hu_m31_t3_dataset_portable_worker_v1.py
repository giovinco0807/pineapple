from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_dataset_contract_v1 as contract
from ofc_regular import hu_m31_t3_dataset_executor_v1 as executor
from ofc_regular import hu_m31_t3_dataset_portable_worker_v1 as subject
from ofc_regular import hu_m31_t3_step6d_fresh_quality_gate_v1 as quality_gate


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    plan = contract.build_dataset_plan()
    fresh = {
        "schema": quality_gate.GATE_SCHEMA,
        "status": "pass",
        "decision": "fresh_quality_pass_open_25_paired_data_shard_only",
        "merge_sha256": "b" * 64,
        "all_gates_passed": True,
        "data_pilot_25_paired_authorized": True,
        "full_9000_paired_fanout_authorized": False,
    }
    fresh_path = tmp_path / "fresh.json"
    fresh_path.write_bytes(subject.canonical_bytes(fresh))
    smoke = {
        "schema": contract.SMOKE_GATE_SCHEMA,
        "status": "pass",
        "decision": "open_remaining_8975_paired_fanout",
        "smoke_shard_done_sha256": "c" * 64,
        "all_gates_passed": True,
        "full_9000_paired_fanout_authorized": True,
    }
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_bytes(subject.canonical_bytes(smoke))
    smoke_directory = tmp_path / "smoke"
    smoke_directory.mkdir()
    (smoke_directory / "SHARD_DONE.json").write_bytes(b"pinned-smoke")
    profile_sha = "d" * 64
    monkeypatch.setattr(executor, "_profile_sha256", lambda: profile_sha)
    monkeypatch.setattr(
        contract,
        "validate_smoke_gate_receipt",
        lambda value, **_kwargs: deepcopy(value),
    )
    inventory = subject._inventory(smoke_directory)  # type: ignore[attr-defined]
    core = {
        "schema": subject.PORTABLE_AUTHORIZATION_SCHEMA,
        "status": "controller_source_replayed_portable_fanout_authorized",
        "plan_sha256": subject.canonical_sha256(plan),
        "fresh_quality_gate_schema": quality_gate.GATE_SCHEMA,
        "fresh_quality_gate_file_sha256": hashlib.sha256(
            fresh_path.read_bytes()
        ).hexdigest(),
        "fresh_quality_gate_canonical_sha256": subject.canonical_sha256(fresh),
        "fresh_quality_merge_sha256": fresh["merge_sha256"],
        "fresh_quality_decision": fresh["decision"],
        "dataset_smoke_gate_schema": contract.SMOKE_GATE_SCHEMA,
        "dataset_smoke_gate_file_sha256": hashlib.sha256(
            smoke_path.read_bytes()
        ).hexdigest(),
        "dataset_smoke_gate_canonical_sha256": subject.canonical_sha256(smoke),
        "dataset_smoke_gate_decision": smoke["decision"],
        "smoke_shard_id": contract.SMOKE_SHARD_ID,
        "smoke_shard_done_sha256": smoke["smoke_shard_done_sha256"],
        "smoke_shard_inventory": inventory,
        "smoke_shard_inventory_sha256": subject.canonical_sha256(inventory),
        "current_profile_registry_sha256": profile_sha,
        "controller_source_replayed": True,
        "worker_replay_boundary": (
            "exact_gate_file_hashes_plus_source_replayed_smoke_shard"
        ),
        "full_9000_paired_fanout_authorized": True,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    receipt = {
        **core,
        "authorization_sha256": subject.canonical_sha256(core),
    }
    receipt_path = tmp_path / "portable.json"
    receipt_path.write_bytes(subject.canonical_bytes(receipt))
    return {
        "plan": plan,
        "fresh_path": fresh_path,
        "smoke": smoke,
        "smoke_path": smoke_path,
        "smoke_directory": smoke_directory,
        "receipt": receipt,
        "receipt_path": receipt_path,
    }


def test_worker_validates_hash_pinned_gate_and_replayed_smoke_without_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path, monkeypatch)
    observed = subject.validate_portable_fanout_authorization(
        values["receipt"],
        plan=values["plan"],
        fresh_quality_gate_path=values["fresh_path"],
        smoke_gate_receipt_path=values["smoke_path"],
        smoke_shard_directory=values["smoke_directory"],
    )
    assert observed["controller_source_replayed"] is True
    assert "exact_gate_file_hashes" in observed["worker_replay_boundary"]

    values["fresh_path"].write_bytes(subject.canonical_bytes({"tampered": True}))
    with pytest.raises((ValueError, PermissionError)):
        subject.validate_portable_fanout_authorization(
            values["receipt"],
            plan=values["plan"],
            fresh_quality_gate_path=values["fresh_path"],
            smoke_gate_receipt_path=values["smoke_path"],
            smoke_shard_directory=values["smoke_directory"],
        )


def test_portable_worker_derives_exact_existing_full_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path, monkeypatch)
    captured = {}

    def run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "partial_safe_to_resume", "new_pair_count": 0}

    monkeypatch.setattr(executor, "_run_authorized_shard", run)
    shard_id = next(
        row["shard_id"]
        for row in values["plan"]["shards"]
        if row["shard_id"] != contract.SMOKE_SHARD_ID
    )
    directory = tmp_path / "worker-shard"
    result = subject.run_portable_dataset_shard(
        plan=values["plan"],
        shard_id=shard_id,
        shard_directory=directory,
        portable_authorization_path=values["receipt_path"],
        fresh_quality_gate_path=values["fresh_path"],
        smoke_gate_receipt_path=values["smoke_path"],
        smoke_shard_directory=values["smoke_directory"],
        max_new_pairs=0,
    )
    authorization = captured["authorization"]
    assert authorization["schema"] == executor.FULL_AUTHORIZATION_SCHEMA
    assert authorization["shard_id"] == shard_id
    assert authorization["source_replayed"] is True
    assert authorization["full_9000_paired_fanout_authorized"] is True
    assert result["cloud_worker_portable_boundary"] is True
    assert (directory / executor.AUTHORIZATION_NAME).is_file()
