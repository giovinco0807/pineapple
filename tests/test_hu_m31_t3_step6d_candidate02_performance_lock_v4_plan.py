from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_candidate02_performance_lock_v4_plan as plan,
)
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plan.canonical_bytes(value))


def _audit() -> dict:
    root_hashes = [f"{index:064x}" for index in range(100)]
    return {
        "root_file_count": 100,
        "paired_hand_count": 100,
        "observation_count": 200,
        "root_hashes": root_hashes,
        "root_hash_aggregate_sha256": plan.canonical_sha256(root_hashes),
        "profile_counts": {
            profile: 20 for profile in plan.M31_T3_BEHAVIOR_PROFILES
        },
        "seat_counts": {"first": 100, "second": 100},
        "seed_set_sha256": runner.CANDIDATE02_PERFORMANCE_LOCK_V4_SEED_SET_SHA256,
        "seed_overlap_counts": {
            "existing_step6d_union": 0,
            "candidate02_development_run009": 0,
            "performance_lock_v1": 0,
            "performance_lock_rearm1": 0,
            "performance_lock_rearm2": 0,
        },
        "observation_fingerprint_count": 200,
        "observation_fingerprint_aggregate_sha256": "a" * 64,
        "all_root_artifacts_valid": True,
        "current_profile_unchanged": True,
        "old_root_or_seed_reuse": False,
    }


def test_real_run009_builds_frozen_balanced_v4_plan() -> None:
    value = plan.build_performance_lock_v4_plan()

    assert plan.validate_performance_lock_v4_plan(value) == value
    assert plan.canonical_sha256(value) == plan.PLAN_SHA256
    assert value["schema"] == plan.PLAN_SCHEMA
    assert value["scope"] == plan.PLAN_SCOPE
    assert value["run_contract_digest"] == plan.RUN_CONTRACT_DIGEST
    assert value["run_contract"]["budget"] == {
        "candidate_samples": 8,
        "evaluation_samples": 32,
        "downstream_t3_samples": 4,
        "downstream_t4_samples": 0,
    }
    assert value["allocation"] == {"workers": 1, "rayon_threads_per_worker": 16}
    assert value["source_identity"]["candidate_library_sha256"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
    )
    assert value["source_identity"]["reference_library_sha256"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
    )
    assert value["seed_contract"]["namespace_bases"] == {
        "hand": 720_108_071_901,
        "behavior": 721_108_071_901,
        "candidate": 722_108_071_901,
        "evaluation": 723_108_071_901,
        "child": 724_108_071_901,
        "confirmation": 725_108_071_901,
    }
    assert value["seed_contract"]["seed_min"] == 720_108_071_901
    assert value["seed_contract"]["seed_max"] == 725_207_072_198
    assert not any(value["seed_contract"]["existing_step6d_schedule_overlap_counts"].values())
    assert len(value["shards"]) == 10
    assert len(value["jobs"]) == 20
    assert sorted(
        index
        for shard in value["shards"]
        for index in shard["work_hand_indices"]
    ) == list(range(100))
    assert all(
        set(shard["profile_counts"].values()) == {2}
        for shard in value["shards"]
    )
    assert all(
        value[field] is False
        for field in (
            "root_content_opened",
            "cloud_started",
            "quality_pilot_authorized",
            "artifact_fanout_authorized",
            "training_authorized",
            "current_profile_changed",
            "named_profile_added",
            "runtime_policy_activated",
            "m31_complete",
        )
    )


def test_v4_runner_contract_and_seed_schedule_are_fresh_and_exact() -> None:
    contract = runner.build_run_contract(
        candidate_library_sha256=plan.CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=plan.REFERENCE_LIBRARY_SHA256,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT,
    )
    assert runner.canonical_sha256(contract) == plan.RUN_CONTRACT_DIGEST
    assert contract["authorizing_gate_receipt_file_sha256"] == (
        plan.RUN009_GATE_RECEIPT_FILE_SHA256
    )
    assert contract["authorizing_gate_receipt_sha256"] == (
        plan.RUN009_GATE_RECEIPT_SHA256
    )
    manifest = runner.build_shard_manifest(
        run_contract=contract,
        source_role="candidate",
        work_hand_indices=range(10),
    )
    assert manifest["schema"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_V4_SHARD_MANIFEST_SCHEMA
    )
    assert runner._source_hand_schema(contract) == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_V4_SOURCE_HAND_SCHEMA
    )
    assert runner._done_schema(contract) == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_V4_DONE_SCHEMA
    )
    assert runner.candidate02_performance_lock_v4_schedule_row(99)["seeds"] == {
        key: base + 99 * 1_000_003
        for key, base in runner.CANDIDATE02_PERFORMANCE_LOCK_V4_NAMESPACE_BASES.items()
    }

    existing_sets = [
        {
            seed
            for index in range(100)
            for seed in values(index).values()
        }
        for values in (
            runner.candidate02_seed_values,
            runner.candidate02_performance_lock_seed_values,
            runner.candidate02_performance_lock_recovery_seed_values,
            runner.candidate02_performance_lock_recovery_v3_seed_values,
        )
    ]
    v4 = {
        seed
        for index in range(100)
        for seed in runner.candidate02_performance_lock_v4_seed_values(index).values()
    }
    assert len(v4) == 600
    assert all(not (v4 & existing) for existing in existing_sets)


def test_receipt_plan_and_current_profile_drift_fail_closed(tmp_path: Path) -> None:
    original = json.loads(
        plan.DEFAULT_RUN009_SCIENTIFIC_GATE_RECEIPT_PATH.read_text(encoding="utf-8")
    )
    tampered_receipt = copy.deepcopy(original)
    tampered_receipt["all_gates_passed"] = False
    tampered_path = tmp_path / "tampered-receipt.json"
    _write(tampered_path, tampered_receipt)
    with pytest.raises(ValueError, match="file hash changed"):
        plan.load_run009_scientific_gate_receipt(tampered_path)

    value = plan.build_performance_lock_v4_plan()
    tampered_plan = copy.deepcopy(value)
    tampered_plan["run009_scientific_gate"]["performance_lock_authorized"] = False
    with pytest.raises(ValueError):
        plan.validate_performance_lock_v4_plan(tampered_plan)
    tampered_plan = copy.deepcopy(value)
    tampered_plan["seed_contract"]["namespace_bases"]["hand"] += 1
    with pytest.raises(ValueError):
        plan.validate_performance_lock_v4_plan(tampered_plan)

    registry = tmp_path / "ai_profiles.py"
    registry.write_bytes(plan.DEFAULT_CURRENT_PROFILE_REGISTRY_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="registry hash changed"):
        plan.build_performance_lock_v4_plan(
            current_profile_registry_path=registry
        )


def test_plan_write_is_canonical_exclusive_and_does_not_touch_roots(
    tmp_path: Path,
) -> None:
    output = tmp_path / "plan.json"
    root_output = tmp_path / "execution"
    before = plan.sha256_file(plan.DEFAULT_CURRENT_PROFILE_REGISTRY_PATH)

    value = plan.write_performance_lock_v4_plan(output_path=output)

    assert output.read_bytes() == plan.canonical_bytes(value)
    assert plan.canonical_sha256(value) == plan.PLAN_SHA256
    assert not root_output.exists()
    assert plan.sha256_file(plan.DEFAULT_CURRENT_PROFILE_REGISTRY_PATH) == before
    with pytest.raises(FileExistsError):
        plan.write_performance_lock_v4_plan(output_path=output)


def test_materialization_claim_precedes_roots_and_receipt_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "execution"
    receipt_path = output / "MATERIALIZATION_RECEIPT.json"
    plan.write_performance_lock_v4_plan(output_path=plan_path)
    calls = []

    def fake_materialize(**kwargs):
        claim_path = Path(kwargs["output_dir"]) / plan.CLAIM_NAME
        assert claim_path.is_file()
        calls.append(tuple(kwargs["indices"]))
        return [{} for _ in range(100)]

    monkeypatch.setattr(runner, "_materialize_roots", fake_materialize)
    monkeypatch.setattr(plan, "_audit_roots", lambda **_kwargs: _audit())

    first = plan.materialize_performance_lock_v4_roots(
        plan_path=plan_path,
        output_dir=output,
        materialization_receipt_path=receipt_path,
    )
    second = plan.materialize_performance_lock_v4_roots(
        plan_path=plan_path,
        output_dir=output,
        materialization_receipt_path=receipt_path,
    )

    assert first == second == plan.validate_materialization_receipt(first)
    assert len(calls) == 2
    assert (output / plan.CLAIM_NAME).is_file()
    assert receipt_path.read_bytes() == plan.canonical_bytes(first)
    assert first["old_root_or_seed_reuse"] is False
    assert not any(first["seed_overlap_counts"].values())


def test_materialization_refuses_unclaimed_root_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "execution"
    rogue = output / "roots" / "hand_000.json"
    plan.write_performance_lock_v4_plan(output_path=plan_path)
    rogue.parent.mkdir(parents=True)
    rogue.write_text("{}", encoding="utf-8")
    called = False

    def fake_materialize(**_kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(runner, "_materialize_roots", fake_materialize)
    with pytest.raises(ValueError, match="exists before"):
        plan.materialize_performance_lock_v4_roots(
            plan_path=plan_path,
            output_dir=output,
            materialization_receipt_path=output / "receipt.json",
        )
    assert called is False
    assert not (output / plan.CLAIM_NAME).exists()


def test_seal_replays_materialization_and_is_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "execution"
    receipt_path = output / "MATERIALIZATION_RECEIPT.json"
    seal_path = output / "ROOT_SEAL.json"
    plan.write_performance_lock_v4_plan(output_path=plan_path)
    monkeypatch.setattr(
        runner, "_materialize_roots", lambda **_kwargs: [{} for _ in range(100)]
    )
    monkeypatch.setattr(plan, "_audit_roots", lambda **_kwargs: _audit())
    plan.materialize_performance_lock_v4_roots(
        plan_path=plan_path,
        output_dir=output,
        materialization_receipt_path=receipt_path,
    )
    monkeypatch.setattr(
        plan, "_load_root_files", lambda **_kwargs: [{} for _ in range(100)]
    )

    first = plan.seal_performance_lock_v4_roots(
        plan_path=plan_path,
        output_dir=output,
        materialization_receipt_path=receipt_path,
        root_seal_path=seal_path,
    )
    second = plan.seal_performance_lock_v4_roots(
        plan_path=plan_path,
        output_dir=output,
        materialization_receipt_path=receipt_path,
        root_seal_path=seal_path,
    )

    assert first == second == plan.validate_root_seal(first)
    assert first["sealed"] is True
    assert first["root_hashes"] == _audit()["root_hashes"]
    assert seal_path.read_bytes() == plan.canonical_bytes(first)
    tampered = copy.deepcopy(first)
    tampered["seat_counts"]["first"] = 99
    with pytest.raises(ValueError):
        plan.validate_root_seal(tampered)
