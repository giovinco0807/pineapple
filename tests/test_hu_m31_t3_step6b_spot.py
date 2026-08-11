from __future__ import annotations

import json
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6b_spot as spot
from ofc_regular.run_hu_m31_t3_step6b_shard import (
    AUTHORIZED_SHARDS,
    EXPECTED_FEATURE_ENCODER_SHA256,
    EXPECTED_NATIVE_LIBRARY_SHA256,
    STEP6B_PACKAGE_SCHEMA,
    STEP6B_PARITY_SCHEMA,
    STEP6B_SHARD_SCHEMA,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> dict:
    return {
        "schema": STEP6B_PACKAGE_SCHEMA,
        "status": "packaged_local_no_gcloud",
        "run_name": "regular-hu-m31-step6b-test",
        "source_sha256": "a" * 64,
        "schedule_sha256": "b" * 64,
        "startup_sha256": "c" * 64,
        "parity_golden_sha256": "d" * 64,
        "step6a_accepted": {
            "done_sha256": "e" * 64,
            "summary_sha256": "f" * 64,
        },
    }


def _write(path: Path, value: dict) -> None:
    path.write_bytes(spot.canonical_bytes(value))


def test_step6b_schedule_reserves_all_ten_rows_but_only_remaining_are_authorized():
    rows = spot.build_schedule("regular-hu-m31-step6b-test")
    assert len(rows) == 10
    assert [row["shard"] for row in rows] == list(range(10))
    assert rows[0]["global_hand_start"] == 0
    assert rows[9]["global_hand_start"] == 225
    assert all(row["schema"] == STEP6B_SHARD_SCHEMA for row in rows)
    assert list(AUTHORIZED_SHARDS) == list(range(1, 10))


@pytest.mark.parametrize(
    "shards",
    [(), (0,), (10,), (1, 1), (1, 2, 3, 4), (True,)],
)
def test_step6b_launch_batch_fails_closed_outside_one_to_three_unique_remaining_shards(
    shards,
):
    with pytest.raises(ValueError, match="1-3 unique shards"):
        spot._bounded_shards(shards)


def test_step6b_launch_batch_accepts_explicit_bounded_waves():
    assert spot._bounded_shards((1, 2, 3)) == (1, 2, 3)
    assert spot._bounded_shards((7, 8, 9)) == (7, 8, 9)
    assert spot._parse_shards("4,5,6") == (4, 5, 6)


def test_step6b_package_prerequisite_binds_real_accepted_step6a_v4():
    evidence = spot._step6a_evidence(
        status_path=REPO_ROOT / "configs/hu_joint_policy_m31_t3_step6a_status.json",
        run_dir=REPO_ROOT / "outputs/gcp_runs/regular-hu-m31-step6a-s0-20260717-004",
        received_dir=REPO_ROOT / "outputs/hu_joint_policy/m31_t3_step6a/received_v4",
    )
    assert evidence["run_name"] == "regular-hu-m31-step6a-s0-20260717-004"
    assert evidence["native_library_sha256"] == EXPECTED_NATIVE_LIBRARY_SHA256
    assert evidence["feature_encoder_sha256"] == EXPECTED_FEATURE_ENCODER_SHA256


def test_step6b_dry_run_receipt_binds_package_source_manifest_and_parity(
    tmp_path, monkeypatch
):
    manifest = _manifest()
    _write(tmp_path / "manifest.json", {"manifest": "fixture"})
    parity_path = tmp_path / "parity.json"
    parity = {
        "schema": STEP6B_PARITY_SCHEMA,
        "all_gates_passed": True,
        "source_package_sha256": manifest["source_sha256"],
        "manifest_sha256": spot.sha256_file(tmp_path / "manifest.json"),
        "parity_golden_sha256": manifest["parity_golden_sha256"],
        "native_library_sha256": EXPECTED_NATIVE_LIBRARY_SHA256,
        "teacher_generation_started": False,
        "current_profile_changed": False,
        "production_fanout_authorized": False,
    }
    _write(parity_path, parity)
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)
    receipt = spot.record_local_dry_run(run_dir=tmp_path, parity_report=parity_path)
    assert receipt["status"] == "pass"
    assert receipt["source_sha256"] == "a" * 64
    assert receipt["production_fanout_authorized"] is False


def test_step6b_authorization_is_exactly_shards_one_through_nine(tmp_path, monkeypatch):
    manifest = _manifest()
    _write(tmp_path / "manifest.json", {"manifest": "fixture"})
    dry = {
        "schema": spot.STEP6B_DRY_RUN_SCHEMA,
        "status": "pass",
        "manifest_sha256": spot.sha256_file(tmp_path / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "teacher_generation_started": False,
        "gcloud_invoked": False,
    }
    _write(tmp_path / "local_dry_run_receipt.json", dry)
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)
    auth = spot.authorize_launch(tmp_path)
    assert auth["authorized_shards"] == list(range(1, 10))
    assert auth["remaining_canary_shards_authorized"] is True
    assert auth["production_fanout_authorized"] is False
    assert auth["current_profile_changed"] is False


def test_step6b_launch_validation_rejects_authorization_fanout_mutation(
    tmp_path, monkeypatch
):
    manifest = _manifest()
    _write(tmp_path / "manifest.json", {"manifest": "fixture"})
    auth = {
        "schema": spot.STEP6B_AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "run_name": manifest["run_name"],
        "manifest_sha256": spot.sha256_file(tmp_path / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "step6a_accepted": manifest["step6a_accepted"],
        "authorized_shards": list(range(1, 10)),
        "spot_authorized": True,
        "remaining_canary_shards_authorized": True,
        "production_fanout_authorized": True,
        "root_execution_started": False,
        "current_profile_changed": False,
        "authorized_unix_seconds": 1.0,
    }
    _write(tmp_path / "launch_authorization.json", auth)
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)
    with pytest.raises(ValueError, match="authorization changed"):
        spot.validate_launch(tmp_path)


def test_step6b_source_declares_bounded_waves_and_no_production_activation():
    source = (REPO_ROOT / "src/ofc_regular/hu_m31_t3_step6b_spot.py").read_text(
        encoding="utf-8"
    )
    assert "MAX_LAUNCH_BATCH = 3" in source
    assert '"production_fanout_authorized": False' in source
    assert '"current_profile_changed": False' in source
    assert '"training_eligible": False' in source
    assert "results/shard-{padded}" in source
