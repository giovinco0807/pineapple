from __future__ import annotations

import json
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6c_spot as spot
from ofc_regular.hu_m31_t3_step6c_contract import (
    ACCEPTED_FEATURE_ENCODER_SHA256,
    ACCEPTED_NATIVE_LIBRARY_SHA256,
    CONFIRMATION_HAND_INDICES,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(spot.canonical_bytes(value))


def _manifest() -> dict:
    return {
        "schema": spot.STEP6C_PACKAGE_SCHEMA,
        "status": "packaged_local_no_gcloud",
        "run_name": "regular-hu-m31-step6c-test",
        "source_sha256": "a" * 64,
        "schedule_sha256": "b" * 64,
        "startup_sha256": "c" * 64,
        "parity_golden_sha256": "9" * 64,
        "step5_contract_canonical_sha256": "d" * 64,
        "step6b_validation_sha256": "e" * 64,
        "step6c_contract_canonical_sha256": "f" * 64,
    }


def _dry_receipt(tmp_path: Path, manifest: dict) -> dict:
    return {
        "schema": spot.STEP6C_DRY_RUN_SCHEMA,
        "status": "pass",
        "run_name": manifest["run_name"],
        "manifest_sha256": spot.sha256_file(tmp_path / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "parity_report_sha256": "1" * 64,
        "resume_report_sha256": "2" * 64,
        "linux_production_budget_parity": True,
        "production_budget_speed_smoke": True,
        "speed_smoke_seconds_by_seat": {"first": 100.0, "second": 2.0},
        "speed_smoke_limits_seconds": {"first": 180.0, "second": 6.0},
        "resume_recovery": True,
        "gcloud_invoked": False,
        "training_eligible": False,
        "production_fanout_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
    }


def _authorization(tmp_path: Path, manifest: dict) -> dict:
    return {
        "schema": spot.STEP6C_AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "run_name": manifest["run_name"],
        "manifest_sha256": spot.sha256_file(tmp_path / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "step5_contract_canonical_sha256": manifest["step5_contract_canonical_sha256"],
        "step6b_validation_sha256": manifest["step6b_validation_sha256"],
        "step6c_contract_canonical_sha256": manifest[
            "step6c_contract_canonical_sha256"
        ],
        "dry_run_receipt_sha256": spot.sha256_file(
            tmp_path / "local_dry_run_receipt.json"
        ),
        "authorized_shards": [0, 1],
        "spot_authorized": True,
        "quality_pilot_only": True,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "root_execution_started": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "authorized_unix_seconds": 1.0,
    }


def _parity_report(*, manifest: dict, manifest_sha256: str) -> dict:
    return {
        "schema": spot.STEP6C_PARITY_SCHEMA,
        "status": "pass",
        "source_package_sha256": manifest["source_sha256"],
        "manifest_sha256": manifest_sha256,
        "parity_golden_sha256": manifest["parity_golden_sha256"],
        "native_library_sha256": ACCEPTED_NATIVE_LIBRARY_SHA256,
        "rows": [
            {
                "seat": "first",
                "observation_fingerprint": "3" * 64,
                "expected_portable_sha256": "4" * 64,
                "observed_portable_sha256": "4" * 64,
                "wall_seconds": 100.0,
                "match": True,
            },
            {
                "seat": "second",
                "observation_fingerprint": "5" * 64,
                "expected_portable_sha256": "6" * 64,
                "observed_portable_sha256": "6" * 64,
                "wall_seconds": 2.0,
                "match": True,
            },
        ],
        "gates": {key: True for key in spot._PARITY_GATE_KEYS},
        "all_gates_passed": True,
        "current_profile_changed": False,
        "teacher_generation_started": False,
        "production_fanout_authorized": False,
    }


def _resume_report(*, manifest: dict, manifest_sha256: str, parity_sha256: str) -> dict:
    return {
        "schema": spot.STEP6C_SUMMARY_SCHEMA,
        "status": "interrupted_for_resume_drill",
        "run_name": manifest["run_name"],
        "shard": 0,
        "hand_indices": list(range(25)),
        "source_package_sha256": manifest["source_sha256"],
        "manifest_sha256": manifest_sha256,
        "parity_report_sha256": parity_sha256,
        "completed_tasks": 2,
        "pending_tasks": 23,
        "resumed_task_count": 1,
        "quality_pilot_authorized": True,
        "training_eligible": False,
        "current_profile_changed": False,
        "production_fanout_authorized": False,
        "m31_complete": False,
    }


def _scientific_shard(
    tmp_path: Path,
    *,
    shard: int = 0,
    manifest_sha256: str = "1" * 64,
    authorization_sha256: str = "2" * 64,
) -> tuple[Path, dict]:
    shard_dir = tmp_path / f"shard-{shard:03d}"
    manifest = _manifest()
    parity = _parity_report(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    _write(shard_dir / "parity.json", parity)
    gates = {key: True for key in spot._SHARD_SUMMARY_GATE_KEYS}
    summary = {
        "schema": spot.STEP6C_SUMMARY_SCHEMA,
        "status": "pass",
        "run_name": manifest["run_name"],
        "shard": shard,
        "source_package_sha256": manifest["source_sha256"],
        "manifest_sha256": manifest_sha256,
        "parity_report_sha256": spot.sha256_file(shard_dir / "parity.json"),
        "resumed_task_count": 1,
        "gates": gates,
        "all_gates_passed": True,
        "teacher_value_status": "diagnostic_not_match_EV",
        "training_eligible": False,
        "production_fanout_authorized": False,
        "current_profile_changed": False,
    }
    for relative in spot._scientific_result_relatives(shard):
        if relative == "summary.json":
            _write(shard_dir / relative, summary)
        elif relative == "parity.json":
            continue
        else:
            _write(shard_dir / relative, {"relative": relative})
    files = {
        relative: {
            "sha256": spot.sha256_file(shard_dir / relative),
            "bytes": (shard_dir / relative).stat().st_size,
        }
        for relative in spot._scientific_result_relatives(shard)
    }
    done = {
        "schema": spot.STEP6C_DONE_SCHEMA,
        "status": "complete",
        "quality_status": "pass",
        "run_name": manifest["run_name"],
        "shard": shard,
        "source_sha256": manifest["source_sha256"],
        "manifest_sha256": manifest_sha256,
        "schedule_sha256": manifest["schedule_sha256"],
        "authorization_sha256": authorization_sha256,
        "native_library_sha256": ACCEPTED_NATIVE_LIBRARY_SHA256,
        "files": files,
        "resume_drill_passed": True,
        "authorized_shards": [0, 1],
        "quality_pilot_only": True,
        "training_eligible": False,
        "production_fanout_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "completed_unix_seconds": 1.0,
    }
    _write(shard_dir / "DONE.json", done)
    return shard_dir, manifest


def _refresh_done_file(shard_dir: Path, relative: str) -> None:
    done_path = shard_dir / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    path = shard_dir / relative
    done["files"][relative] = {
        "sha256": spot.sha256_file(path),
        "bytes": path.stat().st_size,
    }
    done_path.write_bytes(spot.canonical_bytes(done))


def test_step6c_schedule_is_exactly_two_bounded_pilot_shards() -> None:
    rows = spot.build_schedule("regular-hu-m31-step6c-test")
    assert len(rows) == 50
    assert [row["train_hand_index"] for row in rows] == list(range(50))
    assert [row["pilot_shard"] for row in rows[:25]] == [0] * 25
    assert [row["pilot_shard"] for row in rows[25:]] == [1] * 25
    observed_confirmation = {
        row["train_hand_index"] for row in rows if row["confirmation"]
    }
    assert observed_confirmation == set(CONFIRMATION_HAND_INDICES)


@pytest.mark.parametrize(
    "shards",
    [(), (2,), (-1,), (0, 0), (0, 1, 2), (True,)],
)
def test_step6c_launch_rejects_everything_outside_two_shard_pilot(shards) -> None:
    with pytest.raises(ValueError, match="1-2 unique shards from 0..1"):
        spot._bounded_shards(shards)


def test_step6c_launch_accepts_one_or_both_explicit_shards() -> None:
    assert spot._bounded_shards((0,)) == (0,)
    assert spot._bounded_shards((1,)) == (1,)
    assert spot._bounded_shards((0, 1)) == (0, 1)
    assert spot._parse_shards("0,1") == (0, 1)


def test_step6c_rejects_drifted_current_target_linux_binaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_engine = REPO_ROOT / "target/release/libofc_hu_m3_engine.so"
    current_feature = REPO_ROOT / "target/release/libofc_stage3_feature_encoder.so"
    assert spot.sha256_file(current_engine) != ACCEPTED_NATIVE_LIBRARY_SHA256
    assert spot.sha256_file(current_feature) != ACCEPTED_FEATURE_ENCODER_SHA256
    monkeypatch.setattr(
        spot,
        "_validate_prerequisites",
        lambda **_kwargs: {
            "step6c_contract_canonical": "f" * 64,
        },
    )
    monkeypatch.setattr(spot, "_validate_parity_golden", lambda *_a, **_k: {})
    contract = tmp_path / "contract.json"
    validation = tmp_path / "validation.json"
    parity = tmp_path / "parity.json"
    _write(contract, {})
    _write(validation, {})
    _write(parity, {})

    with pytest.raises(ValueError, match="accepted Step 6b Linux binaries"):
        spot.package_step6c(
            run_name="regular-hu-m31-step6c-test",
            run_dir=tmp_path / "run",
            repository_root=REPO_ROOT,
            linux_library=current_engine,
            linux_feature_encoder=current_feature,
            step5_validation=tmp_path / "unused-step5.json",
            step6b_validation=tmp_path / "unused-step6b.json",
            step6b_received_dir=tmp_path / "unused-received",
            step6c_contract=contract,
            step6c_validation=validation,
            parity_golden=parity,
        )


def test_step6c_parity_golden_is_write_once_and_uses_accepted_windows_dll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = []

    class _Decision:
        def __init__(self, fingerprint: str) -> None:
            self.fingerprint = fingerprint

        def to_dict(self) -> dict:
            return {"observation_fingerprint": self.fingerprint, "value": 1.0}

    class _Solver:
        def __init__(self, config) -> None:
            configs.append(config)

        def solve(self, observation):
            return _Decision(observation.fingerprint())

    monkeypatch.setattr(spot, "HuM31T3SearchSolver", _Solver)
    output = tmp_path / "parity_golden.json"
    windows = REPO_ROOT / "target/m31_opt_build/release/ofc_hu_m3_engine.dll"
    contract = REPO_ROOT / "configs/hu_joint_policy_m31_t3_step6c_contract.json"

    golden = spot.build_parity_golden(
        output_path=output,
        windows_library=windows,
        step6c_contract=contract,
    )

    assert golden["source_windows_library_sha256"] == (
        spot.EXPECTED_WINDOWS_LIBRARY_SHA256
    )
    assert golden["runtime_config"]["candidate_samples"] == 8
    assert golden["runtime_config"]["evaluation_samples"] == 32
    assert golden["runtime_config"]["downstream_t3_samples"] == 4
    assert [row["seat"] for row in golden["rows"]] == ["first", "second"]
    assert len(configs) == 1
    with pytest.raises(FileExistsError, match="immutable Step 6c artifact"):
        spot.build_parity_golden(
            output_path=output,
            windows_library=windows,
            step6c_contract=contract,
        )


def test_step6c_authorization_is_explicit_pilot_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    _write(tmp_path / "manifest.json", {"fixture": True})
    dry = _dry_receipt(tmp_path, manifest)
    _write(tmp_path / "local_dry_run_receipt.json", dry)
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)

    authorization = spot.authorize_launch(tmp_path)

    assert authorization["authorized_shards"] == [0, 1]
    assert authorization["quality_pilot_only"] is True
    assert authorization["spot_authorized"] is True
    assert authorization["production_fanout_authorized"] is False
    assert authorization["training_eligible"] is False
    assert authorization["current_profile_changed"] is False
    assert authorization["named_profile_added"] is False


def test_step6c_authorization_rejects_missing_speed_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    _write(tmp_path / "manifest.json", {"fixture": True})
    receipt = _dry_receipt(tmp_path, manifest)
    receipt.pop("production_budget_speed_smoke")
    _write(tmp_path / "local_dry_run_receipt.json", receipt)
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)

    with pytest.raises(ValueError, match="dry-run receipt"):
        spot.authorize_launch(tmp_path)


def test_step6c_dry_run_rejects_unbound_resume_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, {"fixture": True})
    parity_path = tmp_path / "parity.json"
    _write(
        parity_path,
        _parity_report(
            manifest=manifest,
            manifest_sha256=spot.sha256_file(manifest_path),
        ),
    )
    resume_path = tmp_path / "resume.json"
    resume = _resume_report(
        manifest=manifest,
        manifest_sha256=spot.sha256_file(manifest_path),
        parity_sha256=spot.sha256_file(parity_path),
    )
    resume["run_name"] = "unrelated-run"
    _write(resume_path, resume)
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)

    with pytest.raises(ValueError, match="dry-run failed"):
        spot.record_local_dry_run(
            run_dir=tmp_path,
            parity_report=parity_path,
            resume_report=resume_path,
        )


def test_step6c_dry_run_accepts_only_exact_runner_parity_and_interrupted_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, {"fixture": True})
    manifest_sha = spot.sha256_file(manifest_path)
    parity_path = tmp_path / "parity.json"
    _write(
        parity_path,
        _parity_report(manifest=manifest, manifest_sha256=manifest_sha),
    )
    resume_path = tmp_path / "resume.json"
    _write(
        resume_path,
        _resume_report(
            manifest=manifest,
            manifest_sha256=manifest_sha,
            parity_sha256=spot.sha256_file(parity_path),
        ),
    )
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)

    receipt = spot.record_local_dry_run(
        run_dir=tmp_path,
        parity_report=parity_path,
        resume_report=resume_path,
    )

    assert receipt["status"] == "pass"
    assert receipt["resume_recovery"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shard", False),
        ("resumed_task_count", True),
        ("resumed_task_count", 3),
    ],
)
def test_step6c_dry_run_requires_exact_integer_resume_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, {"fixture": True})
    manifest_sha = spot.sha256_file(manifest_path)
    parity_path = tmp_path / "parity.json"
    _write(
        parity_path,
        _parity_report(manifest=manifest, manifest_sha256=manifest_sha),
    )
    resume = _resume_report(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        parity_sha256=spot.sha256_file(parity_path),
    )
    resume[field] = value
    resume_path = tmp_path / "resume.json"
    _write(resume_path, resume)
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)

    with pytest.raises(ValueError, match="local Linux dry-run failed"):
        spot.record_local_dry_run(
            run_dir=tmp_path,
            parity_report=parity_path,
            resume_report=resume_path,
        )


@pytest.mark.parametrize("mutation", ["no_go", "duplicate_seat", "extra_key"])
def test_step6c_dry_run_rejects_nonexact_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, {"fixture": True})
    manifest_sha = spot.sha256_file(manifest_path)
    parity = _parity_report(manifest=manifest, manifest_sha256=manifest_sha)
    if mutation == "no_go":
        parity["status"] = "no_go"
    elif mutation == "duplicate_seat":
        parity["rows"].insert(1, dict(parity["rows"][0]))
    else:
        parity["unexpected"] = False
    parity_path = tmp_path / "parity.json"
    _write(parity_path, parity)
    resume_path = tmp_path / "resume.json"
    _write(
        resume_path,
        _resume_report(
            manifest=manifest,
            manifest_sha256=manifest_sha,
            parity_sha256=spot.sha256_file(parity_path),
        ),
    )
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)

    with pytest.raises(ValueError, match="local Linux dry-run failed"):
        spot.record_local_dry_run(
            run_dir=tmp_path,
            parity_report=parity_path,
            resume_report=resume_path,
        )


@pytest.mark.parametrize("status", ["pass", "no_go"])
def test_step6c_dry_run_rejects_final_summary_as_resume_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, {"fixture": True})
    manifest_sha = spot.sha256_file(manifest_path)
    parity_path = tmp_path / "parity.json"
    _write(
        parity_path,
        _parity_report(manifest=manifest, manifest_sha256=manifest_sha),
    )
    resume = _resume_report(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        parity_sha256=spot.sha256_file(parity_path),
    )
    resume.update(
        status=status,
        completed_tasks=25,
        pending_tasks=0,
    )
    resume_path = tmp_path / "resume.json"
    _write(resume_path, resume)
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)

    with pytest.raises(ValueError, match="local Linux dry-run failed"):
        spot.record_local_dry_run(
            run_dir=tmp_path,
            parity_report=parity_path,
            resume_report=resume_path,
        )


def test_step6c_authorization_rejects_extra_dry_receipt_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    _write(tmp_path / "manifest.json", {"fixture": True})
    receipt = _dry_receipt(tmp_path, manifest)
    receipt["unexpected"] = False
    _write(tmp_path / "local_dry_run_receipt.json", receipt)
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)

    with pytest.raises(ValueError, match="dry-run receipt changed"):
        spot.authorize_launch(tmp_path)


def test_step6c_launch_validation_rejects_fanout_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    _write(tmp_path / "manifest.json", {"fixture": True})
    _write(
        tmp_path / "local_dry_run_receipt.json",
        _dry_receipt(tmp_path, manifest),
    )
    auth = _authorization(tmp_path, manifest)
    auth["production_fanout_authorized"] = True
    _write(tmp_path / "launch_authorization.json", auth)
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)
    with pytest.raises(ValueError, match="authorization changed"):
        spot.validate_launch(tmp_path)


def test_step6c_launch_validation_rejects_extra_authorization_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    _write(tmp_path / "manifest.json", {"fixture": True})
    _write(
        tmp_path / "local_dry_run_receipt.json",
        _dry_receipt(tmp_path, manifest),
    )
    authorization = _authorization(tmp_path, manifest)
    authorization["unexpected"] = False
    _write(tmp_path / "launch_authorization.json", authorization)
    monkeypatch.setattr(spot, "validate_package", lambda _path: manifest)

    with pytest.raises(ValueError, match="authorization changed"):
        spot.validate_launch(tmp_path)


def test_step6c_source_pins_spot_machine_zones_write_once_and_receive() -> None:
    source = (REPO_ROOT / "src/ofc_regular/hu_m31_t3_step6c_spot.py").read_text(
        encoding="utf-8"
    )
    inherited = (REPO_ROOT / "src/ofc_regular/hu_m31_t3_step6a_spot.py").read_text(
        encoding="utf-8"
    )
    assert "MAX_LAUNCH_BATCH = 2" in source
    assert 'DEFAULT_ZONES = ("asia-northeast1-b", "asia-northeast1-c")' in source
    assert '"--provisioning-model=SPOT"' in source
    assert '"--instance-termination-action=DELETE"' in source
    assert '"production_fanout_authorized": False' in source
    assert '"training_eligible": False' in source
    assert "results/shard-{padded}" in source
    assert "receive destination is immutable" in source
    assert '"--if-generation-match=0"' in inherited


def test_step6c_received_shard_accepts_only_exact_scientific_allowlist(
    tmp_path: Path,
) -> None:
    shard_dir, manifest = _scientific_shard(tmp_path)

    result = spot._validate_received_shard(
        shard_dir=shard_dir,
        shard=0,
        manifest=manifest,
        auth_sha256="2" * 64,
        manifest_sha256="1" * 64,
    )

    assert result["task_count"] == 25
    assert result["root_task_count"] == 25
    assert "heartbeat_sha256" not in result
    assert set(spot._scientific_result_relatives(0)) == {
        "parity.json",
        "summary.json",
        *(f"roots/hand_{index:03d}.json" for index in range(25)),
        *(f"tasks/hand_{index:03d}.json" for index in range(25)),
    }


def test_step6c_received_shard_accepts_consistent_no_go_summary(
    tmp_path: Path,
) -> None:
    shard_dir, manifest = _scientific_shard(tmp_path)
    summary_path = shard_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["gates"]["peak_rss_within_1_gib"] = False
    summary["all_gates_passed"] = False
    summary["status"] = "no_go"
    summary_path.write_bytes(spot.canonical_bytes(summary))
    done_path = shard_dir / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["quality_status"] = "no_go"
    done_path.write_bytes(spot.canonical_bytes(done))
    _refresh_done_file(shard_dir, "summary.json")

    result = spot._validate_received_shard(
        shard_dir=shard_dir,
        shard=0,
        manifest=manifest,
        auth_sha256="2" * 64,
        manifest_sha256="1" * 64,
    )

    assert result["shard_status"] == "no_go"


@pytest.mark.parametrize("mutation", ["status_mismatch", "extra_gate"])
def test_step6c_received_shard_rejects_inconsistent_summary_gate_status(
    tmp_path: Path, mutation: str
) -> None:
    shard_dir, manifest = _scientific_shard(tmp_path)
    summary_path = shard_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if mutation == "status_mismatch":
        summary["gates"]["peak_rss_within_1_gib"] = False
        summary["all_gates_passed"] = False
    else:
        summary["gates"]["unexpected_gate"] = False
        summary["all_gates_passed"] = False
        summary["status"] = "no_go"
    summary_path.write_bytes(spot.canonical_bytes(summary))
    done_path = shard_dir / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["quality_status"] = summary["status"]
    done_path.write_bytes(spot.canonical_bytes(done))
    _refresh_done_file(shard_dir, "summary.json")

    with pytest.raises(ValueError, match="received boundary failed"):
        spot._validate_received_shard(
            shard_dir=shard_dir,
            shard=0,
            manifest=manifest,
            auth_sha256="2" * 64,
            manifest_sha256="1" * 64,
        )


def test_step6c_received_shard_rejects_done_quality_status_mismatch(
    tmp_path: Path,
) -> None:
    shard_dir, manifest = _scientific_shard(tmp_path)
    done_path = shard_dir / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["quality_status"] = "no_go"
    done_path.write_bytes(spot.canonical_bytes(done))

    with pytest.raises(ValueError, match="received boundary failed"):
        spot._validate_received_shard(
            shard_dir=shard_dir,
            shard=0,
            manifest=manifest,
            auth_sha256="2" * 64,
            manifest_sha256="1" * 64,
        )


def test_step6c_received_shard_rejects_extra_done_key(tmp_path: Path) -> None:
    shard_dir, manifest = _scientific_shard(tmp_path)
    done_path = shard_dir / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["unexpected"] = False
    done_path.write_bytes(spot.canonical_bytes(done))

    with pytest.raises(ValueError, match="received boundary failed"):
        spot._validate_received_shard(
            shard_dir=shard_dir,
            shard=0,
            manifest=manifest,
            auth_sha256="2" * 64,
            manifest_sha256="1" * 64,
        )


def test_step6c_received_shard_rejects_semantically_invalid_parity(
    tmp_path: Path,
) -> None:
    shard_dir, manifest = _scientific_shard(tmp_path)
    parity_path = shard_dir / "parity.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["status"] = "no_go"
    parity_path.write_bytes(spot.canonical_bytes(parity))
    summary_path = shard_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["parity_report_sha256"] = spot.sha256_file(parity_path)
    summary_path.write_bytes(spot.canonical_bytes(summary))
    _refresh_done_file(shard_dir, "parity.json")
    _refresh_done_file(shard_dir, "summary.json")

    with pytest.raises(ValueError, match="Linux parity provenance changed"):
        spot._validate_received_shard(
            shard_dir=shard_dir,
            shard=0,
            manifest=manifest,
            auth_sha256="2" * 64,
            manifest_sha256="1" * 64,
        )


def test_step6c_received_shard_rejects_summary_parity_hash_mismatch(
    tmp_path: Path,
) -> None:
    shard_dir, manifest = _scientific_shard(tmp_path)
    summary_path = shard_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["parity_report_sha256"] = "0" * 64
    summary_path.write_bytes(spot.canonical_bytes(summary))
    _refresh_done_file(shard_dir, "summary.json")

    with pytest.raises(ValueError, match="received boundary failed"):
        spot._validate_received_shard(
            shard_dir=shard_dir,
            shard=0,
            manifest=manifest,
            auth_sha256="2" * 64,
            manifest_sha256="1" * 64,
        )


@pytest.mark.parametrize(
    "relative",
    ["heartbeat.json", "run.log", "time.txt", "runner_stdout.json"],
)
def test_step6c_received_shard_rejects_mutable_progress_files(
    tmp_path: Path, relative: str
) -> None:
    shard_dir, manifest = _scientific_shard(tmp_path)
    (shard_dir / relative).write_text("mutable\n", encoding="utf-8")

    with pytest.raises(ValueError, match="received file set changed"):
        spot._validate_received_shard(
            shard_dir=shard_dir,
            shard=0,
            manifest=manifest,
            auth_sha256="2" * 64,
            manifest_sha256="1" * 64,
        )


def test_step6c_done_rejects_extra_file_even_when_hash_is_valid(
    tmp_path: Path,
) -> None:
    shard_dir, manifest = _scientific_shard(tmp_path)
    heartbeat = shard_dir / "heartbeat.json"
    _write(heartbeat, {"status": "pass"})
    done = json.loads((shard_dir / "DONE.json").read_text(encoding="utf-8"))
    done["files"]["heartbeat.json"] = {
        "sha256": spot.sha256_file(heartbeat),
        "bytes": heartbeat.stat().st_size,
    }
    (shard_dir / "DONE.json").write_bytes(spot.canonical_bytes(done))

    with pytest.raises(ValueError, match="scientific file set changed"):
        spot._validate_received_shard(
            shard_dir=shard_dir,
            shard=0,
            manifest=manifest,
            auth_sha256="2" * 64,
            manifest_sha256="1" * 64,
        )


def test_step6c_receive_preserves_raw_package_and_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "run"
    source.mkdir()
    manifest = _manifest()
    _write(source / "manifest.json", manifest)
    _write(source / "launch_authorization.json", {"authorized": True})
    manifest_sha = spot.sha256_file(source / "manifest.json")
    auth_sha = spot.sha256_file(source / "launch_authorization.json")
    monkeypatch.setattr(spot, "validate_launch", lambda _path: (manifest, {}))

    def fake_run(args, **_kwargs):
        shard_dir = Path(args[5])
        shard = int(shard_dir.name.removeprefix("shard-"))
        _scientific_shard(
            shard_dir.parent,
            shard=shard,
            manifest_sha256=manifest_sha,
            authorization_sha256=auth_sha,
        )
        return None

    monkeypatch.setattr(spot, "_run", fake_run)
    destination = tmp_path / "received"

    spot.receive_shards(run_dir=source, output_dir=destination)

    assert (destination / "package_manifest.json").read_bytes() == (
        source / "manifest.json"
    ).read_bytes()
    assert (destination / "launch_authorization.json").read_bytes() == (
        source / "launch_authorization.json"
    ).read_bytes()


def test_step6c_accepted_binary_inputs_are_inside_immutable_step6b_package() -> None:
    engine = (
        REPO_ROOT
        / "outputs/gcp_runs/regular-hu-m31-step6b-canary-20260717-001"
        / "package_src/native/release/libofc_hu_m3_engine.so"
    )
    feature = (
        REPO_ROOT
        / "outputs/gcp_runs/regular-hu-m31-step6b-canary-20260717-001"
        / "package_src/target/release/libofc_stage3_feature_encoder.so"
    )
    assert spot.sha256_file(engine) == ACCEPTED_NATIVE_LIBRARY_SHA256
    assert spot.sha256_file(feature) == ACCEPTED_FEATURE_ENCODER_SHA256
