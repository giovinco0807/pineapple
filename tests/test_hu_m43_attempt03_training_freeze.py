from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ofc_regular.build_hu_m43_attempt03_model_spot_package import (
    build_attempt03_model_spot_package,
)
from ofc_regular.hu_m43_attempt03_training import (
    M43_ATTEMPT03_TRAINING_CONFIG_SCHEMA,
    M43_ATTEMPT03_TRAINING_FREEZE_SCHEMA,
    M43_ATTEMPT03_TRAINING_SOURCE_PATHS,
    M43_ATTEMPT03_WORKER_SOURCE_PATHS,
    Attempt03TrainingConfig,
    load_attempt03_training_freeze,
)
from ofc_regular.hu_m43_pilot_contract import canonical_manifest_sha256


ROOT = Path(__file__).resolve().parents[1]
STATIC_FREEZE = (
    ROOT / "configs" / "hu_joint_policy_m43_attempt03_training_freeze.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_payload() -> dict:
    files = {
        relative: _sha(ROOT / relative)
        for relative in M43_ATTEMPT03_TRAINING_SOURCE_PATHS
    }
    workers = {
        relative: files[relative]
        for relative in M43_ATTEMPT03_WORKER_SOURCE_PATHS
    }
    payload = {
        "schema": M43_ATTEMPT03_TRAINING_FREEZE_SCHEMA,
        "milestone": "M4.3-attempt03",
        "status": (
            "frozen_after_one_structural_train_fit_canary_before_"
            "row_valued_design_or_any_holdout_open"
        ),
        "frozen_at": "2026-07-14T00:00:00+09:00",
        "decision_boundary": {
            "cloud_teacher_rows_may_exist": True,
            "attempt03_fit_rows_received_locally": 1,
            "attempt03_fit_rows_structurally_inspected": 1,
            "attempt03_fit_jsonl_content_parse_count": 1,
            "attempt03_fit_row_valued_labels_or_metrics_used_for_design": 0,
            "precalibration_rows_opened": 0,
            "sealed_calibration_rows_opened": 0,
            "inherited_locked_rows_opened": 0,
            "source_or_config_selected_from_row_values": False,
        },
        "parent_model_freeze": {
            "path": "configs/hu_joint_policy_m43_attempt03_model_freeze.json",
            "file_sha256": _sha(
                ROOT
                / "configs"
                / "hu_joint_policy_m43_attempt03_model_freeze.json"
            ),
        },
        "training_config": {
            "schema": M43_ATTEMPT03_TRAINING_CONFIG_SCHEMA,
            "canonical_sha256": canonical_manifest_sha256(
                Attempt03TrainingConfig().to_manifest()
            ),
        },
        "executable_sources": {
            "files": files,
            "files_sha256": canonical_manifest_sha256(files),
            "worker_paths": list(M43_ATTEMPT03_WORKER_SOURCE_PATHS),
            "worker_files_sha256": canonical_manifest_sha256(workers),
        },
        "lifecycle": {
            "teacher_row_values_in_freeze": False,
            "holdout_path_in_freeze": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
            "full_replacement": False,
        },
    }
    payload["freeze_sha256"] = canonical_manifest_sha256(payload)
    return payload


def test_dynamic_freeze_and_package_only_need_no_fit_data(
    tmp_path: Path,
) -> None:
    payload = _freeze_payload()
    freeze_path = tmp_path / "training_freeze.json"
    freeze_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_attempt03_training_freeze(freeze_path, repo_root=ROOT)
    assert loaded["freeze_sha256"] == payload["freeze_sha256"]
    first = build_attempt03_model_spot_package(
        repo_root=ROOT,
        training_freeze_path=freeze_path,
        output_dir=tmp_path / "package-1",
    )
    second = build_attempt03_model_spot_package(
        repo_root=ROOT,
        training_freeze_path=freeze_path,
        output_dir=tmp_path / "package-2",
    )
    assert first["fit_input_count"] == 0
    assert first["holdout_input_count"] == 0
    assert first["cloud_upload_performed"] is False
    assert first["instance_launch_performed"] is False
    assert first["source_package_policy"]["post_archive_source_audit"] == "pass"
    assert first["source_archive"]["file_sha256"] == second["source_archive"][
        "file_sha256"
    ]


@pytest.mark.skipif(
    shutil.which("pwsh") is None and shutil.which("powershell") is None,
    reason="PowerShell unavailable",
)
def test_start_package_only_dry_run_executes_without_fit_data(
    tmp_path: Path,
) -> None:
    freeze_path = tmp_path / "training_freeze.json"
    freeze_path.write_text(
        json.dumps(_freeze_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shell = shutil.which("pwsh") or shutil.which("powershell")
    assert shell is not None
    run_name = "m43-a3-package-test-" + tmp_path.name.replace("_", "-")
    output_root = ROOT / "outputs" / "gcp_runs" / run_name
    assert not output_root.exists()
    try:
        result = subprocess.run(
            [
                shell,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / "scripts" / "Start-GcpHuM43Attempt03ModelRun.ps1"),
                "-RunName",
                run_name,
                "-ModelFreeze",
                str(
                    ROOT
                    / "configs"
                    / "hu_joint_policy_m43_attempt03_model_freeze.json"
                ),
                "-TrainingFreeze",
                str(freeze_path),
                "-PackageOnly",
                "-DryRun",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8-sig",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=240,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        plan = json.loads(result.stdout)
        assert plan["status"] == "pass_no_fit_data_no_cloud"
        assert plan["fit_input_count"] == 0
        assert plan["cloud_upload_performed"] is False
        assert plan["instance_launch_performed"] is False
        assert (
            output_root / "dryrun" / "package" / "source.zip"
        ).is_file()
    finally:
        resolved = output_root.resolve()
        allowed = (ROOT / "outputs" / "gcp_runs").resolve()
        if resolved.parent == allowed and resolved.exists():
            shutil.rmtree(resolved)


def test_static_freeze_self_hash_and_every_source_hash() -> None:
    payload = load_attempt03_training_freeze(STATIC_FREEZE, repo_root=ROOT)
    unsigned = {key: value for key, value in payload.items() if key != "freeze_sha256"}
    assert payload["freeze_sha256"] == canonical_manifest_sha256(unsigned)
    files = payload["executable_sources"]["files"]
    assert tuple(sorted(files)) == M43_ATTEMPT03_TRAINING_SOURCE_PATHS
    for relative in M43_ATTEMPT03_TRAINING_SOURCE_PATHS:
        assert files[relative] == _sha(ROOT / relative)
    assert files[
        "configs/hu_joint_policy_m43_attempt03_training_science_correction.json"
    ] == _sha(
        ROOT
        / "configs"
        / "hu_joint_policy_m43_attempt03_training_science_correction.json"
    )
