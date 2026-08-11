from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
START = REPO_ROOT / "scripts" / "Start-GcpHuM42ModelRun.ps1"
STATUS = REPO_ROOT / "scripts" / "Get-GcpHuM42ModelRunStatus.ps1"
RECEIVE = REPO_ROOT / "scripts" / "Receive-GcpHuM42ModelRun.ps1"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def test_model_start_locks_verified_gate40_inputs_and_bounded_ranker_config():
    text = _read(START)

    for token in (
        'TeacherRunName = "regular-hu-m42-c2e8-gate40-20260713-1335"',
        'ExpectedTeacherSourceSha256 = "902c200873927fb31ffd1204588d4f0492880ab7ff29fb579125742c54e12530"',
        'ExpectedTrainerSha256 = "bda861cd0f72242e904d7bddbb81ca2bbb633f17d1437e4c7b6e56df8625ac5f"',
        "negative_regret_ranker_v2",
        "--cross-fit-folds",
        "--iterations",
        "--minimum-safety-fit-samples 5",
        "--minimum-threshold-lock-samples 5",
        "--minimum-calibration-fires 2",
        "--maximum-false-positive-rate 0.30",
        "--maximum-p95-loss 25",
        "--maximum-p99-loss 40",
        'train = 20; calibration = 10; locked_holdout = 10',
    ):
        assert token in text
    assert 'MachineType = "c4-highcpu-16"' in text
    assert '"--provisioning-model", "SPOT"' in text
    assert '"--instance-termination-action", "DELETE"' in text
    assert "current_profile_mutated = $false" in text
    assert "no_runtime_activation = $true" in text


def test_worker_is_done_last_hash_chained_and_full_job_retry_safe():
    text = _read(START)

    for token in (
        "deterministic_full_job_retry = $true",
        "fold_checkpoint_resume = $false",
        "hu_m42_model_heartbeat_v1",
        "hu_m42_model_validation_v1",
        "hu_m42_model_done_v1",
        "all_outer_validation_identities_excluded",
        "paired_delta_schema_v2_samples",
        "locked_holdout_used_for_threshold_or_training",
        "teacher_value_runtime_gate",
        "python -m pip install 'numpy==2.2.6' 'scikit-learn==1.8.0'",
    ):
        assert token in text
    assert "torch==" not in text
    done_upload = 'gcs_upload_immutable "$OUT/DONE" "$PREFIX/results/DONE"'
    assert text.count(done_upload) == 1
    assert text.index(done_upload) > text.index(
        'gcs_upload "$OUT/validation.json" "$PREFIX/results/validation.json"'
    )
    assert text.index(done_upload) > text.index(
        'gcs_upload "$OUT/status.json" "$PREFIX/results/status.json"'
    )


def test_start_has_explicit_frozen_resume_and_never_overwrites_the_manifest():
    text = _read(START)

    for token in (
        "[switch]$ResumeExisting",
        "ResumeExisting requires CreateInstance",
        "Local frozen model run files already exist; refusing to overwrite them",
        "Local model manifest differs from the frozen remote manifest",
        "Frozen M4.2 model run already has DONE and must not be relaunched",
        '"--if-generation-match=0"',
        "resume_existing = [bool]$ResumeExisting",
    ):
        assert token in text
    assert text.index("$remoteManifestExists = Test-GcsObject $manifestUri") < text.index(
        "Write-Utf8NoBom -Path $manifestPath"
    )
    assert "Use -ResumeExisting -CreateInstance for an incomplete frozen run" in text


def test_status_requires_done_commit_and_tracks_the_frozen_vm_name():
    text = _read(STATUS)

    assert 'elseif ($null -ne $status -and [string]$status.state -eq "complete")' in text
    assert '"finalizing_without_done"' in text
    assert "done_commit_present = ($null -ne $done)" in text
    assert '("name={0}" -f $vmName)' in text
    assert "$manifest.compute.vm_name" in text


def test_receiver_is_constrained_staged_and_never_overwrites_an_output_dir():
    text = _read(RECEIVE)

    for token in (
        "OutputDir must be a child of the frozen M4.2 model_runs directory",
        "refusing a possible profile/current overwrite",
        '".receiving-{0}-{1}"',
        "OutputDir already exists but is not a complete verified receipt; refusing to overwrite it",
        "OutputDir appeared during receive; refusing to overwrite it",
        "Move-Item -LiteralPath $stageDir -Destination $OutputDir",
        "existingReceipt.model_sha256",
        "existingReceipt.training_manifest_sha256",
    ):
        assert token in text
    assert 'Join-Path $stageDir $objects[$name]' in text
    assert 'Join-Path $OutputDir $objects[$name]' not in text


def test_status_and_receive_fail_closed_on_provenance_or_leakage_mismatch():
    status = _read(STATUS)
    receive = _read(RECEIVE)

    for token in (
        "Invalid or stale M4.2 model DONE object",
        "run_manifest_sha256",
        "no_runtime_activation",
        "matched no objects",
    ):
        assert token in status
    for token in (
        "Downloaded M4.2 model output hash chain is invalid",
        "M4.2 worker validation is missing or invalid",
        "Downloaded M4.2 training manifest violates the frozen no-leak contract",
        "identity_leakage_count",
        "locked_holdout_used_for_threshold_or_training",
        "teacher_value_runtime_gate",
        "teacher_metrics_are_realized_match_ev = $false",
        "locked_holdout_used_for_threshold_search = $false",
        "current_profile_mutated = $false",
    ):
        assert token in receive


@pytest.mark.skipif(_powershell() is None, reason="PowerShell is unavailable")
def test_all_m42_model_scripts_parse_in_powershell():
    shell = _powershell()
    assert shell is not None
    paths = ",".join(f"'{path}'" for path in (START, STATUS, RECEIVE))
    command = (
        f"$bad=@(); foreach($f in @({paths})){{"
        "$t=$null;$e=$null;"
        "[System.Management.Automation.Language.Parser]::ParseFile($f,[ref]$t,[ref]$e)|Out-Null;"
        "if($e.Count){$bad += $e}};"
        "if($bad.Count){$bad|ForEach-Object{$_.Message};exit 1}"
    )
    completed = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
