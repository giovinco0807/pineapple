from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "Start-GcpHuM43Attempt02PopulationRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43Attempt02PopulationRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM43Attempt02PopulationRun.ps1"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


@pytest.mark.parametrize("path", (START, STATUS, RECEIVE))
def test_attempt02_population_scripts_parse(path: Path) -> None:
    shell = _powershell()
    if shell is None:
        pytest.skip("PowerShell is unavailable")
    command = (
        "$tokens=$null;$errors=$null;"
        f"[System.Management.Automation.Language.Parser]::ParseFile('{path}',"
        "[ref]$tokens,[ref]$errors)|Out-Null;"
        "if($errors.Count){$errors|%{$_.Message};exit 1}"
    )
    completed = subprocess.run(
        [shell, "-NoProfile", "-Command", command],
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_dry_run_cannot_create_instances() -> None:
    shell = _powershell()
    if shell is None:
        pytest.skip("PowerShell is unavailable")
    completed = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(START),
            "-ModelPath",
            "unused",
            "-TrainingManifestPath",
            "unused",
            "-DataContractPath",
            "unused",
            "-FreezeManifestPath",
            "unused",
            "-LockedReceiptPath",
            "unused",
            "-ConsumptionMarkerPath",
            "unused",
            "-DryRun",
            "-CreateInstances",
        ],
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "DryRun never creates cloud instances" in completed.stdout + completed.stderr


def test_start_packages_hash_receipts_only_and_runs_bound_v4() -> None:
    text = _text(START)
    for token in (
        "ofc_regular.validate_hu_m43_attempt02_acceptance",
        "canonical_global_marker_verified",
        "teacher_calibration_locked_content_packaged",
        "Teacher/calibration/locked JSONL must not enter",
        "Repository configs/current-profile metadata must not enter",
        "Rust source/debug artifacts must not enter",
        "current_profile_artifact_packaged = $false",
        "--expected-model-sha256",
        "--freeze-manifest artifacts/freeze_manifest.json",
        "--training-manifest artifacts/training_manifest.json",
        "all(row.get('runtime_binding_verified') is True for row in rows)",
        "centered_fold_mean_delta_then_safety_gate_v4",
        "hu_m43_t1_joint_model_v4",
        '"--provisioning-model", "SPOT"',
        'unit = "completed_shard"',
        "resume_missing_shards_only = $true",
        "done_commit_last = $true",
        "fanout requires a completed shard-0000 canary",
        "current_profile_mutated = $false",
        "no_runtime_activation = $true",
    ):
        assert token in text
    assert "locked_holdout.jsonl" not in text
    assert "train.jsonl" not in text
    assert "calibration.jsonl" not in text
    assert '"src", "configs"' not in text
    assert 'Copy-Item -LiteralPath (Join-Path $repoRoot "rust/' not in text
    done_upload = 'gcloud storage cp "$OUT/DONE" "$RESULT/DONE" --if-generation-match=0'
    assert text.count(done_upload) == 1
    assert text.index(done_upload) > text.index(
        'gcloud storage cp "$OUT/records.jsonl" "$RESULT/records.jsonl"'
    )
    assert text.index(done_upload) > text.index("write_status complete 0")


def test_embedded_startup_script_has_valid_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    match = re.search(
        r"\$startup = @'\r?\n(?P<script>.*?)\r?\n'@",
        _text(START),
        flags=re.DOTALL,
    )
    assert match is not None
    completed = subprocess.run(
        [bash, "-n"],
        input=(match.group("script") + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")


def test_status_and_receive_fail_closed_on_attempt02_schemas() -> None:
    status = _text(STATUS)
    receive = _text(RECEIVE)
    for token in (
        "hu_m43_attempt02_population_spot_manifest_v1",
        "hu_m43_attempt02_population_spot_done_v1",
        "seen.Add($uriShard)",
        "$rowShard -ne $uriShard",
        "missing_indices",
        "active_instances",
        "Try-GetGcsJson",
        "Invalid or stale Attempt02 population status object",
    ):
        assert token in status
    for token in (
        "outputs/hu_joint_policy/m43_attempt02_population",
        "refusing profile/current overwrite",
        "hu_m43_attempt02_population_spot_done_v1",
        "ofc_regular.merge_hu_m4_population_shards",
        "complete_content_verified",
        "runtime_binding_verified",
        "centered_fold_mean_delta_then_safety_gate_v4",
        "teacher_calibration_locked_content_received = $false",
        "Move-Item -LiteralPath $stage -Destination $OutputDir",
    ):
        assert token in receive
