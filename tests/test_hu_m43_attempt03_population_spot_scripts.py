from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "Start-GcpHuM43Attempt03PopulationRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43Attempt03PopulationRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM43Attempt03PopulationRun.ps1"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _here(text: str, variable: str) -> str:
    match = re.search(
        rf"\${re.escape(variable)}\s*=\s*@'\n(.*?)\n'@",
        text.replace("\r\n", "\n"),
        flags=re.DOTALL,
    )
    assert match is not None, variable
    return match.group(1)


@pytest.mark.parametrize("path", (START, STATUS, RECEIVE))
def test_attempt03_population_scripts_parse(path: Path) -> None:
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


def test_dry_run_cannot_create_attempt03_instances() -> None:
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
            "-FinalTrainingManifestPath",
            "unused",
            "-RuntimeFreezePath",
            "unused",
            "-TrainingFreezePath",
            "unused",
            "-PrecalibrationReceiptPath",
            "unused",
            "-ModelFreezePath",
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


def test_start_packages_runtime_only_and_runs_bound_v5() -> None:
    text = _text(START)
    for token in (
        "ofc_regular.validate_hu_m43_attempt03_population",
        '"--training-freeze", $resolved.training_freeze',
        "canonical_global_marker_verified",
        "teacher_calibration_locked_content_packaged",
        "teacher_valued_receipts_packaged = $false",
        "runtime_artifacts_only = $true",
        "Teacher/calibration/locked JSONL must not enter",
        "Repository configs/current-profile metadata must not enter",
        "runtime artifacts only; teacher-valued receipts and holdouts are forbidden",
        "hu_m43_attempt03_population_post_copy_source_audit_v1",
        "packaged v5 implementation SHA changed",
        "--expected-model-sha256",
        "--freeze-manifest artifacts/runtime_freeze.json",
        "--training-manifest artifacts/final_training_manifest.json",
        "all(row.get('runtime_binding_verified') is True for row in rows)",
        "eligible_stage18_stacked_meta_ranker_v5",
        "hu_m43_t1_joint_model_v5",
        "lightgbm==4.6.0",
        "--opponents stage19_p0 stage9f_p2 stage7_m5_r10 random_exact_final",
        '"--provisioning-model", "SPOT"',
        'unit = "completed_shard"',
        "resume_missing_shards_only = $true",
        "done_commit_last = $true",
        "fanout requires a completed shard-0000 canary",
        "current_profile_mutated = $false",
        "no_runtime_activation = $true",
    ):
        assert token in text
    for forbidden_copy in (
        "Copy-Item -LiteralPath $resolved.training_freeze",
        "Copy-Item -LiteralPath $resolved.precalibration_receipt",
        "Copy-Item -LiteralPath $resolved.locked_receipt",
        "Copy-Item -LiteralPath $resolved.consumption_marker",
        "Copy-Item -LiteralPath $resolved.model_freeze",
        "Copy-Item -LiteralPath $resolved.attempt03_plan",
    ):
        assert forbidden_copy not in text
    assert '"src", "configs"' not in text
    assert 'Copy-Item -LiteralPath (Join-Path $repoRoot "rust/' not in text
    done_upload = 'gcloud storage cp "$OUT/DONE" "$RESULT/DONE" --if-generation-match=0'
    assert text.count(done_upload) == 1
    assert text.index(done_upload) > text.index(
        'gcloud storage cp "$OUT/records.jsonl" "$RESULT/records.jsonl"'
    )
    assert text.index(done_upload) > text.index("write_status complete 0")
    copy_source = text.index('foreach ($item in @("pyproject.toml", "src"))')
    post_copy_verify = text.index("$postCopyV5Verifier | & python")
    build_zip = text.index("New-ZipWithForwardSlashes $packageDir $sourcePath")
    assert copy_source < post_copy_verify < build_zip


def test_population_post_copy_v5_verifier_rejects_tamper(
    tmp_path: Path,
) -> None:
    verifier = _here(_text(START), "postCopyV5Verifier")
    package = tmp_path / "package"
    source = package / "src" / "ofc_regular" / "hu_m43_joint_model_v5.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen-v5")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    freeze = tmp_path / "runtime-freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "executable_model_freeze": {
                    "v5_implementation_sha256": expected
                }
            }
        ),
        encoding="utf-8",
    )
    valid = subprocess.run(
        [sys.executable, "-", str(package), str(freeze)],
        input=verifier,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert valid.returncode == 0, valid.stdout + valid.stderr
    source.write_bytes(b"tampered-after-preflight")
    tampered = subprocess.run(
        [sys.executable, "-", str(package), str(freeze)],
        input=verifier,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert tampered.returncode != 0
    assert "packaged v5 implementation SHA changed" in tampered.stderr


def test_embedded_attempt03_startup_has_valid_bash_syntax() -> None:
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


def test_status_and_receive_bind_attempt03_transport_and_final_validator() -> None:
    status = _text(STATUS)
    receive = _text(RECEIVE)
    for token in (
        "hu_m43_attempt03_population_spot_manifest_v1",
        "hu_m43_attempt03_population_spot_done_v1",
        "hu_m43_attempt03_population_spot_status_v1",
        "seen.Add($uriShard)",
        "$rowShard -ne $uriShard",
        "missing_indices",
        "active_instances",
        "Try-GetGcsJson",
        "Invalid or stale Attempt03 population status object",
    ):
        assert token in status
    for token in (
        "outputs/hu_joint_policy/m43_attempt03_population",
        "refusing profile/current overwrite",
        "hu_m43_attempt03_population_spot_done_v1",
        "hu_m43_attempt03_population_spot_receipt_v1",
        "ofc_regular.merge_hu_m4_population_shards",
        "complete_content_verified",
        "runtime_binding_verified",
        "eligible_stage18_stacked_meta_ranker_v5",
        "teacher_calibration_locked_content_received = $false",
        "ofc_regular.validate_hu_m43_attempt03_population",
        '"--spot-receipt", $receiptPath',
        '"--run-manifest", $runManifestCopy',
        '"--lifecycle-preflight", $preflightPath',
        "$acceptCode -notin @(0, 2)",
        '"complete_go"',
        '"complete_no_go"',
        "Move-Item -LiteralPath $stage -Destination $OutputDir",
    ):
        assert token in receive
