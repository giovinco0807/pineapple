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
COMMON = ROOT / "scripts" / "HuM43Attempt03ModelSpot.Common.ps1"
START = ROOT / "scripts" / "Start-GcpHuM43Attempt03ModelRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43Attempt03ModelRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM43Attempt03ModelRun.ps1"
RUNNER = ROOT / "scripts" / "Run-HuM43Attempt03ModelJobShard.ps1"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _here(text: str, variable: str) -> str:
    match = re.search(
        rf"\${re.escape(variable)}\s*=\s*@'\n(.*?)\n'@",
        text.replace("\r\n", "\n"),
        flags=re.DOTALL,
    )
    assert match is not None, variable
    return match.group(1)


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_attempt03_model_spot_scripts_parse() -> None:
    shell = _powershell()
    assert shell is not None
    quoted = ",".join(f"'{str(path).replace(chr(39), chr(39) * 2)}'" for path in (
        COMMON, START, STATUS, RECEIVE, RUNNER
    ))
    script = (
        f"$files=@({quoted});"
        "foreach($f in $files){$t=$null;$e=$null;"
        "[void][Management.Automation.Language.Parser]::ParseFile($f,[ref]$t,[ref]$e);"
        "if($e.Count){throw (($e|ForEach-Object{$_.ToString()})-join [Environment]::NewLine)}}"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_start_freezes_fit700_canary_fanout_and_resumable_30_jobs() -> None:
    text = _read(START)
    for token in (
        "hu_m43_attempt03_v5_model_spot_run_manifest_v1",
        "ofc_regular.prepare_hu_m43_attempt03_fold_training",
        '"--training-freeze", $resolvedTrainingFreeze',
        '"--inherited-train", $resolvedInherited',
        '"--fresh-train-fit", $resolvedFresh',
        "fit700_only_no_holdouts",
        "$ExpectedJobs = 30",
        "$ExpectedShards = 8",
        "$JobsPerShard = 4",
        "Canary shard 0 must complete before fanout",
        "Canary shard receipt must complete before fanout",
        "JOB_INDICES=",
        "SHARD_EXPECTED_JOBS=",
        "ifGenerationMatch=0",
        "ResumeExisting",
        "Invoke-M43A3GcloudProcess $arguments 30",
        "Create outcome remained ambiguous",
        "current_profile_mutated = $false",
        "runtime_policy_activated = $false",
        "full_replacement = $false",
        "post_copy_source_audit",
        "packaged worker source SHA changed",
    ):
        assert token in text
    assert "--precal" not in text
    assert "--sealed-calibration" not in text
    assert "--locked-holdout" not in text
    assert "ai_profiles.py" not in text
    dryrun_exit = text.index("if ($DryRun) {", text.index("$plan = [ordered]"))
    remote_probe = text.index("Test-M43A3GcsObject $manifestUri", dryrun_exit)
    assert dryrun_exit < remote_probe


def test_start_package_only_dry_run_has_no_fit_or_cloud_requirement() -> None:
    text = _read(START)
    assert "[switch]$PackageOnly" in text
    assert "PackageOnly requires DryRun" in text
    assert 'if ($PackageOnly) { $null } else { Resolve-M43A3Input $InheritedTrain' in text
    branch = text[text.index("if ($PackageOnly) {") : text.index("function Convert-ToVmPrefix")]
    for token in (
        "ofc_regular.build_hu_m43_attempt03_model_spot_package",
        "hu_m43_attempt03_v5_model_spot_package_only_plan_v1",
        'status = "pass_no_fit_data_no_cloud"',
        "fit_input_count = 0",
        "holdout_input_count = 0",
        "cloud_upload_performed = $false",
        "instance_launch_performed = $false",
        "exit 0",
    ):
        assert token in branch
    assert "Invoke-M43A3Gcloud" not in branch
    closure_audit = text.index(
        "$closureOutput = Invoke-LocalPython @($closureScript, $repoRoot, $packageRoot, $preparedContract)"
    )
    zip_builder = text.index("$zipBuilder = Join-Path $temporary")
    assert closure_audit < zip_builder


def test_model_source_closure_rejects_post_preflight_source_tamper(
    tmp_path: Path,
) -> None:
    closure = _here(_read(START), "closureBuilder")
    repo = tmp_path / "repo"
    package = repo / "src" / "ofc_regular"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    worker = package / "train_hu_m43_attempt03_fold_job.py"
    worker.write_text(
        "from . import hu_m43_joint_model_v5\n", encoding="utf-8"
    )
    v5 = package / "hu_m43_joint_model_v5.py"
    v5.write_bytes(b"frozen-v5")
    relative_worker = "src/ofc_regular/train_hu_m43_attempt03_fold_job.py"
    relative_v5 = "src/ofc_regular/hu_m43_joint_model_v5.py"
    worker_sources = {
        relative_worker: hashlib.sha256(worker.read_bytes()).hexdigest(),
        relative_v5: hashlib.sha256(v5.read_bytes()).hexdigest(),
    }
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "training_freeze": {"worker_sources": worker_sources},
                "model_freeze": {
                    "v5_source_sha256": worker_sources[relative_v5]
                },
            }
        ),
        encoding="utf-8",
    )

    valid = subprocess.run(
        [sys.executable, "-", str(repo), str(tmp_path / "valid"), str(contract)],
        input=closure,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert valid.returncode == 0, valid.stdout + valid.stderr
    audit = json.loads(valid.stdout)
    assert audit["post_copy_source_audit"] == "pass"
    assert audit["verified_worker_sources"] == worker_sources

    v5.write_bytes(b"tampered-after-preflight")
    tampered = subprocess.run(
        [
            sys.executable,
            "-",
            str(repo),
            str(tmp_path / "tampered"),
            str(contract),
        ],
        input=closure,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert tampered.returncode != 0
    assert "packaged worker source SHA changed" in tampered.stderr


def test_embedded_startup_is_valid_bash_and_python() -> None:
    startup = _here(_read(START), "startup")
    for index, block in enumerate(re.findall(r"<<'PY'\n(.*?)\nPY", startup, re.DOTALL)):
        compile(block, f"attempt03-model-startup-{index}", "exec")
    assert len(re.findall(r"<<'PY'", startup)) >= 6
    bash = shutil.which("bash")
    if bash is not None:
        result = subprocess.run(
            [bash, "-n"],
            input=(startup + "\n").encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
    closure = _here(_read(START), "closureBuilder")
    compile(closure, "attempt03-model-source-closure", "exec")


def test_startup_publishes_done_last_and_strict_shard_receipt() -> None:
    startup = _here(_read(START), "startup")
    assert startup.index('gcs_upload "$out/estimator.pkl"') < startup.index(
        'gcs_upload_immutable "$out/DONE.json"'
    )
    for token in (
        "hu_m43_attempt03_v5_fold_job_manifest_v1",
        "hu_m43_attempt03_v5_fold_done_v1",
        "hu_m43_attempt03_v5_model_spot_shard_receipt_v1",
        'gcs_upload_immutable "$WORK/shard_receipt.json"',
        "fit700_only",
        "holdout_input_count",
        "scipy==1.16.3",
        "lightgbm==4.6.0",
        "timeout 600",
    ):
        assert token in startup


def test_status_fails_closed_on_done_receipt_and_inactive_partial() -> None:
    text = _read(STATUS)
    for token in (
        "Assert-M43A3RunManifest",
        "Assert-M43A3Done",
        "$seenDone",
        "Duplicate DONE job identity",
        "hu_m43_attempt03_v5_model_spot_shard_receipt_v1",
        "shard receipt exists before all strict DONE artifacts",
        "incomplete_inactive",
        "strict_done_and_receipt_validation = \"pass\"",
        "completed_job_count",
        "canary_complete",
        "fanout_allowed",
    ):
        assert token in text
    assert "gcloud storage rsync" not in text


def test_receive_downloads_exact_chain_then_assembles_once_atomically() -> None:
    text = _read(RECEIVE)
    for token in (
        "OutputDir must remain under m43_attempt03_model_runs",
        "for ($index = 0; $index -lt 30; $index++)",
        'foreach ($name in @("estimator.pkl", "job_manifest.json", "DONE.json"))',
        "Downloaded Attempt03 job hash chain is invalid",
        "for ($shard = 0; $shard -lt 8; $shard++)",
        "hu_m43_attempt03_v5_model_spot_shard_receipt_v1",
        '"ofc_regular.assemble_hu_m43_attempt03_model", "assemble-fit"',
        '"--training-freeze", $resolvedTrainingFreeze',
        "hu_m43_attempt03_v5_model_receive_receipt_v1",
        "fit_candidate_precalibration_unopened",
        "Move-Item -LiteralPath $stageDir -Destination $destination",
        "precalibration_opened = $false",
        "sealed_calibration_opened = $false",
        "inherited_locked_opened = $false",
    ):
        assert token in text
    assert text.count('"ofc_regular.assemble_hu_m43_attempt03_model", "assemble-fit"') == 1
    assert text.index("if ($seen.Count -ne 30)") < text.index("$assemblerArgs = @(")
    assert "--sealed-calibration" not in text
    assert "--locked-holdout" not in text


def test_common_enforces_bounded_gcloud_processes_and_strict_hashes() -> None:
    text = _read(COMMON)
    for token in (
        "Invoke-M43A3ProcessBounded",
        "Stop-M43A3ProcessTree",
        "process timed out after",
        "Invoke-M43A3GcloudProcess",
        "Attempt03 remote DONE hash chain is invalid",
        "training_freeze_file_sha256",
        "fit700_only_no_holdouts",
    ):
        assert token in text
    assert "& gcloud" not in text
