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
START = ROOT / "scripts" / "Start-GcpHuM43ModelRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43ModelRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM43ModelRun.ps1"
RECEIPT_SHA = "9313bdc6ffa33335032ee3d98e279d123d2f3aa70571114e75ee5f38e2594bea"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _extract_here_string(text: str, variable: str) -> str:
    match = re.search(
        rf"\${re.escape(variable)}\s*=\s*@'\r?\n(.*?)\r?\n'@",
        text,
        flags=re.DOTALL,
    )
    assert match is not None, variable
    return match.group(1)


def _canonical(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_start_uses_redacted_projection_and_never_sends_sealed_inputs_to_cloud():
    text = _read(START)
    startup = _extract_here_string(text, "startup")

    for token in (
        "train_hu_m43_fold_job\", \"prepare",
        '"--predeclared-receipt", $predeclaredReceipt',
        '"--output-contract", $preparedContract',
        '--fold-cloud-contract "$WORK/fold_cloud_contract.json"',
        'cloud_input_boundary = "train_and_calibration_only"',
        'if ($manifestText -match \'(?i)locked\')',
        "Source package contains a data/config artifact",
        r"(?i)locked|(^|/)(outputs?|configs?|data)(/|$)|\.jsonl?$",
    ):
        assert token in text
    assert "--m43-data-contract" not in startup
    assert "--m43-plan" not in startup
    assert "--predeclared-receipt" not in startup
    assert "--locked-holdout" not in text
    assert "M43DataContract" not in startup
    assert "M43Plan" not in startup


def test_exact_python_job_grid_and_eight_four_process_groups_are_frozen():
    text = _read(START)

    for token in (
        "$ExpectedJobs = 30",
        "$OuterFolds = 5",
        "$InnerFoldsPerOuter = 5",
        "$JobsPerShard = 4",
        "$ExpectedShards = 8",
        "$expectedOuter = [int][Math]::Floor($i / 6)",
        "$slot = $i % 6",
        '"outer_runtime" } else { "inner_oof_safety"',
        "$projectionJobs = @($cloudProjection.fold_plan.jobs)",
        'job_spec = $jobSpec; job_spec_sha256 = [string]$projectionJob.job_spec_sha256',
        'IFS=\'+\' read -r -a JOB_ARRAY <<< "$JOB_INDICES"',
        'run_job "$job" & pids+=("$!")',
    ):
        assert token in text
    assert "function Get-JobSpec" not in text
    assert '[int[]]$StartShards = @(0, 1, 2, 3, 4, 5, 6, 7)' in text


def test_source_entry_digest_uses_portable_paths_and_ordinal_sorting():
    frozen_pattern = '"(?i)(^|/)(outputs?|configs?|data)(/|$)"'
    for path in (START, STATUS, RECEIVE):
        text = _read(path)
        assert "[Array]::Sort($entries, [StringComparer]::Ordinal)" in text
        assert "non-portable entry separators" in text
        assert "entries_sha256" in text
        assert frozen_pattern in text
        assert '"(?i)(^|/)(outputs?|configs?)(/|$)"' not in text
    assert "Frozen source package entry digest mismatch" in _read(START)
    assert "Frozen source archive violates package policy" in _read(RECEIVE)


def test_every_startup_python_heredoc_compiles():
    startup = _extract_here_string(_read(START), "startup")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", startup, flags=re.DOTALL)
    assert len(blocks) >= 6
    for index, block in enumerate(blocks):
        compile(block, f"startup-heredoc-{index}", "exec")


def test_full_attempt02_configuration_and_abort_receipt_are_immutable():
    start = _read(START)
    receive = _read(RECEIVE)

    for token in (
        '$PositiveGainScoreWeight = 0.25',
        '$DownsideRiskScoreWeight = 0.50',
        '$EnsembleDisagreementScoreWeight = 0.25',
        '$ActionScoreMode = "baseline_paired_delta_risk_ensemble_v3"',
        '$ModelId = "hu-m43-t1-second-regular-hu-m43-c2e16-pilot200-20260713-1810"',
        '$MinimumCalibrationFires = 10',
        '$MaximumP95Loss = 25.0',
        '$MaximumP99Loss = 40.0',
        '$MaximumMaxLoss = 50.0',
        RECEIPT_SHA,
        "training_config = $cloudProjection.hyperparameters",
        "training_config_sha256",
    ):
        assert token in start
    assert '$PositiveGainScoreWeight = 1.0' not in start
    assert '$DownsideRiskScoreWeight = 0.35' not in start
    assert '"--run-manifest", $manifestPath' in receive
    assert '"--predeclared-receipt", $predeclaredReceipt' in receive
    assert '$hp = $manifest.training_config' in receive
    for key in (
        "action_score_mode",
        "model_id",
        "safety_fit_ratio",
        "minimum_threshold_lock_samples",
        "maximum_false_positive_rate",
        "maximum_max_loss",
    ):
        assert f'$hp.{key}' in receive


def test_worker_uploads_each_done_last_and_with_generation_precondition():
    startup = _extract_here_string(_read(START), "startup")

    immutable = 'gcs_upload_immutable "$out/DONE.json" "$prefix/DONE.json"'
    assert startup.count(immutable) == 1
    assert "ifGenerationMatch=0" in startup
    assert startup.index(immutable) > startup.index(
        'gcs_upload "$out/estimator.pkl" "$prefix/estimator.pkl"'
    )
    assert startup.index(immutable) > startup.index(
        'gcs_upload "$out/job_manifest.json" "$prefix/job_manifest.json"'
    )
    assert startup.index(immutable) > startup.index("write_status complete")
    assert '"hu_m43_fold_job_heartbeat_v1"' in startup
    assert '"hu_m43_fold_job_status_v1"' in startup
    assert 'trap \'self_delete\' EXIT' in startup


def test_resume_launches_only_missing_jobs_and_emits_executable_shard_syntax():
    start = _read(START)
    status = _read(STATUS)

    for token in (
        "Assert-FrozenDone $existingDone $manifest.jobs[$job] $job $manifest $manifestSha256",
        "else { $missing += $job }",
        "if ($missing.Count -eq 0) { continue }",
        '("JOB_INDICES=" + ($missing -join "+"))',
        "ResumeExisting requires CreateInstances",
        "M4.3 fold run already exists and is immutable",
    ):
        assert token in start
    assert "-StartShards $($resumeShards -join ',') -ResumeExisting" in status
    assert "-StartShards '$($resumeShards -join ',')'" not in status


def test_status_is_fail_closed_on_uri_job_range_uniqueness_and_hashes():
    text = _read(STATUS)

    for token in (
        '$donePattern = "^" + [regex]::Escape($prefix) + "/job-(\\d{2})/DONE\\.json$"',
        "$seenDoneJobs = [Collections.Generic.HashSet[int]]::new()",
        "DONE URI job is out of range",
        "Duplicate DONE job identity",
        "DONE URI does not match frozen job mapping",
        "job_manifest_sha256",
        "cloud_contract_sha256",
        "input_bundle_sha256",
        "run_manifest_sha256",
        "Get-StrictInteger",
        "must be a JSON integer, not bool/string/float",
        "m[\"training_config\"]==p[\"hyperparameters\"]",
    ):
        assert token in text


def test_receive_verifies_all_jobs_then_invokes_local_assembler_exactly_once():
    text = _read(RECEIVE)

    for token in (
        'if ($doneUris.Count -ne $ExpectedJobs)',
        "$seenDoneJobs = [Collections.Generic.HashSet[int]]::new()",
        'foreach ($name in @("estimator.pkl", "job_manifest.json", "DONE.json"))',
        "Downloaded M4.3 job hash chain is invalid",
        '"--fold-artifacts-dir", $foldArtifactsDir',
        '"--m43-data-contract", $resolvedContract',
        '"--fold-cloud-contract", $cloudContractPath',
        '"--m43-plan", $resolvedPlan',
        '"--repo-root", $repoRoot',
        '"--output-model", $outputModel',
        '"--manifest-output", $trainingManifestOutput',
        'schema = "hu_m43_fold_model_receipt_v1"',
        "OutputDir must remain under m43_fold_model_runs",
        "current_profile_mutated = $false",
        "no_runtime_activation = $true",
    ):
        assert token in text
    assert text.count("& python @assemblerArgs") == 1
    assert text.count(
        '$stageDir = Join-Path $modelRunsRoot (".receiving-{0}-{1}" -f $RunName, [guid]::NewGuid().ToString("N"))'
    ) == 1
    assert "Remove-Item -LiteralPath $candidateRebindPath -Force" in text
    assert text.index("for ($i = 0; $i -lt $ExpectedJobs; $i++)") < text.index(
        "& python @assemblerArgs"
    )
    assert 'Move-Item -LiteralPath $stageDir -Destination $OutputDir' in text


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_strict_integer_guard_rejects_bool_string_and_float():
    shell = _powershell()
    assert shell is not None
    text = _read(STATUS)
    body = text[
        text.index("function Get-StrictInteger {") : text.index("function Assert-Manifest {")
    ]
    command = (
        body
        + "; $values=@('1','1.0','true','\"1\"');"
        + "$r=@();foreach($raw in $values){$v=$raw|ConvertFrom-Json;try{Get-StrictInteger $v x|Out-Null;$r+='pass'}catch{$r+='reject'}};"
        + "$r|ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-Command", command],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == ["pass", "reject", "reject", "reject"]


def test_embedded_config_binding_rejects_hyperparameter_tampering(tmp_path: Path):
    verifier = _extract_here_string(_read(RECEIVE), "configBinding")
    hyperparameters = {
        "cross_fit_folds": 5,
        "iterations": 150,
        "positive_gain_score_weight": 0.25,
    }
    unsigned_projection = {
        "schema": "hu_m43_fold_cloud_contract_v1",
        "status": "frozen_cloud_safe",
        "hyperparameters": hyperparameters,
        "dependencies": {"numpy": "2.2.6", "scikit_learn": "1.8.0"},
        "process_environment": {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "inputs": {"train": [], "calibration": []},
        "input_bundle_sha256": "a" * 64,
        "fold_plan": {"jobs": []},
        "cloud_safe": True,
    }
    manifest_jobs = []
    projection_jobs = []
    for index in range(30):
        kind = "outer_runtime" if index % 6 == 0 else "inner_oof_safety"
        spec = {
            "job_index": index,
            "kind": kind,
            "outer_fold": index // 6,
            "inner_fold": None if index % 6 == 0 else index % 6 - 1,
        }
        digest = hashlib.sha256(str(index).encode()).hexdigest()
        projection_jobs.append({**spec, "job_spec_sha256": digest})
        manifest_jobs.append(
            {
                "job_index": index,
                "job_kind": kind,
                "outer_fold": index // 6,
                "inner_fold": None if index % 6 == 0 else index % 6 - 1,
                "job_spec": spec,
                "job_spec_sha256": digest,
            }
        )
    unsigned_projection["fold_plan"]["jobs"] = projection_jobs
    projection = {
        **unsigned_projection,
        "contract_sha256": _canonical(unsigned_projection),
    }
    manifest = {
        "training_config": dict(hyperparameters),
        "training_config_sha256": _canonical(hyperparameters),
        "dependencies": dict(unsigned_projection["dependencies"]),
        "process_environment": dict(unsigned_projection["process_environment"]),
        "inputs": {"train": [], "calibration": [], "input_bundle_sha256": "a" * 64},
        "jobs": manifest_jobs,
    }
    manifest_path = tmp_path / "manifest.json"
    projection_path = tmp_path / "projection.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    projection_path.write_text(json.dumps(projection), encoding="utf-8")

    good = subprocess.run(
        [sys.executable, "-", str(manifest_path), str(projection_path)],
        input=verifier,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert good.returncode == 0, good.stderr

    manifest["training_config"]["positive_gain_score_weight"] = 1.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bad = subprocess.run(
        [sys.executable, "-", str(manifest_path), str(projection_path)],
        input=verifier,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert bad.returncode != 0


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_all_m43_model_spot_scripts_parse():
    shell = _powershell()
    assert shell is not None
    paths = ",".join(f"'{path}'" for path in (START, STATUS, RECEIVE))
    command = (
        f"$bad=@();foreach($f in @({paths})){{"
        "$t=$null;$e=$null;[Management.Automation.Language.Parser]::ParseFile($f,[ref]$t,[ref]$e)|Out-Null;"
        "if($e.Count){$bad+=$e}};if($bad.Count){$bad|%{$_.Message};exit 1}"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
