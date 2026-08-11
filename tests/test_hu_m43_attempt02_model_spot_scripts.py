from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from ofc_regular import hu_m43_attempt02_fold_training as attempt02_fold
from ofc_regular.train_hu_m4_joint_model import M43FoldJobSpec


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "Start-GcpHuM43Attempt02ModelRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43Attempt02ModelRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM43Attempt02ModelRun.ps1"


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


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_all_attempt02_model_spot_scripts_parse() -> None:
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


def test_start_cloud_surface_is_exactly_twenty_fresh_train_shards() -> None:
    text = _read(START)
    startup = _extract_here_string(text, "startup")

    assert "[Parameter(Mandatory = $true)][string[]]$Train" in text
    assert "$ExpectedTrainShards = 20" in text
    assert "Attempt02 model run requires exactly 20 original fresh train shards" in text
    assert '"ofc_regular.train_hu_m43_attempt02_fold_job", "prepare"' in text
    assert 'foreach ($path in $resolvedTrain) { $prepareArgs += @("--train", $path) }' in text
    assert 'cloud_input_boundary = "fresh_train_only"' in text
    assert 'set(p["inputs"])=={"train"}' in text
    assert 'set(m["inputs"])=={"train","input_bundle_sha256"}' in startup
    assert "--calibration" not in startup
    assert "--attempt02-data-contract" not in startup
    assert "--locked-holdout" not in startup
    param_block = text[: text.index("$ErrorActionPreference")]
    for forbidden in ("$Calibration", "$Attempt02DataContract", "$LockedHoldout"):
        assert forbidden not in param_block


def test_exact_thirty_job_grid_small_spot_shards_and_canary_fanout_are_frozen() -> None:
    text = _read(START)
    startup = _extract_here_string(text, "startup")

    for token in (
        "$ExpectedJobs = 30",
        "$JobsPerShard = 4",
        "$ExpectedShards = 8",
        "[int[]]$StartShards = @(0)",
        "canary_shard = 0",
        "fanout_shards = @(1,2,3,4,5,6,7)",
        "Canary shard 0 must complete before Attempt02 fanout",
        '"--provisioning-model","SPOT"',
        '"--instance-termination-action","DELETE"',
        'trap \'self_delete\' EXIT',
        'IFS=\'+\' read -r -a JOB_ARRAY <<< "$JOB_INDICES"',
        'run_job "$job" & pids+=("$!")',
    ):
        assert token in text or token in startup
    assert "max_parallel_processes = 4" in text
    assert "jobs_per_shard = $JobsPerShard" in text


def test_worker_has_pinned_environment_heartbeat_checkpoint_and_done_last() -> None:
    text = _read(START)
    startup = _extract_here_string(text, "startup")

    assert "'numpy==2.2.6' 'scikit-learn==1.8.0'" in startup
    for name, value in (
        ("PYTHONHASHSEED", "0"),
        ("OMP_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
    ):
        assert f"{name}={value}" in startup
    immutable_done = 'gcs_upload_immutable "$out/DONE.json" "$prefix/DONE.json"'
    assert startup.count(immutable_done) == 1
    assert "ifGenerationMatch=0" in startup
    assert '"hu_m43_attempt02_v4_fold_job_heartbeat_v1"' in startup
    assert '"hu_m43_attempt02_v4_fold_job_checkpoint_v1"' in startup
    assert startup.index(immutable_done) > startup.index(
        'gcs_upload "$out/estimator.pkl" "$prefix/estimator.pkl"'
    )
    assert startup.index(immutable_done) > startup.index(
        'gcs_upload "$out/checkpoint.json" "$prefix/checkpoint.json"'
    )
    assert startup.index(immutable_done) > startup.index("write_status complete")


def test_worker_generated_one_job_outputs_pass_embedded_cloud_contract_validation(
    tmp_path: Path,
) -> None:
    startup = _extract_here_string(_read(START), "startup")
    validators = [
        block
        for block in re.findall(r"<<'PY'\n(.*?)\nPY", startup, flags=re.DOTALL)
        if 'job_manifest.json' in block and 'cloud_contract_file_sha256' in block
    ]
    assert len(validators) == 1
    validator = validators[0]
    assert 'obj["cloud_contract_sha256"]' not in validator

    source_sha = "a" * 64
    contract_file_sha = "b" * 64
    input_bundle_sha = "c" * 64
    run_manifest_sha = "d" * 64
    training_config_sha = "e" * 64
    spec = M43FoldJobSpec(
        job_index=0,
        kind="outer_runtime",
        outer_fold=0,
        inner_fold=None,
        estimator_fold_index=0,
        estimator_seed=2026072801,
        fit_samples=160,
        fit_identity_sha256="f" * 64,
        outer_validation_samples=40,
        outer_validation_identity_sha256="1" * 64,
        inner_validation_samples=0,
        inner_validation_identity_sha256=None,
        outer_assignment_sha256="2" * 64,
        inner_assignment_sha256=None,
    )
    output_dir = tmp_path / "job-00"
    output_dir.mkdir()
    artifact_path = output_dir / "estimator.pkl"
    artifact_path.write_bytes(b"local-one-job-contract-smoke")
    job_manifest = attempt02_fold._job_manifest(
        spec,
        run_name="local-one-job-smoke",
        source_sha256=source_sha,
        run_manifest_sha256=run_manifest_sha,
        contract_file_sha256=contract_file_sha,
        input_bundle_sha256=input_bundle_sha,
        training_config_sha256=training_config_sha,
        artifact_sha256=attempt02_fold._file_sha256(artifact_path),
    )
    manifest_path = output_dir / "job_manifest.json"
    manifest_path.write_text(
        json.dumps(job_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    done = {
        key: value
        for key, value in job_manifest.items()
        if key not in {"job_spec", "schema", "status"}
    }
    done.update(
        {
            "schema": attempt02_fold.M43_ATTEMPT02_FOLD_DONE_SCHEMA,
            "status": "complete",
            "job_manifest_sha256": attempt02_fold._file_sha256(manifest_path),
        }
    )
    (output_dir / "DONE.json").write_text(
        json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert job_manifest["cloud_contract_file_sha256"] == contract_file_sha
    assert done["cloud_contract_file_sha256"] == contract_file_sha
    assert "cloud_contract_sha256" not in job_manifest
    assert "cloud_contract_sha256" not in done

    run_manifest = {
        "run_name": "local-one-job-smoke",
        "jobs": [{"job_spec_sha256": spec.sha256}],
        "source": {"sha256": source_sha},
        "cloud_contract": {"file_sha256": contract_file_sha},
        "inputs": {"input_bundle_sha256": input_bundle_sha},
    }
    run_manifest_path = tmp_path / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-",
            str(run_manifest_path),
            "0",
            str(output_dir),
            run_manifest_sha,
        ],
        input=validator,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_all_spot_job_validators_use_worker_cloud_contract_file_digest_key() -> None:
    start = _read(START)
    status = _read(STATUS)
    receive = _read(RECEIVE)

    assert "$Done.cloud_contract_file_sha256" in start
    assert 'obj["cloud_contract_file_sha256"]' in _extract_here_string(
        start, "startup"
    )
    assert "$Object.cloud_contract_file_sha256" in status
    assert "$object.cloud_contract_file_sha256" in receive
    assert "$Done.cloud_contract_sha256" not in start
    assert 'obj["cloud_contract_sha256"]' not in _extract_here_string(
        start, "startup"
    )
    assert "$Object.cloud_contract_sha256" not in status
    assert "$object.cloud_contract_sha256" not in receive


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_resume_rejects_legacy_frozen_startup_before_remote_audit(
    tmp_path: Path,
) -> None:
    shell = _powershell()
    assert shell is not None
    text = _read(START)
    function_start = text.index("function Assert-CanonicalResumeStartup")
    function_end = text.index("\nfunction Assert-FrozenDone", function_start)
    function_text = text[function_start:function_end]
    resume_start = text.index("if ($ResumeExisting) {")
    guard_call = text.index(
        "Assert-CanonicalResumeStartup -Path $startupPath", resume_start
    )
    remote_audit = text.index("$remoteAudit =", resume_start)
    assert resume_start < guard_call < remote_audit
    assert text.count("Assert-CanonicalResumeStartup -Path $startupPath") == 1

    canonical = tmp_path / "canonical-startup.sh"
    legacy = tmp_path / "legacy-startup.sh"
    missing = tmp_path / "missing-validator-startup.sh"
    canonical.write_text(
        'assert obj["cloud_contract_file_sha256"]==m["cloud_contract"]["file_sha256"]\n',
        encoding="utf-8",
    )
    legacy.write_text(
        'assert obj["cloud_contract_sha256"]==m["cloud_contract"]["file_sha256"]\n',
        encoding="utf-8",
    )
    missing.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    command = (
        function_text
        + "\n"
        + f"Assert-CanonicalResumeStartup -Path {json.dumps(str(canonical))}\n"
        + "$legacyRejected=$false\n"
        + "try { Assert-CanonicalResumeStartup -Path "
        + json.dumps(str(legacy))
        + " } catch { if($_.Exception.Message -notmatch 'create a new RunName'){throw}; $legacyRejected=$true }\n"
        + "if(-not $legacyRejected){throw 'legacy startup was accepted'}\n"
        + "$missingRejected=$false\n"
        + "try { Assert-CanonicalResumeStartup -Path "
        + json.dumps(str(missing))
        + " } catch { if($_.Exception.Message -notmatch 'create a new RunName'){throw}; $missingRejected=$true }\n"
        + "if(-not $missingRejected){throw 'startup without canonical validator was accepted'}\n"
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


def test_ast_source_closure_portable_digest_and_entry_boundary_are_audited() -> None:
    frozen_pattern = '"(?i)(^|/)(outputs?|configs?|data)(/|$)"'
    start = _read(START)
    for path in (START, STATUS, RECEIVE):
        text = _read(path)
        assert "[Array]::Sort($entries,[StringComparer]::Ordinal)" in text or (
            "[Array]::Sort($entries, [StringComparer]::Ordinal)" in text
        )
        assert "non-portable entry separators" in text
        assert "entries_sha256" in text
        assert frozen_pattern in text
    for token in (
        "ast_recursive_local_import_closure_v1",
        "src/ofc_regular/train_hu_m43_attempt02_fold_job.py",
        "src/ofc_regular/assemble_hu_m43_attempt02_model.py",
        "sourceClosureBuilder",
        "local_input_values_embedded = $false",
        "nontrain_cloud_inputs = 0",
    ):
        assert token in start


def test_every_startup_python_heredoc_compiles() -> None:
    startup = _extract_here_string(_read(START), "startup")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", startup, flags=re.DOTALL)
    assert len(blocks) >= 6
    for index, block in enumerate(blocks):
        compile(block, f"attempt02-startup-{index}", "exec")


def test_embedded_startup_has_valid_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    startup = _extract_here_string(_read(START), "startup")
    result = subprocess.run(
        [bash, "-n"],
        input=(startup + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")


def test_dryrun_is_local_only_immutable_and_persists_boundary_receipt() -> None:
    text = _read(START)
    assert '$artifactDir = if ($DryRun) { Join-Path $runDir "dryrun" } else { $runDir }' in text
    assert "DryRun RunName is immutable and already exists" in text
    assert "if (-not $DryRun) { $remoteExists = Test-GcsObject $manifestUri }" in text
    assert "if ($DryRun) {" in text
    assert "$plan.boundary_audit = $boundaryAudit" in text
    assert "Write-Utf8CreateNew -Path $dryRunPlanPath" in text
    dryrun_index = text.index("if ($DryRun) {", text.index("Assert-SourcePackagePolicy"))
    upload_index = text.index('Invoke-Gcloud @("storage", "cp"', dryrun_index)
    assert dryrun_index < upload_index


def test_embedded_dryrun_boundary_audit_passes_clean_and_rejects_leaks(
    tmp_path: Path,
) -> None:
    audit = _extract_here_string(_read(START), "boundaryAuditScript")
    repo = tmp_path / "repo"
    repo.mkdir()
    train = tmp_path / "train-000.jsonl"
    train.write_text("fresh\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    manifest = tmp_path / "manifest.json"
    startup = tmp_path / "startup.sh"
    archive = tmp_path / "source.zip"
    contract.write_text(json.dumps({"inputs": {"train": []}}), encoding="utf-8")
    clean_manifest = {
        "inputs": {"train": [], "input_bundle_sha256": "a" * 64},
        "cloud_contract": {"uri": "gs://bucket/contract.json"},
    }
    manifest.write_text(json.dumps(clean_manifest), encoding="utf-8")
    startup.write_text("#!/bin/sh\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("src/ofc_regular/worker.py", "VALUE = 1\n")

    command = [
        sys.executable,
        "-",
        str(contract),
        str(startup),
        str(manifest),
        str(archive),
        str(repo),
        str(train),
    ]
    good = subprocess.run(
        command,
        input=audit,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout)["nontrain_cloud_inputs"] == 0

    leaked = dict(clean_manifest)
    leaked["calibration"] = {"path": "secret.jsonl"}
    manifest.write_text(json.dumps(leaked), encoding="utf-8")
    bad_metadata = subprocess.run(
        command,
        input=audit,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert bad_metadata.returncode != 0

    manifest.write_text(json.dumps(clean_manifest), encoding="utf-8")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("src/ofc_regular/worker.py", f'VALUE = r"{train}"\n')
    bad_path = subprocess.run(
        command,
        input=audit,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert bad_path.returncode != 0


def test_status_fails_closed_on_uri_identity_and_hash_chain() -> None:
    text = _read(STATUS)
    for token in (
        "$seenDoneJobs=[Collections.Generic.HashSet[int]]::new()",
        "Duplicate DONE job identity",
        "gcloud auth print-access-token --quiet",
        "$snapshotFetcher=@'",
        "deadline=started+55.0",
        "ThreadPoolExecutor(max_workers=8)",
        "generation={generation}",
        "for attempt in range(1,4)",
        '"status":"pass"',
        "refusing to report the run inactive",
        "remote estimator/job-manifest/DONE hash chain is invalid",
        "bulk_snapshot=$snapshotAudit",
        "artifact_sha256",
        "job_manifest_sha256",
        "cloud_contract_file_sha256",
        "input_bundle_sha256",
        "run_manifest_sha256",
        "checkpoint=$observations.checkpoint",
    ):
        assert token in text
    assert "gcloud storage rsync" not in text
    assert "gcloud storage cat" not in text
    assert "gcloud storage ls" not in text
    assert "Start-GcpHuM43Attempt02ModelRun.ps1" in text
    assert "$instances=@()" in text
    assert "$instances=@($parsedInstances)" in text


def test_bounded_snapshot_fetcher_compiles_and_handles_empty_prefix_by_contract() -> None:
    fetcher = _extract_here_string(_read(STATUS), "snapshotFetcher")
    compile(fetcher, "attempt02-bounded-gcs-snapshot", "exec")
    assert 'payload.get("items",[])' in fetcher
    assert "if entries:" in fetcher
    assert '"selected_objects":len(entries)' in fetcher
    assert 'temporary.unlink(missing_ok=True)' in fetcher
    assert 'root.glob("job-*/*")' in fetcher


def test_bounded_snapshot_fetcher_accepts_empty_and_retries_atomic_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetcher = _extract_here_string(_read(STATUS), "snapshotFetcher")

    class Response:
        def __init__(self, payload: bytes) -> None:
            self.stream = io.BytesIO(payload)

        def read(self, size: int = -1) -> bytes:
            return self.stream.read(size)

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def execute(run: str, destination: Path, urlopen: object) -> dict[str, object]:
        monkeypatch.setenv("M43_ATTEMPT02_GCS_TOKEN", "test-token")
        monkeypatch.setattr(sys, "argv", ["snapshot", "test-bucket", run, str(destination)])
        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch("time.sleep", return_value=None):
            exec(compile(fetcher, "attempt02-snapshot", "exec"), {"__name__": "__main__"})
        return json.loads(output.getvalue())

    def empty_urlopen(request: urllib.request.Request, timeout: float) -> Response:
        assert timeout <= 10.0
        assert "alt=media" not in request.full_url
        return Response(b'{"items":[]}')

    empty = execute("empty-run", tmp_path / "empty", empty_urlopen)
    assert empty["status"] == "pass"
    assert empty["attempt_count"] == 1
    assert empty["selected_objects"] == 0

    list_calls = 0

    def changing_urlopen(request: urllib.request.Request, timeout: float) -> Response:
        nonlocal list_calls
        assert timeout <= 10.0
        url = request.full_url
        if "alt=media" not in url:
            list_calls += 1
            generation = "7" if list_calls == 1 else "8"
            payload = {
                "items": [
                    {
                        "name": "runs/changing-run/results/job-00/status.json",
                        "generation": generation,
                        "size": "24",
                    }
                ]
            }
            return Response(json.dumps(payload).encode("utf-8"))
        if "generation=7" in url:
            raise urllib.error.HTTPError(url, 404, "generation replaced", None, None)
        assert "generation=8" in url
        return Response(b'{"generation":8}')

    changing_destination = tmp_path / "changing"
    changing = execute("changing-run", changing_destination, changing_urlopen)
    assert changing["status"] == "pass"
    assert changing["attempt_count"] == 2
    assert changing["selected_objects"] == 1
    assert json.loads(
        (changing_destination / "job-00" / "status.json").read_text(encoding="utf-8")
    ) == {"generation": 8}


def test_start_uses_bounded_async_create_and_fails_closed_on_ambiguity() -> None:
    text = _read(START)
    for token in (
        '"--async","--format=value(name)"',
        "function Wait-AsyncInstanceCreate",
        "function Invoke-ProcessBounded",
        "function Invoke-GcloudProcessBounded",
        "'compute','operations','describe'",
        "'compute','instances','describe'",
        "-TimeoutSeconds 60",
        "operation_done_and_instance_observed",
        "bounded_instance_observed",
        "bounded_instance_observed_after_submission_timeout",
        "Reconcile $vmName before resuming",
        "no fallback was attempted",
        "terminal_submission_failure",
    ):
        assert token in text
    assert "& gcloud" not in text
    assert 'if ($code -eq 0) { $created=$true' not in text
    assert text.index('"--async","--format=value(name)"') < text.index(
        "Wait-AsyncInstanceCreate -Name $vmName"
    )


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_async_create_verifier_distinguishes_success_from_terminal_failure() -> None:
    shell = _powershell()
    assert shell is not None
    text = _read(START)
    functions = text[
        text.index("function Invoke-Gcloud") : text.index("function Get-StrictInteger")
    ]
    script = (
        "$ProjectId='test-project'\n"
        + functions
        + r'''
$Scenario="success"
function Invoke-GcloudProcessBounded {
    param([string[]]$Arguments,[int]$TimeoutSeconds)
    $joined=@($Arguments)-join ' '
    if($joined -match 'compute operations describe'){
        $json=if($Scenario -eq "success"){'{"name":"operation-success","status":"DONE"}'}else{'{"name":"operation-terminal","status":"DONE","error":{"errors":[{"code":"ZONE_RESOURCE_POOL_EXHAUSTED","message":"capacity"}]}}'}
        return [pscustomobject]@{timed_out=$false;exit_code=0;stdout=@($json);stderr=@();output=@($json)}
    }
    if($joined -match 'compute instances describe'){$json='{"name":"vm-test","zone":"zones/us-central1-a","machineType":"machineTypes/c4-standard-8","status":"RUNNING"}';return [pscustomobject]@{timed_out=$false;exit_code=0;stdout=@($json);stderr=@();output=@($json)}}
    if($joined -match 'compute instances list'){return [pscustomobject]@{timed_out=$false;exit_code=0;stdout=@('[]');stderr=@();output=@('[]')}}
    throw "unexpected fake gcloud call: $joined"
}
$success=Wait-AsyncInstanceCreate -Name 'vm-test' -Zone 'us-central1-a' -Machine 'c4-standard-8' -OperationName 'operation-success' -TimeoutSeconds 10
if(-not $success.created -or $success.terminal_failure -or $success.verification -ne 'operation_done_and_instance_observed'){throw 'unexpected async success result'}
$Scenario="terminal"
$terminal=Wait-AsyncInstanceCreate -Name 'vm-test' -Zone 'us-central1-a' -Machine 'c4-standard-8' -OperationName 'operation-terminal' -TimeoutSeconds 10
if($terminal.created -or -not $terminal.terminal_failure -or $terminal.operation_status -ne 'DONE'){throw 'unexpected terminal failure result'}
'''
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
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


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_bounded_process_enforces_deadline_kills_tree_and_captures_success(
    tmp_path: Path,
) -> None:
    shell = _powershell()
    assert shell is not None
    text = _read(START)
    functions = text[
        text.index("function Convert-ToNativeProcessArgument") : text.index(
            "function Invoke-GcloudProcessBounded"
        )
    ]
    child_pid = tmp_path / "child.pid"
    parent = tmp_path / "spawn-child.ps1"
    parent.write_text(
        "param([string]$PidFile)\n"
        "$hostPath=(Get-Process -Id $PID).Path\n"
        "$child=Start-Process -FilePath $hostPath -ArgumentList "
        "@('-NoProfile','-NonInteractive','-Command','Start-Sleep -Seconds 30') -PassThru\n"
        "Set-Content -LiteralPath $PidFile -Value $child.Id\n"
        "Start-Sleep -Seconds 30\n",
        encoding="utf-8",
    )
    parent_ps = str(parent).replace("'", "''")
    child_pid_ps = str(child_pid).replace("'", "''")
    script = (
        functions
        + f"$parentScript='{parent_ps}'\n$childPidFile='{child_pid_ps}'\n"
        + r'''
$hostPath=(Get-Process -Id $PID).Path
$watch=[Diagnostics.Stopwatch]::StartNew()
$timed=Invoke-ProcessBounded -FilePath $hostPath -Arguments @('-NoProfile','-NonInteractive','-File',$parentScript,'-PidFile',$childPidFile) -TimeoutSeconds 2
$watch.Stop()
if(-not $timed.timed_out -or $null -ne $timed.exit_code -or -not $timed.process_tree_terminated -or $watch.Elapsed.TotalSeconds -gt 8){throw 'process-tree deadline was not enforced'}
if(-not (Test-Path -LiteralPath $childPidFile)){throw 'child PID was not recorded before timeout'}
$spawnedPid=[int](Get-Content -LiteralPath $childPidFile -Raw)
Start-Sleep -Milliseconds 300
if($null -ne (Get-Process -Id $spawnedPid -ErrorAction SilentlyContinue)){throw "timed-out child process survived: $spawnedPid"}
$ok=Invoke-ProcessBounded -FilePath $hostPath -Arguments @('-NoProfile','-NonInteractive','-Command','Write-Output "operation-ok"') -TimeoutSeconds 5
if($ok.timed_out -or $ok.exit_code -ne 0 -or @($ok.stdout)[0] -ne 'operation-ok'){throw 'bounded success capture failed'}
'''
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_receive_strict_integer_executes_as_a_return_statement() -> None:
    shell = _powershell()
    assert shell is not None
    text = _read(RECEIVE)
    function = text[
        text.index("function Get-StrictInteger") : text.index(
            "function Write-Utf8CreateNew"
        )
    ]
    script = (
        function
        + "$value=Get-StrictInteger ([int64]30) 'job_count'\n"
        + "if($value -ne 30 -or $value.GetType() -ne [int64]){throw 'integer return failed'}\n"
        + "$rejected=$false\n"
        + "try{Get-StrictInteger $true 'job_count'|Out-Null}catch{$rejected=$true}\n"
        + "if(-not $rejected){throw 'bool was accepted'}\n"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_receive_gcloud_and_uri_helpers_execute_array_returns() -> None:
    shell = _powershell()
    assert shell is not None
    text = _read(RECEIVE)
    functions = text[
        text.index("function Invoke-Gcloud") : text.index(
            "function Get-StrictInteger"
        )
    ]
    script = (
        "$ErrorActionPreference='Stop'\n"
        + "function gcloud { param([Parameter(ValueFromRemainingArguments=$true)]$Rest); "
        + "$global:LASTEXITCODE=0; ' gs://bucket/a '; 'gs://bucket/b' }\n"
        + functions
        + "$raw=@(Invoke-Gcloud @('storage','ls','ignored'))\n"
        + "if($raw.Count -ne 2){throw 'Invoke-Gcloud array return failed'}\n"
        + "$uris=@(Get-GcsUris 'ignored')\n"
        + "if($uris.Count -ne 2 -or $uris[0] -ne 'gs://bucket/a' -or $uris[1] -ne 'gs://bucket/b'){throw 'Get-GcsUris array return failed'}\n"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_receive_verifies_all_jobs_then_runs_frozen_local_assembler_once() -> None:
    text = _read(RECEIVE)
    for token in (
        "$ExpectedTrainShards = 20",
        "$ExpectedCalibrationShards = 10",
        'foreach($name in @("estimator.pkl","job_manifest.json","DONE.json"))',
        "Downloaded Attempt02 job hash chain is invalid",
        '"ofc_regular.assemble_hu_m43_attempt02_model"',
        '"--fold-artifacts-dir",$foldArtifactsDir',
        '"--attempt02-data-contract",$resolvedDataContract',
        '"--fold-cloud-contract",$cloudContractPath',
        '"--output-model",$outputModel',
        '"--manifest-output",$trainingManifestOutput',
        'schema="hu_m43_attempt02_v4_model_receive_receipt_v1"',
        "OutputDir must remain under m43_attempt02_model_runs",
        "precalibration_no_go_opens_local_calibration_or_contract=$false",
        "current_profile_mutated=$false",
        "runtime_policy_activated=$false",
    ):
        assert token in text
    assert text.count("& python @assemblerArgs") == 1
    assert text.count(
        '$stageDir=Join-Path $modelRunsRoot(".receiving-{0}-{1}" -f $RunName,[guid]::NewGuid().ToString("N"))'
    ) == 1
    assert text.index("for($i=0;$i -lt $ExpectedJobs;$i++)") < text.index(
        "& python @assemblerArgs"
    )
    assert "hu_m43_attempt02_v4_assembly_decision_v1" in text
    assert "no_go_precalibration" in text
    assert "data_contract_opened -ne $false" in text
    assert "fresh_rows_opened -ne $false" in text
    assert 'Move-Item -LiteralPath $stageDir -Destination $OutputDir' in text


def test_receive_idempotency_fails_closed_on_model_status_mismatch() -> None:
    text = _read(RECEIVE)

    for token in (
        '$allowedExistingStatuses=@("no_go_precalibration","verified_candidate_ready_for_freeze","verified_no_go_calibration")',
        'existing receipt job_count',
        'cloud_input_boundary -ne "fresh_train_only"',
        'Existing Attempt02 pre-calibration No-Go contains an unexpected model artifact',
        'Existing Attempt02 model hash/status changed',
    ):
        assert token in text
    assert text.index('$null -eq $existing.model_sha256') < text.index(
        '$existing|ConvertTo-Json -Depth 20;exit 0'
    )


def test_attempt02_model_scripts_never_reference_current_profile_path() -> None:
    for path in (START, STATUS, RECEIVE):
        text = _read(path).lower()
        assert "ai_profiles.py" not in text
        assert "profile/current" not in text
        assert "set-current" not in text
