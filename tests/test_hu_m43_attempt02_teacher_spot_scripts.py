from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "Start-GcpHuM43Attempt02TeacherRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43Attempt02TeacherRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM43Attempt02TeacherRun.ps1"
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt02.json"
CONTRACT = ROOT / "src" / "ofc_regular" / "hu_m43_attempt02_contract.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _bash() -> str | None:
    return shutil.which("bash")


def _here_string(text: str, variable: str) -> str:
    match = re.search(
        rf"\${re.escape(variable)}\s*=\s*@'\r?\n(.*?)\r?\n'@",
        text,
        flags=re.DOTALL,
    )
    assert match is not None, variable
    return match.group(1)


def test_start_is_plan_driven_c2e64_train_cal_only_with_exact_profile_quota(
    tmp_path: Path,
):
    text = _read(START)
    builder = _here_string(text, "scheduleBuilder")
    output = tmp_path / "shards.jsonl"
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    roots_per_shard = plan["budget"]["roots_per_shard"]
    projection = {
        "fresh_roots": plan["budget"]["fresh_roots"],
        "roots_per_shard": roots_per_shard,
        "fresh_shards": plan["budget"]["fresh_shards"],
        "splits": {
            split: {
                "roots": plan["fresh_splits"][split]["roots"],
                "shards": plan["fresh_splits"][split]["roots"] // roots_per_shard,
                **{
                    key: plan["fresh_splits"][split][key]
                    for key in (
                        "seed_start",
                        "seed_stride",
                        "candidate_seed_start",
                        "evaluation_seed_start",
                        "child_policy_seed_start",
                    )
                },
                "hand_seed_stride_scope": "root",
                "phase_seed_stride_scope": plan["fresh_splits"][split][
                    "phase_seed_stride_scope"
                ],
            }
            for split in ("train", "calibration")
        },
        "teacher_search": {
            key: plan["teacher_search"][key]
            for key in (
                "candidate_samples",
                "evaluation_samples",
                "common_random_futures",
                "candidate_evaluation_rng_disjoint",
                "batch_child_selectors",
                "native_batch_threads",
            )
        },
        "root_population": [
            {
                "profile": row["profile"],
                "train_roots": row["roots"]["train"],
                "calibration_roots": row["roots"]["calibration"],
            }
            for row in plan["root_population"]
        ],
        "fixed_baseline_profile": plan["fixed_baseline_profile"],
        "fixed_t2_profile": plan["fixed_continuation"]["t2_profile"],
    }
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema": "hu_m43_attempt02_preflight_receipt_v1",
                "status": "pass_frozen_before_fresh_generation",
                "fresh_generation": projection,
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-", str(PLAN), str(preflight), str(output)],
        input=builder,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    specs = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary["split_roots"] == {"train": 200, "calibration": 100}
    assert summary["roots_per_shard"] == 10
    assert summary["total_shards"] == 30
    assert summary["candidate_samples"] == 2
    assert summary["evaluation_samples"] == 64
    assert [row["split"] for row in specs].count("train") == 20
    assert [row["split"] for row in specs].count("calibration") == 10
    assert {row["split"] for row in specs} == {"train", "calibration"}
    assert all(row["roots"] == 10 for row in specs)
    assert all(set(row["profile_quota_per_shard"].values()) == {2} for row in specs)
    assert all("current" not in row["root_profiles"] for row in specs)
    assert all(row["baseline_profile"] == "stage18_p1" for row in specs)
    assert all(row["t2_profile"] == "stage9f_p2" for row in specs)
    for row in specs:
        split_plan = plan["fresh_splits"][row["split"]]
        assert row["seed_start"] == split_plan["seed_start"] + (
            row["root_offset"] * split_plan["seed_stride"]
        )
        for key in ("candidate_seed", "evaluation_seed", "child_policy_seed"):
            plan_key = f"{key}_start"
            assert row[key] == split_plan[plan_key] + (
                row["split_shard"] * split_plan["seed_stride"]
            )
    assert all(
        len(
            {
                row["candidate_seed"],
                row["evaluation_seed"],
                row["child_policy_seed"],
            }
        )
        == 3
        for row in specs
    )


def test_local_preflight_is_mandatory_and_only_hash_binding_crosses_cloud_boundary():
    text = _read(START)
    startup = _here_string(text, "startup")

    for token in (
        '"ofc_regular.hu_m43_attempt02_contract", "preflight"',
        '"--attempt01-data-contract", $Attempt01DataContractPath',
        '"--attempt01-training-manifest", $Attempt01TrainingManifestPath',
        '"--attempt01-train", $Attempt01TrainPath',
        '"--attempt01-calibration", $Attempt01CalibrationPath',
        '"--inherited-locked", $InheritedLockedPath',
        '"--output", $PreflightReceiptPath',
        'schema = "hu_m43_attempt02_teacher_spot_manifest_v1"',
        'local_preflight_binding = [ordered]@{ file_sha256 = $preflightFileSha256; receipt_sha256 = [string]$preflight.receipt_sha256 }',
        'cloud_input_boundary = "fresh_generation_schedule_only"',
        "Attempt02 cloud boundary audit failed",
    ):
        assert token in text

    # Worker metadata/source receives neither the local preflight document nor
    # any prior/inherited private path. Only its two opaque binding hashes are
    # copied into the cloud manifest.
    for forbidden in (
        "InheritedLockedPath",
        "Attempt01DataContractPath",
        "Attempt01TrainingManifestPath",
        "PreflightReceiptPath",
        "locked_holdout",
        "attempt01",
    ):
        assert forbidden.lower() not in startup.lower()
    assert "m43_attempt02_contract" not in startup
    assert 'Copy-Item -LiteralPath $PreflightReceiptPath' not in text
    assert 'storage cp $PreflightReceiptPath' not in text
    assert 'storage cp $InheritedLockedPath' not in text
    assert 'Get-Sha256 $InheritedLockedPath' not in text


def test_worker_contract_is_small_resumable_spot_and_done_is_last_immutable_commit():
    text = _read(START)
    startup = _here_string(text, "startup")

    for token in (
        "teacher.jsonl.partial",
        "checkpoint.json",
        "heartbeat.json",
        "sync_resume",
        'trap cleanup EXIT',
        'gcloud compute instances delete "$INSTANCE_NAME"',
        "--provisioning-model SPOT",
        "--instance-termination-action DELETE",
        "--boot-disk-auto-delete",
        'schema = "hu_m43_attempt02_teacher_spot_manifest_v1"',
        '"hu_m43_attempt02_teacher_done_v1"',
        "--if-generation-match=0",
        "$StartShards",
    ):
        assert token in text
    done_upload = 'gcloud storage cp "$RESULT/DONE.json" "$DONE_URI" --if-generation-match=0'
    assert startup.count(done_upload) == 1
    assert startup.index(done_upload) > startup.index(
        'for file in teacher.jsonl checkpoint.json heartbeat.json generator_summary.json run.log'
    )
    assert 'if gcloud storage ls "$DONE_URI"' in startup
    assert 'if gcloud storage ls "$RESUME_URI/checkpoint.json"' in startup


def test_status_and_receive_fail_closed_on_hash_chain_and_never_create_private_split():
    status = _read(STATUS)
    receive = _read(RECEIVE)

    for token in (
        "hu_m43_attempt02_teacher_spot_manifest_v1",
        "hu_m43_attempt02_teacher_done_v1",
        "manifest_sha256",
        "shards_manifest_sha256",
        "model_manifest_sha256",
        "native_manifest_sha256",
        "Duplicate fresh hand seed in cloud schedule",
        "Start-GcpHuM43Attempt02TeacherRun.ps1",
    ):
        assert token in status
    for token in (
        '"ofc_regular.hu_m43_attempt02_contract", "finalize-fresh"',
        '"--preflight-receipt", $PreflightReceiptPath',
        'foreach ($record in $train) { $finalizeArgs += @("--train", $record.path) }',
        'foreach ($record in $calibration) { $finalizeArgs += @("--calibration", $record.path) }',
        'schema = "hu_m43_attempt02_teacher_receive_receipt_v1"',
        'status = "verified_fresh_train_calibration_only"',
        "current_profile_mutated = $false",
        "no_runtime_activation = $true",
    ):
        assert token in receive
    assert 'Merge-Shards -Records $records -Destination $stageDestination' in receive
    assert 'foreach ($split in @("train", "calibration"))' in receive
    assert '"locked_holdout.jsonl"' not in receive
    assert 'Get-Sha256 $InheritedLockedPath' not in receive


def test_receive_commits_atomically_and_hands_original_shards_to_model_pipeline():
    receive = _read(RECEIVE)

    for token in (
        'RunName is not path-safe',
        'Attempt02 teacher OutputDir is immutable and already exists',
        '$stageDataContractPath = Join-Path $stageDir "data_contract.json"',
        '$finalizeArgs += @("--output", $stageDataContractPath)',
        'Move-Item -LiteralPath $stageDir -Destination $OutputDir',
        '[string]$done.run_name -ne $RunName',
        '[int]$done.roots -ne [int]$spec.roots',
        '$train.Count -ne 20 -or $calibration.Count -ne 10',
        'schema = "hu_m43_attempt02_teacher_downstream_inputs_v1"',
        'train_shards = $originalTrainShards',
        'train_paths = @($originalTrainShards | ForEach-Object { $_.path })',
        'merged_train_is_not_a_model_start_input = $true',
        'calibration_paths = @($originalCalibrationShards | ForEach-Object { $_.path })',
        'attempt02_data_contract = $dataContractPath',
    ):
        assert token in receive

    assert receive.index('Move-Item -LiteralPath $stageDir -Destination $OutputDir') > receive.index(
        'Write-Utf8NoBom -Path $stageReceiptPath'
    )


def test_status_uses_bounded_bulk_checkpoint_snapshot_and_retries_races():
    status = _read(STATUS)

    for token in (
        "function Copy-GcsSnapshot",
        "[int]$MaxParallel = 15",
        "[int]$TimeoutSeconds = 30",
        "Start-Job -ScriptBlock",
        "Wait-Job -Job $jobs -Any -Timeout 1",
        "Copy-GcsSnapshot -Items $doneItems -Required",
        "-MaxParallel 6 -TimeoutSeconds 60",
        "Required GCS snapshot failed",
        "bounded snapshot worker timed out or failed",
        'gcloud storage rsync "$prefix/resume" $resumeSnapshot',
        "--exclude '.*heartbeat[.]json$,.*teacher[.]jsonl[.]partial$'",
        'Get-GcsUris "$prefix/resume/*/checkpoint.json"',
        "Copy-GcsSnapshot -Items $checkpointRetryItems -TimeoutSeconds 15",
        'checkpoint.schema -ne "hu_m4_t1_second_checkpoint_v1"',
        "progress_snapshot_complete",
        "progress_snapshot_unavailable_shards",
        "Unable to obtain authoritative RUNNING VM snapshot",
        '$inactive = @($missing | Where-Object { $running -notcontains $_ })',
    ):
        assert token in status

    # rsync takes one bulk checkpoint snapshot; the following list is a fresh
    # view used to retry only objects lost to atomic generation replacement.
    assert status.count('Get-GcsUris "$prefix/resume/*/checkpoint.json"') == 1
    assert 'Get-GcsUris "$prefix/resume/*/heartbeat.json"' not in status

    # Per-object GCS copies are confined to bounded workers.  DONE content is
    # still required and subsequently checked against the full hash chain.
    assert "& gcloud storage cp $uri $donePath" not in status
    assert "& gcloud storage cp $uri $heartbeatPath" not in status
    assert '[string]$done.manifest_sha256 -ne $ManifestSha256' in status
    assert '[string]$done.shards_manifest_sha256 -ne $ShardsSha256' in status
    assert '[string]$done.model_manifest_sha256 -ne [string]$Manifest.model_manifest_sha256' in status
    assert '[string]$done.native_manifest_sha256 -ne [string]$Manifest.native_manifest_sha256' in status


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_bounded_snapshot_json_array_is_not_collapsed_into_one_uri_argument():
    status = _read(STATUS)

    # In Windows PowerShell, piping ConvertFrom-Json directly into @() leaves a
    # decoded JSON array nested as one item.  Casting that item's .uri property
    # joins all URIs with spaces.  Require the explicit encode/decode boundary
    # used by Copy-GcsSnapshot and exercise that exact round-trip in a job.
    for token in (
        "$chunkJson = ConvertTo-Json -InputObject @($chunk) -Depth 5 -Compress",
        "$decodedItems = ConvertFrom-Json -InputObject ([string]$ItemsJson)",
        "$items = @($decodedItems)",
    ):
        assert token in status
    assert "$items = @($ItemsJson | ConvertFrom-Json)" not in status

    shell = _powershell()
    assert shell is not None
    command = r"""
$chunk = @(
    [pscustomobject]@{ uri = 'gs://bucket/one/DONE.json'; path = 'one.json' },
    [pscustomobject]@{ uri = 'gs://bucket/two/DONE.json'; path = 'two.json' }
)
$chunkJson = ConvertTo-Json -InputObject @($chunk) -Depth 5 -Compress
$job = Start-Job -ScriptBlock {
    param($ItemsJson)
    $decodedItems = ConvertFrom-Json -InputObject ([string]$ItemsJson)
    $items = @($decodedItems)
    foreach ($item in $items) { [string]$item.uri }
} -ArgumentList $chunkJson
Wait-Job -Job $job | Out-Null
$uris = @($job | Receive-Job)
Remove-Job -Job $job -Force
if ($uris.Count -ne 2 -or
    $uris[0] -ne 'gs://bucket/one/DONE.json' -or
    $uris[1] -ne 'gs://bucket/two/DONE.json') {
    $uris | ConvertTo-Json -Compress
    exit 1
}
"""
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


def test_status_rechecks_late_done_before_classifying_a_shard_inactive():
    status = _read(STATUS)

    for token in (
        "function Read-VerifiedDoneRecord",
        "$preliminaryInactive",
        "$lateDoneItems",
        'uri = "$prefix/results/$([string]$spec.output_prefix)/DONE.json"',
        "Copy-GcsSnapshot -Items $lateDoneItems",
        "-TimeoutSeconds 20 -Attempts 2",
        "$done = Read-VerifiedDoneRecord",
        "$doneRecords += $done",
        "$lateDoneReclassified += [int]$item.shard",
        "late_done_reclassified_shards = $lateDoneReclassified",
    ):
        assert token in status

    # The provisional VM-based inactive set is rechecked first.  Only after
    # verified late DONE records update $completed is the final inactive set
    # and relaunch command constructed.
    recheck = status.index("$preliminaryInactive")
    verify = status.index("$done = Read-VerifiedDoneRecord", recheck)
    recompute = status.index("$completed = @($doneRecords", verify)
    classify = status.index("$inactive = @($missing", recompute)
    relaunch = status.index("$relaunch = if ($inactive.Count", classify)
    assert recheck < verify < recompute < classify < relaunch

    # Initial and late DONE records share the same full fail-closed validator.
    assert status.count("Read-VerifiedDoneRecord -Path") == 2
    for field in (
        "manifest_sha256",
        "shards_manifest_sha256",
        "source_sha256",
        "startup_sha256",
        "model_manifest_sha256",
        "native_manifest_sha256",
    ):
        assert f"$done.{field}" in status


def test_every_embedded_python_block_compiles():
    start = _read(START)
    receive = _read(RECEIVE)
    for variable in ("validator", "scheduleBuilder", "closureBuilder", "zipBuilder", "boundaryAudit"):
        compile(_here_string(start, variable), f"start-{variable}", "exec")
    compile(_here_string(receive, "script"), "receive-self-hash", "exec")
    startup = _here_string(start, "startup")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", startup, flags=re.DOTALL)
    assert len(blocks) >= 3
    for index, block in enumerate(blocks):
        compile(block, f"startup-heredoc-{index}", "exec")


@pytest.mark.skipif(_bash() is None, reason="bash unavailable")
def test_embedded_startup_script_parses_as_bash():
    startup = _here_string(_read(START), "startup")
    # WSL interop may add CR on stdin, so strip it before bash parses.
    result = subprocess.run(
        [_bash(), "-c", "tr -d '\\r' | /bin/bash -n"],
        input=startup,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_attempt02_teacher_scripts_parse():
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


@pytest.mark.skipif(
    _powershell() is None or not CONTRACT.exists(),
    reason="PowerShell or Attempt02 contract module unavailable",
)
def test_attempt02_dry_run_executes_local_preflight_without_cloud_or_vm_mutation():
    shell = _powershell()
    assert shell is not None
    run_name = f"pytest-hu-m43-a02-{uuid.uuid4().hex[:10]}"
    run_dir = ROOT / "outputs" / "gcp_runs" / run_name
    try:
        result = subprocess.run(
            [
                shell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(START),
                "-RunName",
                run_name,
                "-DryRun",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8-sig",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
        assert payload["schema"] == "hu_m43_attempt02_teacher_spot_dry_run_v1"
        assert payload["status"] == "pass_no_cloud_mutation"
        assert payload["exclusion_union_count"] == 412
        assert payload["split_roots"] == {"train": 200, "calibration": 100}
        assert payload["total_shards"] == 30
        assert payload["candidate_samples"] == 2
        assert payload["evaluation_samples"] == 64
        assert payload["inherited_private_artifact_uploaded"] is False
        assert payload["source_data_uploaded"] is False
        assert payload["create_instances"] is False
        assert payload["current_profile_mutated"] is False
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
