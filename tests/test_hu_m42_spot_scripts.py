from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
START = REPO_ROOT / "scripts" / "Start-GcpHuM42TeacherRun.ps1"
STATUS = REPO_ROOT / "scripts" / "Get-GcpHuM42TeacherRunStatus.ps1"
RECEIVE = REPO_ROOT / "scripts" / "Receive-GcpHuM42TeacherRun.ps1"

EXPECTED_MODELS = {
    "models/opening_stage7_torch_wide.pt",
    "models/turn1_stage6_torch_wide.pt",
    "models/turn2_stage8.pkl",
    "models/turn3_stage6.pkl",
    "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
    "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl",
    "models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl",
    "models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl",
    "models/hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl",
}
EXPECTED_NATIVE_BINARIES = {
    "target/release/libofc_stage3_feature_encoder.so",
    "target/release/libofc_hu_m3_engine.so",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def test_start_script_locks_gate40_profiles_compute_and_exact_models():
    text = _read(START)

    assert '[int]$TrainRoots = 20' in text
    assert '[int]$CalibrationRoots = 10' in text
    assert '[int]$LockedHoldoutRoots = 10' in text
    assert '[int]$RootsPerShard = 5' in text
    assert '[int]$CandidateSamples = 2' in text
    assert '[int]$EvaluationSamples = 8' in text
    assert '[string]$MachineType = "c4-standard-4"' in text
    assert '[string]$BootDiskType = "hyperdisk-balanced"' in text
    assert '[string]$FallbackBootDiskType = "pd-balanced"' in text
    assert '"--provisioning-model", "SPOT"' in text
    assert '"--instance-termination-action", "DELETE"' in text
    assert '"--boot-disk-auto-delete"' in text
    assert '"--batch-child-selectors"' not in text  # Bash array uses an unquoted flag.
    assert "--batch-child-selectors" in text
    assert "--native-batch-threads" in text
    assert "m42_native_manifest=verified" in text
    assert "Pinned Stage3 feature encoder did not load" in text
    assert 'if [[ "$ALLOW_NATIVE_SOURCE_BUILD_FALLBACK" != "1" ]]' in text
    assert text.rindex("cargo build --release --lib") > text.index(
        "explicit native source-build fallback requested"
    )

    native_block = text.split("$RequiredNativeBinaries = @(", 1)[1].split("\n)", 1)[0]
    observed_native = set(re.findall(r'"(target/release/[^"\r\n]+\.so)"', native_block))
    assert observed_native == EXPECTED_NATIVE_BINARIES
    assert "9e58797bd234f9858fdfeee69fee03f6d0d20b6e3082de5ab54e9a6356ad0ea7" in text
    assert "70523749b757a92b8547a5753460c0d9c3435b6872ca76c231a8c38ecc1bbc36" in text
    assert "Native binary does not match the pinned Docker artifact" in text
    assert "hu_m42_source_native_manifest_v1" in text
    assert "rust@sha256:7d0723df719e7f213b69dc7c8c595985c3f4b060cfbee4f7bc0e347a86fe3b6a" in text
    assert 'abi = "elf64-et_dyn-x86_64-glibc>=2.34"' in text

    required_block = text.split("$RequiredModels = @(", 1)[1].split("\n)", 1)[0]
    observed = set(re.findall(r'"(models/[^"\r\n]+)"', required_block))
    assert observed == EXPECTED_MODELS
    assert "$RequiredModels.Count -ne 11" in text

    profile_block = text.split("$RootProfiles = @(", 1)[1].split("\n)", 1)[0]
    assert set(re.findall(r'"([^"\r\n]+)"', profile_block)) == {
        "stage19_p0",
        "stage9f_p2",
        "stage7_m5_r10",
        "stage3_baseline",
        "random_exact_final",
    }
    assert '$BaselineProfile = "stage18_p1"' in text
    assert '$T2Profile = "stage9f_p2"' in text


def test_start_script_has_resumable_partial_sync_done_commit_and_exact_relaunch():
    text = _read(START)

    for token in (
        "teacher.jsonl.partial",
        "checkpoint.json",
        "heartbeat.json",
        "sync_resume",
        "/resume/${OUTPUT_PREFIX}",
        "download resumable boundary",
        "DONE already exists",
        "DONE is deliberately uploaded last",
        "$StartShards",
        "selected_shards",
    ):
        assert token in text
    assert text.index('gcloud storage cp "${RESULT_DIR}/DONE" "$DONE_URI"') > text.index(
        'gcloud storage cp "${RESULT_DIR}/${file}" "${RESULT_URI}/${file}"'
    )


def test_relaunch_uses_frozen_remote_run_and_never_enters_upload_branch():
    text = _read(START)

    assert '$ErrorActionPreference = "Continue"' in text
    assert "$manifestProbeExit = $LASTEXITCODE" in text
    assert "$strictResume = $explicitRelaunch -or" in text
    assert "StartShards is relaunch-only" in text
    assert "No GCS object is overwritten in this branch" in text
    assert "Frozen M4.2 source/startup/manifest hash verification failed" in text
    assert "Relaunch parameter $ParameterName conflicts with the frozen run manifest" in text
    assert "Launch settings always come from the verified frozen manifest" in text
    outer_else = text.index("else {\nif (Test-Path $packageDir)")
    first_source_upload = text.index("gcloud storage cp $packagePath $sourceUri")
    assert first_source_upload > outer_else
    assert text.count("gcloud storage cp $packagePath $sourceUri") == 1
    assert text.count("gcloud storage cp $manifestPath $manifestUri") == 1
    assert "refusing fail-open upload" in text
    assert "Existing frozen run creation requires explicit StartShards" in text


def test_canary_initial_shards_and_startup_failure_log_contracts():
    text = _read(START)

    for token in (
        "$InitialShards",
        "InitialShards and StartShards are mutually exclusive",
        "InitialShards is new-run-only",
        "initial_shards = $selectedShardIndices",
        "Shard 0 canary has not produced a consistent resumable root",
        "$canaryResumeConsistent",
        "Unable to verify shard 0 DONE state; refusing fan-out",
        "hu_m4_t1_second_heartbeat_v1",
        "hu_m4_t1_second_checkpoint_v1",
        'STARTUP_LOG="/tmp/ofc-hu-m42-startup.log"',
        'exec > >(tee -a "$STARTUP_LOG") 2>&1',
        "upload_startup_log",
        "startup_shard_${SHARD_INDEX}.log",
        "instance/service-accounts/default/token",
    ):
        assert token in text
    assert "PackageOnly and CreateInstances are mutually exclusive" in text
    assert "hu_m42_spot_package_only_result_v1" in text
    assert text.index('exec > >(tee -a "$STARTUP_LOG") 2>&1') < text.index(
        "sudo apt-get update -y"
    )
    cleanup = text.split("cleanup() {", 1)[1].split("trap cleanup EXIT", 1)[0]
    assert "upload_startup_log" in cleanup


def test_status_and_receive_reject_stale_done_hash_chains_and_receive_audits():
    status = _read(STATUS)
    receive = _read(RECEIVE)

    for token in (
        "done.manifest_sha256",
        "done.source_sha256",
        "done.startup_sha256",
        "done.shards_manifest_sha256",
        "done.model_manifest_sha256",
        "done.native_manifest_sha256",
        "Invalid or stale M4.2 DONE object",
    ):
        assert token in status
    for token in (
        "M4.2 frozen source/startup SHA256 mismatch",
        "DONE manifest/source hash chain mismatch",
        "source native manifest SHA256 mismatch",
        "Downloaded result hash mismatch",
        "hu_m4_t1_second_shard_v1",
        "Generator completion config mismatch",
        "ofc_regular.audit_hu_m4_t1_data",
        '"--train"',
        '"--calibration"',
        '"--locked-holdout"',
        "audit.status -ne \"pass\"",
    ):
        assert token in receive


def test_status_exposes_missing_shard_relaunch_command():
    text = _read(STATUS)

    assert "inactive_missing_shards" in text
    assert "-StartShards '$($inactiveMissing -join ',')' -CreateInstances" in text
    assert "resumable_partial_roots" in text
    assert "resumable_progress" in text


@pytest.mark.skipif(_powershell() is None, reason="PowerShell is unavailable")
def test_dry_run_is_side_effect_free_and_emits_expected_gate_plan():
    shell = _powershell()
    assert shell is not None
    run_name = f"pytest-m42-dry-{uuid.uuid4().hex[:12]}"
    forbidden_output = REPO_ROOT / "outputs" / "gcp_runs" / run_name
    command = [
        shell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(START),
        "-RunName",
        run_name,
        "-InitialShards",
        "0",
        "-DryRun",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["execution"] == "dry_run"
    assert plan["total_roots"] == 40
    assert plan["split_roots"] == {
        "train": 20,
        "calibration": 10,
        "locked_holdout": 10,
    }
    assert plan["roots_per_shard"] == 5
    assert plan["total_shards"] == 8
    assert plan["selected_shards"] == [0]
    assert plan["initial_shards"] == [0]
    assert plan["initial_launch_subset"] is True
    assert plan["candidate_samples"] == 2
    assert plan["evaluation_samples"] == 8
    assert plan["required_model_count"] == 11
    assert plan["required_model_bytes"] == 80_128_891
    assert plan["required_native_binary_count"] == 2
    assert plan["required_native_binary_bytes"] == 1_625_488
    assert {row["path"] for row in plan["required_native_binaries"]} == EXPECTED_NATIVE_BINARIES
    assert {row["sha256"] for row in plan["required_native_binaries"]} == {
        "9e58797bd234f9858fdfeee69fee03f6d0d20b6e3082de5ab54e9a6356ad0ea7",
        "70523749b757a92b8547a5753460c0d9c3435b6872ca76c231a8c38ecc1bbc36",
    }
    assert plan["machine_type"] == "c4-standard-4"
    assert plan["boot_disk_type"] == "hyperdisk-balanced"
    assert plan["termination_action"] == "DELETE"
    assert plan["create_instances"] is False
    assert plan["relaunch_existing_run"] is False
    assert [row["split"] for row in plan["shards"]].count("train") == 4
    assert [row["split"] for row in plan["shards"]].count("calibration") == 2
    assert [row["split"] for row in plan["shards"]].count("locked_holdout") == 2
    assert all(row["root_profile_weights"] == [1, 1, 1, 1, 1] for row in plan["shards"])
    assert not forbidden_output.exists()

    conflict = subprocess.run(
        command[:-1] + ["-StartShards", "1", "-DryRun"],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert conflict.returncode != 0
    assert "InitialShards and StartShards are mutually exclusive" in conflict.stderr


@pytest.mark.skipif(_powershell() is None, reason="PowerShell is unavailable")
def test_all_m42_scripts_parse_in_powershell():
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
