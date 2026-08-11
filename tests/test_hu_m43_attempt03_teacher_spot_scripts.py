from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "Start-GcpHuM43Attempt03TeacherRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43Attempt03TeacherRunStatus.ps1"
FREEZE = ROOT / "configs" / "hu_joint_policy_m43_attempt03_model_freeze.json"
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt03.json"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _json(path: Path) -> dict[str, object]:
    return json.loads(_read(path))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _function_block(text: str, name: str, next_marker: str) -> str:
    start = text.index(f"function {name}")
    end = text.index(next_marker, start)
    return text[start:end]


def test_model_freeze_binds_exact_plan_preflight_implementation_and_r2_closure() -> None:
    freeze = _json(FREEZE)
    parent = freeze["parent_plan"]
    teacher = freeze["teacher_run"]
    implementation = freeze["implementation"]

    assert freeze["status"] == "frozen_before_any_attempt03_teacher_row_was_received"
    assert _sha256(ROOT / parent["path"]) == parent["file_sha256"]

    preflight_path = ROOT / parent["preflight_receipt_path"]
    preflight = _json(preflight_path)
    assert _sha256(preflight_path) == parent["preflight_receipt_file_sha256"]
    assert preflight["receipt_sha256"] == parent["preflight_receipt_sha256"]
    assert parent["preflight_receipt_file_sha256"] != parent["preflight_receipt_sha256"]

    manifest_path = ROOT / teacher["manifest_path"]
    schedule_path = ROOT / teacher["schedule_path"]
    source_path = ROOT / teacher["source_path"]
    startup_path = ROOT / teacher["startup_path"]
    manifest = _json(manifest_path)
    assert _sha256(manifest_path) == teacher["manifest_sha256"]
    assert _sha256(schedule_path) == teacher["schedule_sha256"]
    assert _sha256(source_path) == teacher["source_sha256"]
    assert _sha256(startup_path) == teacher["startup_sha256"]
    assert manifest["run_name"] == teacher["run_name"]
    assert manifest["plan_file_sha256"] == parent["file_sha256"]
    assert manifest["preflight_file_sha256"] == parent["preflight_receipt_file_sha256"]
    assert manifest["schedule_sha256"] == teacher["schedule_sha256"]
    assert manifest["source_sha256"] == teacher["source_sha256"]
    assert manifest["startup_sha256"] == teacher["startup_sha256"]
    assert manifest["model_manifest_sha256"] == teacher[
        "source_model_manifest_sha256"
    ]
    assert manifest["native_manifest_sha256"] == teacher[
        "source_native_manifest_sha256"
    ]
    assert teacher["cloud_closure_immutable"] is True

    assert _sha256(ROOT / implementation["module"]) == implementation["file_sha256"]
    stage18 = implementation["stage18_model"]
    assert _sha256(ROOT / stage18["path"]) == stage18["file_sha256"]
    assert freeze["activation_guards"] == {
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "full_replacement_enabled": False,
        "m5_authorized": False,
    }


def test_r2_zip_has_only_posix_members_and_exact_frozen_schedule() -> None:
    freeze = _json(FREEZE)
    teacher = freeze["teacher_run"]
    archive_path = ROOT / teacher["source_path"]
    schedule_path = ROOT / teacher["schedule_path"]
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert names
        assert len(names) == len(set(names))
        assert all("\\" not in name for name in names)
        assert all(not name.startswith(("/", "../")) for name in names)
        assert archive.read("shards_manifest.jsonl") == schedule_path.read_bytes()


def test_attempt03_schedule_exactly_maps_both_logical_splits_and_seed_domains() -> None:
    freeze = _json(FREEZE)
    plan = _json(PLAN)
    specs = [
        json.loads(line)
        for line in (ROOT / freeze["teacher_run"]["schedule_path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    expected_profiles = {
        "stage19_p0",
        "stage9f_p2",
        "stage7_m5_r10",
        "stage3_baseline",
        "random_exact_final",
    }
    assert len(specs) == 70
    assert [row["logical_split"] for row in specs].count("train.fit") == 50
    assert [row["logical_split"] for row in specs].count(
        "train.precal_holdout"
    ) == 20

    hand_seeds: set[int] = set()
    global_shard = 0
    for logical_split in ("train.fit", "train.precal_holdout"):
        split = plan["fresh_splits"][logical_split]
        shard_count = split["roots"] // plan["budget"]["roots_per_shard"]
        for split_shard in range(shard_count):
            row = specs[global_shard]
            assert row["shard"] == global_shard
            assert row["split_shard"] == split_shard
            assert row["logical_split"] == logical_split
            assert row["split"] == split["record_split"] == "train"
            assert row["roots"] == 10
            assert row["seed_start"] == split["seed_start"] + (
                split_shard * 10 * split["seed_stride"]
            )
            for field in ("candidate_seed", "evaluation_seed", "child_policy_seed"):
                assert row[field] == split[f"{field}_start"] + (
                    split_shard * split["seed_stride"]
                )
            assert len(
                {row["candidate_seed"], row["evaluation_seed"], row["child_policy_seed"]}
            ) == 3
            assert set(row["profile_quota_per_shard"]) == expected_profiles
            assert set(row["profile_quota_per_shard"].values()) == {2}
            for root in range(10):
                seed = row["seed_start"] + root * row["seed_stride"]
                assert seed not in hand_seeds
                hand_seeds.add(seed)
            global_shard += 1
    assert global_shard == 70
    assert len(hand_seeds) == 700


def test_start_is_frozen_resume_only_and_revalidates_every_remote_closure_object() -> None:
    text = _read(START)
    for token in (
        "RunName is not path-safe",
        "PackageOnly and CreateInstances are mutually exclusive",
        "Attempt03 cloud closure is frozen; use -ResumeExisting for the bound r2 run",
        "$selection = @(if ($StartShards.Count -gt 0)",
        "function Stop-ProcessTree",
        "gcloud timed out after $TimeoutSeconds seconds",
        "function Assert-GcsObjectMatchesFile",
        "Remote immutable object bytes changed",
        '"$prefix/source/ofc_regular_hu_m43_attempt03_teacher_source.zip"',
        '"$prefix/source/shards_manifest.jsonl"',
        '"$prefix/source/source_model_manifest.json"',
        '"$prefix/source/source_native_manifest.json"',
        "Attempt03 DONE does not match the frozen closure",
        "Existing Attempt03 worker metadata changed",
        "Existing Attempt03 worker compute closure changed",
        "Unable to prove Attempt03 worker absence",
        "--if-generation-match=0",
    ):
        assert token in text
    assert "Compress-Archive -Path" not in text
    assert "path.relative_to(root).as_posix()" in text
    assert 'all("\\\\" not in name' in text


def test_status_is_fail_closed_and_never_treats_worker_status_as_done() -> None:
    text = _read(STATUS)
    for token in (
        "function Stop-ProcessTree",
        "function Invoke-GcloudBounded",
        "function Assert-GcsObjectMatchesFile",
        "Remote immutable object bytes changed",
        "Attempt03 plan/freeze binding changed",
        "Attempt03 schedule mapping changed at shard",
        "Attempt03 schedule contains a duplicate hand seed",
        "Attempt03 DONE does not match the frozen closure",
        "Attempt03 status does not match the frozen schedule",
        "Resolve-ShardState $hasDone $workerStatus $hasActiveVm",
        "authoritative_complete_from_verified_done_only=$true",
        "hu_m43_attempt03_teacher_spot_status_v2",
    ):
        assert token in text
    assert "if ($HasDone) { return 'complete' }" in text
    assert "if ($WorkerStatus -eq 'complete') { return 'inconsistent' }" in text
    assert "if ($WorkerStatus -eq 'running') { return 'interrupted' }" in text
    assert "& gcloud storage" not in text
    assert "& gcloud compute" not in text


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_single_shard_selection_remains_a_json_array() -> None:
    shell = _powershell()
    assert shell is not None
    text = _read(START)
    function = _function_block(text, "Expand-ShardSelection", "function Quote-NativeArgument")
    command = (
        "$ErrorActionPreference='Stop'\n"
        + function
        + "\n$selection = @(Expand-ShardSelection @('0') 70)\n"
        + "[pscustomobject]@{selected_shards=$selection}|ConvertTo-Json -Compress\n"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["selected_shards"] == [0]


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_status_classification_requires_verified_done_and_live_vm() -> None:
    shell = _powershell()
    assert shell is not None
    text = _read(STATUS)
    function = _function_block(text, "Resolve-ShardState", "$repoRoot")
    command = "$ErrorActionPreference='Stop'\n" + function + r'''
$cases = @(
    [pscustomobject]@{done=$true; worker=''; vm=$false},
    [pscustomobject]@{done=$false; worker='complete'; vm=$false},
    [pscustomobject]@{done=$false; worker='running'; vm=$true},
    [pscustomobject]@{done=$false; worker='running'; vm=$false},
    [pscustomobject]@{done=$false; worker='failed'; vm=$true},
    [pscustomobject]@{done=$false; worker='failed'; vm=$false},
    [pscustomobject]@{done=$false; worker=''; vm=$true},
    [pscustomobject]@{done=$false; worker=''; vm=$false}
)
@($cases | ForEach-Object { Resolve-ShardState $_.done $_.worker $_.vm }) | ConvertTo-Json -Compress
'''
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == [
        "complete",
        "inconsistent",
        "running",
        "interrupted",
        "starting",
        "failed",
        "starting",
        "not_started",
    ]


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_attempt03_teacher_scripts_parse_in_powershell() -> None:
    shell = _powershell()
    assert shell is not None
    paths = ",".join(f"'{path}'" for path in (START, STATUS))
    command = (
        f"$bad=@();foreach($f in @({paths})){{"
        "$t=$null;$e=$null;"
        "[Management.Automation.Language.Parser]::ParseFile($f,[ref]$t,[ref]$e)|Out-Null;"
        "if($e.Count){$bad+=$e}};"
        "if($bad.Count){$bad|ForEach-Object{$_.Message};exit 1}"
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
