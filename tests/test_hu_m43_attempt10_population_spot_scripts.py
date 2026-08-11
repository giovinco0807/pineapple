from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from ofc_regular.merge_hu_m4_population_shards import load_population_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt10_population.json"
START = ROOT / "scripts" / "Start-GcpHuM43Attempt10PopulationRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43Attempt10PopulationRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM43Attempt10PopulationRun.ps1"

MODEL_SCHEMA = "hu_m43_attempt10_t1_second_distilled_selector_v1"
ARTIFACT_SCHEMA = "hu_m43_attempt10_t1_second_distilled_pickle_v1"
FEATURE_SCHEMA = "hu_m43_attempt10_lambda_top12_public_infoset_features_v1"
HEAD_SCHEMA = "hu_m43_attempt10_policy_delta_safe_tail_heads_v1"
ACTION_SCORE_MODE = "attempt10_lambda_top12_distilled_safe_selector_v1"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


@pytest.mark.parametrize("path", (START, STATUS, RECEIVE))
def test_attempt10_population_scripts_parse(path: Path) -> None:
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


def test_attempt10_population_plan_is_fixed_before_labels() -> None:
    plan = json.loads(_text(PLAN))
    assert plan["schema"] == "hu_m43_population_acceptance_plan_v1"
    assert plan["status"] == "frozen_before_population_evaluation"
    runtime = plan["runtime_contract"]
    assert runtime == {
        "model_schema": MODEL_SCHEMA,
        "artifact_schema": ARTIFACT_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "head_schema": HEAD_SCHEMA,
        "action_score_mode": ACTION_SCORE_MODE,
        "fixed_safe_probability_threshold": 0.5,
        "fixed_fold_votes_min": 4,
        "folds": 5,
        "runtime_teacher_inputs": False,
        "runtime_teacher_ev_lcb_gate": False,
        "runtime_opponent_private_discard_input": False,
    }
    assert plan["fixed_baseline_profile"] == "stage19_p0"
    assert plan["opponents"] == [
        "stage19_p0",
        "stage9f_p2",
        "stage7_m5_r10",
        "random_exact_final",
    ]
    assert plan["paired_seat_swap"] is True
    assert plan["paired_seeds_per_opponent"] == 1000
    assert plan["shards"] == 20
    assert plan["paired_seeds_per_shard"] == 50
    assert plan["candidate_records"] == 8000
    assert plan["baseline_records"] == 8000
    assert plan["terminal_trace_hands"] == 16000
    assert plan["minimum_valid_overrides"] == 300
    assert plan["sizing_rule"]["optional_stopping_or_posthoc_extension_allowed"] is False
    assert plan["spot_execution"]["immutable_package_only_first"] is True
    assert plan["spot_execution"]["shard_zero_canary_required_before_fanout"] is True
    assert plan["activation_guards"]["current_profile_changed"] is False
    assert plan["activation_guards"]["runtime_policy_activated"] is False
    assert plan["post_acceptance_activation"]["population_complete_go_required"] is True
    assert plan["post_acceptance_activation"]["activation_mode_if_go"] == "explicit_opt_in_only"
    assert plan["post_acceptance_activation"]["automatic_activation_allowed"] is False


def test_attempt10_population_seed_schedule_is_disjoint_from_declared_history() -> None:
    plan = json.loads(_text(PLAN))
    stride = int(plan["seed_stride"])
    population = {
        int(plan["seed"]) + stride * index
        for index in range(int(plan["paired_seeds_per_opponent"]))
    }
    for base in plan["freshness"]["excluded_attempt10_namespace_bases"]:
        attempt10_search = {int(base) + stride * index for index in range(250)}
        assert population.isdisjoint(attempt10_search)
    for schedule in plan["freshness"]["excluded_population_schedules"]:
        prior = {
            int(schedule["seed"]) + int(schedule["seed_stride"]) * index
            for index in range(int(schedule["paired_seeds"]))
        }
        assert population.isdisjoint(prior), schedule["milestone"]


def test_attempt10_population_plan_is_accepted_by_existing_strict_merger() -> None:
    plan = load_population_plan(PLAN)
    assert plan["fixed_baseline_profile"] == "stage19_p0"
    assert plan["shards"] * plan["paired_seeds_per_shard"] == 1000


def test_dry_run_cannot_create_attempt10_instances() -> None:
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
            "-RuntimeFreezePath",
            "unused",
            "-RuntimeSourceArchivePath",
            "unused",
            "-RuntimeSourceManifestPath",
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


def test_fresh_create_requires_prior_immutable_package_only_marker() -> None:
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
            "-RuntimeFreezePath",
            "unused",
            "-RuntimeSourceArchivePath",
            "unused",
            "-RuntimeSourceManifestPath",
            "unused",
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
    assert "Fresh CreateInstances is forbidden" in completed.stdout + completed.stderr


def test_fresh_mutation_without_package_only_has_no_side_effects() -> None:
    shell = _powershell()
    if shell is None:
        pytest.skip("PowerShell is unavailable")
    run_name = f"attempt10-no-mode-{uuid.uuid4().hex}"
    run_dir = ROOT / "outputs" / "gcp_runs" / run_name
    assert not run_dir.exists()
    completed = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(START),
            "-RunName",
            run_name,
            "-ModelPath",
            "unused",
            "-TrainingManifestPath",
            "unused",
            "-RuntimeFreezePath",
            "unused",
            "-RuntimeSourceArchivePath",
            "unused",
            "-RuntimeSourceManifestPath",
            "unused",
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
    assert (
        "Fresh Attempt10 population mutation requires -PackageOnly"
        in completed.stdout + completed.stderr
    )
    assert not run_dir.exists()


def test_start_freezes_runtime_only_then_requires_canary_before_fanout() -> None:
    text = _text(START)
    for token in (
        "ofc_regular.validate_hu_m43_attempt10_acceptance",
        '"preflight"',
        '"--runtime-freeze", $resolved.runtime_freeze',
        'configs/hu_joint_policy_m43_attempt10_population.json',
        MODEL_SCHEMA,
        ARTIFACT_SCHEMA,
        FEATURE_SCHEMA,
        HEAD_SCHEMA,
        ACTION_SCORE_MODE,
        '$BaselineProfile = "stage19_p0"',
        "1000x4, 20x50, stage19_p0-baseline contract",
        "Teacher/calibration/locked JSONL must not enter",
        "Live repository configs/current-profile metadata must not enter",
        'Expand-Archive -LiteralPath $resolvedRuntimeSourceArchive',
        "runtime_source/configs/hu_m43_attempt08_runtime_requirements.txt",
        "validate_distilled_runtime_source_archive",
        "validate_expected_runtime_fingerprint",
        "PYTHONDONTWRITEBYTECODE=1",
        '"--runtime-source-manifest",',
        '"--runtime-source-root",',
        '"--runtime-dependency-root",',
        "source_model_manifest.json",
        "source_native_manifest.json",
        "validate_distilled_runtime_dependencies",
        "validate_frozen_execution_modules",
        "PACKAGE_READY.json",
        "Fresh CreateInstances is forbidden",
        "Fresh Attempt10 population mutation requires -PackageOnly",
        '"--image", $PinnedImage',
        "debian-12-bookworm-v20260609",
        "1449487925682397051",
        "a0ce16d1ab481528ac53f2e94fd9037af7f94004a38b8cf8116537bbf4277c68",
        "8c2cd111bc4e70096ff4f974f684ad146e94329871328e5b5db5d3426256c218",
        "runtime_artifacts_only = $true",
        "Publish-ImmutableObject",
        "PackageOnly deliberately freezes local bytes",
        "--baseline-profile stage19_p0",
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
    assert 'foreach ($item in @("pyproject.toml", "src"))' not in text
    assert 'Copy-Item -LiteralPath (Join-Path $repoRoot "src")' not in text
    assert '$env:PYTHONPATH = Join-Path $repoRoot "src"' not in text
    assert "$RequiredModels" not in text
    assert "$RequiredNativeBinaries" not in text
    assert '"--image-family", "debian-12"' not in text
    assert '"src", "configs"' not in text
    assert "--data-contract" not in text
    assert "--locked-receipt" not in text
    assert "--consumption-marker" not in text
    assert text.index(
        "Fresh Attempt10 population mutation requires -PackageOnly"
    ) < text.index('$repoRoot = (Resolve-Path')
    done_upload = 'gcloud storage cp "$OUT/DONE" "$RESULT/DONE" --if-generation-match=0'
    assert text.count(done_upload) == 1
    assert text.index(done_upload) > text.index(
        'gcloud storage cp "$OUT/records.jsonl" "$RESULT/records.jsonl"'
    )
    assert text.index(done_upload) > text.index("write_status complete 0")
    package_only_exit = text.index("if ($PackageOnly)")
    immutable_publish = text.index("Publish-ImmutableObject $manifestPath $manifestUri")
    canary_check = text.index('$canaryUri = "$gcsPrefix/results/shard-0000/DONE"')
    assert package_only_exit < immutable_publish < canary_check


def test_embedded_attempt10_startup_has_valid_bash_syntax() -> None:
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


def test_status_and_receive_bind_transport_merge_and_final_acceptance() -> None:
    status = _text(STATUS)
    receive = _text(RECEIVE)
    for token in (
        "hu_m43_attempt10_population_spot_manifest_v1",
        "hu_m43_attempt10_population_spot_done_v1",
        "hu_m43_attempt10_population_spot_status_v1",
        MODEL_SCHEMA,
        ACTION_SCORE_MODE,
        'baseline_profile = "stage19_p0"',
        "seen.Add($uriShard)",
        "$rowShard -ne $uriShard",
        "missing_indices",
        "active_instances",
        "Try-GetGcsJson",
        "Invalid or stale Attempt10 population status object",
        "fixed 20x50 schedule",
    ):
        assert token in status
    for token in (
        "outputs/hu_joint_policy/m43_attempt10_population",
        "refusing profile/current overwrite",
        "hu_m43_attempt10_population_spot_done_v1",
        "hu_m43_attempt10_population_spot_receipt_v1",
        "ofc_regular.merge_hu_m4_population_shards",
        "complete_content_verified",
        "runtime_binding_verified",
        MODEL_SCHEMA,
        ARTIFACT_SCHEMA,
        FEATURE_SCHEMA,
        HEAD_SCHEMA,
        ACTION_SCORE_MODE,
        'baseline_profile = "stage19_p0"',
        "teacher_calibration_locked_content_received = $false",
        "ofc_regular.validate_hu_m43_attempt10_acceptance",
        '"finalize"',
        '"--population-plan", $planPath',
        '"--records", $recordsOutput',
        '"--evaluation", $evaluationOutput',
        '"--merge-manifest", $mergeOutput',
        '"--model", $modelPath',
        '"--training-manifest", $trainingManifestPath',
        '"--runtime-freeze", $runtimeFreezePath',
        '"--runtime-source-archive", $runtimeSourceArchivePath',
        '"--runtime-source-manifest", $runtimeSourceManifestPath',
        '"--runtime-source-root", $runtimeSourceRoot',
        '"--runtime-dependency-root", $runtimeDependencyRoot',
        "population_source.zip",
        "$frozenPackageRoot",
        "validate_frozen_execution_modules",
        "validate_distilled_runtime_dependencies",
        '"--output", $statusPath',
        "$acceptCode -notin @(0, 2)",
        '"complete_go"',
        '"complete_no_go"',
        "Move-Item -LiteralPath $stage -Destination $OutputDir",
    ):
        assert token in receive
    assert '$env:PYTHONPATH = Join-Path $repoRoot "src"' not in receive
    assert '$env:PYTHONPATH = Join-Path $runtimeSourceRoot "src"' in receive
