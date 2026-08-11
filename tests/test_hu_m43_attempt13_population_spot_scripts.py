from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt13_population.json"
START = ROOT / "scripts" / "Start-GcpHuM43Attempt13PopulationRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43Attempt13PopulationRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM43Attempt13PopulationRun.ps1"
PLAN_SHA = "bf05a9477e18049883926537b4fb15c7e3ecf350c212072aed7c5b981d593a17"
REGISTRY_SHA = "5872a79f5337d7d252ce1b5f60085939109dd08ff5cdb41737fcfc3acc98cdb0"


@pytest.mark.parametrize("path", [START, STATUS, RECEIVE])
def test_attempt13_population_powershell_parses(path: Path) -> None:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is unavailable")
    command = (
        "$tokens=$null;$errors=$null;"
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{path}',[ref]$tokens,[ref]$errors)|Out-Null;"
        "if($errors.Count){$errors|% Message;exit 1}"
    )
    completed = subprocess.run(
        [shell, "-NoProfile", "-Command", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")


def test_attempt13_population_plan_and_spot_shards_are_fixed() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["seed"] == 250108071901
    assert plan["seed_stride"] == 1000003
    assert plan["paired_seeds_per_opponent"] == 1000
    assert plan["shards"] == 20
    assert plan["paired_seeds_per_shard"] == 50
    assert plan["opponents"] == [
        "stage19_p0",
        "stage9f_p2",
        "stage7_m5_r10",
        "random_exact_final",
    ]
    assert plan["freshness"]["excluded_schedule_registry_sha256"] == REGISTRY_SHA


def test_start_uses_dedicated_attempt13_bound_runner_and_resume_guards() -> None:
    text = START.read_text(encoding="utf-8")
    for token in (
        PLAN_SHA,
        REGISTRY_SHA,
        "ofc_regular.validate_hu_m43_attempt13_acceptance",
        "ofc_regular_promotion.attempt13_population",
        '--shard-index "$SHARD"',
        "--runtime-source-archive artifacts/runtime_source.zip",
        "--development-decision artifacts/development_decision.json",
        "--audit-selector-receipt artifacts/audit50_selector_receipt.json",
        "PACKAGE_READY.json",
        "fanout requires a completed shard-0000 canary",
        '"--provisioning-model", "SPOT"',
        'unit = "completed_shard"',
        "resume_missing_shards_only = $true",
        "done_commit_last = $true",
        "current_profile_mutated = $false",
        "no_runtime_activation = $true",
    ):
        assert token in text
    assert "python -m ofc_regular.evaluate_hu_m4_population" not in text


def test_start_writes_fresh_preflight_receipt_without_rewriting_on_resume() -> None:
    text = START.read_text(encoding="utf-8")
    assert (
        '$preflightOutputPath = Join-Path $runDir "population_launch_preflight.json"'
        in text
    )
    match = re.search(
        r"\$preflightArgs\s*=\s*@\((?P<args>.*?)\r?\n\s*\)",
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    assert '"--output", $preflightOutputPath' in match.group("args")

    resume_start = text.index("if ($ResumeExisting) {")
    fresh_start = text.index("\nelse {", resume_start)
    assert "$preflightArgs" not in text[resume_start:fresh_start]


def test_attempt13_embedded_startup_bash_parses() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    match = re.search(
        r"\$startup = @'\r?\n(?P<script>.*?)\r?\n'@",
        START.read_text(encoding="utf-8"),
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


def test_status_and_receive_bind_merge_and_complete_go_without_activation() -> None:
    status = STATUS.read_text(encoding="utf-8")
    receive = RECEIVE.read_text(encoding="utf-8")
    for token in (
        "hu_m43_attempt13_population_spot_manifest_v1",
        "hu_m43_attempt13_population_spot_done_v1",
        "fixed 20x50 schedule",
        PLAN_SHA,
        REGISTRY_SHA,
    ):
        assert token in status
    for token in (
        "ofc_regular.merge_hu_m4_population_shards",
        "ofc_regular.validate_hu_m43_attempt13_acceptance",
        '"finalize"',
        "complete_content_verified",
        '"complete_go"',
        '"complete_no_go"',
        "current_profile_mutated -ne $false",
        "runtime_policy_activated -ne $false",
        "Move-Item -LiteralPath $stage -Destination $OutputDir",
        PLAN_SHA,
        REGISTRY_SHA,
    ):
        assert token in receive
