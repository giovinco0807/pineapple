from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "Start-GcpHuM31T3Step6dFull100Run.ps1",
    "Get-GcpHuM31T3Step6dFull100RunStatus.ps1",
    "Receive-GcpHuM31T3Step6dFull100Run.ps1",
)
MERGE_SCRIPT = "Merge-GcpHuM31T3Step6dFull100Run.ps1"


def test_full100_powershell_wrappers_parse_and_remain_bounded() -> None:
    for name in (*SCRIPTS, MERGE_SCRIPT):
        path = REPO_ROOT / "scripts" / name
        command = (
            "$errors=$null;"
            f"[System.Management.Automation.Language.Parser]::ParseFile('{path}',"
            "[ref]$null,[ref]$errors) > $null;"
            "if($errors.Count -ne 0){$errors|ForEach-Object{$_.Message};exit 1}"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}\n{result.stderr}"

    start = (REPO_ROOT / "scripts" / SCRIPTS[0]).read_text(encoding="utf-8")
    assert "PackageOnly" in start
    assert "AuthorizeOnly" in start
    assert "Launch" in start
    assert "Resume" in start
    assert "Only full100 Resume accepts -Jobs" in start
    assert "Full100 Resume requires the exact incomplete job set" in start
    assert "ofc_regular.hu_m31_t3_step6d_full100_spot_v1" in start
    assert "set_current" not in start

    merge = (REPO_ROOT / "scripts" / MERGE_SCRIPT).read_text(encoding="utf-8")
    assert "merge_hu_m31_t3_step6d_full100_received_v1" in merge
    assert "--expected-run-name" in merge
    assert "summary.json" in merge
    assert "validation.json" in merge
    assert "set_current" not in merge
    assert "gcloud" not in merge.casefold()


def test_full100_wrappers_pin_project_bucket_and_do_not_touch_tail_scripts() -> None:
    for name in SCRIPTS:
        source = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "ofc-solver-485418" in source
        assert "pokerhu-ofc-solver-485418-training" in source
        assert "current profile" not in source.casefold()
