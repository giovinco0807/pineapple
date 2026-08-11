from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_step6d_v2_start_script_requires_explicit_separate_lifecycle_modes() -> None:
    source = _read("Start-GcpHuM31T3Step6dV2SpotRun.ps1")
    assert "[switch]$PackageOnly" in source
    assert "[switch]$AuthorizeOnly" in source
    assert "[switch]$Launch" in source
    assert "[switch]$Resume" in source
    assert "Choose exactly one" in source
    assert "Initial Launch requires exactly -Jobs all" in source
    assert "Resume requires the exact complete set" in source
    assert '"package"' in source
    assert '"authorize"' in source
    assert '"launch"' in source
    assert '"resume"' in source
    assert '"--jobs", ($Jobs -join ",")' in source
    assert "NoSelfDelete" not in source
    assert (
        "Launch/resume target differs from the fixed Step 6d v2 authorization" in source
    )


def test_step6d_v2_start_script_never_silently_changes_scientific_scope() -> None:
    source = _read("Start-GcpHuM31T3Step6dV2SpotRun.ps1")
    assert "CandidateLibrary" in source
    assert "CandidateSha256" in source
    assert "ContractVariant" in source
    assert "candidate02_compact_scorer" in source
    assert "candidate02_compact_scorer_tail_v2" in source
    assert '"--contract-variant", $ContractVariant' in source
    assert "ReferenceSha256" in source
    assert "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0" in source
    assert "c4-standard" not in source  # allocation is not caller-selectable
    assert "MachineType" not in source
    assert "current" not in source.casefold()


def test_step6d_v2_start_script_single_mode_is_a_real_array() -> None:
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-File",
            str(REPO_ROOT / "scripts" / "Start-GcpHuM31T3Step6dV2SpotRun.ps1"),
            "-RunName",
            "step6d-v2-wrapper-smoke",
            "-PackageOnly",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "PackageOnly requires -CandidateLibrary" in combined
    assert "property 'Count'" not in combined


def test_step6d_v2_get_and_receive_scripts_call_only_the_v2_lifecycle() -> None:
    status = _read("Get-GcpHuM31T3Step6dV2SpotRunStatus.ps1")
    receive = _read("Receive-GcpHuM31T3Step6dV2SpotRun.ps1")
    assert '"ofc_regular.hu_m31_t3_step6d_spot_v2", "status"' in status
    assert '"ofc_regular.hu_m31_t3_step6d_spot_v2", "receive"' in receive
    assert "outputs/hu_joint_policy/m31_t3_step6d/spot_v2" in receive
    assert "Start-" not in status
    assert "Start-" not in receive


def test_step6d_v2_linux_package_smoke_loads_both_release_engines() -> None:
    source = _read("smoke_hu_m31_t3_step6d_package.py")
    assert '("candidate", args.candidate_sha256)' in source
    assert '("reference", args.reference_sha256)' in source
    assert '"release" / "libofc_hu_m3_engine.so"' in source
    assert "HuM31T3SearchSolver" in source
    assert "ctypes.CDLL" in source
