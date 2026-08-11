from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts/Start-GcpHuM43Attempt08DevelopmentRun.ps1"
STATUS = ROOT / "scripts/Get-GcpHuM43Attempt08DevelopmentRunStatus.ps1"
RECEIVE = ROOT / "scripts/Receive-GcpHuM43Attempt08DevelopmentRun.ps1"
STARTUP = ROOT / "scripts/startup_hu_m43_attempt08_development.sh"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_development_scripts_parse() -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:  # pragma: no cover - Windows is the operational host
        pytest.skip("PowerShell is unavailable")
    for path in (START, STATUS, RECEIVE):
        escaped = str(path).replace("'", "''")
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f"[void][scriptblock]::Create([IO.File]::ReadAllText('{escaped}'))",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover
        pytest.skip("bash is unavailable")
    result = subprocess.run(
        [bash, "-n", "-"],
        check=False,
        capture_output=True,
        input=STARTUP.read_bytes().replace(b"\r\n", b"\n"),
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_start_is_three_phase_frozen_source_and_bounded_spot_only() -> None:
    text = _text(START)
    assert "$PackageOnly, $AuthorizeLaunch, $CreateInstances" in text
    assert "Choose exactly one Attempt08 phase" in text
    assert "Join-Path $RunDir 'package_src'" in text
    assert "Start-GcpHuM43Attempt08DevelopmentRun.ps1'" in text
    assert "HuM43Attempt08Spot.Common.ps1'" in text
    assert "HuM43Attempt04Spot.Common.ps1'" in text
    assert "differs from its frozen packaged copy" in text
    assert "c4-highmem-4" in text
    assert "-MaxCount 25" in text
    assert "--provisioning-model','SPOT'" in text
    assert "--instance-termination-action','DELETE'" in text
    assert "--maintenance-policy','TERMINATE'" in text
    assert "--no-restart-on-failure" in text
    assert "--scopes','cloud-platform'" in text
    assert "'--image',$ImageName" in text
    assert "--image-family" not in text
    assert "SkipExisting" not in text
    assert "ResumeExisting" not in text
    assert "selected_wave_created'" in text
    assert "DONE already exists" in text


def test_startup_preserves_exact_package_closure_and_reopens_remote_done() -> None:
    text = _text(STARTUP)
    assert text.index("export PYTHONDONTWRITEBYTECODE=1") < text.index(
        "python -m ofc_regular.hu_m43_attempt08_spot validate-launch"
    )
    assert 'VENV="$RUNMETA/.venv"' in text
    assert 'python3 -m venv "$VENV"' in text
    assert "python3 -m venv .venv" not in text
    assert "validate_runtime_semantic_anchor" in text
    assert "validate_expected_runtime_fingerprint" in text
    assert "validate-completed-bundle" in text
    assert "unable to prove remote object state" in text
    assert "timeout --signal=TERM --kill-after=60s 3600s /usr/bin/time -v" in text
    assert text.index("global-claim") < text.index("run_hu_m43_attempt08_development")
    assert text.index("root-claim") < text.index("run_hu_m43_attempt08_development")


def test_startup_resume_commit_is_content_addressed_and_precedes_final_done() -> None:
    text = _text(STARTUP)
    committed_read = 'download_if_exists "$RESUME_URI/resume_commit.json"'
    object_publish = '"$RESUME_URI/objects/$digest/$name"'
    commit_publish = (
        'upload_once_or_verify "$RESULT/resume_commit.json" '
        '"$RESUME_URI/resume_commit.json"'
    )
    final_support = (
        "for name in teacher.jsonl checkpoint.json heartbeat.json "
        "generator_summary.json run.log boot_image_evidence.json time.txt "
        "resume_commit.json global_claim.json root_claim.json; do"
    )
    final_done = 'upload_once_or_verify "$RESULT/DONE.json" "$RESULT_URI/DONE.json"'
    assert committed_read in text
    assert object_publish in text
    assert commit_publish in text
    assert final_support in text
    assert final_done in text
    assert text.index(object_publish) < text.index(commit_publish)
    assert text.index(commit_publish) < text.index(final_support)
    assert text.index(final_support) < text.index(final_done)
    for field in (
        "output_sha256",
        "checkpoint_sha256",
        "heartbeat_sha256",
        "generator_summary_sha256",
        "run_log_sha256",
        "boot_image_evidence_sha256",
        "time_report_sha256",
        "global_claim_sha256",
        "root_claim_sha256",
    ):
        assert field in text


def test_startup_discards_every_uncommitted_incomplete_completion_cut() -> None:
    text = _text(STARTUP)
    assert "UNCOMMITTED_COMPLETE=0" in text
    assert "required=(checkpoint.json heartbeat.json generator_summary.json time.txt)" in text
    assert 'if [[ "$UNCOMMITTED_COMPLETE" == 0 ]]; then' in text
    cleanup = text[text.index("# No immutable COMMIT exists") :]
    for name in (
        "teacher.jsonl",
        "teacher.jsonl.partial",
        "checkpoint.json",
        "heartbeat.json",
        "generator_summary.json",
        "time.txt",
    ):
        assert f'"$RESULT/{name}"' in cleanup
    assert "partial_pair_valid=0" in text
    assert "hu_m43_attempt08_development_checkpoint_v1" in text
    assert "c['partial_sha256']==hashlib.sha256(raw).hexdigest()" in text


def test_status_is_done_only_and_uses_frozen_lifecycle_code() -> None:
    text = _text(STATUS)
    assert "Join-Path $RunDir 'package_src'" in text
    assert "differs from its frozen packaged copy" in text
    assert '"gs://$Bucket/runs/$RunName/results/$($spec.output_prefix)/DONE.json"' in text
    assert "validate-done-set" in text
    assert "result_content_opened = $false" in text
    assert "teacher.jsonl" not in text
    assert "checkpoint.json" not in text
    assert "resume_commit.json" not in text


def test_receive_claims_exact_done_set_before_any_content_addressing() -> None:
    text = _text(RECEIVE)
    done_download = '"$prefix/results/$($spec.output_prefix)/DONE.json"'
    local_claim = "'claim'"
    remote_publish = "Publish-M43A8ImmutableObject -Source $localClaim"
    claim_reopen = "'validate-claims'"
    content_names = "$contentNames = @("
    done_index = text.index(done_download)
    claim_index = text.index(local_claim, done_index)
    publish_index = text.index(remote_publish, claim_index)
    reopen_index = text.index(claim_reopen, publish_index)
    assert done_index < claim_index < publish_index < reopen_index
    assert reopen_index < text.index(content_names)
    for name in (
        "teacher.jsonl",
        "checkpoint.json",
        "heartbeat.json",
        "generator_summary.json",
        "run.log",
        "boot_image_evidence.json",
        "time.txt",
        "resume_commit.json",
        "global_claim.json",
        "root_claim.json",
    ):
        assert f"'{name}'" in text
    assert "audit-received" in text
    assert "merge-received" in text


def test_receive_uses_frozen_code_and_fail_closed_selector_resume() -> None:
    text = _text(RECEIVE)
    assert "Join-Path $RunDir 'package_src'" in text
    assert "differs from its frozen packaged copy" in text
    assert "[switch]$ResumeSelector" in text
    assert "validate-selector-completion" in text
    assert "validate-merged-receive" in text
    assert "automatic gate reevaluation is forbidden" in text
    assert "development_selector.json" in text
    assert "select-once" in text
    assert "gate_evaluation_count -ne 1" in text
    assert "current_profile_mutated = $false" in text
    assert "runtime_policy_activated = $false" in text
