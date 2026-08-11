from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP = REPO_ROOT / "scripts/startup_hu_m31_t3_step6d_full100_v1.sh"
TAIL_STARTUP = REPO_ROOT / "scripts/startup_hu_m31_t3_step6d_v2.sh"
LIFECYCLE = REPO_ROOT / "src/ofc_regular/hu_m31_t3_step6d_full100_spot_v1.py"


def _source() -> str:
    return STARTUP.read_text(encoding="utf-8")


def test_full100_startup_is_dedicated_and_shell_syntax_is_valid() -> None:
    assert STARTUP.is_file()
    assert TAIL_STARTUP.is_file()
    assert STARTUP != TAIL_STARTUP
    source = _source()
    assert "Dedicated Candidate02 full-100 performance-development worker" in source
    assert "startup_hu_m31_t3_step6d_v2.sh" not in source
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    result = subprocess.run(
        [bash, "-n", STARTUP.relative_to(REPO_ROOT).as_posix()],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_full100_startup_inline_python_is_syntax_valid() -> None:
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", _source(), flags=re.DOTALL)
    assert len(blocks) == 5
    for index, block in enumerate(blocks):
        compile(block, f"{STARTUP.name}#heredoc-{index}", "exec")


def test_full100_startup_consumes_exact_metadata_and_gcs_layout() -> None:
    source = _source()
    for key in (
        "PROJECT_ID",
        "BUCKET",
        "RUN_NAME",
        "JOB_ID",
        "INSTANCE_NAME",
        "ZONE",
        "SOURCE_URI",
        "SOURCE_SHA256",
        "MANIFEST_URI",
        "MANIFEST_SHA256",
        "JOB_MANIFEST_URI",
        "JOB_MANIFEST_SHA256",
        "AUTHORIZATION_URI",
        "AUTHORIZATION_SHA256",
        "ATTEMPT_INDEX",
        "ATTEMPT_CLAIM_URI",
        "ATTEMPT_CLAIM_SHA256",
        "INITIAL_LAUNCH_CLAIM_SHA256",
        "INITIAL_LAUNCH_RESULT_SHA256",
        "RESULT_PREFIX",
        "PROGRESS_PREFIX",
        "COST_GUARD_SHA256",
        "MAX_RUNTIME_SECONDS",
        "SELF_DELETE",
    ):
        assert f'{key}="$(meta {key})"' in source
    assert 'FULL100_PREFIX="gs://$BUCKET/runs/$RUN_NAME/full100"' in source
    assert (
        '"$FULL100_PREFIX/source/ofc_regular_hu_m31_t3_step6d_full100_v1_source.zip"'
        in source
    )
    assert '"$FULL100_PREFIX/manifest.json"' in source
    assert '"$FULL100_PREFIX/source/jobs/$JOB_ID.json"' in source
    assert '"$FULL100_PREFIX/source/launch_authorization.json"' in source
    assert '"$FULL100_PREFIX/launch_authorization.json"' not in source
    lifecycle_source = LIFECYCLE.read_text(encoding="utf-8")
    assert (
        'f"AUTHORIZATION_URI={prefix}/source/{AUTHORIZATION_NAME}"' in lifecycle_source
    )
    assert '"$FULL100_PREFIX/results/jobs/$JOB_ID"' in source
    assert '"$FULL100_PREFIX/progress/jobs/$JOB_ID"' in source
    assert '"$FULL100_PREFIX/control/launch_claim.json"' in source
    assert '"$FULL100_PREFIX/resume/resume_claim.json"' in source
    assert 'exec >>"$LOG" 2>&1' in source
    assert 'exec > >(tee -a "$LOG") 2>&1' not in source
    assert 'trap \'startup_error "$?" "$LINENO" "$BASH_COMMAND"\' ERR' in source
    assert "full100 startup failure: exit=%s line=%s command=%s" in source


def test_full100_startup_verifies_package_plan_job_and_native_hash_chain() -> None:
    source = _source()
    assert "hu_m31_t3_step6d_full100_spot_package_v1" in source
    assert "hu_m31_t3_step6d_full100_launch_authorization_v1" in source
    assert "hu_m31_t3_step6d_candidate02_full100_plan_v1" in source
    assert "9ef14137b97db975bed683bcdd7e53b27414d2efab282c6f98390d7702944758" in source
    assert "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd" in source
    assert "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d" in source
    assert "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0" in source
    assert "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411" in source
    assert '[[ "$(sha /tmp/source.zip)" == "$SOURCE_SHA256" ]]' in source
    assert '[[ "$(sha /tmp/manifest.json)" == "$MANIFEST_SHA256" ]]' in source
    assert '[[ "$(sha /tmp/job.json)" == "$JOB_MANIFEST_SHA256" ]]' in source
    assert '[[ "$(sha /tmp/authorization.json)" == "$AUTHORIZATION_SHA256" ]]' in source
    assert '[[ "$(sha /tmp/attempt_claim.json)" == "$ATTEMPT_CLAIM_SHA256" ]]' in source
    assert 'm["startup_sha256"] == executed_startup_sha' in source
    assert "for index in range(100):" in source
    assert 'relative = f"frozen/full100_roots/hand_{index:03d}.json"' in source
    assert "set(j) == {" in source
    assert '"hu_m31_t3_step6d_performance_shard_manifest_v2"' in source


def test_full100_startup_verifies_attempt_instance_and_cost_boundaries() -> None:
    source = _source()
    assert "hu_m31_t3_step6d_full100_launch_claim_v1" in source
    assert "hu_m31_t3_step6d_full100_resume_claim_v1" in source
    assert 'claim_sha == initial_claim_sha and initial_result_sha == "none"' in source
    assert 'c["initial_launch_claim_sha256"] == initial_claim_sha' in source
    assert 'c["initial_launch_result_sha256"] == initial_result_sha' in source
    assert 'c["third_attempt_authorized"] is False' in source
    assert 'expected_instance = f"{run_name}-j{job_ids.index(job_id):02d}"' in source
    assert '("" if attempt_index == 0 else "-a01")' in source
    assert '[[ "$INSTANCE_NAME" == "$ACTUAL_INSTANCE_NAME" ]]' in source
    assert '"hu_m31_t3_step6d_full100_cost_guard_v1"' in source
    assert '"max_runtime_seconds_per_vm": 4200' in source
    assert '"internal_watchdog_seconds_per_vm": 3900' in source
    assert '"max_cumulative_vm_jobs": 40' in source
    assert '"phase_compute_cap_usd": 25.0' in source
    assert '[[ "$MAX_RUNTIME_SECONDS" == 3900 ]]' in source
    assert 'sleep "$MAX_RUNTIME_SECONDS"' in source


def test_full100_startup_runs_one_source_ten_hand_shard_with_exact_threads() -> None:
    source = _source()
    assert '[[ "$JOB_ID" =~ ^(candidate|reference)-shard-(0[0-9])$ ]]' in source
    assert '[[ "${#WORK_HANDS[@]}" -eq 10 ]]' in source
    assert "export RAYON_NUM_THREADS=16" in source
    assert "export OFC_HU_M3_BATCH_THREADS=1" in source
    assert "export OMP_NUM_THREADS=1" in source
    assert "export MKL_NUM_THREADS=1" in source
    assert "export OPENBLAS_NUM_THREADS=1" in source
    invocation = "python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2"
    assert source.count(invocation) == 2
    assert "--shard-manifest /tmp/job.json" in source
    assert '--library "$WORK/$LIBRARY_RELATIVE"' in source
    assert "--stop-after-hands 1" in source
    assert ".solve_many(" not in source
    assert "gcloud compute instances create" not in source


def test_full100_startup_preseeds_restores_and_checkpoints_each_hand() -> None:
    source = _source()
    preseed = source.index('cp "$WORK/frozen/full100_roots/hand_$hand_pad.json"')
    restore = source.index("restore_or_compare run_contract.json", preseed)
    first_run = source.index(
        "python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2",
        restore,
    )
    checkpoint = source.index('checkpoint_hand "$hand"', first_run)
    second_run = source.index(
        "python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2",
        checkpoint + 1,
    )
    done_upload = source.index(
        'upload_once "$RESULT/DONE.json" "$RESULT_PREFIX/DONE.json"',
        second_run,
    )
    assert preseed < restore < first_run < checkpoint < second_run < done_upload
    assert "restore_or_compare" in source
    assert "--if-generation-match=0" in source
    assert '"$PROGRESS_PREFIX/roots/hand_$hand_pad.json"' in source
    assert '"$PROGRESS_PREFIX/hands/$ROLE/hand_$hand_pad.json"' in source
    assert '"$RESULT_PREFIX/roots/hand_$hand_pad.json"' in source
    assert '"$RESULT_PREFIX/hands/$ROLE/hand_$hand_pad.json"' in source
    assert "a valid checkpoint is never recomputed" in source


def test_full100_startup_heartbeat_done_last_and_failure_safety() -> None:
    source = _source()
    assert '"schema": "hu_m31_t3_step6d_full100_heartbeat_v1"' in source
    assert '"work_hand_indices": work' in source
    assert '"completed_hand_indices": completed' in source
    assert "mark_present_hands_validated" in source
    assert "validated heartbeat index state changed" in source
    assert "sleep 60" in source
    assert '[[ "${#RESULT_FILES[@]}" -eq 23 ]]' in source
    assert 'find "$RESULT/roots"' in source
    assert 'find "$RESULT/hands/$ROLE"' in source
    assert (
        'upload_once "$RESULT/DONE.json"'
        not in source[source.index("cleanup() {") : source.index("trap cleanup EXIT")]
    )
    assert (
        'upload_once "$RESULT/DONE.json"'
        not in source[
            source.index("sync_failure_progress() {") : source.index("upload_log() {")
        ]
    )
    remote_verify = source.rindex(
        '[[ "$(sha "$RESULT/DONE.json")" == "$(sha "$REMOTE_DONE")" ]]'
    )
    assert remote_verify < source.rindex("VALIDATED_DONE=1")
    assert 'if ! gcloud compute instances delete "$INSTANCE_NAME"' in source
    assert "force_shutdown" in source
    for flag in (
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    ):
        assert flag in source
    assert "opponent_private_discards" not in source
