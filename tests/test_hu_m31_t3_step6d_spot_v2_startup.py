from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP = REPO_ROOT / "scripts/startup_hu_m31_t3_step6d_v2.sh"


def _source() -> str:
    return STARTUP.read_text(encoding="utf-8")


def test_step6d_v2_startup_pins_image_contract_allocation_and_tail() -> None:
    source = _source()
    assert "debian-12-bookworm-v20260609" in source
    assert "1449487925682397051" in source
    assert '"machine_type": "c4-standard-16"' in source
    assert '"process_count": 1' in source
    assert '"rayon_threads_per_process": 16' in source
    assert 'tail = contract.get("tail_hand_indices")' in source
    assert "len(tail) == 10" in source
    assert "len(set(tail)) == 10" in source
    assert "index in contract_indices" in source
    assert "tail = [2, 6, 7, 9, 13, 20, 21, 29, 33, 50]" not in source
    assert "export RAYON_NUM_THREADS=16" in source
    assert "export OFC_HU_M3_BATCH_THREADS=1" in source
    assert '"spot_price_ceiling_usd_per_vm_hour": 0.5' in source
    assert '"max_runtime_seconds_per_vm": 3300' in source
    assert '"internal_watchdog_seconds_per_vm": 3000' in source
    assert '"max_attempts_per_job": 2' in source
    assert '"max_cumulative_vm_jobs": 40' in source
    assert '"all_20_estimated_max_compute_usd": 9.166666666666666' in source
    assert '"all_attempts_estimated_max_compute_usd": 18.333333333333332' in source
    assert '"hard_tail_compute_cap_usd": 20.0' in source
    assert '"m31_total_compute_cap_usd": 500.0' in source


def test_step6d_v2_startup_verifies_package_role_work_and_binary_hashes() -> None:
    source = _source()
    assert '[[ "$(sha /tmp/source.zip)" == "$SOURCE_SHA256" ]]' in source
    assert '[[ "$(sha /tmp/manifest.json)" == "$MANIFEST_SHA256" ]]' in source
    assert '[[ "$(sha /tmp/job.json)" == "$JOB_MANIFEST_SHA256" ]]' in source
    assert '[[ "$(sha /tmp/authorization.json)" == "$AUTHORIZATION_SHA256" ]]' in source
    assert "digest(cost_guard) == cost_guard_sha" in source
    assert (
        'max_runtime_seconds == cost_guard["internal_watchdog_seconds_per_vm"]'
        in source
    )
    assert 'expected_instance = f"{run_name}-j{job_ids.index(job_id):02d}"' in source
    assert '("" if attempt_index == 0 else "-a01")' in source
    assert 'attempt_claim.get("selected_job_ids") == job_ids' in source
    assert "attempt_claim_sha == initial_claim_sha" in source
    assert "native/reference/release/libofc_hu_m3_engine.so" in source
    assert "native/candidate/release/libofc_hu_m3_engine.so" in source
    assert (
        'set(j) == {"schema", "run_contract", "run_contract_digest", "source_role", "work_hand_indices"}'
        in source
    )
    assert 'j.get("run_contract_digest") == digest(j.get("run_contract"))' in source
    assert (
        'contract.get("candidate_library_sha256") == candidate.get("sha256")' in source
    )
    assert 'contract.get("reference_library_sha256") == reference["sha256"]' in source
    assert (
        'selection_relative = "configs/hu_joint_policy_m31_t3_candidate02_tail_v2_selection.json"'
        in source
    )
    assert "62cfbe2d95477ed7686a1b583997ee2e35ab23291fd338cfeac597e9a216fed1" in source
    assert 'selection_entry == {"sha256": selection_sha, "bytes": 1523}' in source
    assert "for relative, expected in entries.items()" in source


def test_step6d_v2_startup_restores_and_revalidates_before_done_commit() -> None:
    source = _source()
    restore_progress = source.index('restore_checkpoint "$relative"')
    restore_results = source.index(
        'gcloud storage rsync --recursive "$RESULT_URI" "$RESULT"'
    )
    first_run = source.index(
        "python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2",
        restore_results,
    )
    second_run = source.index(
        "python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2",
        first_run + 1,
    )
    checkpoint = source.index("for relative in", second_run)
    done = source.index('upload_once "$RESULT/DONE.json"', checkpoint)
    assert (
        restore_progress < restore_results < first_run < second_run < checkpoint < done
    )
    assert "--shard-manifest /tmp/job.json" in source
    assert '--library "$WORK/$LIBRARY_RELATIVE"' in source
    assert "--source-library" not in source
    assert "--source-sha256" not in source


def test_step6d_v2_startup_heartbeats_checkpoints_each_hand_and_deletes_last() -> None:
    source = _source()
    assert 'PROGRESS_URI="$PREFIX/progress/jobs/$JOB_ID"' in source
    assert '"schema": "hu_m31_t3_step6d_spot_heartbeat_v2"' in source
    assert '"attempt_index": int(sys.argv[7])' in source
    assert "sleep 60" in source
    assert '"roots/hand_$HAND_PAD.json"' in source
    assert '"hands/$ROLE/hand_$HAND_PAD.json"' in source
    assert "--if-generation-match=0" in source
    assert "VALIDATED_DONE=1" in source
    validation = source.rindex(
        '[[ "$(sha "$RESULT/DONE.json")" == "$(sha "$REMOTE_DONE")" ]]'
    )
    assert validation < source.rindex("VALIDATED_DONE=1")
    assert 'if ! gcloud compute instances delete "$INSTANCE_NAME"' in source


def test_step6d_v2_startup_ignores_prior_progress_startup_log_on_resume() -> None:
    source = _source()

    assert 'gcloud storage rsync --recursive "$PROGRESS_URI" "$RESULT"' not in source
    assert (
        'LOG_URI="$PREFIX/logs/jobs/$JOB_ID/attempt-$ATTEMPT_INDEX/startup.log"'
        in source
    )
    assert '"$PROGRESS_URI/startup.log"' not in source
    restore = source[
        source.index("for relative in", source.index("restore_checkpoint()")) :
    ]
    assert "run_contract.json" in restore
    assert "shard_manifest.json" in restore
    assert '"roots/hand_$HAND_PAD.json"' in restore
    assert '"hands/$ROLE/hand_$HAND_PAD.json"' in restore
    assert "startup.log" not in restore.split("gcloud storage rsync", 1)[0]


def test_step6d_v2_startup_watchdog_and_delete_failure_force_poweroff() -> None:
    source = _source()
    watchdog_start = source.index("start_watchdog\n")
    apt_start = source.index("sudo apt-get")
    cleanup = source[source.index("cleanup() {") : source.index("trap cleanup EXIT")]
    shutdown = source[source.index("force_shutdown() {") : source.index("cleanup() {")]

    assert watchdog_start < apt_start
    assert 'sleep "$MAX_RUNTIME_SECONDS"' in source
    assert 'kill -TERM "$MAIN_PID"' in source
    assert "stop_watchdog" in cleanup
    assert 'if [[ "$code" -ne 0 || "$VALIDATED_DONE" -ne 1 ]]' in cleanup
    assert "sync_failure_state" in cleanup
    assert "upload_startup_log" in cleanup
    assert "force_shutdown" in cleanup
    assert 'if ! gcloud compute instances delete "$INSTANCE_NAME"' in cleanup
    assert "self-delete failed; forcing shutdown" in cleanup
    assert "exit 1" in cleanup
    assert "sudo shutdown -h now" in shutdown
    assert "sudo systemctl poweroff --force --force" in shutdown
    assert "sudo poweroff -f" in shutdown
    assert 'upload_once "$RESULT/DONE.json"' not in cleanup
