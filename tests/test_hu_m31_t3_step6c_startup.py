from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP_PATH = REPO_ROOT / "scripts/startup_hu_m31_t3_step6c.sh"


def _startup() -> str:
    return STARTUP_PATH.read_text(encoding="utf-8")


def test_step6c_startup_strictly_limits_dynamic_shard_to_zero_or_one() -> None:
    startup = _startup()
    assert '[[ ! "$SHARD" =~ ^[01]$ ]]' in startup
    assert "SHARD_NUM=$((10#$SHARD))" in startup
    assert "printf -v SHARD_PAD '%03d'" in startup
    assert "integer shards 0 and 1 only" in startup


def test_step6c_startup_uses_padded_per_shard_remote_paths() -> None:
    startup = _startup()
    assert 'PROGRESS_URI="$PREFIX/progress/shard-$SHARD_PAD"' in startup
    assert 'RESULT_URI="$PREFIX/results/shard-$SHARD_PAD"' in startup
    assert '"$PREFIX/logs/shard-$SHARD_PAD-startup.log"' in startup


def test_step6c_startup_enforces_exact_pilot_authorization_and_budgets() -> None:
    startup = _startup()
    assert "authorized = [0, 1]" in startup
    assert 'm.get("authorized_shards") == authorized' in startup
    assert 'a.get("authorized_shards") == authorized' in startup
    assert 'a.get("quality_pilot_only") is True' in startup
    assert 'm.get("quality_pilot_authorized") is True' in startup
    assert 'a.get("production_fanout_authorized") is False' in startup
    assert 'a.get("training_eligible") is False' in startup
    assert 'm.get("production_label_budget") == {' in startup
    assert 'm.get("confirmation_budget") == {' in startup
    assert 'a.get("step5_contract_canonical_sha256") == m.get(' in startup
    assert 'a.get("step6b_validation_sha256") == m.get(' in startup
    assert 'a.get("step6c_contract_canonical_sha256") == m.get(' in startup


def test_step6c_startup_binds_accepted_binary_and_package_hashes() -> None:
    startup = _startup()
    assert "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0" in startup
    assert "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411" in startup
    assert '[[ "$(sha /tmp/source.zip)" == "$SOURCE_SHA256" ]]' in startup
    assert '[[ "$(sha /tmp/manifest.json)" == "$MANIFEST_SHA256" ]]' in startup
    assert '[[ "$(sha /tmp/shards.jsonl)" == "$SCHEDULE_SHA256" ]]' in startup
    assert (
        '[[ "$(sha /tmp/authorization.json)" == "$AUTHORIZATION_SHA256" ]]' in startup
    )
    assert "for relative, expected in entries.items()" in startup
    assert 'hashlib.sha256(data).hexdigest() == expected.get("sha256")' in startup


def test_step6c_startup_pins_environment_and_runs_production_parity() -> None:
    startup = _startup()
    assert "snapshot.debian.org/archive/debian/20260609T000000Z" in startup
    assert "python -m pip check" in startup
    assert "importlib.metadata.version(name)" in startup
    assert "ofc_regular.run_hu_m31_t3_step6c_shard" in startup
    assert '"$WORK/artifacts/step6c/parity_golden.json"' in startup
    assert "hu_m31_t3_step6c_linux_parity_v1" in startup
    assert "production-budget Linux parity failed" in startup


def test_step6c_startup_runs_parity_then_remote_wipe_recovery_then_full_run() -> None:
    startup = _startup()
    parity = startup.index('"${COMMON_ARGS[@]}" --parity-only')
    drill = startup.index('"${COMMON_ARGS[@]}" --stop-after-tasks 1')
    wipe = startup.index('rm -rf "$RESULT"', drill)
    recover = startup.index(
        'gcloud storage rsync --recursive "$PROGRESS_URI" "$RESULT"', wipe
    )
    full = startup.index('/usr/bin/time -v -o "$RESULT/time.txt"', recover)
    assert parity < drill < wipe < recover < full
    assert '[[ "$drill_code" -ne 75 ]]' in startup
    assert '[[ "$RECOVERED" -lt 1 ]]' in startup


def test_step6c_startup_commits_valid_no_go_and_uploads_done_last() -> None:
    startup = _startup()
    assert 'summary.get("status") in {"pass", "no_go"}' in startup
    assert '"quality_status": summary["status"]' in startup
    ordinary = startup.index(
        'upload_once "$RESULT/parity.json" "$RESULT_URI/parity.json"'
    )
    done = startup.index('upload_once "$RESULT/DONE.json" "$RESULT_URI/DONE.json"')
    assert ordinary < done
    assert 'find "$RESULT" -type f ! -name DONE.json' not in startup
    assert "--if-generation-match=0" in startup
    assert 'objects describe "$RESULT_URI/DONE.json"' in startup


def test_step6c_startup_done_and_upload_use_only_scientific_allowlist() -> None:
    startup = _startup()
    done_builder = startup[
        startup.index("files = {}") : startup.index("upload_once() {")
    ]
    upload = startup[startup.index("upload_once() {") :]
    assert '"parity.json"' in done_builder
    assert '"summary.json"' in done_builder
    assert 'f"roots/hand_{index:03d}.json"' in done_builder
    assert 'f"tasks/hand_{index:03d}.json"' in done_builder
    assert 'r.rglob("*")' not in done_builder
    assert "for directory in roots tasks" in upload
    assert 'relative="$directory/hand_$hand_pad.json"' in upload
    for mutable in ("heartbeat.json", "run.log", "time.txt", "runner_stdout.json"):
        assert mutable not in done_builder
        assert mutable not in upload


def test_step6c_partial_upload_retry_requires_identical_scientific_hash() -> None:
    startup = _startup()
    upload_once = startup[
        startup.index("upload_once() {") : startup.index(
            'upload_once "$RESULT/parity.json"'
        )
    ]
    assert "--if-generation-match=0" in upload_once
    assert 'gcloud storage cp "$target" "$existing"' in upload_once
    assert '[[ "$(sha "$source")" == "$(sha "$existing")" ]]' in upload_once
    assert 'find "$RESULT"' not in startup[startup.index("upload_once() {") :]


def test_step6c_partial_results_are_restored_after_progress_and_revalidated() -> None:
    startup = _startup()
    progress = startup.index(
        'gcloud storage rsync --recursive "$PROGRESS_URI" "$RESULT"'
    )
    partial_results = startup.index(
        'gcloud storage rsync --recursive "$RESULT_URI" "$RESULT"', progress
    )
    pump = startup.index("start_pump", partial_results)
    assert progress < partial_results < pump
    assert (
        '--project "$PROJECT_ID" >/dev/null 2>&1 || true'
        in startup[partial_results:pump]
    )


def test_step6c_summary_upload_follows_complete_root_and_task_uploads() -> None:
    startup = _startup()
    parity = startup.index(
        'upload_once "$RESULT/parity.json" "$RESULT_URI/parity.json"'
    )
    scientific_loop = startup.index("for directory in roots tasks; do", parity)
    loop_end = startup.index("done\n# Summary is published", scientific_loop)
    summary = startup.index(
        'upload_once "$RESULT/summary.json" "$RESULT_URI/summary.json"', loop_end
    )
    done = startup.index(
        'upload_once "$RESULT/DONE.json" "$RESULT_URI/DONE.json"', summary
    )
    assert parity < scientific_loop < loop_end < summary < done


def test_step6c_startup_arms_cleanup_before_first_metadata_read() -> None:
    startup = _startup()
    trap = startup.index("trap cleanup EXIT")
    first_metadata_read = startup.index('PROJECT_ID="$(meta PROJECT_ID)"')
    assert trap < first_metadata_read


def test_step6c_startup_always_self_deletes_without_activation() -> None:
    startup = _startup()
    assert 'if [[ "$SELF_DELETE" == 1 ]]' in startup
    assert 'gcloud compute instances delete "$INSTANCE_NAME"' in startup
    assert '"production_fanout_authorized": False' in startup
    assert '"training_eligible": False' in startup
    assert '"current_profile_changed": False' in startup
    assert '"named_profile_added": False' in startup
