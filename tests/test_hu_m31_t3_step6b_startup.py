from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP_PATH = REPO_ROOT / "scripts/startup_hu_m31_t3_step6b.sh"


def _startup() -> str:
    return STARTUP_PATH.read_text(encoding="utf-8")


def test_step6b_startup_strictly_limits_dynamic_shard_to_one_through_nine():
    startup = _startup()
    assert '[[ ! "$SHARD" =~ ^[1-9]$ ]]' in startup
    assert "SHARD_NUM=$((10#$SHARD))" in startup
    assert "printf -v SHARD_PAD '%03d'" in startup
    assert "integer shards 1 through 9 only" in startup
    assert 'SHARD" != 0' not in startup


def test_step6b_startup_uses_padded_per_shard_remote_paths():
    startup = _startup()
    assert 'PROGRESS_URI="$PREFIX/progress/shard-$SHARD_PAD"' in startup
    assert 'RESULT_URI="$PREFIX/results/shard-$SHARD_PAD"' in startup
    assert '"$PREFIX/logs/shard-$SHARD_PAD-startup.log"' in startup
    assert "progress/shard-000" not in startup
    assert "results/shard-000" not in startup


def test_step6b_startup_enforces_exact_manifest_authorization_membership():
    startup = _startup()
    assert "authorized = list(range(1, 10))" in startup
    assert 'm.get("authorized_shards") == authorized' in startup
    assert 'a.get("authorized_shards") == authorized' in startup
    assert 'a.get("remaining_canary_shards_authorized") is True' in startup
    assert 'a.get("production_fanout_authorized") is False' in startup
    assert "schedule_shards.count(value) == 1 for value in authorized" in startup
    assert 'row.get("global_hand_start") == shard * 25' in startup


def test_step6b_startup_binds_required_runner_and_schemas():
    startup = _startup()
    assert "ofc_regular.run_hu_m31_t3_step6b_shard" in startup
    assert "hu_m31_t3_step6b_spot_package_v1" in startup
    assert "hu_m31_t3_step6b_linux_parity_v1" in startup
    assert "hu_m31_t3_step6b_summary_v1" in startup
    assert "hu_m31_t3_step6b_done_v1" in startup
    assert '"$WORK/artifacts/step6b/parity_golden.json"' in startup


def test_step6b_startup_pins_environment_and_verifies_all_package_hashes():
    startup = _startup()
    assert "snapshot.debian.org/archive/debian/20260609T000000Z" in startup
    assert '[[ "$(sha /tmp/source.zip)" == "$SOURCE_SHA256" ]]' in startup
    assert '[[ "$(sha /tmp/manifest.json)" == "$MANIFEST_SHA256" ]]' in startup
    assert '[[ "$(sha /tmp/shards.jsonl)" == "$SCHEDULE_SHA256" ]]' in startup
    assert (
        '[[ "$(sha /tmp/authorization.json)" == "$AUTHORIZATION_SHA256" ]]' in startup
    )
    assert "for relative, expected in entries.items()" in startup
    assert 'hashlib.sha256(data).hexdigest() == expected.get("sha256")' in startup
    assert "importlib.metadata.version(name)" in startup
    assert "python -m pip check" in startup


def test_step6b_startup_runs_parity_then_remote_wipe_recovery_then_full_run():
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
    assert "start_pump" in startup and "stop_pump" in startup


def test_step6b_startup_uploads_done_last_and_write_once():
    startup = _startup()
    ordinary_upload = startup.index(
        'done < <(find "$RESULT" -type f ! -name DONE.json -print0)'
    )
    done_upload = startup.index(
        'upload_once "$RESULT/DONE.json" "$RESULT_URI/DONE.json"'
    )
    assert ordinary_upload < done_upload
    assert "--if-generation-match=0" in startup
    assert 'objects describe "$RESULT_URI/DONE.json"' in startup


def test_step6b_startup_always_attempts_self_delete_without_enabling_fanout():
    startup = _startup()
    assert 'if [[ "$SELF_DELETE" == 1 ]]' in startup
    assert 'gcloud compute instances delete "$INSTANCE_NAME"' in startup
    assert '"production_fanout_authorized": False' in startup
    assert '"training_eligible": False' in startup
