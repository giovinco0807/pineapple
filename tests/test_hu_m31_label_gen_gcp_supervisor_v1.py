from __future__ import annotations

import hashlib
import json
import datetime as dt
from pathlib import Path

import pytest

from ofc_regular.hu_m31_label_gen_gcp_supervisor_v1 import (
    DEFAULT_BOOT_GRACE_SECONDS,
    DEFAULT_HEARTBEAT_STALE_SECONDS,
    ShardState,
    generation_is_durable,
    inspect_shards,
    load_run_plan,
    supervise_once,
    stale_incomplete_states,
    validate_complete_checkpoint,
    validate_done_marker,
    validate_m7_t2_2048_run_contract,
)
from ofc_regular.hu_m31_label_gen_resume_v1 import (
    build_complete_checkpoint,
    complete_checkpoint_object_name,
    restore_cached_position_object,
)
from ofc_regular.hu_m31_label_gen_worker_v1 import DONE_SCHEMA, canonical_bytes


ROOT = Path(__file__).resolve().parents[1]


def _plan() -> dict:
    startup = ROOT / "scripts/startup_hu_m31_label_gen_v1.sh"
    bindings = [{
        "relative": "static/runtime_archive/runtime.tar.gz",
        "object": "labelgen/test/staging/runtime.tar.gz",
        "generation": "TO_BE_BOUND_AT_STAGE",
        "sha256": "a" * 64,
        "bytes": 123,
    }]
    shards = []
    for index in range(3):
        shard_id = f"{index:03d}"
        shards.append({
            "shard_id": shard_id,
            "start": index * 10,
            "count": 10,
            "object_prefix": f"labelgen/test/shards/{shard_id}",
            "metadata_values": {
                "lg-bucket": "bucket",
                "lg-shard-id": shard_id,
                "lg-object-prefix": f"labelgen/test/shards/{shard_id}",
                "lg-plan-sha256": "b" * 64,
                "lg-watchdog-seconds": "21600",
                "lg-worker-count": "6",
            },
        })
    plan = {
        "schema": "hu_m31_label_gen_run_plan_v1",
        "run_name": "test",
        "bucket": "bucket",
        "machine_type": "c4-standard-8",
        "zone": "asia-northeast1-b",
        "region": "asia-northeast1",
        "network": "default",
        "subnetwork": "default",
        "worker_plan_sha256": "b" * 64,
        "worker_plan_job_id": "test",
        "samples": 2048,
        "startup_script_relative": "scripts/startup_hu_m31_label_gen_v1.sh",
        "startup_script_sha256": hashlib.sha256(startup.read_bytes()).hexdigest(),
        "content_bindings_without_generations": bindings,
        "shards": shards,
        "image_requirement": {},
        "notes": [],
    }
    plan["plan_receipt_sha256"] = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    return plan


def _contract_plan() -> dict:
    plan = _plan()
    plan.update({
        "run_name": "m7v5-t2second-25k-2048p-r2",
        "worker_plan_job_id": "m7v5-t2second-25k-2048p",
        "worker_plan_sha256": (
            "8800ab83b976e117c6a5652a7a8d58fa4dd93b5ed104da05c5b1e59bb7f5573b"
        ),
        "bucket": "pokerhu-ofc-solver-485418-training",
        "samples": 2048,
        "machine_type": "c4-standard-8",
        "zone": "asia-northeast1-b",
        "region": "asia-northeast1",
    })
    plan["content_bindings_without_generations"] = [
        {
            "relative": relative,
            "object": f"labelgen/{plan['run_name']}/staging/{name}",
            "generation": "TO_BE_BOUND_AT_STAGE",
            "sha256": char * 64,
            "bytes": 1,
        }
        for relative, name, char in (
            ("static/runtime_archive/runtime.tar.gz", "runtime.tar.gz", "a"),
            ("static/wheelhouse_archive/wheelhouse.zip", "wheelhouse.zip", "b"),
            ("static/plan/plan.json", "plan.json", "c"),
        )
    ]
    counts = [143] * 56 + [144] * 118
    shards = []
    start = 0
    for index, count in enumerate(counts):
        shard_id = f"{index:03d}"
        prefix = f"labelgen/{plan['run_name']}/shards/{shard_id}"
        shards.append({
            "shard_id": shard_id,
            "start": start,
            "count": count,
            "object_prefix": prefix,
            "metadata_values": {
                "lg-bucket": plan["bucket"],
                "lg-shard-id": shard_id,
                "lg-object-prefix": prefix,
                "lg-plan-sha256": plan["worker_plan_sha256"],
                "lg-watchdog-seconds": "21600",
                "lg-worker-count": "6",
            },
        })
        start += count
    plan["shards"] = shards
    plan.pop("plan_receipt_sha256", None)
    plan["plan_receipt_sha256"] = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    return plan


def _stage(plan: dict) -> dict:
    rows = []
    for row in plan["content_bindings_without_generations"]:
        bound = dict(row)
        bound["generation"] = "12345"
        rows.append(bound)
    return {
        "schema": "hu_m31_label_gen_stage_receipt_v1",
        "run_name": plan["run_name"],
        "bucket": plan["bucket"],
        "content_bindings": rows,
    }


def _done(plan: dict, shard: dict) -> bytes:
    return canonical_bytes({
        "schema": DONE_SCHEMA,
        "plan_sha256": plan["worker_plan_sha256"],
        "shard_id": shard["shard_id"],
        "positions": shard["count"],
    })


def _checkpoint(plan: dict, shard: dict) -> bytes:
    rows = []
    relatives = ["SHARD_DONE.json"] + [
        f"position_{offset:08d}.json"
        for offset in range(shard["start"], shard["start"] + shard["count"])
    ]
    for relative in relatives:
        rows.append({
            "relative_path": relative,
            "object_name": f"{shard['object_prefix']}/files/{relative}",
            "sha256": "d" * 64,
            "bytes": 1,
        })
    payload = build_complete_checkpoint(
        plan_sha256=plan["worker_plan_sha256"],
        shard_id=shard["shard_id"],
        attempt_id="attempt",
        shard_start=shard["start"],
        shard_count=shard["count"],
        files=rows,
    )
    return canonical_bytes(payload)


def _set_complete(adapter, plan: dict, shard: dict) -> None:
    adapter.objects[f"{shard['object_prefix']}/files/SHARD_DONE.json"] = _done(
        plan, shard
    )
    adapter.objects[complete_checkpoint_object_name(shard["object_prefix"])] = (
        _checkpoint(plan, shard)
    )


def _instance(attempt_id: str = "attempt", *, created: str = "2099-08-07T00:00:00Z"):
    return {
        "status": "RUNNING",
        "creationTimestamp": created,
        "metadata": {"items": [{"key": "lg-attempt-id", "value": attempt_id}]},
    }


def _heartbeat(plan: dict, shard: dict, attempt_id: str, sequence: int) -> bytes:
    payload = {
        "schema": "hu_m31_label_gen_heartbeat_v1",
        "plan_sha256": plan["worker_plan_sha256"],
        "shard_id": shard["shard_id"],
        "attempt_id": attempt_id,
        "sequence": sequence,
        "completed_position_count": 0,
        "create_only": True,
        "observed_at_utc": "2020-01-01T00:01:00Z",
    }
    payload["heartbeat_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return canonical_bytes(payload)


class FakeAdapter:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.instances: dict[str, dict] = {}
        self.created: list[tuple[dict, str]] = []
        self.deleted: list[str] = []
        self.listings: dict[str, list[dict]] = {}

    def get_object_bytes(self, *, bucket, object_name, generation=None):
        return self.objects.get(object_name)

    def get_instance(self, *, instance_name):
        return self.instances.get(instance_name)

    def list_prefix(self, *, bucket, prefix):
        return self.listings.get(prefix, [])

    def create_instance(self, *, instance_spec, request_id):
        self.created.append((dict(instance_spec), request_id))
        self.instances[instance_spec["name"]] = {
            "status": "PROVISIONING",
            "creationTimestamp": "2026-08-07T00:00:00Z",
            "metadata": instance_spec["metadata"],
        }
        return {"name": f"operation-{len(self.created)}"}

    def delete_instance(self, *, instance_name, request_id):
        self.deleted.append(instance_name)
        self.instances.pop(instance_name, None)
        return {"name": "delete-operation"}


def test_run_plan_digest_is_fail_closed(tmp_path: Path):
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert load_run_plan(path)["run_name"] == "test"
    plan["samples"] = 512
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(SystemExit, match="digest mismatch"):
        load_run_plan(path)


def test_exact_m7_cloud_contract_and_cap():
    plan = _contract_plan()
    validate_m7_t2_2048_run_contract(
        plan,
        image=(
            "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
            "global/images/debian-12-bookworm-v20260804"
        ),
        max_live=58,
        max_run_seconds=25_200,
    )
    with pytest.raises(SystemExit, match="max_live"):
        validate_m7_t2_2048_run_contract(
            plan,
            image=(
                "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
                "global/images/debian-12-bookworm-v20260804"
            ),
            max_live=59,
            max_run_seconds=25_200,
        )


def test_m7_contract_reuses_immutable_worker_plan_only_under_fresh_r2_run():
    plan = _contract_plan()
    validate_m7_t2_2048_run_contract(
        plan,
        image=(
            "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
            "global/images/debian-12-bookworm-v20260804"
        ),
        max_live=58,
        max_run_seconds=25_200,
    )
    plan["run_name"] = "m7v5-t2second-25k-2048p"
    with pytest.raises(SystemExit, match="refuses run"):
        validate_m7_t2_2048_run_contract(
            plan,
            image=(
                "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
                "global/images/debian-12-bookworm-v20260804"
            ),
            max_live=58,
            max_run_seconds=25_200,
        )
    plan = _contract_plan()
    plan["content_bindings_without_generations"][0]["object"] = (
        "labelgen/m7v5-t2second-25k-2048p/staging/runtime.tar.gz"
    )
    with pytest.raises(SystemExit, match="content bindings drifted"):
        validate_m7_t2_2048_run_contract(
            plan,
            image=(
                "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
                "global/images/debian-12-bookworm-v20260804"
            ),
            max_live=58,
            max_run_seconds=25_200,
        )
    plan = _contract_plan()
    plan["startup_script_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="run identity drifted"):
        validate_m7_t2_2048_run_contract(
            plan,
            image=(
                "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
                "global/images/debian-12-bookworm-v20260804"
            ),
            max_live=58,
            max_run_seconds=25_200,
        )


def test_startup_publishes_done_last_and_checkpoint_after_files():
    source = (ROOT / "scripts/startup_hu_m31_label_gen_v1.sh").read_text(
        encoding="utf-8"
    )
    sort_guard = 'key=lambda p:(p.name=="SHARD_DONE.json"'
    assert sort_guard in source
    assert source.index(sort_guard) < source.index(" for path in paths:")
    assert "from ofc_regular.hu_m31_label_gen_resume_v1" not in source
    assert source.index("def restore_cached_position_object(") < source.index(
        "restore_cached_position_object(out/name,raw"
    )
    assert "restore_cached_position_object(out/name,raw" in source
    assert "complete_checkpoint_object_name(prefix)" in source
    assert 'f"/checkpoints/{count:06d}.json"' not in source
    assert "threading.Thread(target=heartbeat_loop" in source
    assert "heartbeat_stop.wait(120)" in source
    assert '"--shard-directory",str(worker_out)' in source
    assert "promote_worker_positions(strict=True)" in source
    assert source.rindex("promote_worker_positions(strict=True)") < source.rindex(
        "finalize_done_marker()"
    )
    assert source.index("for path in paths:") < source.index(
        "put(complete_checkpoint_object_name(prefix),canon(cp))"
    )


def test_partial_resume_restores_full_local_inventory(tmp_path: Path):
    plan = _plan()
    shard = plan["shards"][0]
    out = tmp_path / "shard"
    out.mkdir()
    for offset in range(shard["start"], shard["start"] + 4):
        raw = canonical_bytes({
            "schema": "hu_m31_label_gen_position_v1",
            "plan_sha256": plan["worker_plan_sha256"],
            "offset": offset,
        })
        name = f"position_{offset:08d}.json"
        restore_cached_position_object(
            out / name,
            raw,
            object_name=f"{shard['object_prefix']}/files/{name}",
            expected_object_name=f"{shard['object_prefix']}/files/{name}",
            generation=str(1000 + offset),
            expected_bytes=len(raw),
            expected_sha256=hashlib.sha256(raw).hexdigest(),
            plan_sha256=plan["worker_plan_sha256"],
            offset=offset,
            position_schema="hu_m31_label_gen_position_v1",
        )
    for offset in range(shard["start"] + 4, shard["start"] + shard["count"]):
        raw = canonical_bytes({
            "schema": "hu_m31_label_gen_position_v1",
            "plan_sha256": plan["worker_plan_sha256"],
            "offset": offset,
        })
        (out / f"position_{offset:08d}.json").write_bytes(raw)
    (out / "SHARD_DONE.json").write_bytes(_done(plan, shard))
    rows = [
        {
            "relative_path": path.name,
            "object_name": f"{shard['object_prefix']}/files/{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(out.iterdir())
    ]
    checkpoint = build_complete_checkpoint(
        plan_sha256=plan["worker_plan_sha256"],
        shard_id=shard["shard_id"],
        attempt_id="replacement-attempt",
        shard_start=shard["start"],
        shard_count=shard["count"],
        files=rows,
    )
    assert validate_complete_checkpoint(
        canonical_bytes(checkpoint), plan=plan, shard=shard
    )
    assert len(list(out.glob("position_*.json"))) == shard["count"]


def test_full_count_without_done_cannot_poison_fixed_complete_checkpoint():
    plan = _plan()
    shard = plan["shards"][0]
    rows = [
        {
            "relative_path": f"position_{offset:08d}.json",
            "object_name": (
                f"{shard['object_prefix']}/files/position_{offset:08d}.json"
            ),
            "sha256": "d" * 64,
            "bytes": 1,
        }
        for offset in range(shard["start"], shard["start"] + shard["count"])
    ]
    with pytest.raises(ValueError, match="whole shard"):
        build_complete_checkpoint(
            plan_sha256=plan["worker_plan_sha256"],
            shard_id=shard["shard_id"],
            attempt_id="attempt",
            shard_start=shard["start"],
            shard_count=shard["count"],
            files=rows,
        )
    assert complete_checkpoint_object_name(shard["object_prefix"]).endswith(
        "/checkpoints/complete.json"
    )


def test_tampered_cached_position_fails_closed_before_local_write(tmp_path: Path):
    plan = _plan()
    shard = plan["shards"][0]
    offset = shard["start"]
    good = canonical_bytes({
        "schema": "hu_m31_label_gen_position_v1",
        "plan_sha256": plan["worker_plan_sha256"],
        "offset": offset,
    })
    tampered = good[:-1] + (b" " if good[-1:] != b" " else b"\n")
    destination = tmp_path / f"position_{offset:08d}.json"
    with pytest.raises(ValueError, match="sha256 drifted"):
        restore_cached_position_object(
            destination,
            tampered,
            object_name=(
                f"{shard['object_prefix']}/files/position_{offset:08d}.json"
            ),
            expected_object_name=(
                f"{shard['object_prefix']}/files/position_{offset:08d}.json"
            ),
            generation="123",
            expected_bytes=len(tampered),
            expected_sha256=hashlib.sha256(good).hexdigest(),
            plan_sha256=plan["worker_plan_sha256"],
            offset=offset,
            position_schema="hu_m31_label_gen_position_v1",
        )
    assert not destination.exists()


def test_done_marker_requires_exact_canonical_provenance():
    plan = _plan()
    shard = plan["shards"][0]
    assert validate_done_marker(_done(plan, shard), plan=plan, shard=shard)

    foreign = json.loads(_done(plan, shard))
    foreign["plan_sha256"] = "c" * 64
    with pytest.raises(SystemExit, match="foreign or non-canonical"):
        validate_done_marker(canonical_bytes(foreign), plan=plan, shard=shard)

    pretty = json.dumps(json.loads(_done(plan, shard)), indent=2).encode()
    with pytest.raises(SystemExit, match="foreign or non-canonical"):
        validate_done_marker(pretty, plan=plan, shard=shard)

    assert validate_complete_checkpoint(
        _checkpoint(plan, shard), plan=plan, shard=shard
    )


def test_supervisor_never_relaunches_completed_shard(tmp_path: Path):
    plan = _plan()
    (tmp_path / "stage_receipt.json").write_text(
        json.dumps(_stage(plan)), encoding="utf-8"
    )
    adapter = FakeAdapter()
    done_shard = plan["shards"][0]
    _set_complete(adapter, plan, done_shard)
    adapter.instances["ofc-lg-test-001"] = _instance()

    snapshot = supervise_once(
        adapter=adapter,
        plan=plan,
        run_dir=tmp_path,
        service_account="worker@example.test",
        image="image",
        max_live=2,
        max_run_seconds=25_200,
        allow_writes=True,
    )

    assert snapshot["complete_shards"] == 1
    assert len(adapter.created) == 1
    assert adapter.created[0][0]["name"] == "ofc-lg-test-002"
    assert all(spec["name"] != "ofc-lg-test-000" for spec, _ in adapter.created)
    assert list(tmp_path.glob("supervisor_intent_*_002_*.json"))
    assert list(tmp_path.glob("supervisor_result_*_002_*.json"))


def test_done_instance_is_deleted_but_not_replaced_same_cycle(tmp_path: Path):
    plan = _plan()
    (tmp_path / "stage_receipt.json").write_text(
        json.dumps(_stage(plan)), encoding="utf-8"
    )
    adapter = FakeAdapter()
    shard = plan["shards"][0]
    _set_complete(adapter, plan, shard)
    adapter.instances["ofc-lg-test-000"] = _instance()
    adapter.instances["ofc-lg-test-001"] = _instance()

    snapshot = supervise_once(
        adapter=adapter,
        plan=plan,
        run_dir=tmp_path,
        service_account="worker@example.test",
        image="image",
        max_live=2,
        max_run_seconds=25_200,
        allow_writes=True,
    )

    assert adapter.deleted == ["ofc-lg-test-000"]
    assert not adapter.created
    assert snapshot["done_instances_delete_submitted"] == ["000"]


def test_inspection_refuses_foreign_done_before_launch():
    plan = _plan()
    adapter = FakeAdapter()
    shard = plan["shards"][1]
    wrong = {
        "schema": DONE_SCHEMA,
        "plan_sha256": plan["worker_plan_sha256"],
        "shard_id": "999",
        "positions": shard["count"],
    }
    adapter.objects[f"{shard['object_prefix']}/files/SHARD_DONE.json"] = canonical_bytes(
        wrong
    )
    with pytest.raises(SystemExit, match="foreign or non-canonical"):
        inspect_shards(adapter, plan)


def test_done_without_after_files_checkpoint_is_not_complete():
    plan = _plan()
    adapter = FakeAdapter()
    shard = plan["shards"][0]
    adapter.objects[f"{shard['object_prefix']}/files/SHARD_DONE.json"] = _done(
        plan, shard
    )
    state = inspect_shards(adapter, plan)[0]
    assert state.done is False


def test_stale_heartbeat_deletes_only_owned_incomplete_vm(tmp_path: Path):
    plan = _plan()
    (tmp_path / "stage_receipt.json").write_text(
        json.dumps(_stage(plan)), encoding="utf-8"
    )
    adapter = FakeAdapter()
    adapter.instances["ofc-lg-test-001"] = _instance(created="2020-01-01T00:00:00Z")
    prefix = f"{plan['shards'][1]['object_prefix']}/heartbeats/attempt-"
    name = f"{prefix}000000.json"
    adapter.listings[prefix] = [{
        "name": name,
        "updated": "2020-01-01T00:01:00Z",
    }]
    adapter.objects[name] = _heartbeat(plan, plan["shards"][1], "attempt", 0)

    snapshot = supervise_once(
        adapter=adapter,
        plan=plan,
        run_dir=tmp_path,
        service_account="worker@example.test",
        image="image",
        max_live=1,
        max_run_seconds=25_200,
        allow_writes=True,
        heartbeat_stale_seconds=60,
    )

    assert adapter.deleted == ["ofc-lg-test-001"]
    assert snapshot["stale_incomplete_delete_submitted"] == ["001"]


def test_historical_attempt_heartbeat_cannot_keep_current_attempt_alive():
    plan = _plan()
    adapter = FakeAdapter()
    shard = plan["shards"][0]
    adapter.instances["ofc-lg-test-000"] = _instance("new-attempt")
    old_prefix = f"{shard['object_prefix']}/heartbeats/old-attempt-"
    adapter.listings[old_prefix] = [{
        "name": f"{old_prefix}000999.json",
        "updated": "2099-01-01T00:00:00Z",
    }]
    states = inspect_shards(adapter, plan)
    assert states[0].attempt_id == "new-attempt"
    assert states[0].latest_heartbeat_at is None


def test_default_stale_windows_do_not_delete_normal_long_startup_or_publish():
    now = dt.datetime(2026, 8, 7, 12, 0, tzinfo=dt.timezone.utc)
    created = (now - dt.timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    heartbeat = (now - dt.timedelta(minutes=30)).isoformat().replace(
        "+00:00", "Z"
    )
    states = [
        ShardState(
            shard_id="000",
            done=False,
            instance_status="RUNNING",
            instance_created_at=created,
            latest_heartbeat_at=None,
            attempt_id="attempt",
        ),
        ShardState(
            shard_id="001",
            done=False,
            instance_status="RUNNING",
            instance_created_at="2020-01-01T00:00:00Z",
            latest_heartbeat_at=heartbeat,
            attempt_id="attempt",
        ),
    ]
    assert stale_incomplete_states(
        states,
        now=now,
        boot_grace_seconds=DEFAULT_BOOT_GRACE_SECONDS,
        heartbeat_stale_seconds=DEFAULT_HEARTBEAT_STALE_SECONDS,
    ) == []


def test_ambiguous_launch_intent_counts_toward_attempt_cap(tmp_path: Path):
    plan = _plan()
    (tmp_path / "stage_receipt.json").write_text(
        json.dumps(_stage(plan)), encoding="utf-8"
    )
    (tmp_path / "supervisor_intent_1_000_ambiguous.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="attempt limit 1"):
        supervise_once(
            adapter=FakeAdapter(),
            plan=plan,
            run_dir=tmp_path,
            service_account="worker@example.test",
            image="image",
            max_live=1,
            max_run_seconds=25_200,
            allow_writes=True,
            max_attempts_per_shard=1,
        )


@pytest.mark.parametrize(
    ("current_attempt", "current_status"),
    (("new-attempt", "TERMINATED"), ("attempt", "RUNNING")),
)
def test_dead_delete_rereads_attempt_and_status(
    tmp_path: Path, current_attempt: str, current_status: str
):
    plan = _plan()
    (tmp_path / "stage_receipt.json").write_text(
        json.dumps(_stage(plan)), encoding="utf-8"
    )
    target = "ofc-lg-test-000"

    class ChangingDeadAdapter(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.reads = 0

        def get_instance(self, *, instance_name):
            if instance_name != target:
                return None
            self.reads += 1
            payload = _instance(
                "attempt" if self.reads == 1 else current_attempt,
                created="2020-01-01T00:00:00Z",
            )
            payload["status"] = "TERMINATED" if self.reads == 1 else current_status
            return payload

    adapter = ChangingDeadAdapter()
    snapshot = supervise_once(
        adapter=adapter,
        plan=plan,
        run_dir=tmp_path,
        service_account="worker@example.test",
        image="image",
        max_live=1,
        max_run_seconds=25_200,
        allow_writes=True,
    )
    assert adapter.deleted == []
    assert snapshot["dead_incomplete_delete_submitted"] == []


def test_stale_delete_rereads_current_attempt_heartbeat(tmp_path: Path):
    plan = _plan()
    (tmp_path / "stage_receipt.json").write_text(
        json.dumps(_stage(plan)), encoding="utf-8"
    )
    shard = plan["shards"][0]
    prefix = f"{shard['object_prefix']}/heartbeats/attempt-"
    old_name = f"{prefix}000000.json"
    fresh_name = f"{prefix}000001.json"
    fresh_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    class RefreshingHeartbeatAdapter(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.list_reads = 0

        def list_prefix(self, *, bucket, prefix):
            self.list_reads += 1
            if self.list_reads == 1:
                return [{"name": old_name, "updated": "2020-01-01T00:01:00Z"}]
            return [{"name": fresh_name, "updated": fresh_at}]

    adapter = RefreshingHeartbeatAdapter()
    adapter.instances["ofc-lg-test-000"] = _instance(
        created="2020-01-01T00:00:00Z"
    )
    adapter.objects[old_name] = _heartbeat(plan, shard, "attempt", 0)
    adapter.objects[fresh_name] = _heartbeat(plan, shard, "attempt", 1)
    snapshot = supervise_once(
        adapter=adapter,
        plan=plan,
        run_dir=tmp_path,
        service_account="worker@example.test",
        image="image",
        max_live=1,
        max_run_seconds=25_200,
        allow_writes=True,
        heartbeat_stale_seconds=60,
    )
    assert adapter.deleted == []
    assert snapshot["stale_incomplete_delete_submitted"] == []


def test_foreign_project_instance_does_not_block_owned_run_durability():
    assert generation_is_durable({
        "complete_shards": 174,
        "total_shards": 174,
        "occupied_instances": 0,
        "project_instances": ["unrelated-vm"],
        "foreign_project_instances": ["unrelated-vm"],
    })
    assert not generation_is_durable({
        "complete_shards": 174,
        "total_shards": 174,
        "occupied_instances": 1,
        "project_instances": ["owned-vm", "unrelated-vm"],
        "foreign_project_instances": ["unrelated-vm"],
    })
