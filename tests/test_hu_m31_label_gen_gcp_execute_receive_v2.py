from __future__ import annotations

import hashlib
import json
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import hu_m31_label_gen_gcp_execute_v1 as subject
from ofc_regular.hu_m31_label_gen_resume_v1 import (
    build_complete_checkpoint,
    complete_checkpoint_object_name,
)
from ofc_regular.hu_m31_label_gen_worker_v1 import (
    DONE_SCHEMA,
    POSITION_SCHEMA,
    T2_VS_FL_KIND,
    T2_VS_FL_POSITION_SCHEMA,
    canonical_bytes,
)


PLAN_SHA = "a" * 64


class FakeAdapter:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.generations = {
            name: str(1000 + index) for index, name in enumerate(sorted(objects))
        }
        self.pinned_reads: list[tuple[str, str | None]] = []
        self.vanish_on_pinned: set[str] = set()
        self.metadata_sha_overrides: dict[str, str] = {}

    def get_object_metadata(self, *, bucket: str, object_name: str):
        raw = self.objects.get(object_name)
        if raw is None:
            return None
        return {
            "name": object_name,
            "generation": self.generations[object_name],
            "size": str(len(raw)),
        }

    def get_object_bytes(
        self, *, bucket: str, object_name: str, generation: str | None = None
    ):
        self.pinned_reads.append((object_name, generation))
        if generation is not None and object_name in self.vanish_on_pinned:
            return None
        if generation is not None and generation != self.generations.get(object_name):
            return None
        return self.objects.get(object_name)

    def list_prefix(self, *, bucket: str, prefix: str):
        return [
            {
                "name": name,
                "generation": self.generations[name],
                "size": str(len(raw)),
                "metadata": {
                    "sha256": self.metadata_sha_overrides.get(
                        name, hashlib.sha256(raw).hexdigest()
                    )
                },
            }
            for name, raw in sorted(self.objects.items())
            if name.startswith(prefix)
        ]


def _plan() -> dict[str, Any]:
    return {
        "run_name": "receive-test",
        "bucket": "bucket",
        "worker_plan_sha256": PLAN_SHA,
        "shards": [
            {
                "shard_id": "000",
                "start": 10,
                "count": 2,
                "object_prefix": "labelgen/receive-test/shards/000",
            }
        ],
    }


def _position(offset: int, *, schema: str = POSITION_SCHEMA) -> bytes:
    return canonical_bytes(
        {
            "schema": schema,
            "plan_sha256": PLAN_SHA,
            "offset": offset,
            "observation": {"street": "T2"},
            "runs": [],
        }
    )


def _objects(
    plan: dict[str, Any], *, position_schema: str = POSITION_SCHEMA
) -> dict[str, bytes]:
    shard = plan["shards"][0]
    prefix = shard["object_prefix"]
    files: dict[str, bytes] = {
        f"position_{offset:08d}.json": _position(offset, schema=position_schema)
        for offset in range(shard["start"], shard["start"] + shard["count"])
    }
    files["SHARD_DONE.json"] = canonical_bytes(
        {
            "schema": DONE_SCHEMA,
            "plan_sha256": PLAN_SHA,
            "shard_id": shard["shard_id"],
            "positions": shard["count"],
        }
    )
    inventory = [
        {
            "relative_path": relative,
            "object_name": f"{prefix}/files/{relative}",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        for relative, raw in sorted(files.items())
    ]
    checkpoint = canonical_bytes(
        build_complete_checkpoint(
            plan_sha256=PLAN_SHA,
            shard_id=shard["shard_id"],
            attempt_id="attempt-1",
            shard_start=shard["start"],
            shard_count=shard["count"],
            files=inventory,
        )
    )
    objects = {f"{prefix}/files/{relative}": raw for relative, raw in files.items()}
    objects[complete_checkpoint_object_name(prefix)] = checkpoint
    return objects


def test_receive_saves_positions_done_checkpoint_evidence_and_final_receipt(
    tmp_path: pathlib.Path,
) -> None:
    plan = _plan()
    adapter = FakeAdapter(_objects(plan))
    run_dir = tmp_path / "run"
    out = tmp_path / "received"

    subject.phase_receive(plan, run_dir, adapter, out)

    shard = out / "shard_000"
    assert (shard / "position_00000010.json").is_file()
    assert (shard / "position_00000011.json").is_file()
    done = shard / "SHARD_DONE.json"
    assert json.loads(done.read_bytes())["plan_sha256"] == PLAN_SHA
    checkpoint = out / "_audit/shard_000/complete.json"
    evidence_path = out / "_audit/shard_000/receive_evidence.json"
    generation_manifest_path = (
        out / "_audit/shard_000/position_generation_manifest.json"
    )
    assert checkpoint.is_file()
    assert generation_manifest_path.is_file()
    manifest = json.loads(generation_manifest_path.read_bytes())
    assert manifest["positions"] == 2
    assert manifest["all_position_gets_generation_pinned"] is True
    assert all(row["generation"].isdigit() for row in manifest["files"])
    evidence = json.loads(evidence_path.read_bytes())
    assert evidence["generation_pinned_done_complete_and_all_positions"] is True
    assert evidence["position_generation_manifest"]["positions"] == 2
    assert evidence["done_object"]["generation"].isdigit()
    assert evidence["complete_checkpoint_object"]["generation"].isdigit()
    receipt = json.loads((run_dir / "receive_receipt.json").read_bytes())
    assert receipt["schema"] == subject.RECEIVE_RECEIPT_SCHEMA
    assert receipt["positions"] == 2
    assert receipt["done_markers"] == 1
    assert receipt["complete_checkpoints"] == 1
    assert receipt["position_generation_manifests"] == 1
    assert receipt["all_expected_artifacts_verified"] is True
    assert receipt["all_positions_generation_pinned"] is True

    done_name = f"{plan['shards'][0]['object_prefix']}/files/SHARD_DONE.json"
    checkpoint_name = complete_checkpoint_object_name(
        plan["shards"][0]["object_prefix"]
    )
    assert (done_name, adapter.generations[done_name]) in adapter.pinned_reads
    assert (
        checkpoint_name,
        adapter.generations[checkpoint_name],
    ) in adapter.pinned_reads
    for offset in (10, 11):
        name = (
            f"{plan['shards'][0]['object_prefix']}/files/"
            f"position_{offset:08d}.json"
        )
        assert (name, adapter.generations[name]) in adapter.pinned_reads


def test_receive_accepts_the_plan_kind_specific_position_schema(
    tmp_path: pathlib.Path,
) -> None:
    plan = _plan()
    plan["plan_kind"] = T2_VS_FL_KIND
    subject.phase_receive(
        plan,
        tmp_path / "run",
        FakeAdapter(_objects(plan, position_schema=T2_VS_FL_POSITION_SCHEMA)),
        tmp_path / "received",
    )
    receipt = json.loads((tmp_path / "run/receive_receipt.json").read_bytes())
    assert receipt["all_expected_artifacts_verified"] is True


def test_receive_reuses_only_identical_local_files(tmp_path: pathlib.Path) -> None:
    plan = _plan()
    objects = _objects(plan)
    run_dir = tmp_path / "run"
    out = tmp_path / "received"
    subject.phase_receive(plan, run_dir, FakeAdapter(objects), out)
    before = (out / "shard_000/position_00000010.json").read_bytes()

    subject.phase_receive(plan, run_dir, FakeAdapter(objects), out)
    assert (out / "shard_000/position_00000010.json").read_bytes() == before

    (out / "shard_000/position_00000010.json").write_bytes(b"tamper")
    with pytest.raises(SystemExit, match="(byte count|digest|existing local).+disagrees"):
        subject.phase_receive(plan, run_dir, FakeAdapter(objects), out)


def test_receive_refuses_noncanonical_done_marker(tmp_path: pathlib.Path) -> None:
    plan = _plan()
    objects = _objects(plan)
    done_name = f"{plan['shards'][0]['object_prefix']}/files/SHARD_DONE.json"
    objects[done_name] = json.dumps(json.loads(objects[done_name]), indent=2).encode()
    with pytest.raises(SystemExit, match="SHARD_DONE is non-canonical"):
        subject.phase_receive(plan, tmp_path / "run", FakeAdapter(objects), tmp_path / "out")


def test_receive_refuses_checkpoint_plan_drift(tmp_path: pathlib.Path) -> None:
    plan = _plan()
    objects = _objects(plan)
    name = complete_checkpoint_object_name(plan["shards"][0]["object_prefix"])
    checkpoint = json.loads(objects[name])
    checkpoint["plan_sha256"] = "b" * 64
    unsigned = dict(checkpoint)
    unsigned.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        canonical_bytes(unsigned)
    ).hexdigest()
    objects[name] = canonical_bytes(checkpoint)
    with pytest.raises(SystemExit, match="provenance drifted"):
        subject.phase_receive(plan, tmp_path / "run", FakeAdapter(objects), tmp_path / "out")


def test_missing_done_never_writes_final_receipt(tmp_path: pathlib.Path) -> None:
    plan = _plan()
    objects = _objects(plan)
    objects.pop(f"{plan['shards'][0]['object_prefix']}/files/SHARD_DONE.json")
    run_dir = tmp_path / "run"
    subject.phase_receive(plan, run_dir, FakeAdapter(objects), tmp_path / "out")
    assert not (run_dir / "receive_receipt.json").exists()


def test_missing_checkpoint_never_writes_final_receipt(tmp_path: pathlib.Path) -> None:
    plan = _plan()
    objects = _objects(plan)
    objects.pop(complete_checkpoint_object_name(plan["shards"][0]["object_prefix"]))
    run_dir = tmp_path / "run"
    out = tmp_path / "out"
    subject.phase_receive(plan, run_dir, FakeAdapter(objects), out)
    assert (out / "shard_000/SHARD_DONE.json").is_file()
    assert not (run_dir / "receive_receipt.json").exists()


def test_missing_position_in_prefix_listing_fails_closed_without_final_receipt(
    tmp_path: pathlib.Path,
) -> None:
    plan = _plan()
    objects = _objects(plan)
    objects.pop(f"{plan['shards'][0]['object_prefix']}/files/position_00000011.json")
    run_dir = tmp_path / "run"
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="listing inventory mismatch"):
        subject.phase_receive(plan, run_dir, FakeAdapter(objects), out)
    assert (out / "shard_000/SHARD_DONE.json").is_file()
    assert (out / "_audit/shard_000/complete.json").is_file()
    assert not (out / "_audit/shard_000/receive_evidence.json").exists()
    assert not (run_dir / "receive_receipt.json").exists()


def test_extra_position_in_prefix_listing_fails_closed(tmp_path: pathlib.Path) -> None:
    plan = _plan()
    objects = _objects(plan)
    objects[
        f"{plan['shards'][0]['object_prefix']}/files/position_99999999.json"
    ] = _position(99_999_999)
    with pytest.raises(SystemExit, match="listing has extra object"):
        subject.phase_receive(
            plan, tmp_path / "run", FakeAdapter(objects), tmp_path / "out"
        )


def test_listed_position_generation_drift_fails_before_local_position_write(
    tmp_path: pathlib.Path,
) -> None:
    plan = _plan()
    adapter = FakeAdapter(_objects(plan))
    name = f"{plan['shards'][0]['object_prefix']}/files/position_00000010.json"
    adapter.vanish_on_pinned.add(name)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="generation .* drifted or vanished"):
        subject.phase_receive(plan, tmp_path / "run", adapter, out)
    assert not (out / "shard_000/position_00000010.json").exists()


def test_position_custom_sha_listing_must_match_checkpoint(
    tmp_path: pathlib.Path,
) -> None:
    plan = _plan()
    adapter = FakeAdapter(_objects(plan))
    name = f"{plan['shards'][0]['object_prefix']}/files/position_00000010.json"
    adapter.metadata_sha_overrides[name] = "b" * 64
    with pytest.raises(SystemExit, match="listing disagrees with complete checkpoint"):
        subject.phase_receive(
            plan, tmp_path / "run", adapter, tmp_path / "out"
        )


def test_position_prefix_listing_fallback_pages_through_rest_adapter() -> None:
    class PagedAdapter:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def _call(self, method: str, url: str):
            self.urls.append(url)
            if "pageToken=" not in url:
                return {"items": [{"name": "a"}], "nextPageToken": "page-2"}
            return {"items": [{"name": "b"}]}

        @staticmethod
        def _json(response, label: str):
            return response

    adapter = PagedAdapter()
    assert subject._list_prefix_metadata(
        adapter, bucket="bucket", prefix="labelgen/run/shards/000/files/position_"
    ) == [{"name": "a"}, {"name": "b"}]
    assert len(adapter.urls) == 2
    assert "items%28name%2Cgeneration%2Csize%2Cmetadata%29" in adapter.urls[0]
    assert "pageToken=page-2" in adapter.urls[1]


def test_existing_legacy_receipt_is_refused(tmp_path: pathlib.Path) -> None:
    plan = _plan()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "receive_receipt.json").write_text(
        json.dumps({"schema": "hu_m31_label_gen_receive_receipt_v1"})
    )
    with pytest.raises(SystemExit, match="predates or disagrees"):
        subject.phase_receive(
            plan, run_dir, FakeAdapter(_objects(plan)), tmp_path / "out"
        )


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_refreshable_adapter_proactively_refreshes_after_45_minutes() -> None:
    clock = FakeClock()
    authorizations: list[str] = []
    provider_calls: list[bool] = []

    def request(method, url, headers, payload, timeout):
        authorizations.append(headers["Authorization"])
        return subject.c4.HttpResponse(status=200, body=b"{}", headers={})

    def provider() -> str:
        provider_calls.append(True)
        return "proactive-token"

    adapter = subject.RefreshableGcpQualityRestAdapter(
        access_token="initial-token",
        zone=subject.c4.ZONE,
        token_provider=provider,
        monotonic=clock,
        request=request,
        sleep=lambda _seconds: None,
    )
    adapter.get_object_metadata(bucket="bucket", object_name="first")
    clock.value = subject.GCLOUD_TOKEN_REFRESH_SECONDS + 1
    adapter.get_object_metadata(bucket="bucket", object_name="second")

    assert provider_calls == [True]
    assert authorizations == [
        "Bearer initial-token",
        "Bearer proactive-token",
    ]


def test_refreshable_adapter_refreshes_and_retries_once_after_401() -> None:
    clock = FakeClock()
    authorizations: list[str] = []
    statuses = iter((401, 200))
    provider_calls: list[bool] = []

    def request(method, url, headers, payload, timeout):
        authorizations.append(headers["Authorization"])
        return subject.c4.HttpResponse(
            status=next(statuses), body=b"{}", headers={}
        )

    def provider() -> str:
        provider_calls.append(True)
        return "recovered-token"

    adapter = subject.RefreshableGcpQualityRestAdapter(
        access_token="expired-token",
        zone=subject.c4.ZONE,
        token_provider=provider,
        monotonic=clock,
        request=request,
        sleep=lambda _seconds: None,
    )
    assert adapter.get_object_metadata(bucket="bucket", object_name="object") == {}
    assert provider_calls == [True]
    assert authorizations == ["Bearer expired-token", "Bearer recovered-token"]


def test_fixed_token_adapter_remains_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_OAUTH_ACCESS_TOKEN", "fixed-token")
    adapter = subject._adapter(_plan())
    assert type(adapter) is subject.GcpQualityRestAdapter
    assert not isinstance(adapter, subject.RefreshableGcpQualityRestAdapter)


def test_gcloud_token_provider_is_quiet_in_memory_and_inherits_environment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="secret-token\n", stderr="")

    executable = r"C:\GoogleCloudSDK\bin\gcloud.CMD"
    assert subject._gcloud_access_token(
        runner=runner,
        executable_resolver=lambda _name: executable,
    ) == "secret-token"
    command, kwargs = calls[0]
    assert command == [executable, "auth", "print-access-token", "--quiet"]
    assert "env" not in kwargs  # inherited CLOUDSDK_CONFIG is not replaced
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_gcloud_token_provider_fails_closed_when_executable_is_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "must-not-appear"

    with pytest.raises(RuntimeError, match="gcloud executable is unavailable") as error:
        subject._gcloud_access_token(
            runner=lambda *_args, **_kwargs: pytest.fail("runner must not be called"),
            executable_resolver=lambda _name: None,
        )

    assert secret not in str(error.value)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
