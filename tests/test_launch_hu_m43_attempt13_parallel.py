from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

import ofc_regular.launch_hu_m43_attempt13_parallel as launcher
from ofc_regular.hu_m43_attempt13_spot import SHARD_SCHEMA


_SOURCE_SHA = "1" * 64
_SCHEDULE_SHA = "2" * 64
_MANIFEST_SHA = "3" * 64


def _completed(command, returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _fixture(tmp_path: Path, *, total: int = 4, mode: str = "development"):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_name = "regular-hu-m43-attempt13-development200-test"
    manifest = {
        "schema": "hu_m43_attempt13_spot_package_v1",
        "run_name": run_name,
        "mode": mode,
        "total_shards": total,
        "source_sha256": _SOURCE_SHA,
        "schedule_sha256": _SCHEDULE_SHA,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "image": {
            "name": launcher.EXPECTED_IMAGE_NAME,
            "id": launcher.EXPECTED_IMAGE_ID,
            "project": "debian-cloud",
            "self_link": launcher.EXPECTED_IMAGE_SELF_LINK,
        },
    }
    authorization = {
        "manifest_sha256": _MANIFEST_SHA,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    rows = []
    root_first = 0 if mode == "development" else 200
    for shard in range(total):
        rows.append(
            {
                "schema": SHARD_SCHEMA,
                "run_name": run_name,
                "mode": mode,
                "shard": shard,
                "root_index": root_first + shard,
                "output_prefix": f"shard-{shard:03d}-root{root_first + shard:03d}",
            }
        )
    (run_dir / "manifest.json").write_text("manifest\n", encoding="utf-8")
    (run_dir / launcher.spot.SOURCE_NAME).write_bytes(b"source")
    (run_dir / launcher.spot.SCHEDULE_NAME).write_bytes(
        b"".join(launcher._canonical_json_bytes(row) for row in rows)
    )
    (run_dir / "launch_authorization.json").write_text("launch\n", encoding="utf-8")
    (run_dir / "execution_authorization.json").write_text(
        "execution\n", encoding="utf-8"
    )
    (run_dir / launcher.spot.STARTUP_NAME).write_text(
        "#!/usr/bin/env bash\n", encoding="utf-8"
    )
    return run_dir, manifest, authorization, rows


def _existing_instance(
    *,
    run_dir: Path,
    manifest: dict,
    authorization: dict,
    shard: int,
    project: str,
    bucket: str,
) -> dict:
    metadata = launcher._metadata_for(
        manifest=manifest,
        authorization=authorization,
        project=project,
        bucket=bucket,
        shard=shard,
        authorization_sha256=hashlib.sha256(
            (run_dir / "execution_authorization.json").read_bytes()
        ).hexdigest(),
        no_self_delete=False,
    )
    return {
        "name": launcher._instance_name(manifest["run_name"], shard),
        "zone": "https://www.googleapis.com/compute/v1/projects/p/zones/us-east1-b",
        "status": "RUNNING",
        "machineType": (
            "https://www.googleapis.com/compute/v1/projects/p/zones/us-east1-b/"
            f"machineTypes/{launcher.EXPECTED_MACHINE_TYPE}"
        ),
        "scheduling": {"provisioningModel": "SPOT"},
        "metadata": {
            "items": [
                {"key": key, "value": value} for key, value in metadata.items()
            ]
        },
    }


def test_shard_and_zone_parsers_allow_full_run_but_reject_ambiguity() -> None:
    assert launcher.parse_shards(["0-2", "4,6"], total_shards=7) == (0, 1, 2, 4, 6)
    assert launcher.parse_shards(["all"], total_shards=200) == tuple(range(200))
    assert launcher.parse_zones(["asia-northeast1-b,us-east1-b"]) == (
        "asia-northeast1-b",
        "us-east1-b",
    )
    with pytest.raises(ValueError, match="duplicate shard"):
        launcher.parse_shards(["0-2", "2"], total_shards=4)
    with pytest.raises(ValueError, match="outside"):
        launcher.parse_shards(["4"], total_shards=4)
    with pytest.raises(ValueError, match="duplicate GCE zone"):
        launcher.parse_zones(["us-east1-b,us-east1-b"])


def test_resume_missing_batches_prechecks_and_parallel_creates_exact_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = "ofc-solver-485418"
    bucket = "pokerhu-ofc-solver-485418-training"
    run_dir, manifest, authorization, rows = _fixture(tmp_path)
    monkeypatch.setattr(
        launcher.spot, "validate_launch", lambda _path: (manifest, authorization)
    )
    monkeypatch.setattr(launcher, "_verify_pinned_image", lambda **_kwargs: None)
    monkeypatch.setattr(
        launcher,
        "publish_immutable_inputs_once",
        lambda **_kwargs: ({"sha256": _SOURCE_SHA, "disposition": "published"},),
    )
    existing = _existing_instance(
        run_dir=run_dir,
        manifest=manifest,
        authorization=authorization,
        shard=1,
        project=project,
        bucket=bucket,
    )
    prefix = f"gs://{bucket}/runs/{manifest['run_name']}"
    done_uri = f"{prefix}/results/{rows[0]['output_prefix']}/DONE.json\n"
    commands: list[list[str]] = []
    lock = threading.Lock()

    def fake_invoke(command, *, timeout):
        del timeout
        command = list(command)
        with lock:
            commands.append(command)
        if command[1:4] == ["compute", "instances", "list"]:
            return _completed(command, stdout=json.dumps([existing]))
        if command[1:4] == ["storage", "ls", "--recursive"]:
            return _completed(command, stdout=done_uri)
        if command[1:4] == ["compute", "instances", "create"]:
            return _completed(command)
        raise AssertionError(command)

    monkeypatch.setattr(launcher, "_invoke", fake_invoke)
    result = launcher.launch_attempt13_parallel(
        run_dir=run_dir,
        project=project,
        bucket=bucket,
        zones=["asia-northeast1-b", "us-east1-b"],
        shards=["0-3"],
        max_workers=2,
        resume_missing=True,
    )
    assert result["status"] == "created"
    assert result["skipped_done_shards"] == [0]
    assert result["skipped_active_shards"] == [1]
    assert [row["shard"] for row in result["created"]] == [2, 3]
    assert result["current_profile_mutated"] is False
    assert result["runtime_policy_activated"] is False
    assert sum(cmd[1:4] == ["compute", "instances", "list"] for cmd in commands) == 1
    assert sum(cmd[1:4] == ["storage", "ls", "--recursive"] for cmd in commands) == 1
    creates = [cmd for cmd in commands if cmd[1:4] == ["compute", "instances", "create"]]
    assert len(creates) == 2
    metadata_rows = []
    for command in creates:
        assert launcher.EXPECTED_MACHINE_TYPE in command
        assert f"--image={launcher.EXPECTED_IMAGE_NAME}" in command
        metadata = next(value for value in command if value.startswith("--metadata="))
        parsed = dict(item.split("=", 1) for item in metadata[11:].split(","))
        metadata_rows.append(parsed)
        assert parsed["SOURCE_SHA256"] == _SOURCE_SHA
        assert parsed["MANIFEST_SHA256"] == _MANIFEST_SHA
        assert parsed["SCHEDULE_SHA256"] == _SCHEDULE_SHA
        assert parsed["AUTHORIZATION_SHA256"] == result["authorization_sha256"]
        assert parsed["RUN_NAME"] == manifest["run_name"]
    assert {row["SHARD"] for row in metadata_rows} == {"2", "3"}


def test_nonresume_conflict_aborts_before_any_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, manifest, authorization, _rows = _fixture(tmp_path)
    monkeypatch.setattr(
        launcher.spot, "validate_launch", lambda _path: (manifest, authorization)
    )
    monkeypatch.setattr(launcher, "_verify_pinned_image", lambda **_kwargs: None)
    monkeypatch.setattr(launcher, "publish_immutable_inputs_once", lambda **_kwargs: ())
    monkeypatch.setattr(
        launcher,
        "_precheck",
        lambda **_kwargs: ((0,), (), ()),
    )
    monkeypatch.setattr(
        launcher,
        "_create_missing_parallel",
        lambda **_kwargs: pytest.fail("create must not run after precheck conflict"),
    )
    with pytest.raises(FileExistsError, match="already have DONE"):
        launcher.launch_attempt13_parallel(
            run_dir=run_dir,
            project="ofc-solver-485418",
            bucket="pokerhu-ofc-solver-485418-training",
            zones=["asia-northeast1-b"],
            shards=["0-3"],
            resume_missing=False,
        )


def test_immutable_inputs_are_published_or_download_verified_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, manifest, _authorization, _rows = _fixture(tmp_path)
    remote: dict[str, bytes] = {}
    for source, uri in launcher._immutable_input_specs(
        run_dir=run_dir, manifest=manifest, bucket="bucket-test"
    ):
        remote[uri] = source.read_bytes()
    uploads = 0
    downloads = 0

    def fake_invoke(command, *, timeout):
        nonlocal uploads, downloads
        del timeout
        command = list(command)
        if "--if-generation-match=0" in command:
            uploads += 1
            return _completed(command, returncode=1, stderr="precondition failed")
        downloads += 1
        Path(command[4]).write_bytes(remote[command[3]])
        return _completed(command)

    monkeypatch.setattr(launcher, "_invoke", fake_invoke)
    result = launcher.publish_immutable_inputs_once(
        run_dir=run_dir,
        manifest=manifest,
        project="ofc-solver-485418",
        bucket="bucket-test",
    )
    assert uploads == downloads == 5
    assert len(result) == 5
    assert {row["disposition"] for row in result} == {"verified_existing"}


def test_immutable_existing_object_mismatch_fails_before_precheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, manifest, _authorization, _rows = _fixture(tmp_path)

    def fake_invoke(command, *, timeout):
        del timeout
        command = list(command)
        if "--if-generation-match=0" in command:
            return _completed(command, returncode=1, stderr="precondition failed")
        Path(command[4]).write_bytes(b"tampered-remote-object")
        return _completed(command)

    monkeypatch.setattr(launcher, "_invoke", fake_invoke)
    with pytest.raises(RuntimeError, match="immutable GCS object differs"):
        launcher.publish_immutable_inputs_once(
            run_dir=run_dir,
            manifest=manifest,
            project="ofc-solver-485418",
            bucket="bucket-test",
        )


def test_existing_instance_must_match_spot_machine_and_all_hash_metadata(
    tmp_path: Path,
) -> None:
    project = "ofc-solver-485418"
    bucket = "pokerhu-ofc-solver-485418-training"
    run_dir, manifest, authorization, _rows = _fixture(tmp_path)
    instance = _existing_instance(
        run_dir=run_dir,
        manifest=manifest,
        authorization=authorization,
        shard=1,
        project=project,
        bucket=bucket,
    )
    metadata = launcher._metadata_for(
        manifest=manifest,
        authorization=authorization,
        project=project,
        bucket=bucket,
        shard=1,
        authorization_sha256=hashlib.sha256(
            (run_dir / "execution_authorization.json").read_bytes()
        ).hexdigest(),
        no_self_delete=False,
    )
    validated = launcher._validate_existing_instance(
        instance, expected_metadata=metadata
    )
    assert validated["status"] == "RUNNING"
    instance["metadata"]["items"][5]["value"] = "f" * 64
    with pytest.raises(ValueError, match="metadata differs"):
        launcher._validate_existing_instance(instance, expected_metadata=metadata)


def test_parallel_create_helper_obeys_worker_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()

    def fake_create(*, shard, zone, root_index, **_kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"shard": shard, "zone": zone, "root_index": root_index}

    monkeypatch.setattr(launcher, "_create_instance", fake_create)
    schedule = [{"root_index": index} for index in range(10)]
    created, failures = launcher._create_missing_parallel(
        shards=tuple(range(10)),
        max_workers=3,
        zones=("asia-northeast1-b", "us-east1-b"),
        schedule=schedule,
        create_kwargs={},
    )
    assert len(created) == 10
    assert failures == ()
    assert maximum == 3


def test_parallel_launcher_rejects_preflight_and_worker_count_above_25(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, manifest, authorization, _rows = _fixture(
        tmp_path, total=7, mode="preflight"
    )
    monkeypatch.setattr(
        launcher.spot, "validate_launch", lambda _path: (manifest, authorization)
    )
    with pytest.raises(ValueError, match="Development200/Audit50"):
        launcher.launch_attempt13_parallel(
            run_dir=run_dir,
            project="ofc-solver-485418",
            bucket="pokerhu-ofc-solver-485418-training",
            zones=["asia-northeast1-b"],
            shards=["all"],
        )
    manifest["mode"] = "development"
    with pytest.raises(ValueError, match="1..25"):
        launcher.launch_attempt13_parallel(
            run_dir=run_dir,
            project="ofc-solver-485418",
            bucket="pokerhu-ofc-solver-485418-training",
            zones=["asia-northeast1-b"],
            shards=["all"],
            max_workers=26,
        )
