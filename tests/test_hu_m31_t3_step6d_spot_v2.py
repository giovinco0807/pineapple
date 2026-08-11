from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6d_spot_v2 as spot
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


def _contract() -> dict:
    return runner.build_run_contract(
        candidate_library_sha256="1" * 64,
        reference_library_sha256=spot.REFERENCE_NATIVE_LIBRARY_SHA256,
        workers=1,
        rayon_threads_per_worker=16,
    )


def _candidate02_contract() -> dict:
    return runner.build_run_contract(
        candidate_library_sha256="1" * 64,
        reference_library_sha256=spot.REFERENCE_NATIVE_LIBRARY_SHA256,
        workers=1,
        rayon_threads_per_worker=16,
        variant=runner.CANDIDATE02_VARIANT,
    )


def _candidate02_tail_v2_contract() -> dict:
    return runner.build_run_contract(
        candidate_library_sha256="1" * 64,
        reference_library_sha256=spot.REFERENCE_NATIVE_LIBRARY_SHA256,
        workers=1,
        rayon_threads_per_worker=16,
        variant=runner.CANDIDATE02_TAIL_V2_VARIANT,
    )


def _elf(suffix: bytes) -> bytes:
    header = bytearray(20)
    header[:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1
    header[16:18] = b"\x03\x00"
    header[18:20] = b"\x3e\x00"
    return bytes(header) + suffix


def _preflight_manifest() -> dict:
    return {
        "run_name": "step6d-v2-preflight-unit",
        "launch_target": spot.build_launch_target(),
        "cost_guard": spot.build_cost_guard(),
        "run_contract": _contract(),
    }


def _lifecycle_manifest() -> dict:
    manifest = _preflight_manifest()
    contract = manifest["run_contract"]
    records = []
    for value in spot.build_job_manifests(contract):
        record = spot._job_record(value)
        record.update({"sha256": "c" * 64, "bytes": 1})
        records.append(record)
    return {
        **manifest,
        "source_sha256": "d" * 64,
        "run_contract_digest": runner.canonical_sha256(contract),
        "job_manifests": records,
    }


def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def _quota_payload(*, cpu_available: int, spot_available: int) -> str:
    return json.dumps(
        {
            "quotas": [
                {"metric": "CPUS", "limit": 1000, "usage": 1000 - cpu_available},
                {
                    "metric": "PREEMPTIBLE_CPUS",
                    "limit": 1000,
                    "usage": 1000 - spot_available,
                },
            ]
        }
    )


def test_step6d_v2_job_matrix_is_exact_tail_roles_and_shared_digest() -> None:
    contract = _contract()
    jobs = spot.build_job_manifests(contract)

    assert len(jobs) == spot.AUTHORIZED_JOB_COUNT == spot.MAX_LOGICAL_JOBS == 20
    assert spot.authorized_job_ids() == tuple(
        f"{role}-hand-{index:03d}"
        for role in ("candidate", "reference")
        for index in (2, 6, 7, 9, 13, 20, 21, 29, 33, 50)
    )
    assert all(set(value) == spot._JOB_MANIFEST_KEYS for value in jobs)
    assert {value["run_contract_digest"] for value in jobs} == {
        runner.canonical_sha256(contract)
    }
    assert [value["work_hand_indices"] for value in jobs[:10]] == [
        [index] for index in spot.TAIL_HAND_INDICES
    ]
    assert [value["source_role"] for value in jobs] == ["candidate"] * 10 + [
        "reference"
    ] * 10


def test_candidate02_uses_same_bounded_tail_lifecycle_with_fresh_contract() -> None:
    contract = _candidate02_contract()
    jobs = spot.build_job_manifests(contract)
    assert len(jobs) == 20
    assert all(
        value["run_contract"]["schema"] == runner.CANDIDATE02_RUN_CONTRACT_SCHEMA
        for value in jobs
    )
    assert {value["run_contract_digest"] for value in jobs} == {
        runner.canonical_sha256(contract)
    }
    assert (
        spot._build_shared_contract(
            candidate_sha256="1" * 64,
            reference_sha256=spot.REFERENCE_NATIVE_LIBRARY_SHA256,
            contract_variant=runner.CANDIDATE02_VARIANT,
        )
        == contract
    )


def test_candidate02_tail_v2_job_matrix_comes_only_from_validated_contract() -> None:
    contract = _candidate02_tail_v2_contract()
    tail = (0, 4, 5, 12, 14, 16, 17, 23, 41, 43)
    expected_ids = tuple(
        f"{role}-hand-{index:03d}"
        for role in ("candidate", "reference")
        for index in tail
    )

    jobs = spot.build_job_manifests(contract)

    assert spot._contract_tail_hand_indices(contract) == tail
    assert spot.authorized_job_ids(contract) == expected_ids
    assert [value["work_hand_indices"] for value in jobs[:10]] == [
        [index] for index in tail
    ]
    assert [spot._job_record(value)["job_id"] for value in jobs] == list(expected_ids)
    assert spot._parse_jobs("all", contract) == expected_ids
    assert (
        spot.authorized_job_ids() != expected_ids
    )  # no-arg legacy helper is immutable


def test_candidate02_tail_v2_rejects_old_tail_and_mixed_contract_mapping() -> None:
    contract = _candidate02_tail_v2_contract()
    with pytest.raises(ValueError, match="job manifest boundary changed"):
        spot.build_job_manifest(
            run_contract=contract,
            source_role="candidate",
            hand_index=2,
        )
    value = spot.build_job_manifest(
        run_contract=contract,
        source_role="candidate",
        hand_index=0,
    )
    value["run_contract"] = _candidate02_contract()
    with pytest.raises(ValueError):
        spot.validate_job_manifest(value)


@pytest.mark.parametrize(
    "mutation",
    ["digest", "extra_key", "role", "non_tail", "multi_hand"],
)
def test_step6d_v2_job_manifest_fails_closed(mutation: str) -> None:
    value = spot.build_job_manifest(
        run_contract=_contract(), source_role="candidate", hand_index=2
    )
    if mutation == "digest":
        value["run_contract_digest"] = "0" * 64
    elif mutation == "extra_key":
        value["unexpected"] = False
    elif mutation == "role":
        value["source_role"] = "combined"
    elif mutation == "non_tail":
        value["work_hand_indices"] = [3]
    else:
        value["work_hand_indices"] = [2, 6]

    with pytest.raises(ValueError):
        spot.validate_job_manifest(value)


def test_step6d_v2_launch_selection_is_bounded_to_twenty_tail_jobs() -> None:
    assert spot._bounded_jobs(["candidate-hand-002"]) == ("candidate-hand-002",)
    assert spot._parse_jobs("all") == spot.authorized_job_ids()
    with pytest.raises(ValueError):
        spot._bounded_jobs([])
    with pytest.raises(ValueError):
        spot._bounded_jobs(["candidate-hand-003"])
    with pytest.raises(ValueError):
        spot._bounded_jobs(["candidate-hand-002", "candidate-hand-002"])


def test_step6d_v2_cost_guard_math_is_exact_and_nested() -> None:
    guard = spot.build_cost_guard()

    assert guard == {
        "schema": "hu_m31_t3_step6d_spot_cost_guard_v1",
        "currency": "USD",
        "spot_price_ceiling_usd_per_vm_hour": 0.5,
        "max_runtime_seconds_per_vm": 3300,
        "internal_watchdog_seconds_per_vm": 3000,
        "max_logical_jobs": 20,
        "max_attempts_per_job": 2,
        "max_cumulative_vm_jobs": 40,
        "all_20_estimated_max_compute_usd": spot.ALL_20_ESTIMATED_MAX_COMPUTE_USD,
        "all_attempts_estimated_max_compute_usd": (
            spot.ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD
        ),
        "hard_tail_compute_cap_usd": 20.0,
        "m31_total_compute_cap_usd": 500.0,
    }
    assert spot._estimated_max_compute_usd(1) == 0.5 * 3300 / 3600
    assert spot._estimated_max_compute_usd(20) == spot.ALL_20_ESTIMATED_MAX_COMPUTE_USD
    assert spot.ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD < 20.0
    assert (
        guard["all_20_estimated_max_compute_usd"]
        <= guard["all_attempts_estimated_max_compute_usd"]
        <= guard["hard_tail_compute_cap_usd"]
        <= guard["m31_total_compute_cap_usd"]
    )


@pytest.mark.parametrize("attempt_index", [0, 1])
def test_step6d_v2_every_attempt_has_infra_and_guest_runtime_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attempt_index: int
) -> None:
    manifest = _lifecycle_manifest()
    authorization = {"package_manifest_sha256": "a" * 64}
    (tmp_path / spot.AUTHORIZATION_NAME).write_bytes(spot.canonical_bytes({}))
    (tmp_path / spot.LAUNCH_CLAIM_NAME).write_bytes(spot.canonical_bytes({}))
    if attempt_index == 1:
        (tmp_path / spot.LAUNCH_RESULT_NAME).write_bytes(spot.canonical_bytes({}))
    prefix = f"gs://{spot.DEFAULT_BUCKET}/runs/{manifest['run_name']}"
    claim_uri = (
        f"{prefix}/control/{spot.LAUNCH_CLAIM_NAME}"
        if attempt_index == 0
        else f"{prefix}/resume/{spot.RESUME_CLAIM_NAME}"
    )
    commands: list[list[str]] = []

    def capture(command, **_kwargs):
        commands.append(list(command))
        return _completed()

    monkeypatch.setattr(spot, "_run", capture)
    spot._create_instance(
        target=tmp_path,
        manifest=manifest,
        authorization=authorization,
        prefix=prefix,
        identifier="candidate-hand-002",
        project=spot.DEFAULT_PROJECT,
        zone=spot.DEFAULT_ZONES[0],
        attempt_index=attempt_index,
        attempt_claim_uri=claim_uri,
        attempt_claim_sha256="b" * 64,
    )

    command = commands[0]
    assert "--instance-termination-action=DELETE" in command
    assert "--max-run-duration=3300s" in command
    metadata = next(value for value in command if value.startswith("--metadata="))
    assert "MAX_RUNTIME_SECONDS=3000" in metadata


def test_step6d_v2_preflight_rejects_insufficient_cpu_or_spot_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        return _completed(stdout=_quota_payload(cpu_available=15, spot_available=320))

    monkeypatch.setattr(spot, "_subprocess_run", fake_run)
    with pytest.raises(RuntimeError, match=r"quota insufficient.*16.*CPUS"):
        spot.preflight_launch(
            manifest=_preflight_manifest(),
            selected=("candidate-hand-002",),
            project=spot.DEFAULT_PROJECT,
            bucket=spot.DEFAULT_BUCKET,
        )

    assert len(calls) == 1
    assert calls[0][1:5] == ["compute", "regions", "describe", "asia-northeast1"]
    assert "instances" not in calls[0]
    assert "storage" not in calls[0]


def test_step6d_v2_preflight_rejects_any_selected_existing_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        values = list(command)
        calls.append(values)
        if values[1:4] == ["compute", "regions", "describe"]:
            return _completed(
                stdout=_quota_payload(cpu_available=320, spot_available=320)
            )
        if values[1:4] == ["compute", "instances", "list"]:
            return _completed(
                stdout=json.dumps(
                    [
                        {
                            "name": "step6d-v2-preflight-unit-j01",
                            "zone": "asia-northeast1-c",
                            "status": "RUNNING",
                        }
                    ]
                )
            )
        raise AssertionError(f"unexpected preflight query: {values}")

    monkeypatch.setattr(spot, "_subprocess_run", fake_run)
    with pytest.raises(FileExistsError, match="selected instances already exist"):
        spot.preflight_launch(
            manifest=_preflight_manifest(),
            selected=("candidate-hand-002", "candidate-hand-006"),
            project=spot.DEFAULT_PROJECT,
            bucket=spot.DEFAULT_BUCKET,
        )

    assert len(calls) == 2
    assert all("storage" not in value for value in calls)


def test_step6d_v2_preflight_checks_all_selected_done_before_rejecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    done_queries: list[str] = []

    def fake_run(command, **_kwargs):
        values = list(command)
        if values[1:4] == ["compute", "regions", "describe"]:
            return _completed(
                stdout=_quota_payload(cpu_available=320, spot_available=320)
            )
        if values[1:4] == ["compute", "instances", "list"]:
            return _completed(stdout="[]")
        if values[1:4] == ["storage", "objects", "describe"]:
            done_queries.append(values[4])
            if "candidate-hand-006" in values[4]:
                return _completed(stdout='{"generation":"1"}')
            return _completed(returncode=1, stderr="404 Not Found")
        raise AssertionError(f"unexpected preflight query: {values}")

    monkeypatch.setattr(spot, "_subprocess_run", fake_run)
    with pytest.raises(FileExistsError, match="selected DONE objects already exist"):
        spot.preflight_launch(
            manifest=_preflight_manifest(),
            selected=("candidate-hand-002", "candidate-hand-006"),
            project=spot.DEFAULT_PROJECT,
            bucket=spot.DEFAULT_BUCKET,
        )

    assert len(done_queries) == 2
    assert done_queries[0].endswith("candidate-hand-002/DONE.json")
    assert done_queries[1].endswith("candidate-hand-006/DONE.json")


def test_step6d_v2_resume_preflight_requires_all_initial_vms_quiescent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _lifecycle_manifest()
    active_unselected = f"{manifest['run_name']}-j19"

    def fake_json_query(command, **_kwargs):
        if command[1:4] == ["compute", "instances", "list"]:
            return [
                {
                    "name": active_unselected,
                    "zone": "asia-northeast1-c",
                    "status": "RUNNING",
                }
            ]
        raise AssertionError(f"unexpected query before quiescence rejection: {command}")

    monkeypatch.setattr(spot, "_json_query", fake_json_query)
    monkeypatch.setattr(
        spot,
        "_subprocess_run",
        lambda *_args, **_kwargs: pytest.fail(
            "DONE queried before all-wave quiescence"
        ),
    )

    with pytest.raises(FileExistsError, match=active_unselected):
        spot.preflight_resume(
            run_dir=tmp_path,
            manifest=manifest,
            selected=("candidate-hand-002",),
            project=spot.DEFAULT_PROJECT,
            bucket=spot.DEFAULT_BUCKET,
        )


def test_step6d_v2_resume_preflight_rejects_any_attempt1_name_across_all_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _lifecycle_manifest()
    existing_unselected = f"{manifest['run_name']}-j19-a01"

    monkeypatch.setattr(
        spot,
        "_json_query",
        lambda *_args, **_kwargs: [
            {
                "name": existing_unselected,
                "zone": "asia-northeast1-c",
                "status": "TERMINATED",
            }
        ],
    )

    with pytest.raises(FileExistsError, match=existing_unselected):
        spot.preflight_resume(
            run_dir=tmp_path,
            manifest=manifest,
            selected=("candidate-hand-002",),
            project=spot.DEFAULT_PROJECT,
            bucket=spot.DEFAULT_BUCKET,
        )


def test_step6d_v2_resume_preflight_derives_exact_missing_set_from_validated_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _lifecycle_manifest()
    completed = "candidate-hand-002"
    expected_missing = tuple(
        identifier
        for identifier in spot.authorized_job_ids()
        if identifier != completed
    )
    done_queries: list[str] = []
    validated: list[str] = []

    def fake_json_query(command, **_kwargs):
        if command[1:4] == ["compute", "instances", "list"]:
            return []
        if command[1:4] == ["compute", "regions", "describe"]:
            return json.loads(_quota_payload(cpu_available=320, spot_available=320))
        raise AssertionError(f"unexpected resume query: {command}")

    def fake_subprocess(command, **_kwargs):
        values = list(command)
        assert values[1:4] == ["storage", "objects", "describe"]
        done_queries.append(values[4])
        if completed in values[4]:
            return _completed(stdout='{"generation":"1"}')
        return _completed(returncode=1, stderr="404 Not Found")

    monkeypatch.setattr(spot, "_json_query", fake_json_query)
    monkeypatch.setattr(spot, "_subprocess_run", fake_subprocess)
    monkeypatch.setattr(spot, "_run", lambda *_args, **_kwargs: _completed())

    def validate_remote_job(*, record, **_kwargs):
        validated.append(record["job_id"])
        return {"job_id": record["job_id"]}

    monkeypatch.setattr(spot, "_validate_received_job", validate_remote_job)

    with pytest.raises(ValueError, match="exactly all validated incomplete jobs"):
        spot.preflight_resume(
            run_dir=tmp_path,
            manifest=manifest,
            selected=expected_missing[:-1],
            project=spot.DEFAULT_PROJECT,
            bucket=spot.DEFAULT_BUCKET,
        )

    preflight = spot.preflight_resume(
        run_dir=tmp_path,
        manifest=manifest,
        selected=expected_missing,
        project=spot.DEFAULT_PROJECT,
        bucket=spot.DEFAULT_BUCKET,
    )

    assert len(done_queries) == 40
    assert validated == [completed, completed]
    assert preflight["validated_completed_job_ids"] == [completed]
    assert preflight["selected_job_ids"] == list(expected_missing)
    assert preflight["all_incomplete_jobs_selected"] is True
    assert preflight["required_vcpus"] == 19 * 16


def test_step6d_v2_launch_never_publishes_when_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _preflight_manifest()
    authorization = {
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
    }
    monkeypatch.setattr(
        spot, "validate_launch", lambda _path: (manifest, authorization)
    )
    monkeypatch.setattr(
        spot,
        "preflight_launch",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("quota insufficient")),
    )
    monkeypatch.setattr(
        spot,
        "_publish_package",
        lambda **_kwargs: pytest.fail("package published before preflight"),
    )

    with pytest.raises(RuntimeError, match="quota insufficient"):
        spot.launch_jobs(run_dir=tmp_path, jobs=spot.authorized_job_ids())


def test_step6d_v2_ambiguous_create_failure_probes_and_deletes_selected_vm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        **_preflight_manifest(),
        "run_contract_digest": "d" * 64,
    }
    authorization = {
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
    }
    commands: list[list[str]] = []
    describe_count = 0

    monkeypatch.setattr(
        spot, "validate_launch", lambda _path: (manifest, authorization)
    )
    monkeypatch.setattr(
        spot,
        "preflight_launch",
        lambda **_kwargs: {
            "schema": spot.LAUNCH_PREFLIGHT_SCHEMA,
            "status": "quota_and_collision_checks_passed",
        },
    )
    monkeypatch.setattr(
        spot,
        "_run",
        lambda *_args, **_kwargs: _completed(
            stdout=json.dumps(
                {
                    "id": spot.EXPECTED_IMAGE_ID,
                    "selfLink": spot.EXPECTED_IMAGE_SELF_LINK,
                }
            )
        ),
    )
    monkeypatch.setattr(spot, "_publish_package", lambda **_kwargs: "gs://unit/run")
    monkeypatch.setattr(spot, "_publish_once", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        spot,
        "_create_instance",
        lambda **_kwargs: (_ for _ in ()).throw(
            TimeoutError("create response timed out after API acceptance")
        ),
    )

    def fake_subprocess(command, **_kwargs):
        nonlocal describe_count
        values = list(command)
        commands.append(values)
        if values[1:4] == ["compute", "instances", "describe"]:
            describe_count += 1
            if describe_count == 1:
                return _completed(
                    stdout=json.dumps(
                        {
                            "name": "step6d-v2-preflight-unit-j00",
                            "status": "RUNNING",
                            "zone": "asia-northeast1-b",
                        }
                    )
                )
            return _completed(returncode=1, stderr="404 instance was not found")
        if values[1:4] == ["compute", "instances", "delete"]:
            return _completed()
        raise AssertionError(f"unexpected cleanup command: {values}")

    monkeypatch.setattr(spot, "_subprocess_run", fake_subprocess)

    with pytest.raises(RuntimeError, match="creation failed; cleanup attempted"):
        spot.launch_jobs(run_dir=tmp_path, jobs=spot.authorized_job_ids())

    receipt = json.loads(
        (tmp_path / spot.LAUNCH_RESULT_NAME).read_text(encoding="utf-8")
    )
    assert receipt["created"] == []
    assert len(receipt["failures"]) == 20
    assert receipt["failures"][0]["error_type"] == "TimeoutError"
    assert len(receipt["cleanup"]) == 20
    assert receipt["cleanup"][0] == {
        "job_id": "candidate-hand-002",
        "instance": "step6d-v2-preflight-unit-j00",
        "zone": "asia-northeast1-b",
        "attempt_index": 0,
        "probe_status": "exists",
        "delete_returncode": 0,
        "stop_returncode": None,
        "final_instance_status": None,
        "status": "deleted",
        "absence_proven": True,
        "compute_stopped_or_absent": True,
    }
    assert all(row["absence_proven"] for row in receipt["cleanup"])
    assert receipt["cleanup_absence_proven"] is True
    assert receipt["cleanup_compute_stopped_or_absent"] is True
    assert any(
        value[1:4] == ["compute", "instances", "delete"]
        and "step6d-v2-preflight-unit-j00" in value
        for value in commands
    )


def test_step6d_v2_cleanup_fails_closed_when_absence_cannot_be_proven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def denied(command, **_kwargs):
        commands.append(list(command))
        return _completed(returncode=1, stderr="permission denied")

    monkeypatch.setattr(spot, "_subprocess_run", denied)
    outcomes = spot._cleanup_selected_instances(
        manifest=_preflight_manifest(),
        selected=("candidate-hand-002",),
        project=spot.DEFAULT_PROJECT,
        zones=spot.DEFAULT_ZONES,
    )

    assert outcomes[0]["probe_status"] == "absence_unproven"
    assert outcomes[0]["status"] == "cleanup_failed_absence_unproven"
    assert outcomes[0]["absence_proven"] is False
    assert outcomes[0]["compute_stopped_or_absent"] is False
    assert [value[3] for value in commands] == [
        "describe",
        "delete",
        "stop",
        "describe",
    ]


def test_step6d_v2_concurrent_launch_loser_never_publishes_or_creates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        **_preflight_manifest(),
        "run_contract_digest": "e" * 64,
    }
    authorization = {
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
    }
    barrier = threading.Barrier(2)
    publish_calls: list[int] = []
    create_calls: list[str] = []

    monkeypatch.setattr(
        spot, "validate_launch", lambda _path: (manifest, authorization)
    )

    def synchronized_preflight(**_kwargs):
        barrier.wait()
        return {
            "schema": spot.LAUNCH_PREFLIGHT_SCHEMA,
            "status": "quota_and_collision_checks_passed",
        }

    monkeypatch.setattr(spot, "preflight_launch", synchronized_preflight)
    monkeypatch.setattr(
        spot,
        "_run",
        lambda *_args, **_kwargs: _completed(
            stdout=json.dumps(
                {
                    "id": spot.EXPECTED_IMAGE_ID,
                    "selfLink": spot.EXPECTED_IMAGE_SELF_LINK,
                }
            )
        ),
    )

    def publish(**_kwargs):
        publish_calls.append(1)
        return "gs://unit/run"

    def create(**kwargs):
        identifier = kwargs["identifier"]
        create_calls.append(identifier)
        ordinal = spot.authorized_job_ids().index(identifier)
        role, hand = identifier.split("-hand-")
        return {
            "job_id": identifier,
            "source_role": role,
            "work_hand_indices": [int(hand)],
            "instance": f"step6d-v2-preflight-unit-j{ordinal:02d}",
            "zone": kwargs["zone"],
            "attempt_index": 0,
            "status": "created",
        }

    monkeypatch.setattr(spot, "_publish_package", publish)
    monkeypatch.setattr(spot, "_publish_once", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(spot, "_create_instance", create)

    def launch() -> str:
        try:
            spot.launch_jobs(run_dir=tmp_path, jobs=spot.authorized_job_ids())
        except FileExistsError as exc:
            assert spot.LAUNCH_CLAIM_NAME in str(exc)
            return "claim_rejected"
        return "launched"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _value: launch(), (0, 1)))

    assert sorted(outcomes) == ["claim_rejected", "launched"]
    assert publish_calls == [1]
    assert set(create_calls) == set(spot.authorized_job_ids())
    assert len(create_calls) == 20
    claim = json.loads((tmp_path / spot.LAUNCH_CLAIM_NAME).read_text(encoding="utf-8"))
    assert claim["crash_reuse_authorized"] is False
    assert (tmp_path / spot.LAUNCH_RESULT_NAME).is_file()


def test_step6d_v2_remote_launch_claim_wins_before_cross_host_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _lifecycle_manifest()
    authorization = {
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
    }
    run_dirs = (tmp_path / "host-a", tmp_path / "host-b")
    for run_dir in run_dirs:
        run_dir.mkdir()
    preflight_barrier = threading.Barrier(2)
    remote_lock = threading.Lock()
    remote_claim: list[bytes] = []
    create_calls: list[str] = []

    monkeypatch.setattr(
        spot, "validate_launch", lambda _path: (manifest, authorization)
    )

    def preflight(**_kwargs):
        preflight_barrier.wait()
        return {
            "schema": spot.LAUNCH_PREFLIGHT_SCHEMA,
            "status": "quota_and_collision_checks_passed",
        }

    monkeypatch.setattr(spot, "preflight_launch", preflight)
    monkeypatch.setattr(
        spot,
        "_run",
        lambda *_args, **_kwargs: _completed(
            stdout=json.dumps(
                {
                    "id": spot.EXPECTED_IMAGE_ID,
                    "selfLink": spot.EXPECTED_IMAGE_SELF_LINK,
                }
            )
        ),
    )

    def publish_once(source, uri, **_kwargs):
        assert uri.endswith("/control/launch_claim.json")
        payload = Path(source).read_bytes()
        with remote_lock:
            if remote_claim:
                assert payload != remote_claim[0]
                raise FileExistsError("remote launch claim generation is nonzero")
            remote_claim.append(payload)

    def create(**kwargs):
        assert remote_claim
        identifier = kwargs["identifier"]
        create_calls.append(identifier)
        ordinal = spot.authorized_job_ids().index(identifier)
        role, hand = identifier.split("-hand-")
        assert kwargs["attempt_claim_uri"].endswith("/control/launch_claim.json")
        return {
            "job_id": identifier,
            "source_role": role,
            "work_hand_indices": [int(hand)],
            "instance": f"{manifest['run_name']}-j{ordinal:02d}",
            "zone": kwargs["zone"],
            "attempt_index": 0,
            "status": "created",
        }

    monkeypatch.setattr(spot, "_publish_once", publish_once)
    monkeypatch.setattr(
        spot,
        "_publish_package",
        lambda **_kwargs: (f"gs://{spot.DEFAULT_BUCKET}/runs/{manifest['run_name']}"),
    )
    monkeypatch.setattr(spot, "_create_instance", create)

    def launch(run_dir: Path) -> str:
        try:
            spot.launch_jobs(run_dir=run_dir, jobs=spot.authorized_job_ids())
        except FileExistsError as exc:
            assert "remote launch claim" in str(exc)
            return "remote_claim_rejected"
        return "launched"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(launch, run_dirs))

    assert sorted(outcomes) == ["launched", "remote_claim_rejected"]
    assert len(remote_claim) == 1
    assert len(create_calls) == 20


def test_step6d_v2_resume_claim_is_published_before_attempt1_and_third_is_forbidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _lifecycle_manifest()
    authorization = {
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
    }
    selected = ("candidate-hand-002", "reference-hand-050")
    (tmp_path / spot.LAUNCH_CLAIM_NAME).write_bytes(
        spot.canonical_bytes({"initial": "claim"})
    )
    (tmp_path / spot.LAUNCH_RESULT_NAME).write_bytes(
        spot.canonical_bytes({"initial": "result"})
    )
    published: list[str] = []
    create_calls: list[str] = []

    monkeypatch.setattr(
        spot, "validate_launch", lambda _path: (manifest, authorization)
    )
    monkeypatch.setattr(
        spot,
        "validate_receive_launch_chain",
        lambda **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        spot,
        "preflight_resume",
        lambda **_kwargs: {
            "schema": spot.RESUME_PREFLIGHT_SCHEMA,
            "selected_job_ids": list(selected),
        },
    )
    monkeypatch.setattr(
        spot,
        "_run",
        lambda *_args, **_kwargs: _completed(
            stdout=json.dumps(
                {
                    "id": spot.EXPECTED_IMAGE_ID,
                    "selfLink": spot.EXPECTED_IMAGE_SELF_LINK,
                }
            )
        ),
    )

    def publish_once(source, uri, **_kwargs):
        assert Path(source) == tmp_path / spot.RESUME_CLAIM_NAME
        assert uri.endswith("/resume/resume_claim.json")
        published.append(uri)

    def create(**kwargs):
        assert published
        assert kwargs["attempt_index"] == 1
        assert kwargs["attempt_claim_uri"] == published[0]
        identifier = kwargs["identifier"]
        create_calls.append(identifier)
        ordinal = spot.authorized_job_ids().index(identifier)
        role, hand = identifier.split("-hand-")
        return {
            "job_id": identifier,
            "source_role": role,
            "work_hand_indices": [int(hand)],
            "instance": f"{manifest['run_name']}-j{ordinal:02d}-a01",
            "zone": kwargs["zone"],
            "attempt_index": 1,
            "status": "created",
        }

    monkeypatch.setattr(spot, "_publish_once", publish_once)
    monkeypatch.setattr(spot, "_create_instance", create)

    result = spot.resume_jobs(run_dir=tmp_path, jobs=selected)

    assert result["attempt_index"] == 1
    assert result["selected_job_ids"] == list(selected)
    assert [row["instance"] for row in result["created"]] == [
        f"{manifest['run_name']}-j00-a01",
        f"{manifest['run_name']}-j19-a01",
    ]
    assert set(create_calls) == set(selected)
    assert result["third_attempt_authorized"] is False

    with pytest.raises(FileExistsError, match="already consumed"):
        spot.resume_jobs(run_dir=tmp_path, jobs=selected)
    assert len(published) == 1
    assert len(create_calls) == 2


def test_step6d_v2_launch_claim_permanently_blocks_crash_reuse(
    tmp_path: Path,
) -> None:
    manifest = {
        **_preflight_manifest(),
        "run_contract_digest": "f" * 64,
    }
    preflight = {
        "schema": spot.LAUNCH_PREFLIGHT_SCHEMA,
        "status": "quota_and_collision_checks_passed",
    }
    first = spot._acquire_launch_claim(
        target=tmp_path,
        manifest=manifest,
        selected=("candidate-hand-002",),
        preflight=preflight,
    )

    assert first["crash_reuse_authorized"] is False
    with pytest.raises(FileExistsError, match=spot.LAUNCH_CLAIM_NAME):
        spot._acquire_launch_claim(
            target=tmp_path,
            manifest=manifest,
            selected=("candidate-hand-006",),
            preflight=preflight,
        )


def test_step6d_v2_write_once_is_atomic_under_concurrent_authorization_writers(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "authorization.json"
    barrier = threading.Barrier(2)

    def write(value: int) -> bool:
        barrier.wait()
        try:
            spot._write_once(destination, {"writer": value})
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write, (1, 2)))

    assert sorted(outcomes) == [False, True]
    assert json.loads(destination.read_text(encoding="utf-8")) in (
        {"writer": 1},
        {"writer": 2},
    )


@pytest.mark.parametrize(
    "contract_variant",
    [
        runner.CANDIDATE01_VARIANT,
        runner.CANDIDATE02_VARIANT,
        runner.CANDIDATE02_TAIL_V2_VARIANT,
    ],
)
def test_step6d_v2_package_and_authorization_bind_every_immutable_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_variant: str,
) -> None:
    reference = tmp_path / "reference.so"
    candidate = tmp_path / "candidate.so"
    feature = tmp_path / "feature.so"
    reference.write_bytes(_elf(b"reference"))
    candidate.write_bytes(_elf(b"candidate"))
    feature.write_bytes(_elf(b"feature"))
    reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    feature_sha = hashlib.sha256(feature.read_bytes()).hexdigest()
    startup = tmp_path / spot.STARTUP_NAME
    startup.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n", encoding="utf-8")
    if contract_variant == runner.CANDIDATE02_TAIL_V2_VARIANT:
        selection = tmp_path / spot.CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH
        selection.parent.mkdir(parents=True)
        selection.write_bytes(
            (
                Path(__file__).resolve().parents[1]
                / spot.CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH
            ).read_bytes()
        )
    monkeypatch.setattr(spot, "REFERENCE_NATIVE_LIBRARY_SHA256", reference_sha)
    monkeypatch.setattr(spot, "EXPECTED_FEATURE_ENCODER_SHA256", feature_sha)
    monkeypatch.setattr(
        spot,
        "_REQUIRED_SOURCE_PATHS",
        frozenset(
            {
                spot.REFERENCE_PACKAGE_PATH,
                spot.CANDIDATE_PACKAGE_PATH,
                spot.FEATURE_PACKAGE_PATH,
            }
        ),
    )

    def fake_copy_source_tree(**kwargs):
        package_root = kwargs["package_root"]
        entries = {}
        for relative, source in (
            (spot.REFERENCE_PACKAGE_PATH, kwargs["reference_library"]),
            (spot.CANDIDATE_PACKAGE_PATH, kwargs["candidate_library"]),
            (spot.FEATURE_PACKAGE_PATH, kwargs["feature_encoder"]),
        ):
            entries[relative] = spot._copy_file(source, package_root / relative)
        selection = kwargs.get("selection_manifest")
        if selection is not None:
            relative = spot.CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH
            entries[relative] = spot._copy_file(
                selection,
                package_root / relative,
            )
        return entries

    monkeypatch.setattr(spot, "_copy_source_tree", fake_copy_source_tree)
    run_dir = tmp_path / "run"
    manifest = spot.package_step6d_v2(
        output_dir=run_dir,
        run_name="step6d-v2-unit-package",
        candidate_library=candidate,
        candidate_sha256=candidate_sha,
        reference_library=reference,
        reference_sha256=reference_sha,
        feature_encoder=feature,
        repository_root=tmp_path,
        startup_script=startup,
        contract_variant=contract_variant,
    )

    assert (
        manifest["run_contract"]["schema"]
        == {
            runner.CANDIDATE01_VARIANT: runner.RUN_CONTRACT_SCHEMA,
            runner.CANDIDATE02_VARIANT: runner.CANDIDATE02_RUN_CONTRACT_SCHEMA,
            runner.CANDIDATE02_TAIL_V2_VARIANT: (
                runner.CANDIDATE02_TAIL_V2_RUN_CONTRACT_SCHEMA
            ),
        }[contract_variant]
    )
    assert manifest["accepted_reference"]["sha256"] == reference_sha
    assert manifest["accepted_candidate"]["sha256"] == candidate_sha
    assert manifest["run_contract"]["allocation"] == {
        "workers": 1,
        "rayon_threads_per_worker": 16,
    }
    assert manifest["allocation"]["machine_type"] == "c4-standard-16"
    assert manifest["accepted_reference"]["package_path"] == (
        "native/reference/release/libofc_hu_m3_engine.so"
    )
    assert manifest["accepted_candidate"]["package_path"] == (
        "native/candidate/release/libofc_hu_m3_engine.so"
    )
    assert manifest["launch_target"] == spot.build_launch_target()
    assert manifest["cost_guard"] == spot.build_cost_guard()
    assert manifest["tail_schedule"]["logical_job_count"] == 20
    assert manifest["spot_execution_authorized"] is False
    if contract_variant == runner.CANDIDATE02_TAIL_V2_VARIANT:
        selection_entry = manifest["source_entries"][
            spot.CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH
        ]
        assert selection_entry == {
            "sha256": runner.CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256,
            "bytes": spot.CANDIDATE02_TAIL_V2_SELECTION_BYTES,
        }
        manifest_path = run_dir / spot.PACKAGE_MANIFEST_NAME
        original_manifest = manifest_path.read_bytes()
        tampered_manifest = json.loads(original_manifest)
        tampered_manifest["source_entries"][
            spot.CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH
        ]["sha256"] = ("0" * 64)
        manifest_path.write_bytes(spot.canonical_bytes(tampered_manifest))
        with pytest.raises(ValueError, match="selection binding changed"):
            spot.validate_package(run_dir)
        manifest_path.write_bytes(original_manifest)

    first_job_path = run_dir / manifest["job_manifests"][0]["path"]
    original_job = first_job_path.read_bytes()
    tampered_job = json.loads(original_job)
    tampered_job["run_contract_digest"] = "0" * 64
    first_job_path.write_bytes(spot.canonical_bytes(tampered_job))
    with pytest.raises(ValueError):
        spot.validate_package(run_dir)
    first_job_path.write_bytes(original_job)

    authorization = spot.authorize_launch(run_dir)
    assert authorization["authorized_job_ids"] == list(
        spot.authorized_job_ids(manifest["run_contract"])
    )
    assert authorization["run_contract_digest"] == manifest["run_contract_digest"]
    assert authorization["launch_target"] == manifest["launch_target"]
    assert authorization["cost_guard"] == manifest["cost_guard"]
    assert authorization["production_fanout_authorized"] is False

    authorization["training_eligible"] = True
    (run_dir / spot.AUTHORIZATION_NAME).write_bytes(spot.canonical_bytes(authorization))
    with pytest.raises(ValueError, match="authorization changed"):
        spot.validate_launch(run_dir)


def test_step6d_v2_status_and_receive_reject_unbound_remote_target_before_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _preflight_manifest()
    monkeypatch.setattr(spot, "validate_launch", lambda _path: (manifest, {}))
    monkeypatch.setattr(
        spot,
        "_subprocess_run",
        lambda *_args, **_kwargs: pytest.fail("status queried an unauthorized target"),
    )
    monkeypatch.setattr(
        spot,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("receive queried an unauthorized target"),
    )

    with pytest.raises(ValueError, match="remote target differs"):
        spot.cloud_status(run_dir=tmp_path, project="wrong-project")
    with pytest.raises(ValueError, match="remote target differs"):
        spot.receive_jobs(
            run_dir=tmp_path,
            output_dir=tmp_path / "received",
            bucket="wrong-bucket",
        )

    assert not (tmp_path / "received").exists()


def test_step6d_v2_receive_receipt_exactly_binds_target_and_result_prefix() -> None:
    contract = _contract()
    digest = runner.canonical_sha256(contract)
    manifest = {
        **_preflight_manifest(),
        "run_contract": contract,
        "run_contract_digest": digest,
    }
    jobs = [
        {
            "job_id": identifier,
            "source_role": identifier.split("-", 1)[0],
            "done_path": f"jobs/{identifier}/DONE.json",
            "run_contract_digest": digest,
        }
        for identifier in spot.authorized_job_ids()
    ]

    receipt = spot.build_receive_receipt(
        manifest=manifest,
        jobs=jobs,
        package_manifest_sha256="b" * 64,
        launch_authorization_sha256="c" * 64,
        launch_claim_sha256="d" * 64,
        launch_result_sha256="e" * 64,
    )

    assert set(receipt) == spot._RECEIVE_RECEIPT_KEYS
    assert receipt["launch_target"] == spot.build_launch_target()
    assert receipt["result_prefix"] == (
        "gs://pokerhu-ofc-solver-485418-training/runs/"
        "step6d-v2-preflight-unit/results"
    )
    changed = dict(receipt)
    changed["result_prefix"] += "/candidate-hand-002"
    with pytest.raises(ValueError, match="receive receipt changed"):
        spot.validate_receive_receipt(changed)
    changed = dict(receipt)
    changed["unexpected"] = False
    with pytest.raises(ValueError, match="keys changed"):
        spot.validate_receive_receipt(changed)


def test_candidate02_tail_v2_receipt_is_bound_to_its_validated_contract() -> None:
    contract = _candidate02_tail_v2_contract()
    digest = runner.canonical_sha256(contract)
    manifest = {
        **_preflight_manifest(),
        "run_contract": contract,
        "run_contract_digest": digest,
    }
    expected_ids = spot.authorized_job_ids(contract)
    jobs = [
        {
            "job_id": identifier,
            "source_role": identifier.split("-", 1)[0],
            "done_path": f"jobs/{identifier}/DONE.json",
            "run_contract_digest": digest,
        }
        for identifier in expected_ids
    ]

    receipt = spot.build_receive_receipt(
        manifest=manifest,
        jobs=jobs,
        package_manifest_sha256="b" * 64,
        launch_authorization_sha256="c" * 64,
        launch_claim_sha256="d" * 64,
        launch_result_sha256="e" * 64,
    )

    assert receipt["work_hand_indices"] == list(
        runner.CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES
    )
    assert [value["job_id"] for value in receipt["jobs"]] == list(expected_ids)
    assert spot.validate_receive_receipt(receipt, run_contract=contract) == receipt
    with pytest.raises(ValueError, match="requires its run contract"):
        spot.validate_receive_receipt(receipt)
    with pytest.raises(ValueError, match="receive receipt changed"):
        spot.validate_receive_receipt(receipt, run_contract=_candidate02_contract())


def test_step6d_v2_received_job_rejects_any_extra_file_before_content_read(
    tmp_path: Path,
) -> None:
    contract = _contract()
    value = spot.build_job_manifest(
        run_contract=contract, source_role="candidate", hand_index=2
    )
    package = tmp_path / "package"
    job_path = package / "jobs/candidate-hand-002.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_bytes(spot.canonical_bytes(value))
    job_dir = tmp_path / "received"
    for relative in spot._expected_received_relatives("candidate", 2):
        path = job_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (job_dir / "run.log").write_text("mutable\n", encoding="utf-8")
    record = {
        "job_id": "candidate-hand-002",
        "source_role": "candidate",
        "work_hand_indices": [2],
        "path": "jobs/candidate-hand-002.json",
    }

    with pytest.raises(ValueError, match="file set changed"):
        spot._validate_received_job(
            job_dir=job_dir, record=record, package_root=package
        )


@pytest.mark.parametrize(
    ("contract_factory", "expected_done_schema", "hand_index"),
    (
        (_contract, runner.DONE_SCHEMA, 50),
        (_candidate02_contract, runner.CANDIDATE02_DONE_SCHEMA, 50),
        (
            _candidate02_tail_v2_contract,
            runner.CANDIDATE02_TAIL_V2_DONE_SCHEMA,
            43,
        ),
    ),
)
def test_step6d_v2_received_job_delegates_semantic_and_done_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_factory,
    expected_done_schema: str,
    hand_index: int,
) -> None:
    contract = contract_factory()
    value = spot.build_job_manifest(
        run_contract=contract, source_role="reference", hand_index=hand_index
    )
    package = tmp_path / "package"
    identifier = f"reference-hand-{hand_index:03d}"
    job_path = package / f"jobs/{identifier}.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_bytes(spot.canonical_bytes(value))
    job_dir = tmp_path / "received"
    done = {
        "schema": expected_done_schema,
        "run_contract_digest": value["run_contract_digest"],
        "source_role": "reference",
        "work_hand_indices": [hand_index],
        "completed_hand_indices": [hand_index],
        "reference_library_sha256": contract["reference_library_sha256"],
        "candidate_library_sha256": contract["candidate_library_sha256"],
        "native_library_sha256": contract["reference_library_sha256"],
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    documents = {
        "DONE.json": done,
        f"hands/reference/hand_{hand_index:03d}.json": {},
        f"roots/hand_{hand_index:03d}.json": {},
        "run_contract.json": contract,
        "shard_manifest.json": value,
    }
    for relative, document in documents.items():
        path = job_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(spot.canonical_bytes(document))
    calls: list[str] = []

    def validate_completed_output(output_dir):
        assert Path(output_dir) == job_dir
        calls.append("completed")
        return json.loads((job_dir / "DONE.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(runner, "validate_completed_output", validate_completed_output)
    record = {
        "job_id": identifier,
        "source_role": "reference",
        "work_hand_indices": [hand_index],
        "path": f"jobs/{identifier}.json",
    }

    result = spot._validate_received_job(
        job_dir=job_dir, record=record, package_root=package
    )

    assert calls == ["completed"]
    assert result["source_role"] == "reference"
    assert result["work_hand_indices"] == [hand_index]
    assert result["run_contract_digest"] == runner.canonical_sha256(contract)

    if expected_done_schema != runner.DONE_SCHEMA:
        done["schema"] = runner.DONE_SCHEMA
        (job_dir / "DONE.json").write_bytes(spot.canonical_bytes(done))
        with pytest.raises(ValueError, match="DONE binding changed"):
            spot._validate_received_job(
                job_dir=job_dir, record=record, package_root=package
            )
        done["schema"] = expected_done_schema

    done["current_profile_changed"] = True
    (job_dir / "DONE.json").write_bytes(spot.canonical_bytes(done))
    with pytest.raises(ValueError, match="DONE binding changed"):
        spot._validate_received_job(
            job_dir=job_dir, record=record, package_root=package
        )
