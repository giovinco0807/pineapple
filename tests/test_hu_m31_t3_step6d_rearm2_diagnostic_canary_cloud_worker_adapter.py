from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package as package_builder,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("r2diag-worker-adapter") / "package"
    package_builder.build_package(output_dir=target)
    return target


def _upload_only(
    preview: dict[str, object],
    snapshot: dict[str, object],
    *,
    sequence: int,
) -> dict[str, object]:
    job = preview["jobs"][0]
    upload = subject.build_fixture_upload(
        preview, job_id=job["job_id"], sequence=sequence
    )
    return subject.fake_publish(
        preview,
        snapshot,
        [subject._record(job["upload_uris"][sequence - 1], upload)],
    )


def _rebind_snapshot(
    preview: dict[str, object], snapshot: dict[str, object]
) -> dict[str, object]:
    value = deepcopy(snapshot)
    value["preview_sha256"] = subject.canonical_sha256(preview)
    value["attempt_index"] = preview["attempt_index"]
    return value


def _complete_job(
    preview: dict[str, object], snapshot: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    job = preview["jobs"][0]
    snapshot = subject.fixture_progress(
        preview,
        snapshot,
        job_id=job["job_id"],
        completed_count=len(job["work_hand_indices"]),
    )
    snapshot = subject.fixture_done(
        preview, snapshot, job_id=job["job_id"]
    )
    done = next(
        row["content"]
        for row in snapshot["objects"]
        if row["uri"] == job["done_uri"]
    )
    return snapshot, done


def test_preview_binds_exact_worker_package_and_enumerated_runner_tree(
    package: Path,
) -> None:
    preview = subject.build_preview(
        package_dir=package, stage_id=plan.STAGE1_ID
    )
    assert preview["package_file_count"] == 8
    assert preview["source_entry_count"] == 60
    assert preview["selected_job_ids"] == ["candidate-shard-00"]
    assert preview["stage_identity"]["retry_invariant"] is True
    assert preview["capabilities"]["cloud_executable"] is False
    assert preview["capabilities"]["launch_ready"] is False
    job = preview["jobs"][0]
    assert len(job["tree_object_manifest"]) == 23
    assert [row["path"] for row in job["tree_object_manifest"][:2]] == [
        "run_contract.json",
        "shard_manifest.json",
    ]
    assert job["tree_object_manifest"][-1]["path"] == "DONE.json"
    assert all(
        row["uri"].startswith(job["tree_prefix"] + "/")
        for row in job["tree_object_manifest"]
    )
    audit = subject.audit(package_dir=package)
    assert audit["package_file_count"] == 8
    assert audit["source_entry_count"] == 60
    assert audit["gcloud_callable"] is False
    assert audit["remote_write_callable"] is False


@pytest.mark.parametrize("attempt_index", [False, True, 0.0, 1.0, -1, 2])
def test_attempt_index_is_an_exact_bounded_integer(
    package: Path, attempt_index: object
) -> None:
    with pytest.raises(ValueError, match="attempt_index"):
        subject.build_preview(
            package_dir=package,
            stage_id=plan.STAGE1_ID,
            attempt_index=attempt_index,
        )


def test_upload_only_is_a_valid_trailing_crash_frontier(package: Path) -> None:
    preview = subject.build_preview(
        package_dir=package, stage_id=plan.STAGE1_ID
    )
    snapshot = _upload_only(
        preview, subject.empty_snapshot(preview), sequence=1
    )
    state = subject.validate_snapshot(preview, snapshot)["job_states"][0]
    assert state == {
        "job_id": "candidate-shard-00",
        "uploaded_sequences": 1,
        "completed_sequences": 0,
        "pending_heartbeat_sequence": 1,
        "state": "crashed_resumable",
    }
    job = preview["jobs"][0]
    forged = deepcopy(snapshot)
    forged["objects"].append(
        subject._record(
            job["upload_uris"][1],
            subject.build_fixture_upload(
                preview, job_id=job["job_id"], sequence=2
            ),
        )
    )
    with pytest.raises(ValueError, match="trailing frontier"):
        subject.validate_snapshot(preview, forged)


def test_attempt1_reuses_exact_tree_and_recovers_pending_heartbeat(
    package: Path,
) -> None:
    attempt0 = subject.build_preview(
        package_dir=package, stage_id=plan.STAGE1_ID
    )
    prior = _upload_only(
        attempt0, subject.empty_snapshot(attempt0), sequence=1
    )
    attempt1 = subject.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        attempt_index=1,
        prior_preview=attempt0,
        prior_snapshot=prior,
    )
    assert (
        attempt0["remote_manifest"]["result_prefix"]
        == attempt1["remote_manifest"]["result_prefix"]
    )
    assert (
        attempt0["remote_manifest"]["attempt_control_prefix"]
        != attempt1["remote_manifest"]["attempt_control_prefix"]
    )
    for key in (
        "upload_uris",
        "heartbeat_uris",
        "done_uri",
        "tree_prefix",
        "tree_object_manifest",
    ):
        assert attempt0["jobs"][0][key] == attempt1["jobs"][0][key]
    upload0 = prior["objects"][0]["content"]
    assert upload0 == subject.build_fixture_upload(
        attempt1, job_id="candidate-shard-00", sequence=1
    )

    resumed = _rebind_snapshot(attempt1, prior)
    resumed = subject.fixture_progress(
        attempt1,
        resumed,
        job_id="candidate-shard-00",
        completed_count=1,
    )
    state = subject.validate_snapshot(attempt1, resumed)["job_states"][0]
    assert state["completed_sequences"] == 1
    assert state["pending_heartbeat_sequence"] is None
    assert len(resumed["objects"]) == 2


def test_full_prefix_without_done_finishes_without_reupload(package: Path) -> None:
    attempt0 = subject.build_preview(
        package_dir=package, stage_id=plan.STAGE1_ID
    )
    job0 = attempt0["jobs"][0]
    prior = subject.fixture_progress(
        attempt0,
        subject.empty_snapshot(attempt0),
        job_id=job0["job_id"],
        completed_count=len(job0["work_hand_indices"]),
    )
    object_count = len(prior["objects"])
    attempt1 = subject.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        attempt_index=1,
        prior_preview=attempt0,
        prior_snapshot=prior,
    )
    resumed = _rebind_snapshot(attempt1, prior)
    resumed = subject.fixture_done(
        attempt1, resumed, job_id=job0["job_id"]
    )
    assert len(resumed["objects"]) == object_count + 1
    assert subject.validate_snapshot(attempt1, resumed)["job_states"][0][
        "state"
    ] == "success_tree_ready"


def test_done_exposes_exact_materialization_manifest_and_receive_is_stable(
    package: Path,
) -> None:
    attempt0 = subject.build_preview(
        package_dir=package, stage_id=plan.STAGE1_ID
    )
    snapshot0, done0 = _complete_job(
        attempt0, subject.empty_snapshot(attempt0)
    )
    materialization = subject.build_materialization_manifest(
        attempt0, done_record=done0
    )
    assert len(materialization["objects"]) == 23
    assert materialization["objects"][0]["path"] == "run_contract.json"
    assert materialization["objects"][-1]["path"] == "DONE.json"
    assert materialization["content_is_not_embedded_in_envelope"] is True
    assert materialization["validate_completed_output_required"] is True

    receive0 = subject.build_receive(attempt0, done_records=[done0])
    prior = subject.fixture_progress(
        attempt0,
        subject.empty_snapshot(attempt0),
        job_id="candidate-shard-00",
        completed_count=1,
    )
    attempt1 = subject.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        attempt_index=1,
        prior_preview=attempt0,
        prior_snapshot=prior,
    )
    assert subject.build_receive(attempt1, done_records=[done0]) == receive0
    assert "attempt_index" not in receive0
    assert "preview_sha256" not in receive0
    assert snapshot0["objects"][-1]["content"] == done0


def test_artifact_identity_rejects_bool_index_and_zero_sha(package: Path) -> None:
    preview = subject.build_preview(
        package_dir=package, stage_id=plan.STAGE1_ID
    )
    job = preview["jobs"][0]
    root = deepcopy(job["root_records"][0])
    root["hand_index"] = False
    with pytest.raises(ValueError, match="identity changed"):
        subject.build_artifact_upload(
            preview,
            job_id=job["job_id"],
            sequence=1,
            root_file=root,
            source_hand_file=subject._fixture_file(
                source_role=job["source_role"],
                hand_index=job["work_hand_indices"][0],
            ),
            transport_fixture_only=True,
        )
    root = deepcopy(job["root_records"][0])
    root["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        subject.build_artifact_upload(
            preview,
            job_id=job["job_id"],
            sequence=1,
            root_file=root,
            source_hand_file=subject._fixture_file(
                source_role=job["source_role"],
                hand_index=job["work_hand_indices"][0],
            ),
            transport_fixture_only=True,
        )


def test_adapter_exports_no_cloud_launch_or_write_capability() -> None:
    forbidden = (
        "gcloud",
        "create_vm",
        "launch",
        "write_claim",
        "write_authorization",
        "upload_remote",
    )
    assert all(not hasattr(subject, name) for name in forbidden)
    assert "build_materialization_manifest" in subject.__all__
