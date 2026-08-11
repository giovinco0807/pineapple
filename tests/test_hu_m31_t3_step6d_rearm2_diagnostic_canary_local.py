from __future__ import annotations

import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_local as subject,
)
from ofc_regular import hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan


def _payload(stage_id: str, job_id: str, index: int) -> dict[str, object]:
    return {
        "schema": "diagnostic_fake_solver_output_v1",
        "identity": f"{stage_id}|{job_id}|{index}",
        "portable_decision_sha256": hashlib.sha256(
            f"{stage_id}|{job_id}|{index}".encode("ascii")
        ).hexdigest(),
    }


def _run_job(
    *,
    package: Path,
    opened: Path,
    stage_id: str,
    job_id: str,
    output: Path,
    split_after: int | None = None,
) -> dict[str, object]:
    job = subject.initialize_local_job(
        package_dir=package,
        stage_open_path=opened,
        stage_id=stage_id,
        job_id=job_id,
        output_dir=output,
    )
    work = job["work_hand_indices"]
    first = work if split_after is None else work[:split_after]
    second = [] if split_after is None else work[split_after:]
    for index in first:
        subject.record_local_upload(
            package_dir=package,
            output_dir=output,
            hand_index=index,
            payload=_payload(stage_id, job_id, index),
        )
    if second:
        assert not (output / subject.DONE_NAME).exists()
        # Resume re-validates an already uploaded artifact and then continues.
        index = first[-1]
        subject.record_local_upload(
            package_dir=package,
            output_dir=output,
            hand_index=index,
            payload=_payload(stage_id, job_id, index),
        )
        for index in second:
            subject.record_local_upload(
                package_dir=package,
                output_dir=output,
                hand_index=index,
                payload=_payload(stage_id, job_id, index),
            )
    return subject.complete_local_job(
        package_dir=package,
        output_dir=output,
    )


def _full_lifecycle(
    base: Path,
    *,
    resume_stage2_candidate: bool,
) -> tuple[Path, Path, dict[str, object]]:
    package = base / "package"
    subject.build_local_package(output_dir=package)

    stage1_open = base / "stage1-open.json"
    subject.open_local_stage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        output_path=stage1_open,
    )
    stage1_job = base / "stage1-job"
    _run_job(
        package=package,
        opened=stage1_open,
        stage_id=plan.STAGE1_ID,
        job_id=plan.STAGE1_JOB_IDS[0],
        output=stage1_job,
    )
    stage1_receive = base / "stage1-received"
    subject.receive_stage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        job_outputs={plan.STAGE1_JOB_IDS[0]: stage1_job},
        destination=stage1_receive,
    )

    stage2_open = base / "stage2-open.json"
    subject.open_local_stage(
        package_dir=package,
        stage_id=plan.STAGE2_ID,
        output_path=stage2_open,
        prerequisite_receive_dir=stage1_receive,
    )
    stage2_outputs: dict[str, Path] = {}
    for job_id in plan.STAGE2_JOB_IDS:
        output = base / f"stage2-{job_id}"
        _run_job(
            package=package,
            opened=stage2_open,
            stage_id=plan.STAGE2_ID,
            job_id=job_id,
            output=output,
            split_after=3
            if resume_stage2_candidate and job_id == plan.STAGE2_JOB_IDS[0]
            else None,
        )
        stage2_outputs[job_id] = output
    stage2_receive = base / "stage2-received"
    subject.receive_stage(
        package_dir=package,
        stage_id=plan.STAGE2_ID,
        job_outputs=stage2_outputs,
        destination=stage2_receive,
    )
    aggregate = subject.aggregate_received_stages(
        package_dir=package,
        received_stages={
            plan.STAGE1_ID: stage1_receive,
            plan.STAGE2_ID: stage2_receive,
        },
    )
    return package, stage2_receive, aggregate


def test_stage1_stage2_happy_path_and_resume_are_byte_identical(
    tmp_path: Path,
) -> None:
    resumed_base = tmp_path / "resumed"
    clean_base = tmp_path / "clean"
    package_a, _receive_a, resumed = _full_lifecycle(
        resumed_base,
        resume_stage2_candidate=True,
    )
    package_b, _receive_b, clean = _full_lifecycle(
        clean_base,
        resume_stage2_candidate=False,
    )

    assert subject.canonical_bytes(resumed) == subject.canonical_bytes(clean)
    assert (resumed_base / "stage2-open.json").read_bytes() == (
        clean_base / "stage2-open.json"
    ).read_bytes()
    assert (
        resumed_base
        / f"stage2-{plan.STAGE2_JOB_IDS[0]}"
        / subject.DONE_NAME
    ).read_bytes() == (
        clean_base
        / f"stage2-{plan.STAGE2_JOB_IDS[0]}"
        / subject.DONE_NAME
    ).read_bytes()
    assert subject.sha256_file(package_a / subject.MANIFEST_NAME) == (
        subject.sha256_file(package_b / subject.MANIFEST_NAME)
    )
    assert resumed["diagnostic_only"] is True
    assert resumed["cloud_execution_performed"] is False
    assert resumed["performance_lock_evidence"] is False
    assert resumed["training_eligible"] is False


def test_stage2_open_binds_the_validated_prerequisite_receipt(
    tmp_path: Path,
) -> None:
    package, _received, _aggregate = _full_lifecycle(
        tmp_path / "run",
        resume_stage2_candidate=True,
    )
    stage2_open = tmp_path / "run" / "stage2-open.json"
    opened = json.loads(stage2_open.read_text(encoding="utf-8"))
    prerequisite = opened["prerequisite_receive_receipt"]
    assert opened["schema"] == subject.STAGE_OPEN_SCHEMA
    assert prerequisite["stage_id"] == plan.STAGE1_ID
    assert opened["prerequisite_receive_receipt_sha256"] == (
        subject.canonical_sha256(prerequisite)
    )

    forged_sha = deepcopy(opened)
    forged_sha["prerequisite_receive_receipt_sha256"] = "0" * 64
    forged_sha_path = tmp_path / "forged-stage2-sha.json"
    forged_sha_path.write_bytes(subject.canonical_bytes(forged_sha))
    with pytest.raises(ValueError, match="receipt hash changed"):
        subject.validate_stage_open(
            package_dir=package,
            stage_open_path=forged_sha_path,
        )
    startup = subject.run_startup_precontent(
        package_dir=package,
        stage_open_path=forged_sha_path,
        stage_id=plan.STAGE2_ID,
        job_id=plan.STAGE2_JOB_IDS[0],
    )
    assert startup.returncode != 0
    assert "invalid prerequisite receipt" in startup.stderr

    forged_receipt = deepcopy(opened)
    forged_receipt["prerequisite_receive_receipt"]["job_records"][0][
        "source_role"
    ] = "reference"
    forged_receipt["prerequisite_receive_receipt"][
        "job_record_aggregate_sha256"
    ] = subject.canonical_sha256(
        forged_receipt["prerequisite_receive_receipt"]["job_records"]
    )
    forged_receipt["prerequisite_receive_receipt_sha256"] = (
        subject.canonical_sha256(
            forged_receipt["prerequisite_receive_receipt"]
        )
    )
    forged_receipt_path = tmp_path / "forged-stage2-receipt.json"
    forged_receipt_path.write_bytes(subject.canonical_bytes(forged_receipt))
    with pytest.raises(ValueError, match="job binding changed"):
        subject.validate_stage_open(
            package_dir=package,
            stage_open_path=forged_receipt_path,
        )


def test_stage2_requires_validated_stage1_receipt(tmp_path: Path) -> None:
    package = tmp_path / "package"
    subject.build_local_package(output_dir=package)
    with pytest.raises(ValueError, match="requires the validated stage1"):
        subject.open_local_stage(
            package_dir=package,
            stage_id=plan.STAGE2_ID,
            output_path=tmp_path / "stage2-open.json",
        )


def test_precontent_startup_succeeds_with_poisoned_root_payloads(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    manifest = subject.build_local_package(output_dir=package)
    stage_open = tmp_path / "stage1-open.json"
    subject.open_local_stage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        output_path=stage_open,
    )

    poison_entries: list[tuple[str, bytes]] = []
    poison_records: list[dict[str, object]] = []
    with zipfile.ZipFile(package / subject.SOURCE_NAME) as archive:
        for info in archive.infolist():
            raw = archive.read(info.filename)
            if info.filename.startswith("roots/"):
                raw = b"POISON: startup must not parse this root\n"
            poison_entries.append((info.filename, raw))
            poison_records.append(
                {
                    "path": info.filename,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                }
            )
    poisoned_source = tmp_path / subject.SOURCE_NAME
    subject._write_deterministic_zip(poisoned_source, poison_entries)
    poisoned_manifest = deepcopy(manifest)
    poisoned_manifest["source"] = {
        "name": subject.SOURCE_NAME,
        "sha256": subject.sha256_file(poisoned_source),
        "bytes": poisoned_source.stat().st_size,
        "entries": poison_records,
    }
    poisoned_manifest_path = tmp_path / "poisoned-manifest.json"
    poisoned_manifest_path.write_bytes(subject.canonical_bytes(poisoned_manifest))
    opened = json.loads(stage_open.read_text(encoding="utf-8"))
    opened["package_manifest_sha256"] = subject.sha256_file(poisoned_manifest_path)
    poisoned_open = tmp_path / "poisoned-stage-open.json"
    poisoned_open.write_bytes(subject.canonical_bytes(opened))

    completed = subject.run_startup_precontent(
        package_dir=package,
        stage_open_path=poisoned_open,
        stage_id=plan.STAGE1_ID,
        job_id=plan.STAGE1_JOB_IDS[0],
        source_override=poisoned_source,
        manifest_override=poisoned_manifest_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith(plan.STAGE1_JOB_IDS[0])
    with zipfile.ZipFile(poisoned_source) as archive:
        with pytest.raises(json.JSONDecodeError):
            json.loads(archive.read("roots/hand_000.json"))


def test_out_of_order_upload_and_received_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    package, received, _aggregate = _full_lifecycle(
        tmp_path / "run",
        resume_stage2_candidate=True,
    )
    stage1_open = tmp_path / "out-of-order-open.json"
    subject.open_local_stage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        output_path=stage1_open,
    )
    job_dir = tmp_path / "out-of-order-job"
    job = subject.initialize_local_job(
        package_dir=package,
        stage_open_path=stage1_open,
        stage_id=plan.STAGE1_ID,
        job_id=plan.STAGE1_JOB_IDS[0],
        output_dir=job_dir,
    )
    index = job["work_hand_indices"][1]
    with pytest.raises(ValueError, match="frozen work-hand order"):
        subject.record_local_upload(
            package_dir=package,
            output_dir=job_dir,
            hand_index=index,
            payload=_payload(plan.STAGE1_ID, plan.STAGE1_JOB_IDS[0], index),
        )

    upload = (
        received
        / "jobs"
        / plan.STAGE2_JOB_IDS[0]
        / "uploads"
        / f"hand_{plan.STAGE2_HAND_INDICES[0]:03d}.json"
    )
    changed = json.loads(upload.read_text(encoding="utf-8"))
    changed["payload"]["identity"] = "tampered"
    upload.write_bytes(subject.canonical_bytes(changed))
    with pytest.raises(ValueError, match="upload changed|completed job changed"):
        subject.validate_received_stage(
            package_dir=package,
            receive_dir=received,
            stage_id=plan.STAGE2_ID,
        )
