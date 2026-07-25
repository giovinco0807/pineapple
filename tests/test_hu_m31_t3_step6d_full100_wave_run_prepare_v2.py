from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_run_prepare_v2 as subject
from ofc_regular.hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
)
from scripts import prepare_hu_m31_t3_step6d_full100_wave_run_v2 as cli


RUN_NAME = "regular-hu-m31-c02-f100wv2-20260723-006"
SALT = "abcdef0123456789abcdef0123456789"
IMAGE = "sha256:" + "3" * 64
OBSERVED = datetime(2026, 7, 23, 2, 0, 0, tzinfo=timezone.utc)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture()
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    science_root = tmp_path / "science"
    science_root.mkdir()
    source_name = subject.scientific.SOURCE_NAME
    source_raw = b"frozen-scientific-payload-for-run006"
    (science_root / source_name).write_bytes(source_raw)
    (science_root / "manifest.json").write_bytes(b"{}")
    (science_root / "PACKAGE_READY.json").write_bytes(b"{}")
    provisional = subject.wave_v2.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256=_sha(source_raw),
        image_digest=IMAGE,
    )
    jobs = []
    for frozen in provisional["full100_plan"]["jobs"]:
        relative = f"jobs/{frozen['job_id']}.json"
        raw = subject.scientific.canonical_bytes(
            subject.scientific._job_manifest(
                provisional["full100_plan"], frozen
            )
        )
        path = science_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        jobs.append(
            {
                "job_id": frozen["job_id"],
                "source_role": frozen["source_role"],
                "shard_index": frozen["shard_index"],
                "work_hand_indices": frozen["work_hand_indices"],
                "path": relative,
                "output_prefix": f"jobs/{frozen['job_id']}",
                "sha256": _sha(raw),
                "bytes": len(raw),
            }
        )
    science_manifest = {
        "schema": subject.scientific.PACKAGE_SCHEMA,
        "run_name": "regular-hu-m31-c02-full100-dev-20260717-002",
        "source_name": source_name,
        "source_sha256": _sha(source_raw),
        "source_bytes": len(source_raw),
        "plan_sha256": provisional["full100_plan_sha256"],
        "run_contract_digest": provisional["run_contract_digest"],
        "job_manifests": jobs,
    }
    (science_root / "manifest.json").write_bytes(
        subject.scientific.canonical_bytes(science_manifest)
    )
    monkeypatch.setattr(
        subject.scientific,
        "validate_package",
        lambda path: deepcopy(science_manifest),
    )
    wheelhouse = tmp_path / "wheelhouse.zip"
    wheelhouse.write_bytes(b"run005-offline-wheelhouse-is-allowed")
    wheelhouse_manifest = tmp_path / "wheelhouse_manifest.json"
    wheelhouse_manifest.write_bytes(
        subject.package_v2.perf_cloud.canonical_bytes(
            {
                "schema": "test-wheelhouse",
                "status": "fixture",
                "entries": [],
                "requirements_sha256": "4" * 64,
            }
        )
    )
    monkeypatch.setattr(
        subject.package_v2.perf_cloud,
        "_validate_wheelhouse_archive",
        lambda archive, manifest: None,
    )
    startup = (
        subject.science_registry.descriptor_for_kind(
            subject.science_registry.DEVELOPMENT_SCIENCE_KIND
        ).resolved_startup_path()
    )
    return {
        "tmp": tmp_path,
        "science": science_root,
        "wheelhouse": wheelhouse,
        "wheelhouse_manifest": wheelhouse_manifest,
        "startup": startup,
        "startup_sha": _sha(startup.read_bytes()),
    }


def _phase_a(
    inputs: Mapping[str, Any], *, name: str = "phase-a",
    execution_scope: str = subject.wave_v2.FULL100_EXECUTION_SCOPE,
) -> Path:
    output = inputs["tmp"] / name
    subject.prepare_phase_a(
        output_dir=output,
        run_name=RUN_NAME,
        identity_salt=SALT,
        scientific_package_dir=inputs["science"],
        startup_script=inputs["startup"],
        wheelhouse_archive=inputs["wheelhouse"],
        wheelhouse_manifest=inputs["wheelhouse_manifest"],
        image_digest=IMAGE,
        expected_startup_sha256=inputs["startup_sha"],
        execution_scope=execution_scope,
    )
    return output


class FakeAbsenceHttp:
    def __init__(self, *, collision_suffix: str | None = None) -> None:
        self.collision_suffix = collision_suffix
        self.calls: list[tuple[str, str, Mapping[str, str], bytes | None, int]] = []

    def __call__(
        self, method: str, url: str, headers: Mapping[str, str],
        body: bytes | None, timeout: int,
    ) -> HttpResponse:
        self.calls.append((method, url, dict(headers), body, timeout))
        status = 200 if self.collision_suffix and url.endswith(self.collision_suffix) else 404
        return HttpResponse(status=status, body=b'{"provider":"secret"}', headers={})


def _absence(
    phase_a: Path, monkeypatch: pytest.MonkeyPatch,
    *, fake: FakeAbsenceHttp | None = None,
) -> tuple[dict[str, Any], FakeAbsenceHttp]:
    prepared = subject.validate_phase_a(
        phase_a,
        expected_startup_sha256=_sha(
            (phase_a / "outer_package/content/startup/"
             "startup_hu_m31_t3_step6d_full100_wave_v2.sh").read_bytes()
        ),
    )
    requester = fake or FakeAbsenceHttp()
    token = "fixture-token-that-is-never-persisted-123"
    monkeypatch.setenv(subject.TOKEN_ENV, token)
    receipt = subject.collect_all_owned_absence(
        wave_plan=prepared["plan"], requester=requester, clock=lambda: OBSERVED
    )
    assert token not in subject.canonical_bytes(receipt).decode("ascii")
    return receipt, requester


def test_phase_a_creates_only_fresh_plan_package_and_stage_plan(
    inputs: Mapping[str, Any],
) -> None:
    root = _phase_a(inputs)
    validated = subject.validate_phase_a(
        root, expected_startup_sha256=inputs["startup_sha"]
    )
    assert {path.name for path in root.iterdir()} == {
        subject.WAVE_PLAN_NAME,
        subject.OUTER_PACKAGE_NAME,
        subject.CONTENT_STAGE_PLAN_NAME,
        subject.PHASE_A_RECEIPT_NAME,
    }
    receipt = validated["receipt"]
    assert receipt["identity_salt_persisted"] is False
    assert receipt["prior_claim_reused"] is False
    assert receipt["prior_control_state_reused"] is False
    assert receipt["prior_random_material_reused"] is False
    assert receipt["prior_iam_handoff_reused"] is False
    assert receipt["prior_content_handoff_reused"] is False
    assert receipt["initial_transition_created"] is False
    assert receipt["attempt_ledger_created"] is False
    assert receipt["resume_plan_created"] is False
    assert validated["plan"]["run_name"] != subject.RUN005_RUN_NAME
    assert (
        validated["outer_manifest"]["content_payload_sha256"]
        != subject.RUN005_CONTENT_PAYLOAD_SHA256
    )
    for path in root.rglob("*"):
        if path.is_file():
            assert SALT.encode() not in path.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        _phase_a(inputs)


def test_real_v4_package_phase_a_uses_plan_derived_scope_and_startup(
    inputs: Mapping[str, Any],
) -> None:
    descriptor = subject.science_registry.descriptor_for_kind(
        subject.science_registry.PERFORMANCE_LOCK_V4_SCIENCE_KIND
    )
    frozen = subject.wave_v2._read_frozen_plan(
        descriptor.resolved_default_plan_path()
    )
    output = inputs["tmp"] / "v4-phase-a"
    receipt = subject.prepare_phase_a(
        output_dir=output,
        run_name="regular-hu-m31-c02-f100wv2-20260723-v4phasea",
        identity_salt=SALT,
        scientific_package_dir=(
            descriptor._package_module().DEFAULT_PACKAGE_DIR
        ),
        startup_script=descriptor.resolved_startup_path(),
        wheelhouse_archive=inputs["wheelhouse"],
        wheelhouse_manifest=inputs["wheelhouse_manifest"],
        image_digest=frozen["source_identity"]["image_digest"],
        full100_plan_path=descriptor.resolved_default_plan_path(),
    )
    validated = subject.validate_phase_a(output)
    assert validated["receipt"] == receipt
    assert validated["plan"]["scope"] == descriptor.execution_scope
    assert validated["outer_manifest"]["expected_startup_sha256"] == (
        descriptor.startup_sha256
    )
    assert receipt["startup_sha256"] == descriptor.startup_sha256
    assert receipt["cloud_mutated"] is False


def test_phase_a_rejects_run005_identity_and_extra_handoff_file(
    inputs: Mapping[str, Any],
) -> None:
    with pytest.raises(ValueError, match="run005 execution identity"):
        subject.prepare_phase_a(
            output_dir=inputs["tmp"] / "forbidden",
            run_name=subject.RUN005_RUN_NAME,
            identity_salt=SALT,
            scientific_package_dir=inputs["science"],
            startup_script=inputs["startup"],
            wheelhouse_archive=inputs["wheelhouse"],
            wheelhouse_manifest=inputs["wheelhouse_manifest"],
            image_digest=IMAGE,
            expected_startup_sha256=inputs["startup_sha"],
        )
    root = _phase_a(inputs, name="extra-handoff")
    (root / "launch_claim.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="extra or missing"):
        subject.validate_phase_a(
            root, expected_startup_sha256=inputs["startup_sha"]
        )


def test_absence_collector_gets_all_40_instance_and_disk_names(
    inputs: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _phase_a(inputs)
    receipt, fake = _absence(root, monkeypatch)
    assert receipt["planned_attempt_count"] == 40
    assert receipt["http_get_count"] == 80
    assert len(fake.calls) == 80
    assert all(method == "GET" and body is None for method, _, _, body, _ in fake.calls)
    assert len({row["instance_name"] for row in receipt["rows"]}) == 40
    assert {row["wave_index"] for row in receipt["rows"]} == {0, 1, 2}
    assert {row["attempt_id"] for row in receipt["rows"]} == {"a00", "a01"}


def test_absence_collector_checks_every_name_then_rejects_any_collision(
    inputs: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _phase_a(inputs)
    prepared = subject.validate_phase_a(
        root, expected_startup_sha256=inputs["startup_sha"]
    )
    collision = prepared["plan"]["waves"][2]["candidate_reference_pairs"][1][
        "reference_attempt_instance_ids"
    ]["a01"]
    fake = FakeAbsenceHttp(collision_suffix=f"/disks/{collision}")
    token = "fixture-token-that-is-never-persisted-456"
    monkeypatch.setenv(subject.TOKEN_ENV, token)
    with pytest.raises(FileExistsError, match="not all absent") as error:
        subject.collect_all_owned_absence(
            wave_plan=prepared["plan"], requester=fake, clock=lambda: OBSERVED
        )
    assert len(fake.calls) == 80
    assert token not in str(error.value)
    assert "provider" not in str(error.value)


def test_phase_b_only_materializes_from_fresh_exact_absence(
    inputs: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_a = _phase_a(inputs)
    absence, _ = _absence(phase_a, monkeypatch)
    phase_b = inputs["tmp"] / "phase-b"
    result = subject.finalize_phase_b(
        phase_a_dir=phase_a,
        output_dir=phase_b,
        absence_receipt=absence,
        current_time_utc="2026-07-23T02:00:01Z",
        expected_startup_sha256=inputs["startup_sha"],
    )
    validated = subject.validate_phase_b(
        phase_a_dir=phase_a,
        phase_b_dir=phase_b,
        expected_startup_sha256=inputs["startup_sha"],
    )
    assert result == validated["receipt"]
    assert validated["initial_transition"]["readback_source"] == "gcloud_readback"
    assert validated["initial_transition"]["owned_vms"] == []
    assert validated["attempt_ledger"]["cloud_claim_persistent"] is False
    assert validated["resume_plan"]["resume_wave_index"] == 0
    assert len(validated["resume_plan"]["selected_attempts"]) == 8
    assert result["all_planned_attempt_names_absent"] is True
    assert result["prior_claim_reused"] is False
    assert result["prior_iam_handoff_reused"] is False


def test_startup_canary_phase_b_keeps_full_absence_and_selects_exactly_one_a00(
    inputs: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_a = _phase_a(
        inputs,
        name="canary-phase-a",
        execution_scope=subject.wave_v2.STARTUP_CANARY_SCOPE,
    )
    prepared = subject.validate_phase_a(
        phase_a, expected_startup_sha256=inputs["startup_sha"]
    )
    plan = subject.wave_v2.validate_startup_canary_plan(prepared["plan"])
    assert prepared["receipt"]["status"] == (
        "fresh_startup_canary_plan_outer_package_and_stage_plan_only"
    )
    assert plan["wave_vm_counts"] == [8, 8, 4]
    assert plan["coverage"]["logical_job_count"] == 20

    absence, fake = _absence(phase_a, monkeypatch)
    assert absence["planned_attempt_count"] == 40
    assert absence["http_get_count"] == 80
    assert len(fake.calls) == 80

    phase_b = inputs["tmp"] / "canary-phase-b"
    result = subject.finalize_phase_b(
        phase_a_dir=phase_a,
        output_dir=phase_b,
        absence_receipt=absence,
        current_time_utc="2026-07-23T02:00:01Z",
        expected_startup_sha256=inputs["startup_sha"],
    )
    validated = subject.validate_phase_b(
        phase_a_dir=phase_a,
        phase_b_dir=phase_b,
        expected_startup_sha256=inputs["startup_sha"],
    )
    selected = validated["resume_plan"]["selected_attempts"]
    assert [(row["job_id"], row["source_role"], row["attempt_id"]) for row in selected] == [
        ("candidate-shard-00", "candidate", "a00")
    ]
    assert "/startup-canary/results/jobs/candidate-shard-00/attempts/a00" in (
        selected[0]["artifact_prefix"]
    )
    assert result["selected_attempt_count"] == 1
    assert result["status"] == (
        "fresh_all_owned_absence_bound_single_attempt_startup_canary"
    )


def test_phase_b_rejects_stale_or_tampered_absence_without_output(
    inputs: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_a = _phase_a(inputs)
    absence, _ = _absence(phase_a, monkeypatch)
    stale_output = inputs["tmp"] / "stale-phase-b"
    with pytest.raises(PermissionError, match="stale"):
        subject.finalize_phase_b(
            phase_a_dir=phase_a,
            output_dir=stale_output,
            absence_receipt=absence,
            current_time_utc="2026-07-23T02:05:01Z",
            expected_startup_sha256=inputs["startup_sha"],
        )
    assert not stale_output.exists()
    tampered = deepcopy(absence)
    tampered["rows"][0]["boot_disk_http_status"] = 200
    unsigned = {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    tampered["receipt_sha256"] = subject.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="absence is incomplete"):
        subject.finalize_phase_b(
            phase_a_dir=phase_a,
            output_dir=inputs["tmp"] / "tampered-phase-b",
            absence_receipt=tampered,
            current_time_utc="2026-07-23T02:00:01Z",
            expected_startup_sha256=inputs["startup_sha"],
        )


def test_cli_requires_environment_only_identity_salt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "phase-a", "--output-dir", str(tmp_path / "out"),
        "--run-name", RUN_NAME, "--scientific-package-dir", str(tmp_path),
        "--wheelhouse-archive", str(tmp_path / "wheel.zip"),
        "--wheelhouse-manifest", str(tmp_path / "wheel.json"),
        "--image-digest", IMAGE,
    ]
    monkeypatch.delenv(subject.IDENTITY_SALT_ENV, raising=False)
    with pytest.raises(PermissionError, match=subject.IDENTITY_SALT_ENV):
        cli.main(argv)
    captured: dict[str, Any] = {}
    monkeypatch.setenv(subject.IDENTITY_SALT_ENV, SALT)

    def fake_prepare(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"schema": subject.PHASE_A_SCHEMA, "current_profile_changed": False}

    monkeypatch.setattr(subject, "prepare_phase_a", fake_prepare)
    assert cli.main(argv) == 0
    assert captured["identity_salt"] == SALT
    assert captured["execution_scope"] is None
    assert captured["expected_startup_sha256"] is None
    assert Path(captured["startup_script"]).resolve() == (
        subject.science_registry.descriptor_for_kind(
            subject.science_registry.DEVELOPMENT_SCIENCE_KIND
        ).resolved_startup_path()
    )
    assert SALT not in capsys.readouterr().out


def test_cli_startup_canary_is_an_explicit_phase_a_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = [
        "phase-a", "--output-dir", str(tmp_path / "out"),
        "--run-name", RUN_NAME, "--scientific-package-dir", str(tmp_path),
        "--wheelhouse-archive", str(tmp_path / "wheel.zip"),
        "--wheelhouse-manifest", str(tmp_path / "wheel.json"),
        "--image-digest", IMAGE, "--startup-canary",
    ]
    captured: dict[str, Any] = {}
    monkeypatch.setenv(subject.IDENTITY_SALT_ENV, SALT)

    def fake_prepare(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"schema": subject.PHASE_A_SCHEMA, "current_profile_changed": False}

    monkeypatch.setattr(subject, "prepare_phase_a", fake_prepare)
    assert cli.main(argv) == 0
    assert captured["execution_scope"] == subject.wave_v2.STARTUP_CANARY_SCOPE
