from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_content_stage_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2


RUN_NAME = "regular-hu-m31-c02-f100wv2-stage-20260722-001"
SALT = "123456789abcdef0123456789abcdef0"
IMAGE = "sha256:" + "3" * 64
BUCKET = "ofc-m31-full100-stage-test"
NOW = "2026-07-22T03:00:00Z"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class FakeBackend:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.next_generation = 100
        self.fail_create_index: int | None = None
        self.create_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.list_complete = True

    def list_prefix(self, *, bucket: str, prefix: str) -> Mapping[str, Any]:
        rows = [
            deepcopy(row)
            for name, row in sorted(self.objects.items())
            if name.startswith(f"{prefix}/")
        ]
        return {
            "bucket": bucket,
            "prefix": prefix,
            "complete": self.list_complete,
            "objects": rows,
        }

    def create_object_from_file(
        self,
        *,
        bucket: str,
        object_name: str,
        source_path: str,
        sha256: str,
        bytes: int,
        if_generation_match: int,
    ) -> Mapping[str, Any]:
        call = {
            "bucket": bucket,
            "object_name": object_name,
            "source_path": source_path,
            "sha256": sha256,
            "bytes": bytes,
            "if_generation_match": if_generation_match,
        }
        self.create_calls.append(call)
        if self.fail_create_index == len(self.create_calls):
            raise RuntimeError("injected create failure")
        if if_generation_match != 0 or object_name in self.objects:
            raise FileExistsError(object_name)
        raw = Path(source_path).read_bytes()
        assert _sha(raw) == sha256
        assert len(raw) == bytes
        generation = str(self.next_generation)
        self.next_generation += 1
        remote = {
            "bucket": bucket,
            "object_name": object_name,
            "generation": generation,
            "sha256": sha256,
            "bytes": bytes,
        }
        self.objects[object_name] = remote
        return {
            **remote,
            "created": True,
            "if_generation_match": if_generation_match,
        }

    def delete_object(
        self,
        *,
        bucket: str,
        object_name: str,
        if_generation_match: str,
    ) -> Mapping[str, Any]:
        remote = self.objects[object_name]
        if remote["generation"] != if_generation_match:
            raise ValueError("generation precondition failed")
        call = {
            "bucket": bucket,
            "object_name": object_name,
            "if_generation_match": if_generation_match,
        }
        self.delete_calls.append(call)
        del self.objects[object_name]
        return {
            "bucket": bucket,
            "object_name": object_name,
            "generation": if_generation_match,
            "deleted": True,
            "if_generation_match": if_generation_match,
        }


@pytest.fixture()
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    package_root = tmp_path / "outer"
    package_root.mkdir()
    wave_plan = wave_v2.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256="4" * 64,
        image_digest=IMAGE,
    )
    relative_paths = [
        package_v2.SOURCE_PATH,
        package_v2.SCIENTIFIC_MANIFEST_PATH,
        package_v2.WHEELHOUSE_PATH,
        package_v2.WHEELHOUSE_MANIFEST_PATH,
        package_v2.STARTUP_PATH,
        package_v2.WAVE_PLAN_PATH,
        *[
            package_v2.JOB_PATH_TEMPLATE.format(job_id=job_id)
            for job_id in wave_plan["coverage"]["job_ids"]
        ],
    ]
    assert len(relative_paths) == 26
    content_payload_sha256 = "5" * 64
    content_prefix = (
        f"{package_v2.CONTENT_PREFIX_ROOT}/{content_payload_sha256}"
    )
    entries = []
    for index, relative in enumerate(relative_paths):
        raw = f"immutable-content-{index:02d}".encode("ascii")
        path = package_root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        entries.append(
            {
                "relative_path": relative,
                "object_name": f"{content_prefix}/{relative}",
                "kind": "job_manifest" if index >= 6 else f"content_{index}",
                "sha256": _sha(raw),
                "bytes": len(raw),
            }
        )
    manifest = {
        "run_name": wave_plan["run_name"],
        "execution_identity_sha256": wave_plan["execution_identity_sha256"],
        "wave_plan_sha256": wave_plan["schedule_sha256"],
        "manifest_sha256": "6" * 64,
        "content_payload_sha256": content_payload_sha256,
        "content_prefix": content_prefix,
        "entry_count": 26,
        "entries": entries,
    }
    calls: list[tuple[Path, str]] = []

    def validate_outer_package(
        package_dir: str | Path,
        supplied_wave_plan: Mapping[str, Any],
        *,
        expected_startup_sha256: str,
    ) -> dict[str, Any]:
        assert wave_v2.validate_wave_plan(supplied_wave_plan) == wave_plan
        calls.append((Path(package_dir), expected_startup_sha256))
        return deepcopy(manifest)

    monkeypatch.setattr(
        subject.package_v2, "validate_outer_package", validate_outer_package
    )
    stage_plan = subject.build_content_stage_plan(
        package_dir=package_root,
        wave_plan=wave_plan,
        expected_startup_sha256="7" * 64,
        bucket=BUCKET,
    )
    backend = FakeBackend()
    return {
        "package_root": package_root,
        "wave_plan": wave_plan,
        "manifest": manifest,
        "stage_plan": stage_plan,
        "backend": backend,
        "validation_calls": calls,
        "startup_sha": "7" * 64,
    }


def _preflight(fixture: dict[str, Any]) -> dict[str, Any]:
    return subject.build_preflight_absence_receipt(
        package_dir=fixture["package_root"],
        wave_plan=fixture["wave_plan"],
        expected_startup_sha256=fixture["startup_sha"],
        stage_plan=fixture["stage_plan"],
        backend=fixture["backend"],
        observed_at_utc=NOW,
    )


def _stage(
    fixture: dict[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    return subject.execute_content_stage(
        package_dir=fixture["package_root"],
        wave_plan=fixture["wave_plan"],
        expected_startup_sha256=fixture["startup_sha"],
        stage_plan=fixture["stage_plan"],
        preflight_receipt=preflight,
        backend=fixture["backend"],
        observed_at_utc=NOW,
    )


def _reseal(value: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result[digest_field] = subject.canonical_sha256(
        {key: val for key, val in result.items() if key != digest_field}
    )
    return result


def test_plan_validates_outer_package_first_and_freezes_exact_26(
    fixture: dict[str, Any],
) -> None:
    plan = fixture["stage_plan"]
    assert len(fixture["validation_calls"]) == 1
    assert plan["entry_count"] == 26
    assert len({row["object_name"] for row in plan["entries"]}) == 26
    assert plan["if_generation_match"] == 0
    assert plan["cloud_launch_authorized"] is False
    assert subject.validate_content_stage_plan(fixture["manifest"], plan) == plan


def test_plan_rejects_extra_missing_duplicate_and_path_traversal(
    fixture: dict[str, Any],
) -> None:
    manifest = deepcopy(fixture["manifest"])
    manifest["entries"].append(deepcopy(manifest["entries"][0]))
    with pytest.raises(ValueError, match="exactly 26"):
        subject.validate_content_stage_plan(manifest, fixture["stage_plan"])

    manifest = deepcopy(fixture["manifest"])
    manifest["entries"].pop()
    with pytest.raises(ValueError, match="exactly 26"):
        subject.validate_content_stage_plan(manifest, fixture["stage_plan"])

    manifest = deepcopy(fixture["manifest"])
    manifest["entries"][1] = deepcopy(manifest["entries"][0])
    with pytest.raises(ValueError, match="duplicated"):
        subject.validate_content_stage_plan(manifest, fixture["stage_plan"])

    manifest = deepcopy(fixture["manifest"])
    manifest["entries"][0]["relative_path"] = "../escape"
    with pytest.raises(ValueError, match="unsafe"):
        subject.validate_content_stage_plan(manifest, fixture["stage_plan"])


def test_preflight_rejects_existing_same_bytes_and_incomplete_listing(
    fixture: dict[str, Any],
) -> None:
    backend = fixture["backend"]
    first = fixture["stage_plan"]["entries"][0]
    backend.objects[first["object_name"]] = {
        "bucket": BUCKET,
        "object_name": first["object_name"],
        "generation": "8",
        "sha256": first["sha256"],
        "bytes": first["bytes"],
    }
    with pytest.raises(FileExistsError, match="never adopted"):
        _preflight(fixture)
    backend.objects.clear()
    backend.list_complete = False
    with pytest.raises(ValueError, match="incomplete"):
        _preflight(fixture)


def test_full_stage_uses_create_only_and_binds_all_readback_metadata(
    fixture: dict[str, Any],
) -> None:
    preflight = _preflight(fixture)
    receipt = _stage(fixture, preflight)
    assert receipt["stage_complete"] is True
    assert receipt["created_entry_count"] == 26
    assert len(receipt["rows"]) == 26
    assert all(call["if_generation_match"] == 0 for call in fixture["backend"].create_calls)
    assert subject.validate_stage_receipt(
        fixture["stage_plan"], preflight, receipt
    ) == receipt
    assert len(fixture["validation_calls"]) == 3


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("bucket", "wrong-bucket"),
        ("run_name", "wrong-run"),
        ("content_payload_sha256", "8" * 64),
    ],
)
def test_stage_receipt_rejects_resealed_wrong_context(
    fixture: dict[str, Any], field: str, replacement: str
) -> None:
    preflight = _preflight(fixture)
    receipt = _stage(fixture, preflight)
    forged = deepcopy(receipt)
    forged[field] = replacement
    forged = _reseal(forged, "receipt_sha256")
    with pytest.raises(ValueError, match="binding"):
        subject.validate_stage_receipt(fixture["stage_plan"], preflight, forged)


@pytest.mark.parametrize("field", ["generation", "sha256", "bytes"])
def test_stage_readback_rejects_wrong_generation_hash_or_bytes(
    fixture: dict[str, Any], field: str
) -> None:
    preflight = _preflight(fixture)
    receipt = _stage(fixture, preflight)
    first = receipt["rows"][0]
    remote = fixture["backend"].objects[first["object_name"]]
    if field == "generation":
        remote[field] = "999"
    elif field == "sha256":
        remote[field] = "9" * 64
    else:
        remote[field] += 1
    with pytest.raises(ValueError, match="generation/hash/bytes"):
        subject.build_stage_readback_receipt(
            stage_plan=fixture["stage_plan"],
            preflight_receipt=preflight,
            created_rows=receipt["rows"],
            backend=fixture["backend"],
            observed_at_utc=NOW,
        )


def test_stage_readback_rejects_extra_missing_and_duplicate_rows(
    fixture: dict[str, Any],
) -> None:
    preflight = _preflight(fixture)
    receipt = _stage(fixture, preflight)
    missing = receipt["rows"][:-1]
    with pytest.raises(ValueError, match="extra, missing"):
        subject.build_stage_readback_receipt(
            stage_plan=fixture["stage_plan"],
            preflight_receipt=preflight,
            created_rows=missing,
            backend=fixture["backend"],
            observed_at_utc=NOW,
        )
    duplicate = [*receipt["rows"], deepcopy(receipt["rows"][0])]
    with pytest.raises(ValueError, match="duplicated"):
        subject.build_stage_readback_receipt(
            stage_plan=fixture["stage_plan"],
            preflight_receipt=preflight,
            created_rows=duplicate,
            backend=fixture["backend"],
            observed_at_utc=NOW,
        )
    fixture["backend"].objects[
        f"{fixture['stage_plan']['content_prefix']}/unexpected"
    ] = {
        "bucket": BUCKET,
        "object_name": f"{fixture['stage_plan']['content_prefix']}/unexpected",
        "generation": "9999",
        "sha256": "a" * 64,
        "bytes": 1,
    }
    with pytest.raises(ValueError, match="extra, missing"):
        subject.build_stage_readback_receipt(
            stage_plan=fixture["stage_plan"],
            preflight_receipt=preflight,
            created_rows=receipt["rows"],
            backend=fixture["backend"],
            observed_at_utc=NOW,
        )


def test_partial_failure_returns_owned_receipt_and_exact_generation_cleanup(
    fixture: dict[str, Any],
) -> None:
    preflight = _preflight(fixture)
    fixture["backend"].fail_create_index = 4
    with pytest.raises(subject.ContentStageIncompleteError) as caught:
        _stage(fixture, preflight)
    partial = caught.value.partial_receipt
    assert partial is not None
    assert partial["stage_complete"] is False
    assert partial["created_entry_count"] == 3
    cleanup = subject.cleanup_staged_content(
        stage_plan=fixture["stage_plan"],
        preflight_receipt=preflight,
        stage_receipt=partial,
        backend=fixture["backend"],
        observed_at_utc="2026-07-22T03:01:00Z",
    )
    assert cleanup["all_objects_absent"] is True
    assert cleanup["delete_attempt_count"] == 3
    assert not fixture["backend"].objects
    assert [call["if_generation_match"] for call in fixture["backend"].delete_calls] == [
        row["generation"] for row in partial["rows"]
    ]
    assert subject.validate_cleanup_receipt(
        fixture["stage_plan"], preflight, partial, cleanup
    ) == cleanup


def test_cleanup_rejects_replaced_generation_and_unowned_extra(
    fixture: dict[str, Any],
) -> None:
    preflight = _preflight(fixture)
    receipt = _stage(fixture, preflight)
    first = receipt["rows"][0]
    fixture["backend"].objects[first["object_name"]]["generation"] = "999"
    with pytest.raises(ValueError, match="owned generation"):
        subject.cleanup_staged_content(
            stage_plan=fixture["stage_plan"],
            preflight_receipt=preflight,
            stage_receipt=receipt,
            backend=fixture["backend"],
            observed_at_utc=NOW,
        )
    fixture["backend"].objects[first["object_name"]]["generation"] = first["generation"]
    extra_name = f"{fixture['stage_plan']['content_prefix']}/extra"
    fixture["backend"].objects[extra_name] = {
        "bucket": BUCKET,
        "object_name": extra_name,
        "generation": "10000",
        "sha256": "b" * 64,
        "bytes": 1,
    }
    with pytest.raises(ValueError, match="unowned"):
        subject.cleanup_staged_content(
            stage_plan=fixture["stage_plan"],
            preflight_receipt=preflight,
            stage_receipt=receipt,
            backend=fixture["backend"],
            observed_at_utc=NOW,
        )


def test_cleanup_is_retry_safe_after_some_owned_objects_are_already_absent(
    fixture: dict[str, Any],
) -> None:
    preflight = _preflight(fixture)
    receipt = _stage(fixture, preflight)
    already_absent = receipt["rows"][0]
    del fixture["backend"].objects[already_absent["object_name"]]
    cleanup = subject.cleanup_staged_content(
        stage_plan=fixture["stage_plan"],
        preflight_receipt=preflight,
        stage_receipt=receipt,
        backend=fixture["backend"],
        observed_at_utc=NOW,
    )
    assert cleanup["already_absent_count"] == 1
    assert cleanup["delete_attempt_count"] == 25
    assert cleanup["wildcard_delete_used"] is False


def test_cleanup_receipt_rejects_resealed_wrong_generation(
    fixture: dict[str, Any],
) -> None:
    preflight = _preflight(fixture)
    receipt = _stage(fixture, preflight)
    cleanup = subject.cleanup_staged_content(
        stage_plan=fixture["stage_plan"],
        preflight_receipt=preflight,
        stage_receipt=receipt,
        backend=fixture["backend"],
        observed_at_utc=NOW,
    )
    forged = deepcopy(cleanup)
    forged["deleted_rows"][0]["generation"] = "999999"
    forged = _reseal(forged, "receipt_sha256")
    with pytest.raises(ValueError, match="exact owned generation"):
        subject.validate_cleanup_receipt(
            fixture["stage_plan"], preflight, receipt, forged
        )
