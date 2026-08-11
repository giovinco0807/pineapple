from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_contract as contract_v1,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_local_package as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_preflight as preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "regular-hu-m31-c02-perfdev-v2-20260722-test01"
OBSERVED_AT = "2026-07-22T12:00:00Z"
OBSERVED_UNIX = int(
    datetime.fromisoformat(OBSERVED_AT[:-1] + "+00:00").timestamp()
)


def _image() -> dict:
    name = "debian-12-bookworm-v20260721"
    return {
        "schema": preflight.IMAGE_OBSERVATION_SCHEMA,
        "observation_id": "image-package-fixture-20260722",
        "observed_at_utc": OBSERVED_AT,
        "project": "debian-cloud",
        "name": name,
        "id": "9021508813201755912",
        "selfLink": (
            "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
            f"global/images/{name}"
        ),
        "status": "READY",
        "deprecation": {"state": "ACTIVE", "replacement": None},
        "guest_os_features": ["GVNIC", "UEFI_COMPATIBLE", "VIRTIO_SCSI_MULTIQUEUE"],
        "read_only": True,
    }


def _runtime(run_name: str = RUN_NAME) -> dict:
    return {
        "schema": preflight.RUNTIME_OBSERVATION_SCHEMA,
        "observation_id": "runtime-package-fixture-20260722",
        "observed_at_utc": OBSERVED_AT,
        "project": "ofc-solver-485418",
        "region": "asia-northeast1",
        "zone": "asia-northeast1-b",
        "machine_type": {
            "name": "c4-standard-16",
            "guest_cpus": 16,
            "memory_mb": 61_440,
        },
        "spot_price": {
            "machine_type": "c4-standard-16",
            "provisioning_model": "SPOT",
            "currency": "USD",
            "unit": "vm_hour",
            "value": "0.53146",
        },
        "quota": {
            "c4_cpus": {"limit": 128, "usage": 0},
            "spot_cpus": {"limit": 468, "usage": 32},
        },
        "namespace": {
            "run_name": run_name,
            "identity_namespace": "perfdev-v2-20260722-test01",
            "result_prefix": f"hu-m31-t3/perfdev-v2/{run_name}/",
            "run_name_collision_count": 0,
            "identity_collision_count": 0,
            "result_prefix_collision_count": 0,
            "inventory_read_only": True,
        },
        "read_only": True,
        "cloud_mutated": False,
    }


def _write_receipt(path: Path, *, run_name: str = RUN_NAME) -> Path:
    receipt = preflight.build_dry_run_receipt(
        image_observation=_image(), runtime_observation=_runtime(run_name)
    )
    path.write_bytes(subject.canonical_bytes(receipt))
    return path


def _tiny_tooling(directory: Path) -> list[tuple[str, Path]]:
    directory.mkdir(parents=True, exist_ok=True)
    first = directory / "runner.py"
    second = directory / "requirements.txt"
    first.write_text("VALUE = 1\n", encoding="utf-8")
    second.write_text("numpy==2.3.1\n", encoding="utf-8")
    return [("src/ofc_regular/runner.py", first), ("configs/requirements.txt", second)]


def _package(
    tmp_path: Path,
    *,
    run_name: str = RUN_NAME,
    root_dir: Path = subject.DEFAULT_ROOT_DIR,
    candidate: Path = subject.DEFAULT_CANDIDATE_PATH,
    selection: Path = subject.DEFAULT_TAIL_SELECTION_MANIFEST_PATH,
    receipt: Path | None = None,
    tooling: list[tuple[str, Path]] | None = None,
    output_parent: Path | None = None,
) -> tuple[Path, dict]:
    receipt_path = receipt or _write_receipt(tmp_path / "receipt.json", run_name=run_name)
    tooling_values = tooling or _tiny_tooling(tmp_path / "tiny-tooling")
    parent = output_parent or tmp_path / "packages"
    result = subject.package_local(
        output_parent=parent,
        run_name=run_name,
        dry_run_receipt_path=receipt_path,
        candidate_library=candidate,
        tail_selection_manifest_path=selection,
        root_dir=root_dir,
        repository_root=REPO_ROOT,
        now_unix_seconds=OBSERVED_UNIX,
        tooling_sources=tooling_values,
        fixture_only=True,
    )
    return parent / run_name, result


@pytest.fixture(scope="module")
def valid_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("perfdev-v2-local-package")
    package, result = _package(directory)
    assert result["status"] == subject.PACKAGE_STATUS
    return package


def _copy_roots(tmp_path: Path) -> Path:
    target = tmp_path / "roots"
    shutil.copytree(subject.DEFAULT_ROOT_DIR, target)
    return target


def _symlink(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")


def test_source_inventory_rehashes_real_accepted_inputs() -> None:
    inventory = subject.audit_source_inventory()
    assert inventory["root_count"] == 100
    assert inventory["root_set_sha256"] == contract_v1.DEVELOPMENT_ROOT_SET_SHA256
    assert (
        inventory["root_topology_sha256"]
        == contract_v1.DEVELOPMENT_ROOT_TOPOLOGY_SHA256
    )
    assert inventory["primary"]["candidate"]["sha256"] == (
        contract_v1.CANDIDATE_LIBRARY_SHA256
    )
    assert inventory["primary"]["reference"]["sha256"] == (
        contract_v1.REFERENCE_LIBRARY_SHA256
    )
    assert inventory["primary"]["selection"]["sha256"] == (
        contract_v1.TAIL_SELECTION_MANIFEST_SHA256
    )
    assert inventory["run_contract_digest"] == contract_v1.TAIL_RUN_CONTRACT_DIGEST
    assert inventory["full100_plan_run_contract_digest"] == (
        contract_v1.FULL_RUN_CONTRACT_DIGEST
    )
    assert inventory["rearm2_overlap"]["roots"]["overlap_count"] == 0
    assert inventory["rearm2_overlap"]["roots"][
        "individual_rearm2_root_hashes_compared"
    ] is True
    assert inventory["rearm2_overlap"]["roots"][
        "individual_file_hash_overlap_count"
    ] == 0
    assert inventory["rearm2_overlap"]["seeds"]["overlap_count"] == 0
    assert inventory["cloud_executable"] is False
    assert inventory["launch_authorized"] is False


def test_local_package_is_postpackage_validated_and_non_executable(
    valid_package: Path,
) -> None:
    result = subject.validate_local_package(valid_package)
    assert result["status"] == "package_ready_local_not_authorized"
    assert result["root_count"] == 100
    assert result["tail_hand_indices"] == [0, 4, 5, 12, 14, 16, 17, 23, 41, 43]
    assert result["run_contract_digest"] == contract_v1.TAIL_RUN_CONTRACT_DIGEST
    assert result["selection_manifest_sha256"] == (
        contract_v1.TAIL_SELECTION_MANIFEST_SHA256
    )
    assert result["source_roles"] == ["candidate", "reference"]
    assert result["source_bytes_equal_packaged_bytes"] is True
    assert result["rearm2_overlap_count"] == 0
    assert result["cloud_executable"] is False
    assert result["launch_authorized"] is False
    startup = valid_package / subject.STARTUP_PACKAGE_PATH
    assert startup.read_text(encoding="utf-8").endswith("exit 97\n")
    selection = valid_package / subject.TAIL_SELECTION_MANIFEST_PACKAGE_PATH
    assert selection.stat().st_size == contract_v1.TAIL_SELECTION_MANIFEST_BYTES
    assert subject.sha256_file(selection) == contract_v1.TAIL_SELECTION_MANIFEST_SHA256
    for role in contract_v1.SOURCE_ROLES:
        manifest = json.loads(
            (valid_package / subject.ROLE_PACKAGE_PATHS[role]).read_text("utf-8")
        )
        assert manifest["run_contract_digest"] == contract_v1.TAIL_RUN_CONTRACT_DIGEST
        assert manifest["run_contract"]["tail_hand_indices"] == list(
            contract_v1.TAIL_HAND_INDICES
        )
        assert manifest["run_contract"]["selection_manifest_sha256"] == (
            contract_v1.TAIL_SELECTION_MANIFEST_SHA256
        )


def test_extracted_validator_uses_packaged_selection_without_checkout_configs(
    valid_package: Path, tmp_path: Path
) -> None:
    isolated = tmp_path / "isolated-runtime"
    shutil.copytree(
        REPO_ROOT / "src" / "ofc_regular",
        isolated / "src" / "ofc_regular",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    assert not (isolated / "configs").exists()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(isolated / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib; "
                "from ofc_regular import "
                "hu_m31_t3_step6d_performance_development_v2_local_package as p; "
                "r=p.validate_local_package(pathlib.Path(__import__('sys').argv[1])); "
                "assert r['selection_manifest_sha256']=="
                f"'{contract_v1.TAIL_SELECTION_MANIFEST_SHA256}'"
            ),
            str(valid_package),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_tampered_candidate_is_rejected_before_package(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.so"
    shutil.copy2(subject.DEFAULT_CANDIDATE_PATH, candidate)
    with candidate.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="digest changed"):
        _package(tmp_path, candidate=candidate)


def test_tampered_tail_selection_is_rejected_before_package(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_bytes(subject.DEFAULT_TAIL_SELECTION_MANIFEST_PATH.read_bytes() + b" ")
    with pytest.raises(ValueError, match="selection manifest (bytes changed|is not canonical JSON)"):
        _package(tmp_path, selection=selection)


def test_tampered_dry_run_receipt_is_rejected(tmp_path: Path) -> None:
    receipt = preflight.build_dry_run_receipt(
        image_observation=_image(), runtime_observation=_runtime()
    )
    receipt["launch_authorized"] = True
    path = tmp_path / "receipt.json"
    path.write_bytes(subject.canonical_bytes(receipt))
    with pytest.raises(ValueError, match="receipt changed"):
        _package(tmp_path, receipt=path)


def test_missing_root_is_rejected(tmp_path: Path) -> None:
    roots = _copy_roots(tmp_path)
    (roots / "hand_050.json").unlink()
    with pytest.raises(ValueError, match="exactly roots"):
        _package(tmp_path, root_dir=roots)


def test_duplicate_or_extra_root_is_rejected(tmp_path: Path) -> None:
    roots = _copy_roots(tmp_path)
    shutil.copy2(roots / "hand_000.json", roots / "hand_100.json")
    with pytest.raises(ValueError, match="exactly roots"):
        _package(tmp_path, root_dir=roots)


def test_duplicate_tooling_package_relative_is_rejected(tmp_path: Path) -> None:
    tooling = _tiny_tooling(tmp_path / "tiny-tooling")
    tooling.append((tooling[0][0], tooling[1][1]))
    with pytest.raises(ValueError, match="duplicate package relative"):
        _package(tmp_path, tooling=tooling)


def test_symlink_candidate_is_rejected(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.so"
    _symlink(candidate, subject.DEFAULT_CANDIDATE_PATH)
    with pytest.raises(ValueError, match="symlink|reparse"):
        _package(tmp_path, candidate=candidate)


def test_symlink_root_is_rejected(tmp_path: Path) -> None:
    roots = _copy_roots(tmp_path)
    target = roots / "real_000.json"
    (roots / "hand_000.json").replace(target)
    _symlink(roots / "hand_000.json", target)
    with pytest.raises(ValueError, match="symlink|reparse"):
        _package(tmp_path, root_dir=roots)


def test_symlink_tooling_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real.py"
    real.write_text("VALUE = 1\n", encoding="utf-8")
    link = tmp_path / "link.py"
    _symlink(link, real)
    with pytest.raises(ValueError, match="symlink|reparse"):
        _package(tmp_path, tooling=[("src/ofc_regular/link.py", link)])


def test_symlink_output_parent_is_rejected(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "output-link"
    _symlink(link, actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|reparse"):
        _package(tmp_path, output_parent=link)


def test_rearm2_or_lock_output_namespace_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "performance_lock_rearm2"
    with pytest.raises(ValueError, match="overlaps rearm2"):
        _package(tmp_path, output_parent=output)


def test_rearm2_root_identity_overlap_helper_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        contract_v1,
        "DEVELOPMENT_ROOT_SET_SHA256",
        contract_v1.REARM2_AGGREGATE_ROOT_SHA256,
    )
    with pytest.raises(ValueError, match="root identity overlap"):
        subject._rearm2_root_overlap()


def test_tail_seed_schedule_rejects_overlap_or_tamper() -> None:
    schedule = subject._build_tail_seed_schedule()
    assert schedule["rearm2_seed_overlap_count"] == 0
    tampered = deepcopy(schedule)
    tampered["rearm2_seed_overlap_count"] = 1
    with pytest.raises(ValueError, match="schedule changed"):
        subject._validate_tail_seed_schedule(tampered)


def test_destination_is_write_once(valid_package: Path) -> None:
    with pytest.raises(FileExistsError, match="immutable"):
        subject.package_local(
            output_parent=valid_package.parent,
            run_name=valid_package.name,
            dry_run_receipt_path=valid_package
            / subject.DRY_RECEIPT_PACKAGE_PATH,
            repository_root=REPO_ROOT,
        )


def test_postpackage_entry_tamper_is_rejected(
    tmp_path: Path, valid_package: Path
) -> None:
    copied = tmp_path / RUN_NAME
    shutil.copytree(valid_package, copied)
    target = copied / subject.ROLE_PACKAGE_PATHS["candidate"]
    with target.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="package entry bytes changed"):
        subject.validate_local_package(copied)


def test_stale_receipt_is_rejected_before_publish(tmp_path: Path) -> None:
    receipt = _write_receipt(tmp_path / "receipt.json")
    with pytest.raises(ValueError, match="stale"):
        subject.package_local(
            output_parent=tmp_path / "packages",
            run_name=RUN_NAME,
            dry_run_receipt_path=receipt,
            repository_root=REPO_ROOT,
            now_unix_seconds=OBSERVED_UNIX + subject.MAX_RECEIPT_AGE_SECONDS + 1,
            tooling_sources=_tiny_tooling(tmp_path / "tiny-tooling"),
            fixture_only=True,
        )
    assert not (tmp_path / "packages" / RUN_NAME).exists()


def test_inventory_cli_is_read_only_and_has_no_cloud_adapter(tmp_path: Path) -> None:
    script = (
        REPO_ROOT
        / "scripts/package_hu_m31_t3_step6d_performance_development_v2.py"
    )
    source = script.read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "google.cloud" not in source
    assert "gcloud" not in source
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(script), "--inventory-only"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "source_inventory_rehashed_local_only"
    assert result["root_count"] == 100
    assert result["cloud_executable"] is False
    assert result["launch_authorized"] is False
    assert list(tmp_path.iterdir()) == []


def test_package_cli_requires_explicit_fresh_receipt(tmp_path: Path) -> None:
    script = (
        REPO_ROOT
        / "scripts/package_hu_m31_t3_step6d_performance_development_v2.py"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    output = tmp_path / "packages"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-parent",
            str(output),
            "--run-name",
            RUN_NAME,
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "explicit fresh --dry-run-receipt" in completed.stderr
    assert not output.exists()
