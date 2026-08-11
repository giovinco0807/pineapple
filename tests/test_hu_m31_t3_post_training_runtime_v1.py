from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_post_training_runtime_v1 as subject
from ofc_regular import hu_m31_t3_post_training_gcp_controller_v1 as controller


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, raw: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _canonical(value: Any) -> bytes:
    return subject.canonical_bytes(value)


def _fake_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    repository = tmp_path / "repository"
    models = tmp_path / "models"
    pyproject = _write(repository / "pyproject.toml", b"[project]\nname='fake'\n")
    required_python = (
        "__init__.py",
        "ai_profiles.py",
        "hu_m31_t3_abr_teacher_v1.py",
        "hu_m31_t3_behavior_roots.py",
        "hu_m31_t3_post_training_gcp_v1.py",
        "hu_m31_t3_post_training_runtime_v1.py",
        "extra_runtime_dependency.py",
    )
    python_paths = {
        name: _write(
            repository / "src" / "ofc_regular" / name,
            f"# {name}\nVALUE = {len(name)}\n".encode("ascii"),
        )
        for name in required_python
    }
    search = _write(
        repository / "rust" / "hu_m3_engine" / "src" / "search.rs",
        b"pub fn exact_search() {}\n",
    )
    monkeypatch.setattr(
        subject, "PROFILE_REGISTRY_SHA256", _sha(python_paths["ai_profiles.py"])
    )
    monkeypatch.setattr(
        subject,
        "ABR_TEACHER_SOURCE_SHA256",
        _sha(python_paths["hu_m31_t3_abr_teacher_v1.py"]),
    )
    monkeypatch.setattr(
        subject, "ACCEPTED_SEARCH_SOURCE_SHA256", _sha(search)
    )

    specs = []
    for index in range(11):
        filename = f"real-model-{index:02d}.bin"
        raw = bytes((index + 1,)) * (index + 3)
        path = _write(models / filename, raw)
        specs.append(
            subject.ModelSpec(
                model_paths_field=f"path_field_{index:02d}",
                bundle_field=f"bundle_field_{index:02d}",
                filename=filename,
                bytes=len(raw),
                sha256=_sha(path),
            )
        )
    monkeypatch.setattr(subject, "MODEL_SPECS", tuple(specs))

    accepted = _write(tmp_path / "accepted.so", b"accepted-real-engine")
    diagnostic = _write(tmp_path / "diagnostic.so", b"diagnostic-real-engine")
    monkeypatch.setattr(
        subject, "ACCEPTED_CANDIDATE_LIBRARY_SHA256", _sha(accepted)
    )
    monkeypatch.setattr(
        subject, "ACCEPTED_CANDIDATE_LIBRARY_BYTES", accepted.stat().st_size
    )

    wheelhouse = tmp_path / "wheelhouse.zip"
    wheel_payloads = {
        "alpha-1.0-py3-none-any.whl": b"alpha-wheel",
        "beta-2.0-py3-none-any.whl": b"beta-wheel",
    }
    with zipfile.ZipFile(wheelhouse, "w") as archive:
        for name in sorted(wheel_payloads):
            archive.writestr(name, wheel_payloads[name])
    entries = [
        {
            "filename": name,
            "sha256": hashlib.sha256(wheel_payloads[name]).hexdigest(),
            "bytes": len(wheel_payloads[name]),
        }
        for name in sorted(wheel_payloads)
    ]
    entries_sha = hashlib.sha256(_canonical(entries) + b"\n").hexdigest()
    wheel_manifest_value = {
        "schema": "hu_m31_t3_step6d_perfdev_v2_wheelhouse_v1",
        "status": "complete_hash_pinned_offline_wheelhouse",
        "python_abi": "cp311",
        "target_os": "linux",
        "target_architecture": "x86_64",
        "network_install_allowed": False,
        "entries": entries,
        "entry_count": len(entries),
        "entries_sha256": entries_sha,
    }
    wheel_manifest = _write(
        tmp_path / "wheelhouse-manifest.json",
        _canonical(wheel_manifest_value),
    )
    monkeypatch.setattr(
        subject, "RAW_WHEELHOUSE_ARCHIVE_SHA256", _sha(wheelhouse)
    )
    monkeypatch.setattr(
        subject, "RAW_WHEELHOUSE_ARCHIVE_BYTES", wheelhouse.stat().st_size
    )
    monkeypatch.setattr(
        subject, "RAW_WHEELHOUSE_MANIFEST_SHA256", _sha(wheel_manifest)
    )
    monkeypatch.setattr(
        subject,
        "RAW_WHEELHOUSE_MANIFEST_BYTES",
        wheel_manifest.stat().st_size,
    )
    monkeypatch.setattr(
        subject, "RAW_WHEELHOUSE_ENTRY_COUNT", len(entries)
    )
    monkeypatch.setattr(
        subject, "RAW_WHEELHOUSE_ENTRIES_SHA256", entries_sha
    )
    monkeypatch.setattr(subject, "_filesystem_type", lambda _path: "ext4")

    def fake_model_load(
        _models: Path, model_inventory_sha256: str
    ) -> dict[str, Any]:
        fields = [spec.bundle_field for spec in subject.MODEL_SPECS]
        return {
            "behavior_profiles": list(subject.M31_T3_BEHAVIOR_PROFILES),
            "required_bundle_fields": fields,
            "loaded_bundle_fields": fields,
            "model_inventory_sha256": model_inventory_sha256,
            "all_required_models_deserialized": True,
            "optional_loader_fallback_used": False,
            "placeholder_or_synthetic_model_used": False,
        }

    monkeypatch.setattr(
        subject, "_validate_model_deserialization", fake_model_load
    )
    return {
        "repository": repository,
        "models": models,
        "pyproject": pyproject,
        "python_paths": python_paths,
        "search": search,
        "accepted": accepted,
        "diagnostic": diagnostic,
        "diagnostic_sha256": _sha(diagnostic),
        "wheelhouse": wheelhouse,
        "wheel_manifest": wheel_manifest,
    }


def _package(inputs: dict[str, Any], output: Path) -> dict[str, Any]:
    return subject.package_runtime_bundle(
        repository_root=inputs["repository"],
        models_root=inputs["models"],
        raw_wheelhouse_path=inputs["wheelhouse"],
        raw_wheelhouse_manifest_path=inputs["wheel_manifest"],
        accepted_library_path=inputs["accepted"],
        diagnostic_library_path=inputs["diagnostic"],
        expected_diagnostic_library_sha256=inputs["diagnostic_sha256"],
        output_root=output,
    )


def test_real_runtime_package_is_deterministic_complete_and_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fake_inputs(tmp_path, monkeypatch)
    first_root = tmp_path / "bundle-a"
    second_root = tmp_path / "bundle-b"
    first = _package(inputs, first_root)
    second = _package(inputs, second_root)

    assert _sha(first["runtime_archive_path"]) == _sha(
        second["runtime_archive_path"]
    )
    assert _sha(first["wheelhouse_archive_path"]) == _sha(
        second["wheelhouse_archive_path"]
    )
    assert first["manifest"] == second["manifest"]
    assert first["ready"] == second["ready"]
    assert first["manifest"]["model_load_receipt"][
        "all_required_models_deserialized"
    ] is True
    assert first["manifest"]["real_models_only"] is True
    assert first["manifest"]["current_profile_changed"] is False
    assert first["ready"]["status"] == "ready_for_real_one_pair_local_smoke"
    assert len(first["manifest"]["model_inventory"]) == 11

    with pytest.raises(FileExistsError, match="create-only"):
        _package(inputs, first_root)


def test_runtime_validator_rejects_payload_manifest_ready_and_extra_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fake_inputs(tmp_path, monkeypatch)

    payload_root = tmp_path / "payload-tamper"
    payload = _package(inputs, payload_root)
    payload["accepted_library_path"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="stored artifact"):
        subject.validate_runtime_bundle(payload["manifest_path"])

    manifest_root = tmp_path / "manifest-tamper"
    manifest = _package(inputs, manifest_root)
    value = json.loads(manifest["manifest_path"].read_text("ascii"))
    value["current_profile_changed"] = True
    manifest["manifest_path"].write_bytes(_canonical(value))
    with pytest.raises(ValueError, match="manifest boundary"):
        subject.validate_runtime_bundle(manifest["manifest_path"])

    ready_root = tmp_path / "ready-tamper"
    ready = _package(inputs, ready_root)
    value = json.loads(ready["ready_path"].read_text("ascii"))
    value["cloud_execution_started"] = True
    ready["ready_path"].write_bytes(_canonical(value))
    with pytest.raises(ValueError, match="READY boundary"):
        subject.validate_runtime_bundle(ready["manifest_path"])

    extra_root = tmp_path / "extra-tamper"
    extra = _package(inputs, extra_root)
    _write(extra_root / "unbound.bin", b"unbound")
    with pytest.raises(ValueError, match="unbound file"):
        subject.validate_runtime_bundle(extra["manifest_path"])


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-model",
        "changed-model",
        "changed-wheelhouse",
        "changed-wheelhouse-manifest",
        "changed-accepted-native",
        "changed-diagnostic-native",
        "changed-profile-registry",
        "changed-search-source",
    ),
)
def test_packager_fails_closed_before_output_for_incomplete_real_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    inputs = _fake_inputs(tmp_path, monkeypatch)
    if mutation == "missing-model":
        (inputs["models"] / subject.MODEL_SPECS[0].filename).unlink()
    elif mutation == "changed-model":
        (inputs["models"] / subject.MODEL_SPECS[0].filename).write_bytes(
            b"changed"
        )
    elif mutation == "changed-wheelhouse":
        with inputs["wheelhouse"].open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "changed-wheelhouse-manifest":
        with inputs["wheel_manifest"].open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "changed-accepted-native":
        inputs["accepted"].write_bytes(b"changed")
    elif mutation == "changed-diagnostic-native":
        inputs["diagnostic"].write_bytes(b"changed")
    elif mutation == "changed-profile-registry":
        inputs["python_paths"]["ai_profiles.py"].write_bytes(b"changed")
    elif mutation == "changed-search-source":
        inputs["search"].write_bytes(b"changed")
    output = tmp_path / "must-not-exist"
    with pytest.raises((ValueError, FileNotFoundError)):
        _package(inputs, output)
    assert not output.exists()


def test_packager_rejects_non_ext4_and_incomplete_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fake_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(subject, "_filesystem_type", lambda _path: "9p")
    with pytest.raises(PermissionError, match="ext4 or xfs"):
        _package(inputs, tmp_path / "wrong-filesystem")

    monkeypatch.setattr(subject, "_filesystem_type", lambda _path: "ext4")

    def incomplete(_models: Path, digest: str) -> dict[str, Any]:
        del digest
        raise ValueError("real ABR behavior model load is incomplete")

    monkeypatch.setattr(subject, "_validate_model_deserialization", incomplete)
    output = tmp_path / "incomplete-model-load"
    with pytest.raises(ValueError, match="model load is incomplete"):
        _package(inputs, output)
    assert not output.exists()


def test_resolve_bundle_requires_plan_diagnostic_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fake_inputs(tmp_path, monkeypatch)
    result = _package(inputs, tmp_path / "bundle")
    checked = subject.resolve_bundle(
        result["manifest_path"],
        expected_diagnostic_library_sha256=inputs["diagnostic_sha256"],
    )
    assert checked["diagnostic_library_path"] == result[
        "diagnostic_library_path"
    ]
    with pytest.raises(ValueError, match="native engine"):
        subject.resolve_bundle(
            result["manifest_path"],
            expected_diagnostic_library_sha256="0" * 64,
        )


def test_package_cli_serializes_resolved_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / subject.MANIFEST_NAME
    monkeypatch.setattr(
        controller.runtime_bundle,
        "package_runtime_bundle",
        lambda **_kwargs: {
            "manifest": {"schema": subject.BUNDLE_SCHEMA},
            "ready": {"schema": subject.READY_SCHEMA},
            "manifest_path": manifest_path,
        },
    )
    result = controller.main(
        [
            "package-abr-runtime",
            "--repository-root",
            "repository",
            "--models-root",
            "models",
            "--raw-wheelhouse",
            "wheelhouse.zip",
            "--raw-wheelhouse-manifest",
            "wheelhouse.json",
            "--accepted-library",
            "accepted.so",
            "--diagnostic-library",
            "diagnostic.so",
            "--expected-diagnostic-library-sha256",
            "1" * 64,
            "--output-root",
            "bundle",
        ]
    )
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["manifest_path"] == str(manifest_path)
