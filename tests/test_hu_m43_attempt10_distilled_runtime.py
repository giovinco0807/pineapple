from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import ofc_regular.hu_m43_attempt10_distilled_runtime as runtime
from ofc_regular.hu_m43_attempt10_contract import (
    ATTEMPT10_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT10_PLAN_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_ROOT = (
    ROOT
    / "outputs"
    / "gcp_runs"
    / "regular-hu-m43-attempt03-c2e64-pilot700-r2-20260714-0228"
    / "package_src"
)
REQUIREMENTS = ROOT / "configs" / "hu_m43_attempt08_runtime_requirements.txt"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
FIXED_MODE = stat.S_IFREG | 0o644


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    (root / "src" / "example").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "pyproject.toml").write_bytes(b"[project]\nname='frozen-test'\n")
    (root / "src" / "example" / "__init__.py").write_bytes(b"VALUE = 1\n")
    (root / "src" / "example" / "runtime.py").write_bytes(
        b"def answer():\n    return 42\n"
    )
    (root / "configs" / REQUIREMENTS.name).write_bytes(REQUIREMENTS.read_bytes())
    for relative in runtime.ATTEMPT10_DISTILLED_RUNTIME_REGISTRY_PATHS:
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # The population contract is deliberately a separate, future artifact.  A
    # source-freeze test needs only its selected bytes, not a live profile.
    plan = root / runtime.ATTEMPT10_DISTILLED_POPULATION_PLAN_PATH
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_bytes(b"{}\n")
    return root


def _freeze(tmp_path: Path, name: str = "freeze") -> tuple[Path, Path, dict]:
    archive = tmp_path / f"{name}.zip"
    manifest = tmp_path / f"{name}.json"
    payload = runtime.build_distilled_runtime_source_freeze(
        source_root=_source(tmp_path / name),
        output_archive=archive,
        output_manifest=manifest,
    )
    return archive, manifest, payload


def _fixed_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = FIXED_MODE << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def _with_extra(source: Path, destination: Path, name: str, encoded: bytes) -> None:
    with zipfile.ZipFile(source, "r") as old, zipfile.ZipFile(
        destination, "x", compression=zipfile.ZIP_STORED
    ) as new:
        for info in old.infolist():
            new.writestr(_fixed_info(info.filename), old.read(info))
        new.writestr(_fixed_info(name), encoded)


def _retarget_archive(manifest: dict, archive: Path) -> dict:
    changed = copy.deepcopy(manifest)
    changed["archive"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    changed["archive"]["bytes"] = archive.stat().st_size
    return changed


def test_source_freeze_is_byte_deterministic_and_exact(tmp_path: Path) -> None:
    source = _source(tmp_path)
    first_archive = tmp_path / "first.zip"
    first_manifest = tmp_path / "first.json"
    second_archive = tmp_path / "second.zip"
    second_manifest = tmp_path / "second.json"

    first = runtime.build_distilled_runtime_source_freeze(
        source_root=source,
        output_archive=first_archive,
        output_manifest=first_manifest,
    )
    second = runtime.build_distilled_runtime_source_freeze(
        source_root=source,
        output_archive=second_archive,
        output_manifest=second_manifest,
    )

    assert first == second
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert [row["path"] for row in first["file_set"]["files"]] == sorted(
        [
            "configs/hu_m43_attempt08_runtime_requirements.txt",
            *runtime.ATTEMPT10_DISTILLED_RUNTIME_REGISTRY_PATHS,
            runtime.ATTEMPT10_DISTILLED_POPULATION_PLAN_PATH,
            "pyproject.toml",
            "src/example/__init__.py",
            "src/example/runtime.py",
        ]
    )
    assert runtime.validate_distilled_runtime_source_archive(
        archive_path=first_archive, manifest=first_manifest
    ) == first


def test_semantic_closure_binds_attempt10_teacher_and_external_runtime(
    tmp_path: Path,
) -> None:
    _, manifest_path, manifest = _freeze(tmp_path)
    semantic = manifest["semantic_closure"]
    assert semantic["teacher_contract"] == {
        "schema": "hu_m43_attempt10_teacher_lineage_v1",
        "plan_sha256": M43_ATTEMPT10_PLAN_SHA256,
        "candidate_model_sha256": ATTEMPT10_LAMBDA_MODEL_SHA256,
    }
    assert semantic["external_runtime"] == {
        "requirements_path": "configs/hu_m43_attempt08_runtime_requirements.txt",
        "requirements_sha256": runtime.ATTEMPT10_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": (
            runtime.ATTEMPT10_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        ),
        "gcp_image": {
            "name": runtime.ATTEMPT10_GCP_IMAGE_NAME,
            "id": runtime.ATTEMPT10_GCP_IMAGE_ID,
            "self_link": runtime.ATTEMPT10_GCP_IMAGE_SELF_LINK,
        },
    }
    assert manifest_path.read_bytes() == runtime.canonical_json_bytes(manifest)


@pytest.mark.parametrize(
    ("section", "key"),
    (
        ("teacher_contract", "plan_sha256"),
        ("teacher_contract", "candidate_model_sha256"),
        ("external_runtime", "requirements_sha256"),
        ("external_runtime", "runtime_fingerprint_sha256"),
    ),
)
def test_manifest_semantic_identity_tamper_fails_closed(
    tmp_path: Path, section: str, key: str
) -> None:
    _, _, manifest = _freeze(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["semantic_closure"][section][key] = "f" * 64
    with pytest.raises(ValueError, match="semantic binding changed"):
        runtime.load_and_validate_distilled_runtime_manifest(changed)


def test_build_rejects_requirements_tamper(tmp_path: Path) -> None:
    source = _source(tmp_path)
    requirements = source / "configs" / REQUIREMENTS.name
    requirements.write_bytes(requirements.read_bytes() + b"# drift\n")
    with pytest.raises(ValueError, match="requirements drifted"):
        runtime.build_distilled_runtime_source_freeze(
            source_root=source,
            output_archive=tmp_path / "bad.zip",
            output_manifest=tmp_path / "bad.json",
        )


def test_archive_path_escape_and_extracted_mutation_fail_closed(
    tmp_path: Path,
) -> None:
    archive, manifest_path, manifest = _freeze(tmp_path)
    malicious = tmp_path / "escape.zip"
    _with_extra(archive, malicious, "../escaped.py", b"ESCAPE = True\n")
    with pytest.raises(ValueError, match="path is unsafe"):
        runtime.extract_distilled_runtime_source_archive(
            archive_path=malicious,
            manifest=_retarget_archive(manifest, malicious),
            output_root=tmp_path / "bad-tree",
        )

    extracted = tmp_path / "tree"
    runtime.extract_distilled_runtime_source_archive(
        archive_path=archive, manifest=manifest_path, output_root=extracted
    )
    target = extracted / "src" / "example" / "runtime.py"
    target.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="source bytes changed"):
        runtime.validate_distilled_runtime_extracted_tree(
            extracted_root=extracted, manifest=manifest_path
        )


def test_pinned_dependency_bytes_are_enforced(tmp_path: Path) -> None:
    validated = runtime.validate_distilled_runtime_dependencies(DEPENDENCY_ROOT)
    assert validated["model_count"] == 11
    assert validated["binary_count"] == 2
    clone = tmp_path / "dependencies"
    shutil.copytree(DEPENDENCY_ROOT, clone, copy_function=os.link)
    victim = clone / validated["models"][0]["path"]
    encoded = victim.read_bytes()
    victim.unlink()
    victim.write_bytes(encoded + b"tamper")
    with pytest.raises(ValueError, match="pinned model bytes changed"):
        runtime.validate_distilled_runtime_dependencies(clone)


def test_only_modules_imported_from_extracted_tree_can_attest(tmp_path: Path) -> None:
    archive, manifest_path, _ = _freeze(tmp_path)
    tree = tmp_path / "tree"
    runtime.extract_distilled_runtime_source_archive(
        archive_path=archive, manifest=manifest_path, output_root=tree
    )
    with pytest.raises(ValueError, match="imported outside frozen tree"):
        runtime.validate_frozen_execution_modules(
            extracted_root=tree,
            manifest=manifest_path,
            module_names=["ofc_regular.hu_m43_attempt10_distilled_runtime"],
        )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tree / "src"), str(ROOT / "src"))
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    script = (
        "from ofc_regular.hu_m43_attempt10_distilled_runtime import "
        "validate_frozen_execution_modules; "
        "a=validate_frozen_execution_modules(extracted_root=r'"
        + str(tree)
        + "',manifest=r'"
        + str(manifest_path)
        + "',module_names=['example.runtime']); "
        "assert a.covers(['example.runtime'])"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_manifest_file_must_be_canonical(tmp_path: Path) -> None:
    _, _, manifest = _freeze(tmp_path)
    noncanonical = tmp_path / "pretty.json"
    noncanonical.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        runtime.load_and_validate_distilled_runtime_manifest(noncanonical)
