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

import ofc_regular.hu_m43_attempt08_distilled_runtime as runtime
from ofc_regular.hu_m43_attempt08_runtime_anchor_contract import (
    ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
)
from ofc_regular.hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT08_GCP_IMAGE_ID,
    ATTEMPT08_GCP_IMAGE_NAME,
    ATTEMPT08_GCP_IMAGE_SELF_LINK,
    ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
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
    (root / "src" / "example" / "ignored.pyc").write_bytes(b"not source")
    (root / "configs" / "ignored.json").write_bytes(b"{}\n")
    (root / "README.md").write_bytes(b"not selected\n")
    (root / "configs" / REQUIREMENTS.name).write_bytes(REQUIREMENTS.read_bytes())
    for relative in runtime.ATTEMPT08_DISTILLED_RUNTIME_REGISTRY_PATHS:
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    plan_source = ROOT / runtime.ATTEMPT08_DISTILLED_POPULATION_PLAN_PATH
    plan_target = root / runtime.ATTEMPT08_DISTILLED_POPULATION_PLAN_PATH
    plan_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plan_source, plan_target)
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


def _fixed_info(name: str, *, timestamp=FIXED_TIME) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=timestamp)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = FIXED_MODE << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def _rewrite_archive(
    source: Path,
    destination: Path,
    *,
    replacements: dict[str, bytes] | None = None,
    timestamp_replacement: str | None = None,
    extra_entries: tuple[tuple[str, bytes], ...] = (),
) -> None:
    replacements = replacements or {}
    with zipfile.ZipFile(source, "r") as old, zipfile.ZipFile(
        destination, "x", compression=zipfile.ZIP_STORED
    ) as new:
        for info in old.infolist():
            timestamp = (1981, 1, 1, 0, 0, 0) if info.filename == timestamp_replacement else FIXED_TIME
            new.writestr(
                _fixed_info(info.filename, timestamp=timestamp),
                replacements.get(info.filename, old.read(info)),
            )
        for name, encoded in extra_entries:
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
            *runtime.ATTEMPT08_DISTILLED_RUNTIME_REGISTRY_PATHS,
            runtime.ATTEMPT08_DISTILLED_POPULATION_PLAN_PATH,
            "pyproject.toml",
            "src/example/__init__.py",
            "src/example/runtime.py",
        ]
    )
    assert runtime.validate_distilled_runtime_source_archive(
        archive_path=first_archive, manifest=first_manifest
    ) == first


def test_semantic_closure_binds_teacher_and_external_runtime(tmp_path: Path) -> None:
    _, manifest_path, manifest = _freeze(tmp_path)
    semantic = manifest["semantic_closure"]
    assert semantic["teacher_runtime"] == {
        "runtime_source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
        "runtime_semantic_anchor_sha256": ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
        "runtime_source_file_count": ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
    }
    assert semantic["external_runtime"] == {
        "requirements_path": "configs/hu_m43_attempt08_runtime_requirements.txt",
        "requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "gcp_image": {
            "name": ATTEMPT08_GCP_IMAGE_NAME,
            "id": ATTEMPT08_GCP_IMAGE_ID,
            "self_link": ATTEMPT08_GCP_IMAGE_SELF_LINK,
        },
    }
    assert manifest_path.read_bytes() == runtime.canonical_json_bytes(manifest)


@pytest.mark.parametrize(
    ("section", "key"),
    (
        ("teacher_runtime", "runtime_source_closure_sha256"),
        ("teacher_runtime", "runtime_semantic_anchor_sha256"),
        ("external_runtime", "requirements_sha256"),
        ("external_runtime", "runtime_fingerprint_sha256"),
    ),
)
def test_manifest_semantic_binding_tamper_fails_closed(
    tmp_path: Path, section: str, key: str
) -> None:
    _, _, manifest = _freeze(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["semantic_closure"][section][key] = "f" * 64
    with pytest.raises(ValueError, match="semantic binding changed"):
        runtime.load_and_validate_distilled_runtime_manifest(changed)


def test_build_rejects_runtime_requirements_tamper(tmp_path: Path) -> None:
    source = _source(tmp_path)
    requirements = source / "configs" / REQUIREMENTS.name
    requirements.write_bytes(requirements.read_bytes() + b"# drift\n")
    with pytest.raises(ValueError, match="requirements drifted"):
        runtime.build_distilled_runtime_source_freeze(
            source_root=source,
            output_archive=tmp_path / "bad.zip",
            output_manifest=tmp_path / "bad.json",
        )


def test_archive_content_and_metadata_tamper_fail_closed(tmp_path: Path) -> None:
    archive, _, manifest = _freeze(tmp_path)
    content_tamper = tmp_path / "content-tamper.zip"
    _rewrite_archive(
        archive,
        content_tamper,
        replacements={"src/example/runtime.py": b"def answer():\n    return 43\n"},
    )
    with pytest.raises(ValueError, match="content changed"):
        runtime.validate_distilled_runtime_source_archive(
            archive_path=content_tamper,
            manifest=_retarget_archive(manifest, content_tamper),
        )

    metadata_tamper = tmp_path / "metadata-tamper.zip"
    _rewrite_archive(
        archive,
        metadata_tamper,
        timestamp_replacement="src/example/runtime.py",
    )
    with pytest.raises(ValueError, match="ZIP metadata changed"):
        runtime.validate_distilled_runtime_source_archive(
            archive_path=metadata_tamper,
            manifest=_retarget_archive(manifest, metadata_tamper),
        )


def test_path_escape_is_rejected_before_extraction(tmp_path: Path) -> None:
    archive, _, manifest = _freeze(tmp_path)
    malicious = tmp_path / "escape.zip"
    _rewrite_archive(
        archive,
        malicious,
        extra_entries=(("../escaped.py", b"ESCAPE = True\n"),),
    )
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="path is unsafe"):
        runtime.extract_distilled_runtime_source_archive(
            archive_path=malicious,
            manifest=_retarget_archive(manifest, malicious),
            output_root=destination,
        )
    assert not destination.exists()
    assert not (tmp_path / "escaped.py").exists()


def test_extracted_tree_validator_rejects_mutation_and_extras(tmp_path: Path) -> None:
    archive, manifest_path, manifest = _freeze(tmp_path)
    extracted = tmp_path / "tree"
    assert runtime.extract_distilled_runtime_source_archive(
        archive_path=archive,
        manifest=manifest_path,
        output_root=extracted,
    ) == manifest
    assert runtime.validate_distilled_runtime_extracted_tree(
        extracted_root=extracted, manifest=manifest_path
    ) == manifest

    target = extracted / "src" / "example" / "runtime.py"
    original = target.read_bytes()
    target.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="source bytes changed"):
        runtime.validate_distilled_runtime_extracted_tree(
            extracted_root=extracted, manifest=manifest_path
        )
    target.write_bytes(original)
    (extracted / "extra.txt").write_bytes(b"extra\n")
    with pytest.raises(ValueError, match="file set changed"):
        runtime.validate_distilled_runtime_extracted_tree(
            extracted_root=extracted, manifest=manifest_path
        )


def test_manifest_file_requires_canonical_bytes(tmp_path: Path) -> None:
    _, _, manifest = _freeze(tmp_path)
    noncanonical = tmp_path / "pretty.json"
    noncanonical.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        runtime.load_and_validate_distilled_runtime_manifest(noncanonical)


def test_pinned_runtime_dependency_manifest_and_bytes_are_enforced(
    tmp_path: Path,
) -> None:
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


def test_live_or_decoy_import_cannot_attest_as_frozen_execution(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "repo.zip"
    manifest = tmp_path / "repo.json"
    tree = tmp_path / "tree"
    runtime.build_distilled_runtime_source_freeze(
        source_root=ROOT, output_archive=archive, output_manifest=manifest
    )
    runtime.extract_distilled_runtime_source_archive(
        archive_path=archive, manifest=manifest, output_root=tree
    )
    with pytest.raises(ValueError, match="imported outside frozen tree"):
        runtime.validate_frozen_execution_modules(
            extracted_root=tree,
            manifest=manifest,
            module_names=["ofc_regular.hu_m43_attempt08_distilled_runtime"],
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tree / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ofc_regular.hu_m43_attempt08_distilled_runtime import "
                "validate_frozen_execution_modules; "
                "validate_frozen_execution_modules(extracted_root=r'"
                + str(tree)
                + "',manifest=r'"
                + str(manifest)
                + "',module_names=['ofc_regular.hu_m43_attempt08_distilled_runtime'])"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
