from __future__ import annotations

import copy
import hashlib
import re
import shutil
from pathlib import Path

import pytest

import ofc_regular.hu_m43_attempt08_runtime_anchor as anchor
import ofc_regular.hu_m43_attempt08_runtime_identity as identity


ROOT = Path(__file__).resolve().parents[1]


def _closure_tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "src" / "ofc_regular", root / "src" / "ofc_regular")
    for relative in anchor.ATTEMPT08_RUNTIME_SEMANTIC_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return root


def test_full_python_tree_and_fixed_orchestration_files_are_in_closure() -> None:
    rows = anchor.runtime_source_closure_rows(ROOT)
    paths = {row["path"] for row in rows}
    expected_python = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src" / "ofc_regular").rglob("*.py")
        if path.relative_to(ROOT).as_posix()
        != anchor.ATTEMPT08_RUNTIME_CONTRACT_RELATIVE
    }
    assert expected_python.issubset(paths)
    assert set(anchor.ATTEMPT08_RUNTIME_SEMANTIC_FILES).issubset(paths)
    assert "scripts/HuM43Attempt08Spot.Common.ps1" in paths
    assert "scripts/HuM43Attempt04Spot.Common.ps1" in paths
    assert anchor.ATTEMPT08_RUNTIME_CONTRACT_RELATIVE not in paths
    assert len(rows) == len(expected_python) + len(
        anchor.ATTEMPT08_RUNTIME_SEMANTIC_FILES
    )


def test_one_byte_source_config_test_or_startup_drift_changes_closure(
    tmp_path: Path,
) -> None:
    root = _closure_tree(tmp_path)
    original = anchor.runtime_source_closure_sha256(root)
    targets = (
        "src/ofc_regular/__init__.py",
        "configs/fl_ev_regular_2k.json",
        "tests/test_run_hu_m43_attempt08_preflight.py",
        "scripts/startup_hu_m43_attempt08_preflight.sh",
        "scripts/HuM43Attempt08Spot.Common.ps1",
        "scripts/HuM43Attempt04Spot.Common.ps1",
    )
    for index, relative in enumerate(targets):
        clone = tmp_path / f"clone-{index}"
        shutil.copytree(root, clone)
        path = clone / relative
        path.write_bytes(path.read_bytes() + b" ")
        assert anchor.runtime_source_closure_sha256(clone) != original


def test_only_three_contract_value_lines_are_normalized(tmp_path: Path) -> None:
    root = _closure_tree(tmp_path)
    original = anchor.normalized_anchor_contract_sha256(root)
    contract = root / anchor.ATTEMPT08_RUNTIME_CONTRACT_RELATIVE
    text = contract.read_text(encoding="utf-8")
    for name, replacement in (
        ("ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256", '"f" * 64'),
        ("ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256", '"e" * 64'),
        ("ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT", "999"),
    ):
        text = re.sub(
            rf"(?m)^{name}\s*=\s*.+$", f"{name} = {replacement}", text, count=1
        )
    contract.write_text(text, encoding="utf-8")
    assert anchor.normalized_anchor_contract_sha256(root) == original
    contract.write_text(text + "# semantic drift\n", encoding="utf-8")
    assert anchor.normalized_anchor_contract_sha256(root) != original


def test_anchor_payload_recomputes_source_rows_and_rejects_tampering() -> None:
    payload = anchor.runtime_semantic_anchor_payload(ROOT)
    rows = payload["source_closure"]["files"]
    source_sha = hashlib.sha256(anchor.canonical_json_bytes(rows)).hexdigest()
    payload_sha = hashlib.sha256(anchor.canonical_json_bytes(payload)).hexdigest()
    anchor.validate_runtime_semantic_anchor_payload(
        payload,
        expected_source_closure_sha256=source_sha,
        expected_anchor_sha256=payload_sha,
        expected_source_file_count=len(rows),
    )
    changed = copy.deepcopy(payload)
    changed["source_closure"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="payload changed"):
        anchor.validate_runtime_semantic_anchor_payload(
            changed,
            expected_source_closure_sha256=source_sha,
            expected_anchor_sha256=payload_sha,
            expected_source_file_count=len(rows),
        )


def test_windows_or_unpinned_local_runtime_cannot_impersonate_spot() -> None:
    with pytest.raises(ValueError, match="external runtime fingerprint changed"):
        identity.validate_expected_runtime_fingerprint()
