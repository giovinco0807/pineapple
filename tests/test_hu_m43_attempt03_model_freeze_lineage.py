from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINEAGE = ROOT / "configs" / "hu_joint_policy_m43_attempt03_model_freeze_lineage.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_attempt03_freeze_lineage_preserves_pre_row_semantics() -> None:
    lineage = json.loads(LINEAGE.read_text(encoding="utf-8"))
    assert lineage["status"] == (
        "pass_administrative_amendment_precedes_first_teacher_row"
    )
    original_path = ROOT / lineage["original_freeze"]["path"]
    amended_path = ROOT / lineage["amended_freeze"]["path"]
    original = json.loads(original_path.read_text(encoding="utf-8"))
    amended = json.loads(amended_path.read_text(encoding="utf-8"))
    assert _sha(original_path) == lineage["original_freeze"]["file_sha256"]
    assert _sha(amended_path) == lineage["amended_freeze"]["file_sha256"]

    fields = lineage["semantic_projection"]["fields"]
    original_projection = {field: original[field] for field in fields}
    amended_projection = {field: amended[field] for field in fields}
    assert original_projection == amended_projection
    assert original["parent_plan"]["file_sha256"] == (
        lineage["semantic_projection"]["parent_plan_file_sha256_original"]
    )
    assert amended["parent_plan"]["file_sha256"] == (
        lineage["semantic_projection"]["parent_plan_file_sha256_amended"]
    )
    assert _canonical_sha(original_projection) == (
        lineage["semantic_projection"]["original_sha256"]
    )
    assert _canonical_sha(amended_projection) == (
        lineage["semantic_projection"]["amended_sha256"]
    )
    assert sorted(
        key for key in set(original) | set(amended) if original.get(key) != amended.get(key)
    ) == lineage["amended_freeze"]["changed_top_level_keys_only"]

    checkpoint_path = ROOT / lineage["first_teacher_row_boundary"]["evidence_path"]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert _sha(checkpoint_path) == (
        lineage["first_teacher_row_boundary"]["evidence_file_sha256"]
    )
    assert checkpoint["completed_roots"] == 1
    freeze_time = datetime.fromisoformat(
        lineage["amended_freeze"]["filesystem_last_write_time_evidence"]
    )
    first_row_time = datetime.fromisoformat(
        lineage["first_teacher_row_boundary"]["completed_at"]
    )
    assert freeze_time < first_row_time
    assert lineage["semantic_projection"]["equal"] is True
    assert not lineage["current_profile_mutated"]
    assert not lineage["runtime_policy_activated"]
    assert not lineage["full_replacement_enabled"]
