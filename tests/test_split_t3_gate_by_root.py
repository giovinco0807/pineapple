import hashlib
import json
from pathlib import Path

import pytest

from ai.tutor.split_t3_gate_by_root import (
    parse_root_spec,
    root_index,
    split_jsonl,
    validate_root_partitions,
)


def _line(root: int | None, source_line: int, position: str, marker: str) -> str:
    record = {
        "source_line": source_line,
        "position": position,
        "marker": marker,
    }
    if root is not None:
        record["branch"] = {"root_index": root}
    return json.dumps(record, separators=(",", ":")) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_split_preserves_rows_and_prefers_branch_root(tmp_path: Path) -> None:
    rows = [
        _line(0, 101, "bb", "mine-1"),
        _line(75, 1, "btn", "dev-1"),
        _line(100, 1, "bb", "final-1"),
        _line(0, 101, "btn", "mine-2"),
        _line(75, 1, "bb", "dev-2"),
        _line(100, 1, "btn", "final-2"),
    ]
    source = tmp_path / "gate.jsonl"
    source.write_text("".join(rows), encoding="utf-8", newline="")

    summary = split_jsonl(
        source,
        tmp_path / "splits",
        expected_counts={"total": 6, "mine": 2, "dev": 2, "final": 2},
    )

    paths = {
        name: Path(details["path"])
        for name, details in summary["outputs"].items()
    }
    assert paths["mine"].read_text(encoding="utf-8") == rows[0] + rows[3]
    assert paths["dev"].read_text(encoding="utf-8") == rows[1] + rows[4]
    assert paths["final"].read_text(encoding="utf-8") == rows[2] + rows[5]
    assert summary["outputs"]["mine"]["positions"] == {"bb": 1, "btn": 1}
    assert summary["outputs"]["dev"]["roots"] == [75]
    assert summary["checks"]["observed_roots_disjoint"] is True
    assert summary["checks"]["all_records_assigned"] is True
    assert summary["outputs"]["final"]["sha256"] == _sha256(paths["final"])
    assert Path(summary["summary_path"]).is_file()


def test_teacher_rows_fall_back_to_one_based_source_line(tmp_path: Path) -> None:
    rows = [
        _line(None, 1, "bb", "mine"),
        _line(None, 76, "btn", "dev"),
        _line(None, 101, "bb", "final"),
    ]
    source = tmp_path / "teacher.jsonl"
    source.write_text("".join(rows), encoding="utf-8", newline="")

    summary = split_jsonl(
        source,
        tmp_path / "teacher-splits",
        expected_counts={"total": 3, "mine": 1, "dev": 1, "final": 1},
    )

    assert summary["outputs"]["mine"]["roots"] == [0]
    assert summary["outputs"]["dev"]["roots"] == [75]
    assert summary["outputs"]["final"]["roots"] == [100]


def test_root_validation_and_expected_counts_fail_closed(tmp_path: Path) -> None:
    assert parse_root_spec("0-2,5") == {0, 1, 2, 5}
    with pytest.raises(ValueError, match="assigned to both"):
        validate_root_partitions({"mine": {0, 1}, "dev": {1, 2}, "final": {3}})

    source = tmp_path / "gate.jsonl"
    source.write_text(_line(0, 1, "bb", "only"), encoding="utf-8")
    with pytest.raises(ValueError, match="expected total=2, got 1"):
        split_jsonl(source, tmp_path / "splits", expected_counts={"total": 2})
    assert not list((tmp_path / "splits").glob("*.jsonl"))


def test_root_index_rejects_missing_or_invalid_fields() -> None:
    with pytest.raises(ValueError, match="missing"):
        root_index({}, line_number=1)
    with pytest.raises(ValueError, match="source_line must be >= 1"):
        root_index({"source_line": 0}, line_number=2)
    with pytest.raises(ValueError, match="branch.root_index must be an integer"):
        root_index({"branch": {"root_index": True}, "source_line": 1}, line_number=3)
