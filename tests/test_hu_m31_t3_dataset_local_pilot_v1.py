from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_dataset_local_pilot_v1 as subject


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.canonical_bytes(value))


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    gate_path = tmp_path / "fresh_quality_gate.json"
    _write(gate_path, {"schema": "synthetic-fresh-quality-gate"})
    library_path = tmp_path / "candidate02.dll"
    library_path.write_bytes(b"synthetic candidate02")
    gate_value = {
        "schema": "synthetic-fresh-quality-gate",
        "merge_sha256": "a" * 64,
        "decision": "fresh_quality_pass_open_25_paired_data_shard_only",
        "full_9000_paired_fanout_authorized": False,
    }
    monkeypatch.setattr(
        subject.executor,
        "_gate_value",
        lambda path: (
            dict(gate_value),
            hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        ),
    )
    monkeypatch.setattr(
        subject.executor,
        "_profile_sha256",
        lambda: subject.contract.CURRENT_PROFILE_REGISTRY_SHA256,
    )
    return gate_path.resolve(), library_path.resolve()


def _install_completed_smoke(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    done = {
        "pair_count": 25,
        "root_count": 50,
        "seat_counts": {"first": 25, "second": 25},
    }
    merge = {
        "data_pilot_25_paired_complete": True,
        "full_9000_paired_fanout_authorized": False,
    }
    gate = {
        "status": "pass",
        "all_gates_passed": True,
        "full_9000_paired_fanout_authorized": True,
    }

    def run_smoke(**kwargs: Any) -> dict[str, Any]:
        events.append("run-smoke")
        assert kwargs["max_new_pairs"] is None
        assert kwargs["plan"]["smoke_gate"]["paired_hand_count"] == 25
        shard = Path(kwargs["shard_directory"])
        shard.mkdir(parents=True, exist_ok=True)
        done_path = shard / "SHARD_DONE.json"
        if done_path.exists():
            return {"status": "already_complete"}
        _write(done_path, done)
        return {"status": "complete_done_published_last"}

    def validate_done(**kwargs: Any) -> dict[str, Any]:
        events.append("validate-done")
        return subject._read_canonical(
            Path(kwargs["shard_directory"]) / "SHARD_DONE.json",
            "synthetic shard done",
        )

    def write_merge(**kwargs: Any) -> dict[str, Any]:
        events.append("write-merge")
        assert (
            Path(kwargs["shard_directory"]) / "SHARD_DONE.json"
        ).is_file()
        _write(Path(kwargs["output_path"]), merge)
        return dict(merge)

    def validate_merge(
        value: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        events.append("validate-merge")
        assert value == merge
        return dict(merge)

    def write_gate(**kwargs: Any) -> dict[str, Any]:
        events.append("write-gate")
        root = Path(kwargs["output_path"]).parent
        assert (root / subject.SMOKE_MERGE_NAME).is_file()
        _write(Path(kwargs["output_path"]), gate)
        return dict(gate)

    def validate_gate(
        value: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        events.append("validate-gate")
        assert value == gate
        return dict(gate)

    monkeypatch.setattr(subject.executor, "run_smoke_shard", run_smoke)
    monkeypatch.setattr(
        subject.contract, "validate_completed_shard", validate_done
    )
    monkeypatch.setattr(subject.executor, "write_smoke_merge", write_merge)
    monkeypatch.setattr(
        subject.executor, "validate_smoke_merge_value", validate_merge
    )
    monkeypatch.setattr(
        subject.contract, "write_smoke_gate_receipt", write_gate
    )
    monkeypatch.setattr(
        subject.contract, "validate_smoke_gate_receipt", validate_gate
    )
    monkeypatch.setattr(
        subject.executor,
        "_pyarrow_modules",
        lambda: (_ for _ in ()).throw(
            AssertionError("local pilot must not import pyarrow")
        ),
    )


def test_local_pilot_is_create_only_replayable_and_needs_no_pyarrow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_gate, library = _inputs(tmp_path, monkeypatch)
    events: list[str] = []
    _install_completed_smoke(monkeypatch, events)
    root = (tmp_path / "pilot").resolve()

    first = subject.run_local_pilot(
        output_root=root,
        fresh_quality_gate_path=fresh_gate,
        library_path=library,
        search_adapter=object(),
    )
    assert first["status"] == "passed_source_replayed_25_paired_local_pilot"
    assert first["paired_hand_count"] == 25
    assert first["root_count"] == 50
    assert first["parquet_required_for_smoke_pass"] is False
    assert first["parquet_exported"] is False
    assert first["full_9000_paired_fanout_authorized"] is True
    assert first["full_fanout_started"] is False
    assert first["training_eligible"] is False
    assert events == [
        "run-smoke",
        "validate-done",
        "write-merge",
        "write-gate",
    ]
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    events.clear()
    second = subject.run_local_pilot(
        output_root=root,
        fresh_quality_gate_path=fresh_gate,
        library_path=library,
        search_adapter=object(),
    )
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert second == first
    assert after == before
    assert events == [
        "run-smoke",
        "validate-done",
        "validate-merge",
        "validate-gate",
    ]
    assert set(path.name for path in root.iterdir()) == {
        subject.PLAN_NAME,
        subject.SHARDS_DIRECTORY_NAME,
        subject.SMOKE_MERGE_NAME,
        subject.SMOKE_GATE_NAME,
    }


def test_unqualified_fresh_gate_stops_before_root_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_gate = tmp_path / "fresh_quality_gate.json"
    _write(fresh_gate, {"schema": "no-go"})
    monkeypatch.setattr(
        subject.executor,
        "_gate_value",
        lambda _path: (_ for _ in ()).throw(
            PermissionError("fresh quality no-go")
        ),
    )
    root = (tmp_path / "must-not-exist").resolve()

    with pytest.raises(PermissionError, match="fresh quality no-go"):
        subject.run_local_pilot(
            output_root=root,
            fresh_quality_gate_path=fresh_gate.resolve(),
            library_path=None,
            search_adapter=object(),
        )
    assert not root.exists()


def test_input_overlap_stops_before_output_root_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "must-not-exist").resolve()
    gate_path = root / "fresh_quality_gate.json"
    gate_path.parent.mkdir()
    _write(gate_path, {"schema": "synthetic-fresh-quality-gate"})
    library_path = tmp_path / "candidate02.dll"
    library_path.write_bytes(b"synthetic candidate02")
    monkeypatch.setattr(
        subject.executor,
        "_gate_value",
        lambda path: (
            {"schema": "synthetic-fresh-quality-gate"},
            hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        ),
    )
    monkeypatch.setattr(
        subject.executor,
        "_profile_sha256",
        lambda: subject.contract.CURRENT_PROFILE_REGISTRY_SHA256,
    )

    with pytest.raises(ValueError, match="overlaps a pinned input"):
        subject.run_local_pilot(
            output_root=root,
            fresh_quality_gate_path=gate_path.resolve(),
            library_path=library_path.resolve(),
            search_adapter=object(),
        )
    assert sorted(path.name for path in root.iterdir()) == [
        "fresh_quality_gate.json"
    ]


def test_production_library_platform_mismatch_fails_before_root_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_gate, _ = _inputs(tmp_path, monkeypatch)
    library = tmp_path / "candidate02.so"
    library.write_bytes(b"accepted synthetic candidate02")
    monkeypatch.setattr(
        subject.contract,
        "ACCEPTED_CANDIDATE_LIBRARY_SHA256",
        hashlib.sha256(library.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(subject.sys, "platform", "win32")
    root = (tmp_path / "must-not-exist").resolve()

    with pytest.raises(ValueError, match="cannot load on this platform"):
        subject.run_local_pilot(
            output_root=root,
            fresh_quality_gate_path=fresh_gate,
            library_path=library.resolve(),
        )
    assert not root.exists()


def test_partial_smoke_never_writes_merge_or_fanout_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_gate, library = _inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        subject.executor,
        "run_smoke_shard",
        lambda **_: {"status": "partial_safe_to_resume"},
    )
    monkeypatch.setattr(
        subject.executor,
        "write_smoke_merge",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("partial shard must not merge")
        ),
    )
    monkeypatch.setattr(
        subject.contract,
        "write_smoke_gate_receipt",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("partial shard must not authorize fanout")
        ),
    )
    root = (tmp_path / "partial").resolve()

    with pytest.raises(RuntimeError, match="before all 25 paired"):
        subject.run_local_pilot(
            output_root=root,
            fresh_quality_gate_path=fresh_gate,
            library_path=library,
            search_adapter=object(),
        )
    assert (root / subject.PLAN_NAME).is_file()
    assert not (root / subject.SMOKE_MERGE_NAME).exists()
    assert not (root / subject.SMOKE_GATE_NAME).exists()


def test_unexpected_entry_or_gate_before_merge_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_gate, library = _inputs(tmp_path, monkeypatch)
    unexpected = (tmp_path / "unexpected").resolve()
    unexpected.mkdir()
    (unexpected / "other-shard.txt").write_text("unsafe", encoding="ascii")
    with pytest.raises(ValueError, match="unexpected entry"):
        subject.run_local_pilot(
            output_root=unexpected,
            fresh_quality_gate_path=fresh_gate,
            library_path=library,
            search_adapter=object(),
        )

    reordered = (tmp_path / "reordered").resolve()
    reordered.mkdir()
    _write(
        reordered / subject.SMOKE_GATE_NAME,
        {"status": "forged-before-merge"},
    )
    with pytest.raises(ValueError, match="before its source-replayed merge"):
        subject.run_local_pilot(
            output_root=reordered,
            fresh_quality_gate_path=fresh_gate,
            library_path=library,
            search_adapter=object(),
        )


def test_cli_exposes_only_the_single_smoke_pilot_surface() -> None:
    help_text = subject._parser().format_help()
    assert "--output-root" in help_text
    assert "--fresh-quality-gate" in help_text
    assert "--library" in help_text
    assert "--max-new-pairs" not in help_text
    assert "run-shard" not in help_text
    assert "export-parquet" not in help_text
