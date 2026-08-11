from __future__ import annotations

from pathlib import Path

from ofc_regular import run_hu_m43_attempt09 as base
from ofc_regular.hu_m43_attempt10_contract import M43_ATTEMPT10_PLAN_SHA256
from ofc_regular.run_hu_m43_attempt10 import (
    ATTEMPT10_ROOT_CONTRACT_SCHEMA,
    ATTEMPT10_ROW_SCHEMA,
    _attempt10_bindings,
    run_attempt10_root,
)


def test_attempt10_bindings_are_scoped_and_schema_specific() -> None:
    old_plan = base.M43_ATTEMPT09_PLAN_SHA256
    old_row = base.ATTEMPT09_ROW_SCHEMA
    old_root = base.ATTEMPT09_ROOT_CONTRACT_SCHEMA
    with _attempt10_bindings():
        assert base.M43_ATTEMPT09_PLAN_SHA256 == M43_ATTEMPT10_PLAN_SHA256
        assert base.ATTEMPT09_ROW_SCHEMA == ATTEMPT10_ROW_SCHEMA
        assert base.ATTEMPT09_ROOT_CONTRACT_SCHEMA == ATTEMPT10_ROOT_CONTRACT_SCHEMA
    assert base.M43_ATTEMPT09_PLAN_SHA256 == old_plan
    assert base.ATTEMPT09_ROW_SCHEMA == old_row
    assert base.ATTEMPT09_ROOT_CONTRACT_SCHEMA == old_root


def test_attempt10_runner_delegates_under_frozen_bindings(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        observed["plan_sha256"] = base.M43_ATTEMPT09_PLAN_SHA256
        observed["row_schema"] = base.ATTEMPT09_ROW_SCHEMA
        return {"status": "complete"}

    monkeypatch.setattr(base, "run_attempt09_root", fake_run)
    result = run_attempt10_root(
        mode="preflight",
        root_index=0,
        output=Path("teacher.jsonl"),
        checkpoint=Path("checkpoint.json"),
        heartbeat=Path("heartbeat.json"),
        model=Path("model.pkl"),
        model_sha256="a" * 64,
        run_id="attempt10-test",
        batch_child_selectors=False,
    )
    assert result == {"status": "complete"}
    assert observed["plan_sha256"] == M43_ATTEMPT10_PLAN_SHA256
    assert observed["row_schema"] == ATTEMPT10_ROW_SCHEMA
    assert observed["mode"] == "preflight"
    assert observed["root_index"] == 0
