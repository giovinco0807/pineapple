from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ofc_regular import select_hu_m43_attempt09_development as base
from ofc_regular.hu_m43_attempt10_contract import M43_ATTEMPT10_PLAN_SHA256
from ofc_regular.select_hu_m43_attempt10_audit50 import (
    ATTEMPT10_AUDIT50_RECEIPT_KEYS,
    ATTEMPT10_AUDIT50_RECEIPT_SCHEMA,
    _attempt10_audit_bindings,
    execute_attempt10_audit50_selector,
)
from ofc_regular.select_hu_m43_attempt10_development import (
    ATTEMPT10_DEVELOPMENT_DECISION_SCHEMA,
    _attempt10_selector_bindings,
    _expected_search_config,
    _expected_seed_domain_provenance,
)
from ofc_regular.hu_m43_attempt10_teacher import Attempt10TeacherConfig


def _config() -> Attempt10TeacherConfig:
    return Attempt10TeacherConfig(
        frozen_model_sha256=(
            "e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3"
        ),
        hand_seed=1,
        rerank_seed=2,
        veto_seed=3,
        stress_seed=4,
        confirmation_seed=5,
        evaluation_seed=6,
        child_policy_seed=7,
        run_id="attempt10-selector-test",
        batch_child_selectors=True,
    )


def test_attempt10_selector_bindings_restore_attempt09() -> None:
    old_plan = base.M43_ATTEMPT09_PLAN_SHA256
    old_schema = base.ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA
    with _attempt10_selector_bindings():
        assert base.M43_ATTEMPT09_PLAN_SHA256 == M43_ATTEMPT10_PLAN_SHA256
        assert base.ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA == ATTEMPT10_DEVELOPMENT_DECISION_SCHEMA
        assert base.POPULATION == "development"
        assert base._ROOTS == 200
    assert base.M43_ATTEMPT09_PLAN_SHA256 == old_plan
    assert base.ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA == old_schema


def test_attempt10_search_and_seed_provenance_are_exact() -> None:
    seeds = {
        "hand": 1,
        "rerank": 2,
        "veto": 3,
        "stress": 4,
        "confirmation": 5,
        "evaluation": 6,
        "child": 7,
    }
    search = _expected_search_config(_config(), seeds)
    assert search["learned_nonbaseline_top_k"] == 12
    assert search["k8_size"] == 8
    assert search["stress_samples"] == 1024
    assert search["confirmation_samples"] == 512
    provenance = _expected_seed_domain_provenance(seeds)
    assert provenance["domain_order"][3:5] == ["stress_x1024", "confirmation_c512"]


def test_attempt10_audit50_binding_is_predeclared_and_restored() -> None:
    old_roots = base._ROOTS
    old_population = base.POPULATION
    with _attempt10_audit_bindings():
        assert base._ROOTS == 50
        assert base.ROOT_INDEX_FIRST == 200
        assert base.ROOTS_PER_PROFILE == 10
        assert base.POPULATION == "future_audit"
        assert base.GATE_SECTION == "future_audit_go_no_go"
        assert base._GATES["fires_total_min"] == 10
    assert base._ROOTS == old_roots
    assert base.POPULATION == old_population


def test_attempt10_audit50_producer_writes_canonical_no_fit_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        base,
        "select_attempt09_development",
        lambda **_kwargs: {
            "decision": "go",
            "search_freeze_authorized": True,
        },
    )
    output = tmp_path / "selector"
    receipt = execute_attempt10_audit50_selector(
        input_path=tmp_path / "unused.jsonl",
        plan_path=tmp_path / "unused-plan.json",
        authorization_path=tmp_path / "unused-authorization.json",
        source_package_sha256="a" * 64,
        run_name="regular-hu-m43-attempt10-audit50-unit",
        output_dir=output,
    )
    decision_bytes = (output / "decision.json").read_bytes()
    receipt_bytes = (output / "decision_receipt.json").read_bytes()
    assert set(receipt) == ATTEMPT10_AUDIT50_RECEIPT_KEYS
    assert receipt["schema"] == ATTEMPT10_AUDIT50_RECEIPT_SCHEMA
    assert receipt["audit_rows_used_for_fit"] is False
    assert receipt["decision_sha256"] == hashlib.sha256(decision_bytes).hexdigest()
    assert receipt_bytes == (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
