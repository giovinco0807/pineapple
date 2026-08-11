from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import select_hu_m43_attempt09_development as base
from ofc_regular.hu_m43_attempt12_contract import M43_ATTEMPT12_PLAN_SHA256
from ofc_regular.hu_m43_attempt12_teacher import (
    Attempt12TeacherConfig,
    validate_attempt12_teacher_output,
)
from ofc_regular import freeze_hu_m43_attempt12_development as freeze
from ofc_regular.select_hu_m43_attempt12_development import (
    ATTEMPT12_DEVELOPMENT_RECEIPT_SCHEMA,
)
from ofc_regular.select_hu_m43_attempt12_audit50 import (
    aggregate_attempt12_audit50_rows,
)
from ofc_regular.select_hu_m43_attempt12_development import (
    _attempt12_exact,
    _attempt12_selector_bindings,
    _expected_search_config,
    aggregate_attempt12_development_rows,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt12.json"
_HASH = "a" * 64


def _plan() -> dict[str, Any]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _validated_row(
    _row: Mapping[str, Any],
    root: int,
    seeds: Mapping[str, int],
    **_kwargs: Any,
) -> dict[str, Any]:
    if root < 200:
        fired = root < 40  # 8 per each of five profiles.
    else:
        fired = root < 210  # 2 per profile in Audit50.
    profile = _plan()["profiles"][root % 5]
    return {
        "root_index": root,
        "profile": profile,
        "hand_seed": seeds["hand"],
        "observation_fingerprint": f"observation-{root}",
        "baseline_action_key": f"baseline-{root}",
        "selected_action_key": f"selected-{root}" if fired else f"baseline-{root}",
        "fired": fired,
        "mean": 1.0 if fired else 0.0,
        "loss": (
            {"p95": 25.0, "p99": 40.0, "max": 50.0}
            if fired
            else {"p95": 0.0, "p99": 0.0, "max": 0.0}
        ),
        "rng_digests": [f"rng-{root}"],
        "belief_digests": [f"belief-{root}"],
        "config_sha256": f"{root:064x}",
    }


def _aggregate_development(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(base, "_validate_row", _validated_row)
    return aggregate_attempt12_development_rows(
        [{} for _ in range(200)],
        plan=_plan(),
        source_input_sha256=_HASH,
        source_plan_sha256=M43_ATTEMPT12_PLAN_SHA256,
        authorization_sha256="b" * 64,
        source_package_sha256="c" * 64,
        run_name="attempt12-development-synthetic",
    )


def test_attempt12_search_config_matches_actual_variable_teacher_schema() -> None:
    seeds = {
        "hand": 1,
        "rerank": 2,
        "veto": 3,
        "stress": 4,
        "confirmation": 5,
        "evaluation": 6,
        "child": 7,
    }
    config = Attempt12TeacherConfig(
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
        run_id="attempt12-selector-schema",
        batch_child_selectors=True,
    )
    actual = {
        "all_legal_candidates": True,
        "candidate_nonbaseline_count": 26,
        "baseline_added_exactly_once": True,
        "rerank_samples": 128,
        "rerank_head_max": 4,
        "shortlist_max": 8,
        "veto_samples": 256,
        "stress_samples": 1024,
        "confirmation_samples": 1024,
        "evaluation_samples": 512,
        "hand_seed": 1,
        "rerank_seed": 2,
        "veto_seed": 3,
        "stress_seed": 4,
        "confirmation_seed": 5,
        "evaluation_seed": 6,
        "child_policy_seed": 7,
        "run_id": "attempt12-selector-schema",
        "batch_child_selectors": True,
    }
    assert _attempt12_exact(actual, _expected_search_config(config, seeds))
    actual["candidate_nonbaseline_count"] = 27
    assert not _attempt12_exact(actual, _expected_search_config(config, seeds))
    with _attempt12_selector_bindings():
        assert base.validate_attempt09_teacher_output is validate_attempt12_teacher_output
        assert base.EVALUATION_SAMPLE_COUNT == 512


def test_attempt12_development_uses_locked_e512_with_nonfires_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _aggregate_development(monkeypatch)
    overall = report["metrics"]["overall"]
    assert report["decision"] == "go"
    assert overall["states"] == 200
    assert overall["fires"] == 40
    assert overall["mean_delta_per_fire"] == 1.0
    assert overall["mean_delta_per_state"] == 0.2
    assert all(metrics["fires"] == 8 for metrics in report["metrics"]["by_profile"].values())
    diagnostics = report["metrics"]["fired_root_diagnostics"]
    assert len(diagnostics) == 40
    assert all("e512_paired_delta_mean" in row for row in diagnostics)
    assert all("e256_paired_delta_mean" not in row for row in diagnostics)
    assert report["science_boundary"]["assessment_source"].startswith(
        "disjoint_E512_locked_final"
    )


def test_attempt12_development_gates_exact_coverage_tails_and_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _aggregate_development(monkeypatch)
    gates = {gate["name"]: gate for gate in report["gates"]}
    assert gates["fires_total"]["observed"] == 40
    assert gates["fires_total"]["requirement"] == ">= 40"
    assert all(value == 8 for value in gates["fires_each_profile"]["observed"].values())
    assert gates["fires_each_profile"]["requirement"] == "each >= 3"
    assert gates["maximum_per_fired_root_p95_loss"]["observed"] == 25
    assert gates["maximum_per_fired_root_p99_loss"]["observed"] == 40
    assert gates["maximum_per_fired_root_max_loss"]["observed"] == 50
    assert gates["pooled_phase_violation_count"]["passed"] is True
    assert report["integrity"]["pooled_phase_violation_count"] == 0
    assert report["integrity"][
        "nonfire_exact_baseline_action_fallback_verified"
    ] is True


def test_attempt12_audit50_uses_ten_total_and_one_per_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base, "_validate_row", _validated_row)
    report = aggregate_attempt12_audit50_rows(
        [{} for _ in range(50)],
        plan=_plan(),
        source_input_sha256=_HASH,
        source_plan_sha256=M43_ATTEMPT12_PLAN_SHA256,
        authorization_sha256="b" * 64,
        source_package_sha256="c" * 64,
        run_name="attempt12-audit50-synthetic",
    )
    gates = {gate["name"]: gate for gate in report["gates"]}
    assert report["decision"] == "go"
    assert report["metrics"]["overall"]["states"] == 50
    assert report["metrics"]["overall"]["fires"] == 10
    assert report["metrics"]["overall"]["mean_delta_per_state"] == 0.2
    assert gates["fires_total"]["requirement"] == ">= 10"
    assert gates["fires_each_profile"]["requirement"] == "each >= 1"
    assert all(value == 2 for value in gates["fires_each_profile"]["observed"].values())
    assert report["integrity"]["pooled_phase_violation_count"] == 0


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def test_attempt12_freeze_reopens_e512_metrics_and_integrity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _aggregate_development(monkeypatch)
    decision_path = tmp_path / "decision.json"
    receipt_path = tmp_path / "receipt.json"
    decision_path.write_bytes(_canonical(report))
    receipt = {
        "schema": ATTEMPT12_DEVELOPMENT_RECEIPT_SCHEMA,
        "status": "single_frozen_gate_evaluation_complete",
        "run_name": report["source"]["run_name"],
        "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
        "decision": "go",
        "search_freeze_authorized": True,
        "gate_evaluation_count": 1,
        "selector_executed": True,
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "runtime_policy_activated": False,
        "current_profile_mutated": False,
    }
    receipt_path.write_bytes(_canonical(receipt))
    with freeze._attempt12_freeze_bindings():
        freeze._validate_attempt12_selector(decision_path, receipt_path)

    report["metrics"]["overall"]["mean_delta_per_state"] = 0.3
    decision_path.write_bytes(_canonical(report))
    receipt["decision_sha256"] = hashlib.sha256(
        decision_path.read_bytes()
    ).hexdigest()
    receipt_path.write_bytes(_canonical(receipt))
    with pytest.raises(ValueError, match="nonfire-zero E512 mean identity"):
        with freeze._attempt12_freeze_bindings():
            freeze._validate_attempt12_selector(decision_path, receipt_path)
