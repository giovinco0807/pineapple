from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ofc_regular.validate_hu_m43_attempt12_acceptance as acceptance


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT11_PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt11_population.json"
ATTEMPT12_PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt12_population.json"


def _future_attempt12_plan() -> dict[str, object]:
    """Build the future contract in memory; this must never write its config."""

    plan = json.loads(ATTEMPT11_PLAN.read_text(encoding="utf-8"))
    plan.update(
        {
            "milestone": "M4.3-attempt12-final-population",
            "purpose": (
                "fresh_realized_match_ev_acceptance_for_attempt12_distilled_"
                "t1_second_selector"
            ),
            "policy_attempt": "attempt12_distilled_v1",
            "seed": acceptance.ATTEMPT12_POPULATION_SEED,
        }
    )
    runtime = plan["runtime_contract"]
    runtime.update(
        {
            "model_schema": acceptance.HU_M43_ATTEMPT12_DISTILLED_MODEL_SCHEMA,
            "artifact_schema": acceptance.HU_M43_ATTEMPT12_DISTILLED_ARTIFACT_SCHEMA,
            "feature_schema": acceptance.HU_M43_ATTEMPT12_DISTILLED_FEATURE_SCHEMA,
            "head_schema": acceptance.HU_M43_ATTEMPT12_DISTILLED_HEAD_SCHEMA,
            "action_score_mode": (
                acceptance.HU_M43_ATTEMPT12_DISTILLED_ACTION_SCORE_MODE
            ),
            "candidate_nonbaseline_count_range": [0, 26],
            "lightgbm_group_size_range": [1, 27],
            "complete_legal_action_set_rebuilt": True,
            "candidate_padding_allowed": False,
            "candidate_duplication_allowed": False,
            "baseline_appended_exactly_once": True,
        }
    )
    freshness = plan["freshness"]
    freshness.pop("exclude_attempt11_preflight_development_and_audit_namespaces")
    freshness["exclude_attempt12_preflight_development_and_audit_namespaces"] = True
    freshness.pop("excluded_attempt11_namespace_bases")
    freshness["excluded_attempt12_namespace_bases"] = [
        200_108_071_901,
        201_108_071_901,
        202_108_071_901,
        203_108_071_901,
        204_108_071_901,
        205_108_071_901,
        206_108_071_901,
        210_108_071_901,
        211_108_071_901,
        212_108_071_901,
        213_108_071_901,
        214_108_071_901,
        215_108_071_901,
        216_108_071_901,
    ]
    freshness["excluded_population_schedules"].append(
        {
            "milestone": "M4.3-attempt11-final-population",
            "seed": 190_108_071_901,
            "seed_stride": 1_000_003,
            "paired_seeds": 1000,
        }
    )
    freshness["excluded_schedule_registry_sources"] = list(
        acceptance.ATTEMPT12_SEED_REGISTRY_SOURCES
    )
    _, registry_rows = acceptance._load_seed_registry(
        acceptance.ATTEMPT12_SEED_REGISTRY_SOURCES
    )
    freshness["excluded_schedule_registry_sha256"] = hashlib.sha256(
        acceptance._canonical(registry_rows).encode("utf-8")
    ).hexdigest()
    return plan


def _audit_go() -> tuple[dict[str, object], dict[str, object], str]:
    digest = "a" * 64
    audit = {
        "schema": acceptance.ATTEMPT12_AUDIT50_DECISION_SCHEMA,
        "status": "go_attempt12_audit50_search_quality",
        "decision": "go",
        "search_freeze_authorized": True,
        "selected_arm": None,
        "selected_threshold": None,
        "science_boundary": {
            "fit_performed": False,
            "threshold_selected": False,
            "runtime_policy_activated": False,
            "current_profile_mutated": False,
            "teacher_values_are_realized_match_ev": False,
        },
        "gates": [{"name": "all", "passed": True}],
        "decision_contract": {
            "gate_evaluation_count": 1,
            "single_frozen_search_architecture": True,
            "arm_selection_performed": False,
            "threshold_selection_performed": False,
            "all_gates_required": True,
        },
    }
    receipt = {
        "schema": acceptance.ATTEMPT12_AUDIT50_SELECTOR_RECEIPT_SCHEMA,
        "status": "single_frozen_gate_evaluation_complete",
        "run_name": "attempt12-audit50",
        "decision_sha256": digest,
        "decision": "go",
        "search_freeze_authorized": True,
        "gate_evaluation_count": 1,
        "selector_executed": True,
        "future_audit_authorized": False,
        "audit_rows_used_for_fit": False,
        "fit_performed": False,
        "threshold_selected": False,
        "runtime_policy_activated": False,
        "current_profile_mutated": False,
    }
    return audit, receipt, digest


def test_attempt12_acceptance_contract_constants() -> None:
    assert callable(acceptance.load_and_validate_attempt12_population_plan)
    assert acceptance.ATTEMPT12_POPULATION_SEEDS == 1000
    assert acceptance.ATTEMPT12_POPULATION_SHARDS == 20
    assert acceptance.ATTEMPT12_POPULATION_SEEDS_PER_SHARD == 50


def test_attempt12_population_plan_is_byte_frozen_and_valid() -> None:
    assert hashlib.sha256(ATTEMPT12_PLAN.read_bytes()).hexdigest() == (
        acceptance.ATTEMPT12_POPULATION_PLAN_SHA256
    )
    plan = acceptance.load_and_validate_attempt12_population_plan(ATTEMPT12_PLAN)
    assert plan["seed"] == 220_108_071_901
    assert plan["seed_stride"] == 1_000_003
    assert plan["paired_seeds_per_opponent"] == 1000
    assert plan["shards"] == 20
    assert plan["paired_seeds_per_shard"] == 50
    assert plan["minimum_valid_overrides"] == 300
    assert plan["runtime_contract"]["candidate_nonbaseline_count_range"] == [0, 26]
    assert plan["runtime_contract"]["lightgbm_group_size_range"] == [1, 27]
    assert plan["activation_guards"]["runtime_policy_activated"] is False
    assert plan["activation_guards"]["current_profile_changed"] is False


def test_attempt12_future_plan_validates_4x1000_20x50_and_min300() -> None:
    plan = acceptance.load_and_validate_attempt12_population_plan(
        _future_attempt12_plan()
    )
    assert len(plan["opponents"]) == 4
    assert plan["paired_seeds_per_opponent"] == 1000
    assert plan["shards"] == 20
    assert plan["paired_seeds_per_shard"] == 50
    assert plan["minimum_valid_overrides"] == 300
    assert plan["_freshness_counts"] == {
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
    }
    assert plan["_freshness_registry"]["planned_overlap_count"] == 0


def test_attempt12_registry_binds_prior_population_and_search_plan() -> None:
    sources = acceptance.ATTEMPT12_SEED_REGISTRY_SOURCES
    assert acceptance.ATTEMPT11_POPULATION_REGISTRY_PATH in sources
    assert acceptance.ATTEMPT12_SEARCH_PLAN_REGISTRY_PATH in sources
    seeds, rows = acceptance._load_seed_registry(sources)
    planned = {
        acceptance.ATTEMPT12_POPULATION_SEED
        + index * acceptance.ATTEMPT12_POPULATION_SEED_STRIDE
        for index in range(acceptance.ATTEMPT12_POPULATION_SEEDS)
    }
    assert not planned.intersection(seeds)
    search_row = next(
        row
        for row in rows
        if row["path"] == acceptance.ATTEMPT12_SEARCH_PLAN_REGISTRY_PATH
    )
    assert search_row["authorized_self_reservation_seed_count"] == 1000


def test_attempt12_plan_fails_closed_on_missing_incomplete_or_drift(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        acceptance.load_and_validate_attempt12_population_plan(
            tmp_path / "population.json"
        )
    with pytest.raises(ValueError, match="identity changed"):
        acceptance.load_and_validate_attempt12_population_plan({})
    changed = _future_attempt12_plan()
    changed["runtime_contract"]["feature_schema"] = "wrong"
    with pytest.raises(ValueError, match="runtime contract changed"):
        acceptance.load_and_validate_attempt12_population_plan(changed)
    changed = _future_attempt12_plan()
    changed["freshness"]["excluded_schedule_registry_sources"].remove(
        acceptance.ATTEMPT11_POPULATION_REGISTRY_PATH
    )
    with pytest.raises(ValueError, match="source list changed"):
        acceptance.load_and_validate_attempt12_population_plan(changed)


def test_attempt12_audit_go_requires_complete_one_shot_binding() -> None:
    audit, receipt, digest = _audit_go()
    acceptance._validate_attempt12_audit50_go_authorization(
        audit, receipt, audit_decision_sha256=digest
    )
    changed = copy.deepcopy(audit)
    changed["gates"][0]["passed"] = False
    with pytest.raises(ValueError, match="Go contract changed"):
        acceptance._validate_attempt12_audit50_go_authorization(
            changed, receipt, audit_decision_sha256=digest
        )
    changed_receipt = copy.deepcopy(receipt)
    changed_receipt.pop("audit_rows_used_for_fit")
    with pytest.raises(ValueError, match="not freeze authorization"):
        acceptance._validate_attempt12_audit50_go_authorization(
            audit, changed_receipt, audit_decision_sha256=digest
        )
    changed = copy.deepcopy(audit)
    changed["decision_contract"]["gate_evaluation_count"] = 2
    with pytest.raises(ValueError, match="Go contract changed"):
        acceptance._validate_attempt12_audit50_go_authorization(
            changed, receipt, audit_decision_sha256=digest
        )


def test_attempt12_population_provenance_reopens_all_five_go_artifacts(
    tmp_path: Path,
) -> None:
    def write(name: str, payload: dict[str, object]) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    development_decision = write(
        "development_decision.json",
        {
            "schema": acceptance.ATTEMPT12_DEVELOPMENT_DECISION_SCHEMA,
            "decision": "go",
            "search_freeze_authorized": True,
        },
    )
    development_receipt = write(
        "development_receipt.json",
        {
            "schema": acceptance.ATTEMPT12_DEVELOPMENT_SELECTOR_RECEIPT_SCHEMA,
            "status": "single_frozen_gate_evaluation_complete",
            "decision_sha256": acceptance._file_sha256(development_decision),
            "decision": "go",
            "search_freeze_authorized": True,
            "gate_evaluation_count": 1,
            "selector_executed": True,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    development_jsonl_sha = "c" * 64
    development_freeze = write(
        "development_freeze.json",
        {
            "schema": acceptance.ATTEMPT12_DEVELOPMENT_PASS_FREEZE_SCHEMA,
            "status": "go_freeze_attempt12_development",
            "decision": "go",
            "future_audit_authorized": False,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
            "bindings": {
                "plan_sha256": acceptance.M43_ATTEMPT12_PLAN_SHA256,
                "selector_decision_sha256": acceptance._file_sha256(
                    development_decision
                ),
                "selector_receipt_sha256": acceptance._file_sha256(
                    development_receipt
                ),
                "merged_input_sha256": development_jsonl_sha,
            },
        },
    )
    audit, audit_receipt, _ = _audit_go()
    audit_decision = write("audit_decision.json", audit)
    audit_receipt["decision_sha256"] = acceptance._file_sha256(audit_decision)
    audit_receipt_path = write("audit_receipt.json", audit_receipt)
    dev_hashes = {
        "development_decision_sha256": acceptance._file_sha256(
            development_decision
        ),
        "development_selector_receipt_sha256": acceptance._file_sha256(
            development_receipt
        ),
        "development_pass_freeze_sha256": acceptance._file_sha256(
            development_freeze
        ),
    }
    audit_hashes = {
        "audit50_decision_sha256": acceptance._file_sha256(audit_decision),
        "audit50_selector_receipt_sha256": acceptance._file_sha256(
            audit_receipt_path
        ),
    }
    result = acceptance._validate_attempt12_population_provenance_bundle(
        training_manifest={
            "source": {**dev_hashes, "development_jsonl_sha256": development_jsonl_sha}
        },
        runtime_freeze=audit_hashes,
        development_decision_path=development_decision,
        development_selector_receipt_path=development_receipt,
        development_pass_freeze_path=development_freeze,
        audit_decision_path=audit_decision,
        audit_selector_receipt_path=audit_receipt_path,
    )
    assert result == {**dev_hashes, **audit_hashes}
    changed = dict(audit_hashes)
    changed["audit50_selector_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audit50 provenance bytes changed"):
        acceptance._validate_attempt12_population_provenance_bundle(
            training_manifest={
                "source": {
                    **dev_hashes,
                    "development_jsonl_sha256": development_jsonl_sha,
                }
            },
            runtime_freeze=changed,
            development_decision_path=development_decision,
            development_selector_receipt_path=development_receipt,
            development_pass_freeze_path=development_freeze,
            audit_decision_path=audit_decision,
            audit_selector_receipt_path=audit_receipt_path,
        )
def test_attempt12_population_preflight_emits_dev_and_audit_binding_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "model.pkl",
            "training.json",
            "freeze.json",
            "plan.json",
            "runtime.zip",
            "runtime_manifest.json",
        )
    }
    dev_hashes = {
        "development_decision_sha256": "1" * 64,
        "development_selector_receipt_sha256": "2" * 64,
        "development_pass_freeze_sha256": "3" * 64,
    }
    audit_hashes = {
        "audit50_decision_sha256": "4" * 64,
        "audit50_selector_receipt_sha256": "5" * 64,
    }
    paths["model.pkl"].write_bytes(b"model")
    paths["training.json"].write_text(
        json.dumps({"source": dev_hashes}), encoding="utf-8"
    )
    paths["freeze.json"].write_text(
        json.dumps(audit_hashes), encoding="utf-8"
    )
    for name in ("plan.json", "runtime.zip", "runtime_manifest.json"):
        paths[name].write_bytes(name.encode("ascii"))
    monkeypatch.setattr(
        acceptance,
        "load_and_validate_attempt12_population_plan",
        lambda _: {
            "_freshness_counts": {
                "teacher_overlap_count": 0,
                "prior_population_overlap_count": 0,
            },
            "_freshness_registry": {
                "sha256": "6" * 64,
                "source_count": 37,
                "seed_count": 1,
                "sources": ["registry"],
            },
        },
    )
    monkeypatch.setattr(
        acceptance,
        "validate_distilled_runtime_source_archive",
        lambda **_: {
            "file_set": {"sha256": "7" * 64},
            "semantic_closure": {"sha256": "8" * 64},
        },
    )
    monkeypatch.setattr(
        acceptance,
        "load_bound_attempt12_distilled_model",
        lambda *_, **__: SimpleNamespace(
            schema=acceptance.HU_M43_ATTEMPT12_DISTILLED_MODEL_SCHEMA,
            artifact_schema=acceptance.HU_M43_ATTEMPT12_DISTILLED_ARTIFACT_SCHEMA,
            feature_schema=acceptance.HU_M43_ATTEMPT12_DISTILLED_FEATURE_SCHEMA,
            head_schema=acceptance.HU_M43_ATTEMPT12_DISTILLED_HEAD_SCHEMA,
            action_score_mode=acceptance.HU_M43_ATTEMPT12_DISTILLED_ACTION_SCORE_MODE,
            model_id="attempt12-test",
            safety_threshold=0.5,
            minimum_fold_votes=4,
            manifest={
                "source_model_manifest_sha256": "9" * 64,
                "source_native_manifest_sha256": "a" * 64,
                "runtime_dependency_closure_sha256": "b" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        acceptance,
        "_validate_attempt12_population_provenance_bundle",
        lambda **_: {**dev_hashes, **audit_hashes},
    )
    receipt = acceptance.build_attempt12_population_preflight(
        model_path=paths["model.pkl"],
        training_manifest_path=paths["training.json"],
        runtime_freeze_path=paths["freeze.json"],
        population_plan_path=paths["plan.json"],
        runtime_source_archive_path=paths["runtime.zip"],
        runtime_source_manifest_path=paths["runtime_manifest.json"],
        runtime_source_root=tmp_path,
        runtime_dependency_root=tmp_path,
        development_decision_path=tmp_path / "development_decision.json",
        development_selector_receipt_path=tmp_path / "development_receipt.json",
        development_pass_freeze_path=tmp_path / "development_freeze.json",
        audit_decision_path=tmp_path / "audit_decision.json",
        audit_selector_receipt_path=tmp_path / "audit_receipt.json",
    )
    assert {name: receipt[name] for name in dev_hashes} == dev_hashes
    assert {name: receipt[name] for name in audit_hashes} == audit_hashes
    assert receipt["development200_full_fit_bound"] is True
    assert receipt["audit50_one_shot_go_bound"] is True
    assert receipt["audit50_fit_rows"] == 0
    assert receipt["threshold_reselection_performed"] is False


def test_attempt12_variable_training_groups_cover_all_legal_actions() -> None:
    groups = [1 + index % 27 for index in range(200)]
    counts = [group - 1 for group in groups]
    diagnostics = {
        "states": 200,
        "rows": sum(groups),
        "group_sizes": groups,
        "candidate_count_min": min(counts),
        "candidate_count_max": max(counts),
        "candidate_count_histogram": {
            str(candidate_count): counts.count(candidate_count)
            for candidate_count in range(27)
        },
    }
    assert acceptance._valid_variable_training_diagnostics(diagnostics)
    changed = copy.deepcopy(diagnostics)
    changed["group_sizes"][0] = 28
    assert not acceptance._valid_variable_training_diagnostics(changed)


def test_attempt12_runtime_freeze_fails_closed_on_missing_artifacts(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        acceptance.freeze_attempt12_distilled_runtime(
            source_model_path=missing,
            training_manifest_path=missing,
            audit_decision_path=missing,
            audit_selector_receipt_path=missing,
            runtime_source_archive_path=missing,
            runtime_source_manifest_path=missing,
            runtime_dependency_root=missing,
            output_model_path=tmp_path / "model.pkl",
            output_runtime_freeze_path=tmp_path / "freeze.json",
        )
    assert not (tmp_path / "model.pkl").exists()
    assert not (tmp_path / "freeze.json").exists()
