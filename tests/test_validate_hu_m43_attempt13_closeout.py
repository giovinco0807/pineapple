from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import ofc_regular.validate_hu_m43_attempt13_closeout as closeout
from ofc_regular.hu_m43_attempt13_contract import (
    M43_ATTEMPT13_PLAN_SHA256,
    M43_ATTEMPT13_PROFILES,
)


ROOT = Path(__file__).resolve().parents[1]


def test_actual_attempt13_development_no_go_closeout_validates() -> None:
    closeout_path = ROOT / "configs" / "hu_joint_policy_m43_attempt13_closeout.json"
    assert hashlib.sha256(closeout_path.read_bytes()).hexdigest() == (
        "00e4189f879f9e5da6f020f7a91e22c2f5cdd2a9628a41a6a36be513213509c1"
    )

    result = closeout.validate_attempt13_closeout(ROOT)

    assert result == {
        "schema": "hu_m43_attempt13_closeout_validation_v1",
        "status": "validated_complete_no_go_development",
        "terminal_stage": "development200",
        "terminal_artifact": "development_decision",
        "validated_artifacts": 10,
        "failed_gates": [
            "maximum_per_fired_root_p95_loss",
            "maximum_per_fired_root_max_loss",
        ],
        "profile_id": "stage20_m4_attempt13",
        "explicit_opt_in_authorized": False,
        "automatic_activation_authorized": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement_enabled": False,
    }


def _bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write(root: Path, relative: str, value: Any, *, raw: bool = False) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if raw else _bytes(value))
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _load(root: Path, identity: dict[str, str]) -> dict[str, Any]:
    return json.loads((root / identity["path"]).read_text(encoding="utf-8"))


def _stub_read_only_revalidators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _validate_launch(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
        directory = Path(run_dir)
        return (
            json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
            json.loads(
                (directory / "launch_authorization.json").read_text(encoding="utf-8")
            ),
        )

    def _select(**kwargs: Any) -> dict[str, Any]:
        return json.loads(
            Path(kwargs["input_path"])
            .with_name("decision.json")
            .read_text(encoding="utf-8")
        )

    monkeypatch.setattr(closeout.spot, "validate_launch", _validate_launch)
    monkeypatch.setattr(
        closeout.development_selector, "select_attempt13_development", _select
    )
    monkeypatch.setattr(closeout.audit_selector, "select_attempt13_audit50", _select)


def _frozen_root(tmp_path: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    root = tmp_path / "repo"
    contracts = copy.deepcopy(closeout._FROZEN_CONTRACTS)
    for identity in contracts.values():
        source = ROOT / identity["path"]
        destination = root / identity["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return root, contracts


def _integrity() -> dict[str, Any]:
    return {
        key: True if key.endswith("fallback_verified") else 0
        for key in closeout._INTEGRITY_KEYS
    }


def _add_run(
    root: Path,
    artifacts: dict[str, dict[str, str]],
    *,
    stage: str,
    decision_value: str,
) -> dict[str, Any]:
    prefix = stage
    mode = "development" if stage == "development" else "future_audit"
    roots = 200 if stage == "development" else 50
    first = 0 if stage == "development" else 200
    last = first + roots - 1
    run_name = f"attempt13-{stage}-synthetic"
    run_dir = f"evidence/{stage}/run"
    artifacts[f"{prefix}_schedule"] = _write(
        root, f"{run_dir}/{closeout.spot.SCHEDULE_NAME}", b"schedule\n", raw=True
    )
    artifacts[f"{prefix}_source_archive"] = _write(
        root, f"{run_dir}/{closeout.spot.SOURCE_NAME}", b"source", raw=True
    )
    artifacts[f"{prefix}_merged_teacher"] = _write(
        root, f"evidence/{stage}/teacher.jsonl", b"{}\n", raw=True
    )
    selector_name = (
        "select_hu_m43_attempt13_development.py"
        if stage == "development"
        else "select_hu_m43_attempt13_audit50.py"
    )
    selector_source = ROOT / "src" / "ofc_regular" / selector_name
    selector_relative = f"evidence/{stage}/{selector_name}"
    selector_destination = root / selector_relative
    selector_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selector_source, selector_destination)
    artifacts[f"{prefix}_selector_source"] = {
        "path": selector_relative,
        "sha256": hashlib.sha256(selector_destination.read_bytes()).hexdigest(),
    }
    manifest = {
        "schema": "hu_m43_attempt13_spot_package_v1",
        "status": "packaged_without_root_or_gcloud",
        "mode": mode,
        "run_name": run_name,
        "plan_sha256": M43_ATTEMPT13_PLAN_SHA256,
        "schedule_sha256": artifacts[f"{prefix}_schedule"]["sha256"],
        "source_sha256": artifacts[f"{prefix}_source_archive"]["sha256"],
        "startup_sha256": "1" * 64,
        "total_shards": roots,
        "ai_profiles_sha256": closeout._AI_PROFILES_SHA256,
        "preceding_gate": {
            "path": closeout.spot.GATE_RELATIVE,
            "sha256": (
                closeout._PREFLIGHT_SHA256
                if stage == "development"
                else artifacts["development_go_freeze"]["sha256"]
            ),
            "status": (
                "pass_correctness_preflight"
                if stage == "development"
                else "go_freeze_attempt13_development"
            ),
            "decision": (
                "authorize_development200_package_only"
                if stage == "development"
                else "go"
            ),
        },
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    artifacts[f"{prefix}_manifest"] = _write(
        root, f"{run_dir}/manifest.json", manifest
    )
    authorization = {
        "schema": "hu_m43_attempt13_execution_authorization_v1",
        "status": "authorized",
        "mode": mode,
        "plan_sha256": M43_ATTEMPT13_PLAN_SHA256,
        "source_package_sha256": artifacts[f"{prefix}_source_archive"]["sha256"],
        "root_index_first": first,
        "root_index_last": last,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    artifacts[f"{prefix}_authorization"] = _write(
        root, f"{run_dir}/execution_authorization.json", authorization
    )
    launch = {
        "schema": "hu_m43_attempt13_spot_launch_authorization_v1",
        "status": "authorized",
        "mode": mode,
        "run_name": run_name,
        "manifest_sha256": artifacts[f"{prefix}_manifest"]["sha256"],
        "schedule_sha256": artifacts[f"{prefix}_schedule"]["sha256"],
        "source_sha256": artifacts[f"{prefix}_source_archive"]["sha256"],
        "total_shards": roots,
        "spot_authorized": True,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    artifacts[f"{prefix}_launch_authorization"] = _write(
        root, f"{run_dir}/launch_authorization.json", launch
    )
    receive = {
        "schema": "hu_m43_attempt13_receive_v1",
        "status": "complete",
        "run_name": run_name,
        "mode": mode,
        "roots": roots,
        "root_indices": list(range(first, last + 1)),
        "profiles": [
            M43_ATTEMPT13_PROFILES[index % 5] for index in range(first, last + 1)
        ],
        "manifest_sha256": artifacts[f"{prefix}_manifest"]["sha256"],
        "authorization_sha256": artifacts[f"{prefix}_authorization"]["sha256"],
        "schedule_sha256": artifacts[f"{prefix}_schedule"]["sha256"],
        "source_sha256": artifacts[f"{prefix}_source_archive"]["sha256"],
        "merged_sha256": artifacts[f"{prefix}_merged_teacher"]["sha256"],
        "audit_sha256": "3" * 64,
        "batch_boundary_validation_count": 1,
        "per_shard_boundary_revalidation_count": 0,
        "selector_executed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    artifacts[f"{prefix}_receive_receipt"] = _write(
        root, f"evidence/{stage}/receive.json", receive
    )
    passed = decision_value == "go"
    population_key = (
        "development_population" if stage == "development" else "audit_population"
    )
    decision_schema = (
        "hu_m43_attempt13_development_go_no_go_v1"
        if stage == "development"
        else "hu_m43_attempt13_audit50_go_no_go_v1"
    )
    decision_status = closeout._selector_expected_status(stage, decision_value)
    requirements = {
        "fires_total": ">= 40" if stage == "development" else ">= 10",
        "fires_each_profile": "each >= 3" if stage == "development" else "each >= 1",
        "mean_delta_per_state": "> 0",
        "mean_delta_per_fire": "> 0",
        "false_positive_rate_per_fire": "<= 0.4",
        "maximum_per_fired_root_p95_loss": "<= 25",
        "maximum_per_fired_root_p99_loss": "<= 40",
        "maximum_per_fired_root_max_loss": "<= 50",
        **{name: "= 0" for name in closeout._INTEGRITY_GATE_NAMES},
        "nonfire_exact_baseline_action_fallback": "required",
    }
    gates = []
    for name in closeout._SELECTOR_GATE_NAMES:
        gate_passed = passed or name != "fires_total"
        gates.append(
            {
                "name": name,
                "passed": gate_passed,
                "observed": (
                    True
                    if name == "nonfire_exact_baseline_action_fallback"
                    else 0
                ),
                "requirement": requirements[name],
            }
        )
    decision = {
        "schema": decision_schema,
        "status": decision_status,
        "decision": decision_value,
        "search_freeze_authorized": passed,
        "selected_arm": None,
        "selected_threshold": None,
        "source": {
            "input_jsonl_sha256": artifacts[f"{prefix}_merged_teacher"]["sha256"],
            "plan_sha256": M43_ATTEMPT13_PLAN_SHA256,
            "authorization_sha256": artifacts[f"{prefix}_authorization"]["sha256"],
            "source_package_sha256": artifacts[f"{prefix}_source_archive"]["sha256"],
            "run_name": run_name,
            "selector_source_sha256": artifacts[f"{prefix}_selector_source"]["sha256"],
            "root_identity_sha256": "2" * 64,
        },
        population_key: {
            "roots": roots,
            "profile_counts": {
                profile: roots // len(M43_ATTEMPT13_PROFILES)
                for profile in M43_ATTEMPT13_PROFILES
            },
        },
        "metrics": {},
        "gates": gates,
        "decision_contract": {
            "single_frozen_search_architecture": True,
            "arm_selection_performed": False,
            "threshold_selection_performed": False,
            "gate_evaluation_count": 1,
            "all_gates_required": True,
        },
        "integrity": _integrity(),
        "science_boundary": {
            "assessment_source": "disjoint_E512_locked_final_nonbaseline_output_vs_explicit_baseline",
            "teacher_values_are_realized_match_ev": False,
            "future_audit_authorized": False,
            "fit_performed": False,
            "threshold_selected": False,
            "runtime_policy_activated": False,
            "current_profile_mutated": False,
            "full_replacement_enabled": False,
        },
    }
    artifacts[f"{prefix}_decision"] = _write(
        root, f"evidence/{stage}/decision.json", decision
    )
    receipt = {
        "schema": (
            "hu_m43_attempt13_development_selector_receipt_v1"
            if stage == "development"
            else "hu_m43_attempt13_audit50_selector_receipt_v1"
        ),
        "status": "single_frozen_gate_evaluation_complete",
        "run_name": run_name,
        "decision_sha256": artifacts[f"{prefix}_decision"]["sha256"],
        "decision": decision_value,
        "search_freeze_authorized": passed,
        "gate_evaluation_count": 1,
        "selector_executed": True,
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "runtime_policy_activated": False,
        "current_profile_mutated": False,
    }
    if stage == "audit":
        receipt["audit_rows_used_for_fit"] = False
    artifacts[f"{prefix}_decision_receipt"] = _write(
        root, f"evidence/{stage}/decision_receipt.json", receipt
    )
    return decision


def _add_development_freeze(
    root: Path,
    artifacts: dict[str, dict[str, str]],
    decision: dict[str, Any],
) -> None:
    manifest = _load(root, artifacts["development_manifest"])
    freeze = {
        "schema": "hu_m43_attempt13_development_go_freeze_v1",
        "status": "go_freeze_attempt13_development",
        "decision": "go",
        "run_name": decision["source"]["run_name"],
        "bindings": {
            "plan_sha256": M43_ATTEMPT13_PLAN_SHA256,
            "manifest_sha256": artifacts["development_manifest"]["sha256"],
            "execution_authorization_sha256": artifacts["development_authorization"]["sha256"],
            "source_package_sha256": artifacts["development_source_archive"]["sha256"],
            "schedule_sha256": artifacts["development_schedule"]["sha256"],
            "startup_sha256": manifest["startup_sha256"],
            "merged_input_sha256": artifacts["development_merged_teacher"]["sha256"],
            "receive_receipt_sha256": artifacts["development_receive_receipt"]["sha256"],
            "selector_decision_sha256": artifacts["development_decision"]["sha256"],
            "selector_receipt_sha256": artifacts["development_decision_receipt"]["sha256"],
            "selector_source_sha256": artifacts["development_selector_source"]["sha256"],
            "root_identity_sha256": decision["source"]["root_identity_sha256"],
        },
        "authorization_scope": {
            "future_audit_package_authorized": True,
            "future_audit_authorization_artifact_authorized": True,
            "future_audit_launch_authorized": False,
            "future_audit_started": False,
            "fit_authorized": False,
            "fit_started": False,
            "threshold_selection_authorized": False,
            "threshold_selection_started": False,
            "runtime_policy_activated": False,
            "current_profile_mutated": False,
            "full_replacement_enabled": False,
        },
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "runtime_policy_activated": False,
        "current_profile_mutated": False,
        "full_replacement_enabled": False,
    }
    artifacts["development_go_freeze"] = _write(
        root, "evidence/development/go_freeze.json", freeze
    )


def _science(stage: str) -> dict[str, Any]:
    population = stage == "population"
    return {
        "public_information_set_only": True,
        "opponent_private_discard_used": False,
        "opponent_profile_runtime_feature_used": False,
        "teacher_values_are_realized_match_ev": False,
        "teacher_ev_or_lcb_runtime_gate": False,
        "threshold_reselection_performed": False,
        "alternate_seed_retry_performed": False,
        "top1_accuracy_is_acceptance_gate": False,
        "candidate_selection_and_evaluation_rng_independent": True,
        "realized_population_evidence": population,
        "nonfire_full_trajectory_cancellation_verified": population,
        "first_seat_candidate_is_exact_baseline_delegate": True,
        "unseen_population_robustness_guaranteed": False,
        "nash_equilibrium_claim": False,
        "exploitability_bound_claim": False,
        "mathematically_proven_optimal": False,
        "evaluation_scope": {
            "development200": "fresh_development200_search_quality_only",
            "audit50": "fresh_audit50_search_quality_only",
            "population": "frozen_four_opponent_1000_paired_seeds_each_realized_match_ev",
        }[stage],
    }


def _activation(stage: str, status: str) -> dict[str, Any]:
    population = stage == "population"
    go = population and status == "complete_go"
    return {
        "profile_id": "stage20_m4_attempt13",
        "runtime_artifact_frozen": population,
        "runtime_binding_verified": population,
        "promotion_eligible": go,
        "explicit_opt_in_authorized": go,
        "automatic_activation_authorized": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement_enabled": False,
    }


def _closeout(
    contracts: dict[str, dict[str, str]],
    artifacts: dict[str, dict[str, str]],
    *,
    stage: str,
    status: str,
    terminal_key: str,
    terminal: dict[str, Any],
    scope: dict[str, Any],
    population_recompute: Any = None,
) -> dict[str, Any]:
    return {
        "schema": "hu_m43_attempt13_closeout_v1",
        "milestone": "M4.3-attempt13",
        "status_date": "2026-07-16",
        "status": status,
        "terminal_stage": stage,
        "frozen_contracts": contracts,
        "immutable_artifacts": artifacts,
        "terminal_gate_snapshot": closeout._gate_snapshot(
            terminal_key, terminal, stage == "population"
        ),
        "opened_scope": scope,
        "science_boundary": _science(stage),
        "activation": _activation(stage, status),
        "population_recompute": population_recompute,
    }


def _development_scope() -> dict[str, Any]:
    return {
        "preflight_passed": True,
        "development_completed": True,
        "development_decision": "no_go",
        "future_audit_started": False,
        "future_audit_decision": None,
        "fit_state": "not_started",
        "audit50_fit_rows": 0,
        "fit_artifact_runtime_eligible": False,
        "runtime_freeze_written": False,
        "population_started": False,
        "population_decision": None,
    }


def _audit_scope() -> dict[str, Any]:
    return {
        "preflight_passed": True,
        "development_completed": True,
        "development_decision": "go",
        "future_audit_started": True,
        "future_audit_decision": "no_go",
        "fit_state": "not_started",
        "audit50_fit_rows": 0,
        "fit_artifact_runtime_eligible": False,
        "runtime_freeze_written": False,
        "population_started": False,
        "population_decision": None,
    }


def test_development_no_go_closeout_reopens_hash_and_receipt_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_read_only_revalidators(monkeypatch)
    root, contracts = _frozen_root(tmp_path)
    artifacts: dict[str, dict[str, str]] = {}
    decision = _add_run(
        root, artifacts, stage="development", decision_value="no_go"
    )
    payload = _closeout(
        contracts,
        artifacts,
        stage="development200",
        status="complete_no_go_development",
        terminal_key="development_decision",
        terminal=decision,
        scope=_development_scope(),
    )
    closeout_path = root / "closeout.json"
    closeout_path.write_bytes(_bytes(payload))

    result = closeout.validate_attempt13_closeout(root, closeout_path=closeout_path)
    assert result["status"] == "validated_complete_no_go_development"
    assert result["terminal_artifact"] == "development_decision"
    assert result["failed_gates"] == ["fires_total"]

    forged = copy.deepcopy(decision)
    forged["metrics"] = {"forged_without_recomputing_rows": True}
    monkeypatch.setattr(
        closeout.development_selector,
        "select_attempt13_development",
        lambda **_kwargs: forged,
    )
    with pytest.raises(ValueError, match="read-only selector recomputation"):
        closeout.validate_attempt13_closeout(root, closeout_path=closeout_path)
    _stub_read_only_revalidators(monkeypatch)

    receipt_path = root / artifacts["development_decision_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["decision_sha256"] = "f" * 64
    receipt_path.write_bytes(_bytes(receipt))
    artifacts["development_decision_receipt"]["sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    closeout_path.write_bytes(_bytes(payload))
    with pytest.raises(ValueError, match="receipt decision SHA"):
        closeout.validate_attempt13_closeout(root, closeout_path=closeout_path)


def test_audit_no_go_closeout_reopens_development_freeze_and_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_read_only_revalidators(monkeypatch)
    root, contracts = _frozen_root(tmp_path)
    artifacts: dict[str, dict[str, str]] = {}
    development = _add_run(
        root, artifacts, stage="development", decision_value="go"
    )
    _add_development_freeze(root, artifacts, development)
    audit = _add_run(root, artifacts, stage="audit", decision_value="no_go")
    payload = _closeout(
        contracts,
        artifacts,
        stage="audit50",
        status="complete_no_go_audit50",
        terminal_key="audit_decision",
        terminal=audit,
        scope=_audit_scope(),
    )
    path = root / "closeout.json"
    path.write_bytes(_bytes(payload))

    result = closeout.validate_attempt13_closeout(root, closeout_path=path)
    assert result["status"] == "validated_complete_no_go_audit50"
    assert result["validated_artifacts"] == len(artifacts)
    assert result["explicit_opt_in_authorized"] is False


@pytest.mark.parametrize("status", ("complete_go", "complete_no_go"))
def test_population_closeout_reuses_record_recomputation_and_preserves_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str
) -> None:
    _stub_read_only_revalidators(monkeypatch)
    root, contracts = _frozen_root(tmp_path)
    artifacts: dict[str, dict[str, str]] = {}
    development = _add_run(
        root, artifacts, stage="development", decision_value="go"
    )
    _add_development_freeze(root, artifacts, development)
    _add_run(root, artifacts, stage="audit", decision_value="go")
    for key, relative, raw in (
        ("training_model", "evidence/population/training.pkl", b"training"),
        ("runtime_source_archive", "evidence/population/runtime.zip", b"runtime"),
        ("runtime_model", "evidence/population/model.pkl", b"model"),
        ("population_source_archive", "evidence/population/source.zip", b"source"),
    ):
        artifacts[key] = _write(root, relative, raw, raw=True)
    artifacts["training_manifest"] = _write(
        root, "evidence/population/training.json", {}
    )
    artifacts["runtime_source_manifest"] = _write(
        root, "evidence/population/runtime_source.json", {}
    )
    artifacts["runtime_freeze"] = _write(
        root, "evidence/population/runtime_freeze.json", {}
    )
    monkeypatch.setattr(closeout, "_validate_training_artifacts", lambda *_a, **_k: None)
    preflight = {
        "schema": "hu_m43_attempt13_population_launch_preflight_v1",
        "status": "pass",
        "profile_id": "stage20_m4_attempt13",
        "baseline_profile": "stage19_p0",
        "model_schema": "hu_m43_attempt13_t1_second_distilled_selector_v1",
        "artifact_schema": "hu_m43_attempt13_t1_second_distilled_pickle_v1",
        "feature_schema": "hu_m43_attempt13_lambda_all_legal_public_infoset_features_v1",
        "head_schema": "hu_m43_attempt13_policy_delta_safe_tail_heads_v1",
        "action_score_mode": "attempt13_lambda_all_legal_distilled_safe_selector_v1",
        "model_sha256": artifacts["runtime_model"]["sha256"],
        "training_manifest_sha256": artifacts["training_manifest"]["sha256"],
        "runtime_freeze_sha256": artifacts["runtime_freeze"]["sha256"],
        "runtime_source_archive_sha256": artifacts["runtime_source_archive"]["sha256"],
        "runtime_source_manifest_sha256": artifacts["runtime_source_manifest"]["sha256"],
        "runtime_source_closure_sha256": "4" * 64,
        "runtime_semantic_closure_sha256": "5" * 64,
        "source_model_manifest_sha256": "6" * 64,
        "source_native_manifest_sha256": "7" * 64,
        "runtime_dependency_closure_sha256": "8" * 64,
        "seed_registry_sha256": "9" * 64,
        "development_decision_sha256": artifacts["development_decision"]["sha256"],
        "development_selector_receipt_sha256": artifacts["development_decision_receipt"]["sha256"],
        "development_pass_freeze_sha256": artifacts["development_go_freeze"]["sha256"],
        "audit50_decision_sha256": artifacts["audit_decision"]["sha256"],
        "audit50_selector_receipt_sha256": artifacts["audit_decision_receipt"]["sha256"],
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    artifacts["population_preflight"] = _write(
        root, "evidence/population/preflight.json", preflight
    )
    run_manifest = {
        "schema": "hu_m43_attempt13_population_spot_manifest_v1",
        "run_name": "attempt13-population-synthetic",
        "population_plan": {"sha256": closeout.acceptance.ATTEMPT13_POPULATION_PLAN_SHA256},
        "source": {"sha256": artifacts["population_source_archive"]["sha256"]},
        "runtime": {
            "model_sha256": artifacts["runtime_model"]["sha256"],
            "training_manifest_sha256": artifacts["training_manifest"]["sha256"],
            "runtime_freeze_sha256": artifacts["runtime_freeze"]["sha256"],
            "runtime_source_archive_sha256": artifacts["runtime_source_archive"]["sha256"],
            "runtime_source_manifest_sha256": artifacts["runtime_source_manifest"]["sha256"],
        },
        "launch_preflight": copy.deepcopy(preflight),
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    artifacts["population_run_manifest"] = _write(
        root, "evidence/population/run_manifest.json", run_manifest
    )
    evaluation = {
        "paired_seeds_per_opponent": 1000,
        "invalid_counterfactuals": 0,
        "nonfire_cancellation_mismatches": 0,
        "population": {"all_seats": {"overrides": 300}},
    }
    artifacts["population_evaluation"] = _write(
        root, "evidence/population/evaluation.json", evaluation
    )
    artifacts["population_records"] = _write(
        root, "evidence/population/records.jsonl", b"{}\n", raw=True
    )
    merge_shards = [
        {
            "evaluation_sha256": f"{index + 10:064x}",
            "records_sha256": f"{index + 40:064x}",
        }
        for index in range(20)
    ]
    artifacts["population_merge_manifest"] = _write(
        root, "evidence/population/merge.json", {"shards": merge_shards}
    )
    acceptance_status = {
        "schema": "hu_m43_attempt13_population_acceptance_status_v1",
        "status": status,
        "gates": [
            {
                "name": "runtime_artifact_hash_chain",
                "passed": True,
                "observed": True,
                "requirement": "bound runtime",
            },
            {
                "name": "nonfire_counterfactual_cancellation",
                "passed": True,
                "observed": True,
                "requirement": "exact cancellation",
            },
            {
                "name": "realized_population",
                "passed": status == "complete_go",
                "observed": status == "complete_go",
                "requirement": "all frozen gates",
            }
        ],
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
        "automatic_activation_authorized": False,
        "explicit_opt_in_authorized": status == "complete_go",
    }
    artifacts["population_acceptance_status"] = _write(
        root, "evidence/population/acceptance.json", acceptance_status
    )
    receipt = {
        "schema": "hu_m43_attempt13_population_spot_receipt_v1",
        "status": "verified_and_merged",
        "run_name": "attempt13-population-synthetic",
        "run_manifest_sha256": artifacts["population_run_manifest"]["sha256"],
        "population_plan_sha256": closeout.acceptance.ATTEMPT13_POPULATION_PLAN_SHA256,
        "model_sha256": artifacts["runtime_model"]["sha256"],
        "population_source_archive_sha256": artifacts["population_source_archive"]["sha256"],
        "runtime_source_archive_sha256": artifacts["runtime_source_archive"]["sha256"],
        "runtime_source_manifest_sha256": artifacts["runtime_source_manifest"]["sha256"],
        "runtime_source_closure_sha256": preflight["runtime_source_closure_sha256"],
        "runtime_semantic_closure_sha256": preflight["runtime_semantic_closure_sha256"],
        "source_model_manifest_sha256": preflight["source_model_manifest_sha256"],
        "source_native_manifest_sha256": preflight["source_native_manifest_sha256"],
        "runtime_dependency_closure_sha256": preflight["runtime_dependency_closure_sha256"],
        "seed_registry_sha256": preflight["seed_registry_sha256"],
        "development_decision_sha256": artifacts["development_decision"]["sha256"],
        "development_selector_receipt_sha256": artifacts["development_decision_receipt"]["sha256"],
        "development_pass_freeze_sha256": artifacts["development_go_freeze"]["sha256"],
        "audit50_decision_sha256": artifacts["audit_decision"]["sha256"],
        "audit50_selector_receipt_sha256": artifacts["audit_decision_receipt"]["sha256"],
        "development200_full_fit_bound": True,
        "audit50_one_shot_go_bound": True,
        "model_schema": preflight["model_schema"],
        "artifact_schema": preflight["artifact_schema"],
        "feature_schema": preflight["feature_schema"],
        "head_schema": preflight["head_schema"],
        "action_score_mode": preflight["action_score_mode"],
        "profile_id": preflight["profile_id"],
        "baseline_profile": preflight["baseline_profile"],
        "evaluation_path": str(
            (root / artifacts["population_evaluation"]["path"]).resolve()
        ),
        "evaluation_sha256": artifacts["population_evaluation"]["sha256"],
        "records_path": str(
            (root / artifacts["population_records"]["path"]).resolve()
        ),
        "records_sha256": artifacts["population_records"]["sha256"],
        "merge_manifest_sha256": artifacts["population_merge_manifest"]["sha256"],
        "paired_seeds_per_opponent": 1000,
        "valid_overrides": 300,
        "invalid_counterfactuals": 0,
        "nonfire_cancellation_mismatches": 0,
        "shards": [
            {
                "shard": index,
                "done_sha256": f"{index + 70:064x}",
                "evaluation_sha256": merge_shards[index]["evaluation_sha256"],
                "records_sha256": merge_shards[index]["records_sha256"],
            }
            for index in range(20)
        ],
        "teacher_calibration_locked_content_received": False,
        "acceptance_status_sha256": artifacts["population_acceptance_status"]["sha256"],
        "acceptance_decision": status,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
        "received_at": "2026-07-16T00:00:00+00:00",
    }
    artifacts["population_receipt"] = _write(
        root, "evidence/population/receipt.json", receipt
    )
    called: dict[str, Any] = {}

    def _recompute(**kwargs: Any) -> dict[str, Any]:
        called.update(kwargs)
        return copy.deepcopy(acceptance_status)

    monkeypatch.setattr(
        closeout.acceptance, "validate_attempt13_population_acceptance", _recompute
    )
    monkeypatch.setattr(
        closeout.acceptance,
        "build_attempt13_population_preflight",
        lambda **_kwargs: copy.deepcopy(preflight),
    )
    scope = {
        "preflight_passed": True,
        "development_completed": True,
        "development_decision": "go",
        "future_audit_started": True,
        "future_audit_decision": "go",
        "fit_state": "completed_development200_only",
        "audit50_fit_rows": 0,
        "fit_artifact_runtime_eligible": False,
        "runtime_freeze_written": True,
        "population_started": True,
        "population_decision": status,
    }
    payload = _closeout(
        contracts,
        artifacts,
        stage="population",
        status=status,
        terminal_key="population_acceptance_status",
        terminal=acceptance_status,
        scope=scope,
        population_recompute={
            "runtime_source_root": ".",
            "runtime_dependency_root": ".",
        },
    )
    path = root / "closeout.json"
    path.write_bytes(_bytes(payload))

    result = closeout.validate_attempt13_closeout(root, closeout_path=path)
    assert result["status"] == f"validated_{status}"
    assert result["explicit_opt_in_authorized"] is (status == "complete_go")
    assert called["records"] == [{}]
    assert called["source_hashes"]["records"] == artifacts["population_records"]["sha256"]

    changed_preflight = copy.deepcopy(preflight)
    changed_preflight["runtime_binding_verified"] = False
    monkeypatch.setattr(
        closeout.acceptance,
        "build_attempt13_population_preflight",
        lambda **_kwargs: changed_preflight,
    )
    with pytest.raises(ValueError, match="full provenance recomputation"):
        closeout.validate_attempt13_closeout(root, closeout_path=path)


def test_population_no_go_claims_follow_the_actual_failed_gates() -> None:
    terminal = {
        "gates": [
            {
                "name": "runtime_artifact_hash_chain",
                "passed": False,
                "observed": False,
                "requirement": "bound runtime",
            },
            {
                "name": "nonfire_counterfactual_cancellation",
                "passed": False,
                "observed": False,
                "requirement": "exact cancellation",
            },
        ]
    }
    science = _science("population")
    science["nonfire_full_trajectory_cancellation_verified"] = False
    activation = _activation("population", "complete_no_go")
    activation["runtime_binding_verified"] = False

    closeout._validate_science(
        science, terminal_stage="population", terminal=terminal
    )
    closeout._validate_activation(
        activation,
        terminal_stage="population",
        status="complete_no_go",
        terminal=terminal,
    )

    science["nonfire_full_trajectory_cancellation_verified"] = True
    with pytest.raises(ValueError, match="full trajectory cancellation"):
        closeout._validate_science(
            science, terminal_stage="population", terminal=terminal
        )


def test_spot_chain_binds_audit_to_the_development_go_freeze(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_read_only_revalidators(monkeypatch)
    root, _contracts = _frozen_root(tmp_path)
    artifacts: dict[str, dict[str, str]] = {}
    development = _add_run(root, artifacts, stage="development", decision_value="go")
    _add_development_freeze(root, artifacts, development)
    _add_run(root, artifacts, stage="audit", decision_value="no_go")
    paths = {
        name: (root / identity["path"]).resolve()
        for name, identity in artifacts.items()
    }

    with pytest.raises(ValueError, match="preceding gate SHA-256"):
        closeout._validate_spot_chain(
            stage="audit",
            paths=paths,
            expected_preceding_gate_sha256="f" * 64,
        )


def test_artifact_resolution_rejects_parent_escape_and_in_repo_symlink(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    identity = {
        "path": "../outside.json",
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="repo-relative"):
        closeout._resolve_file(root, identity, "escaped artifact")

    target = root / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = root / "link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("creating a Windows symlink requires unavailable privileges")
    linked = {
        "path": "link.json",
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="symlink or junction"):
        closeout._resolve_file(root, linked, "linked artifact")


def test_closeout_rejects_any_automatic_or_runtime_activation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_read_only_revalidators(monkeypatch)
    root, contracts = _frozen_root(tmp_path)
    artifacts: dict[str, dict[str, str]] = {}
    decision = _add_run(
        root, artifacts, stage="development", decision_value="no_go"
    )
    payload = _closeout(
        contracts,
        artifacts,
        stage="development200",
        status="complete_no_go_development",
        terminal_key="development_decision",
        terminal=decision,
        scope=_development_scope(),
    )
    payload["activation"]["runtime_policy_activated"] = True
    path = root / "closeout.json"
    path.write_bytes(_bytes(payload))

    with pytest.raises(ValueError, match="runtime_policy_activated"):
        closeout.validate_attempt13_closeout(root, closeout_path=path)
