from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ofc_regular.evaluate_hu_m43_locked_holdout as locked_evaluator
import ofc_regular.freeze_hu_m43_model as model_freezer

from ofc_regular.hu_m43_pilot_contract import (
    M43_DATA_CONTRACT_SCHEMA,
    M43_FREEZE_SCHEMA,
    M43_LOCKED_RECEIPT_SCHEMA,
    audit_frozen_exclusions,
    build_ordered_teacher_shard_binding,
    canonical_manifest_sha256,
    load_and_validate_plan,
    paired_risk_targets,
    partition_calibration_rows,
    validate_locked_holdout_receipt,
    validate_data_contract_binding,
    validate_model_threshold_freeze,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "hu_joint_policy_m43_pilot.json"


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _summary(
    *, mean: float = 1.5, se: float = 0.5, minimum: float = -9.0
) -> dict:
    return {
        "schema": "hu_m4_paired_delta_summary_v1",
        "count": 16,
        "mean": mean,
        "standard_error": se,
        "std": 2.0,
        "min": minimum,
        "p01": -7.0,
        "p05": -4.0,
        "p25": -1.0,
        "p50": 1.0,
        "p75": 2.0,
        "p95": 4.0,
        "p99": 5.0,
        "max": 6.0,
        "lt0_rate": 0.25,
        "le_neg6_rate": 0.1,
        "le_neg12_rate": 0.0,
        "le_neg20_rate": 0.0,
    }


def _freeze(contract: dict) -> dict:
    return {
        "schema": M43_FREEZE_SCHEMA,
        "status": "model_and_threshold_frozen_locked_unopened",
        "data_contract_sha256": contract["contract_sha256"],
        "data_contract_resolved_path": "C:/m43/data_contract.json",
        "locked_consumption_marker_resolved_path": (
            "C:/m43/M43_LOCKED_CONSUMED.json"
        ),
        "plan_sha256": contract["plan_sha256"],
        "prior_freshness_audit_sha256": contract["prior_exclusions"][
            "audit_sha256"
        ],
        "base_audit_sha256": contract["base_audit_sha256"],
        "teacher_shards_all_splits_sha256": contract[
            "teacher_shards_all_splits_sha256"
        ],
        "training_manifest_sha256": "d" * 64,
        "model_sha256": "a" * 64,
        "model_id": "m43-test",
        "frozen_threshold": 0.7,
        "ranker_training_sources": ["train"],
        "ranker_crossfit_source": "train",
        "safety_calibrator_sources": ["train_oof", "calibration.safety_fit"],
        "threshold_selection_source": "calibration.threshold_lock",
        "model_selection_sources": ["train"],
        "locked_holdout_used_for_training_selection_or_threshold": False,
        "model_or_threshold_locked_label_access_count_at_freeze": 0,
        "current_profile_resolved": False,
        "runtime_policy_activated": False,
        "pilot_can_promote_policy": False,
        "fresh_population_acceptance_required": True,
        "minimum_population_valid_overrides": 300,
    }


def _contract() -> dict:
    contract = {
        "schema": M43_DATA_CONTRACT_SCHEMA,
        "status": "pass_fresh_data_sealed_for_model_freeze",
        "plan_sha256": "e" * 64,
        "base_audit_sha256": "f" * 64,
        "teacher_shards_all_splits_sha256": "1" * 64,
        "teacher_shards": {
            "splits": {
                "locked_holdout": {"ordered_shards_sha256": "3" * 64}
            }
        },
        "prior_exclusions": {"audit_sha256": "2" * 64},
        "splits": {"locked_holdout": {"identity_sha256": "c" * 64}},
    }
    contract["contract_sha256"] = canonical_manifest_sha256(contract)
    return contract


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _binding_row(split: str, seed: int, *, score: float = 1.0) -> dict:
    return {
        "split": split,
        "hand_seed": seed,
        "observation_fingerprint": f"{seed:064x}",
        "actions": [
            {
                "action_key": f"action-{seed}",
                "score": score,
                "delta_vs_baseline": score,
                "paired_delta_vs_baseline": {
                    "mean": score,
                    "standard_error": 0.5,
                    "p05": -4.0,
                    "p01": -7.0,
                    "min": -9.0,
                },
            }
        ],
    }


def _sealed_binding_contract(
    tmp_path: Path, split_paths: dict[str, list[Path]]
) -> dict:
    plan = load_and_validate_plan(PLAN_PATH)
    prior = audit_frozen_exclusions(plan, repo_root=ROOT)
    clean_prior = {
        key: value for key, value in prior.items() if not isinstance(value, set)
    }
    clean_prior["audit_sha256"] = canonical_manifest_sha256(clean_prior)
    base_audit = {
        "schema": "hu_m4_t1_second_data_audit_v1",
        "status": "pass",
        "gates": {
            "hidden_discard_safe": True,
            "all_legal_actions_mapped": True,
            "candidate_evaluation_rng_disjoint": True,
            "paired_common_future_delta_contract": True,
            "seed_ranges_disjoint": True,
            "fingerprints_disjoint": True,
            "current_profile_resolved": False,
            "holdout_threshold_search_allowed": False,
        },
    }
    teacher_shards = {
        "schema": "hu_m43_ordered_teacher_shards_v1",
        "splits": {
            split: build_ordered_teacher_shard_binding(
                paths, repo_root=ROOT, expected_split=split
            )
            for split, paths in split_paths.items()
        },
    }
    teacher_shards["all_splits_sha256"] = canonical_manifest_sha256(
        teacher_shards
    )
    contract = {
        "schema": M43_DATA_CONTRACT_SCHEMA,
        "status": "pass_fresh_data_sealed_for_model_freeze",
        "plan_sha256": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
        "base_audit": base_audit,
        "base_audit_sha256": canonical_manifest_sha256(base_audit),
        "prior_exclusions": clean_prior,
        "teacher_shards": teacher_shards,
        "teacher_shards_all_splits_sha256": teacher_shards[
            "all_splits_sha256"
        ],
        "splits": {
            "locked_holdout": {
                "identity_sha256": teacher_shards["splits"]["locked_holdout"][
                    "ordered_shards"
                ][0]["identity_sha256"]
            }
        },
    }
    contract["contract_sha256"] = canonical_manifest_sha256(contract)
    return contract


def test_frozen_plan_has_usable_200_root_pre_promotion_budget() -> None:
    plan = load_and_validate_plan(PLAN_PATH)
    assert plan["budget"] == {
        "kind": "bounded_pre_promotion_pilot",
        "roots": 200,
        "max_roots": 1000,
        "roots_per_shard": 10,
        "shards": 20,
        "spot_allowed_after_local_correctness_and_speed_gates": True,
        "large_scale_allowed": False,
    }
    assert {name: plan["splits"][name]["roots"] for name in plan["splits"]} == {
        "train": 100,
        "calibration": 60,
        "locked_holdout": 40,
    }
    assert plan["calibration_partition"]["safety_fit_roots"] == 30
    assert plan["calibration_partition"]["threshold_lock_roots"] == 30
    assert (
        plan["calibration_partition"][
            "minimum_threshold_lock_fires_for_positive_pilot_signal"
        ]
        == 10
    )
    assert plan["teacher_search"]["candidate_samples"] == 2
    assert plan["teacher_search"]["evaluation_samples"] == 16
    assert plan["promotion_boundary"]["pilot_can_promote_policy"] is False
    assert plan["promotion_boundary"]["minimum_population_valid_overrides"] == 300
    assert plan["activation_guards"]["current_profile_changed"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda plan: plan["activation_guards"].__setitem__(
                "current_profile_changed", True
            ),
            "activation guard",
        ),
        (
            lambda plan: plan["calibration_partition"].__setitem__(
                "threshold_selection_source", "locked_holdout"
            ),
            "thresholds may be selected only",
        ),
        (
            lambda plan: plan["promotion_boundary"].__setitem__(
                "minimum_population_valid_overrides", 29
            ),
            "may not be lowered",
        ),
        (
            lambda plan: plan["teacher_search"].__setitem__(
                "evaluation_samples", 8
            ),
            "evaluation sample count changed",
        ),
    ],
)
def test_plan_rejects_weakened_boundaries(tmp_path: Path, mutate, message: str) -> None:
    plan = _plan()
    mutate(plan)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_and_validate_plan(path)


def test_all_available_prior_teacher_splits_are_hash_bound_and_excluded() -> None:
    plan = load_and_validate_plan(PLAN_PATH)
    audit = audit_frozen_exclusions(plan, repo_root=ROOT)
    assert len(audit["sources"]) == 10
    assert audit["source_records"] == 252
    assert {row["milestone"] for row in audit["sources"]} == {
        "M4",
        "M4.1",
        "M4.2",
    }
    assert audit["unique_identities"] > 0
    assert audit["identity_sha256"] != "0" * 64

    planned_seeds = {
        int(spec["seed_start"]) + int(spec["seed_stride"]) * index
        for spec in plan["splits"].values()
        for index in range(int(spec["roots"]))
    }
    assert planned_seeds.isdisjoint(audit["hand_seeds"])


def test_paired_risk_targets_are_candidate_minus_baseline_mean_se_and_losses() -> None:
    action = {
        "delta_vs_baseline": 1.5,
        "delta_se_vs_baseline": 0.5,
        "paired_delta_vs_baseline": _summary(),
    }
    assert paired_risk_targets(action) == {
        "paired_delta_mean": 1.5,
        "paired_delta_se": 0.5,
        "downside_loss_p95": 4.0,
        "downside_loss_p99": 7.0,
        "downside_loss_max": 9.0,
    }

    broken = copy.deepcopy(action)
    broken["delta_se_vs_baseline"] = 0.25
    with pytest.raises(ValueError, match="SE field mismatch"):
        paired_risk_targets(broken)

    broken = copy.deepcopy(action)
    broken["paired_delta_vs_baseline"]["p01"] = -10.0
    with pytest.raises(ValueError, match="quantiles are not monotone"):
        paired_risk_targets(broken)


def test_calibration_partition_is_deterministic_profile_stratified_and_disjoint() -> None:
    plan = _plan()
    rows = []
    for profile_index, profile in enumerate(plan["root_population"]):
        for row_index in range(12):
            number = profile_index * 12 + row_index + 1
            rows.append(
                {
                    "hand_seed": 4_000_000_000 + number,
                    "observation_fingerprint": f"{number:064x}",
                    "provenance": {"root_profile": profile["profile"]},
                }
            )
    first = partition_calibration_rows(rows, plan)
    second = partition_calibration_rows(list(reversed(rows)), plan)
    assert first == second
    assert first["safety_fit"]["records"] == 30
    assert first["threshold_lock"]["records"] == 30
    assert set(first["safety_fit"]["profile_counts"].values()) == {6}
    assert set(first["threshold_lock"]["profile_counts"].values()) == {6}
    fit = {
        (row["hand_seed"], row["observation_fingerprint"])
        for row in first["safety_fit"]["identities"]
    }
    lock = {
        (row["hand_seed"], row["observation_fingerprint"])
        for row in first["threshold_lock"]["identities"]
    }
    assert len(fit) == len(lock) == 30
    assert fit.isdisjoint(lock)


def test_freeze_requires_calibration_only_threshold_and_zero_holdout_access() -> None:
    contract = _contract()
    freeze = _freeze(contract)
    validate_model_threshold_freeze(freeze, data_contract=contract)

    unsafe = dict(freeze, threshold_selection_source="locked_holdout")
    with pytest.raises(ValueError, match="unsafe freeze source"):
        validate_model_threshold_freeze(unsafe, data_contract=contract)

    unsafe = dict(freeze, model_or_threshold_locked_label_access_count_at_freeze=1)
    with pytest.raises(ValueError, match="reached model or threshold selection"):
        validate_model_threshold_freeze(unsafe, data_contract=contract)


def test_locked_holdout_receipt_is_one_shot_and_cannot_activate() -> None:
    contract = _contract()
    freeze = _freeze(contract)
    receipt = {
        "schema": M43_LOCKED_RECEIPT_SCHEMA,
        "status": "evaluated_once_diagnostic_only_no_activation",
        "freeze_manifest_sha256": canonical_manifest_sha256(freeze),
        "data_contract_sha256": contract["contract_sha256"],
        "model_sha256": freeze["model_sha256"],
        "frozen_threshold": freeze["frozen_threshold"],
        "locked_identity_sha256": "c" * 64,
        "locked_teacher_shards_sha256": "3" * 64,
        "consumption_marker_sha256": "4" * 64,
        "evaluation_pass_count": 1,
        "threshold_search_performed": False,
        "model_selection_performed": False,
        "feature_selection_performed": False,
        "current_profile_resolved": False,
        "runtime_policy_activated": False,
        "policy_promoted": False,
        "requires_fresh_population_acceptance": True,
        "minimum_population_valid_overrides": 300,
        "teacher_value_status": "diagnostic_not_realized_match_ev",
    }
    validate_locked_holdout_receipt(
        receipt, freeze_manifest=freeze, data_contract=contract
    )

    with pytest.raises(ValueError, match="strictly one-shot"):
        validate_locked_holdout_receipt(
            dict(receipt, evaluation_pass_count=2),
            freeze_manifest=freeze,
            data_contract=contract,
        )
    with pytest.raises(ValueError, match="policy_promoted"):
        validate_locked_holdout_receipt(
            dict(receipt, policy_promoted=True),
            freeze_manifest=freeze,
            data_contract=contract,
        )


def test_contract_binds_ordered_shard_bytes_and_identity_preserving_targets(
    tmp_path: Path,
) -> None:
    train_a = tmp_path / "train_a.jsonl"
    train_b = tmp_path / "train_b.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    locked = tmp_path / "locked.jsonl"
    _write_rows(train_a, [_binding_row("train", 1)])
    _write_rows(train_b, [_binding_row("train", 2)])
    _write_rows(calibration, [_binding_row("calibration", 3)])
    _write_rows(locked, [_binding_row("locked_holdout", 4)])
    split_paths = {
        "train": [train_a, train_b],
        "calibration": [calibration],
        "locked_holdout": [locked],
    }
    contract = _sealed_binding_contract(tmp_path, split_paths)
    result = validate_data_contract_binding(
        contract,
        plan_path=PLAN_PATH,
        repo_root=ROOT,
        train=[train_a, train_b],
    )
    assert result["status"] == "pass"
    assert list(result["verified_splits"]) == ["train"]

    with pytest.raises(ValueError, match="ordered path/content binding mismatch"):
        validate_data_contract_binding(
            contract,
            plan_path=PLAN_PATH,
            repo_root=ROOT,
            train=[train_b, train_a],
        )

    mutations = []
    score_mutation = _binding_row("train", 1, score=99.0)
    mutations.append(score_mutation)
    action_mutation = _binding_row("train", 1)
    action_mutation["actions"][0]["action_key"] = "different-action-same-identity"
    mutations.append(action_mutation)
    tail_mutation = _binding_row("train", 1)
    tail_mutation["actions"][0]["paired_delta_vs_baseline"]["p01"] = -35.0
    mutations.append(tail_mutation)
    for mutated in mutations:
        _write_rows(train_a, [mutated])
        with pytest.raises(
            ValueError, match="ordered path/content binding mismatch"
        ):
            validate_data_contract_binding(
                contract,
                plan_path=PLAN_PATH,
                repo_root=ROOT,
                train=[train_a, train_b],
            )


def test_separate_locked_evaluator_claims_once_and_never_selects_or_activates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    locked = tmp_path / "locked.jsonl"
    _write_rows(train, [_binding_row("train", 11)])
    _write_rows(calibration, [_binding_row("calibration", 12)])
    _write_rows(locked, [_binding_row("locked_holdout", 13)])
    contract = _sealed_binding_contract(
        tmp_path,
        {"train": [train], "calibration": [calibration], "locked_holdout": [locked]},
    )
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"already-saved-frozen-model")
    freeze = _freeze(contract)
    freeze["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    freeze["model_id"] = "m43-one-shot-test"
    contract_path = tmp_path / "data_contract.json"
    freeze_path = tmp_path / "freeze.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    freeze["data_contract_resolved_path"] = str(contract_path.resolve())
    freeze["locked_consumption_marker_resolved_path"] = str(
        (tmp_path / "M43_LOCKED_CONSUMED.json").resolve()
    )
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    fake_model = SimpleNamespace(
        action_score_mode="baseline_paired_delta_risk_ensemble_v3",
        model_id="m43-one-shot-test",
        safety_threshold=freeze["frozen_threshold"],
        safety_enabled=False,
    )
    monkeypatch.setattr(
        locked_evaluator,
        "HuM4JointActionModel",
        SimpleNamespace(load=lambda _path: fake_model),
    )
    monkeypatch.setattr(
        locked_evaluator, "prepare_teacher_samples", lambda rows: [object()]
    )
    monkeypatch.setattr(
        locked_evaluator,
        "evaluate_model",
        lambda model, samples, split: {
            "split": split,
            "samples": len(samples),
            "teacher_value_status": "diagnostic_not_match_EV",
        },
    )
    receipt_path = tmp_path / "receipt.json"
    marker_path = tmp_path / "M43_LOCKED_CONSUMED.json"
    receipt = locked_evaluator.evaluate_locked_holdout_once(
        model_path=model_path,
        freeze_manifest_path=freeze_path,
        data_contract_path=contract_path,
        plan_path=PLAN_PATH,
        repo_root=ROOT,
        locked_holdout=[locked],
        receipt_path=receipt_path,
        consumption_marker_path=marker_path,
    )
    assert receipt_path.is_file()
    assert marker_path.is_file()
    assert receipt["evaluation_pass_count"] == 1
    assert receipt["threshold_search_performed"] is False
    assert receipt["model_selection_performed"] is False
    assert receipt["runtime_policy_activated"] is False
    assert receipt["policy_promoted"] is False

    with pytest.raises(FileExistsError, match="receipt already exists"):
        locked_evaluator.evaluate_locked_holdout_once(
            model_path=model_path,
            freeze_manifest_path=freeze_path,
            data_contract_path=contract_path,
            plan_path=PLAN_PATH,
            repo_root=ROOT,
            locked_holdout=[locked],
            receipt_path=receipt_path,
            consumption_marker_path=marker_path,
        )


def test_freeze_producer_binds_saved_model_manifest_contract_and_only_train_cal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    locked = tmp_path / "locked.jsonl"
    _write_rows(train, [_binding_row("train", 21)])
    _write_rows(calibration, [_binding_row("calibration", 22)])
    _write_rows(locked, [_binding_row("locked_holdout", 23)])
    contract = _sealed_binding_contract(
        tmp_path,
        {"train": [train], "calibration": [calibration], "locked_holdout": [locked]},
    )
    binding = validate_data_contract_binding(
        contract,
        plan_path=PLAN_PATH,
        repo_root=ROOT,
        train=[train],
        calibration=[calibration],
    )
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"saved-before-freeze")
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    training_manifest = {
        "action_score_formula": {
            "mode": "baseline_paired_delta_risk_ensemble_v3",
            "baseline_action_score_exact_zero": True,
            "runtime_model": "stored_crossfit_fold_ensemble",
            "teacher_value_runtime_input": False,
            "teacher_lcb_runtime_gate": False,
        },
        "locked_holdout_used_for_threshold_or_training": False,
        "locked_holdout": {"status": "not_evaluated_pre_freeze"},
        "inputs": {"train": {}, "calibration": {}},
        "m43_data_contract": binding,
        "calibration": {
            "status": "go",
            "selected_threshold": 0.7,
            "threshold_selection_source": "calibration.threshold_lock",
            "safety_calibrator_sources": [
                "train_oof",
                "calibration.safety_fit",
            ],
        },
        "runtime_lock": {
            "single_joint_artifact": True,
            "candidate_model_sha256": model_sha,
            "safety_model_sha256": model_sha,
            "safety_threshold": 0.7,
        },
    }
    manifest_path = tmp_path / "training_manifest.json"
    contract_path = tmp_path / "data_contract.json"
    manifest_path.write_text(json.dumps(training_manifest), encoding="utf-8")
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    fake_model = SimpleNamespace(
        action_score_mode="baseline_paired_delta_risk_ensemble_v3",
        model_id="m43-freeze-test",
        safety_threshold=0.7,
        safety_enabled=True,
    )
    monkeypatch.setattr(
        model_freezer,
        "HuM4JointActionModel",
        SimpleNamespace(load=lambda _path: fake_model),
    )
    freeze = model_freezer.build_model_threshold_freeze(
        model_path=model_path,
        training_manifest_path=manifest_path,
        data_contract_path=contract_path,
        plan_path=PLAN_PATH,
        repo_root=ROOT,
        train=[train],
        calibration=[calibration],
    )
    assert freeze["status"] == "model_and_threshold_frozen_locked_unopened"
    assert freeze["model_sha256"] == model_sha
    assert freeze["frozen_threshold"] == 0.7
    assert freeze["model_or_threshold_locked_label_access_count_at_freeze"] == 0
    assert freeze["runtime_policy_activated"] is False
    assert freeze["pilot_can_promote_policy"] is False
