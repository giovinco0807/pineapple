from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from ofc_regular.action_key import action_key_from_payload
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m4_joint_model import (
    DELTA_SCORE_WEIGHT,
    POLICY_SCORE_CENTER,
    POLICY_SCORE_WEIGHT,
    VALUE_SCORE_WEIGHT,
    LEGACY_ACTION_SCORE_MODE,
    NEGATIVE_REGRET_ACTION_SCORE_MODE,
    PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
    PairedDeltaRiskFoldEstimator,
    HuM4JointActionModel,
    build_meta_rank_features,
    build_paired_action_features,
    canonical_action_argmax,
)
from ofc_regular.hu_m43_pilot_contract import canonical_manifest_sha256
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM, sample_to_matrix
from ofc_regular.policy import action_to_json
from ofc_regular.state import Board
from ofc_regular.train_hu_m4_joint_model import (
    TEACHER_VALUE_STATUS,
    _artifact_embedded_manifest,
    _fit_paired_delta_risk_fold,
    _load_m43_role_binding,
    _override_metrics,
    calibrate_safety,
    fit_joint_model,
    fit_joint_model_cross_fitted,
    parse_args,
    prepare_teacher_sample,
    prepare_teacher_samples,
    split_safety_calibration,
    train_from_files,
    validate_disjoint_splits,
)


def _observation(*, seat: str, deal_offset: int) -> ActorObservation:
    hero = Board.from_rows(
        top=["Ah"],
        middle=["Kd"],
        bottom=["2s", "3s", "4s"],
    )
    opponent = (
        Board.from_rows(
            top=["Qh"],
            middle=["Jd", "Td"],
            bottom=["5s", "6s", "7s", "8s"],
        )
        if seat == "second"
        else Board.from_rows(
            top=["Qh"],
            middle=["Jd"],
            bottom=["5s", "6s", "7s"],
        )
    )
    used = {*hero.all_cards(), *opponent.all_cards()}
    remaining = [card for card in ALL_CARDS if card not in used]
    dealt = tuple(remaining[deal_offset : deal_offset + 3])
    return ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_private_discards=(),
        seat=seat,  # type: ignore[arg-type]
        street="T1",
        to_act_order=seat,  # type: ignore[arg-type]
    )


def _row(seed: int, *, seat: str = "second", deal_offset: int = 0) -> dict:
    observation = _observation(seat=seat, deal_offset=deal_offset)
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    payloads = []
    for index, action in enumerate(actions):
        payload = action_to_json(observation.hero_board, action)
        # A deterministic, non-monotone target gives all heads both classes and
        # non-constant regression labels without relying on action enumeration.
        row_value = sum(
            {"top": 0.4, "middle": 1.1, "bottom": 1.7}[row]
            for _card, row in action.placements
        )
        payload["score"] = float(row_value - 0.2 * len(action.discards) + (index % 3) * 0.1)
        payload["score_se"] = float(0.05 + (index % 2) * 0.02)
        payloads.append(payload)
    return {
        "root_seed": seed,
        "hand_seed": seed + 100_000,
        "policy_observation": observation.to_dict(),
        "actions": payloads,
        "baseline_action_row_index": len(payloads) - 1,
    }


def _prepared(start_seed: int, count: int) -> list:
    base_offset = (start_seed // 1_000) % 10 * 3
    rows = [
        _row(
            start_seed + index,
            seat="first" if index % 2 else "second",
            deal_offset=base_offset + index * 3,
        )
        for index in range(count)
    ]
    return prepare_teacher_samples(rows)


def _paired_row(seed: int, *, seat: str = "second", deal_offset: int = 0) -> dict:
    row = _row(seed, seat=seat, deal_offset=deal_offset)
    baseline = row["baseline_action_row_index"]
    baseline_score = float(row["actions"][baseline]["score"])
    for index, action in enumerate(row["actions"]):
        delta = float(action["score"]) - baseline_score
        if index == baseline:
            standard_error = p05 = p01 = minimum = 0.0
        else:
            standard_error = 0.20 + 0.03 * (index % 4)
            p05 = delta - (1.0 + 0.1 * (index % 3))
            p01 = delta - (2.0 + 0.1 * (index % 3))
            minimum = delta - (3.0 + 0.1 * (index % 3))
        action["delta_vs_baseline"] = delta
        action["delta_se_vs_baseline"] = standard_error
        action["paired_delta_vs_baseline"] = {
            "mean": delta,
            "standard_error": standard_error,
            "p05": p05,
            "p01": p01,
            "min": minimum,
        }
    return row


def _paired_prepared(start_seed: int, count: int) -> list:
    base_offset = (start_seed // 1_000) % 10 * 3
    rows = [
        _paired_row(
            start_seed + index,
            seat="first" if index % 2 else "second",
            deal_offset=base_offset + index * 3,
        )
        for index in range(count)
    ]
    return prepare_teacher_samples(rows)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


class _ConstantRegression:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, matrix):
        return np.full(np.asarray(matrix).shape[0], self.value, dtype=np.float64)


def test_joint_model_heads_fixed_score_composition_and_versioned_roundtrip(tmp_path):
    samples = _prepared(1_000, 6)
    model = fit_joint_model(
        samples,
        iterations=3,
        max_leaf_nodes=3,
        learning_rate=0.1,
        seed=9,
    )
    sample = samples[0].policy_sample
    heads = model.predict_heads_sample(sample)

    assert heads.action_score.shape == (len(sample["actions"]),)
    np.testing.assert_allclose(
        heads.action_score,
        VALUE_SCORE_WEIGHT * heads.value
        + DELTA_SCORE_WEIGHT * heads.delta_vs_baseline
        + POLICY_SCORE_WEIGHT * (heads.policy_probability - POLICY_SCORE_CENTER),
    )
    np.testing.assert_allclose(model.predict_sample(sample), heads.action_score)
    assert np.all(heads.predicted_absolute_residual >= 0.0)

    calibrated, report = calibrate_safety(
        model,
        samples[:3],
        threshold_lock_samples=samples[3:],
        thresholds=(0.0, 0.5, 1.0),
        minimum_fires=1,
        iterations=3,
        max_leaf_nodes=3,
        seed=11,
    )
    artifact = tmp_path / "joint.pkl"
    calibrated.save(artifact)
    loaded = HuM4JointActionModel.load(artifact)
    np.testing.assert_allclose(loaded.predict_sample(sample), calibrated.predict_sample(sample))
    assert loaded.safety_threshold == report["selected_threshold"]


def test_teacher_feature_path_ignores_top_level_hidden_truth_and_rejects_nested_truth():
    first = _row(2_000)
    first["opponent_private_discards"] = ["Ac"]
    first["true_dead_cards"] = ["Ac"]
    second = deepcopy(first)
    second["opponent_private_discards"] = ["Kc"]
    second["true_dead_cards"] = ["Kc"]

    first_prepared = prepare_teacher_sample(first)
    second_prepared = prepare_teacher_sample(second)
    first_matrix, _ = sample_to_matrix(first_prepared.policy_sample)
    second_matrix, _ = sample_to_matrix(second_prepared.policy_sample)
    np.testing.assert_array_equal(first_matrix, second_matrix)
    assert "opponent_private_discards" in first_prepared.ignored_truth_keys

    unsafe = deepcopy(first)
    unsafe["policy_observation"]["opponent_private_discards"] = ["Ac"]
    with pytest.raises(ValueError, match="forbidden hidden/world"):
        prepare_teacher_sample(unsafe)


def test_explicit_splits_reject_root_or_hand_seed_and_observation_overlap():
    train = _prepared(3_000, 2)
    calibration = _prepared(4_000, 2)
    holdout = _prepared(5_000, 2)
    assert validate_disjoint_splits(train, calibration, holdout)["status"] == "pass"

    overlapping = [prepare_teacher_sample(_row(3_000, deal_offset=12))]
    with pytest.raises(ValueError, match="split overlap"):
        validate_disjoint_splits(train, overlapping, holdout)

    same_observation_new_seed = [prepare_teacher_sample(_row(9_999, deal_offset=9))]
    with pytest.raises(ValueError, match="observation_fingerprint"):
        validate_disjoint_splits(train, same_observation_new_seed, holdout)


def test_safety_partition_is_deterministic_and_has_zero_identity_overlap():
    samples = _prepared(6_000, 6)
    fit_a, lock_a, audit_a = split_safety_calibration(
        samples,
        safety_fit_ratio=0.5,
        split_seed=77,
        minimum_safety_fit_samples=2,
        minimum_threshold_lock_samples=2,
    )
    fit_b, lock_b, audit_b = split_safety_calibration(
        list(reversed(samples)),
        safety_fit_ratio=0.5,
        split_seed=77,
        minimum_safety_fit_samples=2,
        minimum_threshold_lock_samples=2,
    )

    assert audit_a == audit_b
    assert audit_a["status"] == "pass"
    assert audit_a["overlap"] == {
        "seed_value_count": 0,
        "observation_fingerprint_count": 0,
        "row_hash_count": 0,
    }
    assert [sample.observation_fingerprint for sample in fit_a] == [
        sample.observation_fingerprint for sample in fit_b
    ]
    assert [sample.observation_fingerprint for sample in lock_a] == [
        sample.observation_fingerprint for sample in lock_b
    ]
    for subset_name in ("safety_fit", "threshold_lock"):
        subset = audit_a[subset_name]
        assert subset["rows"] >= 2
        assert len(subset["seed_values"]["sha256"]) == 64
        assert len(subset["observation_fingerprints"]["sha256"]) == 64
        assert len(subset["row_hashes"]["sha256"]) == 64


def test_threshold_sweep_uses_lock_rows_not_fit_teacher_delta_magnitude():
    model = fit_joint_model(
        _prepared(7_000, 6),
        iterations=3,
        max_leaf_nodes=3,
        seed=31,
    )

    def force_delta(samples, values):
        forced = []
        for sample, delta in zip(samples, values, strict=True):
            heads = model.predict_heads_sample(sample.policy_sample)
            candidate = canonical_action_argmax(sample.policy_sample, heads.action_score)
            baseline = (candidate + 1) % len(sample.teacher_scores)
            scores = sample.teacher_scores.copy()
            scores[baseline] = 0.0
            scores[candidate] = float(delta)
            forced.append(
                replace(sample, teacher_scores=scores, baseline_index=baseline)
            )
        return forced

    fit_source = _prepared(8_000, 3)
    lock = force_delta(_prepared(9_000, 3), (2.0, -0.5, 1.0))
    fit_small_delta = force_delta(fit_source, (1.0, 1.0, 1.0))
    fit_large_delta = force_delta(fit_source, (100.0, 200.0, 300.0))
    kwargs = {
        "threshold_lock_samples": lock,
        "thresholds": (0.0, 0.5, 1.0),
        "minimum_fires": 1,
        "maximum_false_positive_rate": 1.0,
        "iterations": 3,
        "max_leaf_nodes": 3,
        "seed": 41,
    }
    _first_model, first = calibrate_safety(model, fit_small_delta, **kwargs)
    _second_model, second = calibrate_safety(model, fit_large_delta, **kwargs)

    assert first["threshold_source"] == "threshold_lock_subset_only"
    assert first["safety_fit"]["used_for_threshold_sweep"] is False
    assert first["threshold_lock"]["used_for_estimator_fit"] is False
    assert first["threshold_sweep"] == second["threshold_sweep"]
    assert first["selected_threshold"] == second["selected_threshold"]


def test_file_training_freezes_calibration_threshold_before_locked_holdout(tmp_path):
    train_rows = [_row(10_000 + i, seat="first" if i % 2 else "second", deal_offset=i * 3) for i in range(4)]
    calibration_rows = [_row(20_000 + i, deal_offset=15 + i * 3) for i in range(3)]
    holdout_rows = [_row(30_000 + i, deal_offset=27 + i * 3) for i in range(3)]
    train_path = tmp_path / "train.jsonl"
    calibration_path = tmp_path / "calibration.jsonl"
    holdout_path = tmp_path / "locked.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(calibration_path, calibration_rows)
    _write_jsonl(holdout_path, holdout_rows)

    model_path = tmp_path / "model.pkl"
    manifest_path = tmp_path / "manifest.json"
    manifest = train_from_files(
        train_path=train_path,
        calibration_path=calibration_path,
        locked_holdout_path=holdout_path,
        output_model=model_path,
        manifest_output=manifest_path,
        iterations=3,
        max_leaf_nodes=3,
        thresholds=(0.0, 0.5, 1.0),
        minimum_calibration_fires=1,
        minimum_safety_fit_samples=1,
        minimum_threshold_lock_samples=1,
        seed=21,
    )

    loaded = HuM4JointActionModel.load(model_path)
    assert loaded.safety_threshold == manifest["calibration"]["selected_threshold"]
    assert manifest["threshold_adaptation_after_calibration"] is False
    assert manifest["locked_holdout_used_for_threshold_or_training"] is False
    assert manifest["teacher_value_status"] == TEACHER_VALUE_STATUS
    assert manifest["teacher_value_runtime_gate"] is False
    assert manifest["hidden_discard_safety"]["status"] == "pass"
    assert manifest["split_integrity"]["seed_overlap_count"] == 0
    assert set(manifest["inputs"]["train"]) == {"path", "sha256", "rows"}
    assert manifest["runtime_lock"]["single_joint_artifact"] is True
    assert (
        manifest["runtime_lock"]["candidate_model_sha256"]
        == manifest["runtime_lock"]["safety_model_sha256"]
    )
    assert len(manifest["runtime_lock"]["candidate_model_sha256"]) == 64
    assert manifest_path.exists()


def test_repeatable_split_cli_preserves_argument_order():
    args = parse_args(
        [
            "--train",
            "train_a.jsonl",
            "--train",
            "train_b.jsonl",
            "--calibration",
            "cal_a.jsonl",
            "--calibration",
            "cal_b.jsonl",
            "--locked-holdout",
            "hold_a.jsonl",
            "--locked-holdout",
            "hold_b.jsonl",
            "--output-model",
            "model.pkl",
            "--manifest-output",
            "manifest.json",
        ]
    )

    assert args.train == [Path("train_a.jsonl"), Path("train_b.jsonl")]
    assert args.calibration == [Path("cal_a.jsonl"), Path("cal_b.jsonl")]
    assert args.locked_holdout == [Path("hold_a.jsonl"), Path("hold_b.jsonl")]
    assert args.action_score_mode == LEGACY_ACTION_SCORE_MODE
    assert args.cross_fit_folds == 1


def test_multi_shard_training_records_deterministic_ordered_provenance(tmp_path):
    split_rows = {
        "train_a": [_row(70_000, deal_offset=0), _row(70_001, deal_offset=3)],
        "train_b": [_row(71_000, deal_offset=6), _row(71_001, deal_offset=9)],
        "cal_a": [_row(72_000, deal_offset=12), _row(72_001, deal_offset=15)],
        "cal_b": [_row(73_000, deal_offset=18), _row(73_001, deal_offset=21)],
        "hold_a": [_row(74_000, deal_offset=24)],
        "hold_b": [_row(75_000, deal_offset=27)],
    }
    paths = {}
    for name, rows in split_rows.items():
        path = tmp_path / f"{name}.jsonl"
        _write_jsonl(path, rows)
        paths[name] = path

    manifest = train_from_files(
        train_path=[paths["train_a"], paths["train_b"]],
        calibration_path=(paths["cal_a"], paths["cal_b"]),
        locked_holdout_path=[paths["hold_a"], paths["hold_b"]],
        output_model=tmp_path / "multi.pkl",
        manifest_output=tmp_path / "multi.json",
        iterations=2,
        max_leaf_nodes=3,
        thresholds=(0.0, 0.5, 1.0),
        minimum_calibration_fires=1,
        minimum_safety_fit_samples=1,
        minimum_threshold_lock_samples=1,
        seed=71,
    )

    expected = {
        "train": [paths["train_a"], paths["train_b"]],
        "calibration": [paths["cal_a"], paths["cal_b"]],
        "locked_holdout": [paths["hold_a"], paths["hold_b"]],
    }
    expected_rows = {"train": 4, "calibration": 4, "locked_holdout": 2}
    for split, ordered_paths in expected.items():
        metadata = manifest["inputs"][split]
        assert metadata["shard_count"] == 2
        assert metadata["rows"] == expected_rows[split]
        assert metadata["concatenation_order"] == "cli_or_api_argument_order"
        assert metadata["paths"] == [str(path.resolve()) for path in ordered_paths]
        assert [row["index"] for row in metadata["shards"]] == [0, 1]
        assert [row["path"] for row in metadata["shards"]] == metadata["paths"]
        assert [row["rows"] for row in metadata["shards"]] == [
            len(split_rows[path.stem]) for path in ordered_paths
        ]
        assert all(len(row["sha256"]) == 64 for row in metadata["shards"])
        assert len(metadata["sha256"]) == 64
    assert manifest["train_metrics"]["samples"] == 4
    assert manifest["locked_holdout"]["samples"] == 2


def test_multi_shard_training_rejects_any_input_or_output_path_collision(tmp_path):
    shared = tmp_path / "shared.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    common = {
        "calibration_path": calibration,
        "locked_holdout_path": holdout,
        "output_model": tmp_path / "model.pkl",
        "manifest_output": tmp_path / "manifest.json",
    }
    with pytest.raises(ValueError, match="every .* shard must be distinct"):
        train_from_files(train_path=[shared, shared], **common)

    with pytest.raises(ValueError, match="distinct from each other and inputs"):
        train_from_files(
            train_path=shared,
            calibration_path=calibration,
            locked_holdout_path=holdout,
            output_model=shared,
            manifest_output=tmp_path / "manifest.json",
        )


def test_locked_holdout_changes_cannot_change_partition_or_threshold(tmp_path):
    train_rows = [
        _row(50_000 + i, deal_offset=i * 3) for i in range(4)
    ]
    calibration_rows = [_row(51_000 + i, deal_offset=15 + i * 3) for i in range(4)]
    holdout_rows = [_row(52_000 + i, deal_offset=30 + i * 3) for i in range(2)]
    changed_holdout_rows = deepcopy(holdout_rows)
    for row in changed_holdout_rows:
        for action in row["actions"]:
            action["score"] *= -10.0
    train_path = tmp_path / "train.jsonl"
    calibration_path = tmp_path / "calibration.jsonl"
    holdout_path = tmp_path / "locked.jsonl"
    changed_holdout_path = tmp_path / "locked_changed.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(calibration_path, calibration_rows)
    _write_jsonl(holdout_path, holdout_rows)
    _write_jsonl(changed_holdout_path, changed_holdout_rows)

    common = {
        "train_path": train_path,
        "calibration_path": calibration_path,
        "iterations": 2,
        "max_leaf_nodes": 3,
        "thresholds": (0.0, 0.5, 1.0),
        "minimum_calibration_fires": 1,
        "minimum_safety_fit_samples": 1,
        "minimum_threshold_lock_samples": 1,
        "safety_split_seed": 123,
        "seed": 51,
    }
    first = train_from_files(
        **common,
        locked_holdout_path=holdout_path,
        output_model=tmp_path / "first.pkl",
        manifest_output=tmp_path / "first.json",
    )
    second = train_from_files(
        **common,
        locked_holdout_path=changed_holdout_path,
        output_model=tmp_path / "second.pkl",
        manifest_output=tmp_path / "second.json",
    )

    assert first["calibration_partition"] == second["calibration_partition"]
    assert first["calibration"] == second["calibration"]
    assert first["locked_holdout"] != second["locked_holdout"]
    assert first["locked_holdout_used_for_threshold_or_training"] is False


def test_too_small_calibration_is_serialized_as_fail_closed_no_go(tmp_path):
    train_rows = [_row(60_000 + i, deal_offset=i * 3) for i in range(4)]
    calibration_rows = [_row(61_000 + i, deal_offset=15 + i * 3) for i in range(2)]
    holdout_rows = [_row(62_000 + i, deal_offset=24 + i * 3) for i in range(2)]
    train_path = tmp_path / "train.jsonl"
    calibration_path = tmp_path / "calibration.jsonl"
    holdout_path = tmp_path / "locked.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(calibration_path, calibration_rows)
    _write_jsonl(holdout_path, holdout_rows)

    manifest = train_from_files(
        train_path=train_path,
        calibration_path=calibration_path,
        locked_holdout_path=holdout_path,
        output_model=tmp_path / "closed.pkl",
        manifest_output=tmp_path / "closed.json",
        iterations=2,
        max_leaf_nodes=3,
        thresholds=(0.0, 0.5, 1.0),
        minimum_safety_fit_samples=2,
        minimum_threshold_lock_samples=2,
        seed=61,
    )
    loaded = HuM4JointActionModel.load(tmp_path / "closed.pkl")

    assert manifest["calibration_partition"]["status"] == "no_go_insufficient_samples"
    assert manifest["calibration"]["status"] == "no_go"
    assert manifest["calibration"]["safety_estimator_source"] == "not_fit_fail_closed"
    assert manifest["promotion_status"] == "no_go_calibration"
    assert loaded.safety_enabled is False


def test_teacher_contract_requires_policy_observation_scores_se_and_baseline():
    row = _row(40_000)
    missing_observation = deepcopy(row)
    missing_observation.pop("policy_observation")
    with pytest.raises(ValueError, match="requires policy_observation"):
        prepare_teacher_sample(missing_observation)

    missing_se = deepcopy(row)
    missing_se["actions"][0].pop("score_se")
    with pytest.raises(ValueError, match="score and score_se"):
        prepare_teacher_sample(missing_se)

    bad_baseline = deepcopy(row)
    bad_baseline["baseline_action_row_index"] = len(row["actions"])
    with pytest.raises(IndexError, match="outside actions"):
        prepare_teacher_sample(bad_baseline)


def test_legacy_pickle_without_m42_fields_loads_with_identical_predictions(tmp_path):
    samples = _prepared(80_000, 4)
    legacy = fit_joint_model(
        samples, iterations=2, max_leaf_nodes=3, seed=801
    )
    expected = legacy.predict_sample(samples[0].policy_sample)
    # Reproduce the instance state stored by a pre-M4.2 pickle.
    legacy.__dict__.pop("action_score_mode", None)
    legacy.__dict__.pop("meta_rank_estimator", None)
    legacy.__dict__.pop("paired_fold_estimators", None)
    legacy.__dict__.pop("positive_gain_score_weight", None)
    legacy.__dict__.pop("downside_risk_score_weight", None)
    legacy.__dict__.pop("ensemble_disagreement_score_weight", None)
    path = tmp_path / "legacy.pkl"
    legacy.save(path)

    loaded = HuM4JointActionModel.load(path)
    assert loaded.action_score_mode == LEGACY_ACTION_SCORE_MODE
    assert loaded.meta_rank_estimator is None
    np.testing.assert_allclose(loaded.predict_sample(samples[0].policy_sample), expected)


def test_cross_fit_folds_are_row_order_invariant_leak_free_and_exact_once():
    samples = [
        replace(
            sample,
            teacher_delta_se_vs_baseline=np.full(
                sample.teacher_scores.shape, 0.30, dtype=np.float64
            ),
        )
        for sample in _prepared(81_000, 6)
    ]
    kwargs = {
        "cross_fit_folds": 3,
        "action_score_mode": NEGATIVE_REGRET_ACTION_SCORE_MODE,
        "iterations": 2,
        "max_leaf_nodes": 3,
        "seed": 811,
    }
    first = fit_joint_model_cross_fitted(samples, **kwargs)
    second = fit_joint_model_cross_fitted(list(reversed(samples)), **kwargs)

    assert first.report["fold_assignment"] == second.report["fold_assignment"]
    assert first.report["predictor_lineage"] == second.report["predictor_lineage"]
    assert first.report["fold_assignment"]["identity_leakage_count"] == 0
    assert all(
        not any(fold["identity_overlap"].values())
        for fold in first.report["fold_assignment"]["fold_audits"]
    )
    assert first.report["oof_coverage"] == {
        "samples": 6,
        "base_head_prediction_counts": [1],
        "meta_rank_prediction_counts": [1],
        "uncertainty_prediction_counts": [1],
        "each_sample_predicted_exactly_once": True,
    }
    assert first.report["uncertainty_target"] == {
        "residual_source": "out_of_fold_action_score_residual",
        "score_se_floor": "1.96_times_action_score_se",
        "paired_delta_se_floor": "1.96_times_paired_delta_se_when_present",
        "paired_delta_schema_v2_samples": 6,
        "teacher_lcb_runtime_gate": False,
    }
    assert first.report["fit_roles"]["train_outer_validation"] == (
        "oof_safety_feature_source_only"
    )
    assert first.report["fit_roles"]["calibration_threshold_lock"] == (
        "not_used_in_cross_fit_stage"
    )
    assert first.report["fit_roles"]["locked_holdout"] == (
        "not_used_in_cross_fit_stage"
    )
    lineage = first.report["predictor_lineage"]
    assert lineage["nested_cross_fit"] is True
    assert lineage["identity_leakage_count"] == 0
    assert lineage["all_outer_validation_identities_excluded"] is True
    assert len(lineage["outer_fold_audits"]) == 3
    for outer in lineage["outer_fold_audits"]:
        assert outer["identity_leakage_count"] == 0
        assert not any(
            outer["outer_train__validation_identity_overlap"].values()
        )
        assert outer["inner_base_fold_assignment"]["identity_leakage_count"] == 0
        strict = outer["strict_residual_prediction_lineage"]
        assert strict["identity_leakage_count"] == 0
        assert strict["prediction_counts"] == [1]
        assert strict["each_sample_predicted_exactly_once"] is True
        for nested_fold in strict["fold_audits"]:
            assert not any(
                nested_fold["target__predictor_lineage_overlap"].values()
            )
            assert not any(nested_fold["base_fit__meta_fit_overlap"].values())
            assert nested_fold["target_identity_used_by_predictor_lineage"] is False
    assert first.model.action_score_mode == NEGATIVE_REGRET_ACTION_SCORE_MODE
    assert first.model.meta_rank_estimator is not None
    prediction = first.model.predict_sample(samples[0].policy_sample)
    assert prediction.shape == samples[0].teacher_scores.shape
    assert np.isfinite(prediction).all()


def test_nested_cross_fit_fails_early_when_outer_training_has_too_few_groups():
    with pytest.raises(
        ValueError,
        match=(
            "requires at least 3 identity groups in every outer training subset"
        ),
    ):
        fit_joint_model_cross_fitted(
            _prepared(81_500, 3),
            cross_fit_folds=2,
            action_score_mode=NEGATIVE_REGRET_ACTION_SCORE_MODE,
            iterations=2,
            max_leaf_nodes=3,
            seed=815,
        )


def test_negative_regret_meta_features_are_state_offset_invariant():
    policy = np.asarray([0.2, 0.8, 0.4])
    value = np.asarray([10.0, 13.0, 11.0])
    delta = np.asarray([-2.0, 1.0, 0.0])
    legacy = value + 0.5 * delta + 0.25 * (policy - 0.5)
    first = build_meta_rank_features(
        policy_probability=policy,
        value=value,
        delta_vs_baseline=delta,
        legacy_score=legacy,
    )
    shifted = build_meta_rank_features(
        policy_probability=policy,
        value=value + 1_000.0,
        delta_vs_baseline=delta + 77.0,
        legacy_score=legacy - 500.0,
    )
    np.testing.assert_allclose(first, shifted, atol=1e-5)


def test_cross_fit_threshold_and_partition_do_not_depend_on_locked_holdout(tmp_path):
    train_rows = [_row(82_000 + i, deal_offset=i * 3) for i in range(6)]
    calibration_rows = [_row(83_000 + i, deal_offset=21 + i * 3) for i in range(4)]
    holdout_rows = [
        _row(84_000, deal_offset=18),
        _row(84_001, deal_offset=33),
    ]
    changed_holdout_rows = deepcopy(holdout_rows)
    for row in changed_holdout_rows:
        for action in row["actions"]:
            action["score"] = -100.0 * float(action["score"])
    paths = {}
    for name, rows in {
        "train": train_rows,
        "calibration": calibration_rows,
        "holdout": holdout_rows,
        "changed": changed_holdout_rows,
    }.items():
        paths[name] = tmp_path / f"{name}.jsonl"
        _write_jsonl(paths[name], rows)
    common = {
        "train_path": paths["train"],
        "calibration_path": paths["calibration"],
        "action_score_mode": NEGATIVE_REGRET_ACTION_SCORE_MODE,
        "cross_fit_folds": 3,
        "iterations": 2,
        "max_leaf_nodes": 3,
        "thresholds": (0.0, 0.5, 1.0),
        "minimum_calibration_fires": 1,
        "minimum_safety_fit_samples": 1,
        "minimum_threshold_lock_samples": 1,
        "seed": 821,
    }
    first = train_from_files(
        **common,
        locked_holdout_path=paths["holdout"],
        output_model=tmp_path / "first-cross.pkl",
        manifest_output=tmp_path / "first-cross.json",
    )
    second = train_from_files(
        **common,
        locked_holdout_path=paths["changed"],
        output_model=tmp_path / "second-cross.pkl",
        manifest_output=tmp_path / "second-cross.json",
    )

    assert first["cross_fit"] == second["cross_fit"]
    assert first["calibration_partition"] == second["calibration_partition"]
    assert first["calibration"] == second["calibration"]
    assert first["locked_holdout"] != second["locked_holdout"]
    assert first["cross_fit"]["split_role_overlap"][
        "train_oof__locked_holdout"
    ] == {
        "seed_value_count": 0,
        "observation_fingerprint_count": 0,
        "row_hash_count": 0,
    }


def test_m43_paired_fold_ensemble_is_baseline_zero_and_action_order_invariant(
    tmp_path,
):
    samples = _paired_prepared(91_000, 6)
    result = fit_joint_model_cross_fitted(
        samples,
        cross_fit_folds=3,
        action_score_mode=PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
        iterations=2,
        max_leaf_nodes=3,
        seed=911,
    )
    model = result.model
    assert model.action_score_mode == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
    assert len(model.paired_fold_estimators) == 3
    assert result.report["runtime_ensemble"] == {
        "source": "stored_crossfit_fold_estimators",
        "fold_count": 3,
        "full_refit_used_at_runtime": False,
        "oof_to_full_refit_distribution_shift": False,
        "oof_safety_aggregation_semantics_match": True,
        "oof_safety_fold_count": 3,
        "baseline_action_score_exact_zero": True,
    }
    assert result.report["oof_coverage"]["paired_head_prediction_counts"] == [1]
    assert all(
        audit["oof_safety_inner_ensemble"]["fold_count"] == 3
        and audit["oof_safety_inner_ensemble"][
            "matches_runtime_fold_count"
        ]
        for audit in result.report["predictor_lineage"]["fold_audits"]
    )
    assert result.report["objectives"]["paired_delta_mean"]["loss"] == "huber"
    assert result.report["objectives"]["downside_tail"] == {
        "downside_loss_p95": "max(0,-paired_delta_p05)",
        "downside_loss_p99": "max(0,-paired_delta_p01)",
        "downside_loss_max": "max(0,-paired_delta_min)",
        "loss": "quantile",
        "quantile": 0.9,
    }

    sample = deepcopy(samples[0].policy_sample)
    baseline = samples[0].baseline_index
    scores = model.predict_sample_with_baseline(sample, baseline_index=baseline)
    assert scores[baseline] == 0.0
    expected = {
        action_key_from_payload(action).to_token(): float(scores[index])
        for index, action in enumerate(sample["actions"])
    }

    reordered = deepcopy(sample)
    reordered["actions"] = list(reversed(reordered["actions"]))
    baseline_key = action_key_from_payload(sample["actions"][baseline]).to_token()
    reordered_baseline = next(
        index
        for index, action in enumerate(reordered["actions"])
        if action_key_from_payload(action).to_token() == baseline_key
    )
    reordered["baseline_action_row_index"] = reordered_baseline
    reordered_scores = model.predict_sample_with_baseline(
        reordered, baseline_index=reordered_baseline
    )
    assert reordered_scores[reordered_baseline] == 0.0
    observed = {
        action_key_from_payload(action).to_token(): float(reordered_scores[index])
        for index, action in enumerate(reordered["actions"])
    }
    assert observed.keys() == expected.keys()
    for key in expected:
        assert observed[key] == pytest.approx(expected[key], abs=1.0e-12)

    paired = build_paired_action_features(sample, baseline_index=baseline)
    assert paired.shape[0] == len(sample["actions"])
    assert np.count_nonzero(paired[baseline, 2 * (paired.shape[1] // 4) :]) == 0
    matrix, _ = sample_to_matrix(sample)
    with pytest.raises(ValueError, match="explicit baseline"):
        model.predict_matrix(matrix)

    artifact = tmp_path / "m43-fold-ensemble.pkl"
    model.save(artifact)
    loaded = HuM4JointActionModel.load(artifact)
    np.testing.assert_allclose(
        loaded.predict_sample_with_baseline(sample, baseline_index=baseline),
        scores,
        rtol=0.0,
        atol=0.0,
    )


def test_m43_centers_each_folds_full_composite_not_only_delta():
    sample = _paired_prepared(91_500, 1)[0]
    legacy = fit_joint_model(
        _prepared(91_600, 2), iterations=1, max_leaf_nodes=2, seed=915
    )
    paired_dim = build_paired_action_features(
        sample.policy_sample, baseline_index=sample.baseline_index
    ).shape[1]
    fold = PairedDeltaRiskFoldEstimator(
        delta_estimator=_ConstantRegression(0.0),
        positive_gain_estimator=_ConstantRegression(0.8),
        downside_p95_estimator=_ConstantRegression(0.0),
        downside_p99_estimator=_ConstantRegression(0.0),
        downside_max_estimator=_ConstantRegression(0.0),
        paired_feature_dim=paired_dim,
        fold_index=0,
    )
    biased = replace(
        legacy,
        action_score_mode=PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
        paired_fold_estimators=(fold,),
    )
    scores = biased.predict_sample_with_baseline(
        sample.policy_sample, baseline_index=sample.baseline_index
    )

    # A constant 0.8 positive head is state-wide bias, not evidence that any
    # candidate beats the baseline. Full-composite centering removes it.
    np.testing.assert_array_equal(scores, np.zeros_like(scores))


def test_m43_threshold_metrics_use_paired_future_tails_and_max_bound():
    rows = [
        {
            "teacher_delta": 3.0,
            "downside_loss_p95": 9.0,
            "downside_loss_p99": 19.0,
            "downside_loss_max": 49.0,
        },
        {
            "teacher_delta": 2.0,
            "downside_loss_p95": 12.0,
            "downside_loss_p99": 22.0,
            "downside_loss_max": 51.0,
        },
    ]
    metrics = _override_metrics(rows, total_states=4, threshold=0.5)
    assert metrics["teacher_mean_delta_per_fire"] == 2.5
    assert metrics["p95_loss"] == 12.0
    assert metrics["p99_loss"] == 22.0
    assert metrics["max_loss"] == 51.0
    assert metrics["loss_metric_source"] == (
        "selected_action_paired_p05_p01_min_maxima"
    )


def test_m43_rejects_teacher_rows_without_paired_mean_se_and_tail_contract():
    with pytest.raises(ValueError, match="requires paired_delta_mean"):
        fit_joint_model_cross_fitted(
            _prepared(92_000, 4),
            cross_fit_folds=2,
            action_score_mode=PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
            iterations=2,
            max_leaf_nodes=3,
            seed=921,
        )


def test_m43_uses_low_capacity_oof_safety_calibrator():
    train = prepare_teacher_samples(
        [_paired_row(93_000 + index, deal_offset=3 * index) for index in range(6)]
    )
    cross_fit = fit_joint_model_cross_fitted(
        train,
        cross_fit_folds=3,
        action_score_mode=PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
        iterations=2,
        max_leaf_nodes=3,
        seed=931,
    )
    examples = cross_fit.safety_examples
    # The calibrator contract is independent of the sampled class mix.  Force
    # both labels while retaining the identity-clean OOF feature provenance.
    features = np.vstack(
        (
            np.zeros(18, dtype=np.float32),
            np.ones(18, dtype=np.float32),
            np.full(18, -0.5, dtype=np.float32),
            np.full(18, 0.5, dtype=np.float32),
        )
    )
    examples = replace(
        examples,
        features=features,
        labels=np.asarray([0, 1, 0, 1], dtype=np.int8),
    )
    calibrated, report = calibrate_safety(
        cross_fit.model,
        prepare_teacher_samples(
            [
                _paired_row(94_000 + index, deal_offset=18 + 3 * index)
                for index in range(3)
            ]
        ),
        threshold_lock_samples=prepare_teacher_samples(
            [
                _paired_row(95_000 + index, deal_offset=27 + 3 * index)
                for index in range(3)
            ]
        ),
        oof_safety_examples=examples,
        thresholds=(0.0, 0.5, 1.0),
        minimum_fires=1,
        iterations=2,
        max_leaf_nodes=3,
        safety_calibrator_c=0.125,
        seed=941,
    )
    assert report["safety_estimator_family"] == (
        "standardized_l2_logistic_low_capacity"
    )
    assert report["safety_calibrator_c"] == 0.125
    assert report["constraints"]["maximum_max_loss"] == 50.0
    assert report["threshold_selection_source"] == "calibration.threshold_lock"
    assert report["safety_calibrator_sources"] == [
        "train_oof",
        "calibration.safety_fit",
    ]
    assert report["runtime_inputs_exclude_teacher_values_and_teacher_lcb"] is True
    assert type(calibrated.safety_estimator).__name__ == "Pipeline"


def test_m43_fold_provider_matches_monolithic_predictions_and_threshold():
    train = _paired_prepared(193_000, 10)
    common = {
        "cross_fit_folds": 5,
        "action_score_mode": PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
        "model_id": "m43-tiny-parity",
        "near_best_margin": 0.5,
        "minimum_safe_teacher_gain": 0.0,
        "iterations": 1,
        "max_leaf_nodes": 3,
        "learning_rate": 0.1,
        "l2_regularization": 1.0,
        "seed": 1931,
        "paired_se_floor": 0.5,
        "paired_huber_alpha": 0.9,
        "downside_quantile": 0.9,
        "positive_gain_score_weight": 0.25,
        "downside_risk_score_weight": 0.5,
        "ensemble_disagreement_score_weight": 0.25,
    }
    monolithic = fit_joint_model_cross_fitted(train, **common)
    consumed = []

    def provider(spec, fit_samples):
        consumed.append(spec.to_manifest())
        return _fit_paired_delta_risk_fold(
            fit_samples,
            fold_index=spec.estimator_fold_index,
            paired_se_floor=common["paired_se_floor"],
            huber_alpha=common["paired_huber_alpha"],
            downside_quantile=common["downside_quantile"],
            iterations=common["iterations"],
            max_leaf_nodes=common["max_leaf_nodes"],
            learning_rate=common["learning_rate"],
            seed=spec.estimator_seed,
        )

    sharded = fit_joint_model_cross_fitted(
        train,
        m43_fold_estimator_provider=provider,
        **common,
    )
    assert len(consumed) == 30
    assert [row["job_index"] for row in consumed] == list(range(30))
    np.testing.assert_array_equal(
        monolithic.safety_examples.features,
        sharded.safety_examples.features,
    )
    np.testing.assert_array_equal(
        monolithic.safety_examples.labels,
        sharded.safety_examples.labels,
    )
    for sample in train:
        np.testing.assert_allclose(
            monolithic.model.predict_sample(sample.policy_sample),
            sharded.model.predict_sample(sample.policy_sample),
            rtol=0.0,
            atol=0.0,
        )

    calibration = _paired_prepared(194_000, 10)
    calibration_kwargs = {
        "threshold_lock_samples": calibration[5:],
        "thresholds": (0.0, 0.5, 1.0),
        "minimum_safe_teacher_gain": 0.0,
        "minimum_fires": 1,
        "maximum_false_positive_rate": 1.0,
        "maximum_p95_loss": 1_000.0,
        "maximum_p99_loss": 1_000.0,
        "maximum_max_loss": 1_000.0,
        "iterations": 1,
        "max_leaf_nodes": 3,
        "learning_rate": 0.1,
        "l2_regularization": 1.0,
        "safety_calibrator_c": 0.25,
        "seed": 1941,
    }
    calibrated_monolithic, monolithic_report = calibrate_safety(
        monolithic.model,
        calibration[:5],
        oof_safety_examples=monolithic.safety_examples,
        **calibration_kwargs,
    )
    calibrated_sharded, sharded_report = calibrate_safety(
        sharded.model,
        calibration[:5],
        oof_safety_examples=sharded.safety_examples,
        **calibration_kwargs,
    )
    assert calibrated_monolithic.safety_threshold == (
        calibrated_sharded.safety_threshold
    )
    assert monolithic_report["selected_threshold"] == sharded_report[
        "selected_threshold"
    ]
    for sample in calibration:
        baseline = sample.baseline_index
        candidate = calibrated_monolithic.choose_action_index(sample.policy_sample)
        np.testing.assert_allclose(
            calibrated_monolithic.predict_safety_probability(
                sample.policy_sample,
                candidate_index=candidate,
                baseline_index=baseline,
            ),
            calibrated_sharded.predict_safety_probability(
                sample.policy_sample,
                candidate_index=candidate,
                baseline_index=baseline,
            ),
            rtol=0.0,
            atol=0.0,
        )


def test_m43_artifact_embedded_manifest_is_independent_of_locked_contract_content(
    tmp_path,
):
    base = {
        "schema": "hu_m4_joint_training_manifest_v1",
        "inputs": {
            "train": {"sha256": "train"},
            "calibration": {"sha256": "calibration"},
            "locked_holdout": {"sha256": "locked-a"},
        },
        "m43_data_contract": {
            "contract_sha256": "contract-a",
            "teacher_shards_all_splits_sha256": "locked-derived-a",
        },
        "calibration_partition": {
            "partition_method": "sealed_explicit_roles",
            "safety_fit": {"identity_digest": "safety"},
            "threshold_lock": {"identity_digest": "lock"},
            "overlap": {"identity": 0},
            "locked_holdout_sha256": "locked-derived-a",
        },
        "locked_holdout": {
            "status": "evaluated",
            "teacher_ev": 999.0,
        },
        "distributed_fold_assembly": {
            "schema": "hu_m43_fold_assembly_v1",
            "status": "pass",
            "job_count": 1,
            "run_manifest_sha256": "locked-derived-run-a",
            "cloud_contract_sha256": "locked-derived-cloud-a",
            "jobs": [
                {
                    "job_index": 0,
                    "artifact_sha256": "train-derived-estimator",
                    "job_manifest_sha256": "locked-derived-job-manifest-a",
                    "job_spec_sha256": "train-derived-job-spec",
                }
            ],
            "exact_outer_inner_coverage": True,
            "current_profile_mutated": False,
            "no_runtime_activation": True,
        },
    }
    changed_locked = deepcopy(base)
    changed_locked["inputs"]["locked_holdout"]["sha256"] = "locked-b"
    changed_locked["m43_data_contract"] = {
        "contract_sha256": "contract-b",
        "teacher_shards_all_splits_sha256": "locked-derived-b",
    }
    changed_locked["calibration_partition"]["locked_holdout_sha256"] = (
        "locked-derived-b"
    )
    changed_locked["locked_holdout"] = {
        "status": "evaluated",
        "teacher_ev": -999.0,
    }
    changed_locked["distributed_fold_assembly"]["run_manifest_sha256"] = (
        "locked-derived-run-b"
    )
    changed_locked["distributed_fold_assembly"]["cloud_contract_sha256"] = (
        "locked-derived-cloud-b"
    )
    changed_locked["distributed_fold_assembly"]["jobs"][0][
        "job_manifest_sha256"
    ] = "locked-derived-job-manifest-b"

    first = _artifact_embedded_manifest(base, is_m43=True)
    second = _artifact_embedded_manifest(changed_locked, is_m43=True)
    assert first == second
    assert "locked_holdout" not in first["inputs"]
    assert first["m43_data_contract"]["embedded"] is False
    assert first["distributed_fold_assembly"] == {
        "schema": "hu_m43_fold_assembly_v1",
        "status": "pass",
        "job_count": 1,
        "jobs": [
            {
                "job_index": 0,
                "artifact_sha256": "train-derived-estimator",
                "job_spec_sha256": "train-derived-job-spec",
            }
        ],
        "exact_outer_inner_coverage": True,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
        "external_spot_hash_chain_required": True,
        "sealed_contract_derived_hashes_embedded": False,
        "job_manifest_hashes_embedded": False,
    }
    assert first["locked_holdout"] == {
        "status": "not_evaluated_pre_freeze",
        "labels_opened_by_trainer": False,
    }
    fold = PairedDeltaRiskFoldEstimator(
        delta_estimator=_ConstantRegression(0.0),
        positive_gain_estimator=_ConstantRegression(0.5),
        downside_p95_estimator=_ConstantRegression(0.0),
        downside_p99_estimator=_ConstantRegression(0.0),
        downside_max_estimator=_ConstantRegression(0.0),
        paired_feature_dim=4 * HU_FEATURE_DIM,
        fold_index=0,
    )
    common_model = {
        "policy_estimator": _ConstantRegression(0.5),
        "value_estimator": _ConstantRegression(0.0),
        "delta_estimator": _ConstantRegression(0.0),
        "uncertainty_estimator": _ConstantRegression(0.0),
        "model_id": "m43-locked-independence",
        "action_score_mode": PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
        "paired_fold_estimators": (fold,),
    }
    model_a = HuM4JointActionModel(manifest=first, **common_model)
    model_b = HuM4JointActionModel(manifest=second, **common_model)
    path_a = tmp_path / "model-a.pkl"
    path_b = tmp_path / "model-b.pkl"
    model_a.save(path_a)
    model_b.save(path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_m43_trainer_rejects_minimal_identity_only_self_hashed_contract(tmp_path):
    contract = {
        "schema": "hu_m43_bounded_pilot_data_contract_v1",
        "status": "pass_fresh_data_sealed_for_model_freeze",
    }
    contract["contract_sha256"] = canonical_manifest_sha256(contract)
    contract_path = tmp_path / "minimal-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="frozen-plan SHA mismatch"):
        _load_m43_role_binding(
            contract_path,
            plan_path=root / "configs" / "hu_joint_policy_m43_pilot.json",
            repo_root=root,
            train_paths=(),
            calibration_paths=(),
            raw={},
            prepared={},
        )


def test_m43_file_training_requires_sealed_profile_stratified_data_contract(tmp_path):
    rows = [
        _paired_row(96_000 + index, deal_offset=3 * index)
        for index in range(12)
    ]
    paths = {
        "train": tmp_path / "train.jsonl",
        "calibration": tmp_path / "calibration.jsonl",
        "locked": tmp_path / "locked.jsonl",
    }
    _write_jsonl(paths["train"], rows[:6])
    _write_jsonl(paths["calibration"], rows[6:10])
    _write_jsonl(paths["locked"], rows[10:])
    with pytest.raises(ValueError, match="requires m43_data_contract_path"):
        train_from_files(
            train_path=paths["train"],
            calibration_path=paths["calibration"],
            locked_holdout_path=paths["locked"],
            output_model=tmp_path / "m43.pkl",
            manifest_output=tmp_path / "m43.json",
            action_score_mode=PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
            cross_fit_folds=3,
            iterations=2,
            max_leaf_nodes=3,
            seed=961,
        )


def test_m43_trainer_rejects_locked_holdout_path_before_opening_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="must not receive locked_holdout_path"):
        train_from_files(
            train_path=tmp_path / "does-not-exist-train.jsonl",
            calibration_path=tmp_path / "does-not-exist-calibration.jsonl",
            locked_holdout_path=tmp_path / "must-never-open.jsonl",
            output_model=tmp_path / "m43.pkl",
            manifest_output=tmp_path / "m43.json",
            m43_data_contract_path=tmp_path / "does-not-exist-contract.json",
            m43_plan_path=root / "configs" / "hu_joint_policy_m43_pilot.json",
            m43_repo_root=root,
            action_score_mode=PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
            cross_fit_folds=3,
            iterations=2,
            max_leaf_nodes=3,
            seed=962,
        )
