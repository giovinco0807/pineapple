from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ofc_regular.action_key import action_key_from_payload
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt03_training import (
    Attempt03OofStatePrediction,
    Attempt03TrainingConfig,
    M43_ATTEMPT03_FIXED_THRESHOLDS,
    audit_attempt03_base_oof_dependencies,
    build_attempt03_oof_safety_dataset,
    claim_attempt03_precalibration,
    evaluate_attempt03_precalibration,
    fit_attempt03_base_oof_runtime_meta,
    fit_attempt03_safety_calibrator,
    load_attempt03_model_freeze,
    load_attempt03_training_science_correction,
    select_attempt03_threshold,
)
from ofc_regular.hu_m43_joint_model_v5 import HuM43JointModelV5
from ofc_regular.hu_m4_joint_model import PairedDeltaRiskFoldEstimator
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM, hu_policy_sample
from ofc_regular.state import Board
from ofc_regular.train_hu_m4_joint_model import (
    PreparedTeacherSample,
    build_m43_fold_training_plan,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs" / "hu_joint_policy_m43_attempt03_model_freeze.json"
SCIENCE_CORRECTION = (
    ROOT
    / "configs"
    / "hu_joint_policy_m43_attempt03_training_science_correction.json"
)


class _ConstantRegression:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, matrix):
        return np.full(np.asarray(matrix).shape[0], self.value, dtype=np.float64)


class _WeightedDifferenceRegression:
    def __init__(self, scale: float) -> None:
        self.scale = float(scale)

    def predict(self, matrix):
        values = np.asarray(matrix, dtype=np.float64)
        difference = values[:, 2 * HU_FEATURE_DIM : 3 * HU_FEATURE_DIM]
        weights = np.linspace(0.25, 1.25, HU_FEATURE_DIM, dtype=np.float64)
        return self.scale * (difference @ weights)


class _FeatureColumnRegression:
    def __init__(self, column: int) -> None:
        self.column = int(column)

    def predict(self, matrix):
        return np.asarray(matrix, dtype=np.float64)[:, self.column]


class _SemanticStage18Scorer:
    def predict_sample(self, sample):
        keys = [action_key_from_payload(action) for action in sample["actions"]]
        ordered = {
            key: rank
            for rank, key in enumerate(sorted(keys, key=lambda value: value.sort_key()))
        }
        return np.asarray(
            [ordered[key] / max(len(keys) - 1, 1) for key in keys],
            dtype=np.float64,
        )


def _fold(index: int, scale: float = 1.0) -> PairedDeltaRiskFoldEstimator:
    return PairedDeltaRiskFoldEstimator(
        delta_estimator=_WeightedDifferenceRegression(scale),
        positive_gain_estimator=_ConstantRegression(0.55),
        downside_p95_estimator=_ConstantRegression(4.0),
        downside_p99_estimator=_ConstantRegression(8.0),
        downside_max_estimator=_ConstantRegression(16.0),
        paired_feature_dim=4 * HU_FEATURE_DIM,
        fold_index=index,
    )


def _folds() -> tuple[PairedDeltaRiskFoldEstimator, ...]:
    return tuple(
        _fold(index, scale)
        for index, scale in enumerate((1.0, 1.0, 1.0, -1.0, -1.0))
    )


def _policy_sample() -> dict:
    hero = Board.from_rows(
        top=["Ah"], middle=["Kd"], bottom=["2s", "3s", "4s"]
    )
    opponent = Board.from_rows(
        top=["Qh"], middle=["Jd", "Td"], bottom=["5s", "6s", "7s", "8s"]
    )
    used = {*hero.all_cards(), *opponent.all_cards()}
    dealt = tuple(card for card in ALL_CARDS if card not in used)[:3]
    observation = ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )
    legal = generate_turn_actions(hero, dealt)
    sample = hu_policy_sample(
        hero,
        dealt,
        legal,
        opponent_board=opponent,
        dead_cards=observation.legacy_dead_cards(),
        seat="second",
        to_act_order="second",
    )
    sample["baseline_action_row_index"] = len(legal) // 2
    return sample


def _prepared(
    index: int,
    *,
    selected: int | None = None,
    selected_delta: float = 1.0,
) -> PreparedTeacherSample:
    policy = _policy_sample()
    count = len(policy["actions"])
    baseline = int(policy["baseline_action_row_index"])
    delta = np.asarray(
        [0.5 if action % 2 == 0 else -0.5 for action in range(count)],
        dtype=np.float64,
    )
    delta[baseline] = 0.0
    if selected is not None:
        delta[selected] = selected_delta
    paired_se = np.full(count, 0.5, dtype=np.float64)
    paired_se[baseline] = 0.0
    p95 = np.full(count, 5.0, dtype=np.float64)
    p99 = np.full(count, 10.0, dtype=np.float64)
    maximum = np.full(count, 20.0, dtype=np.float64)
    p95[baseline] = p99[baseline] = maximum[baseline] = 0.0
    return PreparedTeacherSample(
        policy_sample=policy,
        teacher_scores=delta.copy(),
        teacher_score_se=np.full(count, 0.5, dtype=np.float64),
        teacher_delta_se_vs_baseline=paired_se,
        teacher_paired_delta_mean=delta,
        downside_loss_p95=p95,
        downside_loss_p99=p99,
        downside_loss_max=maximum,
        baseline_index=baseline,
        seat="second",
        root_seed_values=frozenset({str(10_000_000 + index)}),
        observation_fingerprint=f"{index + 1:064x}",
        ignored_truth_keys=(),
    )


def _model() -> HuM43JointModelV5:
    return HuM43JointModelV5(
        paired_fold_estimators=_folds(),
        stage18_scorer=_SemanticStage18Scorer(),
        meta_ranker=_FeatureColumnRegression(0),
        safety_enabled=False,
        safety_threshold=1.0,
        model_id="attempt03-test",
    )


def test_attempt03_freeze_and_config_bind_corrected_grid() -> None:
    freeze = load_attempt03_model_freeze(FREEZE, repo_root=ROOT)
    config = Attempt03TrainingConfig()
    assert freeze["schema"] == "hu_m43_attempt03_model_freeze_v1"
    assert config.thresholds == M43_ATTEMPT03_FIXED_THRESHOLDS
    assert config.to_manifest()["fold_jobs"] == 30
    assert config.to_manifest()["meta_fit_rows"] == "all_nonbaseline_state_balanced"
    assert config.to_manifest()["current_profile_mutated"] is False


def test_attempt03_science_correction_is_pre_fit_and_hash_bound(
    tmp_path: Path,
) -> None:
    correction = load_attempt03_training_science_correction(
        SCIENCE_CORRECTION, repo_root=ROOT
    )
    assert correction["decision_boundary"][
        "attempt03_fit_row_valued_labels_or_metrics_used"
    ] == 0
    assert correction["corrections"]["meta_validation"][
        "internal_meta_oof_performance_claimed"
    ] is False
    assert correction["corrections"]["safety_row_scope"]["fit_rows"] == (
        "hard_eligible_nonbaseline_actions_only"
    )
    tampered = json.loads(SCIENCE_CORRECTION.read_text(encoding="utf-8"))
    tampered["corrections"]["safety_row_scope"]["fit_rows"] = "all_actions"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest changed"):
        load_attempt03_training_science_correction(path, repo_root=ROOT)


def test_attempt03_base_oof_fit_consumes_exact_30_jobs_without_meta_oof_claim() -> None:
    samples = tuple(_prepared(index) for index in range(700))
    consumed: list[int] = []
    meta_fit_calls: list[int] = []

    def provider(spec, _fit_samples):
        consumed.append(spec.job_index)
        return _fold(spec.estimator_fold_index, 1.0)

    def meta_fit(features, targets, states, **_kwargs):
        meta_fit_calls.append(int(features.shape[0]))
        assert features.shape[0] == targets.shape[0] == states.shape[0]
        for state in np.unique(states):
            assert np.sum(states == state) == len(_policy_sample()["actions"]) - 1
        return SimpleNamespace(
            estimator=_FeatureColumnRegression(7),
            manifest={"status": "synthetic_fixed_meta"},
        )

    result = fit_attempt03_base_oof_runtime_meta(
        samples,
        stage18_scorer=_SemanticStage18Scorer(),
        fold_estimator_provider=provider,
        meta_fit=meta_fit,
    )
    assert consumed == list(range(30))
    assert len(meta_fit_calls) == 1
    assert result.report["states"] == 700
    assert result.report["base_fold_jobs"] == 30
    assert result.report["meta_crossfit_performed"] is False
    assert result.report["meta_oof_coverage_exactly_once"] is False
    assert result.report["meta_oof_performance_claimed"] is False
    assert result.report["base_oof_dependency_audit"][
        "own_identity_excluded_from_all_five_base_heads"
    ] is True
    assert result.safety_dataset.manifest["states"] == 700
    assert result.safety_dataset.manifest["baseline_rows"] == 0
    assert result.model.safety_enabled is False
    assert result.report["precalibration_opened"] is False


def test_attempt03_base_oof_dependency_audit_rejects_actual_fit_contamination() -> None:
    samples = tuple(_prepared(index) for index in range(700))
    plan = build_m43_fold_training_plan(
        samples, cross_fit_folds=5, seed=2026072401
    )
    clean = audit_attempt03_base_oof_dependencies(plan)
    assert clean["status"] == "pass"

    held_out = next(
        sample
        for index, sample in enumerate(plan.ordered_samples)
        if plan.outer_fold_ids[index] == 0
    )
    contaminated_job = replace(
        plan.jobs[1], fit_samples=(*plan.jobs[1].fit_samples, held_out)
    )
    contaminated_plan = replace(
        plan, jobs=(plan.jobs[0], contaminated_job, *plan.jobs[2:])
    )
    with pytest.raises(ValueError, match="dependency leak/tamper"):
        audit_attempt03_base_oof_dependencies(contaminated_plan)


def test_attempt03_precal_gate_uses_student_t_lcb_and_all_exact_gates() -> None:
    model = _model()
    probe = _prepared(0)
    proposal = model.predict_heads_sample(
        probe.policy_sample, baseline_index=probe.baseline_index
    ).proposal_index
    samples = tuple(
        _prepared(index, selected=proposal, selected_delta=1.0)
        for index in range(200)
    )
    report = evaluate_attempt03_precalibration(
        model,
        samples,
        fit_seed_values={"fit-only"},
        fit_observation_fingerprints={"f" * 64},
        profile_counts={
            "stage19_p0": 40,
            "stage9f_p2": 40,
            "stage7_m5_r10": 40,
            "stage3_baseline": 40,
            "random_exact_final": 40,
        },
    )
    assert report["status"] == "go"
    assert report["eligible_fires"] == 200
    assert report["raw_proposal_positive_rate"] == 1.0
    assert report["mean_delta_lcb"]["method"] == (
        "one_sided_student_t_over_independent_state_clusters"
    )
    assert report["mean_delta_lcb"]["confidence"] == 0.9
    assert report["mean_delta_lcb"]["lower_bound"] == pytest.approx(1.0)
    assert all(report["gates"].values())
    assert report["sealed_calibration_opened"] is False


def test_attempt03_precal_gate_fails_closed_on_negative_bound() -> None:
    model = _model()
    probe = _prepared(0)
    proposal = model.predict_heads_sample(
        probe.policy_sample, baseline_index=probe.baseline_index
    ).proposal_index
    samples = tuple(
        _prepared(index, selected=proposal, selected_delta=-0.1)
        for index in range(200)
    )
    report = evaluate_attempt03_precalibration(
        model,
        samples,
        fit_seed_values=set(),
        fit_observation_fingerprints=set(),
    )
    assert report["status"] == "no_go"
    assert report["gates"]["minimum_raw_proposal_positive_rate"] is False
    assert report["gates"]["one_sided_student_t_90_lcb_strictly_positive"] is False
    assert report["algorithm_or_gate_change_after_result_allowed"] is False


def test_attempt03_precal_missing_profile_counts_fails_closed() -> None:
    model = _model()
    probe = _prepared(0)
    proposal = model.predict_heads_sample(
        probe.policy_sample, baseline_index=probe.baseline_index
    ).proposal_index
    samples = tuple(
        _prepared(index, selected=proposal, selected_delta=1.0)
        for index in range(200)
    )
    report = evaluate_attempt03_precalibration(
        model,
        samples,
        fit_seed_values={"fit-only"},
        fit_observation_fingerprints={"f" * 64},
        profile_counts=None,
    )
    assert report["status"] == "no_go"
    assert report["gates"]["exact_profile_balance"] is False


def test_attempt03_safety_is_fixed_l2_logistic_then_fixed_grid() -> None:
    model = _model()
    base_samples = tuple(_prepared(index) for index in range(50))
    oof = tuple(
        Attempt03OofStatePrediction(
            sample_index=index,
            fold_index=index % 5,
            predictions=model.predict_heads_sample(
                sample.policy_sample, baseline_index=sample.baseline_index
            ),
            base_identity_excluded_from_fit=True,
            meta_identity_excluded_from_fit=True,
        )
        for index, sample in enumerate(base_samples)
    )
    dataset = build_attempt03_oof_safety_dataset(base_samples, oof)
    probe = _prepared(1000)
    proposal = model.predict_heads_sample(
        probe.policy_sample, baseline_index=probe.baseline_index
    ).proposal_index
    safety_samples = tuple(
        _prepared(1000 + index) for index in range(50)
    )
    safety = fit_attempt03_safety_calibrator(model, dataset, safety_samples)
    assert safety.report["estimator_family"] == "standardized_l2_logistic"
    assert safety.report["c"] == 0.25
    assert safety.model.safety_enabled is False
    independent_oof = tuple(
        Attempt03OofStatePrediction(
            sample_index=index,
            fold_index=-1,
            predictions=model.predict_heads_sample(
                sample.policy_sample, baseline_index=sample.baseline_index
            ),
            base_identity_excluded_from_fit=True,
            meta_identity_excluded_from_fit=True,
        )
        for index, sample in enumerate(safety_samples)
    )
    independent = build_attempt03_oof_safety_dataset(
        safety_samples, independent_oof
    )
    combined_features = np.vstack((dataset.features, independent.features))
    combined_weights = np.concatenate((dataset.weights, independent.weights))
    expected_scaler_mean = np.average(
        combined_features, axis=0, weights=combined_weights
    )
    scaler = safety.model.safety_estimator.named_steps["standardscaler"]
    assert np.allclose(scaler.mean_, expected_scaler_mean)
    assert safety.report["standardizer_uses_state_balanced_weights"] is True

    threshold_samples = tuple(
        _prepared(2000 + index, selected=proposal, selected_delta=1.0)
        for index in range(50)
    )
    threshold = select_attempt03_threshold(safety, threshold_samples)
    assert threshold.report["status"] == "go"
    assert threshold.report["thresholds"] == list(M43_ATTEMPT03_FIXED_THRESHOLDS)
    assert threshold.report["ineligible_top_proposal_rerank_allowed"] is False
    assert threshold.model.safety_enabled is True


def test_attempt03_safety_rows_are_hard_eligible_only_and_zero_state_weight() -> None:
    model = _model()
    samples = (_prepared(3000), _prepared(3001))
    carriers = []
    for index, sample in enumerate(samples):
        heads = model.predict_heads_sample(
            sample.policy_sample, baseline_index=sample.baseline_index
        )
        mask = np.zeros_like(heads.eligible_mask, dtype=bool)
        if index == 0:
            chosen = next(
                action
                for action in range(mask.size)
                if action != sample.baseline_index
            )
            mask[chosen] = True
        heads = replace(heads, eligible_mask=mask)
        carriers.append(
            Attempt03OofStatePrediction(
                sample_index=index,
                fold_index=index,
                predictions=heads,
                base_identity_excluded_from_fit=True,
                meta_identity_excluded_from_fit=False,
            )
        )
    dataset = build_attempt03_oof_safety_dataset(samples, tuple(carriers))
    assert dataset.features.shape[0] == 1
    assert dataset.rows[0]["hard_eligible"] is True
    assert dataset.rows[0]["meta_score_or_proposal_used_as_safety_feature"] is False
    assert dataset.weights.tolist() == [1.0]
    assert dataset.manifest["contributing_states"] == 1
    assert dataset.manifest["zero_eligible_states"] == 1
    assert dataset.manifest["hard_eligibility_used_as_fit_filter"] is True


def test_attempt03_precal_claim_is_exclusive_and_hash_bound(tmp_path: Path) -> None:
    digest = "a" * 64
    marker = tmp_path / "M43_ATTEMPT03_PRECAL_CONSUMED.json"
    payload = claim_attempt03_precalibration(
        marker,
        precal_identity_sha256=digest,
        candidate_model_sha256="b" * 64,
        fit_manifest_sha256="c" * 64,
        data_contract_sha256="d" * 64,
        model_freeze_file_sha256="e" * 64,
        training_freeze_file_sha256="f" * 64,
    )
    assert payload["status"] == "consumed_before_model_evaluation"
    assert payload["candidate_model_sha256"] == "b" * 64
    with pytest.raises(FileExistsError):
        claim_attempt03_precalibration(
            marker,
            precal_identity_sha256=digest,
            candidate_model_sha256="b" * 64,
            fit_manifest_sha256="c" * 64,
            data_contract_sha256="d" * 64,
            model_freeze_file_sha256="e" * 64,
            training_freeze_file_sha256="f" * 64,
        )


def test_attempt03_cli_data_boundaries_are_disjoint() -> None:
    from ofc_regular.assemble_hu_m43_attempt03_model import parse_args

    fit = parse_args(
        [
            "assemble-fit",
            "--fold-artifacts-dir", "folds",
            "--inherited-train", "old.jsonl",
            "--fresh-train-fit", "fresh.jsonl",
            "--fold-cloud-contract", "fold.json",
            "--model-freeze", "freeze.json",
            "--training-freeze", "training-freeze.json",
            "--repo-root", ".",
            "--output-dir", "out",
            "--run-name", "run",
            "--source-sha256", "a" * 64,
            "--run-manifest-sha256", "b" * 64,
        ]
    )
    assert not hasattr(fit, "sealed_calibration")
    assert not hasattr(fit, "one_shot_receive_receipt")
    authorize = parse_args(
        [
            "authorize-precal-open",
            "--candidate-model", "model.pkl",
            "--fit-bundle", "bundle.pkl",
            "--fit-manifest", "fit.json",
            "--fold-cloud-contract", "fold.json",
            "--model-freeze", "freeze.json",
            "--training-freeze", "training-freeze.json",
            "--repo-root", ".",
            "--output", "auth.json",
        ]
    )
    assert not hasattr(authorize, "data_contract")
    assert not hasattr(authorize, "sealed_calibration")
    assert not hasattr(authorize, "locked_holdout")


def test_attempt03_spot_schedule_emits_exact_30_argv_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ofc_regular import build_hu_m43_attempt03_model_job_schedule as schedule

    contract_path = tmp_path / "fold.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    contract = {
        "contract_sha256": "1" * 64,
        "input_bundle_sha256": "2" * 64,
        "training_config_sha256": "3" * 64,
        "model_freeze": {"file_sha256": "4" * 64},
        "training_freeze": {"file_sha256": "5" * 64},
        "process_environment": {"PYTHONHASHSEED": "0"},
        "fold_plan": {
            "jobs": [
                {
                    "job_index": index,
                    "kind": "outer_runtime" if index % 6 == 0 else "inner_oof_safety",
                    "outer_fold": index // 6,
                    "inner_fold": None if index % 6 == 0 else (index % 6) - 1,
                    "job_spec_sha256": f"{index + 10:064x}",
                }
                for index in range(30)
            ]
        },
    }
    monkeypatch.setattr(
        schedule, "load_attempt03_fold_cloud_contract", lambda _path: contract
    )
    result = schedule.build_attempt03_model_job_schedule(
        fold_cloud_contract_path=contract_path,
        run_name="attempt03-model-test",
        source_sha256="a" * 64,
        run_manifest_sha256="b" * 64,
        inherited_train_worker_path="/work/inherited.jsonl",
        fresh_train_fit_worker_path="/work/fresh.jsonl",
        fold_cloud_contract_worker_path="/work/fold.json",
        output_root_worker_path="/work/results",
    )
    assert result["job_count"] == 30
    assert result["shard_count"] == 8
    assert [len(row["job_indices"]) for row in result["shards"]] == [
        4, 4, 4, 4, 4, 4, 4, 2
    ]
    assert result["fanout_before_canary_done_allowed"] is False
    for index, job in enumerate(result["jobs"]):
        assert job["job_index"] == index
        assert job["argv"][:4] == [
            "python", "-B", "-m", "ofc_regular.train_hu_m43_attempt03_fold_job"
        ]
        assert job["job_spec_sha256"] in job["argv"]
