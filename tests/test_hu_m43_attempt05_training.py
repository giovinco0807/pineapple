from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt05_model import Attempt05FoldOutput
from ofc_regular.hu_m43_attempt05_training import (
    ATTEMPT05_PROFILES,
    assign_identity_group_folds,
    evaluate_oof_predictions,
    fit_deepsets_fold,
    fit_lambda_rank_fold,
    select_architecture_winner,
)
from ofc_regular.hu_turn3_model import hu_policy_sample
from ofc_regular.state import Board
from ofc_regular.train_hu_m4_joint_model import PreparedTeacherSample


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepared(offset: int, profile_index: int) -> PreparedTeacherSample:
    hero = Board.from_rows(top=["Ah"], middle=["Kd"], bottom=["2s", "3s", "4s"])
    opponent = Board.from_rows(
        top=["Qh"], middle=["Jd", "Td"], bottom=["5s", "6s", "7s", "8s"]
    )
    used = {*hero.all_cards(), *opponent.all_cards()}
    remaining = [card for card in ALL_CARDS if card not in used]
    start = offset % (len(remaining) - 2)
    observation = ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=tuple(remaining[start : start + 3]),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )
    actions = generate_turn_actions(hero, observation.dealt_cards)
    policy_sample = hu_policy_sample(
        hero,
        observation.dealt_cards,
        actions,
        opponent_board=opponent,
        dead_cards=observation.legacy_dead_cards(),
        seat="second",
        to_act_order="second",
    )
    policy_sample["policy_observation"] = observation.to_dict()
    baseline = len(actions) // 2
    policy_sample["baseline_action_row_index"] = baseline
    count = len(actions)
    delta = np.linspace(-2.0, 3.0, count, dtype=np.float64)
    delta[baseline] = 0.0
    p95 = np.linspace(4.0, 12.0, count)
    p99 = p95 + 6.0
    maximum = p99 + 8.0
    return PreparedTeacherSample(
        policy_sample=policy_sample,
        teacher_scores=delta.copy(),
        teacher_score_se=np.ones(count),
        teacher_delta_se_vs_baseline=np.ones(count),
        teacher_paired_delta_mean=delta,
        downside_loss_p95=p95,
        downside_loss_p99=p99,
        downside_loss_max=maximum,
        baseline_index=baseline,
        seat="second",
        root_seed_values=frozenset({str(offset)}),
        observation_fingerprint=observation.fingerprint(),
        ignored_truth_keys=(),
    )


def _dataset(per_profile: int = 5):
    samples = []
    profiles = []
    for profile_index, profile in enumerate(ATTEMPT05_PROFILES):
        for local in range(per_profile):
            samples.append(_prepared(profile_index * per_profile + local, profile_index))
            profiles.append(profile)
    return samples, profiles


def test_attempt05_identity_folds_are_deterministic_balanced_and_reject_duplicates():
    samples, profiles = _dataset()
    first = assign_identity_group_folds(samples, profiles)
    second = assign_identity_group_folds(samples, profiles)
    np.testing.assert_array_equal(first, second)
    assert set(first) == set(range(5))
    for profile in ATTEMPT05_PROFILES:
        assert sorted(
            first[index] for index, value in enumerate(profiles) if value == profile
        ) == list(range(5))
    duplicated = [*samples[:-1], replace(samples[-1], observation_fingerprint=samples[0].observation_fingerprint)]
    with pytest.raises(ValueError, match="duplicate identities"):
        assign_identity_group_folds(duplicated, profiles)


def test_attempt05_fit_rejects_incomplete_or_poisoned_runtime_features():
    samples, profiles = _dataset(per_profile=1)
    incomplete_policy = deepcopy(samples[0].policy_sample)
    incomplete_policy["actions"].pop()
    incomplete = [replace(samples[0], policy_sample=incomplete_policy), *samples[1:]]
    with pytest.raises(ValueError, match="complete legal action set"):
        fit_lambda_rank_fold(incomplete, profiles, fold_index=0, seed=7, smoke=True)

    poisoned_policy = deepcopy(samples[0].policy_sample)
    poisoned_policy["dead_cards"] = ["As"]
    poisoned = [replace(samples[0], policy_sample=poisoned_policy), *samples[1:]]
    with pytest.raises(ValueError, match="dead_cards disagrees"):
        fit_lambda_rank_fold(poisoned, profiles, fold_index=0, seed=7, smoke=True)


def test_attempt05_lambda_rank_small_fit_and_oof_metrics_smoke():
    samples, profiles = _dataset(per_profile=2)
    predictor = fit_lambda_rank_fold(
        samples, profiles, fold_index=0, seed=7, smoke=True
    )
    predictions = tuple(
        predictor.predict(sample.policy_sample, baseline_index=sample.baseline_index)
        for sample in samples
    )
    metrics, cushions = evaluate_oof_predictions(samples, profiles, predictions)
    assert metrics["states"] == len(samples)
    assert len(cushions) == 3
    assert metrics["teacher_values_reported_as_realized_ev"] is False
    assert set(metrics["profile"]) == set(ATTEMPT05_PROFILES)


def test_attempt05_deepsets_small_cpu_fit_is_paired_and_finite():
    pytest.importorskip("torch")
    samples, profiles = _dataset(per_profile=1)
    predictor, report = fit_deepsets_fold(
        samples,
        profiles,
        fold_index=0,
        seed=11,
        epochs=1,
        device="cpu",
        batch_states=5,
    )
    output = predictor.predict(
        samples[0].policy_sample, baseline_index=samples[0].baseline_index
    )
    assert output.rank_score.shape == (len(samples[0].policy_sample["actions"]),)
    assert np.isfinite(output.rank_score).all()
    assert report["tail_pinball_quantile"] == pytest.approx(0.8)
    other_baseline = (samples[0].baseline_index + 1) % len(
        samples[0].policy_sample["actions"]
    )
    changed = predictor.predict(
        samples[0].policy_sample, baseline_index=other_baseline
    )
    assert not np.array_equal(output.rank_score, changed.rank_score)


def test_attempt05_oof_conformal_is_upper_residual_and_winner_rule_is_fixed():
    samples, profiles = _dataset(per_profile=1)
    predictions = []
    for sample in samples:
        count = len(sample.policy_sample["actions"])
        rank = np.arange(count, dtype=np.float64)
        predictions.append(
            Attempt05FoldOutput(
                rank_score=rank,
                gain_probability=np.full(count, 0.75),
                downside_p95=np.asarray(sample.downside_loss_p95) - 2.0,
                downside_p99=np.asarray(sample.downside_loss_p99) - 3.0,
                downside_max=np.asarray(sample.downside_loss_max) - 4.0,
            )
        )
    metrics, cushions = evaluate_oof_predictions(samples, profiles, predictions)
    assert cushions == pytest.approx((2.0, 3.0, 4.0))
    assert metrics["conformal"]["source"] == "strict_oof_candidate_upper_residuals_only"

    family_reports = {
        "lambda_rank": {
            **metrics,
            "raw_gate_pass": True,
            "raw_selected_positive_rate": 0.5,
            "raw_selected_mean_delta": 0.2,
            "mean_regret_to_best_nonbaseline": 1.0,
        },
        "deepsets": {
            **metrics,
            "raw_gate_pass": True,
            "raw_selected_positive_rate": 0.6,
            "raw_selected_mean_delta": 0.3,
            "mean_regret_to_best_nonbaseline": 0.8,
        },
    }
    assert select_architecture_winner(family_reports) == "deepsets"
    family_reports["lambda_rank"]["raw_gate_pass"] = False
    family_reports["deepsets"]["raw_gate_pass"] = False
    assert select_architecture_winner(family_reports) is None


def test_attempt05_proposal_is_historical_and_binds_authoritative_main_plan():
    proposal_path = REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt05_proposal.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    lifecycle = proposal["lifecycle"]
    authority = lifecycle["authoritative_main_plan"]
    main_path = REPO_ROOT / authority["path"]
    main_bytes = main_path.read_bytes()
    main_plan = json.loads(main_bytes.decode("utf-8"))

    assert lifecycle["classification"] == "historical_pre_contract_smoke_snapshot"
    assert lifecycle["authority"] == "non_authoritative_superseded_by_main_plan"
    assert lifecycle["new_development_seed_schedule"] == (
        "bound_in_authoritative_main_plan_development_roles"
    )
    assert authority["schema"] == main_plan["schema"]
    assert authority["status"] == main_plan["status"]
    assert authority["seed_authority"] == "development_roles"
    assert set(main_plan["development_roles"]) == {
        "pilot_train",
        "pilot_audit",
        "expand_train",
        "final_audit",
    }
    assert hashlib.sha256(main_bytes).hexdigest() == authority["sha256_at_reconciliation"]
