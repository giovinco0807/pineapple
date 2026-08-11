from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ofc_regular.hu_m43_attempt05_model import Attempt05FoldOutput, HuM43Attempt05Model
from ofc_regular.hu_m43_attempt11_distilled_model import (
    HU_M43_ATTEMPT11_DISTILLED_FEATURE_DIM,
)
from ofc_regular.hu_m43_attempt11_teacher import (
    ATTEMPT11_FROZEN_MODEL_ID,
    ATTEMPT11_FROZEN_MODEL_SHA256,
    FrozenAttempt11LambdaRanker,
)
from ofc_regular.train_hu_m43_attempt11_distilled import (
    ATTEMPT11_DISTILLATION_CONFIG_SHA256,
    ATTEMPT11_PROFILES,
    Attempt11DistillationState,
    assign_attempt11_identity_folds,
    fit_attempt11_distilled_model,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hu_joint_policy_m43_attempt11_distillation.json"


class _CandidateFold:
    family = "lambda_rank"

    def __init__(self, fold_index: int) -> None:
        self.fold_index = fold_index

    def predict(self, runtime_sample, *, baseline_index):
        del baseline_index
        count = len(runtime_sample["actions"])
        return Attempt05FoldOutput(
            rank_score=np.arange(count, dtype=np.float64),
            gain_probability=np.full(count, 0.5),
            downside_p95=np.full(count, 2.0),
            downside_p99=np.full(count, 4.0),
            downside_max=np.full(count, 6.0),
        )


def _ranker() -> FrozenAttempt11LambdaRanker:
    model = HuM43Attempt05Model(
        family="lambda_rank",
        fold_predictors=tuple(_CandidateFold(index) for index in range(5)),
        runtime_enabled=False,
        winner_frozen=False,
        model_id=ATTEMPT11_FROZEN_MODEL_ID,
    )
    return FrozenAttempt11LambdaRanker(
        model=model, artifact_sha256=ATTEMPT11_FROZEN_MODEL_SHA256
    )


def _variable_states() -> tuple[Attempt11DistillationState, ...]:
    rng = np.random.default_rng(4311)
    group_sizes = [*range(1, 14), 4, 9]
    states: list[Attempt11DistillationState] = []
    for root, group_size in enumerate(group_sizes):
        relevance = np.zeros(group_size, dtype=np.int8)
        selected = 0 if group_size == 1 else root % (group_size - 1)
        relevance[selected] = 4
        delta = np.linspace(-2.0, 3.0, group_size)
        delta[-1] = 0.0
        safe = np.zeros(group_size, dtype=np.int8)
        if group_size > 1:
            safe[selected] = 1
        tails = np.column_stack(
            (
                np.linspace(2.0, 8.0, group_size),
                np.linspace(5.0, 12.0, group_size),
                np.linspace(9.0, 18.0, group_size),
            )
        )
        tails[-1] = 0.0
        states.append(
            Attempt11DistillationState(
                root_index=root,
                root_profile=ATTEMPT11_PROFILES[root % len(ATTEMPT11_PROFILES)],
                observation_fingerprint=f"{root + 1:064x}",
                sample={},
                baseline_index=group_size - 1,
                proposal_indices=tuple(range(group_size)),
                features=rng.normal(
                    size=(group_size, HU_M43_ATTEMPT11_DISTILLED_FEATURE_DIM)
                ).astype(np.float32),
                relevance=relevance,
                delta=delta,
                safe=safe,
                tails=tails,
                selected_action_key=f"selected-{root}",
                teacher_override_fired=group_size > 1,
            )
        )
    return tuple(states)


def test_attempt11_config_freezes_variable_candidates_without_padding() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    candidate = config["candidate_generation"]
    assert candidate["nonbaseline_max"] == 12
    assert candidate["actual_count_range"] == [0, 12]
    assert candidate["padding_allowed"] is False
    assert candidate["duplicate_action_keys_allowed"] is False
    assert candidate["baseline_in_candidate_set"] is False
    assert candidate["baseline_appended_exactly_once"] is True
    assert candidate["lightgbm_group_size_range"] == [1, 13]
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == (
        ATTEMPT11_DISTILLATION_CONFIG_SHA256
    )


def test_attempt11_variable_lightgbm_groups_match_row_total() -> None:
    states = _variable_states()
    folds = assign_attempt11_identity_folds(states)
    np.testing.assert_array_equal(folds, assign_attempt11_identity_folds(states))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model, report = fit_attempt11_distilled_model(
        states,
        candidate_ranker=_ranker(),
        config=config,
        smoke=True,
    )
    expected_groups = [state.group_size for state in states]
    assert len(model.fold_predictors) == 5
    assert model.safety_enabled is False
    assert model.winner_frozen is False
    assert report["group_sizes"] == expected_groups
    assert report["rows"] == sum(expected_groups)
    assert report["candidate_count_min"] == 0
    assert report["candidate_count_max"] == 12
    assert sum(report["candidate_count_histogram"].values()) == len(states)
    assert report["candidate_count_histogram"]["0"] == 1
    assert report["candidate_count_histogram"]["12"] == 1
