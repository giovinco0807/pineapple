from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ofc_regular.hu_m43_attempt05_model import Attempt05FoldOutput, HuM43Attempt05Model
from ofc_regular.hu_m43_attempt12_distilled_model import (
    HU_M43_ATTEMPT12_DISTILLED_FEATURE_DIM,
)
from ofc_regular.hu_m43_attempt12_teacher import (
    ATTEMPT12_FROZEN_MODEL_ID,
    ATTEMPT12_FROZEN_MODEL_SHA256,
    FrozenAttempt12LambdaRanker,
)
from ofc_regular.train_hu_m43_attempt12_distilled import (
    ATTEMPT12_DISTILLATION_CONFIG_SHA256,
    ATTEMPT12_PROFILES,
    Attempt12DistillationState,
    assign_attempt12_identity_folds,
    fit_attempt12_distilled_model,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hu_joint_policy_m43_attempt12_distillation.json"


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


def _ranker() -> FrozenAttempt12LambdaRanker:
    model = HuM43Attempt05Model(
        family="lambda_rank",
        fold_predictors=tuple(_CandidateFold(index) for index in range(5)),
        runtime_enabled=False,
        winner_frozen=False,
        model_id=ATTEMPT12_FROZEN_MODEL_ID,
    )
    return FrozenAttempt12LambdaRanker(
        model=model, artifact_sha256=ATTEMPT12_FROZEN_MODEL_SHA256
    )


def _variable_states() -> tuple[Attempt12DistillationState, ...]:
    rng = np.random.default_rng(4312)
    group_sizes = [*range(1, 28), 4, 9]
    states: list[Attempt12DistillationState] = []
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
            Attempt12DistillationState(
                root_index=root,
                root_profile=ATTEMPT12_PROFILES[root % len(ATTEMPT12_PROFILES)],
                observation_fingerprint=f"{root + 1:064x}",
                sample={},
                baseline_index=group_size - 1,
                proposal_indices=tuple(range(group_size)),
                features=rng.normal(
                    size=(group_size, HU_M43_ATTEMPT12_DISTILLED_FEATURE_DIM)
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


def test_attempt12_config_freezes_all_legal_candidates_without_padding() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    candidate = config["candidate_generation"]
    assert candidate["nonbaseline_max"] == 26
    assert candidate["actual_count_range"] == [0, 26]
    assert candidate["padding_allowed"] is False
    assert candidate["duplicate_action_keys_allowed"] is False
    assert candidate["baseline_in_candidate_set"] is False
    assert candidate["baseline_appended_exactly_once"] is True
    assert candidate["lightgbm_group_size_range"] == [1, 27]
    assert config["data"]["source_search_plan_sha256"] == (
        "6a862e355d6136f488b6190fd74d1ec87b896d3c07ba6c48d97442cea2ac48c9"
    )
    assert config["data"]["attempt11_rows_may_fit_or_calibrate"] is False
    assert config["labels"]["safe_positive"].endswith("E512_mean_gt_0")
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == (
        ATTEMPT12_DISTILLATION_CONFIG_SHA256
    )


def test_attempt12_variable_lightgbm_groups_match_row_total() -> None:
    states = _variable_states()
    folds = assign_attempt12_identity_folds(states)
    np.testing.assert_array_equal(folds, assign_attempt12_identity_folds(states))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model, report = fit_attempt12_distilled_model(
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
    assert report["candidate_count_max"] == 26
    assert sum(report["candidate_count_histogram"].values()) == len(states)
    assert report["candidate_count_histogram"]["0"] == 1
    assert report["candidate_count_histogram"]["26"] == 1
