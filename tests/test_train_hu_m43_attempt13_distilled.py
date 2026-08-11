from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import ofc_regular.hu_m43_attempt13_teacher as teacher
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt05_model import Attempt05FoldOutput, HuM43Attempt05Model
from ofc_regular.hu_m43_attempt13_distilled_model import (
    HU_M43_ATTEMPT13_DISTILLED_FEATURE_DIM,
)
from ofc_regular.hu_m43_attempt13_teacher import (
    ATTEMPT13_FROZEN_MODEL_ID,
    ATTEMPT13_FROZEN_MODEL_SHA256,
    FrozenAttempt13LambdaRanker,
)
from ofc_regular.hu_m4_t1_teacher import _ActionScores
from ofc_regular.state import Board
from ofc_regular.freeze_hu_m43_attempt13_development import (
    ATTEMPT13_DEVELOPMENT_GO_FREEZE_SCHEMA,
)
from ofc_regular.run_hu_m43_attempt13 import ATTEMPT13_ROW_SCHEMA
from ofc_regular.select_hu_m43_attempt13_audit50 import (
    ATTEMPT13_AUDIT50_DECISION_SCHEMA,
    ATTEMPT13_AUDIT50_RECEIPT_SCHEMA,
)
from ofc_regular.select_hu_m43_attempt13_development import (
    ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA,
    ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA,
)
from ofc_regular.train_hu_m43_attempt13_distilled import (
    ATTEMPT13_AUDIT50_DECISION_SCHEMA as TRAIN_AUDIT50_DECISION_SCHEMA,
    ATTEMPT13_AUDIT50_SELECTOR_RECEIPT_SCHEMA as TRAIN_AUDIT50_RECEIPT_SCHEMA,
    ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA as TRAIN_DEVELOPMENT_DECISION_SCHEMA,
    ATTEMPT13_DEVELOPMENT_PASS_FREEZE_SCHEMA as TRAIN_DEVELOPMENT_FREEZE_SCHEMA,
    ATTEMPT13_DEVELOPMENT_ROW_SCHEMA,
    ATTEMPT13_DEVELOPMENT_SELECTOR_RECEIPT_SCHEMA as TRAIN_DEVELOPMENT_RECEIPT_SCHEMA,
    ATTEMPT13_DISTILLATION_CONFIG_SHA256,
    ATTEMPT13_PROFILES,
    Attempt13DistillationState,
    assign_attempt13_identity_folds,
    fit_attempt13_distilled_model,
    prepare_attempt13_distillation_states,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hu_joint_policy_m43_attempt13_distillation.json"


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


def _ranker() -> FrozenAttempt13LambdaRanker:
    model = HuM43Attempt05Model(
        family="lambda_rank",
        fold_predictors=tuple(_CandidateFold(index) for index in range(5)),
        runtime_enabled=False,
        winner_frozen=False,
        model_id=ATTEMPT13_FROZEN_MODEL_ID,
    )
    return FrozenAttempt13LambdaRanker(
        model=model, artifact_sha256=ATTEMPT13_FROZEN_MODEL_SHA256
    )


def _variable_states() -> tuple[Attempt13DistillationState, ...]:
    rng = np.random.default_rng(4312)
    group_sizes = [*range(1, 28), 4, 9]
    states: list[Attempt13DistillationState] = []
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
            Attempt13DistillationState(
                root_index=root,
                root_profile=ATTEMPT13_PROFILES[root % len(ATTEMPT13_PROFILES)],
                observation_fingerprint=f"{root + 1:064x}",
                sample={},
                baseline_index=group_size - 1,
                proposal_indices=tuple(range(group_size)),
                features=rng.normal(
                    size=(group_size, HU_M43_ATTEMPT13_DISTILLED_FEATURE_DIM)
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


def _observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("9h",), middle=("Th", "Jh"), bottom=("Qh", "Kh")
        ),
        opponent_public_board=Board.from_rows(
            top=("2h",),
            middle=("3h", "4h", "5h"),
            bottom=("6h", "7h", "8h"),
        ),
        dealt_cards=("Ah", "2d", "3d"),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )


class _Stage9fPolicy:
    def __init__(self, seat: str) -> None:
        self.seat = seat
        self.topk_context = {
            "runtime_profile": "stage9f_p2",
            "runtime_status": "p2_fixed",
        }


def _constant_teacher_scores(_observation, actions, batch, _selector):
    rows = []
    for position in range(len(actions)):
        value = 0.0 if position == len(actions) - 1 else float(len(actions) - position)
        rows.append(_ActionScores((value,) * len(batch.particles)))
    return tuple(rows)


def test_attempt13_config_freezes_all_legal_candidates_without_padding() -> None:
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
        "9b860be1de05570840e0fd7b1e4a3e3c2f52e5785be4a882d530b5f9baf5c6c2"
    )
    assert config["data"]["attempt12_rows_may_fit_or_calibrate"] is False
    assert config["labels"]["safe_positive"].endswith("E512_mean_gt_0")
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == (
        ATTEMPT13_DISTILLATION_CONFIG_SHA256
    )


def test_attempt13_trainer_binds_live_selector_freezer_and_row_schemas() -> None:
    assert TRAIN_DEVELOPMENT_DECISION_SCHEMA == ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA
    assert TRAIN_DEVELOPMENT_RECEIPT_SCHEMA == ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA
    assert TRAIN_DEVELOPMENT_FREEZE_SCHEMA == ATTEMPT13_DEVELOPMENT_GO_FREEZE_SCHEMA
    assert TRAIN_AUDIT50_DECISION_SCHEMA == ATTEMPT13_AUDIT50_DECISION_SCHEMA
    assert TRAIN_AUDIT50_RECEIPT_SCHEMA == ATTEMPT13_AUDIT50_RECEIPT_SCHEMA
    assert ATTEMPT13_DEVELOPMENT_ROW_SCHEMA == ATTEMPT13_ROW_SCHEMA


def test_attempt13_trainer_accepts_producer_validated_teacher_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation()
    actions = tuple(generate_turn_actions(observation.hero_board, observation.dealt_cards))
    baseline = action_key(actions[-1]).to_token()
    monkeypatch.setattr(teacher._attempt12, "_score_actions", _constant_teacher_scores)
    monkeypatch.setattr(
        teacher._attempt12, "_score_actions_batched", _constant_teacher_scores
    )
    config = teacher.Attempt13TeacherConfig(
        frozen_model_sha256=ATTEMPT13_FROZEN_MODEL_SHA256,
        hand_seed=13001,
        rerank_seed=13002,
        veto_seed=13003,
        stress_seed=13004,
        confirmation_seed=13005,
        evaluation_seed=13006,
        child_policy_seed=13007,
        run_id="attempt13-distillation-fixture",
    )
    ranker = _ranker()
    payload = teacher.evaluate_attempt13_t1_second(
        observation,
        baseline_action_key=baseline,
        ranker=ranker,
        t2_policies={
            "first": _Stage9fPolicy("first"),
            "second": _Stage9fPolicy("second"),
        },
        config=config,
    )
    rows = [
        {
            "schema": ATTEMPT13_ROW_SCHEMA,
            "root_index": 0,
            "root_profile": ATTEMPT13_PROFILES[0],
            "provenance": {
                "mode": "development",
                "development_only": True,
                "fit_allowed": False,
                "threshold_selection_allowed": False,
                "runtime_activation_allowed": False,
            },
            "policy_observation": observation.to_dict(),
            "observation_fingerprint": observation.fingerprint(),
            "baseline_action_key": baseline,
            "teacher": payload,
        }
    ]
    states = prepare_attempt13_distillation_states(
        rows, ranker=ranker, expected_roots=None
    )
    assert len(states) == 1
    assert states[0].group_size == len(actions)
    assert states[0].teacher_override_fired is True
    assert states[0].selected_action_key == payload["decision"][
        "final_selected_action_key"
    ]


def test_attempt13_variable_lightgbm_groups_match_row_total() -> None:
    states = _variable_states()
    folds = assign_attempt13_identity_folds(states)
    np.testing.assert_array_equal(folds, assign_attempt13_identity_folds(states))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model, report = fit_attempt13_distilled_model(
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
