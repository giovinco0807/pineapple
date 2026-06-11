import random
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

from ofc_regular.apply_hu_self_regret_penalty import penalize_sample
from ofc_regular.hu_teacher_data import build_hu_turn3_sample
from ofc_regular.hu_turn3_model import (
    HU_DERIVED_OFFSET,
    HU_FEATURE_DIM,
    HU_MATCHUP_OFFSET,
    HuSklearnActionValueModel,
    HuTorchActionValueModel,
    hu_policy_sample,
    load_hu_action_value_model,
    sample_to_matrix,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.label_hu_turn3_states import (
    discarded_cards_from_state,
    label_hu_turn3_state,
)
from ofc_regular.hu_self_play_teacher_data import (
    _rollout_after_hero_t3_action,
    build_hu_self_play_turn3_sample,
    evaluate_hu_self_play_turn3_actions,
    hu_turn3_selection_metadata,
    passes_hu_turn3_selection,
    remaining_for_hu_teacher,
    to_act_order_for,
)
from ofc_regular.mine_hu_turn3_states import build_hu_turn3_state_record
from ofc_regular.policy import RegularAiPolicy
from ofc_regular.state import Board
from ofc_regular.train_torch_hu_turn3_streaming import (
    action_weights,
    ranking_loss_for_grouped_predictions,
)
from ofc_regular.turn3_model import _build_torch_mlp


def test_build_hu_turn3_sample_writes_hu_stage1_schema():
    sample = build_hu_turn3_sample(
        rng=random.Random(7),
        sample_id=3,
        future_samples=1,
    )

    assert sample["schema"] == "hu_stage1"
    assert sample["phase"] == "hu_turn3_9card"
    assert sample["seat"] in {"first", "second"}
    assert sample["to_act_order"] == sample["seat"]
    assert sample["opponent_board"]["top"]
    assert sample["dead_cards"]
    assert sample["actions"]


def test_hu_turn3_sample_encodes_to_hu_feature_matrix():
    sample = build_hu_turn3_sample(
        rng=random.Random(8),
        sample_id=4,
        future_samples=1,
    )

    features, targets = sample_to_matrix(sample)

    assert features.shape[0] == len(sample["actions"])
    assert features.shape[1] == HU_FEATURE_DIM
    assert targets.shape[0] == len(sample["actions"])


def test_hu_turn3_features_encode_opponent_top_fl_threat():
    hero = Board.from_rows(
        top=["Ah"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["Qh", "Qd"],
        middle=["3h", "4h", "5h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    dealt = ["Qs", "Ad", "7d"]
    actions = generate_turn_actions(hero, dealt)
    sample = hu_policy_sample(
        hero,
        dealt,
        actions,
        opponent_board=opponent,
        dead_cards=opponent.all_cards(),
        seat="first",
        to_act_order="first",
    )

    features, _targets = sample_to_matrix(sample)
    opponent_offset = HU_DERIVED_OFFSET + 12

    assert features.shape[1] == HU_FEATURE_DIM
    assert features[0, opponent_offset + 3] == 1.0
    assert features[0, opponent_offset + 4] == 1.0
    assert features[0, opponent_offset + 5] > 0.8


def test_hu_turn3_features_encode_row_matchup_strength():
    hero = Board.from_rows(
        top=["Ah"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "3d", "3c", "2s", "2d"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    dealt = ["Qs", "Ad", "7d"]
    actions = generate_turn_actions(hero, dealt)
    sample = hu_policy_sample(
        hero,
        dealt,
        actions,
        opponent_board=opponent,
        dead_cards=opponent.all_cards(),
        seat="first",
        to_act_order="first",
    )

    features, _targets = sample_to_matrix(sample)
    middle_offset = HU_MATCHUP_OFFSET + 12

    assert features.shape[1] == HU_FEATURE_DIM
    assert features[0, middle_offset + 1] == 1.0
    assert features[0, middle_offset + 6] >= 0.75
    assert features[0, middle_offset + 9] > 0.0
    assert features[0, middle_offset + 11] < 0.0


def test_hu_sklearn_model_aligns_expanded_features_for_old_estimators():
    class TinyEstimator:
        n_features_in_ = 3

        def predict(self, features):
            assert features.shape == (2, 3)
            return np.array([1.0, 2.0])

    model = HuSklearnActionValueModel(TinyEstimator())
    predictions = model.predict_matrix(np.ones((2, HU_FEATURE_DIM), dtype=np.float32))

    assert predictions.tolist() == [1.0, 2.0]


def test_hu_torch_model_round_trips_and_loads_generically():
    torch = pytest.importorskip("torch")
    hero = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    dealt = ["Qs", "Ah", "7d"]
    actions = generate_turn_actions(hero, dealt)
    sample = hu_policy_sample(
        hero,
        dealt,
        actions,
        opponent_board=opponent,
        dead_cards=opponent.all_cards(),
        seat="second",
        to_act_order="second",
    )

    net = _build_torch_mlp(torch, HU_FEATURE_DIM, (8,), 0.0)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.fill_(0.0)
    model = HuTorchActionValueModel(
        state_dict={key: value.detach().cpu().clone() for key, value in net.state_dict().items()},
        hidden_layer_sizes=(8,),
        feature_mean=np.zeros(HU_FEATURE_DIM, dtype=np.float32),
        feature_scale=np.ones(HU_FEATURE_DIM, dtype=np.float32),
        target_mean=0.0,
        target_scale=1.0,
    )

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "hu_turn3.pt"
        model.save(path)
        loaded = load_hu_action_value_model(path)

    assert isinstance(loaded, HuTorchActionValueModel)
    predictions = loaded.predict_sample(sample)
    assert predictions.shape == (len(actions),)


def test_ranking_loss_penalizes_wrong_best_order():
    torch = pytest.importorskip("torch")
    targets = torch.tensor([3.0, 1.0, 0.0], dtype=torch.float32)
    good_predictions = torch.tensor([3.0, 1.0, 0.0], dtype=torch.float32)
    bad_predictions = torch.tensor([0.0, 2.0, 3.0], dtype=torch.float32)

    good_loss = ranking_loss_for_grouped_predictions(
        torch,
        good_predictions,
        targets,
        [0, 3],
        max_margin=2.0,
        gap_tolerance=1e-6,
    )
    bad_loss = ranking_loss_for_grouped_predictions(
        torch,
        bad_predictions,
        targets,
        [0, 3],
        max_margin=2.0,
        gap_tolerance=1e-6,
    )

    assert float(good_loss.item()) == 0.0
    assert float(bad_loss.item()) > 0.0


def test_hu_torch_action_weights_include_source_weight():
    weights = action_weights(
        np.array([2.0, 1.0], dtype=np.float64),
        best_action_weight=2.0,
        best_action_tolerance=1e-9,
        source_weight=0.5,
    )

    np.testing.assert_allclose(weights, [1.0, 0.5])


def test_ranking_loss_ignores_zero_weight_groups():
    torch = pytest.importorskip("torch")
    targets = torch.tensor([2.0, 0.0, 2.0, 0.0], dtype=torch.float32)
    predictions = torch.tensor([0.0, 2.0, 2.0, 0.0], dtype=torch.float32)

    unweighted_loss = ranking_loss_for_grouped_predictions(
        torch,
        predictions,
        targets,
        [0, 2, 4],
        max_margin=2.0,
        gap_tolerance=1e-6,
    )
    weighted_loss = ranking_loss_for_grouped_predictions(
        torch,
        predictions,
        targets,
        [0, 2, 4],
        max_margin=2.0,
        gap_tolerance=1e-6,
        group_weights=[0.0, 1.0],
    )

    assert float(unweighted_loss.item()) > 0.0
    assert float(weighted_loss.item()) == 0.0


def test_self_regret_penalty_can_change_hu_teacher_best_action():
    class BaselineModel:
        def predict_sample(self, sample):
            return np.array([0.0, 10.0])

    sample = {
        "rule_set": "regular",
        "schema": "hu_stage1",
        "phase": "hu_turn3_9card",
        "seat": "first",
        "to_act_order": "first",
        "board": {"top": ["Qh"], "middle": ["Kh", "Kd", "6c", "8s"], "bottom": ["9c", "9d", "9s", "Kc"]},
        "opponent_board": {"top": ["2h"], "middle": ["3h"], "bottom": ["4h"]},
        "dead_cards": [],
        "dealt": ["Qs", "Ah", "7d"],
        "best_action": 0,
        "score_gap": 1.0,
        "actions": [
            {
                "placements": [["Qs", "top"], ["Ah", "middle"]],
                "discards": ["7d"],
                "score": 5.0,
                "next_board": {"top": ["Qh", "Qs"], "middle": ["Kh", "Kd", "6c", "8s", "Ah"], "bottom": ["9c", "9d", "9s", "Kc"]},
            },
            {
                "placements": [["Qs", "middle"], ["Ah", "bottom"]],
                "discards": ["7d"],
                "score": 4.5,
                "next_board": {"top": ["Qh"], "middle": ["Kh", "Kd", "6c", "8s", "Qs"], "bottom": ["9c", "9d", "9s", "Kc", "Ah"]},
            },
        ],
    }

    penalized, stats = penalize_sample(
        sample,
        baseline_model=BaselineModel(),
        penalty_weight=1.0,
        free_regret=0.0,
    )

    assert stats["changed_best"] == 1.0
    assert penalized["actions"][0]["placements"] == [["Qs", "middle"], ["Ah", "bottom"]]
    assert penalized["actions"][1]["self_regret_penalty"] == 10.0


def test_hu_self_play_turn3_sample_uses_partial_opponent_board():
    hero = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )

    sample = build_hu_self_play_turn3_sample(
        sample_id=5,
        board=hero,
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=opponent,
        hero_seat="first",
        hero_policy=RegularAiPolicy(seed=1),
        opponent_policy=RegularAiPolicy(seed=2),
        future_samples=1,
        rng=random.Random(9),
    )

    assert sample is not None
    assert sample["schema"] == "hu_stage1"
    assert sample["source"] == "self_play_rollout"
    assert sample["to_act_order"] == "first"
    assert sample["opponent_board"]["middle"] == ["3h", "4h", "5h", "6h"]
    assert sample["actions"]


def test_hu_self_play_turn3_teacher_applies_self_regret_penalty():
    class DescendingSelfModel:
        def predict_sample(self, sample):
            return np.arange(len(sample["actions"]), 0, -1, dtype=np.float64)

    hero = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )

    ranked = evaluate_hu_self_play_turn3_actions(
        board=hero,
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=opponent,
        hero_seat="first",
        hero_policy=RegularAiPolicy(seed=1),
        opponent_policy=RegularAiPolicy(seed=2),
        self_regret_model=DescendingSelfModel(),
        self_regret_penalty_weight=0.5,
        self_regret_free=0.0,
        future_samples=1,
        rng=random.Random(9),
    )

    penalized = [action for action in ranked if action.get("self_regret_penalty", 0.0) > 0.0]
    assert penalized
    for action in penalized:
        assert action["score"] == action["raw_score"] - action["self_regret_penalty"]


def test_hu_turn3_selection_metadata_marks_candidate_disagreement():
    class BaselineModel:
        def choose_action_index(self, sample):
            return 0

    class CandidateModel:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[0] = 2.0
            predictions[-1] = 11.0
            return predictions

    class CompareModel:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[0] = 7.0
            predictions[-1] = 3.0
            return predictions

    hero = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )

    selection = hu_turn3_selection_metadata(
        board=hero,
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=opponent,
        dead_cards=["2c"],
        hero_seat="second",
        baseline_turn3_model=BaselineModel(),
        selection_hu_model=CandidateModel(),
        compare_hu_model=CompareModel(),
    )

    assert selection is not None
    assert selection["baseline_index"] == 0
    assert selection["hu_index"] != selection["baseline_index"]
    assert selection["disagreement"] is True
    assert selection["predicted_margin_vs_baseline"] == 9.0
    assert selection["compare_hu_index"] == 0
    assert selection["compare_hu_disagreement"] is True
    assert selection["compare_predicted_margin_vs_baseline"] == 0.0


def test_hu_turn3_selection_filter_uses_margin_band_and_disagreement():
    selection = {
        "disagreement": True,
        "compare_hu_disagreement": True,
        "predicted_margin_vs_baseline": 8.5,
    }

    assert passes_hu_turn3_selection(
        selection,
        min_margin=8.0,
        max_margin=9.0,
        require_disagreement=True,
        require_compare_disagreement=True,
    )
    assert not passes_hu_turn3_selection(selection, min_margin=9.0)
    assert not passes_hu_turn3_selection(selection, max_margin=8.0)
    assert not passes_hu_turn3_selection(
        {"disagreement": False, "predicted_margin_vs_baseline": 8.5},
        require_disagreement=True,
    )
    assert not passes_hu_turn3_selection(
        {
            "disagreement": True,
            "compare_hu_disagreement": False,
            "predicted_margin_vs_baseline": 8.5,
        },
        require_compare_disagreement=True,
    )


def test_hu_turn3_state_record_keeps_discards_separate_from_visible_dead():
    hero = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    selection = {
        "hu_index": 1,
        "baseline_index": 0,
        "disagreement": True,
        "predicted_margin_vs_baseline": 8.0,
    }

    record = build_hu_turn3_state_record(
        state_id=2,
        seed=10,
        hand_seed=12,
        hand_index=2,
        player=1,
        board=hero,
        opponent_board=opponent,
        dealt_cards=["Qs", "Ah", "7d"],
        discarded_cards=["2c"],
        hero_seat="second",
        selection=selection,
    )

    assert record["schema"] == "hu_stage1_state"
    assert record["hero_seat"] == "second"
    assert record["discarded_cards"] == ["2c"]
    assert "2h" in record["visible_dead_cards"]
    assert "2h" not in record["discarded_cards"]
    assert record["selection"] == selection


def test_label_hu_turn3_state_builds_teacher_sample_from_mined_state():
    hero = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    state = build_hu_turn3_state_record(
        state_id=3,
        seed=10,
        hand_seed=12,
        hand_index=2,
        player=0,
        board=hero,
        opponent_board=opponent,
        dealt_cards=["Qs", "Ah", "7d"],
        discarded_cards=["2c"],
        hero_seat="first",
        selection={
            "hu_index": 1,
            "baseline_index": 0,
            "disagreement": True,
            "predicted_margin_vs_baseline": 8.0,
        },
    )
    policy_bundle = SimpleNamespace(opening=None, turn1=None, turn2=None, turn3=None)

    sample = label_hu_turn3_state(
        state,
        sample_id=4,
        policy_bundle=policy_bundle,
        future_samples=1,
        opening_lookahead_samples=1,
        self_regret_penalty_weight=0.0,
        self_regret_free=0.0,
        rng=random.Random(5),
        policy_seed=6,
    )

    assert sample is not None
    assert sample["schema"] == "hu_stage1"
    assert sample["source"] == "mined_state_rollout"
    assert sample["mined_state"]["state_id"] == 3
    assert sample["selection"]["predicted_margin_vs_baseline"] == 8.0
    assert sample["reference_actions"]["baseline"]["original_index"] == 0
    assert sample["reference_actions"]["selection_hu"]["original_index"] == 1
    assert "delta_best_vs_selection_hu" in sample["reference_actions"]
    assert sample["actions"]


def test_discarded_cards_from_legacy_state_removes_opponent_board_cards():
    opponent = Board.from_rows(top=["2h"], middle=["3h"], bottom=["4h"])
    state = {"dead_cards": ["2h", "3h", "4h", "9c", "Tc"]}

    assert discarded_cards_from_state(state, opponent) == ("9c", "Tc")


def test_remaining_for_hu_teacher_removes_dead_cards():
    hero = Board.from_rows(top=["Qh"])
    opponent = Board.from_rows(middle=["Ah"], bottom=["2c"])

    remaining = remaining_for_hu_teacher(hero, ["Kd", "Ks", "3h"], opponent, ["7d", "8d"])

    assert "7d" not in remaining
    assert "8d" not in remaining
    assert len(remaining) == 44


def test_to_act_order_is_second_when_opponent_is_ahead():
    hero = Board.from_rows(top=["Qh"], middle=["Kh", "Kd", "6c", "8s"], bottom=["9c", "9d", "9s", "Kc"])
    opponent = Board.from_rows(
        top=["2h", "2d"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh", "Qd"],
    )

    assert to_act_order_for(hero, opponent) == "second"


def test_rollout_final_turn_order_respects_hero_seat():
    class LoggingPolicy:
        def __init__(self, label, log):
            self.label = label
            self.log = log

        def choose_action(self, board, dealt, *, dead_cards=(), opponent_board=None):
            self.log.append((self.label, tuple(dead_cards)))
            return generate_turn_actions(board, dealt)[0]

    hero = Board.from_rows(
        top=["Ah", "Kh"],
        middle=["Qh", "Jh", "Th", "9h"],
        bottom=["8h", "7h", "6h", "5h", "4h"],
    )
    opponent = Board.from_rows(
        top=["Ad", "Kd"],
        middle=["Qd", "Jd", "Td", "9d"],
        bottom=["8d", "7d", "6d", "5d", "4d"],
    )
    log = []

    _rollout_after_hero_t3_action(
        hero_board=hero,
        opponent_board=opponent,
        dead_cards=["9c"],
        hero_seat="second",
        future_cards=["2c", "3c", "4c", "5c", "6c", "7c"],
        hero_policy=LoggingPolicy("hero", log),
        opponent_policy=LoggingPolicy("opponent", log),
    )

    assert [entry[0] for entry in log[:2]] == ["opponent", "hero"]
    assert "9c" in log[0][1]
