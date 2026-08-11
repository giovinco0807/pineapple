import math

from ofc_regular.action_space import generate_actions, generate_turn_actions
from ofc_regular.action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.cards import ALL_CARDS
from ofc_regular.counter_rng import COUNTER_RNG_SCHEMA
from ofc_regular.hu_turn0_teacher_pilot import (
    build_turn0_teacher_sample,
    evaluate_turn0_action_subset,
    parse_args,
    remaining_for_turn0_state,
)
from ofc_regular.policy import RegularAiPolicy, action_to_json
from ofc_regular.state import Board
from ofc_regular.teacher import DEFAULT_FL_EV


class FirstLegalPolicy:
    def __init__(self, tag, calls):
        self.tag = tag
        self.calls = calls
        self.decision_context = {}

    def choose_action(
        self,
        board,
        dealt_cards,
        *,
        dead_cards=(),
        opponent_board=None,
        hand_id=None,
        game_id=None,
        decision_seed=None,
        street=None,
    ):
        self.calls.append((self.tag, street, decision_seed))
        dealt = tuple(dealt_cards)
        actions = (
            generate_actions(board, dealt)
            if board.card_count() == 0
            else generate_turn_actions(board, dealt)
        )
        return actions[0]


class IncreasingScoreCandidateModel:
    def predict_sample(self, sample):
        return [float(index) for index, _action in enumerate(sample["actions"])]


def _policies(calls):
    return [FirstLegalPolicy("first", calls), FirstLegalPolicy("second", calls)]


def test_turn0_first_seat_uses_all_action_common_futures_and_action_independent_seeds():
    board = Board.from_rows()
    opponent = Board.from_rows()
    dealt = ("As", "Kd", "Qh", "Jc", "9s")
    remaining = remaining_for_turn0_state(board, dealt, opponent)
    calls = []

    result = evaluate_turn0_action_subset(
        board=board,
        opponent_board=opponent,
        dealt=dealt,
        remaining_cards=remaining,
        hero_player=0,
        policies=_policies(calls),
        hand_seed=101,
        sample_id=3,
        future_samples=2,
        future_seed=202,
        action_indices=(0, 1),
    )

    assert result["total_legal_actions"] == 232
    assert result["evaluated_action_count"] == 2
    assert result["actions_truncated"] is True
    assert result["future_cards_per_rollout"] == 29
    assert result["baseline_action_index"] == 0
    assert result["common_random_futures_verified"] is True
    assert result["rng_schema"] == COUNTER_RNG_SCHEMA
    assert len(result["common_random_future_digest"]) == 64
    assert all(action["rollout_count"] == 2 for action in result["actions"])
    assert all(math.isfinite(action["ev"]) for action in result["actions"])
    assert all(math.isfinite(action["se"]) for action in result["actions"])
    assert all(math.isfinite(action["delta_vs_baseline"]) for action in result["actions"])
    assert all(math.isfinite(action["delta_se_vs_baseline"]) for action in result["actions"])
    assert all(math.isfinite(action["delta_z_vs_baseline"]) for action in result["actions"])
    baseline = next(
        action
        for action in result["actions"]
        if action["original_index"] == result["baseline_action_index"]
    )
    assert baseline["delta_vs_baseline"] == 0.0
    assert baseline["delta_se_vs_baseline"] == 0.0

    # One baseline T0 call, then 2 futures * 9 decisions per candidate action.
    rollout_calls = calls[1:]
    assert len(rollout_calls) == 36
    assert rollout_calls[:18] == rollout_calls[18:]


def test_turn0_candidate_iteration_order_preserves_per_action_values():
    board = Board.from_rows()
    opponent = Board.from_rows()
    dealt = ("As", "Kd", "Qh", "Jc", "9s")
    remaining = remaining_for_turn0_state(board, dealt, opponent)

    def evaluate(indices):
        return evaluate_turn0_action_subset(
            board=board,
            opponent_board=opponent,
            dealt=dealt,
            remaining_cards=remaining,
            hero_player=0,
            policies=[
                RegularAiPolicy(seed=11, seat="first"),
                RegularAiPolicy(seed=12, seat="second"),
            ],
            hand_seed=505,
            sample_id=6,
            future_samples=2,
            future_seed=606,
            action_indices=indices,
        )

    forward = evaluate((0, 1, 2))
    reverse = evaluate((2, 1, 0))
    assert {
        row["canonical_action_key"]: row["score"] for row in forward["actions"]
    } == {
        row["canonical_action_key"]: row["score"] for row in reverse["actions"]
    }


def test_turn0_second_seat_encodes_visible_opponent_board_and_replay_metadata():
    board = Board.from_rows()
    opponent = Board.from_rows(
        top=["2c"],
        middle=["3d", "4h"],
        bottom=["5s", "6c"],
    )
    dealt = ("As", "Kd", "Qh", "Jc", "9s")
    remaining = remaining_for_turn0_state(board, dealt, opponent)
    calls = []

    sample = build_turn0_teacher_sample(
        board=board,
        opponent_board=opponent,
        dealt=dealt,
        remaining_cards=remaining,
        hero_player=1,
        policies=_policies(calls),
        hand_seed=303,
        sample_id=4,
        future_samples=1,
        future_seed=404,
        profile_name="stage18_p1",
        opponent_profile="stage18_p1",
        source_bucket="unit",
        action_indices=(0, 1),
    )

    assert sample["schema"] == "hu_turn0_stage1_teacher_v1"
    assert sample["source"] == "hu_turn0_terminal_rollout_mc1"
    assert sample["phase"] == "hu_turn0_0card"
    assert sample["seat"] == "second"
    assert sample["opponent_board"] == {
        "top": ["2c"],
        "middle": ["3d", "4h"],
        "bottom": ["5s", "6c"],
    }
    assert set(sample["visible_dead_cards"]) == set(opponent.all_cards())
    assert sample["remaining_card_count"] == 42
    assert sample["future_cards_per_rollout"] == 24
    assert sample["total_legal_actions"] == 232
    assert sample["evaluated_action_count"] == 2
    assert sample["common_random_futures_verified"] is True
    assert sample["replay_ready"] is True
    assert sample["t1_continuation"] == "stage18_p1"
    assert sample["t2_continuation"] == "stage9f_p2"
    assert sample["t3_continuation"] == "stage7_m5_r10"
    assert sample["fl_ev"][14] == DEFAULT_FL_EV[14]
    assert math.isfinite(sample["delta_best_vs_baseline_se"])


def test_turn0_teacher_labels_are_deterministic_for_same_state_and_seed():
    board = Board.from_rows()
    opponent = Board.from_rows()
    dealt = ("Ah", "Kh", "Qh", "8c", "2d")
    remaining = remaining_for_turn0_state(board, dealt, opponent)

    def build():
        return build_turn0_teacher_sample(
            board=board,
            opponent_board=opponent,
            dealt=dealt,
            remaining_cards=remaining,
            hero_player=0,
            policies=_policies([]),
            hand_seed=505,
            sample_id=6,
            future_samples=2,
            future_seed=606,
            profile_name="stage18_p1",
            opponent_profile="stage18_p1",
            source_bucket="unit",
            action_indices=(0, 7, 31),
        )

    left = build()
    right = build()
    for sample in (left, right):
        for action in sample["actions"]:
            action.pop("action_eval_seconds", None)
    assert left == right


def test_turn0_candidate_topk_selects_model_actions_and_keeps_baseline(monkeypatch):
    board = Board.from_rows()
    opponent = Board.from_rows(
        top=["2c"],
        middle=["3d", "4h"],
        bottom=["5s", "6c"],
    )
    dealt = ("As", "Kd", "Qh", "Jc", "9s")
    remaining = remaining_for_turn0_state(board, dealt, opponent)

    def fake_evaluate_turn0_action_subset(**kwargs):
        actions = generate_actions(kwargs["board"], kwargs["dealt"])
        selected = list(kwargs["action_indices"])
        if 0 not in selected:
            selected.append(0)
        evaluated = []
        for rank, index in enumerate(selected):
            payload = action_to_json(kwargs["board"], actions[index])
            score = float(len(selected) - rank)
            payload.update(
                {
                    "action_index": index,
                    "original_index": index,
                    "canonical_action_key": action_key(actions[index]).to_token(),
                    "score": score,
                    "ev": score,
                    "se": 0.0,
                    "rollout_count": 1,
                    "common_random_future_digest": "a" * 64,
                    "action_eval_seconds": 0.0,
                }
            )
            evaluated.append(payload)
        return {
            "actions": evaluated,
            "total_legal_actions": len(actions),
            "evaluated_action_count": len(evaluated),
            "actions_truncated": True,
            "action_key_schema": ACTION_KEY_SCHEMA,
            "legal_action_set_digest": legal_action_set_digest(actions),
            "legal_action_order_digest": ordered_action_mapping_digest(actions),
            "baseline_action_index": 0,
            "baseline_action_key": action_key(actions[0]).to_token(),
            "baseline_ev": evaluated[-1]["ev"],
            "best_action_index": evaluated[0]["original_index"],
            "best_action_key": evaluated[0]["canonical_action_key"],
            "best_ev": evaluated[0]["ev"],
            "delta_best_vs_baseline": evaluated[0]["ev"] - evaluated[-1]["ev"],
            "delta_best_vs_baseline_se": 0.0,
            "common_random_future_digest": "a" * 64,
            "common_random_futures_verified": True,
            "future_seed": kwargs["future_seed"],
            "future_cards_per_rollout": 24,
        }

    monkeypatch.setattr(
        "ofc_regular.hu_turn0_teacher_pilot.evaluate_turn0_action_subset",
        fake_evaluate_turn0_action_subset,
    )

    sample = build_turn0_teacher_sample(
        board=board,
        opponent_board=opponent,
        dealt=dealt,
        remaining_cards=remaining,
        hero_player=1,
        policies=_policies([]),
        hand_seed=707,
        sample_id=8,
        future_samples=1,
        future_seed=808,
        profile_name="stage18_p1",
        opponent_profile="stage18_p1",
        source_bucket="unit",
        candidate_model=IncreasingScoreCandidateModel(),
        candidate_topk=2,
    )

    assert sample["candidate_selector"]["selected_indices"] == [231, 230]
    assert sample["evaluated_action_count"] == 3
    assert {action["original_index"] for action in sample["actions"]} == {0, 230, 231}


def test_turn0_remaining_cards_and_cli_defaults():
    board = Board.from_rows()
    opponent = Board.from_rows(top=["2c"], middle=["3d", "4h"], bottom=["5s", "6c"])
    dealt = ("As", "Kd", "Qh", "Jc", "9s")
    remaining = remaining_for_turn0_state(board, dealt, opponent)

    assert len(remaining) == 42
    assert not (set(dealt) | set(opponent.all_cards())) & set(remaining)
    assert set(remaining) <= set(ALL_CARDS)

    parsed = parse_args(
        [
            "--output",
            "outputs/t0.jsonl",
            "--summary-output",
            "outputs/t0_summary.json",
        ]
    )
    assert parsed.profile == "stage18_p1"
    assert parsed.opponent_profile == "stage18_p1"
    assert parsed.future_samples == 1
    assert parsed.max_actions == 0
    assert parsed.seats == ("first", "second")
