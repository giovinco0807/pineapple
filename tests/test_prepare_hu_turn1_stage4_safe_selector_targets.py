import json

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.policy import action_to_json, board_to_json
from ofc_regular.prepare_hu_turn1_stage4_safe_selector_targets import select_rows
from ofc_regular.state import Board


def _decision_row(delta: float, *, seed: int, fired: bool, reason: str = ""):
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd"],
        bottom=["9c", "9d"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h"],
        bottom=["7h", "8h"],
    )
    dealt = ["Qs", "Ah", "7d"]
    actions = generate_turn_actions(board, dealt)
    candidate_index = 2
    baseline_index = 0
    return {
        "schema": "hu_turn1_topk_confirm_decision_v1",
        "hand_id": seed,
        "hand_seed": seed,
        "paired_index": seed,
        "seat_swap": "ab",
        "seat": "first",
        "hero_board": board_to_json(board),
        "opponent_board": board_to_json(opponent),
        "dead_cards": list(opponent.all_cards()),
        "cards_to_place": dealt,
        "override_fired": fired,
        "no_override_reason": reason,
        "realized_delta_valid": True,
        "realized_candidate_seat_delta": delta,
        "candidate_action_index": candidate_index,
        "fallback_action_index": baseline_index,
        "hu_turn1_action": action_to_json(board, actions[candidate_index]),
        "baseline_action": action_to_json(board, actions[baseline_index]),
        "safe_selector_score": 0.25 if reason == "below_safe_selector" else 0.8 if fired else None,
    }


def test_prepare_hu_turn1_stage4_targets_selects_selector_reached_rows(tmp_path):
    rows = [
        _decision_row(4.0, seed=1, fired=True),
        _decision_row(-8.0, seed=2, fired=True),
        _decision_row(0.5, seed=3, fired=False, reason="below_safe_selector"),
        _decision_row(9.0, seed=4, fired=False, reason="below_confirm_delta"),
    ]
    input_path = tmp_path / "decisions.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    selected, summary = select_rows(
        [input_path],
        positive_delta=2.0,
        hard_negative_delta=-5.0,
        include_below_safe_selector=True,
    )

    assert len(selected) == 3
    assert summary["label_counts"] == {"gray": 1, "hard_negative": 1, "positive": 1}
    assert summary["skipped_counts"] == {"selector_not_reached": 1}
    assert selected[0]["hu_turn1_safe_override_label_policy"] == (
        "positive_delta>=2;hard_negative_delta<=-5;gray_between"
    )
