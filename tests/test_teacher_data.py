import random

from ofc_regular.teacher_data import build_final_turn_sample, sample_final_turn_state


def test_sample_final_turn_state_has_board_and_deal_cards():
    board, dealt = sample_final_turn_state(random.Random(7))
    assert board.card_count() == 11
    assert sum(board.open_slots(row) for row in ("top", "middle", "bottom")) == 2
    assert len(dealt) == 3
    assert len(set(board.all_cards()) | set(dealt)) == 14
    assert set(board.all_cards()).isdisjoint(dealt)


def test_build_final_turn_sample_sorts_best_action_first():
    sample = build_final_turn_sample(
        rng=random.Random(11),
        sample_id=3,
        fl_ev={14: 8.0},
    )
    assert sample["sample_id"] == 3
    assert sample["rule_set"] == "regular"
    assert sample["phase"] == "final_turn"
    assert sample["best_action"] == 0
    assert sample["actions"]
    scores = [action["score"] for action in sample["actions"]]
    assert scores == sorted(scores, reverse=True)
    assert sample["score_gap"] == scores[0] - scores[1]
    assert all(len(action["placements"]) == 2 for action in sample["actions"])
