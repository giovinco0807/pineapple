from types import SimpleNamespace
from copy import deepcopy

import pytest

from ofc_regular.annotate_hu_turn1_teacher_baseline import (
    action_signature,
    actor_observation_for_row,
    annotate_rows,
    choose_policy_action_from_row,
)
from ofc_regular.hu_infoset import InformationSetError
from ofc_regular.policy import board_to_json
from ofc_regular.state import Board


def _action(card: str, row: str, *, score: float, index: int) -> dict:
    return {
        "placements": [[card, row], ["2c", "bottom"]],
        "discards": ["3d"],
        "score": score,
        "original_index": index,
    }


def test_annotate_rows_matches_action_by_signature_not_row_order():
    low = _action("As", "top", score=-1.0, index=11)
    baseline = _action("Kh", "middle", score=2.5, index=7)
    rows, summary = annotate_rows(
        [{"seat": "first", "actions": [low, baseline]}],
        choose_baseline=lambda _row: SimpleNamespace(
            placements=(("Kh", "middle"), ("2c", "bottom")),
            discards=("3d",),
        ),
        profile="stage9f_p2",
    )

    assert rows[0]["baseline_action_row_index"] == 1
    assert rows[0]["baseline_action_index"] == 7
    assert rows[0]["baseline_teacher_ev"] == 2.5
    assert summary["missing_actions"] == 0


def test_annotate_rows_rejects_missing_baseline_action():
    with pytest.raises(ValueError, match="missing from legal actions"):
        annotate_rows(
            [{"seat": "first", "actions": [_action("As", "top", score=1.0, index=0)]}],
            choose_baseline=lambda _row: SimpleNamespace(
                placements=(("Qh", "middle"), ("2c", "bottom")),
                discards=("3d",),
            ),
            profile="stage9f_p2",
        )


def test_action_signature_is_order_invariant():
    left = {"placements": [["As", "top"], ["2c", "bottom"]], "discards": ["3d"]}
    right = SimpleNamespace(placements=(("2c", "bottom"), ("As", "top")), discards=("3d",))

    assert action_signature(left) == action_signature(right)


def _teacher_row() -> dict:
    hero = Board.from_rows(
        top=["Ah"],
        middle=["Kd"],
        bottom=["2s", "3s", "9s"],
    )
    opponent = Board.from_rows(
        top=["Qh"],
        middle=["Jd", "Td"],
        bottom=["4s", "5s", "6s", "7s"],
    )
    return {
        "seat": "second",
        "turn": "T1",
        "board": board_to_json(hero),
        "opponent_board": board_to_json(opponent),
        "dealt": ["8c", "9c", "Tc"],
        "visible_dead_cards": list(opponent.all_cards()),
        "hero_private_discards": [],
        # Offline-only fields intentionally disagree across the two test rows.
        "dead_cards": ["Jc"],
        "true_dead_cards": ["Jc"],
        "opponent_private_discards": ["Jc"],
        "hand_seed": 17,
    }


def test_policy_chooser_uses_actor_observation_and_ignores_opponent_truth():
    class RecordingPolicy:
        def __init__(self):
            self.observations = []

        def choose_action_observation(self, observation, **_kwargs):
            self.observations.append(observation)
            return "chosen"

    first = _teacher_row()
    second = deepcopy(first)
    second["dead_cards"] = ["Qc"]
    second["true_dead_cards"] = ["Qc"]
    second["opponent_private_discards"] = ["Qc"]
    policy = RecordingPolicy()

    assert choose_policy_action_from_row(policy, first, default_seed=1) == "chosen"
    assert choose_policy_action_from_row(policy, second, default_seed=1) == "chosen"

    assert policy.observations[0] == policy.observations[1]
    assert "Jc" not in policy.observations[0].known_unavailable_cards()
    assert "Qc" not in policy.observations[0].known_unavailable_cards()


def test_t1_annotation_rejects_ambiguous_dead_cards_without_visible_field():
    row = _teacher_row()
    row.pop("visible_dead_cards")

    with pytest.raises(InformationSetError, match="visible_dead_cards"):
        actor_observation_for_row(row)
