from ofc_regular.action_space import generate_turn_actions
from ofc_regular.convert_hu_turn3_decision_log_to_states import convert_rows
from ofc_regular.hu_infoset import ReplayTruth
from ofc_regular.policy import action_to_json, board_to_json
from ofc_regular.state import Board


def _hero_board():
    return Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )


def _opponent_board():
    return Board.from_rows(
        top=["2h", "3h", "4h"],
        middle=["2d", "3d", "4d", "5d"],
        bottom=["7c", "8c", "Tc", "Jc"],
    )


def _decision_row(*, include_visible_cards: bool = True):
    board = _hero_board()
    dealt = ("Qs", "Ah", "7d")
    actions = generate_turn_actions(board, dealt)
    row = {
        "hand_id": "h1",
        "game_id": "g1",
        "seed": 2026061801,
        "street": "T3",
        "turn": "T3",
        "seat": "second",
        "hero_board": board_to_json(board),
        "opponent_board": board_to_json(_opponent_board()),
        "cards_to_place": list(dealt),
        "stage3_action": action_to_json(board, actions[0]),
        "stage7_action": action_to_json(board, actions[1]),
        "fallback_action": action_to_json(board, actions[0]),
        "final_action": action_to_json(board, actions[1]),
        "override_fired": True,
        "no_override_reason": "",
        "stage7_predicted_margin": 2.5,
        "reference_margin": 12.0,
        "hu_turn3_min_margin": 2.0,
        "hu_turn3_reference_min_margin": 0.0,
    }
    # Deliberately ambiguous/poisoned legacy field: converters must not use it
    # as the actor observation.
    row["dead_cards"] = ["2c", "5c", "3c", "4c", "6c"]
    if include_visible_cards:
        row["visible_dead_cards"] = [*_opponent_board().all_cards(), "2c", "5c"]
        row["hero_private_discards"] = ["2c", "5c"]
    return row


def test_convert_decision_log_uses_visible_cards_and_never_true_dead_cards():
    states, counts = convert_rows([_decision_row()], source_label="c3_runtime")

    assert counts["written"] == 1
    assert counts["replay_ready"] == 1
    state = states[0]
    assert state["schema"] == "hu_stage1_state"
    assert state["phase"] == "hu_turn3_9card"
    assert state["replay_ready"] is True
    assert state["missing_dead_cards"] is False
    expected_visible = [*_opponent_board().all_cards(), "2c", "5c"]
    assert state["dead_cards"] == expected_visible
    assert state["visible_dead_cards"] == expected_visible
    assert "3c" not in state["visible_dead_cards"]
    assert state["policy_observation"]["hero_private_discards"] == ["2c", "5c"]
    assert state["selection"]["baseline_index"] == 0
    assert state["selection"]["hu_index"] == 1
    assert state["selection"]["disagreement"] is True
    assert state["selection"]["predicted_margin_vs_baseline"] == 2.5
    assert state["decision_seed"] == 2026061801
    assert state["seed"] == state["decision_seed"]
    assert state["seed_semantics"] == "policy_decision_seed"
    assert state["hand_seed"] is None
    assert state["hand_seed_source"] == "unavailable"
    assert state["hand_seed_available"] is False


def test_convert_prefers_explicit_hand_seed_over_decision_seed_and_hand_id():
    row = _decision_row()
    row.update(
        {
            "hand_id": 7001,
            "hand_seed": 7002,
            "seed": 9001,
        }
    )

    states, counts = convert_rows([row], source_label="explicit_seed")

    assert counts["written"] == 1
    state = states[0]
    assert state["hand_id"] == 7001
    assert state["hand_seed"] == 7002
    assert state["hand_seed_source"] == "explicit_hand_seed"
    assert state["hand_seed_available"] is True
    assert state["decision_seed"] == 9001
    assert state["seed"] == 9001
    assert state["state_id"] == "explicit_seed:7002:g1:7001:second:T3"


def test_convert_derives_runtime_hand_seed_from_numeric_hand_id_not_policy_seed():
    row = _decision_row()
    row["hand_id"] = 8123
    row["seed"] = 99123

    states, counts = convert_rows([row], source_label="runtime")

    assert counts["written"] == 1
    state = states[0]
    assert state["hand_seed"] == 8123
    assert state["hand_seed_source"] == "numeric_hand_id"
    assert state["decision_seed"] == 99123
    assert state["hand_seed"] != state["decision_seed"]
    assert state["state_id"] == "runtime:8123:g1:8123:second:T3"


def test_convert_opaque_hand_id_does_not_promote_policy_seed_to_hand_seed():
    first = _decision_row()
    second = _decision_row()
    second["seed"] = first["seed"] + 1

    first_states, _ = convert_rows([first], source_label="opaque")
    second_states, _ = convert_rows([second], source_label="opaque")

    assert first_states[0]["hand_id"] == "h1"
    assert first_states[0]["hand_seed"] is None
    assert second_states[0]["hand_seed"] is None
    assert first_states[0]["decision_seed"] != second_states[0]["decision_seed"]
    assert first_states[0]["state_id"] == second_states[0]["state_id"]


def test_convert_invalid_explicit_hand_seed_fails_closed():
    row = _decision_row()
    row["hand_seed"] = {"not": "a seed"}

    states, counts = convert_rows([row], source_label="invalid")

    assert states == []
    assert counts["invalid_hand_seed"] == 1
    assert counts["written"] == 0


def test_convert_legacy_decision_log_marks_replay_ineligible():
    states, counts = convert_rows([_decision_row(include_visible_cards=False)], source_label="legacy")

    assert counts["written"] == 1
    assert counts["replay_ready"] == 0
    assert counts["replay_ineligible"] == 1
    assert states[0]["replay_ready"] is False
    assert states[0]["missing_dead_cards"] is True
    assert states[0]["legacy_runtime_log"] is True
    assert states[0]["exclude_from_exact_replay"] is True


def test_convert_can_filter_replay_ineligible_rows():
    states, counts = convert_rows(
        [_decision_row(include_visible_cards=False)],
        source_label="legacy",
        require_replay_ready=True,
    )

    assert states == []
    assert counts["filtered_replay_ineligible"] == 1
    assert counts["written"] == 0


def test_convert_can_recover_visible_observation_from_versioned_replay_truth():
    row = _decision_row(include_visible_cards=False)
    visible = (*_opponent_board().all_cards(), "2c", "5c")
    truth = ReplayTruth(
        true_dead_cards=("2c", "5c", "3c", "4c", "6c"),
        visible_dead_cards=visible,
        hero_private_discards=("2c", "5c"),
        opponent_private_discards=("3c", "4c", "6c"),
    )
    row.update(
        {
            "visibility_model": "hidden_discard",
            "replay_truth": truth.to_dict(),
            "true_dead_cards": ["2c", "5c", "3c", "4c", "6c"],
        }
    )

    states, counts = convert_rows([row], source_label="legacy_attached")

    assert counts["replay_ready"] == 1
    assert states[0]["visible_dead_cards"] == list(visible)
    assert states[0]["replay_truth_available"] is True


def test_convert_fails_closed_on_conflicting_replay_truth():
    row = _decision_row()
    truth = ReplayTruth(
        true_dead_cards=("2c", "5c", "3c", "4c", "6c"),
        visible_dead_cards=(*_opponent_board().all_cards(), "2c", "5c"),
        hero_private_discards=("2c", "5c"),
        opponent_private_discards=("3c", "4c", "6c"),
    )
    row["replay_truth"] = truth.to_dict()
    row["true_dead_cards"] = ["2c", "5c", "3c", "4c", "9h"]

    states, counts = convert_rows([row], source_label="corrupt")

    assert counts["replay_ready"] == 0
    assert states[0]["replay_ready"] is False
    assert "disagrees with replay_truth" in states[0]["observation_error"]
