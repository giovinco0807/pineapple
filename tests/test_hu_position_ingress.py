import pytest

from ai.training.action_feature_encoding import board_context_feature_vector
from ai.training.convert_action_value_teacher import post_action_observation
from ai.tutor.hybrid_t1t2 import observation_from_payload


def _t1_payload(*, position: str = "btn", is_btn=...) -> dict:
    payload = {
        "turn": 1,
        "position": position,
        "board": {
            "top": ["Ah"],
            "middle": ["2h", "3h"],
            "bottom": ["4h", "5h"],
        },
        "opponent_board": {
            "top": ["Kh"],
            "middle": ["6h", "7h", "8h"],
            "bottom": ["9h", "Th", "Jh"],
        },
        "dealt": ["Qc", "Kd", "As"],
    }
    if is_btn is not ...:
        payload["is_btn"] = is_btn
    return payload


def _t2_feature_decision(*, position: str = "btn", is_btn=...) -> dict:
    decision = {
        "turn": 2,
        "position": position,
        "board": {
            "top": ["Ah"],
            "middle": ["2h", "3h", "4h"],
            "bottom": ["5h", "6h", "7h"],
        },
        "opponent_board": {
            "top": ["Kh", "Qh"],
            "middle": ["8h", "9h", "Th"],
            "bottom": ["Jh", "2c", "3c", "4c"],
        },
        "dealt": ["5c", "6c", "7c"],
    }
    if is_btn is not ...:
        decision["is_btn"] = is_btn
    return decision


def test_hybrid_position_only_canonicalizes_btn_to_second_actor() -> None:
    obs = observation_from_payload(_t1_payload())

    assert obs.is_btn is True
    assert len(obs.board_self.all_cards()) == 5
    assert len(obs.board_opponent.all_cards()) == 7


def test_hybrid_is_btn_only_preserves_bb_first_actor_compatibility() -> None:
    payload = _t1_payload()
    payload.pop("position")
    payload["is_btn"] = False
    payload["opponent_board"]["bottom"] = ["9h"]

    obs = observation_from_payload(payload)

    assert obs.is_btn is False
    assert len(obs.board_self.all_cards()) == 5
    assert len(obs.board_opponent.all_cards()) == 5


def test_hybrid_rejects_position_flag_contradiction_and_wrong_public_shape() -> None:
    with pytest.raises(ValueError, match="contradictory HU position"):
        observation_from_payload(_t1_payload(is_btn=False))

    invalid = _t1_payload()
    invalid["opponent_board"]["bottom"].pop()
    with pytest.raises(ValueError, match="requires hero/opponent board counts 5/7"):
        observation_from_payload(invalid)


def test_teacher_converter_uses_canonical_position_for_observation() -> None:
    record = _t1_payload()
    candidate = {
        "placements": [["Qc", "top"], ["Kd", "middle"]],
        "discard": "As",
    }

    obs = post_action_observation(record, candidate)

    assert obs.is_btn is True
    assert len(obs.board_self.all_cards()) == 7
    assert len(obs.board_opponent.all_cards()) == 7


def test_teacher_converter_rejects_contradictory_position_fields() -> None:
    with pytest.raises(ValueError, match="contradictory HU position"):
        post_action_observation(
            _t1_payload(is_btn=False),
            {"placements": [["Qc", "top"], ["Kd", "middle"]], "discard": "As"},
        )


def test_board_context_position_bit_uses_canonical_position() -> None:
    decision = _t2_feature_decision()
    candidate = {
        "placements": [["5c", "top"], ["6c", "middle"]],
        "discard": "7c",
    }

    features = board_context_feature_vector(decision, candidate)

    assert features[-1] == pytest.approx(1.0)


def test_board_context_bb_first_actor_uses_zero_position_bit() -> None:
    decision = _t2_feature_decision(position="bb")
    decision["opponent_board"]["bottom"] = ["Jh", "2c"]
    candidate = {
        "placements": [["5c", "top"], ["6c", "middle"]],
        "discard": "7c",
    }

    features = board_context_feature_vector(decision, candidate)

    assert features[-1] == pytest.approx(0.0)


def test_board_context_rejects_contradiction_and_ambiguous_role() -> None:
    candidate = {
        "placements": [["5c", "top"], ["6c", "middle"]],
        "discard": "7c",
    }
    with pytest.raises(ValueError, match="contradictory HU position"):
        board_context_feature_vector(_t2_feature_decision(is_btn=False), candidate)

    ambiguous = _t2_feature_decision()
    ambiguous.pop("position")
    with pytest.raises(ValueError, match="requires explicit position or is_btn"):
        board_context_feature_vector(ambiguous, candidate)
