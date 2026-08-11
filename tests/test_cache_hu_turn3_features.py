import json

import numpy as np
import pytest

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cache_hu_turn3_features import _encode_chunk
from ofc_regular.hu_infoset import ActorObservation, InformationSetError
from ofc_regular.hu_turn3_model import hu_policy_sample
from ofc_regular.state import Board


def _safe_t3_sample(*, poisoned_dead: str) -> dict:
    hero = Board.from_rows(
        top=("Qh",),
        middle=("Kh", "Kd", "6c", "8s"),
        bottom=("9c", "9d", "9s", "Kc"),
    )
    opponent = Board.from_rows(
        top=("2h", "3h"),
        middle=("2d", "3d", "4d"),
        bottom=("7c", "8c", "Tc", "Jc"),
    )
    dealt = ("Qs", "Ah", "7d")
    hero_private = ("2c", "5c")
    observation = ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_private_discards=hero_private,
        seat="first",
        street="T3",
        to_act_order="first",
    )
    actions = generate_turn_actions(hero, dealt)
    sample = hu_policy_sample(
        hero,
        dealt,
        actions,
        opponent_board=opponent,
        dead_cards=observation.legacy_dead_cards(),
        seat="first",
        to_act_order="first",
    )
    for index, action in enumerate(sample["actions"]):
        action["score"] = float(index)
    sample.update(
        {
            "turn": "T3",
            "hero_private_discards": list(hero_private),
            "visible_dead_cards": list(observation.legacy_dead_cards()),
            "policy_observation": observation.to_dict(),
            "dead_cards": [*observation.legacy_dead_cards(), poisoned_dead],
        }
    )
    return sample


def test_t3_cache_features_ignore_offline_hidden_truth():
    first = _encode_chunk(
        0, [json.dumps(_safe_t3_sample(poisoned_dead="4c"))], "float32"
    )
    second = _encode_chunk(
        0, [json.dumps(_safe_t3_sample(poisoned_dead="5d"))], "float32"
    )

    assert np.array_equal(first["features"], second["features"])
    assert np.array_equal(first["targets"], second["targets"])


def test_t3_cache_rejects_legacy_unsplit_dead_cards():
    sample = _safe_t3_sample(poisoned_dead="4c")
    sample.pop("policy_observation")
    sample.pop("visible_dead_cards")
    sample.pop("hero_private_discards")

    with pytest.raises(InformationSetError, match="hero_private_discards"):
        _encode_chunk(0, [json.dumps(sample)], "float32")
