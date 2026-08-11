import pytest

from ofc_regular.hu_turn1_training_augmentation import augment_samples_by_suit, suit_mappings


def _sample() -> dict:
    return {
        "board": {"top": ["Ac"], "middle": ["Kd"], "bottom": ["Qh"]},
        "opponent_board": {"top": ["Js"], "middle": [], "bottom": []},
        "dealt": ["2c", "3d", "4h"],
        "dead_cards": ["5s"],
        "actions": [
            {
                "placements": [["2c", "top"], ["3d", "middle"]],
                "discards": ["4h"],
                "next_board": {"top": ["Ac", "2c"], "middle": ["Kd", "3d"], "bottom": ["Qh"]},
                "score": 1.25,
            }
        ],
    }


def test_suit_augmentation_preserves_scores_and_card_relationships():
    rows = augment_samples_by_suit([_sample()], count=1, seed=17)

    assert len(rows) == 2
    assert rows[0]["board"]["top"] == ["Ac"]
    assert rows[1]["actions"][0]["score"] == 1.25
    mapping = rows[1]["training_augmentation"]["mapping"]
    assert rows[1]["board"]["top"] == ["A" + mapping["c"]]
    assert rows[1]["dealt"][0] == "2" + mapping["c"]
    assert rows[1]["actions"][0]["next_board"]["top"][1] == "2" + mapping["c"]


def test_suit_mappings_are_deterministic_and_non_identity():
    first = suit_mappings(count=5, seed=99)
    second = suit_mappings(count=5, seed=99)

    assert first == second
    assert all(any(source != target for source, target in mapping.items()) for mapping in first)


def test_suit_augmentation_rejects_more_than_all_non_identity_permutations():
    with pytest.raises(ValueError, match="must be <= 23"):
        suit_mappings(count=24, seed=1)
