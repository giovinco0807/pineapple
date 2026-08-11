from itertools import permutations
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ofc_regular.action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    generate_canonical_actions,
    legal_action_set_digest,
    ordered_action_mapping_digest,
    resolve_action_index,
    resolve_action_key,
)
from ofc_regular.action_space import Action, generate_actions
from ofc_regular.policy import RegularAiPolicy
from ofc_regular.state import Board


@pytest.mark.parametrize(
    ("board", "dealt"),
    [
        (Board.from_rows(), ("Ah", "Kd", "Qc", "Js", "Th")),
        (
            Board.from_rows(top=("2h",), middle=("3h", "4d"), bottom=("5c", "6s")),
            ("7h", "8d", "9c"),
        ),
        (
            Board.from_rows(top=("2h", "3d"), middle=("4c", "5s"), bottom=("6h", "7d", "8c")),
            ("9s", "Th", "Jd"),
        ),
        (
            Board.from_rows(top=("2h", "3d"), middle=("4c", "5s", "6h"), bottom=("7d", "8c", "9s", "Th")),
            ("Jd", "Qc", "Ks"),
        ),
        (
            Board.from_rows(top=("2h", "3d"), middle=("4c", "5s", "6h", "7d"), bottom=("8c", "9s", "Th", "Jd", "Qc")),
            ("Ks", "Ah", "2d"),
        ),
    ],
    ids=("t0", "t1", "t2", "t3", "t4"),
)
def test_canonical_action_order_is_invariant_to_dealt_card_permutation(board, dealt):
    expected = [action_key(action) for action in generate_canonical_actions(board, dealt)]
    expected_set_digest = legal_action_set_digest(generate_actions(board, dealt))

    for permuted in permutations(dealt):
        actions = generate_actions(board, permuted)
        assert legal_action_set_digest(actions) == expected_set_digest
        assert [action_key(action) for action in generate_canonical_actions(board, permuted)] == expected


def test_t0_canonical_generator_preserves_all_232_semantic_actions():
    actions = generate_canonical_actions(
        Board.from_rows(), ("Ah", "Kd", "Qc", "Js", "Th")
    )

    assert len(actions) == 232
    assert len({action_key(action) for action in actions}) == 232


def test_key_resolves_across_different_legacy_action_orders():
    board = Board.from_rows(
        top=("2h", "3d"),
        middle=("4c", "5s"),
        bottom=("6h", "7d", "8c"),
    )
    original = generate_actions(board, ("9s", "Th", "Jd"))
    permuted = generate_actions(board, ("Jd", "9s", "Th"))
    key = action_key(original[4])

    resolved = resolve_action_key(permuted, key)
    resolution = resolve_action_index(
        permuted, key=key.to_token(), legacy_index=4
    )

    assert action_key(permuted[resolved]) == key
    assert resolution.index == resolved
    assert resolution.source == "key_search"
    assert ordered_action_mapping_digest(original) != ordered_action_mapping_digest(permuted)
    assert legal_action_set_digest(original) == legal_action_set_digest(permuted)


def test_action_key_token_and_json_round_trip_uses_fixed_width_masks():
    action = Action(
        placements=(("Ah", "top"), ("2h", "top"), ("Kd", "middle")),
        discards=("Qs", "3c"),
    )
    key = action_key(action)
    token = key.to_token()

    assert token.startswith("rak1:")
    assert [len(part) for part in token.split(":")[1:]] == [13, 13, 13, 13]
    assert key.cards("top") == ("2h", "Ah")
    assert key.cards("middle") == ("Kd",)
    assert key.cards("discards") == ("3c", "Qs")
    assert ActionKey.from_token(token) == key
    assert ActionKey.from_dict({"schema": ACTION_KEY_SCHEMA, "token": token}) == key
    assert key.to_dict() == {"schema": ACTION_KEY_SCHEMA, "token": token}


def test_legacy_index_only_resolution_requires_matching_order_digest():
    board = Board.from_rows(top=("2h",), middle=("3h", "4d"), bottom=("5c", "6s"))
    actions = generate_actions(board, ("7h", "8d", "9c"))

    with pytest.raises(ValueError, match="requires an order digest"):
        resolve_action_index(actions, legacy_index=3)
    with pytest.raises(ValueError, match="order digest mismatch"):
        resolve_action_index(actions, legacy_index=3, expected_order_digest="0" * 64)

    resolution = resolve_action_index(
        actions,
        legacy_index=3,
        expected_order_digest=ordered_action_mapping_digest(actions),
    )
    assert resolution.index == 3
    assert resolution.key == action_key(actions[3])
    assert resolution.source == "verified_legacy_index"


def test_action_key_rejects_overlapping_masks_and_duplicate_semantic_actions():
    with pytest.raises(ValueError, match="pairwise disjoint"):
        ActionKey(top_mask=1, discard_mask=1)

    duplicate = Action(placements=(("Ah", "top"),), discards=("Kd",))
    with pytest.raises(ValueError, match="duplicate semantic action key"):
        legal_action_set_digest([duplicate, duplicate])


def test_canonical_action_output_digest_is_identical_across_processes():
    script = r'''
import hashlib, sys
sys.path.insert(0, "src")
from ofc_regular.action_key import action_key, generate_canonical_actions
from ofc_regular.state import Board
actions = generate_canonical_actions(
    Board.from_rows(top=("2h", "3d"), middle=("4c", "5s"), bottom=("6h", "7d", "8c")),
    ("9s", "Th", "Jd"),
)
print(hashlib.sha256("\n".join(action_key(action).to_token() for action in actions).encode("ascii")).hexdigest())
'''
    root = Path(__file__).resolve().parents[1]

    def run():
        return subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=root,
            text=True,
            encoding="utf-8",
        ).strip()

    assert run() == run()


@pytest.mark.parametrize(
    ("model_field", "board", "dealt"),
    (
        ("opening_model", Board.from_rows(), ("Ah", "Kd", "Qc", "Js", "Th")),
        (
            "turn1_model",
            Board.from_rows(top=("2h",), middle=("3h", "4d"), bottom=("5c", "6s")),
            ("7h", "8d", "9c"),
        ),
        (
            "turn2_model",
            Board.from_rows(
                top=("2h", "3d"),
                middle=("4c", "5s"),
                bottom=("6h", "7d", "8c"),
            ),
            ("9s", "Th", "Jd"),
        ),
        (
            "turn3_model",
            Board.from_rows(
                top=("2h", "3d"),
                middle=("4c", "5s", "6h"),
                bottom=("7d", "8c", "9s", "Th"),
            ),
            ("Jd", "Qc", "Ks"),
        ),
    ),
)
def test_runtime_fallback_ties_are_action_key_invariant(
    model_field, board, dealt
):
    class PositionalTieModel:
        def predict_sample(self, sample):
            return np.zeros(len(sample["actions"]), dtype=np.float64)

        def choose_action_index(self, _sample):
            return 0

    policy = RegularAiPolicy(**{model_field: PositionalTieModel()})
    selected = {
        action_key(policy.choose_action(board, permuted)).to_token()
        for permuted in permutations(dealt)
    }

    assert len(selected) == 1
