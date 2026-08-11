from copy import deepcopy
from itertools import permutations

import pytest

from ofc_regular.action_key import ActionKey
from ofc_regular.action_space import Action
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_rl_contract import (
    HU_RL_ACTOR_VIEW_SCHEMA,
    HuRlActorViewV1,
    HuRlContractError,
    LegalActionMappingV1,
    PublicPlacement,
)
from ofc_regular.state import Board


def _observation(*, permuted: bool = False) -> ActorObservation:
    hero = Board.from_rows(
        top=("3h", "2h") if permuted else ("2h", "3h"),
        middle=("5h", "4h") if permuted else ("4h", "5h"),
        bottom=("8h", "7h", "6h") if permuted else ("6h", "7h", "8h"),
    )
    opponent = Board.from_rows(
        top=("3d", "2d") if permuted else ("2d", "3d"),
        middle=("5d", "4d") if permuted else ("4d", "5d"),
        bottom=("8d", "7d", "6d") if permuted else ("6d", "7d", "8d"),
    )
    return ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=("Qh", "Jh", "Th") if permuted else ("Th", "Jh", "Qh"),
        hero_private_discards=("9h",),
        seat="first",
        street="T2",
        to_act_order="first",
    )


def _history(*, reverse_placements: bool = False) -> tuple[PublicPlacement, ...]:
    def placements(values):
        result = tuple(values)
        return tuple(reversed(result)) if reverse_placements else result

    return (
        PublicPlacement.from_action(
            street="T0",
            acting_seat="first",
            action=Action(
                placements=placements(
                    (
                        ("2h", "top"),
                        ("3h", "top"),
                        ("4h", "middle"),
                        ("5h", "middle"),
                        ("6h", "bottom"),
                    )
                )
            ),
        ),
        PublicPlacement.from_action(
            street="T0",
            acting_seat="second",
            action=Action(
                placements=placements(
                    (
                        ("2d", "top"),
                        ("3d", "top"),
                        ("4d", "middle"),
                        ("5d", "middle"),
                        ("6d", "bottom"),
                    )
                )
            ),
        ),
        PublicPlacement.from_action(
            street="T1",
            acting_seat="first",
            action=Action(
                placements=placements((("7h", "bottom"), ("8h", "bottom"))),
                discards=("Ac",),
            ),
        ),
        PublicPlacement.from_action(
            street="T1",
            acting_seat="second",
            action=Action(
                placements=placements((("7d", "bottom"), ("8d", "bottom"))),
                discards=("Ad",),
            ),
        ),
    )


def _view(*, permuted: bool = False) -> HuRlActorViewV1:
    return HuRlActorViewV1.from_observation(
        _observation(permuted=permuted),
        public_history=_history(reverse_placements=permuted),
    )


def test_public_placement_erases_discard_identity_and_keeps_semantics() -> None:
    first = PublicPlacement.from_action(
        street="T1",
        acting_seat="second",
        action=Action(
            placements=(("7d", "bottom"), ("8d", "bottom")),
            discards=("Ad",),
        ),
    )
    different_hidden_discard = PublicPlacement.from_action(
        street="T1",
        acting_seat="second",
        action=Action(
            placements=(("8d", "bottom"), ("7d", "bottom")),
            discards=("Kc",),
        ),
    )

    assert first == different_hidden_discard
    assert first.cards("bottom") == ("7d", "8d")
    assert first.discard_count == 1
    assert "discard_mask" not in first.__dict__
    assert set(first.to_dict()) == {
        "schema",
        "street",
        "acting_seat",
        "top_placement_mask",
        "middle_placement_mask",
        "bottom_placement_mask",
        "discard_count",
    }
    assert "Ad" not in first.canonical_json()
    assert "Kc" not in first.canonical_json()
    assert PublicPlacement.from_dict(first.to_dict()) == first


@pytest.mark.parametrize(
    "forbidden_field",
    ("discard_mask", "opponent_legal_digest", "audit_truth"),
)
def test_public_history_rejects_hidden_or_audit_fields(forbidden_field: str) -> None:
    payload = _view().to_dict()
    payload["public_history"][2][forbidden_field] = "must-not-enter-actor-view"

    with pytest.raises(HuRlContractError, match="forbidden hidden/audit fields"):
        HuRlActorViewV1.from_dict(payload)


def test_actor_view_rejects_hidden_truth_at_every_typed_boundary() -> None:
    payload = _view().to_dict()
    payload["observation"]["opponent_private_discards"] = ["Ac"]
    with pytest.raises(HuRlContractError, match="opponent_private_discards"):
        HuRlActorViewV1.from_dict(payload)

    payload = _view().to_dict()
    payload["audit_truth"] = {"deck_tail": ["Ac"]}
    with pytest.raises(HuRlContractError, match="unknown fields: audit_truth"):
        HuRlActorViewV1.from_dict(payload)


def test_canonical_actor_view_is_stable_under_card_and_action_permutation() -> None:
    expected = _view()
    permuted = _view(permuted=True)

    assert expected.canonical_json() == permuted.canonical_json()
    assert expected.digest() == permuted.digest()
    assert expected.legal_action_mapping == permuted.legal_action_mapping

    for dealt in permutations(_observation().dealt_cards):
        source = _observation()
        observation = ActorObservation(
            hero_board=source.hero_board,
            opponent_public_board=source.opponent_public_board,
            dealt_cards=dealt,
            hero_private_discards=source.hero_private_discards,
            seat=source.seat,
            street=source.street,
            to_act_order=source.to_act_order,
            scoring=source.scoring,
        )
        candidate = HuRlActorViewV1.from_observation(
            observation, public_history=_history()
        )
        assert candidate.digest() == expected.digest()


def test_legal_mapping_round_trip_and_tamper_detection_are_fail_closed() -> None:
    mapping = LegalActionMappingV1.for_observation(_observation())
    payload = mapping.to_dict()

    assert 0 < mapping.action_count <= 232
    assert LegalActionMappingV1.from_dict(payload) == mapping
    assert mapping.index_for(mapping.key_at(3)) == 3
    with pytest.raises(IndexError, match="outside the action mapping"):
        mapping.key_at(-1)
    with pytest.raises(IndexError, match="outside the action mapping"):
        mapping.key_at(mapping.action_count)

    wrong_digest = deepcopy(payload)
    wrong_digest["action_order_digest"] = "0" * 64
    with pytest.raises(HuRlContractError, match="order digest mismatch"):
        LegalActionMappingV1.from_dict(wrong_digest)

    duplicate = deepcopy(payload)
    duplicate["action_keys"][1] = duplicate["action_keys"][0]
    with pytest.raises(HuRlContractError, match="duplicate ActionKeys"):
        LegalActionMappingV1.from_dict(duplicate)

    reordered = LegalActionMappingV1(tuple(reversed(mapping.action_keys)))
    with pytest.raises(HuRlContractError, match="canonical legal actions"):
        HuRlActorViewV1(_observation(), _history(), reordered)


def test_public_history_must_be_past_chronological_and_match_public_boards() -> None:
    history = _history()

    with pytest.raises(HuRlContractError, match="chronological order"):
        HuRlActorViewV1.from_observation(
            _observation(), public_history=(history[1], history[0])
        )

    future = PublicPlacement.from_action(
        street="T2",
        acting_seat="first",
        action=Action(
            placements=(("Th", "top"), ("Jh", "middle")), discards=("Qh",)
        ),
    )
    with pytest.raises(HuRlContractError, match="current or a future action"):
        HuRlActorViewV1.from_observation(
            _observation(), public_history=(*history, future)
        )

    wrong_public_card = PublicPlacement.from_action(
        street="T1",
        acting_seat="second",
        action=Action(
            placements=(("7d", "bottom"), ("Kd", "bottom")), discards=("Ad",)
        ),
    )
    with pytest.raises(HuRlContractError, match="disagree with the public boards"):
        HuRlActorViewV1.from_observation(
            _observation(), public_history=(*history[:3], wrong_public_card)
        )

    wrong_row = PublicPlacement.from_action(
        street="T1",
        acting_seat="first",
        action=Action(
            placements=(("7h", "bottom"), ("8h", "middle")),
            discards=("Ac",),
        ),
    )
    with pytest.raises(HuRlContractError, match="placement rows disagree"):
        HuRlActorViewV1.from_observation(
            _observation(), public_history=(*history[:2], wrong_row, history[3])
        )

    with pytest.raises(HuRlContractError, match="complete trajectory prefix"):
        HuRlActorViewV1.from_observation(
            _observation(), public_history=history[:-1]
        )

    incomplete_board_history = (
        history[0],
        history[1],
        PublicPlacement.from_action(
            street="T1",
            acting_seat="first",
            action=Action(
                placements=(("7h", "bottom"), ("4h", "middle")),
                discards=("Ac",),
            ),
        ),
        history[3],
    )
    with pytest.raises(HuRlContractError, match="same card more than once"):
        HuRlActorViewV1.from_observation(
            _observation(), public_history=incomplete_board_history
        )


def test_actor_view_round_trip_preserves_schema_mapping_and_digest() -> None:
    view = _view()
    payload = view.to_dict()
    restored = HuRlActorViewV1.from_dict(payload)

    assert payload["schema"] == HU_RL_ACTOR_VIEW_SCHEMA
    assert restored == view
    assert restored.digest() == view.digest()
    assert "opponent_private_discard" not in view.canonical_json()
    assert "true_dead_cards" not in view.canonical_json()


def test_public_placement_geometry_and_mask_overlap_are_rejected() -> None:
    key = ActionKey(top_mask=1, middle_mask=2)
    with pytest.raises(HuRlContractError, match="requires 5 placed cards"):
        PublicPlacement.from_action_key(street="T0", acting_seat="first", key=key)

    with pytest.raises(HuRlContractError, match="pairwise disjoint"):
        PublicPlacement(
            street="T1",
            acting_seat="first",
            top_placement_mask=1,
            middle_placement_mask=1,
            bottom_placement_mask=0,
            discard_count=1,
        )
