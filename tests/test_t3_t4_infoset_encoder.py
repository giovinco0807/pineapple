import hashlib
import json

import numpy as np
import pytest

from ai.engine.action_space import (
    REGULAR_TURN_ACTIONS,
    get_action_from_semantic_index_if_valid,
    get_semantic_action_index,
)
from ai.engine.encoding import ALL_CARDS, Board, Observation, encode_state
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.exact_late import action_key
from ai.tutor.t3_hu_public_cfr import InfoSetKey, PrivateRecall
from ai.tutor.t3_t4_infoset_encoder import (
    ACTION_SEMANTICS_SHA256,
    FANTASY_STATE_MAX_UTF8_BYTES,
    INFOSET_ENCODER_MANIFEST_SHA256,
    INFOSET_ENCODER_SCHEMA,
    INFOSET_ENCODER_SECTIONS,
    INFOSET_VECTOR_DIM,
    decode_infoset_key,
    encode_infoset_key,
    infoset_encoder_manifest,
    legal_action_mask,
    semantic_action_ids,
    validate_infoset_encoder_manifest,
)


BB_T0 = (
    ("4h", "top"),
    ("2c", "top"),
    ("3d", "top"),
    ("7c", "middle"),
    ("Qc", "bottom"),
)
BTN_T0 = (
    ("8c", "top"),
    ("5c", "top"),
    ("6d", "top"),
    ("9c", "middle"),
    ("As", "bottom"),
)
BB_T1 = (("7d", "middle"), ("8h", "middle"))
BTN_T1 = (("9d", "middle"), ("Jh", "middle"))
BB_T2 = (("9s", "middle"), ("Tc", "middle"))
BTN_T2 = (("Qs", "middle"), ("Kc", "middle"))
BB_T3 = (("Qd", "bottom"), ("X1", "bottom"))
BTN_T3 = (("2d", "bottom"), ("3h", "bottom"))
BB_T4 = (("Ad", "bottom"), ("Kd", "bottom"))

HISTORY_T3_FIRST = (
    (0, "bb", BB_T0),
    (0, "btn", BTN_T0),
    (1, "bb", BB_T1),
    (1, "btn", BTN_T1),
    (2, "bb", BB_T2),
    (2, "btn", BTN_T2),
)
HISTORY_T3_SECOND = HISTORY_T3_FIRST + ((3, "bb", BB_T3),)
HISTORY_T4_FIRST = HISTORY_T3_SECOND + ((3, "btn", BTN_T3),)
HISTORY_T4_SECOND = HISTORY_T4_FIRST + ((4, "bb", BB_T4),)

BB_BOARD_9 = (
    ("2c", "3d", "4h"),
    ("7c", "7d", "8h", "9s", "Tc"),
    ("Qc",),
)
BTN_BOARD_9 = (
    ("5c", "6d", "8c"),
    ("9c", "9d", "Jh", "Kc", "Qs"),
    ("As",),
)
BB_BOARD_11 = (BB_BOARD_9[0], BB_BOARD_9[1], ("Qc", "Qd", "X1"))
BTN_BOARD_11 = (BTN_BOARD_9[0], BTN_BOARD_9[1], ("2d", "3h", "As"))
BB_BOARD_13 = (
    BB_BOARD_9[0],
    BB_BOARD_9[1],
    ("Ad", "Kd", "Qc", "Qd", "X1"),
)


def _recall(actor: str, through_turn: int) -> PrivateRecall:
    public_cards = {
        "bb": {
            1: ("7d", "8h"),
            2: ("9s", "Tc"),
            3: ("Qd", "X1"),
        },
        "btn": {
            1: ("9d", "Jh"),
            2: ("Qs", "Kc"),
            3: ("2d", "3h"),
        },
    }[actor]
    discards = {
        "bb": {1: "6c", 2: "6s", 3: "X2"},
        "btn": {1: "4s", 2: "7h", 3: "4d"},
    }[actor]
    return PrivateRecall(
        dealt_by_turn=tuple(
            (turn, (*public_cards[turn], discards[turn]))
            for turn in range(1, through_turn + 1)
        ),
        discards_by_turn=tuple(
            (turn, discards[turn]) for turn in range(1, through_turn + 1)
        ),
    )


def _key(phase: str, *, fantasy_state: str | None = None) -> InfoSetKey:
    spec = {
        "t3_first": {
            "actor": "bb",
            "turn": 3,
            "board_bb": BB_BOARD_9,
            "board_btn": BTN_BOARD_9,
            "history": HISTORY_T3_FIRST,
            "recall": _recall("bb", 2),
            "draw": ("Qd", "X1", "X2"),
        },
        "t3_second": {
            "actor": "btn",
            "turn": 3,
            "board_bb": BB_BOARD_11,
            "board_btn": BTN_BOARD_9,
            "history": HISTORY_T3_SECOND,
            "recall": _recall("btn", 2),
            "draw": ("2d", "3h", "4d"),
        },
        "t4_first": {
            "actor": "bb",
            "turn": 4,
            "board_bb": BB_BOARD_11,
            "board_btn": BTN_BOARD_11,
            "history": HISTORY_T4_FIRST,
            "recall": _recall("bb", 3),
            "draw": ("Ad", "Jd", "Kd"),
        },
        "t4_second": {
            "actor": "btn",
            "turn": 4,
            "board_bb": BB_BOARD_13,
            "board_btn": BTN_BOARD_11,
            "history": HISTORY_T4_SECOND,
            "recall": _recall("btn", 3),
            "draw": ("Ah", "Qh", "Th"),
        },
    }[phase]
    return InfoSetKey(
        contract_version=POSITION_CONTRACT_VERSION,
        actor=spec["actor"],
        turn=spec["turn"],
        phase=phase,
        board_bb=spec["board_bb"],
        board_btn=spec["board_btn"],
        public_action_history=spec["history"],
        own_recall=spec["recall"],
        current_draw=spec["draw"],
        fantasy_state=fantasy_state,
    )


def _slice(name: str) -> slice:
    section = INFOSET_ENCODER_SECTIONS[name]
    return slice(section["offset"], section["offset"] + section["width"])


def _old_522_vector(key: InfoSetKey) -> np.ndarray:
    own_rows = key.board_bb if key.actor == "bb" else key.board_btn
    opponent_rows = key.board_btn if key.actor == "bb" else key.board_bb
    return encode_state(
        Observation(
            board_self=Board(
                top=list(own_rows[0]),
                middle=list(own_rows[1]),
                bottom=list(own_rows[2]),
            ),
            board_opponent=Board(
                top=list(opponent_rows[0]),
                middle=list(opponent_rows[1]),
                bottom=list(opponent_rows[2]),
            ),
            dealt_cards=list(key.current_draw),
            known_discards_self=[
                card for _turn, card in key.own_recall.discards_by_turn
            ],
            turn=key.turn,
            is_btn=key.actor == "btn",
        )
    )


@pytest.mark.parametrize(
    ("phase", "fantasy_state"),
    (
        ("t3_first", None),
        ("t3_second", "btn:継続"),
        ("t4_first", ""),
        ("t4_second", None),
    ),
)
def test_all_late_hu_phases_roundtrip_exact_canonical_information(
    phase: str,
    fantasy_state: str | None,
):
    key = _key(phase, fantasy_state=fantasy_state)

    encoded = encode_infoset_key(key)
    decoded = decode_infoset_key(encoded)

    assert encoded.dtype == np.float32
    assert encoded.shape == (INFOSET_VECTOR_DIM,) == (3313,)
    assert set(np.unique(encoded)).issubset({0.0, 1.0})
    assert decoded == key
    assert decoded.canonical_json() == key.canonical_json()
    assert decoded.digest() == key.digest()
    assert np.array_equal(encode_infoset_key(decoded), encoded)


def test_manifest_is_content_bound_explicit_and_non_promotional():
    manifest = infoset_encoder_manifest()
    validate_infoset_encoder_manifest(manifest)
    payload = dict(manifest)
    supplied_hash = payload.pop("manifest_sha256")
    calculated_hash = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert manifest["schema"] == INFOSET_ENCODER_SCHEMA
    assert supplied_hash == calculated_hash == INFOSET_ENCODER_MANIFEST_SHA256
    assert manifest["vector"]["dimension"] == INFOSET_VECTOR_DIM == 3313
    assert manifest["action_semantics_sha256"] == ACTION_SEMANTICS_SHA256
    assert manifest["fixed_layout"]["physical_joker_ids"] == ["X1", "X2"]
    assert manifest["equivariance"] == {
        "seat": False,
        "suit": False,
        "joker_exchange": False,
        "contract": "absolute_identity_only",
    }
    assert manifest["serving_default"] is False
    assert manifest["truth_boundaries"] == {
        "strategic_strength_claimed": False,
        "equilibrium_claimed": False,
        "global_policy_claimed": False,
        "encoder_correctness_only": True,
    }

    tampered = json.loads(json.dumps(manifest))
    tampered["vector"]["dimension"] += 1
    with pytest.raises(ValueError, match="payload mismatch"):
        validate_infoset_encoder_manifest(tampered)


def test_x1_x2_have_distinct_absolute_physical_card_slots():
    both = _key("t3_first")
    x1_only = InfoSetKey(
        **{
            **both.__dict__,
            "current_draw": ("Kh", "Qd", "X1"),
        }
    )
    x2_only = InfoSetKey(
        **{
            **both.__dict__,
            "current_draw": ("Kh", "Qd", "X2"),
        }
    )
    encoded_x1 = encode_infoset_key(x1_only)
    encoded_x2 = encode_infoset_key(x2_only)
    draw_x1 = encoded_x1[_slice("current_draw")]
    draw_x2 = encoded_x2[_slice("current_draw")]

    assert draw_x1[ALL_CARDS.index("X1")] == 1
    assert draw_x1[ALL_CARDS.index("X2")] == 0
    assert draw_x2[ALL_CARDS.index("X1")] == 0
    assert draw_x2[ALL_CARDS.index("X2")] == 1
    assert not np.array_equal(encoded_x1, encoded_x2)
    assert decode_infoset_key(encoded_x1).current_draw == tuple(sorted(("Kh", "Qd", "X1")))
    assert decode_infoset_key(encoded_x2).current_draw == tuple(sorted(("Kh", "Qd", "X2")))


@pytest.mark.parametrize("phase", ("t3_first", "t3_second", "t4_first", "t4_second"))
def test_27_action_mask_and_semantic_action_ids_are_bound_to_each_key(phase: str):
    key = _key(phase)
    encoded = encode_infoset_key(key)
    mask = legal_action_mask(key)
    ids = semantic_action_ids(key)
    board_rows = key.board_bb if key.actor == "bb" else key.board_btn
    board = Board(
        top=list(board_rows[0]),
        middle=list(board_rows[1]),
        bottom=list(board_rows[2]),
    )

    assert len(mask) == len(ids) == REGULAR_TURN_ACTIONS == 27
    assert np.array_equal(encoded[_slice("legal_action_mask")], mask.astype(np.float32))
    assert tuple(item is not None for item in ids) == tuple(mask)
    for index in range(REGULAR_TURN_ACTIONS):
        action = get_action_from_semantic_index_if_valid(
            index,
            list(key.current_draw),
            board,
        )
        if action is None:
            assert ids[index] is None
        else:
            assert get_semantic_action_index(action, list(key.current_draw)) == index
            assert ids[index] == action_key(action)


def test_old_522_encoder_collides_when_opponent_public_action_order_changes():
    base = _key("t3_first")
    changed_history = (
        (0, "bb", BB_T0),
        (0, "btn", BTN_T0),
        (1, "bb", BB_T1),
        (1, "btn", BTN_T2),
        (2, "bb", BB_T2),
        (2, "btn", BTN_T1),
    )
    changed = InfoSetKey(
        contract_version=base.contract_version,
        actor=base.actor,
        turn=base.turn,
        phase=base.phase,
        board_bb=base.board_bb,
        board_btn=base.board_btn,
        public_action_history=changed_history,
        own_recall=base.own_recall,
        current_draw=base.current_draw,
        fantasy_state=base.fantasy_state,
    )

    assert base != changed
    assert np.array_equal(_old_522_vector(base), _old_522_vector(changed))
    assert not np.array_equal(encode_infoset_key(base), encode_infoset_key(changed))


def test_old_522_encoder_collides_when_own_recall_turn_order_changes():
    base = _key("t3_first")
    changed_history = (
        (0, "bb", BB_T0),
        (0, "btn", BTN_T0),
        (1, "bb", BB_T2),
        (1, "btn", BTN_T1),
        (2, "bb", BB_T1),
        (2, "btn", BTN_T2),
    )
    changed_recall = PrivateRecall(
        dealt_by_turn=(
            (1, ("9s", "Tc", "6s")),
            (2, ("7d", "8h", "6c")),
        ),
        discards_by_turn=((1, "6s"), (2, "6c")),
    )
    changed = InfoSetKey(
        contract_version=base.contract_version,
        actor=base.actor,
        turn=base.turn,
        phase=base.phase,
        board_bb=base.board_bb,
        board_btn=base.board_btn,
        public_action_history=changed_history,
        own_recall=changed_recall,
        current_draw=base.current_draw,
        fantasy_state=base.fantasy_state,
    )

    assert base != changed
    assert np.array_equal(_old_522_vector(base), _old_522_vector(changed))
    assert not np.array_equal(encode_infoset_key(base), encode_infoset_key(changed))


def _map_card(card: str, suit_map: dict[str, str]) -> str:
    return card if card.startswith("X") else f"{card[0]}{suit_map[card[1]]}"


def _map_key_suits(key: InfoSetKey, suit_map: dict[str, str]) -> InfoSetKey:
    return InfoSetKey(
        contract_version=key.contract_version,
        actor=key.actor,
        turn=key.turn,
        phase=key.phase,
        board_bb=tuple(
            tuple(_map_card(card, suit_map) for card in row)
            for row in key.board_bb
        ),
        board_btn=tuple(
            tuple(_map_card(card, suit_map) for card in row)
            for row in key.board_btn
        ),
        public_action_history=tuple(
            (
                turn,
                actor,
                tuple((_map_card(card, suit_map), row) for card, row in placements),
            )
            for turn, actor, placements in key.public_action_history
        ),
        own_recall=PrivateRecall(
            dealt_by_turn=tuple(
                (
                    turn,
                    tuple(_map_card(card, suit_map) for card in cards),
                )
                for turn, cards in key.own_recall.dealt_by_turn
            ),
            discards_by_turn=tuple(
                (turn, _map_card(card, suit_map))
                for turn, card in key.own_recall.discards_by_turn
            ),
        ),
        current_draw=tuple(_map_card(card, suit_map) for card in key.current_draw),
        fantasy_state=key.fantasy_state,
    )


def test_seat_and_suit_identities_are_deterministic_absolute_channels():
    base = _key("t3_first")
    suit_mapped = _map_key_suits(
        base,
        {"h": "d", "d": "h", "c": "s", "s": "c"},
    )
    by_turn_actor = {
        (turn, actor): placements
        for turn, actor, placements in base.public_action_history
    }
    seat_swapped_history = tuple(
        (
            turn,
            actor,
            by_turn_actor[(turn, "btn" if actor == "bb" else "bb")],
        )
        for turn in range(3)
        for actor in ("bb", "btn")
    )
    seat_swapped = InfoSetKey(
        contract_version=base.contract_version,
        actor="bb",
        turn=3,
        phase="t3_first",
        board_bb=base.board_btn,
        board_btn=base.board_bb,
        public_action_history=seat_swapped_history,
        own_recall=_recall("btn", 2),
        current_draw=base.current_draw,
        fantasy_state=base.fantasy_state,
    )

    encoded_base = encode_infoset_key(base)
    encoded_suit = encode_infoset_key(suit_mapped)
    encoded_seat = encode_infoset_key(seat_swapped)

    assert np.array_equal(encoded_base, encode_infoset_key(base))
    assert np.array_equal(encoded_suit, encode_infoset_key(suit_mapped))
    assert np.array_equal(encoded_seat, encode_infoset_key(seat_swapped))
    assert not np.array_equal(encoded_base, encoded_suit)
    assert not np.array_equal(encoded_base, encoded_seat)
    assert decode_infoset_key(encoded_suit) == suit_mapped
    assert decode_infoset_key(encoded_seat) == seat_swapped
    assert infoset_encoder_manifest()["equivariance"]["suit"] is False
    assert infoset_encoder_manifest()["equivariance"]["seat"] is False


def test_encoder_rejects_fantasy_state_overflow_and_invalid_utf8_text():
    overflow = _key(
        "t3_first",
        fantasy_state="a" * (FANTASY_STATE_MAX_UTF8_BYTES + 1),
    )
    with pytest.raises(ValueError, match="exceeds fixed schema capacity"):
        encode_infoset_key(overflow)

    invalid_unicode = _key("t3_first", fantasy_state="\ud800")
    with pytest.raises(ValueError, match="valid UTF-8"):
        encode_infoset_key(invalid_unicode)


def test_decoder_rejects_wrong_type_dtype_shape_and_nonbinary_values():
    encoded = encode_infoset_key(_key("t3_first"))
    with pytest.raises(TypeError, match="numpy.ndarray"):
        decode_infoset_key(encoded.tolist())
    with pytest.raises(TypeError, match="float32"):
        decode_infoset_key(encoded.astype(np.float64))
    with pytest.raises(ValueError, match="shape"):
        decode_infoset_key(encoded[:-1])

    nonbinary = encoded.copy()
    nonbinary[0] = 0.5
    with pytest.raises(ValueError, match="exact binary"):
        decode_infoset_key(nonbinary)

    nonfinite = encoded.copy()
    nonfinite[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        decode_infoset_key(nonfinite)


def test_decoder_rejects_wrong_schema_onehots_padding_and_action_binding():
    encoded = encode_infoset_key(_key("t3_first"))

    wrong_schema = encoded.copy()
    header = wrong_schema[_slice("manifest_sha256_bits")]
    header[0] = 1.0 - header[0]
    with pytest.raises(ValueError, match="manifest SHA-256 header mismatch"):
        decode_infoset_key(wrong_schema)

    bad_actor = encoded.copy()
    bad_actor[_slice("actor")] = 1.0
    with pytest.raises(ValueError, match="actor must contain exactly one"):
        decode_infoset_key(bad_actor)

    bad_none_padding = encoded.copy()
    bad_none_padding[_slice("fantasy_utf8_bytes")][0] = 1.0
    with pytest.raises(ValueError, match="None fantasy_state"):
        decode_infoset_key(bad_none_padding)

    bad_mask = encoded.copy()
    mask = bad_mask[_slice("legal_action_mask")]
    mask[0] = 1.0 - mask[0]
    with pytest.raises(ValueError, match="legal action mask"):
        decode_infoset_key(bad_mask)

    future_history = encoded.copy()
    history = future_history[_slice("public_action_history")]
    t4_btn_slot_offset = 9 * (1 + 3 * len(ALL_CARDS))
    history[t4_btn_slot_offset] = 1.0
    history[t4_btn_slot_offset + 1] = 1.0
    with pytest.raises(ValueError, match="history through T2 btn"):
        decode_infoset_key(future_history)
