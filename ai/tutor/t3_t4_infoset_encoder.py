"""Lossless, opt-in fixed-schema encoding for late-HU ``InfoSetKey`` values.

This module is a distillation data contract.  It deliberately does not replace
``ai.engine.encoding.encode_state`` and is not selected by any serving route.

The vector is a fixed-width binary ``float32`` representation of every field in
an ``InfoSetKey``.  Public actions retain their absolute T0..T4 BB/BTN slots,
private recall retains its turn slots, and X1/X2 retain distinct physical-card
slots.  A redundant 27-action legal mask is checked on decode so a dataset row
cannot silently combine one observation with another observation's targets.

No seat- or suit-equivariance is claimed.  BB/BTN, the four physical suits, X1,
and X2 are encoded as absolute identities.  Symmetry augmentation, if desired,
must happen after a leakage-safe dataset split and must create a newly encoded
row with a separately recomputed legal-action binding.
"""
from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ai.engine.action_space import (
    POSITIONS,
    REGULAR_TURN_ACTIONS,
    create_regular_turn_mask,
    get_action_from_semantic_index_if_valid,
)
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_public_cfr import (
    FORBIDDEN_INFOSET_FIELDS,
    InfoSetKey,
    PHASES,
    PrivateRecall,
    ROWS,
)
from ai.tutor.runtime_semantic_anchor import register_module_function_anchor


INFOSET_ENCODER_SCHEMA = "ofc_t3_t4_infoset_fixed_binary/v1"
ACTION_SEMANTICS_SCHEMA = "ofc_regular_turn_semantic_actions/v1"
INFOSET_ENCODER_DTYPE = np.dtype(np.float32)
FANTASY_STATE_MAX_UTF8_BYTES = 64
HISTORY_SLOTS = tuple(
    (turn, actor)
    for turn in range(5)
    for actor in ("bb", "btn")
)
ACTORS = ("bb", "btn")
TURNS = tuple(range(5))
PHASE_VOCAB = tuple(PHASES)
CARD_VOCAB = tuple(ALL_CARDS)
ROW_VOCAB = tuple(ROWS)

_CARD_TO_INDEX = {card: index for index, card in enumerate(CARD_VOCAB)}
_ROW_TO_INDEX = {row: index for index, row in enumerate(ROW_VOCAB)}
_HISTORY_TO_INDEX = {slot: index for index, slot in enumerate(HISTORY_SLOTS)}
_MANIFEST_HASH_WIDTH = 256
_FANTASY_LENGTH_WIDTH = FANTASY_STATE_MAX_UTF8_BYTES + 1
_FANTASY_BYTES_WIDTH = FANTASY_STATE_MAX_UTF8_BYTES * 8
_BOARD_WIDTH = len(ROW_VOCAB) * len(CARD_VOCAB)
_HISTORY_SLOT_WIDTH = 1 + _BOARD_WIDTH
_HISTORY_WIDTH = len(HISTORY_SLOTS) * _HISTORY_SLOT_WIDTH
_RECALL_WIDTH = 4 * len(CARD_VOCAB)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


_ACTION_SEMANTICS_PAYLOAD: dict[str, Any] = {
    "schema": ACTION_SEMANTICS_SCHEMA,
    "action_count": REGULAR_TURN_ACTIONS,
    "dealt_card_order": "python_unicode_lexicographic_ascending",
    "row_order": list(POSITIONS),
    "index_formula": "discard_slot*9+remaining_card_0_row*3+remaining_card_1_row",
    "semantic_slots": [
        {
            "index": index,
            "discard_slot": index // 9,
            "remaining_card_0_row": POSITIONS[(index % 9) // 3],
            "remaining_card_1_row": POSITIONS[index % 3],
        }
        for index in range(REGULAR_TURN_ACTIONS)
    ],
    "legality": "row_capacity_after_both_placements",
    "action_id": {
        "format": "canonical_json",
        "placement_order": "top_then_middle_then_bottom_cards_lexicographic",
        "fields": ["discard", "placements"],
    },
}
ACTION_SEMANTICS_SHA256 = _sha256(_ACTION_SEMANTICS_PAYLOAD)


_SECTION_WIDTHS = (
    ("manifest_sha256_bits", _MANIFEST_HASH_WIDTH),
    ("contract_version", 1),
    ("actor", len(ACTORS)),
    ("turn_t0_t4", len(TURNS)),
    ("phase", len(PHASE_VOCAB)),
    ("fantasy_is_none", 1),
    ("fantasy_utf8_length", _FANTASY_LENGTH_WIDTH),
    ("fantasy_utf8_bytes", _FANTASY_BYTES_WIDTH),
    ("board_bb", _BOARD_WIDTH),
    ("board_btn", _BOARD_WIDTH),
    ("public_action_history", _HISTORY_WIDTH),
    ("own_dealt_by_turn", _RECALL_WIDTH),
    ("own_discards_by_turn", _RECALL_WIDTH),
    ("current_draw", len(CARD_VOCAB)),
    ("legal_action_mask", REGULAR_TURN_ACTIONS),
)


def _build_sections() -> dict[str, dict[str, int]]:
    offset = 0
    sections: dict[str, dict[str, int]] = {}
    for name, width in _SECTION_WIDTHS:
        sections[name] = {"offset": offset, "width": width}
        offset += width
    return sections


_INFOSET_ENCODER_SECTIONS = _build_sections()
INFOSET_ENCODER_SECTIONS: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        name: MappingProxyType(dict(spec))
        for name, spec in _INFOSET_ENCODER_SECTIONS.items()
    }
)
INFOSET_VECTOR_DIM = sum(width for _name, width in _SECTION_WIDTHS)

_INFOSET_ENCODER_MANIFEST_PAYLOAD: dict[str, Any] = {
    "schema": INFOSET_ENCODER_SCHEMA,
    "scope": "opt_in_t3_t4_hu_distillation_only",
    "serving_default": False,
    "lossless_for_accepted_inputs": True,
    "vector": {
        "dimension": INFOSET_VECTOR_DIM,
        "dtype": "float32",
        "allowed_values": [0, 1],
        "sections": {
            name: dict(spec)
            for name, spec in _INFOSET_ENCODER_SECTIONS.items()
        },
        "manifest_hash_embedding": "sha256_msb_first_256_bits",
    },
    "position_contract_version": POSITION_CONTRACT_VERSION,
    "fixed_layout": {
        "turns": list(TURNS),
        "actors": list(ACTORS),
        "phases": list(PHASE_VOCAB),
        "history_slots": [
            {"turn": turn, "actor": actor}
            for turn, actor in HISTORY_SLOTS
        ],
        "rows": list(ROW_VOCAB),
        "card_vocab": list(CARD_VOCAB),
        "physical_joker_ids": ["X1", "X2"],
        "fantasy_state": {
            "encoding": "utf8_msb_first_bits",
            "max_bytes": FANTASY_STATE_MAX_UTF8_BYTES,
            "none_distinct_from_empty_string": True,
        },
    },
    "field_contract": {
        "boards": "absolute_actor_then_row_then_physical_card_multihot",
        "public_action_history": "absolute_T0_T4_actor_slot_then_row_then_physical_card",
        "own_recall": "absolute_turn_then_physical_card",
        "current_draw": "physical_card_multihot",
        "forbidden_hidden_fields": sorted(FORBIDDEN_INFOSET_FIELDS),
    },
    "action_semantics": _ACTION_SEMANTICS_PAYLOAD,
    "action_semantics_sha256": ACTION_SEMANTICS_SHA256,
    "equivariance": {
        "seat": False,
        "suit": False,
        "joker_exchange": False,
        "contract": "absolute_identity_only",
    },
    "truth_boundaries": {
        "strategic_strength_claimed": False,
        "equilibrium_claimed": False,
        "global_policy_claimed": False,
        "encoder_correctness_only": True,
    },
}
INFOSET_ENCODER_MANIFEST_SHA256 = _sha256(_INFOSET_ENCODER_MANIFEST_PAYLOAD)


def infoset_encoder_manifest() -> dict[str, Any]:
    """Return a detached canonical manifest including its content hash."""
    manifest = json.loads(_canonical_json(_INFOSET_ENCODER_MANIFEST_PAYLOAD))
    manifest["manifest_sha256"] = INFOSET_ENCODER_MANIFEST_SHA256
    return manifest


def _section(name: str) -> slice:
    spec = _INFOSET_ENCODER_SECTIONS[name]
    return slice(spec["offset"], spec["offset"] + spec["width"])


def _sha256_bits(digest: str) -> np.ndarray:
    return np.asarray(
        [
            (byte >> shift) & 1
            for byte in bytes.fromhex(digest)
            for shift in range(7, -1, -1)
        ],
        dtype=np.float32,
    )


_MANIFEST_HASH_BITS = _sha256_bits(INFOSET_ENCODER_MANIFEST_SHA256)


def _one_hot(width: int, index: int) -> np.ndarray:
    if not 0 <= index < width:
        raise ValueError(f"one-hot index {index} is outside width {width}")
    result = np.zeros(width, dtype=np.float32)
    result[index] = 1.0
    return result


def _board_for_actor(key: InfoSetKey) -> Board:
    rows = key.board_bb if key.actor == "bb" else key.board_btn
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def legal_action_mask(key: InfoSetKey) -> np.ndarray:
    """Return the bound 27-slot legality mask for ``key``."""
    if not isinstance(key, InfoSetKey):
        raise TypeError("key must be an InfoSetKey")
    mask = np.asarray(
        create_regular_turn_mask(list(key.current_draw), _board_for_actor(key)),
        dtype=np.bool_,
    )
    if mask.shape != (REGULAR_TURN_ACTIONS,):
        raise AssertionError("regular-turn action mask has an unexpected shape")
    return mask


def _canonical_action_id(action: Any) -> str:
    by_row: dict[str, list[str]] = {row: [] for row in ROW_VOCAB}
    for card, row in action.placements:
        if row not in by_row:
            raise ValueError(f"action contains unsupported row {row!r}")
        by_row[row].append(str(card))
    payload = {
        "placements": [
            [card, row]
            for row in ROW_VOCAB
            for card in sorted(by_row[row])
        ],
        "discard": None if action.discard is None else str(action.discard),
    }
    return _canonical_json(payload)


def semantic_action_ids(key: InfoSetKey) -> tuple[str | None, ...]:
    """Bind each semantic index to its legal canonical action ID, or ``None``."""
    if not isinstance(key, InfoSetKey):
        raise TypeError("key must be an InfoSetKey")
    board = _board_for_actor(key)
    action_ids: list[str | None] = []
    for index in range(REGULAR_TURN_ACTIONS):
        action = get_action_from_semantic_index_if_valid(
            index,
            list(key.current_draw),
            board,
        )
        action_ids.append(None if action is None else _canonical_action_id(action))
    if tuple(item is not None for item in action_ids) != tuple(legal_action_mask(key)):
        raise AssertionError("semantic action IDs do not match the legal action mask")
    return tuple(action_ids)


def _encode_fantasy_state(value: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    is_none = np.asarray([1.0 if value is None else 0.0], dtype=np.float32)
    if value is None:
        encoded = b""
    elif isinstance(value, str):
        try:
            encoded = value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("fantasy_state must be valid UTF-8 text") from exc
    else:  # Defensive: InfoSetKey currently canonicalizes non-None values to str.
        raise TypeError("fantasy_state must be a string or None")
    if len(encoded) > FANTASY_STATE_MAX_UTF8_BYTES:
        raise ValueError(
            "fantasy_state UTF-8 encoding exceeds fixed schema capacity "
            f"{FANTASY_STATE_MAX_UTF8_BYTES}"
        )
    length = _one_hot(_FANTASY_LENGTH_WIDTH, len(encoded))
    byte_bits = np.zeros(_FANTASY_BYTES_WIDTH, dtype=np.float32)
    for byte_index, byte in enumerate(encoded):
        for bit_index, shift in enumerate(range(7, -1, -1)):
            byte_bits[byte_index * 8 + bit_index] = (byte >> shift) & 1
    return is_none, length, byte_bits


def _write_board(vector: np.ndarray, section_name: str, board: Sequence[Sequence[str]]) -> None:
    target = vector[_section(section_name)]
    for row_index, cards in enumerate(board):
        for card in cards:
            try:
                card_index = _CARD_TO_INDEX[card]
            except KeyError as exc:  # InfoSetKey normally rejects this first.
                raise ValueError(f"unknown physical card {card!r}") from exc
            target[row_index * len(CARD_VOCAB) + card_index] = 1.0


def encode_infoset_key(key: InfoSetKey) -> np.ndarray:
    """Encode one key as a 3,313-dimensional binary ``float32`` vector."""
    if not isinstance(key, InfoSetKey):
        raise TypeError("key must be an InfoSetKey")
    # Exercise the information-safe canonical serializer before producing data.
    key.canonical_json()
    if key.contract_version != POSITION_CONTRACT_VERSION:
        raise ValueError("unsupported position contract")

    vector = np.zeros(INFOSET_VECTOR_DIM, dtype=np.float32)
    vector[_section("manifest_sha256_bits")] = _MANIFEST_HASH_BITS
    vector[_section("contract_version")] = 1.0
    vector[_section("actor")] = _one_hot(len(ACTORS), ACTORS.index(key.actor))
    vector[_section("turn_t0_t4")] = _one_hot(len(TURNS), TURNS.index(key.turn))
    vector[_section("phase")] = _one_hot(len(PHASE_VOCAB), PHASE_VOCAB.index(key.phase))

    fantasy_none, fantasy_length, fantasy_bytes = _encode_fantasy_state(key.fantasy_state)
    vector[_section("fantasy_is_none")] = fantasy_none
    vector[_section("fantasy_utf8_length")] = fantasy_length
    vector[_section("fantasy_utf8_bytes")] = fantasy_bytes

    _write_board(vector, "board_bb", key.board_bb)
    _write_board(vector, "board_btn", key.board_btn)

    history = vector[_section("public_action_history")]
    for turn, actor, placements in key.public_action_history:
        try:
            slot_index = _HISTORY_TO_INDEX[(turn, actor)]
        except KeyError as exc:
            raise ValueError(f"public history slot T{turn} {actor} exceeds T0-T4 schema") from exc
        slot_offset = slot_index * _HISTORY_SLOT_WIDTH
        if history[slot_offset] != 0:
            raise ValueError(f"public history repeats T{turn} {actor}")
        history[slot_offset] = 1.0
        for card, row in placements:
            card_index = _CARD_TO_INDEX[card]
            row_index = _ROW_TO_INDEX[row]
            history[
                slot_offset + 1 + row_index * len(CARD_VOCAB) + card_index
            ] = 1.0

    dealt = vector[_section("own_dealt_by_turn")]
    for turn, cards in key.own_recall.dealt_by_turn:
        if not 1 <= turn <= 4:
            raise ValueError(f"private recall turn {turn} exceeds fixed T1-T4 schema")
        turn_offset = (turn - 1) * len(CARD_VOCAB)
        for card in cards:
            dealt[turn_offset + _CARD_TO_INDEX[card]] = 1.0

    discards = vector[_section("own_discards_by_turn")]
    for turn, card in key.own_recall.discards_by_turn:
        if not 1 <= turn <= 4:
            raise ValueError(f"private discard turn {turn} exceeds fixed T1-T4 schema")
        turn_offset = (turn - 1) * len(CARD_VOCAB)
        discards[turn_offset + _CARD_TO_INDEX[card]] = 1.0

    current_draw = vector[_section("current_draw")]
    for card in key.current_draw:
        current_draw[_CARD_TO_INDEX[card]] = 1.0

    vector[_section("legal_action_mask")] = legal_action_mask(key).astype(np.float32)
    if not np.all((vector == 0.0) | (vector == 1.0)):
        raise AssertionError("encoder emitted a non-binary value")
    return vector


def _strict_binary_vector(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("encoded infoset must be a numpy.ndarray")
    if value.dtype != INFOSET_ENCODER_DTYPE:
        raise TypeError("encoded infoset dtype must be float32")
    if value.ndim != 1 or value.shape != (INFOSET_VECTOR_DIM,):
        raise ValueError(
            f"encoded infoset shape must be ({INFOSET_VECTOR_DIM},), got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("encoded infoset contains a non-finite value")
    if not np.all((value == 0.0) | (value == 1.0)):
        raise ValueError("encoded infoset must contain only exact binary values")
    return value


def _one_hot_index(values: np.ndarray, *, label: str) -> int:
    indices = np.flatnonzero(values)
    if len(indices) != 1:
        raise ValueError(f"{label} must contain exactly one active bit")
    return int(indices[0])


def _decode_fantasy_state(vector: np.ndarray) -> str | None:
    is_none = bool(vector[_section("fantasy_is_none")][0])
    length = _one_hot_index(
        vector[_section("fantasy_utf8_length")],
        label="fantasy UTF-8 length",
    )
    raw_bits = vector[_section("fantasy_utf8_bytes")]
    if is_none:
        if length != 0 or np.any(raw_bits):
            raise ValueError("None fantasy_state must have zero length and zero payload")
        return None
    used_width = length * 8
    if np.any(raw_bits[used_width:]):
        raise ValueError("fantasy_state has nonzero padding bits")
    encoded = bytearray()
    for byte_index in range(length):
        byte = 0
        for bit in raw_bits[byte_index * 8 : (byte_index + 1) * 8]:
            byte = (byte << 1) | int(bit)
        encoded.append(byte)
    try:
        decoded = bytes(encoded).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("fantasy_state payload is not valid UTF-8") from exc
    if decoded.encode("utf-8") != bytes(encoded):
        raise ValueError("fantasy_state payload is not canonical UTF-8")
    return decoded


def _decode_board(vector: np.ndarray, section_name: str) -> tuple[tuple[str, ...], ...]:
    source = vector[_section(section_name)]
    rows: list[tuple[str, ...]] = []
    for row_index in range(len(ROW_VOCAB)):
        start = row_index * len(CARD_VOCAB)
        active = np.flatnonzero(source[start : start + len(CARD_VOCAB)])
        rows.append(tuple(sorted(CARD_VOCAB[int(index)] for index in active)))
    return tuple(rows)


def _decode_history(vector: np.ndarray) -> tuple[tuple[int, str, tuple[tuple[str, str], ...]], ...]:
    source = vector[_section("public_action_history")]
    history = []
    for slot_index, (turn, actor) in enumerate(HISTORY_SLOTS):
        slot_offset = slot_index * _HISTORY_SLOT_WIDTH
        present = bool(source[slot_offset])
        placements_bits = source[
            slot_offset + 1 : slot_offset + _HISTORY_SLOT_WIDTH
        ]
        if not present:
            if np.any(placements_bits):
                raise ValueError(f"absent public history slot T{turn} {actor} has payload bits")
            continue
        placements: list[tuple[str, str]] = []
        for row_index, row in enumerate(ROW_VOCAB):
            start = row_index * len(CARD_VOCAB)
            active = np.flatnonzero(
                placements_bits[start : start + len(CARD_VOCAB)]
            )
            placements.extend(
                (CARD_VOCAB[int(card_index)], row)
                for card_index in active
            )
        history.append((turn, actor, tuple(placements)))
    return tuple(history)


def _decode_recall(vector: np.ndarray) -> PrivateRecall:
    dealt_source = vector[_section("own_dealt_by_turn")]
    discard_source = vector[_section("own_discards_by_turn")]
    dealt: list[tuple[int, tuple[str, ...]]] = []
    discards: list[tuple[int, str]] = []
    for turn in range(1, 5):
        start = (turn - 1) * len(CARD_VOCAB)
        dealt_indices = np.flatnonzero(
            dealt_source[start : start + len(CARD_VOCAB)]
        )
        discard_indices = np.flatnonzero(
            discard_source[start : start + len(CARD_VOCAB)]
        )
        if len(discard_indices) > 1:
            raise ValueError(f"private recall T{turn} has multiple discards")
        if len(dealt_indices):
            dealt.append(
                (
                    turn,
                    tuple(sorted(CARD_VOCAB[int(index)] for index in dealt_indices)),
                )
            )
        if len(discard_indices):
            discards.append((turn, CARD_VOCAB[int(discard_indices[0])]))
    return PrivateRecall(tuple(dealt), tuple(discards))


def decode_infoset_key(vector: np.ndarray) -> InfoSetKey:
    """Strictly decode and canonically re-encode one fixed-schema vector."""
    source = _strict_binary_vector(vector)
    if not np.array_equal(source[_section("manifest_sha256_bits")], _MANIFEST_HASH_BITS):
        raise ValueError("encoded infoset manifest SHA-256 header mismatch")
    if not np.array_equal(
        source[_section("contract_version")],
        np.asarray([1.0], dtype=np.float32),
    ):
        raise ValueError("encoded infoset position contract bit is invalid")

    actor = ACTORS[
        _one_hot_index(source[_section("actor")], label="actor")
    ]
    turn = TURNS[
        _one_hot_index(source[_section("turn_t0_t4")], label="turn")
    ]
    phase = PHASE_VOCAB[
        _one_hot_index(source[_section("phase")], label="phase")
    ]
    fantasy_state = _decode_fantasy_state(source)
    board_bb = _decode_board(source, "board_bb")
    board_btn = _decode_board(source, "board_btn")
    public_action_history = _decode_history(source)
    own_recall = _decode_recall(source)
    draw_indices = np.flatnonzero(source[_section("current_draw")])
    current_draw = tuple(sorted(CARD_VOCAB[int(index)] for index in draw_indices))

    key = InfoSetKey(
        contract_version=POSITION_CONTRACT_VERSION,
        actor=actor,
        turn=turn,
        phase=phase,
        board_bb=board_bb,
        board_btn=board_btn,
        public_action_history=public_action_history,
        own_recall=own_recall,
        current_draw=current_draw,
        fantasy_state=fantasy_state,
    )
    expected_mask = legal_action_mask(key).astype(np.float32)
    if not np.array_equal(source[_section("legal_action_mask")], expected_mask):
        raise ValueError("encoded infoset legal action mask does not match its observation")
    canonical = encode_infoset_key(key)
    if not np.array_equal(source, canonical):
        raise ValueError("encoded infoset is not in canonical fixed-schema form")
    return key


def validate_infoset_encoder_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed unless ``manifest`` is exactly this encoder contract."""
    if not isinstance(manifest, Mapping):
        raise TypeError("encoder manifest must be a mapping")
    supplied = dict(manifest)
    supplied_hash = supplied.pop("manifest_sha256", None)
    if supplied_hash != INFOSET_ENCODER_MANIFEST_SHA256:
        raise ValueError("encoder manifest SHA-256 field mismatch")
    if supplied != _INFOSET_ENCODER_MANIFEST_PAYLOAD:
        raise ValueError("encoder manifest payload mismatch")
    if _sha256(supplied) != supplied_hash:
        raise ValueError("encoder manifest content SHA-256 mismatch")


_INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR = register_module_function_anchor(
    __name__, globals()
)
_INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR_MIRROR = (
    _INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR
)


__all__ = [
    "ACTION_SEMANTICS_SCHEMA",
    "ACTION_SEMANTICS_SHA256",
    "FANTASY_STATE_MAX_UTF8_BYTES",
    "HISTORY_SLOTS",
    "INFOSET_ENCODER_DTYPE",
    "INFOSET_ENCODER_MANIFEST_SHA256",
    "INFOSET_ENCODER_SCHEMA",
    "INFOSET_ENCODER_SECTIONS",
    "INFOSET_VECTOR_DIM",
    "decode_infoset_key",
    "encode_infoset_key",
    "infoset_encoder_manifest",
    "legal_action_mask",
    "semantic_action_ids",
    "validate_infoset_encoder_manifest",
]
