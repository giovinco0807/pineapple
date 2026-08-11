"""Stable, card-order-independent identities for legal OFC actions.

Legacy action generators intentionally keep their existing enumeration order so
that trained models and saved artifacts are not reinterpreted. New search and
dataset code should persist :class:`ActionKey` and resolve positional indices at
the boundary where an ordered action list is required.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .action_space import Action, generate_actions
from .cards import ALL_CARDS
from .state import Board, ROWS


ACTION_KEY_SCHEMA = "regular_ofc_action_key_v1"
_ACTION_KEY_PREFIX = "rak1"
_MASK_HEX_WIDTH = 13  # 52 cards / four bits per hex digit.
_MAX_MASK = (1 << len(ALL_CARDS)) - 1
_CARD_INDEX = {card: index for index, card in enumerate(ALL_CARDS)}


@dataclass(frozen=True)
class ActionKey:
    """Four disjoint 52-bit masks forming a semantic action identity."""

    top_mask: int = 0
    middle_mask: int = 0
    bottom_mask: int = 0
    discard_mask: int = 0

    def __post_init__(self) -> None:
        masks = self.masks
        if any(not isinstance(mask, int) or isinstance(mask, bool) for mask in masks):
            raise TypeError("ActionKey masks must be integers")
        if any(mask < 0 or mask > _MAX_MASK for mask in masks):
            raise ValueError("ActionKey mask is outside the 52-card domain")
        union = 0
        for mask in masks:
            if union & mask:
                raise ValueError("ActionKey masks must be pairwise disjoint")
            union |= mask

    @property
    def masks(self) -> tuple[int, int, int, int]:
        return (
            self.top_mask,
            self.middle_mask,
            self.bottom_mask,
            self.discard_mask,
        )

    @classmethod
    def from_action(cls, action: Action) -> "ActionKey":
        masks = {row: 0 for row in ROWS}
        used = 0
        for card, row in action.placements:
            if row not in masks:
                raise ValueError(f"unknown action row: {row}")
            bit = _card_bit(card)
            if used & bit:
                raise ValueError(f"duplicate card in action: {card!r}")
            masks[row] |= bit
            used |= bit
        discard_mask = 0
        for card in action.discards:
            bit = _card_bit(card)
            if used & bit:
                raise ValueError(f"duplicate card in action: {card!r}")
            discard_mask |= bit
            used |= bit
        return cls(
            top_mask=masks["top"],
            middle_mask=masks["middle"],
            bottom_mask=masks["bottom"],
            discard_mask=discard_mask,
        )

    @classmethod
    def from_token(cls, token: str) -> "ActionKey":
        parts = token.split(":")
        if len(parts) != 5 or parts[0] != _ACTION_KEY_PREFIX:
            raise ValueError(f"invalid ActionKey token: {token!r}")
        if any(
            len(part) != _MASK_HEX_WIDTH
            or any(character not in "0123456789abcdef" for character in part)
            for part in parts[1:]
        ):
            raise ValueError(f"invalid ActionKey mask encoding: {token!r}")
        return cls(*(int(part, 16) for part in parts[1:]))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionKey":
        schema = payload.get("schema")
        if schema != ACTION_KEY_SCHEMA:
            raise ValueError(f"unsupported action key schema: {schema!r}")
        token = payload.get("token")
        if not isinstance(token, str):
            raise ValueError("ActionKey payload requires a string token")
        return cls.from_token(token)

    def to_token(self) -> str:
        encoded = ":".join(f"{mask:0{_MASK_HEX_WIDTH}x}" for mask in self.masks)
        return f"{_ACTION_KEY_PREFIX}:{encoded}"

    def stable_token(self) -> str:
        return self.to_token()

    def to_dict(self) -> dict[str, str]:
        return {"schema": ACTION_KEY_SCHEMA, "token": self.to_token()}

    def sort_key(self) -> tuple[int, int, int, int]:
        return self.masks

    def cards(self, group: str) -> tuple[str, ...]:
        try:
            index = ("top", "middle", "bottom", "discards").index(group)
        except ValueError as exc:
            raise ValueError(f"unknown ActionKey group: {group}") from exc
        mask = self.masks[index]
        return tuple(
            card for card_index, card in enumerate(ALL_CARDS) if mask & (1 << card_index)
        )


@dataclass(frozen=True)
class ActionResolution:
    index: int
    key: ActionKey
    source: str


def action_key(action: Action) -> ActionKey:
    return ActionKey.from_action(action)


def action_key_from_payload(payload: Mapping[str, Any]) -> ActionKey:
    placements = payload.get("placements", ())
    discards = payload.get("discards", ())
    if isinstance(placements, (str, bytes)) or isinstance(discards, (str, bytes)):
        raise ValueError("action payload placements/discards must be sequences")
    action = Action(
        placements=tuple((str(card), str(row)) for card, row in placements),
        discards=tuple(str(card) for card in discards),
    )
    return action_key(action)


def canonicalize_actions(actions: Iterable[Action]) -> list[Action]:
    """Return actions in stable semantic order, rejecting duplicate identities."""
    keyed = [(action_key(action), action) for action in actions]
    _ensure_unique_keys(key for key, _ in keyed)
    keyed.sort(key=lambda item: item[0].sort_key())
    return [action for _, action in keyed]


def canonical_argmax_index(values: Sequence[float], actions: Sequence[Action]) -> int:
    """Argmax with a semantic, enumeration-independent tie break."""
    if len(values) == 0 or len(values) != len(actions):
        raise ValueError("values/actions length mismatch")
    best = max(float(value) for value in values)
    tied = [index for index, value in enumerate(values) if float(value) == best]
    return min(tied, key=lambda index: action_key(actions[index]).sort_key())


def canonical_descending_indices(
    values: Sequence[float], actions: Sequence[Action]
) -> list[int]:
    """Rank values descending, breaking ties by semantic ActionKey."""
    if len(values) != len(actions):
        raise ValueError("values/actions length mismatch")
    return sorted(
        range(len(actions)),
        key=lambda index: (-float(values[index]), action_key(actions[index]).sort_key()),
    )


def generate_canonical_actions(board: Board, dealt_cards: Iterable[str]) -> list[Action]:
    """Opt-in canonical alternative to the legacy ordered action generator."""
    return canonicalize_actions(generate_actions(board, dealt_cards))


def index_actions_by_key(actions: Sequence[Action]) -> dict[ActionKey, int]:
    """Map each semantic key to its current positional index."""
    mapping: dict[ActionKey, int] = {}
    for index, action in enumerate(actions):
        key = action_key(action)
        if key in mapping:
            raise ValueError(
                f"duplicate semantic action key at indices {mapping[key]} and {index}"
            )
        mapping[key] = index
    return mapping


def resolve_action_key(actions: Sequence[Action], key: ActionKey) -> int:
    """Resolve *key* against a freshly generated ordered action list."""
    try:
        return index_actions_by_key(actions)[key]
    except KeyError as exc:
        raise KeyError(f"action key is not legal in this state: {key.to_token()}") from exc


def resolve_action_index(
    actions: Sequence[Action],
    *,
    key: ActionKey | str | None = None,
    payload: Mapping[str, Any] | None = None,
    legacy_index: int | None = None,
    expected_order_digest: str | None = None,
) -> ActionResolution:
    """Fail-closed resolver for new key-first and legacy index artifacts."""
    desired_key = ActionKey.from_token(key) if isinstance(key, str) else key
    if payload is not None:
        payload_key = action_key_from_payload(payload)
        if desired_key is not None and desired_key != payload_key:
            raise ValueError("canonical action key and action payload disagree")
        desired_key = payload_key

    if desired_key is not None:
        if legacy_index is not None and 0 <= legacy_index < len(actions):
            if action_key(actions[legacy_index]) == desired_key:
                return ActionResolution(legacy_index, desired_key, "key_at_legacy_index")
        index = resolve_action_key(actions, desired_key)
        return ActionResolution(index, desired_key, "key_search")

    if legacy_index is None:
        raise ValueError("action resolution requires a key, payload, or legacy index")
    if expected_order_digest is None:
        raise ValueError("legacy-index-only resolution requires an order digest")
    actual_digest = ordered_action_mapping_digest(actions)
    if actual_digest != expected_order_digest:
        raise ValueError("legacy action order digest mismatch")
    if not 0 <= legacy_index < len(actions):
        raise IndexError(f"legacy action index is out of range: {legacy_index}")
    resolved_key = action_key(actions[legacy_index])
    return ActionResolution(legacy_index, resolved_key, "verified_legacy_index")


def ordered_action_mapping_digest(actions: Sequence[Action]) -> str:
    """Digest an index-to-key mapping; changes whenever positional mapping changes."""
    payload = "\n".join(action_key(action).to_token() for action in actions).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def legal_action_set_digest(actions: Sequence[Action]) -> str:
    """Digest the legal semantic action set independent of enumeration order."""
    keys = [action_key(action) for action in actions]
    _ensure_unique_keys(keys)
    payload = "\n".join(
        key.to_token() for key in sorted(keys, key=ActionKey.sort_key)
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _card_bit(card: str) -> int:
    try:
        return 1 << _CARD_INDEX[card]
    except KeyError as exc:
        raise ValueError(f"invalid regular-rule card: {card!r}") from exc


def _ensure_unique_keys(keys: Iterable[ActionKey]) -> None:
    seen: set[ActionKey] = set()
    for key in keys:
        if key in seen:
            raise ValueError(f"duplicate semantic action key: {key.to_token()}")
        seen.add(key)
