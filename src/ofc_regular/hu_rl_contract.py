"""Fail-closed actor-facing contract for full-hand HU RL.

This module deliberately separates public action history from the simulator's
hidden world.  A historical placement records where public cards were placed
and how many cards were discarded, but never which card an opponent discarded.
The current actor's legal actions remain full :class:`ActionKey` values because
all cards in those actions are visible to that actor.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    generate_canonical_actions,
)
from .action_space import Action
from .cards import ALL_CARDS
from .hu_infoset import ActorObservation, InformationSetError, Seat, Street
from .state import ROWS


PUBLIC_PLACEMENT_SCHEMA = "regular_ofc_hu_rl_public_placement_v1"
LEGAL_ACTION_MAPPING_SCHEMA = "regular_ofc_hu_rl_legal_action_mapping_v1"
HU_RL_ACTOR_VIEW_SCHEMA = "regular_ofc_hu_rl_actor_view_v1"
MAX_LEGAL_ACTIONS = 232

_MASK_HEX_WIDTH = 13
_MAX_MASK = (1 << len(ALL_CARDS)) - 1
_CARD_INDEX = {card: index for index, card in enumerate(ALL_CARDS)}
_STREET_ORDER = {street: index for index, street in enumerate(("T0", "T1", "T2", "T3", "T4", "FL"))}
_SEAT_ORDER = {"first": 0, "second": 1}
_PUBLIC_ACTION_GEOMETRY = {
    "T0": (5, 0),
    "T1": (2, 1),
    "T2": (2, 1),
    "T3": (2, 1),
    "T4": (2, 1),
    "FL": (13, 1),
}
_FORBIDDEN_PUBLIC_HISTORY_FIELDS = {
    "audit_truth",
    "discard_card",
    "discard_cards",
    "discard_mask",
    "opponent_discard",
    "opponent_discard_card",
    "opponent_discard_cards",
    "opponent_legal_action_digest",
    "opponent_legal_digest",
    "opponent_private_discard",
    "opponent_private_discards",
    "replay_truth",
    "true_dead_cards",
    "true_opponent_private_discards",
    "world_state",
}


class HuRlContractError(InformationSetError):
    """Actor-view payload is unsafe or inconsistent with its public state."""


@dataclass(frozen=True)
class PublicPlacement:
    """One public action with discard identity intentionally erased.

    The three masks contain only cards placed face-up on the board.  There is
    no discard mask field on this object; only the public discard count remains.
    """

    street: Street
    acting_seat: Seat
    top_placement_mask: int
    middle_placement_mask: int
    bottom_placement_mask: int
    discard_count: int

    def __post_init__(self) -> None:
        if self.street not in _PUBLIC_ACTION_GEOMETRY:
            raise HuRlContractError(f"invalid public-placement street: {self.street!r}")
        if self.acting_seat not in _SEAT_ORDER:
            raise HuRlContractError(
                f"invalid public-placement acting_seat: {self.acting_seat!r}"
            )
        masks = self.placement_masks
        if any(not _strict_int(mask) for mask in masks):
            raise HuRlContractError("public placement masks must be integers")
        if any(mask < 0 or mask > _MAX_MASK for mask in masks):
            raise HuRlContractError("public placement mask is outside the 52-card domain")
        union = 0
        for mask in masks:
            if union & mask:
                raise HuRlContractError("public placement masks must be pairwise disjoint")
            union |= mask
        if not _strict_int(self.discard_count) or self.discard_count < 0:
            raise HuRlContractError("public discard_count must be a non-negative integer")
        expected_placements, expected_discards = _PUBLIC_ACTION_GEOMETRY[self.street]
        if union.bit_count() != expected_placements:
            raise HuRlContractError(
                f"{self.street} public action requires {expected_placements} placed cards"
            )
        if self.discard_count != expected_discards:
            raise HuRlContractError(
                f"{self.street} public action requires discard_count={expected_discards}"
            )

    @property
    def placement_masks(self) -> tuple[int, int, int]:
        return (
            self.top_placement_mask,
            self.middle_placement_mask,
            self.bottom_placement_mask,
        )

    @property
    def placement_mask(self) -> int:
        return self.top_placement_mask | self.middle_placement_mask | self.bottom_placement_mask

    @classmethod
    def from_action_key(
        cls, *, street: Street, acting_seat: Seat, key: ActionKey
    ) -> "PublicPlacement":
        if type(key) is not ActionKey:
            raise TypeError("PublicPlacement requires an ActionKey")
        return cls(
            street=street,
            acting_seat=acting_seat,
            top_placement_mask=key.top_mask,
            middle_placement_mask=key.middle_mask,
            bottom_placement_mask=key.bottom_mask,
            discard_count=key.discard_mask.bit_count(),
        )

    @classmethod
    def from_action(
        cls, *, street: Street, acting_seat: Seat, action: Action
    ) -> "PublicPlacement":
        return cls.from_action_key(
            street=street,
            acting_seat=acting_seat,
            key=action_key(action),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicPlacement":
        if not isinstance(payload, Mapping):
            raise HuRlContractError("public placement must be a mapping")
        forbidden = sorted(str(key) for key in payload if key in _FORBIDDEN_PUBLIC_HISTORY_FIELDS)
        if forbidden:
            raise HuRlContractError(
                "public history contains forbidden hidden/audit fields: "
                + ", ".join(forbidden)
            )
        _require_exact_fields(
            payload,
            {
                "schema",
                "street",
                "acting_seat",
                "top_placement_mask",
                "middle_placement_mask",
                "bottom_placement_mask",
                "discard_count",
            },
            context="public placement",
        )
        if payload["schema"] != PUBLIC_PLACEMENT_SCHEMA:
            raise HuRlContractError("unsupported public placement schema")
        street = payload["street"]
        acting_seat = payload["acting_seat"]
        if not isinstance(street, str) or not isinstance(acting_seat, str):
            raise HuRlContractError("public placement street/acting_seat must be strings")
        discard_count = payload["discard_count"]
        if not _strict_int(discard_count):
            raise HuRlContractError("public discard_count must be an integer")
        return cls(
            street=street,  # type: ignore[arg-type]
            acting_seat=acting_seat,  # type: ignore[arg-type]
            top_placement_mask=_decode_mask(payload["top_placement_mask"]),
            middle_placement_mask=_decode_mask(payload["middle_placement_mask"]),
            bottom_placement_mask=_decode_mask(payload["bottom_placement_mask"]),
            discard_count=discard_count,
        )

    def cards(self, row: str) -> tuple[str, ...]:
        if row not in ROWS:
            raise ValueError(f"unknown row: {row}")
        mask = self.placement_masks[ROWS.index(row)]
        return tuple(
            card for index, card in enumerate(ALL_CARDS) if mask & (1 << index)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PUBLIC_PLACEMENT_SCHEMA,
            "street": self.street,
            "acting_seat": self.acting_seat,
            "top_placement_mask": _encode_mask(self.top_placement_mask),
            "middle_placement_mask": _encode_mask(self.middle_placement_mask),
            "bottom_placement_mask": _encode_mask(self.bottom_placement_mask),
            "discard_count": self.discard_count,
        }

    def canonical_json(self) -> str:
        return canonical_json(self)

    def digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True)
class LegalActionMappingV1:
    """Exact index-to-ActionKey mapping for the actor's legal action tensor."""

    action_keys: tuple[ActionKey, ...]

    def __post_init__(self) -> None:
        keys = tuple(self.action_keys)
        if not keys:
            raise HuRlContractError("legal action mapping must not be empty")
        if len(keys) > MAX_LEGAL_ACTIONS:
            raise HuRlContractError(
                f"legal action mapping exceeds capacity {MAX_LEGAL_ACTIONS}"
            )
        if any(type(key) is not ActionKey for key in keys):
            raise TypeError("legal action mapping entries must be ActionKey values")
        if len(set(keys)) != len(keys):
            raise HuRlContractError("legal action mapping contains duplicate ActionKeys")
        object.__setattr__(self, "action_keys", keys)

    @classmethod
    def from_ordered_actions(cls, actions: Sequence[Action]) -> "LegalActionMappingV1":
        return cls(tuple(action_key(action) for action in actions))

    @classmethod
    def for_observation(cls, observation: ActorObservation) -> "LegalActionMappingV1":
        _require_actor_observation(observation)
        if observation.street == "FL":
            raise HuRlContractError(
                "HuRlActorViewV1 does not yet define the 14-card FL action tensor"
            )
        return cls.from_ordered_actions(
            generate_canonical_actions(observation.hero_board, observation.dealt_cards)
        )

    @property
    def action_count(self) -> int:
        return len(self.action_keys)

    @property
    def action_set_digest(self) -> str:
        tokens = sorted(
            (key.to_token() for key in self.action_keys),
            key=lambda token: ActionKey.from_token(token).sort_key(),
        )
        return _tokens_digest(tokens)

    @property
    def action_order_digest(self) -> str:
        return _tokens_digest(key.to_token() for key in self.action_keys)

    @property
    def legal_action_set_digest(self) -> str:
        return self.action_set_digest

    @property
    def ordered_action_mapping_digest(self) -> str:
        return self.action_order_digest

    def key_at(self, index: int) -> ActionKey:
        if not _strict_int(index):
            raise TypeError("legal action index must be an integer")
        if index < 0 or index >= self.action_count:
            raise IndexError("legal action index is outside the action mapping")
        return self.action_keys[index]

    def index_for(self, key: ActionKey | str) -> int:
        desired = ActionKey.from_token(key) if isinstance(key, str) else key
        if type(desired) is not ActionKey:
            raise TypeError("legal action lookup requires an ActionKey or token")
        try:
            return self.action_keys.index(desired)
        except ValueError as exc:
            raise KeyError(f"ActionKey is not legal: {desired.to_token()}") from exc

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LegalActionMappingV1":
        if not isinstance(payload, Mapping):
            raise HuRlContractError("legal action mapping must be a mapping")
        _require_exact_fields(
            payload,
            {
                "schema",
                "action_key_schema",
                "max_actions",
                "action_count",
                "action_keys",
                "action_set_digest",
                "action_order_digest",
            },
            context="legal action mapping",
        )
        if payload["schema"] != LEGAL_ACTION_MAPPING_SCHEMA:
            raise HuRlContractError("unsupported legal action mapping schema")
        if payload["action_key_schema"] != ACTION_KEY_SCHEMA:
            raise HuRlContractError("unsupported legal ActionKey schema")
        if payload["max_actions"] != MAX_LEGAL_ACTIONS or not _strict_int(
            payload["max_actions"]
        ):
            raise HuRlContractError("legal action mapping max_actions mismatch")
        raw_keys = payload["action_keys"]
        if isinstance(raw_keys, (str, bytes)) or not isinstance(raw_keys, Sequence):
            raise HuRlContractError("legal action_keys must be a sequence")
        if any(not isinstance(token, str) for token in raw_keys):
            raise HuRlContractError("legal action_keys must contain only strings")
        try:
            mapping = cls(tuple(ActionKey.from_token(token) for token in raw_keys))
        except (TypeError, ValueError) as exc:
            raise HuRlContractError(f"invalid legal ActionKey mapping: {exc}") from exc
        if not _strict_int(payload["action_count"]) or payload["action_count"] != mapping.action_count:
            raise HuRlContractError("legal action mapping action_count mismatch")
        _require_sha256(payload["action_set_digest"], "action_set_digest")
        _require_sha256(payload["action_order_digest"], "action_order_digest")
        if payload["action_set_digest"] != mapping.action_set_digest:
            raise HuRlContractError("legal action set digest mismatch")
        if payload["action_order_digest"] != mapping.action_order_digest:
            raise HuRlContractError("legal action order digest mismatch")
        return mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEGAL_ACTION_MAPPING_SCHEMA,
            "action_key_schema": ACTION_KEY_SCHEMA,
            "max_actions": MAX_LEGAL_ACTIONS,
            "action_count": self.action_count,
            "action_keys": [key.to_token() for key in self.action_keys],
            "action_set_digest": self.action_set_digest,
            "action_order_digest": self.action_order_digest,
        }

    def canonical_json(self) -> str:
        return canonical_json(self)

    def digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True)
class HuRlActorViewV1:
    """The complete policy-facing input for one regular HU decision."""

    observation: ActorObservation
    public_history: tuple[PublicPlacement, ...]
    legal_action_mapping: LegalActionMappingV1

    def __post_init__(self) -> None:
        _require_actor_observation(self.observation)
        history = tuple(self.public_history)
        if any(type(event) is not PublicPlacement for event in history):
            raise TypeError("public_history entries must be PublicPlacement values")
        if type(self.legal_action_mapping) is not LegalActionMappingV1:
            raise TypeError("legal_action_mapping must be LegalActionMappingV1")
        object.__setattr__(self, "public_history", history)

        expected_mapping = LegalActionMappingV1.for_observation(self.observation)
        if self.legal_action_mapping != expected_mapping:
            raise HuRlContractError(
                "legal ActionKey index mapping disagrees with canonical legal actions"
            )
        _validate_public_history(self.observation, history)

    @classmethod
    def from_observation(
        cls,
        observation: ActorObservation,
        *,
        public_history: Sequence[PublicPlacement] = (),
    ) -> "HuRlActorViewV1":
        return cls(
            observation=observation,
            public_history=tuple(public_history),
            legal_action_mapping=LegalActionMappingV1.for_observation(observation),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HuRlActorViewV1":
        if not isinstance(payload, Mapping):
            raise HuRlContractError("HU RL actor view must be a mapping")
        _require_exact_fields(
            payload,
            {"schema", "observation", "public_history", "legal_action_mapping"},
            context="HU RL actor view",
        )
        if payload["schema"] != HU_RL_ACTOR_VIEW_SCHEMA:
            raise HuRlContractError("unsupported HU RL actor view schema")
        raw_observation = payload["observation"]
        if not isinstance(raw_observation, Mapping):
            raise HuRlContractError("actor view observation must be a mapping")
        raw_history = payload["public_history"]
        if isinstance(raw_history, (str, bytes)) or not isinstance(raw_history, Sequence):
            raise HuRlContractError("actor view public_history must be a sequence")
        raw_mapping = payload["legal_action_mapping"]
        if not isinstance(raw_mapping, Mapping):
            raise HuRlContractError("actor view legal_action_mapping must be a mapping")
        try:
            observation = ActorObservation.from_dict(raw_observation)
        except (TypeError, ValueError) as exc:
            raise HuRlContractError(f"invalid actor observation: {exc}") from exc
        history = tuple(PublicPlacement.from_dict(event) for event in raw_history)
        mapping = LegalActionMappingV1.from_dict(raw_mapping)
        return cls(observation, history, mapping)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HU_RL_ACTOR_VIEW_SCHEMA,
            "observation": _canonical_observation_dict(self.observation),
            "public_history": [event.to_dict() for event in self.public_history],
            "legal_action_mapping": self.legal_action_mapping.to_dict(),
        }

    def canonical_json(self) -> str:
        return canonical_json(self)

    def digest(self) -> str:
        return canonical_digest(self)

    def fingerprint(self) -> str:
        return self.digest()


ContractValue = PublicPlacement | LegalActionMappingV1 | HuRlActorViewV1


def canonical_json(value: ContractValue) -> str:
    """Return byte-stable JSON for an RL actor-contract value."""
    if type(value) not in {PublicPlacement, LegalActionMappingV1, HuRlActorViewV1}:
        raise TypeError("canonical_json requires an HU RL contract value")
    return json.dumps(
        value.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_digest(value: ContractValue) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _validate_public_history(
    observation: ActorObservation, history: tuple[PublicPlacement, ...]
) -> None:
    current_order = _event_order(observation.street, observation.seat)
    previous_order = -1
    seen_events: set[tuple[Street, Seat]] = set()
    seen_cards = 0
    seen_by_seat = {"first": 0, "second": 0}
    seen_by_seat_row = {
        seat: {row: 0 for row in ROWS} for seat in ("first", "second")
    }
    hero_board_mask = _cards_mask(observation.hero_board.all_cards())
    opponent_board_mask = _cards_mask(observation.opponent_public_board.all_cards())
    for event in history:
        order = _event_order(event.street, event.acting_seat)
        identity = (event.street, event.acting_seat)
        if identity in seen_events:
            raise HuRlContractError("public_history contains a duplicate seat/street event")
        if order <= previous_order:
            raise HuRlContractError("public_history is not in canonical chronological order")
        if order >= current_order:
            raise HuRlContractError("public_history contains the current or a future action")
        if seen_cards & event.placement_mask:
            raise HuRlContractError("public_history places the same card more than once")
        visible_board = (
            observation.hero_board
            if event.acting_seat == observation.seat
            else observation.opponent_public_board
        )
        for row, placement_mask in zip(ROWS, event.placement_masks):
            visible_row_mask = _cards_mask(getattr(visible_board, row))
            if placement_mask & ~visible_row_mask:
                raise HuRlContractError(
                    "public_history placement rows disagree with the public boards"
                )
            seen_by_seat_row[event.acting_seat][row] |= placement_mask
        previous_order = order
        seen_events.add(identity)
        seen_cards |= event.placement_mask
        seen_by_seat[event.acting_seat] |= event.placement_mask

    # A policy-facing actor view is a complete public trajectory prefix, not a
    # best-effort collection of otherwise valid events.  Accepting a missing
    # event would make two payloads with the same public boards disagree about
    # what the actor was told and would break deterministic replay/belief
    # reconstruction.  With the fixed first/second schedule, a prefix ending
    # at ``current_order`` contains exactly that many events.
    if len(history) != current_order:
        raise HuRlContractError(
            "public_history is not the complete trajectory prefix for this decision"
        )
    if seen_by_seat[observation.seat] != hero_board_mask:
        raise HuRlContractError(
            "public_history does not exactly reconstruct the hero public board"
        )
    opponent_seat: Seat = "second" if observation.seat == "first" else "first"
    if seen_by_seat[opponent_seat] != opponent_board_mask:
        raise HuRlContractError(
            "public_history does not exactly reconstruct the opponent public board"
        )
    if seen_cards != hero_board_mask | opponent_board_mask:
        raise HuRlContractError(
            "public_history does not exactly reconstruct both public boards"
        )
    board_by_seat = {
        observation.seat: observation.hero_board,
        opponent_seat: observation.opponent_public_board,
    }
    for seat, board in board_by_seat.items():
        for row in ROWS:
            if seen_by_seat_row[seat][row] != _cards_mask(getattr(board, row)):
                raise HuRlContractError(
                    "public_history does not exactly reconstruct every public board row"
                )


def _event_order(street: Street, seat: Seat) -> int:
    return _STREET_ORDER[street] * 2 + _SEAT_ORDER[seat]


def _canonical_observation_dict(observation: ActorObservation) -> dict[str, Any]:
    payload = observation.to_dict()
    for board_name in ("hero_board", "opponent_public_board"):
        board = payload[board_name]
        for row in ROWS:
            board[row] = _sort_cards(board[row])
    payload["dealt_cards"] = _sort_cards(payload["dealt_cards"])
    payload["hero_private_discards"] = _sort_cards(payload["hero_private_discards"])
    return payload


def _sort_cards(cards: Sequence[str]) -> list[str]:
    return sorted(cards, key=_CARD_INDEX.__getitem__)


def _cards_mask(cards: Sequence[str]) -> int:
    mask = 0
    for card in cards:
        mask |= 1 << _CARD_INDEX[card]
    return mask


def _encode_mask(mask: int) -> str:
    return f"{mask:0{_MASK_HEX_WIDTH}x}"


def _decode_mask(value: Any) -> int:
    if (
        not isinstance(value, str)
        or len(value) != _MASK_HEX_WIDTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HuRlContractError("public placement masks require fixed-width lowercase hex")
    return int(value, 16)


def _tokens_digest(tokens: Any) -> str:
    return hashlib.sha256("\n".join(tokens).encode("ascii")).hexdigest()


def _require_actor_observation(observation: ActorObservation) -> None:
    if type(observation) is not ActorObservation:
        raise TypeError("HuRlActorViewV1 requires exactly ActorObservation")


def _require_exact_fields(
    payload: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(str(key) for key in actual - expected)
    if missing:
        raise HuRlContractError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise HuRlContractError(f"{context} contains unknown fields: {', '.join(unknown)}")


def _require_sha256(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HuRlContractError(f"{field_name} must be a lowercase SHA-256 digest")


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
