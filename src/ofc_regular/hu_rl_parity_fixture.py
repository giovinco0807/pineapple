"""Versioned deterministic full-hand golden fixture for Rust parity.

The fixture is generated entirely in memory from :class:`HuRlReferenceEnv`.
Its explicit 52-card deck is isolated below the clearly named ``oracle_only``
object.  Decision records contain actor-safe views plus audit outputs needed to
compare another implementation; they are parity artifacts, not policy inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .action_key import ACTION_KEY_SCHEMA, ActionKey
from .cards import ALL_CARDS, validate_cards
from .hu_infoset import Seat, Street
from .hu_rl_contract import HuRlActorViewV1, PublicPlacement
from .hu_rl_reference import HuRlReferenceEnv
from .state import Board, ROWS


HU_RL_PARITY_FIXTURE_SCHEMA = "regular_ofc_hu_rl_parity_fixture_v1"
HU_RL_PARITY_ORACLE_ONLY_SCHEMA = "regular_ofc_hu_rl_parity_oracle_only_v1"
HU_RL_PARITY_SELECTION_SCHEMA = "regular_ofc_hu_rl_parity_selection_v1"
HU_RL_PARITY_FIXTURE_ID = "regular_hu_full_hand_all_cards_middle_key_v1"
HU_RL_PARITY_SELECTION_ID = "canonical_middle_action_key_v1"
GOLDEN_HU_RL_PARITY_FIXTURE_SHA256 = (
    "60c634aa8fb91f1ebdbf507f508a573c337f6e6838e45e8d0c8b80ab62e27e3e"
)

_EXPECTED_SELECTION = {
    "schema": HU_RL_PARITY_SELECTION_SCHEMA,
    "policy_id": HU_RL_PARITY_SELECTION_ID,
    "rule": "ordered_action_keys[floor(action_count/2)]",
}
_CARD_INDEX = {card: index for index, card in enumerate(ALL_CARDS)}
_ORACLE_ONLY_PINNED_EXPLICIT_DECK = (*ALL_CARDS[5:], *ALL_CARDS[:5])


class HuRlParityFixtureError(ValueError):
    """A parity fixture is malformed, unsafe, or inconsistent on replay."""


@dataclass(frozen=True, repr=False)
class _OracleOnlyV1:
    explicit_deck: tuple[str, ...]

    def __post_init__(self) -> None:
        deck = tuple(self.explicit_deck)
        if deck != _ORACLE_ONLY_PINNED_EXPLICIT_DECK:
            raise HuRlParityFixtureError(
                "v1 golden oracle_only.explicit_deck must equal the pinned oracle deck"
            )
        try:
            validate_cards(deck)
        except ValueError as exc:  # defensive if the global deck ever changes.
            raise HuRlParityFixtureError(f"invalid oracle-only deck: {exc}") from exc
        object.__setattr__(self, "explicit_deck", deck)

    def __repr__(self) -> str:
        return "_OracleOnlyV1(explicit_deck=<redacted>)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HU_RL_PARITY_ORACLE_ONLY_SCHEMA,
            "explicit_deck": list(self.explicit_deck),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_OracleOnlyV1":
        _require_mapping(payload, "oracle_only")
        _require_exact_fields(
            payload,
            {"schema", "explicit_deck"},
            context="oracle_only",
        )
        if payload["schema"] != HU_RL_PARITY_ORACLE_ONLY_SCHEMA:
            raise HuRlParityFixtureError("unsupported oracle_only schema")
        deck = _string_sequence(payload["explicit_deck"], "oracle_only.explicit_deck")
        return cls(deck)


@dataclass(frozen=True)
class _DecisionV1:
    ordinal: int
    actor: int
    street: Street
    actor_view: HuRlActorViewV1
    actor_view_digest: str
    ordered_action_keys: tuple[str, ...]
    action_set_digest: str
    action_order_digest: str
    selected_legal_index: int
    selected_legal_action_key: str
    public_step_event: PublicPlacement
    done: bool
    rewards: tuple[float, float]

    def __post_init__(self) -> None:
        if not _strict_int(self.ordinal) or not 0 <= self.ordinal < 10:
            raise HuRlParityFixtureError("decision ordinal must be an integer in [0, 9]")
        expected_actor = self.ordinal % 2
        expected_street = f"T{self.ordinal // 2}"
        if not _strict_int(self.actor) or self.actor != expected_actor:
            raise HuRlParityFixtureError("decision actor disagrees with ordinal")
        if self.street != expected_street:
            raise HuRlParityFixtureError("decision street disagrees with ordinal")
        if type(self.actor_view) is not HuRlActorViewV1:
            raise TypeError("decision actor_view must be HuRlActorViewV1")
        if type(self.public_step_event) is not PublicPlacement:
            raise TypeError("decision public_step_event must be PublicPlacement")
        _require_sha256(self.actor_view_digest, "actor_view_digest")
        if self.actor_view_digest != self.actor_view.digest():
            raise HuRlParityFixtureError("decision actor_view digest mismatch")

        expected_seat: Seat = "first" if self.actor == 0 else "second"
        observation = self.actor_view.observation
        if (observation.street, observation.seat) != (self.street, expected_seat):
            raise HuRlParityFixtureError(
                "decision actor_view street/seat disagrees with decision"
            )
        if (self.public_step_event.street, self.public_step_event.acting_seat) != (
            self.street,
            expected_seat,
        ):
            raise HuRlParityFixtureError(
                "decision public step event street/seat disagrees with decision"
            )

        tokens = tuple(self.ordered_action_keys)
        if not tokens or any(not isinstance(token, str) for token in tokens):
            raise HuRlParityFixtureError(
                "decision ordered_action_keys must be a non-empty string sequence"
            )
        try:
            parsed = tuple(ActionKey.from_token(token) for token in tokens)
        except ValueError as exc:
            raise HuRlParityFixtureError(f"invalid decision ActionKey token: {exc}") from exc
        if len(set(parsed)) != len(parsed):
            raise HuRlParityFixtureError("decision ordered_action_keys are not unique")
        expected_mapping = self.actor_view.legal_action_mapping
        if parsed != expected_mapping.action_keys:
            raise HuRlParityFixtureError(
                "decision ordered_action_keys disagree with actor_view mapping"
            )
        _require_sha256(self.action_set_digest, "action_set_digest")
        _require_sha256(self.action_order_digest, "action_order_digest")
        if self.action_set_digest != expected_mapping.action_set_digest:
            raise HuRlParityFixtureError("decision action set digest mismatch")
        if self.action_order_digest != expected_mapping.action_order_digest:
            raise HuRlParityFixtureError("decision action order digest mismatch")

        deterministic_index = len(tokens) // 2
        if (
            not _strict_int(self.selected_legal_index)
            or self.selected_legal_index != deterministic_index
        ):
            raise HuRlParityFixtureError(
                "decision selected legal index violates the deterministic policy"
            )
        if self.selected_legal_action_key != tokens[deterministic_index]:
            raise HuRlParityFixtureError(
                "decision selected legal ActionKey disagrees with its index"
            )
        selected = parsed[deterministic_index]
        if self.public_step_event.placement_masks != selected.masks[:3] or (
            self.public_step_event.discard_count != selected.discard_mask.bit_count()
        ):
            raise HuRlParityFixtureError(
                "decision public step event disagrees with selected ActionKey"
            )

        if type(self.done) is not bool or self.done != (self.ordinal == 9):
            raise HuRlParityFixtureError("decision done flag disagrees with ordinal")
        rewards = _reward_pair(self.rewards, "decision rewards")
        if not self.done and rewards != (0.0, 0.0):
            raise HuRlParityFixtureError("nonterminal decision rewards must be zero")
        object.__setattr__(self, "ordered_action_keys", tokens)
        object.__setattr__(self, "rewards", rewards)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_DecisionV1":
        _require_mapping(payload, "decision")
        _require_exact_fields(
            payload,
            {
                "ordinal",
                "actor",
                "street",
                "actor_view",
                "actor_view_digest",
                "legal_action_mapping",
                "selected_legal_index",
                "selected_legal_action_key",
                "step",
            },
            context="decision",
        )
        actor_view_payload = _mapping_value(payload["actor_view"], "decision.actor_view")
        try:
            actor_view = HuRlActorViewV1.from_dict(actor_view_payload)
        except (TypeError, ValueError) as exc:
            raise HuRlParityFixtureError(f"invalid decision actor_view: {exc}") from exc

        mapping = _mapping_value(
            payload["legal_action_mapping"], "decision.legal_action_mapping"
        )
        _require_exact_fields(
            mapping,
            {
                "action_key_schema",
                "action_count",
                "ordered_action_keys",
                "action_set_digest",
                "action_order_digest",
            },
            context="decision legal_action_mapping",
        )
        if mapping["action_key_schema"] != ACTION_KEY_SCHEMA:
            raise HuRlParityFixtureError("unsupported decision ActionKey schema")
        ordered = _string_sequence(
            mapping["ordered_action_keys"],
            "decision.legal_action_mapping.ordered_action_keys",
        )
        if not _strict_int(mapping["action_count"]) or mapping["action_count"] != len(
            ordered
        ):
            raise HuRlParityFixtureError("decision legal action_count mismatch")

        step = _mapping_value(payload["step"], "decision.step")
        _require_exact_fields(
            step,
            {"public_event", "done", "rewards"},
            context="decision step",
        )
        public_event_payload = _mapping_value(
            step["public_event"], "decision.step.public_event"
        )
        try:
            public_event = PublicPlacement.from_dict(public_event_payload)
        except (TypeError, ValueError) as exc:
            raise HuRlParityFixtureError(f"invalid decision public event: {exc}") from exc
        street = payload["street"]
        if not isinstance(street, str):
            raise HuRlParityFixtureError("decision street must be a string")
        done = step["done"]
        if type(done) is not bool:
            raise HuRlParityFixtureError("decision step.done must be boolean")
        return cls(
            ordinal=payload["ordinal"],
            actor=payload["actor"],
            street=street,  # type: ignore[arg-type]
            actor_view=actor_view,
            actor_view_digest=payload["actor_view_digest"],
            ordered_action_keys=ordered,
            action_set_digest=mapping["action_set_digest"],
            action_order_digest=mapping["action_order_digest"],
            selected_legal_index=payload["selected_legal_index"],
            selected_legal_action_key=payload["selected_legal_action_key"],
            public_step_event=public_event,
            done=done,
            rewards=_reward_pair(step["rewards"], "decision step.rewards"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "actor": self.actor,
            "street": self.street,
            "actor_view": self.actor_view.to_dict(),
            "actor_view_digest": self.actor_view_digest,
            "legal_action_mapping": {
                "action_key_schema": ACTION_KEY_SCHEMA,
                "action_count": len(self.ordered_action_keys),
                "ordered_action_keys": list(self.ordered_action_keys),
                "action_set_digest": self.action_set_digest,
                "action_order_digest": self.action_order_digest,
            },
            "selected_legal_index": self.selected_legal_index,
            "selected_legal_action_key": self.selected_legal_action_key,
            "step": {
                "public_event": self.public_step_event.to_dict(),
                "done": self.done,
                "rewards": list(self.rewards),
            },
        }


@dataclass(frozen=True)
class _TerminalV1:
    boards: tuple[Board, Board]
    rewards: tuple[float, float]

    def __post_init__(self) -> None:
        boards = tuple(self.boards)
        if len(boards) != 2 or any(type(board) is not Board for board in boards):
            raise HuRlParityFixtureError("terminal requires exactly two Boards")
        if not all(board.is_complete() for board in boards):
            raise HuRlParityFixtureError("terminal boards must both be complete")
        try:
            validate_cards((*boards[0].all_cards(), *boards[1].all_cards()))
        except ValueError as exc:
            raise HuRlParityFixtureError(f"invalid terminal boards: {exc}") from exc
        object.__setattr__(self, "boards", boards)
        object.__setattr__(self, "rewards", _reward_pair(self.rewards, "terminal rewards"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_TerminalV1":
        _require_mapping(payload, "terminal")
        _require_exact_fields(payload, {"boards", "rewards"}, context="terminal")
        raw_boards = payload["boards"]
        if isinstance(raw_boards, (str, bytes)) or not isinstance(raw_boards, Sequence):
            raise HuRlParityFixtureError("terminal boards must be a sequence")
        if len(raw_boards) != 2:
            raise HuRlParityFixtureError("terminal requires exactly two board payloads")
        boards = tuple(_board_from_dict(item) for item in raw_boards)
        return cls(
            boards=(boards[0], boards[1]),
            rewards=_reward_pair(payload["rewards"], "terminal rewards"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "boards": [_board_to_dict(board) for board in self.boards],
            "rewards": list(self.rewards),
        }


@dataclass(frozen=True, repr=False)
class HuRlParityFixtureV1:
    """Immutable golden fixture that validates itself by exact replay."""

    oracle_only: _OracleOnlyV1
    decisions: tuple[_DecisionV1, ...]
    terminal: _TerminalV1

    def __post_init__(self) -> None:
        if type(self.oracle_only) is not _OracleOnlyV1:
            raise TypeError("fixture oracle_only must be the versioned oracle type")
        decisions = tuple(self.decisions)
        if len(decisions) != 10 or any(type(row) is not _DecisionV1 for row in decisions):
            raise HuRlParityFixtureError("fixture must contain exactly 10 decisions")
        if type(self.terminal) is not _TerminalV1:
            raise TypeError("fixture terminal must be the versioned terminal type")
        object.__setattr__(self, "decisions", decisions)
        _validate_exact_replay(self)

    def __repr__(self) -> str:
        return (
            "HuRlParityFixtureV1("
            f"decisions={len(self.decisions)}, sha256={self.digest()}, "
            "oracle_only=<redacted>)"
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HuRlParityFixtureV1":
        _require_mapping(payload, "parity fixture")
        _require_exact_fields(
            payload,
            {
                "schema",
                "fixture_id",
                "oracle_only",
                "deterministic_selection",
                "decisions",
                "terminal",
            },
            context="parity fixture",
        )
        if payload["schema"] != HU_RL_PARITY_FIXTURE_SCHEMA:
            raise HuRlParityFixtureError("unsupported HU RL parity fixture schema")
        if payload["fixture_id"] != HU_RL_PARITY_FIXTURE_ID:
            raise HuRlParityFixtureError("unsupported HU RL parity fixture id")
        selection = _mapping_value(
            payload["deterministic_selection"], "deterministic_selection"
        )
        _require_exact_fields(
            selection,
            set(_EXPECTED_SELECTION),
            context="deterministic_selection",
        )
        if dict(selection) != _EXPECTED_SELECTION:
            raise HuRlParityFixtureError("deterministic selection contract mismatch")
        raw_decisions = payload["decisions"]
        if isinstance(raw_decisions, (str, bytes)) or not isinstance(
            raw_decisions, Sequence
        ):
            raise HuRlParityFixtureError("fixture decisions must be a sequence")
        oracle = _OracleOnlyV1.from_dict(
            _mapping_value(payload["oracle_only"], "oracle_only")
        )
        decisions = tuple(_DecisionV1.from_dict(row) for row in raw_decisions)
        terminal = _TerminalV1.from_dict(
            _mapping_value(payload["terminal"], "terminal")
        )
        fixture = cls(oracle, decisions, terminal)
        expected_sha = GOLDEN_HU_RL_PARITY_FIXTURE_SHA256
        if expected_sha != "0" * 64 and fixture.digest() != expected_sha:
            raise HuRlParityFixtureError("canonical golden fixture SHA-256 mismatch")
        return fixture

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HU_RL_PARITY_FIXTURE_SCHEMA,
            "fixture_id": HU_RL_PARITY_FIXTURE_ID,
            "oracle_only": self.oracle_only.to_dict(),
            "deterministic_selection": dict(_EXPECTED_SELECTION),
            "decisions": [row.to_dict() for row in self.decisions],
            "terminal": self.terminal.to_dict(),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()


def build_golden_hu_rl_parity_fixture_v1() -> HuRlParityFixtureV1:
    """Build the one pinned V1 full-hand fixture without filesystem I/O."""

    oracle = _OracleOnlyV1(_ORACLE_ONLY_PINNED_EXPLICIT_DECK)
    env = HuRlReferenceEnv(oracle.explicit_deck)
    decisions: list[_DecisionV1] = []
    for ordinal in range(10):
        view = env.observe()
        mapping = view.legal_action_mapping
        selected_index = mapping.action_count // 2
        selected = mapping.key_at(selected_index)
        step = env.step(selected)
        decisions.append(
            _DecisionV1(
                ordinal=ordinal,
                actor=step.actor,
                street=step.street,
                actor_view=view,
                actor_view_digest=view.digest(),
                ordered_action_keys=tuple(
                    key.to_token() for key in mapping.action_keys
                ),
                action_set_digest=mapping.action_set_digest,
                action_order_digest=mapping.action_order_digest,
                selected_legal_index=selected_index,
                selected_legal_action_key=selected.to_token(),
                public_step_event=step.public_placement,
                done=step.done,
                rewards=step.rewards,
            )
        )
    terminal = _TerminalV1(
        boards=env.boards,
        rewards=(float(env.terminal_rewards()[0]), float(env.terminal_rewards()[1])),
    )
    fixture = HuRlParityFixtureV1(oracle, tuple(decisions), terminal)
    expected_sha = GOLDEN_HU_RL_PARITY_FIXTURE_SHA256
    if expected_sha != "0" * 64 and fixture.digest() != expected_sha:
        raise HuRlParityFixtureError("generated golden fixture SHA-256 drifted")
    return fixture


def _validate_exact_replay(fixture: HuRlParityFixtureV1) -> None:
    env = HuRlReferenceEnv(fixture.oracle_only.explicit_deck)
    for ordinal, decision in enumerate(fixture.decisions):
        view = env.observe()
        mapping = view.legal_action_mapping
        if decision.ordinal != ordinal:
            raise HuRlParityFixtureError("decision ordinals are not contiguous")
        if decision.actor_view.to_dict() != view.to_dict():
            raise HuRlParityFixtureError(
                f"decision {ordinal} actor_view failed exact replay"
            )
        if decision.actor_view_digest != view.digest():
            raise HuRlParityFixtureError(
                f"decision {ordinal} actor_view digest failed exact replay"
            )
        replay_tokens = tuple(key.to_token() for key in mapping.action_keys)
        if decision.ordered_action_keys != replay_tokens:
            raise HuRlParityFixtureError(
                f"decision {ordinal} legal mapping failed exact replay"
            )
        selected = mapping.key_at(mapping.action_count // 2)
        if decision.selected_legal_action_key != selected.to_token():
            raise HuRlParityFixtureError(
                f"decision {ordinal} selection failed exact replay"
            )
        step = env.step(selected)
        if (
            decision.actor != step.actor
            or decision.street != step.street
            or decision.public_step_event != step.public_placement
            or decision.done != step.done
            or decision.rewards != step.rewards
        ):
            raise HuRlParityFixtureError(
                f"decision {ordinal} public step failed exact replay"
            )
    if fixture.terminal.boards != env.boards:
        raise HuRlParityFixtureError("terminal boards failed exact replay")
    if fixture.terminal.rewards != tuple(env.terminal_rewards()):
        raise HuRlParityFixtureError("terminal rewards failed exact replay")


def _board_to_dict(board: Board) -> dict[str, list[str]]:
    return {
        row: sorted(getattr(board, row), key=_CARD_INDEX.__getitem__) for row in ROWS
    }


def _board_from_dict(payload: Any) -> Board:
    mapping = _mapping_value(payload, "terminal board")
    _require_exact_fields(mapping, set(ROWS), context="terminal board")
    try:
        return Board.from_rows(
            top=_string_sequence(mapping["top"], "terminal board.top"),
            middle=_string_sequence(mapping["middle"], "terminal board.middle"),
            bottom=_string_sequence(mapping["bottom"], "terminal board.bottom"),
        )
    except ValueError as exc:
        raise HuRlParityFixtureError(f"invalid terminal board: {exc}") from exc


def _reward_pair(value: Any, context: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise HuRlParityFixtureError(f"{context} must contain exactly two numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise HuRlParityFixtureError(f"{context} must contain only numbers")
    rewards = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in rewards):
        raise HuRlParityFixtureError(f"{context} must be finite")
    if rewards[0] + rewards[1] != 0.0:
        raise HuRlParityFixtureError(f"{context} must be exactly zero-sum")
    return rewards


def _string_sequence(value: Any, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HuRlParityFixtureError(f"{context} must be a sequence")
    if any(not isinstance(item, str) for item in value):
        raise HuRlParityFixtureError(f"{context} must contain only strings")
    return tuple(value)


def _mapping_value(value: Any, context: str) -> Mapping[str, Any]:
    _require_mapping(value, context)
    return value


def _require_mapping(value: Any, context: str) -> None:
    if not isinstance(value, Mapping):
        raise HuRlParityFixtureError(f"{context} must be a mapping")


def _require_exact_fields(
    payload: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(str(key) for key in actual - expected)
    if missing:
        raise HuRlParityFixtureError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise HuRlParityFixtureError(f"{context} contains unknown fields: {', '.join(unknown)}")


def _require_sha256(value: Any, context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HuRlParityFixtureError(f"{context} must be a lowercase SHA-256 digest")


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
