"""Information-set-safe state types for heads-up regular OFC.

``WorldState`` belongs to the simulator and may contain both players' private
discards. ``ActorObservation`` is the only card-bearing state intended for a
policy, feature encoder, or search root.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from collections.abc import Set as SetABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from .cards import ALL_CARDS, validate_cards
from .state import Board, ROWS


Seat = Literal["first", "second"]
ActOrder = Literal["first", "second"]
Street = Literal["T0", "T1", "T2", "T3", "T4", "FL"]
OBSERVATION_SCHEMA = "regular_ofc_actor_observation_v1"
SCORING_CONTEXT_SCHEMA = "regular_ofc_scoring_context_v1"
POLICY_FEATURE_SAMPLE_SCHEMA = "regular_ofc_policy_feature_sample_v1"
REPLAY_TRUTH_SCHEMA = "regular_ofc_replay_truth_v1"

# The production FL EV lives in one file and is read here, not copied.  Every
# ``ActorObservation`` built without an explicit ``ScoringContext`` -- which is
# every behavior root the label-generation fleets evaluate -- carries this
# value, so a literal here would be a second source of truth that silently
# outranks the config.  ``FALLBACK_FL_EV`` matches the config and exists only
# for trees shipped without ``configs/``; it deliberately does not fall back to
# any superseded value.
#
# v4 (2026-08-06) supersedes v3's 9.109.  v3 was measured against a STATIC
# Fantasyland side and was therefore a floor -- the static bias was later
# measured at +0.42 to +0.46 per hand.  v4 is the fixed point of 150,000 hands
# of self-play with Fantasyland actually played, adaptively, on both sides.
# Both superseded configs stay on disk unmodified: corpora labelled under them
# validate against them, and the loaders reject a mismatch by design.
FL_EV_CONFIG_RELPATH = "configs/fl_ev_regular_v4_selfplay.json"
FL_EV_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "fl_ev_regular_v4_selfplay.json"
FALLBACK_FL_EV = {14: 9.6}


def load_default_fl_ev() -> dict[int, float]:
    """Read the pinned FL EV table, falling back only to ``FALLBACK_FL_EV``."""
    if not FL_EV_CONFIG_PATH.exists():
        return dict(FALLBACK_FL_EV)
    payload = json.loads(FL_EV_CONFIG_PATH.read_text(encoding="utf-8"))
    raw = payload.get("fl_ev", {})
    if not raw:
        return dict(FALLBACK_FL_EV)
    return {int(cards): float(value) for cards, value in raw.items()}


_DEFAULT_FL_EV = tuple(sorted(load_default_fl_ev().items()))
_CARD_INDEX = {card: index for index, card in enumerate(ALL_CARDS)}
_STREETS = {"T0", "T1", "T2", "T3", "T4", "FL"}
_REGULAR_DECISION_GEOMETRY = {
    ("T0", "first"): (0, 0, 5, 0),
    ("T0", "second"): (0, 5, 5, 0),
    ("T1", "first"): (5, 5, 3, 0),
    ("T1", "second"): (5, 7, 3, 0),
    ("T2", "first"): (7, 7, 3, 1),
    ("T2", "second"): (7, 9, 3, 1),
    ("T3", "first"): (9, 9, 3, 2),
    ("T3", "second"): (9, 11, 3, 2),
    ("T4", "first"): (11, 11, 3, 3),
    ("T4", "second"): (11, 13, 3, 3),
}
_FORBIDDEN_POLICY_METADATA_KEYS = {
    "board",
    "cards_to_place",
    "dead_cards",
    "dealt",
    "draw_pile",
    "draw_order",
    "deck",
    "deck_tail",
    "future_cards",
    "future_rollouts",
    "hero_board",
    "hero_private_discards",
    "opponent_board",
    "opponent_private_discards",
    "private_discards",
    "policy_observation",
    "remaining_cards",
    "remaining_deck",
    "replay_truth",
    "replay_world",
    "true_dead_cards",
    "true_hero_private_discards",
    "true_opponent_private_discards",
    "visible_dead_cards",
    "world_state",
    "world",
}


class InformationSetError(ValueError):
    """Raised when a policy-facing record is missing or contains unsafe cards."""


class CardFreeMetadata(dict[str, Any]):
    """Mutable compatibility mapping that rejects card/world fields on update."""

    def __init__(self, initial: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        payload = dict(initial or {})
        payload.update(kwargs)
        validate_card_free_metadata(payload)
        super().__init__(
            (key, _metadata_value(value)) for key, value in payload.items()
        )

    def __setitem__(self, key: str, value: Any) -> None:
        validate_card_free_metadata({key: value})
        super().__setitem__(key, _metadata_value(value))

    def update(self, *args: Any, **kwargs: Any) -> None:
        payload = dict(*args, **kwargs)
        validate_card_free_metadata(payload)
        for key, value in payload.items():
            super().__setitem__(key, _metadata_value(value))

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def __ior__(self, other: Mapping[str, Any]):
        self.update(other)
        return self


@dataclass(frozen=True)
class ScoringContext:
    fl_ev: tuple[tuple[int, float], ...] = _DEFAULT_FL_EV
    middle_trips_royalty: int = 2
    hu_line_points: bool = True
    scoop_bonus: int = 3
    foul_enabled: bool = True
    fantasyland_cards: int = 14

    def __post_init__(self) -> None:
        normalized = tuple(sorted((int(cards), float(value)) for cards, value in self.fl_ev))
        if not normalized or any(cards <= 0 for cards, _value in normalized):
            raise ValueError("fl_ev must contain positive card counts")
        if len({cards for cards, _value in normalized}) != len(normalized):
            raise ValueError("fl_ev card counts must be unique")
        if self.middle_trips_royalty != 2:
            raise ValueError("regular OFC middle trips royalty must be 2")
        if self.fantasyland_cards != 14:
            raise ValueError("regular HU fantasyland uses 14 cards")
        object.__setattr__(self, "fl_ev", normalized)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScoringContext":
        _reject_unknown_fields(
            payload,
            {
                "schema",
                "fl_ev",
                "middle_trips_royalty",
                "hu_line_points",
                "scoop_bonus",
                "foul_enabled",
                "fantasyland_cards",
            },
            context="scoring context",
        )
        if payload.get("schema") != SCORING_CONTEXT_SCHEMA:
            raise ValueError("unsupported scoring context schema")
        raw_fl_ev = payload.get("fl_ev", {})
        if not isinstance(raw_fl_ev, Mapping):
            raise ValueError("scoring context fl_ev must be a mapping")
        return cls(
            fl_ev=tuple((int(cards), float(value)) for cards, value in raw_fl_ev.items()),
            middle_trips_royalty=int(payload.get("middle_trips_royalty", 2)),
            hu_line_points=bool(payload.get("hu_line_points", True)),
            scoop_bonus=int(payload.get("scoop_bonus", 3)),
            foul_enabled=bool(payload.get("foul_enabled", True)),
            fantasyland_cards=int(payload.get("fantasyland_cards", 14)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCORING_CONTEXT_SCHEMA,
            "fl_ev": {str(cards): value for cards, value in self.fl_ev},
            "middle_trips_royalty": self.middle_trips_royalty,
            "hu_line_points": self.hu_line_points,
            "scoop_bonus": self.scoop_bonus,
            "foul_enabled": self.foul_enabled,
            "fantasyland_cards": self.fantasyland_cards,
        }


@dataclass(frozen=True)
class ActorObservation:
    hero_board: Board
    opponent_public_board: Board
    dealt_cards: tuple[str, ...]
    hero_private_discards: tuple[str, ...]
    seat: Seat
    street: Street
    to_act_order: ActOrder
    scoring: ScoringContext = field(default_factory=ScoringContext)
    hero_in_fantasyland: bool = False
    opponent_in_fantasyland: bool = False

    def __post_init__(self) -> None:
        dealt = tuple(self.dealt_cards)
        hero_private = tuple(self.hero_private_discards)
        if self.seat not in {"first", "second"}:
            raise ValueError(f"invalid seat: {self.seat!r}")
        if self.to_act_order not in {"first", "second"}:
            raise ValueError(f"invalid to_act_order: {self.to_act_order!r}")
        if self.street not in _STREETS:
            raise ValueError(f"invalid street: {self.street!r}")
        self.hero_board.validate()
        self.opponent_public_board.validate()
        validate_cards(
            (
                *self.hero_board.all_cards(),
                *self.opponent_public_board.all_cards(),
                *dealt,
                *hero_private,
            )
        )
        if self.street != "FL":
            expected = _REGULAR_DECISION_GEOMETRY[(self.street, self.to_act_order)]
            if self.opponent_in_fantasyland:
                # A second table, not a looser bound on the first. An opponent
                # in Fantasyland takes fourteen cards face down and never places
                # one where the hero can see it, so its public board holds zero
                # cards at EVERY street -- not fewer than usual, zero. Reading
                # it as a relaxation would also accept a partial board, which is
                # a hand nobody is playing. The hero's own three columns are
                # untouched: same player, same decisions, same cards, and only
                # the information opposite is gone. Mirrors
                # `regular_decision_geometry` in the Rust engine.
                hero, _opponent, dealt_count, discard_count = expected
                expected = (hero, 0, dealt_count, discard_count)
            actual = (
                self.hero_board.card_count(),
                self.opponent_public_board.card_count(),
                len(dealt),
                len(hero_private),
            )
            if actual != expected:
                raise InformationSetError(
                    "inconsistent regular decision geometry: "
                    f"street/order={self.street}/{self.to_act_order}"
                    f"{' vs-fantasyland' if self.opponent_in_fantasyland else ''} "
                    f"expected hero/opponent/dealt/hero-discards={expected}, got {actual}"
                )
            if self.seat != self.to_act_order:
                raise InformationSetError(
                    "regular HU seat must match within-street action order"
                )
            # The hero's own Fantasyland is a different GAME, not a different
            # view of this one: fourteen cards arrive and thirteen are set in a
            # single action, which this action space cannot describe. It stays
            # refused. The opponent's is a different VIEW, and is now
            # representable -- see the geometry branch above.
            if self.hero_in_fantasyland:
                raise InformationSetError(
                    "hero_in_fantasyland requires the FL observation schema: the "
                    "hero sets thirteen of fourteen cards in one action, which "
                    "this action space cannot describe"
                )
        object.__setattr__(self, "dealt_cards", dealt)
        object.__setattr__(self, "hero_private_discards", hero_private)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActorObservation":
        _reject_unknown_fields(
            payload,
            {
                "schema",
                "hero_board",
                "opponent_public_board",
                "dealt_cards",
                "hero_private_discards",
                "seat",
                "street",
                "to_act_order",
                "scoring",
                "hero_in_fantasyland",
                "opponent_in_fantasyland",
                "opponent_discard_count",
            },
            context="actor observation",
        )
        if payload.get("schema") != OBSERVATION_SCHEMA:
            raise ValueError("unsupported actor observation schema")
        observation = cls(
            hero_board=_board_from_dict(
                _mapping(payload, "hero_board"), reject_unknown=True
            ),
            opponent_public_board=_board_from_dict(
                _mapping(payload, "opponent_public_board"), reject_unknown=True
            ),
            dealt_cards=tuple(str(card) for card in _sequence(payload, "dealt_cards")),
            hero_private_discards=tuple(
                str(card) for card in _sequence(payload, "hero_private_discards")
            ),
            seat=str(payload.get("seat", "")),  # type: ignore[arg-type]
            street=str(payload.get("street", "")),  # type: ignore[arg-type]
            to_act_order=str(payload.get("to_act_order", "")),  # type: ignore[arg-type]
            scoring=ScoringContext.from_dict(_mapping(payload, "scoring")),
            hero_in_fantasyland=bool(payload.get("hero_in_fantasyland", False)),
            opponent_in_fantasyland=bool(
                payload.get("opponent_in_fantasyland", False)
            ),
        )
        declared_count = payload.get("opponent_discard_count")
        if declared_count is not None:
            if (
                not isinstance(declared_count, int)
                or isinstance(declared_count, bool)
                or declared_count != observation.opponent_discard_count
            ):
                raise ValueError(
                    "opponent_discard_count disagrees with public board"
                )
        return observation

    @property
    def opponent_discard_count(self) -> int:
        count = self.opponent_public_board.card_count()
        return max(0, min(4, (count - 5) // 2))

    def legacy_dead_cards(self) -> tuple[str, ...]:
        """Safe adapter for existing models' ambiguously named dead-card field."""
        return (
            *self.opponent_public_board.all_cards(),
            *self.hero_private_discards,
        )

    def known_unavailable_cards(self) -> tuple[str, ...]:
        known = {
            *self.hero_board.all_cards(),
            *self.opponent_public_board.all_cards(),
            *self.dealt_cards,
            *self.hero_private_discards,
        }
        return tuple(card for card in ALL_CARDS if card in known)

    def fingerprint(self) -> str:
        payload = _canonical_observation_payload(self)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OBSERVATION_SCHEMA,
            "hero_board": _board_to_dict(self.hero_board),
            "opponent_public_board": _board_to_dict(self.opponent_public_board),
            "dealt_cards": list(self.dealt_cards),
            "hero_private_discards": list(self.hero_private_discards),
            "seat": self.seat,
            "street": self.street,
            "to_act_order": self.to_act_order,
            "scoring": self.scoring.to_dict(),
            "hero_in_fantasyland": self.hero_in_fantasyland,
            "opponent_in_fantasyland": self.opponent_in_fantasyland,
            "opponent_discard_count": self.opponent_discard_count,
        }


@dataclass(frozen=True)
class WorldState:
    boards: tuple[Board, Board]
    private_discards: tuple[tuple[str, ...], tuple[str, ...]]
    street: Street
    next_player: int
    scoring: ScoringContext = field(default_factory=ScoringContext)
    fantasyland: tuple[bool, bool] = (False, False)

    def __post_init__(self) -> None:
        boards = tuple(self.boards)
        discards = tuple(tuple(cards) for cards in self.private_discards)
        fantasyland = tuple(bool(value) for value in self.fantasyland)
        if len(boards) != 2 or len(discards) != 2 or len(fantasyland) != 2:
            raise ValueError("WorldState requires exactly two players")
        if self.next_player not in (0, 1):
            raise ValueError("next_player must be 0 or 1")
        if self.street not in _STREETS:
            raise ValueError(f"invalid street: {self.street!r}")
        for board in boards:
            board.validate()
        validate_cards(
            (
                *boards[0].all_cards(),
                *boards[1].all_cards(),
                *discards[0],
                *discards[1],
            )
        )
        object.__setattr__(self, "boards", boards)
        object.__setattr__(self, "private_discards", discards)
        object.__setattr__(self, "fantasyland", fantasyland)

    def observe(self, actor: int, dealt_cards: Iterable[str]) -> ActorObservation:
        if actor not in (0, 1):
            raise ValueError("actor must be 0 or 1")
        if actor != self.next_player:
            raise ValueError("actor does not match WorldState.next_player")
        dealt = tuple(dealt_cards)
        validate_cards(
            (
                *self.boards[0].all_cards(),
                *self.boards[1].all_cards(),
                *self.private_discards[0],
                *self.private_discards[1],
                *dealt,
            )
        )
        hero = self.boards[actor]
        opponent = self.boards[1 - actor]
        return ActorObservation(
            hero_board=hero,
            opponent_public_board=opponent,
            dealt_cards=dealt,
            hero_private_discards=self.private_discards[actor],
            seat="first" if actor == 0 else "second",
            street=self.street,
            to_act_order="second" if opponent.card_count() > hero.card_count() else "first",
            scoring=self.scoring,
            hero_in_fantasyland=self.fantasyland[actor],
            opponent_in_fantasyland=self.fantasyland[1 - actor],
        )


@dataclass(frozen=True)
class ReplayTruth:
    """Offline-only hidden state attached after a policy has acted."""

    true_dead_cards: tuple[str, ...]
    visible_dead_cards: tuple[str, ...]
    hero_private_discards: tuple[str, ...]
    opponent_private_discards: tuple[str, ...]

    def __post_init__(self) -> None:
        true_dead = tuple(self.true_dead_cards)
        visible = tuple(self.visible_dead_cards)
        hero_private = tuple(self.hero_private_discards)
        opponent_private = tuple(self.opponent_private_discards)
        validate_cards(true_dead)
        validate_cards(visible)
        validate_cards(hero_private)
        validate_cards(opponent_private)
        validate_cards((*hero_private, *opponent_private))
        if not set(hero_private).issubset(true_dead):
            raise ValueError("hero private discards must be included in true dead cards")
        if not set(opponent_private).issubset(true_dead):
            raise ValueError("opponent private discards must be included in true dead cards")
        if len(true_dead) != len(hero_private) + len(opponent_private) or set(
            true_dead
        ) != {*hero_private, *opponent_private}:
            raise ValueError(
                "true dead cards must exactly match both players' private discards"
            )
        if set(opponent_private).intersection(visible):
            raise ValueError("opponent private discards cannot be actor-visible")
        if not set(hero_private).issubset(visible):
            raise ValueError("hero private discards must be actor-visible")
        object.__setattr__(self, "true_dead_cards", true_dead)
        object.__setattr__(self, "visible_dead_cards", visible)
        object.__setattr__(self, "hero_private_discards", hero_private)
        object.__setattr__(self, "opponent_private_discards", opponent_private)

    @classmethod
    def from_world(
        cls,
        world: WorldState,
        *,
        actor: int,
        observation: ActorObservation,
    ) -> "ReplayTruth":
        expected = world.observe(actor, observation.dealt_cards)
        if expected != observation:
            raise ValueError("observation does not match replay world")
        return cls(
            true_dead_cards=(
                *world.private_discards[0],
                *world.private_discards[1],
            ),
            visible_dead_cards=observation.legacy_dead_cards(),
            hero_private_discards=world.private_discards[actor],
            opponent_private_discards=world.private_discards[1 - actor],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPLAY_TRUTH_SCHEMA,
            "true_dead_cards": list(self.true_dead_cards),
            "visible_dead_cards": list(self.visible_dead_cards),
            "hero_private_discards": list(self.hero_private_discards),
            "opponent_private_discards": list(self.opponent_private_discards),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayTruth":
        if payload.get("schema") != REPLAY_TRUTH_SCHEMA:
            raise InformationSetError("unsupported replay truth schema")
        required = (
            "true_dead_cards",
            "visible_dead_cards",
            "hero_private_discards",
            "opponent_private_discards",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise InformationSetError(
                f"replay truth is missing required fields: {', '.join(missing)}"
            )
        try:
            return cls(
                true_dead_cards=_card_sequence(payload, "true_dead_cards"),
                visible_dead_cards=_card_sequence(payload, "visible_dead_cards"),
                hero_private_discards=_card_sequence(
                    payload, "hero_private_discards"
                ),
                opponent_private_discards=_card_sequence(
                    payload, "opponent_private_discards"
                ),
            )
        except ValueError as exc:
            raise InformationSetError(f"invalid replay truth: {exc}") from exc

    def to_legacy_record_fields(self) -> dict[str, Any]:
        """Explicit replay fields added only after the policy has returned.

        Actor-visible compatibility fields (especially ``dead_cards``) are
        intentionally absent, so attachment cannot replace what the actor saw.
        """
        return {
            "true_dead_cards": list(self.true_dead_cards),
            "true_hero_private_discards": list(self.hero_private_discards),
            "true_opponent_private_discards": list(
                self.opponent_private_discards
            ),
            "replay_truth": self.to_dict(),
            "replay_ready": True,
        }


def replay_truth_from_record(record: Mapping[str, Any]) -> ReplayTruth:
    """Read replay-only truth without guessing what ``dead_cards`` meant.

    New rows carry the versioned nested object. Flat rows are accepted only
    when visible cards and both players' private discards are all explicit and
    mutually consistent. Anything incomplete or contradictory fails closed.
    """
    nested = record.get("replay_truth")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise InformationSetError("replay_truth must be a mapping")
        truth = ReplayTruth.from_dict(nested)
        _validate_record_truth_fields(record, truth, allow_legacy_dead=True)
        return truth

    true_dead = _optional_card_sequence(record, "true_dead_cards")
    visible = _optional_card_sequence(record, "visible_dead_cards")
    true_hero = _optional_card_sequence(record, "true_hero_private_discards")
    true_opponent = _optional_card_sequence(
        record, "true_opponent_private_discards"
    )
    if true_hero is not None or true_opponent is not None:
        if any(
            value is None
            for value in (true_dead, visible, true_hero, true_opponent)
        ):
            raise InformationSetError("incomplete flattened replay truth fields")
        truth = _make_replay_truth(
            true_dead=true_dead,
            visible=visible,
            hero_private=true_hero,
            opponent_private=true_opponent,
        )
        _validate_record_truth_fields(record, truth, allow_legacy_dead=False)
        return truth

    hero = _optional_card_sequence(record, "hero_private_discards")
    opponent = _optional_card_sequence(record, "opponent_private_discards")
    if true_dead is not None:
        if any(value is None for value in (visible, hero, opponent)):
            raise InformationSetError("incomplete legacy replay truth fields")
        truth = _make_replay_truth(
            true_dead=true_dead,
            visible=visible,
            hero_private=hero,
            opponent_private=opponent,
        )
        _validate_record_truth_fields(record, truth, allow_legacy_dead=True)
        return truth

    legacy_dead = _optional_card_sequence(record, "dead_cards")
    if all(value is not None for value in (legacy_dead, visible, hero, opponent)):
        combined = (*hero, *opponent)
        if not _same_cards(legacy_dead, combined):
            raise InformationSetError(
                "legacy dead_cards do not uniquely identify private discards"
            )
        truth = _make_replay_truth(
            true_dead=legacy_dead,
            visible=visible,
            hero_private=hero,
            opponent_private=opponent,
        )
        _validate_record_truth_fields(record, truth, allow_legacy_dead=True)
        return truth

    raise InformationSetError("record has no unambiguous replay truth")


def _make_replay_truth(
    *,
    true_dead: tuple[str, ...],
    visible: tuple[str, ...],
    hero_private: tuple[str, ...],
    opponent_private: tuple[str, ...],
) -> ReplayTruth:
    try:
        return ReplayTruth(
            true_dead_cards=true_dead,
            visible_dead_cards=visible,
            hero_private_discards=hero_private,
            opponent_private_discards=opponent_private,
        )
    except ValueError as exc:
        raise InformationSetError(f"invalid replay truth: {exc}") from exc


def _validate_record_truth_fields(
    record: Mapping[str, Any],
    truth: ReplayTruth,
    *,
    allow_legacy_dead: bool,
) -> None:
    expected = {
        "true_dead_cards": truth.true_dead_cards,
        "true_hero_private_discards": truth.hero_private_discards,
        "true_opponent_private_discards": truth.opponent_private_discards,
        "visible_dead_cards": truth.visible_dead_cards,
        "hero_private_discards": truth.hero_private_discards,
    }
    for name, cards in expected.items():
        actual = _optional_card_sequence(record, name)
        if actual is not None and not _same_cards(actual, cards):
            raise InformationSetError(f"{name} disagrees with replay_truth")

    opponent = _optional_card_sequence(record, "opponent_private_discards")
    if opponent is not None and not _same_cards(
        opponent, truth.opponent_private_discards
    ):
        raise InformationSetError(
            "opponent_private_discards disagrees with replay_truth"
        )

    dead = _optional_card_sequence(record, "dead_cards")
    if dead is None or _same_cards(dead, truth.visible_dead_cards):
        return
    if (
        allow_legacy_dead
        and str(record.get("visibility_model", "")) == "hidden_discard"
        and _same_cards(dead, truth.true_dead_cards)
    ):
        return
    raise InformationSetError(
        "dead_cards is neither the actor-visible set nor an explicit legacy truth set"
    )


def _optional_card_sequence(
    payload: Mapping[str, Any], name: str
) -> tuple[str, ...] | None:
    if name not in payload or payload.get(name) is None:
        return None
    return _card_sequence(payload, name)


def _card_sequence(payload: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = payload.get(name)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InformationSetError(f"{name} must be a card sequence")
    cards = tuple(str(card) for card in value)
    try:
        validate_cards(cards)
    except ValueError as exc:
        raise InformationSetError(f"invalid {name}: {exc}") from exc
    return cards


def _same_cards(left: Sequence[str], right: Sequence[str]) -> bool:
    return len(left) == len(right) and set(left) == set(right)


def validate_card_free_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    path: str = "metadata",
) -> None:
    """Reject card/world fields from mutable policy metadata recursively."""
    if metadata is None:
        return
    for raw_key, value in metadata.items():
        key = str(raw_key)
        child_path = f"{path}.{key}"
        if key.casefold() in _FORBIDDEN_POLICY_METADATA_KEYS:
            raise InformationSetError(
                f"policy metadata contains forbidden card/world field: {child_path}"
            )
        if isinstance(value, (WorldState, ActorObservation, ReplayTruth, Board)):
            raise InformationSetError(
                f"policy metadata contains a card-bearing object: {child_path}"
            )
        if isinstance(value, MappingABC):
            validate_card_free_metadata(value, path=child_path)
        elif isinstance(value, (SequenceABC, SetABC)) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                item_path = f"{child_path}[{index}]"
                if isinstance(item, (WorldState, ActorObservation, ReplayTruth, Board)):
                    raise InformationSetError(
                        f"policy metadata contains a card-bearing object: {item_path}"
                    )
                if isinstance(item, MappingABC):
                    validate_card_free_metadata(item, path=item_path)
                elif isinstance(item, (SequenceABC, SetABC)) and not isinstance(
                    item, (str, bytes, bytearray)
                ):
                    validate_card_free_metadata(
                        {"items": item},
                        path=item_path,
                    )


def card_free_metadata(metadata: Mapping[str, Any] | None) -> CardFreeMetadata:
    validate_card_free_metadata(metadata)
    return CardFreeMetadata(metadata)


def _metadata_value(value: Any) -> Any:
    if isinstance(value, CardFreeMetadata):
        return value
    if isinstance(value, Mapping):
        return CardFreeMetadata(value)
    if isinstance(value, list):
        return tuple(_metadata_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_metadata_value(item) for item in value)
    return value


def actor_observation_from_record(record: Mapping[str, Any]) -> ActorObservation:
    """Build and validate an observation from a legacy teacher/runtime record."""
    if "hero_private_discards" not in record:
        raise InformationSetError("record lacks explicit hero_private_discards")
    board = _board_from_dict(_mapping(record, "board"))
    opponent = _board_from_dict(_mapping(record, "opponent_board"))
    dealt_raw = record.get("dealt", record.get("cards_to_place"))
    if dealt_raw is None or isinstance(dealt_raw, (str, bytes)):
        raise InformationSetError("record lacks a dealt-card sequence")
    dealt = tuple(str(card) for card in dealt_raw)
    hero_private = tuple(
        str(card) for card in _sequence(record, "hero_private_discards")
    )
    observation = ActorObservation(
        hero_board=board,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_private_discards=hero_private,
        seat=str(record.get("seat", "")),  # type: ignore[arg-type]
        street=str(record.get("turn", _street_from_board(board))),  # type: ignore[arg-type]
        to_act_order=str(record.get("to_act_order", _to_act_order(board, opponent))),  # type: ignore[arg-type]
        scoring=_scoring_from_record(record),
    )
    visible_raw = record.get("visible_dead_cards")
    if visible_raw is not None:
        if isinstance(visible_raw, (str, bytes)):
            raise InformationSetError("visible_dead_cards must be a sequence")
        visible = tuple(str(card) for card in visible_raw)
        validate_cards(visible)
        if set(visible) != set(observation.legacy_dead_cards()) or len(visible) != len(
            observation.legacy_dead_cards()
        ):
            raise InformationSetError(
                "visible_dead_cards disagree with opponent public board plus hero private discards"
            )
    return observation


def policy_feature_sample_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the minimal safe legacy-shaped sample consumed by HU encoders."""
    derived = actor_observation_from_record(record)
    nested = record.get("policy_observation")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise InformationSetError("policy_observation must be a mapping")
        declared = ActorObservation.from_dict(nested)
        if declared != derived:
            raise InformationSetError("policy_observation disagrees with public record fields")
        observation = declared
    else:
        observation = derived
    actions = record.get("actions", ())
    if isinstance(actions, (str, bytes)) or not actions:
        raise InformationSetError("record has no action sequence")
    return {
        "rule_set": "regular",
        "schema": POLICY_FEATURE_SAMPLE_SCHEMA,
        "phase": record.get("phase", "hu_turn2_7card"),
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "board": _board_to_dict(observation.hero_board),
        "opponent_board": _board_to_dict(observation.opponent_public_board),
        "dead_cards": list(observation.legacy_dead_cards()),
        "dealt": list(observation.dealt_cards),
        "best_action": record.get("best_action", 0),
        "score_gap": record.get("score_gap", 0.0),
        "actions": list(actions),
    }


def _scoring_from_record(record: Mapping[str, Any]) -> ScoringContext:
    raw = record.get("fl_ev")
    if isinstance(raw, Mapping) and raw:
        fl_ev = tuple((int(cards), float(value)) for cards, value in raw.items())
    elif "fl_ev_14" in record:
        fl_ev = ((14, float(record["fl_ev_14"])),)
    else:
        fl_ev = _DEFAULT_FL_EV
    return ScoringContext(fl_ev=fl_ev)


def _canonical_observation_payload(observation: ActorObservation) -> dict[str, Any]:
    payload = observation.to_dict()
    payload.pop("opponent_discard_count", None)
    for board_key in ("hero_board", "opponent_public_board"):
        for row in ROWS:
            payload[board_key][row] = _sort_cards(payload[board_key][row])
    payload["dealt_cards"] = _sort_cards(payload["dealt_cards"])
    payload["hero_private_discards"] = _sort_cards(
        payload["hero_private_discards"]
    )
    return payload


def _sort_cards(cards: Sequence[str]) -> list[str]:
    return sorted(cards, key=_CARD_INDEX.__getitem__)


def _street_from_board(board: Board) -> Street:
    return {0: "T0", 5: "T1", 7: "T2", 9: "T3", 11: "T4"}.get(
        board.card_count(), "T4"
    )  # type: ignore[return-value]


def _to_act_order(board: Board, opponent: Board) -> ActOrder:
    return "second" if opponent.card_count() > board.card_count() else "first"


def _board_to_dict(board: Board) -> dict[str, list[str]]:
    return {row: list(getattr(board, row)) for row in ROWS}


def _board_from_dict(
    payload: Mapping[str, Any], *, reject_unknown: bool = False
) -> Board:
    if reject_unknown:
        _reject_unknown_fields(
            payload,
            {"top", "middle", "bottom"},
            context="public board",
        )
    return Board.from_rows(
        top=_sequence(payload, "top"),
        middle=_sequence(payload, "middle"),
        bottom=_sequence(payload, "bottom"),
    )


def _reject_unknown_fields(
    payload: Mapping[str, Any], allowed: set[str], *, context: str
) -> None:
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise ValueError(
            f"{context} contains unknown fields: {', '.join(unknown)}"
        )


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise InformationSetError(f"{name} must be a mapping")
    return value


def _sequence(payload: Mapping[str, Any], name: str) -> Sequence[Any]:
    value = payload.get(name, ())
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InformationSetError(f"{name} must be a sequence")
    return value
