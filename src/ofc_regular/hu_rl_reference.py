"""Slow, fail-closed Python correctness oracle for full-hand HU RL.

The environment intentionally mirrors the deal order used by :mod:`play_ai`
and :mod:`evaluate_matchups`: T0 first/second receive five cards, followed by
T1--T4 first/second receiving three cards each.  Policies see only
``HuRlActorViewV1``.  The explicit deck, its unrealized tail, and both players'
private discards remain simulator state and must never be attached to the actor
view or public history.

This module is a scalar reference, not a production rollout engine.  Its job is
to provide an unambiguous oracle for the future Rust/batch implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .action_key import ActionKey, action_key, generate_canonical_actions
from .cards import ALL_CARDS, validate_cards
from .hu_infoset import ScoringContext, Seat, Street, WorldState
from .hu_rl_contract import (
    HuRlActorViewV1,
    LegalActionMappingV1,
    PublicPlacement,
)
from .state import Board, ROWS
from .teacher import DEFAULT_FL_EV, terminal_score


_DECISION_SCHEDULE: tuple[tuple[Street, int], ...] = (
    ("T0", 0),
    ("T0", 1),
    ("T1", 0),
    ("T1", 1),
    ("T2", 0),
    ("T2", 1),
    ("T3", 0),
    ("T3", 1),
    ("T4", 0),
    ("T4", 1),
)
_DEAL_SIZES = (5, 5, 3, 3, 3, 3, 3, 3, 3, 3)
_DEAL_STARTS = tuple(sum(_DEAL_SIZES[:index]) for index in range(10))
_CARDS_USED_PER_NORMAL_HAND = sum(_DEAL_SIZES)
_SCORING = ScoringContext(
    fl_ev=tuple((int(cards), float(value)) for cards, value in DEFAULT_FL_EV.items())
)


class HuRlReferenceError(ValueError):
    """The scalar environment received an invalid deck, state, or action."""


@dataclass(frozen=True)
class HuRlReferenceStep:
    """Result of one legal decision.

    ``public_placement`` deliberately erases discard identity.  Rewards are
    zero until the final decision and then contain the zero-sum p0/p1 result.
    """

    actor: int
    street: Street
    action_key: ActionKey
    public_placement: PublicPlacement
    done: bool
    rewards: tuple[float, float]


@dataclass(frozen=True, repr=False)
class HuRlReferenceSnapshot:
    """Opaque simulator checkpoint; never a policy or replay input.

    Hidden fields use ``repr=False`` so ordinary logging cannot accidentally
    disclose the deck or private discards.  The type intentionally has no
    serialization method.
    """

    _explicit_deck: tuple[str, ...] = field(repr=False)
    _boards: tuple[Board, Board] = field(repr=False)
    _private_discards: tuple[tuple[str, ...], tuple[str, ...]] = field(repr=False)
    _decision_count: int
    _public_history: tuple[PublicPlacement, ...] = field(repr=False)

    def __repr__(self) -> str:
        return (
            "HuRlReferenceSnapshot("
            f"decision_count={self._decision_count}, hidden_state=<redacted>)"
        )


class HuRlReferenceEnv:
    """One deterministic, normal-hand HU Regular OFC reference environment."""

    def __init__(self, explicit_deck: Sequence[str]) -> None:
        self._explicit_deck: tuple[str, ...] = ()
        self._boards = (Board.from_rows(), Board.from_rows())
        self._private_discards: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
        self._decision_count = 0
        self._public_history: tuple[PublicPlacement, ...] = ()
        self.reset(explicit_deck)

    def reset(self, explicit_deck: Sequence[str] | None = None) -> HuRlActorViewV1:
        """Reset to T0-first and return its actor-safe view.

        Passing ``None`` reuses the previously validated deck.  Construction
        always requires an explicit 52-card permutation; this reference never
        owns an RNG or silently shuffles.
        """

        if explicit_deck is not None:
            self._explicit_deck = _validated_explicit_deck(explicit_deck)
        elif not self._explicit_deck:
            raise HuRlReferenceError("reset requires an explicit 52-card deck")
        self._boards = (Board.from_rows(), Board.from_rows())
        self._private_discards = ((), ())
        self._decision_count = 0
        self._public_history = ()
        self._validate_state()
        return self.observe()

    @property
    def decision_count(self) -> int:
        return self._decision_count

    @property
    def done(self) -> bool:
        return self._decision_count == len(_DECISION_SCHEDULE)

    @property
    def boards(self) -> tuple[Board, Board]:
        """The two public boards, in fixed first/second seat order."""

        return self._boards

    @property
    def public_history(self) -> tuple[PublicPlacement, ...]:
        return self._public_history

    def observe(self) -> HuRlActorViewV1:
        """Return the current policy-facing view with no hidden world fields."""

        self._require_active()
        street, actor = _DECISION_SCHEDULE[self._decision_count]
        world = WorldState(
            boards=self._boards,
            private_discards=self._private_discards,
            street=street,
            next_player=actor,
            scoring=_SCORING,
        )
        observation = world.observe(actor, self._current_deal())
        return HuRlActorViewV1.from_observation(
            observation,
            public_history=self._public_history,
        )

    def legal_mapping(self) -> LegalActionMappingV1:
        """Return the exact canonical index-to-ActionKey mapping."""

        return self.observe().legal_action_mapping

    def legal_actions(self) -> tuple[ActionKey, ...]:
        """Return every legal semantic action in canonical ActionKey order."""

        return self.legal_mapping().action_keys

    def step(self, selected: ActionKey) -> HuRlReferenceStep:
        """Apply one exact legal ActionKey; illegal keys never fall back."""

        self._require_active()
        if type(selected) is not ActionKey:
            raise TypeError("step requires an ActionKey")

        view = self.observe()
        actions = tuple(
            generate_canonical_actions(
                view.observation.hero_board,
                view.observation.dealt_cards,
            )
        )
        by_key = {action_key(candidate): candidate for candidate in actions}
        if len(by_key) != len(actions):  # defensive: canonical generator rejects this too.
            raise HuRlReferenceError("canonical legal actions contain duplicate ActionKeys")
        try:
            action = by_key[selected]
        except KeyError as exc:
            raise HuRlReferenceError(
                f"ActionKey is not legal at this decision: {selected.to_token()}"
            ) from exc

        street, actor = _DECISION_SCHEDULE[self._decision_count]
        seat: Seat = "first" if actor == 0 else "second"
        public_placement = PublicPlacement.from_action_key(
            street=street,
            acting_seat=seat,
            key=selected,
        )

        boards = list(self._boards)
        boards[actor] = boards[actor].place(action.placements)
        private_discards = [list(cards) for cards in self._private_discards]
        private_discards[actor].extend(action.discards)

        self._boards = (boards[0], boards[1])
        self._private_discards = (
            tuple(private_discards[0]),
            tuple(private_discards[1]),
        )
        self._public_history = (*self._public_history, public_placement)
        self._decision_count += 1
        self._validate_state()

        rewards = tuple(self.terminal_rewards()) if self.done else (0.0, 0.0)
        return HuRlReferenceStep(
            actor=actor,
            street=street,
            action_key=selected,
            public_placement=public_placement,
            done=self.done,
            rewards=(float(rewards[0]), float(rewards[1])),
        )

    def terminal_rewards(self) -> list[float]:
        """Return ``[p0, -p0]`` using the current 14-card FL bootstrap."""

        if not self.done:
            raise HuRlReferenceError("terminal rewards require a completed hand")
        if not all(board.is_complete() for board in self._boards):
            raise HuRlReferenceError("terminal boards are incomplete")
        score_p0, _board_score = terminal_score(
            self._boards[0],
            self._boards[1],
            fl_ev=DEFAULT_FL_EV,
        )
        return [float(score_p0), -float(score_p0)]

    def snapshot(self) -> HuRlReferenceSnapshot:
        """Capture a simulator-only checkpoint with hidden fields redacted in repr."""

        return HuRlReferenceSnapshot(
            _explicit_deck=self._explicit_deck,
            _boards=self._boards,
            _private_discards=self._private_discards,
            _decision_count=self._decision_count,
            _public_history=self._public_history,
        )

    def restore(self, snapshot: HuRlReferenceSnapshot) -> None:
        """Restore an exact simulator checkpoint and validate it fail-closed."""

        if type(snapshot) is not HuRlReferenceSnapshot:
            raise TypeError("restore requires a HuRlReferenceSnapshot")
        previous = self.snapshot()
        try:
            self._explicit_deck = _validated_explicit_deck(snapshot._explicit_deck)
            self._boards = snapshot._boards
            self._private_discards = snapshot._private_discards
            self._decision_count = snapshot._decision_count
            self._public_history = snapshot._public_history
            self._validate_state()
        except Exception:
            self._explicit_deck = previous._explicit_deck
            self._boards = previous._boards
            self._private_discards = previous._private_discards
            self._decision_count = previous._decision_count
            self._public_history = previous._public_history
            raise

    def _current_deal(self) -> tuple[str, ...]:
        self._require_active()
        start = _DEAL_STARTS[self._decision_count]
        size = _DEAL_SIZES[self._decision_count]
        return self._explicit_deck[start : start + size]

    def _require_active(self) -> None:
        if self.done:
            raise HuRlReferenceError("the hand is already terminal")

    def _validate_state(self) -> None:
        if type(self._decision_count) is not int or not 0 <= self._decision_count <= len(
            _DECISION_SCHEDULE
        ):
            raise HuRlReferenceError("decision_count is outside the 10-decision hand")
        if (
            type(self._boards) is not tuple
            or len(self._boards) != 2
            or any(type(board) is not Board for board in self._boards)
        ):
            raise HuRlReferenceError("state requires exactly two Board lanes")
        if (
            type(self._private_discards) is not tuple
            or len(self._private_discards) != 2
            or any(type(cards) is not tuple for cards in self._private_discards)
        ):
            raise HuRlReferenceError(
                "state requires exactly two private-discard tuple lanes"
            )
        if type(self._public_history) is not tuple:
            raise HuRlReferenceError("public history must be a tuple")
        try:
            for board in self._boards:
                board.validate()
        except ValueError as exc:
            raise HuRlReferenceError(f"invalid board state: {exc}") from exc
        if len(self._public_history) != self._decision_count:
            raise HuRlReferenceError("public history length disagrees with decision_count")
        if any(type(event) is not PublicPlacement for event in self._public_history):
            raise HuRlReferenceError("public history must contain only PublicPlacement")

        expected_board_counts = [0, 0]
        expected_discard_counts = [0, 0]
        expected_private_discards: list[list[str]] = [[], []]
        expected_history_rows = [
            {row: set() for row in ROWS},
            {row: set() for row in ROWS},
        ]
        for index, (street, actor) in enumerate(_DECISION_SCHEDULE[: self._decision_count]):
            expected_board_counts[actor] += 5 if street == "T0" else 2
            expected_discard_counts[actor] += 0 if street == "T0" else 1
            event = self._public_history[index]
            expected_seat: Seat = "first" if actor == 0 else "second"
            if (event.street, event.acting_seat) != (street, expected_seat):
                raise HuRlReferenceError("public history event order/seat changed")
            for row in ROWS:
                row_cards = set(event.cards(row))
                if not row_cards.issubset(getattr(self._boards[actor], row)):
                    raise HuRlReferenceError(
                        "public history placement is on the wrong actor board row"
                    )
                expected_history_rows[actor][row].update(row_cards)
            placement_cards = tuple(
                card
                for row in ("top", "middle", "bottom")
                for card in event.cards(row)
            )
            deal_start = _DEAL_STARTS[index]
            dealt_cards = self._explicit_deck[
                deal_start : deal_start + _DEAL_SIZES[index]
            ]
            if not set(placement_cards).issubset(dealt_cards):
                raise HuRlReferenceError(
                    "public history placement violates its per-decision deal"
                )
            discarded = tuple(card for card in dealt_cards if card not in placement_cards)
            expected_discard_count = 0 if street == "T0" else 1
            if len(discarded) != expected_discard_count:
                raise HuRlReferenceError(
                    "public history does not consume its exact per-decision deal"
                )
            expected_private_discards[actor].extend(discarded)

        for actor in (0, 1):
            if self._boards[actor].card_count() != expected_board_counts[actor]:
                raise HuRlReferenceError("board geometry disagrees with decision_count")
            if len(self._private_discards[actor]) != expected_discard_counts[actor]:
                raise HuRlReferenceError(
                    "private-discard geometry disagrees with decision_count"
                )
            if self._private_discards[actor] != tuple(expected_private_discards[actor]):
                raise HuRlReferenceError(
                    "private discards disagree with the per-decision deals"
                )
            for row in ROWS:
                if expected_history_rows[actor][row] != set(
                    getattr(self._boards[actor], row)
                ):
                    raise HuRlReferenceError(
                        "public history does not exactly reconstruct every board row"
                    )

        placed_cards = tuple(
            card for board in self._boards for card in board.all_cards()
        )
        private_discards = (
            *self._private_discards[0],
            *self._private_discards[1],
        )
        validate_cards((*placed_cards, *private_discards))
        consumed_end = (
            _CARDS_USED_PER_NORMAL_HAND
            if self.done
            else _DEAL_STARTS[self._decision_count]
        )
        consumed = (*placed_cards, *private_discards)
        if len(consumed) != consumed_end or set(consumed) != set(
            self._explicit_deck[:consumed_end]
        ):
            raise HuRlReferenceError("completed actions violate deck-prefix conservation")

        history_cards = tuple(
            card
            for event in self._public_history
            for row in ("top", "middle", "bottom")
            for card in event.cards(row)
        )
        if len(history_cards) != len(placed_cards) or set(history_cards) != set(
            placed_cards
        ):
            raise HuRlReferenceError("public history disagrees with the public boards")


def _validated_explicit_deck(explicit_deck: Sequence[str]) -> tuple[str, ...]:
    if isinstance(explicit_deck, (str, bytes)):
        raise TypeError("explicit_deck must be a card sequence")
    deck = tuple(explicit_deck)
    if len(deck) != len(ALL_CARDS):
        raise HuRlReferenceError("explicit_deck must contain exactly 52 cards")
    try:
        validate_cards(deck)
    except ValueError as exc:
        raise HuRlReferenceError(f"invalid explicit_deck: {exc}") from exc
    if set(deck) != set(ALL_CARDS):  # length + validation already imply this.
        raise HuRlReferenceError("explicit_deck is not a complete regular deck")
    return deck
