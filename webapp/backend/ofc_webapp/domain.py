"""Pure domain state machine for the regular, no-joker HU web game.

The state machine owns dealing and turn progression but deliberately does not
know how AI decisions or final scores are produced.  Those capabilities are
provided through the protocols in :mod:`ofc_webapp.ports`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import random
import secrets
from typing import Any, Literal, Mapping, TypeAlias
from uuid import uuid4

from ofc_regular.action_space import (
    Action,
    generate_actions,
    generate_turn_actions,
)
from ofc_regular.cards import create_deck, validate_cards
from ofc_regular.hu_infoset import ActorObservation, ScoringContext
from ofc_regular.state import Board


Player: TypeAlias = Literal["human", "ai"]
Seat: TypeAlias = Literal["first", "second"]
Street: TypeAlias = Literal["T0", "T1", "T2", "T3", "T4", "FL"]

PLAYERS: tuple[Player, Player] = ("human", "ai")
NORMAL_STREETS: tuple[Street, ...] = ("T0", "T1", "T2", "T3", "T4")
REGULAR_FL_CARDS = 14
STARTING_STACK = 200


class MatchStatus(StrEnum):
    READY = "ready"
    IN_HAND = "in_hand"
    AWAITING_CONTINUE = "awaiting_continue"
    COMPLETED = "completed"


class HandStatus(StrEnum):
    PLAYING = "playing"
    AWAITING_SCORE = "awaiting_score"
    COMPLETE = "complete"


class DomainError(ValueError):
    """Base error for invalid state-machine requests."""


class InvalidTransition(DomainError):
    """The requested transition is not valid in the current state."""


class IllegalAction(DomainError):
    """A submitted placement is not one of the engine-generated legal moves."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def other_player(player: Player) -> Player:
    return "ai" if player == "human" else "human"


def player_index(player: Player) -> int:
    if player not in PLAYERS:
        raise DomainError(f"unknown player: {player!r}")
    return 0 if player == "human" else 1


def board_to_dict(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


@dataclass(frozen=True)
class MatchState:
    id: str
    created_at: str
    seed: int
    stacks: tuple[int, int] = (STARTING_STACK, STARTING_STACK)
    status: MatchStatus = MatchStatus.READY
    first_hand_first: Player = "human"
    assembly_sha: str = ""
    app_version: str = ""
    hand_count: int = 0
    current_hand_id: str | None = None
    pending_fantasyland: tuple[bool, bool] = (False, False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("match id is required")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not 0 <= self.seed <= (2**63 - 1):
            raise ValueError("seed must fit SQLite's non-negative signed integer")
        if len(self.stacks) != 2 or any(
            isinstance(stack, bool) or not isinstance(stack, int) or stack < 0
            for stack in self.stacks
        ):
            raise ValueError("stacks must contain two non-negative integers")
        if sum(self.stacks) != STARTING_STACK * 2:
            raise ValueError("HU stacks must conserve 400 total points")
        if self.first_hand_first not in PLAYERS:
            raise ValueError("first_hand_first must be human or ai")
        if self.hand_count < 0:
            raise ValueError("hand_count cannot be negative")
        if len(self.pending_fantasyland) != 2:
            raise ValueError("pending_fantasyland must contain two flags")
        if self.status == MatchStatus.IN_HAND and not self.current_hand_id:
            raise ValueError("an in-hand match requires current_hand_id")
        if self.status != MatchStatus.IN_HAND and self.current_hand_id is not None:
            raise ValueError("only an in-hand match may have current_hand_id")
        if self.status == MatchStatus.COMPLETED and 0 in self.stacks:
            object.__setattr__(self, "pending_fantasyland", (False, False))

    def stack_for(self, player: Player) -> int:
        return self.stacks[player_index(player)]

    def pending_fl_for(self, player: Player) -> bool:
        return self.pending_fantasyland[player_index(player)]

    def first_player_for_hand(self, hand_index: int | None = None) -> Player:
        index = self.hand_count if hand_index is None else hand_index
        if index < 0:
            raise ValueError("hand index cannot be negative")
        return (
            self.first_hand_first
            if index % 2 == 0
            else other_player(self.first_hand_first)
        )

    def positions_for_hand(
        self, hand_index: int | None = None
    ) -> tuple[Seat, Seat]:
        first = self.first_player_for_hand(hand_index)
        return ("first", "second") if first == "human" else ("second", "first")


@dataclass(frozen=True)
class Turn:
    actor: Player
    street: Street
    dealt_cards: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.actor not in PLAYERS:
            raise ValueError(f"invalid turn actor: {self.actor!r}")
        if self.street not in (*NORMAL_STREETS, "FL"):
            raise ValueError(f"invalid street: {self.street!r}")
        expected = (
            REGULAR_FL_CARDS
            if self.street == "FL"
            else 5
            if self.street == "T0"
            else 3
        )
        if len(self.dealt_cards) != expected:
            raise ValueError(
                f"{self.street} requires {expected} cards, got "
                f"{len(self.dealt_cards)}"
            )
        validate_cards(self.dealt_cards)


@dataclass(frozen=True)
class HandResult:
    """Authoritative engine result plus table-stakes settlement.

    ``raw_score`` and ``capped_score`` are both from the first seat's
    perspective.  The breakdown is copied from the score adapter and is never
    recomputed in this layer.
    """

    raw_score: int
    capped_score: int
    breakdown: Mapping[str, Any]
    stacks_after: tuple[int, int]
    next_fantasyland: tuple[bool, bool]


@dataclass(frozen=True)
class HandState:
    id: str
    match_id: str
    index: int
    positions: tuple[Seat, Seat]
    fantasyland: tuple[bool, bool]
    deck_order: tuple[str, ...]
    turns: tuple[Turn, ...]
    started_at: str
    boards: tuple[Board, Board] = field(
        default_factory=lambda: (Board(), Board())
    )
    private_discards: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    current_turn_index: int = 0
    status: HandStatus = HandStatus.PLAYING
    ended_at: str | None = None
    result: HandResult | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.match_id:
            raise ValueError("hand id and match id are required")
        if self.index < 0:
            raise ValueError("hand index cannot be negative")
        if len(self.positions) != 2 or set(self.positions) != {"first", "second"}:
            raise ValueError("positions must assign first and second exactly once")
        if len(self.fantasyland) != 2:
            raise ValueError("fantasyland must contain two flags")
        if len(self.boards) != 2 or len(self.private_discards) != 2:
            raise ValueError("hand state requires two boards and discard piles")
        validate_cards(self.deck_order)
        if len(self.deck_order) != 52:
            raise ValueError("deck_order must contain a full 52-card deck")
        dealt_cards = tuple(card for turn in self.turns for card in turn.dealt_cards)
        validate_cards(dealt_cards)
        for board in self.boards:
            board.validate()
        validate_cards(
            (
                *self.boards[0].all_cards(),
                *self.boards[1].all_cards(),
                *self.private_discards[0],
                *self.private_discards[1],
            )
        )
        if not set(
            (
                *self.boards[0].all_cards(),
                *self.boards[1].all_cards(),
                *self.private_discards[0],
                *self.private_discards[1],
            )
        ).issubset(dealt_cards):
            raise ValueError("placed and discarded cards must come from the deal")
        if not 0 <= self.current_turn_index <= len(self.turns):
            raise ValueError("current_turn_index is outside the turn list")
        completed_turns = self.turns[: self.current_turn_index]
        for player in PLAYERS:
            expected = {
                card
                for turn in completed_turns
                if turn.actor == player
                for card in turn.dealt_cards
            }
            actual = {
                *self.board_for(player).all_cards(),
                *self.discards_for(player),
            }
            if actual != expected:
                raise ValueError(
                    f"{player} board/discards do not match completed deals"
                )
        if self.status == HandStatus.PLAYING:
            if self.current_turn_index >= len(self.turns):
                raise ValueError("playing hand requires an outstanding turn")
            if self.result is not None or self.ended_at is not None:
                raise ValueError("playing hand cannot have a result")
        elif self.status == HandStatus.AWAITING_SCORE:
            if self.current_turn_index != len(self.turns):
                raise ValueError("awaiting-score hand must have completed all turns")
            if not all(board.is_complete() for board in self.boards):
                raise ValueError("awaiting-score hand requires complete boards")
            if self.result is not None or self.ended_at is not None:
                raise ValueError("awaiting-score hand cannot already have a result")
        elif self.status == HandStatus.COMPLETE:
            if self.result is None or self.ended_at is None:
                raise ValueError("complete hand requires result and ended_at")

    def board_for(self, player: Player) -> Board:
        return self.boards[player_index(player)]

    def discards_for(self, player: Player) -> tuple[str, ...]:
        return self.private_discards[player_index(player)]

    def seat_for(self, player: Player) -> Seat:
        return self.positions[player_index(player)]

    def in_fantasyland(self, player: Player) -> bool:
        return self.fantasyland[player_index(player)]

    @property
    def current_turn(self) -> Turn | None:
        if self.status != HandStatus.PLAYING:
            return None
        return self.turns[self.current_turn_index]


@dataclass(frozen=True)
class ActionSubmission:
    placements: tuple[tuple[str, str], ...]
    discards: tuple[str, ...] = ()

    @classmethod
    def from_parts(
        cls,
        placements: list[list[str]] | tuple[tuple[str, str], ...],
        discards: list[str] | tuple[str, ...] = (),
    ) -> "ActionSubmission":
        try:
            normalized = tuple((str(card), str(row)) for card, row in placements)
        except (TypeError, ValueError) as exc:
            raise IllegalAction(
                "placements must contain [card, row] pairs"
            ) from exc
        return cls(normalized, tuple(str(card) for card in discards))


@dataclass(frozen=True)
class AppliedDecision:
    actor: Player
    street: Street
    seat: Seat
    dealt_cards: tuple[str, ...]
    placements: tuple[tuple[str, str], ...]
    discards: tuple[str, ...]
    sequence_no: int


@dataclass(frozen=True)
class ActionTransition:
    hand: HandState
    decision: AppliedDecision


@dataclass(frozen=True)
class FinalScore:
    """Normalized output required from the authoritative score adapter.

    ``hu_score`` is from the first seat's perspective.  The adapter also
    supplies FL entry/stay decisions because this layer must not infer fouls or
    poker hand classes independently of the authoritative evaluator.
    """

    hu_score: int
    breakdown: Mapping[str, Any]
    first_next_fantasyland: bool = False
    second_next_fantasyland: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.hu_score, bool) or not isinstance(self.hu_score, int):
            raise ValueError("hu_score must be an integer table score")


@dataclass(frozen=True)
class HiddenFantasylandObservation:
    """Information-safe fallback for a normal player facing hidden FL.

    The existing ``ActorObservation`` rejects FL flags on a normal street, so
    this small adapter preserves the same safe fields without inventing a
    public opponent board.  AI adapters can explicitly route this case.
    """

    hero_board: Board
    opponent_public_board: Board
    dealt_cards: tuple[str, ...]
    hero_private_discards: tuple[str, ...]
    seat: Seat
    street: Street
    to_act_order: Seat
    scoring: ScoringContext
    hero_in_fantasyland: bool
    opponent_in_fantasyland: bool

    def __post_init__(self) -> None:
        if self.street == "FL":
            raise ValueError("hidden-FL fallback is only for normal streets")
        if not self.opponent_in_fantasyland:
            raise ValueError("hidden-FL fallback requires an FL opponent")
        if self.opponent_public_board.card_count() != 0:
            raise ValueError("an in-progress FL board must remain hidden")
        validate_cards(
            (
                *self.hero_board.all_cards(),
                *self.dealt_cards,
                *self.hero_private_discards,
            )
        )

    @property
    def opponent_discard_count(self) -> int:
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ofc_webapp.hidden_fantasyland_observation.v1",
            "hero_board": board_to_dict(self.hero_board),
            "opponent_public_board": board_to_dict(
                self.opponent_public_board
            ),
            "dealt_cards": list(self.dealt_cards),
            "hero_private_discards": list(self.hero_private_discards),
            "seat": self.seat,
            "street": self.street,
            "to_act_order": self.to_act_order,
            "scoring": self.scoring.to_dict(),
            "hero_in_fantasyland": self.hero_in_fantasyland,
            "opponent_in_fantasyland": True,
            "opponent_discard_count": 0,
        }


DecisionObservation: TypeAlias = (
    ActorObservation | HiddenFantasylandObservation
)


def create_match(
    *,
    seed: int | None = None,
    match_id: str | None = None,
    assembly_sha: str = "",
    app_version: str = "",
    created_at: str | None = None,
) -> MatchState:
    actual_seed = secrets.randbits(63) if seed is None else seed
    if isinstance(actual_seed, bool) or not isinstance(actual_seed, int):
        raise ValueError("seed must be an integer")
    if not 0 <= actual_seed <= (2**63 - 1):
        raise ValueError("seed must fit SQLite's non-negative signed integer")
    first_rng = random.Random(_derived_seed(actual_seed, -1))
    first: Player = PLAYERS[first_rng.randrange(2)]
    return MatchState(
        id=match_id or str(uuid4()),
        created_at=created_at or utc_now(),
        seed=actual_seed,
        first_hand_first=first,
        assembly_sha=assembly_sha,
        app_version=app_version,
    )


def start_hand(
    match: MatchState,
    *,
    hand_id: str | None = None,
    started_at: str | None = None,
) -> tuple[MatchState, HandState]:
    if match.status != MatchStatus.READY:
        raise InvalidTransition(
            f"cannot start a hand while match status is {match.status}"
        )
    if 0 in match.stacks:
        raise InvalidTransition("cannot start a hand after a stack reaches zero")

    hand_index = match.hand_count
    positions = match.positions_for_hand(hand_index)
    first = match.first_player_for_hand(hand_index)
    order = (first, other_player(first))
    rng = random.Random(_derived_seed(match.seed, hand_index))
    deck = tuple(create_deck(shuffle=True, rng=rng))
    cursor = 0
    turns: list[Turn] = []

    # Opening packets are allocated and acted in position order.  An FL packet
    # replaces that player's T0-T4 packets and remains private until showdown.
    for actor in order:
        in_fl = match.pending_fl_for(actor)
        street: Street = "FL" if in_fl else "T0"
        count = REGULAR_FL_CARDS if in_fl else 5
        dealt = deck[cursor : cursor + count]
        cursor += count
        turns.append(Turn(actor=actor, street=street, dealt_cards=dealt))

    for street in ("T1", "T2", "T3", "T4"):
        for actor in order:
            if match.pending_fl_for(actor):
                continue
            dealt = deck[cursor : cursor + 3]
            cursor += 3
            turns.append(Turn(actor=actor, street=street, dealt_cards=dealt))

    new_hand_id = hand_id or str(uuid4())
    hand = HandState(
        id=new_hand_id,
        match_id=match.id,
        index=hand_index,
        positions=positions,
        fantasyland=match.pending_fantasyland,
        deck_order=deck,
        turns=tuple(turns),
        started_at=started_at or utc_now(),
    )
    return (
        replace(
            match,
            status=MatchStatus.IN_HAND,
            current_hand_id=new_hand_id,
        ),
        hand,
    )


def legal_actions(hand: HandState) -> tuple[Action, ...]:
    turn = _require_current_turn(hand)
    if turn.street == "FL":
        raise InvalidTransition("FL placement uses submit_fantasyland")
    board = hand.board_for(turn.actor)
    if turn.street == "T0":
        generated = generate_actions(board, turn.dealt_cards)
    else:
        generated = generate_turn_actions(board, turn.dealt_cards)
    return tuple(generated)


def submit_normal_action(
    hand: HandState,
    submission: ActionSubmission,
    *,
    actor: Player | None = None,
) -> ActionTransition:
    turn = _require_current_turn(hand)
    if turn.street == "FL":
        raise InvalidTransition("current turn requires an FL placement")
    _require_actor(turn, actor)

    submitted_key = _action_key(submission.placements, submission.discards)
    legal = {
        _action_key(action.placements, action.discards): action
        for action in legal_actions(hand)
    }
    if submitted_key not in legal:
        raise IllegalAction("submitted placement is not a generated legal action")
    chosen = legal[submitted_key]
    return _apply_placement(
        hand,
        turn,
        placements=chosen.placements,
        discards=chosen.discards,
    )


def submit_fantasyland(
    hand: HandState,
    submission: ActionSubmission,
    *,
    actor: Player | None = None,
) -> ActionTransition:
    turn = _require_current_turn(hand)
    if turn.street != "FL":
        raise InvalidTransition("current turn is not a fantasyland placement")
    _require_actor(turn, actor)

    if len(turn.dealt_cards) != REGULAR_FL_CARDS:
        raise IllegalAction("regular fantasyland must deal exactly 14 cards")
    if len(submission.placements) != 13 or len(submission.discards) != 1:
        raise IllegalAction(
            "regular fantasyland requires 13 placements and one discard"
        )
    submitted_cards = tuple(card for card, _row in submission.placements)
    try:
        validate_cards((*submitted_cards, *submission.discards))
    except ValueError as exc:
        raise IllegalAction(str(exc)) from exc
    if set((*submitted_cards, *submission.discards)) != set(turn.dealt_cards):
        raise IllegalAction("FL placement must use every dealt card exactly once")
    try:
        board = Board().place(submission.placements)
    except ValueError as exc:
        raise IllegalAction(str(exc)) from exc
    if not board.is_complete():
        raise IllegalAction("FL placement must fill top 3, middle 5, bottom 5")
    return _apply_placement(
        hand,
        turn,
        placements=submission.placements,
        discards=submission.discards,
        replacement_board=board,
    )


def apply_action(
    hand: HandState,
    submission: ActionSubmission,
    *,
    actor: Player | None = None,
) -> ActionTransition:
    turn = _require_current_turn(hand)
    if turn.street == "FL":
        return submit_fantasyland(hand, submission, actor=actor)
    return submit_normal_action(hand, submission, actor=actor)


def build_observation(
    hand: HandState,
    *,
    actor: Player | None = None,
    scoring: ScoringContext | None = None,
) -> DecisionObservation:
    turn = _require_current_turn(hand)
    actual_actor = turn.actor if actor is None else actor
    _require_actor(turn, actual_actor)
    opponent = other_player(actual_actor)
    opponent_public = _opponent_public_board(hand, actual_actor)
    context = scoring or ScoringContext(fantasyland_cards=REGULAR_FL_CARDS)
    common = {
        "hero_board": hand.board_for(actual_actor),
        "opponent_public_board": opponent_public,
        "dealt_cards": turn.dealt_cards,
        "hero_private_discards": hand.discards_for(actual_actor),
        "seat": hand.seat_for(actual_actor),
        "street": turn.street,
        "to_act_order": hand.seat_for(actual_actor),
        "scoring": context,
        "hero_in_fantasyland": hand.in_fantasyland(actual_actor),
        "opponent_in_fantasyland": hand.in_fantasyland(opponent),
    }
    if turn.street != "FL" and hand.in_fantasyland(opponent):
        return HiddenFantasylandObservation(**common)
    return ActorObservation(**common)


def settle_hand(
    match: MatchState,
    hand: HandState,
    score: FinalScore,
    *,
    ended_at: str | None = None,
) -> tuple[MatchState, HandState]:
    if match.status != MatchStatus.IN_HAND or match.current_hand_id != hand.id:
        raise InvalidTransition("match does not identify this hand as active")
    if hand.status != HandStatus.AWAITING_SCORE:
        raise InvalidTransition("hand is not ready for final scoring")
    if hand.index != match.hand_count:
        raise InvalidTransition("hand index does not match match progression")

    first = "human" if hand.positions[0] == "first" else "ai"
    second = other_player(first)
    first_index = player_index(first)
    second_index = player_index(second)
    stacks = list(match.stacks)
    raw = score.hu_score
    if raw > 0:
        transfer = min(raw, stacks[second_index])
        stacks[first_index] += transfer
        stacks[second_index] -= transfer
        capped = transfer
    elif raw < 0:
        transfer = min(-raw, stacks[first_index])
        stacks[first_index] -= transfer
        stacks[second_index] += transfer
        capped = -transfer
    else:
        capped = 0

    next_by_player = [False, False]
    next_by_player[first_index] = bool(score.first_next_fantasyland)
    next_by_player[second_index] = bool(score.second_next_fantasyland)
    next_fl = (next_by_player[0], next_by_player[1])
    result = HandResult(
        raw_score=raw,
        capped_score=capped,
        breakdown=dict(score.breakdown),
        stacks_after=(stacks[0], stacks[1]),
        next_fantasyland=next_fl,
    )
    completed_hand = replace(
        hand,
        status=HandStatus.COMPLETE,
        ended_at=ended_at or utc_now(),
        result=result,
    )

    forced_end = 0 in stacks
    if forced_end:
        next_status = MatchStatus.COMPLETED
        pending_fl = (False, False)
    elif any(next_fl):
        next_status = MatchStatus.READY
        pending_fl = next_fl
    else:
        next_status = MatchStatus.AWAITING_CONTINUE
        pending_fl = (False, False)
    settled_match = replace(
        match,
        stacks=(stacks[0], stacks[1]),
        status=next_status,
        current_hand_id=None,
        pending_fantasyland=pending_fl,
        hand_count=match.hand_count + 1,
    )
    return settled_match, completed_hand


def continue_match(match: MatchState, *, should_continue: bool) -> MatchState:
    if match.status != MatchStatus.AWAITING_CONTINUE:
        raise InvalidTransition(
            "continue/finish is only available when neither player has FL"
        )
    return replace(
        match,
        status=(
            MatchStatus.READY if should_continue else MatchStatus.COMPLETED
        ),
    )


def _require_current_turn(hand: HandState) -> Turn:
    if hand.status != HandStatus.PLAYING or hand.current_turn is None:
        raise InvalidTransition("hand has no outstanding action")
    return hand.current_turn


def _require_actor(turn: Turn, actor: Player | None) -> None:
    if actor is not None and actor != turn.actor:
        raise InvalidTransition(
            f"it is {turn.actor}'s turn, not {actor}'s turn"
        )


def _apply_placement(
    hand: HandState,
    turn: Turn,
    *,
    placements: tuple[tuple[str, str], ...],
    discards: tuple[str, ...],
    replacement_board: Board | None = None,
) -> ActionTransition:
    index = player_index(turn.actor)
    boards = list(hand.boards)
    private_discards = list(hand.private_discards)
    try:
        boards[index] = (
            replacement_board
            if replacement_board is not None
            else boards[index].place(placements)
        )
        private_discards[index] = (
            *private_discards[index],
            *tuple(discards),
        )
    except ValueError as exc:
        raise IllegalAction(str(exc)) from exc
    next_turn_index = hand.current_turn_index + 1
    next_status = (
        HandStatus.AWAITING_SCORE
        if next_turn_index == len(hand.turns)
        else HandStatus.PLAYING
    )
    updated = replace(
        hand,
        boards=(boards[0], boards[1]),
        private_discards=(
            tuple(private_discards[0]),
            tuple(private_discards[1]),
        ),
        current_turn_index=next_turn_index,
        status=next_status,
    )
    decision = AppliedDecision(
        actor=turn.actor,
        street=turn.street,
        seat=hand.seat_for(turn.actor),
        dealt_cards=turn.dealt_cards,
        placements=tuple(placements),
        discards=tuple(discards),
        sequence_no=hand.current_turn_index,
    )
    return ActionTransition(hand=updated, decision=decision)


def _action_key(
    placements: tuple[tuple[str, str], ...],
    discards: tuple[str, ...],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    return tuple(sorted(placements)), tuple(sorted(discards))


def _opponent_public_board(hand: HandState, viewer: Player) -> Board:
    opponent = other_player(viewer)
    if hand.status != HandStatus.COMPLETE and hand.in_fantasyland(opponent):
        return Board()
    return hand.board_for(opponent)


def _derived_seed(match_seed: int, hand_index: int) -> int:
    material = f"ofc-webapp-v1:{match_seed}:{hand_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


__all__ = [
    "ActionSubmission",
    "ActionTransition",
    "AppliedDecision",
    "DecisionObservation",
    "DomainError",
    "FinalScore",
    "HandResult",
    "HandState",
    "HandStatus",
    "HiddenFantasylandObservation",
    "IllegalAction",
    "InvalidTransition",
    "MatchState",
    "MatchStatus",
    "NORMAL_STREETS",
    "PLAYERS",
    "Player",
    "REGULAR_FL_CARDS",
    "STARTING_STACK",
    "Seat",
    "Street",
    "Turn",
    "apply_action",
    "board_to_dict",
    "build_observation",
    "continue_match",
    "create_match",
    "legal_actions",
    "other_player",
    "player_index",
    "settle_hand",
    "start_hand",
    "submit_fantasyland",
    "submit_normal_action",
    "utc_now",
]
