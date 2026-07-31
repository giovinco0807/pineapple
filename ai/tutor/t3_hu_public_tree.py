"""Strict physical transitions for the reduced T3/T4 public tree.

This module covers only the transition skeleton

``t3_first (BB) -> t3_second (BTN) -> t4_first (BB)``
``-> t4_second (BTN) -> terminal``

under ``bb_first_v1``.  It does not run CFR, choose a strategy, enumerate the
full 54-card deck, or evaluate terminal utility.

Remaining-card contract
-----------------------

At a :class:`PublicTreeDecisionState`, ``particle.undealt_cards`` is a reduced
or complete physical remainder *after the current actor's three-card draw has
already been removed*.  Applying the current action therefore leaves that
remainder unchanged while moving the draw into the actor's perfect recall.
Resolving the next supplied chance outcome removes its three cards and creates
the next decision state.  Thus no dealt card remains available to a later
chance node.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

from ai.engine.action_space import Action, get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.exact_late import action_key, apply_action
from ai.tutor.t3_hu_public_cfr import (
    CardRows,
    InfoSetKey,
    JointParticle,
    PrivateRecall,
    PublicHistoryEntry,
)


ExactInput = Fraction | int | str
ROWS = ("top", "middle", "bottom")
ROW_LIMITS = (3, 5, 5)
VALID_CARDS = frozenset(ALL_CARDS)
SUPPORTED_PHASES = ("t3_first", "t3_second", "t4_first", "t4_second")
TERMINAL_SNAPSHOT = "terminal"
ACTION_TRANSITIONS: Mapping[str, tuple[str, str, int]] = {
    # completed phase: (next phase, acting actor, completed turn)
    "t3_first": ("t3_second", "bb", 3),
    "t3_second": ("t4_first", "btn", 3),
    "t4_first": ("t4_second", "bb", 4),
}
PHASE_DECISION: Mapping[str, tuple[str, int]] = {
    "t3_first": ("bb", 3),
    "t3_second": ("btn", 3),
    "t4_first": ("bb", 4),
    "t4_second": ("btn", 4),
}
PHASE_BOARD_COUNTS: Mapping[str, tuple[int, int]] = {
    "t3_first": (9, 9),
    "t3_second": (11, 9),
    "t4_first": (11, 11),
    "t4_second": (13, 11),
}
PHASE_LAST_ACTION: Mapping[str, tuple[int, str]] = {
    "t3_first": (2, "btn"),
    "t3_second": (3, "bb"),
    "t4_first": (3, "btn"),
    "t4_second": (4, "bb"),
}
SNAPSHOT_BOARD_COUNTS: Mapping[str, tuple[int, int]] = {
    **PHASE_BOARD_COUNTS,
    TERMINAL_SNAPSHOT: (13, 13),
}
SNAPSHOT_LAST_ACTION: Mapping[str, tuple[int, str]] = {
    **PHASE_LAST_ACTION,
    TERMINAL_SNAPSHOT: (4, "btn"),
}


def _exact(value: ExactInput, *, label: str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError(f"{label} must be an exact Fraction/int/str")
    try:
        return value if isinstance(value, Fraction) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise TypeError(f"{label} is not an exact rational: {value!r}") from exc


def _canonical_cards(cards: Iterable[str], *, label: str) -> tuple[str, ...]:
    canonical = tuple(sorted(str(card) for card in cards))
    invalid = sorted(card for card in canonical if card not in VALID_CARDS)
    if invalid:
        raise ValueError(f"{label} contains invalid physical cards: {invalid}")
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"{label} contains duplicate physical cards")
    return canonical


def _board_from_rows(rows: CardRows) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def _rows_from_board(board: Board) -> CardRows:
    return (
        tuple(sorted(board.top)),
        tuple(sorted(board.middle)),
        tuple(sorted(board.bottom)),
    )


def _expected_history_pairs(phase: str) -> list[tuple[int, str]]:
    last_turn, last_actor = SNAPSHOT_LAST_ACTION[phase]
    pairs: list[tuple[int, str]] = []
    for turn in range(last_turn + 1):
        pairs.append((turn, "bb"))
        if turn < last_turn or last_actor == "btn":
            pairs.append((turn, "btn"))
    return pairs


def _validate_completed_physical_state(
    *,
    next_phase: str,
    board_bb: CardRows,
    board_btn: CardRows,
    history: tuple[PublicHistoryEntry, ...],
    particle: JointParticle,
) -> None:
    """Validate the no-current-draw state between action and chance."""
    if next_phase not in SNAPSHOT_BOARD_COUNTS:
        raise ValueError(f"unsupported pending next phase: {next_phase!r}")
    actual_pairs = [(turn, actor) for turn, actor, _placements in history]
    if actual_pairs != _expected_history_pairs(next_phase):
        raise ValueError(
            f"pending {next_phase} history has wrong action order: {actual_pairs}"
        )

    reconstructed = {
        "bb": {row: [] for row in ROWS},
        "btn": {row: [] for row in ROWS},
    }
    public_by_actor_turn: dict[str, dict[int, set[str]]] = {"bb": {}, "btn": {}}
    seen_public: set[str] = set()
    for turn, actor, placements in history:
        expected_count = 5 if turn == 0 else 2
        if len(placements) != expected_count:
            raise ValueError(
                f"T{turn} {actor} history requires {expected_count} placements"
            )
        turn_cards: set[str] = set()
        for card, row in placements:
            if card not in VALID_CARDS or row not in ROWS:
                raise ValueError("pending history contains an invalid card or row")
            if card in seen_public or card in turn_cards:
                raise ValueError(f"public card {card!r} is placed more than once")
            seen_public.add(card)
            turn_cards.add(card)
            reconstructed[actor][row].append(card)
        public_by_actor_turn[actor][turn] = turn_cards

    canonical_boards = {
        "bb": tuple(
            tuple(sorted(reconstructed["bb"][row])) for row in ROWS
        ),
        "btn": tuple(
            tuple(sorted(reconstructed["btn"][row])) for row in ROWS
        ),
    }
    if canonical_boards["bb"] != board_bb or canonical_boards["btn"] != board_btn:
        raise ValueError("pending boards do not match placements-only public history")
    for label, board in (("bb", board_bb), ("btn", board_btn)):
        lengths = tuple(len(row) for row in board)
        if any(actual > limit for actual, limit in zip(lengths, ROW_LIMITS)):
            raise ValueError(f"{label} board exceeds row capacity: {lengths}")
    counts = (
        sum(len(row) for row in board_bb),
        sum(len(row) for row in board_btn),
    )
    if counts != SNAPSHOT_BOARD_COUNTS[next_phase]:
        raise ValueError(
            f"pending {next_phase} requires board counts "
            f"{SNAPSHOT_BOARD_COUNTS[next_phase]}, got {counts}"
        )

    all_discards: set[str] = set()
    for actor, recall in (("bb", particle.bb_recall), ("btn", particle.btn_recall)):
        expected_turns = tuple(
            sorted(turn for turn in public_by_actor_turn[actor] if turn > 0)
        )
        dealt_turns = tuple(turn for turn, _cards in recall.dealt_by_turn)
        discard_turns = tuple(turn for turn, _card in recall.discards_by_turn)
        if dealt_turns != expected_turns or discard_turns != expected_turns:
            raise ValueError(
                f"pending {actor} recall does not match completed turns {expected_turns}"
            )
        discard_by_turn = dict(recall.discards_by_turn)
        for turn, cards in recall.dealt_by_turn:
            expected_deal = public_by_actor_turn[actor][turn] | {
                discard_by_turn[turn]
            }
            if set(cards) != expected_deal:
                raise ValueError(
                    f"pending {actor} T{turn} recall is not placements plus discard"
                )
        discards = set(discard_by_turn.values())
        if discards & seen_public or discards & all_discards:
            raise ValueError("pending private discards are not physically unique")
        all_discards.update(discards)

    remaining = set(particle.undealt_cards)
    occupied = seen_public | all_discards
    overlap = remaining & occupied
    if overlap:
        raise ValueError(
            f"pending remaining cards overlap already dealt cards: {sorted(overlap)}"
        )


@dataclass(frozen=True)
class PublicTreeDecisionState:
    """One strict physical particle at a public-tree decision point."""

    infoset_key: InfoSetKey
    particle: JointParticle

    def __post_init__(self) -> None:
        key = self.infoset_key
        if key.contract_version != POSITION_CONTRACT_VERSION:
            raise ValueError("public tree requires bb_first_v1")
        if key.phase not in SUPPORTED_PHASES:
            raise ValueError(f"unsupported public-tree decision phase: {key.phase!r}")
        expected_actor, expected_turn = PHASE_DECISION[key.phase]
        if key.actor != expected_actor or key.turn != expected_turn:
            raise ValueError(f"decision key does not match phase {key.phase!r}")

        # Rebuilding through for_particle repeats all physical compatibility
        # checks while demonstrating that no physical-only field enters key.
        rebuilt = InfoSetKey.for_particle(
            self.particle,
            contract_version=key.contract_version,
            actor=key.actor,
            turn=key.turn,
            phase=key.phase,
            board_bb=key.board_bb,
            board_btn=key.board_btn,
            public_action_history=key.public_action_history,
            current_draw=key.current_draw,
            fantasy_state=key.fantasy_state,
        )
        if rebuilt != key:
            raise ValueError("decision key is not canonical for the supplied particle")

    @property
    def remaining_cards(self) -> tuple[str, ...]:
        """Cards available after the current private draw was removed."""
        return self.particle.undealt_cards

    @classmethod
    def from_particle(
        cls,
        particle: JointParticle,
        *,
        phase: str,
        board_bb: CardRows,
        board_btn: CardRows,
        public_action_history: tuple[PublicHistoryEntry, ...],
        current_draw: Sequence[str],
        fantasy_state: str | None = None,
    ) -> "PublicTreeDecisionState":
        if phase not in PHASE_DECISION:
            raise ValueError(f"unsupported public-tree decision phase: {phase!r}")
        actor, turn = PHASE_DECISION[phase]
        key = InfoSetKey.for_particle(
            particle,
            contract_version=POSITION_CONTRACT_VERSION,
            actor=actor,
            turn=turn,
            phase=phase,
            board_bb=board_bb,
            board_btn=board_btn,
            public_action_history=public_action_history,
            current_draw=current_draw,
            fantasy_state=fantasy_state,
        )
        return cls(infoset_key=key, particle=particle)


@dataclass(frozen=True)
class PendingChanceState:
    """Physical state after an action and before the next three-card draw."""

    completed_phase: str
    next_phase: str
    board_bb: CardRows
    board_btn: CardRows
    public_action_history: tuple[PublicHistoryEntry, ...]
    particle: JointParticle
    applied_action_key: str
    fantasy_state: str | None = None

    def __post_init__(self) -> None:
        expected = ACTION_TRANSITIONS.get(self.completed_phase)
        if expected is None or expected[0] != self.next_phase:
            raise ValueError("completed_phase and next_phase are not consecutive")
        if not self.applied_action_key:
            raise ValueError("applied_action_key must be non-empty")
        _validate_completed_physical_state(
            next_phase=self.next_phase,
            board_bb=self.board_bb,
            board_btn=self.board_btn,
            history=self.public_action_history,
            particle=self.particle,
        )

    @property
    def remaining_cards(self) -> tuple[str, ...]:
        """Cards available to the immediately following chance node."""
        return self.particle.undealt_cards


@dataclass(frozen=True)
class PublicTreeTerminalState:
    """Complete physical state after BTN's final T4 placement."""

    completed_phase: str
    board_bb: CardRows
    board_btn: CardRows
    public_action_history: tuple[PublicHistoryEntry, ...]
    particle: JointParticle
    applied_action_key: str
    fantasy_state: str | None = None

    def __post_init__(self) -> None:
        if self.completed_phase != "t4_second":
            raise ValueError("terminal public-tree state must complete t4_second")
        if not self.applied_action_key:
            raise ValueError("terminal applied_action_key must be non-empty")
        _validate_completed_physical_state(
            next_phase=TERMINAL_SNAPSHOT,
            board_bb=self.board_bb,
            board_btn=self.board_btn,
            history=self.public_action_history,
            particle=self.particle,
        )

    @property
    def remaining_cards(self) -> tuple[str, ...]:
        """Cards never dealt after both final T4 draws."""
        return self.particle.undealt_cards


@dataclass(frozen=True)
class SuppliedChanceDraw:
    """One exact externally supplied three-card chance outcome."""

    cards: tuple[str, ...]
    probability: Fraction

    def __init__(self, cards: Iterable[str], probability: ExactInput) -> None:
        canonical = _canonical_cards(cards, label="supplied chance draw")
        if len(canonical) != 3:
            raise ValueError("supplied chance draw must contain exactly three cards")
        exact_probability = _exact(probability, label="chance probability")
        if exact_probability <= 0:
            raise ValueError("chance probability must be positive")
        object.__setattr__(self, "cards", canonical)
        object.__setattr__(self, "probability", exact_probability)


@dataclass(frozen=True)
class PublicTreeChanceBranch:
    """One conditional chance branch and its next strict decision state."""

    probability: Fraction
    state: PublicTreeDecisionState


def _strict_legal_action(state: PublicTreeDecisionState, action: Action) -> Action:
    if not isinstance(action, Action):
        raise TypeError("public-tree transition requires an Action")
    if len(action.placements) != 2 or action.discard is None:
        raise ValueError("regular public-tree action requires two placements and one discard")
    placement_cards = [str(card) for card, _row in action.placements]
    placement_rows = [str(row) for _card, row in action.placements]
    if len(set(placement_cards)) != 2 or any(row not in ROWS for row in placement_rows):
        raise ValueError("action placements must contain two unique cards in valid rows")
    used_cards = set(placement_cards) | {str(action.discard)}
    if len(used_cards) != 3 or used_cards != set(state.infoset_key.current_draw):
        raise ValueError("action must place two and discard one from the exact current draw")

    actor_board = (
        state.infoset_key.board_bb
        if state.infoset_key.actor == "bb"
        else state.infoset_key.board_btn
    )
    legal = {
        action_key(candidate): candidate
        for candidate in get_turn_actions(
            list(state.infoset_key.current_draw), _board_from_rows(actor_board)
        )
    }
    canonical = legal.get(action_key(action))
    if canonical is None:
        raise ValueError("action is not legal for the current board capacity")
    return canonical


def _append_recall(
    recall: PrivateRecall,
    *,
    turn: int,
    draw: tuple[str, ...],
    discard: str,
) -> PrivateRecall:
    return PrivateRecall(
        dealt_by_turn=recall.dealt_by_turn + ((turn, draw),),
        discards_by_turn=recall.discards_by_turn + ((turn, discard),),
    )


def apply_public_tree_action(
    state: PublicTreeDecisionState,
    action: Action,
) -> PendingChanceState:
    """Apply a strict non-terminal action and return the pre-chance state."""
    key = state.infoset_key
    transition = ACTION_TRANSITIONS.get(key.phase)
    if transition is None:
        raise ValueError(f"phase {key.phase!r} has no transition in this prototype")
    next_phase, acting_actor, completed_turn = transition
    if key.actor != acting_actor or key.turn != completed_turn:
        raise ValueError("decision actor/turn does not match the transition contract")
    canonical_action = _strict_legal_action(state, action)

    if acting_actor == "bb":
        updated_board = apply_action(_board_from_rows(key.board_bb), canonical_action)
        board_bb = _rows_from_board(updated_board)
        board_btn = key.board_btn
        bb_recall = _append_recall(
            state.particle.bb_recall,
            turn=completed_turn,
            draw=key.current_draw,
            discard=str(canonical_action.discard),
        )
        btn_recall = state.particle.btn_recall
    else:
        updated_board = apply_action(_board_from_rows(key.board_btn), canonical_action)
        board_bb = key.board_bb
        board_btn = _rows_from_board(updated_board)
        bb_recall = state.particle.bb_recall
        btn_recall = _append_recall(
            state.particle.btn_recall,
            turn=completed_turn,
            draw=key.current_draw,
            discard=str(canonical_action.discard),
        )

    placements_only = tuple(
        sorted(
            ((str(card), str(row)) for card, row in canonical_action.placements),
            key=lambda item: (item[1], item[0]),
        )
    )
    history = key.public_action_history + (
        (completed_turn, acting_actor, placements_only),
    )
    particle = JointParticle(
        bb_recall=bb_recall,
        btn_recall=btn_recall,
        # The current draw was already absent at the decision point.
        undealt_cards=state.particle.undealt_cards,
        weight=state.particle.weight,
    )
    return PendingChanceState(
        completed_phase=key.phase,
        next_phase=next_phase,
        board_bb=board_bb,
        board_btn=board_btn,
        public_action_history=history,
        particle=particle,
        applied_action_key=action_key(canonical_action),
        fantasy_state=key.fantasy_state,
    )


def apply_public_tree_terminal_action(
    state: PublicTreeDecisionState,
    action: Action,
) -> PublicTreeTerminalState:
    """Apply BTN's final T4 action and return a validated 13/13 state."""
    key = state.infoset_key
    if key.phase != "t4_second" or key.actor != "btn" or key.turn != 4:
        raise ValueError("terminal public-tree action requires BTN t4_second")
    canonical_action = _strict_legal_action(state, action)

    updated_btn = apply_action(_board_from_rows(key.board_btn), canonical_action)
    board_btn = _rows_from_board(updated_btn)
    btn_recall = _append_recall(
        state.particle.btn_recall,
        turn=4,
        draw=key.current_draw,
        discard=str(canonical_action.discard),
    )
    placements_only = tuple(
        sorted(
            ((str(card), str(row)) for card, row in canonical_action.placements),
            key=lambda item: (item[1], item[0]),
        )
    )
    history = key.public_action_history + ((4, "btn", placements_only),)
    particle = JointParticle(
        bb_recall=state.particle.bb_recall,
        btn_recall=btn_recall,
        # The final BTN draw was already absent at the decision point.
        undealt_cards=state.particle.undealt_cards,
        weight=state.particle.weight,
    )
    return PublicTreeTerminalState(
        completed_phase="t4_second",
        board_bb=key.board_bb,
        board_btn=board_btn,
        public_action_history=history,
        particle=particle,
        applied_action_key=action_key(canonical_action),
        fantasy_state=key.fantasy_state,
    )


def resolve_supplied_chance(
    pending: PendingChanceState,
    outcomes: Iterable[SuppliedChanceDraw],
) -> tuple[PublicTreeChanceBranch, ...]:
    """Resolve exact supplied draws in stable card order.

    Probabilities are conditional on ``pending`` and must sum to exactly one.
    Alternative branches may share cards, but every individual draw must be a
    physically unique subset of the pending remainder.
    """
    supplied = tuple(outcomes)
    if not supplied:
        raise ValueError("at least one supplied chance outcome is required")
    if any(not isinstance(outcome, SuppliedChanceDraw) for outcome in supplied):
        raise TypeError("outcomes must contain SuppliedChanceDraw instances")
    if sum((outcome.probability for outcome in supplied), Fraction(0, 1)) != 1:
        raise ValueError("supplied chance probabilities must sum exactly to 1")
    if len({outcome.cards for outcome in supplied}) != len(supplied):
        raise ValueError("supplied chance draws must be unique")

    remaining_set = set(pending.particle.undealt_cards)
    actor, turn = PHASE_DECISION[pending.next_phase]
    branches: list[PublicTreeChanceBranch] = []
    for outcome in sorted(supplied, key=lambda item: item.cards):
        draw_set = set(outcome.cards)
        unavailable = draw_set - remaining_set
        if unavailable:
            raise ValueError(
                f"chance draw uses cards outside the physical remainder: {sorted(unavailable)}"
            )
        next_remaining = tuple(
            card
            for card in pending.particle.undealt_cards
            if card not in draw_set
        )
        particle = JointParticle(
            bb_recall=pending.particle.bb_recall,
            btn_recall=pending.particle.btn_recall,
            undealt_cards=next_remaining,
            weight=pending.particle.weight * outcome.probability,
        )
        key = InfoSetKey.for_particle(
            particle,
            contract_version=POSITION_CONTRACT_VERSION,
            actor=actor,
            turn=turn,
            phase=pending.next_phase,
            board_bb=pending.board_bb,
            board_btn=pending.board_btn,
            public_action_history=pending.public_action_history,
            current_draw=outcome.cards,
            fantasy_state=pending.fantasy_state,
        )
        branches.append(
            PublicTreeChanceBranch(
                probability=outcome.probability,
                state=PublicTreeDecisionState(infoset_key=key, particle=particle),
            )
        )
    return tuple(branches)


def transition_public_tree_action(
    state: PublicTreeDecisionState,
    action: Action,
    outcomes: Iterable[SuppliedChanceDraw],
) -> tuple[PublicTreeChanceBranch, ...]:
    """Apply one public action and its explicitly supplied next chance node."""
    return resolve_supplied_chance(apply_public_tree_action(state, action), outcomes)
