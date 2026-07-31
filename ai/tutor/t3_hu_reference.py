"""Small-deck root-grouped PIMC reference evaluator for T3 HU tests.

This intentionally favors clarity over runtime.  It fixes the chance/action
ordering for BB-first heads-up play and is used as a golden oracle on reduced
live-card sets.  Individual worlds are PIMC kernels; public decisions are made
only after the same root action has been averaged across all supplied worlds.
Future policies are still solved separately inside each world, so this detects
root-level strategy fusion only.  The public-tree CFR reference lives in
``ai.tutor.t3_hu_public_cfr``.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from ai.engine.action_space import Action, get_turn_actions
from ai.engine.encoding import Board
from ai.engine.turn_order import normalize_position, validate_decision_board_counts
from ai.tutor.exact_late import action_key, apply_action, board_card_count, terminal_metrics


BoardKey = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


def _board_key(board: Board) -> BoardKey:
    return (
        tuple(sorted(board.top)),
        tuple(sorted(board.middle)),
        tuple(sorted(board.bottom)),
    )


def _board_from_key(key: BoardKey) -> Board:
    return Board(top=list(key[0]), middle=list(key[1]), bottom=list(key[2]))


def _remaining_after_draw(live: Sequence[str], draw: Iterable[str]) -> tuple[str, ...]:
    removed = set(draw)
    return tuple(card for card in live if card not in removed)


@dataclass(frozen=True)
class PlayerView:
    """Only information legally available at the root T3 decision."""

    actor: str
    board_self: BoardKey
    board_opponent: BoardKey
    dealt_cards: tuple[str, ...]
    known_discards_self: tuple[str, ...] = ()
    turn: int = 3
    first_actor: str = "bb"
    public_action_history: tuple[tuple[int, str, tuple[tuple[str, str], ...]], ...] = ()

    @classmethod
    def from_boards(
        cls,
        *,
        actor: str,
        board_self: Board,
        board_opponent: Board,
        dealt_cards: Sequence[str],
        known_discards_self: Sequence[str] = (),
    ) -> "PlayerView":
        position = normalize_position(actor)
        validate_decision_board_counts(
            3,
            position,
            board_card_count(board_self),
            board_card_count(board_opponent),
        )
        if len(dealt_cards) != 3:
            raise ValueError(f"T3 requires exactly 3 dealt cards, got {len(dealt_cards)}")
        return cls(
            actor=position,
            board_self=_board_key(board_self),
            board_opponent=_board_key(board_opponent),
            dealt_cards=tuple(dealt_cards),
            known_discards_self=tuple(known_discards_self),
        )


@dataclass(frozen=True)
class BeliefWorld:
    """One reduced physical world; never expose this object to a policy."""

    live: tuple[str, ...]
    opponent_private_discards: tuple[str, ...]
    weight: Fraction = Fraction(1, 1)


@dataclass(frozen=True)
class PublicBeliefEvaluation:
    action_values: Mapping[str, float]
    actions: Mapping[str, Action]
    best_action_key: str
    best_action: Action
    best_value: float
    world_action_values: tuple[Mapping[str, float], ...]
    evaluation_scope: str = "hu_reduced_root_grouped_pimc"
    information_model: str = "explicit_worlds_root_grouped_future_pimc"
    root_strategy_fusion_free: bool = True
    future_strategy_fusion_free: bool = False
    equilibrium_approx: bool = False
    hu_exact: bool = False
    inner_t4_exact: bool = True


class ReducedT3HUSolver:
    """Exact reduced-deck evaluator with canonical BB-first action order."""

    @lru_cache(maxsize=None)
    def terminal_bb_score(self, bb: BoardKey, btn: BoardKey) -> float:
        return float(terminal_metrics(_board_from_key(bb), _board_from_key(btn))["score"])

    @lru_cache(maxsize=None)
    def t4_bb_value(self, bb: BoardKey, btn: BoardKey, live: tuple[str, ...]) -> float:
        """Return ``E[BB draw] max_BB E[BTN draw] min_BTN u_BB`` exactly."""
        bb_board = _board_from_key(bb)
        btn_board = _board_from_key(btn)
        if board_card_count(bb_board) != 11 or board_card_count(btn_board) != 11:
            raise ValueError("reduced T4 HU value requires two 11-card boards")
        if len(live) < 6:
            raise ValueError("reduced T4 HU value requires at least 6 live cards")

        draw_values: list[float] = []
        for bb_draw in combinations(live, 3):
            remaining = _remaining_after_draw(live, bb_draw)
            bb_actions = get_turn_actions(list(bb_draw), bb_board)
            if not bb_actions:
                raise ValueError("BB T4 draw has no legal actions")
            bb_action_values: list[float] = []
            for bb_action in bb_actions:
                final_bb = _board_key(apply_action(bb_board, bb_action))
                btn_draw_values: list[float] = []
                for btn_draw in combinations(remaining, 3):
                    btn_actions = get_turn_actions(list(btn_draw), btn_board)
                    if not btn_actions:
                        raise ValueError("BTN T4 draw has no legal actions")
                    # BTN maximizes its own value, hence minimizes BB utility.
                    btn_draw_values.append(
                        min(
                            self.terminal_bb_score(final_bb, _board_key(apply_action(btn_board, action)))
                            for action in btn_actions
                        )
                    )
                bb_action_values.append(sum(btn_draw_values) / len(btn_draw_values))
            draw_values.append(max(bb_action_values))
        return sum(draw_values) / len(draw_values)

    def _validate_world(self, view: PlayerView, world: BeliefWorld) -> None:
        known = {
            *view.board_self[0],
            *view.board_self[1],
            *view.board_self[2],
            *view.board_opponent[0],
            *view.board_opponent[1],
            *view.board_opponent[2],
            *view.dealt_cards,
            *view.known_discards_self,
            *world.opponent_private_discards,
        }
        if len(known) != (
            sum(len(row) for row in view.board_self)
            + sum(len(row) for row in view.board_opponent)
            + len(view.dealt_cards)
            + len(view.known_discards_self)
            + len(world.opponent_private_discards)
        ):
            raise ValueError("reduced T3 world contains duplicate known/dead cards")
        if known.intersection(world.live):
            raise ValueError("reduced T3 world redeals a board, dealt, or discarded card")
        if len(world.live) != len(set(world.live)):
            raise ValueError("reduced T3 world contains duplicate live cards")

    def world_action_values(self, view: PlayerView, world: BeliefWorld) -> dict[str, float]:
        """Evaluate every root action inside one reduced physical world."""
        self._validate_world(view, world)
        self_board = _board_from_key(view.board_self)
        opponent_board = _board_from_key(view.board_opponent)
        actions = get_turn_actions(list(view.dealt_cards), self_board)
        values: dict[str, float] = {}

        if view.actor == "btn":
            # BB has already made T3 public; BTN completes its 11-card board.
            for action in actions:
                btn11 = _board_key(apply_action(self_board, action))
                values[action_key(action)] = -self.t4_bb_value(
                    _board_key(opponent_board),
                    btn11,
                    tuple(world.live),
                )
            return values

        if view.actor != "bb":
            raise ValueError(f"unsupported T3 actor: {view.actor!r}")

        # BB acts at T3, then BTN sees BB's placement and makes its T3 action.
        for action in actions:
            bb11 = _board_key(apply_action(self_board, action))
            btn_draw_values: list[float] = []
            for btn_t3_draw in combinations(world.live, 3):
                remaining = _remaining_after_draw(world.live, btn_t3_draw)
                btn_actions = get_turn_actions(list(btn_t3_draw), opponent_board)
                if not btn_actions:
                    raise ValueError("BTN T3 draw has no legal actions")
                # BTN chooses one T3 action before either T4 draw is known.
                btn_draw_values.append(
                    min(
                        self.t4_bb_value(
                            bb11,
                            _board_key(apply_action(opponent_board, btn_action)),
                            remaining,
                        )
                        for btn_action in btn_actions
                    )
                )
            values[action_key(action)] = sum(btn_draw_values) / len(btn_draw_values)
        return values

    def evaluate_public_belief(
        self,
        view: PlayerView,
        worlds: Sequence[BeliefWorld],
    ) -> PublicBeliefEvaluation:
        """Average each root action across worlds before selecting one action."""
        if not worlds:
            raise ValueError("at least one belief world is required")
        total_weight = sum((world.weight for world in worlds), Fraction(0, 1))
        if total_weight <= 0:
            raise ValueError("belief-world weights must sum to a positive value")

        root_actions = get_turn_actions(list(view.dealt_cards), _board_from_key(view.board_self))
        actions_by_key = {action_key(action): action for action in root_actions}
        per_world = tuple(self.world_action_values(view, world) for world in worlds)
        expected: dict[str, float] = {}
        for key in actions_by_key:
            expected[key] = sum(
                float(world.weight / total_weight) * float(values[key])
                for world, values in zip(worlds, per_world)
            )
        best_key = max(expected, key=lambda key: (expected[key], key))
        return PublicBeliefEvaluation(
            action_values=expected,
            actions=actions_by_key,
            best_action_key=best_key,
            best_action=actions_by_key[best_key],
            best_value=expected[best_key],
            world_action_values=per_world,
        )


def values_by_discard(evaluation: PublicBeliefEvaluation) -> dict[str, float]:
    return {
        evaluation.actions[key].discard: value
        for key, value in evaluation.action_values.items()
    }


def world_values_by_discard(
    evaluation: PublicBeliefEvaluation,
) -> tuple[dict[str, float], ...]:
    return tuple(
        {
            evaluation.actions[key].discard: value
            for key, value in values.items()
        }
        for values in evaluation.world_action_values
    )
