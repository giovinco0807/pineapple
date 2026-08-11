"""Information-safe client serialization.

These functions are the only intended source for API match/hand payloads.
Internal state snapshots and repository rows contain private cards and must
never be returned directly.
"""

from __future__ import annotations

from typing import Any, Iterable

from ofc_regular.state import Board

from .domain import (
    HandResult,
    HandState,
    HandStatus,
    MatchState,
    MatchStatus,
    Player,
    board_to_dict,
    other_player,
    player_index,
)


def serialize_match(
    match: MatchState,
    *,
    hands: Iterable[HandState] = (),
) -> dict[str, Any]:
    """Return the public match view without its replay seed."""

    first_positions = match.positions_for_hand(0)
    next_positions = match.positions_for_hand(match.hand_count)
    summaries = [
        serialize_hand_summary(hand)
        for hand in hands
        if hand.match_id == match.id
    ]
    return {
        "id": match.id,
        "match_id": match.id,
        "created_at": match.created_at,
        "status": match.status.value,
        "stacks": {
            "human": match.stacks[0],
            "ai": match.stacks[1],
        },
        "first_hand_positions": {
            "human": first_positions[0],
            "ai": first_positions[1],
        },
        "next_hand_positions": {
            "human": next_positions[0],
            "ai": next_positions[1],
        },
        "hand_count": match.hand_count,
        "current_hand_id": match.current_hand_id,
        "fl_status": {
            "human": match.pending_fantasyland[0],
            "ai": match.pending_fantasyland[1],
            "cards": 14,
        },
        "can_start_hand": match.status == MatchStatus.READY,
        "can_continue": match.status == MatchStatus.AWAITING_CONTINUE,
        "continue_required": match.status == MatchStatus.AWAITING_CONTINUE,
        "hands": summaries,
    }


def serialize_hand(
    hand: HandState,
    *,
    viewer: Player = "human",
    ai_pending: bool | None = None,
) -> dict[str, Any]:
    """Return one player's safe hand view.

    The deck, future deals, the opponent's dealt cards/private discards, and an
    in-progress opponent FL board are intentionally omitted.
    """

    viewer_index = player_index(viewer)
    opponent = other_player(viewer)
    current = hand.current_turn
    visible_deal = (
        list(current.dealt_cards)
        if current is not None and current.actor == viewer
        else []
    )
    human_board = _visible_board(hand, player="human", viewer=viewer)
    ai_board = _visible_board(hand, player="ai", viewer=viewer)
    result = (
        _serialize_result(hand, hand.result)
        if hand.status == HandStatus.COMPLETE and hand.result is not None
        else None
    )
    pending = (
        current is not None and current.actor == "ai"
        if ai_pending is None
        else bool(ai_pending)
    )
    return {
        "id": hand.id,
        "hand_id": hand.id,
        "match_id": hand.match_id,
        "index": hand.index,
        "status": hand.status.value,
        "positions": {
            "human": hand.positions[0],
            "ai": hand.positions[1],
        },
        "fl_status": {
            "human": hand.fantasyland[0],
            "ai": hand.fantasyland[1],
            "cards": 14,
        },
        "street": current.street if current is not None else None,
        "to_act": current.actor if current is not None else None,
        "boards": {
            "human": board_to_dict(human_board),
            "ai": board_to_dict(ai_board),
        },
        "dealt_cards": visible_deal,
        "private_discards": list(hand.private_discards[viewer_index]),
        "opponent_discard_count": _visible_opponent_discard_count(
            hand, opponent
        ),
        "action_required": (
            ("fl" if current.street == "FL" else "normal")
            if current is not None and current.actor == viewer
            else None
        ),
        "ai_pending": pending,
        "result": result,
    }


def serialize_hand_summary(hand: HandState) -> dict[str, Any]:
    """Return replay-list metadata without private in-progress state."""

    payload: dict[str, Any] = {
        "id": hand.id,
        "hand_id": hand.id,
        "index": hand.index,
        "status": hand.status.value,
        "positions": {
            "human": hand.positions[0],
            "ai": hand.positions[1],
        },
        "fl_status": {
            "human": hand.fantasyland[0],
            "ai": hand.fantasyland[1],
            "cards": 14,
        },
        "started_at": hand.started_at,
        "ended_at": hand.ended_at,
        "result": None,
    }
    if hand.status == HandStatus.COMPLETE and hand.result is not None:
        payload["result"] = _serialize_result(hand, hand.result)
    return payload


def _visible_board(
    hand: HandState, *, player: Player, viewer: Player
) -> Board:
    if (
        player != viewer
        and hand.status != HandStatus.COMPLETE
        and hand.in_fantasyland(player)
    ):
        return Board()
    return hand.board_for(player)


def _visible_opponent_discard_count(
    hand: HandState, opponent: Player
) -> int:
    if (
        hand.status != HandStatus.COMPLETE
        and hand.in_fantasyland(opponent)
    ):
        return 0
    return len(hand.discards_for(opponent))


def _serialize_result(
    hand: HandState, result: HandResult
) -> dict[str, Any]:
    human_is_first = hand.positions[0] == "first"
    human_raw = result.raw_score if human_is_first else -result.raw_score
    human_capped = (
        result.capped_score if human_is_first else -result.capped_score
    )
    breakdown = dict(result.breakdown)
    public_breakdown = _humanize_breakdown(
        breakdown, human_is_first=human_is_first
    )
    return {
        "raw_score": result.raw_score,
        "capped_score": result.capped_score,
        "human_raw_score": human_raw,
        "human_capped_score": human_capped,
        "breakdown": breakdown,
        **public_breakdown,
        "stacks_after": {
            "human": result.stacks_after[0],
            "ai": result.stacks_after[1],
        },
        "fl_entries": {
            "human": result.next_fantasyland[0],
            "ai": result.next_fantasyland[1],
            "cards": 14,
        },
    }


def _humanize_breakdown(
    breakdown: dict[str, Any], *, human_is_first: bool
) -> dict[str, Any]:
    """Expose first/second engine details using human/AI labels for the UI."""

    def player_for(value: Any) -> Any:
        if value == "tie" or value is None:
            return value
        if value == "first":
            return "human" if human_is_first else "ai"
        if value == "second":
            return "ai" if human_is_first else "human"
        return value

    raw_rows = breakdown.get("row_results", {})
    row_wins = {
        row: player_for(raw_rows.get(row))
        for row in ("top", "middle", "bottom")
    }
    raw_scoop = breakdown.get("scoop", {})
    scoop: str | None = None
    if raw_scoop.get("first"):
        scoop = player_for("first")
    elif raw_scoop.get("second"):
        scoop = player_for("second")

    raw_royalties = breakdown.get("royalties", {})
    raw_fouls = breakdown.get("fouls", {})
    raw_components = breakdown.get("point_components")
    point_components: dict[str, Any] | None = None
    if isinstance(raw_components, dict):
        component_names = (
            "foul_base",
            "line_total",
            "scoop_bonus",
            "royalty_delta",
            "total",
        )
        if all(
            isinstance(raw_components.get(name), (int, float))
            and not isinstance(raw_components.get(name), bool)
            for name in component_names
        ):
            sign = 1 if human_is_first else -1
            point_components = {
                "perspective": "human",
                **{
                    name: sign * raw_components[name]
                    for name in component_names
                },
            }
    first_label = "human" if human_is_first else "ai"
    second_label = "ai" if human_is_first else "human"
    return {
        "row_wins": row_wins,
        "scoop": scoop,
        "royalties": {
            first_label: raw_royalties.get("first", {}),
            second_label: raw_royalties.get("second", {}),
        },
        "fouls": {
            first_label: bool(raw_fouls.get("first", False)),
            second_label: bool(raw_fouls.get("second", False)),
        },
        "point_components": point_components,
    }


__all__ = [
    "serialize_hand",
    "serialize_hand_summary",
    "serialize_match",
]
