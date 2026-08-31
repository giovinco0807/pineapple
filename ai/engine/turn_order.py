"""Canonical heads-up action-order contract for normal OFC Pineapple hands.

The non-button seat acts first on every street and the button acts second after
seeing the first player's public placement.  Historical project code used
``btn`` as an alias for "first" in a few places; new code must use the explicit
``first``/``second`` semantics below and keep ``is_btn`` purely positional.
"""
from __future__ import annotations

from typing import Any

FIRST_POSITION = "bb"
SECOND_POSITION = "btn"
POSITIONS = (FIRST_POSITION, SECOND_POSITION)
POSITION_CONTRACT_VERSION = "bb_first_v1"

_FIRST_ALIASES = {
    "bb",
    "blind",
    "non_button",
    "non-button",
    "oop",
    "first",
    "senkou",
    "先行",
}
_SECOND_ALIASES = {
    "btn",
    "button",
    "ip",
    "second",
    "koukou",
    "後攻",
}


def normalize_is_btn(value: Any) -> bool | None:
    """Parse a positional flag without Python's unsafe ``bool('false')`` coercion."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in {"true", "1"}:
        return True
    if raw in {"false", "0"}:
        return False
    raise ValueError(f"is_btn must be a boolean or 0/1, got {value!r}")


def normalize_position(value: Any = None, *, is_btn: Any = None) -> str:
    """Return ``bb`` (first) or ``btn`` (second), rejecting contradictions."""
    raw = str(value or "").strip().lower()
    position: str | None
    if not raw:
        position = None
    elif raw in _FIRST_ALIASES:
        position = FIRST_POSITION
    elif raw in _SECOND_ALIASES:
        position = SECOND_POSITION
    else:
        raise ValueError(f"unknown HU position: {value!r}")

    flag = normalize_is_btn(is_btn)
    flag_position = None if flag is None else (SECOND_POSITION if flag else FIRST_POSITION)
    if position is not None and flag_position is not None and position != flag_position:
        raise ValueError(
            f"contradictory HU position: position={position!r}, is_btn={flag!r}"
        )
    return position or flag_position or FIRST_POSITION


def action_order(btn_seat: int) -> tuple[int, int]:
    """Return the normal-hand seat order: non-button first, button second."""
    button = int(btn_seat)
    if button not in (0, 1):
        raise ValueError(f"heads-up button seat must be 0 or 1, got {btn_seat!r}")
    return (1 - button, button)


def position_for_seat(seat: int, btn_seat: int) -> str:
    seat_value = int(seat)
    if seat_value not in (0, 1):
        raise ValueError(f"heads-up seat must be 0 or 1, got {seat!r}")
    button = int(btn_seat)
    if button not in (0, 1):
        raise ValueError(f"heads-up button seat must be 0 or 1, got {btn_seat!r}")
    return SECOND_POSITION if seat_value == button else FIRST_POSITION


def board_cards_before_turn(turn: int) -> int:
    """Number of public cards on the acting player's board before T0..T4."""
    turn_value = int(turn)
    if turn_value < 0 or turn_value > 4:
        raise ValueError(f"normal OFC turn must be 0 through 4, got {turn!r}")
    return 0 if turn_value == 0 else 5 + 2 * (turn_value - 1)


def expected_decision_board_counts(turn: int, position: Any) -> tuple[int, int]:
    """Return expected ``(hero, opponent)`` public-card counts at a decision."""
    canonical = normalize_position(position)
    hero = board_cards_before_turn(turn)
    opponent = hero if canonical == FIRST_POSITION else (5 if int(turn) == 0 else hero + 2)
    return hero, opponent


def validate_decision_board_counts(
    turn: int,
    position: Any,
    hero_cards: int,
    opponent_cards: int,
) -> None:
    expected = expected_decision_board_counts(turn, position)
    actual = (int(hero_cards), int(opponent_cards))
    if actual != expected:
        raise ValueError(
            f"T{int(turn)} {normalize_position(position)} decision requires "
            f"hero/opponent board counts {expected[0]}/{expected[1]}, got {actual[0]}/{actual[1]}"
        )
