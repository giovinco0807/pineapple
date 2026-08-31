"""
OFC Pineapple - Scoring Functions

Backend-specific scoring logic that builds on the core game engine.
Includes joker bust-prevention, full round scoring, FL stay checks,
session end detection, and hand name display.
"""
from typing import Optional

from .game_engine import (
    evaluate_hand, hand_category, check_fl_entry,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    evaluate_board_with_joker_constraint,
    evaluate_row_with_joker_constraint,
    _B, _B5,
)


def _evaluate_with_joker_constraint(cards: list, expected_count: int,
                                    max_value: int) -> int:
    """Compatibility wrapper around the canonical row evaluator."""
    _cards, value = evaluate_row_with_joker_constraint(
        list(cards), expected_count, max_value
    )
    return value


def hand_name_from_val(val: int, expected_count: int) -> str:
    """Get hand name from pre-computed encoded value (no re-evaluation)."""
    if val == 0:
        return "---"
    cat = hand_category(val)
    rank_names = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                  9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    r1 = (val // (_B ** 4)) % _B
    r_name = rank_names.get(r1, '?')
    if expected_count == 3:
        if cat == 3:
            return f"{r_name}のスリーカード"
        elif cat == 1:
            return f"{r_name}のペア"
        else:
            return "ハイカード"
    else:
        names = {8: "ストレートフラッシュ", 7: "フォーカード", 6: "フルハウス",
                 5: "フラッシュ", 4: "ストレート", 3: "スリーカード",
                 2: "ツーペア", 1: "ワンペア", 0: "ハイカード"}
        name = names.get(cat, "ハイカード")
        if cat == 8 and r1 == 14:
            name = "ロイヤルフラッシュ"
        return name


def check_fl_stay(board: dict, hand_vals: dict, current_fl_cards: int) -> tuple:
    """Check if a player already in FL qualifies to STAY.

    FL Stay conditions (any one triggers stay, preserving original card count):
    - Trips on top
    - Quads or better on bottom (cat >= 7)

    Returns (qualifies: bool, card_count: int)
    """
    # Check trips on top
    top_cat = hand_category(hand_vals["top"])
    if top_cat >= 3:  # Trips (cat 3 for 3-card hand)
        return True, current_fl_cards

    # Check Quads+ on bottom
    bot_cat = hand_category(hand_vals["bottom"])
    if bot_cat >= 7:  # Quads (7), Str Flush (8)
        return True, current_fl_cards

    return False, 0


def check_fl_stay_from_cards(top_cards: list, bottom_cards: list,
                             current_fl_cards: int,
                             middle_cards: Optional[list] = None) -> tuple:
    """Check FL stay from raw card lists (evaluates cards internally).

    Convenience wrapper for eval scripts and self-play code.
    Returns (qualifies: bool, card_count: int)
    """
    if middle_cards is not None:
        evaluated = evaluate_board_with_joker_constraint(
            list(top_cards), list(middle_cards), list(bottom_cards)
        )
        if evaluated["busted"]:
            return False, 0
        top_val = int(evaluated["values"]["top"])
        bot_val = int(evaluated["values"]["bottom"])
    else:
        # Backward-compatible fallback for callers that do not have a complete
        # middle row.  Complete-board callers must pass ``middle_cards`` so a
        # Joker on Top is constrained by the canonical Middle value.
        top_val = evaluate_hand(top_cards, 3)
        bot_val = evaluate_hand(bottom_cards, 5)
    hand_vals = {"top": top_val, "bottom": bot_val}
    return check_fl_stay({}, hand_vals, current_fl_cards)


def calculate_scores(game) -> dict:
    """Calculate hand scores with full implementation.

    Args:
        game: Object with attributes: boards (list of dicts with top/middle/bottom),
              is_fantasyland (list of bool), fl_card_count (list of int),
              chips (list of int), btn (int)

    Returns:
        dict with boards, busted, royalties, hand_names, line_results,
        scoop, raw_score, actual_score, chips, fl_entry, fl_card_count
    """
    fl_entry = [False, False]
    fl_card_count = [0, 0]
    busted = [False, False]
    royalties = [
        {"top": 0, "middle": 0, "bottom": 0, "total": 0},
        {"top": 0, "middle": 0, "bottom": 0, "total": 0}
    ]

    # Evaluate every complete board through the single canonical path.
    hand_values = [{}, {}]
    for seat in [0, 1]:
        board = game.boards[seat]
        evaluated = evaluate_board_with_joker_constraint(
            board["top"], board["middle"], board["bottom"]
        )
        hand_values[seat] = dict(evaluated["values"])
        busted[seat] = bool(evaluated["busted"])
        royalties[seat] = dict(evaluated["royalties"])

        if game.is_fantasyland[seat] and not busted[seat]:
            fl_s, fl_c = check_fl_stay(
                board, hand_values[seat], game.fl_card_count[seat]
            )
            fl_entry[seat] = fl_s
            fl_card_count[seat] = fl_c
            print(f"[DEBUG] Seat {seat} FL STAY check: stay={fl_s}, cards={fl_c}")
        elif not busted[seat]:
            fl_entry[seat] = bool(evaluated["fl_entry"])
            fl_card_count[seat] = int(evaluated["fl_card_count"])

    # Calculate line results (P0 perspective: +1 win, -1 loss, 0 tie)
    line_results = [0, 0, 0]
    if busted[0] and busted[1]:
        pass  # Both busted, no lines
    elif busted[0]:
        line_results = [-1, -1, -1]  # P0 loses all
    elif busted[1]:
        line_results = [1, 1, 1]  # P0 wins all
    else:
        # Compare each line
        for i, line in enumerate(["top", "middle", "bottom"]):
            if hand_values[0][line] > hand_values[1][line]:
                line_results[i] = 1
            elif hand_values[0][line] < hand_values[1][line]:
                line_results[i] = -1

    # Calculate total score
    line_total = sum(line_results)
    scoop = abs(line_total) == 3
    scoop_bonus = 3 if scoop else 0

    if busted[0] and not busted[1]:
        raw_score = [-6 - royalties[1]["total"], 6 + royalties[1]["total"]]
    elif busted[1] and not busted[0]:
        raw_score = [6 + royalties[0]["total"], -6 - royalties[0]["total"]]
    elif busted[0] and busted[1]:
        raw_score = [0, 0]
    else:
        p0_score = line_total + (scoop_bonus if line_total > 0 else -scoop_bonus if line_total < 0 else 0)
        p0_score += royalties[0]["total"] - royalties[1]["total"]
        raw_score = [p0_score, -p0_score]

    # Update chips (zero-sum: transfer capped by loser's available chips)
    old_chips = game.chips.copy()
    if raw_score[0] >= 0:
        # Seat 0 wins - transfer capped by seat 1's chips
        transfer = min(raw_score[0], old_chips[1])
        game.chips[0] = old_chips[0] + transfer
        game.chips[1] = old_chips[1] - transfer
    else:
        # Seat 1 wins - transfer capped by seat 0's chips
        transfer = min(raw_score[1], old_chips[0])
        game.chips[1] = old_chips[1] + transfer
        game.chips[0] = old_chips[0] - transfer
    actual_score = [game.chips[0] - old_chips[0], game.chips[1] - old_chips[1]]

    # Update FL state in game
    game.is_fantasyland = fl_entry
    game.fl_card_count = fl_card_count

    # Get hand names for display - use constrained values, not raw re-evaluation
    hand_names = [{}, {}]
    for seat in [0, 1]:
        hand_names[seat] = {
            "top": hand_name_from_val(hand_values[seat]["top"], 3),
            "middle": hand_name_from_val(hand_values[seat]["middle"], 5),
            "bottom": hand_name_from_val(hand_values[seat]["bottom"], 5)
        }

    result = {
        "boards": game.boards,
        "busted": busted,
        "royalties": royalties,
        "hand_names": hand_names,
        "line_results": line_results,
        "scoop": scoop,
        "raw_score": raw_score,
        "actual_score": actual_score,
        "chips": game.chips.copy(),
        "fl_entry": fl_entry,
        "fl_card_count": fl_card_count
    }
    print(f"[DEBUG] Score calculated: {result}")
    return result


def check_session_end(game) -> Optional[dict]:
    """Check if session should end.

    Bankruptcy always ends; chip_lead skipped during FL.

    Args:
        game: Object with attributes: chips (list of int),
              hands_played (int), is_fantasyland (list of bool)
    """
    # Bankruptcy always ends the game, even during FL
    if game.chips[0] <= 0:
        return {"winner": 1, "final_chips": game.chips, "hands_played": game.hands_played, "reason": "bankrupt"}
    if game.chips[1] <= 0:
        return {"winner": 0, "final_chips": game.chips, "hands_played": game.hands_played, "reason": "bankrupt"}

    # Don't end on chip_lead if either player qualifies for FL
    if game.is_fantasyland[0] or game.is_fantasyland[1]:
        return None

    if abs(game.chips[0] - game.chips[1]) >= 40:
        winner = 0 if game.chips[0] > game.chips[1] else 1
        return {"winner": winner, "final_chips": game.chips, "hands_played": game.hands_played, "reason": "chip_lead"}
    return None
