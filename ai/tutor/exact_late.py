"""Exact late-turn evaluator for OFC Pineapple tutor positions.

This module is intentionally independent from the Rust probability engine so
the tutor API can provide deterministic T3/T4 labels and tests can run without
launching a subprocess.  A complete opponent board is compared directly.  At
T4, an 11-card opponent board means Hero is acting first (BB), so every possible
opponent draw and the opponent's best legal reply are included in Hero's EV.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ai.engine.action_space import Action, get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import (
    POSITION_CONTRACT_VERSION,
    normalize_position,
    validate_decision_board_counts,
)
from ai.engine.game_engine import (
    check_fl_entry,
    evaluate_board_with_joker_constraint,
    evaluate_hand,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
)
from ai.tutor.fl_ev_table import FL_EV

ROWS = ("top", "middle", "bottom")
FL_TYPE_BY_CARD_COUNT = {14: "qq", 15: "kk", 16: "aa", 17: "trips"}
PUBLIC_CFR_T4_LEAF_SCHEMA = "ofc_uniform_public_range_t4_leaf/v1"
UNIFORM_T4_RANGE_MODEL = "uniform_hidden_discards_no_history_v1"
PHYSICAL_BB_T4_ACTION_VECTOR_SCHEMA = "ofc_physical_bb_t4_action_vector/v1"
PHYSICAL_T4_KNOWN_DISCARDS_RANGE_MODEL = "explicit_both_hidden_discard_sets_v1"
PHYSICAL_T4_REMAINING_CARDS_RANGE_MODEL = "explicit_remaining_cards_v1"
_PHYSICAL_CARD_SET = frozenset(ALL_CARDS)


class CardNormalizer:
    """Normalize legacy joker labels while preserving two unique jokers."""

    def __init__(self):
        self.used: set[str] = set()

    def card(self, card: str) -> str:
        if card in ("Xj", "JK"):
            if "X1" not in self.used:
                normalized = "X1"
            elif "X2" not in self.used:
                normalized = "X2"
            else:
                normalized = "X1"
        else:
            normalized = card
        self.used.add(normalized)
        return normalized

    def cards(self, cards: Iterable[str]) -> List[str]:
        return [self.card(str(card)) for card in cards if card]


def normalize_board(board: Optional[Dict[str, Any]], normalizer: Optional[CardNormalizer] = None) -> Board:
    normalizer = normalizer or CardNormalizer()
    board = board or {}
    return Board(
        top=normalizer.cards(board.get("top", []) or []),
        middle=normalizer.cards((board.get("middle", []) or []) + (board.get("mid", []) or [])),
        bottom=normalizer.cards((board.get("bottom", []) or []) + (board.get("bot", []) or [])),
    )


def board_to_dict(board: Board) -> Dict[str, List[str]]:
    return {"top": list(board.top), "middle": list(board.middle), "bottom": list(board.bottom)}


def board_card_count(board: Board) -> int:
    return len(board.top) + len(board.middle) + len(board.bottom)


def action_to_dict(action: Action) -> Dict[str, Any]:
    return {
        "placements": [[card, row] for card, row in action.placements],
        "discard": action.discard,
    }


def action_key(action: Action | Dict[str, Any]) -> str:
    if isinstance(action, Action):
        placements = action.placements
        discard = action.discard
    else:
        placements = action.get("placements") or []
        discard = action.get("discard")
    by_row: Dict[str, List[str]] = {"top": [], "middle": [], "bottom": []}
    row_alias = {"mid": "middle", "bot": "bottom"}
    for card, row in placements:
        row_name = row_alias.get(str(row), str(row))
        if row_name in by_row:
            by_row[row_name].append(str(card))
    normalized = {
        "placements": [
            [card, row]
            for row in ("top", "middle", "bottom")
            for card in sorted(by_row[row])
        ],
        "discard": None if discard is None else str(discard),
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _metric_rank_tuple(metrics: Dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("score", metrics.get("ev", 0.0)) or 0.0),
        -float(metrics.get("bust_rate", 1.0 if metrics.get("bust") else 0.0) or 0.0),
        float(metrics.get("fl_rate", 1.0 if metrics.get("fl_any") else 0.0) or 0.0),
        float(metrics.get("raw_score", 0.0) or 0.0),
    )


def _candidate_sort_key(candidate: Dict[str, Any]) -> tuple[float, float, float, float, str]:
    score, neg_bust_rate, fl_rate, raw_score = _metric_rank_tuple(candidate["metrics"])
    return (-score, -neg_bust_rate, -fl_rate, -raw_score, action_key(candidate["action"]))


def _candidate_is_better(candidate: Dict[str, Any], current: Dict[str, Any]) -> bool:
    candidate_rank = _metric_rank_tuple(candidate["metrics"])
    current_rank = _metric_rank_tuple(current["metrics"])
    if candidate_rank != current_rank:
        return candidate_rank > current_rank
    return action_key(candidate["action"]) < action_key(current["action"])


def filter_actions_by_payload(actions: List[Action], candidate_actions: Optional[Iterable[Dict[str, Any]]]) -> List[Action]:
    requested = list(candidate_actions or [])
    if not requested:
        return actions
    legal_by_key = {action_key(action): action for action in actions}
    out: List[Action] = []
    seen: set[str] = set()
    for payload in requested:
        key = action_key(payload)
        if key in seen:
            continue
        seen.add(key)
        action = legal_by_key.get(key)
        if action is not None:
            out.append(action)
    return out


def apply_action(board: Board, action: Action) -> Board:
    out = board.copy()
    for card, row in action.placements:
        getattr(out, row).append(card)
    return out


def _complete_opponent(opp_board: Board) -> bool:
    return len(opp_board.top) == 3 and len(opp_board.middle) == 5 and len(opp_board.bottom) == 5


def late_exact_scope(turn: int, opponent_board: Optional[Board]) -> Tuple[str, bool]:
    """Describe what is exact without overstating incomplete T3 HU play."""
    if turn == 3:
        return "t3_self_board_all_t4_draws_best_t4", False
    opponent_cards = board_card_count(opponent_board) if opponent_board is not None else 0
    if opponent_cards == 11:
        return "t4_all_opponent_draws_best_response_given_exclude", False
    if opponent_board is not None and _complete_opponent(opponent_board):
        return "t4_terminal_vs_complete_opponent", True
    return "t4_terminal_self_board", False


def _is_busted(board: Board) -> bool:
    if not board.is_complete():
        return False
    eval_res = evaluate_board_with_joker_constraint(board.top, board.middle, board.bottom)
    return bool(eval_res["busted"])


def _constrained_board(board: Board) -> Tuple[Board, bool, Dict[str, int]]:
    eval_res = evaluate_board_with_joker_constraint(board.top, board.middle, board.bottom)
    constrained = Board(
        top=list(eval_res["top"]),
        middle=list(eval_res["middle"]),
        bottom=list(eval_res["bottom"]),
    )
    hand_values = {
        "top": evaluate_hand(constrained.top, 3),
        "middle": evaluate_hand(constrained.middle, 5),
        "bottom": evaluate_hand(constrained.bottom, 5),
    }
    return constrained, bool(eval_res["busted"]), hand_values


def _board_royalty(board: Board, busted: bool) -> int:
    if busted:
        return 0
    return (
        get_top_royalty(board.top)
        + get_middle_royalty(board.middle)
        + get_bottom_royalty(board.bottom)
    )


def _fl_card_count(top_cards: List[str], busted: bool) -> int:
    if busted:
        return 0
    _qualified, card_count = check_fl_entry(top_cards)
    return card_count


def _score_against_complete_opponent(board: Board, opponent_board: Board, include_fl_ev: bool) -> float:
    my_board, my_busted, my_vals = _constrained_board(board)
    opp_board, opp_busted, opp_vals = _constrained_board(opponent_board)
    my_royalty = _board_royalty(my_board, my_busted)
    opp_royalty = _board_royalty(opp_board, opp_busted)

    if my_busted and opp_busted:
        score = 0.0
    elif my_busted:
        score = float(-6 - opp_royalty)
    elif opp_busted:
        score = float(6 + my_royalty)
    else:
        line_total = 0
        for line in ("top", "middle", "bottom"):
            if my_vals[line] > opp_vals[line]:
                line_total += 1
            elif my_vals[line] < opp_vals[line]:
                line_total -= 1
        scoop_bonus = 3 if abs(line_total) == 3 else 0
        score = float(line_total)
        score += scoop_bonus if line_total > 0 else (-scoop_bonus if line_total < 0 else 0)
        score += my_royalty - opp_royalty

    if include_fl_ev:
        my_fl = _fl_card_count(my_board.top, my_busted)
        opp_fl = _fl_card_count(opp_board.top, opp_busted)
        score += FL_EV.get(my_fl, 0)
        score -= FL_EV.get(opp_fl, 0)
    return float(score)


def terminal_metrics(board: Board, opponent_board: Optional[Board] = None) -> Dict[str, Any]:
    """Return exact metrics for a completed board."""
    if not board.is_complete():
        raise ValueError(f"terminal_metrics requires 13 cards, got {board_card_count(board)}")

    eval_board, busted, _hand_values = _constrained_board(board)
    fl_card_count = _fl_card_count(eval_board.top, busted)
    fl_type = FL_TYPE_BY_CARD_COUNT.get(fl_card_count)
    royalty = _board_royalty(eval_board, busted)

    if opponent_board is not None and _complete_opponent(opponent_board):
        score = _score_against_complete_opponent(board, opponent_board, include_fl_ev=True)
        raw_score = _score_against_complete_opponent(board, opponent_board, include_fl_ev=False)
    else:
        score = float(0 if busted else royalty + FL_EV.get(fl_card_count, 0))
        raw_score = float(0 if busted else royalty)

    return {
        "score": float(score),
        "raw_score": float(raw_score),
        "royalty": float(royalty),
        "bust": bool(busted),
        "fl_any": bool(fl_card_count),
        "fl_type": fl_type,
        "fl_card_count": fl_card_count,
    }


def _accumulate(total: Dict[str, float], metrics: Dict[str, Any]) -> None:
    total["score"] += float(metrics["score"])
    total["raw_score"] += float(metrics["raw_score"])
    total["royalty"] += float(metrics["royalty"])
    total["bust"] += 1.0 if metrics["bust"] else 0.0
    total["fl_any"] += 1.0 if metrics["fl_any"] else 0.0
    for fl_type in ("qq", "kk", "aa", "trips"):
        total[f"fl_{fl_type}"] += 1.0 if metrics["fl_type"] == fl_type else 0.0


def _averages(total: Dict[str, float], n: int, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    denom = max(n, 1)
    out = {
        "score": total["score"] / denom,
        "ev": total["score"] / denom,
        "raw_score": total["raw_score"] / denom,
        "royalty": total["royalty"] / denom,
        "bust_rate": total["bust"] / denom,
        "fl_rate": total["fl_any"] / denom,
        "fl_type_rates": {
            "qq": total["fl_qq"] / denom,
            "kk": total["fl_kk"] / denom,
            "aa": total["fl_aa"] / denom,
            "trips": total["fl_trips"] / denom,
        },
        "samples": n,
        "source": "exact",
    }
    if extra:
        out.update(extra)
    return out


def _remaining_deck(
    board: Board,
    opponent_board: Optional[Board] = None,
    dealt: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    used = set(board.all_cards())
    if opponent_board is not None:
        used.update(opponent_board.all_cards())
    if dealt:
        used.update(dealt)
    if exclude:
        used.update(exclude)
    return [card for card in ALL_CARDS if card not in used]


def remaining_deck_size(
    board: Board,
    opponent_board: Optional[Board] = None,
    dealt: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> int:
    return len(_remaining_deck(board, opponent_board=opponent_board, dealt=dealt, exclude=exclude))


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_rust_t3_exact_solver_path() -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return _repo_root() / "ai" / "rust_solver" / "target" / "release" / f"t3_exact_solver{suffix}"


def _hero_terminal_from_opponent_reply(
    hero_metrics: Dict[str, Any],
    opponent_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Mirror a best opponent terminal result while retaining Hero statistics."""
    return {
        "score": -float(opponent_metrics["score"]),
        "raw_score": -float(opponent_metrics["raw_score"]),
        "royalty": float(hero_metrics["royalty"]),
        "bust": bool(hero_metrics["bust"]),
        "fl_any": bool(hero_metrics["fl_any"]),
        "fl_type": hero_metrics.get("fl_type"),
        "fl_card_count": int(hero_metrics.get("fl_card_count", 0) or 0),
    }


def exact_t4_opponent_response_distribution(
    final_board: Board,
    opponent_board: Board,
    exclude: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Score a completed BB board against every legal BTN T4 response.

    ``opponent_board`` must contain the 11 public cards visible before BTN's
    final draw.  For each remaining three-card draw BTN chooses the placement
    that maximizes BTN's canonical heads-up score; Hero receives its negation.
    """
    if not final_board.is_complete():
        raise ValueError("T4 opponent-response evaluation requires a complete hero board")
    if board_card_count(opponent_board) != 11:
        raise ValueError(
            "T4 opponent-response evaluation requires an 11-card opponent board, "
            f"got {board_card_count(opponent_board)}"
        )

    deck = _remaining_deck(final_board, opponent_board=opponent_board, exclude=exclude)
    hero_terminal = terminal_metrics(final_board)
    totals = {
        k: 0.0
        for k in (
            "score",
            "raw_score",
            "royalty",
            "bust",
            "fl_any",
            "fl_qq",
            "fl_kk",
            "fl_aa",
            "fl_trips",
        )
    }
    n = 0
    for draw in combinations(deck, 3):
        opponent_best = best_t4_completion(opponent_board, list(draw), final_board)
        hero_result = _hero_terminal_from_opponent_reply(hero_terminal, opponent_best["metrics"])
        _accumulate(totals, hero_result)
        n += 1
    if n == 0:
        raise ValueError(f"T4 opponent-response evaluation has fewer than 3 live cards ({len(deck)})")
    return _averages(
        totals,
        n,
        {
            "source": "exact_hu_response",
            "enumerated_draws": n,
            "remaining_deck_size": len(deck),
            "start_turn": 4,
            "opponent_response": True,
        },
    )


def best_t4_completion(board: Board, dealt: List[str], opponent_board: Optional[Board] = None) -> Dict[str, Any]:
    """Choose the best legal T4 placement for a concrete 3-card draw."""
    actions = get_turn_actions(dealt, board)
    if not actions:
        raise ValueError("No legal T4 actions")
    best: Optional[Dict[str, Any]] = None
    for action in actions:
        final_board = apply_action(board, action)
        metrics = terminal_metrics(final_board, opponent_board)
        candidate = {
            "action": action_to_dict(action),
            "board": board_to_dict(final_board),
            "metrics": metrics,
        }
        if best is None or _candidate_is_better(candidate, best):
            best = candidate
    return best or {}


def exact_t4_draw_distribution(
    board: Board,
    opponent_board: Optional[Board] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Enumerate every possible T4 draw from an 11-card hero board."""
    if board_card_count(board) != 11:
        raise ValueError(f"T4 draw distribution requires 11 board cards, got {board_card_count(board)}")

    deck = _remaining_deck(board, opponent_board=opponent_board, exclude=exclude)
    totals = {k: 0.0 for k in ("score", "raw_score", "royalty", "bust", "fl_any", "fl_qq", "fl_kk", "fl_aa", "fl_trips")}
    n = 0
    for draw in combinations(deck, 3):
        best = best_t4_completion(board, list(draw), opponent_board)
        _accumulate(totals, best["metrics"])
        n += 1
    return _averages(totals, n, {"enumerated_draws": n, "remaining_deck_size": len(deck), "start_turn": 4})


def exact_candidate_metrics(
    board: Board,
    dealt: List[str],
    action: Action,
    turn: int,
    opponent_board: Optional[Board] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Evaluate one T3 or T4 action exactly."""
    next_board = apply_action(board, action)
    next_exclude = list(exclude or [])
    if action.discard:
        next_exclude.append(action.discard)

    if turn == 4:
        opponent_cards = board_card_count(opponent_board) if opponent_board is not None else 0
        if opponent_cards == 11:
            return exact_t4_opponent_response_distribution(
                next_board,
                opponent_board,
                exclude=next_exclude,
            )
        if opponent_cards and not _complete_opponent(opponent_board):
            raise ValueError(
                "T4 opponent board must be empty, an 11-card pre-response board, "
                "or a complete 3/5/5 board"
            )
        metrics = terminal_metrics(next_board, opponent_board)
        return _averages(
            {
                "score": float(metrics["score"]),
                "raw_score": float(metrics["raw_score"]),
                "royalty": float(metrics["royalty"]),
                "bust": 1.0 if metrics["bust"] else 0.0,
                "fl_any": 1.0 if metrics["fl_any"] else 0.0,
                "fl_qq": 1.0 if metrics["fl_type"] == "qq" else 0.0,
                "fl_kk": 1.0 if metrics["fl_type"] == "kk" else 0.0,
                "fl_aa": 1.0 if metrics["fl_type"] == "aa" else 0.0,
                "fl_trips": 1.0 if metrics["fl_type"] == "trips" else 0.0,
            },
            1,
            {
                "enumerated_draws": 1,
                "remaining_deck_size": 0,
                "start_turn": 5,
                "opponent_response": False,
            },
        )

    if turn == 3:
        return exact_t4_draw_distribution(next_board, opponent_board=opponent_board, exclude=next_exclude)

    raise ValueError("exact_candidate_metrics only supports turns 3 and 4")


def evaluate_late_position_rust(
    board: Board,
    dealt: List[str],
    turn: int,
    opponent_board: Optional[Board] = None,
    exclude: Optional[Iterable[str]] = None,
    candidate_actions: Optional[Iterable[Dict[str, Any]]] = None,
    top_n: int = 20,
    rust_solver_path: str | Path | None = None,
    timeout_s: float = 5.0,
) -> Dict[str, Any]:
    """Evaluate a concrete T3/T4 position with the Rust exact solver.

    T3 enumerates Hero's possible T4 draws and best T4 completion.  T4 compares
    directly with a complete opponent or, for an 11-card opponent, enumerates
    every opponent draw and best legal response.
    """
    if turn not in (3, 4):
        raise ValueError("evaluate_late_position_rust only supports T3/T4")
    if len(dealt) != 3:
        raise ValueError(f"T{turn} Rust exact evaluation requires exactly 3 dealt cards")
    expected_cards = 9 if turn == 3 else 11
    if board_card_count(board) != expected_cards:
        raise ValueError(
            f"T{turn} Rust exact evaluation requires {expected_cards} board cards, "
            f"got {board_card_count(board)}"
        )

    solver = Path(rust_solver_path) if rust_solver_path is not None else default_rust_t3_exact_solver_path()
    if not solver.exists():
        raise FileNotFoundError(f"missing Rust T3 exact solver: {solver}")

    payload = {
        "turn": turn,
        "board": board_to_dict(board),
        "opponent_board": board_to_dict(opponent_board or Board()),
        "dealt": list(dealt),
        "exclude": list(exclude or []),
    }
    requested_actions = list(candidate_actions or [])
    if requested_actions:
        payload["candidate_actions"] = requested_actions
    with tempfile.TemporaryDirectory(prefix="ofc_t3_exact_") as tmp:
        input_path = Path(tmp) / "input.jsonl"
        output_path = Path(tmp) / "output.jsonl"
        input_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        cmd = [
            str(solver),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--top-n",
            str(max(0, int(top_n))),
        ]
        subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(0.25, float(timeout_s)),
        )
        lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Rust T3 exact solver produced no output")

    result = json.loads(lines[0])
    best = result.get("best")
    exact_scope, hu_exact = late_exact_scope(turn, opponent_board)
    return {
        "turn": turn,
        "board": board_to_dict(board),
        "dealt": list(dealt),
        "legal_actions": int(result.get("legal_actions") or 0),
        "evaluated_actions": int(result.get("evaluated_actions") or len(result.get("candidates") or [])),
        "requested_actions": int(result.get("requested_actions") or len(requested_actions)),
        "chosen_action": (best or {}).get("action"),
        "best": best,
        "candidates": list(result.get("candidates") or []),
        "candidate_count": int(result.get("evaluated_actions") or len(result.get("candidates") or [])),
        "source": "rust_exact",
        "exact_scope": exact_scope,
        "hu_exact": hu_exact,
        "elapsed_ms": float(result.get("elapsed_ms", 0.0) or 0.0),
        "rust_solver": str(solver),
    }


def evaluate_late_positions_rust_batch(
    payloads: Sequence[Dict[str, Any]],
    *,
    top_n: int = 1,
    rust_solver_path: str | Path | None = None,
    timeout_s: float = 120.0,
    position_parallel: bool = True,
) -> List[Dict[str, Any]]:
    """Evaluate many T2/T3/T4 payloads in one Rust process.

    This is the low-overhead leaf interface used by sampled HU search.  Payload
    order is preserved through Rust's ``record_index`` field.
    """
    records = list(payloads)
    if not records:
        return []
    solver = Path(rust_solver_path) if rust_solver_path is not None else default_rust_t3_exact_solver_path()
    if not solver.exists():
        raise FileNotFoundError(f"missing Rust late exact solver: {solver}")
    with tempfile.TemporaryDirectory(prefix="ofc_late_exact_batch_") as tmp:
        input_path = Path(tmp) / "input.jsonl"
        output_path = Path(tmp) / "output.jsonl"
        input_path.write_text(
            "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in records),
            encoding="utf-8",
        )
        cmd = [
            str(solver),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--top-n",
            str(max(0, int(top_n))),
        ]
        if position_parallel:
            cmd.append("--position-parallel")
        subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(0.25, float(timeout_s)),
        )
        results = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    results.sort(key=lambda result: int(result.get("record_index", -1)))
    if len(results) != len(records):
        raise RuntimeError(f"Rust late batch returned {len(results)}/{len(records)} records")
    for expected_index, result in enumerate(results):
        if int(result.get("record_index", -1)) != expected_index:
            raise RuntimeError(
                "Rust late batch record-index mismatch: "
                f"expected {expected_index}, got {result.get('record_index')}"
            )
    return results


def _required_payload_alias(payload: Dict[str, Any], names: Sequence[str], label: str) -> Any:
    present = [name for name in names if name in payload and payload[name] is not None]
    if not present:
        raise ValueError(f"public CFR BB T4 PlayerView is missing {label}")
    if len(present) > 1:
        raise ValueError(f"public CFR BB T4 PlayerView has ambiguous {label} aliases: {present}")
    return payload[present[0]]


def _canonical_public_board_payload(raw: Any, label: str) -> Dict[str, List[str]]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object with top/middle/bottom rows")
    if "middle" in raw and "mid" in raw:
        raise ValueError(f"{label} has ambiguous middle/mid rows")
    if "bottom" in raw and "bot" in raw:
        raise ValueError(f"{label} has ambiguous bottom/bot rows")

    def row(canonical: str, alias: Optional[str] = None) -> List[str]:
        value = raw.get(canonical, raw.get(alias, []) if alias else []) or []
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{label}.{canonical} must be a card list")
        return [str(card) for card in value]

    return {
        "top": row("top"),
        "middle": row("middle", "mid"),
        "bottom": row("bottom", "bot"),
    }


def _reject_non_public_cfr_fields(payload: Dict[str, Any]) -> None:
    for field in ("candidate_actions", "candidates", "source_candidate_top_k", "top_n"):
        if field in payload:
            raise ValueError(f"public CFR BB T4 leaf does not accept candidate filtering field {field!r}")
    if "exclude" in payload:
        raise ValueError(
            "ambiguous exclude is forbidden for public CFR leaves; use "
            "known_discards_self or public_exclude"
        )

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for raw_key, child in value.items():
                key = str(raw_key).strip().lower().replace("-", "_")
                child_path = f"{path}.{raw_key}" if path else str(raw_key)
                if not path and key == "known_discards_self":
                    continue
                if "discard" in key:
                    raise ValueError(
                        "private or ambiguous discard field is forbidden in public CFR leaves: "
                        f"{child_path}"
                    )
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload)


def _normalize_public_cfr_bb_t4_player_view(
    payload: Dict[str, Any],
    *,
    allow_synthetic_public_exclude: bool = False,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("public CFR BB T4 PlayerView must be an object")
    _reject_non_public_cfr_fields(payload)

    if "turn" not in payload or int(payload["turn"]) != 4:
        raise ValueError("public CFR BB T4 PlayerView requires turn=4")
    supplied_contract = payload.get("position_contract_version")
    if supplied_contract != POSITION_CONTRACT_VERSION:
        raise ValueError(
            "public CFR BB T4 PlayerView requires explicit position contract: "
            f"{supplied_contract!r}; expected {POSITION_CONTRACT_VERSION!r}"
        )

    positions: List[str] = []
    for field in ("actor", "position", "player_position"):
        if field in payload and payload[field] not in (None, ""):
            positions.append(normalize_position(payload[field]))
    if "is_btn" in payload and payload["is_btn"] is not None:
        positions.append(normalize_position(None, is_btn=payload["is_btn"]))
    if not positions:
        raise ValueError(
            "public CFR BB T4 PlayerView requires an explicit actor/position; "
            "11/11 board shapes are role-ambiguous"
        )
    if len(set(positions)) != 1:
        raise ValueError(f"contradictory public CFR position fields: {positions}")
    position = positions[0]
    if position != "bb":
        raise ValueError("public CFR T4 leaf adapter only accepts BB/first-actor decisions")
    if "first_actor" not in payload:
        raise ValueError("public CFR T4 leaf adapter requires explicit first_actor='bb'")
    if normalize_position(payload["first_actor"]) != "bb":
        raise ValueError("public CFR T4 leaf adapter requires first_actor='bb'")

    raw_board = _required_payload_alias(payload, ("board_self", "board"), "BB board")
    raw_opponent = _required_payload_alias(
        payload,
        ("board_opponent", "opponent_board"),
        "BTN board",
    )
    raw_dealt = _required_payload_alias(payload, ("dealt_cards", "dealt"), "BB dealt cards")
    raw_known_self = _required_payload_alias(
        payload,
        ("known_discards_self",),
        "BB known discards",
    )
    raw_public_exclude = payload.get("public_exclude") or []
    if not isinstance(raw_dealt, (list, tuple)):
        raise ValueError("BB dealt cards must be a card list")
    if not isinstance(raw_known_self, (list, tuple)):
        raise ValueError("known_discards_self must be a card list")
    if not isinstance(raw_public_exclude, (list, tuple)):
        raise ValueError("public_exclude must be a card list")

    normalizer = CardNormalizer()
    board = normalize_board(
        _canonical_public_board_payload(raw_board, "board_self"),
        normalizer,
    )
    opponent = normalize_board(
        _canonical_public_board_payload(raw_opponent, "board_opponent"),
        normalizer,
    )
    dealt = normalizer.cards(raw_dealt)
    known_self = normalizer.cards(raw_known_self)
    public_exclude = normalizer.cards(raw_public_exclude)
    if public_exclude and not allow_synthetic_public_exclude:
        raise ValueError(
            "public_exclude is disabled for production PlayerView inputs; "
            "pass allow_synthetic_public_exclude=True only for an explicit public deck variant"
        )

    for label, candidate_board in (("BB", board), ("BTN", opponent)):
        row_lengths = (
            len(candidate_board.top),
            len(candidate_board.middle),
            len(candidate_board.bottom),
        )
        if any(actual > limit for actual, limit in zip(row_lengths, (3, 5, 5))):
            raise ValueError(f"{label} board exceeds row capacity: {row_lengths}")
    validate_decision_board_counts(4, position, board_card_count(board), board_card_count(opponent))
    if len(dealt) != 3:
        raise ValueError(f"public CFR BB T4 PlayerView requires 3 dealt cards, got {len(dealt)}")
    if len(known_self) != 3:
        raise ValueError(
            "public CFR BB T4 PlayerView requires exactly 3 prior BB discards, "
            f"got {len(known_self)}"
        )

    zones: List[Tuple[str, Iterable[str]]] = [
        ("board_self", board.all_cards()),
        ("board_opponent", opponent.all_cards()),
        ("dealt_cards", dealt),
        ("known_discards_self", known_self),
        ("public_exclude", public_exclude),
    ]
    seen: Dict[str, str] = {}
    valid_cards = set(ALL_CARDS)
    for zone, cards in zones:
        for card in cards:
            if card not in valid_cards:
                raise ValueError(f"unknown physical card {card!r} in {zone}")
            prior = seen.get(card)
            if prior is not None:
                raise ValueError(f"duplicate physical card {card!r} in {prior} and {zone}")
            seen[card] = zone
    if len(valid_cards - set(seen)) < 6:
        raise ValueError(
            "public CFR BB T4 PlayerView requires at least 6 possible unseen cards "
            "for BTN's 3 hidden discards plus 3-card final draw"
        )

    legal_actions = get_turn_actions(dealt, board)
    if not legal_actions:
        raise ValueError("public CFR BB T4 PlayerView has no legal BB actions")
    return {
        "board": board,
        "opponent": opponent,
        "dealt": dealt,
        "known_self": known_self,
        "public_exclude": public_exclude,
        "legal_actions": legal_actions,
        "rust_payload": {
            "turn": 4,
            "board": board_to_dict(board),
            "opponent_board": board_to_dict(opponent),
            "dealt": list(dealt),
            "exclude": [*known_self, *public_exclude],
        },
    }


def _strict_public_cfr_t4_action_key(
    action: Action | Dict[str, Any],
    dealt: Sequence[str],
) -> str:
    """Validate the cross-language T4 action contract before keying it."""
    if isinstance(action, Action):
        placements = list(action.placements)
        discard = action.discard
    elif isinstance(action, dict):
        placements = action.get("placements")
        discard = action.get("discard")
    else:
        raise ValueError("T4 action must be an Action or object")
    if not isinstance(placements, (list, tuple)) or len(placements) != 2:
        raise ValueError("T4 action requires exactly 2 placements")
    normalized: List[Tuple[str, str]] = []
    for placement in placements:
        if not isinstance(placement, (list, tuple)) or len(placement) != 2:
            raise ValueError("T4 placement must be [card, row]")
        card, row = str(placement[0]), str(placement[1])
        if row not in {"top", "middle", "bottom"}:
            raise ValueError(f"T4 placement has invalid row {row!r}")
        normalized.append((card, row))
    if discard is None:
        raise ValueError("T4 action requires one discard")
    discard_card = str(discard)
    action_cards = [card for card, _row in normalized] + [discard_card]
    if len(set(action_cards)) != 3:
        raise ValueError("T4 action must consume three distinct cards")
    if set(action_cards) != set(dealt) or len(set(dealt)) != 3:
        raise ValueError("T4 action cards do not match the dealt cards")
    return action_key({"placements": normalized, "discard": discard_card})


def evaluate_public_cfr_bb_t4_leaves_rust_batch(
    payloads: Sequence[Dict[str, Any]],
    *,
    rust_solver_path: str | Path | None = None,
    timeout_s: float = 120.0,
    position_parallel: bool = True,
    allow_synthetic_public_exclude: bool = False,
) -> List[Dict[str, Any]]:
    """Return every uniform-range BB T4 action value without selecting an action.

    Inputs contain only the acting BB's private knowledge plus public state.
    Opponent discard identities and ambiguous ``exclude`` fields are rejected.
    Unknown BTN discards are marginalized as an exchangeable uniform range;
    a history-weighted public CFR must instead aggregate physical particle
    vectors outside this adapter.
    ``action_keys`` and ``metrics_vector`` are aligned in lexical action-key
    order, independent of Rust's EV-ranked candidate order.
    """
    normalized: List[Dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        try:
            normalized.append(
                _normalize_public_cfr_bb_t4_player_view(
                    payload,
                    allow_synthetic_public_exclude=allow_synthetic_public_exclude,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"public CFR BB T4 payload {index}: {exc}") from exc
    if not normalized:
        return []

    top_n = max(len(item["legal_actions"]) for item in normalized)
    rust_results = evaluate_late_positions_rust_batch(
        [item["rust_payload"] for item in normalized],
        top_n=top_n,
        rust_solver_path=rust_solver_path,
        timeout_s=timeout_s,
        position_parallel=position_parallel,
    )
    if len(rust_results) != len(normalized):
        raise RuntimeError(
            f"public CFR Rust leaf batch returned {len(rust_results)}/{len(normalized)} positions"
        )

    outputs: List[Dict[str, Any]] = []
    for index, (item, result) in enumerate(zip(normalized, rust_results)):
        expected_keys = {
            _strict_public_cfr_t4_action_key(action, item["dealt"])
            for action in item["legal_actions"]
        }
        candidates = list(result.get("candidates") or [])
        actual: Dict[str, Dict[str, Any]] = {}
        actions_by_key: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            action = candidate.get("action") or {}
            try:
                key = _strict_public_cfr_t4_action_key(action, item["dealt"])
            except ValueError as exc:
                raise RuntimeError(
                    f"public CFR Rust leaf {index} returned an invalid action: {exc}"
                ) from exc
            if key in actual:
                raise RuntimeError(f"public CFR Rust leaf {index} returned duplicate action {key}")
            metrics = dict(candidate.get("metrics") or {})
            if metrics.get("source") != "exact_hu_response":
                raise RuntimeError(
                    f"public CFR Rust leaf {index} action {key} has non-HU source "
                    f"{metrics.get('source')!r}"
                )
            if metrics.get("opponent_response") is not True or int(metrics.get("start_turn") or 0) != 4:
                raise RuntimeError(
                    f"public CFR Rust leaf {index} action {key} is not an exact T4 response leaf"
                )
            actual[key] = metrics
            actions_by_key[key] = action

        reported_legal = int(result.get("legal_actions") or 0)
        reported_evaluated = int(result.get("evaluated_actions") or 0)
        if reported_legal != len(expected_keys) or reported_evaluated != len(expected_keys):
            raise RuntimeError(
                f"public CFR Rust leaf {index} coverage counters are "
                f"{reported_evaluated}/{reported_legal}, expected {len(expected_keys)}"
            )
        if set(actual) != expected_keys:
            missing = sorted(expected_keys - set(actual))
            extra = sorted(set(actual) - expected_keys)
            raise RuntimeError(
                f"public CFR Rust leaf {index} action coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )

        stable_keys = sorted(expected_keys)
        stable_metrics = {key: actual[key] for key in stable_keys}
        stable_actions = {key: actions_by_key[key] for key in stable_keys}
        outputs.append(
            {
                "schema": PUBLIC_CFR_T4_LEAF_SCHEMA,
                "turn": 4,
                "actor": "bb",
                "position": "bb",
                "is_btn": False,
                "first_actor": "bb",
                "position_contract_version": POSITION_CONTRACT_VERSION,
                "candidate_scope": "all_legal",
                "selection_performed": False,
                "source": "rust_exact_uniform_public_range_leaf",
                "metrics_source": "exact_hu_response",
                "exact_scope": "t4_uniform_hidden_discard_marginal_all_draws_best_response",
                "range_model": UNIFORM_T4_RANGE_MODEL,
                "hidden_discard_marginal": "uniform_exchangeable",
                "uniform_range_exact": True,
                "terminal_response_exact": True,
                "hu_exact": False,
                "synthetic_public_exclude_used": bool(item["public_exclude"]),
                "legal_actions": len(stable_keys),
                "evaluated_actions": len(stable_keys),
                "candidate_count": len(stable_keys),
                "action_keys": stable_keys,
                "metrics_vector": [stable_metrics[key] for key in stable_keys],
                "metrics_by_action_key": stable_metrics,
                "actions_by_action_key": stable_actions,
                "board": board_to_dict(item["board"]),
                "opponent_board": board_to_dict(item["opponent"]),
                "dealt": list(item["dealt"]),
                "known_discards_self": list(item["known_self"]),
                "public_exclude": list(item["public_exclude"]),
                "elapsed_ms": float(result.get("elapsed_ms", 0.0) or 0.0),
            }
        )
    return outputs


def evaluate_public_cfr_bb_t4_leaf_rust(
    payload: Dict[str, Any],
    *,
    rust_solver_path: str | Path | None = None,
    timeout_s: float = 120.0,
    allow_synthetic_public_exclude: bool = False,
) -> Dict[str, Any]:
    """Single-position wrapper for :func:`evaluate_public_cfr_bb_t4_leaves_rust_batch`."""
    return evaluate_public_cfr_bb_t4_leaves_rust_batch(
        [payload],
        rust_solver_path=rust_solver_path,
        timeout_s=timeout_s,
        position_parallel=False,
        allow_synthetic_public_exclude=allow_synthetic_public_exclude,
    )[0]


_PHYSICAL_BB_T4_FIELDS = frozenset(
    {
        "turn",
        "actor",
        "is_btn",
        "first_actor",
        "position_contract_version",
        "particle_id",
        "board_self",
        "board_opponent",
        "dealt_cards",
        "known_discards_self",
        "known_discards_opponent",
        "remaining_cards",
    }
)


def _strict_physical_card_list(raw: Any, label: str) -> List[str]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{label} must be a physical card list")
    cards: List[str] = []
    for card in raw:
        if not isinstance(card, str):
            raise ValueError(f"{label} contains a non-string physical card")
        if card not in _PHYSICAL_CARD_SET:
            raise ValueError(
                f"unknown or ambiguous physical card {card!r} in {label}; "
                "Jokers must be identified as X1 or X2"
            )
        cards.append(card)
    return cards


def _strict_physical_board(raw: Any, label: str) -> Board:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object with top/middle/bottom rows")
    supplied_rows = set(raw)
    expected_rows = set(ROWS)
    if supplied_rows != expected_rows:
        missing = sorted(expected_rows - supplied_rows)
        extra = sorted(supplied_rows - expected_rows)
        raise ValueError(
            f"{label} must use exactly top/middle/bottom rows; "
            f"missing={missing}, extra={extra}"
        )
    return Board(
        top=_strict_physical_card_list(raw["top"], f"{label}.top"),
        middle=_strict_physical_card_list(raw["middle"], f"{label}.middle"),
        bottom=_strict_physical_card_list(raw["bottom"], f"{label}.bottom"),
    )


def _physical_bb_t4_state_commitment(
    board: Board,
    opponent: Board,
    dealt: Sequence[str],
    remaining_cards: Sequence[str],
) -> str:
    """Commit to the exact physical T4 utility query without exposing its range."""
    canonical = {
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "turn": 4,
        "actor": "bb",
        "board_self": {
            key: sorted(value) for key, value in board_to_dict(board).items()
        },
        "board_opponent": {
            key: sorted(value) for key, value in board_to_dict(opponent).items()
        },
        "dealt_cards": sorted(dealt),
        "remaining_cards": sorted(remaining_cards),
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_physical_bb_t4_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one private physical range before crossing the Rust boundary.

    Two mutually exclusive representations are accepted:

    * both players' three prior hidden discards, which imply the full live deck;
    * the complete set of cards that may form BTN's final draw.

    The second form is useful for reduced-deck/reference particles.  It is not
    a public PlayerView: private physical cards are deliberately accepted here
    and deliberately omitted from the returned action-vector record.
    """
    if not isinstance(payload, dict):
        raise ValueError("physical BB T4 particle must be an object")
    unsupported = sorted(set(payload) - _PHYSICAL_BB_T4_FIELDS)
    if unsupported:
        raise ValueError(f"physical BB T4 particle has unsupported fields: {unsupported}")

    if type(payload.get("turn")) is not int or payload.get("turn") != 4:
        raise ValueError("physical BB T4 particle requires integer turn=4")
    if payload.get("position_contract_version") != POSITION_CONTRACT_VERSION:
        raise ValueError(
            "physical BB T4 particle requires explicit position contract: "
            f"{payload.get('position_contract_version')!r}; "
            f"expected {POSITION_CONTRACT_VERSION!r}"
        )
    if payload.get("actor") != "bb":
        raise ValueError("physical T4 action vectors only accept canonical actor='bb'")
    if payload.get("is_btn") is not False:
        raise ValueError("physical BB T4 particle requires canonical is_btn=false")
    if payload.get("first_actor") != "bb":
        raise ValueError("physical BB T4 particle requires canonical first_actor='bb'")

    particle_id = payload.get("particle_id")
    if not isinstance(particle_id, str) or not particle_id.strip():
        raise ValueError("physical BB T4 particle requires a non-empty particle_id")

    for field in ("board_self", "board_opponent", "dealt_cards"):
        if field not in payload:
            raise ValueError(f"physical BB T4 particle is missing {field}")
    board = _strict_physical_board(payload["board_self"], "board_self")
    opponent = _strict_physical_board(payload["board_opponent"], "board_opponent")
    dealt = _strict_physical_card_list(payload["dealt_cards"], "dealt_cards")

    for label, candidate_board in (("BB", board), ("BTN", opponent)):
        row_lengths = (
            len(candidate_board.top),
            len(candidate_board.middle),
            len(candidate_board.bottom),
        )
        if any(actual > limit for actual, limit in zip(row_lengths, (3, 5, 5))):
            raise ValueError(f"{label} board exceeds row capacity: {row_lengths}")
    validate_decision_board_counts(4, "bb", board_card_count(board), board_card_count(opponent))
    if len(dealt) != 3:
        raise ValueError(f"physical BB T4 particle requires 3 dealt cards, got {len(dealt)}")

    has_self_discards = "known_discards_self" in payload
    has_opponent_discards = "known_discards_opponent" in payload
    has_remaining = "remaining_cards" in payload
    if has_remaining and (has_self_discards or has_opponent_discards):
        raise ValueError(
            "physical BB T4 particle must use either both hidden-discard sets "
            "or remaining_cards, not both"
        )
    if has_self_discards != has_opponent_discards:
        raise ValueError(
            "physical BB T4 particle requires both known_discards_self and "
            "known_discards_opponent"
        )
    if not has_remaining and not has_self_discards:
        raise ValueError(
            "physical BB T4 particle requires both hidden-discard sets or "
            "an explicit complete remaining_cards list"
        )

    zones: List[Tuple[str, Iterable[str]]] = [
        ("board_self", board.all_cards()),
        ("board_opponent", opponent.all_cards()),
        ("dealt_cards", dealt),
    ]
    if has_remaining:
        remaining = _strict_physical_card_list(payload["remaining_cards"], "remaining_cards")
        zones.append(("remaining_cards", remaining))
        range_model = PHYSICAL_T4_REMAINING_CARDS_RANGE_MODEL
    else:
        known_self = _strict_physical_card_list(
            payload["known_discards_self"], "known_discards_self"
        )
        known_opponent = _strict_physical_card_list(
            payload["known_discards_opponent"], "known_discards_opponent"
        )
        if len(known_self) != 3 or len(known_opponent) != 3:
            raise ValueError(
                "physical BB T4 hidden-discard mode requires exactly 3 prior "
                "discards for each player"
            )
        zones.extend(
            (
                ("known_discards_self", known_self),
                ("known_discards_opponent", known_opponent),
            )
        )
        remaining = []
        range_model = PHYSICAL_T4_KNOWN_DISCARDS_RANGE_MODEL

    seen: Dict[str, str] = {}
    for zone, cards in zones:
        for card in cards:
            prior = seen.get(card)
            if prior is not None:
                raise ValueError(f"duplicate physical card {card!r} in {prior} and {zone}")
            seen[card] = zone

    valid_cards = set(_PHYSICAL_CARD_SET)
    if has_remaining:
        remaining_count = len(remaining)
        # Every omitted card is explicitly dead.  This makes the Rust exclude
        # set complete rather than silently applying a uniform hidden range.
        exclude = sorted(valid_cards - set(seen))
        live_remaining = sorted(remaining)
    else:
        remaining_count = len(valid_cards - set(seen))
        exclude = [*known_self, *known_opponent]
        live_remaining = sorted(valid_cards - set(seen))
    if remaining_count < 3:
        raise ValueError(
            "physical BB T4 particle requires at least 3 BTN final-draw cards, "
            f"got {remaining_count}"
        )

    legal_actions = get_turn_actions(dealt, board)
    if not legal_actions:
        raise ValueError("physical BB T4 particle has no legal BB actions")
    return {
        "particle_id": particle_id,
        "board": board,
        "opponent": opponent,
        "dealt": dealt,
        "legal_actions": legal_actions,
        "range_model": range_model,
        "remaining_card_count": remaining_count,
        "physical_state_commitment": _physical_bb_t4_state_commitment(
            board,
            opponent,
            dealt,
            live_remaining,
        ),
        "rust_payload": {
            "turn": 4,
            "board": board_to_dict(board),
            "opponent_board": board_to_dict(opponent),
            "dealt": list(dealt),
            "exclude": exclude,
        },
    }


def physical_bb_t4_state_commitment(payload: Dict[str, Any]) -> str:
    """Return the commitment used to bind a physical leaf to an infoset/range."""
    return str(_normalize_physical_bb_t4_payload(payload)["physical_state_commitment"])


def evaluate_physical_bb_t4_action_vectors_rust_batch(
    payloads: Sequence[Dict[str, Any]],
    *,
    rust_solver_path: str | Path | None = None,
    timeout_s: float = 120.0,
    position_parallel: bool = True,
) -> List[Dict[str, Any]]:
    """Evaluate all BB T4 actions for explicit private physical ranges.

    Each result is a complete action-key-aligned metric vector for one
    conditioned physical particle/range.  This function never selects an
    action and never aggregates particles; callers must combine whole vectors
    at a shared public information set.  Exactness is limited to the supplied
    T4 physical range and BTN's optimal terminal response.  ``particle_id`` is
    echoed verbatim and must therefore be an opaque, non-secret caller handle;
    raw discard/remaining-card fields are never copied to the result.
    """
    normalized: List[Dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        try:
            normalized.append(_normalize_physical_bb_t4_payload(payload))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"physical BB T4 payload {index}: {exc}") from exc
    if not normalized:
        return []
    particle_ids = [item["particle_id"] for item in normalized]
    if len(particle_ids) != len(set(particle_ids)):
        raise ValueError("physical BB T4 particle_id values must be unique within a batch")

    top_n = max(len(item["legal_actions"]) for item in normalized)
    rust_results = evaluate_late_positions_rust_batch(
        [item["rust_payload"] for item in normalized],
        top_n=top_n,
        rust_solver_path=rust_solver_path,
        timeout_s=timeout_s,
        position_parallel=position_parallel,
    )
    if len(rust_results) != len(normalized):
        raise RuntimeError(
            f"physical BB T4 Rust batch returned {len(rust_results)}/{len(normalized)} positions"
        )

    outputs: List[Dict[str, Any]] = []
    for index, (item, result) in enumerate(zip(normalized, rust_results)):
        expected_keys = {
            _strict_public_cfr_t4_action_key(action, item["dealt"])
            for action in item["legal_actions"]
        }
        metrics_by_key: Dict[str, Dict[str, Any]] = {}
        actions_by_key: Dict[str, Dict[str, Any]] = {}
        for candidate in list(result.get("candidates") or []):
            action = candidate.get("action") or {}
            try:
                key = _strict_public_cfr_t4_action_key(action, item["dealt"])
            except ValueError as exc:
                raise RuntimeError(
                    f"physical BB T4 Rust result {index} returned an invalid action: {exc}"
                ) from exc
            if key in metrics_by_key:
                raise RuntimeError(
                    f"physical BB T4 Rust result {index} returned duplicate action {key}"
                )
            metrics = dict(candidate.get("metrics") or {})
            if metrics.get("source") != "exact_hu_response":
                raise RuntimeError(
                    f"physical BB T4 Rust result {index} action {key} has non-HU source "
                    f"{metrics.get('source')!r}"
                )
            if metrics.get("opponent_response") is not True or int(
                metrics.get("start_turn") or 0
            ) != 4:
                raise RuntimeError(
                    f"physical BB T4 Rust result {index} action {key} is not an exact "
                    "T4 response leaf"
                )
            expected_draws = math.comb(item["remaining_card_count"], 3)
            if int(metrics.get("remaining_deck_size") or -1) != item["remaining_card_count"]:
                raise RuntimeError(
                    f"physical BB T4 Rust result {index} action {key} used "
                    f"remaining_deck_size={metrics.get('remaining_deck_size')!r}; "
                    f"expected {item['remaining_card_count']}"
                )
            if int(metrics.get("enumerated_draws") or -1) != expected_draws:
                raise RuntimeError(
                    f"physical BB T4 Rust result {index} action {key} enumerated "
                    f"{metrics.get('enumerated_draws')!r} draws; expected {expected_draws}"
                )
            metrics_by_key[key] = metrics
            actions_by_key[key] = action

        reported_legal = int(result.get("legal_actions") or 0)
        reported_evaluated = int(result.get("evaluated_actions") or 0)
        if reported_legal != len(expected_keys) or reported_evaluated != len(expected_keys):
            raise RuntimeError(
                f"physical BB T4 Rust result {index} coverage counters are "
                f"{reported_evaluated}/{reported_legal}, expected {len(expected_keys)}"
            )
        if set(metrics_by_key) != expected_keys:
            missing = sorted(expected_keys - set(metrics_by_key))
            extra = sorted(set(metrics_by_key) - expected_keys)
            raise RuntimeError(
                f"physical BB T4 Rust result {index} action coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )

        stable_keys = sorted(expected_keys)
        stable_metrics = {key: metrics_by_key[key] for key in stable_keys}
        stable_actions = {key: actions_by_key[key] for key in stable_keys}
        outputs.append(
            {
                "schema": PHYSICAL_BB_T4_ACTION_VECTOR_SCHEMA,
                "turn": 4,
                "actor": "bb",
                "position": "bb",
                "is_btn": False,
                "first_actor": "bb",
                "position_contract_version": POSITION_CONTRACT_VERSION,
                "particle_id": item["particle_id"],
                "physical_state_commitment": item["physical_state_commitment"],
                "candidate_scope": "all_legal",
                "selection_performed": False,
                "particle_aggregation_performed": False,
                "source": "rust_exact_physical_t4_action_vector",
                "metrics_source": "exact_hu_response",
                "exact_scope": "t4_conditioned_physical_range_all_draws_best_response",
                "range_model": item["range_model"],
                "conditioned_range_exact": True,
                "terminal_response_exact": True,
                "hu_exact": False,
                "equilibrium_exact": False,
                "private_physical_input_fields_omitted": True,
                "particle_id_contract": "caller_supplied_opaque_diagnostic_only",
                "public_policy_safe": False,
                "requires_infoset_aggregation": True,
                "remaining_card_count": item["remaining_card_count"],
                "legal_actions": len(stable_keys),
                "evaluated_actions": len(stable_keys),
                "candidate_count": len(stable_keys),
                "action_keys": stable_keys,
                "metrics_vector": [stable_metrics[key] for key in stable_keys],
                "metrics_by_action_key": stable_metrics,
                "actions_by_action_key": stable_actions,
                "elapsed_ms": float(result.get("elapsed_ms", 0.0) or 0.0),
            }
        )
    return outputs


def evaluate_physical_bb_t4_action_vector_rust(
    payload: Dict[str, Any],
    *,
    rust_solver_path: str | Path | None = None,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    """Single-particle wrapper for the physical BB T4 action-vector API."""
    return evaluate_physical_bb_t4_action_vectors_rust_batch(
        [payload],
        rust_solver_path=rust_solver_path,
        timeout_s=timeout_s,
        position_parallel=False,
    )[0]


def evaluate_late_position(
    board: Board,
    dealt: List[str],
    turn: int,
    opponent_board: Optional[Board] = None,
    exclude: Optional[Iterable[str]] = None,
    candidate_actions: Optional[Iterable[Dict[str, Any]]] = None,
    top_n: int = 20,
    prefer_rust: bool = True,
    rust_solver_path: str | Path | None = None,
    rust_timeout_s: float = 5.0,
    fallback_on_rust_error: bool = True,
) -> Dict[str, Any]:
    """Evaluate all legal T3/T4 actions exactly and rank candidates."""
    if turn not in (3, 4):
        raise ValueError("evaluate_late_position only supports turns 3 and 4")
    requested_actions = list(candidate_actions or [])
    rust_error = None
    use_rust = turn == 3 or (
        turn == 4
        and opponent_board is not None
        and board_card_count(opponent_board) == 11
    )
    if prefer_rust and use_rust:
        try:
            return evaluate_late_position_rust(
                board,
                dealt,
                turn,
                opponent_board=opponent_board,
                exclude=exclude,
                candidate_actions=requested_actions,
                top_n=top_n,
                rust_solver_path=rust_solver_path,
                timeout_s=rust_timeout_s,
            )
        except Exception as exc:
            rust_error = f"{type(exc).__name__}: {exc}"
            if not fallback_on_rust_error:
                raise RuntimeError(f"rust_exact_failed:{rust_error}") from exc
    legal_actions = get_turn_actions(dealt, board)
    actions = filter_actions_by_payload(legal_actions, requested_actions)
    candidates = []
    for action in actions:
        next_board = apply_action(board, action)
        metrics = exact_candidate_metrics(board, dealt, action, turn, opponent_board, exclude)
        candidates.append(
            {
                "action": action_to_dict(action),
                "board": board_to_dict(next_board),
                "metrics": metrics,
            }
        )
    candidates.sort(key=_candidate_sort_key)
    exact_scope, hu_exact = late_exact_scope(turn, opponent_board)
    out = {
        "turn": turn,
        "board": board_to_dict(board),
        "dealt": list(dealt),
        "legal_actions": len(legal_actions),
        "evaluated_actions": len(actions),
        "requested_actions": len(requested_actions),
        "chosen_action": candidates[0]["action"] if candidates else None,
        "best": candidates[0] if candidates else None,
        "candidates": candidates[:top_n],
        "candidate_count": len(candidates),
        "source": "exact",
        "exact_scope": exact_scope,
        "hu_exact": hu_exact,
    }
    if rust_error is not None:
        out["rust_fallback_error"] = rust_error
    return out


def normalize_position_payload(payload: Dict[str, Any]) -> Tuple[Board, Board, List[str], List[str]]:
    normalizer = CardNormalizer()
    board = normalize_board(payload.get("board") or payload, normalizer)
    opponent = normalize_board(payload.get("opponent_board") or {}, normalizer)
    dealt = normalizer.cards(payload.get("dealt", []) or [])
    exclude = normalizer.cards((payload.get("exclude", []) or []) + (payload.get("known_discards", []) or []))
    return board, opponent, dealt, exclude


def stable_cache_key(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def exact_sample_size(
    turn: int,
    remaining_deck_size: int,
    *,
    opponent_response: bool = False,
) -> int:
    if turn == 4:
        return math.comb(remaining_deck_size, 3) if opponent_response else 1
    if turn == 3:
        return math.comb(remaining_deck_size, 3)
    return 0
