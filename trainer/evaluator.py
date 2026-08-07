"""Candidate evaluation for the regular OFC trainer.

Monte-Carlo evaluator over the full legal action set with common random
futures: every candidate is scored against the SAME sampled continuations,
which removes almost all ranking variance between candidates.

Continuation model: both players complete their boards with uniform random
legal placements (information-set correct: the opponent's hidden cards are
drawn from the remaining deck).  Scores use the regular head-to-head
contract (lines + scoop + royalties, +/-6 fouls) plus the fixed-point
Fantasyland EV bonus for FL entries.
"""

from __future__ import annotations

import random
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ofc_regular.action_space import generate_actions  # noqa: E402
from ofc_regular.cards import ALL_CARDS  # noqa: E402
from ofc_regular.evaluator import (  # noqa: E402
    evaluate_3_card,
    evaluate_5_card,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
)
from ofc_regular.rules import fl_entry_type  # noqa: E402
from ofc_regular.state import ROW_CAPACITY  # noqa: E402

from trainer.fl_ev import FL_EV_14  # noqa: E402  (configs/fl_ev_regular_v4_selfplay.json)

FOUL_PENALTY = 6

ROWS = ("top", "middle", "bottom")

# Simulations per precision and street (T4 futures are cheap; early streets
# need more rollouts but each is longer, so budgets are balanced by hand).
SIMS = {
    "fast": {0: 48, 1: 64, 2: 96, 3: 160, 4: 400},
    "standard": {0: 160, 1: 220, 2: 300, 3: 600, 4: 1500},
    "high": {0: 400, 1: 600, 2: 800, 3: 1600, 4: 4000},
}

_rng_lock = threading.Lock()


@lru_cache(maxsize=400_000)
def _eval3(key: Tuple[str, ...]) -> Tuple[int, Tuple[int, ...]]:
    return evaluate_3_card(key)


@lru_cache(maxsize=400_000)
def _eval5(key: Tuple[str, ...]) -> Tuple[int, Tuple[int, ...]]:
    return evaluate_5_card(key)


@lru_cache(maxsize=400_000)
def _royalty_top(key: Tuple[str, ...]) -> int:
    return get_top_royalty(key)


@lru_cache(maxsize=400_000)
def _royalty_mid(key: Tuple[str, ...]) -> int:
    return get_middle_royalty(key)


@lru_cache(maxsize=400_000)
def _royalty_bot(key: Tuple[str, ...]) -> int:
    return get_bottom_royalty(key)


@lru_cache(maxsize=100_000)
def _fl_type(key: Tuple[str, ...]) -> Optional[str]:
    return fl_entry_type(key)


def _board_terminal(top: Tuple[str, ...], mid: Tuple[str, ...], bot: Tuple[str, ...]):
    """(busted, values, royalty, fl_entry) for a complete 13-card board."""
    tv = _eval3(top)
    mv = _eval5(mid)
    bv = _eval5(bot)
    busted = tv > mv or mv > bv
    if busted:
        return True, (tv, mv, bv), 0, False
    royalty = _royalty_top(top) + _royalty_mid(mid) + _royalty_bot(bot)
    fl = _fl_type(top) is not None
    return False, (tv, mv, bv), royalty, fl


def _score_pair(hero, opp) -> float:
    """Head-to-head score from hero's perspective, incl. FL EV bonus."""
    h_bust, h_vals, h_roy, h_fl = hero
    o_bust, o_vals, o_roy, o_fl = opp
    if h_bust and o_bust:
        score = 0.0
    elif h_bust:
        score = -(FOUL_PENALTY + o_roy)
    elif o_bust:
        score = FOUL_PENALTY + h_roy
    else:
        lines = 0
        for hv, ov in zip(h_vals, o_vals):
            lines += 1 if hv > ov else -1 if hv < ov else 0
        score = lines + (3 if lines == 3 else -3 if lines == -3 else 0) + h_roy - o_roy
    if h_fl:
        score += FL_EV_14
    if o_fl:
        score -= FL_EV_14
    return score


class _MutableRows:
    __slots__ = ("rows", "open")

    def __init__(self, rows: Dict[str, Sequence[str]]):
        self.rows = {r: list(rows[r]) for r in ROWS}
        self.open = [r for r in ROWS for _ in range(ROW_CAPACITY[r] - len(rows[r]))]

    def sorted_key(self):
        return (
            tuple(sorted(self.rows["top"])),
            tuple(sorted(self.rows["middle"])),
            tuple(sorted(self.rows["bottom"])),
        )


def _random_fill(rows: Dict[str, List[str]], cards: List[str], rng: random.Random):
    """Fill all open slots with the given cards in random slot order."""
    slots = [r for r in ROWS for _ in range(ROW_CAPACITY[r] - len(rows[r]))]
    rng.shuffle(slots)
    for card, row in zip(cards, slots):
        rows[row].append(card)


def _hero_continue(rows: Dict[str, List[str]], future: List[str], streets: int, rng: random.Random):
    """Pineapple continuation: per street take 3 cards, place 2 random-legal, discard 1."""
    idx = 0
    for _ in range(streets):
        dealt = future[idx : idx + 3]
        idx += 3
        keep = rng.sample(dealt, 2)
        for card in keep:
            open_rows = [r for r in ROWS if len(rows[r]) < ROW_CAPACITY[r]]
            row = rng.choice(open_rows)
            rows[row].append(card)


def evaluate_position(
    *,
    hero_board: Dict[str, Sequence[str]],
    opp_board: Dict[str, Sequence[str]],
    dealt: Sequence[str],
    dead: Sequence[str] = (),
    turn: int,
    position: str = "first",
    precision: str = "standard",
    sims_override: Optional[int] = None,
    top_n: int = 0,
) -> Dict[str, Any]:
    """Rank every legal action: m3 engine for T1-T4, Monte-Carlo for T0/fallback."""
    if turn in (1, 2, 3, 4):
        try:
            from trainer import engine_eval

            return engine_eval.evaluate_with_engine(
                hero_board=hero_board,
                opp_board=opp_board,
                dealt=dealt,
                dead=dead,
                turn=turn,
                position=position,
                precision=precision,
            )
        except Exception as exc:  # engine refusal (e.g. T1-first) -> MC fallback
            import logging

            logging.getLogger("trainer.evaluator").info(
                "engine eval unavailable for T%s/%s (%s); falling back to MC", turn, position, exc
            )
    return evaluate_position_mc(
        hero_board=hero_board,
        opp_board=opp_board,
        dealt=dealt,
        dead=dead,
        turn=turn,
        position=position,
        precision=precision,
        sims_override=sims_override,
        top_n=top_n,
    )


def evaluate_position_mc(
    *,
    hero_board: Dict[str, Sequence[str]],
    opp_board: Dict[str, Sequence[str]],
    dealt: Sequence[str],
    dead: Sequence[str] = (),
    turn: int,
    position: str = "first",
    precision: str = "standard",
    sims_override: Optional[int] = None,
    top_n: int = 0,
) -> Dict[str, Any]:
    """Rank every legal action of the current street by Monte-Carlo EV."""
    if turn not in (0, 1, 2, 3, 4):
        raise ValueError("turn must be 0..4")
    expected_dealt = 5 if turn == 0 else 3
    if len(dealt) != expected_dealt:
        raise ValueError(f"T{turn} は配札 {expected_dealt} 枚が必要です")

    from ofc_regular.state import Board  # local import to keep module load light

    hero = Board.from_rows(hero_board.get("top", ()), hero_board.get("middle", ()), hero_board.get("bottom", ()))
    opp = Board.from_rows(opp_board.get("top", ()), opp_board.get("middle", ()), opp_board.get("bottom", ()))

    expected_board = 0 if turn == 0 else 5 + 2 * (turn - 1)
    if hero.card_count() != expected_board:
        raise ValueError(f"T{turn} の自分の盤面は {expected_board} 枚のはずです（現在 {hero.card_count()} 枚）")

    seen = set(hero.all_cards()) | set(opp.all_cards()) | set(dealt) | set(dead)
    if len(seen) != hero.card_count() + opp.card_count() + len(dealt) + len(set(dead)):
        raise ValueError("カードが重複しています")
    deck = [c for c in ALL_CARDS if c not in seen]

    actions = generate_actions(hero, list(dealt))
    if not actions:
        raise ValueError("合法手がありません")

    sims = sims_override or SIMS.get(precision, SIMS["standard"]).get(turn, 200)
    hero_streets_left = 4 - turn  # future pineapple streets after this action
    hero_future_need = 3 * hero_streets_left
    opp_need = 13 - opp.card_count()
    # opponent's own future discards also leave the deck unseen; drawing only
    # the fill cards is the standard information-set approximation.

    rng = random.Random(0xC0FFEE ^ (turn * 7919) ^ len(deck))

    # Pre-draw common futures: same continuation cards for every candidate.
    futures = []
    for _ in range(sims):
        sample = rng.sample(deck, min(len(deck), hero_future_need + opp_need))
        futures.append((sample[:hero_future_need], sample[hero_future_need : hero_future_need + opp_need]))

    candidates = []
    for action in actions:
        placed = {r: list(getattr(hero, r)) for r in ROWS}
        for card, row in action.placements:
            placed[row].append(card)

        total = 0.0
        busts = 0
        fls = 0
        roy_sum = 0.0
        for hero_future, opp_fill in futures:
            rows = {r: list(placed[r]) for r in ROWS}
            _hero_continue(rows, hero_future, hero_streets_left, rng)
            opp_rows = {r: list(getattr(opp, r)) for r in ROWS}
            _random_fill(opp_rows, opp_fill, rng)
            h_term = _board_terminal(
                tuple(sorted(rows["top"])), tuple(sorted(rows["middle"])), tuple(sorted(rows["bottom"]))
            )
            o_term = _board_terminal(
                tuple(sorted(opp_rows["top"])),
                tuple(sorted(opp_rows["middle"])),
                tuple(sorted(opp_rows["bottom"])),
            )
            total += _score_pair(h_term, o_term)
            busts += h_term[0]
            fls += h_term[3]
            roy_sum += h_term[2]

        n = max(len(futures), 1)
        discard = action.discards[0] if action.discards else None
        candidates.append(
            {
                "action": {
                    "placements": [[c, r] for c, r in action.placements],
                    "discard": discard,
                },
                "board": {r: placed[r] for r in ROWS},
                "metrics": {
                    "ev": total / n,
                    "bust_rate": busts / n,
                    "fl_rate": fls / n,
                    "royalty": roy_sum / n,
                    "sims": n,
                },
            }
        )

    candidates.sort(key=lambda c: c["metrics"]["ev"], reverse=True)
    for cand in candidates:
        cand["key"] = "|".join(",".join(sorted(cand["board"][r])) for r in ROWS)
    if top_n:
        candidates = candidates[:top_n]
    return {
        "candidates": candidates,
        "evaluator": f"mc-random({sims} sims)",
        "sims": sims,
    }


def describe() -> str:
    try:
        from trainer import engine_eval

        if engine_eval.available():
            return "m3 engine (T1-T4) + mc-random (T0/fallback)"
    except Exception:
        pass
    return "mc-random common-futures evaluator (pure python)"
