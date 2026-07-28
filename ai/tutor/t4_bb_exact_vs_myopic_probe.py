"""T4 BB component strength probe: exact uniform-deal choice vs legacy myopic rule.

Compares two T4 BB decision rules on generated physical roots:

- ``myopic``: the legacy playout completion rule ``best_t4_completion(board,
  draw, opponent_board=None)`` — maximize own royalty + own FL EV, ignoring the
  opponent board and the opponent's final draw/response.
- ``exact``: the uniform-deal exact rule of ``t4_bb_exact_resolver`` — maximize
  the mean over all C(26,3) opponent draws of the negated exact opponent best
  response.

Both choices are scored under the same declared uniform exchangeable restart
belief, so the per-root regret ``EV[exact] - EV[myopic] >= 0`` is exact and
needs no simulation.  No hidden opponent information is used anywhere.

Root generators:

- ``random_legal``: seeded 54-card shuffles assigned to random legal two-open
  row shapes for both seats.  Board quality is intentionally unfiltered.
- ``fl_live``: the bust-free FL-live fixture family from
  ``t4_bb_fl_ev_sensitivity`` with seeded draws.

This is a diagnostic component probe under the declared belief; it is not a
full-game paired self-play strength result, changes no serving, and promotes
nothing (``promotion_eligible=false``).

Usage:
    python -m ai.tutor.t4_bb_exact_vs_myopic_probe --out ai/reports/<dir>/report.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import ai.tutor.exact_late as exact_late
import ai.tutor.t4_bb_fl_ev_sensitivity as fl_live_fixture
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board

SCHEMA = "ofc_t4_bb_exact_vs_myopic_probe/v1"

# (top, middle, bottom) fills with exactly two open slots.
TWO_OPEN_SHAPES = (
    (3, 5, 3),
    (3, 4, 4),
    (3, 3, 5),
    (2, 5, 4),
    (2, 4, 5),
    (1, 5, 5),
)


def sample_random_root(seed: int) -> dict:
    """One physical T4 BB root from a seeded 54-card shuffle."""
    rng = random.Random(seed)
    deck = list(ALL_CARDS)
    rng.shuffle(deck)

    bb_shape = rng.choice(TWO_OPEN_SHAPES)
    btn_shape = rng.choice(TWO_OPEN_SHAPES)

    def take(count: int) -> list[str]:
        cards = deck[:count]
        del deck[:count]
        return cards

    bb_board = Board(
        top=take(bb_shape[0]),
        middle=take(bb_shape[1]),
        bottom=take(bb_shape[2]),
    )
    bb_discards = tuple(take(3))
    btn_board = Board(
        top=take(btn_shape[0]),
        middle=take(btn_shape[1]),
        bottom=take(btn_shape[2]),
    )
    draw = tuple(take(3))
    return {
        "generator": "random_legal",
        "seed": seed,
        "bb_board": [list(bb_board.top), list(bb_board.middle), list(bb_board.bottom)],
        "btn_board": [
            list(btn_board.top),
            list(btn_board.middle),
            list(btn_board.bottom),
        ],
        "bb_discards": list(bb_discards),
        "draw": list(draw),
    }


def fl_live_root(draw: tuple[str, str, str]) -> dict:
    return {
        "generator": "fl_live",
        "seed": None,
        "bb_board": [list(row) for row in fl_live_fixture.BB_BOARD_11],
        "btn_board": [list(row) for row in fl_live_fixture.BTN_BOARD_11],
        "bb_discards": list(fl_live_fixture.BB_DISCARDS),
        "draw": list(draw),
    }


def _root_boards(root: dict) -> tuple[Board, Board]:
    bb = Board(
        top=list(root["bb_board"][0]),
        middle=list(root["bb_board"][1]),
        bottom=list(root["bb_board"][2]),
    )
    btn = Board(
        top=list(root["btn_board"][0]),
        middle=list(root["btn_board"][1]),
        bottom=list(root["btn_board"][2]),
    )
    return bb, btn


def exact_action_table(root: dict) -> dict[str, float]:
    """Exact uniform-deal EV for every legal T4 BB action of this root."""
    bb_board, btn_board = _root_boards(root)
    evs: dict[str, float] = {}
    for action in get_turn_actions(list(root["draw"]), bb_board):
        action_id = exact_late.action_key(action)
        if action_id in evs:
            continue
        final_bb = exact_late.apply_action(bb_board, action)
        distribution = exact_late.exact_t4_opponent_response_distribution(
            final_bb,
            btn_board,
            exclude=tuple(root["bb_discards"]) + (action.discard,),
        )
        evs[action_id] = float(distribution["score"])
    if not evs:
        raise RuntimeError("probe root has no legal T4 BB action")
    return evs


def myopic_action_id(root: dict) -> str:
    """Legacy playout completion rule: no opponent, no response model."""
    bb_board, _btn_board = _root_boards(root)
    best = exact_late.best_t4_completion(bb_board, list(root["draw"]), None)
    return exact_late.action_key(best["action"])


def _best(evs: dict[str, float]) -> tuple[str, float]:
    best_value = max(evs.values())
    best_action = min(
        action_id for action_id, value in evs.items() if value == best_value
    )
    return best_action, best_value


def _evaluate_root(root: dict) -> dict:
    evs = exact_action_table(root)
    myopic_id = myopic_action_id(root)
    if myopic_id not in evs:
        raise RuntimeError("myopic action is not in the exact legal table")
    exact_id, exact_value = _best(evs)
    myopic_value = evs[myopic_id]
    regret = exact_value - myopic_value
    all_cards = (
        [card for row in root["bb_board"] for card in row]
        + [card for row in root["btn_board"] for card in row]
        + list(root["draw"])
    )
    return {
        **{k: root[k] for k in ("generator", "seed", "draw")},
        "legal_action_count": len(evs),
        "visible_joker_count": sum(1 for card in all_cards if card in ("X1", "X2")),
        "myopic_action": myopic_id,
        "myopic_ev": myopic_value,
        "exact_action": exact_id,
        "exact_ev": exact_value,
        "fired": exact_id != myopic_id,
        "myopic_regret": regret,
    }


def _aggregate(rows: list[dict]) -> dict:
    fired = [row for row in rows if row["fired"]]
    regrets = [row["myopic_regret"] for row in rows]
    fired_regrets = [row["myopic_regret"] for row in fired]
    return {
        "roots": len(rows),
        "fired": len(fired),
        "fire_rate": len(fired) / len(rows) if rows else 0.0,
        "mean_regret": sum(regrets) / len(regrets) if regrets else 0.0,
        "max_regret": max(regrets) if regrets else 0.0,
        "mean_regret_when_fired": (
            sum(fired_regrets) / len(fired_regrets) if fired_regrets else 0.0
        ),
        "min_regret": min(regrets) if regrets else 0.0,
    }


def run(
    *,
    seed: int,
    random_count: int,
    fl_live_count: int,
    workers: int,
    out_path: Path,
) -> dict:
    roots = [sample_random_root(seed + index) for index in range(random_count)]
    fl_draws = fl_live_fixture.sample_draws(
        seed,
        max(fl_live_count - len(fl_live_fixture.HAND_PICKED_DRAWS), 0),
    )[:fl_live_count]
    roots.extend(fl_live_root(tuple(draw)) for draw in fl_draws)

    started = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_evaluate_root, roots))
    else:
        rows = [_evaluate_root(root) for root in roots]
    elapsed = time.time() - started

    for row in rows:
        if row["myopic_regret"] < 0:
            raise RuntimeError("exact table regret must be non-negative")

    by_generator = {}
    for name in ("random_legal", "fl_live"):
        subset = [row for row in rows if row["generator"] == name]
        if subset:
            by_generator[name] = _aggregate(subset)
    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "serving_changed": False,
        "full_game_strength_claim": False,
        "belief_model": "uniform_exchangeable_restart_v1",
        "baseline_rule": "best_t4_completion(opponent_board=None)  # legacy myopic",
        "candidate_rule": "t4_bb_exact_uniform_deal_response_v1",
        "seed": seed,
        "random_count": random_count,
        "fl_live_count": fl_live_count,
        "elapsed_seconds": elapsed,
        "summary": {
            "all": _aggregate(rows),
            "by_generator": by_generator,
        },
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--random-count", type=int, default=40)
    parser.add_argument("--fl-live-count", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    report = run(
        seed=args.seed,
        random_count=args.random_count,
        fl_live_count=args.fl_live_count,
        workers=args.workers,
        out_path=args.out,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"roots={len(report['rows'])} elapsed={report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
