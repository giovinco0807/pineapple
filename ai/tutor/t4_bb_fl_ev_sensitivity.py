"""Diagnostic FL EV sensitivity pilot for the T4 BB exact decision.

Measures how much the exact uniform-deal T4 BB policy (see
``t4_bb_exact_resolver``) changes when the Fantasyland EV table is scaled by
+/-20%.  The fixture family keeps Fantasyland live on both sides: BB holds a
QQ top pair with an open top slot (trips-top upgrade possible), BTN holds an
AA top pair with an open top slot, so both the hero decision and the exact
opponent response depend on the FL EV constants.

This is a diagnostic only: it patches ``RolloutEvaluator.FL_EV`` in-process,
bypasses the fail-closed resolver guards on purpose, and must never feed
teachers, gates, or serving.  Its output artifact carries
``diagnostic_only=true`` and ``promotion_eligible=false``.

Usage:
    python -m ai.tutor.t4_bb_fl_ev_sensitivity --out ai/reports/<dir>/report.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import ai.engine.action_space as action_space
import ai.tutor.exact_late as exact_late
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.tutor.t3_hu_public_cfr import InfoSetKey, PrivateRecall
from ai.tutor.t3_t4_infoset_encoder import semantic_action_ids

SCHEMA = "ofc_t4_bb_fl_ev_sensitivity/v1"
DEFAULT_SCALES = (0.8, 1.0, 1.2)
BASELINE_SCALE = 1.0

# FL-live, bust-free fixture: BB top QQ + open slot with a made middle
# straight and made bottom quads, BTN top AA + open slot likewise.  Every
# completion is legal (bottom quads > middle straight > any top pair/trips),
# so action EVs differ only through line wins, royalty, and FL EV.
BB_BOARD_11 = (
    ("Qs", "Qh"),
    ("5c", "6c", "7c", "8h", "9h"),
    ("4c", "4d", "4h", "4s"),
)
BTN_BOARD_11 = (
    ("Ah", "Ad"),
    ("7d", "8d", "9c", "Tc", "Jh"),
    ("2c", "2d", "2h", "2s"),
)
PUBLIC_HISTORY = (
    (
        0,
        "bb",
        (
            ("Qs", "top"),
            ("Qh", "top"),
            ("5c", "middle"),
            ("6c", "middle"),
            ("4c", "bottom"),
        ),
    ),
    (
        0,
        "btn",
        (
            ("Ah", "top"),
            ("Ad", "top"),
            ("7d", "middle"),
            ("8d", "middle"),
            ("2c", "bottom"),
        ),
    ),
    (1, "bb", (("7c", "middle"), ("8h", "middle"))),
    (1, "btn", (("9c", "middle"), ("Tc", "middle"))),
    (2, "bb", (("9h", "middle"), ("4d", "bottom"))),
    (2, "btn", (("Jh", "middle"), ("2d", "bottom"))),
    (3, "bb", (("4h", "bottom"), ("4s", "bottom"))),
    (3, "btn", (("2h", "bottom"), ("2s", "bottom"))),
)
BB_DISCARDS = ("3c", "3d", "3h")
BTN_HIDDEN_DISCARDS = ("3s", "6d", "6h")

# Hand-picked draws that make the FL trade-off explicit (Qd = trips-top).
HAND_PICKED_DRAWS = (
    ("Qd", "Ks", "5s"),
    ("Qd", "X1", "5s"),
    ("X1", "X2", "5s"),
    ("Ac", "X1", "Ts"),
    ("Qd", "Ac", "X2"),
    ("Ts", "9d", "5s"),
)


def _bb_recall() -> PrivateRecall:
    return PrivateRecall(
        dealt_by_turn=(
            (1, ("7c", "8h", "3c")),
            (2, ("9h", "4d", "3d")),
            (3, ("4h", "4s", "3h")),
        ),
        discards_by_turn=((1, "3c"), (2, "3d"), (3, "3h")),
    )


def build_infoset(draw: tuple[str, str, str]) -> InfoSetKey:
    return InfoSetKey(
        contract_version=POSITION_CONTRACT_VERSION,
        actor="bb",
        turn=4,
        phase="t4_first",
        board_bb=BB_BOARD_11,
        board_btn=BTN_BOARD_11,
        public_action_history=PUBLIC_HISTORY,
        own_recall=_bb_recall(),
        current_draw=draw,
        fantasy_state=None,
    )


def physically_live_pool() -> list[str]:
    """Cards a physical dealer could still hand BB at T4."""
    known = (
        {card for row in BB_BOARD_11 for card in row}
        | {card for row in BTN_BOARD_11 for card in row}
        | set(BB_DISCARDS)
        | set(BTN_HIDDEN_DISCARDS)
    )
    return [card for card in ALL_CARDS if card not in known]


def sample_draws(seed: int, random_count: int) -> list[tuple[str, str, str]]:
    pool = physically_live_pool()
    pool_set = set(pool)
    for draw in HAND_PICKED_DRAWS:
        missing = [card for card in draw if card not in pool_set]
        if missing:
            raise ValueError(f"hand-picked draw {draw} uses dead cards {missing}")
    seen = {tuple(sorted(draw)) for draw in HAND_PICKED_DRAWS}
    rng = random.Random(seed)
    all_combos = list(combinations(pool, 3))
    rng.shuffle(all_combos)
    draws = list(HAND_PICKED_DRAWS)
    for combo in all_combos:
        if len(draws) >= len(HAND_PICKED_DRAWS) + random_count:
            break
        key = tuple(sorted(combo))
        if key in seen:
            continue
        seen.add(key)
        draws.append(combo)
    return draws


def compute_action_evs(
    draw: tuple[str, str, str],
    fl_ev_scale: float,
) -> dict[str, float]:
    """Exact uniform-deal EV per legal action under a scaled FL EV table."""
    information = build_infoset(draw)
    bb_board = Board(
        top=list(information.board_bb[0]),
        middle=list(information.board_bb[1]),
        bottom=list(information.board_bb[2]),
    )
    btn_board = Board(
        top=list(information.board_btn[0]),
        middle=list(information.board_btn[1]),
        bottom=list(information.board_btn[2]),
    )
    original_fl_ev = RolloutEvaluator.FL_EV
    scaled_fl_ev = {
        count: float(value) * float(fl_ev_scale)
        for count, value in original_fl_ev.items()
    }
    RolloutEvaluator.FL_EV = scaled_fl_ev
    try:
        evs: dict[str, float] = {}
        for index, action_id in enumerate(semantic_action_ids(information)):
            if action_id is None:
                continue
            action = action_space.get_action_from_semantic_index_if_valid(
                index,
                list(information.current_draw),
                bb_board,
            )
            if action is None:
                raise RuntimeError("semantic legal action cannot be reconstructed")
            final_bb = exact_late.apply_action(bb_board, action)
            distribution = exact_late.exact_t4_opponent_response_distribution(
                final_bb,
                btn_board,
                exclude=tuple(BB_DISCARDS) + (action.discard,),
            )
            evs[action_id] = float(distribution["score"])
        if not evs:
            raise RuntimeError("no legal T4 BB action in sensitivity fixture")
        return evs
    finally:
        RolloutEvaluator.FL_EV = original_fl_ev


def _best(evs: dict[str, float]) -> tuple[str, float]:
    best_value = max(evs.values())
    best_action = min(
        action_id for action_id, value in evs.items() if value == best_value
    )
    return best_action, best_value


def _worker(task: tuple[tuple[str, str, str], float]) -> tuple[
    tuple[str, str, str], float, dict[str, float]
]:
    draw, scale = task
    return draw, scale, compute_action_evs(draw, scale)


def run(
    *,
    seed: int,
    random_count: int,
    scales: tuple[float, ...],
    workers: int,
    out_path: Path,
) -> dict:
    if BASELINE_SCALE not in scales:
        raise ValueError("scales must include the 1.0 baseline")
    draws = sample_draws(seed, random_count)
    tasks = [(draw, scale) for draw in draws for scale in scales]
    started = time.time()
    results: dict[tuple[tuple[str, str, str], float], dict[str, float]] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for draw, scale, evs in pool.map(_worker, tasks):
                results[(tuple(draw), scale)] = evs
    else:
        for task in tasks:
            draw, scale, evs = _worker(task)
            results[(tuple(draw), scale)] = evs
    elapsed = time.time() - started

    rows = []
    for draw in draws:
        key = tuple(draw)
        baseline = results[(key, BASELINE_SCALE)]
        baseline_action, baseline_value = _best(baseline)
        row = {
            "draw": list(draw),
            "joker_count": sum(1 for card in draw if card in ("X1", "X2")),
            "legal_action_count": len(baseline),
            "baseline_best_action": baseline_action,
            "baseline_best_ev": baseline_value,
            "scales": {},
        }
        for scale in scales:
            if scale == BASELINE_SCALE:
                continue
            variant = results[(key, scale)]
            variant_action, variant_value = _best(variant)
            regret = variant_value - variant[baseline_action]
            row["scales"][f"{scale:g}"] = {
                "best_action": variant_action,
                "best_ev": variant_value,
                "action_changed": variant_action != baseline_action,
                "baseline_action_regret": regret,
                "best_ev_delta_vs_baseline": variant_value - baseline_value,
            }
        rows.append(row)

    summary = {}
    for scale in scales:
        if scale == BASELINE_SCALE:
            continue
        scale_key = f"{scale:g}"
        changed = [row for row in rows if row["scales"][scale_key]["action_changed"]]
        regrets = [row["scales"][scale_key]["baseline_action_regret"] for row in rows]
        summary[scale_key] = {
            "roots": len(rows),
            "top_action_changed": len(changed),
            "top_action_change_rate": len(changed) / len(rows),
            "changed_draws": [row["draw"] for row in changed],
            "mean_baseline_action_regret": sum(regrets) / len(regrets),
            "max_baseline_action_regret": max(regrets),
        }

    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "serving_changed": False,
        "method": "t4_bb_exact_uniform_deal_response_v1 (unguarded diagnostic path)",
        "fl_ev_base": {str(k): float(v) for k, v in RolloutEvaluator.FL_EV.items()},
        "fl_ev_scales": [float(s) for s in scales],
        "seed": seed,
        "hand_picked_draws": [list(d) for d in HAND_PICKED_DRAWS],
        "random_draw_count": random_count,
        "fixture": {
            "board_bb": [list(row) for row in BB_BOARD_11],
            "board_btn": [list(row) for row in BTN_BOARD_11],
            "bb_hidden_discards": list(BB_DISCARDS),
            "physical_pool_size": len(physically_live_pool()),
            "fl_live_bb": "QQ top pair, open top slot (trips upgrade possible)",
            "fl_live_btn": "AA top pair, open top slot",
        },
        "elapsed_seconds": elapsed,
        "summary": summary,
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
    parser.add_argument("--random-count", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCALES),
    )
    args = parser.parse_args()
    report = run(
        seed=args.seed,
        random_count=args.random_count,
        scales=tuple(args.scales),
        workers=args.workers,
        out_path=args.out,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"roots={len(report['rows'])} elapsed={report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
