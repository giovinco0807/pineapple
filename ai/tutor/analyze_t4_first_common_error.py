"""Diagnose the per-root COMMON error of the T4 first-seat evaluator.

The measured deficiency (see the pass-1 report) is not action ranking but the
node value that backs up into T3 search: the error component that is IDENTICAL
across all actions of a root averages 1.079 pt (78% of the total), so it must
live in the blocks shared within a root -- the opponent block and the context
block.  This script hunts for the missing action-independent facts:

1. Over fresh roots (seed base 5,000,000, disjoint from training and earlier
   analysis seeds) it computes each root's common error
   ``mean over legal actions of (predicted - exact)``.
2. It computes, per root, a set of CANDIDATE action-independent facts that the
   current encoder does not carry -- chiefly the opponent's JOINT best
   completion over all C(26,3) draws (foul rate, royalty, Fantasy Land EV),
   which the per-row independent histograms cannot express.
3. It reports correlations of every candidate with the common error, plus the
   incremental R^2 of the new facts over a control regression built only from
   facts the current features already express.
4. It prints the worst positive and negative common-error roots in full so the
   diagnosis can be read from real hands, not only from aggregates.

The joint opponent metrics are exact enumerations (all C(26,3) = 2600 draws,
all discards, all arrangements) with the same joker-constraint evaluation the
scorer uses, so a candidate feature measured here is the same number a future
encoder would compute.

Usage:
    python -m ai.tutor.analyze_t4_first_common_error --roots 600 --show 10
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import ai.tutor.exact_late as exact_late
import ai.tutor.t4_bb_exact_vs_myopic_probe as probe
import ai.tutor.t4_first_features as features
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.engine.game_engine import (
    check_fl_entry,
    evaluate_board_with_joker_constraint,
    evaluate_hand,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
    hand_category,
    is_joker,
)
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.tutor.generate_t4_first_teacher import label_roots
from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

ROW_CAPACITY = (3, 5, 5)
FL_EV = RolloutEvaluator.FL_EV
FOUL_SELF_VALUE = -6.0
CATEGORY_NAMES = (
    "high", "pair", "two-pair", "trips", "straight",
    "flush", "full-house", "quads", "str-flush",
)


# ---------------------------------------------------------------------------
# Exact joint opponent completion metrics (the candidate missing facts).
# ---------------------------------------------------------------------------

def _row_royalty(index: int, cards) -> float:
    if index == 0:
        return float(get_top_royalty(list(cards)))
    if index == 1:
        return float(get_middle_royalty(list(cards)))
    return float(get_bottom_royalty(list(cards)))


def _top_fl_ev(cards) -> float:
    qualified, count = check_fl_entry(list(cards))
    return float(FL_EV.get(count, 0)) if qualified else 0.0


@lru_cache(maxsize=1_000_000)
def _constrained_board_facts(top: tuple, mid: tuple, bot: tuple):
    """Busted flag, royalty, FL EV under the canonical joker-constraint eval."""
    res = evaluate_board_with_joker_constraint(list(top), list(mid), list(bot))
    if bool(res["busted"]):
        return True, 0.0, 0.0
    royalty = (
        _row_royalty(0, res["top"])
        + _row_royalty(1, res["middle"])
        + _row_royalty(2, res["bottom"])
    )
    return False, royalty, _top_fl_ev(res["top"])


def opponent_joint_metrics(opp_rows, pool) -> dict:
    """Exact joint outlook of the opponent's final placement over all draws.

    For every C(26,3) draw the opponent keeps 2 of 3 cards and fills the two
    open slots; among all legal finals it takes the one maximizing its own
    (royalty + FL EV), with a foul worth -6.  This is the self-max response --
    the exact solver's response also includes line wins against the hero, but
    that part is action-dependent and small next to royalty/FL/foul.
    """
    rooms = [ROW_CAPACITY[i] - len(opp_rows[i]) for i in range(3)]
    open_rows = [i for i in range(3) if rooms[i] > 0]
    board_has_joker = any(is_joker(c) for row in opp_rows for c in row)

    # Complete-row facts are fixed for the whole enumeration.
    fixed_value = {}
    fixed_royalty = {}
    for i in range(3):
        if rooms[i] == 0:
            fixed_value[i] = evaluate_hand(list(opp_rows[i]), ROW_CAPACITY[i])
            fixed_royalty[i] = _row_royalty(i, opp_rows[i])
    fixed_top_fl = _top_fl_ev(opp_rows[0]) if rooms[0] == 0 else None

    # Per-row completion tables over the pool.
    one_card = {}   # row -> {card: (value, royalty, fl_ev_if_top)}
    two_card = {}   # row -> {(c1, c2) sorted: (value, royalty, fl_ev_if_top)}
    for i in open_rows:
        base = list(opp_rows[i])
        if rooms[i] == 1:
            table = {}
            for card in pool:
                filled = base + [card]
                value = evaluate_hand(filled, ROW_CAPACITY[i])
                fl = _top_fl_ev(filled) if i == 0 else 0.0
                table[card] = (value, _row_royalty(i, filled), fl)
            one_card[i] = table
        else:
            table = {}
            for pair in combinations(pool, 2):
                key = tuple(sorted(pair))
                filled = base + list(key)
                value = evaluate_hand(filled, ROW_CAPACITY[i])
                fl = _top_fl_ev(filled) if i == 0 else 0.0
                table[key] = (value, _row_royalty(i, filled), fl)
            two_card[i] = table

    def finals_for(kept: tuple) -> list:
        """(top_val, mid_val, bot_val, royalty, fl_ev, placed_cards) tuples."""
        out = []
        if len(open_rows) == 2:
            r1, r2 = open_rows
            for a, b in ((kept[0], kept[1]), (kept[1], kept[0])):
                va, ra, fa = one_card[r1][a]
                vb, rb, fb = one_card[r2][b]
                vals = [0, 0, 0]
                roys = [0.0, 0.0, 0.0]
                fl = fixed_top_fl if fixed_top_fl is not None else 0.0
                for i in range(3):
                    if rooms[i] == 0:
                        vals[i] = fixed_value[i]
                        roys[i] = fixed_royalty[i]
                vals[r1], roys[r1] = va, ra
                vals[r2], roys[r2] = vb, rb
                if r1 == 0:
                    fl = fa
                elif r2 == 0:
                    fl = fb
                out.append((vals, sum(roys), fl, (a, b)))
        else:
            row = open_rows[0]
            key = tuple(sorted(kept))
            value, royalty, fl_row = two_card[row][key]
            vals = [0, 0, 0]
            roys = [0.0, 0.0, 0.0]
            fl = fixed_top_fl if fixed_top_fl is not None else 0.0
            for i in range(3):
                if rooms[i] == 0:
                    vals[i] = fixed_value[i]
                    roys[i] = fixed_royalty[i]
            vals[row], roys[row] = value, royalty
            if row == 0:
                fl = fl_row
            out.append((vals, sum(roys), fl, key))
        return out

    def final_self_value(vals, royalty, fl, kept, top_row_cards) -> float:
        """Self value of one final, honoring the joker-constraint scorer."""
        if vals[0] <= vals[1] <= vals[2]:
            return royalty + fl
        # Raw ordering violated: only a joker somewhere on the final board can
        # rescue it, through the canonical constrained evaluation.
        if not (board_has_joker or any(is_joker(c) for c in kept)):
            return FOUL_SELF_VALUE
        rows_final = [list(opp_rows[i]) for i in range(3)]
        if len(open_rows) == 2:
            rows_final[open_rows[0]].append(kept[0])
            rows_final[open_rows[1]].append(kept[1])
        else:
            rows_final[open_rows[0]].extend(kept)
        busted, c_royalty, c_fl = _constrained_board_facts(
            tuple(rows_final[0]), tuple(rows_final[1]), tuple(rows_final[2])
        )
        return FOUL_SELF_VALUE if busted else c_royalty + c_fl

    n = 0
    forced_fouls = 0
    best_values = []
    sum_royalty = 0.0
    sum_fl = 0.0
    for draw in combinations(pool, 3):
        best = None
        best_parts = (0.0, 0.0)
        for kept in combinations(draw, 2):
            for vals, royalty, fl, placed in finals_for(kept):
                value = final_self_value(vals, royalty, fl, placed, None)
                if best is None or value > best:
                    best = value
                    best_parts = (0.0, 0.0) if value == FOUL_SELF_VALUE else (royalty, fl)
        n += 1
        best_values.append(best)
        if best == FOUL_SELF_VALUE:
            forced_fouls += 1
        else:
            sum_royalty += best_parts[0]
            sum_fl += best_parts[1]

    best_arr = np.asarray(best_values, dtype=np.float64)

    # Uniform marginal completion values per row, for head-to-head win rates.
    row_value_arrays = []
    for i in range(3):
        if rooms[i] == 0:
            values = np.asarray([fixed_value[i]], dtype=np.float64)
        elif rooms[i] == 1:
            values = np.asarray(
                [entry[0] for entry in one_card[i].values()], dtype=np.float64
            )
        else:
            values = np.asarray(
                [entry[0] for entry in two_card[i].values()], dtype=np.float64
            )
        values.sort()
        row_value_arrays.append(values)

    return {
        "draws": n,
        "joint_ev": float(best_arr.mean()),
        "joint_std": float(best_arr.std()),
        "foul_rate": forced_fouls / n,
        "joint_royalty": sum_royalty / n,
        "joint_fl": sum_fl / n,
        "row_value_arrays": row_value_arrays,
    }


# ---------------------------------------------------------------------------
# Facts the current encoder already expresses (controls).
# ---------------------------------------------------------------------------

def current_feature_facts(cache: features.NodeCache) -> dict:
    """Per-root quantities reconstructible from the existing 47 shared dims."""
    opp = cache._opponent
    # Opponent block layout: 3 rows x (hist 9 + royalty + fl + room) then
    # 3 suit maxima + opp jokers + pool jokers.
    indep_royalty = sum(opp[r * 12 + 9] for r in range(3)) * 25.0
    indep_fl = sum(opp[r * 12 + 10] for r in range(3)) * 63.5
    cats = cache._categories
    rooms = [ROW_CAPACITY[i] - len(cache.opponent_board[i]) for i in range(3)]
    locked_middle = rooms[2] == 0 and cats[1] > cats[2]
    locked_top = rooms[1] == 0 and cats[0] > cats[1]
    return {
        "indep_royalty": indep_royalty,
        "indep_fl": indep_fl,
        "locked_cat_any": 1.0 if (locked_middle or locked_top) else 0.0,
        "opp_jokers": opp[-2] * 2.0,
        "pool_jokers": opp[-1] * 2.0,
        "cat_slack_mid_bot": (cats[1] - cats[2]) / 8.0,
        "cat_slack_top_mid": (cats[0] - cats[1]) / 8.0,
        "pool_high_frac": cache._context[2],
    }


# ---------------------------------------------------------------------------
# Model plumbing.
# ---------------------------------------------------------------------------

def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = T4FirstEvaluator(checkpoint["input_dim"], tuple(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["input_mean"], checkpoint["input_std"]


def analyze_root(root: dict, model, mean, std, exact: dict) -> dict:
    board = Board(
        top=list(root["bb_board"][0]),
        middle=list(root["bb_board"][1]),
        bottom=list(root["bb_board"][2]),
    )
    cache = features.NodeCache.for_root(
        root["bb_board"], root["btn_board"], root["draw"], root["bb_discards"]
    )
    keys, vectors = [], []
    seen = set()
    for action in get_turn_actions(list(root["draw"]), board):
        key = exact_late.action_key(action)
        if key in seen:
            continue
        seen.add(key)
        final = exact_late.apply_action(board, action)
        keys.append(key)
        vectors.append(
            features.encode_action((final.top, final.middle, final.bottom), cache)
        )
    matrix = np.asarray(vectors, dtype=np.float32)
    scaled = (torch.tensor(matrix) - mean) / std
    with torch.no_grad():
        predicted = model(scaled).numpy().astype(np.float64)
    exact_arr = np.asarray([exact[k] for k in keys], dtype=np.float64)
    residual = predicted - exact_arr

    hero_bust = matrix[:, 0].astype(np.float64)
    hero_royalty = matrix[:, 1].astype(np.float64) * 25.0
    hero_fl = matrix[:, 7].astype(np.float64) * 63.5

    joint = opponent_joint_metrics(cache.opponent_board, cache.pool)
    controls = current_feature_facts(cache)

    # Expected head-to-head line outcome per action, against the UNIFORM
    # marginal completion of each opponent row (opponent choice ignored).
    row_values = joint.pop("row_value_arrays")
    exp_lines = []
    for key in keys:
        payload = json.loads(key)
        final = Board(
            top=list(root["bb_board"][0]),
            middle=list(root["bb_board"][1]),
            bottom=list(root["bb_board"][2]),
        )
        for card, row in payload["placements"]:
            getattr(final, row).append(card)
        _c_board, _c_bust, hero_vals = exact_late._constrained_board(final)
        total = 0.0
        for i, name in enumerate(("top", "middle", "bottom")):
            values = row_values[i]
            n = len(values)
            v = float(hero_vals[name])
            wins = float(np.searchsorted(values, v, side="left"))
            losses = float(n - np.searchsorted(values, v, side="right"))
            total += (wins - losses) / n
        exp_lines.append(total)
    exp_lines_mean = float(np.mean(exp_lines))

    # Composed approximate node EV from the joint facts alone (no learning).
    foul = joint["foul_rate"]
    opp_total = joint["joint_royalty"] + joint["joint_fl"]
    hb = float(hero_bust.mean())
    hr = float(hero_royalty.mean())
    hf = float(hero_fl.mean())
    bust_payoff = -6.0 * (1.0 - foul) - opp_total
    survive_payoff = (
        foul * (6.0 + hr + hf)
        + (1.0 - foul) * (hr + hf + exp_lines_mean)
        - opp_total
    )
    approx_ev = hb * bust_payoff + (1.0 - hb) * survive_payoff

    return {
        "seed": root["seed"],
        "keys": keys,
        "exact": exact_arr,
        "predicted": predicted,
        "common": float(residual.mean()),
        "spread": float(np.abs(residual - residual.mean()).mean()),
        "exact_mean": float(exact_arr.mean()),
        "exact_best": float(exact_arr.max()),
        "node_error": float(predicted.max() - exact_arr.max()),
        "hero_bust_mean": float(hero_bust.mean()),
        "hero_royalty_mean": float(hero_royalty.mean()),
        "hero_fl_mean": float(hero_fl.mean()),
        "joint": joint,
        "controls": controls,
        "exp_lines_mean": exp_lines_mean,
        "bust_payoff": bust_payoff,
        "approx_ev": approx_ev,
        "pool": list(cache.pool),
        "root": root,
    }


# ---------------------------------------------------------------------------
# Statistics.
# ---------------------------------------------------------------------------

def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _r2(design: np.ndarray, target: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(target)), design])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ beta
    ss_res = float(((target - fitted) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def row_summary(cards, capacity: int) -> str:
    if not cards:
        return "(empty)"
    if len(cards) == capacity:
        category = hand_category(evaluate_hand(list(cards), capacity))
        return f"{' '.join(cards)}  [{CATEGORY_NAMES[min(category, 8)]}]"
    return f"{' '.join(cards)}  [{len(cards)}/{capacity}]"


def print_case(case: dict) -> None:
    root = case["root"]
    joint = case["joint"]
    controls = case["controls"]
    print("\n" + "=" * 96)
    print(
        f"seed {root['seed']}   common {case['common']:+.3f}   spread {case['spread']:.3f}   "
        f"exact_mean {case['exact_mean']:+.3f}   node_err {case['node_error']:+.3f}"
    )
    for label, rows in (("BB (hero)", root["bb_board"]), ("BTN (opp)", root["btn_board"])):
        print(f"  {label}")
        for index, name in enumerate(("top", "mid", "bot")):
            print(f"    {name}: {row_summary(rows[index], ROW_CAPACITY[index])}")
    print(f"  draw {list(root['draw'])}   hero dead {list(root['bb_discards'])}")
    pool = case["pool"]
    print(f"  pool ({len(pool)}): {' '.join(pool)}")
    print(
        f"  OPP JOINT: ev {joint['joint_ev']:+.3f}  foul {joint['foul_rate']:.3f}  "
        f"royalty {joint['joint_royalty']:.3f}  fl {joint['joint_fl']:.3f}  std {joint['joint_std']:.2f}"
    )
    print(
        f"  OPP INDEP (current features): royalty {controls['indep_royalty']:.3f}  "
        f"fl {controls['indep_fl']:.3f}  locked_cat {controls['locked_cat_any']:.0f}  "
        f"overstatement {(controls['indep_royalty'] + controls['indep_fl'] - joint['joint_ev']):+.3f}"
    )
    print(
        f"  hero: bust {case['hero_bust_mean']:.2f}  royalty {case['hero_royalty_mean']:.2f}  "
        f"fl {case['hero_fl_mean']:.2f}   exp_lines {case['exp_lines_mean']:+.2f}   "
        f"approx_ev {case['approx_ev']:+.2f}"
    )
    order = np.argsort(-case["exact"])
    print(f"  {'exact':>9}{'pred':>9}{'resid':>8}   action")
    for i in order[:5]:
        payload = json.loads(case["keys"][i])
        placed = ", ".join(f"{card}->{row}" for card, row in payload["placements"])
        print(
            f"  {case['exact'][i]:>9.3f}{case['predicted'][i]:>9.3f}"
            f"{case['predicted'][i] - case['exact'][i]:>+8.2f}   "
            f"{placed} (discard {payload['discard']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=Path("D:/ofc_data/t4_first_model_pass1/evaluator_best.pt"),
    )
    parser.add_argument("--roots", type=int, default=600)
    parser.add_argument("--seed-base", type=int, default=5_000_000)
    parser.add_argument("--show", type=int, default=10)
    parser.add_argument(
        "--out", type=Path,
        default=Path("ai/reports/t4_first_node_error_diagnosis_20260729/analysis.json"),
    )
    args = parser.parse_args()

    started = time.time()
    model, mean, std = load_model(args.model)
    roots = [probe.sample_random_root(args.seed_base + i) for i in range(args.roots)]
    print(f"labelling {len(roots)} roots with the Rust exact solver ...", flush=True)
    labelled = label_roots(
        roots,
        workspace_root=Path.cwd(),
        scratch=Path.cwd() / "ai" / "reports" / "_common_error_scratch",
        chunk_size=256,
    )
    tables = [
        {row["action_key"]: float(row["ev"]) for row in payload["actions"]}
        for payload in labelled
    ]
    print(f"labelled in {time.time() - started:.1f}s; computing joint metrics ...", flush=True)

    cases = []
    for index, (root, table) in enumerate(zip(roots, tables)):
        cases.append(analyze_root(root, model, mean, std, table))
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(roots)}  ({time.time() - started:.0f}s)", flush=True)

    common = np.asarray([c["common"] for c in cases])
    spread = np.asarray([c["spread"] for c in cases])
    print(
        f"\nroots {len(cases)}   mean|common| {np.abs(common).mean():.3f}   "
        f"bias {common.mean():+.3f}   mean spread {spread.mean():.3f}   "
        f"share within 1pt {(np.abs(common) <= 1.0).mean():.3f}"
    )

    # Candidate and control variable matrices.
    def col(fn):
        return np.asarray([fn(c) for c in cases], dtype=np.float64)

    variables = {
        # --- new joint facts (not in current features) ---
        "joint_ev": col(lambda c: c["joint"]["joint_ev"]),
        "joint_foul_rate": col(lambda c: c["joint"]["foul_rate"]),
        "joint_royalty": col(lambda c: c["joint"]["joint_royalty"]),
        "joint_fl": col(lambda c: c["joint"]["joint_fl"]),
        "joint_std": col(lambda c: c["joint"]["joint_std"]),
        "overstate_royfl": col(
            lambda c: c["controls"]["indep_royalty"] + c["controls"]["indep_fl"]
            - c["joint"]["joint_royalty"] - c["joint"]["joint_fl"]
        ),
        "overstate_ev": col(
            lambda c: c["controls"]["indep_royalty"] + c["controls"]["indep_fl"]
            - c["joint"]["joint_ev"]
        ),
        "exp_lines_mean": col(lambda c: c["exp_lines_mean"]),
        "bust_payoff": col(lambda c: c["bust_payoff"]),
        "approx_ev": col(lambda c: c["approx_ev"]),
        # --- controls: expressible from current features ---
        "indep_royalty": col(lambda c: c["controls"]["indep_royalty"]),
        "indep_fl": col(lambda c: c["controls"]["indep_fl"]),
        "locked_cat_any": col(lambda c: c["controls"]["locked_cat_any"]),
        "opp_jokers": col(lambda c: c["controls"]["opp_jokers"]),
        "pool_jokers": col(lambda c: c["controls"]["pool_jokers"]),
        "cat_slack_mid_bot": col(lambda c: c["controls"]["cat_slack_mid_bot"]),
        "cat_slack_top_mid": col(lambda c: c["controls"]["cat_slack_top_mid"]),
        "pool_high_frac": col(lambda c: c["controls"]["pool_high_frac"]),
        "hero_bust_mean": col(lambda c: c["hero_bust_mean"]),
        "hero_royalty_mean": col(lambda c: c["hero_royalty_mean"]),
        "hero_fl_mean": col(lambda c: c["hero_fl_mean"]),
        # --- reference ---
        "exact_mean": col(lambda c: c["exact_mean"]),
    }

    print("\n### correlation with signed common error")
    for name, values in variables.items():
        print(f"  {name:>20}: r = {_corr(values, common):+.3f}")

    control_names = [
        "indep_royalty", "indep_fl", "locked_cat_any", "opp_jokers",
        "pool_jokers", "cat_slack_mid_bot", "cat_slack_top_mid",
        "pool_high_frac", "hero_bust_mean", "hero_royalty_mean", "hero_fl_mean",
    ]
    new_names = [
        "joint_ev", "joint_foul_rate", "joint_royalty", "joint_fl", "joint_std",
        "exp_lines_mean", "bust_payoff", "approx_ev",
    ]
    controls_x = np.column_stack([variables[n] for n in control_names])
    r2_controls = _r2(controls_x, common)
    full_x = np.column_stack([controls_x] + [variables[n] for n in new_names])
    r2_full = _r2(full_x, common)
    print(f"\n### linear R^2 against common error")
    print(f"  controls only (current-feature facts): {r2_controls:.3f}")
    print(f"  controls + joint opponent facts:       {r2_full:.3f}")
    for name in new_names:
        r2_one = _r2(np.column_stack([controls_x, variables[name]]), common)
        print(f"    + {name:>16} alone: {r2_one:.3f}  (delta {r2_one - r2_controls:+.3f})")

    exact_mean_arr = variables["exact_mean"]
    approx = variables["approx_ev"]
    print("\n### composed approx_ev as a node value (no learning)")
    print(f"  corr(approx_ev, exact_mean) = {_corr(approx, exact_mean_arr):+.3f}")
    print(f"  MAE(approx_ev - exact_mean) = {np.abs(approx - exact_mean_arr).mean():.3f}")
    pred_mean = exact_mean_arr + common
    print(f"  model:  corr(pred_mean, exact_mean) = {_corr(pred_mean, exact_mean_arr):+.3f}   "
          f"MAE {np.abs(common).mean():.3f}")

    print("\n### stratified common error by hero bust share")
    hb = variables["hero_bust_mean"]
    for low, high, label in ((0.0, 0.001, "never"), (0.001, 0.999, "partial"), (0.999, 1.01, "always")):
        mask = (hb >= low) & (hb < high)
        if mask.sum():
            print(
                f"  hero bust {label:>7}: n={int(mask.sum()):4d}  "
                f"mean|common| {np.abs(common[mask]).mean():.3f}  "
                f"bias {common[mask].mean():+.3f}"
            )

    print("\n### stratified |common| by joker counts")
    for label, values in (("pool_jokers", variables["pool_jokers"]),
                          ("opp_jokers", variables["opp_jokers"])):
        for level in sorted(set(values.tolist())):
            mask = values == level
            print(
                f"  {label}={level:.0f}: n={int(mask.sum()):4d}  "
                f"mean|common| {np.abs(common[mask]).mean():.3f}  "
                f"bias {common[mask].mean():+.3f}"
            )

    print("\n### stratified common error by joint foul rate")
    foul = variables["joint_foul_rate"]
    for low, high in ((0.0, 0.001), (0.001, 0.05), (0.05, 0.25), (0.25, 0.75), (0.75, 1.01)):
        mask = (foul >= low) & (foul < high)
        if mask.sum():
            print(
                f"  foul in [{low:.3f}, {high:.3f}): n={int(mask.sum()):4d}  "
                f"mean|common| {np.abs(common[mask]).mean():.3f}  "
                f"bias {common[mask].mean():+.3f}"
            )

    order = np.argsort(common)
    print(f"\n### {args.show} most NEGATIVE common error (model too pessimistic)")
    for i in order[: args.show]:
        print_case(cases[i])
    print(f"\n### {args.show} most POSITIVE common error (model too optimistic)")
    for i in order[::-1][: args.show]:
        print_case(cases[i])

    payload = {
        "schema": "ofc_t4_first_common_error_analysis/v1",
        "model": str(args.model),
        "roots": len(cases),
        "seed_base": args.seed_base,
        "mean_abs_common": float(np.abs(common).mean()),
        "bias": float(common.mean()),
        "correlations": {
            name: _corr(values, common) for name, values in variables.items()
        },
        "r2_controls": r2_controls,
        "r2_full": r2_full,
        "elapsed_seconds": time.time() - started,
        "per_root": [
            {
                "seed": c["seed"],
                "common": c["common"],
                "spread": c["spread"],
                "exact_mean": c["exact_mean"],
                "node_error": c["node_error"],
                "joint": c["joint"],
                "controls": c["controls"],
                "exp_lines_mean": c["exp_lines_mean"],
                "bust_payoff": c["bust_payoff"],
                "approx_ev": c["approx_ev"],
                "hero_bust_mean": c["hero_bust_mean"],
                "hero_royalty_mean": c["hero_royalty_mean"],
                "hero_fl_mean": c["hero_fl_mean"],
            }
            for c in cases
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}   elapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
