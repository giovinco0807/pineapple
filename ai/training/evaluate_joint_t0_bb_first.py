"""Evaluate BB-first joint T0 placements for review hands.

This is a fast row-aware T0 diagnostic.  It does not hard-code placement rules.
For each hand:

1. Deal five T0 cards to BB, then five to BTN.
2. Evaluate all T0 placements for each seat with prob_engine PE.
3. Keep a compact pool of promising BB/BTN placements.
4. Score every BB-pool x BTN-pool pair using row distributions, royalties,
   FL EV, bust probability, and approximate line/scoop equity.
5. Because BB acts first, select BB by its best value after BTN chooses the
   response that is worst for BB.

The output is intended for human review before building a deeper coupled
rollout teacher.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.engine.encoding import ALL_CARDS  # noqa: E402
from ai.prob_engine_wrapper import PROB_ENGINE_PATH, evaluate_candidates  # noqa: E402


ROWS = ("top", "middle", "bottom")


def row_cards(candidate: dict[str, Any], row: str) -> list[str]:
    return [card for card, pos in candidate.get("placements", []) if normalize_row(pos) == row]


def normalize_row(row: str) -> str:
    if row in ("mid", "middle"):
        return "middle"
    if row in ("bot", "bottom"):
        return "bottom"
    return row


def board_from_candidate(candidate: dict[str, Any]) -> dict[str, list[str]]:
    board = {"top": [], "middle": [], "bottom": []}
    for card, pos in candidate.get("placements", []):
        board[normalize_row(pos)].append(card)
    for row in ROWS:
        board[row].sort(key=card_sort_key)
    return board


def card_sort_key(card: str) -> tuple[int, str]:
    ranks = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
    if card.startswith("X"):
        return (99, card)
    return (ranks.get(card[0], 0), card[1:])


def action_label(candidate: dict[str, Any]) -> str:
    board = board_from_candidate(candidate)
    return (
        f"T:{' '.join(board['top']) or '-'} | "
        f"M:{' '.join(board['middle']) or '-'} | "
        f"B:{' '.join(board['bottom']) or '-'}"
    )


def board_request(req_id: dict[str, Any], candidate: dict[str, Any], exclude: list[str], position: str) -> dict[str, Any]:
    board = board_from_candidate(candidate)
    return {
        "id": req_id,
        "mode": "board",
        "top": ",".join(board["top"]),
        "mid": ",".join(board["middle"]),
        "bot": ",".join(board["bottom"]),
        "exclude": ",".join(exclude),
        "turn": 1,
        "position": position,
    }


def run_batch(requests: list[dict[str, Any]], out_dir: Path, engine_path: str | None) -> dict[str, dict[str, Any]]:
    if not requests:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}_{random.randint(0, 999999):06d}"
    req_path = out_dir / f"joint_t0_batch_{stamp}.requests.jsonl"
    resp_path = out_dir / f"joint_t0_batch_{stamp}.responses.jsonl"
    with req_path.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    exe = engine_path or str(PROB_ENGINE_PATH)
    result = subprocess.run(
        [exe, "--mode", "batch", "--input", str(req_path), "--output", str(resp_path)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(f"prob_engine batch failed: {result.stderr}")

    responses: dict[str, dict[str, Any]] = {}
    with resp_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            key = json.dumps(item.get("id"), sort_keys=True)
            if not item.get("ok"):
                raise RuntimeError(f"prob_engine request failed: {item.get('error')}")
            responses[key] = item["result"]
    req_path.unlink(missing_ok=True)
    resp_path.unlink(missing_ok=True)
    return responses


def select_pool(candidates: list[dict[str, Any]], pool_size: int) -> list[dict[str, Any]]:
    """Keep EV leaders plus enough FL-heavy alternatives for review."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(items: list[dict[str, Any]], limit: int) -> None:
        for cand in items:
            if len(selected) >= pool_size:
                return
            key = action_label(cand)
            if key in seen:
                continue
            selected.append(cand)
            seen.add(key)
            if len(selected) >= limit:
                return

    ev_sorted = sorted(candidates, key=lambda c: float(c.get("ev", -1e9)), reverse=True)
    fl_sorted = sorted(candidates, key=lambda c: float(c.get("fl_rate", 0.0)), reverse=True)
    safe_sorted = sorted(candidates, key=lambda c: (float(c.get("bust_prob", 1.0)), -float(c.get("ev", -1e9))))

    add(ev_sorted, max(pool_size * 2 // 3, 1))
    add(fl_sorted, max(pool_size * 5 // 6, 1))
    add(safe_sorted, pool_size)
    if len(selected) < pool_size:
        add(ev_sorted, pool_size)
    return selected


def hist_gt_lt(a: list[float], b: list[float]) -> tuple[float, float, float]:
    p_gt = 0.0
    p_lt = 0.0
    p_tie = 0.0
    prefix_b = []
    total = 0.0
    for v in b:
        prefix_b.append(total)
        total += v
    suffix_b = []
    total = 0.0
    for v in reversed(b):
        suffix_b.append(total)
        total += v
    suffix_b.reverse()
    for i, pa in enumerate(a):
        if pa == 0:
            continue
        p_gt += pa * prefix_b[i]
        p_lt += pa * suffix_b[i]
        p_tie += pa * b[i]
    return p_gt, p_lt, p_tie


def pair_score(bb_eval: dict[str, Any], btn_eval: dict[str, Any]) -> dict[str, float]:
    bb = bb_eval["evaluation"]
    btn = btn_eval["evaluation"]
    line_ev = 0.0
    p_sweep = 1.0
    p_swept = 1.0
    row_parts: dict[str, float] = {}
    for row in ("top", "mid", "bot"):
        row_key = "middle" if row == "mid" else "bottom" if row == "bot" else row
        bb_hist = bb[row]["fine_hist"]
        btn_hist = btn[row]["fine_hist"]
        p_gt, p_lt, _p_tie = hist_gt_lt(bb_hist, btn_hist)
        row_ev = p_gt - p_lt
        row_parts[f"{row_key}_line_ev"] = row_ev
        line_ev += row_ev
        p_sweep *= p_gt
        p_swept *= p_lt

    live_weight = (1.0 - float(bb["bust_prob"])) * (1.0 - float(btn["bust_prob"]))
    scoop_ev = 3.0 * (p_sweep - p_swept)
    board_ev = float(bb["ev"]) - float(btn["ev"])
    utility = board_ev + live_weight * (line_ev + scoop_ev)

    return {
        "utility": utility,
        "board_ev_diff": board_ev,
        "line_ev": line_ev,
        "scoop_ev": scoop_ev,
        "live_weight": live_weight,
        "bb_ev": float(bb["ev"]),
        "btn_ev": float(btn["ev"]),
        "bb_fl": float(bb["fl_rate"]),
        "btn_fl": float(btn["fl_rate"]),
        "bb_bust": float(bb["bust_prob"]),
        "btn_bust": float(btn["bust_prob"]),
        **row_parts,
    }


def get_eval(evals: dict[str, dict[str, Any]], side: str, idx: int) -> dict[str, Any]:
    key = json.dumps({"side": side, "idx": idx}, sort_keys=True)
    return evals[key]


def evaluate_hand(
    hand_no: int,
    rng: random.Random,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    deck = list(ALL_CARDS)
    rng.shuffle(deck)
    bb_dealt = deck[:5]
    btn_dealt = deck[5:10]

    bb_raw = evaluate_candidates(
        top=[],
        mid=[],
        bot=[],
        dealt=bb_dealt,
        exclude=btn_dealt,
        turn=0,
        position="bb",
        engine_path=args.engine_path,
    )["candidates"]
    btn_raw = evaluate_candidates(
        top=[],
        mid=[],
        bot=[],
        dealt=btn_dealt,
        exclude=bb_dealt,
        turn=0,
        position="btn",
        engine_path=args.engine_path,
    )["candidates"]

    bb_pool = select_pool(bb_raw, args.pool_size)
    btn_pool = select_pool(btn_raw, args.pool_size)

    requests: list[dict[str, Any]] = []
    for idx, cand in enumerate(bb_pool):
        requests.append(board_request({"side": "bb", "idx": idx}, cand, btn_dealt, "bb"))
    for idx, cand in enumerate(btn_pool):
        requests.append(board_request({"side": "btn", "idx": idx}, cand, bb_dealt, "btn"))
    evals = run_batch(requests, out_dir, args.engine_path)

    bb_options = []
    for bb_idx, bb_cand in enumerate(bb_pool):
        responses = []
        for btn_idx, btn_cand in enumerate(btn_pool):
            metrics = pair_score(get_eval(evals, "bb", bb_idx), get_eval(evals, "btn", btn_idx))
            responses.append({
                "btn_idx": btn_idx,
                "btn_action": action_label(btn_cand),
                "metrics": metrics,
                "btn_candidate_pe": {
                    "ev": float(btn_cand.get("ev", 0.0)),
                    "fl_rate": float(btn_cand.get("fl_rate", 0.0)),
                    "bust_prob": float(btn_cand.get("bust_prob", 0.0)),
                },
            })
        responses.sort(key=lambda r: r["metrics"]["utility"])
        best_response = responses[0]
        bb_options.append({
            "bb_idx": bb_idx,
            "bb_action": action_label(bb_cand),
            "bb_candidate_pe": {
                "ev": float(bb_cand.get("ev", 0.0)),
                "fl_rate": float(bb_cand.get("fl_rate", 0.0)),
                "bust_prob": float(bb_cand.get("bust_prob", 0.0)),
            },
            "btn_best_response": best_response,
            "top_btn_responses": responses[: args.show_responses],
        })

    bb_options.sort(key=lambda r: r["btn_best_response"]["metrics"]["utility"], reverse=True)
    return {
        "hand": hand_no,
        "bb_dealt": bb_dealt,
        "btn_dealt": btn_dealt,
        "bb_pool_size": len(bb_pool),
        "btn_pool_size": len(btn_pool),
        "chosen": bb_options[0],
        "bb_options": bb_options[: args.show_top],
    }


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def write_markdown(records: list[dict[str, Any]], path: Path, args: argparse.Namespace, elapsed_s: float) -> None:
    lines = [
        "# BB-first Joint T0 Review",
        "",
        f"- hands: {len(records)}",
        f"- seed: {args.seed}",
        f"- pool_size: {args.pool_size}",
        f"- scoring: PE board EV diff + approximate row line/scoop equity",
        f"- elapsed_s: {elapsed_s:.1f}",
        "",
        "Note: BTN response is selected after BB's T0 placement. This is row-aware, but it is still an approximate T0 diagnostic, not a full coupled T1-T4 rollout teacher.",
        "",
    ]
    for rec in records:
        chosen = rec["chosen"]
        br = chosen["btn_best_response"]
        m = br["metrics"]
        lines.extend([
            f"## Hand {rec['hand']}",
            "",
            f"- BB dealt: {' '.join(rec['bb_dealt'])}",
            f"- BTN dealt: {' '.join(rec['btn_dealt'])}",
            f"- Selected BB: `{chosen['bb_action']}`",
            f"- BTN response: `{br['btn_action']}`",
            f"- utility={m['utility']:+.3f}, board_ev_diff={m['board_ev_diff']:+.3f}, line_ev={m['line_ev']:+.3f}, scoop_ev={m['scoop_ev']:+.3f}",
            f"- BB FL/bust={pct(m['bb_fl'])}/{pct(m['bb_bust'])}, BTN FL/bust={pct(m['btn_fl'])}/{pct(m['btn_bust'])}",
            "",
            "| rank | BB action | BTN best response | utility | BB FL | BB bust | BTN FL | BTN bust |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ])
        for rank, opt in enumerate(rec["bb_options"], start=1):
            bm = opt["btn_best_response"]["metrics"]
            lines.append(
                f"| {rank} | `{opt['bb_action']}` | `{opt['btn_best_response']['btn_action']}` | "
                f"{bm['utility']:+.3f} | {pct(bm['bb_fl'])} | {pct(bm['bb_bust'])} | "
                f"{pct(bm['btn_fl'])} | {pct(bm['btn_bust'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BB-first joint T0 review hands")
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--pool-size", type=int, default=24)
    parser.add_argument("--show-top", type=int, default=5)
    parser.add_argument("--show-responses", type=int, default=3)
    parser.add_argument("--out-dir", default="ai/models/candidate_runs/fl-route-reranker-v2-medium-20260520")
    parser.add_argument("--name", default="joint_t0_bb_first_10")
    parser.add_argument("--engine-path", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    start = time.time()
    records = []
    for hand_no in range(1, args.hands + 1):
        rec = evaluate_hand(hand_no, rng, args, out_dir)
        records.append(rec)
        chosen = rec["chosen"]
        metric = chosen["btn_best_response"]["metrics"]
        print(
            f"hand={hand_no} utility={metric['utility']:+.3f} "
            f"bb_fl={metric['bb_fl']:.3f} btn_fl={metric['btn_fl']:.3f} "
            f"bb={chosen['bb_action']} btn={chosen['btn_best_response']['btn_action']}",
            flush=True,
        )

    elapsed_s = time.time() - start
    jsonl_path = out_dir / f"{args.name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "hands": args.hands,
        "seed": args.seed,
        "pool_size": args.pool_size,
        "elapsed_s": elapsed_s,
        "jsonl": str(jsonl_path),
        "avg_selected_utility": sum(r["chosen"]["btn_best_response"]["metrics"]["utility"] for r in records) / max(len(records), 1),
        "avg_selected_bb_fl": sum(r["chosen"]["btn_best_response"]["metrics"]["bb_fl"] for r in records) / max(len(records), 1),
        "avg_selected_btn_fl": sum(r["chosen"]["btn_best_response"]["metrics"]["btn_fl"] for r in records) / max(len(records), 1),
        "avg_selected_bb_bust": sum(r["chosen"]["btn_best_response"]["metrics"]["bb_bust"] for r in records) / max(len(records), 1),
        "avg_selected_btn_bust": sum(r["chosen"]["btn_best_response"]["metrics"]["btn_bust"] for r in records) / max(len(records), 1),
    }
    summary_path = out_dir / f"{args.name}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = out_dir / f"{args.name}.md"
    write_markdown(records, md_path, args, elapsed_s)
    print(f"wrote {jsonl_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
