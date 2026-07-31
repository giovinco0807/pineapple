"""Evaluate a union of T3 model pools against exact teacher JSONL.

This reports whether exact reranking over the union pool can recover the exact
best action.  It is meant for practical pruning checks where several specialist
models contribute candidates.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker
from ai.tutor.diagnose_t3_model_from_t2_miss import compact_action_key, score_model_t3
from ai.tutor.evaluate_t3_exact_teacher_models import parse_model_pair, parse_topks, position_of, stats
from ai.tutor.exact_late import CardNormalizer, normalize_board


RANK_VALUE = {rank: value for value, rank in enumerate("23456789TJQKA", start=2)}


def row_name(row: Any) -> str:
    return {"mid": "middle", "bot": "bottom"}.get(str(row), str(row))


def rank_value(card: Any) -> int:
    text = str(card or "")
    if text.startswith("X"):
        return 15
    return RANK_VALUE.get(text[:1], 0)


def low_card_score(card: Any) -> float:
    value = rank_value(card)
    if value <= 0:
        return 0.0
    return max(0.0, 15.0 - float(value))


def insurance_priority(action: dict[str, Any]) -> float:
    """Heuristic insurance for T3 candidate pools.

    This intentionally uses only the candidate action shape, not teacher EV or
    candidate ordering.  The target is the observed miss pattern where models
    over-promote top placements and drop low/medium bottom-middle placements.
    """
    placements = [(str(card), row_name(row)) for card, row in (action.get("placements") or [])]
    discard = action.get("discard")
    row_counts = {"top": 0, "middle": 0, "bottom": 0}
    row_low = {"top": 0.0, "middle": 0.0, "bottom": 0.0}
    row_high = {"top": 0.0, "middle": 0.0, "bottom": 0.0}
    for card, row in placements:
        if row not in row_counts:
            continue
        value = rank_value(card)
        row_counts[row] += 1
        row_low[row] += low_card_score(card)
        row_high[row] += max(0.0, float(value) - 10.0)

    priority = 0.0
    if row_counts["top"] == 0:
        priority += 100.0
    if row_counts["bottom"] == 2:
        priority += 35.0
    if row_counts["middle"] == 2:
        priority += 12.0
    priority += row_low["bottom"] * 2.0
    priority += row_low["middle"] * 1.0
    priority -= row_high["top"] * 4.0
    priority -= row_counts["top"] * 8.0
    priority += max(0.0, rank_value(discard) - 11.0) * 0.5
    return priority


def high_top_anchor_priority(action: dict[str, Any]) -> float:
    placements = [(str(card), row_name(row)) for card, row in (action.get("placements") or [])]
    top_values = [rank_value(card) for card, row in placements if row == "top"]
    side_cards = [(card, row) for card, row in placements if row in {"middle", "bottom"}]
    if not top_values or not side_cards or max(top_values) < 11:
        return 0.0
    priority = 100.0 + max(top_values) * 2.0
    for card, row in side_cards:
        if row == "bottom":
            priority += 8.0
        priority += low_card_score(card) * 1.5
    return priority


def low_top_side_priority(action: dict[str, Any]) -> float:
    placements = [(str(card), row_name(row)) for card, row in (action.get("placements") or [])]
    top_values = [rank_value(card) for card, row in placements if row == "top"]
    side_cards = [(card, row) for card, row in placements if row in {"middle", "bottom"}]
    if not top_values or not side_cards or min(top_values) > 8:
        return 0.0
    priority = 100.0 + sum(max(0.0, 9.0 - float(value)) for value in top_values)
    for card, row in side_cards:
        value = rank_value(card)
        if row == "bottom":
            priority += 5.0
        priority += max(0.0, float(value) - 8.0) * 1.5
    return priority


def pair_middle_bottom_priority(action: dict[str, Any]) -> float:
    placements = [(str(card), row_name(row)) for card, row in (action.get("placements") or [])]
    if len(placements) != 2:
        return 0.0
    if {row for _card, row in placements} != {"middle", "bottom"}:
        return 0.0
    return 100.0 + sum(low_card_score(card) for card, _row in placements)


def select_insurance_keys(actions_by_key: dict[Any, dict[str, Any]], count: int, mode: str = "general") -> list[Any]:
    if count <= 0:
        return []
    if mode != "category":
        return sorted(
            actions_by_key,
            key=lambda key: (-insurance_priority(actions_by_key[key]), key),
        )[:count]

    selected: list[Any] = []
    selected_set: set[Any] = set()
    categories = (
        insurance_priority,
        high_top_anchor_priority,
        low_top_side_priority,
        pair_middle_bottom_priority,
    )
    for scorer in categories:
        if len(selected) >= count:
            break
        ranked = sorted(
            actions_by_key,
            key=lambda key: (-scorer(actions_by_key[key]), key),
        )
        for key in ranked:
            if key in selected_set:
                continue
            if scorer(actions_by_key[key]) <= 0.0:
                break
            selected.append(key)
            selected_set.add(key)
            break

    if len(selected) < count:
        for key in sorted(actions_by_key, key=lambda key: (-insurance_priority(actions_by_key[key]), key)):
            if key in selected_set:
                continue
            selected.append(key)
            selected_set.add(key)
            if len(selected) >= count:
                break
    return selected


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_model_pairs(values: list[str], device: torch.device) -> dict[str, dict[str, ActionValueReranker]]:
    loaded: dict[str, dict[str, ActionValueReranker]] = {}
    for value in values:
        name, paths = parse_model_pair(value)
        loaded[name] = {
            pos: ActionValueReranker.from_checkpoint(path, map_location=device).to(device).eval()
            for pos, path in paths.items()
        }
    return loaded


def evaluate_union(
    *,
    rows: list[dict[str, Any]],
    model_args: list[str],
    topks: list[int],
    insurance_candidates: int,
    insurance_mode: str,
    near_full_expand_gap: int,
    miss_threshold: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    normalizer = CardNormalizer()
    models = load_model_pairs(model_args, device)
    max_k = max(topks)

    losses = {k: [] for k in topks}
    recalls = {k: 0 for k in topks}
    pool_sizes = {k: [] for k in topks}
    by_position: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "losses": {k: [] for k in topks}, "recalls": {k: 0 for k in topks}, "pool_sizes": {k: [] for k in topks}}
    )
    worst = {k: None for k in topks}
    misses: list[dict[str, Any]] = []

    for row_i, row in enumerate(rows):
        position = position_of(row)
        board = normalize_board(row["board"], normalizer)
        opponent_board = normalize_board(row.get("opponent_board") or {}, normalizer)

        exact_items = []
        for candidate in row.get("candidates") or []:
            action = {"placements": candidate.get("placements") or [], "discard": candidate.get("discard")}
            exact_items.append((compact_action_key(action), float(candidate.get("ev", 0.0)), action))
        if not exact_items:
            continue
        exact_by_key = {key: (score, action) for key, score, action in exact_items}
        exact_best_key, exact_best_score, exact_best_action = exact_items[0]

        model_tops: dict[str, list[dict[str, Any]]] = {}
        model_rank_by_key: dict[str, dict[str, int]] = {}
        for name, pair in models.items():
            model_eval = score_model_t3(
                model=pair[position],
                device=device,
                board=board,
                opponent_board=opponent_board,
                draw=row["dealt"],
                dead=row.get("exclude") or [],
                is_btn=position == "btn",
                batch_size=batch_size,
                top_n=max_k,
            )
            model_tops[name] = list(model_eval.get("top") or [])
            for rank, item in enumerate(model_tops[name], start=1):
                key = compact_action_key(item["action"])
                model_rank_by_key.setdefault(key, {})[name] = rank

        pos_bucket = by_position[position]
        pos_bucket["rows"] += 1

        for k in topks:
            pool: dict[Any, dict[str, Any]] = {}
            for top in model_tops.values():
                for item in top[: min(k, len(top))]:
                    key = compact_action_key(item["action"])
                    pool.setdefault(key, item["action"])
            if insurance_candidates > 0:
                legal_by_key = {key: action for key, _score, action in exact_items}
                added = 0
                for key in select_insurance_keys(legal_by_key, insurance_candidates, mode=insurance_mode):
                    if key in pool:
                        continue
                    pool[key] = legal_by_key[key]
                    added += 1
            if near_full_expand_gap > 0 and len(pool) >= max(0, len(exact_items) - near_full_expand_gap):
                for key, _score, action in exact_items:
                    pool.setdefault(key, action)
            pool_items = [
                (key, exact_by_key[key][0], exact_by_key[key][1])
                for key in pool
                if key in exact_by_key
            ]
            best_pool = max(pool_items, key=lambda item: item[1]) if pool_items else None
            best_pool_score = float(best_pool[1]) if best_pool else -999.0
            loss = max(0.0, exact_best_score - best_pool_score)
            hit = exact_best_key in pool

            losses[k].append(float(loss))
            pool_sizes[k].append(len(pool))
            pos_bucket["losses"][k].append(float(loss))
            pos_bucket["pool_sizes"][k].append(len(pool))
            if hit:
                recalls[k] += 1
                pos_bucket["recalls"][k] += 1
            if worst[k] is None or loss > worst[k]["loss"]:
                worst[k] = {
                    "row": row_i,
                    "position": position,
                    "loss": float(loss),
                    "dealt": row["dealt"],
                    "pool_size": len(pool),
                    "exact_action": exact_best_action,
                }
            if loss > miss_threshold:
                best_pool_action = exact_by_key[best_pool[0]][1] if best_pool else None
                misses.append(
                    {
                        "row": row_i,
                        "position": position,
                        "k": int(k),
                        "loss": float(loss),
                        "dealt": row["dealt"],
                        "pool_size": len(pool),
                        "exact_action": exact_best_action,
                        "best_pool_action": best_pool_action,
                        "exact_best_model_ranks": model_rank_by_key.get(exact_best_key, {}),
                    }
                )

    n = len(next(iter(losses.values()))) if losses else 0
    out = {
        "rows": n,
        "models": [parse_model_pair(value)[0] for value in model_args],
        "insurance_candidates": int(insurance_candidates),
        "insurance_mode": str(insurance_mode),
        "near_full_expand_gap": int(near_full_expand_gap),
        "pool": {},
        "by_position": {},
        "miss_threshold": float(miss_threshold),
        "misses": misses,
    }
    for k in topks:
        sizes = pool_sizes[k]
        out["pool"][str(k)] = {
            "recall": recalls[k] / n if n else 0.0,
            "ev_loss": stats(losses[k]),
            "pool_size": {
                "mean": float(np.mean(sizes)) if sizes else 0.0,
                "p95": float(np.percentile(sizes, 95)) if sizes else 0.0,
                "max": int(max(sizes)) if sizes else 0,
            },
            "worst": worst[k],
        }
    for pos, bucket in by_position.items():
        pos_n = int(bucket["rows"])
        out["by_position"][pos] = {
            "rows": pos_n,
            "pool": {
                str(k): {
                    "recall": bucket["recalls"][k] / pos_n if pos_n else 0.0,
                    "ev_loss": stats(bucket["losses"][k]),
                    "pool_size": {
                        "mean": float(np.mean(bucket["pool_sizes"][k])) if bucket["pool_sizes"][k] else 0.0,
                        "p95": float(np.percentile(bucket["pool_sizes"][k], 95)) if bucket["pool_sizes"][k] else 0.0,
                        "max": int(max(bucket["pool_sizes"][k])) if bucket["pool_sizes"][k] else 0,
                    },
                }
                for k in topks
            },
        }
    return out


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate union T3 exact-rerank pools")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--model", action="append", required=True, help="name=bb_path,btn_path")
    parser.add_argument("--topks", default="5,8,10,15,20")
    parser.add_argument("--insurance-candidates", type=int, default=0)
    parser.add_argument("--insurance-mode", choices=("general", "category"), default="general")
    parser.add_argument("--near-full-expand-gap", type=int, default=0)
    parser.add_argument("--miss-threshold", type=float, default=0.1)
    parser.add_argument("--miss-output", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = list(iter_jsonl(Path(args.teacher)))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    result = {
        "teacher": str(args.teacher),
        "topks": parse_topks(args.topks),
        "union": evaluate_union(
            rows=rows,
            model_args=args.model,
            topks=parse_topks(args.topks),
            insurance_candidates=max(0, int(args.insurance_candidates)),
            insurance_mode=str(args.insurance_mode),
            near_full_expand_gap=max(0, int(args.near_full_expand_gap)),
            miss_threshold=float(args.miss_threshold),
            device=device,
            batch_size=args.batch_size,
        ),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.miss_output:
        miss_path = Path(args.miss_output)
        miss_path.parent.mkdir(parents=True, exist_ok=True)
        with miss_path.open("w", encoding="utf-8") as handle:
            for miss in result["union"]["misses"]:
                handle.write(json.dumps(miss, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
