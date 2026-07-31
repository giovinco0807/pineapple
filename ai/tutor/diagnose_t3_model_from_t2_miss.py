"""Diagnose T3 model errors behind a T2 capped-exact miss.

Given a T2 decision row and a miss report from ``run_t2_exact_oracle.py``, this
script expands selected T2 actions into sampled T3 deals.  For each concrete T3
position it compares the single T3 action-value model's best action with Rust
T3 exact.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker
from ai.tutor.benchmark_t2_t3_value_model import (
    choose_draws,
    encode_post_t3_candidates,
    make_action,
    normalized_dead_cards,
    predict_scores,
    remaining_deck,
)
from ai.tutor.exact_late import CardNormalizer, action_key, apply_action, board_to_dict, evaluate_late_position, normalize_board
from ai.tutor.run_t2_exact_oracle import candidate_score


DEFAULT_T3_BB_MODEL = (
    "D:/ofc-pineapple-data/t3_jokerfix_extra_20260605/models/"
    "t3-20k-plus-gen5k-hardneginit-bb/action_value_best.pt"
)
DEFAULT_T3_BTN_MODEL = (
    "D:/ofc-pineapple-data/t3_jokerfix_extra_20260605/models/"
    "t3-20k-plus-gen5k-hardneginit-btn/action_value_best.pt"
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def row_name(row: Any) -> str:
    raw = str(row)
    return {"mid": "middle", "bot": "bottom"}.get(raw, raw)


def compact_action_key(action: dict[str, Any] | None) -> str:
    action = action or {}
    return json.dumps(
        {
            "placements": tuple(sorted((str(card), row_name(row)) for card, row in (action.get("placements") or []))),
            "discard": action.get("discard"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def select_t2_actions(miss: dict[str, Any], source: dict[str, Any], max_actions: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(label: str, item: dict[str, Any] | None) -> None:
        if not item:
            return
        action = item.get("action") or item
        key = compact_action_key(action)
        if key in seen:
            return
        seen.add(key)
        selected.append({"label": label, "action": action, "score": candidate_score(item)})

    add("source_best", miss.get("source_best"))
    add("oracle_best", miss.get("oracle_best"))
    add("source_best_under_oracle", miss.get("source_best_under_oracle"))
    add("oracle_best_under_source", miss.get("oracle_best_under_source"))
    for idx, item in enumerate(miss.get("oracle_top_candidates") or [], start=1):
        add(f"oracle_top{idx}", item)
        if len(selected) >= max_actions:
            break
    if len(selected) < max_actions:
        for idx, item in enumerate(source.get("candidates") or [], start=1):
            add(f"source_row_top{idx}", item)
            if len(selected) >= max_actions:
                break
    return selected[:max_actions]


def metric_score(candidate: dict[str, Any] | None) -> float:
    candidate = candidate or {}
    metrics = candidate.get("metrics") or {}
    if metrics.get("score") is not None:
        return float(metrics["score"])
    return candidate_score(candidate)


def metric_value(candidate: dict[str, Any], *names: str, default: float = 0.0) -> float:
    metrics = candidate.get("metrics") or {}
    for name in names:
        if metrics.get(name) is not None:
            return float(metrics[name])
        if candidate.get(name) is not None:
            return float(candidate[name])
    return float(default)


def flatten_exact_candidate(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    action = candidate.get("action") or {}
    metrics = candidate.get("metrics") or {}
    fl_type_rates = metrics.get("fl_type_rates") or candidate.get("fl_type_rates") or {}
    out = {
        "placements": action.get("placements") or candidate.get("placements") or [],
        "discard": action.get("discard", candidate.get("discard")),
        "ev": metric_value(candidate, "ev", "score"),
        "score": metric_value(candidate, "score", "ev"),
        "bust_prob": metric_value(candidate, "bust_rate", "bust_prob", "bust"),
        "fl_rate": metric_value(candidate, "fl_rate", "fl_any"),
        "fl_type_rates": {
            "qq": float(fl_type_rates.get("qq", 0.0)),
            "kk": float(fl_type_rates.get("kk", 0.0)),
            "aa": float(fl_type_rates.get("aa", 0.0)),
            "trips": float(fl_type_rates.get("trips", 0.0)),
        },
        "rank": int(rank),
        "board": candidate.get("board"),
        "metrics": metrics,
    }
    return out


def build_teacher_record(
    row: dict[str, Any],
    exact_candidates: list[dict[str, Any]],
    source_label: str = "t2_miss_t3_exact_hardcase",
) -> dict[str, Any]:
    sorted_candidates = sorted(exact_candidates, key=metric_score, reverse=True)
    return {
        "source": source_label,
        "turn": 3,
        "board": row["t3_payload"]["board"],
        "opponent_board": row["t3_payload"]["opponent_board"],
        "dealt": row["t3_payload"]["dealt"],
        "known_discards": row["t3_payload"].get("known_discards") or [],
        "exclude": row["t3_payload"].get("exclude") or [],
        "is_btn": bool(row["t3_payload"].get("is_btn")),
        "position": row["position"],
        "eval_mode": "exact",
        "best_idx": 0,
        "source_record_index": row["record_index"],
        "source_t2_action_label": row["t2_action_label"],
        "source_t2_action": row["t2_action"],
        "source_t3_draw_index": row["t3_draw_index"],
        "model_best_exact_rank": row["model_best_exact_rank"],
        "model_best_ev_loss": row["ev_loss"],
        "candidates": [
            flatten_exact_candidate(candidate, rank)
            for rank, candidate in enumerate(sorted_candidates, start=1)
        ],
    }


def score_model_t3(
    *,
    model: ActionValueReranker,
    device: torch.device,
    board,
    opponent_board,
    draw: Iterable[str],
    dead: list[str],
    is_btn: bool,
    batch_size: int,
    top_n: int = 5,
) -> dict[str, Any]:
    states, rows = encode_post_t3_candidates(
        board,
        opponent_board,
        draw,
        dead,
        is_btn=is_btn,
        target_dim=int(getattr(model, "input_dim", 520)),
    )
    scores, bust, fl = predict_scores(model, states, device=device, batch_size=batch_size)
    if len(scores) == 0:
        raise RuntimeError("no T3 states scored")
    order = np.argsort(-scores)
    best_i = int(order[0])
    out_rows = []
    for rank, idx in enumerate(order[: min(top_n, len(order))], start=1):
        row = rows[int(idx)]
        out_rows.append(
            {
                "rank": rank,
                "action": row["action"],
                "board": row["board"],
                "model_score": float(scores[int(idx)]),
                "model_bust": float(bust[int(idx)]),
                "model_fl": float(fl[int(idx)]),
            }
        )
    best = out_rows[0]
    return {
        "best": best,
        "top": out_rows,
        "legal_actions": len(rows),
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = load_jsonl(Path(args.source))
    misses = load_jsonl(Path(args.misses))
    if not misses:
        raise ValueError("miss file is empty")
    miss_indices = (
        list(range(len(misses)))
        if str(args.miss_index).lower() == "all"
        else [int(args.miss_index)]
    )

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cache: dict[str, ActionValueReranker] = {}

    def model_for_position(position: str) -> tuple[ActionValueReranker, str]:
        model_path = args.model or (args.model_t3_btn if position == "btn" else args.model_t3_bb)
        model_key = str(model_path)
        model = model_cache.get(model_key)
        if model is None:
            model = ActionValueReranker.from_checkpoint(model_path, map_location=device).to(device)
            model.eval()
            model_cache[model_key] = model
        return model, model_key

    normalizer = CardNormalizer()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    teacher_output = Path(args.teacher_output) if args.teacher_output else None
    if teacher_output is not None:
        teacher_output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    selected_total = 0
    positions_seen: set[str] = set()
    models_used: set[str] = set()
    record_indexes: list[int] = []
    for miss_index in miss_indices:
        if miss_index < 0 or miss_index >= len(misses):
            raise ValueError(f"miss_index out of range: {miss_index}")
        miss = misses[miss_index]
        record_index = int(miss.get("record_index", -1))
        if record_index < 0 or record_index >= len(source_rows):
            raise ValueError(f"record_index out of range: {record_index}")
        record_indexes.append(record_index)
        source = source_rows[record_index]
        position = str(source.get("position") or ("btn" if source.get("is_btn") else "bb")).lower()
        positions_seen.add(position)
        model, model_path = model_for_position(position)
        models_used.add(model_path)

        base_board = normalize_board(source.get("board") or {}, normalizer)
        opponent_board = normalize_board(source.get("opponent_board") or source.get("board_opponent") or {}, normalizer)
        selected = select_t2_actions(miss, source, args.max_t2_actions)
        selected_total += len(selected)
        for action_i, item in enumerate(selected):
            action = make_action(item["action"], normalizer)
            board_after_t2 = apply_action(base_board, action)
            dead = normalized_dead_cards(
                source,
                board_after_t2,
                opponent_board,
                [action.discard] if action.discard else [],
                normalizer,
            )
            deck = remaining_deck(board_after_t2, opponent_board, dead)
            seed = args.seed + record_index * 1009 + miss_index * 131 + action_i * 37
            draws = choose_draws(deck, args.draw_limit, seed)
            for draw_i, draw in enumerate(draws):
                t3_payload = {
                    "turn": 3,
                    "board": board_to_dict(board_after_t2),
                    "opponent_board": board_to_dict(opponent_board),
                    "dealt": list(draw),
                    "known_discards": dead,
                    "exclude": dead,
                    "is_btn": position == "btn",
                    "position": position,
                }
                model_eval = score_model_t3(
                    model=model,
                    device=device,
                    board=board_after_t2,
                    opponent_board=opponent_board,
                    draw=draw,
                    dead=dead,
                    is_btn=position == "btn",
                    batch_size=args.batch_size,
                )
                exact = evaluate_late_position(
                    board_after_t2,
                    list(draw),
                    3,
                    opponent_board=opponent_board,
                    exclude=dead,
                    top_n=args.exact_top_n,
                    prefer_rust=True,
                    rust_solver_path=args.rust_solver or None,
                    rust_timeout_s=args.rust_timeout_s,
                )
                exact_candidates = list(exact.get("candidates") or [])
                exact_by_key = {
                    compact_action_key(c.get("action") or {}): (rank, c)
                    for rank, c in enumerate(exact_candidates, start=1)
                }
                exact_best = exact.get("best") or (exact_candidates[0] if exact_candidates else {})
                model_best_action = model_eval["best"]["action"]
                model_rank, model_exact = exact_by_key.get(compact_action_key(model_best_action), (None, None))
                exact_best_score = metric_score(exact_best)
                model_exact_score = metric_score(model_exact)
                ev_loss = (
                    max(0.0, exact_best_score - model_exact_score)
                    if model_exact is not None
                    else None
                )
                row = {
                    "miss_index": miss_index,
                    "record_index": record_index,
                    "position": position,
                    "t2_action_label": item["label"],
                    "t2_action": item["action"],
                    "t2_source_score": item.get("score"),
                    "t3_draw_index": draw_i,
                    "t3_payload": t3_payload,
                    "model_best": model_eval["best"],
                    "model_top": model_eval["top"],
                    "exact_best": exact_best,
                    "model_best_exact_rank": model_rank,
                    "model_best_exact_score": model_exact_score if model_exact is not None else None,
                    "exact_best_score": exact_best_score,
                    "ev_loss": ev_loss,
                    "exact_elapsed_ms": exact.get("elapsed_ms"),
                    "legal_actions": exact.get("legal_actions"),
                    "exact_candidate_count": len(exact_candidates),
                    "exact_candidates": [
                        flatten_exact_candidate(candidate, rank)
                        for rank, candidate in enumerate(
                            sorted(exact_candidates, key=metric_score, reverse=True),
                            start=1,
                        )
                    ],
                    }
                rows.append(row)
                if teacher_output is not None:
                    teacher_rows.append(build_teacher_record(row, exact_candidates))

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    if teacher_output is not None:
        with teacher_output.open("w", encoding="utf-8") as handle:
            for row in teacher_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    losses = [float(row["ev_loss"]) for row in rows if row.get("ev_loss") is not None]
    summary = {
        "source": str(args.source),
        "misses": str(args.misses),
        "miss_index": str(args.miss_index),
        "miss_indices": miss_indices,
        "record_indexes": record_indexes,
        "positions": sorted(positions_seen),
        "models": sorted(models_used),
        "output": str(output),
        "teacher_output": str(teacher_output) if teacher_output is not None else None,
        "rows": len(rows),
        "teacher_rows": len(teacher_rows),
        "t2_actions": selected_total,
        "draw_limit": int(args.draw_limit),
        "ev_loss_mean": float(np.mean(losses)) if losses else 0.0,
        "ev_loss_p95": float(np.percentile(losses, 95)) if losses else 0.0,
        "ev_loss_max": float(np.max(losses)) if losses else 0.0,
        "ev_loss_gt_0p1": int(sum(loss > 0.1 for loss in losses)),
        "ev_loss_gt_1": int(sum(loss > 1.0 for loss in losses)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose T3 model errors from a T2 exact miss")
    parser.add_argument("--source", required=True, help="Original T2 source JSONL")
    parser.add_argument("--misses", required=True, help="Miss JSONL from run_t2_exact_oracle.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-output", default="", help="Optional exact T3 teacher JSONL output")
    parser.add_argument("--miss-index", default="0", help="Miss index or 'all'")
    parser.add_argument("--draw-limit", type=int, default=5)
    parser.add_argument("--max-t2-actions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="", help="Override single T3 model")
    parser.add_argument("--model-t3-bb", default=DEFAULT_T3_BB_MODEL)
    parser.add_argument("--model-t3-btn", default=DEFAULT_T3_BTN_MODEL)
    parser.add_argument("--rust-solver", default="ai/rust_solver/target/release/t3_exact_solver.exe")
    parser.add_argument("--rust-timeout-s", type=float, default=5.0)
    parser.add_argument("--exact-top-n", type=int, default=100)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(diagnose(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
