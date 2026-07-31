"""Build split T3 exact teacher data from generated T2 rows.

This expands practical T2 branches into concrete T3 deals and evaluates every
legal T3 placement with the Rust exact solver.  Source rows are split before
expansion so train/dev/holdout do not share the same T2 board.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker
from ai.tutor.benchmark_t2_t3_value_model import (
    choose_draws,
    make_action,
    normalized_dead_cards,
    remaining_deck,
)
from ai.tutor.diagnose_t3_model_from_t2_miss import (
    build_teacher_record,
    compact_action_key,
    metric_score,
    score_model_t3,
)
from ai.tutor.exact_late import CardNormalizer, apply_action, board_to_dict, evaluate_late_position, normalize_board
from ai.tutor.run_t2_exact_oracle import candidate_score


DEFAULT_T3_BB_MODEL = (
    "D:/ofc-pineapple-data/t3_jokerfix_extra_20260605/models/"
    "t3-20k-plus-gen5k-hardneginit-bb/action_value_best.pt"
)
DEFAULT_T3_BTN_MODEL = (
    "D:/ofc-pineapple-data/t3_jokerfix_extra_20260605/models/"
    "t3-20k-plus-gen5k-hardneginit-btn/action_value_best.pt"
)
SPLIT_NAMES = ("mine_train", "dev", "final_holdout")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_positions(value: str) -> set[str]:
    raw = value.lower().strip()
    if raw == "both":
        return {"bb", "btn"}
    if raw not in {"bb", "btn"}:
        raise argparse.ArgumentTypeError("--position must be bb, btn, or both")
    return {raw}


def parse_split_fracs(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--split-fracs must have three comma-separated values")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("--split-fracs must sum to a positive value")
    return tuple(part / total for part in parts)  # type: ignore[return-value]


def position_of(row: dict[str, Any]) -> str:
    return str(row.get("position") or ("btn" if row.get("is_btn") else "bb")).lower()


def load_source_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    positions = parse_positions(args.position)
    rows = [
        row
        for row in iter_jsonl(Path(args.source))
        if int(row.get("turn", -1)) == 2
        and position_of(row) in positions
        and row.get("candidates")
    ]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_source_rows > 0:
        rows = rows[: args.max_source_rows]
    return rows


def split_rows(rows: list[dict[str, Any]], fracs: tuple[float, float, float]) -> dict[str, list[dict[str, Any]]]:
    n = len(rows)
    n_train = int(round(n * fracs[0]))
    n_dev = int(round(n * fracs[1]))
    if n_train + n_dev > n:
        n_dev = max(0, n - n_train)
    n_holdout = n - n_train - n_dev
    if n > 0 and n_holdout == 0:
        n_holdout = 1
        if n_dev > 0:
            n_dev -= 1
        else:
            n_train = max(0, n_train - 1)
    return {
        "mine_train": rows[:n_train],
        "dev": rows[n_train : n_train + n_dev],
        "final_holdout": rows[n_train + n_dev :],
    }


def select_t2_candidates(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = list(row.get("candidates") or [])
    if args.t2_actions_per_row <= 0 or len(candidates) <= args.t2_actions_per_row:
        selected = candidates
    elif args.selection == "top":
        selected = candidates[: args.t2_actions_per_row]
    else:
        top_n = max(1, min(args.top_always, args.t2_actions_per_row, len(candidates)))
        selected = candidates[:top_n]
        remaining = args.t2_actions_per_row - len(selected)
        if remaining > 0:
            start = top_n
            if start < len(candidates):
                spread = np.linspace(start, len(candidates) - 1, num=remaining, dtype=np.int64).tolist()
                for idx in spread:
                    selected.append(candidates[int(idx)])

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, candidate in enumerate(selected):
        action = candidate.get("action") or candidate
        key = compact_action_key(action)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "label": f"source_candidate_{idx + 1}",
                "action": action,
                "score": candidate_score(candidate),
                "source_candidate": candidate,
            }
        )
    return out


def load_model(
    *,
    position: str,
    args: argparse.Namespace,
    device: torch.device,
    cache: dict[str, ActionValueReranker],
) -> tuple[ActionValueReranker | None, str | None]:
    if args.no_model_diagnostics:
        return None, None
    model_path = args.model or (args.model_t3_btn if position == "btn" else args.model_t3_bb)
    model = cache.get(model_path)
    if model is None:
        model = ActionValueReranker.from_checkpoint(model_path, map_location=device).to(device)
        model.eval()
        cache[model_path] = model
    return model, model_path


def loss_stats(losses: list[float]) -> dict[str, Any]:
    if not losses:
        return {
            "mean": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "gt_0p1": 0,
            "gt_0p5": 0,
            "gt_1": 0,
        }
    return {
        "mean": float(np.mean(losses)),
        "p95": float(np.percentile(losses, 95)),
        "p99": float(np.percentile(losses, 99)),
        "max": float(np.max(losses)),
        "gt_0p1": int(sum(loss > 0.1 for loss in losses)),
        "gt_0p5": int(sum(loss > 0.5 for loss in losses)),
        "gt_1": int(sum(loss > 1.0 for loss in losses)),
    }


def write_outputs(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_source_rows(args)
    splits = split_rows(rows, parse_split_fracs(args.split_fracs))

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cache: dict[str, ActionValueReranker] = {}
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "source": str(args.source),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "source_rows": len(rows),
        "split_source_rows": {name: len(items) for name, items in splits.items()},
        "t2_actions_per_row": int(args.t2_actions_per_row),
        "draw_limit": int(args.draw_limit),
        "selection": args.selection,
        "splits": {},
        "models": set(),
    }

    all_losses: list[float] = []
    for split_name in SPLIT_NAMES:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        teacher_path = split_dir / "t3_exact_teacher.jsonl"
        diag_path = split_dir / "t3_diagnostics.jsonl"
        split_losses: list[float] = []
        records = 0
        candidate_samples = 0
        positions = defaultdict(int)
        legal_counts: list[int] = []
        with teacher_path.open("w", encoding="utf-8") as teacher_handle, diag_path.open("w", encoding="utf-8") as diag_handle:
            for row_index, source in enumerate(splits[split_name]):
                position = position_of(source)
                positions[position] += 1
                normalizer = CardNormalizer()
                base_board = normalize_board(source.get("board") or {}, normalizer)
                opponent_board = normalize_board(source.get("opponent_board") or source.get("board_opponent") or {}, normalizer)
                model, model_path = load_model(position=position, args=args, device=device, cache=model_cache)
                if model_path:
                    summary["models"].add(model_path)
                for action_i, item in enumerate(select_t2_candidates(source, args)):
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
                    seed = args.seed + int(source.get("source_line", row_index)) * 1009 + action_i * 37
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
                        exact_best = exact.get("best") or (exact_candidates[0] if exact_candidates else {})
                        exact_by_key = {
                            compact_action_key(candidate.get("action") or {}): (rank, candidate)
                            for rank, candidate in enumerate(exact_candidates, start=1)
                        }
                        model_eval = None
                        model_rank = None
                        model_exact_score = None
                        ev_loss = None
                        if model is not None:
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
                            model_rank, model_exact = exact_by_key.get(
                                compact_action_key(model_eval["best"]["action"]),
                                (None, None),
                            )
                            if model_exact is not None:
                                model_exact_score = metric_score(model_exact)
                                ev_loss = max(0.0, metric_score(exact_best) - model_exact_score)
                                split_losses.append(float(ev_loss))
                                all_losses.append(float(ev_loss))

                        diag_row = {
                            "split": split_name,
                            "source_row_index": row_index,
                            "source_line": source.get("source_line"),
                            "position": position,
                            "t2_action_label": item["label"],
                            "t2_action": item["action"],
                            "t2_source_score": item.get("score"),
                            "t3_draw_index": draw_i,
                            "t3_payload": t3_payload,
                            "model_best": (model_eval or {}).get("best"),
                            "model_top": (model_eval or {}).get("top"),
                            "exact_best": exact_best,
                            "model_best_exact_rank": model_rank,
                            "model_best_exact_score": model_exact_score,
                            "exact_best_score": metric_score(exact_best),
                            "ev_loss": ev_loss,
                            "exact_elapsed_ms": exact.get("elapsed_ms"),
                            "legal_actions": exact.get("legal_actions"),
                            "exact_candidate_count": len(exact_candidates),
                        }
                        teacher = build_teacher_record(
                            {
                                "record_index": source.get("source_line", row_index),
                                "position": position,
                                "t2_action_label": item["label"],
                                "t2_action": item["action"],
                                "t3_draw_index": draw_i,
                                "t3_payload": t3_payload,
                                "model_best_exact_rank": model_rank,
                                "ev_loss": ev_loss,
                            },
                            exact_candidates,
                            source_label="t2_branch_t3_exact",
                        )
                        teacher["split"] = split_name
                        teacher["source_line"] = source.get("source_line")
                        teacher["source_candidate_score"] = item.get("score")
                        teacher_handle.write(json.dumps(teacher, ensure_ascii=False, separators=(",", ":")) + "\n")
                        diag_handle.write(json.dumps(diag_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        records += 1
                        candidate_samples += len(exact_candidates)
                        if exact.get("legal_actions") is not None:
                            legal_counts.append(int(exact["legal_actions"]))

        split_summary = {
            "teacher": str(teacher_path),
            "diagnostics": str(diag_path),
            "source_rows": len(splits[split_name]),
            "t3_records": records,
            "candidate_samples": candidate_samples,
            "positions": dict(positions),
            "legal_actions_avg": float(np.mean(legal_counts)) if legal_counts else 0.0,
            "ev_loss": loss_stats(split_losses),
        }
        summary["splits"][split_name] = split_summary
        split_dir.joinpath("summary.json").write_text(
            json.dumps(split_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    summary["models"] = sorted(summary["models"])
    summary["overall_ev_loss"] = loss_stats(all_losses)
    summary["elapsed_seconds"] = time.perf_counter() - started
    output_dir.joinpath("summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build split T3 exact teacher data from T2 rows")
    parser.add_argument("--source", required=True, help="T2 source JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-source-rows", type=int, default=60)
    parser.add_argument("--position", default="both", help="bb, btn, or both")
    parser.add_argument("--split-fracs", default="0.7,0.15,0.15")
    parser.add_argument("--t2-actions-per-row", type=int, default=2)
    parser.add_argument("--selection", choices=["top", "top_spread"], default="top_spread")
    parser.add_argument("--top-always", type=int, default=1)
    parser.add_argument("--draw-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--model", default="", help="Override T3 diagnostic model for both positions")
    parser.add_argument("--model-t3-bb", default=DEFAULT_T3_BB_MODEL)
    parser.add_argument("--model-t3-btn", default=DEFAULT_T3_BTN_MODEL)
    parser.add_argument("--no-model-diagnostics", action="store_true")
    parser.add_argument("--rust-solver", default="ai/rust_solver/target/release/t3_exact_solver.exe")
    parser.add_argument("--rust-timeout-s", type=float, default=10.0)
    parser.add_argument("--exact-top-n", type=int, default=100)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(write_outputs(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
