"""Generate teacher labels for only the candidates kept by self-play pruning.

The input is produced by ai.tutor.extract_selfplay_targets and must contain
top_k_indices from a pruned self-play run.  Output follows the candidate record
schema consumed by ai.training.convert_action_value_teacher, but candidates are
restricted to the saved top-k set instead of every legal action.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import encode_action, get_initial_actions, get_turn_actions
from ai.engine.encoding import Board, Observation
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.models.networks import PolicyNetwork


FL_TYPE_KEYS = {14: "qq", 15: "kk", 16: "aa", 17: "trips"}

_EVALUATOR: RolloutEvaluator | None = None
_SIMS = 0
_SEED = 0


def _row(board: dict[str, Any], *names: str) -> list[str]:
    out: list[str] = []
    for name in names:
        out.extend(board.get(name, []) or [])
    return out


def _board_from_target(board: dict[str, Any]) -> Board:
    return Board(
        top=list(board.get("top", [])),
        middle=_row(board, "middle", "mid"),
        bottom=_row(board, "bottom", "bot"),
    )


def _load_policy(model_path: str) -> PolicyNetwork:
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    input_dim = state_dict.get("net.0.weight", torch.empty(0, 522)).shape[1]
    policy = PolicyNetwork(input_dim=input_dim)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def init_worker(args_dict: dict[str, Any]) -> None:
    global _EVALUATOR, _SIMS, _SEED
    _SIMS = int(args_dict["sims"])
    _SEED = int(args_dict["seed"])
    policy = _load_policy(args_dict["model"])
    _EVALUATOR = RolloutEvaluator(
        policy_net=policy,
        n_rollouts=max(1, _SIMS),
        top_k=int(args_dict["top_k"]),
        device="cpu",
        use_policy_playout=bool(args_dict["policy_playout"]),
        full_width=True,
    )


def _observation(target: dict[str, Any]) -> Observation:
    return Observation(
        board_self=_board_from_target(target.get("board") or {}),
        board_opponent=_board_from_target(target.get("opponent_board") or {}),
        dealt_cards=list(target.get("dealt") or []),
        known_discards_self=list(target.get("known_discards") or []),
        turn=int(target.get("turn", 0)),
        is_btn=bool(target.get("is_btn", False)),
        is_fl=False,
        opp_is_fl=False,
        chips_self=200,
        chips_opponent=200,
    )


def _valid_actions(obs: Observation):
    if int(obs.turn) == 0:
        return get_initial_actions(obs.dealt_cards, obs.board_self)
    return get_turn_actions(obs.dealt_cards, obs.board_self)


def _candidate_ids(target: dict[str, Any], valid_actions, args_dict: dict[str, Any]) -> list[int]:
    turn = int(target.get("turn", 0))
    dealt = list(target.get("dealt") or [])
    index_by_action = [
        int(encode_action(action, valid_actions, turn=turn, dealt_cards=dealt))
        for action in valid_actions
    ]
    available = set(index_by_action)
    if args_dict.get("all_actions"):
        return index_by_action
    selected = []
    seen = set()
    for raw_idx in target.get("top_k_indices") or []:
        idx = int(raw_idx)
        if idx in available and idx not in seen:
            selected.append(idx)
            seen.add(idx)
    return selected


def _action_lookup(obs: Observation, valid_actions) -> dict[int, Any]:
    return {
        int(encode_action(action, valid_actions, turn=int(obs.turn), dealt_cards=obs.dealt_cards)): action
        for action in valid_actions
    }


def _candidate_record(action, metrics: dict[str, Any], sims: int) -> dict[str, Any]:
    fl_counts = metrics.get("fl_type_dist", {}) or {}
    fl_rates = {
        name: float(fl_counts.get(card_count, 0)) / max(1, sims)
        for card_count, name in FL_TYPE_KEYS.items()
    }
    return {
        "placements": [[card, pos] for card, pos in action.placements],
        "discard": action.discard,
        "mc": {
            "avg_score": float(metrics.get("avg_score", 0.0)),
            "std_score": float(metrics.get("std_score", 0.0)),
            "bust_rate": float(metrics.get("bust_prob", 0.0)),
            "fl_rate": float(metrics.get("fl_prob", 0.0)),
            "fl_type_rates": fl_rates,
            "avg_royalty": float(metrics.get("avg_royalty", 0.0)),
            "opp_bust_rate": float(metrics.get("opp_bust_prob", 0.0)),
            "n_rollouts": int(metrics.get("n_rollouts", sims)),
        },
    }


def evaluate_target(payload: tuple[int, dict[str, Any], dict[str, Any]]) -> dict[str, Any] | None:
    index, target, args_dict = payload
    if _EVALUATOR is None:
        raise RuntimeError("worker evaluator is not initialized")
    obs = _observation(target)
    valid_actions = _valid_actions(obs)
    if not valid_actions:
        return None

    candidate_ids = _candidate_ids(target, valid_actions, args_dict)
    if not candidate_ids:
        return None
    lookup = _action_lookup(obs, valid_actions)

    random.seed(_SEED + int(target.get("source_line", index)) * 1009 + int(target.get("turn", 0)) * 37)
    candidates = []
    for action_idx in candidate_ids:
        action = lookup.get(action_idx)
        if action is None:
            continue
        metrics = _EVALUATOR._evaluate_action_detailed(obs, action, n_rollouts=max(1, _SIMS))
        candidate = _candidate_record(action, metrics, max(1, _SIMS))
        candidate["action_idx"] = int(action_idx)
        candidates.append(candidate)

    if not candidates:
        return None
    candidates.sort(key=lambda cand: float(cand.get("mc", {}).get("avg_score", 0.0)), reverse=True)

    return {
        "source": target.get("source"),
        "source_line": target.get("source_line"),
        "active_reasons": list(target.get("reasons", [])),
        "turn": int(target.get("turn", 0)),
        "board": {
            "top": list((target.get("board") or {}).get("top", [])),
            "mid": _row(target.get("board") or {}, "middle", "mid"),
            "bot": _row(target.get("board") or {}, "bottom", "bot"),
        },
        "opponent_board": {
            "top": list((target.get("opponent_board") or {}).get("top", [])),
            "mid": _row(target.get("opponent_board") or {}, "middle", "mid"),
            "bot": _row(target.get("opponent_board") or {}, "bottom", "bot"),
        },
        "dealt": list(target.get("dealt") or []),
        "known_discards": list(target.get("known_discards") or []),
        "exclude": list(target.get("exclude") or []),
        "is_btn": bool(target.get("is_btn", False)),
        "position": "btn" if bool(target.get("is_btn", False)) else "bb",
        "n_candidates": len(candidates),
        "candidates": candidates,
        "best_idx": 0,
        "eval_mode": f"mc{max(1, _SIMS)}",
        "elapsed_s": 0.0,
        "candidate_source": "selfplay_top_k_indices",
        "pruned_top_k": int(target.get("pruned_top_k") or 0),
        "original_n_actions": int(target.get("n_actions") or 0),
        "original_evaluated_actions": int(target.get("evaluated_actions") or 0),
    }


def load_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    turns = {int(part) for part in args.turns.split(",") if part.strip()}
    targets = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            target = json.loads(line)
            if int(target.get("turn", -1)) not in turns:
                continue
            targets.append(target)
    if args.max_records > 0:
        targets = targets[: args.max_records]
    return targets


def generate(args: argparse.Namespace) -> dict[str, Any]:
    targets = load_targets(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    args_dict = {
        "model": args.model,
        "sims": int(args.sims),
        "top_k": int(args.top_k),
        "policy_playout": bool(args.policy_playout),
        "all_actions": bool(args.all_actions),
        "seed": int(args.seed),
    }

    stats = {
        "input": args.input,
        "output": str(output),
        "targets": len(targets),
        "written": 0,
        "skipped": 0,
        "sims": int(args.sims),
        "turns": {},
        "elapsed_s": 0.0,
        "all_actions": bool(args.all_actions),
    }
    start = time.time()
    work = [(i, target, args_dict) for i, target in enumerate(targets, start=1)]
    with output.open("w", encoding="utf-8") as dst:
        if args.workers <= 1:
            init_worker(args_dict)
            iterator = (evaluate_target(item) for item in work)
            for i, record in enumerate(iterator, start=1):
                if record is None:
                    stats["skipped"] += 1
                    continue
                record["elapsed_s"] = round(time.time() - start, 3)
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["written"] += 1
                key = str(record["turn"])
                stats["turns"][key] = stats["turns"].get(key, 0) + 1
                if i % args.progress_every == 0:
                    dst.flush()
                    print(f"  {i:,}/{len(work):,} written={stats['written']:,}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args_dict,)) as pool:
                futures = [pool.submit(evaluate_target, item) for item in work]
                for i, future in enumerate(as_completed(futures), start=1):
                    try:
                        record = future.result()
                    except Exception as exc:
                        stats["skipped"] += 1
                        if args.print_errors:
                            print(f"  error: {exc}", flush=True)
                        continue
                    if record is None:
                        stats["skipped"] += 1
                        continue
                    dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                    key = str(record["turn"])
                    stats["turns"][key] = stats["turns"].get(key, 0) + 1
                    if i % args.progress_every == 0:
                        dst.flush()
                        print(f"  {i:,}/{len(work):,} written={stats['written']:,}", flush=True)

    stats["elapsed_s"] = round(time.time() - start, 3)
    output.with_suffix(".summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate top-k-pruned rollout teacher labels")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="ai/models/selfplay_iter18/bc_policy_best.pt")
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--turns", default="0,1,2,3,4")
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--policy-playout", action="store_true")
    parser.add_argument("--all-actions", action="store_true", help="Ignore top_k_indices and evaluate every legal action")
    parser.add_argument("--print-errors", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(generate(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
