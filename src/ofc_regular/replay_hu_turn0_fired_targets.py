"""Replay fired HU T0 baseline/candidate pairs with independent common futures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .action_space import generate_actions
from .ai_profiles import ModelPaths, build_policy, load_model_bundle
from .hu_turn0_teacher_pilot import (
    build_turn0_teacher_sample,
    remaining_for_turn0_state,
)
from .state import Board


BuildSample = Callable[..., dict[str, Any]]


def _board(payload: dict[str, Any]) -> Board:
    return Board.from_rows(
        payload.get("top", ()), payload.get("middle", ()), payload.get("bottom", ())
    )


def _action_signature(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple((str(card), str(row)) for card, row in payload.get("placements", ())),
        tuple(str(card) for card in payload.get("discards", ())),
    )


def _generated_action_signature(action: Any) -> tuple[Any, ...]:
    return tuple(action.placements), tuple(action.discards)


def _future_seed(replay_seed: int, target_id: str) -> int:
    payload = f"{replay_seed}|{target_id}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & (
        (1 << 63) - 1
    )


def replay_target(
    target: dict[str, Any],
    *,
    policies: Sequence[Any],
    future_samples: int,
    replay_seed: int,
    build_sample_fn: BuildSample = build_turn0_teacher_sample,
) -> dict[str, Any]:
    board = _board(target["hero_board"])
    opponent_board = _board(target["opponent_board"])
    dealt = tuple(target["cards_to_place"])
    actions = generate_actions(board, dealt)
    baseline_index = int(target["baseline_action_index"])
    candidate_index = int(target["candidate_action_index"])
    if not 0 <= baseline_index < len(actions) or not 0 <= candidate_index < len(actions):
        raise ValueError(f"{target['target_id']} action index is out of range")
    if _generated_action_signature(actions[baseline_index]) != _action_signature(
        target["baseline_action"]
    ):
        raise ValueError(f"{target['target_id']} baseline action mapping changed")
    if _generated_action_signature(actions[candidate_index]) != _action_signature(
        target["candidate_action"]
    ):
        raise ValueError(f"{target['target_id']} candidate action mapping changed")

    hero_player = 0 if target["seat"] == "first" else 1
    remaining = remaining_for_turn0_state(board, dealt, opponent_board)
    started = time.perf_counter()
    sample = build_sample_fn(
        board=board,
        opponent_board=opponent_board,
        dealt=dealt,
        remaining_cards=remaining,
        hero_player=hero_player,
        policies=policies,
        hand_seed=int(target["hand_seed"]),
        sample_id=int(target["target_id"][:15], 16),
        future_samples=future_samples,
        future_seed=_future_seed(replay_seed, target["target_id"]),
        profile_name="stage18_p1",
        opponent_profile="stage18_p1",
        source_bucket="runtime_fired_pair_replay",
        action_indices=(baseline_index, candidate_index),
    )
    by_index = {int(action["original_index"]): action for action in sample["actions"]}
    if set(by_index) != {baseline_index, candidate_index}:
        raise ValueError(f"{target['target_id']} replay returned the wrong action set")
    if int(sample["baseline_action_index"]) != baseline_index:
        raise ValueError(f"{target['target_id']} baseline policy action changed")
    candidate = by_index[candidate_index]
    delta = float(candidate["delta_vs_baseline"])
    delta_se = float(candidate["delta_se_vs_baseline"])
    if not math.isfinite(delta) or not math.isfinite(delta_se):
        raise ValueError(f"{target['target_id']} produced non-finite replay values")
    lcb164 = delta - 1.64 * delta_se
    lcb196 = delta - 1.96 * delta_se
    ucb196 = delta + 1.96 * delta_se
    if lcb196 > 0.0:
        safe_label = "positive"
        safe_label_id = 1
    elif ucb196 < 0.0 or delta <= -0.25:
        safe_label = "negative"
        safe_label_id = 0
    else:
        safe_label = "gray"
        safe_label_id = -1
    return {
        **target,
        "schema": "hu_turn0_fired_pair_replay_v1",
        "future_samples": int(future_samples),
        "replay_seed": int(replay_seed),
        "future_seed": int(sample["future_seed"]),
        "common_random_future_digest": sample["common_random_future_digest"],
        "common_random_futures_verified": bool(
            sample["common_random_futures_verified"]
        ),
        "action_mapping_verified": True,
        "baseline_ev": float(by_index[baseline_index]["ev"]),
        "candidate_ev": float(candidate["ev"]),
        "candidate_delta_vs_baseline": delta,
        "candidate_delta_se_vs_baseline": delta_se,
        "candidate_delta_z_vs_baseline": float(candidate["delta_z_vs_baseline"]),
        "candidate_delta_lcb164": lcb164,
        "candidate_delta_lcb196": lcb196,
        "candidate_delta_ucb196": ucb196,
        "safe_override_label": safe_label,
        "safe_override_label_id": safe_label_id,
        "replay_seconds": time.perf_counter() - started,
        "replay_actions": sample["actions"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_policies(seed: int) -> list[Any]:
    bundle = load_model_bundle(ModelPaths(), {"stage18_p1"})
    return [
        build_policy(
            "stage18_p1", bundle, seed=seed * 2, seat="first", opening_lookahead_samples=1
        ),
        build_policy(
            "stage18_p1",
            bundle,
            seed=seed * 2 + 1,
            seat="second",
            opening_lookahead_samples=1,
        ),
    ]


def replay_targets(
    targets: Sequence[dict[str, Any]],
    *,
    future_samples: int,
    replay_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if future_samples <= 0:
        raise ValueError("future_samples must be positive")
    policies = build_policies(replay_seed)
    started = time.perf_counter()
    rows = [
        replay_target(
            target,
            policies=policies,
            future_samples=future_samples,
            replay_seed=replay_seed,
        )
        for target in targets
    ]
    elapsed = time.perf_counter() - started
    summary = {
        "schema": "hu_turn0_fired_pair_replay_summary_v1",
        "targets": len(targets),
        "written": len(rows),
        "missing": len(targets) - len(rows),
        "future_samples": int(future_samples),
        "replay_seed": int(replay_seed),
        "common_random_futures_verified": all(
            row["common_random_futures_verified"] for row in rows
        ),
        "action_mapping_verified": all(row["action_mapping_verified"] for row in rows),
        "label_counts": dict(
            sorted(Counter(row["safe_override_label"] for row in rows).items())
        ),
        "seat_counts": dict(sorted(Counter(row["seat"] for row in rows).items())),
        "elapsed_seconds": elapsed,
        "seconds_per_target": elapsed / len(rows) if rows else 0.0,
    }
    return rows, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("inputs/targets.jsonl"))
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--skip-records", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--future-samples", type=int, default=128)
    parser.add_argument("--replay-seed", type=int)
    parser.add_argument("--seed", type=int)
    # Compatibility with the generic GCP pilot sharder. Replay always uses
    # the Stage18 chain and the exact action pair saved in each target.
    parser.add_argument("--max-actions", type=int, default=0)
    parser.add_argument("--profile")
    parser.add_argument("--opponent-profile")
    parser.add_argument("--source-bucket")
    parser.add_argument("--opening-lookahead-samples", type=int, default=1)
    parser.add_argument("--seats", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    targets = _read_jsonl(args.input)
    start_index = (
        args.start_index
        if args.start_index is not None
        else (args.skip_records if args.skip_records is not None else 0)
    )
    count = args.count if args.count is not None else (args.samples or 0)
    replay_seed = (
        args.replay_seed
        if args.replay_seed is not None
        else (args.seed if args.seed is not None else 2026607001)
    )
    if start_index < 0 or start_index > len(targets):
        raise SystemExit("--start-index is outside the target range")
    stop = len(targets) if count <= 0 else min(len(targets), start_index + count)
    selected = targets[start_index:stop]
    if not selected:
        raise SystemExit("no replay targets selected")
    rows, summary = replay_targets(
        selected,
        future_samples=args.future_samples,
        replay_seed=replay_seed,
    )
    summary.update(
        {
            "input": str(args.input),
            "start_index": start_index,
            "count": len(selected),
            "seconds_per_sample": summary["seconds_per_target"],
        }
    )
    _write_jsonl(args.output, rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
