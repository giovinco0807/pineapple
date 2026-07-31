"""Evaluate a fixed-slot T3 policy-value model on tutor teacher JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state
from ai.training.action_feature_encoding import adapt_np_state
from ai.training.convert_mc_teacher import action_to_index_t1plus
from ai.training.train_t3_oracle_v2 import T3PolicyValueNet
from ai.tutor.evaluate_teacher_model import (
    MetricBucket,
    action_label,
    iter_decisions,
    iter_jsonl,
    priority_score,
    summarize_decision,
    true_metric,
    update_bucket,
    write_markdown,
)


def board_from_dict(board: dict[str, Any]) -> Board:
    return Board(
        top=list(board.get("top", []) or []),
        middle=list(board.get("middle", []) or []) + list(board.get("mid", []) or []),
        bottom=list(board.get("bottom", []) or []) + list(board.get("bot", []) or []),
    )


def load_policy_model(path: Path, device: torch.device) -> T3PolicyValueNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt if isinstance(ckpt, dict) else {}
    state_dict = config.get("model_state_dict", ckpt)
    model = T3PolicyValueNet(
        state_dim=int(config.get("state_dim", 522)),
        n_actions=int(config.get("n_actions", 27)),
        hidden=int(config.get("hidden", 1024)),
        n_blocks=int(config.get("n_blocks", 4)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def pre_action_observation(decision: dict[str, Any]) -> Observation:
    return Observation(
        board_self=board_from_dict(decision.get("board") or {}),
        board_opponent=board_from_dict(decision.get("opponent_board") or {}),
        dealt_cards=list(decision.get("dealt", []) or []),
        known_discards_self=list(decision.get("known_discards", []) or []),
        turn=int(decision.get("turn", 3)),
        is_btn=bool(decision.get("is_btn", True)),
        is_fl=False,
        opp_is_fl=False,
        chips_self=200,
        chips_opponent=200,
    )


def score_policy_candidates(
    model: T3PolicyValueNet,
    decision: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], int]:
    if int(decision.get("turn", -1)) != 3:
        return [], len(decision.get("candidates") or [])
    candidates = decision.get("candidates") or []
    if not candidates:
        return [], 0

    obs = pre_action_observation(decision)
    state = adapt_np_state(np.asarray(encode_state(obs), dtype=np.float32), model.input_proj[0].in_features)
    with torch.no_grad():
        tensor = torch.from_numpy(state[None, :]).to(device)
        logits, _value = model(tensor)
        scores = logits.detach().cpu().numpy()[0]

    rows = []
    skipped = 0
    for idx, candidate in enumerate(candidates):
        try:
            action_idx = int(action_to_index_t1plus(candidate, list(decision.get("dealt", []) or [])))
            rows.append(
                {
                    "candidate_index": idx,
                    "candidate": candidate,
                    "true": true_metric(candidate),
                    "pred": {
                        "score": float(scores[action_idx]),
                        "bust_rate": 0.0,
                        "fl_rate": 0.0,
                        "fl_type_rates": {"qq": 0.0, "kk": 0.0, "aa": 0.0, "trips": 0.0},
                    },
                    "policy_action_idx": action_idx,
                }
            )
        except Exception:
            skipped += 1
    return rows, skipped


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_turns = {int(v) for v in args.turns.split(",") if v.strip()}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = Path(args.model)
    model = load_policy_model(model_path, device)

    overall = MetricBucket()
    by_turn: dict[int, MetricBucket] = {}
    weak_spots: list[dict[str, Any]] = []
    decision_count = 0
    record_count = 0

    for record_index, record in enumerate(iter_jsonl(input_path), start=1):
        record_count += 1
        for decision in iter_decisions(record, include_turns):
            scored, skipped = score_policy_candidates(model, decision, device)
            summary = summarize_decision(record_index, decision_count + 1, decision, scored, skipped)
            if summary is None:
                overall.skipped_candidates += skipped
                continue
            decision_count += 1
            update_bucket(overall, summary)
            turn = int(summary["turn"])
            by_turn.setdefault(turn, MetricBucket())
            update_bucket(by_turn[turn], summary)
            weak_spots.append(summary)
        if args.limit_records and record_count >= args.limit_records:
            break

    weak_spots.sort(key=priority_score, reverse=True)
    weak_path = output_dir / "weak_spots.jsonl"
    with weak_path.open("w", encoding="utf-8") as f:
        for item in weak_spots[: args.weak_spots]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "input": str(input_path),
        "model": str(model_path),
        "model_type": "t3_policyvalue_fixed_slot",
        "device": str(device),
        "records": record_count,
        "turns": sorted(include_turns),
        "metric_note": (
            "Policy logits are used only for ranking. score_mae is logit-vs-teacher "
            "scale and should not be used as calibration."
        ),
        "overall": overall.as_dict(),
        "by_turn": {str(k): v.as_dict() for k, v in sorted(by_turn.items())},
        "weak_spots_path": str(weak_path),
        "top_actions_note": "model_top3 actions are ranked by fixed-slot policy logits.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_markdown(output_dir / "summary.md", report, weak_spots)
    print(json.dumps(report, indent=2))
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate T3 policy-value model against tutor teacher data")
    parser.add_argument("input", help="Tutor teacher JSONL")
    parser.add_argument("--model", required=True, help="Path to t3_policyvalue_v2_best.pt")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--turns", default="3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--weak-spots", type=int, default=50)
    args = parser.parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
