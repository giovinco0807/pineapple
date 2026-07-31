"""Benchmark reranker suit-ensemble scoring latency for one decision state."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board, Observation
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.models.action_value_reranker import ActionValueReranker


class DummyPolicy(torch.nn.Module):
    def forward(self, state, mask=None):
        width = mask.shape[-1] if mask is not None else 1
        return torch.ones((state.shape[0], width), dtype=torch.float32, device=state.device)


def sample_t2_observation() -> Observation:
    return Observation(
        board_self=Board(
            top=["Js"],
            middle=["3h", "5h", "5s", "Ks"],
            bottom=["8d", "9d"],
        ),
        board_opponent=Board(
            top=["Ac", "Th"],
            middle=["3s", "Kc", "6s"],
            bottom=["2d", "Jd", "Jh", "Kd"],
        ),
        dealt_cards=["2s", "4d", "4h"],
        known_discards_self=["7s"],
        turn=2,
        is_btn=False,
    )


def parse_sizes(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def benchmark(args: argparse.Namespace) -> dict:
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionValueReranker.from_checkpoint(args.model, map_location=device)
    model.to(device)
    model.eval()

    obs = sample_t2_observation()
    actions = list(enumerate(get_turn_actions(obs.dealt_cards, obs.board_self)))
    sizes = parse_sizes(args.sizes)
    results = []

    for size in sizes:
        turns = {2} if size > 0 else set()
        evaluator = RolloutEvaluator(
            policy_net=DummyPolicy(),
            action_value_net=model,
            device=device,
            action_value_suit_ensemble_turns=turns,
            action_value_suit_ensemble_size=max(size, 1),
        )
        for _ in range(args.warmup):
            evaluator._score_candidates_action_value(obs, actions)
        if device == "cuda":
            torch.cuda.synchronize()

        timings = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            evaluator._score_candidates_action_value(obs, actions)
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

        label = "off" if size <= 0 else str(size)
        states_scored = len(actions) * (max(size, 1) if size > 0 else 1)
        results.append(
            {
                "ensemble_size": label,
                "turn": 2,
                "candidates": len(actions),
                "states_scored": states_scored,
                "iterations": int(args.iterations),
                "mean_ms": statistics.fmean(timings),
                "median_ms": statistics.median(timings),
                "min_ms": min(timings),
                "max_ms": max(timings),
            }
        )

    return {
        "model": args.model,
        "device": device,
        "torch_num_threads": torch.get_num_threads(),
        "results": results,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark action-value suit ensemble runtime")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sizes", default="0,4,8,12,24")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = benchmark(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
