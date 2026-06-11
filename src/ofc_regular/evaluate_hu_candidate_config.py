"""Evaluate a HU Turn3 candidate config against the current baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .ai_profiles import (
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
)
from .evaluate_model_set_matchup import evaluate_matchup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name-b", default="baseline")
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--trace-limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_model_path(config: dict[str, Any]) -> Path | None:
    if "model" in config:
        return Path(config["model"]["path"])
    if "primary_model" in config:
        return Path(config["primary_model"]["path"])
    return None


def support_model_path(config: dict[str, Any]) -> Path | None:
    if "support_model" not in config:
        return None
    return Path(config["support_model"]["path"])


def reference_model_path(config: dict[str, Any]) -> Path | None:
    if "reference_model" not in config:
        return None
    return Path(config["reference_model"]["path"])


def baseline_model_paths(config: dict[str, Any]) -> dict[str, Path]:
    payload = config.get("baseline_models", {})
    return {
        "opening": Path(payload.get("opening", DEFAULT_OPENING_MODEL)),
        "turn1": Path(payload.get("turn1", DEFAULT_TURN1_MODEL)),
        "turn2": Path(payload.get("turn2", DEFAULT_TURN2_MODEL)),
        "turn3": Path(payload.get("turn3", DEFAULT_TURN3_MODEL)),
    }


def namespace_from_config(config: dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    runtime = config.get("runtime", {})
    baseline = baseline_model_paths(config)
    hu_path = candidate_model_path(config)
    if hu_path is None:
        raise SystemExit("candidate config must contain model.path or primary_model.path")

    return argparse.Namespace(
        games=args.games,
        seed=args.seed,
        name_a=config.get("name", "hu_candidate"),
        name_b=args.name_b,
        opening_a=baseline["opening"],
        turn1_a=baseline["turn1"],
        turn2_a=baseline["turn2"],
        turn3_a=baseline["turn3"],
        hu_turn3_a=hu_path,
        hu_turn3_reference_a=reference_model_path(config),
        hu_turn3_support_a=support_model_path(config),
        hu_turn3_gate_a=None,
        hu_turn3_min_margin_a=float(runtime.get("hu_turn3_min_margin", 0.0) or 0.0),
        hu_turn3_reference_min_margin_a=float(
            runtime.get("hu_turn3_reference_min_margin", 0.0) or 0.0
        ),
        hu_turn3_min_support_margin_a=float(
            runtime.get("hu_turn3_min_support_margin", 0.0) or 0.0
        ),
        hu_turn3_min_gate_probability_a=0.0,
        hu_turn3_max_self_regret_a=runtime.get("hu_turn3_max_self_regret"),
        disable_hu_turn3_stage7_a=not bool(runtime.get("hu_turn3_stage7_enabled", True)),
        opening_b=baseline["opening"],
        turn1_b=baseline["turn1"],
        turn2_b=baseline["turn2"],
        turn3_b=baseline["turn3"],
        hu_turn3_b=None,
        hu_turn3_reference_b=None,
        hu_turn3_support_b=None,
        hu_turn3_gate_b=None,
        hu_turn3_min_margin_b=0.0,
        hu_turn3_reference_min_margin_b=0.0,
        hu_turn3_min_support_margin_b=0.0,
        hu_turn3_min_gate_probability_b=0.0,
        hu_turn3_max_self_regret_b=None,
        disable_hu_turn3_stage7_b=False,
        opening_lookahead_samples=64,
        prediction_threads=args.prediction_threads,
        progress_every=args.progress_every,
        trace_output=args.trace_output,
        trace_limit=args.trace_limit,
        output=args.output,
    )


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    matchup_args = namespace_from_config(config, args)
    summary = evaluate_matchup(matchup_args)
    summary["candidate_config"] = str(args.config)
    summary["candidate_status"] = config.get("status")
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
