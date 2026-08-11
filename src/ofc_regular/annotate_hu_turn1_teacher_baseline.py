"""Annotate all-action HU Turn1 teacher rows with the fixed baseline action."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .ai_profiles import ModelPaths, build_policy, load_model_bundle, required_profiles
from .evaluate_matchups import PROFILE_CHOICES
from .hu_infoset import (
    ActorObservation,
    InformationSetError,
    actor_observation_from_record,
)
from .play_ai import _choose_from_observation
from .state import Board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="stage9f_p2")
    parser.add_argument("--seed", type=int, default=2026071702)
    parser.add_argument("--opening-lookahead-samples", type=int, default=1)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return rows


def board_from_json(payload: dict[str, Any]) -> Board:
    return Board.from_rows(
        top=payload.get("top", ()),
        middle=payload.get("middle", ()),
        bottom=payload.get("bottom", ()),
    )


def action_signature(action: Any) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    if isinstance(action, dict):
        raw_placements = action.get("placements", ())
        raw_discards = action.get("discards", ())
    else:
        raw_placements = getattr(action, "placements", ())
        raw_discards = getattr(action, "discards", ())
    placements = tuple(sorted((str(card), str(row)) for card, row in raw_placements))
    discards = tuple(sorted(str(card) for card in raw_discards))
    return placements, discards


def action_index(action: dict[str, Any], fallback: int) -> int:
    raw = action.get("original_index", action.get("action_index", fallback))
    return int(raw)


def actor_observation_for_row(row: dict[str, Any]) -> ActorObservation:
    """Return only the cards that were available to the row's acting seat."""
    nested = row.get("policy_observation")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise InformationSetError("policy_observation must be a mapping")
        declared = ActorObservation.from_dict(nested)
        derived = actor_observation_from_record(row)
        if declared != derived:
            raise InformationSetError(
                "policy_observation disagrees with teacher row"
            )
        return declared
    if "visible_dead_cards" not in row:
        raise InformationSetError(
            "T1 annotation requires explicit visible_dead_cards"
        )
    return actor_observation_from_record(row)


def choose_policy_action_from_row(
    policy: Any,
    row: dict[str, Any],
    *,
    default_seed: int,
) -> Any:
    observation = actor_observation_for_row(row)
    hand_seed = int(row.get("hand_seed", row.get("hand_id", default_seed)))
    return _choose_from_observation(
        policy,
        observation,
        hand_id=hand_seed,
        game_id=row.get("game_id", hand_seed),
        decision_seed=hand_seed,
    )


def annotate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    choose_baseline: Callable[[dict[str, Any]], Any],
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seat_counts: Counter[str] = Counter()
    duplicate_matches = 0
    for row_number, row in enumerate(rows):
        actions = row.get("actions", ()) or ()
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"row {row_number} has no actions")
        selected = choose_baseline(row)
        selected_signature = action_signature(selected)
        matches = [
            (index, action)
            for index, action in enumerate(actions)
            if action_signature(action) == selected_signature
        ]
        if not matches:
            raise ValueError(f"row {row_number} baseline action is missing from legal actions")
        duplicate_matches += max(0, len(matches) - 1)
        row_index, matched = matches[0]
        annotated = dict(row)
        annotated.update(
            {
                "baseline_profile": profile,
                "baseline_action": matched,
                "baseline_action_row_index": int(row_index),
                "baseline_action_index": action_index(matched, row_index),
                "baseline_teacher_ev": float(matched.get("score", matched.get("ev", 0.0))),
            }
        )
        output.append(annotated)
        seat_counts[str(row.get("seat", "unknown"))] += 1
    summary = {
        "schema": "hu_turn1_teacher_baseline_annotation_summary_v1",
        "rows": len(output),
        "baseline_profile": profile,
        "seat_counts": dict(sorted(seat_counts.items())),
        "missing_actions": 0,
        "duplicate_action_matches": int(duplicate_matches),
    }
    return output, summary


def make_policy_chooser(*, profile: str, seed: int, opening_lookahead_samples: int) -> Callable[[dict[str, Any]], Any]:
    bundle = load_model_bundle(ModelPaths(), required_profiles(profile, profile))
    policies = {
        seat: build_policy(
            profile,
            bundle,
            seed=seed + index,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
        )
        for index, seat in enumerate(("first", "second"))
    }

    def choose(row: dict[str, Any]) -> Any:
        seat = str(row.get("seat", "first"))
        policy = policies[seat]
        return choose_policy_action_from_row(
            policy,
            row,
            default_seed=seed,
        )

    return choose


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    chooser = make_policy_chooser(
        profile=args.profile,
        seed=int(args.seed),
        opening_lookahead_samples=int(args.opening_lookahead_samples),
    )
    annotated, summary = annotate_rows(rows, choose_baseline=chooser, profile=args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in annotated:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
