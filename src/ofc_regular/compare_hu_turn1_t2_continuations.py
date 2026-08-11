"""Compare HU Turn1 pilot continuation profiles on the same sampled states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate_matchups import PROFILE_CHOICES
from .hu_turn1_teacher_pilot import build_turn1_pilot_samples


def _action_signature(action: dict[str, Any] | None) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]] | None:
    if not action:
        return None
    placements = tuple((str(card), str(row)) for card, row in action.get("placements", ()))
    discards = tuple(str(card) for card in action.get("discards", ()))
    return placements, discards


def _state_signature(row: dict[str, Any]) -> str:
    payload = {
        "hand_seed": row.get("hand_seed"),
        "player": row.get("player"),
        "seat": row.get("seat"),
        "board": row.get("board"),
        "opponent_board": row.get("opponent_board"),
        "dealt": row.get("dealt"),
        "visible_dead_cards": row.get("visible_dead_cards"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _ranked_actions(row: dict[str, Any]) -> dict[tuple[tuple[tuple[str, str], ...], tuple[str, ...]], dict[str, Any]]:
    ranked: dict[tuple[tuple[tuple[str, str], ...], tuple[str, ...]], dict[str, Any]] = {}
    for rank, action in enumerate(row.get("actions") or (), start=1):
        signature = _action_signature(action)
        if signature is None:
            continue
        ranked[signature] = {
            "rank": rank,
            "score": float(action.get("score", action.get("ev", 0.0)) or 0.0),
        }
    return ranked


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_profile_comparison(
    profile_rows: dict[str, list[dict[str, Any]]],
    profile_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    profiles = list(profile_rows)
    baseline = profiles[0]
    baseline_rows = profile_rows[baseline]
    baseline_best = [
        _action_signature(row.get("actions", [None])[0] if row.get("actions") else None)
        for row in baseline_rows
    ]
    baseline_states = [_state_signature(row) for row in baseline_rows]

    comparisons: dict[str, Any] = {}
    for profile in profiles:
        rows = profile_rows[profile]
        best_matches = 0
        state_matches = 0
        comparable_states = 0
        profile_best_in_baseline_top3 = 0
        profile_best_baseline_ranks: list[float] = []
        profile_best_baseline_regrets: list[float] = []
        baseline_best_profile_ranks: list[float] = []
        baseline_best_profile_regrets: list[float] = []
        missing_profile_best_in_baseline = 0
        missing_baseline_best_in_profile = 0
        compared = min(len(rows), len(baseline_rows))
        for index in range(compared):
            state_match = _state_signature(rows[index]) == baseline_states[index]
            state_matches += int(state_match)
            candidate_best = _action_signature(
                rows[index].get("actions", [None])[0] if rows[index].get("actions") else None
            )
            best_matches += int(candidate_best == baseline_best[index])
            if not state_match:
                continue
            comparable_states += 1
            baseline_ranked = _ranked_actions(baseline_rows[index])
            profile_ranked = _ranked_actions(rows[index])
            baseline_best_signature = baseline_best[index]
            profile_best_signature = candidate_best
            if profile_best_signature is not None and profile_best_signature in baseline_ranked:
                rank = float(baseline_ranked[profile_best_signature]["rank"])
                profile_best_baseline_ranks.append(rank)
                profile_best_in_baseline_top3 += int(rank <= 3.0)
                baseline_best_score = float(
                    baseline_ranked[baseline_best_signature]["score"]
                    if baseline_best_signature in baseline_ranked
                    else 0.0
                )
                profile_best_baseline_regrets.append(
                    baseline_best_score - float(baseline_ranked[profile_best_signature]["score"])
                )
            else:
                missing_profile_best_in_baseline += 1
            if baseline_best_signature is not None and baseline_best_signature in profile_ranked:
                rank = float(profile_ranked[baseline_best_signature]["rank"])
                baseline_best_profile_ranks.append(rank)
                profile_best_score = float(
                    profile_ranked[profile_best_signature]["score"]
                    if profile_best_signature in profile_ranked
                    else 0.0
                )
                baseline_best_profile_regrets.append(
                    profile_best_score - float(profile_ranked[baseline_best_signature]["score"])
                )
            else:
                missing_baseline_best_in_profile += 1
        comparisons[profile] = {
            "samples": len(rows),
            "seconds_per_sample": profile_summaries[profile].get("seconds_per_sample"),
            "mean_action_count": profile_summaries[profile].get("mean_action_count"),
            "topk_decisions": profile_summaries[profile].get("topk_decisions", 0),
            "topk_overrides": profile_summaries[profile].get("topk_overrides", 0),
            "t2_choose_action_seconds": (profile_summaries[profile].get("profile_stats") or {}).get(
                "choose_action_T2_seconds", 0.0
            ),
            "compared_to": baseline,
            "state_match_rate": (state_matches / compared) if compared else 0.0,
            "comparable_state_count": comparable_states,
            "best_action_match_rate": (best_matches / compared) if compared else 0.0,
            "profile_best_in_baseline_top3_rate": (
                profile_best_in_baseline_top3 / comparable_states if comparable_states else 0.0
            ),
            "profile_best_baseline_rank_mean": _mean(profile_best_baseline_ranks),
            "profile_best_baseline_regret_mean": _mean(profile_best_baseline_regrets),
            "profile_best_baseline_regret_max": (
                max(profile_best_baseline_regrets) if profile_best_baseline_regrets else 0.0
            ),
            "baseline_best_profile_rank_mean": _mean(baseline_best_profile_ranks),
            "baseline_best_profile_regret_mean": _mean(baseline_best_profile_regrets),
            "baseline_best_profile_regret_max": (
                max(baseline_best_profile_regrets) if baseline_best_profile_regrets else 0.0
            ),
            "missing_profile_best_in_baseline": missing_profile_best_in_baseline,
            "missing_baseline_best_in_profile": missing_baseline_best_in_profile,
        }
    return {
        "schema": "hu_turn1_t2_continuation_comparison_v1",
        "baseline_profile": baseline,
        "profiles": profiles,
        "comparisons": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        default="stage9f_p2,stage9f_fast_t2_t1_teacher",
        help="Comma-separated profiles; first profile is the comparison baseline.",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--future-samples", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026062401)
    parser.add_argument("--opening-lookahead-samples", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = [profile.strip() for profile in args.profiles.split(",") if profile.strip()]
    if len(profiles) < 2:
        raise SystemExit("--profiles must contain at least two profiles")
    unknown = [profile for profile in profiles if profile not in PROFILE_CHOICES]
    if unknown:
        raise SystemExit(f"unknown profile(s): {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_rows: dict[str, list[dict[str, Any]]] = {}
    profile_summaries: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        rows, summary = build_turn1_pilot_samples(
            samples=args.samples,
            seed=args.seed,
            profile=profile,
            opponent_profile=profile,
            future_samples=args.future_samples,
            max_actions=args.max_actions,
            opening_lookahead_samples=args.opening_lookahead_samples,
            collect_topk_log=True,
        )
        profile_rows[profile] = rows
        profile_summaries[profile] = summary
        prefix = args.output_dir / profile
        with (prefix.with_suffix(".jsonl")).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        prefix.with_name(prefix.name + "_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    comparison = summarize_profile_comparison(profile_rows, profile_summaries)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
