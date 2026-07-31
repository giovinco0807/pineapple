"""Add exact late-turn metrics to a Route10 review JSON file.

By default this enriches only the chosen T3/T4 actions.  Use --all-candidates
when preparing a slower, full diagnostic artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from ai.tutor.exact_late import (
    CardNormalizer,
    exact_t4_draw_distribution,
    normalize_board,
    remaining_deck_size,
    terminal_metrics,
)


def _normalize_exclude(cards: Iterable[str], normalizer: CardNormalizer) -> list[str]:
    return normalizer.cards([str(card) for card in cards if card])


def _raw_cards(board: Dict[str, Any]) -> list[str]:
    return (
        list(board.get("top", []) or [])
        + list(board.get("middle", []) or board.get("mid", []) or [])
        + list(board.get("bottom", []) or board.get("bot", []) or [])
    )


def _dead_cards_without_visible(trace: Dict[str, Any], candidate: Dict[str, Any]) -> list[str]:
    visible = _raw_cards(candidate.get("next_board") or {}) + _raw_cards(trace.get("opponent_board") or {})
    visible_counts: dict[str, int] = {}
    for card in visible:
        visible_counts[str(card)] = visible_counts.get(str(card), 0) + 1
    out: list[str] = []
    for card in trace.get("dead_before", []) or []:
        card = str(card)
        if visible_counts.get(card, 0) > 0:
            visible_counts[card] -= 1
            continue
        out.append(card)
    return out


def _metrics_for_candidate(trace: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any] | None:
    turn = int(trace.get("turn", -1))
    if turn not in (3, 4):
        return None

    normalizer = CardNormalizer()
    board = normalize_board(candidate.get("next_board") or {}, normalizer)
    opponent = normalize_board(trace.get("opponent_board") or {}, normalizer)
    exclude = _normalize_exclude(_dead_cards_without_visible(trace, candidate), normalizer)
    discard = candidate.get("discard")
    if discard:
        exclude.extend(_normalize_exclude([discard], normalizer))

    target_remaining = (
        (candidate.get("recursive") or {}).get("remaining_deck_size")
        or ((candidate.get("recursive") or {}).get("mc") or {}).get("remaining_deck_size")
    )

    # Older Route10 artifacts collapse one or two jokers into "Xj".  If the
    # saved generator says the remaining deck is one card smaller, add the
    # unused second joker as dead so exact counts match that run's deck model.
    if target_remaining is not None and turn == 3:
        deck_size = remaining_deck_size(board, opponent_board=opponent, exclude=exclude)
        if deck_size == int(target_remaining) + 1 and "X2" not in set(exclude + board.all_cards() + opponent.all_cards()):
            exclude.append("X2")

    if turn == 3:
        return exact_t4_draw_distribution(board, opponent_board=opponent, exclude=exclude)
    if board.is_complete():
        metrics = terminal_metrics(board, opponent)
        return {
            "score": metrics["score"],
            "ev": metrics["score"],
            "raw_score": metrics["raw_score"],
            "royalty": metrics["royalty"],
            "bust_rate": 1.0 if metrics["bust"] else 0.0,
            "fl_rate": 1.0 if metrics["fl_any"] else 0.0,
            "fl_type_rates": {
                "qq": 1.0 if metrics["fl_type"] == "qq" else 0.0,
                "kk": 1.0 if metrics["fl_type"] == "kk" else 0.0,
                "aa": 1.0 if metrics["fl_type"] == "aa" else 0.0,
                "trips": 1.0 if metrics["fl_type"] == "trips" else 0.0,
            },
            "samples": 1,
            "source": "exact",
            "enumerated_draws": 1,
            "remaining_deck_size": 0,
            "start_turn": 5,
        }
    return None


def enrich_review(input_path: Path, output_path: Path, all_candidates: bool = False, max_traces: int = 0) -> Dict[str, Any]:
    review = json.loads(input_path.read_text(encoding="utf-8"))
    enriched_traces = 0
    enriched_candidates = 0

    for hand in review.get("hands", []):
        for decision in hand.get("decisions", []):
            decision.setdefault("metric_source", "estimated")
            for trace in decision.get("turn_traces", []):
                if int(trace.get("turn", -1)) not in (3, 4):
                    continue
                if max_traces and enriched_traces >= max_traces:
                    break
                chosen = trace.get("chosen")
                if chosen:
                    exact = _metrics_for_candidate(trace, chosen)
                    if exact is not None:
                        chosen["exact_terminal"] = exact
                        enriched_candidates += 1
                if all_candidates:
                    for candidate in trace.get("candidates", []):
                        if candidate is chosen:
                            continue
                        exact = _metrics_for_candidate(trace, candidate)
                        if exact is not None:
                            candidate["exact_terminal"] = exact
                            enriched_candidates += 1
                trace["exact_terminal_available"] = bool(chosen and chosen.get("exact_terminal"))
                enriched_traces += 1

    review["data_quality"] = {
        "t0_candidates": "estimated",
        "t0_sims": 300,
        "turn_traces": "estimated",
        "turn_trace_sims": 96,
        "exact_terminal_available": enriched_candidates > 0,
        "exact_terminal_scope": "all_candidates" if all_candidates else "chosen_actions",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "enriched_traces": enriched_traces,
        "enriched_candidates": enriched_candidates,
        "all_candidates": all_candidates,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich Route10 review JSON with exact T3/T4 metrics")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-candidates", action="store_true")
    parser.add_argument("--max-traces", type=int, default=0)
    args = parser.parse_args()
    summary = enrich_review(Path(args.input), Path(args.output), args.all_candidates, args.max_traces)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
