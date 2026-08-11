"""Replay HU T2 Stage8b TopK hard negatives under the current objective.

This tool is intentionally narrow. It does not train a model and it does not
approve a runtime gate. It re-evaluates replay-ready TopK false positives with
the current FL EV and an explicit T3 continuation so they can be audited or
converted into future supervised hard-negative labels. Hidden-discard default is
the HU Stage3 reference action; legacy Stage7_candidate_A m5_r10 is opt-in.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .action_space import Action, generate_turn_actions
from .ai_profiles import (
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
)
from .evaluate_hu_turn2_stage8_seat_swap import (
    DEFAULT_STAGE3_REFERENCE,
    DEFAULT_STAGE7_MODEL,
    DEFAULT_T3_CONTINUATION,
    T3ContinuationMode,
    _t3_continuation_policy_name,
    _t3_policy_kwargs,
)
from .final_turn_decision_cache import FinalTurnDecisionCache
from .hu_belief import sample_hidden_card_particles, turn2_actor_observation
from .hu_turn2_teacher_data import _t3_continuation_metadata, evaluate_hu_turn2_actions
from .hu_turn3_batch_continuation import (
    HuTurn3ActionCache,
    HuTurn3DecisionCache,
    HuTurn3Stage3ReferenceCache,
    HuTurn3Stage7BatchConfig,
    Stage3StateFeatureCache,
)
from .hu_turn3_model import load_hu_action_value_model
from .play_ai import _prediction_thread_context
from .policy import RegularAiPolicy, action_to_json, policy_sample
from .state import Board
from .turn3_model import load_action_value_model


DEFAULT_INPUT = Path("outputs/evals/hu_turn2_stage8b_topk_hard_negatives/topk_false_positive_hard_negatives.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8b_topk_hard_negative_replay")


@dataclass
class ReplayModelParts:
    opening: Any
    turn1: Any
    turn2_baseline: Any
    turn3: Any
    hu_turn3_stage7: Any
    hu_turn3_reference: Any


def belief_replay_ready(row: dict[str, Any]) -> bool:
    """Return whether actor-visible state is sufficient for belief replay.

    Historical ``replay_ready`` flags also required realized opponent-private
    truth.  Do not carry that obsolete requirement into the new evaluator.
    """

    required = (
        "hero_board",
        "opponent_board",
        "cards_to_place",
        "baseline_action",
        "candidate_action",
    )
    return all(row.get(field) for field in required) and bool(
        row.get("hero_private_discards") or row.get("visible_dead_cards")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--future-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026062301)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "Number of input rows to skip before replaying. The skipped row count "
            "is still used as the global row_index so chunked runs keep the same "
            "deterministic replay seeds as a full unchunked run."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-baseline-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--hu-turn3-stage7-model", type=Path, default=DEFAULT_STAGE7_MODEL)
    parser.add_argument("--hu-turn3-reference-model", type=Path, default=DEFAULT_STAGE3_REFERENCE)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--opening-lookahead-samples", type=int, default=32)
    parser.add_argument(
        "--t3-continuation",
        choices=("stage3_reference_default", "stage7_m5_r10"),
        default=DEFAULT_T3_CONTINUATION,
        help=(
            "T3 continuation used for replay rollouts. Hidden-discard default "
            "is stage3_reference_default; stage7_m5_r10 is legacy opt-in."
        ),
    )
    parser.add_argument("--batched-continuation-batch-size", type=int, default=8192)
    parser.add_argument("--stage3-feature-encoder-mode", default="rust_direct")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_replay_rows(rows: list[dict[str, Any]], *, offset: int = 0, limit: int = 0) -> list[tuple[int, dict[str, Any]]]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    selected = list(enumerate(rows))[offset:]
    if limit > 0:
        selected = selected[:limit]
    return selected


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def board_from_json(value: dict[str, Any] | None) -> Board:
    data = value or {}
    return Board.from_rows(data.get("top", ()), data.get("middle", ()), data.get("bottom", ()))


def action_from_json(value: dict[str, Any] | None) -> Action:
    data = value or {}
    placements = tuple((str(card), str(row)) for card, row in data.get("placements", ()))
    discards = tuple(str(card) for card in data.get("discards", ()))
    return Action(placements=placements, discards=discards)


def action_signature_from_parts(placements: Iterable[Iterable[Any]], discards: Iterable[Any]) -> str:
    placement_key = tuple(sorted((str(card), str(row)) for card, row in placements))
    discard_key = tuple(sorted(str(card) for card in discards))
    return json.dumps({"placements": placement_key, "discards": discard_key}, sort_keys=True)


def action_signature(action: Action) -> str:
    return action_signature_from_parts(action.placements, action.discards)


def logged_action_signature(value: dict[str, Any] | None) -> str:
    data = value or {}
    return action_signature_from_parts(data.get("placements", ()), data.get("discards", ()))


def resolve_action_index(actions: list[Action], logged_action: dict[str, Any] | None) -> int | None:
    target = logged_action_signature(logged_action)
    matches = [index for index, action in enumerate(actions) if action_signature(action) == target]
    if len(matches) != 1:
        return None
    return matches[0]


def baseline_model_index(
    *,
    board: Board,
    dealt_cards: tuple[str, ...],
    actions: list[Action],
    baseline_model: Any,
) -> int:
    sample = policy_sample(board, dealt_cards, actions)
    predictions = baseline_model.predict_sample(sample)
    return int(np.argmax(predictions))


def deterministic_replay_seed(*, base_seed: int, row: dict[str, Any], row_index: int, future_samples: int) -> int:
    payload = {
        "base_seed": base_seed,
        "row_index": row_index,
        "future_samples": future_samples,
        "state_signature": row.get("state_signature"),
        "action_signature": row.get("action_signature"),
        "hand_seed": row.get("hand_seed"),
        "config_id": row.get("config_id"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 2_147_483_647


def find_action_record(sample: dict[str, Any], original_index: int) -> dict[str, Any] | None:
    for action in sample.get("legal_actions", ()):
        if int(action.get("original_index", -1)) == int(original_index):
            return action
    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        output = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(output):
        return default
    return output


def label_from_delta(delta: float, se: float) -> dict[str, Any]:
    lcb196 = delta - 1.96 * se
    lcb164 = delta - 1.64 * se
    return {
        "replay_delta_lcb196": lcb196,
        "replay_delta_lcb164": lcb164,
        "safe_lcb196_label": "positive" if lcb196 > 0.0 else ("negative" if delta <= 0.0 else "gray"),
        "safe_lcb164_label": "positive" if lcb164 > 0.0 else ("negative" if delta <= 0.0 else "gray"),
        "hard_negative_label": int(delta < 0.0),
    }


def replay_source_metadata(row: dict[str, Any], *, row_index: int, candidate_index: int) -> dict[str, Any]:
    hand_seed = row.get("hand_seed", row_index)
    return {
        "source": "hu_turn2_stage8b_topk_hard_negative_replay",
        "source_bucket": "topk_hard_negative_replay",
        "source_bucket_requested": "topk_hard_negative_replay",
        "source_bucket_actual": "topk_hard_negative_replay",
        "hand_seed": row.get("hand_seed"),
        "hand_id": row.get("hand_id", hand_seed),
        "game_id": row.get("game_id", hand_seed),
        "sample_id": f"topk_hard_negative_replay:{hand_seed}:{candidate_index}",
        "state_id": f"topk_hard_negative_replay:{hand_seed}",
        "source_config_id": row.get("config_id", ""),
    }


def load_parts(args: argparse.Namespace) -> ReplayModelParts:
    return ReplayModelParts(
        opening=load_action_value_model(args.opening_model),
        turn1=load_action_value_model(args.turn1_model),
        turn2_baseline=load_action_value_model(args.turn2_baseline_model),
        turn3=load_action_value_model(args.turn3_model),
        hu_turn3_stage7=load_hu_action_value_model(args.hu_turn3_stage7_model),
        hu_turn3_reference=load_hu_action_value_model(args.hu_turn3_reference_model),
    )


def make_rollout_policy(
    parts: ReplayModelParts,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
    t3_continuation: T3ContinuationMode,
) -> RegularAiPolicy:
    return RegularAiPolicy(
        opening_model=parts.opening,
        turn1_model=parts.turn1,
        turn2_model=parts.turn2_baseline,
        turn3_model=parts.turn3,
        **_t3_policy_kwargs(parts, t3_continuation),
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
    )


def replay_one(
    row: dict[str, Any],
    *,
    row_index: int,
    args: argparse.Namespace,
    parts: ReplayModelParts,
    batch_config: HuTurn3Stage7BatchConfig,
    t3_cache: HuTurn3DecisionCache,
    t3_reference_cache: HuTurn3Stage3ReferenceCache,
    t3_state_feature_cache: Stage3StateFeatureCache,
    t3_action_cache: HuTurn3ActionCache,
    final_turn_cache: FinalTurnDecisionCache,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started_at = time.perf_counter()
    summary: dict[str, Any] = {
        "row_index": row_index,
        "source_log": row.get("source_log", ""),
        "source_config_id": row.get("config_id", ""),
        "hand_seed": row.get("hand_seed", ""),
        "seat": row.get("seat", ""),
        "t3_continuation": args.t3_continuation,
        "t3_continuation_policy": _t3_continuation_policy_name(args.t3_continuation),
        "status": "failed",
    }
    if not belief_replay_ready(row):
        summary["status"] = "not_replay_ready"
        return summary, None

    board = board_from_json(row.get("hero_board"))
    opponent_board = board_from_json(row.get("opponent_board"))
    dealt = tuple(str(card) for card in row.get("cards_to_place", ()))
    visible_dead_cards = tuple(str(card) for card in row.get("visible_dead_cards", ()))
    hero_private_discards = tuple(str(card) for card in row.get("hero_private_discards", ()))
    seat = str(row.get("seat") or "first")
    if seat not in {"first", "second"}:
        summary["status"] = "invalid_seat"
        return summary, None

    actions = generate_turn_actions(board, dealt)
    logged_baseline_index = resolve_action_index(actions, row.get("baseline_action"))
    candidate_index = resolve_action_index(actions, row.get("candidate_action"))
    if logged_baseline_index is None or candidate_index is None:
        summary.update(
            {
                "status": "action_mapping_failed",
                "logged_baseline_index": logged_baseline_index,
                "candidate_index": candidate_index,
                "legal_action_count": len(actions),
            }
        )
        return summary, None

    current_baseline_index = baseline_model_index(
        board=board,
        dealt_cards=dealt,
        actions=actions,
        baseline_model=parts.turn2_baseline,
    )
    action_indices = sorted({logged_baseline_index, candidate_index, current_baseline_index})
    replay_seed = deterministic_replay_seed(
        base_seed=args.seed,
        row=row,
        row_index=row_index,
        future_samples=args.future_samples,
    )
    observation = turn2_actor_observation(
        hero_board=board,
        opponent_public_board=opponent_board,
        dealt_cards=dealt,
        hero_seat=seat,
        hero_private_discards=hero_private_discards,
        visible_dead_cards=visible_dead_cards or None,
    )
    belief_batch = sample_hidden_card_particles(
        observation,
        base_seed=replay_seed,
        run_id=(
            "replay_hu_turn2_stage8b_topk_hard_negatives"
            f"|row={row_index}|state={row.get('state_signature', '')}"
        ),
        sample_count=args.future_samples,
    )
    opponent_seat = "second" if seat == "first" else "first"
    sample = evaluate_hu_turn2_actions(
        board=board,
        dealt_cards=dealt,
        opponent_board=opponent_board,
        hero_seat=seat,
        continuation_policy=make_rollout_policy(
            parts,
            seed=replay_seed * 4 + 1,
            seat=seat,
            opening_lookahead_samples=args.opening_lookahead_samples,
            t3_continuation=args.t3_continuation,
        ),
        opponent_policy=make_rollout_policy(
            parts,
            seed=replay_seed * 4 + 2,
            seat=opponent_seat,
            opening_lookahead_samples=args.opening_lookahead_samples,
            t3_continuation=args.t3_continuation,
        ),
        baseline_turn2_model=parts.turn2_baseline,
        future_samples=args.future_samples,
        future_rollout_seed=replay_seed,
        action_indices=action_indices,
        use_batched_continuation=True,
        batched_continuation_config=batch_config,
        batched_continuation_cache=t3_cache,
        batched_reference_cache=t3_reference_cache,
        batched_state_feature_cache=t3_state_feature_cache,
        batched_action_cache=t3_action_cache,
        batched_continuation_batch_size=args.batched_continuation_batch_size,
        final_turn_cache=final_turn_cache,
        use_final_turn_cache=True,
        continuation_policy_name=_t3_continuation_policy_name(args.t3_continuation),
        continuation_metadata=_t3_continuation_metadata(args.t3_continuation),
        observation=observation,
        belief_batch=belief_batch,
    )
    if sample is None:
        summary.update(
            {
                "status": "teacher_replay_failed",
                "logged_baseline_index": logged_baseline_index,
                "candidate_index": candidate_index,
                "current_baseline_index": current_baseline_index,
                "evaluated_action_count": len(action_indices),
            }
        )
        return summary, None

    candidate = find_action_record(sample, candidate_index)
    logged_baseline = find_action_record(sample, logged_baseline_index)
    current_baseline = find_action_record(sample, current_baseline_index)
    if candidate is None or current_baseline is None:
        summary.update(
            {
                "status": "replayed_action_missing",
                "logged_baseline_index": logged_baseline_index,
                "candidate_index": candidate_index,
                "current_baseline_index": current_baseline_index,
                "evaluated_action_count": len(action_indices),
            }
        )
        return summary, sample

    candidate_ev = safe_float(candidate.get("ev"))
    current_baseline_ev = safe_float(current_baseline.get("ev"))
    logged_baseline_ev = safe_float(logged_baseline.get("ev")) if logged_baseline is not None else current_baseline_ev
    delta_current = candidate_ev - current_baseline_ev
    current_baseline_se = safe_float(current_baseline.get("ev_standard_error"))
    candidate_se = safe_float(candidate.get("ev_standard_error"))
    unpaired_se = math.sqrt(candidate_se * candidate_se + current_baseline_se * current_baseline_se)
    paired_count = int(sample.get("paired_delta_count", 0) or 0)
    paired_delta = safe_float(sample.get("paired_delta_mean"), delta_current)
    paired_se = safe_float(sample.get("paired_delta_standard_error"), unpaired_se)
    delta_for_label = paired_delta if paired_count > 0 else delta_current
    se_for_label = paired_se if paired_count > 0 else unpaired_se
    labels = label_from_delta(delta_for_label, se_for_label)
    action_mapping_status = "ok"
    if current_baseline_index != logged_baseline_index:
        action_mapping_status = "current_baseline_mismatch"
    elif int(row.get("baseline_action_index", logged_baseline_index)) != logged_baseline_index:
        action_mapping_status = "logged_index_mismatch"
    elif int(row.get("candidate_action_index", candidate_index)) != candidate_index:
        action_mapping_status = "logged_index_mismatch"

    sample.update(
        {
            **replay_source_metadata(row, row_index=row_index, candidate_index=candidate_index),
            "topk_hard_negative_source": row,
            "topk_hard_negative_replay": {
                "future_samples": args.future_samples,
                "future_rollout_seed": replay_seed,
                "logged_baseline_index": logged_baseline_index,
                "candidate_index": candidate_index,
                "current_baseline_index": current_baseline_index,
                "action_mapping_status": action_mapping_status,
                "candidate_ev": candidate_ev,
                "logged_baseline_ev": logged_baseline_ev,
                "current_baseline_ev": current_baseline_ev,
                "delta_vs_logged_baseline": candidate_ev - logged_baseline_ev,
                "delta_vs_current_baseline": delta_current,
                "delta_for_label": delta_for_label,
                "delta_standard_error_for_label": se_for_label,
                **labels,
            },
        }
    )
    summary.update(
        {
            "status": "ok",
            "logged_baseline_index": logged_baseline_index,
            "candidate_index": candidate_index,
            "current_baseline_index": current_baseline_index,
            "action_mapping_status": action_mapping_status,
            "evaluated_action_count": len(action_indices),
            "future_samples": args.future_samples,
            "replay_seed": replay_seed,
            "candidate_ev": candidate_ev,
            "logged_baseline_ev": logged_baseline_ev,
            "current_baseline_ev": current_baseline_ev,
            "delta_vs_logged_baseline": candidate_ev - logged_baseline_ev,
            "delta_vs_current_baseline": delta_current,
            "paired_delta_count": paired_count,
            "paired_delta_mean": paired_delta if paired_count > 0 else "",
            "paired_delta_standard_error": paired_se if paired_count > 0 else "",
            "delta_for_label": delta_for_label,
            "delta_standard_error_for_label": se_for_label,
            **labels,
            "old_realized_delta": row.get("realized_delta", ""),
            "old_confirm_delta": row.get("confirm_delta", ""),
            "old_predicted_delta": row.get("predicted_delta", ""),
            "old_gate_probability": row.get("gate_probability", ""),
            "runtime_seconds": time.perf_counter() - started_at,
        }
    )
    return summary, sample


def write_summary_markdown(path: Path, rows: list[dict[str, Any]], *, args: argparse.Namespace) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    deltas = [safe_float(row.get("delta_for_label")) for row in ok_rows]
    hard_negatives = sum(1 for row in ok_rows if int(row.get("hard_negative_label", 0)) == 1)
    lines = [
        "# HU T2 Stage8b TopK Hard Negative Replay",
        "",
        "This is a replay/audit artifact only. It does not approve T2 production, P2 fixed status, T1, or 50k teacher generation.",
        "",
        f"- input: `{args.input_jsonl}`",
        f"- future samples: {args.future_samples}",
        f"- T3 continuation: `{_t3_continuation_policy_name(args.t3_continuation)}`",
        f"- t3_continuation: `{args.t3_continuation}`",
        f"- rows: {len(rows)}",
        f"- ok rows: {len(ok_rows)}",
        f"- hard-negative labels after replay: {hard_negatives}",
        f"- replay delta mean: {float(np.mean(deltas)) if deltas else 0.0:.4f}",
        "",
        "| status | rows |",
        "|---|---:|",
    ]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    for status, count in sorted(counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use `delta_for_label` and `delta_standard_error_for_label` as replay diagnostics.",
            "- If `action_mapping_status` is not `ok`, inspect the row before using it for training.",
            "- Rows with `hard_negative_label=1` are candidates for the next Stage8c hard-negative training pass after feature generation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_source_rows = read_jsonl(args.input_jsonl)
    selected_rows = select_replay_rows(all_source_rows, offset=args.offset, limit=args.limit)
    started_at = time.time()
    parts = load_parts(args)
    stage7_enabled = args.t3_continuation == "stage7_m5_r10"
    batch_config = HuTurn3Stage7BatchConfig(
        stage7_enabled=stage7_enabled,
        hu_turn3_min_margin=5.0 if stage7_enabled else 0.0,
        hu_turn3_reference_min_margin=10.0 if stage7_enabled else 0.0,
        batch_size=args.batched_continuation_batch_size,
        stage3_feature_encoder_mode=args.stage3_feature_encoder_mode,
    )
    summaries: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    with _prediction_thread_context(args.prediction_threads):
        t3_cache = HuTurn3DecisionCache()
        t3_reference_cache = HuTurn3Stage3ReferenceCache()
        t3_state_feature_cache = Stage3StateFeatureCache()
        t3_action_cache = HuTurn3ActionCache()
        final_turn_cache = FinalTurnDecisionCache()
        for row_index, row in selected_rows:
            summary, sample = replay_one(
                row,
                row_index=row_index,
                args=args,
                parts=parts,
                batch_config=batch_config,
                t3_cache=t3_cache,
                t3_reference_cache=t3_reference_cache,
                t3_state_feature_cache=t3_state_feature_cache,
                t3_action_cache=t3_action_cache,
                final_turn_cache=final_turn_cache,
            )
            summaries.append(summary)
            if sample is not None:
                samples.append(sample)

    write_jsonl(args.output_dir / "topk_hard_negative_replay_teacher.jsonl", samples)
    write_csv(args.output_dir / "topk_hard_negative_replay_summary.csv", summaries)
    write_summary_markdown(args.output_dir / "topk_hard_negative_replay_summary.md", summaries, args=args)
    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(args.output_dir),
        "future_samples": args.future_samples,
        "seed": args.seed,
        "source_total_rows": len(all_source_rows),
        "offset": args.offset,
        "limit": args.limit,
        "rows": len(selected_rows),
        "samples_written": len(samples),
        "elapsed_seconds": time.time() - started_at,
        "t3_continuation": args.t3_continuation,
        "t3_continuation_policy": _t3_continuation_policy_name(args.t3_continuation),
        "t3_continuation_metadata": _t3_continuation_metadata(args.t3_continuation),
    }
    (args.output_dir / "topk_hard_negative_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
