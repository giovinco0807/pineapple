"""Replay Stage9f tail-guard targets with independent MC.

This runner consumes the runtime-log targets produced by
``prepare_hu_turn2_stage9f_tail_guard_targets``.  It re-evaluates the exact
baseline action versus the exact Stage9f fired candidate action with new common
future decks.  The output is intended for tail-risk guard labels or selected
high-MC audit, not for direct production approval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .action_space import Action, generate_turn_actions
from .ai_profiles import ModelPaths, build_policy, load_model_bundle
from .final_turn_decision_cache import FinalTurnDecisionCache
from .hu_belief import sample_hidden_card_particles, turn2_actor_observation
from .hu_turn2_teacher_data import (
    _t3_continuation_metadata,
    _t3_continuation_policy_name,
    evaluate_hu_turn2_actions,
)
from .hu_turn3_batch_continuation import (
    HuTurn3ActionCache,
    HuTurn3DecisionCache,
    HuTurn3Stage3ReferenceCache,
    HuTurn3Stage7BatchConfig,
    Stage3StateFeatureCache,
)
from .state import Board


DEFAULT_TARGETS = Path(
    "outputs/training/hu_turn2_stage9f_cse2_csemax2_tail_guard_targets_10000/stage9f_tail_guard_targets.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage9f_tail_guard_replay")


@dataclass(frozen=True)
class PreparedTarget:
    row: dict[str, Any]
    board: Board | None
    opponent_board: Board | None
    dealt: tuple[str, ...]
    dead_cards: tuple[str, ...]
    visible_dead_cards: tuple[str, ...]
    hero_private_discards: tuple[str, ...]
    opponent_private_discards: tuple[str, ...]
    actions: tuple[Action, ...]
    baseline_index: int | None
    candidate_index: int | None
    replay_ready: bool
    failure_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mc-samples", type=int, default=512)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target-group", action="append", default=[])
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--partial-results-name", default="replay_results.partial.jsonl")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=9_000_000)
    parser.add_argument("--opening-lookahead-samples", type=int, default=1)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--disable-batched-continuation", action="store_true")
    parser.add_argument("--batched-continuation-batch-size", type=int, default=8192)
    parser.add_argument("--disable-continuation-cache", action="store_true")
    parser.add_argument("--continuation-cache-size", type=int, default=200_000)
    parser.add_argument("--disable-stage3-feature-fast-path", action="store_true")
    parser.add_argument("--stage3-feature-encoder-mode", default="rust_direct")
    parser.add_argument("--disable-final-turn-cache", action="store_true")
    parser.add_argument("--final-turn-cache-size", type=int, default=200_000)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def dedupe_rows_by_target_id(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    for ordinal, row in enumerate(rows):
        target_id = str(row.get("target_id") or f"row{ordinal}")
        if target_id not in by_id:
            ordered_ids.append(target_id)
        by_id[target_id] = row
    return [by_id[target_id] for target_id in ordered_ids]


def board_from_payload(payload: Any) -> Board | None:
    if not isinstance(payload, dict):
        return None
    try:
        return Board.from_rows(
            payload.get("top") or (),
            payload.get("middle") or (),
            payload.get("bottom") or (),
        )
    except Exception:
        return None


def canonical_action_from_payload(payload: Any) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    if not isinstance(payload, dict):
        return (), ()
    placements = tuple(sorted((str(card), str(row)) for card, row in (payload.get("placements") or ())))
    discards = tuple(sorted(str(card) for card in (payload.get("discards") or ())))
    return placements, discards


def canonical_action(action: Action) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    return tuple(sorted(action.placements)), tuple(sorted(action.discards))


def resolve_action_index(
    actions: tuple[Action, ...],
    payload: Any,
    hinted_index: Any,
) -> tuple[int | None, str]:
    target = canonical_action_from_payload(payload)
    hinted = safe_int(hinted_index)
    if hinted is not None and 0 <= hinted < len(actions):
        if canonical_action(actions[hinted]) == target:
            return hinted, "hint_match"
    matches = [index for index, action in enumerate(actions) if canonical_action(action) == target]
    if len(matches) == 1:
        return matches[0], "searched_match"
    if not matches:
        return None, "no_action_match"
    return None, "ambiguous_action_match"


def prepare_target(row: dict[str, Any]) -> PreparedTarget:
    missing: list[str] = []
    board = board_from_payload(row.get("hero_board"))
    opponent = board_from_payload(row.get("opponent_board"))
    if board is None:
        missing.append("hero_board")
    if opponent is None:
        missing.append("opponent_board")
    dealt = tuple(row.get("cards_to_place") or ())
    if len(dealt) != 3:
        missing.append("cards_to_place")
    actions: tuple[Action, ...] = ()
    baseline_index: int | None = None
    candidate_index: int | None = None
    baseline_status = "not_checked"
    candidate_status = "not_checked"
    if board is not None and len(dealt) == 3:
        try:
            actions = tuple(generate_turn_actions(board, dealt))
        except Exception:
            missing.append("legal_action_generation")
            actions = ()
    if actions:
        baseline_index, baseline_status = resolve_action_index(
            actions,
            row.get("baseline_action"),
            row.get("baseline_action_index"),
        )
        candidate_index, candidate_status = resolve_action_index(
            actions,
            row.get("candidate_action"),
            row.get("candidate_action_index"),
        )
        if baseline_index is None:
            missing.append(f"baseline_action:{baseline_status}")
        if candidate_index is None:
            missing.append(f"candidate_action:{candidate_status}")
    elif "legal_action_generation" not in missing:
        missing.append("legal_actions_empty")
    if not row.get("hero_private_discards") and not row.get("visible_dead_cards"):
        missing.append("hero_visible_discard")
    failure_reason = "ready" if not missing else "missing_" + "|".join(sorted(set(missing)))
    return PreparedTarget(
        row=row,
        board=board,
        opponent_board=opponent,
        dealt=dealt,
        dead_cards=tuple(row.get("dead_cards") or ()),
        visible_dead_cards=tuple(row.get("visible_dead_cards") or ()),
        hero_private_discards=tuple(row.get("hero_private_discards") or ()),
        opponent_private_discards=tuple(row.get("opponent_private_discards") or ()),
        actions=actions,
        baseline_index=baseline_index,
        candidate_index=candidate_index,
        replay_ready=not missing,
        failure_reason=failure_reason,
    )


def readiness_row(target: PreparedTarget) -> dict[str, Any]:
    row = target.row
    return {
        "target_id": row.get("target_id", ""),
        "target_group": row.get("target_group", ""),
        "config_id": row.get("config_id", ""),
        "seed": row.get("seed", ""),
        "hand_id": row.get("hand_id", ""),
        "seat": row.get("seat", ""),
        "replay_ready": int(target.replay_ready),
        "failure_reason": target.failure_reason,
        "action_count": len(target.actions),
        "baseline_action_index": target.baseline_index if target.baseline_index is not None else "",
        "candidate_action_index": target.candidate_index if target.candidate_index is not None else "",
        "realized_delta": row.get("realized_delta", ""),
        "tail_loss_label": row.get("tail_loss_label", ""),
        "severe_tail_loss_label": row.get("severe_tail_loss_label", ""),
        "safe_positive_label": row.get("safe_positive_label", ""),
    }


def replay_seed(target: PreparedTarget, *, seed_offset: int, mc_samples: int) -> int:
    original = safe_int(target.row.get("hand_seed"), safe_int(target.row.get("seed"), 0)) or 0
    return original + seed_offset + mc_samples


def build_batched_config(args: argparse.Namespace) -> HuTurn3Stage7BatchConfig:
    return HuTurn3Stage7BatchConfig(
        stage7_enabled=True,
        hu_turn3_min_margin=5.0,
        hu_turn3_reference_min_margin=10.0,
        batch_size=args.batched_continuation_batch_size,
        use_cache=not args.disable_continuation_cache,
        use_stage3_feature_fast_path=not args.disable_stage3_feature_fast_path,
        stage3_feature_encoder_mode=args.stage3_feature_encoder_mode,
        stage3_feature_replay_source="stage9f_tail_guard_replay",
        stage3_feature_replay_teacher_run_hash=f"stage9f_tail_guard_mc{args.mc_samples}",
    )


def action_by_original(actions: list[dict[str, Any]], original_index: int | None) -> dict[str, Any] | None:
    if original_index is None:
        return None
    for action in actions:
        if safe_int(action.get("original_index")) == original_index:
            return action
    return None


def action_rank(actions: list[dict[str, Any]], original_index: int | None) -> int | None:
    if original_index is None:
        return None
    for rank, action in enumerate(actions, start=1):
        if safe_int(action.get("original_index")) == original_index:
            return rank
    return None


def replay_target(
    target: PreparedTarget,
    *,
    args: argparse.Namespace,
    bundle: object,
    batched_config: HuTurn3Stage7BatchConfig,
    batched_cache: HuTurn3DecisionCache | None,
    batched_reference_cache: HuTurn3Stage3ReferenceCache | None,
    batched_state_feature_cache: Stage3StateFeatureCache | None,
    batched_action_cache: HuTurn3ActionCache | None,
    final_turn_cache: FinalTurnDecisionCache | None,
) -> dict[str, Any]:
    if not target.replay_ready:
        raise ValueError(target.failure_reason)
    assert target.board is not None
    assert target.opponent_board is not None
    seed = replay_seed(target, seed_offset=args.seed_offset, mc_samples=args.mc_samples)
    hero_seat = str(target.row.get("seat") or "first")
    opponent_seat = "second" if hero_seat == "first" else "first"
    observation = turn2_actor_observation(
        hero_board=target.board,
        opponent_public_board=target.opponent_board,
        dealt_cards=target.dealt,
        hero_seat=hero_seat,
        hero_private_discards=target.hero_private_discards,
        visible_dead_cards=target.visible_dead_cards or None,
    )
    belief_batch = sample_hidden_card_particles(
        observation,
        base_seed=seed,
        run_id=(
            "replay_hu_turn2_stage9f_tail_guard_targets"
            f"|target={target.row.get('target_id', '')}"
        ),
        sample_count=args.mc_samples,
    )
    hero_policy = build_policy(
        "stage7_m5_r10",
        bundle,
        seed=seed * 4 + 2,
        seat=hero_seat,
        opening_lookahead_samples=args.opening_lookahead_samples,
    )
    opponent_policy = build_policy(
        "stage7_m5_r10",
        bundle,
        seed=seed * 4 + 3,
        seat=opponent_seat,
        opening_lookahead_samples=args.opening_lookahead_samples,
    )
    started_at = time.perf_counter()
    sample = evaluate_hu_turn2_actions(
        board=target.board,
        dealt_cards=target.dealt,
        opponent_board=target.opponent_board,
        hero_seat=hero_seat,
        continuation_policy=hero_policy,
        opponent_policy=opponent_policy,
        baseline_turn2_model=bundle.turn2,
        future_samples=args.mc_samples,
        future_rollout_seed=seed,
        action_indices=(int(target.baseline_index), int(target.candidate_index)),
        use_batched_continuation=not args.disable_batched_continuation,
        batched_continuation_config=batched_config,
        batched_continuation_cache=batched_cache,
        batched_reference_cache=batched_reference_cache,
        batched_state_feature_cache=batched_state_feature_cache,
        batched_action_cache=batched_action_cache,
        batched_continuation_batch_size=args.batched_continuation_batch_size,
        final_turn_cache=final_turn_cache,
        use_final_turn_cache=final_turn_cache is not None,
        continuation_policy_name=_t3_continuation_policy_name("stage7_m5_r10"),
        continuation_metadata=_t3_continuation_metadata("stage7_m5_r10"),
        observation=observation,
        belief_batch=belief_batch,
    )
    elapsed = time.perf_counter() - started_at
    if sample is None:
        raise ValueError("evaluate_returned_none")
    actions = sample.get("actions") or []
    baseline_action = action_by_original(actions, target.baseline_index)
    candidate_action = action_by_original(actions, target.candidate_index)
    if baseline_action is None:
        raise ValueError("baseline_action_missing_after_replay")
    if candidate_action is None:
        raise ValueError("candidate_action_missing_after_replay")
    baseline_ev = safe_float(baseline_action.get("score", baseline_action.get("ev")))
    candidate_ev = safe_float(candidate_action.get("score", candidate_action.get("ev")))
    gain = candidate_ev - baseline_ev
    paired_mean = safe_float(sample.get("paired_delta_mean"), gain)
    paired_se = safe_float(sample.get("paired_delta_standard_error"))
    lower90 = paired_mean - 1.64 * paired_se
    lower95 = paired_mean - 1.96 * paired_se
    target_realized = safe_float(target.row.get("realized_delta"))
    return {
        "target_id": target.row.get("target_id", ""),
        "target_group": target.row.get("target_group", ""),
        "config_id": target.row.get("config_id", ""),
        "seed": target.row.get("seed", ""),
        "hand_id": target.row.get("hand_id", ""),
        "seat": hero_seat,
        "mc_n": args.mc_samples,
        "future_rollout_seed": seed,
        "belief_conditioned": bool(sample.get("belief_conditioned")),
        "observation_fingerprint": sample.get("observation_fingerprint", ""),
        "belief_batch_digest": sample.get("belief_batch_digest", ""),
        "baseline_action_index": target.baseline_index,
        "candidate_action_index": target.candidate_index,
        "baseline_ev": baseline_ev,
        "candidate_ev": candidate_ev,
        "gain_mean": paired_mean,
        "gain_stderr": paired_se,
        "gain_lower90": lower90,
        "gain_lower95": lower95,
        "gain_positive": int(paired_mean > 0.0),
        "gain_negative": int(paired_mean < 0.0),
        "gain_lcb95_positive": int(lower95 > 0.0),
        "high_mc_tail_loss_label": int(paired_mean < 0.0),
        "high_mc_severe_tail_loss_label": int(paired_mean <= -8.0),
        "input_realized_delta": target_realized,
        "input_tail_loss_label": target.row.get("tail_loss_label", ""),
        "input_severe_tail_loss_label": target.row.get("severe_tail_loss_label", ""),
        "input_safe_positive_label": target.row.get("safe_positive_label", ""),
        "sign_flip_vs_input_realized": int((paired_mean > 0.0) != (target_realized > 0.0)),
        "candidate_rank_high_mc": action_rank(actions, target.candidate_index),
        "baseline_rank_high_mc": action_rank(actions, target.baseline_index),
        "common_random_future_digest": sample.get("common_random_future_digest", ""),
        "paired_delta_count": sample.get("paired_delta_count", ""),
        "paired_delta_min": sample.get("paired_delta_min", ""),
        "paired_delta_p05": sample.get("paired_delta_p05", ""),
        "paired_delta_p50": sample.get("paired_delta_p50", ""),
        "paired_delta_p95": sample.get("paired_delta_p95", ""),
        "paired_delta_max": sample.get("paired_delta_max", ""),
        "paired_delta_le_neg6_rate": sample.get("paired_delta_le_neg6_rate", ""),
        "paired_delta_le_neg12_rate": sample.get("paired_delta_le_neg12_rate", ""),
        "paired_delta_le_neg20_rate": sample.get("paired_delta_le_neg20_rate", ""),
        "component_delta_summaries_json": json.dumps(
            (sample.get("paired_delta_by_action") or [{}])[0].get("component_delta_summaries", {}),
            sort_keys=True,
        )
        if sample.get("paired_delta_by_action")
        else "",
        "replay_status": "success",
        "failure_reason": "",
        "elapsed_seconds": elapsed,
    }


def failure_row(target: PreparedTarget, reason: str, *, mc_samples: int) -> dict[str, Any]:
    return {
        "target_id": target.row.get("target_id", ""),
        "target_group": target.row.get("target_group", ""),
        "config_id": target.row.get("config_id", ""),
        "seed": target.row.get("seed", ""),
        "hand_id": target.row.get("hand_id", ""),
        "seat": target.row.get("seat", ""),
        "mc_n": mc_samples,
        "gain_mean": "",
        "gain_stderr": "",
        "replay_status": "failed",
        "failure_reason": reason,
    }


def summarize_results(rows: list[dict[str, Any]], readiness: list[PreparedTarget], *, readiness_only: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    successes = [row for row in rows if row.get("replay_status") == "success"]
    failures = [row for row in rows if row.get("replay_status") == "failed"]
    for group in sorted({str(row.row.get("target_group", "")) for row in readiness} | {str(row.get("target_group", "")) for row in rows}):
        group_successes = [row for row in successes if row.get("target_group") == group]
        gains = [safe_float(row.get("gain_mean")) for row in group_successes]
        summary_rows.append(
            {
                "target_group": group,
                "readiness_rows": sum(1 for item in readiness if item.row.get("target_group") == group),
                "replay_successes": len(group_successes),
                "replay_failures": sum(1 for row in failures if row.get("target_group") == group),
                "gain_mean": sum(gains) / len(gains) if gains else 0.0,
                "gain_min": min(gains, default=0.0),
                "gain_max": max(gains, default=0.0),
                "tail_loss_count": sum(1 for row in group_successes if safe_float(row.get("gain_mean")) < 0.0),
                "lcb95_positive_count": sum(safe_int(row.get("gain_lcb95_positive"), 0) or 0 for row in group_successes),
                "sign_flip_count": sum(safe_int(row.get("sign_flip_vs_input_realized"), 0) or 0 for row in group_successes),
            }
        )
    manifest = {
        "schema": "hu_turn2_stage9f_tail_guard_replay_manifest_v1",
        "readiness_only": readiness_only,
        "targets": len(readiness),
        "readiness_ready": sum(1 for item in readiness if item.replay_ready),
        "replay_successes": len(successes),
        "replay_failures": len(failures),
        "failure_reasons": dict(Counter(str(row.get("failure_reason", "")) for row in failures)),
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    return summary_rows, manifest


def write_summary_md(path: Path, *, manifest: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# HU T2 Stage9f Tail-Guard Replay",
        "",
        "This is selected high-MC / readiness evidence for tail-risk guard labels.",
        "It does not approve production, P2 fixed, 50k teacher, or T1.",
        "",
        f"- readiness only: `{manifest['readiness_only']}`",
        f"- targets: `{manifest['targets']}`",
        f"- readiness-ready: `{manifest['readiness_ready']}`",
        f"- replay successes: `{manifest['replay_successes']}`",
        f"- replay failures: `{manifest['replay_failures']}`",
        "",
        "## Breakdown",
        "",
        "| group | readiness | successes | failures | gain mean | min | max | high-MC losses | LCB95+ | sign flips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {group} | {ready} | {success} | {fail} | {mean:+.4f} | {min_gain:+.4f} | {max_gain:+.4f} | {losses} | {lcb} | {flips} |".format(
                group=row["target_group"],
                ready=safe_int(row["readiness_rows"], 0) or 0,
                success=safe_int(row["replay_successes"], 0) or 0,
                fail=safe_int(row["replay_failures"], 0) or 0,
                mean=safe_float(row["gain_mean"]),
                min_gain=safe_float(row["gain_min"]),
                max_gain=safe_float(row["gain_max"]),
                losses=safe_int(row["tail_loss_count"], 0) or 0,
                lcb=safe_int(row["lcb95_positive_count"], 0) or 0,
                flips=safe_int(row["sign_flip_count"], 0) or 0,
            )
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "- production/P2 fixed: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_targets = read_jsonl(args.targets)
    target_groups = {str(group) for group in args.target_group if str(group)}
    if target_groups:
        raw_targets = [row for row in raw_targets if str(row.get("target_group", "")) in target_groups]
    if args.offset < 0:
        raise SystemExit("--offset must be non-negative")
    if args.offset > 0:
        raw_targets = raw_targets[args.offset :]
    if args.limit > 0:
        raw_targets = raw_targets[: args.limit]
    prepared = [prepare_target(row) for row in raw_targets]
    readiness_rows = [readiness_row(target) for target in prepared]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "readiness.csv", readiness_rows)
    results: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    if not args.readiness_only:
        partial_path = args.output_dir / args.partial_results_name
        completed_ids = set()
        if args.resume:
            existing_rows = read_jsonl(partial_path)
            completed_ids = {str(row.get("target_id", "")) for row in existing_rows if row.get("target_id")}
        bundle = load_model_bundle(ModelPaths(), profiles={"stage7_m5_r10"})
        batched_config = build_batched_config(args)
        batched_cache = None if args.disable_continuation_cache else HuTurn3DecisionCache(args.continuation_cache_size)
        reference_cache = None if args.disable_continuation_cache else HuTurn3Stage3ReferenceCache(args.continuation_cache_size)
        state_feature_cache = Stage3StateFeatureCache()
        action_cache = HuTurn3ActionCache()
        final_turn_cache = None if args.disable_final_turn_cache else FinalTurnDecisionCache(args.final_turn_cache_size)
        for ordinal, target in enumerate(prepared, start=1):
            target_id = str(target.row.get("target_id", ""))
            if target_id and target_id in completed_ids:
                continue
            try:
                result = replay_target(
                    target,
                    args=args,
                    bundle=bundle,
                    batched_config=batched_config,
                    batched_cache=batched_cache,
                    batched_reference_cache=reference_cache,
                    batched_state_feature_cache=state_feature_cache,
                    batched_action_cache=action_cache,
                    final_turn_cache=final_turn_cache,
                )
            except Exception as exc:
                result = failure_row(target, str(exc), mc_samples=args.mc_samples)
            results.append(result)
            append_jsonl(partial_path, result)
            if args.progress_every > 0 and (ordinal % args.progress_every == 0 or ordinal == len(prepared)):
                write_json(
                    args.output_dir / "progress.json",
                    {
                        "targets": len(prepared),
                        "processed_this_run": len(results),
                        "ordinal": ordinal,
                        "resume_skipped": len(completed_ids),
                        "partial_results": str(partial_path),
                        "elapsed_seconds": time.perf_counter() - started_at,
                    },
                )
        if partial_path.exists():
            results = dedupe_rows_by_target_id(read_jsonl(partial_path))
    summary_rows, manifest = summarize_results(results, prepared, readiness_only=args.readiness_only)
    manifest.update(
        {
            "targets_path": str(args.targets),
            "output_dir": str(args.output_dir),
            "mc_samples": args.mc_samples,
            "offset": args.offset,
            "limit": args.limit,
            "target_groups": sorted(target_groups),
            "elapsed_seconds": time.perf_counter() - started_at,
        }
    )
    write_jsonl(args.output_dir / "replay_results.jsonl", results)
    write_csv(args.output_dir / "replay_results.csv", results)
    write_csv(args.output_dir / "replay_summary.csv", summary_rows)
    write_json(args.output_dir / "replay_manifest.json", manifest)
    write_summary_md(args.output_dir / "replay_summary.md", manifest=manifest, summary_rows=summary_rows)


if __name__ == "__main__":
    main()
