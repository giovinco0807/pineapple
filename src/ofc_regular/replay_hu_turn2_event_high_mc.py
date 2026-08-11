"""Replay selected HU Turn2 teacher events with higher Monte Carlo samples.

Gate C1c-R is evaluation-only. It replays event-level candidates from the
Gate C1b autopsy, records exact replay blockers, and, when requested, reruns
the original Turn2 teacher evaluation at a larger future sample count.
It never starts production training, T1 training, or a 50k teacher run.
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

from .ai_profiles import (
    DEFAULT_HU_TURN3_STAGE7_MODEL,
    DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL,
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
    ModelPaths,
    load_model_bundle,
)
from .final_turn_decision_cache import FinalTurnDecisionCache
from .hu_belief import sample_hidden_card_particles, turn2_actor_observation
from .hu_turn2_teacher_data import (
    DEFAULT_T3_CONTINUATION,
    T3ContinuationMode,
    _build_policy_for_profile,
    _build_t3_continuation_policy,
    _stage3_feature_replay_source,
    _stage3_feature_teacher_run_hash,
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
from .play_ai import _prediction_thread_context
from .state import Board


DEFAULT_CANDIDATES = Path(
    "outputs/hu_turn2_stage1_pilot_training_5000_c1b_false_positive_autopsy/high_mc_recheck_candidates.csv"
)
DEFAULT_CACHE_DIR = Path(
    "D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_gate_c1_5k_mc512"
)
DEFAULT_OUTPUT_DIR = Path("outputs/hu_turn2_stage1_pilot_training_5000_c1b_false_positive_autopsy")


@dataclass(frozen=True)
class ReplayEvent:
    candidate_row: dict[str, str]
    sample_row: dict[str, Any] | None
    candidate_id: str
    state_index: int
    sample_id: str
    source: str
    position: str
    baseline_local_index: int | None
    candidate_local_index: int | None
    teacher_local_index: int | None
    baseline_original_index: int | None
    candidate_original_index: int | None
    teacher_original_index: int | None
    replay_ready: bool
    failure_reason: str
    missing_fields: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mc-samples", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--seed-mode", choices=("reuse_original", "offset_by_mc"), default="reuse_original")
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--continuation-profile", default="current")
    parser.add_argument("--opponent-profile", default="current")
    parser.add_argument(
        "--t3-continuation",
        choices=("stage3_reference_default", "stage7_m5_r10"),
        default=DEFAULT_T3_CONTINUATION,
        help=(
            "T3 continuation used for replay rollouts. Hidden-discard default "
            "is stage3_reference_default; stage7_m5_r10 is legacy opt-in."
        ),
    )
    parser.add_argument("--opening-lookahead-samples", type=int, default=128)
    parser.add_argument("--disable-batched-continuation", action="store_true")
    parser.add_argument("--batched-continuation-batch-size", type=int, default=8192)
    parser.add_argument("--disable-continuation-cache", action="store_true")
    parser.add_argument("--continuation-cache-size", type=int, default=200_000)
    parser.add_argument("--disable-stage3-feature-fast-path", action="store_true")
    parser.add_argument("--stage3-feature-encoder-mode", default="rust_direct")
    parser.add_argument("--disable-final-turn-cache", action="store_true")
    parser.add_argument("--final-turn-cache-size", type=int, default=200_000)
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--hu-turn3-stage7-model", type=Path, default=DEFAULT_HU_TURN3_STAGE7_MODEL)
    parser.add_argument(
        "--hu-turn3-stage7-reference-model",
        type=Path,
        default=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL,
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def bool_int(value: bool) -> int:
    return 1 if value else 0


def candidate_id_for(row: dict[str, str], ordinal: int) -> str:
    existing = row.get("candidate_id") or row.get("event_id")
    if existing:
        return existing
    state_index = row.get("state_index", f"row{ordinal}")
    candidate = row.get("candidate_action_local_index", "na")
    baseline = row.get("baseline_action_local_index", "na")
    return f"state{state_index}_cand{candidate}_base{baseline}_row{ordinal}"


def load_cache_metadata(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"cache metadata not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_input_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root / path


def load_teacher_rows_for_state_indices(
    *,
    metadata: dict[str, Any],
    repo_root: Path,
    target_state_indices: set[int],
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    state_index = 0
    input_files = metadata.get("input_files") or [{"path": metadata.get("input", "")}]
    for entry in input_files:
        input_path = resolve_input_path(str(entry.get("path", "")), repo_root)
        if not input_path.exists():
            raise FileNotFoundError(f"teacher input file not found: {input_path}")
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if state_index in target_state_indices:
                    rows[state_index] = json.loads(line)
                state_index += 1
        if target_state_indices.issubset(rows.keys()):
            break
    return rows


def board_from_json(payload: Any, field_name: str, missing: list[str]) -> Board | None:
    if not isinstance(payload, dict):
        missing.append(field_name)
        return None
    try:
        return Board.from_rows(
            payload.get("top") or (),
            payload.get("middle") or (),
            payload.get("bottom") or (),
        )
    except Exception:
        missing.append(field_name)
        return None


def actions_by_original_index(actions: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for action in actions:
        original_index = safe_int(action.get("original_index"))
        if original_index is not None:
            output[original_index] = action
    return output


def action_at_local(actions: list[dict[str, Any]], local_index: int | None) -> dict[str, Any] | None:
    if local_index is None or local_index < 0 or local_index >= len(actions):
        return None
    return actions[local_index]


def local_to_original(actions: list[dict[str, Any]], local_index: int | None) -> int | None:
    action = action_at_local(actions, local_index)
    if action is None:
        return None
    return safe_int(action.get("original_index"))


def build_replay_events(
    candidates: list[dict[str, str]],
    sample_rows: dict[int, dict[str, Any]],
) -> list[ReplayEvent]:
    events: list[ReplayEvent] = []
    for ordinal, row in enumerate(candidates):
        missing: list[str] = []
        state_index = safe_int(row.get("state_index"))
        if state_index is None:
            state_index = -1
            missing.append("state_index")
        sample_row = sample_rows.get(state_index)
        if sample_row is None:
            missing.append("sample_row")
        actions = sample_row.get("actions") if isinstance(sample_row, dict) else None
        if not isinstance(actions, list) or not actions:
            actions = []
            missing.append("actions")

        baseline_local = safe_int(row.get("baseline_action_local_index"))
        candidate_local = safe_int(row.get("candidate_action_local_index"))
        teacher_local = safe_int(row.get("teacher_best_action_local_index"))
        if baseline_local is None:
            missing.append("baseline_action_local_index")
        if candidate_local is None:
            missing.append("candidate_action_local_index")
        if teacher_local is None:
            missing.append("teacher_best_action_local_index")

        baseline_original = local_to_original(actions, baseline_local)
        candidate_original = local_to_original(actions, candidate_local)
        teacher_original = local_to_original(actions, teacher_local)
        if baseline_original is None:
            missing.append("baseline_original_index")
        if candidate_original is None:
            missing.append("candidate_original_index")
        if teacher_original is None:
            missing.append("teacher_original_index")

        if isinstance(sample_row, dict):
            for field in ("board", "opponent_board", "future_rollout_seed"):
                if field not in sample_row:
                    missing.append(field)
            if "dealt" not in sample_row and "dealt_cards" not in sample_row:
                missing.append("dealt")
            if not sample_row.get("hero_private_discards") and not sample_row.get(
                "visible_dead_cards"
            ):
                missing.append("hero_visible_discard")

        unique_missing = tuple(sorted(set(missing)))
        failure_reason = "ready" if not unique_missing else "missing_" + "|".join(unique_missing)
        events.append(
            ReplayEvent(
                candidate_row=row,
                sample_row=sample_row,
                candidate_id=candidate_id_for(row, ordinal),
                state_index=state_index,
                sample_id=row.get("sample_id") or (str(sample_row.get("sample_id")) if sample_row else ""),
                source=row.get("source_group") or row.get("run_bucket") or row.get("source") or "",
                position=row.get("seat") or row.get("position") or (str(sample_row.get("seat")) if sample_row else ""),
                baseline_local_index=baseline_local,
                candidate_local_index=candidate_local,
                teacher_local_index=teacher_local,
                baseline_original_index=baseline_original,
                candidate_original_index=candidate_original,
                teacher_original_index=teacher_original,
                replay_ready=not unique_missing,
                failure_reason=failure_reason,
                missing_fields=unique_missing,
            )
        )
    return events


def readiness_rows(events: list[ReplayEvent]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        actions = event.sample_row.get("actions") if event.sample_row else []
        rows.append(
            {
                "candidate_id": event.candidate_id,
                "priority": event.candidate_row.get("priority", ""),
                "reason": event.candidate_row.get("reason", ""),
                "state_index": event.state_index,
                "sample_id": event.sample_id,
                "source": event.source,
                "position": event.position,
                "replay_ready": bool_int(event.replay_ready),
                "failure_reason": event.failure_reason,
                "missing_fields": "|".join(event.missing_fields),
                "candidate_local_index": event.candidate_local_index,
                "baseline_local_index": event.baseline_local_index,
                "teacher_local_index": event.teacher_local_index,
                "candidate_original_index": event.candidate_original_index,
                "baseline_original_index": event.baseline_original_index,
                "teacher_original_index": event.teacher_original_index,
                "action_count": len(actions) if isinstance(actions, list) else 0,
                "future_rollout_seed": event.sample_row.get("future_rollout_seed", "") if event.sample_row else "",
                "original_mc_samples": event.sample_row.get("rollout_count", "") if event.sample_row else "",
            }
        )
    return rows


def missing_field_rows(events: list[ReplayEvent]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        for field in event.missing_fields:
            rows.append(
                {
                    "candidate_id": event.candidate_id,
                    "state_index": event.state_index,
                    "sample_id": event.sample_id,
                    "source": event.source,
                    "position": event.position,
                    "field": field,
                    "failure_reason": event.failure_reason,
                }
            )
    return rows


def stderr_for_delta(candidate_action: dict[str, Any], baseline_action: dict[str, Any]) -> float:
    candidate_se = safe_float(
        candidate_action.get("ev_standard_error", candidate_action.get("standard_error", 0.0))
    )
    baseline_se = safe_float(
        baseline_action.get("ev_standard_error", baseline_action.get("standard_error", 0.0))
    )
    return math.sqrt(candidate_se * candidate_se + baseline_se * baseline_se)


def rank_of_original(actions: list[dict[str, Any]], original_index: int | None) -> int | None:
    if original_index is None:
        return None
    for rank, action in enumerate(actions, start=1):
        if safe_int(action.get("original_index")) == original_index:
            return rank
    return None


def sign_label(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def replay_seed(event: ReplayEvent, mc_samples: int, seed_mode: str) -> int:
    original = safe_int(event.sample_row.get("future_rollout_seed") if event.sample_row else None)
    if original is None:
        original = 0
    if seed_mode == "offset_by_mc":
        return original + mc_samples
    return original


def build_batched_config(args: argparse.Namespace) -> HuTurn3Stage7BatchConfig:
    replay_source = _stage3_feature_replay_source(args.mc_samples, 0)
    teacher_run_hash = _stage3_feature_teacher_run_hash(
        seed=0,
        samples=0,
        future_samples=args.mc_samples,
        source_bucket="event_replay",
        stage3_feature_encoder_mode=args.stage3_feature_encoder_mode,
        disable_stage3_feature_fast_path=args.disable_stage3_feature_fast_path,
        t3_continuation=args.t3_continuation,
    )
    stage7_enabled = args.t3_continuation == "stage7_m5_r10"
    return HuTurn3Stage7BatchConfig(
        stage7_enabled=stage7_enabled,
        hu_turn3_min_margin=5.0 if stage7_enabled else 0.0,
        hu_turn3_reference_min_margin=10.0 if stage7_enabled else 0.0,
        batch_size=args.batched_continuation_batch_size,
        use_cache=not args.disable_continuation_cache,
        use_stage3_feature_fast_path=not args.disable_stage3_feature_fast_path,
        stage3_feature_encoder_mode=args.stage3_feature_encoder_mode,
        stage3_feature_replay_model_path=str(args.hu_turn3_stage7_reference_model),
        stage3_feature_replay_stage7_model_path=str(args.hu_turn3_stage7_model),
        stage3_feature_replay_source=replay_source,
        stage3_feature_replay_teacher_run_hash=teacher_run_hash,
    )


def build_replay_policy(
    profile: str,
    bundle: object,
    *,
    args: argparse.Namespace,
    seed: int,
    seat: str,
) -> object:
    if profile == "random_exact_final":
        return _build_policy_for_profile(
            profile,
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=args.opening_lookahead_samples,
        )
    return _build_t3_continuation_policy(
        args.t3_continuation,
        bundle,
        seed=seed,
        seat=seat,
        opening_lookahead_samples=args.opening_lookahead_samples,
    )


def board_payloads(
    event: ReplayEvent,
) -> tuple[Board, Board, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    assert event.sample_row is not None
    missing: list[str] = []
    board = board_from_json(event.sample_row.get("board"), "board", missing)
    opponent = board_from_json(event.sample_row.get("opponent_board"), "opponent_board", missing)
    dealt = tuple(event.sample_row.get("dealt") or event.sample_row.get("dealt_cards") or ())
    visible_dead = tuple(event.sample_row.get("visible_dead_cards") or ())
    hero_private = tuple(event.sample_row.get("hero_private_discards") or ())
    if board is None or opponent is None or len(dealt) != 3:
        raise ValueError(f"invalid replay state: {','.join(missing) or 'dealt'}")
    return board, opponent, dealt, visible_dead, hero_private


def replay_event(
    event: ReplayEvent,
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
    if not event.replay_ready:
        raise ValueError(event.failure_reason)
    board, opponent_board, dealt, visible_dead, hero_private = board_payloads(event)
    seed = replay_seed(event, args.mc_samples, args.seed_mode)
    hero_seat = event.position if event.position in {"first", "second"} else str(event.sample_row.get("seat", "first"))
    observation = turn2_actor_observation(
        hero_board=board,
        opponent_public_board=opponent_board,
        dealt_cards=dealt,
        hero_seat=hero_seat,
        hero_private_discards=hero_private,
        visible_dead_cards=visible_dead or None,
    )
    belief_batch = sample_hidden_card_particles(
        observation,
        base_seed=seed,
        run_id=(
            "replay_hu_turn2_event_high_mc"
            f"|state={event.state_index}|sample={event.sample_id}"
        ),
        sample_count=args.mc_samples,
    )
    hero_policy = build_replay_policy(
        args.continuation_profile,
        bundle,
        args=args,
        seed=seed * 4 + 2,
        seat=hero_seat,
    )
    opponent_seat = "second" if hero_seat == "first" else "first"
    opponent_policy = build_replay_policy(
        args.opponent_profile,
        bundle,
        args=args,
        seed=seed * 4 + 3,
        seat=opponent_seat,
    )
    started_at = time.perf_counter()
    sample = evaluate_hu_turn2_actions(
        board=board,
        dealt_cards=dealt,
        opponent_board=opponent_board,
        hero_seat=hero_seat,
        continuation_policy=hero_policy,
        opponent_policy=opponent_policy,
        baseline_turn2_model=bundle.turn2,
        future_samples=args.mc_samples,
        future_rollout_seed=seed,
        use_batched_continuation=not args.disable_batched_continuation,
        batched_continuation_config=batched_config,
        batched_continuation_cache=batched_cache,
        batched_reference_cache=batched_reference_cache,
        batched_state_feature_cache=batched_state_feature_cache,
        batched_action_cache=batched_action_cache,
        batched_continuation_batch_size=args.batched_continuation_batch_size,
        final_turn_cache=final_turn_cache,
        use_final_turn_cache=final_turn_cache is not None,
        continuation_policy_name=_t3_continuation_policy_name(args.t3_continuation),
        continuation_metadata=_t3_continuation_metadata(args.t3_continuation),
        observation=observation,
        belief_batch=belief_batch,
    )
    elapsed = time.perf_counter() - started_at
    if sample is None:
        raise ValueError("evaluate_returned_none")
    actions = sample.get("actions") or []
    by_original = actions_by_original_index(actions)
    baseline_action = by_original.get(int(event.baseline_original_index))
    candidate_action = by_original.get(int(event.candidate_original_index))
    teacher_action = by_original.get(int(event.teacher_original_index))
    if baseline_action is None:
        raise ValueError("baseline_action_missing_after_replay")
    if candidate_action is None:
        raise ValueError("candidate_action_missing_after_replay")
    if teacher_action is None:
        raise ValueError("teacher_action_missing_after_replay")

    baseline_ev = safe_float(baseline_action.get("score", baseline_action.get("ev")))
    candidate_ev = safe_float(candidate_action.get("score", candidate_action.get("ev")))
    teacher_ev = safe_float(teacher_action.get("score", teacher_action.get("ev")))
    gain = candidate_ev - baseline_ev
    stderr = stderr_for_delta(candidate_action, baseline_action)
    current_gain = safe_float(event.candidate_row.get("teacher_gain_mean_current_mc512"))
    current_sign = sign_label(current_gain)
    high_sign = sign_label(gain)
    best_action = actions[0] if actions else {}
    best_original = safe_int(best_action.get("original_index"))
    lower90 = gain - 1.64 * stderr
    lower95 = gain - 1.96 * stderr
    return {
        "candidate_id": event.candidate_id,
        "sample_id": event.sample_id,
        "state_index": event.state_index,
        "source": event.source,
        "position": event.position,
        "baseline_action": event.candidate_row.get("baseline_action_text", ""),
        "override_action": event.candidate_row.get("candidate_action_text", ""),
        "teacher_action": event.candidate_row.get("teacher_best_action_text", ""),
        "baseline_original_index": event.baseline_original_index,
        "candidate_original_index": event.candidate_original_index,
        "teacher_original_index": event.teacher_original_index,
        "mc_n": args.mc_samples,
        "t3_continuation": args.t3_continuation,
        "t3_continuation_policy": _t3_continuation_policy_name(args.t3_continuation),
        "gain_mean": gain,
        "gain_stderr": stderr,
        "lower_bound_90": lower90,
        "lower_bound_95": lower95,
        "current_mc512_gain_mean": current_gain,
        "current_mc512_gain_stderr_proxy": event.candidate_row.get("teacher_gain_stderr_proxy_current_mc512", ""),
        "current_mc512_sign": current_sign,
        "high_mc_sign": high_sign,
        "sign_flip_vs_mc512": bool_int(current_sign != high_sign),
        "false_positive_after_high_mc": bool_int(gain < 0.0),
        "still_positive_after_high_mc": bool_int(gain > 0.0),
        "still_negative_after_high_mc": bool_int(gain < 0.0),
        "lower_bound_90_positive": bool_int(lower90 > 0.0),
        "lower_bound_95_positive": bool_int(lower95 > 0.0),
        "baseline_ev": baseline_ev,
        "candidate_ev": candidate_ev,
        "teacher_ev": teacher_ev,
        "new_best_original_index": best_original,
        "new_best_action_score": safe_float(best_action.get("score", best_action.get("ev", 0.0))),
        "candidate_rank_high_mc": rank_of_original(actions, event.candidate_original_index),
        "baseline_rank_high_mc": rank_of_original(actions, event.baseline_original_index),
        "teacher_rank_high_mc": rank_of_original(actions, event.teacher_original_index),
        "replay_status": "success",
        "failure_reason": "",
        "elapsed_seconds": elapsed,
        "future_rollout_seed": seed,
        "common_random_future_digest": sample.get("common_random_future_digest", ""),
        "belief_conditioned": bool_int(bool(sample.get("belief_conditioned"))),
        "observation_fingerprint": sample.get("observation_fingerprint", ""),
        "belief_batch_digest": sample.get("belief_batch_digest", ""),
        "stage7_t3_eval_count": candidate_action.get("stage7_t3_eval_count", ""),
        "stage7_t3_fired_count": candidate_action.get("stage7_t3_fired_count", ""),
        "stage7_t3_override_rate": candidate_action.get("stage7_t3_override_rate", ""),
    }


def failure_row(event: ReplayEvent, reason: str, *, mc_samples: int) -> dict[str, Any]:
    return {
        "candidate_id": event.candidate_id,
        "sample_id": event.sample_id,
        "state_index": event.state_index,
        "source": event.source,
        "position": event.position,
        "baseline_action": event.candidate_row.get("baseline_action_text", ""),
        "override_action": event.candidate_row.get("candidate_action_text", ""),
        "teacher_action": event.candidate_row.get("teacher_best_action_text", ""),
        "mc_n": mc_samples,
        "gain_mean": "",
        "gain_stderr": "",
        "lower_bound_90": "",
        "lower_bound_95": "",
        "sign_flip_vs_mc512": "",
        "false_positive_after_high_mc": "",
        "replay_status": "failed",
        "failure_reason": reason or "unknown",
    }


def write_replay_blocker_summary(path: Path, events: list[ReplayEvent], *, readiness_only: bool) -> None:
    reason_counts = Counter(event.failure_reason for event in events if not event.replay_ready)
    ready_count = sum(1 for event in events if event.replay_ready)
    unknown_failures = sum(1 for event in events if (not event.replay_ready and not event.failure_reason))
    lines = [
        "# Gate C1c-R Event Replay Readiness",
        "",
        f"- candidates: `{len(events)}`",
        f"- replay-ready events: `{ready_count}`",
        f"- blocked events: `{len(events) - ready_count}`",
        f"- unknown failure reasons: `{unknown_failures}`",
        f"- mode: `{'readiness_only' if readiness_only else 'high_mc_replay'}`",
        "",
        "## Blockers",
        "",
    ]
    if reason_counts:
        for reason, count in reason_counts.most_common():
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Training Gates",
            "",
            "- production training: `No-Go`",
            "- T1 training: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- C2-small: `No-Go until high-MC replay evidence is reviewed`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_high_mc_summary(
    path: Path,
    *,
    events: list[ReplayEvent],
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    mc_samples: int,
    t3_continuation: T3ContinuationMode,
    readiness_only: bool,
    elapsed_seconds: float,
) -> None:
    successes = [row for row in results if row.get("replay_status") == "success"]
    source_position: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in successes:
        key = (str(row.get("source", "")), str(row.get("position", "")))
        source_position[key]["events"] += 1
        if int(row.get("false_positive_after_high_mc", 0) or 0):
            source_position[key]["false_positive_after_high_mc"] += 1
        if int(row.get("lower_bound_95_positive", 0) or 0):
            source_position[key]["lower_bound_95_positive"] += 1

    false_positive_count = sum(int(row.get("false_positive_after_high_mc", 0) or 0) for row in successes)
    positive_count = sum(int(row.get("still_positive_after_high_mc", 0) or 0) for row in successes)
    negative_count = sum(int(row.get("still_negative_after_high_mc", 0) or 0) for row in successes)
    sign_flip_count = sum(int(row.get("sign_flip_vs_mc512", 0) or 0) for row in successes)
    lcb90_count = sum(int(row.get("lower_bound_90_positive", 0) or 0) for row in successes)
    lcb95_count = sum(int(row.get("lower_bound_95_positive", 0) or 0) for row in successes)
    unknown_failures = sum(1 for row in failures if not row.get("failure_reason") or row.get("failure_reason") == "unknown")
    c2_status = "No-Go"
    if successes and false_positive_count == 0 and lcb95_count >= 30:
        c2_status = "Conditional Go"
    elif successes and false_positive_count == 0 and lcb95_count > 0:
        c2_status = "Conditional Go for a narrower heldout smoke only"

    lines = [
        "# Gate C1c-R High-MC Event Replay Summary",
        "",
        f"- mode: `{'readiness_only' if readiness_only else 'high_mc_replay'}`",
        f"- requested MC samples: `{mc_samples}`",
        f"- T3 continuation: `{_t3_continuation_policy_name(t3_continuation)}`",
        f"- t3_continuation: `{t3_continuation}`",
        f"- candidates: `{len(events)}`",
        f"- replay successes: `{len(successes)}`",
        f"- replay failures: `{len(failures)}`",
        f"- unknown failure reasons: `{unknown_failures}`",
        f"- elapsed seconds: `{elapsed_seconds:.2f}`",
        "",
        "## High-MC Outcomes",
        "",
        f"- false positives after high-MC: `{false_positive_count}`",
        f"- still positive: `{positive_count}`",
        f"- still negative: `{negative_count}`",
        f"- sign flips vs MC512: `{sign_flip_count}`",
        f"- lower_bound_90 > 0: `{lcb90_count}`",
        f"- lower_bound_95 > 0: `{lcb95_count}`",
        "",
        "## Source / Position",
        "",
    ]
    if source_position:
        lines.append("| source | position | events | false_positive_after_high_mc | lower_bound_95_positive |")
        lines.append("|---|---:|---:|---:|---:|")
        for (source, position), counts in sorted(source_position.items()):
            lines.append(
                f"| {source} | {position} | {counts['events']} | "
                f"{counts['false_positive_after_high_mc']} | {counts['lower_bound_95_positive']} |"
            )
    else:
        lines.append("- no successful high-MC replay rows")
    lines.extend(
        [
            "",
            "## Gate Decisions",
            "",
            "- production training: `No-Go`",
            "- T1 training: `No-Go`",
            "- 50k teacher: `No-Go`",
            f"- C2-small heldout: `{c2_status}`",
            "",
            "C2-small remains gated on fixed thresholds and heldout rerun; this replay does not authorize production or 50k teacher training.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    repo_root = Path.cwd()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_csv(args.candidates)
    if args.limit > 0:
        candidates = candidates[: args.limit]
    metadata = load_cache_metadata(args.cache_dir)
    target_state_indices = {
        state_index
        for row in candidates
        if (state_index := safe_int(row.get("state_index"))) is not None
    }
    sample_rows = load_teacher_rows_for_state_indices(
        metadata=metadata,
        repo_root=repo_root,
        target_state_indices=target_state_indices,
    )
    events = build_replay_events(candidates, sample_rows)
    write_csv(args.output_dir / "event_replay_readiness.csv", readiness_rows(events))
    write_csv(args.output_dir / "missing_fields_by_event.csv", missing_field_rows(events))
    write_replay_blocker_summary(
        args.output_dir / "replay_blocker_summary.md",
        events,
        readiness_only=args.readiness_only,
    )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for event in events:
        if not event.replay_ready:
            failures.append(failure_row(event, event.failure_reason, mc_samples=args.mc_samples))

    if not args.readiness_only:
        paths = ModelPaths(
            opening=args.opening_model,
            turn1=args.turn1_model,
            turn2=args.turn2_model,
            turn3=args.turn3_model,
            hu_turn3_stage7=args.hu_turn3_stage7_model,
            hu_turn3_stage7_reference=args.hu_turn3_stage7_reference_model,
        )
        bundle = load_model_bundle(paths, {"current"})
        batched_config = build_batched_config(args)
        batched_cache = (
            None
            if args.disable_continuation_cache
            else HuTurn3DecisionCache(max_size=args.continuation_cache_size)
        )
        batched_reference_cache = (
            None
            if args.disable_continuation_cache
            else HuTurn3Stage3ReferenceCache(max_size=args.continuation_cache_size)
        )
        batched_state_feature_cache = (
            None
            if args.disable_continuation_cache
            else Stage3StateFeatureCache(max_size=args.continuation_cache_size)
        )
        batched_action_cache = (
            None
            if args.disable_continuation_cache
            else HuTurn3ActionCache(max_size=args.continuation_cache_size)
        )
        final_turn_cache = (
            None
            if args.disable_final_turn_cache
            else FinalTurnDecisionCache(max_size=args.final_turn_cache_size)
        )
        with _prediction_thread_context(args.prediction_threads):
            for event in events:
                if not event.replay_ready:
                    continue
                try:
                    results.append(
                        replay_event(
                            event,
                            args=args,
                            bundle=bundle,
                            batched_config=batched_config,
                            batched_cache=batched_cache,
                            batched_reference_cache=batched_reference_cache,
                            batched_state_feature_cache=batched_state_feature_cache,
                            batched_action_cache=batched_action_cache,
                            final_turn_cache=final_turn_cache,
                        )
                    )
                except Exception as exc:
                    failures.append(failure_row(event, str(exc) or "unknown", mc_samples=args.mc_samples))

    result_path = args.output_dir / f"high_mc_recheck_results_mc{args.mc_samples}.csv"
    write_csv(result_path, results)
    write_csv(args.output_dir / "event_replay_failures.csv", failures)
    write_high_mc_summary(
        args.output_dir / "high_mc_recheck_summary.md",
        events=events,
        results=results,
        failures=failures,
        mc_samples=args.mc_samples,
        t3_continuation=args.t3_continuation,
        readiness_only=args.readiness_only,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
