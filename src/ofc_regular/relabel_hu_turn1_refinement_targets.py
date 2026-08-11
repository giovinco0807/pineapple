"""Relabel selected HU Turn1 refinement targets with a stronger continuation."""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import ActionKey, resolve_action_index
from .action_space import generate_turn_actions
from .ai_profiles import ModelPaths, build_policy, load_model_bundle, required_profiles
from .cards import ALL_CARDS, validate_cards
from .evaluate_matchups import PROFILE_CHOICES, board_to_json
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation, actor_observation_from_record
from .hu_turn1_teacher_pilot import CANDIDATE_UNION_MODES, evaluate_turn1_action_subset
from .hu_turn3_model import load_hu_action_value_model
from .state import Board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="stage9f_p2")
    parser.add_argument("--opponent-profile", choices=PROFILE_CHOICES, default="stage9f_p2")
    parser.add_argument("--future-samples", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=0)
    parser.add_argument(
        "--source-actions-only",
        action="store_true",
        help="Evaluate only action_index values already present in each target actions list.",
    )
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--candidate-models", type=Path, nargs="+")
    parser.add_argument("--candidate-topk", type=int, default=0)
    parser.add_argument("--candidate-union-cap", type=int, default=0)
    parser.add_argument("--candidate-union-mode", choices=CANDIDATE_UNION_MODES, default="min_rank")
    parser.add_argument("--skip-targets", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026062602)
    parser.add_argument("--opening-lookahead-samples", type=int, default=1)
    parser.add_argument(
        "--target-timeout-seconds",
        type=float,
        default=0.0,
        help="Per-target wall-clock timeout. Uses SIGALRM where available; 0 disables.",
    )
    parser.add_argument(
        "--skip-output",
        type=Path,
        default=None,
        help="Optional JSONL path for timed-out or skipped target records.",
    )
    parser.add_argument(
        "--continue-on-target-error",
        action="store_true",
        help="Skip failed targets instead of failing the whole relabel job.",
    )
    return parser.parse_args()


class TargetTimeoutError(TimeoutError):
    pass


def target_timeout_supported() -> bool:
    return all(hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "setitimer"))


@contextlib.contextmanager
def target_timeout(seconds: float):
    if seconds <= 0 or not target_timeout_supported():
        yield
        return

    def _handle_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise TargetTimeoutError(f"target exceeded {seconds:.3f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
        signal.signal(signal.SIGALRM, previous_handler)


def read_jsonl(path: Path, *, skip_targets: int = 0, max_targets: int = 0) -> list[dict[str, Any]]:
    if skip_targets < 0:
        raise ValueError("skip_targets must be non-negative")
    rows: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            if seen < skip_targets:
                seen += 1
                continue
            rows.append(json.loads(line))
            seen += 1
            if max_targets > 0 and len(rows) >= max_targets:
                break
    return rows


def board_from_json(payload: dict[str, Sequence[str]]) -> Board:
    return Board.from_rows(
        payload.get("top", ()),
        payload.get("middle", ()),
        payload.get("bottom", ()),
    )


def private_discards_from_record(record: dict[str, Any]) -> list[list[str]]:
    player = player_from_record(record)
    hero = list(record.get("hero_private_discards", ()) or ())
    opponent = list(record.get("opponent_private_discards", ()) or ())
    if player == 0:
        return [hero, opponent]
    return [opponent, hero]


def player_from_record(record: dict[str, Any]) -> int:
    raw_player = record.get("player")
    if raw_player is not None:
        return int(raw_player)
    return 1 if record.get("seat") == "second" else 0


def dealt_from_record(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(record.get("dealt") or record.get("cards_to_place") or ())


def actor_observation_for_record(record: dict[str, Any]) -> ActorObservation:
    """Build the relabel root exclusively from actor-visible record fields."""
    if "visible_dead_cards" not in record:
        raise ValueError("T1 relabel requires explicit visible_dead_cards")
    observation = actor_observation_from_record(record)
    nested = record.get("policy_observation")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("policy_observation must be a mapping")
        declared = ActorObservation.from_dict(nested)
        if declared != observation:
            raise ValueError("policy_observation disagrees with relabel target")
        observation = declared
    if observation.street != "T1":
        raise ValueError("T1 relabel requires a T1 actor observation")
    return observation


def remaining_cards_for_record(
    *,
    board: Board,
    opponent_board: Board,
    dealt: Sequence[str],
    private_discards: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    used_cards = [
        *board.all_cards(),
        *opponent_board.all_cards(),
        *dealt,
        *private_discards[0],
        *private_discards[1],
    ]
    validate_cards(used_cards)
    used = set(used_cards)
    return tuple(card for card in ALL_CARDS if card not in used)


def action_signature(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    placements = tuple((str(card), str(row)) for card, row in action.get("placements", ()))
    discards = tuple(str(card) for card in action.get("discards", ()))
    return json.dumps({"placements": placements, "discards": discards}, sort_keys=True, separators=(",", ":"))


def source_action_indices(
    record: dict[str, Any], legal_actions: Sequence[Any] | None = None
) -> list[int]:
    if legal_actions is not None:
        resolved: list[int] = []
        seen_resolved: set[int] = set()
        candidates: list[tuple[Any, Any, Any]] = [
            (
                record.get("runtime_candidate_action_key"),
                record.get("runtime_candidate_action"),
                record.get("runtime_candidate_action_index"),
            ),
            (
                record.get("runtime_baseline_action_key"),
                record.get("runtime_baseline_action"),
                record.get("runtime_baseline_action_index"),
            ),
        ]
        for action in record.get("actions", ()) or ():
            if not isinstance(action, dict):
                continue
            candidates.append(
                (
                    action.get("canonical_action_key", action.get("action_key")),
                    action.get("action") if isinstance(action.get("action"), dict) else action,
                    action.get("action_index", action.get("original_index")),
                )
            )
        order_digest = record.get("legal_action_order_digest")
        for raw_key, payload, legacy_index in candidates:
            if raw_key is None and payload is None and legacy_index is None:
                continue
            key = None
            if isinstance(raw_key, str):
                key = ActionKey.from_token(raw_key)
            payload_mapping = payload if isinstance(payload, Mapping) else None
            resolution = resolve_action_index(
                legal_actions,
                key=key,
                payload=payload_mapping,
                legacy_index=legacy_index,
                expected_order_digest=order_digest,
            )
            if resolution.index not in seen_resolved:
                seen_resolved.add(resolution.index)
                resolved.append(resolution.index)
        return resolved

    indices: list[int] = []
    seen: set[int] = set()
    raw_indices: list[Any] = [
        record.get("runtime_candidate_action_index"),
        record.get("runtime_baseline_action_index"),
    ]
    for action in record.get("actions", ()) or ():
        if isinstance(action, dict):
            raw_indices.append(action.get("action_index", action.get("original_index")))
    for raw_index in raw_indices:
        if raw_index is None:
            continue
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if index < 0 or index in seen:
            continue
        seen.add(index)
        indices.append(index)
    return indices


def compare_source_to_relabel(source: dict[str, Any], relabeled_actions: list[dict[str, Any]]) -> dict[str, Any]:
    source_actions = source.get("actions", ()) or ()
    source_best_index = int(source.get("best_action", 0) or 0)
    source_best = source_actions[source_best_index] if 0 <= source_best_index < len(source_actions) else None
    source_best_signature = action_signature(source_best)
    relabeled_by_signature = {
        action_signature(action): (rank, action)
        for rank, action in enumerate(relabeled_actions, start=1)
    }
    source_by_signature = {
        action_signature(action): (rank, action)
        for rank, action in enumerate(source_actions, start=1)
    }
    new_best = relabeled_actions[0] if relabeled_actions else None
    new_best_signature = action_signature(new_best)
    relabeled_best_score = float(new_best.get("score", 0.0)) if isinstance(new_best, dict) else 0.0
    source_best_new_rank = 0
    source_best_new_regret = 0.0
    if source_best_signature in relabeled_by_signature:
        source_best_new_rank, action = relabeled_by_signature[source_best_signature]
        source_best_new_regret = relabeled_best_score - float(action.get("score", 0.0))
    new_best_source_rank = 0
    if new_best_signature in source_by_signature:
        new_best_source_rank, _ = source_by_signature[new_best_signature]
    return {
        "source_best_signature": source_best_signature,
        "new_best_signature": new_best_signature,
        "best_action_changed": bool(source_best_signature and new_best_signature and source_best_signature != new_best_signature),
        "source_best_new_rank": int(source_best_new_rank),
        "source_best_new_regret": float(source_best_new_regret),
        "new_best_source_rank": int(new_best_source_rank),
    }


def skipped_target_record(
    record: dict[str, Any],
    *,
    reason: str,
    message: str,
    profile: str,
    opponent_profile: str,
    future_samples: int,
    max_actions: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": "hu_turn1_stage1_refinement_relabel_skip_v1",
        "target_id": record.get("target_id", record.get("sample_id")),
        "sample_id": record.get("sample_id"),
        "hand_seed": record.get("hand_seed"),
        "player": record.get("player"),
        "seat": record.get("seat"),
        "source_bucket": record.get("source_bucket"),
        "source_bucket_group": record.get("source_bucket_group"),
        "selection_reasons": list(record.get("selection_reasons", ()) or ()),
        "skip_reason": reason,
        "skip_message": message,
        "profile": profile,
        "opponent_profile": opponent_profile,
        "future_samples": future_samples,
        "max_actions": max_actions,
        "target_timeout_seconds": timeout_seconds,
    }


def build_policies(profile: str, opponent_profile: str, seed: int, opening_lookahead_samples: int) -> list[Any]:
    profiles = required_profiles(profile, opponent_profile)
    bundle = load_model_bundle(ModelPaths(), profiles)
    return [
        build_policy(profile, bundle, seed=seed, seat="first", opening_lookahead_samples=opening_lookahead_samples),
        build_policy(
            opponent_profile,
            bundle,
            seed=seed + 1,
            seat="second",
            opening_lookahead_samples=opening_lookahead_samples,
        ),
    ]


def relabel_record(
    record: dict[str, Any],
    *,
    policies: Sequence[Any],
    profile: str,
    opponent_profile: str,
    future_samples: int,
    max_actions: int,
    candidate_model: Any | None,
    candidate_models: Sequence[Any],
    candidate_topk: int,
    candidate_union_cap: int,
    candidate_union_mode: str,
    source_actions_only: bool,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    board = board_from_json(record["board"])
    opponent_board = board_from_json(record["opponent_board"])
    dealt = dealt_from_record(record)
    player = player_from_record(record)
    observation = actor_observation_for_record(record)
    if (
        observation.hero_board != board
        or observation.opponent_public_board != opponent_board
        or observation.dealt_cards != dealt
        or observation.seat != ("first" if player == 0 else "second")
    ):
        raise ValueError("actor observation disagrees with relabel target identity")
    target_id = int(record.get("target_id", record.get("sample_id", 0)) or 0)
    belief_seed = seed + target_id * 1_000_003 + int(
        record.get("hand_seed", 0) or 0
    )
    belief_batch = sample_hidden_card_particles(
        observation,
        base_seed=belief_seed,
        run_id="hu_turn1_refinement_relabel",
        sample_count=future_samples,
    )
    rng = random.Random(belief_seed)
    profile_stats: dict[str, Any] = {}
    legal_actions = generate_turn_actions(board, dealt)
    action_indices = (
        source_action_indices(record, legal_actions) if source_actions_only else None
    )
    if source_actions_only and not action_indices:
        raise ValueError("source-actions-only requested but target has no valid source action indices")
    started = time.perf_counter()
    action_result = evaluate_turn1_action_subset(
        board=board,
        opponent_board=opponent_board,
        dealt=dealt,
        hero_player=player,
        policies=policies,
        hand_seed=int(record.get("hand_seed", seed) or seed),
        sample_id=target_id,
        future_samples=future_samples,
        action_indices=action_indices,
        max_actions=max_actions,
        candidate_model=candidate_model,
        candidate_models=candidate_models,
        candidate_topk=candidate_topk,
        candidate_union_cap=candidate_union_cap,
        candidate_union_mode=candidate_union_mode,
        rng=rng,
        profile=profile_stats,
        observation=observation,
        belief_batch=belief_batch,
    )
    actions = list(action_result["actions"])
    truncated = bool(action_result["actions_truncated"])
    elapsed = time.perf_counter() - started
    best = actions[0] if actions else {}
    second = actions[1] if len(actions) > 1 else best
    score_gap = float(best.get("score", 0.0)) - float(second.get("score", 0.0))
    comparison = compare_source_to_relabel(record, actions)
    relabeled = {
        **record,
        "schema": "hu_turn1_stage1_refinement_relabel_v1",
        "source_schema": record.get("schema"),
        "source_profile": record.get("profile"),
        "source_opponent_profile": record.get("opponent_profile"),
        "source_t2_continuation_profile": record.get("t2_continuation_profile"),
        "profile": profile,
        "opponent_profile": opponent_profile,
        "t2_continuation_profile": profile,
        "t3_continuation": "stage7_m5_r10",
        "future_samples": future_samples,
        "max_actions": max_actions,
        "candidate_topk": candidate_topk,
        "candidate_union_cap": candidate_union_cap,
        "candidate_union_mode": candidate_union_mode,
        "source_actions_only": source_actions_only,
        "source_action_indices": action_indices or [],
        "policy_observation": observation.to_dict(),
        "visible_dead_cards": list(observation.legacy_dead_cards()),
        "belief_conditioned": True,
        "teacher_conditioning": "actor_observation_belief_v1",
        "replay_truth_audit_only": True,
        "replay_truth_used_for_label": False,
        "observation_fingerprint": observation.fingerprint(),
        "belief_base_seed": belief_seed,
        "belief_run_id": belief_batch.run_id,
        "belief_start_index": belief_batch.start_index,
        "belief_sample_count": future_samples,
        "belief_batch_digest": action_result.get(
            "belief_batch_digest", belief_batch.digest()
        ),
        "belief_schema": action_result.get(
            "belief_schema", belief_batch.to_dict()["belief_schema"]
        ),
        "belief_prior": action_result.get("belief_prior", belief_batch.prior),
        "action_count": len(actions),
        "total_legal_actions": int(action_result.get("total_legal_actions", len(actions))),
        "evaluated_action_count": int(action_result.get("evaluated_action_count", len(actions))),
        "actions_truncated": truncated,
        "best_action": 0,
        "score_gap": score_gap,
        "actions": actions,
        "relabel_compare": comparison,
        "relabel_seconds": elapsed,
    }
    if "candidate_selector" in action_result:
        relabeled["candidate_selector"] = action_result["candidate_selector"]
    elif "candidate_selector" in relabeled:
        del relabeled["candidate_selector"]
    return relabeled, profile_stats


def summarize(
    rows: list[dict[str, Any]],
    profile_stats: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    changed = sum(1 for row in rows if row.get("relabel_compare", {}).get("best_action_changed"))
    source_regrets = [
        float(row.get("relabel_compare", {}).get("source_best_new_regret", 0.0) or 0.0)
        for row in rows
    ]
    reason_counts: Counter[str] = Counter()
    seat_counts: Counter[str] = Counter()
    for row in rows:
        seat_counts[str(row.get("seat", "unknown"))] += 1
        for reason in row.get("selection_reasons", ()) or ():
            reason_counts[str(reason)] += 1
    skip_reason_counts: Counter[str] = Counter()
    for row in skipped:
        skip_reason_counts[str(row.get("skip_reason", "unknown"))] += 1
    t2_seconds = sum(float(stats.get("choose_action_T2_seconds", 0.0) or 0.0) for stats in profile_stats)
    return {
        "schema": "hu_turn1_stage1_refinement_relabel_summary_v1",
        "input": str(args.input),
        "output": str(args.output),
        "skip_output": str(args.skip_output) if args.skip_output else None,
        "target_attempts": len(rows) + len(skipped),
        "records": len(rows),
        "skipped_targets": len(skipped),
        "timed_out_targets": skip_reason_counts.get("target_timeout", 0),
        "failed_targets": skip_reason_counts.get("target_error", 0),
        "skip_reason_counts": dict(sorted(skip_reason_counts.items())),
        "profile": args.profile,
        "opponent_profile": args.opponent_profile,
        "future_samples": args.future_samples,
        "max_actions": args.max_actions,
        "candidate_model_path": str(args.candidate_model) if args.candidate_model else None,
        "candidate_model_paths": [str(path) for path in args.candidate_models or ()],
        "candidate_topk": args.candidate_topk,
        "candidate_union_cap": args.candidate_union_cap,
        "candidate_union_mode": args.candidate_union_mode,
        "source_actions_only": bool(getattr(args, "source_actions_only", False)),
        "target_timeout_seconds": args.target_timeout_seconds,
        "target_timeout_supported": target_timeout_supported(),
        "continue_on_target_error": bool(args.continue_on_target_error),
        "best_action_changed": changed,
        "best_action_changed_rate": changed / len(rows) if rows else 0.0,
        "source_best_new_regret_mean": sum(source_regrets) / len(source_regrets) if source_regrets else 0.0,
        "source_best_new_regret_max": max(source_regrets) if source_regrets else 0.0,
        "seat_counts": dict(sorted(seat_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "relabel_seconds_sum": sum(float(row.get("relabel_seconds", 0.0) or 0.0) for row in rows),
        "t2_choose_action_seconds_sum": t2_seconds,
        "seed": args.seed,
    }


def main() -> None:
    args = parse_args()
    if args.future_samples <= 0:
        raise SystemExit("--future-samples must be positive")
    if args.max_actions < 0:
        raise SystemExit("--max-actions must be non-negative")
    if args.candidate_topk < 0:
        raise SystemExit("--candidate-topk must be non-negative")
    if args.candidate_union_cap < 0:
        raise SystemExit("--candidate-union-cap must be non-negative")
    if args.candidate_model is not None and args.candidate_models:
        raise SystemExit("--candidate-model and --candidate-models are mutually exclusive")
    if args.candidate_model is not None and args.candidate_topk <= 0:
        raise SystemExit("--candidate-topk must be positive when --candidate-model is set")
    if args.candidate_models and args.candidate_topk <= 0:
        raise SystemExit("--candidate-topk must be positive when --candidate-models is set")
    if args.target_timeout_seconds < 0:
        raise SystemExit("--target-timeout-seconds must be non-negative")
    targets = read_jsonl(args.input, skip_targets=args.skip_targets, max_targets=args.max_targets)
    if not targets:
        raise SystemExit("no targets")
    policies = build_policies(args.profile, args.opponent_profile, args.seed, args.opening_lookahead_samples)
    candidate_model = load_hu_action_value_model(args.candidate_model) if args.candidate_model else None
    candidate_models = [load_hu_action_value_model(path) for path in args.candidate_models or ()]
    rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for record in targets:
        try:
            with target_timeout(args.target_timeout_seconds):
                row, profile_stats = relabel_record(
                    record,
                    policies=policies,
                    profile=args.profile,
                    opponent_profile=args.opponent_profile,
                    future_samples=args.future_samples,
                    max_actions=args.max_actions,
                    candidate_model=candidate_model,
                    candidate_models=candidate_models,
                    candidate_topk=args.candidate_topk,
                    candidate_union_cap=args.candidate_union_cap,
                    candidate_union_mode=args.candidate_union_mode,
                    source_actions_only=args.source_actions_only,
                    seed=args.seed,
                )
            rows.append(row)
            stats.append(profile_stats)
        except TargetTimeoutError as exc:
            skipped.append(
                skipped_target_record(
                    record,
                    reason="target_timeout",
                    message=str(exc),
                    profile=args.profile,
                    opponent_profile=args.opponent_profile,
                    future_samples=args.future_samples,
                    max_actions=args.max_actions,
                    timeout_seconds=args.target_timeout_seconds,
                )
            )
        except Exception as exc:
            if not args.continue_on_target_error:
                raise
            skipped.append(
                skipped_target_record(
                    record,
                    reason="target_error",
                    message=f"{type(exc).__name__}: {exc}",
                    profile=args.profile,
                    opponent_profile=args.opponent_profile,
                    future_samples=args.future_samples,
                    max_actions=args.max_actions,
                    timeout_seconds=args.target_timeout_seconds,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    if args.skip_output:
        args.skip_output.parent.mkdir(parents=True, exist_ok=True)
        with args.skip_output.open("w", encoding="utf-8") as handle:
            for row in skipped:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    result = summarize(rows, stats, skipped, args)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
