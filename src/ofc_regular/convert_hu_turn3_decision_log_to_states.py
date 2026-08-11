"""Convert HU T3 runtime decision logs into joint-exact teacher states.

Only an explicit actor observation or ``visible_dead_cards`` is accepted as a
policy/search input. The historically ambiguous ``dead_cards`` field is never
promoted to actor-visible state. Legacy rows remain available for audit, but
are marked replay-ineligible unless their observation can be reconstructed
without hidden-card inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .action_space import Action, generate_turn_actions
from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key as semantic_action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .hu_infoset import (
    ActorObservation,
    InformationSetError,
    ReplayTruth,
    actor_observation_from_record,
    replay_truth_from_record,
)
from .state import Board


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def action_payload_key(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    normalized = {
        "placements": sorted([list(item) for item in payload.get("placements", ())]),
        "discards": sorted(str(card) for card in payload.get("discards", ())),
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def action_key(action: Action) -> str:
    """Legacy JSON identity used only to map historical decision-log payloads."""
    return action_payload_key(
        {
            "placements": action.placements,
            "discards": action.discards,
        }
    ) or ""


def board_from_json(payload: Mapping[str, Any] | None) -> Board:
    payload = payload or {}
    return Board.from_rows(
        top=payload.get("top", ()),
        middle=payload.get("middle", ()),
        bottom=payload.get("bottom", ()),
    )


def legal_action_index_by_key(actions: list[Action]) -> dict[str, int]:
    return {action_key(action): index for index, action in enumerate(actions)}


def _street_is_t3(row: dict[str, Any]) -> bool:
    street = str(row.get("street") or row.get("turn") or "").upper()
    return street in {"T3", "TURN3", "HU_TURN3", "TURN3_9CARD"} or street.endswith("T3")


def _declares_replay_truth(row: dict[str, Any]) -> bool:
    return any(
        name in row and row.get(name) is not None
        for name in (
            "replay_truth",
            "true_hero_private_discards",
            "true_opponent_private_discards",
        )
    ) or bool(row.get("true_dead_cards"))


def _replay_truth_if_declared(row: dict[str, Any]) -> ReplayTruth | None:
    if not _declares_replay_truth(row):
        return None
    return replay_truth_from_record(row)


def _source_hand_seed(row: Mapping[str, Any]) -> tuple[int | str | None, str]:
    """Resolve the hand seed without confusing it with policy RNG.

    Runtime decision logs historically call the per-decision policy RNG value
    ``seed``.  ``play_hand`` passes the actual shuffle seed as ``hand_id``;
    newer producers may provide an explicit ``hand_seed``.  The legacy
    ``seed`` field is therefore never a valid fallback for ``hand_seed``.

    Opaque string hand IDs remain useful identities, but are not assumed to be
    reproducible RNG seeds.  Explicit string hand seeds are preserved for
    compatibility with existing JSONL artifacts that serialize integer seeds
    as strings.
    """

    if "hand_seed" in row and row.get("hand_seed") is not None:
        value = row.get("hand_seed")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, str))
            or (isinstance(value, str) and not value.strip())
        ):
            raise ValueError("hand_seed must be a non-empty integer or string")
        return value, "explicit_hand_seed"

    hand_id = row.get("hand_id")
    if isinstance(hand_id, int) and not isinstance(hand_id, bool):
        return hand_id, "numeric_hand_id"
    return None, "unavailable"


def _source_decision_seed(row: Mapping[str, Any]) -> Any:
    """Return the policy-decision seed while preserving legacy log shape."""

    explicit = row.get("decision_seed")
    return explicit if explicit is not None else row.get("seed")


def _actor_observation_for_row(
    row: dict[str, Any],
    *,
    board: Any,
    opponent_board: Any,
    dealt: tuple[str, ...],
    truth: ReplayTruth | None,
) -> ActorObservation:
    nested = row.get("policy_observation")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise InformationSetError("policy_observation must be a mapping")
        observation = ActorObservation.from_dict(nested)
        if (
            observation.hero_board != board
            or observation.opponent_public_board != opponent_board
            or observation.dealt_cards != dealt
        ):
            raise InformationSetError(
                "policy_observation disagrees with T3 decision state"
            )
        if observation.street != "T3":
            raise InformationSetError("policy_observation is not a T3 state")
        return observation

    visible_raw = row.get("visible_dead_cards")
    hero_private_raw = row.get("hero_private_discards")
    if visible_raw is None and truth is not None:
        visible_raw = truth.visible_dead_cards
    if hero_private_raw is None and truth is not None:
        hero_private_raw = truth.hero_private_discards
    if visible_raw is None:
        raise InformationSetError("record lacks explicit visible_dead_cards")
    if hero_private_raw is None:
        raise InformationSetError("record lacks explicit hero_private_discards")

    normalized = dict(row)
    normalized.update(
        {
            "board": row["hero_board"],
            "opponent_board": row["opponent_board"],
            "dealt": list(dealt),
            "hero_private_discards": list(hero_private_raw),
            "visible_dead_cards": list(visible_raw),
            "turn": "T3",
        }
    )
    return actor_observation_from_record(normalized)


def convert_decision_row(
    row: dict[str, Any],
    *,
    source_label: str,
    source_input_path: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if not _street_is_t3(row):
        return None, "non_t3"
    if not row.get("hero_board") or not row.get("opponent_board") or not row.get("cards_to_place"):
        return None, "missing_state_fields"

    board = board_from_json(row.get("hero_board"))
    opponent_board = board_from_json(row.get("opponent_board"))
    dealt = tuple(str(card) for card in row.get("cards_to_place", ()))
    try:
        actions = generate_turn_actions(board, dealt)
    except Exception:
        return None, "legal_action_generation_failed"
    index_by_key = legal_action_index_by_key(actions)

    baseline_key = action_payload_key(row.get("stage3_action"))
    hu_key = action_payload_key(row.get("stage7_action") or row.get("final_action"))
    final_key = action_payload_key(row.get("final_action"))
    fallback_key = action_payload_key(row.get("fallback_action"))
    baseline_index = index_by_key.get(baseline_key or "")
    hu_index = index_by_key.get(hu_key or "")
    final_index = index_by_key.get(final_key or "")
    fallback_index = index_by_key.get(fallback_key or "")
    if baseline_index is None or hu_index is None:
        return None, "action_mapping_failed"

    try:
        hand_seed, hand_seed_source = _source_hand_seed(row)
    except ValueError:
        return None, "invalid_hand_seed"
    decision_seed = _source_decision_seed(row)

    truth: ReplayTruth | None = None
    observation: ActorObservation | None = None
    observation_error: str | None = None
    try:
        truth = _replay_truth_if_declared(row)
        observation = _actor_observation_for_row(
            row,
            board=board,
            opponent_board=opponent_board,
            dealt=dealt,
            truth=truth,
        )
    except (InformationSetError, ValueError) as exc:
        observation_error = str(exc)
    visible_dead_cards = (
        list(observation.legacy_dead_cards()) if observation is not None else []
    )
    replay_ready = observation is not None
    declared_state_id = row.get("state_id")
    state_id = str(
        declared_state_id
        if declared_state_id is not None
        else (
            f"{source_label}:{hand_seed}:{row.get('game_id')}:"
            f"{row.get('hand_id')}:{row.get('seat')}:T3"
        )
    )
    selection = {
        "action_key_schema": ACTION_KEY_SCHEMA,
        "legal_action_set_digest": legal_action_set_digest(actions),
        "legal_action_order_digest": ordered_action_mapping_digest(actions),
        "baseline_index": int(baseline_index),
        "baseline_action_key": semantic_action_key(
            actions[baseline_index]
        ).to_token(),
        "hu_index": int(hu_index),
        "hu_action_key": semantic_action_key(actions[hu_index]).to_token(),
        "final_index": int(final_index) if final_index is not None else None,
        "final_action_key": (
            semantic_action_key(actions[final_index]).to_token()
            if final_index is not None
            else None
        ),
        "fallback_index": int(fallback_index) if fallback_index is not None else None,
        "fallback_action_key": (
            semantic_action_key(actions[fallback_index]).to_token()
            if fallback_index is not None
            else None
        ),
        "disagreement": bool(baseline_index != hu_index),
        "override_fired": bool(row.get("override_fired")),
        "no_override_reason": row.get("no_override_reason", ""),
        "predicted_margin_vs_baseline": row.get("stage7_predicted_margin"),
        "reference_margin": row.get("reference_margin"),
        "model_score": row.get("model_score"),
        "runtime_latency_ms": row.get("runtime_latency_ms"),
        "hu_turn3_min_margin": row.get("hu_turn3_min_margin"),
        "hu_turn3_reference_min_margin": row.get("hu_turn3_reference_min_margin"),
    }
    state = {
        "schema": "hu_stage1_state",
        "rule_set": "regular",
        "phase": "hu_turn3_9card",
        "source": source_label,
        "state_id": state_id,
        # Keep ``seed`` as a compatibility alias for the source policy-decision
        # seed, but name and tag it explicitly.  It must never be promoted to a
        # hand/deck seed.
        "seed": decision_seed,
        "decision_seed": decision_seed,
        "seed_semantics": "policy_decision_seed",
        "hand_seed": hand_seed,
        "hand_seed_source": hand_seed_source,
        "hand_seed_available": hand_seed is not None,
        "hand_id": row.get("hand_id"),
        "game_id": row.get("game_id"),
        "hero_seat": row.get("seat"),
        "seat": row.get("seat"),
        "to_act_order": (
            observation.to_act_order if observation is not None else row.get("to_act_order")
        ),
        "board": row.get("hero_board"),
        "opponent_board": row.get("opponent_board"),
        "dealt": list(dealt),
        "dead_cards": visible_dead_cards,
        "visible_dead_cards": visible_dead_cards,
        "hero_private_discards": (
            list(observation.hero_private_discards)
            if observation is not None
            else []
        ),
        "policy_observation": (
            observation.to_dict() if observation is not None else None
        ),
        "selection": selection,
        "runtime_decision": {
            "stage3_action": row.get("stage3_action"),
            "stage7_action": row.get("stage7_action"),
            "fallback_action": row.get("fallback_action"),
            "final_action": row.get("final_action"),
            "legality_check_result": row.get("legality_check_result"),
        },
        "replay_ready": replay_ready,
        "replay_truth_available": truth is not None,
        "replay_truth": truth.to_dict() if truth is not None else None,
        "true_dead_cards": (
            list(truth.true_dead_cards) if truth is not None else None
        ),
        "observation_error": observation_error,
        "missing_dead_cards": not replay_ready,
        "legacy_runtime_log": not replay_ready,
        "exclude_from_exact_replay": not replay_ready,
        "source_input_path": source_input_path,
        "visibility_model": (
            "actor_observation_v1" if replay_ready else "legacy_ambiguous_dead_cards"
        ),
        "discard_visibility": "own_private_only" if replay_ready else "unknown_legacy",
    }
    return state, "converted"


def convert_rows(
    rows: Iterable[dict[str, Any]],
    *,
    source_label: str,
    source_input_path: str | None = None,
    only_overrides: bool = False,
    require_replay_ready: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        if only_overrides and not row.get("override_fired"):
            counts["filtered_non_override"] = counts.get("filtered_non_override", 0) + 1
            continue
        state, reason = convert_decision_row(row, source_label=source_label, source_input_path=source_input_path)
        counts[reason] = counts.get(reason, 0) + 1
        if state is None:
            continue
        if require_replay_ready and not state.get("replay_ready"):
            counts["filtered_replay_ineligible"] = counts.get("filtered_replay_ineligible", 0) + 1
            continue
        out.append(state)
    counts["written"] = len(out)
    counts["replay_ready"] = sum(1 for row in out if row.get("replay_ready"))
    counts["replay_ineligible"] = sum(1 for row in out if not row.get("replay_ready"))
    return out, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="HU T3 decision log JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output hu_stage1_state JSONL.")
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--source-label", default="runtime_decision_log")
    parser.add_argument("--only-overrides", action="store_true")
    parser.add_argument(
        "--require-replay-ready",
        action="store_true",
        help="Skip logs without a validated actor-visible observation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    states, counts = convert_rows(
        rows,
        source_label=args.source_label,
        source_input_path=str(args.input),
        only_overrides=args.only_overrides,
        require_replay_ready=args.require_replay_ready,
    )
    write_jsonl(args.output, states)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "source_label": args.source_label,
        "only_overrides": bool(args.only_overrides),
        "require_replay_ready": bool(args.require_replay_ready),
        **counts,
    }
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
