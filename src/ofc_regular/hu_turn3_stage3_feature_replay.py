"""Replay artifacts for HU Turn3 Stage3 feature generation benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .action_space import Action
from .hu_turn3_batch_continuation import HuTurn3State, Stage3ReferenceDecision
from .hu_turn3_stage3_feature_manifest import build_hu_turn3_stage3_feature_manifest
from .policy import board_to_json
from .state import Board


@dataclass
class Stage3FeatureReplay:
    states: list[HuTurn3State]
    actions_by_state: list[list[Action]]
    state_keys: list[tuple[Any, ...]]
    metadata: dict[str, Any]


_EXPECTED_SCORE_SAMPLES: dict[str, list[tuple[int, int, int, float]]] = {}
_EXPECTED_ROW_OFFSETS: dict[str, int] = {}


def append_stage3_feature_replay(
    path: str | Path,
    *,
    states: Sequence[HuTurn3State],
    actions_by_state: Sequence[list[Action]],
    state_keys: Sequence[tuple[Any, ...]],
    decisions: Sequence[Stage3ReferenceDecision],
    predictions: Sequence[np.ndarray | None],
    feature_schema_version: str,
    feature_column_names: Sequence[str],
    feature_dtype: str,
    profile: dict[str, Any],
    metadata: dict[str, Any],
    sample_limit: int = 0,
) -> None:
    output = resolve_stage3_feature_replay_path(path, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = _selected_count(len(states), sample_limit)
    selected_states = list(states[:selected])
    selected_actions = list(actions_by_state[:selected])
    selected_keys = list(state_keys[:selected])
    selected_decisions = list(decisions[:selected])
    selected_predictions = list(predictions[:selected])

    row_to_state_index: list[int] = []
    row_to_action_index: list[int] = []
    expected_scores: list[dict[str, Any]] = []
    for state_index, (actions, prediction) in enumerate(zip(selected_actions, selected_predictions)):
        for action_index, _action in enumerate(actions):
            row_to_state_index.append(state_index)
            row_to_action_index.append(action_index)
            if prediction is not None and action_index < len(prediction):
                expected_scores.append(
                    {
                        "state_index": state_index,
                        "action_index": action_index,
                        "score": float(prediction[action_index]),
                    }
                )

    manifest = build_hu_turn3_stage3_feature_manifest()
    action_counts = [len(actions) for actions in selected_actions]
    record = {
        "schema": "hu_turn3_stage3_feature_replay_v1",
        "feature_schema_version": feature_schema_version,
        "feature_column_names": list(feature_column_names),
        "feature_dtype": feature_dtype,
        "replay_source": metadata.get("replay_source", ""),
        "unique_t3_states": [_state_to_json(state) for state in selected_states],
        "legal_actions_by_state": [
            [_action_to_json(action) for action in actions] for actions in selected_actions
        ],
        "canonical_t3_state_key": [_jsonable_key(key) for key in selected_keys],
        "row_to_state_index": row_to_state_index,
        "row_to_action_index": row_to_action_index,
        "scalar_parity_sample_rows": _scalar_parity_sample_rows(
            row_to_state_index,
            row_to_action_index,
            expected_scores,
        ),
        "expected_stage3_reference": [
            _decision_to_json(decision) for decision in selected_decisions
        ],
        "expected_hgb_score_sample": expected_scores[: min(len(expected_scores), 4096)],
        "profile": {
            "replay_source": metadata.get("replay_source", ""),
            "raw_t3_states": int(metadata.get("raw_t3_states", len(states))),
            "unique_t3_states": len(selected_states),
            "legal_action_states": len(selected_actions),
            "legal_action_count_mean": float(sum(action_counts) / len(action_counts)) if action_counts else 0.0,
            "legal_action_count_max": int(max(action_counts)) if action_counts else 0,
            "total_feature_rows": len(row_to_state_index),
            "stage3_feature_rows": int(profile.get("stage3_feature_rows", 0)),
            "feature_column_count": int(profile.get("feature_column_count", 0)),
            "stage3_feature_mode": profile.get("stage3_feature_mode", ""),
            "direct_column_count": int(profile.get("direct_column_count", 0)),
            "scalar_fallback_column_count": int(profile.get("scalar_fallback_column_count", 0)),
            "direct_column_coverage_ratio": float(profile.get("direct_column_coverage_ratio", 0.0)),
            "stage3_feature_generation_total": float(profile.get("stage3_feature_generation_total", 0.0)),
            "stage3_encoder_matrix_build_seconds": float(profile.get("stage3_encoder_matrix_build_seconds", 0.0)),
            "stage3_after_board_construction_seconds": float(profile.get("stage3_after_board_construction_seconds", 0.0)),
            "stage3_row_summary_seconds": float(profile.get("stage3_row_summary_seconds", 0.0)),
            "stage3_global_summary_seconds": float(profile.get("stage3_global_summary_seconds", 0.0)),
            "stage3_action_delta_seconds": float(profile.get("stage3_action_delta_seconds", 0.0)),
            "stage3_scalar_fallback_seconds": float(profile.get("stage3_scalar_fallback_seconds", 0.0)),
            "stage3_cache_lookup_update_seconds": float(profile.get("stage3_cache_lookup_update_seconds", 0.0)),
            "stage3_column_validation_seconds": float(profile.get("stage3_column_validation_seconds", 0.0)),
            "stage3_numpy_allocation_seconds": float(profile.get("stage3_numpy_allocation_seconds", 0.0)),
            "stage3_non_encoder_overhead_seconds": float(profile.get("stage3_non_encoder_overhead_seconds", 0.0)),
            "stage3_state_feature_cache_hit": int(profile.get("stage3_state_feature_cache_hit", 0)),
            "stage3_state_feature_cache_miss": int(profile.get("stage3_state_feature_cache_miss", 0)),
            "row_summary_cache_hit": int(profile.get("row_summary_cache_hit", 0)),
            "row_summary_cache_miss": int(profile.get("row_summary_cache_miss", 0)),
            "top_summary_cache_hit": int(profile.get("top_summary_cache_hit", 0)),
            "top_summary_cache_miss": int(profile.get("top_summary_cache_miss", 0)),
            "complete_row_summary_cache_hit": int(profile.get("complete_row_summary_cache_hit", 0)),
            "complete_row_summary_cache_miss": int(profile.get("complete_row_summary_cache_miss", 0)),
            "feature_group_distribution": manifest.get("groups", {}),
            "source_teacher_run_hash": metadata.get("source_teacher_run_hash", ""),
        },
        "metadata": dict(metadata),
    }
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    _write_schema(output, record, metadata)
    _append_expected_npz(output, expected_scores, row_count=len(row_to_state_index))


def load_stage3_feature_replay(path: str | Path, *, limit_states: int = 0) -> Stage3FeatureReplay:
    replay_path = Path(path)
    states: list[HuTurn3State] = []
    actions_by_state: list[list[Action]] = []
    state_keys: list[tuple[Any, ...]] = []
    metadata: dict[str, Any] = {}
    for line in replay_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metadata.update(record.get("metadata", {}))
        for state_payload, action_payloads, key_payload in zip(
            record.get("unique_t3_states", ()),
            record.get("legal_actions_by_state", ()),
            record.get("canonical_t3_state_key", ()),
        ):
            if limit_states > 0 and len(states) >= limit_states:
                return Stage3FeatureReplay(states, actions_by_state, state_keys, metadata)
            states.append(_state_from_json(state_payload))
            actions_by_state.append([_action_from_json(action) for action in action_payloads])
            state_keys.append(_key_from_jsonable(key_payload))
    return Stage3FeatureReplay(states, actions_by_state, state_keys, metadata)


def resolve_stage3_feature_replay_path(path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    replay_path = Path(path)
    if replay_path.suffix.lower() == ".jsonl":
        return replay_path
    source = "replay"
    if metadata:
        source = str(metadata.get("replay_source") or source)
    return replay_path / f"stage3_feature_replay_{_safe_filename(source)}.jsonl"


def schema_path_for_replay(path: str | Path) -> Path:
    return Path(path).with_name("stage3_feature_replay_schema.json")


def expected_scores_path_for_replay(path: str | Path) -> Path:
    return Path(path).with_name("stage3_feature_replay_expected_scores.npz")


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return safe or "replay"


def _selected_count(count: int, sample_limit: int) -> int:
    if sample_limit <= 0:
        return count
    return min(count, sample_limit)


def _state_to_json(state: HuTurn3State) -> dict[str, Any]:
    return {
        "board": board_to_json(state.board),
        "dealt_cards": list(state.dealt_cards),
        "opponent_board": board_to_json(state.opponent_board),
        "dead_cards": list(state.dead_cards),
        "seat": state.seat,
        "to_act_order": state.to_act_order,
        "hand_id": state.hand_id,
        "game_id": state.game_id,
        "decision_seed": state.decision_seed,
        "street": state.street,
    }


def _state_from_json(payload: dict[str, Any]) -> HuTurn3State:
    return HuTurn3State(
        board=_board_from_json(payload["board"]),
        dealt_cards=tuple(payload["dealt_cards"]),
        opponent_board=_board_from_json(payload["opponent_board"]),
        dead_cards=tuple(payload.get("dead_cards", ())),
        seat=payload.get("seat", "first"),
        to_act_order=payload.get("to_act_order"),
        hand_id=payload.get("hand_id"),
        game_id=payload.get("game_id"),
        decision_seed=payload.get("decision_seed"),
        street=payload.get("street", "T3"),
    )


def _board_from_json(payload: dict[str, Sequence[str]]) -> Board:
    return Board.from_rows(
        top=payload.get("top", ()),
        middle=payload.get("middle", ()),
        bottom=payload.get("bottom", ()),
    )


def _action_to_json(action: Action) -> dict[str, Any]:
    return {
        "placements": [list(placement) for placement in action.placements],
        "discards": list(action.discards),
    }


def _action_from_json(payload: dict[str, Any]) -> Action:
    return Action(
        placements=tuple((str(card), str(row)) for card, row in payload.get("placements", ())),
        discards=tuple(str(card) for card in payload.get("discards", ())),
    )


def _decision_to_json(decision: Stage3ReferenceDecision) -> dict[str, Any]:
    return {
        "action_index": decision.action_index,
        "reference_margin": decision.reference_margin,
        "score": decision.score,
        "rank_score": decision.rank_score,
        "action_count": decision.action_count,
        "legality_status": decision.legality_status,
        "fallback_reason": decision.fallback_reason,
    }


def _jsonable_key(key: tuple[Any, ...]) -> Any:
    if isinstance(key, tuple):
        return [_jsonable_key(value) for value in key]
    return key


def _key_from_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_key_from_jsonable(item) for item in value)
    return value


def _scalar_parity_sample_rows(
    row_to_state_index: Sequence[int],
    row_to_action_index: Sequence[int],
    expected_scores: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    score_by_row = {
        (int(item["state_index"]), int(item["action_index"])): float(item["score"])
        for item in expected_scores[:64]
    }
    rows: list[dict[str, Any]] = []
    for row_index, (state_index, action_index) in enumerate(
        zip(row_to_state_index[:64], row_to_action_index[:64])
    ):
        rows.append(
            {
                "row_index": row_index,
                "state_index": int(state_index),
                "action_index": int(action_index),
                "expected_score": score_by_row.get((int(state_index), int(action_index))),
            }
        )
    return rows


def _write_schema(output: Path, record: dict[str, Any], metadata: dict[str, Any]) -> None:
    schema = {
        "schema": "hu_turn3_stage3_feature_replay_v1",
        "feature_schema_version": record["feature_schema_version"],
        "feature_column_count": len(record["feature_column_names"]),
        "feature_dtype": record["feature_dtype"],
        "metadata": dict(metadata),
        "jsonl": str(output),
        "expected_scores_npz": str(expected_scores_path_for_replay(output)),
    }
    schema_path_for_replay(output).write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _append_expected_npz(output: Path, scores: Sequence[dict[str, Any]], *, row_count: int) -> None:
    key = str(output.resolve())
    rows = _EXPECTED_SCORE_SAMPLES.setdefault(key, [])
    row_offset = _EXPECTED_ROW_OFFSETS.get(key, 0)
    for local_row_index, item in enumerate(scores[:4096]):
        rows.append(
            (
                row_offset + local_row_index,
                int(item["state_index"]),
                int(item["action_index"]),
                float(item["score"]),
            )
        )
    _EXPECTED_ROW_OFFSETS[key] = row_offset + row_count
    if not rows:
        return
    array = np.asarray(rows, dtype=np.float64)
    np.savez_compressed(
        expected_scores_path_for_replay(output),
        row_index=array[:, 0].astype(np.int64),
        state_index=array[:, 1].astype(np.int64),
        action_index=array[:, 2].astype(np.int64),
        score=array[:, 3].astype(np.float64),
    )
