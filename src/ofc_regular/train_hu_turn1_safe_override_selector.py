"""Train a small HU Turn1 safe-override selector from realized fired rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    action_key_from_payload,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .hu_infoset import ActorObservation
from .state import Board
from .hu_turn3_model import HU_FEATURE_DIM, sample_to_matrix


META_FEATURE_NAMES = (
    "meta_hu_turn1_predicted_margin",
    "meta_candidate_score",
    "meta_fallback_score",
    "meta_score_delta",
    "meta_action_count_log1p",
    "meta_seat_is_second",
)


@dataclass(frozen=True)
class TrainingData:
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    deltas: np.ndarray
    regrets: np.ndarray
    rows: list[dict[str, Any]]
    feature_mode: str
    label_mode: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _strict_index(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    try:
        if float(value) != float(index):
            raise ValueError(f"{field} must be a non-negative integer")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if index < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return index


def _consistent_index(
    payload: Mapping[str, Any], fields: Sequence[str], *, role: str
) -> int | None:
    indexed = [
        (field, _strict_index(payload[field], field=field))
        for field in fields
        if payload.get(field) is not None
    ]
    if not indexed:
        return None
    expected = indexed[0][1]
    if any(index != expected for _field, index in indexed[1:]):
        details = ", ".join(f"{field}={index}" for field, index in indexed)
        raise ValueError(f"{role} saved action indices disagree: {details}")
    return expected


def _parse_action_key(value: Any, *, field: str) -> ActionKey:
    if isinstance(value, str):
        return ActionKey.from_token(value)
    if isinstance(value, Mapping):
        return ActionKey.from_dict(value)
    raise ValueError(f"{field} must be an ActionKey token or payload")


def _semantic_action_key(
    payload: Mapping[str, Any],
    *,
    key_fields: Sequence[str],
    action_fields: Sequence[str],
    role: str,
    payload_is_action: bool = False,
) -> ActionKey:
    evidence: list[tuple[str, ActionKey]] = []
    for field in key_fields:
        if payload.get(field) is not None:
            evidence.append((field, _parse_action_key(payload[field], field=field)))
    for field in action_fields:
        action_payload = payload.get(field)
        if action_payload is None:
            continue
        if not isinstance(action_payload, Mapping):
            raise ValueError(f"{role} {field} must be an action payload")
        evidence.append((field, action_key_from_payload(action_payload)))
        for embedded_field in ("canonical_action_key", "action_key"):
            if action_payload.get(embedded_field) is not None:
                evidence.append(
                    (
                        f"{field}.{embedded_field}",
                        _parse_action_key(
                            action_payload[embedded_field],
                            field=f"{field}.{embedded_field}",
                        ),
                    )
                )
    if payload_is_action:
        if "placements" not in payload or "discards" not in payload:
            raise ValueError(f"{role} score row is missing its action payload")
        evidence.append(("action_payload", action_key_from_payload(payload)))
    if not evidence:
        raise ValueError(f"{role} requires a semantic action key or payload")
    expected = evidence[0][1]
    for field, key in evidence[1:]:
        if key != expected:
            details = ", ".join(
                f"{name}={candidate.to_token()}" for name, candidate in evidence
            )
            raise ValueError(f"{role} semantic action evidence disagrees: {details}")
    return expected


def _strict_action_score(action: Mapping[str, Any], *, role: str) -> float:
    values: list[tuple[str, float]] = []
    for field in ("score", "ev"):
        if action.get(field) is None:
            continue
        try:
            value = float(action[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{role} {field} must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"{role} {field} must be finite")
        values.append((field, value))
    if not values:
        raise ValueError(f"{role} score row is missing score/ev")
    expected = values[0][1]
    if any(value != expected for _field, value in values[1:]):
        raise ValueError(f"{role} score and ev disagree")
    return expected


def _legal_turn1_actions(row: Mapping[str, Any]) -> list[Action]:
    board_payload = row.get("hero_board") or row.get("board")
    if not isinstance(board_payload, Mapping):
        raise ValueError("Stage10 MC32 row is missing the hero board")
    dealt = row.get("cards_to_place") or row.get("dealt")
    if not isinstance(dealt, Sequence) or isinstance(dealt, (str, bytes)):
        raise ValueError("Stage10 MC32 row is missing dealt cards")
    board = Board.from_rows(
        board_payload.get("top", ()),
        board_payload.get("middle", ()),
        board_payload.get("bottom", ()),
    )
    return generate_turn_actions(board, tuple(str(card) for card in dealt))


def _validate_action_mapping_metadata(
    row: Mapping[str, Any], legal_actions: Sequence[Action]
) -> None:
    schema = row.get("action_key_schema")
    if schema is not None and schema != ACTION_KEY_SCHEMA:
        raise ValueError(f"unsupported action key schema: {schema!r}")
    expected_set_digest = row.get("legal_action_set_digest")
    if expected_set_digest is not None:
        actual_set_digest = legal_action_set_digest(legal_actions)
        if expected_set_digest != actual_set_digest:
            raise ValueError("legal action set digest mismatch")
    expected_order_digest = row.get("legal_action_order_digest")
    if expected_order_digest is not None:
        actual_order_digest = ordered_action_mapping_digest(legal_actions)
        if expected_order_digest != actual_order_digest:
            raise ValueError("legal action order digest mismatch")


def _resolve_stage10_action_scores(row: Mapping[str, Any]) -> tuple[float, float]:
    actions = row.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("Stage10 MC32 relabel row requires evaluated actions")
    legal_actions = _legal_turn1_actions(row)
    _validate_action_mapping_metadata(row, legal_actions)
    legal_indices = {action_key(action): index for index, action in enumerate(legal_actions)}
    if len(legal_indices) != len(legal_actions):
        raise ValueError("generated legal action mapping contains duplicate keys")

    candidate_key = _semantic_action_key(
        row,
        key_fields=(
            "candidate_action_key",
            "hu_turn1_action_key",
            "runtime_candidate_action_key",
        ),
        action_fields=("hu_turn1_action", "runtime_candidate_action"),
        role="candidate",
    )
    baseline_key = _semantic_action_key(
        row,
        key_fields=(
            "baseline_action_key",
            "fallback_action_key",
            "runtime_baseline_action_key",
        ),
        action_fields=("baseline_action", "fallback_action", "runtime_baseline_action"),
        role="baseline",
    )
    for role, key in (("candidate", candidate_key), ("baseline", baseline_key)):
        if key not in legal_indices:
            raise ValueError(f"{role} semantic action is not legal in the saved state")

    candidate_index = _consistent_index(
        row,
        ("candidate_action_index", "runtime_candidate_action_index"),
        role="candidate",
    )
    baseline_index = _consistent_index(
        row,
        (
            "baseline_action_index",
            "fallback_action_index",
            "runtime_baseline_action_index",
        ),
        role="baseline",
    )
    for role, saved_index, key in (
        ("candidate", candidate_index, candidate_key),
        ("baseline", baseline_index, baseline_key),
    ):
        if saved_index is not None and saved_index != legal_indices[key]:
            raise ValueError(
                f"{role} saved action index disagrees with its semantic action"
            )

    scores: dict[ActionKey, float] = {}
    action_row_keys: list[ActionKey] = []
    for row_index, action_row in enumerate(actions):
        if not isinstance(action_row, Mapping):
            raise ValueError(f"actions[{row_index}] must be an action payload")
        key = _semantic_action_key(
            action_row,
            key_fields=("canonical_action_key", "action_key"),
            action_fields=("action",),
            role=f"actions[{row_index}]",
            payload_is_action="action" not in action_row,
        )
        if key not in legal_indices:
            raise ValueError(f"actions[{row_index}] is not legal in the saved state")
        saved_index = _consistent_index(
            action_row,
            ("action_index", "original_index"),
            role=f"actions[{row_index}]",
        )
        if saved_index is not None and saved_index != legal_indices[key]:
            raise ValueError(
                f"actions[{row_index}] saved index disagrees with its semantic action"
            )
        if key in scores:
            raise ValueError(f"duplicate evaluated semantic action: {key.to_token()}")
        scores[key] = _strict_action_score(action_row, role=f"actions[{row_index}]")
        action_row_keys.append(key)

    for role, field, key in (
        ("candidate", "candidate_action_row_index", candidate_key),
        ("baseline", "baseline_action_row_index", baseline_key),
    ):
        if row.get(field) is None:
            continue
        saved_row_index = _strict_index(row[field], field=field)
        if saved_row_index >= len(action_row_keys) or action_row_keys[saved_row_index] != key:
            raise ValueError(f"{role} saved action row index disagrees with its semantic action")

    if candidate_key not in scores:
        raise ValueError("candidate semantic action has no evaluated score")
    if baseline_key not in scores:
        raise ValueError("baseline semantic action has no evaluated score")
    return scores[candidate_key], scores[baseline_key]


def stage10_mc32_candidate_delta(row: dict[str, Any]) -> float:
    """Return candidate minus baseline MC relabel score.

    Semantic action identity is authoritative.  When evaluated action rows are
    present, every saved positional index and action-order digest is checked
    against a freshly generated legal mapping before a score is returned.
    Corrupt or ambiguous rows raise instead of silently producing a label.
    """

    actions_present = row.get("actions") is not None
    resolved_delta: float | None = None
    if actions_present:
        candidate_score, baseline_score = _resolve_stage10_action_scores(row)
        resolved_delta = candidate_score - baseline_score

    if "stage10_mc32_candidate_delta" in row:
        try:
            saved_delta = float(row["stage10_mc32_candidate_delta"])
        except (TypeError, ValueError) as exc:
            raise ValueError("stage10_mc32_candidate_delta must be finite") from exc
        if not math.isfinite(saved_delta):
            raise ValueError("stage10_mc32_candidate_delta must be finite")
        if resolved_delta is not None and saved_delta != resolved_delta:
            raise ValueError(
                "saved stage10_mc32_candidate_delta disagrees with semantic action scores"
            )
        return saved_delta

    if resolved_delta is None:
        raise ValueError(
            "Stage10 MC32 row requires a saved delta or semantically mapped action scores"
        )
    return resolved_delta


def row_to_sample(row: dict[str, Any]) -> dict[str, Any]:
    candidate_action = row.get("hu_turn1_action") or row.get("runtime_candidate_action") or row.get("final_action")
    baseline_action = row.get("baseline_action") or row.get("runtime_baseline_action") or row.get("fallback_action")
    if not isinstance(candidate_action, dict) or not isinstance(baseline_action, dict):
        raise ValueError("row is missing candidate/baseline action JSON")
    board = row.get("hero_board") or row.get("board")
    opponent = row.get("opponent_board")
    if not isinstance(board, dict) or not isinstance(opponent, dict):
        raise ValueError("row is missing hero/opponent board JSON")
    nested_observation = row.get("policy_observation")
    if not isinstance(nested_observation, dict):
        raise ValueError(
            "safe-selector training requires a versioned policy_observation"
        )
    observation = ActorObservation.from_dict(nested_observation)
    dealt = tuple(row.get("cards_to_place") or row.get("dealt") or ())
    if (
        observation.hero_board != Board.from_rows(**board)
        or observation.opponent_public_board != Board.from_rows(**opponent)
        or observation.dealt_cards != dealt
        or observation.street != "T1"
    ):
        raise ValueError("safe-selector row disagrees with policy_observation")
    opponent_cards = sum(len(opponent.get(row_name, ())) for row_name in ("top", "middle", "bottom"))
    hero_cards = sum(len(board.get(row_name, ())) for row_name in ("top", "middle", "bottom"))
    to_act_order = "second" if opponent_cards > hero_cards else "first"
    return {
        "rule_set": "regular",
        "schema": "hu_stage1",
        "phase": "hu_turn1_5card",
        "seat": row.get("seat") or "first",
        "to_act_order": to_act_order,
        "board": board,
        "opponent_board": opponent,
        "dead_cards": list(observation.legacy_dead_cards()),
        "dealt": list(dealt),
        "best_action": 0,
        "score_gap": 0.0,
        "actions": [candidate_action, baseline_action],
    }


def row_to_feature_vector(row: dict[str, Any], *, feature_mode: str) -> np.ndarray:
    sample = row_to_sample(row)
    features, _targets = sample_to_matrix(sample)
    if features.shape[0] != 2:
        raise ValueError("expected candidate and baseline feature rows")
    candidate = features[0].astype(np.float32, copy=False)
    baseline = features[1].astype(np.float32, copy=False)
    delta = candidate - baseline
    meta = np.asarray(
        [
            safe_float(row.get("hu_turn1_predicted_margin", row.get("runtime_predicted_margin"))),
            safe_float(row.get("candidate_score", row.get("runtime_candidate_score"))),
            safe_float(row.get("fallback_score", row.get("runtime_baseline_score"))),
            safe_float(row.get("candidate_score", row.get("runtime_candidate_score")))
            - safe_float(row.get("fallback_score", row.get("runtime_baseline_score"))),
            math.log1p(max(0.0, safe_float(row.get("action_count")))),
            1.0 if row.get("seat") == "second" else 0.0,
        ],
        dtype=np.float32,
    )
    if feature_mode == "delta_plus_meta":
        return np.concatenate([delta, meta]).astype(np.float32, copy=False)
    if feature_mode == "candidate_delta_plus_meta":
        return np.concatenate([candidate, delta, meta]).astype(np.float32, copy=False)
    runtime_meta = np.asarray(
        [
            safe_float(row.get("stage_a_delta")),
            safe_float(row.get("stage_a_delta_se")),
            safe_float(row.get("confirm_delta")),
            safe_float(row.get("confirm_delta_se")),
            safe_float(row.get("confirm_delta_count")),
            safe_float(row.get("confirm_delta")) - safe_float(row.get("confirm_delta_se")),
            (
                safe_float(row.get("confirm_delta"))
                / max(1.0e-6, safe_float(row.get("confirm_delta_se")))
                if safe_float(row.get("confirm_delta_se")) > 0.0
                else 0.0
            ),
        ],
        dtype=np.float32,
    )
    if feature_mode == "candidate_delta_plus_runtime_meta":
        return np.concatenate([candidate, delta, meta, runtime_meta]).astype(np.float32, copy=False)
    if feature_mode == "delta_plus_runtime_meta":
        return np.concatenate([delta, meta, runtime_meta]).astype(np.float32, copy=False)
    if feature_mode == "meta_only":
        return meta
    raise ValueError(f"unsupported feature mode: {feature_mode}")


def label_for_row(
    row: dict[str, Any],
    *,
    label_mode: str,
    teacher_accept_regret: float,
    teacher_gray_regret: float,
    accept_delta: float,
    hard_negative_delta: float,
    gray_weight: float,
) -> tuple[int, float, float, float, bool]:
    delta = safe_float(row.get("hu_turn1_realized_delta", row.get("realized_candidate_seat_delta")))
    regret = safe_float(row.get("relabel_compare", {}).get("source_best_new_regret"))
    if label_mode in {
        "stage10_mc32_delta",
        "teacher_delta_thresholded",
        "high_mc_delta_thresholded",
    }:
        delta = stage10_mc32_candidate_delta(row)
        if label_mode in {"teacher_delta_thresholded", "high_mc_delta_thresholded"}:
            if delta >= accept_delta:
                return 1, 1.0, delta, regret, False
            if delta <= hard_negative_delta:
                negative_scale = max(abs(hard_negative_delta), abs(accept_delta), 1e-9)
                weight = min(10.0, max(2.0, abs(delta) / negative_scale))
                return 0, weight, delta, regret, False
            return 0, gray_weight, delta, regret, True
        if delta > 0.0:
            return 1, 1.0, delta, regret, False
        if delta < 0.0:
            return 0, 2.0, delta, regret, False
        return 0, gray_weight, delta, regret, True
    if label_mode == "teacher_regret":
        if regret <= teacher_accept_regret:
            return 1, 1.0, delta, regret, False
        if regret <= teacher_gray_regret:
            return 0, 0.35, delta, regret, True
        weight = min(8.0, max(1.0, regret / 4.0))
        return 0, weight, delta, regret, False
    if label_mode == "teacher_regret_or_accept_gray_delta" and isinstance(row.get("relabel_compare"), dict):
        if regret <= teacher_accept_regret:
            return 1, 1.0, delta, regret, False
        if regret <= teacher_gray_regret:
            return 0, gray_weight, delta, regret, True
        weight = min(8.0, max(1.0, regret / 4.0))
        return 0, weight, delta, regret, False
    if label_mode == "teacher_regret_or_accept_gray_delta":
        label_mode = "accept_gray_delta"
    if label_mode == "accept_gray_delta":
        if delta >= accept_delta:
            return 1, 1.0, delta, regret, False
        if delta <= hard_negative_delta:
            negative_scale = max(abs(hard_negative_delta), abs(accept_delta), 1e-9)
            weight = min(10.0, max(2.0, abs(delta) / negative_scale))
            return 0, weight, delta, regret, False
        return 0, gray_weight, delta, regret, True
    if label_mode != "realized_delta":
        raise ValueError(f"unsupported label mode: {label_mode}")
    label = 1 if delta > 0.0 else 0
    if delta < 0.0:
        weight = 2.0
    elif delta == 0.0:
        weight = 0.35
    else:
        weight = 1.0
    return label, weight, delta, regret, delta == 0.0


def build_training_data(
    rows: Iterable[dict[str, Any]],
    *,
    feature_mode: str = "delta_plus_meta",
    include_neutral: bool = True,
    include_non_fired_candidates: bool = False,
    label_mode: str = "realized_delta",
    teacher_accept_regret: float = 0.25,
    teacher_gray_regret: float = 1.0,
    accept_delta: float = 2.0,
    hard_negative_delta: float = -5.0,
    gray_weight: float = 0.35,
) -> TrainingData:
    materialized: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    deltas: list[float] = []
    regrets: list[float] = []
    for row in rows:
        if not row.get("override_fired", True) and not (
            include_non_fired_candidates and _is_usable_non_fired_candidate(row)
        ):
            continue
        label, weight, delta, regret, is_gray = label_for_row(
            row,
            label_mode=label_mode,
            teacher_accept_regret=teacher_accept_regret,
            teacher_gray_regret=teacher_gray_regret,
            accept_delta=accept_delta,
            hard_negative_delta=hard_negative_delta,
            gray_weight=gray_weight,
        )
        if is_gray and not include_neutral:
            continue
        features.append(row_to_feature_vector(row, feature_mode=feature_mode))
        labels.append(label)
        weights.append(weight)
        deltas.append(delta)
        regrets.append(regret)
        materialized.append(row)
    if not features:
        raise ValueError("no usable T1 safe-override rows")
    return TrainingData(
        features=np.vstack(features).astype(np.float32, copy=False),
        labels=np.asarray(labels, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float64),
        deltas=np.asarray(deltas, dtype=np.float64),
        regrets=np.asarray(regrets, dtype=np.float64),
        rows=materialized,
        feature_mode=feature_mode,
        label_mode=label_mode,
    )


def _is_usable_non_fired_candidate(row: dict[str, Any]) -> bool:
    """Return true for gated-off candidate-vs-baseline rows with labels."""

    if row.get("override_fired", False):
        return False
    if not row.get("realized_delta_valid", False):
        return False
    if not isinstance(row.get("hu_turn1_action") or row.get("runtime_candidate_action"), dict):
        return False
    candidate_index = row.get("candidate_action_index", row.get("runtime_candidate_action_index"))
    baseline_index = row.get(
        "baseline_action_index",
        row.get("fallback_action_index", row.get("runtime_baseline_action_index")),
    )
    if candidate_index is None or baseline_index is None:
        return False
    return candidate_index != baseline_index


def deterministic_split(rows: list[dict[str, Any]], labels: np.ndarray | None = None) -> np.ndarray:
    """Build a deterministic train/val/test split.

    T1 fired datasets are small enough that a pure hash bucket can put all
    positives outside the validation split.  When labels are available, split
    within each class after sorting by a stable row hash.
    """

    if labels is None:
        split = np.zeros(len(rows), dtype=np.int8)
        for index, row in enumerate(rows):
            bucket = _row_hash_bucket(row)
            if bucket < 70:
                split[index] = 0
            elif bucket < 85:
                split[index] = 1
            else:
                split[index] = 2
        return split

    split = np.zeros(len(rows), dtype=np.int8)
    label_values = labels.tolist()
    for label in sorted(set(int(value) for value in label_values)):
        indices = [index for index, value in enumerate(label_values) if int(value) == label]
        indices.sort(key=lambda index: _row_hash_key(rows[index]))
        train_count, val_count, _test_count = _stratified_counts(len(indices))
        for position, index in enumerate(indices):
            if position < train_count:
                split[index] = 0
            elif position < train_count + val_count:
                split[index] = 1
            else:
                split[index] = 2
    return split


def _row_hash_key(row: dict[str, Any]) -> str:
    key = "|".join(
        str(row.get(field, ""))
        for field in ("hand_seed", "paired_index", "seat_swap", "seat", "candidate_action_index")
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _row_hash_bucket(row: dict[str, Any]) -> int:
    return int(_row_hash_key(row)[:8], 16) % 100


def _stratified_counts(count: int) -> tuple[int, int, int]:
    if count <= 1:
        return count, 0, 0
    if count == 2:
        return 1, 0, 1
    val_count = max(1, int(round(count * 0.15)))
    test_count = max(1, int(round(count * 0.15)))
    while val_count + test_count >= count:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    train_count = count - val_count - test_count
    return train_count, val_count, test_count


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return 0.0
    hit = 0
    total = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            hit += 1
            total += hit / rank
    return float(total / positives)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0
    wins = 0.0
    for positive in positives:
        wins += float(np.sum(positive > negatives))
        wins += 0.5 * float(np.sum(positive == negatives))
    return float(wins / (len(positives) * len(negatives)))


def threshold_rows(
    labels: np.ndarray,
    scores: np.ndarray,
    deltas: np.ndarray,
    *,
    split_name: str,
    thresholds: Iterable[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for threshold in thresholds:
        fired = scores >= threshold
        fired_deltas = deltas[fired]
        if fired_deltas.size:
            mean_delta = float(np.mean(fired_deltas))
            loss_count = int(np.sum(fired_deltas < 0))
            win_count = int(np.sum(fired_deltas > 0))
            zero_count = int(np.sum(fired_deltas == 0))
            precision = float(np.mean(labels[fired] == 1))
        else:
            mean_delta = 0.0
            loss_count = win_count = zero_count = 0
            precision = 0.0
        output.append(
            {
                "split": split_name,
                "threshold": threshold,
                "fires": int(np.sum(fired)),
                "fire_rate": float(np.mean(fired)) if labels.size else 0.0,
                "precision_positive": precision,
                "mean_realized_delta": mean_delta,
                "loss_count": loss_count,
                "win_count": win_count,
                "zero_count": zero_count,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_selector(
    data: TrainingData,
    *,
    output_dir: Path,
    model_output: Path,
    random_state: int = 20260624,
) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    split = deterministic_split(data.rows, data.labels)
    models = {
        "logistic_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
            ),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        ),
        "hist_gradient": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }
    metrics: list[dict[str, Any]] = []
    predictions_for_best: np.ndarray | None = None
    best_name = ""
    best_model: Any | None = None
    best_score = -1.0
    thresholds = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    threshold_output: list[dict[str, Any]] = []

    for name, model in models.items():
        train_mask = split == 0
        if len(set(data.labels[train_mask].tolist())) < 2:
            continue
        model.fit(data.features[train_mask], data.labels[train_mask], **_fit_kwargs(model, data.weights[train_mask]))
        scores = _predict_probability(model, data.features)
        for split_id, split_name in ((0, "train"), (1, "val"), (2, "test")):
            mask = split == split_id
            if not np.any(mask):
                continue
            labels = data.labels[mask]
            split_scores = scores[mask]
            split_deltas = data.deltas[mask]
            metrics.append(
                {
                    "model": name,
                    "split": split_name,
                    "rows": int(np.sum(mask)),
                    "positives": int(np.sum(labels == 1)),
                    "negatives": int(np.sum(labels == 0)),
                    "average_precision": average_precision(labels, split_scores),
                    "roc_auc": roc_auc(labels, split_scores),
                    "score_mean": float(np.mean(split_scores)),
                }
            )
            threshold_output.extend(
                {
                    "model": name,
                    **row,
                }
                for row in threshold_rows(
                    labels,
                    split_scores,
                    split_deltas,
                    split_name=split_name,
                    thresholds=thresholds,
                )
            )
        val_mask = split == 1
        val_score = (
            average_precision(data.labels[val_mask], scores[val_mask])
            if np.any(val_mask)
            else average_precision(data.labels, scores)
        )
        if val_score > best_score:
            best_score = val_score
            best_name = name
            best_model = model
            predictions_for_best = scores

    if best_model is None or predictions_for_best is None:
        raise ValueError("no model could be trained")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "safe_selector_metrics.csv", metrics)
    write_csv(output_dir / "safe_selector_thresholds.csv", threshold_output)
    prediction_rows = [
        {
            "row_index": index,
            "split": ("train" if split[index] == 0 else "val" if split[index] == 1 else "test"),
            "label": int(data.labels[index]),
            "realized_delta": float(data.deltas[index]),
            "teacher_regret": float(data.regrets[index]),
            "safe_probability": float(predictions_for_best[index]),
            "seat": data.rows[index].get("seat", ""),
            "margin": safe_float(data.rows[index].get("hu_turn1_predicted_margin", data.rows[index].get("runtime_predicted_margin"))),
        }
        for index in range(len(data.rows))
    ]
    write_csv(output_dir / "safe_selector_predictions.csv", prediction_rows)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("wb") as handle:
        pickle.dump(
            {
                "model_kind": "hu_turn1_safe_override_selector_sklearn",
                "model_name": best_name,
                "feature_mode": data.feature_mode,
                "label_mode": data.label_mode,
                "feature_dim": int(data.features.shape[1]),
                "meta_feature_names": META_FEATURE_NAMES,
                "base_hu_feature_dim": HU_FEATURE_DIM,
                "estimator": best_model,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    summary = {
        "rows": len(data.rows),
        "positives": int(np.sum(data.labels == 1)),
        "negatives": int(np.sum(data.labels == 0)),
        "feature_mode": data.feature_mode,
        "label_mode": data.label_mode,
        "feature_dim": int(data.features.shape[1]),
        "best_model": best_name,
        "best_val_average_precision": best_score,
        "model_output": str(model_output),
        "metrics": str(output_dir / "safe_selector_metrics.csv"),
        "thresholds": str(output_dir / "safe_selector_thresholds.csv"),
        "predictions": str(output_dir / "safe_selector_predictions.csv"),
    }
    (output_dir / "safe_selector_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_markdown(output_dir / "safe_selector_summary.md", summary, metrics)
    return summary


def _fit_kwargs(model: Any, weights: np.ndarray) -> dict[str, Any]:
    # Pipelines need step-qualified fit parameters.
    if hasattr(model, "steps"):
        final_name = model.steps[-1][0]
        return {f"{final_name}__sample_weight": weights}
    return {"sample_weight": weights}


def _predict_probability(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
    decision = np.asarray(model.decision_function(features), dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-decision))


def write_summary_markdown(path: Path, summary: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    lines = [
        "# HU Turn1 Safe Override Selector Smoke",
        "",
        f"- rows: `{summary['rows']}`",
        f"- positives: `{summary['positives']}`",
        f"- negatives: `{summary['negatives']}`",
        f"- feature mode: `{summary['feature_mode']}`",
        f"- label mode: `{summary['label_mode']}`",
        f"- best model: `{summary['best_model']}`",
        f"- best val AP: `{summary['best_val_average_precision']:.4f}`",
        "",
        "This is a diagnostic smoke model, not a runtime approval.",
        "",
        "## Metrics",
        "",
        "| model | split | rows | positives | AP | AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['model']} | {row['split']} | {row['rows']} | {row['positives']} | "
            f"{row['average_precision']:.4f} | {row['roc_auc']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Training JSONL. Repeat to combine independent seat/source datasets.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument(
        "--feature-mode",
        choices=(
            "delta_plus_meta",
            "candidate_delta_plus_meta",
            "candidate_delta_plus_runtime_meta",
            "delta_plus_runtime_meta",
            "meta_only",
        ),
        default="delta_plus_meta",
    )
    parser.add_argument("--exclude-neutral", action="store_true")
    parser.add_argument(
        "--include-non-fired-candidates",
        action="store_true",
        help="Also train on gated-off rows that have a concrete candidate action and realized counterfactual delta.",
    )
    parser.add_argument(
        "--label-mode",
        choices=(
            "realized_delta",
            "stage10_mc32_delta",
            "teacher_delta_thresholded",
            "high_mc_delta_thresholded",
            "teacher_regret",
            "accept_gray_delta",
            "teacher_regret_or_accept_gray_delta",
        ),
        default="realized_delta",
    )
    parser.add_argument("--teacher-accept-regret", type=float, default=0.25)
    parser.add_argument("--teacher-gray-regret", type=float, default=1.0)
    parser.add_argument("--accept-delta", type=float, default=2.0)
    parser.add_argument("--hard-negative-delta", type=float, default=-5.0)
    parser.add_argument("--gray-weight", type=float, default=0.35)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    rows = [row for path in args.input for row in read_jsonl(path)]
    data = build_training_data(
        rows,
        feature_mode=args.feature_mode,
        include_neutral=not args.exclude_neutral,
        include_non_fired_candidates=args.include_non_fired_candidates,
        label_mode=args.label_mode,
        teacher_accept_regret=args.teacher_accept_regret,
        teacher_gray_regret=args.teacher_gray_regret,
        accept_delta=args.accept_delta,
        hard_negative_delta=args.hard_negative_delta,
        gray_weight=args.gray_weight,
    )
    summary = train_selector(data, output_dir=args.output_dir, model_output=args.model_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
