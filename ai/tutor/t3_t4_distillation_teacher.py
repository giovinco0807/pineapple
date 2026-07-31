"""Content-addressed, leakage-safe T3/T4 distillation teacher contract.

This is an opt-in artifact contract for bounded, in-memory teacher batches.  It
does not run MCCFR, train a model, change serving, or promote a policy.  Split
assignment is derived only from an immutable pre-augmentation full-deal
commitment before any label is accepted.  Descendants, private types, solver
replicates, seat swaps, and suit augmentations carry that same assignment.

Only an information-safe ``InfoSetKey`` and its lossless fixed-schema vector
enter ``model_input``.  Hidden worlds may be represented elsewhere only by
lowercase SHA-256 commitments.  Standard non-Fantasyland OFC is the sole
accepted scope: ``fantasy_state`` must be ``None``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ai.tutor.t3_hu_public_cfr import FORBIDDEN_INFOSET_FIELDS, InfoSetKey
from ai.tutor.t3_t4_infoset_encoder import (
    ACTION_SEMANTICS_SHA256,
    INFOSET_ENCODER_MANIFEST_SHA256,
    INFOSET_ENCODER_SCHEMA,
    INFOSET_VECTOR_DIM,
    decode_infoset_key,
    encode_infoset_key,
    legal_action_mask,
    semantic_action_ids,
)
from ai.tutor.runtime_semantic_anchor import register_module_function_anchor
from ai.tutor.t4_btn_exact_resolver import (
    T4_BTN_EXACT_METHOD,
    T4BtnExactResolution,
    T4BtnExactResolveError,
    resolve_t4_second_btn_exact,
    verify_t4_second_btn_exact,
)


SPLIT_ASSIGNMENT_SCHEMA = "ofc_t3_t4_teacher_split_assignment/v1"
MODEL_INPUT_SCHEMA = "ofc_t3_t4_teacher_model_input/v1"
TEACHER_ROW_SCHEMA = "ofc_t3_t4_distillation_teacher_row/v2"
TEACHER_SHARD_SCHEMA = "ofc_t3_t4_distillation_teacher_shard/v2"
TEACHER_SHARD_ENTRY_SCHEMA = "ofc_t3_t4_distillation_teacher_shard_entry/v2"
TEACHER_BUNDLE_SCHEMA = "ofc_t3_t4_distillation_teacher_bundle/v2"

SPLIT_NAMESPACE = "ofc_t3_t4_distillation_full_deal_split/v1"
SOLVER_SEED_NAMESPACE = "ofc_t3_t4_distillation_solver_seed/v1"
PAYOFF_SEED_NAMESPACE = "ofc_t3_t4_distillation_payoff_seed/v1"
MCCFR_SOLVER_METHOD = "external_sampling_mccfr_average_strategy_v1"
SOLVER_METHOD_CONTRACT_IDS = (MCCFR_SOLVER_METHOD, T4_BTN_EXACT_METHOD)
SPLIT_NAMES = ("fit", "dev", "test")
SPLIT_INVARIANTS = (
    "descendants",
    "private_types",
    "solver_seeds",
    "payoff_seeds",
    "seat_swaps",
    "suit_augmentations",
)

Q32_DENOMINATOR = 1 << 32
MAX_ROWS_PER_SHARD = 256
MAX_IN_MEMORY_ROWS = 4096
MANIFEST_NAME = "teacher-manifest.json"
_FLOAT_TOL = 1e-12

_COMMON_BINDING_HASH_FIELDS = frozenset(
    {
        "public_root_commitment_sha256",
        "public_root_mixture_sha256",
        "range_content_sha256",
        "range_build_sha256",
        "behavior_model_sha256",
        "source_manifest_sha256",
    }
)
_MCCFR_BINDING_HASH_FIELDS = frozenset(
    {
        *_COMMON_BINDING_HASH_FIELDS,
        "solver_source_sha256",
        "solver_config_sha256",
        "solver_checkpoint_sha256",
    }
)
_LINEAGE_HASH_FIELDS = frozenset(
    {
        "descendant_public_path_sha256",
        "restricted_variant_root_commitment_sha256",
        "restricted_private_type_commitment_sha256",
    }
)
_OVERLAP_BINDINGS = (
    "public_root_commitment_sha256",
    "public_root_mixture_sha256",
    "range_content_sha256",
    "range_build_sha256",
)


class TeacherContractError(ValueError):
    """A teacher row or immutable bundle failed closed."""


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TeacherContractError("value is not canonical-JSON serializable") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def self_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return canonical_sha256(unsigned)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TeacherContractError(f"{label}: object required")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise TeacherContractError(
            f"{label}: exact fields required; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TeacherContractError(f"{label}: lowercase SHA-256 required")
    return value


def _integer(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TeacherContractError(f"{label}: integer >= {minimum} required")
    if maximum is not None and value > maximum:
        raise TeacherContractError(f"{label}: integer <= {maximum} required")
    return value


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeacherContractError(f"{label}: finite number required")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise TeacherContractError(f"{label}: invalid finite number")
    return 0.0 if result == 0.0 else result


def _raw_moment_mean_and_standard_error(
    *,
    count: int,
    total: float,
    squares: float,
    label: str,
) -> tuple[float, float]:
    """Derive raw-moment statistics without overflowing ``total**2``.

    ``Fraction.from_float`` preserves the exact finite binary inputs while
    keeping the Cauchy lower bound ``sum_squares >= sum**2 / count`` in an
    unbounded rational domain.  The relative tolerance retains the previous
    allowance for ordinary floating-point accumulation error.
    """

    exact_total = Fraction.from_float(total)
    exact_squares = Fraction.from_float(squares)
    minimum_squares = exact_total * exact_total / count
    scale = max(Fraction(1, 1), abs(exact_squares), abs(minimum_squares))
    tolerance = Fraction.from_float(_FLOAT_TOL) * scale
    if exact_squares + tolerance < minimum_squares:
        raise TeacherContractError(f"{label}: impossible raw moments")
    centered = max(Fraction(0, 1), exact_squares - minimum_squares)
    variance_of_mean = centered / (count - 1) / count
    return total / count, math.sqrt(float(variance_of_mean))


def _exact_fraction(value: Fraction | int | str, *, label: str) -> Fraction:
    if isinstance(value, (bool, float)):
        raise TeacherContractError(f"{label}: exact Fraction/int/str required")
    try:
        result = value if isinstance(value, Fraction) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise TeacherContractError(f"{label}: invalid exact rational") from exc
    return result


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _split_group_sha256(full_deal_commitment_sha256: str) -> str:
    return canonical_sha256(
        {
            "schema": "ofc_t3_t4_distillation_split_group/v1",
            "namespace": SPLIT_NAMESPACE,
            "full_deal_commitment_sha256": full_deal_commitment_sha256,
        }
    )


def _split_bucket(full_deal_commitment_sha256: str) -> int:
    material = (SPLIT_NAMESPACE + full_deal_commitment_sha256).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % 10_000


def _split_name(bucket: int) -> str:
    if bucket < 7_000:
        return "fit"
    if bucket < 8_500:
        return "dev"
    return "test"


def build_split_assignment(
    *,
    full_deal_commitment_sha256: str,
    public_root_family_commitment_sha256: str,
) -> dict[str, Any]:
    """Assign a locked split using only a pre-label full-deal commitment."""
    full_deal = _sha(
        full_deal_commitment_sha256,
        label="full_deal_commitment_sha256",
    )
    root_family = _sha(
        public_root_family_commitment_sha256,
        label="public_root_family_commitment_sha256",
    )
    bucket = _split_bucket(full_deal)
    assignment: dict[str, Any] = {
        "schema": SPLIT_ASSIGNMENT_SCHEMA,
        "namespace": SPLIT_NAMESPACE,
        "assigned_before_labels": True,
        "split_material": "original_pre_augmentation_full_deal_commitment_sha256",
        "full_deal_commitment_sha256": full_deal,
        "public_root_family_commitment_sha256": root_family,
        "split_group_sha256": _split_group_sha256(full_deal),
        "split_bucket": bucket,
        "split": _split_name(bucket),
        "invariant_axes": list(SPLIT_INVARIANTS),
    }
    assignment["assignment_sha256"] = self_hash(
        assignment,
        "assignment_sha256",
    )
    return assignment


_SPLIT_ASSIGNMENT_KEYS = {
    "schema",
    "namespace",
    "assigned_before_labels",
    "split_material",
    "full_deal_commitment_sha256",
    "public_root_family_commitment_sha256",
    "split_group_sha256",
    "split_bucket",
    "split",
    "invariant_axes",
    "assignment_sha256",
}


def verify_split_assignment(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _require_mapping(value, label="split_assignment")
    _exact_keys(raw, _SPLIT_ASSIGNMENT_KEYS, label="split_assignment")
    expected = build_split_assignment(
        full_deal_commitment_sha256=_sha(
            raw.get("full_deal_commitment_sha256"),
            label="split_assignment.full_deal_commitment_sha256",
        ),
        public_root_family_commitment_sha256=_sha(
            raw.get("public_root_family_commitment_sha256"),
            label="split_assignment.public_root_family_commitment_sha256",
        ),
    )
    if dict(raw) != expected:
        raise TeacherContractError("split_assignment: stale or label-dependent assignment")
    return expected


def _derive_seed(split_group_sha256: str, *, namespace: str, index: int, parity: int) -> int:
    _sha(split_group_sha256, label="split_group_sha256")
    _integer(index, label="seed index", maximum=(1 << 31) - 1)
    digest = hashlib.sha256(
        canonical_json(
            {
                "namespace": namespace,
                "split_group_sha256": split_group_sha256,
                "index": index,
            }
        ).encode("utf-8")
    ).digest()
    base = int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 62) - 1)
    return (base << 1) | parity


def derive_solver_seed(split_group_sha256: str, index: int) -> int:
    return _derive_seed(
        split_group_sha256,
        namespace=SOLVER_SEED_NAMESPACE,
        index=index,
        parity=0,
    )


def derive_payoff_seed(split_group_sha256: str, index: int) -> int:
    return _derive_seed(
        split_group_sha256,
        namespace=PAYOFF_SEED_NAMESPACE,
        index=index,
        parity=1,
    )


def _seed_plan(
    assignment: Mapping[str, Any],
    *,
    solver_seed_index: int,
    payoff_seed_index: int,
) -> dict[str, Any]:
    solver_index = _integer(
        solver_seed_index,
        label="solver_seed_index",
        maximum=(1 << 31) - 1,
    )
    payoff_index = _integer(
        payoff_seed_index,
        label="payoff_seed_index",
        maximum=(1 << 31) - 1,
    )
    group = assignment["split_group_sha256"]
    return {
        "solver": {
            "namespace": SOLVER_SEED_NAMESPACE,
            "index": solver_index,
            "seed": derive_solver_seed(group, solver_index),
        },
        "payoff": {
            "namespace": PAYOFF_SEED_NAMESPACE,
            "index": payoff_index,
            "seed": derive_payoff_seed(group, payoff_index),
        },
        "numeric_seed_sets_disjoint": True,
    }


def _validate_hash_bindings(
    value: Mapping[str, Any],
    *,
    fields: frozenset[str] = _MCCFR_BINDING_HASH_FIELDS,
) -> dict[str, str]:
    raw = _require_mapping(value, label="bindings")
    _exact_keys(raw, set(fields), label="bindings")
    return {
        field: _sha(raw[field], label=f"bindings.{field}")
        for field in sorted(fields)
    }


def _validate_lineage(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _require_mapping(value, label="lineage")
    expected = set(_LINEAGE_HASH_FIELDS) | {
        "seat_swap_index",
        "suit_augmentation_index",
    }
    _exact_keys(raw, expected, label="lineage")
    lineage: dict[str, Any] = {
        field: _sha(raw[field], label=f"lineage.{field}")
        for field in sorted(_LINEAGE_HASH_FIELDS)
    }
    lineage["seat_swap_index"] = _integer(
        raw["seat_swap_index"],
        label="lineage.seat_swap_index",
        maximum=1,
    )
    lineage["suit_augmentation_index"] = _integer(
        raw["suit_augmentation_index"],
        label="lineage.suit_augmentation_index",
        maximum=23,
    )
    return lineage


def _visible_joker_count(key: InfoSetKey) -> int:
    visible: set[str] = set(key.current_draw)
    for board in (key.board_bb, key.board_btn):
        for row in board:
            visible.update(row)
    for _turn, cards in key.own_recall.dealt_by_turn:
        visible.update(cards)
    visible.update(card for _turn, card in key.own_recall.discards_by_turn)
    return len(visible & {"X1", "X2"})


def _model_input(key: InfoSetKey) -> dict[str, Any]:
    if key.fantasy_state is not None:
        raise TeacherContractError(
            "teacher model_input supports standard OFC only; fantasy_state must be None"
        )
    encoded = encode_infoset_key(key)
    vector = [int(value) for value in encoded]
    model_input: dict[str, Any] = {
        "schema": MODEL_INPUT_SCHEMA,
        "information_canonical_json": key.canonical_json(),
        "information_digest": key.digest(),
        "encoder_schema": INFOSET_ENCODER_SCHEMA,
        "encoder_manifest_sha256": INFOSET_ENCODER_MANIFEST_SHA256,
        "encoded_vector_dimension": INFOSET_VECTOR_DIM,
        "encoded_vector": vector,
        "encoded_vector_sha256": canonical_sha256(vector),
        "opponent_hidden_cards_included": False,
        "restricted_commitments_in_input": False,
    }
    model_input["model_input_sha256"] = self_hash(
        model_input,
        "model_input_sha256",
    )
    return model_input


_MODEL_INPUT_KEYS = {
    "schema",
    "information_canonical_json",
    "information_digest",
    "encoder_schema",
    "encoder_manifest_sha256",
    "encoded_vector_dimension",
    "encoded_vector",
    "encoded_vector_sha256",
    "opponent_hidden_cards_included",
    "restricted_commitments_in_input",
    "model_input_sha256",
}


def _verify_model_input(value: Mapping[str, Any]) -> tuple[dict[str, Any], InfoSetKey]:
    raw = _require_mapping(value, label="model_input")
    _exact_keys(raw, _MODEL_INPUT_KEYS, label="model_input")
    fixed = {
        "schema": MODEL_INPUT_SCHEMA,
        "encoder_schema": INFOSET_ENCODER_SCHEMA,
        "encoder_manifest_sha256": INFOSET_ENCODER_MANIFEST_SHA256,
        "encoded_vector_dimension": INFOSET_VECTOR_DIM,
        "opponent_hidden_cards_included": False,
        "restricted_commitments_in_input": False,
    }
    for field, expected in fixed.items():
        if raw.get(field) != expected:
            raise TeacherContractError(f"model_input.{field}: mismatch")
    vector = raw.get("encoded_vector")
    if (
        not isinstance(vector, list)
        or len(vector) != INFOSET_VECTOR_DIM
        or any(isinstance(bit, bool) or bit not in (0, 1) for bit in vector)
    ):
        raise TeacherContractError("model_input.encoded_vector: exact binary vector required")
    if raw.get("encoded_vector_sha256") != canonical_sha256(vector):
        raise TeacherContractError("model_input.encoded_vector: SHA-256 mismatch")
    try:
        key = decode_infoset_key(np.asarray(vector, dtype=np.float32))
    except (TypeError, ValueError) as exc:
        raise TeacherContractError(
            f"model_input: invalid fixed-schema vector ({exc})"
        ) from exc
    if key.fantasy_state is not None:
        raise TeacherContractError("model_input: fantasy_state must be None")
    canonical = raw.get("information_canonical_json")
    if not isinstance(canonical, str) or canonical != key.canonical_json():
        raise TeacherContractError("model_input: canonical information mismatch")
    if raw.get("information_digest") != key.digest():
        raise TeacherContractError("model_input: information digest mismatch")
    if raw.get("model_input_sha256") != self_hash(raw, "model_input_sha256"):
        raise TeacherContractError("model_input: self-hash mismatch")
    serialized = canonical_json(raw).lower()
    leaked = [
        field
        for field in FORBIDDEN_INFOSET_FIELDS
        if f'"{field}"' in serialized
    ]
    if leaked:
        raise TeacherContractError(f"model_input: forbidden hidden fields {sorted(leaked)}")
    return dict(raw), key


def _strategy_weight(value: Any, *, label: str) -> Fraction:
    if isinstance(value, bool):
        raise TeacherContractError(f"{label}: nonnegative numeric weight required")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TeacherContractError(f"{label}: finite weight required")
        result = Fraction(str(value))
    elif isinstance(value, (Fraction, int, str)):
        try:
            result = value if isinstance(value, Fraction) else Fraction(value)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise TeacherContractError(f"{label}: invalid strategy weight") from exc
    else:
        raise TeacherContractError(f"{label}: numeric weight required")
    if result < 0:
        raise TeacherContractError(f"{label}: nonnegative weight required")
    return result


def quantize_average_strategy_q32(
    key: InfoSetKey,
    strategy_by_action_id: Mapping[str, Any],
) -> tuple[int, ...]:
    """Largest-remainder exact-Q32 quantization in semantic-index order."""
    raw = _require_mapping(strategy_by_action_id, label="average_strategy")
    action_ids = semantic_action_ids(key)
    legal_ids = {action_id for action_id in action_ids if action_id is not None}
    if set(raw) != legal_ids:
        raise TeacherContractError("average_strategy: exact legal action support required")
    weights = {
        action_id: _strategy_weight(
            raw[action_id],
            label=f"average_strategy.{action_id}",
        )
        for action_id in legal_ids
    }
    total = sum(weights.values(), Fraction(0, 1))
    if total <= 0:
        raise TeacherContractError("average_strategy: positive total weight required")
    floors = [0] * len(action_ids)
    remainders: list[tuple[Fraction, int]] = []
    for index, action_id in enumerate(action_ids):
        if action_id is None:
            continue
        scaled = weights[action_id] * Q32_DENOMINATOR / total
        floor = scaled.numerator // scaled.denominator
        floors[index] = floor
        remainders.append((scaled - floor, index))
    remaining = Q32_DENOMINATOR - sum(floors)
    for _remainder, index in sorted(
        remainders,
        key=lambda item: (-item[0], item[1]),
    )[:remaining]:
        floors[index] += 1
    if sum(floors) != Q32_DENOMINATOR:
        raise AssertionError("Q32 strategy quantization did not conserve mass")
    return tuple(floors)


def _action_statistics(
    key: InfoSetKey,
    moments_by_action_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any] | None]:
    raw = _require_mapping(moments_by_action_id, label="action_payoff_moments")
    action_ids = semantic_action_ids(key)
    legal_ids = {action_id for action_id in action_ids if action_id is not None}
    if set(raw) != legal_ids:
        raise TeacherContractError(
            "action_payoff_moments: exact legal action support required"
        )
    output: list[dict[str, Any] | None] = []
    for action_id in action_ids:
        if action_id is None:
            output.append(None)
            continue
        moment = _require_mapping(
            raw[action_id],
            label=f"action_payoff_moments.{action_id}",
        )
        _exact_keys(
            moment,
            {"count", "sum", "sum_squares"},
            label=f"action_payoff_moments.{action_id}",
        )
        count = _integer(
            moment.get("count"),
            label=f"action_payoff_moments.{action_id}.count",
            minimum=2,
        )
        total = _finite(
            moment.get("sum"),
            label=f"action_payoff_moments.{action_id}.sum",
        )
        squares = _finite(
            moment.get("sum_squares"),
            label=f"action_payoff_moments.{action_id}.sum_squares",
            nonnegative=True,
        )
        mean, standard_error = _raw_moment_mean_and_standard_error(
            count=count,
            total=total,
            squares=squares,
            label=f"action_payoff_moments.{action_id}",
        )
        output.append(
            {
                "action_id": action_id,
                "count": count,
                "sum": total,
                "sum_squares": squares,
                "mean": mean,
                "standard_error": standard_error,
            }
        )
    return output


def _target_payload(
    key: InfoSetKey,
    *,
    average_strategy_by_action_id: Mapping[str, Any],
    action_payoff_moments_by_action_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    strategy = quantize_average_strategy_q32(
        key,
        average_strategy_by_action_id,
    )
    statistics = _action_statistics(key, action_payoff_moments_by_action_id)
    means = [None if row is None else float(row["mean"]) for row in statistics]
    legal_indices = [index for index, mean in enumerate(means) if mean is not None]
    best_index = max(legal_indices, key=lambda index: (means[index], -index))
    policy_value = math.fsum(
        strategy[index] / Q32_DENOMINATOR * float(means[index])
        for index in legal_indices
    )
    best_value = float(means[best_index])
    regret = max(0.0, best_value - policy_value)
    action_ids = semantic_action_ids(key)
    return {
        "average_strategy_quantization": "largest_remainder_exact_q32",
        "average_strategy_denominator": Q32_DENOMINATOR,
        "average_strategy_q32": list(strategy),
        "payoff_perspective": "acting_player",
        "action_payoff_semantics": (
            "one_step_action_then_frozen_teacher_average_strategy_continuation"
        ),
        "policy_value_uses_quantized_average_strategy": True,
        "action_payoff_statistics": statistics,
        "policy_value_estimate": policy_value,
        "best_action_semantic_index": best_index,
        "best_action_id": action_ids[best_index],
        "best_action_value_estimate": best_value,
        "one_step_deviation_regret_estimate": regret,
        "exact_exploitability_computed": False,
    }


def _quality_payload(
    targets: Mapping[str, Any],
    *,
    infoset_visit_count: int,
    infoset_reach_probability: Fraction | int | str,
) -> dict[str, Any]:
    visits = _integer(
        infoset_visit_count,
        label="infoset_visit_count",
    )
    reach = _exact_fraction(
        infoset_reach_probability,
        label="infoset_reach_probability",
    )
    if not 0 <= reach <= 1:
        raise TeacherContractError("infoset_reach_probability: [0,1] required")
    strategy = targets["average_strategy_q32"]
    statistics = [
        row for row in targets["action_payoff_statistics"] if row is not None
    ]
    support = sum(weight > 0 for weight in strategy)
    min_samples = min(int(row["count"]) for row in statistics)
    max_error = max(float(row["standard_error"]) for row in statistics)
    structural = visits > 0 and reach > 0 and support > 0 and min_samples >= 2
    return {
        "infoset_visit_count": visits,
        "infoset_reach_probability_exact": _fraction_text(reach),
        "legal_action_count": len(statistics),
        "average_strategy_support_count": support,
        "min_action_payoff_sample_count": min_samples,
        "max_action_payoff_standard_error": max_error,
        "all_legal_actions_sampled": True,
        "structural_quality_passed": structural,
        "quality_status": "structural_pass" if structural else "structural_fail",
        "promotion_quality_gate_evaluated": False,
    }


def _strict_float_hex(value: Any, *, label: str) -> str:
    if type(value) is not float or not math.isfinite(value):
        raise TeacherContractError(f"{label}: finite float required")
    return value.hex()


def _exact_resolution_sha256(resolution: T4BtnExactResolution) -> str:
    utility = [
        {
            "action_id": action_id,
            "utility_float_hex": _strict_float_hex(
                value,
                label=f"exact resolution utility {action_id}",
            ),
        }
        for action_id, value in sorted(resolution.utility_by_action_id.items())
    ]
    policy = [
        {
            "action_id": action_id,
            "probability_float_hex": _strict_float_hex(
                value,
                label=f"exact resolution policy {action_id}",
            ),
        }
        for action_id, value in sorted(resolution.action_probabilities.items())
    ]
    metrics = {
        action_id: dict(row)
        for action_id, row in sorted(resolution.terminal_metrics_by_action_id.items())
    }
    return canonical_sha256(
        {
            "schema": "ofc_t4_btn_exact_teacher_resolution_binding/v1",
            "infoset_sha256": resolution.information.digest(),
            "utility": utility,
            "terminal_metrics_sha256": canonical_sha256(metrics),
            "optimal_action_ids": list(resolution.optimal_action_ids),
            "selected_action_id": resolution.selected_action_id,
            "policy": policy,
            "resolver_manifest_sha256": resolution.manifest_sha256,
        }
    )


def _exact_solver_payload(resolution: T4BtnExactResolution) -> dict[str, Any]:
    return {
        "method": T4_BTN_EXACT_METHOD,
        "policy_source": "exact_terminal_argmax",
        "payoff_source": "exact_terminal_utility",
        "seed_usage": "none",
        "resolver_manifest_sha256": resolution.manifest_sha256,
        "resolver_result_sha256": _exact_resolution_sha256(resolution),
        "label_authenticity_scope": "fresh_resolver_reexecution_and_manifest_binding",
    }


def _exact_target_payload(
    key: InfoSetKey,
    resolution: T4BtnExactResolution,
) -> dict[str, Any]:
    action_ids = semantic_action_ids(key)
    policy_q32: list[int] = []
    utilities: list[dict[str, Any] | None] = []
    for action_id in action_ids:
        if action_id is None:
            policy_q32.append(0)
            utilities.append(None)
            continue
        probability = resolution.action_probabilities[action_id]
        if probability not in (0.0, 1.0):
            raise TeacherContractError("exact resolver policy must be deterministic")
        policy_q32.append(Q32_DENOMINATOR if probability == 1.0 else 0)
        utilities.append(
            {
                "action_id": action_id,
                "utility": resolution.utility_by_action_id[action_id],
            }
        )
    if sum(policy_q32) != Q32_DENOMINATOR:
        raise TeacherContractError("exact resolver policy must select exactly one action")
    selected_index = action_ids.index(resolution.selected_action_id)
    return {
        "policy_semantics": "exact_terminal_argmax",
        "policy_denominator": Q32_DENOMINATOR,
        "exact_argmax_policy_q32": policy_q32,
        "payoff_perspective": "acting_player",
        "action_payoff_semantics": "exact_terminal_utility",
        "terminal_utility_by_action": utilities,
        "selected_action_semantic_index": selected_index,
        "selected_action_id": resolution.selected_action_id,
        "selected_action_utility": resolution.utility_by_action_id[
            resolution.selected_action_id
        ],
        "optimal_action_ids": list(resolution.optimal_action_ids),
        "tie_break": "canonical_action_id_ascending",
        "all_legal_actions_enumerated": True,
        "exact_exploitability_computed": False,
    }


def _exact_quality_payload(resolution: T4BtnExactResolution) -> dict[str, Any]:
    return {
        "legal_action_count": len(resolution.utility_by_action_id),
        "exact_terminal_result_reverified": True,
        "all_legal_actions_evaluated_exactly": True,
        "structural_quality_passed": True,
        "quality_status": "exact_terminal_verified",
        "promotion_quality_gate_evaluated": False,
    }


def _row_identity(
    *,
    assignment: Mapping[str, Any],
    lineage: Mapping[str, Any],
    model_input: Mapping[str, Any],
    bindings: Mapping[str, Any],
    solver: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "schema": "ofc_t3_t4_teacher_row_identity/v2",
            "split_assignment_sha256": assignment["assignment_sha256"],
            "lineage": dict(lineage),
            "model_input_sha256": model_input["model_input_sha256"],
            "bindings": dict(bindings),
            "solver": dict(solver),
        }
    )


def _assemble_teacher_row(
    information: InfoSetKey,
    *,
    assignment: Mapping[str, Any],
    lineage: Mapping[str, Any],
    bindings: Mapping[str, Any],
    solver: Mapping[str, Any],
    targets: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    model_input = _model_input(information)
    action_ids = semantic_action_ids(information)
    action_mask = legal_action_mask(information)
    joker_count = _visible_joker_count(information)
    row: dict[str, Any] = {
        "schema": TEACHER_ROW_SCHEMA,
        "artifact_kind": "algorithm_teacher_row_for_generalization_distillation",
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "global_policy_claimed": False,
        "training_performed": False,
        "serving_changed": False,
        "solver_checkpoint_replayed": False,
        "opponent_hidden_cards_in_model_input": False,
        "restricted_hidden_information_is_commitment_only": True,
        "split_assignment": dict(assignment),
        "lineage": dict(lineage),
        "bindings": dict(bindings),
        "model_input": model_input,
        "stratum": {
            "phase": information.phase,
            "turn": information.turn,
            "actor": information.actor,
            "visible_joker_count": joker_count,
            "joker_stratum": f"joker_{joker_count}",
        },
        "solver": dict(solver),
        "action_contract": {
            "semantic_action_count": 27,
            "action_semantics_sha256": ACTION_SEMANTICS_SHA256,
            "semantic_action_ids": list(action_ids),
            "legal_action_mask": [bool(value) for value in action_mask],
        },
        "targets": dict(targets),
        "quality": dict(quality),
    }
    row["row_identity_sha256"] = _row_identity(
        assignment=assignment,
        lineage=lineage,
        model_input=model_input,
        bindings=bindings,
        solver=solver,
    )
    row["row_sha256"] = self_hash(row, "row_sha256")
    return verify_teacher_row(row)


def build_teacher_row(
    information: InfoSetKey,
    *,
    split_assignment: Mapping[str, Any],
    lineage: Mapping[str, Any],
    bindings: Mapping[str, Any],
    solver_method: str,
    solver_iterations_completed: int,
    solver_seed_index: int,
    payoff_seed_index: int,
    average_strategy_by_action_id: Mapping[str, Any],
    action_payoff_moments_by_action_id: Mapping[str, Mapping[str, Any]],
    infoset_visit_count: int,
    infoset_reach_probability: Fraction | int | str,
) -> dict[str, Any]:
    """Build one sampled MCCFR teacher row.

    Exact T4 BTN rows use :func:`build_t4_btn_exact_teacher_row`; accepting
    exact labels through this sampled interface would fabricate iterations,
    seed consumption, and payoff samples that the terminal resolver never used.
    """
    if not isinstance(information, InfoSetKey):
        raise TypeError("information must be an InfoSetKey")
    if information.fantasy_state is not None:
        raise TeacherContractError("teacher rows require fantasy_state=None")
    assignment = verify_split_assignment(split_assignment)
    canonical_lineage = _validate_lineage(lineage)
    canonical_bindings = _validate_hash_bindings(bindings)
    if not isinstance(solver_method, str) or solver_method not in SOLVER_METHOD_CONTRACT_IDS:
        raise TeacherContractError(
            "solver_method: known contract ID required; "
            f"allowed={list(SOLVER_METHOD_CONTRACT_IDS)}"
        )
    if solver_method != MCCFR_SOLVER_METHOD:
        raise TeacherContractError(
            "solver_method: exact terminal labels require "
            "build_t4_btn_exact_teacher_row"
        )
    iterations = _integer(
        solver_iterations_completed,
        label="solver_iterations_completed",
        minimum=1,
    )
    seeds = _seed_plan(
        assignment,
        solver_seed_index=solver_seed_index,
        payoff_seed_index=payoff_seed_index,
    )
    targets = _target_payload(
        information,
        average_strategy_by_action_id=average_strategy_by_action_id,
        action_payoff_moments_by_action_id=action_payoff_moments_by_action_id,
    )
    quality = _quality_payload(
        targets,
        infoset_visit_count=infoset_visit_count,
        infoset_reach_probability=infoset_reach_probability,
    )
    solver = {
        "method": solver_method,
        "iterations_completed": iterations,
        "seed_plan": seeds,
        "average_strategy_source": "mccfr_average_strategy",
        "payoff_source": "independent_payoff_seed_namespace",
        "label_authenticity_scope": "content_binding_not_solver_reexecution",
    }
    return _assemble_teacher_row(
        information,
        assignment=assignment,
        lineage=canonical_lineage,
        bindings=canonical_bindings,
        solver=solver,
        targets=targets,
        quality=quality,
    )


def build_t4_btn_exact_teacher_row(
    information: InfoSetKey,
    *,
    resolution: T4BtnExactResolution,
    split_assignment: Mapping[str, Any],
    lineage: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt one freshly reverified T4 BTN exact result without sampled provenance."""

    if not isinstance(information, InfoSetKey):
        raise TypeError("information must be an InfoSetKey")
    if not isinstance(resolution, T4BtnExactResolution):
        raise TypeError("resolution must be T4BtnExactResolution")
    if information.fantasy_state is not None:
        raise TeacherContractError("teacher rows require fantasy_state=None")
    if resolution.information != information:
        raise TeacherContractError("exact resolution infoset does not match teacher input")
    try:
        verify_t4_second_btn_exact(
            information,
            resolution,
            expected_manifest_sha256=resolution.manifest_sha256,
        )
    except (T4BtnExactResolveError, TypeError, ValueError) as exc:
        raise TeacherContractError(f"exact resolver result verification failed: {exc}") from exc
    assignment = verify_split_assignment(split_assignment)
    canonical_lineage = _validate_lineage(lineage)
    canonical_bindings = _validate_hash_bindings(
        bindings,
        fields=_COMMON_BINDING_HASH_FIELDS,
    )
    return _assemble_teacher_row(
        information,
        assignment=assignment,
        lineage=canonical_lineage,
        bindings=canonical_bindings,
        solver=_exact_solver_payload(resolution),
        targets=_exact_target_payload(information, resolution),
        quality=_exact_quality_payload(resolution),
    )


_TEACHER_ROW_KEYS = {
    "schema",
    "artifact_kind",
    "algorithm_teacher_only",
    "promotion_eligible",
    "global_policy_claimed",
    "training_performed",
    "serving_changed",
    "solver_checkpoint_replayed",
    "opponent_hidden_cards_in_model_input",
    "restricted_hidden_information_is_commitment_only",
    "split_assignment",
    "lineage",
    "bindings",
    "model_input",
    "stratum",
    "solver",
    "action_contract",
    "targets",
    "quality",
    "row_identity_sha256",
    "row_sha256",
}


def _verify_seed_plan(
    value: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _require_mapping(value, label="seed_plan")
    _exact_keys(
        raw,
        {"solver", "payoff", "numeric_seed_sets_disjoint"},
        label="seed_plan",
    )
    for role in ("solver", "payoff"):
        part = _require_mapping(raw.get(role), label=f"seed_plan.{role}")
        _exact_keys(part, {"namespace", "index", "seed"}, label=f"seed_plan.{role}")
    solver_index = _integer(
        raw["solver"].get("index"),
        label="seed_plan.solver.index",
        maximum=(1 << 31) - 1,
    )
    payoff_index = _integer(
        raw["payoff"].get("index"),
        label="seed_plan.payoff.index",
        maximum=(1 << 31) - 1,
    )
    expected = _seed_plan(
        assignment,
        solver_seed_index=solver_index,
        payoff_seed_index=payoff_index,
    )
    if dict(raw) != expected:
        raise TeacherContractError("seed_plan: namespace, derivation, or parity mismatch")
    if expected["solver"]["seed"] == expected["payoff"]["seed"]:
        raise TeacherContractError("seed_plan: solver/payoff numeric overlap")
    return expected


def _verify_mccfr_targets(
    value: Mapping[str, Any],
    key: InfoSetKey,
) -> dict[str, Any]:
    raw = _require_mapping(value, label="targets")
    expected_fields = {
        "average_strategy_quantization",
        "average_strategy_denominator",
        "average_strategy_q32",
        "action_payoff_statistics",
        "policy_value_estimate",
        "payoff_perspective",
        "action_payoff_semantics",
        "policy_value_uses_quantized_average_strategy",
        "best_action_semantic_index",
        "best_action_id",
        "best_action_value_estimate",
        "one_step_deviation_regret_estimate",
        "exact_exploitability_computed",
    }
    _exact_keys(raw, expected_fields, label="targets")
    if raw.get("average_strategy_quantization") != "largest_remainder_exact_q32":
        raise TeacherContractError("targets: strategy quantization mismatch")
    if raw.get("average_strategy_denominator") != Q32_DENOMINATOR:
        raise TeacherContractError("targets: strategy denominator mismatch")
    if raw.get("payoff_perspective") != "acting_player":
        raise TeacherContractError("targets: payoff perspective mismatch")
    if raw.get("action_payoff_semantics") != (
        "one_step_action_then_frozen_teacher_average_strategy_continuation"
    ):
        raise TeacherContractError("targets: action payoff semantics mismatch")
    if raw.get("policy_value_uses_quantized_average_strategy") is not True:
        raise TeacherContractError("targets: policy value strategy binding mismatch")
    if raw.get("exact_exploitability_computed") is not False:
        raise TeacherContractError("targets: exact exploitability claim forbidden")
    action_ids = semantic_action_ids(key)
    mask = legal_action_mask(key)
    strategy = raw.get("average_strategy_q32")
    if (
        not isinstance(strategy, list)
        or len(strategy) != 27
        or any(
            isinstance(weight, bool) or not isinstance(weight, int) or weight < 0
            for weight in strategy
        )
        or sum(strategy) != Q32_DENOMINATOR
    ):
        raise TeacherContractError("targets: invalid exact-Q32 strategy")
    for index, legal in enumerate(mask):
        if not legal and strategy[index] != 0:
            raise TeacherContractError("targets: illegal action has strategy mass")
    statistics = raw.get("action_payoff_statistics")
    if not isinstance(statistics, list) or len(statistics) != 27:
        raise TeacherContractError("targets: 27 payoff statistic slots required")
    means: dict[int, float] = {}
    for index, (action_id, legal) in enumerate(zip(action_ids, mask)):
        item = statistics[index]
        if not legal:
            if item is not None:
                raise TeacherContractError("targets: illegal action has payoff statistics")
            continue
        stat = _require_mapping(item, label=f"targets.action_payoff_statistics[{index}]")
        _exact_keys(
            stat,
            {"action_id", "count", "sum", "sum_squares", "mean", "standard_error"},
            label=f"targets.action_payoff_statistics[{index}]",
        )
        if stat.get("action_id") != action_id:
            raise TeacherContractError("targets: action statistic identity mismatch")
        count = _integer(
            stat.get("count"),
            label=f"targets.action_payoff_statistics[{index}].count",
            minimum=2,
        )
        total = _finite(
            stat.get("sum"),
            label=f"targets.action_payoff_statistics[{index}].sum",
        )
        squares = _finite(
            stat.get("sum_squares"),
            label=f"targets.action_payoff_statistics[{index}].sum_squares",
            nonnegative=True,
        )
        mean, error = _raw_moment_mean_and_standard_error(
            count=count,
            total=total,
            squares=squares,
            label="targets",
        )
        published_mean = _finite(
            stat.get("mean"),
            label=f"targets.action_payoff_statistics[{index}].mean",
        )
        published_error = _finite(
            stat.get("standard_error"),
            label=f"targets.action_payoff_statistics[{index}].standard_error",
            nonnegative=True,
        )
        if not math.isclose(mean, published_mean, rel_tol=0.0, abs_tol=_FLOAT_TOL):
            raise TeacherContractError("targets: payoff mean/raw moment mismatch")
        if not math.isclose(error, published_error, rel_tol=0.0, abs_tol=_FLOAT_TOL):
            raise TeacherContractError("targets: payoff SE/raw moment mismatch")
        means[index] = mean
    policy_value = math.fsum(
        strategy[index] / Q32_DENOMINATOR * mean
        for index, mean in means.items()
    )
    best_index = max(means, key=lambda index: (means[index], -index))
    best_value = means[best_index]
    regret = max(0.0, best_value - policy_value)
    derived = {
        "policy_value_estimate": policy_value,
        "best_action_semantic_index": best_index,
        "best_action_id": action_ids[best_index],
        "best_action_value_estimate": best_value,
        "one_step_deviation_regret_estimate": regret,
    }
    for field, expected in derived.items():
        actual = raw.get(field)
        if isinstance(expected, float):
            published = _finite(actual, label=f"targets.{field}")
            if not math.isclose(expected, published, rel_tol=0.0, abs_tol=_FLOAT_TOL):
                raise TeacherContractError(f"targets.{field}: derived mismatch")
        elif actual != expected:
            raise TeacherContractError(f"targets.{field}: derived mismatch")
    return dict(raw)


def _verify_mccfr_quality(
    value: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _require_mapping(value, label="quality")
    expected_fields = {
        "infoset_visit_count",
        "infoset_reach_probability_exact",
        "legal_action_count",
        "average_strategy_support_count",
        "min_action_payoff_sample_count",
        "max_action_payoff_standard_error",
        "all_legal_actions_sampled",
        "structural_quality_passed",
        "quality_status",
        "promotion_quality_gate_evaluated",
    }
    _exact_keys(raw, expected_fields, label="quality")
    visits = _integer(raw.get("infoset_visit_count"), label="quality.infoset_visit_count")
    reach = _exact_fraction(
        raw.get("infoset_reach_probability_exact"),
        label="quality.infoset_reach_probability_exact",
    )
    if not 0 <= reach <= 1 or raw.get("infoset_reach_probability_exact") != _fraction_text(reach):
        raise TeacherContractError("quality: noncanonical reach probability")
    statistics = [row for row in targets["action_payoff_statistics"] if row is not None]
    strategy = targets["average_strategy_q32"]
    derived = {
        "legal_action_count": len(statistics),
        "average_strategy_support_count": sum(weight > 0 for weight in strategy),
        "min_action_payoff_sample_count": min(int(row["count"]) for row in statistics),
        "max_action_payoff_standard_error": max(
            float(row["standard_error"]) for row in statistics
        ),
        "all_legal_actions_sampled": True,
    }
    structural = (
        visits > 0
        and reach > 0
        and derived["average_strategy_support_count"] > 0
        and derived["min_action_payoff_sample_count"] >= 2
    )
    derived.update(
        {
            "structural_quality_passed": structural,
            "quality_status": "structural_pass" if structural else "structural_fail",
            "promotion_quality_gate_evaluated": False,
        }
    )
    for field, expected in derived.items():
        actual = raw.get(field)
        if isinstance(expected, float):
            published = _finite(actual, label=f"quality.{field}", nonnegative=True)
            if not math.isclose(expected, published, rel_tol=0.0, abs_tol=_FLOAT_TOL):
                raise TeacherContractError(f"quality.{field}: derived mismatch")
        elif actual != expected:
            raise TeacherContractError(f"quality.{field}: derived mismatch")
    return dict(raw)


def _verify_exact_targets(
    value: Mapping[str, Any],
    key: InfoSetKey,
    resolution: T4BtnExactResolution,
) -> dict[str, Any]:
    raw = _require_mapping(value, label="targets")
    expected = _exact_target_payload(key, resolution)
    if canonical_json(raw) != canonical_json(expected):
        raise TeacherContractError("targets: exact terminal resolver binding mismatch")
    return dict(raw)


def _verify_exact_quality(
    value: Mapping[str, Any],
    resolution: T4BtnExactResolution,
) -> dict[str, Any]:
    raw = _require_mapping(value, label="quality")
    expected = _exact_quality_payload(resolution)
    if canonical_json(raw) != canonical_json(expected):
        raise TeacherContractError("quality: exact terminal verification mismatch")
    return dict(raw)


def verify_teacher_row(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _require_mapping(value, label="teacher_row")
    _exact_keys(raw, _TEACHER_ROW_KEYS, label="teacher_row")
    fixed = {
        "schema": TEACHER_ROW_SCHEMA,
        "artifact_kind": "algorithm_teacher_row_for_generalization_distillation",
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "global_policy_claimed": False,
        "training_performed": False,
        "serving_changed": False,
        "solver_checkpoint_replayed": False,
        "opponent_hidden_cards_in_model_input": False,
        "restricted_hidden_information_is_commitment_only": True,
    }
    for field, expected in fixed.items():
        if raw.get(field) != expected:
            raise TeacherContractError(f"teacher_row.{field}: mismatch")
    if raw.get("row_sha256") != self_hash(raw, "row_sha256"):
        raise TeacherContractError("teacher_row: self-hash mismatch")
    assignment = verify_split_assignment(raw.get("split_assignment"))
    lineage = _validate_lineage(raw.get("lineage"))
    raw_bindings = _require_mapping(raw.get("bindings"), label="bindings")
    model_input, key = _verify_model_input(raw.get("model_input"))

    stratum = _require_mapping(raw.get("stratum"), label="stratum")
    _exact_keys(
        stratum,
        {"phase", "turn", "actor", "visible_joker_count", "joker_stratum"},
        label="stratum",
    )
    joker_count = _visible_joker_count(key)
    expected_stratum = {
        "phase": key.phase,
        "turn": key.turn,
        "actor": key.actor,
        "visible_joker_count": joker_count,
        "joker_stratum": f"joker_{joker_count}",
    }
    if dict(stratum) != expected_stratum:
        raise TeacherContractError("stratum: observation binding mismatch")

    solver = _require_mapping(raw.get("solver"), label="solver")
    method = solver.get("method")
    if not isinstance(method, str) or method not in SOLVER_METHOD_CONTRACT_IDS:
        raise TeacherContractError(
            "solver.method: known contract ID required; "
            f"allowed={list(SOLVER_METHOD_CONTRACT_IDS)}"
        )
    exact_resolution: T4BtnExactResolution | None = None
    if method == MCCFR_SOLVER_METHOD:
        bindings = _validate_hash_bindings(raw_bindings)
        _exact_keys(
            solver,
            {
                "method",
                "iterations_completed",
                "seed_plan",
                "average_strategy_source",
                "payoff_source",
                "label_authenticity_scope",
            },
            label="solver",
        )
        _integer(
            solver.get("iterations_completed"),
            label="solver.iterations_completed",
            minimum=1,
        )
        if solver.get("average_strategy_source") != "mccfr_average_strategy":
            raise TeacherContractError("solver: average strategy source mismatch")
        if solver.get("payoff_source") != "independent_payoff_seed_namespace":
            raise TeacherContractError("solver: payoff source mismatch")
        if solver.get("label_authenticity_scope") != (
            "content_binding_not_solver_reexecution"
        ):
            raise TeacherContractError("solver: label authenticity scope mismatch")
        _verify_seed_plan(solver.get("seed_plan"), assignment)
    else:
        bindings = _validate_hash_bindings(
            raw_bindings,
            fields=_COMMON_BINDING_HASH_FIELDS,
        )
        if key.phase != "t4_second" or key.actor != "btn" or key.turn != 4:
            raise TeacherContractError("exact terminal method requires BTN t4_second")
        _exact_keys(
            solver,
            {
                "method",
                "policy_source",
                "payoff_source",
                "seed_usage",
                "resolver_manifest_sha256",
                "resolver_result_sha256",
                "label_authenticity_scope",
            },
            label="solver",
        )
        try:
            exact_resolution = resolve_t4_second_btn_exact(key)
        except (T4BtnExactResolveError, TypeError, ValueError) as exc:
            raise TeacherContractError(f"exact resolver reexecution failed: {exc}") from exc
        expected_solver = _exact_solver_payload(exact_resolution)
        if canonical_json(solver) != canonical_json(expected_solver):
            raise TeacherContractError("solver: exact resolver provenance mismatch")

    action_contract = _require_mapping(raw.get("action_contract"), label="action_contract")
    _exact_keys(
        action_contract,
        {
            "semantic_action_count",
            "action_semantics_sha256",
            "semantic_action_ids",
            "legal_action_mask",
        },
        label="action_contract",
    )
    expected_action_contract = {
        "semantic_action_count": 27,
        "action_semantics_sha256": ACTION_SEMANTICS_SHA256,
        "semantic_action_ids": list(semantic_action_ids(key)),
        "legal_action_mask": [bool(value) for value in legal_action_mask(key)],
    }
    if dict(action_contract) != expected_action_contract:
        raise TeacherContractError("action_contract: observation binding mismatch")
    if method == MCCFR_SOLVER_METHOD:
        targets = _verify_mccfr_targets(raw.get("targets"), key)
        _verify_mccfr_quality(raw.get("quality"), targets)
    else:
        if exact_resolution is None:
            raise AssertionError("exact resolution was not reexecuted")
        _verify_exact_targets(raw.get("targets"), key, exact_resolution)
        _verify_exact_quality(raw.get("quality"), exact_resolution)
    identity = _row_identity(
        assignment=assignment,
        lineage=lineage,
        model_input=model_input,
        bindings=bindings,
        solver=solver,
    )
    if raw.get("row_identity_sha256") != identity:
        raise TeacherContractError("teacher_row: identity mismatch")
    return json.loads(canonical_json(raw))


def _overlap_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    owners: dict[str, dict[str, str]] = {
        "full_deal_commitment_sha256": {},
        "public_root_family_commitment_sha256": {},
        "information_digest": {},
        **{field: {} for field in _OVERLAP_BINDINGS},
        **{field: {} for field in sorted(_LINEAGE_HASH_FIELDS)},
    }
    root_family_deal: dict[str, str] = {}
    lineage_deal_owners: dict[str, dict[str, str]] = {
        field: {} for field in sorted(_LINEAGE_HASH_FIELDS)
    }
    cross_split_counts = {field: 0 for field in owners}
    for row in rows:
        assignment = row["split_assignment"]
        split = assignment["split"]
        values = {
            "full_deal_commitment_sha256": assignment[
                "full_deal_commitment_sha256"
            ],
            "public_root_family_commitment_sha256": assignment[
                "public_root_family_commitment_sha256"
            ],
            "information_digest": row["model_input"]["information_digest"],
            **{field: row["bindings"][field] for field in _OVERLAP_BINDINGS},
            **{field: row["lineage"][field] for field in sorted(_LINEAGE_HASH_FIELDS)},
        }
        family = values["public_root_family_commitment_sha256"]
        deal = values["full_deal_commitment_sha256"]
        prior_deal = root_family_deal.setdefault(family, deal)
        if prior_deal != deal:
            raise TeacherContractError(
                "public root family is bound to multiple full-deal commitments"
            )
        for field in sorted(_LINEAGE_HASH_FIELDS):
            commitment = values[field]
            prior_lineage_deal = lineage_deal_owners[field].setdefault(
                commitment,
                deal,
            )
            if prior_lineage_deal != deal:
                raise TeacherContractError(
                    f"{field} lineage commitment is bound to multiple "
                    "full-deal commitments"
                )
        for field, commitment in values.items():
            prior = owners[field].setdefault(commitment, split)
            if prior != split:
                cross_split_counts[field] += 1
    if any(cross_split_counts.values()):
        raise TeacherContractError(
            f"teacher rows overlap locked splits: {cross_split_counts}"
        )
    return {
        "checked": True,
        "fields": sorted(owners),
        "cross_split_overlap_counts": cross_split_counts,
        "cross_split_overlap_count": 0,
        "each_public_root_family_has_one_full_deal": True,
        "each_lineage_commitment_has_one_full_deal": True,
    }


def _audit_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence of teacher rows")
    if not 1 <= len(rows) <= MAX_IN_MEMORY_ROWS:
        raise TeacherContractError(
            f"bounded bundle requires 1..{MAX_IN_MEMORY_ROWS} rows"
        )
    verified = [verify_teacher_row(row) for row in rows]
    identities = [row["row_identity_sha256"] for row in verified]
    if len(identities) != len(set(identities)):
        raise TeacherContractError("duplicate teacher row identity")
    overlap = _overlap_audit(verified)
    ordered = sorted(verified, key=lambda row: row["row_identity_sha256"])
    split_counts = {split: 0 for split in SPLIT_NAMES}
    group_sets = {split: set() for split in SPLIT_NAMES}
    strata = {
        f"{phase}_{actor}_joker_{joker}": 0
        for phase, actor in (
            ("t3_first", "bb"),
            ("t3_second", "btn"),
            ("t4_first", "bb"),
            ("t4_second", "btn"),
        )
        for joker in range(3)
    }
    solver_seeds: set[int] = set()
    payoff_seeds: set[int] = set()
    for row in ordered:
        assignment = row["split_assignment"]
        split = assignment["split"]
        split_counts[split] += 1
        group_sets[split].add(assignment["split_group_sha256"])
        stratum = row["stratum"]
        cell = (
            f"{stratum['phase']}_{stratum['actor']}_"
            f"joker_{stratum['visible_joker_count']}"
        )
        if cell not in strata:
            raise TeacherContractError("teacher row lies outside T3/T4 actor/Joker strata")
        strata[cell] += 1
        if row["solver"]["method"] == MCCFR_SOLVER_METHOD:
            solver_seeds.add(row["solver"]["seed_plan"]["solver"]["seed"])
            payoff_seeds.add(row["solver"]["seed_plan"]["payoff"]["seed"])
    if solver_seeds & payoff_seeds:
        raise TeacherContractError("solver/payoff numeric seed sets overlap")
    stats = {
        "row_count": len(ordered),
        "split_row_counts": split_counts,
        "split_group_counts": {
            split: len(group_sets[split]) for split in SPLIT_NAMES
        },
        "phase_actor_joker_counts": strata,
        "full_deal_count": len(
            {
                row["split_assignment"]["full_deal_commitment_sha256"]
                for row in ordered
            }
        ),
        "public_root_family_count": len(
            {
                row["split_assignment"]["public_root_family_commitment_sha256"]
                for row in ordered
            }
        ),
        "row_identity_set_sha256": canonical_sha256(sorted(identities)),
        "row_content_set_sha256": canonical_sha256(
            sorted(row["row_sha256"] for row in ordered)
        ),
        "solver_seed_set_sha256": canonical_sha256(sorted(solver_seeds)),
        "payoff_seed_set_sha256": canonical_sha256(sorted(payoff_seeds)),
        "seed_namespace_overlap_count": 0,
        "overlap_audit": overlap,
    }
    return ordered, stats


def _build_shard(
    *,
    index: int,
    row_start: int,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    shard: dict[str, Any] = {
        "schema": TEACHER_SHARD_SCHEMA,
        "artifact_kind": "bounded_content_addressed_teacher_shard",
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "index": index,
        "row_start": row_start,
        "row_stop_exclusive": row_start + len(rows),
        "row_count": len(rows),
        "row_identity_order_sha256": canonical_sha256(
            [row["row_identity_sha256"] for row in rows]
        ),
        "rows": list(rows),
    }
    shard["shard_sha256"] = self_hash(shard, "shard_sha256")
    return shard


def _shard_relative_path(index: int, shard_sha256: str) -> str:
    return f"shards/shard-{index:06d}-{shard_sha256}.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_shard_entry(
    *,
    shard: Mapping[str, Any],
    relative_path: str,
    file_sha256: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "schema": TEACHER_SHARD_ENTRY_SCHEMA,
        "index": shard["index"],
        "relative_path": relative_path,
        "row_start": shard["row_start"],
        "row_stop_exclusive": shard["row_stop_exclusive"],
        "row_count": shard["row_count"],
        "shard_sha256": shard["shard_sha256"],
        "file_sha256": file_sha256,
        "row_identity_order_sha256": shard["row_identity_order_sha256"],
    }
    entry["entry_sha256"] = self_hash(entry, "entry_sha256")
    return entry


def _build_manifest(
    *,
    shard_size: int,
    shard_entries: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": TEACHER_BUNDLE_SCHEMA,
        "artifact_kind": "bounded_in_memory_t3_t4_distillation_teacher_bundle",
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "global_policy_claimed": False,
        "training_performed": False,
        "serving_changed": False,
        "solver_checkpoint_replayed": False,
        "solver_output_authenticity_claimed": False,
        "exact_exploitability_computed": False,
        "manifest_written_last": True,
        "opponent_hidden_cards_in_model_input": False,
        "restricted_hidden_information_is_commitment_only": True,
        "encoder_contract": {
            "schema": INFOSET_ENCODER_SCHEMA,
            "manifest_sha256": INFOSET_ENCODER_MANIFEST_SHA256,
            "vector_dimension": INFOSET_VECTOR_DIM,
        },
        "split_contract": {
            "schema": SPLIT_ASSIGNMENT_SCHEMA,
            "namespace": SPLIT_NAMESPACE,
            "assignment_before_labels": True,
            "material": "original_pre_augmentation_full_deal_commitment_sha256",
            "bucket": "uint64_be(first_8_sha256_bytes)%10000",
            "fit": [0, 6999],
            "dev": [7000, 8499],
            "test": [8500, 9999],
            "invariant_axes": list(SPLIT_INVARIANTS),
        },
        "seed_contract": {
            "applies_to_method": MCCFR_SOLVER_METHOD,
            "seedless_methods": [T4_BTN_EXACT_METHOD],
            "solver_namespace": SOLVER_SEED_NAMESPACE,
            "payoff_namespace": PAYOFF_SEED_NAMESPACE,
            "solver_numeric_parity": 0,
            "payoff_numeric_parity": 1,
            "overlap_forbidden": True,
        },
        "solver_method_contract_ids": list(SOLVER_METHOD_CONTRACT_IDS),
        "action_semantics_sha256": ACTION_SEMANTICS_SHA256,
        "shard_size": shard_size,
        "max_rows_per_shard": MAX_ROWS_PER_SHARD,
        "max_in_memory_rows": MAX_IN_MEMORY_ROWS,
        "shard_count": len(shard_entries),
        "row_count": stats["row_count"],
        "split_row_counts": stats["split_row_counts"],
        "split_group_counts": stats["split_group_counts"],
        "phase_actor_joker_counts": stats["phase_actor_joker_counts"],
        "full_deal_count": stats["full_deal_count"],
        "public_root_family_count": stats["public_root_family_count"],
        "row_identity_set_sha256": stats["row_identity_set_sha256"],
        "row_content_set_sha256": stats["row_content_set_sha256"],
        "solver_seed_set_sha256": stats["solver_seed_set_sha256"],
        "payoff_seed_set_sha256": stats["payoff_seed_set_sha256"],
        "seed_namespace_overlap_count": stats["seed_namespace_overlap_count"],
        "overlap_audit": stats["overlap_audit"],
        "shards": list(shard_entries),
        "remaining_integration": [
            "streaming_or_resumable_teacher_generation",
            "mccfr_result_to_row_adapter",
            "locked_model_promotion_gate",
        ],
    }
    manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
    return manifest


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TeacherContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_canonical(path: Path) -> Any:
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise TeacherContractError(f"cannot read {path}: {exc}") from exc
    if not text.endswith("\n") or text.count("\n") != 1:
        raise TeacherContractError(f"{path}: one canonical JSON line required")
    try:
        value = json.loads(
            text[:-1],
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except json.JSONDecodeError as exc:
        raise TeacherContractError(f"{path}: invalid JSON") from exc
    if canonical_json(value) != text[:-1]:
        raise TeacherContractError(f"{path}: noncanonical JSON")
    return value


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    if _read_canonical(path) != value:
        raise TeacherContractError(f"{path}: atomic write readback mismatch")


def _safe_regular_file(root: Path, relative_path: str, *, label: str) -> Path:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
    ):
        raise TeacherContractError(f"{label}: canonical POSIX path required")
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != relative_path
    ):
        raise TeacherContractError(f"{label}: unsafe relative path")
    if root.is_symlink():
        raise TeacherContractError(f"{label}: symlink root forbidden")
    target = root / relative
    cursor = target
    while cursor != root and cursor != cursor.parent:
        if cursor.is_symlink():
            raise TeacherContractError(f"{label}: symlinks forbidden")
        cursor = cursor.parent
    try:
        resolved_root = root.resolve(strict=True)
        resolved = target.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise TeacherContractError(f"{label}: missing or escaping file") from exc
    if not resolved.is_file():
        raise TeacherContractError(f"{label}: regular file required")
    return resolved


def _artifact_paths(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise TeacherContractError("bundle contains a forbidden symlink")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            files.add(relative)
        else:
            raise TeacherContractError("bundle contains a non-regular artifact")
    return files, directories


@dataclass(frozen=True)
class VerifiedTeacherBundle:
    root: Path
    manifest: dict[str, Any]
    rows: tuple[dict[str, Any], ...]


def write_teacher_bundle(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    shard_size: int,
) -> VerifiedTeacherBundle:
    """Atomically publish content-addressed shards, then the manifest marker."""
    size = _integer(
        shard_size,
        label="shard_size",
        minimum=1,
        maximum=MAX_ROWS_PER_SHARD,
    )
    ordered, stats = _audit_rows(rows)
    supplied_root = Path(output_dir).absolute()
    if supplied_root.is_symlink():
        raise TeacherContractError("output root symlink forbidden")
    root = supplied_root.resolve()
    if root.exists():
        if any(root.iterdir()):
            raise TeacherContractError("output directory must be empty")
    else:
        root.mkdir(parents=True)

    entries: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(ordered), size)):
        chunk = ordered[start : start + size]
        shard = _build_shard(index=index, row_start=start, rows=chunk)
        relative_path = _shard_relative_path(index, shard["shard_sha256"])
        path = root / relative_path
        _atomic_write(path, shard)
        entries.append(
            _build_shard_entry(
                shard=shard,
                relative_path=relative_path,
                file_sha256=_file_sha256(path),
            )
        )
    manifest = _build_manifest(
        shard_size=size,
        shard_entries=entries,
        stats=stats,
    )
    _atomic_write(root / MANIFEST_NAME, manifest)
    return verify_teacher_bundle(root)


_SHARD_KEYS = {
    "schema",
    "artifact_kind",
    "algorithm_teacher_only",
    "promotion_eligible",
    "index",
    "row_start",
    "row_stop_exclusive",
    "row_count",
    "row_identity_order_sha256",
    "rows",
    "shard_sha256",
}
_SHARD_ENTRY_KEYS = {
    "schema",
    "index",
    "relative_path",
    "row_start",
    "row_stop_exclusive",
    "row_count",
    "shard_sha256",
    "file_sha256",
    "row_identity_order_sha256",
    "entry_sha256",
}


def _verify_shard(
    value: Mapping[str, Any],
    *,
    index: int,
    expected_start: int,
    shard_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _require_mapping(value, label="teacher_shard")
    _exact_keys(raw, _SHARD_KEYS, label="teacher_shard")
    fixed = {
        "schema": TEACHER_SHARD_SCHEMA,
        "artifact_kind": "bounded_content_addressed_teacher_shard",
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "index": index,
        "row_start": expected_start,
    }
    for field, expected in fixed.items():
        if raw.get(field) != expected:
            raise TeacherContractError(f"teacher_shard.{field}: mismatch")
    rows = raw.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= shard_size:
        raise TeacherContractError("teacher_shard: invalid bounded row list")
    stop = expected_start + len(rows)
    if raw.get("row_count") != len(rows) or raw.get("row_stop_exclusive") != stop:
        raise TeacherContractError("teacher_shard: gap, overlap, or row-count mismatch")
    verified_rows = [verify_teacher_row(row) for row in rows]
    if [row["row_identity_sha256"] for row in verified_rows] != sorted(
        row["row_identity_sha256"] for row in verified_rows
    ):
        raise TeacherContractError("teacher_shard: rows are not canonically ordered")
    rebuilt = _build_shard(
        index=index,
        row_start=expected_start,
        rows=verified_rows,
    )
    if dict(raw) != rebuilt:
        raise TeacherContractError("teacher_shard: content or self-hash mismatch")
    return rebuilt, verified_rows


_MANIFEST_KEYS = {
    "schema",
    "artifact_kind",
    "algorithm_teacher_only",
    "promotion_eligible",
    "global_policy_claimed",
    "training_performed",
    "serving_changed",
    "solver_checkpoint_replayed",
    "solver_output_authenticity_claimed",
    "exact_exploitability_computed",
    "manifest_written_last",
    "opponent_hidden_cards_in_model_input",
    "restricted_hidden_information_is_commitment_only",
    "encoder_contract",
    "split_contract",
    "seed_contract",
    "solver_method_contract_ids",
    "action_semantics_sha256",
    "shard_size",
    "max_rows_per_shard",
    "max_in_memory_rows",
    "shard_count",
    "row_count",
    "split_row_counts",
    "split_group_counts",
    "phase_actor_joker_counts",
    "full_deal_count",
    "public_root_family_count",
    "row_identity_set_sha256",
    "row_content_set_sha256",
    "solver_seed_set_sha256",
    "payoff_seed_set_sha256",
    "seed_namespace_overlap_count",
    "overlap_audit",
    "shards",
    "remaining_integration",
    "manifest_sha256",
}


def verify_teacher_bundle(output_dir: str | Path) -> VerifiedTeacherBundle:
    supplied_root = Path(output_dir).absolute()
    if supplied_root.is_symlink():
        raise TeacherContractError("teacher bundle root symlink forbidden")
    root = supplied_root.resolve()
    if not root.exists() or not root.is_dir():
        raise TeacherContractError("teacher bundle root must be a real directory")
    manifest_path = _safe_regular_file(root, MANIFEST_NAME, label="teacher manifest")
    manifest = _require_mapping(
        _read_canonical(manifest_path),
        label="teacher manifest",
    )
    _exact_keys(manifest, _MANIFEST_KEYS, label="teacher manifest")
    fixed = {
        "schema": TEACHER_BUNDLE_SCHEMA,
        "artifact_kind": "bounded_in_memory_t3_t4_distillation_teacher_bundle",
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "global_policy_claimed": False,
        "training_performed": False,
        "serving_changed": False,
        "solver_checkpoint_replayed": False,
        "solver_output_authenticity_claimed": False,
        "exact_exploitability_computed": False,
        "manifest_written_last": True,
        "opponent_hidden_cards_in_model_input": False,
        "restricted_hidden_information_is_commitment_only": True,
        "max_rows_per_shard": MAX_ROWS_PER_SHARD,
        "max_in_memory_rows": MAX_IN_MEMORY_ROWS,
        "solver_method_contract_ids": list(SOLVER_METHOD_CONTRACT_IDS),
        "action_semantics_sha256": ACTION_SEMANTICS_SHA256,
    }
    for field, expected in fixed.items():
        if manifest.get(field) != expected:
            raise TeacherContractError(f"teacher manifest.{field}: mismatch")
    if manifest.get("manifest_sha256") != self_hash(manifest, "manifest_sha256"):
        raise TeacherContractError("teacher manifest: self-hash mismatch")
    size = _integer(
        manifest.get("shard_size"),
        label="teacher manifest.shard_size",
        minimum=1,
        maximum=MAX_ROWS_PER_SHARD,
    )
    raw_entries = manifest.get("shards")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise TeacherContractError("teacher manifest: non-empty shard list required")
    if manifest.get("shard_count") != len(raw_entries):
        raise TeacherContractError("teacher manifest: shard count mismatch")
    declared_rows = _integer(
        manifest.get("row_count"),
        label="teacher manifest.row_count",
        minimum=1,
        maximum=MAX_IN_MEMORY_ROWS,
    )
    expected_shard_count = (declared_rows + size - 1) // size
    if len(raw_entries) != expected_shard_count:
        raise TeacherContractError("teacher manifest: bounded shard count mismatch")

    expected_files = {MANIFEST_NAME}
    entries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    expected_start = 0
    for index, raw_entry in enumerate(raw_entries):
        entry = _require_mapping(raw_entry, label=f"shards[{index}]")
        _exact_keys(entry, _SHARD_ENTRY_KEYS, label=f"shards[{index}]")
        if entry.get("entry_sha256") != self_hash(entry, "entry_sha256"):
            raise TeacherContractError("teacher shard entry: self-hash mismatch")
        if entry.get("schema") != TEACHER_SHARD_ENTRY_SCHEMA or entry.get("index") != index:
            raise TeacherContractError("teacher shard entry: duplicate/out-of-order index")
        if entry.get("row_start") != expected_start:
            raise TeacherContractError("teacher shard entry: gap or overlap")
        entry_count = _integer(
            entry.get("row_count"),
            label=f"shards[{index}].row_count",
            minimum=1,
            maximum=size,
        )
        if index < len(raw_entries) - 1 and entry_count != size:
            raise TeacherContractError("teacher shard entry: non-final shard is incomplete")
        shard_hash = _sha(entry.get("shard_sha256"), label="shard entry.shard_sha256")
        expected_path = _shard_relative_path(index, shard_hash)
        if entry.get("relative_path") != expected_path:
            raise TeacherContractError("teacher shard entry: content-addressed path mismatch")
        path = _safe_regular_file(root, expected_path, label=f"shards[{index}]")
        if _file_sha256(path) != _sha(entry.get("file_sha256"), label="shard entry.file_sha256"):
            raise TeacherContractError("teacher shard entry: file SHA-256 mismatch")
        shard, shard_rows = _verify_shard(
            _read_canonical(path),
            index=index,
            expected_start=expected_start,
            shard_size=size,
        )
        rebuilt_entry = _build_shard_entry(
            shard=shard,
            relative_path=expected_path,
            file_sha256=_file_sha256(path),
        )
        if dict(entry) != rebuilt_entry:
            raise TeacherContractError("teacher shard entry: immutable artifact mismatch")
        entries.append(rebuilt_entry)
        rows.extend(shard_rows)
        expected_start = shard["row_stop_exclusive"]
        expected_files.add(expected_path)
    if declared_rows != expected_start:
        raise TeacherContractError("teacher manifest: terminal row gap")

    ordered, stats = _audit_rows(rows)
    if rows != ordered:
        raise TeacherContractError("teacher bundle: global row order mismatch")
    rebuilt_manifest = _build_manifest(
        shard_size=size,
        shard_entries=entries,
        stats=stats,
    )
    if dict(manifest) != rebuilt_manifest:
        raise TeacherContractError("teacher manifest: verified content mismatch")
    actual_files, actual_directories = _artifact_paths(root)
    if actual_files != expected_files:
        raise TeacherContractError(
            "teacher bundle: missing or orphan files; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"orphan={sorted(actual_files - expected_files)}"
        )
    if actual_directories != {"shards"}:
        raise TeacherContractError(
            f"teacher bundle: orphan directories {sorted(actual_directories - {'shards'})}"
        )
    return VerifiedTeacherBundle(
        root=root,
        manifest=dict(manifest),
        rows=tuple(ordered),
    )


_CANONICAL_T4_EXACT_RESOLVER_ALIASES = (
    ("resolve_t4_second_btn_exact", resolve_t4_second_btn_exact),
    ("verify_t4_second_btn_exact", verify_t4_second_btn_exact),
)


def _guard_exact_resolver_aliases(function: Callable[..., Any]) -> Callable[..., Any]:
    """Reject substitution of the exact resolver used to authenticate labels."""

    canonical_aliases = _CANONICAL_T4_EXACT_RESOLVER_ALIASES

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if (
            globals().get("_CANONICAL_T4_EXACT_RESOLVER_ALIASES")
            is not canonical_aliases
        ):
            raise TeacherContractError("canonical exact resolver alias set drifted")
        for name, expected in canonical_aliases:
            if globals().get(name) is not expected:
                raise TeacherContractError(
                    f"canonical exact resolver alias drifted: {name}"
                )
        return function(*args, **kwargs)

    return guarded


build_t4_btn_exact_teacher_row = _guard_exact_resolver_aliases(
    build_t4_btn_exact_teacher_row
)
verify_teacher_row = _guard_exact_resolver_aliases(verify_teacher_row)


_TEACHER_RUNTIME_SEMANTIC_ANCHOR = register_module_function_anchor(
    __name__, globals()
)
_TEACHER_RUNTIME_SEMANTIC_ANCHOR_MIRROR = _TEACHER_RUNTIME_SEMANTIC_ANCHOR


__all__ = [
    "MANIFEST_NAME",
    "MAX_IN_MEMORY_ROWS",
    "MAX_ROWS_PER_SHARD",
    "MCCFR_SOLVER_METHOD",
    "MODEL_INPUT_SCHEMA",
    "PAYOFF_SEED_NAMESPACE",
    "Q32_DENOMINATOR",
    "SOLVER_SEED_NAMESPACE",
    "SOLVER_METHOD_CONTRACT_IDS",
    "SPLIT_ASSIGNMENT_SCHEMA",
    "SPLIT_NAMESPACE",
    "TEACHER_BUNDLE_SCHEMA",
    "TEACHER_ROW_SCHEMA",
    "TeacherContractError",
    "VerifiedTeacherBundle",
    "build_split_assignment",
    "build_t4_btn_exact_teacher_row",
    "build_teacher_row",
    "canonical_json",
    "canonical_sha256",
    "derive_payoff_seed",
    "derive_solver_seed",
    "quantize_average_strategy_q32",
    "self_hash",
    "verify_split_assignment",
    "verify_teacher_bundle",
    "verify_teacher_row",
    "write_teacher_bundle",
]
