"""Verified, split-locked T3/T4 policy/value/Q distillation dataset.

The teacher bundle is the authority for both labels and split ownership.  This
module never invents a random train/dev/test split: it materializes the locked
``fit``/``dev``/``test`` assignments only after replaying the teacher bundle's
content, row, encoder, hidden-information, and provenance verification.

The training interface intentionally exposes only the lossless information-set
vector and label tensors.  Full-deal, private-type, and public-root commitments
remain audit-only metadata and can therefore never become model features by
accident.  Loading is opt-in, bounded by the teacher contract, and does not
train, promote, or change serving defaults.
"""
from __future__ import annotations

import hashlib
import math
import types
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

import ai.tutor.t3_t4_distillation_teacher as _teacher_module
import ai.tutor.t3_t4_infoset_encoder as _encoder_module
from ai.tutor.runtime_semantic_anchor import (
    registered_module_function_anchor,
    verify_module_function_anchor,
)
from ai.tutor.t3_t4_distillation_teacher import (
    MCCFR_SOLVER_METHOD,
    Q32_DENOMINATOR,
    SPLIT_NAMES,
    TEACHER_BUNDLE_SCHEMA,
    TeacherContractError,
    VerifiedTeacherBundle,
    canonical_json,
    canonical_sha256,
    verify_teacher_bundle,
)
from ai.tutor.t3_t4_infoset_encoder import (
    ACTION_SEMANTICS_SHA256,
    INFOSET_ENCODER_MANIFEST_SHA256,
    INFOSET_ENCODER_SCHEMA,
    INFOSET_VECTOR_DIM,
    decode_infoset_key,
    legal_action_mask,
)
from ai.tutor.t4_btn_exact_resolver import T4_BTN_EXACT_METHOD


DISTILLATION_DATASET_SCHEMA = "ofc_t3_t4_split_locked_distillation_dataset/v1"
DISTILLATION_SPLIT_SCHEMA = "ofc_t3_t4_split_locked_distillation_split/v1"
TARGET_METHOD_CODES = MappingProxyType(
    {
        MCCFR_SOLVER_METHOD: 0,
        T4_BTN_EXACT_METHOD: 1,
    }
)
TRAINING_SAMPLE_KEYS = (
    "state",
    "legal_action_mask",
    "policy_target",
    "value_target",
    "q_target",
    "q_target_mask",
    "q_standard_error",
    "best_action_index",
    "target_method_code",
)

_F32 = np.dtype("<f4")
_F64 = np.dtype("<f8")
_U64 = np.dtype("<u8")
_I64 = np.dtype("<i8")
_I8 = np.dtype("i1")
_BOOL = np.dtype("?")
_PHASE_CONTRACT = {
    "t3_first": ("bb", 3),
    "t3_second": ("btn", 3),
    "t4_first": ("bb", 4),
    "t4_second": ("btn", 4),
}
_VERIFY_TEACHER_BUNDLE = verify_teacher_bundle


class DistillationDatasetError(ValueError):
    """A source bundle or materialized training dataset failed closed."""


def _sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DistillationDatasetError(f"{label}: lowercase SHA-256 required")
    return value


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _array_sha256(value: np.ndarray) -> str:
    header = canonical_json(
        {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "order": "C",
        }
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _visible_joker_count(key: Any) -> int:
    visible: set[str] = set(key.current_draw)
    for board in (key.board_bb, key.board_btn):
        for row in board:
            visible.update(row)
    for _turn, cards in key.own_recall.dealt_by_turn:
        visible.update(cards)
    visible.update(card for _turn, card in key.own_recall.discards_by_turn)
    return len(visible & {"X1", "X2"})


@dataclass(frozen=True)
class DistillationSplitProvenance:
    """Audit-only commitments; none are returned by the training interface."""

    row_identity_sha256: tuple[str, ...]
    row_sha256: tuple[str, ...]
    split_group_sha256: tuple[str, ...]
    full_deal_commitment_sha256: tuple[str, ...]
    public_root_family_commitment_sha256: tuple[str, ...]
    information_digest: tuple[str, ...]
    phase: tuple[str, ...]
    actor: tuple[str, ...]
    visible_joker_count: tuple[int, ...]


@dataclass(frozen=True)
class DistillationSplit:
    """One immutable locked split, compatible with ``torch`` DataLoader."""

    name: str
    states: np.ndarray
    legal_action_masks: np.ndarray
    policy_q32: np.ndarray
    policy_targets: np.ndarray
    value_targets: np.ndarray
    q_targets: np.ndarray
    q_target_masks: np.ndarray
    q_standard_errors: np.ndarray
    best_action_indices: np.ndarray
    target_method_codes: np.ndarray
    provenance: DistillationSplitProvenance

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("distillation sample index must be an integer")
        position = int(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("distillation sample index out of range")
        # Copies keep PyTorch's default collator from receiving non-writeable
        # NumPy views while preserving the audit-locked backing arrays.
        return {
            "state": np.array(self.states[position], copy=True),
            "legal_action_mask": np.array(
                self.legal_action_masks[position], copy=True
            ),
            "policy_target": np.array(self.policy_targets[position], copy=True),
            "value_target": np.asarray(self.value_targets[position]).copy(),
            "q_target": np.array(self.q_targets[position], copy=True),
            "q_target_mask": np.array(self.q_target_masks[position], copy=True),
            "q_standard_error": np.array(
                self.q_standard_errors[position], copy=True
            ),
            "best_action_index": np.asarray(
                self.best_action_indices[position]
            ).copy(),
            "target_method_code": np.asarray(
                self.target_method_codes[position]
            ).copy(),
        }

    def training_batch(self) -> dict[str, np.ndarray]:
        """Return detached arrays containing no hidden-world commitments."""

        return {
            "state": np.array(self.states, copy=True),
            "legal_action_mask": np.array(self.legal_action_masks, copy=True),
            "policy_target": np.array(self.policy_targets, copy=True),
            "value_target": np.array(self.value_targets, copy=True),
            "q_target": np.array(self.q_targets, copy=True),
            "q_target_mask": np.array(self.q_target_masks, copy=True),
            "q_standard_error": np.array(self.q_standard_errors, copy=True),
            "best_action_index": np.array(self.best_action_indices, copy=True),
            "target_method_code": np.array(self.target_method_codes, copy=True),
        }


@dataclass(frozen=True)
class VerifiedDistillationDataset:
    source_root: Path
    source_teacher_manifest_sha256: str
    splits: Mapping[str, DistillationSplit]
    manifest: Mapping[str, Any]

    def for_split(self, name: str) -> DistillationSplit:
        if name not in SPLIT_NAMES:
            raise KeyError(f"unknown locked split {name!r}")
        return self.splits[name]


def _method_targets(
    row: Mapping[str, Any],
) -> tuple[list[int], float, list[float], list[float], int, int]:
    method = row["solver"]["method"]
    targets = row["targets"]
    if method == MCCFR_SOLVER_METHOD:
        policy_q32 = targets["average_strategy_q32"]
        statistics = targets["action_payoff_statistics"]
        q_values = [math.nan if item is None else float(item["mean"]) for item in statistics]
        q_errors = [
            math.nan if item is None else float(item["standard_error"])
            for item in statistics
        ]
        value = float(targets["policy_value_estimate"])
        best_index = int(targets["best_action_semantic_index"])
    elif method == T4_BTN_EXACT_METHOD:
        policy_q32 = targets["exact_argmax_policy_q32"]
        utilities = targets["terminal_utility_by_action"]
        q_values = [math.nan if item is None else float(item["utility"]) for item in utilities]
        q_errors = [math.nan if item is None else 0.0 for item in utilities]
        value = float(targets["selected_action_utility"])
        best_index = int(targets["selected_action_semantic_index"])
    else:  # The teacher verifier should already have rejected this.
        raise DistillationDatasetError(f"unsupported teacher method {method!r}")
    return (
        list(policy_q32),
        value,
        q_values,
        q_errors,
        best_index,
        TARGET_METHOD_CODES[method],
    )


def _empty_provenance() -> DistillationSplitProvenance:
    return DistillationSplitProvenance(
        row_identity_sha256=(),
        row_sha256=(),
        split_group_sha256=(),
        full_deal_commitment_sha256=(),
        public_root_family_commitment_sha256=(),
        information_digest=(),
        phase=(),
        actor=(),
        visible_joker_count=(),
    )


def _build_split(name: str, rows: Sequence[Mapping[str, Any]]) -> DistillationSplit:
    if not rows:
        return DistillationSplit(
            name=name,
            states=_readonly(np.empty((0, INFOSET_VECTOR_DIM)), _F32),
            legal_action_masks=_readonly(np.empty((0, 27)), _BOOL),
            policy_q32=_readonly(np.empty((0, 27)), _U64),
            policy_targets=_readonly(np.empty((0, 27)), _F64),
            value_targets=_readonly(np.empty((0,)), _F64),
            q_targets=_readonly(np.empty((0, 27)), _F64),
            q_target_masks=_readonly(np.empty((0, 27)), _BOOL),
            q_standard_errors=_readonly(np.empty((0, 27)), _F64),
            best_action_indices=_readonly(np.empty((0,)), _I64),
            target_method_codes=_readonly(np.empty((0,)), _I8),
            provenance=_empty_provenance(),
        )

    states: list[list[int]] = []
    masks: list[list[bool]] = []
    policies_q32: list[list[int]] = []
    values: list[float] = []
    q_values: list[list[float]] = []
    q_errors: list[list[float]] = []
    best_indices: list[int] = []
    method_codes: list[int] = []
    provenance: dict[str, list[Any]] = {
        "row_identity_sha256": [],
        "row_sha256": [],
        "split_group_sha256": [],
        "full_deal_commitment_sha256": [],
        "public_root_family_commitment_sha256": [],
        "information_digest": [],
        "phase": [],
        "actor": [],
        "visible_joker_count": [],
    }
    for row in rows:
        assignment = row["split_assignment"]
        if assignment["split"] != name:
            raise DistillationDatasetError("row placed in a split it does not own")
        if row["quality"].get("structural_quality_passed") is not True:
            raise DistillationDatasetError(
                "structural-fail teacher rows are forbidden from training datasets"
            )
        mask = [bool(value) for value in row["action_contract"]["legal_action_mask"]]
        policy_q32, value, q_row, error_row, best, method_code = _method_targets(row)
        q_mask = [math.isfinite(item) for item in q_row]
        if q_mask != mask:
            raise DistillationDatasetError("Q targets must cover every legal action only")
        if any(
            legal and (not math.isfinite(error) or error < 0.0)
            for legal, error in zip(mask, error_row)
        ):
            raise DistillationDatasetError("legal Q uncertainty must be finite and nonnegative")
        if any((not legal) and not math.isnan(error) for legal, error in zip(mask, error_row)):
            raise DistillationDatasetError("illegal Q uncertainty slots must be NaN")
        if sum(policy_q32) != Q32_DENOMINATOR:
            raise DistillationDatasetError("policy target must conserve exact Q32 mass")
        if any((not legal) and weight != 0 for legal, weight in zip(mask, policy_q32)):
            raise DistillationDatasetError("illegal action has policy target mass")
        if not 0 <= best < 27 or not mask[best]:
            raise DistillationDatasetError("best-action target is not legal")
        policy_value = math.fsum(
            weight / Q32_DENOMINATOR * q
            for weight, q, legal in zip(policy_q32, q_row, mask)
            if legal
        )
        if not math.isclose(
            policy_value,
            value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise DistillationDatasetError("value target is not policy/Q consistent")

        states.append(list(row["model_input"]["encoded_vector"]))
        masks.append(mask)
        policies_q32.append(policy_q32)
        values.append(value)
        q_values.append(q_row)
        q_errors.append(error_row)
        best_indices.append(best)
        method_codes.append(method_code)
        provenance["row_identity_sha256"].append(row["row_identity_sha256"])
        provenance["row_sha256"].append(row["row_sha256"])
        provenance["split_group_sha256"].append(assignment["split_group_sha256"])
        provenance["full_deal_commitment_sha256"].append(
            assignment["full_deal_commitment_sha256"]
        )
        provenance["public_root_family_commitment_sha256"].append(
            assignment["public_root_family_commitment_sha256"]
        )
        provenance["information_digest"].append(
            row["model_input"]["information_digest"]
        )
        provenance["phase"].append(row["stratum"]["phase"])
        provenance["actor"].append(row["stratum"]["actor"])
        provenance["visible_joker_count"].append(
            int(row["stratum"]["visible_joker_count"])
        )

    policy_array = _readonly(np.asarray(policies_q32), _U64)
    q_array = _readonly(np.asarray(q_values), _F64)
    return DistillationSplit(
        name=name,
        states=_readonly(np.asarray(states), _F32),
        legal_action_masks=_readonly(np.asarray(masks), _BOOL),
        policy_q32=policy_array,
        policy_targets=_readonly(
            policy_array.astype(_F64) / float(Q32_DENOMINATOR),
            _F64,
        ),
        value_targets=_readonly(np.asarray(values), _F64),
        q_targets=q_array,
        q_target_masks=_readonly(np.isfinite(q_array), _BOOL),
        q_standard_errors=_readonly(np.asarray(q_errors), _F64),
        best_action_indices=_readonly(np.asarray(best_indices), _I64),
        target_method_codes=_readonly(np.asarray(method_codes), _I8),
        provenance=DistillationSplitProvenance(
            **{field: tuple(items) for field, items in provenance.items()}
        ),
    )


def _verify_split(split: DistillationSplit) -> None:
    if not isinstance(split, DistillationSplit) or split.name not in SPLIT_NAMES:
        raise DistillationDatasetError("invalid locked split object")
    count = len(split)
    arrays = {
        "states": (split.states, (count, INFOSET_VECTOR_DIM), _F32),
        "legal_action_masks": (split.legal_action_masks, (count, 27), _BOOL),
        "policy_q32": (split.policy_q32, (count, 27), _U64),
        "policy_targets": (split.policy_targets, (count, 27), _F64),
        "value_targets": (split.value_targets, (count,), _F64),
        "q_targets": (split.q_targets, (count, 27), _F64),
        "q_target_masks": (split.q_target_masks, (count, 27), _BOOL),
        "q_standard_errors": (split.q_standard_errors, (count, 27), _F64),
        "best_action_indices": (split.best_action_indices, (count,), _I64),
        "target_method_codes": (split.target_method_codes, (count,), _I8),
    }
    for label, (array, shape, dtype) in arrays.items():
        if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != dtype:
            raise DistillationDatasetError(
                f"{split.name}.{label}: exact shape/dtype contract required"
            )
        if array.flags.writeable or not array.flags.c_contiguous:
            raise DistillationDatasetError(
                f"{split.name}.{label}: immutable C-contiguous array required"
            )
    if not np.all(np.isfinite(split.states)) or not np.all(
        (split.states == 0.0) | (split.states == 1.0)
    ):
        raise DistillationDatasetError("state vectors must be finite exact binary values")
    if count:
        if not np.all(split.legal_action_masks.any(axis=1)):
            raise DistillationDatasetError("every row requires at least one legal action")
        if not np.array_equal(split.q_target_masks, split.legal_action_masks):
            raise DistillationDatasetError("Q target mask must equal the legal-action mask")
        if not np.all(split.policy_q32.sum(axis=1) == Q32_DENOMINATOR):
            raise DistillationDatasetError("policy Q32 rows do not conserve mass")
        if np.any(split.policy_q32[~split.legal_action_masks] != 0):
            raise DistillationDatasetError("illegal actions contain policy mass")
        expected_policy = split.policy_q32.astype(_F64) / float(Q32_DENOMINATOR)
        if not np.array_equal(split.policy_targets, expected_policy):
            raise DistillationDatasetError("float policy targets are not exact Q32 projections")
        if not np.all(np.isfinite(split.value_targets)):
            raise DistillationDatasetError("value targets must be finite")
        if not np.all(np.isfinite(split.q_targets[split.q_target_masks])):
            raise DistillationDatasetError("legal Q targets must be finite")
        if not np.all(np.isnan(split.q_targets[~split.q_target_masks])):
            raise DistillationDatasetError("illegal Q target slots must be NaN")
        if not np.all(np.isfinite(split.q_standard_errors[split.q_target_masks])) or np.any(
            split.q_standard_errors[split.q_target_masks] < 0.0
        ):
            raise DistillationDatasetError("legal Q standard errors are invalid")
        if not np.all(np.isnan(split.q_standard_errors[~split.q_target_masks])):
            raise DistillationDatasetError("illegal Q standard-error slots must be NaN")
        if np.any(split.best_action_indices < 0) or np.any(split.best_action_indices >= 27):
            raise DistillationDatasetError("best action index is out of range")
        if not np.all(
            split.legal_action_masks[
                np.arange(count), split.best_action_indices.astype(np.intp)
            ]
        ):
            raise DistillationDatasetError("best action index is illegal")
        if not set(int(value) for value in split.target_method_codes).issubset(
            set(TARGET_METHOD_CODES.values())
        ):
            raise DistillationDatasetError("unknown target method code")
        recomputed_values = np.sum(
            split.policy_targets
            * np.where(split.q_target_masks, split.q_targets, 0.0),
            axis=1,
            dtype=np.float64,
        )
        if not np.allclose(
            split.value_targets,
            recomputed_values,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise DistillationDatasetError("policy/value/Q targets are inconsistent")

    provenance = split.provenance
    for field in DistillationSplitProvenance.__dataclass_fields__:
        values = getattr(provenance, field)
        if not isinstance(values, tuple) or len(values) != count:
            raise DistillationDatasetError(
                f"{split.name}.provenance.{field}: row-aligned tuple required"
            )
    for field in (
        "row_identity_sha256",
        "row_sha256",
        "split_group_sha256",
        "full_deal_commitment_sha256",
        "public_root_family_commitment_sha256",
        "information_digest",
    ):
        for value in getattr(provenance, field):
            _sha(value, label=f"{split.name}.provenance.{field}")
    for index in range(count):
        try:
            key = decode_infoset_key(split.states[index])
        except (TypeError, ValueError) as exc:
            raise DistillationDatasetError(
                f"{split.name}: state {index} fails lossless decoder"
            ) from exc
        phase = provenance.phase[index]
        actor = provenance.actor[index]
        if phase not in _PHASE_CONTRACT or _PHASE_CONTRACT[phase] != (actor, key.turn):
            raise DistillationDatasetError("phase/actor/turn provenance mismatch")
        if key.phase != phase or key.actor != actor:
            raise DistillationDatasetError("state and phase/actor provenance differ")
        if key.digest() != provenance.information_digest[index]:
            raise DistillationDatasetError("state information digest mismatch")
        joker_count = provenance.visible_joker_count[index]
        if joker_count not in (0, 1, 2) or _visible_joker_count(key) != joker_count:
            raise DistillationDatasetError("state and Joker stratum provenance differ")
        expected_mask = np.asarray(legal_action_mask(key), dtype=_BOOL)
        if not np.array_equal(expected_mask, split.legal_action_masks[index]):
            raise DistillationDatasetError("state and legal-action mask differ")


def _provenance_audit(
    splits: Mapping[str, DistillationSplit],
) -> dict[str, Any]:
    owner_fields = (
        "full_deal_commitment_sha256",
        "public_root_family_commitment_sha256",
        "split_group_sha256",
        "information_digest",
    )
    owners: dict[str, dict[str, str]] = {field: {} for field in owner_fields}
    root_family_deal: dict[str, str] = {}
    counts = {field: 0 for field in owner_fields}
    per_split: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        provenance = splits[name].provenance
        per_split[name] = {
            "row_count": len(splits[name]),
            "split_group_count": len(set(provenance.split_group_sha256)),
            "full_deal_count": len(set(provenance.full_deal_commitment_sha256)),
            "public_root_family_count": len(
                set(provenance.public_root_family_commitment_sha256)
            ),
            "row_identity_order_sha256": canonical_sha256(
                list(provenance.row_identity_sha256)
            ),
            "row_content_order_sha256": canonical_sha256(
                list(provenance.row_sha256)
            ),
            "split_group_set_sha256": canonical_sha256(
                sorted(set(provenance.split_group_sha256))
            ),
            "full_deal_set_sha256": canonical_sha256(
                sorted(set(provenance.full_deal_commitment_sha256))
            ),
            "public_root_family_set_sha256": canonical_sha256(
                sorted(set(provenance.public_root_family_commitment_sha256))
            ),
        }
        for index in range(len(splits[name])):
            family = provenance.public_root_family_commitment_sha256[index]
            deal = provenance.full_deal_commitment_sha256[index]
            prior_deal = root_family_deal.setdefault(family, deal)
            if prior_deal != deal:
                raise DistillationDatasetError(
                    "one public-root family maps to multiple full deals"
                )
            for field in owner_fields:
                value = getattr(provenance, field)[index]
                prior = owners[field].setdefault(value, name)
                if prior != name:
                    counts[field] += 1
    if any(counts.values()):
        raise DistillationDatasetError(
            f"locked dataset has cross-split provenance overlap: {counts}"
        )
    return {
        "checked": True,
        "assignment_source": "teacher_locked_prelabel_full_deal_split_only",
        "random_resplit_performed": False,
        "cross_split_overlap_counts": counts,
        "cross_split_overlap_count": 0,
        "public_root_family_disjoint": True,
        "full_deal_disjoint": True,
        "information_digest_disjoint": True,
        "each_public_root_family_has_one_full_deal": True,
        "per_split": per_split,
    }


def _split_tensor_manifest(split: DistillationSplit) -> dict[str, Any]:
    arrays = {
        "state": split.states,
        "legal_action_mask": split.legal_action_masks,
        "policy_q32": split.policy_q32,
        "policy_target": split.policy_targets,
        "value_target": split.value_targets,
        "q_target": split.q_targets,
        "q_target_mask": split.q_target_masks,
        "q_standard_error": split.q_standard_errors,
        "best_action_index": split.best_action_indices,
        "target_method_code": split.target_method_codes,
    }
    return {
        "schema": DISTILLATION_SPLIT_SCHEMA,
        "row_count": len(split),
        "array_sha256": {
            name: _array_sha256(array) for name, array in sorted(arrays.items())
        },
        "method_counts": {
            method: int(np.sum(split.target_method_codes == code))
            for method, code in TARGET_METHOD_CODES.items()
        },
    }


def _build_manifest(
    source: VerifiedTeacherBundle,
    splits: Mapping[str, DistillationSplit],
) -> dict[str, Any]:
    audit = _provenance_audit(splits)
    stratum_counts = {
        f"{phase}_{actor}_joker_{joker}": 0
        for phase, (actor, _turn) in _PHASE_CONTRACT.items()
        for joker in range(3)
    }
    for split in splits.values():
        for phase, actor, joker in zip(
            split.provenance.phase,
            split.provenance.actor,
            split.provenance.visible_joker_count,
        ):
            stratum_counts[f"{phase}_{actor}_joker_{joker}"] += 1
    manifest: dict[str, Any] = {
        "schema": DISTILLATION_DATASET_SCHEMA,
        "artifact_kind": "verified_in_memory_t3_t4_policy_value_q_dataset",
        "opt_in_only": True,
        "training_performed": False,
        "promotion_eligible": False,
        "serving_changed": False,
        "global_policy_claimed": False,
        "opponent_hidden_cards_in_model_input": False,
        "audit_commitments_exposed_as_model_features": False,
        "source_contract": {
            "teacher_bundle_schema": TEACHER_BUNDLE_SCHEMA,
            "teacher_manifest_sha256": source.manifest["manifest_sha256"],
            "teacher_row_content_set_sha256": source.manifest[
                "row_content_set_sha256"
            ],
            "teacher_row_identity_set_sha256": source.manifest[
                "row_identity_set_sha256"
            ],
        },
        "encoder_contract": {
            "schema": INFOSET_ENCODER_SCHEMA,
            "manifest_sha256": INFOSET_ENCODER_MANIFEST_SHA256,
            "vector_dimension": INFOSET_VECTOR_DIM,
        },
        "action_semantics_sha256": ACTION_SEMANTICS_SHA256,
        "split_names": list(SPLIT_NAMES),
        "split_assignment_source": "verified_teacher_split_assignment",
        "random_resplit_performed": False,
        "training_sample_keys": list(TRAINING_SAMPLE_KEYS),
        "model_feature_keys": ["state"],
        "target_keys": list(TRAINING_SAMPLE_KEYS[1:]),
        "policy_target_semantics": "teacher_policy_exact_q32_projection",
        "value_target_semantics": "teacher_policy_value_acting_player",
        "q_target_semantics": (
            "all_legal_actions_teacher_payoff_with_method_code_and_uncertainty"
        ),
        "target_method_codes": dict(TARGET_METHOD_CODES),
        "split_tensors": {
            name: _split_tensor_manifest(splits[name]) for name in SPLIT_NAMES
        },
        "phase_actor_joker_counts": stratum_counts,
        "provenance_overlap_audit": audit,
        "remaining_integration": [
            "generalizing_policy_value_q_model_training",
            "root_family_disjoint_model_evaluation",
            "locked_model_promotion_gate",
        ],
    }
    manifest["dataset_manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _materialize(
    source: VerifiedTeacherBundle,
) -> tuple[Mapping[str, DistillationSplit], Mapping[str, Any]]:
    rows_by_split = {
        name: [
            row
            for row in source.rows
            if row["split_assignment"]["split"] == name
        ]
        for name in SPLIT_NAMES
    }
    splits = MappingProxyType(
        {name: _build_split(name, rows_by_split[name]) for name in SPLIT_NAMES}
    )
    for split in splits.values():
        _verify_split(split)
    manifest = MappingProxyType(_build_manifest(source, splits))
    return splits, manifest


def _verify_required_splits(
    splits: Mapping[str, DistillationSplit],
    required_splits: Sequence[str],
) -> None:
    if isinstance(required_splits, (str, bytes)) or not isinstance(
        required_splits, Sequence
    ):
        raise TypeError("required_splits must be a sequence of locked split names")
    names = tuple(required_splits)
    if len(names) != len(set(names)) or any(name not in SPLIT_NAMES for name in names):
        raise DistillationDatasetError("required_splits contains duplicates or unknown names")
    empty = [name for name in names if len(splits[name]) == 0]
    if empty:
        raise DistillationDatasetError(f"required locked splits are empty: {empty}")


def load_distillation_dataset(
    teacher_bundle_dir: str | Path,
    *,
    expected_teacher_manifest_sha256: str,
    required_splits: Sequence[str] = SPLIT_NAMES,
) -> VerifiedDistillationDataset:
    """Load a pinned teacher bundle into immutable split-specific tensors."""

    expected = _sha(
        expected_teacher_manifest_sha256,
        label="expected_teacher_manifest_sha256",
    )
    try:
        source = _VERIFY_TEACHER_BUNDLE(teacher_bundle_dir)
    except (TeacherContractError, OSError, TypeError, ValueError) as exc:
        raise DistillationDatasetError(f"teacher bundle verification failed: {exc}") from exc
    if source.manifest["manifest_sha256"] != expected:
        raise DistillationDatasetError("teacher manifest does not match the pinned SHA-256")
    splits, manifest = _materialize(source)
    _verify_required_splits(splits, required_splits)
    dataset = VerifiedDistillationDataset(
        source_root=source.root,
        source_teacher_manifest_sha256=expected,
        splits=splits,
        manifest=manifest,
    )
    return _verify_in_memory(dataset, source=source)


def _verify_in_memory(
    dataset: VerifiedDistillationDataset,
    *,
    source: VerifiedTeacherBundle,
) -> VerifiedDistillationDataset:
    if not isinstance(dataset, VerifiedDistillationDataset):
        raise TypeError("dataset must be VerifiedDistillationDataset")
    _sha(
        dataset.source_teacher_manifest_sha256,
        label="source_teacher_manifest_sha256",
    )
    if source.root != dataset.source_root:
        raise DistillationDatasetError("dataset source root changed")
    if source.manifest["manifest_sha256"] != dataset.source_teacher_manifest_sha256:
        raise DistillationDatasetError("dataset source manifest provenance mismatch")
    if set(dataset.splits) != set(SPLIT_NAMES):
        raise DistillationDatasetError("dataset must contain exactly the locked splits")
    for name in SPLIT_NAMES:
        if dataset.splits[name].name != name:
            raise DistillationDatasetError("dataset split name/key mismatch")
        _verify_split(dataset.splits[name])
    rebuilt_current = _build_manifest(source, dataset.splits)
    if dict(dataset.manifest) != rebuilt_current:
        raise DistillationDatasetError("dataset manifest or tensor content mismatch")
    expected_splits, expected_manifest = _materialize(source)
    if dict(dataset.manifest) != dict(expected_manifest):
        raise DistillationDatasetError(
            "dataset tensors do not deterministically match the verified teacher source"
        )
    # The manifest binds every expected array hash and every ordered row/content
    # commitment, so equality is a complete deterministic materialization check.
    for name in SPLIT_NAMES:
        if _split_tensor_manifest(dataset.splits[name]) != _split_tensor_manifest(
            expected_splits[name]
        ):
            raise DistillationDatasetError("dataset split differs from teacher materialization")
    return dataset


def verify_distillation_dataset(
    dataset: VerifiedDistillationDataset,
) -> VerifiedDistillationDataset:
    """Reverify both the current source directory and the in-memory tensors."""

    if not isinstance(dataset, VerifiedDistillationDataset):
        raise TypeError("dataset must be VerifiedDistillationDataset")
    try:
        source = _VERIFY_TEACHER_BUNDLE(dataset.source_root)
    except (TeacherContractError, OSError, TypeError, ValueError) as exc:
        raise DistillationDatasetError(f"teacher source re-verification failed: {exc}") from exc
    return _verify_in_memory(dataset, source=source)


def _runtime_semantic_state(
    value: Any,
    active: set[int] | None = None,
) -> tuple[Any, ...]:
    """Build an immutable code/default/closure descriptor without a hasher."""

    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if value is Ellipsis:
        return ("ellipsis",)
    if value is NotImplemented:
        return ("not_implemented",)

    seen = active if active is not None else set()
    identity = id(value)
    if identity in seen:
        value_type = type(value)
        return (
            "recursive_reference",
            value_type.__module__,
            value_type.__qualname__,
        )
    seen.add(identity)
    try:
        if isinstance(value, types.CodeType):
            return (
                "code",
                value.co_argcount,
                value.co_posonlyargcount,
                value.co_kwonlyargcount,
                value.co_nlocals,
                value.co_stacksize,
                value.co_flags,
                value.co_code,
                getattr(value, "co_exceptiontable", b""),
                tuple(_runtime_semantic_state(item, seen) for item in value.co_consts),
                tuple(value.co_names),
                tuple(value.co_varnames),
                tuple(value.co_freevars),
                tuple(value.co_cellvars),
            )
        if isinstance(value, types.FunctionType):
            closure: list[tuple[Any, ...]] = []
            for cell in value.__closure__ or ():
                try:
                    contents = cell.cell_contents
                except ValueError:
                    closure.append(("empty_cell",))
                else:
                    closure.append(_runtime_semantic_state(contents, seen))
            wrapped = getattr(value, "__wrapped__", None)
            return (
                "python_function",
                value.__module__,
                value.__qualname__,
                value.__name__,
                _runtime_semantic_state(value.__code__, seen),
                _runtime_semantic_state(value.__defaults__, seen),
                _runtime_semantic_state(value.__kwdefaults__, seen),
                tuple(closure),
                (
                    ("none",)
                    if wrapped is None
                    else _runtime_semantic_state(wrapped, seen)
                ),
            )
        if isinstance(value, types.MethodType):
            owner = value.__self__
            owner_type = owner if isinstance(owner, type) else type(owner)
            return (
                "bound_method",
                _runtime_semantic_state(value.__func__, seen),
                owner_type.__module__,
                owner_type.__qualname__,
            )
        if isinstance(value, type):
            return ("class", value.__module__, value.__qualname__)
        if isinstance(value, tuple):
            return (
                "tuple",
                tuple(_runtime_semantic_state(item, seen) for item in value),
            )
        if isinstance(value, list):
            return (
                "list",
                tuple(_runtime_semantic_state(item, seen) for item in value),
            )
        if isinstance(value, (set, frozenset)):
            items = [_runtime_semantic_state(item, seen) for item in value]
            items.sort(key=repr)
            return (type(value).__name__, tuple(items))
        if isinstance(value, Mapping):
            items = [
                (
                    _runtime_semantic_state(key, seen),
                    _runtime_semantic_state(item, seen),
                )
                for key, item in value.items()
            ]
            items.sort(key=lambda pair: repr(pair[0]))
            return ("mapping", tuple(items))
        if isinstance(value, types.ModuleType):
            return ("module", value.__name__)
        if callable(value):
            owner_class = getattr(value, "__objclass__", None)
            return (
                "native_callable",
                type(value).__module__,
                type(value).__qualname__,
                getattr(value, "__module__", None),
                getattr(value, "__qualname__", None),
                getattr(value, "__name__", None),
                getattr(value, "__text_signature__", None),
                (
                    None
                    if owner_class is None
                    else (owner_class.__module__, owner_class.__qualname__)
                ),
            )
        raise DistillationDatasetError(
            "unsupported transitive runtime semantic value: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )
    finally:
        seen.remove(identity)


def _module_owned_semantic_anchors() -> tuple[
    tuple[str, types.ModuleType, str, str, tuple[Any, ...]], ...
]:
    specifications = (
        (
            "teacher",
            _teacher_module,
            "_TEACHER_RUNTIME_SEMANTIC_ANCHOR",
            "_TEACHER_RUNTIME_SEMANTIC_ANCHOR_MIRROR",
        ),
        (
            "encoder",
            _encoder_module,
            "_INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR",
            "_INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR_MIRROR",
        ),
    )
    anchors: list[
        tuple[str, types.ModuleType, str, str, tuple[Any, ...]]
    ] = []
    for label, module, primary_name, mirror_name in specifications:
        primary = getattr(module, primary_name, None)
        mirror = getattr(module, mirror_name, None)
        try:
            registered = registered_module_function_anchor(
                module.__name__, vars(module)
            )
            verify_module_function_anchor(vars(module), registered)
        except RuntimeError as exc:
            raise DistillationDatasetError(
                f"{label} pre-consumer-import runtime semantic registry drift"
            ) from exc
        if (
            primary is not registered
            or mirror is not registered
            or not isinstance(registered, tuple)
            or not registered
        ):
            raise DistillationDatasetError(
                f"{label} pre-consumer-import runtime semantic registry drift"
            )
        anchors.append((label, module, primary_name, mirror_name, registered))
    return tuple(anchors)


_CANONICAL_TRANSITIVE_MODULE_ANCHORS = _module_owned_semantic_anchors()


def _capture_transitive_module_functions() -> tuple[
    tuple[
        str,
        types.ModuleType,
        str,
        types.FunctionType,
        types.CodeType,
        tuple[Any, ...],
    ],
    ...,
]:
    captured: list[
        tuple[
            str,
            types.ModuleType,
            str,
            types.FunctionType,
            types.CodeType,
            tuple[Any, ...],
        ]
    ] = []
    for label, module, _primary, _mirror, anchor in (
        _CANONICAL_TRANSITIVE_MODULE_ANCHORS
    ):
        names: set[str] = set()
        for entry in anchor:
            if not isinstance(entry, tuple) or len(entry) != 4:
                raise DistillationDatasetError(
                    f"{label} module-owned semantic anchor entry malformed"
                )
            name, target, original_code, expected_state = entry
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(target, types.FunctionType)
                or not isinstance(original_code, types.CodeType)
                or not isinstance(expected_state, tuple)
            ):
                raise DistillationDatasetError(
                    f"{label} module-owned semantic anchor entry invalid"
                )
            names.add(name)
            current = getattr(module, name, None)
            if current is not target or current.__code__ is not original_code:
                raise DistillationDatasetError(
                    f"{label} pre-consumer-import runtime semantic drift: {name}"
                )
            if _runtime_semantic_state(current) != expected_state:
                raise DistillationDatasetError(
                    f"{label} pre-consumer-import runtime state drift: {name}"
                )
            captured.append(
                (
                    f"{label}.{name}",
                    module,
                    name,
                    target,
                    original_code,
                    expected_state,
                )
            )
    return tuple(captured)


_CANONICAL_TRANSITIVE_MODULE_FUNCTIONS = _capture_transitive_module_functions()


def _assert_transitive_runtime_anchors() -> None:
    """Verify producer import records without reading or mutating artifacts."""

    for label, module, primary_name, mirror_name, anchor in (
        _CANONICAL_TRANSITIVE_MODULE_ANCHORS
    ):
        try:
            registered = registered_module_function_anchor(
                module.__name__, vars(module)
            )
        except RuntimeError as exc:
            raise DistillationDatasetError(
                f"dataset transitive runtime registry drift: {label}"
            ) from exc
        if (
            registered is not anchor
            or getattr(module, primary_name, None) is not anchor
            or getattr(module, mirror_name, None) is not anchor
        ):
            raise DistillationDatasetError(
                f"dataset transitive runtime anchor alias drift: {label}"
            )
        try:
            verify_module_function_anchor(vars(module), registered)
        except RuntimeError as exc:
            raise DistillationDatasetError(
                f"dataset transitive runtime semantic drift: {label}"
            ) from exc
    for (
        label,
        module,
        name,
        canonical,
        original_code,
        expected_state,
    ) in _CANONICAL_TRANSITIVE_MODULE_FUNCTIONS:
        current = getattr(module, name, None)
        if current is not canonical:
            raise DistillationDatasetError(
                f"dataset transitive runtime alias drift: {label}"
            )
        if current.__code__ is not original_code:
            raise DistillationDatasetError(
                f"dataset transitive runtime code drift: {label}"
            )
        if _runtime_semantic_state(current) != expected_state:
            raise DistillationDatasetError(
                f"dataset transitive runtime semantics drift: {label}"
            )


_CANONICAL_RUNTIME_CALLABLES = (
    ("canonical_json", canonical_json),
    ("canonical_sha256", canonical_sha256),
    ("verify_teacher_bundle", verify_teacher_bundle),
    ("_VERIFY_TEACHER_BUNDLE", _VERIFY_TEACHER_BUNDLE),
    ("decode_infoset_key", decode_infoset_key),
    ("legal_action_mask", legal_action_mask),
    ("_sha", _sha),
    ("_readonly", _readonly),
    ("_array_sha256", _array_sha256),
    ("_visible_joker_count", _visible_joker_count),
    ("_method_targets", _method_targets),
    ("_empty_provenance", _empty_provenance),
    ("_build_split", _build_split),
    ("_verify_split", _verify_split),
    ("_provenance_audit", _provenance_audit),
    ("_split_tensor_manifest", _split_tensor_manifest),
    ("_build_manifest", _build_manifest),
    ("_materialize", _materialize),
    ("_verify_required_splits", _verify_required_splits),
    ("_verify_in_memory", _verify_in_memory),
    ("_runtime_semantic_state", _runtime_semantic_state),
    (
        "registered_module_function_anchor",
        registered_module_function_anchor,
    ),
    ("verify_module_function_anchor", verify_module_function_anchor),
    (
        "_capture_transitive_module_functions",
        _capture_transitive_module_functions,
    ),
    ("_module_owned_semantic_anchors", _module_owned_semantic_anchors),
    ("_assert_transitive_runtime_anchors", _assert_transitive_runtime_anchors),
)
_CANONICAL_RUNTIME_VALUES = (
    ("DISTILLATION_DATASET_SCHEMA", DISTILLATION_DATASET_SCHEMA),
    ("DISTILLATION_SPLIT_SCHEMA", DISTILLATION_SPLIT_SCHEMA),
    ("MCCFR_SOLVER_METHOD", MCCFR_SOLVER_METHOD),
    ("T4_BTN_EXACT_METHOD", T4_BTN_EXACT_METHOD),
    ("Q32_DENOMINATOR", Q32_DENOMINATOR),
    ("SPLIT_NAMES", SPLIT_NAMES),
    ("INFOSET_ENCODER_SCHEMA", INFOSET_ENCODER_SCHEMA),
    ("INFOSET_ENCODER_MANIFEST_SHA256", INFOSET_ENCODER_MANIFEST_SHA256),
    ("INFOSET_VECTOR_DIM", INFOSET_VECTOR_DIM),
    ("ACTION_SEMANTICS_SHA256", ACTION_SEMANTICS_SHA256),
    ("TARGET_METHOD_CODES", TARGET_METHOD_CODES),
    ("TRAINING_SAMPLE_KEYS", TRAINING_SAMPLE_KEYS),
    (
        "_CANONICAL_TRANSITIVE_MODULE_FUNCTIONS",
        _CANONICAL_TRANSITIVE_MODULE_FUNCTIONS,
    ),
    (
        "_CANONICAL_TRANSITIVE_MODULE_ANCHORS",
        _CANONICAL_TRANSITIVE_MODULE_ANCHORS,
    ),
)


def _guard_runtime_bindings(function: Any) -> Any:
    """Close over import-time verifier semantics for public trust boundaries."""

    callables = _CANONICAL_RUNTIME_CALLABLES
    values = _CANONICAL_RUNTIME_VALUES
    callable_states = tuple(
        (name, _runtime_semantic_state(canonical))
        for name, canonical in callables
    )
    semantic_state = _runtime_semantic_state
    assert_transitive_runtime_anchors = _assert_transitive_runtime_anchors

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        for name, canonical in callables:
            if globals().get(name) is not canonical:
                raise DistillationDatasetError(
                    f"dataset runtime callable alias drift: {name}"
                )
        for name, expected_state in callable_states:
            if semantic_state(globals().get(name)) != expected_state:
                raise DistillationDatasetError(
                    f"dataset runtime callable semantics drift: {name}"
                )
        assert_transitive_runtime_anchors()
        for name, canonical in values:
            current = globals().get(name)
            if type(current) is not type(canonical) or current != canonical:
                raise DistillationDatasetError(
                    f"dataset runtime value binding drift: {name}"
                )
        return function(*args, **kwargs)

    return guarded


load_distillation_dataset = _guard_runtime_bindings(load_distillation_dataset)
verify_distillation_dataset = _guard_runtime_bindings(verify_distillation_dataset)


__all__ = [
    "DISTILLATION_DATASET_SCHEMA",
    "DISTILLATION_SPLIT_SCHEMA",
    "DistillationDatasetError",
    "DistillationSplit",
    "DistillationSplitProvenance",
    "TARGET_METHOD_CODES",
    "TRAINING_SAMPLE_KEYS",
    "VerifiedDistillationDataset",
    "load_distillation_dataset",
    "verify_distillation_dataset",
]
