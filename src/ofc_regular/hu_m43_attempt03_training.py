"""Fail-closed training lifecycle for M4.3 Attempt03 v5.

This module deliberately separates four data-access phases:

* base-fold workers receive only the inherited 200 fit rows and fresh 500
  ``train.fit`` rows;
* fit assembly builds nested OOF base features, five-way meta cross-fit
  predictions, and a disabled runtime candidate without opening any holdout;
* the fresh 200-state pre-calibration split is consumed once by an explicit
  caller and decides whether calibration may be opened;
* only after that Go may an explicit calibration caller fit the fixed L2
  safety head and lock one threshold on the already-sealed 50/50 roles.

The inherited locked holdout is not accepted by any API in this module.
Teacher deltas are offline targets and diagnostics, never runtime inputs or
realized match EV.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import lightgbm
import numpy as np
import scipy
import sklearn
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .action_key import action_key_from_payload
from .hu_m43_fold_training import (
    M43_FROZEN_PROCESS_ENVIRONMENT,
)
from .hu_m43_joint_model_v4 import (
    V4FoldWorkerConfig,
    fit_v4_fold_worker_compatible,
)
from .hu_m43_joint_model_v5 import (
    HU_M43_V5_ARTIFACT_SCHEMA,
    HU_M43_V5_FEATURE_SCHEMA,
    HU_M43_V5_META_FEATURE_DIM,
    HU_M43_V5_MODEL_SCHEMA,
    HU_M43_V5_PROPOSAL_SCHEMA,
    HuM43JointModelV5,
    V5ActionPredictions,
    V5MetaRankerConfig,
    build_v5_stacked_features,
    fit_v5_meta_ranker,
)
from .hu_m43_pilot_contract import canonical_manifest_sha256
from .hu_m4_joint_model import PairedDeltaRiskFoldEstimator
from .train_hu_m4_joint_model import (
    M43FoldJobDefinition,
    M43FoldJobSpec,
    M43FoldTrainingPlan,
    PreparedTeacherSample,
    _m43_sample_identity_sha256,
    _sample_membership_hash,
    build_m43_fold_training_plan,
    prepare_teacher_samples,
    read_teacher_jsonl,
)


M43_ATTEMPT03_FOLD_CLOUD_CONTRACT_SCHEMA = (
    "hu_m43_attempt03_v5_fold_cloud_contract_v1"
)
M43_ATTEMPT03_FOLD_ESTIMATOR_SCHEMA = (
    "hu_m43_attempt03_v5_base_fold_estimator_artifact_v1"
)
M43_ATTEMPT03_FOLD_JOB_MANIFEST_SCHEMA = (
    "hu_m43_attempt03_v5_fold_job_manifest_v1"
)
M43_ATTEMPT03_FOLD_DONE_SCHEMA = "hu_m43_attempt03_v5_fold_done_v1"
M43_ATTEMPT03_FOLD_ASSEMBLY_SCHEMA = (
    "hu_m43_attempt03_v5_fold_assembly_v1"
)
M43_ATTEMPT03_TRAINING_CONFIG_SCHEMA = (
    "hu_m43_attempt03_v5_training_config_v1"
)
M43_ATTEMPT03_TRAINING_FREEZE_SCHEMA = (
    "hu_m43_attempt03_training_pipeline_freeze_v1"
)
M43_ATTEMPT03_FIT_BUNDLE_SCHEMA = "hu_m43_attempt03_v5_fit_bundle_v1"
M43_ATTEMPT03_FIT_MANIFEST_SCHEMA = "hu_m43_attempt03_v5_fit_manifest_v1"
M43_ATTEMPT03_PRECAL_REPORT_SCHEMA = (
    "hu_m43_attempt03_v5_precalibration_report_v1"
)
M43_ATTEMPT03_PRECAL_RECEIPT_SCHEMA = (
    "hu_m43_attempt03_v5_precalibration_receipt_v1"
)
M43_ATTEMPT03_PRECAL_MARKER_SCHEMA = (
    "hu_m43_attempt03_v5_precalibration_consumption_v1"
)
M43_ATTEMPT03_SAFETY_DATASET_SCHEMA = (
    "hu_m43_attempt03_v5_safety_dataset_v1"
)
M43_ATTEMPT03_SAFETY_FIT_SCHEMA = "hu_m43_attempt03_v5_safety_fit_v1"
M43_ATTEMPT03_THRESHOLD_REPORT_SCHEMA = (
    "hu_m43_attempt03_v5_threshold_lock_v1"
)
M43_ATTEMPT03_FINAL_MANIFEST_SCHEMA = (
    "hu_m43_attempt03_v5_final_training_manifest_v1"
)
M43_ATTEMPT03_SCIENCE_CORRECTION_SCHEMA = (
    "hu_m43_attempt03_training_science_correction_v1"
)
M43_ATTEMPT03_SCIENCE_CORRECTION_PATH = (
    "configs/hu_joint_policy_m43_attempt03_training_science_correction.json"
)

M43_ATTEMPT03_FIT_STATES = 700
M43_ATTEMPT03_INHERITED_FIT_STATES = 200
M43_ATTEMPT03_FRESH_FIT_STATES = 500
M43_ATTEMPT03_PRECAL_STATES = 200
M43_ATTEMPT03_FOLDS = 5
M43_ATTEMPT03_FOLD_JOBS = 30
M43_ATTEMPT03_FIXED_THRESHOLDS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.975,
    0.99,
    1.0,
)
M43_ATTEMPT03_STAGE18_PATH = (
    "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl"
)
M43_ATTEMPT03_STAGE18_SHA256 = (
    "e910131715efbec7a116e411a8dc23dcfc9a904f2aef9e0eee9d4913a02dfee7"
)
M43_ATTEMPT03_V5_SOURCE_SHA256 = (
    "64c5da4b694796dd0804c8e3fcc3578e6ceca9d99ff9453d162facd2c3940695"
)
M43_ATTEMPT03_TEACHER_STATUS = (
    "offline_diagnostic_only_not_realized_match_ev_not_runtime_gate"
)

M43_ATTEMPT03_TRAINING_SOURCE_PATHS = (
    M43_ATTEMPT03_SCIENCE_CORRECTION_PATH,
    "scripts/Get-GcpHuM43Attempt03ModelRunStatus.ps1",
    "scripts/HuM43Attempt03ModelSpot.Common.ps1",
    "scripts/Receive-GcpHuM43Attempt03ModelRun.ps1",
    "scripts/Run-HuM43Attempt03ModelJobShard.ps1",
    "scripts/Start-GcpHuM43Attempt03ModelRun.ps1",
    "src/ofc_regular/action_key.py",
    "src/ofc_regular/assemble_hu_m43_attempt03_model.py",
    "src/ofc_regular/build_hu_m43_attempt03_model_job_schedule.py",
    "src/ofc_regular/build_hu_m43_attempt03_model_spot_package.py",
    "src/ofc_regular/hu_m43_attempt03_training.py",
    "src/ofc_regular/hu_m43_fold_training.py",
    "src/ofc_regular/hu_m43_joint_model_v4.py",
    "src/ofc_regular/hu_m43_joint_model_v5.py",
    "src/ofc_regular/hu_m43_pilot_contract.py",
    "src/ofc_regular/hu_m4_joint_model.py",
    "src/ofc_regular/hu_turn3_model.py",
    "src/ofc_regular/prepare_hu_m43_attempt03_fold_training.py",
    "src/ofc_regular/train_hu_m43_attempt03_fold_job.py",
    "src/ofc_regular/train_hu_m4_joint_model.py",
)
M43_ATTEMPT03_WORKER_SOURCE_PATHS = (
    "src/ofc_regular/action_key.py",
    "src/ofc_regular/hu_m43_attempt03_training.py",
    "src/ofc_regular/hu_m43_fold_training.py",
    "src/ofc_regular/hu_m43_joint_model_v4.py",
    "src/ofc_regular/hu_m43_joint_model_v5.py",
    "src/ofc_regular/hu_m43_pilot_contract.py",
    "src/ofc_regular/hu_m4_joint_model.py",
    "src/ofc_regular/hu_turn3_model.py",
    "src/ofc_regular/train_hu_m43_attempt03_fold_job.py",
    "src/ofc_regular/train_hu_m4_joint_model.py",
)

_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
_HEX = frozenset("0123456789abcdef")
_WORKER_FORBIDDEN_KEYS = frozenset(
    {
        "precalibration_path",
        "precal_path",
        "calibration_path",
        "calibration_paths",
        "sealed_calibration_path",
        "locked_holdout",
        "locked_holdout_path",
        "inherited_locked",
    }
)


@dataclass(frozen=True)
class Attempt03TrainingConfig:
    """Executable values bound by the immutable Attempt03 model freeze."""

    model_id: str = "hu-m43-t1-v5-attempt03"
    fold_seed: int = 2026072401
    worker: V4FoldWorkerConfig = field(default_factory=V4FoldWorkerConfig)
    meta: V5MetaRankerConfig = field(default_factory=V5MetaRankerConfig)
    safety_calibrator_c: float = 0.25
    safety_seed: int = 2026072405
    thresholds: tuple[float, ...] = M43_ATTEMPT03_FIXED_THRESHOLDS
    minimum_threshold_fires: int = 10
    maximum_false_positive_rate: float = 0.35
    maximum_p95_loss: float = 25.0
    maximum_p99_loss: float = 40.0
    maximum_max_loss: float = 50.0
    schema: str = M43_ATTEMPT03_TRAINING_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != M43_ATTEMPT03_TRAINING_CONFIG_SCHEMA:
            raise ValueError("unsupported Attempt03 training config schema")
        if not self.model_id:
            raise ValueError("Attempt03 model_id must be non-empty")
        if self.fold_seed != 2026072401:
            raise ValueError("Attempt03 fold seed changed")
        if self.worker.to_manifest() != V4FoldWorkerConfig().to_manifest():
            raise ValueError("Attempt03 v4-compatible base worker changed")
        if self.meta.to_manifest() != V5MetaRankerConfig().to_manifest():
            raise ValueError("Attempt03 v5 meta ranker changed")
        if float(self.safety_calibrator_c) != 0.25:
            raise ValueError("Attempt03 safety C changed")
        if self.safety_seed != 2026072405:
            raise ValueError("Attempt03 safety seed changed")
        normalized = tuple(sorted({float(value) for value in self.thresholds}))
        if normalized != M43_ATTEMPT03_FIXED_THRESHOLDS:
            raise ValueError("Attempt03 corrected threshold grid changed")
        actual = (
            self.minimum_threshold_fires,
            float(self.maximum_false_positive_rate),
            float(self.maximum_p95_loss),
            float(self.maximum_p99_loss),
            float(self.maximum_max_loss),
        )
        if actual != (10, 0.35, 25.0, 40.0, 50.0):
            raise ValueError("Attempt03 calibration constraints changed")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_id": self.model_id,
            "model_schema": HU_M43_V5_MODEL_SCHEMA,
            "proposal_schema": HU_M43_V5_PROPOSAL_SCHEMA,
            "feature_schema": HU_M43_V5_FEATURE_SCHEMA,
            "feature_dim": HU_M43_V5_META_FEATURE_DIM,
            "fit_states": M43_ATTEMPT03_FIT_STATES,
            "cross_fit_folds": M43_ATTEMPT03_FOLDS,
            "fold_jobs": M43_ATTEMPT03_FOLD_JOBS,
            "fold_seed": self.fold_seed,
            "base_worker": self.worker.to_manifest(),
            "meta_ranker": self.meta.to_manifest(),
            "meta_fit_rows": "all_nonbaseline_state_balanced",
            "runtime_meta_fit_source": "fit_identity_clean_base_oof_all_nonbaseline",
            "safety": {
                "estimator": "standardized_l2_logistic",
                "c": float(self.safety_calibrator_c),
                "seed": self.safety_seed,
                "fit_sources": [
                    "fit_base_oof_hard_eligible_only",
                    "sealed_calibration.safety_fit_50",
                ],
            },
            "threshold_lock": {
                "source": "sealed_calibration.threshold_lock_50_only",
                "thresholds": [float(value) for value in self.thresholds],
                "minimum_fires": self.minimum_threshold_fires,
                "maximum_false_positive_rate": self.maximum_false_positive_rate,
                "maximum_p95_loss": self.maximum_p95_loss,
                "maximum_p99_loss": self.maximum_p99_loss,
                "maximum_max_loss": self.maximum_max_loss,
            },
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
            "full_replacement": False,
        }

    @classmethod
    def from_manifest(cls, value: Any) -> "Attempt03TrainingConfig":
        payload = _mapping(value, "Attempt03 training config")
        result = cls(model_id=str(payload.get("model_id", "")))
        if dict(payload) != result.to_manifest():
            raise ValueError("Attempt03 training config manifest changed")
        return result


@dataclass(frozen=True)
class Attempt03OofStatePrediction:
    sample_index: int
    fold_index: int
    predictions: V5ActionPredictions
    base_identity_excluded_from_fit: bool
    meta_identity_excluded_from_fit: bool


@dataclass(frozen=True)
class Attempt03SafetyDataset:
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    rows: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]
    seed_values: frozenset[str]
    observation_fingerprints: frozenset[str]


@dataclass(frozen=True)
class Attempt03FitResult:
    model: HuM43JointModelV5
    oof_predictions: tuple[Attempt03OofStatePrediction, ...]
    safety_dataset: Attempt03SafetyDataset
    report: Mapping[str, Any]


@dataclass(frozen=True)
class Attempt03SafetyFitResult:
    model: HuM43JointModelV5
    report: Mapping[str, Any]
    fit_seed_values: frozenset[str]
    fit_observation_fingerprints: frozenset[str]


@dataclass(frozen=True)
class Attempt03ThresholdResult:
    model: HuM43JointModelV5
    report: Mapping[str, Any]


@dataclass
class Attempt03FoldArtifactProvider:
    estimators: dict[int, PairedDeltaRiskFoldEstimator]
    expected_specs: dict[int, M43FoldJobSpec]
    training_samples: tuple[PreparedTeacherSample, ...]
    assembly_receipt: dict[str, Any]
    consumed: set[int]

    def __call__(
        self,
        spec: M43FoldJobSpec,
        fit_samples: Sequence[PreparedTeacherSample],
    ) -> PairedDeltaRiskFoldEstimator:
        if spec.job_index in self.consumed:
            raise ValueError(f"Attempt03 fold consumed twice: {spec.job_index}")
        expected = self.expected_specs.get(spec.job_index)
        if expected is None or expected.to_manifest() != spec.to_manifest():
            raise ValueError("Attempt03 assembler job spec changed")
        if _m43_sample_identity_sha256(fit_samples) != spec.fit_identity_sha256:
            raise ValueError("Attempt03 assembler fit identities changed")
        estimator = self.estimators.get(spec.job_index)
        if estimator is None:
            raise ValueError(f"Attempt03 fold missing: {spec.job_index}")
        self.consumed.add(spec.job_index)
        return estimator

    def assert_complete(self) -> None:
        if self.consumed != set(range(M43_ATTEMPT03_FOLD_JOBS)):
            raise ValueError("Attempt03 assembler did not consume exact 30 jobs")


def load_attempt03_model_freeze(
    path: str | Path, *, repo_root: str | Path
) -> dict[str, Any]:
    """Validate required semantics without depending on optional added keys."""

    root = Path(repo_root).resolve()
    source = Path(path).resolve()
    payload = _read_mapping(source, "Attempt03 model freeze")
    if (
        payload.get("schema") != "hu_m43_attempt03_model_freeze_v1"
        or payload.get("milestone") != "M4.3-attempt03"
    ):
        raise ValueError("Attempt03 model freeze identity changed")
    implementation = _mapping(payload.get("implementation"), "implementation")
    if implementation.get("module") != "src/ofc_regular/hu_m43_joint_model_v5.py":
        raise ValueError("Attempt03 v5 implementation path changed")
    model_source = (root / str(implementation["module"])).resolve()
    if (
        _file_sha256(model_source) != M43_ATTEMPT03_V5_SOURCE_SHA256
        or implementation.get("file_sha256") != M43_ATTEMPT03_V5_SOURCE_SHA256
    ):
        raise ValueError("Attempt03 v5 implementation SHA changed")
    stage18 = _mapping(implementation.get("stage18_model"), "stage18_model")
    if (
        stage18.get("path") != M43_ATTEMPT03_STAGE18_PATH
        or stage18.get("file_sha256") != M43_ATTEMPT03_STAGE18_SHA256
        or _file_sha256(root / M43_ATTEMPT03_STAGE18_PATH)
        != M43_ATTEMPT03_STAGE18_SHA256
    ):
        raise ValueError("Attempt03 fixed Stage18 model changed")
    base = _mapping(payload.get("base_crossfit"), "base_crossfit")
    worker = _mapping(base.get("worker"), "base_crossfit.worker")
    expected_worker = {
        "paired_se_floor": 0.5,
        "loss": "huber",
        "huber_alpha": 0.9,
        "iterations": 150,
        "max_leaf_nodes": 31,
        "learning_rate": 0.05,
    }
    for key, expected in {
        "folds": 5,
        "fold_seed": 2026072401,
        "outer_inner_jobs": 30,
        "fit_states": 700,
        "baseline_rows_included": False,
    }.items():
        if base.get(key) != expected:
            raise ValueError(f"Attempt03 frozen base field changed: {key}")
    if dict(worker) != expected_worker:
        raise ValueError("Attempt03 frozen base worker changed")
    meta = _mapping(payload.get("meta_ranker"), "meta_ranker")
    expected_meta = {
        "features": "stage18_aware_22_in_diagnostic_order",
        "fit_rows": "all_nonbaseline_actions",
        "baseline_rows_included": False,
        "state_weight_sum": 1.0,
        "target": "offline_raw_teacher_paired_delta",
        "objective": "lightgbm_huber",
        "trees": 180,
        "learning_rate": 0.035,
        "num_leaves": 7,
        "min_child_samples": 50,
        "l2": 8.0,
        "l1": 0.5,
        "random_state": 20260714,
        "proposal_order": "canonical_argmax_across_all_nonbaseline_then_vote_gate",
        "positive_vote_gate": (
            "centered_base_delta_strictly_positive_at_least_3_of_5"
        ),
        "rerank_to_next_eligible_action": False,
    }
    if dict(meta) != expected_meta:
        raise ValueError("Attempt03 frozen meta ranker changed")
    gate = _mapping(payload.get("precalibration_gate"), "precalibration_gate")
    expected_gate = {
        "source": "fresh_one_shot_train.precal_holdout_200",
        "minimum_raw_proposal_positive_rate": 0.4,
        "minimum_eligible_fires": 30,
        "minimum_eligible_mean_delta_per_fire": 0.0,
        "minimum_eligible_delta_per_state": 0.0,
        "maximum_false_positive_rate_delta_le_zero": 0.35,
        "post_result_model_feature_gate_or_threshold_change_allowed": False,
    }
    for key, expected in expected_gate.items():
        if gate.get(key) != expected:
            raise ValueError(f"Attempt03 frozen pre-cal field changed: {key}")
    lcb = _mapping(gate.get("mean_delta_lcb"), "mean_delta_lcb")
    if dict(lcb) != {
        "method": "one_sided_student_t_over_independent_state_clusters",
        "confidence": 0.9,
        "strictly_greater_than": 0.0,
    }:
        raise ValueError("Attempt03 Student-t LCB changed")
    if dict(_mapping(gate.get("selected_action_downside_maximum"), "tails")) != {
        "p95": 25.0,
        "p99": 40.0,
        "max": 50.0,
    }:
        raise ValueError("Attempt03 pre-cal tail gates changed")
    safety = _mapping(payload.get("safety_calibration"), "safety_calibration")
    expected_safety = {
        "estimator": "standardized_l2_logistic",
        "c": 0.25,
        "seed": 2026072405,
        "threshold_grid": list(M43_ATTEMPT03_FIXED_THRESHOLDS),
        "minimum_fires": 10,
        "maximum_false_positive_rate_delta_le_zero": 0.35,
        "minimum_delta_per_state": 0.0,
        "maximum_p95_loss": 25.0,
        "maximum_p99_loss": 40.0,
        "maximum_max_loss": 50.0,
        "repeat_threshold_search_on_same_holdout_allowed": False,
    }
    for key, expected in expected_safety.items():
        if safety.get(key) != expected:
            raise ValueError(f"Attempt03 frozen safety field changed: {key}")
    guards = _mapping(payload.get("activation_guards"), "activation_guards")
    if any(guards.get(key) is not False for key in (
        "current_profile_changed",
        "runtime_policy_activated",
        "full_replacement_enabled",
        "m5_authorized",
    )):
        raise ValueError("Attempt03 activation guard changed")
    return payload


def load_attempt03_training_science_correction(
    path: str | Path, *, repo_root: str | Path
) -> dict[str, Any]:
    """Validate the pre-fit, result-independent scientific correction."""

    root = Path(repo_root).resolve()
    source = Path(path).resolve()
    payload = _read_mapping(source, "Attempt03 training science correction")
    expected_keys = {
        "schema",
        "milestone",
        "status",
        "frozen_at",
        "decision_boundary",
        "parent_plan",
        "parent_model_freeze",
        "corrections",
        "unchanged_frozen_semantics",
        "correction_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("Attempt03 science correction key set changed")
    unsigned = dict(payload)
    declared = _require_sha256(
        unsigned.pop("correction_sha256", None), "science correction"
    )
    if canonical_manifest_sha256(unsigned) != declared:
        raise ValueError("Attempt03 science correction digest changed")
    if (
        payload.get("schema") != M43_ATTEMPT03_SCIENCE_CORRECTION_SCHEMA
        or payload.get("milestone") != "M4.3-attempt03"
        or payload.get("status") != "frozen_pre_fit_without_row_valued_evidence"
        or not isinstance(payload.get("frozen_at"), str)
    ):
        raise ValueError("Attempt03 science correction identity changed")
    boundary = _mapping(payload.get("decision_boundary"), "decision_boundary")
    if dict(boundary) != {
        "cloud_teacher_rows_may_exist": True,
        "attempt03_fit_rows_structurally_inspected": 1,
        "attempt03_fit_row_valued_labels_or_metrics_used": 0,
        "attempt03_model_fit_started": False,
        "precalibration_rows_opened": 0,
        "sealed_calibration_rows_opened": 0,
        "inherited_locked_rows_opened": 0,
        "correction_selected_from_result_values": False,
    }:
        raise ValueError("Attempt03 science correction boundary changed")
    for key, relative, expected_sha in (
        (
            "parent_plan",
            "configs/hu_joint_policy_m43_attempt03.json",
            "2bd1666596ec6e7bf2cab7578006c96587977a8277969edb9cecdb54e24ef6d5",
        ),
        (
            "parent_model_freeze",
            "configs/hu_joint_policy_m43_attempt03_model_freeze.json",
            "8aed10b143172c9d3b2a98fca199e4fbe4324f5956406fea51c2d17fcb614b9d",
        ),
    ):
        binding = _mapping(payload.get(key), key)
        bound_path = (root / relative).resolve()
        if (
            dict(binding) != {"path": relative, "file_sha256": expected_sha}
            or _file_sha256(bound_path) != expected_sha
        ):
            raise ValueError(f"Attempt03 science correction {key} changed")
    corrections = _mapping(payload.get("corrections"), "corrections")
    meta = _mapping(corrections.get("meta_validation"), "meta_validation")
    safety = _mapping(corrections.get("safety_row_scope"), "safety_row_scope")
    weighted = _mapping(
        corrections.get("weighted_standardization"), "weighted_standardization"
    )
    profile = _mapping(corrections.get("profile_gate"), "profile_gate")
    if (
        set(corrections)
        != {
            "meta_validation",
            "safety_row_scope",
            "weighted_standardization",
            "profile_gate",
        }
        or meta.get("internal_meta_oof_performance_claimed") is not False
        or meta.get("base_fold_job_count") != M43_ATTEMPT03_FOLD_JOBS
        or safety.get("fit_rows") != "hard_eligible_nonbaseline_actions_only"
        or safety.get("state_weight_sum_if_any_eligible") != 1.0
        or safety.get("state_weight_sum_if_zero_eligible") != 0.0
        or weighted.get("standard_scaler_sample_weight")
        != "same_state_balanced_weights_as_logistic"
        or profile.get("missing_profile_counts")
        != "fail_exact_profile_balance_gate"
    ):
        raise ValueError("Attempt03 science correction semantics changed")
    unchanged = _mapping(
        payload.get("unchanged_frozen_semantics"), "unchanged_frozen_semantics"
    )
    if (
        unchanged.get("v5_core_path")
        != "src/ofc_regular/hu_m43_joint_model_v5.py"
        or unchanged.get("v5_core_file_sha256")
        != M43_ATTEMPT03_V5_SOURCE_SHA256
        or _file_sha256(root / "src/ofc_regular/hu_m43_joint_model_v5.py")
        != M43_ATTEMPT03_V5_SOURCE_SHA256
        or any(
            unchanged.get(key) is not False
            for key in (
                "base_worker_hyperparameters_changed",
                "meta_ranker_hyperparameters_changed",
                "positive_vote_gate_changed",
                "precalibration_gate_changed",
                "threshold_grid_changed",
                "safety_logistic_c_or_seed_changed",
                "current_profile_mutated",
                "runtime_policy_activated",
                "full_replacement_enabled",
            )
        )
    ):
        raise ValueError("Attempt03 frozen semantic invariants changed")
    return payload


def load_attempt03_training_freeze(
    path: str | Path,
    *,
    repo_root: str | Path,
    expected_model_freeze_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the immutable pre-receive executable training closure.

    The semantic hyperparameters and v5 core were frozen before the cloud
    teacher produced its first row.  This addendum binds the later pipeline
    implementation after one train.fit canary row was structurally inspected
    but before any row-valued label or metric influenced design.  It contains
    no teacher-valued metric and cannot authorize a holdout open.
    """

    root = Path(repo_root).resolve()
    load_attempt03_training_science_correction(
        root / M43_ATTEMPT03_SCIENCE_CORRECTION_PATH,
        repo_root=root,
    )
    source = Path(path).resolve()
    payload = _read_mapping(source, "Attempt03 training pipeline freeze")
    if set(payload) != {
        "schema",
        "milestone",
        "status",
        "frozen_at",
        "decision_boundary",
        "parent_model_freeze",
        "training_config",
        "executable_sources",
        "lifecycle",
        "freeze_sha256",
    }:
        raise ValueError("Attempt03 training freeze key set changed")
    unsigned = dict(payload)
    declared = _require_sha256(
        unsigned.pop("freeze_sha256", None), "training freeze"
    )
    if canonical_manifest_sha256(unsigned) != declared:
        raise ValueError("Attempt03 training freeze digest mismatch")
    if (
        payload.get("schema") != M43_ATTEMPT03_TRAINING_FREEZE_SCHEMA
        or payload.get("milestone") != "M4.3-attempt03"
        or payload.get("status")
        != (
            "frozen_after_one_structural_train_fit_canary_before_"
            "row_valued_design_or_any_holdout_open"
        )
        or not isinstance(payload.get("frozen_at"), str)
        or not payload["frozen_at"]
    ):
        raise ValueError("Attempt03 training freeze identity changed")
    boundary = _mapping(payload.get("decision_boundary"), "decision_boundary")
    expected_boundary = {
        "cloud_teacher_rows_may_exist": True,
        "attempt03_fit_rows_received_locally": 1,
        "attempt03_fit_rows_structurally_inspected": 1,
        "attempt03_fit_jsonl_content_parse_count": 1,
        "attempt03_fit_row_valued_labels_or_metrics_used_for_design": 0,
        "precalibration_rows_opened": 0,
        "sealed_calibration_rows_opened": 0,
        "inherited_locked_rows_opened": 0,
        "source_or_config_selected_from_row_values": False,
    }
    if dict(boundary) != expected_boundary:
        raise ValueError("Attempt03 training freeze data boundary changed")
    parent = _mapping(payload.get("parent_model_freeze"), "parent_model_freeze")
    if set(parent) != {"path", "file_sha256"}:
        raise ValueError("Attempt03 training freeze parent projection changed")
    parent_path = (root / str(parent.get("path"))).resolve()
    if root not in parent_path.parents:
        raise ValueError("Attempt03 parent model freeze escapes repo root")
    parent_sha = _require_sha256(
        parent.get("file_sha256"), "parent model freeze"
    )
    if _file_sha256(parent_path) != parent_sha:
        raise ValueError("Attempt03 parent model freeze SHA changed")
    if expected_model_freeze_path is not None:
        expected_path = Path(expected_model_freeze_path).resolve()
        if expected_path != parent_path or _file_sha256(expected_path) != parent_sha:
            raise ValueError("Attempt03 expected parent model freeze changed")
    load_attempt03_model_freeze(parent_path, repo_root=root)
    training = _mapping(payload.get("training_config"), "training_config")
    expected_config_sha = canonical_manifest_sha256(
        Attempt03TrainingConfig().to_manifest()
    )
    if (
        set(training) != {"schema", "canonical_sha256"}
        or training.get("schema") != M43_ATTEMPT03_TRAINING_CONFIG_SCHEMA
        or training.get("canonical_sha256") != expected_config_sha
    ):
        raise ValueError("Attempt03 training freeze config changed")
    sources = _mapping(payload.get("executable_sources"), "executable_sources")
    if set(sources) != {
        "files",
        "files_sha256",
        "worker_paths",
        "worker_files_sha256",
    }:
        raise ValueError("Attempt03 training source projection changed")
    files = _mapping(sources.get("files"), "executable_sources.files")
    if tuple(sorted(files)) != M43_ATTEMPT03_TRAINING_SOURCE_PATHS:
        raise ValueError("Attempt03 training source path set changed")
    validated_files: dict[str, str] = {}
    for relative in M43_ATTEMPT03_TRAINING_SOURCE_PATHS:
        declared_source_sha = _require_sha256(
            files.get(relative), f"training source {relative}"
        )
        source_path = (root / relative).resolve()
        if root not in source_path.parents or _file_sha256(source_path) != declared_source_sha:
            raise ValueError(f"Attempt03 training source SHA changed: {relative}")
        validated_files[relative] = declared_source_sha
    if sources.get("files_sha256") != canonical_manifest_sha256(validated_files):
        raise ValueError("Attempt03 training source bundle digest changed")
    if tuple(sources.get("worker_paths", ())) != M43_ATTEMPT03_WORKER_SOURCE_PATHS:
        raise ValueError("Attempt03 worker source path set changed")
    worker_files = {
        relative: validated_files[relative]
        for relative in M43_ATTEMPT03_WORKER_SOURCE_PATHS
    }
    if sources.get("worker_files_sha256") != canonical_manifest_sha256(
        worker_files
    ):
        raise ValueError("Attempt03 worker source bundle digest changed")
    lifecycle = _mapping(payload.get("lifecycle"), "lifecycle")
    if dict(lifecycle) != {
        "teacher_row_values_in_freeze": False,
        "holdout_path_in_freeze": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }:
        raise ValueError("Attempt03 training freeze lifecycle changed")
    return payload


def build_attempt03_fold_cloud_contract(
    *,
    inherited_train_path: str | Path,
    fresh_train_fit_path: str | Path,
    fit_receive_receipt_path: str | Path,
    model_freeze_path: str | Path,
    training_freeze_path: str | Path,
    repo_root: str | Path,
    config: Attempt03TrainingConfig = Attempt03TrainingConfig(),
) -> dict[str, Any]:
    """Build a worker contract containing fit700 and no holdout location."""

    root = Path(repo_root).resolve()
    freeze_path = Path(model_freeze_path).resolve()
    load_attempt03_model_freeze(freeze_path, repo_root=root)
    training_freeze_path = Path(training_freeze_path).resolve()
    training_freeze = load_attempt03_training_freeze(
        training_freeze_path,
        repo_root=root,
        expected_model_freeze_path=freeze_path,
    )
    fit_receipt_path = Path(fit_receive_receipt_path).resolve()
    fit_receipt = validate_attempt03_fit_receive_receipt(
        fit_receipt_path,
        inherited_train_path=inherited_train_path,
        fresh_train_fit_path=fresh_train_fit_path,
    )
    paths, rows_by_path, samples, plan = _read_fit_and_plan(
        inherited_train_path,
        fresh_train_fit_path,
        config=config,
        repo_root=root,
    )
    entries = {
        role: _input_entry(path, rows, role=role)
        for role, path, rows in zip(
            ("inherited_attempt02_train", "fresh_train_fit"),
            paths,
            rows_by_path,
            strict=True,
        )
    }
    fold_plan = {
        "outer_folds": 5,
        "inner_folds_per_outer": 5,
        "total_jobs": 30,
        "fit_identity_sha256": _m43_sample_identity_sha256(
            plan.ordered_samples
        ),
        "jobs": [
            {**job.spec.to_manifest(), "job_spec_sha256": job.spec.sha256}
            for job in plan.jobs
        ],
    }
    fold_plan["fold_plan_sha256"] = canonical_manifest_sha256(fold_plan)
    config_manifest = config.to_manifest()
    if training_freeze["training_config"]["canonical_sha256"] != (
        canonical_manifest_sha256(config_manifest)
    ):
        raise ValueError("Attempt03 supplied training config is not frozen")
    unsigned = {
        "schema": M43_ATTEMPT03_FOLD_CLOUD_CONTRACT_SCHEMA,
        "status": "frozen_fit700_only",
        "inputs": entries,
        "input_bundle_sha256": canonical_manifest_sha256(entries),
        "fold_plan": fold_plan,
        "training_config": config_manifest,
        "training_config_sha256": canonical_manifest_sha256(config_manifest),
        "model_freeze": {
            "file_sha256": _file_sha256(freeze_path),
            "v5_source_sha256": M43_ATTEMPT03_V5_SOURCE_SHA256,
            "stage18_sha256": M43_ATTEMPT03_STAGE18_SHA256,
        },
        "training_freeze": {
            "file_sha256": _file_sha256(training_freeze_path),
            "parent_model_freeze_file_sha256": _file_sha256(freeze_path),
            "training_config_sha256": training_freeze["training_config"][
                "canonical_sha256"
            ],
            "worker_sources": {
                relative: training_freeze["executable_sources"]["files"][relative]
                for relative in M43_ATTEMPT03_WORKER_SOURCE_PATHS
            },
            "worker_sources_sha256": training_freeze["executable_sources"][
                "worker_files_sha256"
            ],
        },
        "fit_receive_receipt": {
            "file_sha256": _file_sha256(fit_receipt_path),
            "run_name": fit_receipt["run_name"],
            "teacher_manifest_sha256": fit_receipt["manifest_sha256"],
            "teacher_schedule_sha256": fit_receipt["schedule_sha256"],
            "teacher_source_sha256": fit_receipt["source_sha256"],
            "fresh_train_fit_sha256": fit_receipt["fresh_train_fit"]["sha256"],
            "verified_roots": 500,
            "precalibration_opened": False,
        },
        "dependencies": _dependency_versions(),
        "process_environment": dict(M43_FROZEN_PROCESS_ENVIRONMENT),
        "worker_input_boundary": {
            "fit_rows": len(samples),
            "fit_inputs": 2,
            "holdout_inputs": 0,
            "teacher_labels_used_for": "base_fold_fit_only",
        },
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    _reject_worker_sensitive(unsigned)
    return {**unsigned, "contract_sha256": canonical_manifest_sha256(unsigned)}


def write_attempt03_fold_cloud_contract(
    path: str | Path, **kwargs: Any
) -> dict[str, Any]:
    payload = build_attempt03_fold_cloud_contract(**kwargs)
    _write_json_exclusive(Path(path), payload)
    return payload


def load_attempt03_fold_cloud_contract(
    path: str | Path,
    *,
    verify_process_environment: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    payload = _read_mapping(Path(path), "Attempt03 fold cloud contract")
    expected_keys = {
        "schema",
        "status",
        "inputs",
        "input_bundle_sha256",
        "fold_plan",
        "training_config",
        "training_config_sha256",
        "model_freeze",
        "training_freeze",
        "fit_receive_receipt",
        "dependencies",
        "process_environment",
        "worker_input_boundary",
        "current_profile_mutated",
        "runtime_policy_activated",
        "full_replacement",
        "contract_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("Attempt03 fold cloud contract key set changed")
    unsigned = dict(payload)
    declared = _require_sha256(unsigned.pop("contract_sha256", None), "contract")
    if canonical_manifest_sha256(unsigned) != declared:
        raise ValueError("Attempt03 fold cloud contract digest mismatch")
    if (
        payload.get("schema") != M43_ATTEMPT03_FOLD_CLOUD_CONTRACT_SCHEMA
        or payload.get("status") != "frozen_fit700_only"
        or any(payload.get(key) is not False for key in (
            "current_profile_mutated",
            "runtime_policy_activated",
            "full_replacement",
        ))
    ):
        raise ValueError("Attempt03 worker lifecycle flags changed")
    _reject_worker_sensitive(payload)
    entries = _mapping(payload.get("inputs"), "inputs")
    if set(entries) != {"inherited_attempt02_train", "fresh_train_fit"}:
        raise ValueError("Attempt03 worker must receive exact fit sources")
    expected_rows = {
        "inherited_attempt02_train": 200,
        "fresh_train_fit": 500,
    }
    for role, rows in expected_rows.items():
        entry = _mapping(entries.get(role), role)
        if (
            set(entry) != {"role", "sha256", "bytes", "rows", "identity_sha256"}
            or entry.get("role") != role
            or _integer(entry.get("rows"), f"{role}.rows") != rows
            or _integer(entry.get("bytes"), f"{role}.bytes", minimum=1) < 1
        ):
            raise ValueError(f"Attempt03 {role} input changed")
        _require_sha256(entry.get("sha256"), f"{role}.sha256")
        _require_sha256(entry.get("identity_sha256"), f"{role}.identity")
    if payload.get("input_bundle_sha256") != canonical_manifest_sha256(entries):
        raise ValueError("Attempt03 input bundle digest changed")
    config = Attempt03TrainingConfig.from_manifest(payload.get("training_config"))
    if payload.get("training_config_sha256") != canonical_manifest_sha256(
        config.to_manifest()
    ):
        raise ValueError("Attempt03 training config digest changed")
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    model_freeze = _mapping(payload.get("model_freeze"), "model_freeze")
    if set(model_freeze) != {
        "file_sha256",
        "v5_source_sha256",
        "stage18_sha256",
    }:
        raise ValueError("Attempt03 model freeze projection changed")
    _require_sha256(model_freeze.get("file_sha256"), "model freeze file")
    if (
        model_freeze.get("v5_source_sha256") != M43_ATTEMPT03_V5_SOURCE_SHA256
        or model_freeze.get("stage18_sha256") != M43_ATTEMPT03_STAGE18_SHA256
        or _file_sha256(root / "src/ofc_regular/hu_m43_joint_model_v5.py")
        != M43_ATTEMPT03_V5_SOURCE_SHA256
    ):
        raise ValueError("Attempt03 model freeze executable projection changed")
    training_freeze = _mapping(
        payload.get("training_freeze"), "training_freeze"
    )
    if set(training_freeze) != {
        "file_sha256",
        "parent_model_freeze_file_sha256",
        "training_config_sha256",
        "worker_sources",
        "worker_sources_sha256",
    }:
        raise ValueError("Attempt03 training freeze projection changed")
    _require_sha256(training_freeze.get("file_sha256"), "training freeze file")
    if (
        training_freeze.get("parent_model_freeze_file_sha256")
        != model_freeze["file_sha256"]
        or training_freeze.get("training_config_sha256")
        != payload["training_config_sha256"]
    ):
        raise ValueError("Attempt03 training/model/config freeze chain changed")
    worker_sources = _mapping(
        training_freeze.get("worker_sources"), "worker_sources"
    )
    if tuple(sorted(worker_sources)) != M43_ATTEMPT03_WORKER_SOURCE_PATHS:
        raise ValueError("Attempt03 worker source projection changed")
    validated_worker_sources: dict[str, str] = {}
    for relative in M43_ATTEMPT03_WORKER_SOURCE_PATHS:
        declared_source_sha = _require_sha256(
            worker_sources.get(relative), f"worker source {relative}"
        )
        if _file_sha256(root / relative) != declared_source_sha:
            raise ValueError(f"Attempt03 worker source SHA changed: {relative}")
        validated_worker_sources[relative] = declared_source_sha
    if training_freeze.get("worker_sources_sha256") != canonical_manifest_sha256(
        validated_worker_sources
    ):
        raise ValueError("Attempt03 worker source bundle digest changed")
    receipt = _mapping(payload.get("fit_receive_receipt"), "fit receive receipt")
    if set(receipt) != {
        "file_sha256",
        "run_name",
        "teacher_manifest_sha256",
        "teacher_schedule_sha256",
        "teacher_source_sha256",
        "fresh_train_fit_sha256",
        "verified_roots",
        "precalibration_opened",
    }:
        raise ValueError("Attempt03 fit receipt projection changed")
    for key in (
        "file_sha256",
        "teacher_manifest_sha256",
        "teacher_schedule_sha256",
        "teacher_source_sha256",
        "fresh_train_fit_sha256",
    ):
        _require_sha256(receipt.get(key), f"fit receipt {key}")
    if (
        not isinstance(receipt.get("run_name"), str)
        or not receipt["run_name"]
        or receipt.get("verified_roots") != 500
        or receipt.get("precalibration_opened") is not False
    ):
        raise ValueError("Attempt03 fit receipt projection lifecycle changed")
    _validate_fold_plan(payload.get("fold_plan"))
    if dict(_mapping(payload.get("dependencies"), "dependencies")) != (
        _dependency_versions()
    ):
        raise RuntimeError("Attempt03 training dependency versions changed")
    if dict(_mapping(payload.get("process_environment"), "environment")) != dict(
        M43_FROZEN_PROCESS_ENVIRONMENT
    ):
        raise ValueError("Attempt03 process environment contract changed")
    if _mapping(payload.get("worker_input_boundary"), "worker boundary") != {
        "fit_rows": 700,
        "fit_inputs": 2,
        "holdout_inputs": 0,
        "teacher_labels_used_for": "base_fold_fit_only",
    }:
        raise ValueError("Attempt03 worker input boundary changed")
    if verify_process_environment:
        actual = {
            key: os.environ.get(key) for key in M43_FROZEN_PROCESS_ENVIRONMENT
        }
        if actual != M43_FROZEN_PROCESS_ENVIRONMENT:
            raise RuntimeError("Attempt03 worker process environment is not frozen")
    return payload


def validate_attempt03_fit_receive_receipt(
    path: str | Path,
    *,
    inherited_train_path: str | Path,
    fresh_train_fit_path: str | Path,
) -> dict[str, Any]:
    """Bind the fit-only receiver output without addressing pre-cal results."""

    payload = _read_mapping(Path(path), "Attempt03 fit receive receipt")
    required_false = (
        "precal_result_prefix_listed",
        "precal_result_downloaded",
        "precal_result_opened",
        "attempt03_contract_finalized",
        "sealed_calibration_opened",
        "inherited_locked_opened",
        "current_profile_mutated",
        "runtime_policy_activated",
    )
    if (
        payload.get("schema")
        != "hu_m43_attempt03_teacher_fit_receive_receipt_v1"
        or payload.get("status")
        != "verified_fresh_train_fit_only_precal_unopened"
        or payload.get("verified_shards") != 50
        or payload.get("verified_roots") != 500
        or any(payload.get(key) is not False for key in required_false)
    ):
        raise ValueError("Attempt03 fit receive lifecycle changed")
    for key in (
        "manifest_sha256",
        "schedule_sha256",
        "source_sha256",
        "startup_sha256",
        "plan_sha256",
        "preflight_receipt_sha256",
    ):
        _require_sha256(payload.get(key), f"fit receipt {key}")
    fresh = _mapping(payload.get("fresh_train_fit"), "fresh_train_fit")
    fresh_path = Path(fresh_train_fit_path).resolve()
    if (
        Path(str(fresh.get("path"))).resolve() != fresh_path
        or fresh.get("rows") != 500
        or fresh.get("sha256") != _file_sha256(fresh_path)
        or len(fresh.get("shards", ())) != 50
    ):
        raise ValueError("Attempt03 fresh fit receipt binding changed")
    downstream = _mapping(payload.get("downstream_inputs"), "downstream_inputs")
    inherited = Path(inherited_train_path).resolve()
    if (
        downstream.get("schema") != "hu_m43_attempt03_fit_downstream_inputs_v1"
        or Path(str(downstream.get("inherited_attempt02_train"))).resolve()
        != inherited
        or Path(str(downstream.get("fresh_train_fit"))).resolve() != fresh_path
        or [Path(str(value)).resolve() for value in downstream.get("fit_source_paths", ())]
        != [inherited, fresh_path]
        or downstream.get("combined_fit_roots") != 700
        or len(downstream.get("original_fresh_shard_paths", ())) != 50
    ):
        raise ValueError("Attempt03 downstream fit inputs changed")
    return payload


def run_attempt03_fold_job(
    *,
    job_index: int,
    inherited_train_path: str | Path,
    fresh_train_fit_path: str | Path,
    fold_cloud_contract_path: str | Path,
    output_dir: str | Path,
    run_name: str,
    source_sha256: str,
    run_manifest_sha256: str,
    input_bundle_sha256: str,
    job_spec_sha256: str,
) -> dict[str, Any]:
    """Fit one frozen base job and publish DONE only after hash validation."""

    index = _integer(job_index, "job_index")
    if index >= M43_ATTEMPT03_FOLD_JOBS:
        raise ValueError("Attempt03 job index is outside exact 30-job grid")
    _validate_run_name(run_name)
    values = {
        "source_sha256": source_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "job_spec_sha256": job_spec_sha256,
    }
    bound = {key: _require_sha256(value, key) for key, value in values.items()}
    contract_path = Path(fold_cloud_contract_path).resolve()
    contract = load_attempt03_fold_cloud_contract(
        contract_path, verify_process_environment=True
    )
    if bound["input_bundle_sha256"] != contract["input_bundle_sha256"]:
        raise ValueError("Attempt03 runner input bundle changed")
    config = Attempt03TrainingConfig.from_manifest(contract["training_config"])
    samples, plan = _load_fit_plan(
        contract,
        inherited_train_path=inherited_train_path,
        fresh_train_fit_path=fresh_train_fit_path,
        config=config,
    )
    definition = plan.jobs[index]
    if definition.spec.sha256 != bound["job_spec_sha256"]:
        raise ValueError("Attempt03 job spec digest changed")
    destination = Path(output_dir)
    artifact_path, manifest_path, done_path = _job_files(destination)
    contract_file_sha = _file_sha256(contract_path)
    if done_path.exists():
        return _validate_existing_done(
            destination,
            definition=definition,
            contract_file_sha256=contract_file_sha,
            training_config_sha256=contract["training_config_sha256"],
            run_name=run_name,
            source_sha256=bound["source_sha256"],
            run_manifest_sha256=bound["run_manifest_sha256"],
            input_bundle_sha256=bound["input_bundle_sha256"],
        )
    if artifact_path.exists() or manifest_path.exists():
        raise FileExistsError("Attempt03 partial fold output exists without DONE")
    estimator = fit_v4_fold_worker_compatible(
        definition.spec, definition.fit_samples, config=config.worker
    )
    artifact = {
        "schema": M43_ATTEMPT03_FOLD_ESTIMATOR_SCHEMA,
        "model_schema": HU_M43_V5_MODEL_SCHEMA,
        "proposal_schema": HU_M43_V5_PROPOSAL_SCHEMA,
        "training_config_sha256": contract["training_config_sha256"],
        "job_spec_sha256": definition.spec.sha256,
        "estimator": estimator,
    }
    _write_pickle_exclusive(artifact_path, artifact)
    manifest = _job_manifest(
        definition.spec,
        run_name=run_name,
        source_sha256=bound["source_sha256"],
        run_manifest_sha256=bound["run_manifest_sha256"],
        contract_file_sha256=contract_file_sha,
        input_bundle_sha256=bound["input_bundle_sha256"],
        training_config_sha256=contract["training_config_sha256"],
        artifact_sha256=_file_sha256(artifact_path),
    )
    _write_json_exclusive(manifest_path, manifest)
    done = _done_from_manifest(manifest, manifest_path)
    _write_json_exclusive(done_path, done)
    return done


def load_attempt03_fold_artifact_provider(
    *,
    artifacts_dir: str | Path,
    fold_cloud_contract_path: str | Path,
    inherited_train_path: str | Path,
    fresh_train_fit_path: str | Path,
    expected_run_name: str,
    expected_source_sha256: str,
    expected_run_manifest_sha256: str,
) -> Attempt03FoldArtifactProvider:
    _validate_run_name(expected_run_name)
    source_sha = _require_sha256(expected_source_sha256, "source_sha256")
    run_manifest_sha = _require_sha256(
        expected_run_manifest_sha256, "run_manifest_sha256"
    )
    contract_path = Path(fold_cloud_contract_path).resolve()
    contract = load_attempt03_fold_cloud_contract(contract_path)
    config = Attempt03TrainingConfig.from_manifest(contract["training_config"])
    samples, plan = _load_fit_plan(
        contract,
        inherited_train_path=inherited_train_path,
        fresh_train_fit_path=fresh_train_fit_path,
        config=config,
    )
    root = Path(artifacts_dir).resolve()
    expected_names = {f"job-{index:02d}" for index in range(30)}
    actual_names = {path.name for path in root.glob("job-*") if path.is_dir()}
    if actual_names != expected_names:
        raise ValueError("Attempt03 fold artifact directory coverage mismatch")
    contract_file_sha = _file_sha256(contract_path)
    estimators: dict[int, PairedDeltaRiskFoldEstimator] = {}
    receipts: list[dict[str, Any]] = []
    for definition in plan.jobs:
        estimator, receipt = _load_job_artifact(
            root / f"job-{definition.spec.job_index:02d}",
            definition=definition,
            contract_file_sha256=contract_file_sha,
            input_bundle_sha256=contract["input_bundle_sha256"],
            training_config_sha256=contract["training_config_sha256"],
        )
        if (
            receipt["run_name"] != expected_run_name
            or receipt["source_sha256"] != source_sha
            or receipt["run_manifest_sha256"] != run_manifest_sha
        ):
            raise ValueError("Attempt03 immutable run lineage mismatch")
        estimators[definition.spec.job_index] = estimator
        receipts.append(receipt)
    assembly = {
        "schema": M43_ATTEMPT03_FOLD_ASSEMBLY_SCHEMA,
        "status": "verified_unconsumed",
        "job_count": 30,
        "run_name": expected_run_name,
        "source_sha256": source_sha,
        "run_manifest_sha256": run_manifest_sha,
        "cloud_contract_file_sha256": contract_file_sha,
        "cloud_contract_sha256": contract["contract_sha256"],
        "training_config_sha256": contract["training_config_sha256"],
        "input_bundle_sha256": contract["input_bundle_sha256"],
        "jobs": receipts,
        "exact_outer_inner_coverage": True,
        "fit700_only": True,
        "holdout_input_count": 0,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    return Attempt03FoldArtifactProvider(
        estimators=estimators,
        expected_specs={job.spec.job_index: job.spec for job in plan.jobs},
        training_samples=tuple(samples),
        assembly_receipt=assembly,
        consumed=set(),
    )


def audit_attempt03_base_oof_dependencies(
    plan: M43FoldTrainingPlan,
) -> dict[str, Any]:
    """Prove every five-head base OOF row excludes its own information set.

    This inspects the actual job fit samples rather than trusting job labels or
    declared hashes.  It also makes the 30-job limit explicit: the audit proves
    base-feature OOF lineage only and does not claim a second nested meta OOF.
    """

    if len(plan.ordered_samples) != M43_ATTEMPT03_FIT_STATES:
        raise ValueError("Attempt03 dependency audit requires exactly 700 states")
    if len(plan.jobs) != M43_ATTEMPT03_FOLD_JOBS:
        raise ValueError("Attempt03 dependency audit requires exactly 30 jobs")
    by_job = {job.spec.job_index: job for job in plan.jobs}
    if set(by_job) != set(range(M43_ATTEMPT03_FOLD_JOBS)):
        raise ValueError("Attempt03 dependency audit job coverage changed")
    coverage = np.zeros(len(plan.ordered_samples), dtype=np.int8)
    folds: list[dict[str, Any]] = []
    for outer in range(M43_ATTEMPT03_FOLDS):
        validation_indices = [
            index
            for index, assigned in enumerate(plan.outer_fold_ids)
            if assigned == outer
        ]
        validation = [plan.ordered_samples[index] for index in validation_indices]
        validation_identity = _m43_sample_identity_sha256(validation)
        outer_job = by_job[outer * (M43_ATTEMPT03_FOLDS + 1)]
        if (
            outer_job.spec.kind != "outer_runtime"
            or outer_job.spec.outer_fold != outer
            or outer_job.spec.inner_fold is not None
            or outer_job.spec.outer_validation_identity_sha256
            != validation_identity
            or not _identity_disjoint(outer_job.fit_samples, validation)
        ):
            raise ValueError(
                f"Attempt03 outer runtime dependency leak/tamper in fold {outer}"
            )
        source_jobs: list[int] = []
        for inner in range(M43_ATTEMPT03_FOLDS):
            job_index = outer * (M43_ATTEMPT03_FOLDS + 1) + 1 + inner
            job = by_job[job_index]
            if (
                job.spec.kind != "inner_oof_safety"
                or job.spec.outer_fold != outer
                or job.spec.inner_fold != inner
                or job.spec.outer_validation_identity_sha256
                != validation_identity
                or not _identity_disjoint(job.fit_samples, validation)
            ):
                raise ValueError(
                    f"Attempt03 base OOF dependency leak/tamper in job {job_index}"
                )
            source_jobs.append(job_index)
        coverage[validation_indices] += 1
        folds.append(
            {
                "outer_fold": outer,
                "validation_states": len(validation_indices),
                "validation_identity_sha256": validation_identity,
                "base_feature_source_jobs": source_jobs,
                "all_five_source_jobs_exclude_validation_identities": True,
            }
        )
    if np.any(coverage != 1):
        raise ValueError("Attempt03 base OOF dependency coverage is not exactly once")
    return {
        "schema": "hu_m43_attempt03_base_oof_dependency_audit_v1",
        "status": "pass",
        "states": len(plan.ordered_samples),
        "jobs": len(plan.jobs),
        "base_feature_heads_per_state": M43_ATTEMPT03_FOLDS,
        "coverage_exactly_once": True,
        "own_identity_excluded_from_all_five_base_heads": True,
        "meta_oof_claimed": False,
        "folds": folds,
    }


def fit_attempt03_base_oof_runtime_meta(
    samples: Sequence[PreparedTeacherSample],
    *,
    stage18_scorer: Any,
    config: Attempt03TrainingConfig = Attempt03TrainingConfig(),
    fold_estimator_provider: Callable[
        [M43FoldJobSpec, Sequence[PreparedTeacherSample]],
        PairedDeltaRiskFoldEstimator,
    ]
    | None = None,
    meta_fit: Callable[..., Any] = fit_v5_meta_ranker,
) -> Attempt03FitResult:
    """Build identity-clean base OOF features, then fit the runtime meta model.

    The 30-job grid can produce five identity-clean base heads for every fit
    state, but it cannot also produce a strictly nested five-head meta OOF
    estimate.  In particular, reusing another state's globally cached base OOF
    row in a meta outer fold lets that row's upstream base estimators depend on
    the meta-validation identities.  We therefore make no internal meta-OOF
    claim.  The independent one-shot pre-calibration split is the model-level
    generalization gate; the fit safety rows below consume only the clean
    22-column pre-meta feature matrix and hard vote mask.
    """

    if len(samples) != M43_ATTEMPT03_FIT_STATES:
        raise ValueError("Attempt03 fit requires exactly 700 states")
    plan = build_m43_fold_training_plan(
        samples, cross_fit_folds=5, seed=config.fold_seed
    )
    jobs = {job.spec.job_index: job for job in plan.jobs}
    dependency_audit = audit_attempt03_base_oof_dependencies(plan)

    def obtain(job_index: int) -> PairedDeltaRiskFoldEstimator:
        job = jobs[job_index]
        estimator = (
            fit_v4_fold_worker_compatible(job.spec, job.fit_samples, config=config.worker)
            if fold_estimator_provider is None
            else fold_estimator_provider(job.spec, job.fit_samples)
        )
        if not isinstance(estimator, PairedDeltaRiskFoldEstimator):
            raise TypeError("Attempt03 fold provider returned wrong estimator")
        if estimator.fold_index != job.spec.estimator_fold_index:
            raise ValueError("Attempt03 fold estimator index changed")
        return estimator

    runtime_folds: list[PairedDeltaRiskFoldEstimator] = []
    stacked_by_state: list[Any | None] = [None] * len(plan.ordered_samples)
    fold_audits: list[dict[str, Any]] = []
    for outer in range(5):
        base_job = outer * 6
        runtime_folds.append(obtain(base_job))
        inner = tuple(obtain(base_job + 1 + index) for index in range(5))
        validation = [
            index for index, assigned in enumerate(plan.outer_fold_ids)
            if assigned == outer
        ]
        training = [
            index for index, assigned in enumerate(plan.outer_fold_ids)
            if assigned != outer
        ]
        excluded = _identity_disjoint(
            [plan.ordered_samples[index] for index in training],
            [plan.ordered_samples[index] for index in validation],
        )
        if not excluded:
            raise AssertionError("Attempt03 base outer fold identity leakage")
        for index in validation:
            sample = plan.ordered_samples[index]
            stacked_by_state[index] = build_v5_stacked_features(
                sample.policy_sample,
                paired_fold_estimators=inner,
                stage18_scorer=stage18_scorer,
                baseline_index=sample.baseline_index,
            )
        fold_audits.append(
            {
                "fold": outer,
                "training_states": len(training),
                "validation_states": len(validation),
                "inner_fold_estimators": len(inner),
                "base_outer_validation_identity_excluded": True,
            }
        )
    if any(value is None for value in stacked_by_state):
        raise AssertionError("Attempt03 nested base OOF coverage is incomplete")
    stacked = tuple(value for value in stacked_by_state if value is not None)

    all_states = list(range(len(stacked)))
    all_features, all_targets, all_state_ids = _meta_training_arrays(
        plan.ordered_samples, stacked, all_states
    )
    runtime_fit = meta_fit(
        all_features, all_targets, all_state_ids, config=config.meta
    )
    runtime_meta = getattr(runtime_fit, "estimator", runtime_fit)
    # These predictions are retained only as a feature-carrier compatibility
    # envelope.  Their meta scores are an in-fit diagnostic and are never used
    # to construct safety features or make a fit-time performance claim.
    feature_carriers = tuple(
        Attempt03OofStatePrediction(
            sample_index=state_index,
            fold_index=int(plan.outer_fold_ids[state_index]),
            predictions=_predictions_from_stacked(stacked[state_index], runtime_meta),
            base_identity_excluded_from_fit=True,
            meta_identity_excluded_from_fit=False,
        )
        for state_index in all_states
    )
    model = HuM43JointModelV5(
        paired_fold_estimators=tuple(runtime_folds),
        stage18_scorer=stage18_scorer,
        meta_ranker=runtime_meta,
        safety_estimator=None,
        safety_threshold=1.0,
        safety_enabled=False,
        model_id=config.model_id,
        manifest={
            "status": "fit_only_precalibration_unopened",
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
            "full_replacement": False,
        },
    )
    safety = build_attempt03_oof_safety_dataset(
        plan.ordered_samples, feature_carriers
    )
    report = {
        "schema": "hu_m43_attempt03_v5_base_oof_runtime_meta_fit_v1",
        "status": "fit_complete_precalibration_unopened",
        "states": len(stacked),
        "folds": 5,
        "base_fold_jobs": len(plan.jobs),
        "exact_outer_inner_job_grid": len(plan.jobs) == 30,
        "nested_base_oof_coverage_exactly_once": len(stacked) == 700,
        "base_oof_dependency_audit": dependency_audit,
        "meta_crossfit_performed": False,
        "meta_oof_coverage_exactly_once": False,
        "meta_oof_performance_claimed": False,
        "model_level_generalization_gate": "fresh_one_shot_precalibration_200",
        "meta_fit_rows": int(all_features.shape[0]),
        "meta_fit_row_scope": "all_nonbaseline_actions",
        "meta_state_balanced": True,
        "baseline_training_rows_included": False,
        "base_fold_audits": fold_audits,
        "runtime_meta_fit": dict(getattr(runtime_fit, "manifest", {})),
        "safety_fit_feature_scope": (
            "identity_clean_base_oof_22_columns_no_meta_score_or_proposal"
        ),
        "precalibration_opened": False,
        "sealed_calibration_opened": False,
        "inherited_locked_opened": False,
        "runtime_teacher_inputs": False,
        "teacher_value_status": M43_ATTEMPT03_TEACHER_STATUS,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    return Attempt03FitResult(
        model=model,
        oof_predictions=feature_carriers,
        safety_dataset=safety,
        report=report,
    )


def build_attempt03_oof_safety_dataset(
    samples: Sequence[PreparedTeacherSample],
    oof_predictions: Sequence[Attempt03OofStatePrediction],
    *,
    maximum_p95_loss: float = 25.0,
    maximum_p99_loss: float = 40.0,
    maximum_max_loss: float = 50.0,
) -> Attempt03SafetyDataset:
    ordered = _ordered_oof(samples, oof_predictions)
    features: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    rows: list[dict[str, Any]] = []
    contributing_states: list[int] = []
    zero_eligible_states: list[int] = []
    for state_index, (sample, oof) in enumerate(zip(samples, ordered, strict=True)):
        _require_targets(sample)
        if not oof.base_identity_excluded_from_fit:
            raise ValueError("Attempt03 safety feature row is not base-OOF clean")
        heads = oof.predictions
        action_count = len(sample.policy_sample["actions"])
        baseline = sample.baseline_index
        eligible = [
            index
            for index in range(action_count)
            if index != baseline and bool(heads.eligible_mask[index])
        ]
        if not eligible:
            zero_eligible_states.append(state_index)
            continue
        contributing_states.append(state_index)
        weight = 1.0 / len(eligible)
        for action in eligible:
            delta = float(sample.teacher_paired_delta_mean[action])
            p95 = float(sample.downside_loss_p95[action])
            p99 = float(sample.downside_loss_p99[action])
            maximum = float(sample.downside_loss_max[action])
            safe = bool(
                delta > 0.0
                and p95 <= maximum_p95_loss
                and p99 <= maximum_p99_loss
                and maximum <= maximum_max_loss
            )
            features.append(np.asarray(heads.meta_features[action], dtype=np.float32))
            labels.append(int(safe))
            weights.append(weight)
            rows.append(
                {
                    "state_index": state_index,
                    "fold_index": oof.fold_index,
                    "action_index": action,
                    "baseline_index": baseline,
                    "teacher_delta": delta,
                    "downside_loss_p95": p95,
                    "downside_loss_p99": p99,
                    "downside_loss_max": maximum,
                    "hard_eligible": bool(heads.eligible_mask[action]),
                    "safe_label": int(safe),
                    "sample_weight": weight,
                    "base_identity_excluded_from_fit": bool(
                        oof.base_identity_excluded_from_fit
                    ),
                    "meta_identity_excluded_from_fit": bool(
                        oof.meta_identity_excluded_from_fit
                    ),
                    "meta_score_or_proposal_used_as_safety_feature": False,
                    "teacher_value_status": M43_ATTEMPT03_TEACHER_STATUS,
                }
            )
    matrix = (
        np.vstack(features).astype(np.float32, copy=False)
        if features
        else np.empty((0, HU_M43_V5_META_FEATURE_DIM), dtype=np.float32)
    )
    label_array = np.asarray(labels, dtype=np.int8)
    weight_array = np.asarray(weights, dtype=np.float64)
    for state in contributing_states:
        mask = np.asarray([row["state_index"] == state for row in rows])
        if not math.isclose(
            float(np.sum(weight_array[mask])), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise AssertionError("Attempt03 safety state weights changed")
    return Attempt03SafetyDataset(
        features=matrix,
        labels=label_array,
        weights=weight_array,
        rows=tuple(rows),
        manifest={
            "schema": M43_ATTEMPT03_SAFETY_DATASET_SCHEMA,
            "states": len(samples),
            "contributing_states": len(contributing_states),
            "zero_eligible_states": len(zero_eligible_states),
            "zero_eligible_state_indices": zero_eligible_states,
            "rows": len(rows),
            "feature_schema": HU_M43_V5_FEATURE_SCHEMA,
            "feature_dim": HU_M43_V5_META_FEATURE_DIM,
            "row_source": "fit_base_oof_hard_eligible_actions_only",
            "row_scope": "hard_eligible_nonbaseline_actions_only",
            "state_weighting": (
                "sum_one_per_contributing_information_set_zero_if_no_eligible_action"
            ),
            "baseline_rows": 0,
            "safe_rows": int(np.sum(label_array == 1)),
            "unsafe_rows": int(np.sum(label_array == 0)),
            "hard_eligibility_used_as_fit_filter": True,
            "meta_score_or_proposal_used_as_fit_feature": False,
            "teacher_value_status": M43_ATTEMPT03_TEACHER_STATUS,
            "runtime_teacher_inputs": False,
        },
        seed_values=frozenset().union(*(sample.root_seed_values for sample in samples)),
        observation_fingerprints=frozenset(
            sample.observation_fingerprint for sample in samples
        ),
    )


def evaluate_attempt03_precalibration(
    model: HuM43JointModelV5,
    samples: Sequence[PreparedTeacherSample],
    *,
    fit_seed_values: Iterable[str],
    fit_observation_fingerprints: Iterable[str],
    profile_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen one-shot gates over independent state clusters."""

    fit_seeds = frozenset(str(value) for value in fit_seed_values)
    fit_fingerprints = frozenset(str(value) for value in fit_observation_fingerprints)
    precal_seeds = frozenset().union(*(sample.root_seed_values for sample in samples))
    precal_fingerprints = frozenset(
        sample.observation_fingerprint for sample in samples
    )
    identity_disjoint = not (
        fit_seeds & precal_seeds or fit_fingerprints & precal_fingerprints
    )
    raw_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    mapping_exact = True
    baseline_exact = True
    tie_exact = True
    for state_index, sample in enumerate(samples):
        _require_targets(sample)
        heads = model.predict_heads_sample(
            sample.policy_sample, baseline_index=sample.baseline_index
        )
        proposal = int(heads.proposal_index)
        baseline = sample.baseline_index
        mapping_exact = mapping_exact and proposal != baseline and (
            0 <= proposal < len(sample.policy_sample["actions"])
        )
        baseline_exact = baseline_exact and (
            heads.meta_score[baseline] == 0.0
            and heads.action_score[baseline] == 0.0
            and heads.base_delta[baseline] == 0.0
            and heads.delta_positive_votes[baseline] == 0
            and not bool(heads.eligible_mask[baseline])
        )
        values = np.asarray(heads.meta_score, dtype=np.float64)
        candidates = [index for index in range(values.size) if index != baseline]
        best = max(float(values[index]) for index in candidates)
        tied = [index for index in candidates if float(values[index]) == best]
        canonical = min(
            tied,
            key=lambda index: action_key_from_payload(
                sample.policy_sample["actions"][index]
            ).sort_key(),
        )
        tie_exact = tie_exact and proposal == canonical
        row = {
            "state_index": state_index,
            "proposal_index": proposal,
            "baseline_index": baseline,
            "teacher_delta": float(sample.teacher_paired_delta_mean[proposal]),
            "downside_loss_p95": float(sample.downside_loss_p95[proposal]),
            "downside_loss_p99": float(sample.downside_loss_p99[proposal]),
            "downside_loss_max": float(sample.downside_loss_max[proposal]),
            "eligible": bool(heads.proposal_eligible),
            "positive_votes": int(heads.delta_positive_votes[proposal]),
            "teacher_value_status": M43_ATTEMPT03_TEACHER_STATUS,
        }
        raw_rows.append(row)
        if row["eligible"]:
            eligible_rows.append(row)
    raw_deltas = np.asarray(
        [row["teacher_delta"] for row in raw_rows], dtype=np.float64
    )
    eligible_deltas = np.asarray(
        [row["teacher_delta"] for row in eligible_rows], dtype=np.float64
    )
    fires = len(eligible_rows)
    if fires >= 2:
        mean = float(np.mean(eligible_deltas))
        standard_error = float(np.std(eligible_deltas, ddof=1) / math.sqrt(fires))
        critical = float(student_t.ppf(0.90, fires - 1))
        lcb = mean - critical * standard_error
    else:
        mean = float(np.mean(eligible_deltas)) if fires else 0.0
        standard_error = None
        critical = None
        lcb = None
    false_positive = (
        float(np.mean(eligible_deltas <= 0.0)) if fires else 0.0
    )
    p95 = max((float(row["downside_loss_p95"]) for row in eligible_rows), default=0.0)
    p99 = max((float(row["downside_loss_p99"]) for row in eligible_rows), default=0.0)
    maximum = max((float(row["downside_loss_max"]) for row in eligible_rows), default=0.0)
    expected_profiles = {profile: 40 for profile in _PROFILES}
    profile_exact = (
        profile_counts is not None and dict(profile_counts) == expected_profiles
    )
    gates = {
        "exact_state_count": len(samples) == 200,
        "exact_profile_balance": profile_exact,
        "fit_precal_identity_disjoint": identity_disjoint,
        "semantic_action_mapping_exact": mapping_exact,
        "baseline_quantities_exact_zero": baseline_exact,
        "canonical_tie_break_exact": tie_exact,
        "nonbaseline_raw_proposal_every_state": len(raw_rows) == len(samples),
        "minimum_raw_proposal_positive_rate": (
            float(np.mean(raw_deltas > 0.0)) >= 0.40
            if raw_deltas.size else False
        ),
        "minimum_eligible_fires": fires >= 30,
        "eligible_mean_delta_per_fire_strictly_positive": mean > 0.0,
        "eligible_delta_per_state_strictly_positive": (
            float(np.sum(eligible_deltas) / len(samples)) > 0.0
            if samples else False
        ),
        "eligible_false_positive_rate_at_most_0_35": false_positive <= 0.35,
        "one_sided_student_t_90_lcb_strictly_positive": (
            lcb is not None and lcb > 0.0
        ),
        "selected_p95_loss_at_most_25": p95 <= 25.0,
        "selected_p99_loss_at_most_40": p99 <= 40.0,
        "selected_max_loss_at_most_50": maximum <= 50.0,
    }
    status = "go" if all(gates.values()) else "no_go"
    report = {
        "schema": M43_ATTEMPT03_PRECAL_REPORT_SCHEMA,
        "status": status,
        "source": "fresh_one_shot_train.precal_holdout_200",
        "states": len(samples),
        "raw_proposals": len(raw_rows),
        "raw_proposal_positive_count": int(np.sum(raw_deltas > 0.0)),
        "raw_proposal_positive_rate": (
            float(np.mean(raw_deltas > 0.0)) if raw_deltas.size else 0.0
        ),
        "eligible_fires": fires,
        "eligible_mean_delta_per_fire": mean,
        "eligible_delta_per_state": (
            float(np.sum(eligible_deltas) / len(samples)) if samples else 0.0
        ),
        "eligible_false_positive_rate_delta_le_zero": false_positive,
        "eligible_selected_action_downside_maximum": {
            "p95": p95,
            "p99": p99,
            "max": maximum,
        },
        "mean_delta_lcb": {
            "method": "one_sided_student_t_over_independent_state_clusters",
            "confidence": 0.90,
            "clusters": fires,
            "degrees_of_freedom": fires - 1 if fires >= 2 else None,
            "sample_standard_error": standard_error,
            "critical_value": critical,
            "lower_bound": lcb,
        },
        "gates": gates,
        "raw_proposal_rows": raw_rows,
        "eligible_rows": eligible_rows,
        "algorithm_or_gate_change_after_result_allowed": False,
        "sealed_calibration_opened": False,
        "inherited_locked_opened": False,
        "runtime_teacher_inputs": False,
        "teacher_value_status": M43_ATTEMPT03_TEACHER_STATUS,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    return report


def fit_attempt03_safety_calibrator(
    model: HuM43JointModelV5,
    train_oof_dataset: Attempt03SafetyDataset,
    safety_fit_samples: Sequence[PreparedTeacherSample],
    *,
    config: Attempt03TrainingConfig = Attempt03TrainingConfig(),
) -> Attempt03SafetyFitResult:
    """Fit the exact standardized L2 logistic only after pre-cal Go."""

    if len(safety_fit_samples) != 50:
        raise ValueError("Attempt03 safety-fit role must contain exactly 50 states")
    independent = _build_independent_safety_dataset(
        model, safety_fit_samples, source="sealed_calibration.safety_fit"
    )
    if train_oof_dataset.features.shape[1] != HU_M43_V5_META_FEATURE_DIM:
        raise ValueError("Attempt03 OOF safety feature shape changed")
    if train_oof_dataset.seed_values & independent.seed_values or (
        train_oof_dataset.observation_fingerprints
        & independent.observation_fingerprints
    ):
        raise ValueError("Attempt03 fit/safety calibration identity overlap")
    features = np.vstack((train_oof_dataset.features, independent.features)).astype(
        np.float32, copy=False
    )
    labels = np.concatenate((train_oof_dataset.labels, independent.labels)).astype(
        np.int8, copy=False
    )
    weights = np.concatenate((train_oof_dataset.weights, independent.weights)).astype(
        np.float64, copy=False
    )
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Attempt03 fixed logistic requires both safety classes")
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.safety_calibrator_c,
            solver="lbfgs",
            max_iter=500,
            random_state=config.safety_seed,
        ),
    ).fit(
        features,
        labels,
        standardscaler__sample_weight=weights,
        logisticregression__sample_weight=weights,
    )
    provisional = model.with_frozen_safety(
        estimator, threshold=1.0, enabled=False, manifest=model.manifest
    )
    report = {
        "schema": M43_ATTEMPT03_SAFETY_FIT_SCHEMA,
        "status": "fit_threshold_unselected",
        "estimator_family": "standardized_l2_logistic",
        "c": config.safety_calibrator_c,
        "seed": config.safety_seed,
        "feature_schema": HU_M43_V5_FEATURE_SCHEMA,
        "feature_dim": HU_M43_V5_META_FEATURE_DIM,
        "sources": {
            "fit_base_oof_hard_eligible_only": {
                "states": train_oof_dataset.manifest["states"],
                "contributing_states": train_oof_dataset.manifest[
                    "contributing_states"
                ],
                "zero_eligible_states": train_oof_dataset.manifest[
                    "zero_eligible_states"
                ],
                "rows": int(train_oof_dataset.features.shape[0]),
            },
            "sealed_calibration.safety_fit_50": {
                "states": 50,
                "rows": int(independent.features.shape[0]),
            },
            "sealed_calibration.threshold_lock_50": {
                "opened": False,
                "used": False,
            },
        },
        "combined_rows": int(features.shape[0]),
        "combined_safe_rows": int(np.sum(labels == 1)),
        "combined_unsafe_rows": int(np.sum(labels == 0)),
        "state_balanced_weights": True,
        "standardizer_uses_state_balanced_weights": True,
        "logistic_uses_state_balanced_weights": True,
        "safety_enabled": False,
        "threshold_selected": False,
        "inherited_locked_opened": False,
        "runtime_teacher_inputs": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    return Attempt03SafetyFitResult(
        model=provisional,
        report=report,
        fit_seed_values=(
            train_oof_dataset.seed_values | independent.seed_values
        ),
        fit_observation_fingerprints=(
            train_oof_dataset.observation_fingerprints
            | independent.observation_fingerprints
        ),
    )


def select_attempt03_threshold(
    safety_fit: Attempt03SafetyFitResult,
    threshold_lock_samples: Sequence[PreparedTeacherSample],
    *,
    config: Attempt03TrainingConfig = Attempt03TrainingConfig(),
) -> Attempt03ThresholdResult:
    """Run the fixed grid once; ineligible top proposals never fall through."""

    if len(threshold_lock_samples) != 50:
        raise ValueError("Attempt03 threshold-lock role must contain 50 states")
    lock_seeds = frozenset().union(
        *(sample.root_seed_values for sample in threshold_lock_samples)
    )
    lock_fingerprints = frozenset(
        sample.observation_fingerprint for sample in threshold_lock_samples
    )
    if safety_fit.fit_seed_values & lock_seeds or (
        safety_fit.fit_observation_fingerprints & lock_fingerprints
    ):
        raise ValueError("Attempt03 safety-fit/threshold-lock identity overlap")
    rows: list[dict[str, Any]] = []
    for state_index, sample in enumerate(threshold_lock_samples):
        _require_targets(sample)
        heads = safety_fit.model.predict_heads_sample(
            sample.policy_sample, baseline_index=sample.baseline_index
        )
        proposal = heads.proposal_index
        probability = None
        if heads.proposal_eligible:
            probability = safety_fit.model.predict_safety_probability(
                sample.policy_sample,
                candidate_index=proposal,
                baseline_index=sample.baseline_index,
            )
        rows.append(
            {
                "state_index": state_index,
                "proposal_index": proposal,
                "baseline_index": sample.baseline_index,
                "eligible": bool(heads.proposal_eligible),
                "probability": probability,
                "teacher_delta": float(sample.teacher_paired_delta_mean[proposal]),
                "downside_loss_p95": float(sample.downside_loss_p95[proposal]),
                "downside_loss_p99": float(sample.downside_loss_p99[proposal]),
                "downside_loss_max": float(sample.downside_loss_max[proposal]),
            }
        )
    sweep: list[dict[str, Any]] = []
    for threshold in config.thresholds:
        selected = [
            row for row in rows
            if row["eligible"]
            and row["probability"] is not None
            and float(row["probability"]) >= threshold
        ]
        sweep.append(
            _threshold_metrics(selected, total_states=50, threshold=threshold)
        )
    eligible = [
        row for row in sweep
        if row["fires"] >= config.minimum_threshold_fires
        and row["teacher_delta_per_state"] > 0.0
        and row["false_positive_rate"] <= config.maximum_false_positive_rate
        and row["p95_loss"] <= config.maximum_p95_loss
        and row["p99_loss"] <= config.maximum_p99_loss
        and row["max_loss"] <= config.maximum_max_loss
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (row["teacher_delta_per_state"], row["threshold"]),
        )
        status = "go"
        enabled = True
    else:
        selected = next(row for row in sweep if row["threshold"] == 1.0)
        status = "no_go"
        enabled = False
    final_model = safety_fit.model.with_frozen_safety(
        safety_fit.model.safety_estimator,
        threshold=float(selected["threshold"]),
        enabled=enabled,
        manifest=safety_fit.model.manifest,
    )
    report = {
        "schema": M43_ATTEMPT03_THRESHOLD_REPORT_SCHEMA,
        "status": status,
        "source": "sealed_calibration.threshold_lock_50_only",
        "states": 50,
        "eligible_proposals": sum(bool(row["eligible"]) for row in rows),
        "thresholds": list(config.thresholds),
        "selected_threshold": selected["threshold"],
        "selected_metrics": selected,
        "threshold_sweep": sweep,
        "constraints": {
            "minimum_fires": config.minimum_threshold_fires,
            "maximum_false_positive_rate": config.maximum_false_positive_rate,
            "minimum_delta_per_state": "strictly_greater_than_zero",
            "maximum_p95_loss": config.maximum_p95_loss,
            "maximum_p99_loss": config.maximum_p99_loss,
            "maximum_max_loss": config.maximum_max_loss,
        },
        "ineligible_top_proposal_rerank_allowed": False,
        "threshold_adaptation_after_selection": False,
        "safety_enabled": enabled,
        "inherited_locked_opened": False,
        "runtime_teacher_inputs": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    return Attempt03ThresholdResult(model=final_model, report=report)


def save_attempt03_fit_bundle(
    path: str | Path,
    result: Attempt03FitResult,
    *,
    training_config_sha256: str,
    model_freeze_file_sha256: str,
    training_freeze_file_sha256: str,
) -> str:
    payload = {
        "schema": M43_ATTEMPT03_FIT_BUNDLE_SCHEMA,
        "training_config_sha256": _require_sha256(
            training_config_sha256, "training_config_sha256"
        ),
        "model_freeze_file_sha256": _require_sha256(
            model_freeze_file_sha256, "model_freeze_file_sha256"
        ),
        "training_freeze_file_sha256": _require_sha256(
            training_freeze_file_sha256, "training_freeze_file_sha256"
        ),
        "fit_result": result,
        "precalibration_opened": False,
        "sealed_calibration_opened": False,
        "inherited_locked_opened": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write_pickle_exclusive(Path(path), payload)
    return _file_sha256(Path(path))


def load_attempt03_fit_bundle(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> Attempt03FitResult:
    source = Path(path)
    if expected_sha256 is not None and _file_sha256(source) != _require_sha256(
        expected_sha256, "fit_bundle_sha256"
    ):
        raise ValueError("Attempt03 fit bundle SHA mismatch")
    payload = pickle.loads(source.read_bytes())
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "training_config_sha256",
        "model_freeze_file_sha256",
        "training_freeze_file_sha256",
        "fit_result",
        "precalibration_opened",
        "sealed_calibration_opened",
        "inherited_locked_opened",
        "current_profile_mutated",
        "runtime_policy_activated",
    }:
        raise ValueError("Attempt03 fit bundle header changed")
    if payload["schema"] != M43_ATTEMPT03_FIT_BUNDLE_SCHEMA or any(
        payload[key] is not False for key in (
            "precalibration_opened",
            "sealed_calibration_opened",
            "inherited_locked_opened",
            "current_profile_mutated",
            "runtime_policy_activated",
        )
    ):
        raise ValueError("Attempt03 fit bundle lifecycle changed")
    result = payload.get("fit_result")
    if not isinstance(result, Attempt03FitResult):
        raise TypeError("Attempt03 fit bundle payload type changed")
    result.model.__post_init__()
    if result.model.safety_enabled or result.model.safety_estimator is not None:
        raise ValueError("Attempt03 fit bundle must remain fail-closed")
    return result


def claim_attempt03_precalibration(
    marker_path: str | Path,
    *,
    precal_identity_sha256: str,
    candidate_model_sha256: str,
    fit_manifest_sha256: str,
    data_contract_sha256: str,
    model_freeze_file_sha256: str,
    training_freeze_file_sha256: str,
) -> dict[str, Any]:
    """Irreversibly claim the one-shot split before parsing its JSONL."""

    payload = {
        "schema": M43_ATTEMPT03_PRECAL_MARKER_SCHEMA,
        "status": "consumed_before_model_evaluation",
        "precal_identity_sha256": _require_sha256(
            precal_identity_sha256, "precal_identity_sha256"
        ),
        "candidate_model_sha256": _require_sha256(
            candidate_model_sha256, "candidate_model_sha256"
        ),
        "fit_manifest_sha256": _require_sha256(
            fit_manifest_sha256, "fit_manifest_sha256"
        ),
        "data_contract_sha256": _require_sha256(
            data_contract_sha256, "data_contract_sha256"
        ),
        "model_freeze_file_sha256": _require_sha256(
            model_freeze_file_sha256, "model_freeze_file_sha256"
        ),
        "training_freeze_file_sha256": _require_sha256(
            training_freeze_file_sha256, "training_freeze_file_sha256"
        ),
        "algorithm_or_gate_change_after_result_allowed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    payload["marker_sha256"] = canonical_manifest_sha256(payload)
    _write_json_exclusive(Path(marker_path), payload)
    return payload


def build_attempt03_fit_manifest(
    *,
    result: Attempt03FitResult,
    model_sha256: str,
    fit_bundle_sha256: str,
    fold_assembly: Mapping[str, Any],
    fold_cloud_contract_sha256: str,
    training_config_sha256: str,
    model_freeze_file_sha256: str,
    training_freeze_file_sha256: str,
) -> dict[str, Any]:
    manifest = {
        "schema": M43_ATTEMPT03_FIT_MANIFEST_SCHEMA,
        "status": "fit_candidate_precalibration_unopened",
        "promotion_status": "not_eligible_precalibration_unopened",
        "model_schema": HU_M43_V5_MODEL_SCHEMA,
        "model_artifact_schema": HU_M43_V5_ARTIFACT_SCHEMA,
        "proposal_schema": HU_M43_V5_PROPOSAL_SCHEMA,
        "model_sha256": _require_sha256(model_sha256, "model_sha256"),
        "fit_bundle_sha256": _require_sha256(
            fit_bundle_sha256, "fit_bundle_sha256"
        ),
        "fold_cloud_contract_sha256": _require_sha256(
            fold_cloud_contract_sha256, "fold_cloud_contract_sha256"
        ),
        "training_config_sha256": _require_sha256(
            training_config_sha256, "training_config_sha256"
        ),
        "model_freeze_file_sha256": _require_sha256(
            model_freeze_file_sha256, "model_freeze_file_sha256"
        ),
        "training_freeze_file_sha256": _require_sha256(
            training_freeze_file_sha256, "training_freeze_file_sha256"
        ),
        "fold_assembly": dict(fold_assembly),
        "crossfit": dict(result.report),
        "safety_dataset": dict(result.safety_dataset.manifest),
        "precalibration": {"opened": False, "evaluated": False},
        "sealed_calibration": {"opened": False, "evaluated": False},
        "inherited_locked": {"opened": False, "evaluated": False},
        "runtime_teacher_inputs": False,
        "teacher_value_status": M43_ATTEMPT03_TEACHER_STATUS,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    manifest["manifest_sha256"] = canonical_manifest_sha256(manifest)
    return manifest


def build_attempt03_precalibration_receipt(
    *,
    report: Mapping[str, Any],
    marker: Mapping[str, Any],
    candidate_model_sha256: str,
    fit_manifest_sha256: str,
) -> dict[str, Any]:
    status = str(report.get("status"))
    if status not in {"go", "no_go"}:
        raise ValueError("Attempt03 pre-calibration report status changed")
    payload = {
        "schema": M43_ATTEMPT03_PRECAL_RECEIPT_SCHEMA,
        "status": "go_precalibration" if status == "go" else "no_go_precalibration",
        "promotion_status": (
            "eligible_to_open_sealed_calibration"
            if status == "go"
            else "no_go_precalibration"
        ),
        "candidate_model_sha256": _require_sha256(
            candidate_model_sha256, "candidate_model_sha256"
        ),
        "fit_manifest_sha256": _require_sha256(
            fit_manifest_sha256, "fit_manifest_sha256"
        ),
        "precalibration_consumption_marker_sha256": marker["marker_sha256"],
        "precalibration_report": dict(report),
        "sealed_calibration_open_allowed": status == "go",
        "sealed_calibration_opened": False,
        "inherited_locked_opened": False,
        "algorithm_or_gate_change_after_result_allowed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    payload["receipt_sha256"] = canonical_manifest_sha256(payload)
    return payload


def build_attempt03_final_manifest(
    *,
    threshold: Attempt03ThresholdResult,
    safety_fit: Attempt03SafetyFitResult,
    precalibration_receipt: Mapping[str, Any],
    model_sha256: str | None,
) -> dict[str, Any]:
    go = threshold.report.get("status") == "go"
    if go != (model_sha256 is not None):
        raise ValueError("Attempt03 final model publication status mismatch")
    manifest = {
        "schema": M43_ATTEMPT03_FINAL_MANIFEST_SCHEMA,
        "status": "candidate_ready_for_freeze" if go else "no_go_calibration",
        "promotion_status": (
            "candidate_ready_for_freeze" if go else "no_go_calibration"
        ),
        "model_schema": HU_M43_V5_MODEL_SCHEMA,
        "proposal_schema": HU_M43_V5_PROPOSAL_SCHEMA,
        "model_sha256": model_sha256,
        "precalibration_receipt_sha256": precalibration_receipt.get(
            "receipt_sha256"
        ),
        "safety_fit": dict(safety_fit.report),
        "threshold_selection": dict(threshold.report),
        "locked_holdout": {
            "status": "not_evaluated_pre_freeze",
            "opened": False,
        },
        "runtime_teacher_inputs": False,
        "teacher_value_status": M43_ATTEMPT03_TEACHER_STATUS,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }
    manifest["manifest_sha256"] = canonical_manifest_sha256(manifest)
    return manifest


def _read_fit_and_plan(
    inherited_train_path: str | Path,
    fresh_train_fit_path: str | Path,
    *,
    config: Attempt03TrainingConfig,
    repo_root: Path | None = None,
) -> tuple[
    tuple[Path, Path],
    tuple[list[dict[str, Any]], list[dict[str, Any]]],
    list[PreparedTeacherSample],
    M43FoldTrainingPlan,
]:
    inherited = Path(inherited_train_path).resolve()
    fresh = Path(fresh_train_fit_path).resolve()
    rows = (read_teacher_jsonl(inherited), read_teacher_jsonl(fresh))
    if len(rows[0]) != 200 or len(rows[1]) != 500:
        raise ValueError("Attempt03 fit inputs must contain inherited200 + fresh500")
    if any(row.get("split") not in (None, "train") for block in rows for row in block):
        raise ValueError("Attempt03 fit worker received a non-train row")
    if repo_root is not None:
        expected_inherited = (
            repo_root
            / "outputs/hu_joint_policy/m43_attempt02_teacher/"
            "regular-hu-m43-attempt02-c2e64-pilot300-20260713-2201/train.jsonl"
        ).resolve()
        if inherited == expected_inherited and _file_sha256(inherited) != (
            "0a55541b26305c5dfe50b9122263bb92ca1ee60a5f085d2020b38b31a4cdb996"
        ):
            raise ValueError("Attempt03 inherited train200 SHA changed")
    combined_rows = [*rows[0], *rows[1]]
    counts = Counter(
        str(_mapping(row.get("provenance"), "fit provenance").get("root_profile"))
        for row in combined_rows
    )
    if dict(counts) != {profile: 140 for profile in _PROFILES}:
        raise ValueError("Attempt03 fit700 profile balance changed")
    if any(
        _mapping(row.get("provenance"), "fit provenance").get(
            "current_profile_resolved"
        )
        is not False
        for row in combined_rows
    ):
        raise ValueError("Attempt03 fit data resolved current profile")
    samples = prepare_teacher_samples(combined_rows)
    memberships = [_sample_membership_hash(sample) for sample in samples]
    if len(set(memberships)) != 700:
        raise ValueError("Attempt03 fit membership is duplicated")
    fingerprints = {sample.observation_fingerprint for sample in samples}
    if len(fingerprints) != 700:
        raise ValueError("Attempt03 fit fingerprint is duplicated")
    seed_sets = [sample.root_seed_values for sample in samples]
    all_seeds = set().union(*seed_sets)
    if sum(len(values) for values in seed_sets) != len(all_seeds):
        raise ValueError("Attempt03 fit seed is duplicated")
    plan = build_m43_fold_training_plan(samples, cross_fit_folds=5, seed=config.fold_seed)
    return (inherited, fresh), rows, samples, plan


def _load_fit_plan(
    contract: Mapping[str, Any],
    *,
    inherited_train_path: str | Path,
    fresh_train_fit_path: str | Path,
    config: Attempt03TrainingConfig,
) -> tuple[list[PreparedTeacherSample], M43FoldTrainingPlan]:
    paths, rows, samples, plan = _read_fit_and_plan(
        inherited_train_path, fresh_train_fit_path, config=config
    )
    actual = {
        role: _input_entry(path, block, role=role)
        for role, path, block in zip(
            ("inherited_attempt02_train", "fresh_train_fit"),
            paths,
            rows,
            strict=True,
        )
    }
    if actual != contract["inputs"]:
        raise ValueError("Attempt03 fit input bytes/rows changed")
    expected_plan = _mapping(contract.get("fold_plan"), "fold_plan")
    jobs = [
        {**job.spec.to_manifest(), "job_spec_sha256": job.spec.sha256}
        for job in plan.jobs
    ]
    if (
        expected_plan.get("fit_identity_sha256")
        != _m43_sample_identity_sha256(plan.ordered_samples)
        or expected_plan.get("jobs") != jobs
    ):
        raise ValueError("Attempt03 recomputed fold plan changed")
    return samples, plan


def _meta_training_arrays(
    samples: Sequence[PreparedTeacherSample],
    stacked: Sequence[Any],
    state_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    states: list[int] = []
    for state in state_indices:
        sample = samples[state]
        _require_targets(sample)
        stack = stacked[state]
        for action in range(stack.features.shape[0]):
            if action == sample.baseline_index:
                continue
            features.append(stack.features[action])
            targets.append(float(sample.teacher_paired_delta_mean[action]))
            states.append(state)
    matrix = np.vstack(features).astype(np.float32, copy=False)
    target = np.asarray(targets, dtype=np.float64)
    state_array = np.asarray(states, dtype=np.int64)
    if matrix.shape[1] != 22 or target.shape != state_array.shape:
        raise AssertionError("Attempt03 meta array shape changed")
    return matrix, target, state_array


def _predictions_from_stacked(stacked: Any, estimator: Any) -> V5ActionPredictions:
    predictor = getattr(estimator, "booster_", estimator)
    score = np.asarray(predictor.predict(stacked.features), dtype=np.float64).reshape(-1)
    if score.shape != (stacked.features.shape[0],) or not np.isfinite(score).all():
        raise ValueError("Attempt03 meta OOF prediction is invalid")
    score[stacked.baseline_index] = 0.0
    candidates = [index for index in range(score.size) if index != stacked.baseline_index]
    best = max(float(score[index]) for index in candidates)
    tied = [index for index in candidates if float(score[index]) == best]
    proposal = min(tied, key=lambda index: stacked.action_keys[index].sort_key())
    eligible = bool(stacked.base.eligible_mask[proposal])
    action_score = np.full(score.size, -1.0, dtype=np.float64)
    action_score[stacked.baseline_index] = 0.0
    if eligible:
        action_score[proposal] = 1.0
    return V5ActionPredictions(
        meta_features=stacked.features,
        meta_score=score,
        action_score=action_score,
        base_delta=stacked.base.delta,
        base_positive=stacked.base.positive,
        downside_p95=stacked.base.downside_p95,
        downside_p99=stacked.base.downside_p99,
        downside_max=stacked.base.downside_max,
        delta_disagreement=stacked.base.delta_disagreement,
        delta_positive_votes=stacked.base.delta_positive_votes,
        eligible_mask=stacked.base.eligible_mask,
        stage18_score=stacked.stage18_score,
        proposal_index=int(proposal),
        proposal_eligible=eligible,
    )


def _build_independent_safety_dataset(
    model: HuM43JointModelV5,
    samples: Sequence[PreparedTeacherSample],
    *,
    source: str,
) -> Attempt03SafetyDataset:
    if source != "sealed_calibration.safety_fit":
        raise ValueError("Attempt03 independent safety source changed")
    oof = tuple(
        Attempt03OofStatePrediction(
            sample_index=index,
            fold_index=-1,
            predictions=model.predict_heads_sample(
                sample.policy_sample, baseline_index=sample.baseline_index
            ),
            base_identity_excluded_from_fit=True,
            meta_identity_excluded_from_fit=True,
        )
        for index, sample in enumerate(samples)
    )
    dataset = build_attempt03_oof_safety_dataset(samples, oof)
    return replace(
        dataset,
        rows=tuple({**dict(row), "source": source} for row in dataset.rows),
        manifest={
            **dict(dataset.manifest),
            "row_source": "sealed_calibration_safety_fit_hard_eligible_only",
            "source": source,
            "used_for_threshold_selection": False,
        },
    )


def _threshold_metrics(
    rows: Sequence[Mapping[str, Any]], *, total_states: int, threshold: float
) -> dict[str, Any]:
    deltas = np.asarray([row["teacher_delta"] for row in rows], dtype=np.float64)
    fires = int(deltas.size)
    return {
        "threshold": float(threshold),
        "fires": fires,
        "fire_rate": fires / total_states,
        "teacher_mean_delta_per_fire": float(np.mean(deltas)) if fires else 0.0,
        "teacher_delta_per_state": float(np.sum(deltas) / total_states),
        "false_positive_rate": float(np.mean(deltas <= 0.0)) if fires else 0.0,
        "p95_loss": max((float(row["downside_loss_p95"]) for row in rows), default=0.0),
        "p99_loss": max((float(row["downside_loss_p99"]) for row in rows), default=0.0),
        "max_loss": max((float(row["downside_loss_max"]) for row in rows), default=0.0),
        "teacher_value_status": M43_ATTEMPT03_TEACHER_STATUS,
    }


def _ordered_oof(
    samples: Sequence[PreparedTeacherSample],
    values: Sequence[Attempt03OofStatePrediction],
) -> tuple[Attempt03OofStatePrediction, ...]:
    if len(samples) != len(values):
        raise ValueError("Attempt03 OOF count changed")
    by_index = {row.sample_index: row for row in values}
    if len(by_index) != len(values) or set(by_index) != set(range(len(samples))):
        raise ValueError("Attempt03 OOF coverage changed")
    return tuple(by_index[index] for index in range(len(samples)))


def _identity_disjoint(
    left: Sequence[PreparedTeacherSample], right: Sequence[PreparedTeacherSample]
) -> bool:
    left_seeds = set().union(*(sample.root_seed_values for sample in left))
    right_seeds = set().union(*(sample.root_seed_values for sample in right))
    left_fp = {sample.observation_fingerprint for sample in left}
    right_fp = {sample.observation_fingerprint for sample in right}
    return not (left_seeds & right_seeds or left_fp & right_fp)


def _require_targets(sample: PreparedTeacherSample) -> None:
    if any(
        value is None
        for value in (
            sample.teacher_delta_se_vs_baseline,
            sample.teacher_paired_delta_mean,
            sample.downside_loss_p95,
            sample.downside_loss_p99,
            sample.downside_loss_max,
        )
    ):
        raise ValueError("Attempt03 requires complete paired delta/tail targets")


def _load_job_artifact(
    job_dir: Path,
    *,
    definition: M43FoldJobDefinition,
    contract_file_sha256: str,
    input_bundle_sha256: str,
    training_config_sha256: str,
) -> tuple[PairedDeltaRiskFoldEstimator, dict[str, Any]]:
    artifact_path, manifest_path, done_path = _job_files(job_dir)
    if not all(path.is_file() for path in (artifact_path, manifest_path, done_path)):
        raise ValueError("Attempt03 fold artifact set is incomplete")
    manifest = _read_mapping(manifest_path, "Attempt03 job manifest")
    expected = _job_manifest(
        definition.spec,
        run_name=str(manifest.get("run_name", "")),
        source_sha256=str(manifest.get("source_sha256", "")),
        run_manifest_sha256=str(manifest.get("run_manifest_sha256", "")),
        contract_file_sha256=contract_file_sha256,
        input_bundle_sha256=input_bundle_sha256,
        training_config_sha256=training_config_sha256,
        artifact_sha256=_file_sha256(artifact_path),
    )
    if manifest != expected or _read_mapping(done_path, "Attempt03 DONE") != (
        _done_from_manifest(expected, manifest_path)
    ):
        raise ValueError("Attempt03 fold artifact hash chain changed")
    payload = pickle.loads(artifact_path.read_bytes())
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "model_schema",
        "proposal_schema",
        "training_config_sha256",
        "job_spec_sha256",
        "estimator",
    }:
        raise ValueError("Attempt03 fold artifact header changed")
    if (
        payload["schema"] != M43_ATTEMPT03_FOLD_ESTIMATOR_SCHEMA
        or payload["model_schema"] != HU_M43_V5_MODEL_SCHEMA
        or payload["proposal_schema"] != HU_M43_V5_PROPOSAL_SCHEMA
        or payload["training_config_sha256"] != training_config_sha256
        or payload["job_spec_sha256"] != definition.spec.sha256
    ):
        raise ValueError("Attempt03 fold artifact binding changed")
    estimator = payload.get("estimator")
    if not isinstance(estimator, PairedDeltaRiskFoldEstimator):
        raise TypeError("Attempt03 fold estimator type changed")
    estimator.__post_init__()
    if estimator.fold_index != definition.spec.estimator_fold_index:
        raise ValueError("Attempt03 fold estimator index changed")
    return estimator, {
        "job_index": definition.spec.job_index,
        "artifact_sha256": _file_sha256(artifact_path),
        "job_manifest_sha256": _file_sha256(manifest_path),
        "job_spec_sha256": definition.spec.sha256,
        "run_name": manifest["run_name"],
        "source_sha256": manifest["source_sha256"],
        "run_manifest_sha256": manifest["run_manifest_sha256"],
    }


def _validate_existing_done(
    output_dir: Path,
    *,
    definition: M43FoldJobDefinition,
    contract_file_sha256: str,
    training_config_sha256: str,
    run_name: str,
    source_sha256: str,
    run_manifest_sha256: str,
    input_bundle_sha256: str,
) -> dict[str, Any]:
    artifact, manifest_path, done_path = _job_files(output_dir)
    if not artifact.is_file() or not manifest_path.is_file():
        raise ValueError("Attempt03 completed fold is incomplete")
    expected = _job_manifest(
        definition.spec,
        run_name=run_name,
        source_sha256=source_sha256,
        run_manifest_sha256=run_manifest_sha256,
        contract_file_sha256=contract_file_sha256,
        input_bundle_sha256=input_bundle_sha256,
        training_config_sha256=training_config_sha256,
        artifact_sha256=_file_sha256(artifact),
    )
    if _read_mapping(manifest_path, "job manifest") != expected:
        raise ValueError("Attempt03 existing job manifest changed")
    done = _done_from_manifest(expected, manifest_path)
    if _read_mapping(done_path, "DONE") != done:
        raise ValueError("Attempt03 existing DONE changed")
    return done


def _job_manifest(
    spec: M43FoldJobSpec,
    *,
    run_name: str,
    source_sha256: str,
    run_manifest_sha256: str,
    contract_file_sha256: str,
    input_bundle_sha256: str,
    training_config_sha256: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    _validate_run_name(run_name)
    digests = {
        key: _require_sha256(value, key)
        for key, value in {
            "source_sha256": source_sha256,
            "run_manifest_sha256": run_manifest_sha256,
            "cloud_contract_file_sha256": contract_file_sha256,
            "input_bundle_sha256": input_bundle_sha256,
            "training_config_sha256": training_config_sha256,
            "artifact_sha256": artifact_sha256,
        }.items()
    }
    return {
        "schema": M43_ATTEMPT03_FOLD_JOB_MANIFEST_SCHEMA,
        "status": "pass",
        "run_name": run_name,
        "job_index": spec.job_index,
        "job_kind": spec.kind,
        "outer_fold": spec.outer_fold,
        "inner_fold": spec.inner_fold,
        "job_spec": spec.to_manifest(),
        "job_spec_sha256": spec.sha256,
        **digests,
        "fit700_only": True,
        "holdout_input_count": 0,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def _done_from_manifest(manifest: Mapping[str, Any], path: Path) -> dict[str, Any]:
    done = {
        key: value
        for key, value in manifest.items()
        if key not in {"job_spec", "schema", "status"}
    }
    done.update(
        {
            "schema": M43_ATTEMPT03_FOLD_DONE_SCHEMA,
            "status": "complete",
            "job_manifest_sha256": _file_sha256(path),
        }
    )
    return done


def _validate_fold_plan(value: Any) -> None:
    plan = _mapping(value, "fold_plan")
    if set(plan) != {
        "outer_folds",
        "inner_folds_per_outer",
        "total_jobs",
        "fit_identity_sha256",
        "jobs",
        "fold_plan_sha256",
    }:
        raise ValueError("Attempt03 fold plan key set changed")
    unsigned = dict(plan)
    declared = _require_sha256(unsigned.pop("fold_plan_sha256"), "fold plan")
    if canonical_manifest_sha256(unsigned) != declared:
        raise ValueError("Attempt03 fold plan digest changed")
    if (plan["outer_folds"], plan["inner_folds_per_outer"], plan["total_jobs"]) != (
        5,
        5,
        30,
    ):
        raise ValueError("Attempt03 fold grid is not 5x(1+5)")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 30:
        raise ValueError("Attempt03 fold plan job count changed")
    for index, raw in enumerate(jobs):
        job = dict(_mapping(raw, f"job {index}"))
        digest = _require_sha256(job.pop("job_spec_sha256", None), "job spec")
        if canonical_manifest_sha256(job) != digest or job.get("job_index") != index:
            raise ValueError("Attempt03 fold job order/digest changed")


def _input_entry(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> dict[str, Any]:
    identities = []
    for row in rows:
        seed = row.get("hand_seed", row.get("root_seed"))
        fingerprint = row.get("observation_fingerprint")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Attempt03 {role} seed is invalid")
        identities.append(f"{seed}\t{_require_sha256(fingerprint, 'fingerprint')}")
    identity_sha = hashlib.sha256(
        ("\n".join(sorted(identities)) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "role": role,
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "identity_sha256": identity_sha,
    }


def _dependency_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "scipy": scipy.__version__,
    }


def _reject_worker_sensitive(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _WORKER_FORBIDDEN_KEYS:
                raise ValueError("Attempt03 worker contract contains holdout input")
            _reject_worker_sensitive(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_worker_sensitive(child)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_run_name(value: str) -> None:
    if not value or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in value
    ):
        raise ValueError("Attempt03 run_name contains unsafe characters")


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _job_files(root: Path) -> tuple[Path, Path, Path]:
    return root / "estimator.pkl", root / "job_manifest.json", root / "DONE.json"


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_pickle_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "Attempt03FitResult",
    "Attempt03FoldArtifactProvider",
    "Attempt03OofStatePrediction",
    "Attempt03SafetyDataset",
    "Attempt03SafetyFitResult",
    "Attempt03ThresholdResult",
    "Attempt03TrainingConfig",
    "M43_ATTEMPT03_FIXED_THRESHOLDS",
    "M43_ATTEMPT03_FOLD_JOBS",
    "M43_ATTEMPT03_SCIENCE_CORRECTION_PATH",
    "M43_ATTEMPT03_SCIENCE_CORRECTION_SCHEMA",
    "M43_ATTEMPT03_TRAINING_FREEZE_SCHEMA",
    "M43_ATTEMPT03_TRAINING_SOURCE_PATHS",
    "M43_ATTEMPT03_WORKER_SOURCE_PATHS",
    "audit_attempt03_base_oof_dependencies",
    "build_attempt03_fit_manifest",
    "build_attempt03_final_manifest",
    "build_attempt03_fold_cloud_contract",
    "build_attempt03_oof_safety_dataset",
    "build_attempt03_precalibration_receipt",
    "claim_attempt03_precalibration",
    "evaluate_attempt03_precalibration",
    "fit_attempt03_base_oof_runtime_meta",
    "fit_attempt03_safety_calibrator",
    "load_attempt03_fit_bundle",
    "load_attempt03_fold_artifact_provider",
    "load_attempt03_fold_cloud_contract",
    "load_attempt03_model_freeze",
    "load_attempt03_training_science_correction",
    "load_attempt03_training_freeze",
    "run_attempt03_fold_job",
    "save_attempt03_fit_bundle",
    "select_attempt03_threshold",
    "write_attempt03_fold_cloud_contract",
]
