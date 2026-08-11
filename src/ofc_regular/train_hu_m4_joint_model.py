"""Train/calibrate the M4 HU T1 joint action model on explicit split files.

The CLI never creates or re-splits a holdout.  ``--train`` and
``--calibration`` accept repeatable, explicitly ordered JSONL shards; all
input/output paths must be distinct and the combined splits must have disjoint
root/hand seeds.  Legacy/M4.2 may additionally receive ``--locked-holdout`` as
a read-only report at the fixed threshold.  M4.3 rejects that argument and
leaves locked evaluation to the separate post-freeze one-shot command.  Its
sealed calibration roles are consumed exactly as declared by the data
contract.  The safety head sees ``safety_fit`` and strictly nested OOF train
examples, while the threshold sweep sees only ``threshold_lock``.

Teacher scores in every report are search-teacher diagnostics, not realized
match EV and not a permitted runtime gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .action_key import action_key_from_payload
from .hu_infoset import ActorObservation
from .hu_m43_pilot_contract import (
    M43_DATA_CONTRACT_SCHEMA,
    canonical_manifest_sha256,
    validate_data_contract_binding,
)
from .hu_m4_joint_model import (
    ACTION_SCORE_MODES,
    DELTA_SCORE_WEIGHT,
    DEFAULT_DOWNSIDE_RISK_SCORE_WEIGHT,
    DEFAULT_ENSEMBLE_DISAGREEMENT_SCORE_WEIGHT,
    DEFAULT_POSITIVE_GAIN_SCORE_WEIGHT,
    HU_M4_JOINT_ARTIFACT_SCHEMA,
    HU_M4_JOINT_FEATURE_SCHEMA,
    HU_M4_JOINT_MODEL_SCHEMA,
    HU_M4_META_RANK_FEATURE_SCHEMA,
    HU_M4_PAIRED_ACTION_FEATURE_SCHEMA,
    LEGACY_ACTION_SCORE_MODE,
    NEGATIVE_REGRET_ACTION_SCORE_MODE,
    PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
    POLICY_SCORE_CENTER,
    POLICY_SCORE_WEIGHT,
    VALUE_SCORE_WEIGHT,
    ConstantProbabilityEstimator,
    HuM4JointActionModel,
    JointHeadPredictions,
    PairedDeltaRiskFoldEstimator,
    build_joint_safety_features,
    build_meta_rank_features,
    build_paired_action_features,
    canonical_action_argmax,
)
from .hu_turn3_model import HU_FEATURE_DIM, sample_to_matrix
from .policy import board_to_json


HU_M4_JOINT_TRAINING_MANIFEST_SCHEMA = "hu_m4_t1_joint_training_manifest_v2"
TEACHER_VALUE_STATUS = "diagnostic_only_not_realized_match_ev_not_runtime_gate"
M43_DEFAULT_PILOT_MINIMUM_FIRES = 10
LEGACY_DEFAULT_MINIMUM_CALIBRATION_FIRES = 30
_SEED_KEYS = ("root_seed", "hand_seed")
_TOP_LEVEL_TRUTH_KEYS = frozenset(
    {
        "opponent_private_discards",
        "true_opponent_private_discards",
        "true_dead_cards",
        "replay_truth",
        "replay_world",
        "world_state",
        "future_cards",
        "remaining_deck",
        "deck_tail",
    }
)
_FORBIDDEN_OBSERVATION_KEYS = _TOP_LEVEL_TRUTH_KEYS | {
    "dead_cards",
    "visible_dead_cards",
    "private_discards",
}


@dataclass(frozen=True)
class PreparedTeacherSample:
    policy_sample: dict[str, Any]
    teacher_scores: np.ndarray
    teacher_score_se: np.ndarray
    teacher_delta_se_vs_baseline: np.ndarray | None
    teacher_paired_delta_mean: np.ndarray | None
    downside_loss_p95: np.ndarray | None
    downside_loss_p99: np.ndarray | None
    downside_loss_max: np.ndarray | None
    baseline_index: int
    seat: str
    root_seed_values: frozenset[str]
    observation_fingerprint: str
    ignored_truth_keys: tuple[str, ...]


@dataclass(frozen=True)
class CrossFitSafetyExamples:
    """OOF-only safety rows derived from the training split."""

    features: np.ndarray
    labels: np.ndarray
    rows: tuple[Mapping[str, Any], ...]
    seed_values: frozenset[str]
    observation_fingerprints: frozenset[str]
    row_hashes: frozenset[str]


@dataclass(frozen=True)
class CrossFitTrainingResult:
    """Final full-data model plus auditable out-of-fold by-products."""

    model: HuM4JointActionModel
    safety_examples: CrossFitSafetyExamples
    report: Mapping[str, Any]


@dataclass(frozen=True)
class M43FoldJobSpec:
    """Deterministic identity and fit contract for one M4.3 fold estimator."""

    job_index: int
    kind: str
    outer_fold: int
    inner_fold: int | None
    estimator_fold_index: int
    estimator_seed: int
    fit_samples: int
    fit_identity_sha256: str
    outer_validation_samples: int
    outer_validation_identity_sha256: str
    inner_validation_samples: int
    inner_validation_identity_sha256: str | None
    outer_assignment_sha256: str
    inner_assignment_sha256: str | None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "job_index": self.job_index,
            "kind": self.kind,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "estimator_fold_index": self.estimator_fold_index,
            "estimator_seed": self.estimator_seed,
            "fit_samples": self.fit_samples,
            "fit_identity_sha256": self.fit_identity_sha256,
            "outer_validation_samples": self.outer_validation_samples,
            "outer_validation_identity_sha256": (
                self.outer_validation_identity_sha256
            ),
            "inner_validation_samples": self.inner_validation_samples,
            "inner_validation_identity_sha256": (
                self.inner_validation_identity_sha256
            ),
            "outer_assignment_sha256": self.outer_assignment_sha256,
            "inner_assignment_sha256": self.inner_assignment_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_manifest_sha256(self.to_manifest())


@dataclass(frozen=True)
class M43FoldJobDefinition:
    spec: M43FoldJobSpec
    fit_samples: tuple[PreparedTeacherSample, ...]


@dataclass(frozen=True)
class M43FoldTrainingPlan:
    ordered_samples: tuple[PreparedTeacherSample, ...]
    outer_fold_ids: tuple[int, ...]
    outer_fold_report: Mapping[str, Any]
    inner_fold_reports: tuple[Mapping[str, Any], ...]
    jobs: tuple[M43FoldJobDefinition, ...]


M43FoldEstimatorProvider = Callable[
    [M43FoldJobSpec, Sequence[PreparedTeacherSample]],
    PairedDeltaRiskFoldEstimator,
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    parser.add_argument(
        "--locked-holdout",
        type=Path,
        action="append",
        help=(
            "Legacy/M4.2 only. M4.3 rejects locked labels during training; use "
            "the separate post-freeze one-shot evaluator."
        ),
    )
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--m43-data-contract",
        type=Path,
        help=(
            "Required for baseline_paired_delta_risk_ensemble_v3; binds the "
            "audited profile-stratified calibration role identities."
        ),
    )
    parser.add_argument("--m43-plan", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--model-id", default="hu-m4-t1-joint-v1")
    parser.add_argument("--near-best-margin", type=float, default=0.5)
    parser.add_argument("--minimum-safe-teacher-gain", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026071801)
    parser.add_argument(
        "--action-score-mode",
        choices=sorted(ACTION_SCORE_MODES),
        default=LEGACY_ACTION_SCORE_MODE,
    )
    parser.add_argument(
        "--cross-fit-folds",
        type=int,
        default=1,
        help="1 preserves legacy fitting; use 2+ for identity-group OOF fitting.",
    )
    parser.add_argument("--paired-se-floor", type=float, default=0.50)
    parser.add_argument("--paired-huber-alpha", type=float, default=0.90)
    parser.add_argument("--downside-quantile", type=float, default=0.90)
    parser.add_argument(
        "--positive-gain-score-weight",
        type=float,
        default=DEFAULT_POSITIVE_GAIN_SCORE_WEIGHT,
    )
    parser.add_argument(
        "--downside-risk-score-weight",
        type=float,
        default=DEFAULT_DOWNSIDE_RISK_SCORE_WEIGHT,
    )
    parser.add_argument(
        "--ensemble-disagreement-score-weight",
        type=float,
        default=DEFAULT_ENSEMBLE_DISAGREEMENT_SCORE_WEIGHT,
    )
    parser.add_argument(
        "--safety-calibrator-c",
        type=float,
        default=0.25,
        help="L2-regularized logistic safety-head inverse regularization for M4.3.",
    )
    parser.add_argument("--safety-fit-ratio", type=float, default=0.5)
    parser.add_argument("--safety-split-seed", type=int, default=2026071802)
    parser.add_argument("--minimum-safety-fit-samples", type=int, default=30)
    parser.add_argument("--minimum-threshold-lock-samples", type=int, default=30)
    parser.add_argument(
        "--thresholds",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99,1",
    )
    parser.add_argument(
        "--minimum-calibration-fires",
        type=int,
        default=None,
        help=(
            "Threshold-lock pilot signal minimum. Defaults to 10 for M4.3 and "
            "30 for legacy/M4.2; final promotion fire count is an external gate."
        ),
    )
    parser.add_argument("--maximum-false-positive-rate", type=float, default=0.30)
    parser.add_argument("--maximum-p95-loss", type=float, default=25.0)
    parser.add_argument("--maximum-p99-loss", type=float, default=40.0)
    parser.add_argument("--maximum-max-loss", type=float, default=50.0)
    return parser.parse_args(argv)


def read_teacher_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{source}: no teacher rows")
    return rows


def prepare_teacher_sample(row: Mapping[str, Any]) -> PreparedTeacherSample:
    """Build features from the nested ActorObservation and nothing else."""

    nested = row.get("policy_observation")
    if not isinstance(nested, Mapping):
        raise ValueError("M4 teacher row requires policy_observation")
    nested_forbidden = sorted(_find_forbidden_keys(nested, _FORBIDDEN_OBSERVATION_KEYS))
    if nested_forbidden:
        raise ValueError(
            "policy_observation contains forbidden hidden/world keys: "
            + ",".join(nested_forbidden)
        )
    observation = ActorObservation.from_dict(nested)
    if observation.street != "T1":
        raise ValueError("M4 joint trainer accepts only T1 observations")

    raw_actions = row.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("M4 teacher row requires a non-empty actions list")
    actions: list[dict[str, Any]] = []
    action_keys: set[str] = set()
    scores = np.empty(len(raw_actions), dtype=np.float64)
    standard_errors = np.empty(len(raw_actions), dtype=np.float64)
    paired_delta_standard_errors = np.empty(len(raw_actions), dtype=np.float64)
    paired_delta_se_present: list[bool] = []
    paired_delta_means = np.empty(len(raw_actions), dtype=np.float64)
    downside_loss_p95 = np.empty(len(raw_actions), dtype=np.float64)
    downside_loss_p99 = np.empty(len(raw_actions), dtype=np.float64)
    downside_loss_max = np.empty(len(raw_actions), dtype=np.float64)
    paired_summary_present: list[bool] = []
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, Mapping):
            raise ValueError(f"action {index} must be an object")
        if "score" not in raw_action or "score_se" not in raw_action:
            raise ValueError(f"action {index} requires score and score_se")
        score = float(raw_action["score"])
        score_se = float(raw_action["score_se"])
        if not math.isfinite(score) or not math.isfinite(score_se) or score_se < 0.0:
            raise ValueError(f"action {index} has invalid score/score_se")
        action = dict(raw_action)
        semantic_key = action_key_from_payload(action).to_token()
        declared_key = action.get("action_key")
        if declared_key is not None and declared_key != semantic_key:
            raise ValueError(f"action {index} ActionKey disagrees with payload")
        if semantic_key in action_keys:
            raise ValueError(f"action {index} duplicates a semantic legal action")
        action_keys.add(semantic_key)
        # The common encoder reads score only as its returned training target.
        # It receives no other raw teacher-row fields.
        action["score"] = score
        actions.append(action)
        scores[index] = score
        standard_errors[index] = score_se
        has_paired_delta_se = "delta_se_vs_baseline" in raw_action
        paired_delta_se_present.append(has_paired_delta_se)
        if has_paired_delta_se:
            paired_delta_se = float(raw_action["delta_se_vs_baseline"])
            if not math.isfinite(paired_delta_se) or paired_delta_se < 0.0:
                raise ValueError(f"action {index} has invalid delta_se_vs_baseline")
            paired_delta_standard_errors[index] = paired_delta_se
        else:
            paired_delta_standard_errors[index] = 0.0
        raw_summary = raw_action.get("paired_delta_vs_baseline")
        has_summary = isinstance(raw_summary, Mapping)
        paired_summary_present.append(has_summary)
        if has_summary:
            summary = raw_summary
            required = ("mean", "standard_error", "p05", "p01", "min")
            if any(key not in summary for key in required):
                raise ValueError(
                    f"action {index} paired delta summary is missing required targets"
                )
            values = {key: float(summary[key]) for key in required}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"action {index} paired delta summary is non-finite")
            if has_paired_delta_se and not math.isclose(
                values["standard_error"],
                paired_delta_standard_errors[index],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"action {index} paired summary standard error disagrees"
                )
            paired_delta_means[index] = values["mean"]
            downside_loss_p95[index] = max(0.0, -values["p05"])
            downside_loss_p99[index] = max(0.0, -values["p01"])
            downside_loss_max[index] = max(0.0, -values["min"])
        else:
            paired_delta_means[index] = 0.0
            downside_loss_p95[index] = 0.0
            downside_loss_p99[index] = 0.0
            downside_loss_max[index] = 0.0

    if any(paired_delta_se_present) and not all(paired_delta_se_present):
        raise ValueError("schema-v2 paired delta SE must be present on every action")
    if any(paired_summary_present) and not all(paired_summary_present):
        raise ValueError("paired delta summary must be present on every action")

    raw_baseline = row.get("baseline_action_row_index")
    if isinstance(raw_baseline, bool) or not isinstance(raw_baseline, int):
        raise ValueError("baseline_action_row_index must be an integer")
    baseline_index = int(raw_baseline)
    if not 0 <= baseline_index < len(actions):
        raise IndexError("baseline_action_row_index is outside actions")
    if all(paired_summary_present):
        expected_delta = scores - float(scores[baseline_index])
        if not np.allclose(
            paired_delta_means, expected_delta, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError("paired delta mean disagrees with score minus baseline")
        baseline_targets = (
            paired_delta_means[baseline_index],
            paired_delta_standard_errors[baseline_index],
            downside_loss_p95[baseline_index],
            downside_loss_p99[baseline_index],
            downside_loss_max[baseline_index],
        )
        if any(value != 0.0 for value in baseline_targets):
            raise ValueError("baseline paired delta/downside targets must be exactly zero")

    root_seed_values = frozenset(
        str(row[key]) for key in _SEED_KEYS if key in row and row[key] is not None
    )
    if not root_seed_values:
        raise ValueError("M4 split rows require root_seed or hand_seed")

    policy_sample = {
        "rule_set": "regular",
        "schema": HU_M4_JOINT_FEATURE_SCHEMA,
        "phase": "hu_turn1_5card",
        "seat": observation.seat,
        "to_act_order": observation.to_act_order,
        "board": board_to_json(observation.hero_board),
        "opponent_board": board_to_json(observation.opponent_public_board),
        # The legacy field name is retained only at the common encoder boundary.
        # Its contents are public opponent cards plus hero's own discards.
        "dead_cards": list(observation.legacy_dead_cards()),
        "dealt": list(observation.dealt_cards),
        "best_action": int(np.argmax(scores)),
        "score_gap": float(np.max(scores) - np.partition(scores, -2)[-2])
        if scores.size > 1
        else 0.0,
        "actions": actions,
        # Metadata only; ignored by the common encoder.  M4.3 uses the exact
        # semantic baseline mapping to construct candidate-paired features.
        "baseline_action_row_index": baseline_index,
    }
    features, encoded_targets = sample_to_matrix(policy_sample)
    if features.shape != (len(actions), HU_FEATURE_DIM):
        raise ValueError("common HU encoder returned an unexpected shape")
    if not np.isfinite(features).all() or not np.array_equal(encoded_targets, scores):
        raise ValueError("common HU encoder target/feature validation failed")
    return PreparedTeacherSample(
        policy_sample=policy_sample,
        teacher_scores=scores,
        teacher_score_se=standard_errors,
        teacher_delta_se_vs_baseline=(
            paired_delta_standard_errors if all(paired_delta_se_present) else None
        ),
        teacher_paired_delta_mean=(
            paired_delta_means if all(paired_summary_present) else None
        ),
        downside_loss_p95=(
            downside_loss_p95 if all(paired_summary_present) else None
        ),
        downside_loss_p99=(
            downside_loss_p99 if all(paired_summary_present) else None
        ),
        downside_loss_max=(
            downside_loss_max if all(paired_summary_present) else None
        ),
        baseline_index=baseline_index,
        seat=observation.seat,
        root_seed_values=root_seed_values,
        observation_fingerprint=observation.fingerprint(),
        ignored_truth_keys=tuple(sorted(set(row) & _TOP_LEVEL_TRUTH_KEYS)),
    )


def prepare_teacher_samples(
    rows: Iterable[Mapping[str, Any]],
) -> list[PreparedTeacherSample]:
    prepared = [prepare_teacher_sample(row) for row in rows]
    if not prepared:
        raise ValueError("no prepared M4 teacher samples")
    return prepared


def validate_disjoint_splits(
    train: Sequence[PreparedTeacherSample],
    calibration: Sequence[PreparedTeacherSample],
    locked_holdout: Sequence[PreparedTeacherSample],
) -> dict[str, Any]:
    named = {
        "train": train,
        "calibration": calibration,
        "locked_holdout": locked_holdout,
    }
    seeds = {
        name: set().union(*(sample.root_seed_values for sample in samples))
        for name, samples in named.items()
    }
    fingerprints = {
        name: {sample.observation_fingerprint for sample in samples}
        for name, samples in named.items()
    }
    names = tuple(named)
    overlaps: list[str] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            seed_overlap = seeds[left] & seeds[right]
            fingerprint_overlap = fingerprints[left] & fingerprints[right]
            if seed_overlap:
                overlaps.append(f"{left}/{right}:seed={sorted(seed_overlap)[:5]}")
            if fingerprint_overlap:
                overlaps.append(
                    f"{left}/{right}:observation_fingerprint={len(fingerprint_overlap)}"
                )
    if overlaps:
        raise ValueError("M4 split overlap detected: " + "; ".join(overlaps))
    return {
        "status": "pass",
        "seed_overlap_count": 0,
        "observation_fingerprint_overlap_count": 0,
        "split_rows": {name: len(samples) for name, samples in named.items()},
        "split_unique_seed_values": {name: len(values) for name, values in seeds.items()},
    }


def _load_m43_role_binding(
    path: str | Path,
    *,
    plan_path: str | Path,
    repo_root: str | Path,
    train_paths: Sequence[str | Path],
    calibration_paths: Sequence[str | Path],
    raw: Mapping[str, Sequence[Mapping[str, Any]]],
    prepared: Mapping[str, Sequence[PreparedTeacherSample]],
) -> tuple[
    list[PreparedTeacherSample],
    list[PreparedTeacherSample],
    dict[str, Any],
    dict[str, Any],
]:
    """Bind training to the audited profile-stratified 30/30 identities."""

    source = Path(path).resolve()
    contract = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(contract, dict):
        raise ValueError("M4.3 data contract must be a mapping")
    if contract.get("schema") != M43_DATA_CONTRACT_SCHEMA:
        raise ValueError("unsupported M4.3 data contract schema")
    if contract.get("status") != "pass_fresh_data_sealed_for_model_freeze":
        raise ValueError("M4.3 data contract is not sealed for model freeze")
    declared_contract_sha = contract.get("contract_sha256")
    if not isinstance(declared_contract_sha, str) or len(declared_contract_sha) != 64:
        raise ValueError("M4.3 data contract SHA-256 is missing")
    unsigned = dict(contract)
    unsigned.pop("contract_sha256", None)
    if canonical_manifest_sha256(unsigned) != declared_contract_sha:
        raise ValueError("M4.3 data contract canonical SHA-256 mismatch")
    binding_validation = validate_data_contract_binding(
        contract,
        plan_path=plan_path,
        repo_root=repo_root,
        train=train_paths,
        calibration=calibration_paths,
    )

    expected_records = {"train": 100, "calibration": 60, "locked_holdout": 40}
    split_identities: dict[str, set[tuple[int, str]]] = {}
    split_entries = contract.get("splits")
    if not isinstance(split_entries, Mapping):
        raise ValueError("M4.3 data contract splits are missing")
    for split in ("train", "calibration"):
        expected = expected_records[split]
        raw_rows = raw[split]
        samples = prepared[split]
        if len(raw_rows) != expected or len(samples) != expected:
            raise ValueError(f"M4.3 {split} must contain exactly {expected} rows")
        identities = {
            _m43_training_identity(row, sample)
            for row, sample in zip(raw_rows, samples, strict=True)
        }
        if len(identities) != expected:
            raise ValueError(f"M4.3 {split} identities are not unique")
        declared = split_entries.get(split)
        if not isinstance(declared, Mapping):
            raise ValueError(f"M4.3 data contract split is missing: {split}")
        if int(declared.get("records", -1)) != expected:
            raise ValueError(f"M4.3 data contract record count changed: {split}")
        if declared.get("identity_sha256") != _m43_identity_digest(identities):
            raise ValueError(f"M4.3 data contract identity digest mismatch: {split}")
        split_identities[split] = identities
    locked_declared = split_entries.get("locked_holdout")
    if not isinstance(locked_declared, Mapping):
        raise ValueError("M4.3 sealed locked-holdout metadata is missing")
    if int(locked_declared.get("records", -1)) != 40:
        raise ValueError("M4.3 sealed locked-holdout record count changed")
    locked_identity_sha = locked_declared.get("identity_sha256")
    if not isinstance(locked_identity_sha, str) or len(locked_identity_sha) != 64:
        raise ValueError("M4.3 sealed locked-holdout identity digest is invalid")

    partition = contract.get("calibration_partition")
    if not isinstance(partition, Mapping) or partition.get("schema") != (
        "hu_m43_calibration_partition_v1"
    ):
        raise ValueError("M4.3 calibration partition is missing")
    calibration_rows = raw["calibration"]
    calibration_samples = prepared["calibration"]
    by_identity = {
        _m43_training_identity(row, sample): (row, sample)
        for row, sample in zip(calibration_rows, calibration_samples, strict=True)
    }
    roles: dict[str, list[PreparedTeacherSample]] = {}
    role_report: dict[str, Any] = {}
    role_sets: dict[str, set[tuple[int, str]]] = {}
    for role in ("safety_fit", "threshold_lock"):
        declared = partition.get(role)
        if not isinstance(declared, Mapping):
            raise ValueError(f"M4.3 calibration role is missing: {role}")
        if int(declared.get("records", -1)) != 30:
            raise ValueError(f"M4.3 {role} must contain exactly 30 identities")
        identity_rows = declared.get("identities")
        if not isinstance(identity_rows, list) or len(identity_rows) != 30:
            raise ValueError(f"M4.3 {role} identity list is invalid")
        ordered_identities = [_m43_declared_identity(row) for row in identity_rows]
        identity_set = set(ordered_identities)
        if len(identity_set) != 30:
            raise ValueError(f"M4.3 {role} identities are duplicated")
        if declared.get("identity_sha256") != _m43_identity_digest(identity_set):
            raise ValueError(f"M4.3 {role} identity digest mismatch")
        if not identity_set <= split_identities["calibration"]:
            raise ValueError(f"M4.3 {role} contains an unaudited calibration identity")
        missing = identity_set - set(by_identity)
        if missing:
            raise ValueError(f"M4.3 {role} rows are missing from trainer input")
        profile_counts: dict[str, int] = {}
        for identity in identity_set:
            raw_row, _sample = by_identity[identity]
            provenance = raw_row.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError("M4.3 calibration row provenance is missing")
            profile = str(provenance.get("root_profile", ""))
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
        expected_profiles = declared.get("profile_counts")
        if not isinstance(expected_profiles, Mapping) or {
            str(key): int(value) for key, value in expected_profiles.items()
        } != dict(sorted(profile_counts.items())):
            raise ValueError(f"M4.3 {role} profile counts disagree with contract")
        if set(profile_counts.values()) != {6} or len(profile_counts) != 5:
            raise ValueError(f"M4.3 {role} is not five-profile balanced")
        roles[role] = [by_identity[identity][1] for identity in ordered_identities]
        role_sets[role] = identity_set
        role_report[role] = {
            "records": 30,
            "identity_sha256": declared["identity_sha256"],
            "profile_counts": dict(sorted(profile_counts.items())),
        }
    if role_sets["safety_fit"] & role_sets["threshold_lock"]:
        raise ValueError("M4.3 calibration role identities overlap")
    if role_sets["safety_fit"] | role_sets["threshold_lock"] != split_identities[
        "calibration"
    ]:
        raise ValueError("M4.3 calibration roles do not cover the sealed split")
    audit = {
        "schema": "hu_m43_trainer_data_contract_binding_v1",
        "status": "pass",
        "path": str(source),
        "file_sha256": _sha256(source),
        "contract_sha256": declared_contract_sha,
        "partition_method": partition.get("method"),
        "legacy_hash_partition_used": False,
        **role_report,
        "overlap": 0,
        "locked_holdout_labels_used_for_role_assignment": False,
        "locked_holdout_labels_opened_by_trainer": False,
        "sealed_locked_holdout_identity_sha256": locked_identity_sha,
    }
    return (
        roles["safety_fit"],
        roles["threshold_lock"],
        audit,
        binding_validation,
    )


def _m43_training_identity(
    row: Mapping[str, Any], sample: PreparedTeacherSample
) -> tuple[int, str]:
    hand_seed = row.get("hand_seed")
    if isinstance(hand_seed, bool) or not isinstance(hand_seed, int):
        raise ValueError("M4.3 trainer row requires integer hand_seed")
    declared_fingerprint = row.get("observation_fingerprint")
    if declared_fingerprint is not None and declared_fingerprint != (
        sample.observation_fingerprint
    ):
        raise ValueError("M4.3 trainer row observation fingerprint mismatch")
    return int(hand_seed), sample.observation_fingerprint


def _m43_declared_identity(row: Any) -> tuple[int, str]:
    if not isinstance(row, Mapping):
        raise ValueError("M4.3 declared identity must be a mapping")
    seed = row.get("hand_seed")
    fingerprint = row.get("observation_fingerprint")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("M4.3 declared identity hand_seed must be an integer")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("M4.3 declared identity fingerprint is invalid")
    return int(seed), fingerprint


def _m43_identity_digest(values: Iterable[tuple[int, str]]) -> str:
    encoded = "".join(
        f"{seed}\t{fingerprint}\n" for seed, fingerprint in sorted(values)
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def split_safety_calibration(
    samples: Sequence[PreparedTeacherSample],
    *,
    safety_fit_ratio: float = 0.5,
    split_seed: int = 2026071802,
    minimum_safety_fit_samples: int = 30,
    minimum_threshold_lock_samples: int = 30,
) -> tuple[list[PreparedTeacherSample], list[PreparedTeacherSample], dict[str, Any]]:
    """Deterministically split calibration without breaking an identity group.

    Rows connected by *either* a root/hand seed or an observation fingerprint
    form one indivisible component.  Hash-ranked components, rather than input
    row order, determine membership.  This makes the split reproducible and
    prevents a repeated root from leaking into the threshold lock.
    """

    if not samples:
        raise ValueError("calibration partition requires samples")
    if not math.isfinite(safety_fit_ratio) or not 0.0 < safety_fit_ratio < 1.0:
        raise ValueError("safety_fit_ratio must be finite and strictly between 0 and 1")
    if minimum_safety_fit_samples < 1 or minimum_threshold_lock_samples < 1:
        raise ValueError("minimum safety subset sizes must be positive")

    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    seed_owner: dict[str, int] = {}
    fingerprint_owner: dict[str, int] = {}
    for index, sample in enumerate(samples):
        for seed_value in sample.root_seed_values:
            previous = seed_owner.setdefault(seed_value, index)
            union(index, previous)
        previous = fingerprint_owner.setdefault(sample.observation_fingerprint, index)
        union(index, previous)

    components: dict[int, list[PreparedTeacherSample]] = {}
    for index, sample in enumerate(samples):
        components.setdefault(find(index), []).append(sample)
    ranked_components: list[tuple[str, str, list[PreparedTeacherSample]]] = []
    for component in components.values():
        ordered = sorted(component, key=_sample_membership_hash)
        identity = _digest_strings(_sample_membership_hash(sample) for sample in ordered)
        rank = hashlib.sha256(f"{split_seed}\0{identity}".encode("utf-8")).hexdigest()
        ranked_components.append((rank, identity, ordered))
    ranked_components.sort(key=lambda item: (item[0], item[1]))

    target_fit_rows = safety_fit_ratio * len(samples)
    prefix_rows = [0]
    for _rank, _identity, component in ranked_components:
        prefix_rows.append(prefix_rows[-1] + len(component))
    feasible = [
        index
        for index, fit_rows in enumerate(prefix_rows)
        if fit_rows >= minimum_safety_fit_samples
        and len(samples) - fit_rows >= minimum_threshold_lock_samples
    ]
    candidate_cuts = feasible or list(range(len(prefix_rows)))
    cut = min(
        candidate_cuts,
        key=lambda index: (
            abs(prefix_rows[index] - target_fit_rows),
            abs(index - len(ranked_components) * safety_fit_ratio),
            index,
        ),
    )
    fit = [
        sample
        for _rank, _identity, component in ranked_components[:cut]
        for sample in component
    ]
    lock = [
        sample
        for _rank, _identity, component in ranked_components[cut:]
        for sample in component
    ]
    fit.sort(key=_sample_membership_hash)
    lock.sort(key=_sample_membership_hash)
    overlap = _safety_subset_overlap(fit, lock)
    if any(overlap.values()):
        raise AssertionError(f"safety calibration partition leaked identities: {overlap}")
    status = "pass" if feasible else "no_go_insufficient_samples"
    audit = {
        "schema": "hu_m4_safety_calibration_partition_v1",
        "status": status,
        "method": "connected_seed_fingerprint_groups_hash_ranked_v1",
        "split_seed": int(split_seed),
        "requested_safety_fit_ratio": float(safety_fit_ratio),
        "realized_safety_fit_ratio": float(len(fit) / len(samples)),
        "minimum_samples": {
            "safety_fit": int(minimum_safety_fit_samples),
            "threshold_lock": int(minimum_threshold_lock_samples),
        },
        "component_count": len(ranked_components),
        "safety_fit": _safety_subset_manifest(fit),
        "threshold_lock": _safety_subset_manifest(lock),
        "overlap": overlap,
        "locked_holdout_used": False,
    }
    return fit, lock, audit


def fit_joint_model(
    samples: Sequence[PreparedTeacherSample],
    *,
    model_id: str = "hu-m4-t1-joint-v1",
    near_best_margin: float = 0.5,
    iterations: int = 150,
    max_leaf_nodes: int = 31,
    learning_rate: float = 0.05,
    l2_regularization: float = 1.0,
    seed: int = 2026071801,
) -> HuM4JointActionModel:
    if not samples:
        raise ValueError("joint model training requires samples")
    if not math.isfinite(near_best_margin) or near_best_margin < 0.0:
        raise ValueError("near_best_margin must be non-negative")
    if iterations <= 0 or max_leaf_nodes < 2 or learning_rate <= 0.0:
        raise ValueError("invalid histogram boosting configuration")

    features, scores, standard_errors, delta_targets, policy_targets, weights = (
        _training_arrays(samples, near_best_margin=near_best_margin)
    )
    policy_estimator = _fit_classifier(
        features,
        policy_targets,
        weights,
        iterations=iterations,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        seed=seed,
    )
    value_estimator = _fit_regressor(
        features,
        scores,
        weights,
        iterations=iterations,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        seed=seed + 1,
    )
    delta_estimator = _fit_regressor(
        features,
        delta_targets,
        weights,
        iterations=iterations,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        seed=seed + 2,
    )
    zero_uncertainty = DummyRegressor(strategy="constant", constant=0.0).fit(
        features, np.zeros(features.shape[0], dtype=np.float64)
    )
    provisional = HuM4JointActionModel(
        policy_estimator=policy_estimator,
        value_estimator=value_estimator,
        delta_estimator=delta_estimator,
        uncertainty_estimator=zero_uncertainty,
        model_id=model_id,
    )
    provisional_heads = provisional.predict_heads_matrix(features)
    # The risk head predicts a conservative absolute error proxy.  score_se is
    # used only as an offline training label and never supplied at runtime.
    uncertainty_targets = np.maximum(
        np.abs(scores - provisional_heads.action_score),
        1.96 * standard_errors,
    )
    uncertainty_estimator = _fit_regressor(
        features,
        uncertainty_targets,
        weights,
        iterations=iterations,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        seed=seed + 3,
    )
    return replace(provisional, uncertainty_estimator=uncertainty_estimator)


def _fit_identity_oof_base_heads(
    samples: Sequence[PreparedTeacherSample],
    *,
    folds: int,
    model_id: str,
    near_best_margin: float,
    iterations: int,
    max_leaf_nodes: int,
    learning_rate: float,
    l2_regularization: float,
    seed: int,
) -> tuple[list[JointHeadPredictions], list[int], dict[str, Any], np.ndarray]:
    """Return identity-group OOF base heads for one already isolated subset."""

    fold_ids, fold_report = _assign_cross_fit_folds(samples, folds=folds, seed=seed)
    predictions: list[JointHeadPredictions | None] = [None] * len(samples)
    coverage = np.zeros(len(samples), dtype=np.int16)
    for fold in range(folds):
        fit_samples = [
            sample for index, sample in enumerate(samples) if fold_ids[index] != fold
        ]
        validation_indices = [
            index for index in range(len(samples)) if fold_ids[index] == fold
        ]
        model = fit_joint_model(
            fit_samples,
            model_id=f"{model_id}:fold-{fold}",
            near_best_margin=near_best_margin,
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            seed=seed + 100 * (fold + 1),
        )
        for index in validation_indices:
            heads = model.predict_heads_sample(samples[index].policy_sample)
            predictions[index] = replace(
                heads,
                predicted_absolute_residual=np.zeros_like(
                    heads.predicted_absolute_residual
                ),
            )
            coverage[index] += 1
    if np.any(coverage != 1) or any(heads is None for heads in predictions):
        raise AssertionError("base OOF coverage must predict every sample exactly once")
    return (
        [heads for heads in predictions if heads is not None],
        fold_ids,
        fold_report,
        coverage,
    )


def _fit_three_way_oof_rank_scores(
    samples: Sequence[PreparedTeacherSample],
    *,
    fold_ids: Sequence[int],
    model_id: str,
    near_best_margin: float,
    iterations: int,
    max_leaf_nodes: int,
    learning_rate: float,
    l2_regularization: float,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Produce strict OOF rank scores with disjoint base/meta/target roles.

    For target fold C, the base heads fit only role A.  Their predictions on
    role B train the meta ranker, which then predicts role C.  C therefore has
    no direct or feature-lineage path into either estimator that predicts it.
    """

    unique_folds = sorted(set(int(value) for value in fold_ids))
    if unique_folds != [0, 1, 2]:
        raise ValueError("strict nested rank residuals require exactly 3 inner folds")
    relative_targets = [
        sample.teacher_scores - float(np.max(sample.teacher_scores))
        for sample in samples
    ]
    action_weights = [
        np.full(len(sample.teacher_scores), 1.0 / len(sample.teacher_scores))
        for sample in samples
    ]
    predictions: list[np.ndarray | None] = [None] * len(samples)
    coverage = np.zeros(len(samples), dtype=np.int16)
    lineage_audits: list[dict[str, Any]] = []
    for target_fold in unique_folds:
        meta_fold = (target_fold + 1) % 3
        base_folds = [
            fold for fold in unique_folds if fold not in (target_fold, meta_fold)
        ]
        base_indices = [
            index for index, fold in enumerate(fold_ids) if fold in base_folds
        ]
        meta_indices = [
            index for index, fold in enumerate(fold_ids) if fold == meta_fold
        ]
        target_indices = [
            index for index, fold in enumerate(fold_ids) if fold == target_fold
        ]
        base_samples = [samples[index] for index in base_indices]
        meta_samples = [samples[index] for index in meta_indices]
        target_samples = [samples[index] for index in target_indices]
        target_lineage_overlap = _safety_subset_overlap(
            target_samples, (*base_samples, *meta_samples)
        )
        base_meta_overlap = _safety_subset_overlap(base_samples, meta_samples)
        if any(target_lineage_overlap.values()) or any(base_meta_overlap.values()):
            raise AssertionError(
                "three-way nested rank lineage contains an identity overlap"
            )

        base_model = fit_joint_model(
            base_samples,
            model_id=f"{model_id}:strict-base-target-{target_fold}",
            near_best_margin=near_best_margin,
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            seed=seed + 100 * (target_fold + 1),
        )
        meta_features = []
        for index in meta_indices:
            heads = base_model.predict_heads_sample(samples[index].policy_sample)
            meta_features.append(
                build_meta_rank_features(
                    policy_probability=heads.policy_probability,
                    value=heads.value,
                    delta_vs_baseline=heads.delta_vs_baseline,
                    legacy_score=heads.action_score,
                )
            )
        meta_estimator = _fit_regressor(
            np.vstack(meta_features),
            np.concatenate([relative_targets[index] for index in meta_indices]),
            np.concatenate([action_weights[index] for index in meta_indices]),
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            seed=seed + 1_000 + target_fold,
        )
        for index in target_indices:
            heads = base_model.predict_heads_sample(samples[index].policy_sample)
            features = build_meta_rank_features(
                policy_probability=heads.policy_probability,
                value=heads.value,
                delta_vs_baseline=heads.delta_vs_baseline,
                legacy_score=heads.action_score,
            )
            predictions[index] = np.asarray(
                meta_estimator.predict(features), dtype=np.float64
            ).reshape(-1)
            coverage[index] += 1
        lineage_audits.append(
            {
                "target_fold": target_fold,
                "meta_fit_fold": meta_fold,
                "base_fit_folds": base_folds,
                "base_fit_samples": len(base_indices),
                "meta_fit_samples": len(meta_indices),
                "target_samples": len(target_indices),
                "target__predictor_lineage_overlap": target_lineage_overlap,
                "base_fit__meta_fit_overlap": base_meta_overlap,
                "target_identity_used_by_predictor_lineage": False,
            }
        )
    if np.any(coverage != 1) or any(value is None for value in predictions):
        raise AssertionError(
            "strict nested rank coverage must predict every sample exactly once"
        )
    return (
        [value for value in predictions if value is not None],
        {
            "schema": "hu_m4_three_way_rank_lineage_v1",
            "method": "disjoint_base_fit_meta_fit_target_rotate_3fold",
            "prediction_counts": sorted(set(coverage.tolist())),
            "each_sample_predicted_exactly_once": True,
            "identity_leakage_count": 0,
            "fold_audits": lineage_audits,
        },
    )


def _paired_delta_risk_training_arrays(
    samples: Sequence[PreparedTeacherSample],
    *,
    paired_se_floor: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return explicit pair features and M4.3 precision-weighted targets."""

    if not math.isfinite(paired_se_floor) or paired_se_floor <= 0.0:
        raise ValueError("paired_se_floor must be finite and positive")
    feature_blocks: list[np.ndarray] = []
    delta_blocks: list[np.ndarray] = []
    soft_positive_blocks: list[np.ndarray] = []
    downside_p95_blocks: list[np.ndarray] = []
    downside_p99_blocks: list[np.ndarray] = []
    downside_max_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    for sample in samples:
        if (
            sample.teacher_delta_se_vs_baseline is None
            or sample.teacher_paired_delta_mean is None
            or sample.downside_loss_p95 is None
            or sample.downside_loss_p99 is None
            or sample.downside_loss_max is None
        ):
            raise ValueError(
                "baseline_paired_delta_risk_ensemble_v3 requires paired_delta_mean, "
                "paired_delta_se, downside_loss_p95, downside_loss_p99, and "
                "downside_loss_max targets on every action"
            )
        paired_features = build_paired_action_features(
            sample.policy_sample, baseline_index=sample.baseline_index
        )
        delta = np.asarray(sample.teacher_paired_delta_mean, dtype=np.float64).copy()
        paired_se = np.asarray(
            sample.teacher_delta_se_vs_baseline, dtype=np.float64
        ).copy()
        p95 = np.asarray(sample.downside_loss_p95, dtype=np.float64).copy()
        p99 = np.asarray(sample.downside_loss_p99, dtype=np.float64).copy()
        maximum = np.asarray(sample.downside_loss_max, dtype=np.float64).copy()
        baseline = sample.baseline_index
        if any(
            values.shape != sample.teacher_scores.shape
            for values in (delta, paired_se, p95, p99, maximum)
        ):
            raise ValueError("M4.3 paired target shape mismatch")
        if any(
            not np.isfinite(values).all()
            for values in (delta, paired_se, p95, p99, maximum)
        ) or np.any(paired_se < 0.0):
            raise ValueError("M4.3 paired targets must be finite and SE non-negative")
        if any(
            value != 0.0
            for value in (
                delta[baseline],
                paired_se[baseline],
                p95[baseline],
                p99[baseline],
                maximum[baseline],
            )
        ):
            raise ValueError("M4.3 baseline paired targets must be exactly zero")

        scale = np.maximum(paired_se, paired_se_floor)
        standardized = np.clip(delta / scale, -40.0, 40.0)
        soft_positive = 1.0 / (1.0 + np.exp(-standardized))
        soft_positive[baseline] = 0.5
        precision = 1.0 / np.square(scale)
        # Every information set retains equal total weight; paired precision
        # only reallocates weight among its legal actions.
        precision /= float(np.sum(precision))

        feature_blocks.append(paired_features)
        delta_blocks.append(delta)
        soft_positive_blocks.append(soft_positive)
        downside_p95_blocks.append(p95)
        downside_p99_blocks.append(p99)
        downside_max_blocks.append(maximum)
        weight_blocks.append(precision)
    return (
        np.vstack(feature_blocks).astype(np.float32, copy=False),
        np.concatenate(delta_blocks),
        np.concatenate(soft_positive_blocks),
        np.concatenate(downside_p95_blocks),
        np.concatenate(downside_p99_blocks),
        np.concatenate(downside_max_blocks),
        np.concatenate(weight_blocks),
    )


def _m43_sample_identity_sha256(
    samples: Sequence[PreparedTeacherSample],
) -> str:
    return _digest_strings(_sample_membership_hash(sample) for sample in samples)


def _m43_fold_assignment_sha256(
    samples: Sequence[PreparedTeacherSample], fold_ids: Sequence[int]
) -> str:
    if len(samples) != len(fold_ids):
        raise ValueError("M4.3 fold assignment length mismatch")
    payload = [
        {
            "sample_sha256": _sample_membership_hash(sample),
            "fold": int(fold),
        }
        for sample, fold in zip(samples, fold_ids, strict=True)
    ]
    return canonical_manifest_sha256({"assignments": payload})


def build_m43_fold_training_plan(
    samples: Sequence[PreparedTeacherSample],
    *,
    cross_fit_folds: int,
    seed: int,
) -> M43FoldTrainingPlan:
    """Build the exact outer+inner estimator job grid without fitting it."""

    if cross_fit_folds < 2:
        raise ValueError("M4.3 fold plan requires at least two folds")
    ordered = tuple(sorted(samples, key=_sample_membership_hash))
    outer_fold_ids_array, outer_report = _assign_cross_fit_folds(
        ordered, folds=cross_fit_folds, seed=seed + 10_000
    )
    outer_fold_ids = tuple(int(value) for value in outer_fold_ids_array)
    outer_assignment_sha = _m43_fold_assignment_sha256(ordered, outer_fold_ids)
    jobs: list[M43FoldJobDefinition] = []
    inner_reports: list[Mapping[str, Any]] = []
    for outer_fold in range(cross_fit_folds):
        training_indices = [
            index
            for index, assigned in enumerate(outer_fold_ids)
            if assigned != outer_fold
        ]
        validation_indices = [
            index
            for index, assigned in enumerate(outer_fold_ids)
            if assigned == outer_fold
        ]
        fit_samples = tuple(ordered[index] for index in training_indices)
        validation_samples = tuple(ordered[index] for index in validation_indices)
        if not fit_samples or not validation_samples:
            raise ValueError(f"M4.3 outer fold {outer_fold} is empty")
        jobs.append(
            M43FoldJobDefinition(
                spec=M43FoldJobSpec(
                    job_index=outer_fold * (cross_fit_folds + 1),
                    kind="outer_runtime",
                    outer_fold=outer_fold,
                    inner_fold=None,
                    estimator_fold_index=outer_fold,
                    estimator_seed=seed + 1_000 * (outer_fold + 1),
                    fit_samples=len(fit_samples),
                    fit_identity_sha256=_m43_sample_identity_sha256(fit_samples),
                    outer_validation_samples=len(validation_samples),
                    outer_validation_identity_sha256=(
                        _m43_sample_identity_sha256(validation_samples)
                    ),
                    inner_validation_samples=0,
                    inner_validation_identity_sha256=None,
                    outer_assignment_sha256=outer_assignment_sha,
                    inner_assignment_sha256=None,
                ),
                fit_samples=fit_samples,
            )
        )
        try:
            inner_fold_ids_array, inner_report = _assign_cross_fit_folds(
                fit_samples,
                folds=cross_fit_folds,
                seed=seed + 50_000 * (outer_fold + 1),
            )
        except ValueError as error:
            raise ValueError(
                f"M4.3 outer fold {outer_fold} requires at least "
                f"{cross_fit_folds} identity groups for its OOF safety inner ensemble"
            ) from error
        inner_fold_ids = tuple(int(value) for value in inner_fold_ids_array)
        inner_assignment_sha = _m43_fold_assignment_sha256(
            fit_samples, inner_fold_ids
        )
        inner_reports.append(inner_report)
        for inner_fold in range(cross_fit_folds):
            inner_fit = tuple(
                sample
                for index, sample in enumerate(fit_samples)
                if inner_fold_ids[index] != inner_fold
            )
            inner_validation = tuple(
                sample
                for index, sample in enumerate(fit_samples)
                if inner_fold_ids[index] == inner_fold
            )
            if not inner_fit or not inner_validation:
                raise ValueError(
                    f"M4.3 inner fold {outer_fold}/{inner_fold} is empty"
                )
            jobs.append(
                M43FoldJobDefinition(
                    spec=M43FoldJobSpec(
                        job_index=(
                            outer_fold * (cross_fit_folds + 1) + 1 + inner_fold
                        ),
                        kind="inner_oof_safety",
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                        estimator_fold_index=inner_fold,
                        estimator_seed=(
                            seed
                            + 100_000 * (outer_fold + 1)
                            + 1_000 * (inner_fold + 1)
                        ),
                        fit_samples=len(inner_fit),
                        fit_identity_sha256=_m43_sample_identity_sha256(inner_fit),
                        outer_validation_samples=len(validation_samples),
                        outer_validation_identity_sha256=(
                            _m43_sample_identity_sha256(validation_samples)
                        ),
                        inner_validation_samples=len(inner_validation),
                        inner_validation_identity_sha256=(
                            _m43_sample_identity_sha256(inner_validation)
                        ),
                        outer_assignment_sha256=outer_assignment_sha,
                        inner_assignment_sha256=inner_assignment_sha,
                    ),
                    fit_samples=inner_fit,
                )
            )
    expected_jobs = cross_fit_folds * (cross_fit_folds + 1)
    if len(jobs) != expected_jobs or [job.spec.job_index for job in jobs] != list(
        range(expected_jobs)
    ):
        raise AssertionError("M4.3 fold job grid is not exact and contiguous")
    return M43FoldTrainingPlan(
        ordered_samples=ordered,
        outer_fold_ids=outer_fold_ids,
        outer_fold_report=outer_report,
        inner_fold_reports=tuple(inner_reports),
        jobs=tuple(jobs),
    )


def _fit_paired_delta_risk_fold(
    samples: Sequence[PreparedTeacherSample],
    *,
    fold_index: int,
    paired_se_floor: float,
    huber_alpha: float,
    downside_quantile: float,
    iterations: int,
    max_leaf_nodes: int,
    learning_rate: float,
    seed: int,
) -> PairedDeltaRiskFoldEstimator:
    if not 0.5 <= huber_alpha < 1.0:
        raise ValueError("paired_huber_alpha must be in [0.5, 1.0)")
    if not 0.5 <= downside_quantile < 1.0:
        raise ValueError("downside_quantile must be in [0.5, 1.0)")
    (
        features,
        delta,
        soft_positive,
        downside_p95,
        downside_p99,
        downside_max,
        weights,
    ) = _paired_delta_risk_training_arrays(
        samples, paired_se_floor=paired_se_floor
    )
    return PairedDeltaRiskFoldEstimator(
        delta_estimator=_fit_gradient_regressor(
            features,
            delta,
            weights,
            loss="huber",
            alpha=huber_alpha,
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            seed=seed,
        ),
        positive_gain_estimator=_fit_gradient_regressor(
            features,
            soft_positive,
            weights,
            loss="huber",
            alpha=huber_alpha,
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            seed=seed + 1,
        ),
        downside_p95_estimator=_fit_gradient_regressor(
            features,
            downside_p95,
            weights,
            loss="quantile",
            alpha=downside_quantile,
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            seed=seed + 2,
        ),
        downside_p99_estimator=_fit_gradient_regressor(
            features,
            downside_p99,
            weights,
            loss="quantile",
            alpha=downside_quantile,
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            seed=seed + 3,
        ),
        downside_max_estimator=_fit_gradient_regressor(
            features,
            downside_max,
            weights,
            loss="quantile",
            alpha=downside_quantile,
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            seed=seed + 4,
        ),
        paired_feature_dim=features.shape[1],
        fold_index=fold_index,
    )


def _paired_runtime_model(
    samples: Sequence[PreparedTeacherSample],
    folds: Sequence[PairedDeltaRiskFoldEstimator],
    *,
    model_id: str,
    positive_gain_score_weight: float,
    downside_risk_score_weight: float,
    ensemble_disagreement_score_weight: float,
) -> HuM4JointActionModel:
    raw = np.vstack(
        [sample_to_matrix(sample.policy_sample)[0] for sample in samples]
    ).astype(np.float32, copy=False)
    zero = DummyRegressor(strategy="constant", constant=0.0).fit(
        raw, np.zeros(raw.shape[0], dtype=np.float64)
    )
    return HuM4JointActionModel(
        policy_estimator=ConstantProbabilityEstimator(0.5),
        value_estimator=zero,
        delta_estimator=zero,
        uncertainty_estimator=zero,
        model_id=model_id,
        action_score_mode=PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
        paired_fold_estimators=tuple(folds),
        positive_gain_score_weight=positive_gain_score_weight,
        downside_risk_score_weight=downside_risk_score_weight,
        ensemble_disagreement_score_weight=ensemble_disagreement_score_weight,
    )


def _fit_paired_delta_risk_ensemble_cross_fitted(
    samples: Sequence[PreparedTeacherSample],
    *,
    cross_fit_folds: int,
    model_id: str,
    minimum_safe_teacher_gain: float,
    paired_se_floor: float,
    paired_huber_alpha: float,
    downside_quantile: float,
    positive_gain_score_weight: float,
    downside_risk_score_weight: float,
    ensemble_disagreement_score_weight: float,
    iterations: int,
    max_leaf_nodes: int,
    learning_rate: float,
    seed: int,
    fold_estimator_provider: M43FoldEstimatorProvider | None = None,
) -> CrossFitTrainingResult:
    """Fit OOF paired heads and preserve those same folds for runtime."""

    fold_plan = build_m43_fold_training_plan(
        samples, cross_fit_folds=cross_fit_folds, seed=seed
    )
    ordered = fold_plan.ordered_samples
    fold_ids = fold_plan.outer_fold_ids
    fold_report = fold_plan.outer_fold_report
    jobs_by_index = {job.spec.job_index: job for job in fold_plan.jobs}

    def obtain_estimator(job: M43FoldJobDefinition) -> PairedDeltaRiskFoldEstimator:
        if fold_estimator_provider is None:
            estimator = _fit_paired_delta_risk_fold(
                job.fit_samples,
                fold_index=job.spec.estimator_fold_index,
                paired_se_floor=paired_se_floor,
                huber_alpha=paired_huber_alpha,
                downside_quantile=downside_quantile,
                iterations=iterations,
                max_leaf_nodes=max_leaf_nodes,
                learning_rate=learning_rate,
                seed=job.spec.estimator_seed,
            )
        else:
            estimator = fold_estimator_provider(job.spec, job.fit_samples)
        if not isinstance(estimator, PairedDeltaRiskFoldEstimator):
            raise TypeError("M4.3 fold provider returned the wrong estimator type")
        estimator.__post_init__()
        if estimator.fold_index != job.spec.estimator_fold_index:
            raise ValueError("M4.3 fold estimator index disagrees with its job spec")
        return estimator

    fold_estimators: list[PairedDeltaRiskFoldEstimator] = []
    oof_heads: list[JointHeadPredictions | None] = [None] * len(ordered)
    coverage = np.zeros(len(ordered), dtype=np.int16)
    lineage_audits: list[dict[str, Any]] = []
    for fold in range(cross_fit_folds):
        training_indices = [
            index for index, assigned in enumerate(fold_ids) if assigned != fold
        ]
        validation_indices = [
            index for index, assigned in enumerate(fold_ids) if assigned == fold
        ]
        fit_samples = [ordered[index] for index in training_indices]
        validation_samples = [ordered[index] for index in validation_indices]
        overlap = _safety_subset_overlap(fit_samples, validation_samples)
        if any(overlap.values()):
            raise AssertionError(f"M4.3 fold {fold} identity leak: {overlap}")
        outer_job = jobs_by_index[fold * (cross_fit_folds + 1)]
        estimator = obtain_estimator(outer_job)
        fold_estimators.append(estimator)
        # Safety OOF rows must see the same K-fold aggregation semantics as
        # runtime/calibration rows.  Build an identity-clean inner K-fold
        # ensemble wholly inside the outer-training subset; a single outer
        # estimator would create a material calibrator feature shift.
        inner_fold_report = fold_plan.inner_fold_reports[fold]
        inner_estimators: list[PairedDeltaRiskFoldEstimator] = []
        for inner_fold in range(cross_fit_folds):
            inner_job = jobs_by_index[
                fold * (cross_fit_folds + 1) + 1 + inner_fold
            ]
            inner_estimators.append(
                obtain_estimator(inner_job)
            )
        fold_model = _paired_runtime_model(
            fit_samples,
            inner_estimators,
            model_id=f"{model_id}:paired-oof-inner-ensemble-{fold}",
            positive_gain_score_weight=positive_gain_score_weight,
            downside_risk_score_weight=downside_risk_score_weight,
            ensemble_disagreement_score_weight=ensemble_disagreement_score_weight,
        )
        for index in validation_indices:
            sample = ordered[index]
            heads = fold_model.predict_heads_sample(
                sample.policy_sample, baseline_index=sample.baseline_index
            )
            if heads.action_score[sample.baseline_index] != 0.0:
                raise AssertionError("M4.3 OOF baseline score is not exactly zero")
            oof_heads[index] = heads
            coverage[index] += 1
        lineage_audits.append(
            {
                "fold": fold,
                "training_samples": len(training_indices),
                "validation_samples": len(validation_indices),
                "train__validation_identity_overlap": overlap,
                "validation_identity_excluded_from_all_paired_heads": True,
                "oof_safety_inner_ensemble": {
                    "fold_count": len(inner_estimators),
                    "aggregation": "mean_plus_cross_fold_disagreement",
                    "matches_runtime_fold_count": (
                        len(inner_estimators) == cross_fit_folds
                    ),
                    "fold_assignment": inner_fold_report,
                },
                "identity_leakage_count": 0,
            }
        )
    if np.any(coverage != 1) or any(heads is None for heads in oof_heads):
        raise AssertionError("M4.3 OOF coverage must predict every sample once")

    runtime_model = _paired_runtime_model(
        ordered,
        fold_estimators,
        model_id=model_id,
        positive_gain_score_weight=positive_gain_score_weight,
        downside_risk_score_weight=downside_risk_score_weight,
        ensemble_disagreement_score_weight=ensemble_disagreement_score_weight,
    )
    safety_features: list[np.ndarray] = []
    safety_labels: list[int] = []
    safety_rows: list[Mapping[str, Any]] = []
    clean_heads = [heads for heads in oof_heads if heads is not None]
    for index, sample in enumerate(ordered):
        heads = clean_heads[index]
        candidate = canonical_action_argmax(sample.policy_sample, heads.action_score)
        baseline = sample.baseline_index
        if candidate == baseline:
            continue
        teacher_delta = float(
            sample.teacher_paired_delta_mean[candidate]  # type: ignore[index]
        )
        safety_features.append(
            build_joint_safety_features(
                heads,
                candidate_index=candidate,
                baseline_index=baseline,
                seat=sample.seat,
            )
        )
        safety_labels.append(int(teacher_delta > minimum_safe_teacher_gain))
        safety_rows.append(
            {
                "teacher_delta": teacher_delta,
                "downside_loss_p95": float(
                    sample.downside_loss_p95[candidate]  # type: ignore[index]
                ),
                "downside_loss_p99": float(
                    sample.downside_loss_p99[candidate]  # type: ignore[index]
                ),
                "downside_loss_max": float(
                    sample.downside_loss_max[candidate]  # type: ignore[index]
                ),
                "seat": sample.seat,
                "candidate_index": candidate,
                "baseline_index": baseline,
                "source": "train_oof_paired_fold",
                "fold": int(fold_ids[index]),
                "row_hash": _sample_membership_hash(sample),
            }
        )
    seed_values, fingerprints, row_hashes = _safety_subset_identity(ordered)
    safety_examples = CrossFitSafetyExamples(
        features=(
            np.vstack(safety_features).astype(np.float32, copy=False)
            if safety_features
            else np.empty((0, 18), dtype=np.float32)
        ),
        labels=np.asarray(safety_labels, dtype=np.int8),
        rows=tuple(safety_rows),
        seed_values=frozenset(seed_values),
        observation_fingerprints=frozenset(fingerprints),
        row_hashes=frozenset(row_hashes),
    )
    report = {
        "schema": "hu_m4_identity_group_paired_delta_cross_fit_v3",
        "status": "pass",
        "folds": int(cross_fit_folds),
        "action_score_mode": PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
        "paired_action_feature_schema": HU_M4_PAIRED_ACTION_FEATURE_SCHEMA,
        "fold_assignment": fold_report,
        "oof_coverage": {
            "samples": len(ordered),
            "paired_head_prediction_counts": sorted(set(coverage.tolist())),
            "each_sample_predicted_exactly_once": True,
        },
        "predictor_lineage": {
            "schema": "hu_m4_paired_outer_fold_predictor_lineage_v1",
            "identity_leakage_count": 0,
            "all_outer_validation_identities_excluded": True,
            "fold_audits": lineage_audits,
        },
        "runtime_ensemble": {
            "source": "stored_crossfit_fold_estimators",
            "fold_count": len(fold_estimators),
            "full_refit_used_at_runtime": False,
            "oof_to_full_refit_distribution_shift": False,
            "oof_safety_aggregation_semantics_match": True,
            "oof_safety_fold_count": int(cross_fit_folds),
            "baseline_action_score_exact_zero": True,
        },
        "objectives": {
            "paired_delta_mean": {
                "loss": "huber",
                "sample_weight": "paired_delta_se_inverse_variance_within_state",
                "paired_delta_se_floor": float(paired_se_floor),
                "huber_alpha": float(paired_huber_alpha),
            },
            "soft_pairwise_positive_gain": {
                "target": "sigmoid(paired_delta_mean/max(paired_delta_se,floor))",
                "loss": "huber",
            },
            "downside_tail": {
                "downside_loss_p95": "max(0,-paired_delta_p05)",
                "downside_loss_p99": "max(0,-paired_delta_p01)",
                "downside_loss_max": "max(0,-paired_delta_min)",
                "loss": "quantile",
                "quantile": float(downside_quantile),
            },
        },
        "runtime_score": {
            "positive_gain_score_weight": float(positive_gain_score_weight),
            "downside_risk_score_weight": float(downside_risk_score_weight),
            "ensemble_disagreement_score_weight": float(
                ensemble_disagreement_score_weight
            ),
            "teacher_value_runtime_input": False,
            "teacher_lcb_runtime_gate": False,
        },
        "safety_examples": {
            "source": "train_oof_only",
            "rows": int(safety_examples.labels.size),
            "used_for_threshold_lock": False,
            "used_for_locked_holdout": False,
            "identity_manifest": _safety_examples_identity_manifest(
                safety_examples
            ),
        },
        "fit_roles": {
            "train_outer_fit": "fit_stored_runtime_paired_fold_heads",
            "train_outer_validation": "oof_safety_feature_source_only",
            "calibration_safety_fit": "not_used_in_cross_fit_stage",
            "calibration_threshold_lock": "not_used_in_cross_fit_stage",
            "locked_holdout": "not_used_in_cross_fit_stage",
        },
        "final_model_fit_source": "stored_crossfit_fold_ensemble_no_full_refit",
        "locked_holdout_used": False,
        "threshold_lock_used": False,
    }
    return CrossFitTrainingResult(
        model=runtime_model,
        safety_examples=safety_examples,
        report=report,
    )


def fit_joint_model_cross_fitted(
    samples: Sequence[PreparedTeacherSample],
    *,
    cross_fit_folds: int = 5,
    action_score_mode: str = NEGATIVE_REGRET_ACTION_SCORE_MODE,
    model_id: str = "hu-m4-t1-joint-v1",
    near_best_margin: float = 0.5,
    minimum_safe_teacher_gain: float = 0.0,
    iterations: int = 150,
    max_leaf_nodes: int = 31,
    learning_rate: float = 0.05,
    l2_regularization: float = 1.0,
    seed: int = 2026071801,
    paired_se_floor: float = 0.50,
    paired_huber_alpha: float = 0.90,
    downside_quantile: float = 0.90,
    positive_gain_score_weight: float = DEFAULT_POSITIVE_GAIN_SCORE_WEIGHT,
    downside_risk_score_weight: float = DEFAULT_DOWNSIDE_RISK_SCORE_WEIGHT,
    ensemble_disagreement_score_weight: float = (
        DEFAULT_ENSEMBLE_DISAGREEMENT_SCORE_WEIGHT
    ),
    m43_fold_estimator_provider: M43FoldEstimatorProvider | None = None,
) -> CrossFitTrainingResult:
    """Fit final heads plus strictly nested OOF risk/safety predictions.

    Every outer validation identity is excluded from the entire predictor
    lineage used for its safety example.  Ranker residual labels inside an
    outer training subset use a separate three-way base/meta/target rotation,
    preventing the subtle feature-lineage leak of ordinary stacked OOF fits.
    """

    if not samples:
        raise ValueError("cross fitting requires samples")
    if action_score_mode not in ACTION_SCORE_MODES:
        raise ValueError(f"unsupported action_score_mode: {action_score_mode!r}")
    if cross_fit_folds < 2:
        raise ValueError("cross_fit_folds must be at least 2 for cross fitting")
    if action_score_mode == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE:
        return _fit_paired_delta_risk_ensemble_cross_fitted(
            samples,
            cross_fit_folds=cross_fit_folds,
            model_id=model_id,
            minimum_safe_teacher_gain=minimum_safe_teacher_gain,
            paired_se_floor=paired_se_floor,
            paired_huber_alpha=paired_huber_alpha,
            downside_quantile=downside_quantile,
            positive_gain_score_weight=positive_gain_score_weight,
            downside_risk_score_weight=downside_risk_score_weight,
            ensemble_disagreement_score_weight=(
                ensemble_disagreement_score_weight
            ),
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            seed=seed,
            fold_estimator_provider=m43_fold_estimator_provider,
        )

    ordered = sorted(samples, key=_sample_membership_hash)
    fold_ids, fold_report = _assign_cross_fit_folds(
        ordered, folds=cross_fit_folds, seed=seed + 10_000
    )
    required_inner_groups = (
        3 if action_score_mode == NEGATIVE_REGRET_ACTION_SCORE_MODE else 2
    )
    insufficient_outer_folds = [
        {
            "fold": int(audit["fold"]),
            "outer_training_identity_groups": int(
                fold_report["component_count"] - audit["validation_components"]
            ),
        }
        for audit in fold_report["fold_audits"]
        if fold_report["component_count"] - audit["validation_components"]
        < required_inner_groups
    ]
    if insufficient_outer_folds:
        raise ValueError(
            "strict nested cross fitting requires at least "
            f"{required_inner_groups} identity groups in every outer training "
            f"subset; insufficient outer folds: {insufficient_outer_folds}"
        )
    oof_base_heads: list[JointHeadPredictions | None] = [None] * len(ordered)
    oof_action_scores: list[np.ndarray | None] = [None] * len(ordered)
    oof_uncertainty: list[np.ndarray | None] = [None] * len(ordered)
    base_coverage = np.zeros(len(ordered), dtype=np.int16)
    meta_coverage = np.zeros(len(ordered), dtype=np.int16)
    uncertainty_coverage = np.zeros(len(ordered), dtype=np.int16)
    outer_lineage_audits: list[dict[str, Any]] = []
    for fold in range(cross_fit_folds):
        training_indices = [
            index for index in range(len(ordered)) if fold_ids[index] != fold
        ]
        validation_indices = [
            index for index in range(len(ordered)) if fold_ids[index] == fold
        ]
        fit_samples = [ordered[index] for index in training_indices]
        validation_samples = [ordered[index] for index in validation_indices]
        outer_overlap = _safety_subset_overlap(fit_samples, validation_samples)
        if any(outer_overlap.values()):
            raise AssertionError(f"outer fold {fold} identity leak: {outer_overlap}")

        outer_base_model = fit_joint_model(
            fit_samples,
            model_id=f"{model_id}:oof-base-fold-{fold}",
            near_best_margin=near_best_margin,
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            seed=seed + 100 * (fold + 1),
        )
        outer_base_validation_heads: dict[int, JointHeadPredictions] = {}
        for index in validation_indices:
            heads = outer_base_model.predict_heads_sample(
                ordered[index].policy_sample
            )
            clean_heads = replace(
                heads,
                predicted_absolute_residual=np.zeros_like(
                    heads.predicted_absolute_residual
                ),
            )
            oof_base_heads[index] = clean_heads
            outer_base_validation_heads[index] = clean_heads
            base_coverage[index] += 1
        inner_folds = required_inner_groups
        try:
            inner_heads, inner_fold_ids, inner_report, inner_coverage = (
                _fit_identity_oof_base_heads(
                    fit_samples,
                    folds=inner_folds,
                    model_id=f"{model_id}:outer-{fold}:inner-base",
                    near_best_margin=near_best_margin,
                    iterations=iterations,
                    max_leaf_nodes=max_leaf_nodes,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                    seed=seed + 10_000 * (fold + 1),
                )
            )
        except ValueError as error:
            raise ValueError(
                f"outer fold {fold} needs at least {inner_folds} identity groups "
                "for strict nested cross fitting"
            ) from error
        inner_meta_features = [
            build_meta_rank_features(
                policy_probability=heads.policy_probability,
                value=heads.value,
                delta_vs_baseline=heads.delta_vs_baseline,
                legacy_score=heads.action_score,
            )
            for heads in inner_heads
        ]
        inner_relative_targets = [
            sample.teacher_scores - float(np.max(sample.teacher_scores))
            for sample in fit_samples
        ]
        inner_weights = [
            np.full(len(sample.teacher_scores), 1.0 / len(sample.teacher_scores))
            for sample in fit_samples
        ]
        if action_score_mode == NEGATIVE_REGRET_ACTION_SCORE_MODE:
            strict_train_scores, strict_rank_report = _fit_three_way_oof_rank_scores(
                fit_samples,
                fold_ids=inner_fold_ids,
                model_id=f"{model_id}:outer-{fold}:residual-rank",
                near_best_margin=near_best_margin,
                iterations=iterations,
                max_leaf_nodes=max_leaf_nodes,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
                seed=seed + 20_000 * (fold + 1),
            )
            outer_meta = _fit_regressor(
                np.vstack(inner_meta_features),
                np.concatenate(inner_relative_targets),
                np.concatenate(inner_weights),
                iterations=iterations,
                max_leaf_nodes=max_leaf_nodes,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
                seed=seed + 30_000 + fold,
            )
            for index in validation_indices:
                heads = outer_base_validation_heads[index]
                features = build_meta_rank_features(
                    policy_probability=heads.policy_probability,
                    value=heads.value,
                    delta_vs_baseline=heads.delta_vs_baseline,
                    legacy_score=heads.action_score,
                )
                oof_action_scores[index] = np.asarray(
                    outer_meta.predict(features), dtype=np.float64
                ).reshape(-1)
                meta_coverage[index] += 1
            inner_score_targets = inner_relative_targets
        else:
            strict_train_scores = [heads.action_score for heads in inner_heads]
            strict_rank_report = {
                "schema": "hu_m4_legacy_base_residual_lineage_v1",
                "method": "identity_group_inner_oof_base_prediction",
                "prediction_counts": sorted(set(inner_coverage.tolist())),
                "each_sample_predicted_exactly_once": True,
                "identity_leakage_count": 0,
            }
            for index in validation_indices:
                oof_action_scores[index] = outer_base_validation_heads[index].action_score
                meta_coverage[index] += 1
            inner_score_targets = [sample.teacher_scores for sample in fit_samples]

        inner_uncertainty_targets = [
            _oof_uncertainty_target(
                sample,
                target_scores=inner_score_targets[index],
                predicted_scores=strict_train_scores[index],
            )
            for index, sample in enumerate(fit_samples)
        ]
        inner_raw_features = [
            sample_to_matrix(sample.policy_sample)[0] for sample in fit_samples
        ]
        outer_uncertainty = _fit_regressor(
            np.vstack(inner_raw_features),
            np.concatenate(inner_uncertainty_targets),
            np.concatenate(inner_weights),
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            seed=seed + 40_000 + fold,
        )
        for index in validation_indices:
            raw = sample_to_matrix(ordered[index].policy_sample)[0]
            oof_uncertainty[index] = np.maximum(
                0.0,
                np.asarray(outer_uncertainty.predict(raw), dtype=np.float64).reshape(-1),
            )
            uncertainty_coverage[index] += 1
        outer_lineage_audits.append(
            {
                "outer_fold": fold,
                "training_samples": len(training_indices),
                "validation_samples": len(validation_indices),
                "outer_train__validation_identity_overlap": outer_overlap,
                "inner_base_fold_assignment": inner_report,
                "strict_residual_prediction_lineage": strict_rank_report,
                "outer_validation_excluded_from_base_fit": True,
                "outer_validation_excluded_from_meta_fit": True,
                "outer_validation_excluded_from_uncertainty_fit": True,
                "outer_validation_excluded_from_residual_target_lineage": True,
                "identity_leakage_count": 0,
            }
        )

    if np.any(base_coverage != 1) or any(heads is None for heads in oof_base_heads):
        raise AssertionError("base OOF coverage must predict every sample exactly once")
    if np.any(meta_coverage != 1) or any(
        values is None for values in oof_action_scores
    ):
        raise AssertionError("action-score OOF coverage must predict every sample once")
    if np.any(uncertainty_coverage != 1) or any(
        values is None for values in oof_uncertainty
    ):
        raise AssertionError(
            "uncertainty OOF coverage must predict every sample exactly once"
        )

    base_heads = [heads for heads in oof_base_heads if heads is not None]
    action_scores = [values for values in oof_action_scores if values is not None]
    raw_features = [sample_to_matrix(sample.policy_sample)[0] for sample in ordered]
    relative_targets = [
        sample.teacher_scores - float(np.max(sample.teacher_scores))
        for sample in ordered
    ]
    action_weights = [
        np.full(len(sample.teacher_scores), 1.0 / len(sample.teacher_scores))
        for sample in ordered
    ]
    score_targets = (
        relative_targets
        if action_score_mode == NEGATIVE_REGRET_ACTION_SCORE_MODE
        else [sample.teacher_scores for sample in ordered]
    )
    uncertainty_targets = [
        _oof_uncertainty_target(
            sample,
            target_scores=score_targets[index],
            predicted_scores=action_scores[index],
        )
        for index, sample in enumerate(ordered)
    ]

    final_meta_rank_estimator: Any | None = None
    if action_score_mode == NEGATIVE_REGRET_ACTION_SCORE_MODE:
        final_meta_features = [
            build_meta_rank_features(
                policy_probability=heads.policy_probability,
                value=heads.value,
                delta_vs_baseline=heads.delta_vs_baseline,
                legacy_score=heads.action_score,
            )
            for heads in base_heads
        ]
        final_meta_rank_estimator = _fit_regressor(
            np.vstack(final_meta_features),
            np.concatenate(relative_targets),
            np.concatenate(action_weights),
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            seed=seed + 50_000,
        )

    final_uncertainty = _fit_regressor(
        np.vstack(raw_features),
        np.concatenate(uncertainty_targets),
        np.concatenate(action_weights),
        iterations=iterations,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        seed=seed + 2_500,
    )
    full_model = fit_joint_model(
        ordered,
        model_id=model_id,
        near_best_margin=near_best_margin,
        iterations=iterations,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        seed=seed,
    )
    final_model = replace(
        full_model,
        uncertainty_estimator=final_uncertainty,
        meta_rank_estimator=final_meta_rank_estimator,
        action_score_mode=action_score_mode,
    )

    safety_features: list[np.ndarray] = []
    safety_labels: list[int] = []
    safety_rows: list[Mapping[str, Any]] = []
    for index, sample in enumerate(ordered):
        heads = replace(
            base_heads[index],
            action_score=np.asarray(action_scores[index], dtype=np.float64),
            predicted_absolute_residual=np.asarray(
                oof_uncertainty[index], dtype=np.float64
            ),
        )
        candidate = canonical_action_argmax(sample.policy_sample, heads.action_score)
        baseline = sample.baseline_index
        if candidate == baseline:
            continue
        teacher_delta = float(
            sample.teacher_scores[candidate] - sample.teacher_scores[baseline]
        )
        safety_features.append(
            build_joint_safety_features(
                heads,
                candidate_index=candidate,
                baseline_index=baseline,
                seat=sample.seat,
            )
        )
        safety_labels.append(int(teacher_delta > minimum_safe_teacher_gain))
        safety_rows.append(
            {
                "teacher_delta": teacher_delta,
                "seat": sample.seat,
                "candidate_index": candidate,
                "baseline_index": baseline,
                "source": "train_oof",
                "fold": int(fold_ids[index]),
                "row_hash": _sample_membership_hash(sample),
            }
        )

    seed_values, fingerprints, row_hashes = _safety_subset_identity(ordered)
    safety_examples = CrossFitSafetyExamples(
        features=(
            np.vstack(safety_features).astype(np.float32, copy=False)
            if safety_features
            else np.empty((0, 18), dtype=np.float32)
        ),
        labels=np.asarray(safety_labels, dtype=np.int8),
        rows=tuple(safety_rows),
        seed_values=frozenset(seed_values),
        observation_fingerprints=frozenset(fingerprints),
        row_hashes=frozenset(row_hashes),
    )
    paired_delta_rows = sum(
        sample.teacher_delta_se_vs_baseline is not None for sample in ordered
    )
    report = {
        "schema": "hu_m4_identity_group_nested_cross_fit_v2",
        "status": "pass",
        "folds": int(cross_fit_folds),
        "action_score_mode": action_score_mode,
        "meta_rank_feature_schema": (
            HU_M4_META_RANK_FEATURE_SCHEMA
            if action_score_mode == NEGATIVE_REGRET_ACTION_SCORE_MODE
            else None
        ),
        "rank_target": (
            "teacher_score_minus_state_max_negative_regret"
            if action_score_mode == NEGATIVE_REGRET_ACTION_SCORE_MODE
            else "legacy_absolute_teacher_score"
        ),
        "fold_assignment": fold_report,
        "oof_coverage": {
            "samples": len(ordered),
            "base_head_prediction_counts": sorted(set(base_coverage.tolist())),
            "meta_rank_prediction_counts": (
                sorted(set(meta_coverage.tolist()))
            ),
            "uncertainty_prediction_counts": sorted(
                set(uncertainty_coverage.tolist())
            ),
            "each_sample_predicted_exactly_once": True,
        },
        "predictor_lineage": {
            "schema": "hu_m4_outer_fold_predictor_lineage_v1",
            "contract": (
                "outer_validation_identity_absent_from_all_base_meta_"
                "uncertainty_and_residual_target_fit_lineages"
            ),
            "nested_cross_fit": True,
            "identity_leakage_count": 0,
            "all_outer_validation_identities_excluded": True,
            "outer_fold_audits": outer_lineage_audits,
        },
        "uncertainty_target": {
            "residual_source": "out_of_fold_action_score_residual",
            "score_se_floor": "1.96_times_action_score_se",
            "paired_delta_se_floor": "1.96_times_paired_delta_se_when_present",
            "paired_delta_schema_v2_samples": int(paired_delta_rows),
            "teacher_lcb_runtime_gate": False,
        },
        "safety_examples": {
            "source": "train_oof_only",
            "rows": int(safety_examples.labels.size),
            "used_for_threshold_lock": False,
            "used_for_locked_holdout": False,
            "identity_manifest": _safety_examples_identity_manifest(
                safety_examples
            ),
        },
        "fit_roles": {
            "train_outer_fit": (
                "predict_outer_validation_base_meta_uncertainty_without_"
                "validation_identity_lineage"
            ),
            "train_outer_validation": "oof_safety_feature_source_only",
            "train_inner_base_fit": "produce_outer_meta_fit_features",
            "train_inner_three_way_roles": (
                "produce_identity_clean_rank_residual_targets"
            ),
            "calibration_safety_fit": "not_used_in_cross_fit_stage",
            "calibration_threshold_lock": "not_used_in_cross_fit_stage",
            "locked_holdout": "not_used_in_cross_fit_stage",
        },
        "final_model_fit_source": "all_train_rows_after_oof_target_construction",
        "locked_holdout_used": False,
        "threshold_lock_used": False,
    }
    return CrossFitTrainingResult(
        model=final_model,
        safety_examples=safety_examples,
        report=report,
    )


def calibrate_safety(
    model: HuM4JointActionModel,
    safety_fit_samples: Sequence[PreparedTeacherSample],
    *,
    threshold_lock_samples: Sequence[PreparedTeacherSample],
    oof_safety_examples: CrossFitSafetyExamples | None = None,
    thresholds: Sequence[float],
    minimum_safe_teacher_gain: float = 0.0,
    minimum_fires: int = 30,
    maximum_false_positive_rate: float = 0.30,
    maximum_p95_loss: float = 25.0,
    maximum_p99_loss: float = 40.0,
    maximum_max_loss: float = 50.0,
    iterations: int = 150,
    max_leaf_nodes: int = 31,
    learning_rate: float = 0.05,
    l2_regularization: float = 1.0,
    safety_calibrator_c: float = 0.25,
    seed: int = 2026071805,
) -> tuple[HuM4JointActionModel, dict[str, Any]]:
    if not safety_fit_samples or not threshold_lock_samples:
        raise ValueError("safety fit and threshold lock both require samples")
    overlap = _safety_subset_overlap(safety_fit_samples, threshold_lock_samples)
    if any(overlap.values()):
        raise ValueError(f"safety fit/threshold lock overlap detected: {overlap}")
    normalized_thresholds = _normalize_thresholds(thresholds)
    if not math.isfinite(safety_calibrator_c) or safety_calibrator_c <= 0.0:
        raise ValueError("safety_calibrator_c must be finite and positive")
    fit_features, fit_labels, fit_rows = _safety_rows(
        model,
        safety_fit_samples,
        minimum_safe_teacher_gain=minimum_safe_teacher_gain,
    )
    oof_overlap: dict[str, int] | None = None
    if oof_safety_examples is not None:
        oof_overlap = _oof_safety_identity_overlap(
            oof_safety_examples,
            (*safety_fit_samples, *threshold_lock_samples),
        )
        if any(oof_overlap.values()):
            raise ValueError(
                f"OOF train safety/calibration identity overlap detected: {oof_overlap}"
            )
        if oof_safety_examples.features.ndim != 2 or (
            oof_safety_examples.features.shape[0]
            != oof_safety_examples.labels.shape[0]
        ):
            raise ValueError("OOF safety feature/label shape mismatch")
        if fit_features.size and (
            oof_safety_examples.features.shape[1] != fit_features.shape[1]
        ):
            raise ValueError("OOF/calibration safety feature dimensions disagree")
    fit_feature_blocks = [fit_features] if fit_features.size else []
    fit_label_blocks = [np.asarray(fit_labels, dtype=np.int8)] if fit_labels else []
    if oof_safety_examples is not None and oof_safety_examples.features.size:
        fit_feature_blocks.insert(0, oof_safety_examples.features)
        fit_label_blocks.insert(0, oof_safety_examples.labels)
    combined_fit_features = (
        np.vstack(fit_feature_blocks).astype(np.float32, copy=False)
        if fit_feature_blocks
        else np.empty((0, 0), dtype=np.float32)
    )
    safety_labels = (
        np.concatenate(fit_label_blocks).astype(np.int8, copy=False)
        if fit_label_blocks
        else np.empty(0, dtype=np.int8)
    )
    if combined_fit_features.size:
        if np.unique(safety_labels).size < 2:
            safety_estimator: Any = ConstantProbabilityEstimator(
                float(safety_labels[0])
            )
        elif (
            model.action_score_mode
            == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
        ):
            # Deliberately low capacity: one standardized L2-logistic layer.
            # Threshold-lock and locked-holdout rows never enter this fit.
            safety_estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=safety_calibrator_c,
                    solver="lbfgs",
                    max_iter=max(100, iterations),
                    random_state=seed,
                ),
            ).fit(combined_fit_features, safety_labels)
        else:
            safety_estimator = _fit_classifier(
                combined_fit_features,
                safety_labels,
                np.ones(safety_labels.shape[0], dtype=np.float64),
                iterations=iterations,
                max_leaf_nodes=max_leaf_nodes,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
                seed=seed,
            )
        provisional = replace(model, safety_estimator=safety_estimator)
    else:
        safety_estimator = ConstantProbabilityEstimator(0.0)
        provisional = replace(model, safety_estimator=safety_estimator)

    lock_features, lock_labels, lock_rows = _safety_rows(
        model,
        threshold_lock_samples,
        minimum_safe_teacher_gain=minimum_safe_teacher_gain,
    )
    probabilities = (
        _classifier_probabilities(safety_estimator, lock_features)
        if lock_features.size
        else np.zeros(0, dtype=np.float64)
    )

    sweep = []
    for threshold in normalized_thresholds:
        selected = [
            row
            for row, probability in zip(lock_rows, probabilities, strict=True)
            if float(probability) >= threshold
        ]
        sweep.append(
            _override_metrics(
                selected,
                total_states=len(threshold_lock_samples),
                threshold=threshold,
            )
        )
    eligible = [
        row
        for row in sweep
        if row["fires"] >= minimum_fires
        and row["teacher_mean_delta_per_fire"] > 0.0
        and row["false_positive_rate"] <= maximum_false_positive_rate
        and row["p95_loss"] <= maximum_p95_loss
        and row["p99_loss"] <= maximum_p99_loss
        and row["max_loss"] <= maximum_max_loss
    ]
    if eligible:
        selected_metric = max(
            eligible,
            key=lambda row: (row["teacher_delta_per_state"], row["threshold"]),
        )
        calibration_status = "go"
        safety_enabled = True
    else:
        selected_metric = next(
            row for row in reversed(sweep) if row["threshold"] == max(normalized_thresholds)
        )
        calibration_status = "no_go"
        safety_enabled = False
    calibrated = replace(
        provisional,
        safety_threshold=float(selected_metric["threshold"]),
        safety_enabled=safety_enabled,
    )
    report = {
        "status": calibration_status,
        "teacher_value_status": TEACHER_VALUE_STATUS,
        # Kept for report readers written against v1; this is now lock-only.
        "candidate_overrides": len(lock_rows),
        "safe_label_rate": float(np.mean(lock_labels)) if lock_labels else 0.0,
        "safety_fit": {
            "samples": len(safety_fit_samples),
            "candidate_overrides": len(fit_rows),
            "safe_label_rate": float(np.mean(fit_labels)) if fit_labels else 0.0,
            "used_for_estimator_fit": True,
            "used_for_threshold_sweep": False,
        },
        "threshold_lock": {
            "samples": len(threshold_lock_samples),
            "candidate_overrides": len(lock_rows),
            "safe_label_rate": float(np.mean(lock_labels)) if lock_labels else 0.0,
            "used_for_estimator_fit": False,
            "used_for_threshold_sweep": True,
        },
        "fit_lock_overlap": overlap,
        "oof_train_calibration_overlap": oof_overlap,
        "minimum_safe_teacher_gain": float(minimum_safe_teacher_gain),
        "selected_threshold": float(selected_metric["threshold"]),
        "selected_metrics": selected_metric,
        "threshold_sweep": sweep,
        "constraints": {
            "minimum_fires": int(minimum_fires),
            "maximum_false_positive_rate": float(maximum_false_positive_rate),
            "maximum_p95_loss": float(maximum_p95_loss),
            "maximum_p99_loss": float(maximum_p99_loss),
            "maximum_max_loss": float(maximum_max_loss),
        },
        "safety_estimator_source": (
            "train_oof_plus_safety_fit_subset"
            if oof_safety_examples is not None
            else "safety_fit_subset_only"
        ),
        "safety_estimator_family": (
            "standardized_l2_logistic_low_capacity"
            if model.action_score_mode
            == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
            and combined_fit_features.size
            and np.unique(safety_labels).size >= 2
            else (
                "constant_probability_one_class"
                if combined_fit_features.size
                else "constant_probability_no_candidates"
            )
            if model.action_score_mode
            == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
            else "hist_gradient_boosting_or_constant_legacy"
        ),
        "safety_calibrator_c": (
            float(safety_calibrator_c)
            if model.action_score_mode
            == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
            else None
        ),
        "runtime_inputs_exclude_teacher_values_and_teacher_lcb": True,
        "safety_estimator_training_sources": {
            "train_oof": {
                "candidate_overrides": (
                    int(oof_safety_examples.labels.size)
                    if oof_safety_examples is not None
                    else 0
                ),
                "used_for_estimator_fit": oof_safety_examples is not None,
                "used_for_threshold_sweep": False,
            },
            "calibration_safety_fit": {
                "candidate_overrides": len(fit_rows),
                "used_for_estimator_fit": True,
                "used_for_threshold_sweep": False,
            },
            "calibration_threshold_lock": {
                "candidate_overrides": len(lock_rows),
                "used_for_estimator_fit": False,
                "used_for_threshold_sweep": True,
            },
            "locked_holdout": {
                "used_for_estimator_fit": False,
                "used_for_threshold_sweep": False,
            },
        },
        "threshold_source": "threshold_lock_subset_only",
        "threshold_selection_source": (
            "calibration.threshold_lock"
            if model.action_score_mode
            == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
            else "threshold_lock_subset_only"
        ),
        "safety_calibrator_sources": (
            ["train_oof", "calibration.safety_fit"]
            if model.action_score_mode
            == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
            else (
                ["train_oof", "calibration.safety_fit"]
                if oof_safety_examples is not None
                else ["calibration.safety_fit"]
            )
        ),
        "locked_holdout_used": False,
    }
    return calibrated, report


def fail_closed_safety(
    model: HuM4JointActionModel,
    *,
    thresholds: Sequence[float],
    reason: str,
) -> tuple[HuM4JointActionModel, dict[str, Any]]:
    """Return a serializable candidate that cannot fire any override."""

    normalized_thresholds = _normalize_thresholds(thresholds)
    threshold = float(max(normalized_thresholds))
    closed = replace(
        model,
        safety_estimator=ConstantProbabilityEstimator(0.0),
        safety_threshold=threshold,
        safety_enabled=False,
    )
    empty_metrics = _override_metrics((), total_states=0, threshold=threshold)
    return closed, {
        "status": "no_go",
        "reason": reason,
        "teacher_value_status": TEACHER_VALUE_STATUS,
        "candidate_overrides": 0,
        "safe_label_rate": 0.0,
        "selected_threshold": threshold,
        "selected_metrics": empty_metrics,
        "threshold_sweep": [],
        "safety_estimator_source": "not_fit_fail_closed",
        "threshold_source": "not_swept_fail_closed",
        "locked_holdout_used": False,
    }


def _safety_rows(
    model: HuM4JointActionModel,
    samples: Sequence[PreparedTeacherSample],
    *,
    minimum_safe_teacher_gain: float,
) -> tuple[np.ndarray, list[int], list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, Any]] = []
    for sample in samples:
        heads = model.predict_heads_sample(sample.policy_sample)
        candidate = canonical_action_argmax(sample.policy_sample, heads.action_score)
        baseline = sample.baseline_index
        if candidate == baseline:
            continue
        teacher_delta = float(
            sample.teacher_scores[candidate] - sample.teacher_scores[baseline]
        )
        features.append(
            build_joint_safety_features(
                heads,
                candidate_index=candidate,
                baseline_index=baseline,
                seat=sample.seat,
            )
        )
        labels.append(int(teacher_delta > minimum_safe_teacher_gain))
        rows.append(
            {
                "teacher_delta": teacher_delta,
                "seat": sample.seat,
                "candidate_index": candidate,
                "baseline_index": baseline,
                **(
                    {
                        "downside_loss_p95": float(
                            sample.downside_loss_p95[candidate]
                        ),
                        "downside_loss_p99": float(
                            sample.downside_loss_p99[candidate]
                        ),
                        "downside_loss_max": float(
                            sample.downside_loss_max[candidate]
                        ),
                    }
                    if model.action_score_mode
                    == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
                    and sample.downside_loss_p95 is not None
                    and sample.downside_loss_p99 is not None
                    and sample.downside_loss_max is not None
                    else {}
                ),
            }
        )
    matrix = (
        np.vstack(features).astype(np.float32, copy=False)
        if features
        else np.empty((0, 0), dtype=np.float32)
    )
    return matrix, labels, rows


def _normalize_thresholds(thresholds: Sequence[float]) -> list[float]:
    normalized = sorted({float(value) for value in thresholds})
    if not normalized or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in normalized
    ):
        raise ValueError("safety thresholds must be finite values in [0, 1]")
    return normalized


def evaluate_model(
    model: HuM4JointActionModel,
    samples: Sequence[PreparedTeacherSample],
    *,
    split: str,
) -> dict[str, Any]:
    action_errors: list[float] = []
    uncertainty_targets: list[float] = []
    uncertainty_predictions: list[float] = []
    top1 = 0
    rows: list[dict[str, Any]] = []
    seats = {"first": [], "second": []}
    for sample in samples:
        heads = model.predict_heads_sample(sample.policy_sample)
        candidate = canonical_action_argmax(sample.policy_sample, heads.action_score)
        true_best = float(np.max(sample.teacher_scores))
        top1 += int(sample.teacher_scores[candidate] >= true_best - 1e-9)
        score_target = (
            sample.teacher_scores - float(np.max(sample.teacher_scores))
            if model.action_score_mode == NEGATIVE_REGRET_ACTION_SCORE_MODE
            else (
                np.asarray(sample.teacher_paired_delta_mean, dtype=np.float64)
                if model.action_score_mode
                == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
                and sample.teacher_paired_delta_mean is not None
                else sample.teacher_scores
            )
        )
        action_errors.extend((heads.action_score - score_target).tolist())
        if (
            model.action_score_mode
            == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
            and sample.downside_loss_p95 is not None
            and sample.downside_loss_p99 is not None
            and sample.downside_loss_max is not None
        ):
            uncertainty_targets.extend(
                (
                    0.50 * sample.downside_loss_p95
                    + 0.30 * sample.downside_loss_p99
                    + 0.20 * sample.downside_loss_max
                ).tolist()
            )
        else:
            uncertainty_targets.extend(
                _evaluation_uncertainty_target(
                    sample,
                    target_scores=score_target,
                    predicted_scores=heads.action_score,
                ).tolist()
            )
        uncertainty_predictions.extend(heads.predicted_absolute_residual.tolist())
        baseline = sample.baseline_index
        probability = (
            model.predict_safety_probability(
                sample.policy_sample,
                candidate_index=candidate,
                baseline_index=baseline,
            )
            if candidate != baseline and model.safety_estimator is not None
            else 0.0
        )
        fired = bool(
            model.safety_enabled
            and candidate != baseline
            and probability >= model.safety_threshold
        )
        if fired:
            row = {
                "teacher_delta": float(
                    sample.teacher_scores[candidate]
                    - sample.teacher_scores[baseline]
                ),
                "seat": sample.seat,
            }
            rows.append(row)
            seats[sample.seat].append(row)
    errors = np.asarray(action_errors, dtype=np.float64)
    uncertainty_error = np.asarray(uncertainty_predictions) - np.asarray(
        uncertainty_targets
    )
    report = {
        "split": split,
        "samples": len(samples),
        "actions": int(errors.size),
        "top1_accuracy": float(top1 / len(samples)) if samples else 0.0,
        "action_score_mae": float(np.mean(np.abs(errors))) if errors.size else 0.0,
        "action_score_rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else 0.0,
        "action_score_target": (
            "state_relative_negative_regret"
            if model.action_score_mode == NEGATIVE_REGRET_ACTION_SCORE_MODE
            else (
                "paired_delta_mean_vs_explicit_baseline"
                if model.action_score_mode
                == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
                else "absolute_teacher_score"
            )
        ),
        "uncertainty_mae": float(np.mean(np.abs(uncertainty_error)))
        if uncertainty_error.size
        else 0.0,
        "frozen_safety_threshold": float(model.safety_threshold),
        "safety_enabled": bool(model.safety_enabled),
        "override_metrics": _override_metrics(
            rows,
            total_states=len(samples),
            threshold=model.safety_threshold,
        ),
        "seat_override_metrics": {
            seat: _override_metrics(
                seat_rows,
                total_states=sum(sample.seat == seat for sample in samples),
                threshold=model.safety_threshold,
            )
            for seat, seat_rows in seats.items()
        },
        "teacher_value_status": TEACHER_VALUE_STATUS,
    }
    return report


def train_from_files(
    *,
    train_path: str | Path | Sequence[str | Path],
    calibration_path: str | Path | Sequence[str | Path],
    locked_holdout_path: str | Path | Sequence[str | Path] | None,
    output_model: Path,
    manifest_output: Path,
    m43_data_contract_path: str | Path | None = None,
    m43_plan_path: str | Path | None = None,
    m43_repo_root: str | Path | None = None,
    m43_fold_estimator_provider: M43FoldEstimatorProvider | None = None,
    m43_distributed_fold_assembly: Mapping[str, Any] | None = None,
    model_id: str = "hu-m4-t1-joint-v1",
    near_best_margin: float = 0.5,
    minimum_safe_teacher_gain: float = 0.0,
    iterations: int = 150,
    max_leaf_nodes: int = 31,
    learning_rate: float = 0.05,
    l2_regularization: float = 1.0,
    seed: int = 2026071801,
    action_score_mode: str = LEGACY_ACTION_SCORE_MODE,
    cross_fit_folds: int = 1,
    paired_se_floor: float = 0.50,
    paired_huber_alpha: float = 0.90,
    downside_quantile: float = 0.90,
    positive_gain_score_weight: float = DEFAULT_POSITIVE_GAIN_SCORE_WEIGHT,
    downside_risk_score_weight: float = DEFAULT_DOWNSIDE_RISK_SCORE_WEIGHT,
    ensemble_disagreement_score_weight: float = (
        DEFAULT_ENSEMBLE_DISAGREEMENT_SCORE_WEIGHT
    ),
    safety_calibrator_c: float = 0.25,
    safety_fit_ratio: float = 0.5,
    safety_split_seed: int = 2026071802,
    minimum_safety_fit_samples: int = 30,
    minimum_threshold_lock_samples: int = 30,
    thresholds: Sequence[float] = (0.0, 0.5, 0.9, 1.0),
    minimum_calibration_fires: int | None = None,
    maximum_false_positive_rate: float = 0.30,
    maximum_p95_loss: float = 25.0,
    maximum_p99_loss: float = 40.0,
    maximum_max_loss: float = 50.0,
) -> dict[str, Any]:
    if action_score_mode not in ACTION_SCORE_MODES:
        raise ValueError(f"unsupported action_score_mode: {action_score_mode!r}")
    if cross_fit_folds < 1:
        raise ValueError("cross_fit_folds must be positive")
    if action_score_mode in {
        NEGATIVE_REGRET_ACTION_SCORE_MODE,
        PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
    } and cross_fit_folds < 2:
        raise ValueError(
            f"{action_score_mode} requires cross_fit_folds of at least 2"
        )
    resolved_minimum_calibration_fires = (
        int(minimum_calibration_fires)
        if minimum_calibration_fires is not None
        else (
            M43_DEFAULT_PILOT_MINIMUM_FIRES
            if action_score_mode
            == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
            else LEGACY_DEFAULT_MINIMUM_CALIBRATION_FIRES
        )
    )
    if resolved_minimum_calibration_fires <= 0:
        raise ValueError("minimum_calibration_fires must be positive")
    if (
        action_score_mode == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
        and (
            m43_data_contract_path is None
            or m43_plan_path is None
            or m43_repo_root is None
        )
    ):
        raise ValueError(
            "baseline_paired_delta_risk_ensemble_v3 requires "
            "m43_data_contract_path, m43_plan_path, and m43_repo_root"
        )
    if (
        action_score_mode != PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
        and any(
            value is not None
            for value in (m43_data_contract_path, m43_plan_path, m43_repo_root)
        )
    ):
        raise ValueError("M4.3 contract/plan/repo inputs are valid only for M4.3 mode")
    is_m43 = action_score_mode == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
    if (m43_fold_estimator_provider is None) != (
        m43_distributed_fold_assembly is None
    ):
        raise ValueError(
            "M4.3 distributed fold provider and assembly receipt must be supplied together"
        )
    if m43_fold_estimator_provider is not None and (
        not is_m43 or cross_fit_folds != 5
    ):
        raise ValueError("distributed fold assembly is valid only for M4.3 exact 5-fold")
    if is_m43 and locked_holdout_path is not None:
        raise ValueError(
            "M4.3 trainer must not receive locked_holdout_path; locked labels "
            "are evaluated only after artifact and threshold freeze"
        )
    if not is_m43 and locked_holdout_path is None:
        raise ValueError("legacy/M4.2 training requires locked_holdout_path")
    paths: dict[str, tuple[Path, ...]] = {
        "train": _normalize_input_paths(train_path, split="train"),
        "calibration": _normalize_input_paths(calibration_path, split="calibration"),
    }
    if locked_holdout_path is not None:
        paths["locked_holdout"] = _normalize_input_paths(
            locked_holdout_path, split="locked_holdout"
        )
    flat_input_paths = [path for split_paths in paths.values() for path in split_paths]
    if m43_data_contract_path is not None:
        contract_path = Path(m43_data_contract_path).resolve()
        if contract_path in set(flat_input_paths):
            raise ValueError("M4.3 data contract must be distinct from teacher shards")
    else:
        contract_path = None
    if len(set(flat_input_paths)) != len(flat_input_paths):
        raise ValueError("every train/calibration/locked-holdout shard must be distinct")
    protected_inputs = set(flat_input_paths)
    if contract_path is not None:
        protected_inputs.add(contract_path)
    if m43_plan_path is not None:
        protected_inputs.add(Path(m43_plan_path).resolve())
    output_paths = {output_model.resolve(), manifest_output.resolve()}
    if len(output_paths) != 2 or output_paths & protected_inputs:
        raise ValueError("output model/manifest must be distinct from each other and inputs")
    raw_shards = {
        name: [read_teacher_jsonl(path) for path in split_paths]
        for name, split_paths in paths.items()
    }
    # Concatenation order is exactly the repeatable CLI/API argument order.  Do
    # not sort paths: train_a/train_b ordering is provenance and reproducibility
    # state even when both contain disjoint roots.
    raw = {
        name: [row for shard_rows in split_shards for row in shard_rows]
        for name, split_shards in raw_shards.items()
    }
    prepared = {name: prepare_teacher_samples(rows) for name, rows in raw.items()}
    split_integrity = validate_disjoint_splits(
        prepared["train"],
        prepared["calibration"],
        prepared.get("locked_holdout", ()),
    )
    if is_m43:
        split_integrity["locked_holdout"] = (
            "sealed_contract_only_not_opened_by_trainer"
        )
    if action_score_mode == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE:
        assert contract_path is not None
        (
            safety_fit,
            threshold_lock,
            calibration_partition,
            m43_binding_validation,
        ) = (
            _load_m43_role_binding(
                contract_path,
                plan_path=m43_plan_path,
                repo_root=m43_repo_root,
                train_paths=paths["train"],
                calibration_paths=paths["calibration"],
                raw=raw,
                prepared=prepared,
            )
        )
    else:
        m43_binding_validation = None
        safety_fit, threshold_lock, calibration_partition = split_safety_calibration(
            prepared["calibration"],
            safety_fit_ratio=safety_fit_ratio,
            split_seed=safety_split_seed,
            minimum_safety_fit_samples=minimum_safety_fit_samples,
            minimum_threshold_lock_samples=minimum_threshold_lock_samples,
        )
    hidden_audit = {
        "status": "pass",
        "feature_source": "policy_observation_only",
        "opponent_private_discards_in_feature_contract": False,
        "top_level_truth_fields_ignored": {
            name: int(sum(bool(sample.ignored_truth_keys) for sample in samples))
            for name, samples in prepared.items()
        },
        "nested_forbidden_field_count": 0,
    }

    oof_safety_examples: CrossFitSafetyExamples | None = None
    if cross_fit_folds > 1:
        cross_fit_result = fit_joint_model_cross_fitted(
            prepared["train"],
            cross_fit_folds=cross_fit_folds,
            action_score_mode=action_score_mode,
            model_id=model_id,
            near_best_margin=near_best_margin,
            minimum_safe_teacher_gain=minimum_safe_teacher_gain,
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            seed=seed,
            paired_se_floor=paired_se_floor,
            paired_huber_alpha=paired_huber_alpha,
            downside_quantile=downside_quantile,
            positive_gain_score_weight=positive_gain_score_weight,
            downside_risk_score_weight=downside_risk_score_weight,
            ensemble_disagreement_score_weight=(
                ensemble_disagreement_score_weight
            ),
            m43_fold_estimator_provider=m43_fold_estimator_provider,
        )
        model = cross_fit_result.model
        oof_safety_examples = cross_fit_result.safety_examples
        cross_fit_report = dict(cross_fit_result.report)
        calibration_overlap = _oof_safety_identity_overlap(
            oof_safety_examples, prepared["calibration"]
        )
        holdout_overlap = (
            {
                "status": "sealed_contract_only_not_opened_by_trainer",
                "identity_overlap_not_computed_from_locked_labels": True,
            }
            if is_m43
            else _oof_safety_identity_overlap(
                oof_safety_examples, prepared["locked_holdout"]
            )
        )
        if any(calibration_overlap.values()) or (
            not is_m43 and any(holdout_overlap.values())
        ):
            raise AssertionError("cross-fit train identities leaked across locked splits")
        cross_fit_report["split_role_overlap"] = {
            "train_oof__calibration": calibration_overlap,
            "train_oof__locked_holdout": holdout_overlap,
        }
    else:
        model = fit_joint_model(
            prepared["train"],
            model_id=model_id,
            near_best_margin=near_best_margin,
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            seed=seed,
        )
        cross_fit_report = {
            "schema": "hu_m4_identity_group_cross_fit_v1",
            "status": "disabled_legacy",
            "folds": 1,
            "action_score_mode": LEGACY_ACTION_SCORE_MODE,
            "legacy_behavior_preserved": True,
            "oof_safety_examples": 0,
            "locked_holdout_used": False,
            "threshold_lock_used": False,
        }
    train_metrics = evaluate_model(model, prepared["train"], split="train")
    if calibration_partition["status"] == "pass":
        model, calibration_report = calibrate_safety(
            model,
            safety_fit,
            threshold_lock_samples=threshold_lock,
            oof_safety_examples=oof_safety_examples,
            thresholds=thresholds,
            minimum_safe_teacher_gain=minimum_safe_teacher_gain,
            minimum_fires=resolved_minimum_calibration_fires,
            maximum_false_positive_rate=maximum_false_positive_rate,
            maximum_p95_loss=maximum_p95_loss,
            maximum_p99_loss=maximum_p99_loss,
            maximum_max_loss=maximum_max_loss,
            iterations=iterations,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            safety_calibrator_c=safety_calibrator_c,
            seed=seed + 100,
        )
    else:
        model, calibration_report = fail_closed_safety(
            model,
            thresholds=thresholds,
            reason="insufficient_disjoint_safety_fit_or_threshold_lock_samples",
        )
    if is_m43:
        # P0 lifecycle boundary: model training never opens locked labels.  A
        # separate command first verifies the saved artifact/threshold freeze,
        # atomically claims the one-shot marker, and only then evaluates them.
        locked_holdout_report = {
            "status": "not_evaluated_pre_freeze",
        }
    else:
        # Legacy/M4.2 contract retained exactly for explicit legacy modes.
        locked_holdout_report = evaluate_model(
            model, prepared["locked_holdout"], split="locked_holdout"
        )
    manifest = {
        "schema": HU_M4_JOINT_TRAINING_MANIFEST_SCHEMA,
        "artifact_schema": HU_M4_JOINT_ARTIFACT_SCHEMA,
        "model_schema": HU_M4_JOINT_MODEL_SCHEMA,
        "feature_schema": HU_M4_JOINT_FEATURE_SCHEMA,
        "model_id": model_id,
        "feature_dim": HU_FEATURE_DIM,
        "heads": {
            "policy": "near_best_probability",
            "value": (
                "paired_delta_mean_compatibility_slot"
                if is_m43
                else "absolute_teacher_score"
            ),
            "delta": (
                "paired_delta_mean_vs_explicit_baseline"
                if action_score_mode
                == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
                else "teacher_score_minus_state_baseline_teacher_score"
            ),
            "uncertainty": (
                "paired_downside_tail_plus_cross_fold_disagreement"
                if is_m43
                else (
                    "oof_residual_or_1.96_score_se_or_1.96_paired_delta_se"
                    if cross_fit_folds > 1
                    else "predicted_absolute_residual_or_1.96_score_se"
                )
            ),
            "safety": (
                "low_capacity_oof_positive_paired_delta_probability"
                if action_score_mode
                == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
                else "positive_teacher_delta_probability"
            ),
        },
        "action_score_formula": (
            {
                "mode": NEGATIVE_REGRET_ACTION_SCORE_MODE,
                "feature_schema": HU_M4_META_RANK_FEATURE_SCHEMA,
                "target": "teacher_score_minus_state_max_negative_regret",
                "state_offset_invariant": True,
                "meta_fit_source": "identity_group_out_of_fold_head_predictions",
            }
            if action_score_mode == NEGATIVE_REGRET_ACTION_SCORE_MODE
            else (
                {
                    "mode": PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
                    "feature_schema": HU_M4_PAIRED_ACTION_FEATURE_SCHEMA,
                    "target": "paired_delta_mean_vs_explicit_baseline",
                    "runtime_model": "stored_crossfit_fold_ensemble",
                    "baseline_action_score_exact_zero": True,
                    "positive_gain_score_weight": float(
                        positive_gain_score_weight
                    ),
                    "downside_risk_score_weight": float(
                        downside_risk_score_weight
                    ),
                    "ensemble_disagreement_score_weight": float(
                        ensemble_disagreement_score_weight
                    ),
                    "teacher_value_runtime_input": False,
                    "teacher_lcb_runtime_gate": False,
                }
                if action_score_mode
                == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
                else {
                "value_weight": VALUE_SCORE_WEIGHT,
                "delta_weight": DELTA_SCORE_WEIGHT,
                "policy_weight": POLICY_SCORE_WEIGHT,
                "policy_center": POLICY_SCORE_CENTER,
                "formula": "value + 0.5*delta + 0.25*(near_best_probability-0.5)",
                }
            )
        ),
        "teacher_value_status": TEACHER_VALUE_STATUS,
        "teacher_value_runtime_gate": False,
        "threshold_adaptation_after_calibration": False,
        "locked_holdout_used_for_threshold_or_training": False,
        "split_integrity": split_integrity,
        "calibration_partition": calibration_partition,
        "cross_fit": cross_fit_report,
        "hidden_discard_safety": hidden_audit,
        "action_mapping": {
            "identity": "regular_ofc_action_key_v1",
            "duplicate_semantic_actions": 0,
            "baseline_mapping": "validated_baseline_action_row_index",
            "runtime_tie_break": "canonical_action_key",
        },
        "inputs": {
            name: _input_split_manifest(
                paths=paths[name],
                shard_rows=raw_shards[name],
            )
            for name in paths
        },
        "m43_data_contract": (
            m43_binding_validation
            if action_score_mode
            == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
            else None
        ),
        "distributed_fold_assembly": (
            dict(m43_distributed_fold_assembly)
            if m43_distributed_fold_assembly is not None
            else None
        ),
        "training_config": {
            "near_best_margin": float(near_best_margin),
            "iterations": int(iterations),
            "max_leaf_nodes": int(max_leaf_nodes),
            "learning_rate": float(learning_rate),
            "l2_regularization": float(l2_regularization),
            "seed": int(seed),
            "action_score_mode": action_score_mode,
            "cross_fit_folds": int(cross_fit_folds),
            "paired_se_floor": float(paired_se_floor),
            "paired_huber_alpha": float(paired_huber_alpha),
            "downside_quantile": float(downside_quantile),
            "positive_gain_score_weight": float(positive_gain_score_weight),
            "downside_risk_score_weight": float(downside_risk_score_weight),
            "ensemble_disagreement_score_weight": float(
                ensemble_disagreement_score_weight
            ),
            "safety_calibrator_c": float(safety_calibrator_c),
            "minimum_calibration_fires": int(
                resolved_minimum_calibration_fires
            ),
            "calibration_fire_gate_scope": (
                "bounded_pilot_signal_not_final_acceptance"
                if action_score_mode
                == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
                else "legacy_calibration_gate"
            ),
            "safety_fit_ratio": float(safety_fit_ratio),
            "safety_split_seed": int(safety_split_seed),
            "minimum_safety_fit_samples": int(minimum_safety_fit_samples),
            "minimum_threshold_lock_samples": int(minimum_threshold_lock_samples),
        },
        "train_metrics": train_metrics,
        "calibration": calibration_report,
        "locked_holdout": locked_holdout_report,
        "promotion_status": (
            "candidate_for_realized_ev_evaluation"
            if calibration_report["status"] == "go"
            else "no_go_calibration"
        ),
    }
    if is_m43:
        manifest["artifact_embedded_manifest_excludes_locked_content"] = True
        manifest["locked_evaluation_command"] = (
            "separate_post_freeze_one_shot_evaluator"
        )
    model = replace(
        model,
        manifest=_artifact_embedded_manifest(manifest, is_m43=is_m43),
    )
    model.save(output_model)
    model_sha256 = _sha256(output_model)
    # The external manifest is the immutable runtime lock. Embedding the file's
    # own digest inside the pickle would be circular, so both action and safety
    # roles point to the same already-written joint artifact here.
    manifest["runtime_lock"] = {
        "single_joint_artifact": True,
        "model_path": str(output_model.resolve()),
        "candidate_model_sha256": model_sha256,
        "safety_model_sha256": model_sha256,
        "safety_threshold": float(model.safety_threshold),
    }
    _write_json_atomic(manifest_output, manifest)
    return manifest


def _artifact_embedded_manifest(
    manifest: Mapping[str, Any], *, is_m43: bool
) -> dict[str, Any]:
    """Keep the M4.3 pickle/hash independent of sealed locked content.

    The external training manifest may carry the audited data-contract binding;
    the later freeze manifest binds that file, the model hash, and the sealed
    locked contract.  Embedding the contract SHA or locked shard hashes in the
    pickle would make the pre-freeze artifact depend on locked content.
    """

    copied = json.loads(json.dumps(manifest, sort_keys=True))
    if not is_m43:
        return copied
    binding = copied.get("calibration_partition")
    role_only: dict[str, Any] = {
        "schema": "hu_m43_artifact_calibration_roles_v1",
        "external_freeze_binding_required": True,
        "locked_content_embedded": False,
    }
    if isinstance(binding, Mapping):
        for key in ("partition_method", "safety_fit", "threshold_lock", "overlap"):
            if key in binding:
                role_only[key] = binding[key]
    copied["m43_data_contract"] = {
        "schema": "hu_m43_external_data_contract_binding_required_v1",
        "embedded": False,
        "locked_content_embedded": False,
    }
    copied["calibration_partition"] = role_only
    copied["locked_holdout"] = {
        "status": "not_evaluated_pre_freeze",
        "labels_opened_by_trainer": False,
    }
    distributed = copied.get("distributed_fold_assembly")
    if isinstance(distributed, Mapping):
        # The external manifest/freeze receipt retains the complete Spot hash
        # chain.  Run-manifest, cloud-contract, and job-manifest hashes can be
        # derived from the sealed contract, so embedding them would make the
        # pre-freeze model bytes depend on locked-only metadata.  Estimator and
        # deterministic job-spec hashes depend only on train/calibration state.
        safe_jobs: list[dict[str, Any]] = []
        raw_jobs = distributed.get("jobs")
        if isinstance(raw_jobs, list):
            for raw_job in raw_jobs:
                if not isinstance(raw_job, Mapping):
                    continue
                safe_jobs.append(
                    {
                        key: raw_job[key]
                        for key in (
                            "job_index",
                            "artifact_sha256",
                            "job_spec_sha256",
                        )
                        if key in raw_job
                    }
                )
        copied["distributed_fold_assembly"] = {
            "schema": distributed.get("schema"),
            "status": distributed.get("status"),
            "job_count": distributed.get("job_count"),
            "jobs": safe_jobs,
            "exact_outer_inner_coverage": distributed.get(
                "exact_outer_inner_coverage"
            ),
            "current_profile_mutated": distributed.get(
                "current_profile_mutated"
            ),
            "no_runtime_activation": distributed.get("no_runtime_activation"),
            "external_spot_hash_chain_required": True,
            "sealed_contract_derived_hashes_embedded": False,
            "job_manifest_hashes_embedded": False,
        }
    inputs = copied.get("inputs")
    if isinstance(inputs, dict):
        inputs.pop("locked_holdout", None)
    return copied


def parse_thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one safety threshold is required")
    return values


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = train_from_files(
        train_path=args.train,
        calibration_path=args.calibration,
        locked_holdout_path=args.locked_holdout,
        output_model=args.output_model,
        manifest_output=args.manifest_output,
        m43_data_contract_path=args.m43_data_contract,
        m43_plan_path=args.m43_plan,
        m43_repo_root=args.repo_root,
        model_id=args.model_id,
        near_best_margin=args.near_best_margin,
        minimum_safe_teacher_gain=args.minimum_safe_teacher_gain,
        iterations=args.iterations,
        max_leaf_nodes=args.max_leaf_nodes,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
        seed=args.seed,
        action_score_mode=args.action_score_mode,
        cross_fit_folds=args.cross_fit_folds,
        paired_se_floor=args.paired_se_floor,
        paired_huber_alpha=args.paired_huber_alpha,
        downside_quantile=args.downside_quantile,
        positive_gain_score_weight=args.positive_gain_score_weight,
        downside_risk_score_weight=args.downside_risk_score_weight,
        ensemble_disagreement_score_weight=(
            args.ensemble_disagreement_score_weight
        ),
        safety_calibrator_c=args.safety_calibrator_c,
        safety_fit_ratio=args.safety_fit_ratio,
        safety_split_seed=args.safety_split_seed,
        minimum_safety_fit_samples=args.minimum_safety_fit_samples,
        minimum_threshold_lock_samples=args.minimum_threshold_lock_samples,
        thresholds=parse_thresholds(args.thresholds),
        minimum_calibration_fires=args.minimum_calibration_fires,
        maximum_false_positive_rate=args.maximum_false_positive_rate,
        maximum_p95_loss=args.maximum_p95_loss,
        maximum_p99_loss=args.maximum_p99_loss,
        maximum_max_loss=args.maximum_max_loss,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _training_arrays(
    samples: Sequence[PreparedTeacherSample], *, near_best_margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_blocks: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    standard_errors: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    policy: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for sample in samples:
        matrix, _targets = sample_to_matrix(sample.policy_sample)
        feature_blocks.append(matrix)
        scores.append(sample.teacher_scores)
        standard_errors.append(sample.teacher_score_se)
        baseline = float(sample.teacher_scores[sample.baseline_index])
        deltas.append(sample.teacher_scores - baseline)
        best = float(np.max(sample.teacher_scores))
        policy.append((sample.teacher_scores >= best - near_best_margin).astype(np.int8))
        weights.append(np.full(len(sample.teacher_scores), 1.0 / len(sample.teacher_scores)))
    return (
        np.vstack(feature_blocks).astype(np.float32, copy=False),
        np.concatenate(scores),
        np.concatenate(standard_errors),
        np.concatenate(deltas),
        np.concatenate(policy),
        np.concatenate(weights),
    )


def _fit_regressor(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    iterations: int,
    max_leaf_nodes: int,
    learning_rate: float,
    l2_regularization: float,
    seed: int,
) -> Any:
    if np.allclose(targets, targets[0]):
        return DummyRegressor(strategy="constant", constant=float(targets[0])).fit(
            features, targets
        )
    estimator = HistGradientBoostingRegressor(
        max_iter=iterations,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        random_state=seed,
    )
    return estimator.fit(features, targets, sample_weight=weights)


def _fit_gradient_regressor(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    loss: str,
    alpha: float,
    iterations: int,
    max_leaf_nodes: int,
    learning_rate: float,
    seed: int,
) -> Any:
    """Fit the explicit robust/quantile objectives used only by M4.3."""

    if np.allclose(targets, targets[0]):
        return DummyRegressor(strategy="constant", constant=float(targets[0])).fit(
            features, targets
        )
    estimator = GradientBoostingRegressor(
        loss=loss,
        alpha=alpha,
        n_estimators=iterations,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        random_state=seed,
    )
    return estimator.fit(features, targets, sample_weight=weights)


def _fit_classifier(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    iterations: int,
    max_leaf_nodes: int,
    learning_rate: float,
    l2_regularization: float,
    seed: int,
) -> Any:
    unique = np.unique(targets)
    if unique.size < 2:
        return DummyClassifier(strategy="constant", constant=int(unique[0])).fit(
            features, targets
        )
    estimator = HistGradientBoostingClassifier(
        max_iter=iterations,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        random_state=seed,
    )
    return estimator.fit(features, targets, sample_weight=weights)


def _classifier_probabilities(estimator: Any, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = list(getattr(estimator, "classes_", ()))
    if raw.ndim != 2 or raw.shape[0] != features.shape[0]:
        raise ValueError("invalid safety predict_proba output")
    if 1 in classes:
        return raw[:, classes.index(1)]
    if classes == [0]:
        return np.zeros(features.shape[0], dtype=np.float64)
    return raw[:, -1]


def _override_metrics(
    rows: Sequence[Mapping[str, Any]], *, total_states: int, threshold: float
) -> dict[str, Any]:
    deltas = np.asarray([float(row["teacher_delta"]) for row in rows], dtype=np.float64)
    losses = np.maximum(0.0, -deltas)
    fires = int(deltas.size)
    paired_tail_flags = [
        all(
            key in row
            for key in (
                "downside_loss_p95",
                "downside_loss_p99",
                "downside_loss_max",
            )
        )
        for row in rows
    ]
    if any(paired_tail_flags) and not all(paired_tail_flags):
        raise ValueError("override rows mix paired-tail and legacy loss metrics")
    if fires and all(paired_tail_flags):
        paired_p95 = np.asarray(
            [float(row["downside_loss_p95"]) for row in rows], dtype=np.float64
        )
        paired_p99 = np.asarray(
            [float(row["downside_loss_p99"]) for row in rows], dtype=np.float64
        )
        paired_max = np.asarray(
            [float(row["downside_loss_max"]) for row in rows], dtype=np.float64
        )
        if not all(
            np.isfinite(values).all() and np.all(values >= 0.0)
            for values in (paired_p95, paired_p99, paired_max)
        ):
            raise ValueError("paired downside metrics must be finite and non-negative")
        # Conservative threshold lock: every selected action must satisfy the
        # within-state paired-future p05/p01/min loss bounds.
        p95_loss = float(np.max(paired_p95))
        p99_loss = float(np.max(paired_p99))
        max_loss = float(np.max(paired_max))
        loss_metric_source = "selected_action_paired_p05_p01_min_maxima"
    else:
        p95_loss = float(np.quantile(losses, 0.95)) if fires else 0.0
        p99_loss = float(np.quantile(losses, 0.99)) if fires else 0.0
        max_loss = float(np.max(losses)) if fires else 0.0
        loss_metric_source = "cross_state_negative_mean_delta_legacy"
    return {
        "threshold": float(threshold),
        "fires": fires,
        "fire_rate": float(fires / total_states) if total_states else 0.0,
        "teacher_mean_delta_per_fire": float(np.mean(deltas)) if fires else 0.0,
        "teacher_delta_per_state": float(np.sum(deltas) / total_states)
        if total_states
        else 0.0,
        "false_positive_rate": float(np.mean(deltas <= 0.0)) if fires else 0.0,
        "p95_loss": p95_loss,
        "p99_loss": p99_loss,
        "max_loss": max_loss,
        "loss_metric_source": loss_metric_source,
        "teacher_value_status": TEACHER_VALUE_STATUS,
    }


def _find_forbidden_keys(value: Any, forbidden: frozenset[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in forbidden:
                found.add(key_text)
            found.update(_find_forbidden_keys(nested, forbidden))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.update(_find_forbidden_keys(nested, forbidden))
    return found


def _assign_cross_fit_folds(
    samples: Sequence[PreparedTeacherSample], *, folds: int, seed: int
) -> tuple[list[int], dict[str, Any]]:
    """Assign connected identities to balanced, deterministic OOF folds."""

    if folds < 2:
        raise ValueError("folds must be at least 2")
    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    seed_owner: dict[str, int] = {}
    fingerprint_owner: dict[str, int] = {}
    for index, sample in enumerate(samples):
        for seed_value in sample.root_seed_values:
            union(index, seed_owner.setdefault(seed_value, index))
        union(
            index,
            fingerprint_owner.setdefault(sample.observation_fingerprint, index),
        )

    components: dict[int, list[int]] = {}
    for index in range(len(samples)):
        components.setdefault(find(index), []).append(index)
    if len(components) < folds:
        raise ValueError(
            "cross_fit_folds exceeds the number of connected identity groups"
        )

    ranked: list[tuple[str, str, list[int]]] = []
    for indices in components.values():
        row_hashes = sorted(_sample_membership_hash(samples[index]) for index in indices)
        seeds = sorted(
            set().union(*(samples[index].root_seed_values for index in indices))
        )
        fingerprints = sorted(
            {samples[index].observation_fingerprint for index in indices}
        )
        identity_payload = {
            "row_hashes": row_hashes,
            "seed_values": seeds,
            "observation_fingerprints": fingerprints,
        }
        identity = hashlib.sha256(
            json.dumps(
                identity_payload, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
        ).hexdigest()
        rank = hashlib.sha256(f"{seed}\0{identity}".encode("ascii")).hexdigest()
        ranked.append((rank, identity, sorted(indices)))
    ranked.sort(key=lambda item: (item[0], item[1]))

    fold_loads = [0] * folds
    fold_components: list[list[tuple[str, list[int]]]] = [[] for _ in range(folds)]
    for component_index, (_rank, identity, indices) in enumerate(ranked):
        fold = (
            component_index
            if component_index < folds
            else min(range(folds), key=lambda value: (fold_loads[value], value))
        )
        fold_loads[fold] += len(indices)
        fold_components[fold].append((identity, indices))

    fold_ids = [-1] * len(samples)
    assignments: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []
    for fold, components_for_fold in enumerate(fold_components):
        validation_indices = sorted(
            index for _identity, indices in components_for_fold for index in indices
        )
        validation_set = set(validation_indices)
        training_indices = [
            index for index in range(len(samples)) if index not in validation_set
        ]
        for identity, indices in components_for_fold:
            for index in indices:
                if fold_ids[index] != -1:
                    raise AssertionError("sample assigned to more than one OOF fold")
                fold_ids[index] = fold
                assignments.append(
                    {
                        "row_hash": _sample_membership_hash(samples[index]),
                        "component_identity": identity,
                        "fold": fold,
                    }
                )
        train_seeds, train_fingerprints, train_hashes = _safety_subset_identity(
            [samples[index] for index in training_indices]
        )
        valid_seeds, valid_fingerprints, valid_hashes = _safety_subset_identity(
            [samples[index] for index in validation_indices]
        )
        overlap = {
            "seed_value_count": len(train_seeds & valid_seeds),
            "observation_fingerprint_count": len(
                train_fingerprints & valid_fingerprints
            ),
            "row_hash_count": len(train_hashes & valid_hashes),
        }
        if any(overlap.values()):
            raise AssertionError(f"cross-fit identity leak in fold {fold}: {overlap}")
        fold_audits.append(
            {
                "fold": fold,
                "training_samples": len(training_indices),
                "validation_samples": len(validation_indices),
                "validation_components": len(components_for_fold),
                "identity_overlap": overlap,
            }
        )
    if any(fold < 0 for fold in fold_ids):
        raise AssertionError("sample missing from OOF fold assignment")
    assignments.sort(
        key=lambda item: (item["row_hash"], item["component_identity"], item["fold"])
    )
    return fold_ids, {
        "method": "connected_seed_fingerprint_groups_hash_balanced_v1",
        "seed": int(seed),
        "folds": int(folds),
        "component_count": len(ranked),
        "fold_sample_counts": fold_loads,
        "identity_leakage_count": 0,
        "assignments": assignments,
        "fold_audits": fold_audits,
    }


def _oof_uncertainty_target(
    sample: PreparedTeacherSample,
    *,
    target_scores: np.ndarray,
    predicted_scores: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_scores, dtype=np.float64).reshape(-1)
    prediction = np.asarray(predicted_scores, dtype=np.float64).reshape(-1)
    if target.shape != sample.teacher_scores.shape or prediction.shape != target.shape:
        raise ValueError("OOF score/target shape mismatch")
    floors = [
        np.abs(target - prediction),
        1.96 * sample.teacher_score_se,
    ]
    if sample.teacher_delta_se_vs_baseline is not None:
        floors.append(1.96 * sample.teacher_delta_se_vs_baseline)
    result = np.maximum.reduce(floors)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError("OOF uncertainty target is invalid")
    return result


def _evaluation_uncertainty_target(
    sample: PreparedTeacherSample,
    *,
    target_scores: np.ndarray,
    predicted_scores: np.ndarray,
) -> np.ndarray:
    """Use the same conservative diagnostic target when reporting a model."""

    return _oof_uncertainty_target(
        sample,
        target_scores=target_scores,
        predicted_scores=predicted_scores,
    )


def _safety_examples_identity_manifest(
    examples: CrossFitSafetyExamples,
) -> dict[str, Any]:
    return {
        "seed_values": {
            "count": len(examples.seed_values),
            "sha256": _digest_strings(examples.seed_values),
        },
        "observation_fingerprints": {
            "count": len(examples.observation_fingerprints),
            "sha256": _digest_strings(examples.observation_fingerprints),
        },
        "row_hashes": {
            "count": len(examples.row_hashes),
            "sha256": _digest_strings(examples.row_hashes),
        },
    }


def _sample_membership_hash(sample: PreparedTeacherSample) -> str:
    payload = {
        "root_seed_values": sorted(sample.root_seed_values),
        "observation_fingerprint": sample.observation_fingerprint,
        "baseline_index": sample.baseline_index,
        "teacher_scores": sample.teacher_scores.tolist(),
        "teacher_score_se": sample.teacher_score_se.tolist(),
        "teacher_delta_se_vs_baseline": (
            sample.teacher_delta_se_vs_baseline.tolist()
            if sample.teacher_delta_se_vs_baseline is not None
            else None
        ),
        "teacher_paired_delta_mean": (
            sample.teacher_paired_delta_mean.tolist()
            if sample.teacher_paired_delta_mean is not None
            else None
        ),
        "downside_loss_p95": (
            sample.downside_loss_p95.tolist()
            if sample.downside_loss_p95 is not None
            else None
        ),
        "downside_loss_p99": (
            sample.downside_loss_p99.tolist()
            if sample.downside_loss_p99 is not None
            else None
        ),
        "downside_loss_max": (
            sample.downside_loss_max.tolist()
            if sample.downside_loss_max is not None
            else None
        ),
        "actions": sample.policy_sample["actions"],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _safety_subset_identity(
    samples: Sequence[PreparedTeacherSample],
) -> tuple[set[str], set[str], set[str]]:
    seeds = set().union(*(sample.root_seed_values for sample in samples)) if samples else set()
    fingerprints = {sample.observation_fingerprint for sample in samples}
    row_hashes = {_sample_membership_hash(sample) for sample in samples}
    return seeds, fingerprints, row_hashes


def _safety_subset_overlap(
    safety_fit: Sequence[PreparedTeacherSample],
    threshold_lock: Sequence[PreparedTeacherSample],
) -> dict[str, int]:
    fit_seeds, fit_fingerprints, fit_hashes = _safety_subset_identity(safety_fit)
    lock_seeds, lock_fingerprints, lock_hashes = _safety_subset_identity(threshold_lock)
    return {
        "seed_value_count": len(fit_seeds & lock_seeds),
        "observation_fingerprint_count": len(fit_fingerprints & lock_fingerprints),
        "row_hash_count": len(fit_hashes & lock_hashes),
    }


def _oof_safety_identity_overlap(
    examples: CrossFitSafetyExamples,
    samples: Sequence[PreparedTeacherSample],
) -> dict[str, int]:
    seeds, fingerprints, row_hashes = _safety_subset_identity(samples)
    return {
        "seed_value_count": len(set(examples.seed_values) & seeds),
        "observation_fingerprint_count": len(
            set(examples.observation_fingerprints) & fingerprints
        ),
        "row_hash_count": len(set(examples.row_hashes) & row_hashes),
    }


def _safety_subset_manifest(
    samples: Sequence[PreparedTeacherSample],
) -> dict[str, Any]:
    seeds, fingerprints, row_hashes = _safety_subset_identity(samples)
    return {
        "rows": len(samples),
        "seed_values": {
            "count": len(seeds),
            "sha256": _digest_strings(seeds),
        },
        "observation_fingerprints": {
            "count": len(fingerprints),
            "sha256": _digest_strings(fingerprints),
        },
        "row_hashes": {
            "count": len(row_hashes),
            "sha256": _digest_strings(row_hashes),
        },
    }


def _normalize_input_paths(
    value: str | Path | Sequence[str | Path], *, split: str
) -> tuple[Path, ...]:
    """Normalize a legacy single path or an explicitly ordered shard list."""

    if isinstance(value, (str, Path)):
        raw_paths: Sequence[str | Path] = (value,)
    else:
        raw_paths = value
    paths = tuple(Path(path).resolve() for path in raw_paths)
    if not paths:
        raise ValueError(f"{split} requires at least one input shard")
    return paths


def _input_split_manifest(
    *, paths: Sequence[Path], shard_rows: Sequence[Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    """Return legacy metadata for one shard and ordered metadata for many."""

    if len(paths) != len(shard_rows) or not paths:
        raise ValueError("input shard paths/rows disagree")
    shards = [
        {
            "index": index,
            "path": str(path),
            "sha256": _sha256(path),
            "rows": len(rows),
        }
        for index, (path, rows) in enumerate(zip(paths, shard_rows, strict=True))
    ]
    if len(shards) == 1:
        # Preserve the original v2 single-file manifest contract exactly.
        shard = shards[0]
        return {
            "path": shard["path"],
            "sha256": shard["sha256"],
            "rows": shard["rows"],
        }
    aggregate_payload = [
        {"sha256": shard["sha256"], "rows": shard["rows"]} for shard in shards
    ]
    aggregate_sha256 = hashlib.sha256(
        json.dumps(
            aggregate_payload, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest()
    return {
        "paths": [shard["path"] for shard in shards],
        "sha256": aggregate_sha256,
        "rows": sum(int(shard["rows"]) for shard in shards),
        "shard_count": len(shards),
        "concatenation_order": "cli_or_api_argument_order",
        "aggregate_sha256_contract": "ordered_shard_content_hashes_and_row_counts_v1",
        "shards": shards,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()


__all__ = [
    "CrossFitSafetyExamples",
    "CrossFitTrainingResult",
    "HU_M4_JOINT_TRAINING_MANIFEST_SCHEMA",
    "PreparedTeacherSample",
    "TEACHER_VALUE_STATUS",
    "calibrate_safety",
    "evaluate_model",
    "fail_closed_safety",
    "fit_joint_model",
    "fit_joint_model_cross_fitted",
    "parse_thresholds",
    "prepare_teacher_sample",
    "prepare_teacher_samples",
    "read_teacher_jsonl",
    "split_safety_calibration",
    "train_from_files",
    "validate_disjoint_splits",
]
