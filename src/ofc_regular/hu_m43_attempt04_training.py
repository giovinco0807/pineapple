"""Attempt04 meta-tail proposal and continuous-confidence development pilot.

This module intentionally consumes only the already-open Attempt02/03 fit rows
and the already-consumed Attempt03 pre-calibration rows.  Its reports are
development evidence, never a fresh generalization claim.  A new Attempt04
pre-calibration split remains the sole model-level Go/No-Go gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .action_key import action_key_from_payload
from .hu_m43_attempt03_training import Attempt03OofStatePrediction
from .hu_m43_joint_model_v5 import HuM43JointModelV5, V5ActionPredictions
from .hu_m4_joint_model import build_paired_action_features_matrix
from .hu_turn3_model import sample_to_matrix
from .train_hu_m4_joint_model import (
    PreparedTeacherSample,
    build_m43_fold_training_plan,
    prepare_teacher_samples,
    read_teacher_jsonl,
)


M43_ATTEMPT04_DEV_PILOT_SCHEMA = "hu_m43_attempt04_meta_tail_confidence_dev900_v1"
M43_ATTEMPT04_FINAL_MANIFEST_SCHEMA = (
    "hu_m43_attempt04_v6_final_training_manifest_v1"
)
M43_ATTEMPT04_THRESHOLD_LOCK_SCHEMA = "hu_m43_attempt04_v6_threshold_lock_v1"
M43_ATTEMPT04_CONFIDENCE_FEATURE_SCHEMA = (
    "hu_m43_attempt04_meta_tail_confidence_features_v1"
)
M43_ATTEMPT04_CONFIDENCE_FEATURE_DIM = 31
M43_ATTEMPT04_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
M43_ATTEMPT04_TAIL_LIMITS = (25.0, 40.0, 50.0)
M43_ATTEMPT04_CONFIDENCE_THRESHOLDS = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
    0.99,
)
M43_ATTEMPT04_FINAL_DEV_CONFORMAL_QUANTILES = (0.90, 0.95, 0.975)

DEFAULT_ATTEMPT02_TRAIN = Path(
    "outputs/hu_joint_policy/m43_attempt02_teacher/"
    "regular-hu-m43-attempt02-c2e64-pilot300-20260713-2201/train.jsonl"
)
DEFAULT_ATTEMPT03_FIT = Path(
    "outputs/hu_joint_policy/m43_attempt03_teacher/"
    "regular-hu-m43-attempt03-c2e64-pilot700-r2-20260714-0228/"
    "fresh_train_fit.jsonl"
)
DEFAULT_ATTEMPT03_CONSUMED_PRECAL = Path(
    "outputs/hu_joint_policy/m43_attempt03_precal/"
    "regular-hu-m43-attempt03-c2e64-pilot700-r2-20260714-0228/"
    "fresh_precal_holdout.jsonl"
)
DEFAULT_ATTEMPT03_FIT_BUNDLE = Path(
    "outputs/hu_joint_policy/m43_attempt03_model_runs/"
    "regular-hu-m43-attempt03-v5-model-20260714-063835/candidate/fit_bundle.pkl"
)


@dataclass(frozen=True)
class MetaTailProposal:
    index: int
    tail_pool_eligible: bool
    tail_pool_count: int
    confidence_features: np.ndarray


@dataclass(frozen=True)
class DevData:
    fit_samples: tuple[PreparedTeacherSample, ...]
    fit_profiles: tuple[str, ...]
    fit_fold_ids: tuple[int, ...]
    fit_predictions: tuple[V5ActionPredictions, ...]
    consumed_samples: tuple[PreparedTeacherSample, ...]
    consumed_profiles: tuple[str, ...]
    consumed_predictions: tuple[V5ActionPredictions, ...]
    source_manifest: Mapping[str, Any]


class _ConstantConfidence:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)
        self.classes_ = np.asarray([0, 1], dtype=np.int8)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features).shape[0]
        positive = np.full(rows, self.probability, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def load_dev900(
    *,
    attempt02_train: str | Path = DEFAULT_ATTEMPT02_TRAIN,
    attempt03_fit: str | Path = DEFAULT_ATTEMPT03_FIT,
    consumed_precal: str | Path = DEFAULT_ATTEMPT03_CONSUMED_PRECAL,
    fit_bundle: str | Path = DEFAULT_ATTEMPT03_FIT_BUNDLE,
) -> DevData:
    """Load exactly 700 fit + 200 consumed rows; no sealed path is accepted."""

    paths = tuple(Path(value) for value in (attempt02_train, attempt03_fit, consumed_precal))
    raw_blocks = tuple(read_teacher_jsonl(path) for path in paths)
    if tuple(len(rows) for rows in raw_blocks) != (200, 500, 200):
        raise ValueError("Attempt04 dev pilot requires exact 200/500/200 sources")
    if any("locked_holdout" in str(path).lower() or "calibration.jsonl" in str(path).lower() for path in paths):
        raise ValueError("Attempt04 dev pilot refuses sealed calibration/locked paths")

    fit_rows = [*raw_blocks[0], *raw_blocks[1]]
    fit_prepared = prepare_teacher_samples(fit_rows)
    consumed_prepared = prepare_teacher_samples(raw_blocks[2])
    for row, sample in zip(fit_rows, fit_prepared, strict=True):
        sample.policy_sample["policy_observation"] = dict(row["policy_observation"])
    for row, sample in zip(raw_blocks[2], consumed_prepared, strict=True):
        sample.policy_sample["policy_observation"] = dict(row["policy_observation"])

    profile_by_fingerprint = {
        sample.observation_fingerprint: _profile(row)
        for row, sample in zip(fit_rows, fit_prepared, strict=True)
    }
    consumed_profiles = tuple(_profile(row) for row in raw_blocks[2])
    plan = build_m43_fold_training_plan(
        fit_prepared, cross_fit_folds=5, seed=2026072401
    )
    ordered = tuple(plan.ordered_samples)
    fit_profiles = tuple(
        profile_by_fingerprint[sample.observation_fingerprint] for sample in ordered
    )
    _validate_identity_and_profiles(
        (*ordered, *consumed_prepared), (*fit_profiles, *consumed_profiles)
    )

    envelope = pickle.loads(Path(fit_bundle).read_bytes())
    if not isinstance(envelope, dict) or envelope.get("schema") != (
        "hu_m43_attempt03_v5_fit_bundle_v1"
    ):
        raise ValueError("Attempt03 fit bundle schema mismatch")
    fit_result = envelope.get("fit_result")
    raw_oof = tuple(getattr(fit_result, "oof_predictions", ()))
    if len(raw_oof) != 700 or any(
        not isinstance(row, Attempt03OofStatePrediction) for row in raw_oof
    ):
        raise ValueError("Attempt03 fit bundle OOF predictions are incomplete")
    by_index = {row.sample_index: row for row in raw_oof}
    if set(by_index) != set(range(700)):
        raise ValueError("Attempt03 OOF sample index mapping changed")
    fit_predictions = tuple(by_index[index].predictions for index in range(700))

    model = getattr(fit_result, "model", None)
    if not isinstance(model, HuM43JointModelV5):
        raise TypeError("Attempt03 fit bundle does not contain a v5 model")
    consumed_predictions = tuple(
        model.predict_heads_sample(sample.policy_sample, baseline_index=sample.baseline_index)
        for sample in consumed_prepared
    )
    sources = []
    for path, rows in zip(paths, raw_blocks, strict=True):
        sources.append(
            {
                "path": path.as_posix(),
                "rows": len(rows),
                "file_sha256": _file_sha256(path),
            }
        )
    sources.append(
        {
            "path": Path(fit_bundle).as_posix(),
            "file_sha256": _file_sha256(Path(fit_bundle)),
        }
    )
    return DevData(
        fit_samples=ordered,
        fit_profiles=fit_profiles,
        fit_fold_ids=tuple(int(value) for value in plan.outer_fold_ids),
        fit_predictions=fit_predictions,
        consumed_samples=tuple(consumed_prepared),
        consumed_profiles=consumed_profiles,
        consumed_predictions=consumed_predictions,
        source_manifest={
            "classification": "consumed_development_only_not_fresh_generalization",
            "sealed_calibration_opened": False,
            "locked_holdout_opened": False,
            "sources": sources,
        },
    )


def build_meta_tail_proposal(
    sample: PreparedTeacherSample,
    predictions: V5ActionPredictions,
    *,
    tail_limits: Sequence[float] = M43_ATTEMPT04_TAIL_LIMITS,
    tail_cushions: Sequence[float] = (0.0, 0.0, 0.0),
    rerank_inside_tail_pool: bool = True,
) -> MetaTailProposal:
    baseline = sample.baseline_index
    action_count = len(sample.policy_sample["actions"])
    arrays = (
        predictions.meta_score,
        predictions.base_delta,
        predictions.base_positive,
        predictions.downside_p95,
        predictions.downside_p99,
        predictions.downside_max,
        predictions.delta_disagreement,
        predictions.stage18_score,
    )
    if any(np.asarray(values).shape != (action_count,) for values in arrays):
        raise ValueError("meta-tail prediction/action shape mismatch")
    nonbaseline = np.asarray(
        [index for index in range(action_count) if index != baseline], dtype=np.int32
    )
    p95 = np.asarray(predictions.downside_p95, dtype=np.float64)
    p99 = np.asarray(predictions.downside_p99, dtype=np.float64)
    maximum = np.asarray(predictions.downside_max, dtype=np.float64)
    limits = np.asarray(tail_limits, dtype=np.float64)
    cushions = np.asarray(tail_cushions, dtype=np.float64)
    if cushions.shape != (3,) or np.any(cushions < 0.0):
        raise ValueError("Attempt04 tail cushions must be three non-negative values")
    mask = (p95 + cushions[0] <= limits[0]) & (
        p99 + cushions[1] <= limits[1]
    ) & (maximum + cushions[2] <= limits[2])
    mask[baseline] = False
    pool = nonbaseline[mask[nonbaseline]]
    candidates = pool if rerank_inside_tail_pool and pool.size else nonbaseline
    candidate = _meta_base_action_key_argmax(sample, predictions, candidates)
    features = build_confidence_features(
        sample, predictions, candidate_index=candidate, tail_pool_mask=mask
    )
    return MetaTailProposal(
        index=candidate,
        tail_pool_eligible=bool(pool.size and mask[candidate]),
        tail_pool_count=int(pool.size),
        confidence_features=features,
    )


def build_confidence_features(
    sample: PreparedTeacherSample,
    predictions: V5ActionPredictions,
    *,
    candidate_index: int,
    tail_pool_mask: np.ndarray,
) -> np.ndarray:
    baseline = sample.baseline_index
    candidate = int(candidate_index)
    nonbaseline = np.asarray(
        [index for index in range(len(predictions.meta_score)) if index != baseline],
        dtype=np.int32,
    )
    others = nonbaseline[nonbaseline != candidate]
    meta = np.asarray(predictions.meta_score, dtype=np.float64)
    delta = np.asarray(predictions.base_delta, dtype=np.float64)
    stage18 = np.asarray(predictions.stage18_score, dtype=np.float64)
    candidate_meta = np.asarray(predictions.meta_features[candidate], dtype=np.float32)
    if candidate_meta.shape != (22,):
        raise ValueError("Attempt04 confidence requires the frozen 22 meta features")
    meta_margin = float(meta[candidate] - np.max(meta[others])) if others.size else 0.0
    base_margin = float(delta[candidate] - np.max(delta[others])) if others.size else 0.0
    scalars = np.asarray(
        [
            meta[candidate],
            meta_margin,
            float(np.mean(meta[nonbaseline] < meta[candidate])),
            base_margin,
            stage18[candidate] - stage18[baseline],
            float(np.sum(tail_pool_mask[nonbaseline])),
            float(np.mean(tail_pool_mask[nonbaseline])),
            float(bool(tail_pool_mask[candidate])),
            float(len(nonbaseline)),
        ],
        dtype=np.float32,
    )
    result = np.concatenate((candidate_meta, scalars)).astype(np.float32, copy=False)
    if result.shape != (M43_ATTEMPT04_CONFIDENCE_FEATURE_DIM,):
        raise AssertionError("Attempt04 confidence feature dimension changed")
    if not np.isfinite(result).all():
        raise ValueError("Attempt04 confidence features are non-finite")
    return result


def run_compact_confidence_pilot(
    data: DevData,
    *,
    rerank_inside_tail_pool: bool = True,
    include_direct_paired: bool = False,
) -> dict[str, Any]:
    fit_proposals = tuple(
        build_meta_tail_proposal(
            sample,
            prediction,
            rerank_inside_tail_pool=rerank_inside_tail_pool,
        )
        for sample, prediction in zip(
            data.fit_samples, data.fit_predictions, strict=True
        )
    )
    consumed_proposals = tuple(
        build_meta_tail_proposal(
            sample,
            prediction,
            rerank_inside_tail_pool=rerank_inside_tail_pool,
        )
        for sample, prediction in zip(
            data.consumed_samples, data.consumed_predictions, strict=True
        )
    )
    cushion_variants: dict[str, Any] = {}
    residuals = np.asarray(
        [
            (
                float(sample.downside_loss_p95[proposal.index])
                - float(prediction.downside_p95[proposal.index]),
                float(sample.downside_loss_p99[proposal.index])
                - float(prediction.downside_p99[proposal.index]),
                float(sample.downside_loss_max[proposal.index])
                - float(prediction.downside_max[proposal.index]),
            )
            for sample, prediction, proposal in zip(
                data.fit_samples,
                data.fit_predictions,
                fit_proposals,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    for quantile in (0.90, 0.95, 0.975):
        cushions = tuple(
            max(0.0, float(np.quantile(residuals[:, column], quantile, method="higher")))
            for column in range(3)
        )
        variant_fit = tuple(
            build_meta_tail_proposal(
                sample,
                prediction,
                tail_cushions=cushions,
                rerank_inside_tail_pool=rerank_inside_tail_pool,
            )
            for sample, prediction in zip(
                data.fit_samples, data.fit_predictions, strict=True
            )
        )
        variant_consumed = tuple(
            build_meta_tail_proposal(
                sample,
                prediction,
                tail_cushions=cushions,
                rerank_inside_tail_pool=rerank_inside_tail_pool,
            )
            for sample, prediction in zip(
                data.consumed_samples, data.consumed_predictions, strict=True
            )
        )
        cushion_variants[f"q{quantile:g}"] = {
            "residual_quantile": quantile,
            "tail_cushions": list(cushions),
            **_confidence_comparison(
                data,
                fit_proposals=variant_fit,
                consumed_proposals=variant_consumed,
            ),
        }
    fit_x = np.vstack([proposal.confidence_features for proposal in fit_proposals])
    fit_y = np.asarray(
        [
            _safe_label(sample, proposal.index)
            for sample, proposal in zip(data.fit_samples, fit_proposals, strict=True)
        ],
        dtype=np.int8,
    )
    consumed_x = np.vstack(
        [proposal.confidence_features for proposal in consumed_proposals]
    )
    split_gain_tail = _split_gain_tail_comparison(
        data,
        fit_proposals=fit_proposals,
        consumed_proposals=consumed_proposals,
    )

    model_reports: dict[str, Any] = {}
    for kind in ("l2_logistic", "lightgbm_binary"):
        oof_probability = np.full(len(data.fit_samples), np.nan, dtype=np.float64)
        for fold in range(5):
            train = np.asarray(
                [index for index, value in enumerate(data.fit_fold_ids) if value != fold],
                dtype=np.int32,
            )
            validation = np.asarray(
                [index for index, value in enumerate(data.fit_fold_ids) if value == fold],
                dtype=np.int32,
            )
            estimator = _fit_confidence(kind, fit_x[train], fit_y[train], seed=2026072601 + fold)
            oof_probability[validation] = _positive_probability(
                estimator, fit_x[validation]
            )
        if not np.isfinite(oof_probability).all():
            raise AssertionError("Attempt04 grouped confidence OOF coverage incomplete")
        final_estimator = _fit_confidence(kind, fit_x, fit_y, seed=2026072699)
        consumed_probability = _positive_probability(final_estimator, consumed_x)
        fit_table = _threshold_table(
            data.fit_samples,
            data.fit_profiles,
            fit_proposals,
            oof_probability,
        )
        consumed_table = _threshold_table(
            data.consumed_samples,
            data.consumed_profiles,
            consumed_proposals,
            consumed_probability,
        )
        model_reports[kind] = {
            "confidence_fit_target": (
                "teacher_delta_gt_zero_and_actual_p95_p99_max_within_limits"
            ),
            "fit_positive_labels": int(np.sum(fit_y)),
            "fit_grouped_oof": fit_table,
            "consumed_attempt03_precal_development_only": consumed_table,
            "fit_validation_identity_excluded_from_direct_confidence_fit": True,
            "fully_nested_upstream_meta_oof_claimed": False,
        }

    fit_raw = _metrics(
        data.fit_samples,
        data.fit_profiles,
        [proposal.index for proposal in fit_proposals],
        np.ones(len(fit_proposals), dtype=bool),
    )
    consumed_raw = _metrics(
        data.consumed_samples,
        data.consumed_profiles,
        [proposal.index for proposal in consumed_proposals],
        np.ones(len(consumed_proposals), dtype=bool),
    )
    report = {
        "schema": M43_ATTEMPT04_DEV_PILOT_SCHEMA,
        "status": "development_comparison_complete_fresh_precal_still_required",
        "proposal": {
            "kind": (
                "huber_meta_argmax_inside_predicted_tail_pool"
                if rerank_inside_tail_pool
                else "huber_meta_all_argmax_then_predicted_tail_veto"
            ),
            "tie_break": "meta_score_then_base_delta_then_action_key",
            "tail_limits": list(M43_ATTEMPT04_TAIL_LIMITS),
            "fold_absolute_zero_vote_gate": False,
            "direct_delta_only": False,
            "rerank_inside_tail_pool": bool(rerank_inside_tail_pool),
        },
        "raw_proposal": {
            "fit700": fit_raw,
            "consumed_attempt03_precal200": consumed_raw,
            "go_gate": (
                "fresh_only_positive_rate_overall_ge_0.40_and_each_profile_ge_0.30"
            ),
            "mean_and_lcb": "required_diagnostic_not_go_gate",
        },
        "confidence_feature_schema": M43_ATTEMPT04_CONFIDENCE_FEATURE_SCHEMA,
        "confidence_feature_dim": M43_ATTEMPT04_CONFIDENCE_FEATURE_DIM,
        "confidence_thresholds": list(M43_ATTEMPT04_CONFIDENCE_THRESHOLDS),
        "confidence_models": model_reports,
        "split_gain_tail_models": split_gain_tail,
        "conformal_tail_cushion_variants": cushion_variants,
        "development_sources": data.source_manifest,
        "claim_boundary": {
            "dev900_is_fresh_generalization": False,
            "consumed_attempt03_precal_reused_as_fresh_gate": False,
            "fresh_attempt04_precal_is_sole_model_level_gate": True,
            "sealed_calibration_opened": False,
            "locked_holdout_opened": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    }
    if include_direct_paired:
        report["direct_paired_proposal_comparison"] = (
            run_direct_paired_proposal_pilot(data)
        )
    return report


def run_direct_paired_proposal_pilot(data: DevData) -> dict[str, Any]:
    """Reproduce the direct paired-delta proposal with grouped state OOF."""

    fit_features = tuple(_paired_features(sample) for sample in data.fit_samples)
    consumed_features = tuple(
        _paired_features(sample) for sample in data.consumed_samples
    )
    row_features = []
    row_targets = []
    row_weights = []
    row_states = []
    row_tail_safe = []
    for state, (sample, features) in enumerate(
        zip(data.fit_samples, fit_features, strict=True)
    ):
        nonbaseline = [
            index for index in range(len(sample.teacher_scores)) if index != sample.baseline_index
        ]
        row_features.append(features[nonbaseline])
        row_targets.append(
            np.asarray(sample.teacher_paired_delta_mean, dtype=np.float64)[nonbaseline]
        )
        row_weights.append(
            np.full(len(nonbaseline), 1.0 / len(nonbaseline), dtype=np.float64)
        )
        row_states.append(np.full(len(nonbaseline), state, dtype=np.int32))
        row_tail_safe.append(
            np.asarray(
                [
                    all(
                        value <= limit
                        for value, limit in zip(
                            (
                                float(sample.downside_loss_p95[index]),
                                float(sample.downside_loss_p99[index]),
                                float(sample.downside_loss_max[index]),
                            ),
                            M43_ATTEMPT04_TAIL_LIMITS,
                            strict=True,
                        )
                    )
                    for index in nonbaseline
                ],
                dtype=np.int8,
            )
        )
    x = np.vstack(row_features).astype(np.float32, copy=False)
    y = np.concatenate(row_targets)
    weights = np.concatenate(row_weights)
    states = np.concatenate(row_states)
    gain_labels = (y > 0.0).astype(np.int8)
    tail_labels = np.concatenate(row_tail_safe)
    oof_scores: list[np.ndarray | None] = [None] * len(data.fit_samples)
    for fold in range(5):
        train_rows = np.asarray(
            [data.fit_fold_ids[int(state)] != fold for state in states], dtype=bool
        )
        estimator = _fit_direct_delta(
            x[train_rows], y[train_rows], weights[train_rows], seed=2026073301 + fold
        )
        for state, assigned in enumerate(data.fit_fold_ids):
            if assigned != fold:
                continue
            prediction = np.asarray(
                estimator.predict(fit_features[state]), dtype=np.float64
            )
            prediction[data.fit_samples[state].baseline_index] = 0.0
            oof_scores[state] = prediction
    if any(value is None for value in oof_scores):
        raise AssertionError("direct paired grouped OOF coverage incomplete")
    final = _fit_direct_delta(x, y, weights, seed=2026073399)
    consumed_scores = []
    for sample, features in zip(
        data.consumed_samples, consumed_features, strict=True
    ):
        prediction = np.asarray(final.predict(features), dtype=np.float64)
        prediction[sample.baseline_index] = 0.0
        consumed_scores.append(prediction)

    meta_fit_proposals = tuple(
        build_meta_tail_proposal(
            sample, prediction, rerank_inside_tail_pool=False
        )
        for sample, prediction in zip(
            data.fit_samples, data.fit_predictions, strict=True
        )
    )
    meta_consumed_proposals = tuple(
        build_meta_tail_proposal(
            sample, prediction, rerank_inside_tail_pool=False
        )
        for sample, prediction in zip(
            data.consumed_samples, data.consumed_predictions, strict=True
        )
    )
    action_gain_oof = np.full(len(data.fit_samples), np.nan, dtype=np.float64)
    action_tail_oof = np.full(len(data.fit_samples), np.nan, dtype=np.float64)
    for fold in range(5):
        train_rows = np.asarray(
            [data.fit_fold_ids[int(state)] != fold for state in states], dtype=bool
        )
        gain_model = _fit_action_classifier(
            x[train_rows],
            gain_labels[train_rows],
            weights[train_rows],
            seed=2026073401 + fold,
            class_weight_balanced=True,
        )
        tail_model = _fit_action_classifier(
            x[train_rows],
            tail_labels[train_rows],
            weights[train_rows],
            seed=2026073501 + fold,
            class_weight_balanced=True,
        )
        for state, assigned in enumerate(data.fit_fold_ids):
            if assigned != fold:
                continue
            candidate = meta_fit_proposals[state].index
            action_gain_oof[state] = _positive_probability(
                gain_model, fit_features[state]
            )[candidate]
            action_tail_oof[state] = _positive_probability(
                tail_model, fit_features[state]
            )[candidate]
    gain_final = _fit_action_classifier(
        x,
        gain_labels,
        weights,
        seed=2026073499,
        class_weight_balanced=True,
    )
    tail_final = _fit_action_classifier(
        x,
        tail_labels,
        weights,
        seed=2026073599,
        class_weight_balanced=True,
    )
    action_gain_consumed = np.asarray(
        [
            _positive_probability(gain_final, features)[proposal.index]
            for features, proposal in zip(
                consumed_features, meta_consumed_proposals, strict=True
            )
        ]
    )
    action_tail_consumed = np.asarray(
        [
            _positive_probability(tail_final, features)[proposal.index]
            for features, proposal in zip(
                consumed_features, meta_consumed_proposals, strict=True
            )
        ]
    )
    no_legacy_tail_fit = tuple(
        replace(proposal, tail_pool_eligible=True)
        for proposal in meta_fit_proposals
    )
    no_legacy_tail_consumed = tuple(
        replace(proposal, tail_pool_eligible=True)
        for proposal in meta_consumed_proposals
    )

    fit_predictions = tuple(
        replace(prediction, meta_score=np.asarray(score, dtype=np.float64))
        for prediction, score in zip(data.fit_predictions, oof_scores, strict=True)
    )
    consumed_predictions = tuple(
        replace(prediction, meta_score=np.asarray(score, dtype=np.float64))
        for prediction, score in zip(
            data.consumed_predictions, consumed_scores, strict=True
        )
    )
    fit_proposals = tuple(
        build_meta_tail_proposal(
            sample, prediction, rerank_inside_tail_pool=False
        )
        for sample, prediction in zip(
            data.fit_samples, fit_predictions, strict=True
        )
    )
    consumed_proposals = tuple(
        build_meta_tail_proposal(
            sample, prediction, rerank_inside_tail_pool=False
        )
        for sample, prediction in zip(
            data.consumed_samples, consumed_predictions, strict=True
        )
    )
    direct_data = replace(
        data,
        fit_predictions=fit_predictions,
        consumed_predictions=consumed_predictions,
    )
    return {
        "proposal": "direct_paired_delta_huber_argmax_then_tail_veto",
        "fit_base_identity_excluded": True,
        "upstream_fully_nested_confidence_claimed": False,
        "raw_fit700": _metrics(
            data.fit_samples,
            data.fit_profiles,
            [proposal.index for proposal in fit_proposals],
            np.ones(len(fit_proposals), dtype=bool),
        ),
        "raw_consumed200": _metrics(
            data.consumed_samples,
            data.consumed_profiles,
            [proposal.index for proposal in consumed_proposals],
            np.ones(len(consumed_proposals), dtype=bool),
        ),
        "confidence": _confidence_comparison(
            direct_data,
            fit_proposals=fit_proposals,
            consumed_proposals=consumed_proposals,
        ),
        "split_gain_tail_models": _split_gain_tail_comparison(
            direct_data,
            fit_proposals=fit_proposals,
            consumed_proposals=consumed_proposals,
        ),
        "meta_all_action_head_gain_tail_veto": {
            "proposal": "canonical_meta_all_raw_unchanged",
            "gain_head": "paired_action_lightgbm_binary_balanced",
            "tail_head": "paired_action_lightgbm_binary_balanced_all_three_limits",
            "legacy_predicted_tail_pool_used": False,
            "fit_grouped_oof": _two_threshold_table(
                data.fit_samples,
                data.fit_profiles,
                no_legacy_tail_fit,
                action_gain_oof,
                action_tail_oof,
            ),
            "consumed_attempt03_precal_development_only": _two_threshold_table(
                data.consumed_samples,
                data.consumed_profiles,
                no_legacy_tail_consumed,
                action_gain_consumed,
                action_tail_consumed,
            ),
        },
    }


def run_final_meta_all_gain_conformal_comparison(data: DevData) -> dict[str, Any]:
    """Run the last bounded pre-fresh family comparison on consumed dev900.

    The raw action remains the canonical meta-all proposal.  A gain-only model
    may abstain, and a separately fitted one-sided conformal cushion may veto
    unsafe predicted tails.  Root profile provenance is used only to construct
    training weights; it is never an input feature or a runtime dependency.

    This is intentionally a fixed 2 x 2 x 3 x 13 comparison.  It is development
    evidence, not a fresh generalization result, and it must not be expanded or
    retuned against the already-consumed Attempt03 pre-calibration rows.
    """

    samples = (*data.fit_samples, *data.consumed_samples)
    profiles = (*data.fit_profiles, *data.consumed_profiles)
    predictions = (*data.fit_predictions, *data.consumed_predictions)
    proposals = tuple(
        build_meta_tail_proposal(
            sample, prediction, rerank_inside_tail_pool=False
        )
        for sample, prediction in zip(samples, predictions, strict=True)
    )
    if len(samples) != 900:
        raise AssertionError("Attempt04 final dev comparison requires dev900")

    # Only the paired action representation of the already-selected meta-all
    # candidate is visible to the gain head.  Profile/source provenance is not.
    features = np.vstack(
        [
            _paired_features(sample)[proposal.index]
            for sample, proposal in zip(samples, proposals, strict=True)
        ]
    ).astype(np.float32, copy=False)
    gain_labels = np.asarray(
        [
            float(sample.teacher_paired_delta_mean[proposal.index]) > 0.0
            for sample, proposal in zip(samples, proposals, strict=True)
        ],
        dtype=np.int8,
    )
    predicted_tails = np.asarray(
        [
            (
                float(prediction.downside_p95[proposal.index]),
                float(prediction.downside_p99[proposal.index]),
                float(prediction.downside_max[proposal.index]),
            )
            for prediction, proposal in zip(predictions, proposals, strict=True)
        ],
        dtype=np.float64,
    )
    actual_tails = np.asarray(
        [
            (
                float(sample.downside_loss_p95[proposal.index]),
                float(sample.downside_loss_p99[proposal.index]),
                float(sample.downside_loss_max[proposal.index]),
            )
            for sample, proposal in zip(samples, proposals, strict=True)
        ],
        dtype=np.float64,
    )
    fold_ids = _balanced_identity_fold_ids(samples, profiles, folds=5)
    selections = [proposal.index for proposal in proposals]
    reports: dict[str, Any] = {}

    for kind in ("l2_logistic", "lightgbm_binary"):
        for weighting in ("class_balanced", "profile_x_class_balanced"):
            key = f"{kind}__{weighting}"
            gain_oof = np.full(len(samples), np.nan, dtype=np.float64)
            tail_oof = {
                quantile: np.zeros(len(samples), dtype=bool)
                for quantile in M43_ATTEMPT04_FINAL_DEV_CONFORMAL_QUANTILES
            }
            fold_cushions: dict[str, list[list[float]]] = {
                f"q{quantile:g}": []
                for quantile in M43_ATTEMPT04_FINAL_DEV_CONFORMAL_QUANTILES
            }
            for fold in range(5):
                train = fold_ids != fold
                validation = fold_ids == fold
                train_weights = _gain_training_weights(
                    gain_labels[train],
                    np.asarray(profiles, dtype=object)[train],
                    weighting=weighting,
                )
                estimator = _fit_gain_only_classifier(
                    kind,
                    features[train],
                    gain_labels[train],
                    train_weights,
                    seed=2026073701 + 100 * (kind == "lightgbm_binary") + fold,
                )
                gain_oof[validation] = _positive_probability(
                    estimator, features[validation]
                )
                residuals = actual_tails[train] - predicted_tails[train]
                for quantile in M43_ATTEMPT04_FINAL_DEV_CONFORMAL_QUANTILES:
                    cushions = np.asarray(
                        [
                            max(
                                0.0,
                                float(
                                    np.quantile(
                                        residuals[:, column],
                                        quantile,
                                        method="higher",
                                    )
                                ),
                            )
                            for column in range(3)
                        ],
                        dtype=np.float64,
                    )
                    tail_oof[quantile][validation] = np.all(
                        predicted_tails[validation] + cushions
                        <= np.asarray(M43_ATTEMPT04_TAIL_LIMITS),
                        axis=1,
                    )
                    fold_cushions[f"q{quantile:g}"].append(cushions.tolist())
            if not np.isfinite(gain_oof).all():
                raise AssertionError("Attempt04 dev900 gain OOF coverage incomplete")

            quantile_reports: dict[str, Any] = {}
            for quantile in M43_ATTEMPT04_FINAL_DEV_CONFORMAL_QUANTILES:
                rows = []
                for threshold in M43_ATTEMPT04_CONFIDENCE_THRESHOLDS:
                    fired = tail_oof[quantile] & (gain_oof >= threshold)
                    row = _metrics(samples, profiles, selections, fired)
                    row.update(
                        {
                            "gain_threshold": float(threshold),
                            "conformal_quantile": float(quantile),
                            "passes_scaled_dev900_gate": _passes_scaled_dev900_gate(
                                row
                            ),
                        }
                    )
                    rows.append(row)
                quantile_reports[f"q{quantile:g}"] = {
                    "fold_tail_cushions": fold_cushions[f"q{quantile:g}"],
                    "rows": rows,
                }
            reports[key] = {
                "classifier": kind,
                "training_weighting": weighting,
                "runtime_profile_feature": False,
                "gain_target_positive_rate": float(np.mean(gain_labels)),
                "quantiles": quantile_reports,
            }

    eligible = []
    for key, model_report in reports.items():
        for quantile_key, quantile_report in model_report["quantiles"].items():
            for row in quantile_report["rows"]:
                if row["passes_scaled_dev900_gate"]:
                    eligible.append(
                        {
                            "model": key,
                            "quantile": quantile_key,
                            **row,
                        }
                    )
    eligible.sort(
        key=lambda row: (
            -min(value["fires"] for value in row["profile"].values()),
            -float(row["mean_delta_per_state"]),
            float(row["false_positive_rate"]),
            str(row["model"]),
            str(row["quantile"]),
            float(row["gain_threshold"]),
        )
    )
    return {
        "schema": "hu_m43_attempt04_final_bounded_dev900_comparison_v1",
        "status": (
            "development_go_candidate_exists_fresh_precal_required"
            if eligible
            else "development_no_go_more_consumed_family_search_forbidden"
        ),
        "raw_proposal": "canonical_meta_all_huber_argmax_unchanged",
        "gain_head_input": "paired_action_features_of_meta_all_candidate",
        "tail_gate": "fold_train_only_one_sided_conformal_residual_cushion",
        "folding": "profile_stratified_identity_hash_round_robin_5fold",
        "exploration_budget": {
            "classifier_families": 2,
            "training_weight_schemes": 2,
            "conformal_quantiles": list(
                M43_ATTEMPT04_FINAL_DEV_CONFORMAL_QUANTILES
            ),
            "absolute_gain_thresholds": list(M43_ATTEMPT04_CONFIDENCE_THRESHOLDS),
            "total_rows": 156,
            "additional_family_or_feature_retuning_on_dev900_allowed": False,
        },
        "scaled_dev900_gate": {
            "fires_min": 90,
            "each_profile_fires_min": 9,
            "each_profile_mean_delta_per_fire_gte": 0.0,
            "each_profile_mean_delta_per_state_gte": 0.0,
            "false_positive_rate_max": 0.30,
            "mean_delta_per_state_gt": 0.0,
            "cluster_lcb90_per_fire_gt": 0.0,
            "actual_selected_tail_maxima": dict(
                zip(("p95", "p99", "max"), M43_ATTEMPT04_TAIL_LIMITS, strict=True)
            ),
        },
        "eligible_count": len(eligible),
        "pre_registered_selection_order": (
            "max_min_profile_fires_then_mean_delta_per_state_then_lower_fp"
        ),
        "selected_development_candidate": eligible[0] if eligible else None,
        "all_eligible": eligible,
        "models": reports,
        "claim_boundary": {
            "dev900_is_fresh_generalization": False,
            "upstream_meta_is_fully_nested": False,
            "fresh_attempt04_precal_is_sole_model_level_gate": True,
            "teacher_values_reported_as_realized_ev": False,
            "sealed_calibration_opened": False,
            "locked_holdout_opened": False,
            "runtime_policy_activated": False,
        },
    }


def _balanced_identity_fold_ids(
    samples: Sequence[PreparedTeacherSample],
    profiles: Sequence[str],
    *,
    folds: int,
) -> np.ndarray:
    result = np.full(len(samples), -1, dtype=np.int8)
    for profile in M43_ATTEMPT04_PROFILES:
        indices = [index for index, value in enumerate(profiles) if value == profile]
        indices.sort(
            key=lambda index: hashlib.sha256(
                samples[index].observation_fingerprint.encode("utf-8")
            ).digest()
        )
        for rank, index in enumerate(indices):
            result[index] = rank % folds
    if np.any(result < 0):
        raise AssertionError("Attempt04 balanced fold assignment incomplete")
    return result


def _gain_training_weights(
    labels: np.ndarray,
    profiles: np.ndarray,
    *,
    weighting: str,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    profiles = np.asarray(profiles, dtype=object)
    if weighting == "class_balanced":
        keys = [(int(label),) for label in labels]
    elif weighting == "profile_x_class_balanced":
        keys = [
            (str(profile), int(label))
            for profile, label in zip(profiles, labels, strict=True)
        ]
    else:
        raise ValueError(f"unknown Attempt04 gain weighting: {weighting}")
    counts: dict[tuple[Any, ...], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)
    return weights / np.mean(weights)


def _fit_gain_only_classifier(
    kind: str,
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
) -> Any:
    if np.unique(labels).size == 1:
        return _ConstantConfidence(float(labels[0]))
    if kind == "l2_logistic":
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.25,
                solver="lbfgs",
                max_iter=1000,
                random_state=seed,
            ),
        )
        estimator.fit(
            features,
            labels,
            logisticregression__sample_weight=weights,
        )
        return estimator
    if kind == "lightgbm_binary":
        return LGBMClassifier(
            objective="binary",
            n_estimators=160,
            learning_rate=0.035,
            num_leaves=7,
            min_child_samples=50,
            reg_lambda=8.0,
            reg_alpha=0.5,
            random_state=seed,
            n_jobs=2,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ).fit(features, labels, sample_weight=weights)
    raise ValueError(f"unknown Attempt04 gain classifier: {kind}")


def _passes_scaled_dev900_gate(row: Mapping[str, Any]) -> bool:
    tails = row["actual_selected_tail_maxima"]
    return bool(
        int(row["fires"]) >= 90
        and all(
            int(value["fires"]) >= 9
            and value["mean_delta_per_fire"] is not None
            and float(value["mean_delta_per_fire"]) >= 0.0
            and value["mean_delta_per_state"] is not None
            and float(value["mean_delta_per_state"]) >= 0.0
            for value in row["profile"].values()
        )
        and row["false_positive_rate"] is not None
        and float(row["false_positive_rate"]) <= 0.30
        and float(row["mean_delta_per_state"]) > 0.0
        and row["cluster_lcb90_per_fire"] is not None
        and float(row["cluster_lcb90_per_fire"]) > 0.0
        and all(
            tails[name] is not None and float(tails[name]) <= limit
            for name, limit in zip(
                ("p95", "p99", "max"), M43_ATTEMPT04_TAIL_LIMITS, strict=True
            )
        )
    )


def _paired_features(sample: PreparedTeacherSample) -> np.ndarray:
    matrix, _targets = sample_to_matrix(sample.policy_sample)
    return build_paired_action_features_matrix(
        matrix, baseline_index=sample.baseline_index
    )


def _fit_direct_delta(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
) -> LGBMRegressor:
    return LGBMRegressor(
        objective="huber",
        alpha=0.90,
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=7,
        min_child_samples=50,
        reg_lambda=8.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    ).fit(features, targets, sample_weight=weights)


def _fit_action_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    class_weight_balanced: bool,
) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=160,
        learning_rate=0.035,
        num_leaves=7,
        min_child_samples=50,
        reg_lambda=8.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        class_weight="balanced" if class_weight_balanced else None,
    ).fit(features, labels, sample_weight=weights)


def _split_gain_tail_comparison(
    data: DevData,
    *,
    fit_proposals: Sequence[MetaTailProposal],
    consumed_proposals: Sequence[MetaTailProposal],
) -> dict[str, Any]:
    fit_x = np.vstack([proposal.confidence_features for proposal in fit_proposals])
    consumed_x = np.vstack(
        [proposal.confidence_features for proposal in consumed_proposals]
    )
    fit_gain = np.asarray(
        [
            float(sample.teacher_paired_delta_mean[proposal.index]) > 0.0
            for sample, proposal in zip(data.fit_samples, fit_proposals, strict=True)
        ],
        dtype=np.int8,
    )
    fit_tail = np.asarray(
        [
            all(
                value <= limit
                for value, limit in zip(
                    (
                        float(sample.downside_loss_p95[proposal.index]),
                        float(sample.downside_loss_p99[proposal.index]),
                        float(sample.downside_loss_max[proposal.index]),
                    ),
                    M43_ATTEMPT04_TAIL_LIMITS,
                    strict=True,
                )
            )
            for sample, proposal in zip(data.fit_samples, fit_proposals, strict=True)
        ],
        dtype=np.int8,
    )
    reports = {}
    for kind in ("l2_logistic", "lightgbm_binary"):
        gain_oof = np.full(len(fit_x), np.nan, dtype=np.float64)
        tail_oof = np.full(len(fit_x), np.nan, dtype=np.float64)
        for fold in range(5):
            train = np.asarray(
                [i for i, value in enumerate(data.fit_fold_ids) if value != fold],
                dtype=np.int32,
            )
            validation = np.asarray(
                [i for i, value in enumerate(data.fit_fold_ids) if value == fold],
                dtype=np.int32,
            )
            gain_model = _fit_confidence(
                kind, fit_x[train], fit_gain[train], seed=2026073001 + fold
            )
            tail_model = _fit_confidence(
                kind,
                fit_x[train],
                fit_tail[train],
                seed=2026073101 + fold,
                class_weight_balanced=True,
            )
            gain_oof[validation] = _positive_probability(
                gain_model, fit_x[validation]
            )
            tail_oof[validation] = _positive_probability(
                tail_model, fit_x[validation]
            )
        gain_final = _fit_confidence(
            kind, fit_x, fit_gain, seed=2026073099
        )
        tail_final = _fit_confidence(
            kind,
            fit_x,
            fit_tail,
            seed=2026073199,
            class_weight_balanced=True,
        )
        reports[kind] = {
            "gain_target_positive_rate": float(np.mean(fit_gain)),
            "tail_target_safe_rate": float(np.mean(fit_tail)),
            "fit_grouped_oof": _two_threshold_table(
                data.fit_samples,
                data.fit_profiles,
                fit_proposals,
                gain_oof,
                tail_oof,
            ),
            "consumed_attempt03_precal_development_only": _two_threshold_table(
                data.consumed_samples,
                data.consumed_profiles,
                consumed_proposals,
                _positive_probability(gain_final, consumed_x),
                _positive_probability(tail_final, consumed_x),
            ),
        }
    return reports


def _two_threshold_table(
    samples: Sequence[PreparedTeacherSample],
    profiles: Sequence[str],
    proposals: Sequence[MetaTailProposal],
    gain_probability: np.ndarray,
    tail_probability: np.ndarray,
) -> list[dict[str, Any]]:
    selections = [proposal.index for proposal in proposals]
    base_eligible = np.asarray(
        [proposal.tail_pool_eligible for proposal in proposals], dtype=bool
    )
    rows = []
    for gain_threshold in M43_ATTEMPT04_CONFIDENCE_THRESHOLDS:
        for tail_threshold in M43_ATTEMPT04_CONFIDENCE_THRESHOLDS:
            fired = (
                base_eligible
                & (gain_probability >= gain_threshold)
                & (tail_probability >= tail_threshold)
            )
            metrics = _metrics(samples, profiles, selections, fired)
            metrics["gain_threshold"] = float(gain_threshold)
            metrics["tail_threshold"] = float(tail_threshold)
            rows.append(metrics)
    return rows


def _confidence_comparison(
    data: DevData,
    *,
    fit_proposals: Sequence[MetaTailProposal],
    consumed_proposals: Sequence[MetaTailProposal],
) -> dict[str, Any]:
    fit_x = np.vstack([proposal.confidence_features for proposal in fit_proposals])
    fit_y = np.asarray(
        [
            _safe_label(sample, proposal.index)
            for sample, proposal in zip(data.fit_samples, fit_proposals, strict=True)
        ],
        dtype=np.int8,
    )
    consumed_x = np.vstack(
        [proposal.confidence_features for proposal in consumed_proposals]
    )
    reports = {}
    for kind in ("l2_logistic", "lightgbm_binary"):
        oof_probability = np.full(len(data.fit_samples), np.nan, dtype=np.float64)
        for fold in range(5):
            train = np.asarray(
                [i for i, value in enumerate(data.fit_fold_ids) if value != fold],
                dtype=np.int32,
            )
            validation = np.asarray(
                [i for i, value in enumerate(data.fit_fold_ids) if value == fold],
                dtype=np.int32,
            )
            estimator = _fit_confidence(
                kind, fit_x[train], fit_y[train], seed=2026072801 + fold
            )
            oof_probability[validation] = _positive_probability(
                estimator, fit_x[validation]
            )
        final = _fit_confidence(kind, fit_x, fit_y, seed=2026072899)
        reports[kind] = {
            "fit_grouped_oof": _threshold_table(
                data.fit_samples,
                data.fit_profiles,
                fit_proposals,
                oof_probability,
            ),
            "consumed_attempt03_precal_development_only": _threshold_table(
                data.consumed_samples,
                data.consumed_profiles,
                consumed_proposals,
                _positive_probability(final, consumed_x),
            ),
        }
    scalar_reports = {}
    for name, fit_score, consumed_score in _scalar_confidence_scores(
        fit_proposals, consumed_proposals
    ):
        scalar_reports[name] = {
            "fit700": _threshold_table(
                data.fit_samples,
                data.fit_profiles,
                fit_proposals,
                fit_score,
            ),
            "consumed_attempt03_precal_development_only": _threshold_table(
                data.consumed_samples,
                data.consumed_profiles,
                consumed_proposals,
                consumed_score,
            ),
        }
    return {
        "raw_fit700": _metrics(
            data.fit_samples,
            data.fit_profiles,
            [proposal.index for proposal in fit_proposals],
            np.ones(len(fit_proposals), dtype=bool),
        ),
        "raw_consumed200": _metrics(
            data.consumed_samples,
            data.consumed_profiles,
            [proposal.index for proposal in consumed_proposals],
            np.ones(len(consumed_proposals), dtype=bool),
        ),
        "confidence_models": reports,
        "fixed_formula_confidence": scalar_reports,
    }


def _scalar_confidence_scores(
    fit_proposals: Sequence[MetaTailProposal],
    consumed_proposals: Sequence[MetaTailProposal],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    def blocks(proposals: Sequence[MetaTailProposal]) -> tuple[np.ndarray, ...]:
        matrix = np.vstack([proposal.confidence_features for proposal in proposals])
        positive = np.clip(matrix[:, 1].astype(np.float64), 0.0, 1.0)
        delta = 1.0 / (1.0 + np.exp(-np.clip(matrix[:, 0] / 2.0, -40.0, 40.0)))
        meta = 1.0 / (1.0 + np.exp(-np.clip(matrix[:, 22] / 2.0, -40.0, 40.0)))
        return positive, delta, meta, 0.5 * (positive + meta), np.minimum(positive, meta)

    fit = blocks(fit_proposals)
    consumed = blocks(consumed_proposals)
    names = (
        "base_positive",
        "base_delta_sigmoid_scale2",
        "meta_score_sigmoid_scale2",
        "base_positive_meta_mean",
        "base_positive_meta_min",
    )
    return [
        (name, fit[index], consumed[index]) for index, name in enumerate(names)
    ]


def _threshold_table(
    samples: Sequence[PreparedTeacherSample],
    profiles: Sequence[str],
    proposals: Sequence[MetaTailProposal],
    probability: np.ndarray,
) -> list[dict[str, Any]]:
    result = []
    selections = [proposal.index for proposal in proposals]
    tail_eligible = np.asarray(
        [proposal.tail_pool_eligible for proposal in proposals], dtype=bool
    )
    for threshold in M43_ATTEMPT04_CONFIDENCE_THRESHOLDS:
        fired = tail_eligible & (probability >= threshold)
        row = _metrics(samples, profiles, selections, fired)
        row["threshold"] = float(threshold)
        result.append(row)
    return result


def _metrics(
    samples: Sequence[PreparedTeacherSample],
    profiles: Sequence[str],
    selections: Sequence[int],
    fired: np.ndarray,
) -> dict[str, Any]:
    flags = np.asarray(fired, dtype=bool)
    deltas = np.asarray(
        [
            float(sample.teacher_paired_delta_mean[index])
            for sample, index in zip(samples, selections, strict=True)
        ],
        dtype=np.float64,
    )
    selected = deltas[flags]
    zeros = np.where(flags, deltas, 0.0)
    fire_count = int(np.sum(flags))
    false_positives = int(np.sum(selected <= 0.0))
    tails = _tail_maxima(samples, selections, flags)
    by_profile = {}
    for profile in M43_ATTEMPT04_PROFILES:
        mask = np.asarray([value == profile for value in profiles], dtype=bool)
        profile_selected = deltas[flags & mask]
        profile_states = int(np.sum(mask))
        by_profile[profile] = {
            "states": profile_states,
            "fires": int(np.sum(flags & mask)),
            "positive_rate": (
                float(np.mean(profile_selected > 0.0))
                if profile_selected.size
                else None
            ),
            "mean_delta_per_fire": (
                float(np.mean(profile_selected)) if profile_selected.size else None
            ),
            "mean_delta_per_state": (
                float(np.sum(profile_selected) / profile_states)
                if profile_states
                else None
            ),
        }
    return {
        "states": len(samples),
        "fires": fire_count,
        "positive_fires": int(np.sum(selected > 0.0)),
        "positive_rate": float(np.mean(selected > 0.0)) if selected.size else None,
        "false_positive_rate": (
            float(false_positives / fire_count) if fire_count else None
        ),
        "mean_delta_per_fire": float(np.mean(selected)) if selected.size else None,
        "mean_delta_per_state": float(np.mean(zeros)),
        "cluster_lcb90_per_fire": _lcb90(selected),
        "cluster_lcb90_per_state": _lcb90(zeros),
        "actual_selected_tail_maxima": tails,
        "profile": by_profile,
    }


def _fit_confidence(
    kind: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    class_weight_balanced: bool = False,
) -> Any:
    if np.unique(y).size == 1:
        return _ConstantConfidence(float(y[0]))
    if kind == "l2_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.25,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=seed,
                class_weight="balanced" if class_weight_balanced else None,
            ),
        ).fit(x, y)
    if kind == "lightgbm_binary":
        return LGBMClassifier(
            objective="binary",
            n_estimators=120,
            learning_rate=0.035,
            num_leaves=7,
            min_child_samples=50,
            reg_lambda=8.0,
            reg_alpha=0.5,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            class_weight="balanced" if class_weight_balanced else None,
        ).fit(x, y)
    raise ValueError(f"unknown Attempt04 confidence model: {kind}")


def _positive_probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = list(getattr(estimator, "classes_", ()))
    if not classes and hasattr(estimator, "named_steps"):
        classes = list(estimator.named_steps["logisticregression"].classes_)
    if raw.shape[0] != len(features) or 1 not in classes:
        raise ValueError("Attempt04 confidence probability contract changed")
    result = raw[:, classes.index(1)]
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("Attempt04 confidence probability is invalid")
    return result


def _meta_base_action_key_argmax(
    sample: PreparedTeacherSample,
    predictions: V5ActionPredictions,
    candidates: np.ndarray,
) -> int:
    indices = [int(value) for value in candidates]
    meta = np.asarray(predictions.meta_score, dtype=np.float64)
    base = np.asarray(predictions.base_delta, dtype=np.float64)
    best_meta = max(float(meta[index]) for index in indices)
    meta_tied = [index for index in indices if float(meta[index]) == best_meta]
    best_base = max(float(base[index]) for index in meta_tied)
    base_tied = [index for index in meta_tied if float(base[index]) == best_base]
    return min(
        base_tied,
        key=lambda index: action_key_from_payload(
            sample.policy_sample["actions"][index]
        ).sort_key(),
    )


def _safe_label(sample: PreparedTeacherSample, index: int) -> int:
    delta = float(sample.teacher_paired_delta_mean[index])
    tails = (
        float(sample.downside_loss_p95[index]),
        float(sample.downside_loss_p99[index]),
        float(sample.downside_loss_max[index]),
    )
    return int(
        delta > 0.0
        and all(value <= limit for value, limit in zip(tails, M43_ATTEMPT04_TAIL_LIMITS, strict=True))
    )


def _tail_maxima(
    samples: Sequence[PreparedTeacherSample],
    selections: Sequence[int],
    fired: np.ndarray,
) -> dict[str, float | None]:
    result = []
    for attribute in ("downside_loss_p95", "downside_loss_p99", "downside_loss_max"):
        values = [
            float(getattr(sample, attribute)[index])
            for sample, index, flag in zip(samples, selections, fired, strict=True)
            if flag
        ]
        result.append(max(values) if values else None)
    return dict(zip(("p95", "p99", "max"), result, strict=True))


def _lcb90(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return None
    standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size))
    return float(
        np.mean(array) - student_t.ppf(0.90, array.size - 1) * standard_error
    )


def _profile(row: Mapping[str, Any]) -> str:
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Attempt04 dev row lacks provenance")
    value = str(provenance.get("root_profile", ""))
    if value not in M43_ATTEMPT04_PROFILES:
        raise ValueError(f"Attempt04 dev row has invalid root profile: {value!r}")
    return value


def _validate_identity_and_profiles(
    samples: Sequence[PreparedTeacherSample], profiles: Sequence[str]
) -> None:
    if len(samples) != 900 or len(profiles) != 900:
        raise ValueError("Attempt04 dev identity audit requires exactly 900 states")
    fingerprints = [sample.observation_fingerprint for sample in samples]
    if len(set(fingerprints)) != 900:
        raise ValueError("Attempt04 dev observation identity overlap detected")
    seeds: set[str] = set()
    for sample in samples:
        if seeds & sample.root_seed_values:
            raise ValueError("Attempt04 dev root/hand seed overlap detected")
        seeds.update(sample.root_seed_values)
    counts = {profile: profiles.count(profile) for profile in M43_ATTEMPT04_PROFILES}
    if any(value != 180 for value in counts.values()):
        raise ValueError(f"Attempt04 dev profile balance changed: {counts}")
    if {sample.seat for sample in samples} != {"second"}:
        raise ValueError("Attempt04 dev pilot is scoped to T1 second seat")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("compact-confidence-pilot")
    pilot.add_argument("--attempt02-train", type=Path, default=DEFAULT_ATTEMPT02_TRAIN)
    pilot.add_argument("--attempt03-fit", type=Path, default=DEFAULT_ATTEMPT03_FIT)
    pilot.add_argument(
        "--consumed-attempt03-precal",
        type=Path,
        default=DEFAULT_ATTEMPT03_CONSUMED_PRECAL,
    )
    pilot.add_argument("--include-direct-paired", action="store_true")
    pilot.add_argument("--final-bounded-comparison", action="store_true")
    pilot.add_argument("--fit-bundle", type=Path, default=DEFAULT_ATTEMPT03_FIT_BUNDLE)
    pilot.add_argument(
        "--proposal-mode",
        choices=("meta-tail-rerank", "meta-all-tail-veto"),
        default="meta-tail-rerank",
    )
    pilot.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command != "compact-confidence-pilot":
        raise AssertionError("unhandled Attempt04 command")
    data = load_dev900(
        attempt02_train=args.attempt02_train,
        attempt03_fit=args.attempt03_fit,
        consumed_precal=args.consumed_attempt03_precal,
        fit_bundle=args.fit_bundle,
    )
    report = run_compact_confidence_pilot(
        data,
        rerank_inside_tail_pool=args.proposal_mode == "meta-tail-rerank",
        include_direct_paired=bool(args.include_direct_paired),
    )
    if args.final_bounded_comparison:
        report["final_bounded_comparison"] = (
            run_final_meta_all_gain_conformal_comparison(data)
        )
    _write_json_no_clobber(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "M43_ATTEMPT04_CONFIDENCE_FEATURE_DIM",
    "M43_ATTEMPT04_CONFIDENCE_FEATURE_SCHEMA",
    "M43_ATTEMPT04_FINAL_MANIFEST_SCHEMA",
    "M43_ATTEMPT04_THRESHOLD_LOCK_SCHEMA",
    "MetaTailProposal",
    "build_confidence_features",
    "build_meta_tail_proposal",
    "load_dev900",
    "run_compact_confidence_pilot",
    "run_direct_paired_proposal_pilot",
    "run_final_meta_all_gain_conformal_comparison",
]
