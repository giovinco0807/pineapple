"""Attempt05 pre-registered A/B architecture comparison and smoke trainer.

This module may consume the already-open 900 development states only as one
identity-grouped five-fold architecture comparison.  It does not split out or
re-query the consumed Attempt03 pre-calibration 200, sweep runtime thresholds,
or make a fresh generalization claim.  After one winner is frozen, exactly one
new audit is permitted by the external lifecycle contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor

from .action_key import action_key_from_payload
from .hu_infoset import ActorObservation
from .hu_m43_attempt05_model import (
    HU_M43_ATTEMPT05_CONFORMAL_SCHEMA,
    HU_M43_ATTEMPT05_TAIL_LIMITS,
    HU_M43_ATTEMPT05_TAIL_SCALES,
    Attempt05FoldOutput,
    DeepSetsFoldPredictor,
    HuM43Attempt05Model,
    LambdaRankFoldPredictor,
    _runtime_policy_projection,
    build_deepsets_network,
    encode_deepsets_runtime_sample,
)
from .hu_m4_joint_model import build_paired_action_features_matrix
from .hu_turn3_model import sample_to_matrix
from .train_hu_m4_joint_model import (
    PreparedTeacherSample,
    prepare_teacher_samples,
    read_teacher_jsonl,
)


ATTEMPT05_DEV_SCHEMA = "hu_m43_attempt05_old_dev900_architecture_comparison_v1"
ATTEMPT05_OOF_SCHEMA = "hu_m43_attempt05_identity_grouped_oof_v1"
ATTEMPT05_SMOKE_SCHEMA = "hu_m43_attempt05_correctness_smoke_fit_v1"
ATTEMPT05_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
ATTEMPT05_FOLDS = 5
ATTEMPT05_CONFORMAL_QUANTILE = 0.95
ATTEMPT05_TAIL_PINBALL_QUANTILE = 0.80
ATTEMPT05_FAMILIES = ("lambda_rank", "deepsets")

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


@dataclass(frozen=True)
class Attempt05DevData:
    samples: tuple[PreparedTeacherSample, ...]
    profiles: tuple[str, ...]
    source_roles: tuple[str, ...]
    fold_ids: tuple[int, ...]
    source_manifest: Mapping[str, Any]


@dataclass(frozen=True)
class _ConstantProbability:
    probability: float
    classes_: tuple[int, int] = (0, 1)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features).shape[0]
        positive = np.full(rows, self.probability, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def load_attempt05_dev900(
    *,
    attempt02_train: str | Path = DEFAULT_ATTEMPT02_TRAIN,
    attempt03_fit: str | Path = DEFAULT_ATTEMPT03_FIT,
    consumed_precal: str | Path = DEFAULT_ATTEMPT03_CONSUMED_PRECAL,
) -> Attempt05DevData:
    """Load the immutable old dev900 as one comparison set, never as fresh."""

    paths = tuple(Path(value) for value in (attempt02_train, attempt03_fit, consumed_precal))
    if any(
        "locked" in str(path).lower()
        or "calibration.jsonl" in str(path).lower()
        for path in paths
    ):
        raise ValueError("Attempt05 dev loader refuses calibration/locked paths")
    blocks = tuple(read_teacher_jsonl(path) for path in paths)
    if tuple(len(block) for block in blocks) != (200, 500, 200):
        raise ValueError("Attempt05 requires the exact old 200/500/200 dev sources")
    roles = (
        *("attempt02_train" for _ in blocks[0]),
        *("attempt03_fit" for _ in blocks[1]),
        *("attempt03_consumed_precal_dev_only" for _ in blocks[2]),
    )
    rows = [*blocks[0], *blocks[1], *blocks[2]]
    samples = tuple(prepare_teacher_samples(rows))
    profiles = tuple(_profile(row) for row in rows)
    if len(samples) != 900 or len(set(sample.observation_fingerprint for sample in samples)) != 900:
        raise ValueError("Attempt05 dev identities must be exactly 900 unique states")
    counts = {profile: profiles.count(profile) for profile in ATTEMPT05_PROFILES}
    if counts != {profile: 180 for profile in ATTEMPT05_PROFILES}:
        raise ValueError(f"Attempt05 dev profile balance changed: {counts}")
    for row, sample in zip(rows, samples, strict=True):
        observation_payload = row.get("policy_observation")
        if not isinstance(observation_payload, Mapping):
            raise ValueError("Attempt05 row lacks policy_observation")
        observation = ActorObservation.from_dict(observation_payload)
        if observation.street != "T1" or observation.seat != "second":
            raise ValueError("Attempt05 development data must be T1-second")
        if observation.to_dict() != dict(observation_payload):
            raise ValueError("Attempt05 policy_observation is non-canonical")
        sample.policy_sample["policy_observation"] = observation.to_dict()
        _require_targets(sample)
        projected = _safe_training_projection(sample)
        sample.policy_sample.clear()
        sample.policy_sample.update(projected)
    fold_ids = tuple(assign_identity_group_folds(samples, profiles, folds=ATTEMPT05_FOLDS))
    return Attempt05DevData(
        samples=samples,
        profiles=profiles,
        source_roles=roles,
        fold_ids=fold_ids,
        source_manifest={
            role: {
                "path": path.as_posix(),
                "file_sha256": _file_sha256(path),
                "rows": len(block),
            }
            for role, path, block in zip(
                ("attempt02_train", "attempt03_fit", "attempt03_consumed_precal_dev_only"),
                paths,
                blocks,
                strict=True,
            )
        },
    )


def assign_identity_group_folds(
    samples: Sequence[PreparedTeacherSample],
    profiles: Sequence[str],
    *,
    folds: int = ATTEMPT05_FOLDS,
) -> np.ndarray:
    """Deterministic profile-stratified identity folds with no row leakage."""

    if len(samples) != len(profiles) or folds < 2:
        raise ValueError("Attempt05 fold inputs are invalid")
    fingerprints = [sample.observation_fingerprint for sample in samples]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Attempt05 fold assignment rejects duplicate identities")
    result = np.full(len(samples), -1, dtype=np.int8)
    for profile in ATTEMPT05_PROFILES:
        indices = [index for index, value in enumerate(profiles) if value == profile]
        indices.sort(
            key=lambda index: hashlib.sha256(fingerprints[index].encode("ascii")).digest()
        )
        for rank, index in enumerate(indices):
            result[index] = rank % folds
    if np.any(result < 0):
        raise ValueError("Attempt05 fold assignment found an unknown profile")
    return result


def fit_lambda_rank_fold(
    samples: Sequence[PreparedTeacherSample],
    profiles: Sequence[str],
    *,
    fold_index: int,
    seed: int,
    smoke: bool = False,
) -> LambdaRankFoldPredictor:
    safe_samples = _validate_and_project_training_inputs(samples, profiles)
    features: list[np.ndarray] = []
    relevance: list[np.ndarray] = []
    gains: list[np.ndarray] = []
    tails: list[np.ndarray] = []
    groups: list[int] = []
    state_weights = _profile_weights(profiles)
    row_weights: list[np.ndarray] = []
    for sample, runtime_sample, state_weight in zip(
        samples, safe_samples, state_weights, strict=True
    ):
        matrix, _unused = sample_to_matrix(runtime_sample)
        paired = build_paired_action_features_matrix(matrix, baseline_index=sample.baseline_index)
        delta = np.asarray(sample.teacher_paired_delta_mean, dtype=np.float64)
        feature_rows = paired.astype(np.float32, copy=False)
        features.append(feature_rows)
        relevance.append(_graded_relevance(delta))
        gains.append((delta > 0.0).astype(np.int8))
        tails.append(
            np.column_stack(
                (
                    sample.downside_loss_p95,
                    sample.downside_loss_p99,
                    sample.downside_loss_max,
                )
            ).astype(np.float64)
        )
        groups.append(len(delta))
        row_weights.append(
            np.full(len(delta), state_weight / len(delta), dtype=np.float64)
        )
    x = np.vstack(features)
    y_rank = np.concatenate(relevance)
    y_gain = np.concatenate(gains)
    y_tail = np.vstack(tails)
    weight = np.concatenate(row_weights)
    gain_weight = weight * _binary_balance_weights(y_gain)
    iterations = 24 if smoke else 180
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 3, 7, 15],
        n_estimators=iterations,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=30,
        reg_lambda=8.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    ).fit(x, y_rank, group=groups, sample_weight=weight)
    gain_head: Any
    if np.unique(y_gain).size == 1:
        gain_head = _ConstantProbability(float(y_gain[0]))
    else:
        gain_head = LGBMClassifier(
            objective="binary",
            n_estimators=iterations,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=30,
            reg_lambda=8.0,
            reg_alpha=0.5,
            random_state=seed + 1,
            n_jobs=2,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ).fit(x, y_gain, sample_weight=gain_weight)
    tail_heads = []
    for column in range(3):
        tail_heads.append(
            LGBMRegressor(
                objective="quantile",
                alpha=ATTEMPT05_TAIL_PINBALL_QUANTILE,
                n_estimators=iterations,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=30,
                reg_lambda=8.0,
                reg_alpha=0.5,
                random_state=seed + 10 + column,
                n_jobs=2,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            ).fit(x, y_tail[:, column], sample_weight=weight)
        )
    return LambdaRankFoldPredictor(
        ranker=ranker,
        gain_head=gain_head,
        tail_p95_head=tail_heads[0],
        tail_p99_head=tail_heads[1],
        tail_max_head=tail_heads[2],
        fold_index=fold_index,
    )


def fit_deepsets_fold(
    samples: Sequence[PreparedTeacherSample],
    profiles: Sequence[str],
    *,
    fold_index: int,
    seed: int,
    epochs: int,
    device: str,
    batch_states: int = 24,
) -> tuple[DeepSetsFoldPredictor, dict[str, Any]]:
    safe_samples = _validate_and_project_training_inputs(samples, profiles)
    torch = _import_torch()
    resolved_device = _resolve_device(torch, device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net = build_deepsets_network(
        torch, token_hidden_dim=32, trunk_hidden_dim=64, dropout=0.05
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-4)
    state_weights = _profile_weights(profiles)
    all_gain = np.concatenate(
        [
            (np.asarray(sample.teacher_paired_delta_mean) > 0.0).astype(np.int8)
            for sample in samples
        ]
    )
    negative = max(1, int(np.sum(all_gain == 0)))
    positive = max(1, int(np.sum(all_gain == 1)))
    pos_weight = torch.tensor(negative / positive, dtype=torch.float32, device=resolved_device)
    rng = np.random.default_rng(seed)
    epoch_losses: list[float] = []
    for _epoch in range(epochs):
        net.train()
        order = rng.permutation(len(samples))
        total = 0.0
        total_weight = 0.0
        for start in range(0, len(order), batch_states):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = None
            batch_weight = 0.0
            for raw_index in order[start : start + batch_states]:
                index = int(raw_index)
                sample = samples[index]
                state_tokens, action_tokens, context = encode_deepsets_runtime_sample(
                    safe_samples[index], baseline_index=sample.baseline_index
                )
                rank, gain_logit, tail_norm = net(
                    torch.from_numpy(state_tokens).to(resolved_device),
                    torch.from_numpy(action_tokens).to(resolved_device),
                    torch.from_numpy(context).to(resolved_device),
                )
                delta = torch.as_tensor(
                    sample.teacher_paired_delta_mean,
                    dtype=torch.float32,
                    device=resolved_device,
                )
                gain_target = (delta > 0.0).to(torch.float32)
                tail_target = torch.as_tensor(
                    np.column_stack(
                        (
                            sample.downside_loss_p95,
                            sample.downside_loss_p99,
                            sample.downside_loss_max,
                        )
                    )
                    / np.asarray(HU_M43_ATTEMPT05_TAIL_SCALES),
                    dtype=torch.float32,
                    device=resolved_device,
                )
                target_probability = torch.softmax(delta / 2.0, dim=0)
                listwise = -(target_probability * torch.log_softmax(rank, dim=0)).sum()
                gain_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    gain_logit, gain_target, pos_weight=pos_weight
                )
                error = tail_target - tail_norm
                tail_loss = torch.maximum(
                    ATTEMPT05_TAIL_PINBALL_QUANTILE * error,
                    (ATTEMPT05_TAIL_PINBALL_QUANTILE - 1.0) * error,
                ).mean()
                loss = listwise + 0.35 * gain_loss + 0.50 * tail_loss
                weight = float(state_weights[index])
                weighted = weight * loss
                batch_loss = weighted if batch_loss is None else batch_loss + weighted
                batch_weight += weight
            if batch_loss is None:
                continue
            normalized = batch_loss / max(batch_weight, 1e-9)
            normalized.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(normalized.detach().cpu()) * batch_weight
            total_weight += batch_weight
        epoch_losses.append(total / max(total_weight, 1e-9))
    state_dict = {key: value.detach().cpu() for key, value in net.state_dict().items()}
    predictor = DeepSetsFoldPredictor(state_dict=state_dict, fold_index=fold_index)
    return predictor, {
        "epochs": epochs,
        "device": resolved_device,
        "epoch_losses": epoch_losses,
        "listwise_temperature": 2.0,
        "gain_loss_weight": 0.35,
        "tail_pinball_weight": 0.50,
        "tail_pinball_quantile": ATTEMPT05_TAIL_PINBALL_QUANTILE,
    }


def run_architecture_comparison(
    data: Attempt05DevData,
    *,
    output_dir: str | Path,
    deepsets_epochs: int = 12,
    device: str = "auto",
    seed: int = 2026074001,
) -> dict[str, Any]:
    """Run the sole old-dev900 five-fold A/B comparison, without a gate sweep."""

    if len(data.samples) != 900 or set(data.fold_ids) != set(range(5)):
        raise ValueError("Attempt05 architecture comparison requires exact dev900/5fold")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    family_reports: dict[str, Any] = {}
    family_models: dict[str, HuM43Attempt05Model] = {}
    for family_offset, family in enumerate(ATTEMPT05_FAMILIES):
        oof: list[Attempt05FoldOutput | None] = [None] * len(data.samples)
        folds = []
        fold_reports = []
        for fold in range(ATTEMPT05_FOLDS):
            train_indices = [index for index, value in enumerate(data.fold_ids) if value != fold]
            validation_indices = [index for index, value in enumerate(data.fold_ids) if value == fold]
            train_ids = {data.samples[index].observation_fingerprint for index in train_indices}
            validation_ids = {data.samples[index].observation_fingerprint for index in validation_indices}
            if train_ids & validation_ids:
                raise AssertionError("Attempt05 OOF identity leakage")
            train_samples = [data.samples[index] for index in train_indices]
            train_profiles = [data.profiles[index] for index in train_indices]
            fit_seed = seed + 1000 * family_offset + fold
            if family == "lambda_rank":
                predictor = fit_lambda_rank_fold(
                    train_samples,
                    train_profiles,
                    fold_index=fold,
                    seed=fit_seed,
                )
                fit_report: Mapping[str, Any] = {
                    "loss": "lambdarank+binary_logloss+three_q0.80_pinball_heads"
                }
            else:
                predictor, fit_report = fit_deepsets_fold(
                    train_samples,
                    train_profiles,
                    fold_index=fold,
                    seed=fit_seed,
                    epochs=deepsets_epochs,
                    device=device,
                )
            folds.append(predictor)
            for index in validation_indices:
                oof[index] = predictor.predict(
                    data.samples[index].policy_sample,
                    baseline_index=data.samples[index].baseline_index,
                )
            fold_reports.append(
                {
                    "fold": fold,
                    "train_states": len(train_indices),
                    "validation_states": len(validation_indices),
                    "identity_intersection": 0,
                    "fit": dict(fit_report),
                }
            )
        if any(value is None for value in oof):
            raise AssertionError("Attempt05 OOF coverage incomplete")
        predictions = tuple(value for value in oof if value is not None)
        family_report, cushions = evaluate_oof_predictions(
            data.samples, data.profiles, predictions
        )
        model = HuM43Attempt05Model(
            family=family,
            fold_predictors=tuple(folds),
            conformal_cushions=cushions,
            conformal_quantile=ATTEMPT05_CONFORMAL_QUANTILE,
            gain_threshold=1.0,
            uncertainty_max=0.0,
            runtime_enabled=False,
            winner_frozen=False,
            model_id=f"hu-m43-attempt05-{family}-dev900-oof",
            manifest={
                "architecture_comparison_only": True,
                "runtime_enabled": False,
                "current_profile_mutated": False,
                "profile_runtime_feature": False,
                "teacher_ev_runtime_gate": False,
                "teacher_lcb_runtime_gate": False,
            },
        )
        artifact_path = output / f"{family}_candidate.pkl"
        artifact_sha256 = model.save(artifact_path)
        family_report.update(
            {
                "folds": fold_reports,
                "candidate_artifact": artifact_path.as_posix(),
                "candidate_artifact_sha256": artifact_sha256,
            }
        )
        family_reports[family] = family_report
        family_models[family] = model
    winner = select_architecture_winner(family_reports)
    report = {
        "schema": ATTEMPT05_DEV_SCHEMA,
        "status": (
            "architecture_winner_selected_freeze_required_before_one_new_audit"
            if winner is not None
            else "development_no_go_no_new_audit_authorized"
        ),
        "comparison_families": list(ATTEMPT05_FAMILIES),
        "states": 900,
        "folds": 5,
        "fold_strategy": "profile_stratified_observation_identity_round_robin_v1",
        "threshold_sweep_performed": False,
        "old_consumed200_separate_selection_performed": False,
        "winner_selection_rule": (
            "raw_gate_then_max_min_profile_positive_rate_then_overall_positive_rate_"
            "then_mean_delta_then_lower_regret_then_family_name"
        ),
        "selected_family": winner,
        "winner_is_runtime_frozen": False,
        "fresh_audit_authorized_before_winner_freeze": False,
        "new_audit_budget_after_winner_freeze": 1 if winner is not None else 0,
        "families": family_reports,
        "source_manifest": data.source_manifest,
        "claim_boundary": {
            "old_dev900_is_fresh_generalization": False,
            "teacher_values_are_realized_ev": False,
            "teacher_ev_or_lcb_runtime_gate": False,
            "profile_runtime_feature": False,
            "current_profile_mutated": False,
            "runtime_enabled": False,
        },
    }
    _write_json_no_clobber(output / "architecture_comparison.json", report)
    return report


def run_smoke_fit(
    data: Attempt05DevData,
    *,
    family: str,
    max_states: int,
    epochs: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Small correctness/performance smoke; never an architecture decision."""

    if family not in ATTEMPT05_FAMILIES:
        raise ValueError("Attempt05 smoke family is invalid")
    indices = _balanced_subset_indices(data.profiles, max_states)
    subset_samples = [data.samples[index] for index in indices]
    subset_profiles = [data.profiles[index] for index in indices]
    subset_folds = assign_identity_group_folds(subset_samples, subset_profiles, folds=5)
    train = [index for index, fold in enumerate(subset_folds) if fold != 0]
    validation = [index for index, fold in enumerate(subset_folds) if fold == 0]
    if family == "lambda_rank":
        predictor = fit_lambda_rank_fold(
            [subset_samples[index] for index in train],
            [subset_profiles[index] for index in train],
            fold_index=0,
            seed=seed,
            smoke=True,
        )
        fit_report: Mapping[str, Any] = {"iterations": 24, "device": "cpu"}
    else:
        predictor, fit_report = fit_deepsets_fold(
            [subset_samples[index] for index in train],
            [subset_profiles[index] for index in train],
            fold_index=0,
            seed=seed,
            epochs=epochs,
            device=device,
            batch_states=min(16, max(1, len(train))),
        )
    predictions = tuple(
        predictor.predict(
            subset_samples[index].policy_sample,
            baseline_index=subset_samples[index].baseline_index,
        )
        for index in validation
    )
    metrics, cushions = evaluate_oof_predictions(
        [subset_samples[index] for index in validation],
        [subset_profiles[index] for index in validation],
        predictions,
        require_all_profiles=False,
    )
    return {
        "schema": ATTEMPT05_SMOKE_SCHEMA,
        "family": family,
        "train_states": len(train),
        "validation_states": len(validation),
        "identity_intersection": 0,
        "fit": dict(fit_report),
        "metrics_diagnostic_only": metrics,
        "conformal_cushions_diagnostic_only": list(cushions),
        "architecture_selected": False,
        "fresh_audit_consumed": False,
        "runtime_enabled": False,
    }


def evaluate_oof_predictions(
    samples: Sequence[PreparedTeacherSample],
    profiles: Sequence[str],
    predictions: Sequence[Attempt05FoldOutput],
    *,
    require_all_profiles: bool = True,
) -> tuple[dict[str, Any], tuple[float, float, float]]:
    if not (len(samples) == len(profiles) == len(predictions)) or not samples:
        raise ValueError("Attempt05 OOF evaluation inputs disagree")
    rows = []
    all_gain_probability = []
    all_gain_label = []
    all_tail_prediction = []
    all_tail_target = []
    for sample, profile, prediction in zip(samples, profiles, predictions, strict=True):
        action_count = len(sample.policy_sample["actions"])
        _validate_prediction_shapes(prediction, action_count)
        nonbaseline = [index for index in range(action_count) if index != sample.baseline_index]
        rank = np.asarray(prediction.rank_score)
        best_score = max(float(rank[index]) for index in nonbaseline)
        candidate = min(
            (index for index in nonbaseline if float(rank[index]) == best_score),
            key=lambda index: action_key_from_payload(
                sample.policy_sample["actions"][index]
            ).sort_key(),
        )
        delta = np.asarray(sample.teacher_paired_delta_mean, dtype=np.float64)
        actual_tails = np.column_stack(
            (
                sample.downside_loss_p95,
                sample.downside_loss_p99,
                sample.downside_loss_max,
            )
        ).astype(np.float64)
        predicted_tails = np.column_stack(
            (
                prediction.downside_p95,
                prediction.downside_p99,
                prediction.downside_max,
            )
        ).astype(np.float64)
        rows.append(
            {
                "fingerprint": sample.observation_fingerprint,
                "profile": profile,
                "candidate_index": candidate,
                "candidate_action_key": action_key_from_payload(
                    sample.policy_sample["actions"][candidate]
                ).to_token(),
                "selected_delta": float(delta[candidate]),
                "regret_to_best_nonbaseline": float(
                    np.max(delta[nonbaseline]) - delta[candidate]
                ),
                "ndcg": _ndcg(rank[nonbaseline], _graded_relevance(delta[nonbaseline])),
                "gain_probability": float(prediction.gain_probability[candidate]),
                "predicted_tails": predicted_tails[candidate].tolist(),
                "actual_tails": actual_tails[candidate].tolist(),
            }
        )
        all_gain_probability.append(np.asarray(prediction.gain_probability))
        all_gain_label.append((delta > 0.0).astype(np.float64))
        all_tail_prediction.append(predicted_tails)
        all_tail_target.append(actual_tails)
    selected_delta = np.asarray([row["selected_delta"] for row in rows])
    residual = np.asarray(
        [
            np.asarray(row["actual_tails"]) - np.asarray(row["predicted_tails"])
            for row in rows
        ]
    )
    cushions = tuple(
        max(
            0.0,
            float(
                np.quantile(
                    residual[:, column],
                    ATTEMPT05_CONFORMAL_QUANTILE,
                    method="higher",
                )
            ),
        )
        for column in range(3)
    )
    gain_probability = np.concatenate(all_gain_probability)
    gain_label = np.concatenate(all_gain_label)
    tail_prediction = np.vstack(all_tail_prediction)
    tail_target = np.vstack(all_tail_target)
    by_profile = {}
    for profile in ATTEMPT05_PROFILES:
        values = np.asarray(
            [row["selected_delta"] for row in rows if row["profile"] == profile],
            dtype=np.float64,
        )
        if require_all_profiles and values.size == 0:
            raise ValueError("Attempt05 OOF evaluation lacks a required profile")
        by_profile[profile] = {
            "states": int(values.size),
            "positive_rate": float(np.mean(values > 0.0)) if values.size else None,
            "mean_delta": float(np.mean(values)) if values.size else None,
        }
    upper = np.asarray(
        [
            np.asarray(row["predicted_tails"]) + np.asarray(cushions)
            for row in rows
        ]
    )
    actual = np.asarray([row["actual_tails"] for row in rows])
    conformal_coverage = np.mean(actual <= upper, axis=0)
    metrics = {
        "schema": ATTEMPT05_OOF_SCHEMA,
        "states": len(rows),
        "raw_selected_positive_rate": float(np.mean(selected_delta > 0.0)),
        "raw_selected_mean_delta": float(np.mean(selected_delta)),
        "mean_regret_to_best_nonbaseline": float(
            np.mean([row["regret_to_best_nonbaseline"] for row in rows])
        ),
        "mean_ndcg": float(np.mean([row["ndcg"] for row in rows])),
        "gain_brier_all_actions": float(np.mean((gain_probability - gain_label) ** 2)),
        "tail_pinball_q80_all_actions": [
            _pinball(tail_target[:, column], tail_prediction[:, column], ATTEMPT05_TAIL_PINBALL_QUANTILE)
            for column in range(3)
        ],
        "conformal": {
            "schema": HU_M43_ATTEMPT05_CONFORMAL_SCHEMA,
            "quantile": ATTEMPT05_CONFORMAL_QUANTILE,
            "source": "strict_oof_candidate_upper_residuals_only",
            "cushions": list(cushions),
            "empirical_marginal_coverage": conformal_coverage.tolist(),
            "tail_limits_not_used_for_architecture_threshold_selection": list(
                HU_M43_ATTEMPT05_TAIL_LIMITS
            ),
        },
        "profile": by_profile,
        "raw_gate_pass": bool(
            np.mean(selected_delta > 0.0) >= 0.40
            and all(
                value["positive_rate"] is not None
                and float(value["positive_rate"]) >= 0.30
                for value in by_profile.values()
                if value["states"] > 0
            )
        ),
        "oof_identity_prediction_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "teacher_values_reported_as_realized_ev": False,
        "rows": rows,
    }
    return metrics, cushions


def select_architecture_winner(family_reports: Mapping[str, Mapping[str, Any]]) -> str | None:
    eligible = [
        (name, report)
        for name, report in family_reports.items()
        if bool(report.get("raw_gate_pass"))
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda item: (
            -min(
                float(value["positive_rate"])
                for value in item[1]["profile"].values()
            ),
            -float(item[1]["raw_selected_positive_rate"]),
            -float(item[1]["raw_selected_mean_delta"]),
            float(item[1]["mean_regret_to_best_nonbaseline"]),
            item[0],
        )
    )
    return eligible[0][0]


def _profile(row: Mapping[str, Any]) -> str:
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Attempt05 teacher row lacks provenance")
    value = str(provenance.get("root_profile", ""))
    if value not in ATTEMPT05_PROFILES:
        raise ValueError(f"Attempt05 unknown root profile: {value!r}")
    return value


def _require_targets(sample: PreparedTeacherSample) -> None:
    arrays = (
        sample.teacher_paired_delta_mean,
        sample.downside_loss_p95,
        sample.downside_loss_p99,
        sample.downside_loss_max,
    )
    if any(value is None for value in arrays):
        raise ValueError("Attempt05 requires paired delta and all downside targets")
    action_count = len(sample.policy_sample["actions"])
    if any(np.asarray(value).shape != (action_count,) for value in arrays):
        raise ValueError("Attempt05 target/action dimensions disagree")


def _safe_training_projection(sample: PreparedTeacherSample) -> dict[str, Any]:
    """Rebuild one feature sample solely from its canonical ActorObservation.

    Teacher labels stay on ``PreparedTeacherSample`` and never enter this
    projection.  Requiring the complete legal ActionKey set here prevents a
    direct fit API caller from bypassing the runtime projection contract.
    """

    source = dict(sample.policy_sample)
    actions = source.get("actions")
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise ValueError("Attempt05 training sample requires legal actions")
    baseline = int(sample.baseline_index)
    if not 0 <= baseline < len(actions):
        raise ValueError("Attempt05 training baseline index is invalid")
    baseline_key = action_key_from_payload(actions[baseline]).to_token()
    declared_key = source.get("baseline_action_key")
    if declared_key is not None and declared_key != baseline_key:
        raise ValueError("Attempt05 training baseline ActionKey disagrees")
    source["baseline_action_row_index"] = baseline
    source["baseline_action_key"] = baseline_key
    projected, observation = _runtime_policy_projection(source)
    if observation.street != "T1" or observation.seat != "second":
        raise ValueError("Attempt05 training projection authorizes only T1-second")
    if observation.fingerprint() != sample.observation_fingerprint:
        raise ValueError("Attempt05 training observation identity disagrees")
    return projected


def _validate_and_project_training_inputs(
    samples: Sequence[PreparedTeacherSample], profiles: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    if not samples or len(samples) != len(profiles):
        raise ValueError("Attempt05 training samples/profiles disagree")
    identities = [sample.observation_fingerprint for sample in samples]
    if len(set(identities)) != len(identities):
        raise ValueError("Attempt05 fit rejects duplicate identities")
    if any(profile not in ATTEMPT05_PROFILES for profile in profiles):
        raise ValueError("Attempt05 fit contains an unknown profile")
    return tuple(_safe_training_projection(sample) for sample in samples)


def _profile_weights(profiles: Sequence[str]) -> np.ndarray:
    counts = {profile: profiles.count(profile) for profile in set(profiles)}
    result = np.asarray([1.0 / counts[profile] for profile in profiles], dtype=np.float64)
    return result / np.mean(result)


def _binary_balance_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    counts = {value: max(1, int(np.sum(labels == value))) for value in (0, 1)}
    result = np.asarray([1.0 / counts[int(value)] for value in labels])
    return result / np.mean(result)


def _graded_relevance(delta: np.ndarray) -> np.ndarray:
    values = np.asarray(delta, dtype=np.float64)
    unique = np.unique(values)
    if unique.size == 1:
        return np.zeros(values.shape, dtype=np.int32)
    ranks = np.searchsorted(unique, values, side="left")
    return np.minimum(4, np.floor(5.0 * ranks / unique.size)).astype(np.int32)


def _ndcg(rank_score: np.ndarray, relevance: np.ndarray) -> float:
    order = np.argsort(-np.asarray(rank_score), kind="mergesort")
    ideal = np.argsort(-np.asarray(relevance), kind="mergesort")
    discount = 1.0 / np.log2(np.arange(len(order), dtype=np.float64) + 2.0)
    gain = np.power(2.0, relevance.astype(np.float64)) - 1.0
    dcg = float(np.sum(gain[order] * discount))
    idcg = float(np.sum(gain[ideal] * discount))
    return dcg / idcg if idcg > 0.0 else 1.0


def _pinball(target: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = np.asarray(target) - np.asarray(prediction)
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def _validate_prediction_shapes(output: Attempt05FoldOutput, actions: int) -> None:
    for name in (
        "rank_score", "gain_probability", "downside_p95", "downside_p99", "downside_max"
    ):
        value = np.asarray(getattr(output, name))
        if value.shape != (actions,) or not np.isfinite(value).all():
            raise ValueError(f"Attempt05 OOF {name} shape/value is invalid")


def _balanced_subset_indices(profiles: Sequence[str], max_states: int) -> list[int]:
    if max_states < 25:
        raise ValueError("Attempt05 smoke requires at least 25 states")
    per_profile = max_states // len(ATTEMPT05_PROFILES)
    result = []
    for profile in ATTEMPT05_PROFILES:
        candidates = [index for index, value in enumerate(profiles) if value == profile]
        result.extend(candidates[:per_profile])
    return result


def _resolve_device(torch: Any, value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Attempt05 requested CUDA but it is unavailable")
    if value not in {"cpu", "cuda"}:
        raise ValueError("Attempt05 device must be auto/cpu/cuda")
    return value


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Attempt05 DeepSets training requires PyTorch") from error
    return torch


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
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
    parser.add_argument("--attempt02-train", type=Path, default=DEFAULT_ATTEMPT02_TRAIN)
    parser.add_argument("--attempt03-fit", type=Path, default=DEFAULT_ATTEMPT03_FIT)
    parser.add_argument("--consumed-precal", type=Path, default=DEFAULT_ATTEMPT03_CONSUMED_PRECAL)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke-fit")
    smoke.add_argument("--family", choices=ATTEMPT05_FAMILIES, required=True)
    smoke.add_argument("--max-states", type=int, default=50)
    smoke.add_argument("--epochs", type=int, default=2)
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    smoke.add_argument("--seed", type=int, default=2026074001)
    smoke.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("architecture-compare")
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--deepsets-epochs", type=int, default=12)
    compare.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    compare.add_argument("--seed", type=int, default=2026074001)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_attempt05_dev900(
        attempt02_train=args.attempt02_train,
        attempt03_fit=args.attempt03_fit,
        consumed_precal=args.consumed_precal,
    )
    if args.command == "smoke-fit":
        report = run_smoke_fit(
            data,
            family=args.family,
            max_states=args.max_states,
            epochs=args.epochs,
            device=args.device,
            seed=args.seed,
        )
        _write_json_no_clobber(args.output, report)
        print(json.dumps({"status": "smoke_complete", "family": args.family, "output": str(args.output)}))
        return 0
    report = run_architecture_comparison(
        data,
        output_dir=args.output_dir,
        deepsets_epochs=args.deepsets_epochs,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "selected_family": report["selected_family"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT05_CONFORMAL_QUANTILE",
    "ATTEMPT05_DEV_SCHEMA",
    "ATTEMPT05_FAMILIES",
    "Attempt05DevData",
    "assign_identity_group_folds",
    "evaluate_oof_predictions",
    "fit_deepsets_fold",
    "fit_lambda_rank_fold",
    "load_attempt05_dev900",
    "run_architecture_comparison",
    "run_smoke_fit",
    "select_architecture_winner",
]
