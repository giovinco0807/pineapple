"""Train the fixed Attempt12 public-infoset distillation ensemble.

This trainer consumes only the immutable development200 teacher rows.  The
disjoint audit50 may authorize the frozen search architecture, but it is never
used as fit, calibration, fold, feature, or threshold data.
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

from .action_key import ActionKey, resolve_action_key
from .action_space import generate_turn_actions
from .hu_infoset import ActorObservation
from .hu_m43_attempt05_training import _ConstantProbability
from .hu_m43_attempt12_distilled_model import (
    HU_M43_ATTEMPT12_DISTILLED_ACTION_SCORE_MODE,
    HU_M43_ATTEMPT12_DISTILLED_FEATURE_DIM,
    HU_M43_ATTEMPT12_DISTILLED_FEATURE_SCHEMA,
    HU_M43_ATTEMPT12_DISTILLED_HEAD_SCHEMA,
    HU_M43_ATTEMPT12_DISTILLED_MODEL_SCHEMA,
    Attempt12DistilledFoldPredictor,
    HuM43Attempt12DistilledModel,
    build_attempt12_distilled_features,
)
from .hu_m43_attempt12_contract import (
    ATTEMPT12_CANDIDATE_MAX,
    M43_ATTEMPT12_PLAN_SHA256,
)
from .hu_m43_attempt12_distilled_runtime import (
    ATTEMPT12_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT12_RUNTIME_REQUIREMENTS_SHA256,
    build_distilled_runtime_source_freeze,
    validate_distilled_runtime_dependencies,
)
from .hu_m43_attempt12_teacher import (
    ATTEMPT12_FROZEN_MODEL_SHA256,
    ATTEMPT12_TEACHER_SCHEMA,
    Attempt12TeacherConfig,
    FrozenAttempt12LambdaRanker,
    validate_attempt12_teacher_output,
)
from .hu_turn3_model import hu_policy_sample


ATTEMPT12_DISTILLATION_CONFIG_SCHEMA = "hu_m43_attempt12_distillation_config_v1"
ATTEMPT12_DISTILLATION_CONFIG_SHA256 = (
    "a62b075d2f86e9ec751ea013495b0d6c109371cf49d4c6eec7c95e386e194fb5"
)
ATTEMPT12_DISTILLATION_FULL_ITERATIONS = 180
ATTEMPT12_DISTILLATION_FULL_STATES = 200
ATTEMPT12_DEVELOPMENT_ROW_SCHEMA = "hu_m43_attempt12_search_root_v1"
ATTEMPT12_DISTILLED_TRAINING_MANIFEST_SCHEMA = (
    "hu_m43_attempt12_distilled_training_manifest_v1"
)
ATTEMPT12_DEVELOPMENT_DECISION_SCHEMA = (
    "hu_m43_attempt12_development_go_no_go_v1"
)
ATTEMPT12_DEVELOPMENT_SELECTOR_RECEIPT_SCHEMA = (
    "hu_m43_attempt12_development_selector_receipt_v1"
)
ATTEMPT12_DEVELOPMENT_PASS_FREEZE_SCHEMA = (
    "hu_m43_attempt12_development_go_freeze_v1"
)
ATTEMPT12_AUDIT50_DECISION_SCHEMA = "hu_m43_attempt12_audit50_go_no_go_v1"
ATTEMPT12_AUDIT50_SELECTOR_RECEIPT_SCHEMA = (
    "hu_m43_attempt12_audit50_selector_receipt_v1"
)
ATTEMPT12_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
ATTEMPT12_FOLDS = 5
ATTEMPT12_PROPOSAL_ROWS_MAX = ATTEMPT12_CANDIDATE_MAX + 1


@dataclass(frozen=True)
class Attempt12DistillationState:
    root_index: int
    root_profile: str
    observation_fingerprint: str
    sample: Mapping[str, Any]
    baseline_index: int
    proposal_indices: tuple[int, ...]
    features: np.ndarray
    relevance: np.ndarray
    delta: np.ndarray
    safe: np.ndarray
    tails: np.ndarray
    selected_action_key: str
    teacher_override_fired: bool

    @property
    def group_size(self) -> int:
        return len(self.proposal_indices)


@dataclass(frozen=True)
class _ConstantRegressor:
    value: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(features).shape[0], self.value, dtype=np.float64)


def read_attempt12_jsonl(path: str | Path) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("Attempt12 distillation JSONL must be non-empty and newline terminated")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Attempt12 distillation JSONL line {line_number} is invalid") from exc
        if not isinstance(value, dict):
            raise ValueError("Attempt12 distillation JSONL rows must be mappings")
        rows.append(value)
    return rows


def prepare_attempt12_distillation_states(
    rows: Sequence[Mapping[str, Any]],
    *,
    ranker: FrozenAttempt12LambdaRanker,
    expected_roots: int | None = 200,
) -> tuple[Attempt12DistillationState, ...]:
    """Project teacher rows to complete legal public-infoset training states."""

    if expected_roots is not None and len(rows) != expected_roots:
        raise ValueError(f"Attempt12 distillation requires exactly {expected_roots} rows")
    by_root: dict[int, Mapping[str, Any]] = {}
    for raw in rows:
        root_index = _strict_int(raw.get("root_index"), "root_index")
        if root_index in by_root:
            raise ValueError("Attempt12 distillation rejects duplicate root indices")
        by_root[root_index] = raw
    if expected_roots is not None and set(by_root) != set(range(expected_roots)):
        raise ValueError("Attempt12 distillation root indices are not contiguous")

    states: list[Attempt12DistillationState] = []
    seen_fingerprints: set[str] = set()
    for root_index in sorted(by_root):
        row = by_root[root_index]
        _reject_hidden_information(row)
        if row.get("schema") != ATTEMPT12_DEVELOPMENT_ROW_SCHEMA:
            raise ValueError("Attempt12 distillation requires canonical search rows")
        provenance = _mapping(row.get("provenance"), "provenance")
        if (
            provenance.get("mode") != "development"
            or provenance.get("development_only") is not True
            or provenance.get("fit_allowed") is not False
            or provenance.get("threshold_selection_allowed") is not False
            or provenance.get("runtime_activation_allowed") is not False
        ):
            raise ValueError(
                "Attempt12 distillation accepts Development200 rows only"
            )
        profile = str(row.get("root_profile"))
        if profile not in ATTEMPT12_PROFILES:
            raise ValueError("Attempt12 distillation root profile is unknown")
        if expected_roots == 200 and profile != ATTEMPT12_PROFILES[root_index % 5]:
            raise ValueError("Attempt12 distillation profile schedule changed")
        observation_payload = _mapping(row.get("policy_observation"), "policy_observation")
        observation = ActorObservation.from_dict(observation_payload)
        if observation.to_dict() != observation_payload:
            raise ValueError("Attempt12 distillation observation is non-canonical")
        if observation.street != "T1" or observation.seat != "second":
            raise ValueError("Attempt12 distillation authorizes only T1-second")
        fingerprint = observation.fingerprint()
        if row.get("observation_fingerprint") not in (None, fingerprint):
            raise ValueError("Attempt12 distillation observation fingerprint changed")
        if fingerprint in seen_fingerprints:
            raise ValueError("Attempt12 distillation rejects duplicate information sets")
        seen_fingerprints.add(fingerprint)

        legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
        baseline_token = str(row.get("baseline_action_key"))
        baseline_index = resolve_action_key(legal, ActionKey.from_token(baseline_token))
        sample = hu_policy_sample(
            observation.hero_board,
            observation.dealt_cards,
            legal,
            opponent_board=observation.opponent_public_board,
            dead_cards=observation.legacy_dead_cards(),
            seat=observation.seat,
            to_act_order=observation.to_act_order,
        )
        sample["policy_observation"] = observation_payload
        sample["baseline_action_row_index"] = baseline_index
        sample["baseline_action_key"] = baseline_token

        teacher = _mapping(row.get("teacher"), "teacher")
        if teacher.get("schema") != ATTEMPT12_TEACHER_SCHEMA:
            raise ValueError("Attempt12 distillation teacher schema changed")
        if teacher.get("policy_observation") != observation_payload:
            raise ValueError("Attempt12 distillation teacher observation changed")
        if teacher.get("baseline_action_key") != baseline_token:
            raise ValueError("Attempt12 distillation teacher baseline changed")
        if teacher.get("runtime_gate_allowed") is not False:
            raise ValueError("Attempt12 teacher data cannot be runtime authorization")
        _validate_attempt12_public_teacher(
            observation,
            baseline_action_key=baseline_token,
            teacher=teacher,
        )
        built = build_attempt12_distilled_features(
            sample,
            candidate_generator=ranker.model,
            source_candidate_sha256=ranker.artifact_sha256,
            baseline_index=baseline_index,
        )
        teacher_candidates = tuple(str(value) for value in _sequence(
            teacher.get("all_legal_nonbaseline_action_keys"),
            "all_legal_nonbaseline_action_keys",
        ))
        built_candidates = tuple(
            built.action_keys[index].to_token() for index in built.candidate_indices
        )
        if teacher_candidates != built_candidates:
            raise ValueError("Attempt12 distillation all-legal candidate mapping changed")
        candidate_count = len(built.candidate_indices)
        if (
            teacher.get("candidate_nonbaseline_count") != candidate_count
            or not 0 <= candidate_count <= ATTEMPT12_CANDIDATE_MAX
            or len(set(teacher_candidates)) != candidate_count
            or baseline_token in teacher_candidates
        ):
            raise ValueError("Attempt12 distillation candidate count/uniqueness changed")
        proposal_indices = (*built.candidate_indices, baseline_index)
        group_size = len(proposal_indices)

        rerank = _mapping(teacher.get("rerank"), "rerank")
        rerank_rows = _sequence(rerank.get("actions"), "rerank.actions")
        if len(rerank_rows) != group_size:
            raise ValueError("Attempt12 distillation requires all legal candidates plus baseline R128 rows")
        expected_tokens = tuple(built.action_keys[index].to_token() for index in proposal_indices)
        if tuple(str(_mapping(item, "rerank action").get("action_key")) for item in rerank_rows) != expected_tokens:
            raise ValueError("Attempt12 distillation R128 action mapping changed")

        decision = _mapping(teacher.get("decision"), "decision")
        selected_token = str(decision.get("final_selected_action_key"))
        fired = decision.get("override_fired")
        fallback = decision.get("exact_baseline_fallback")
        if type(fired) is not bool or type(fallback) is not bool:
            raise ValueError("Attempt12 distillation decision flags are invalid")
        if fired == fallback or (not fired and selected_token != baseline_token):
            raise ValueError("Attempt12 distillation baseline fallback changed")
        if selected_token not in expected_tokens:
            raise ValueError("Attempt12 distillation selected action is outside proposals")

        delta = np.zeros(group_size, dtype=np.float64)
        tails = np.zeros((group_size, 3), dtype=np.float64)
        for index, raw_rerank in enumerate(rerank_rows):
            rerank_row = _mapping(raw_rerank, "rerank action")
            raw_delta = np.asarray(
                _sequence(
                    rerank_row.get("raw_paired_deltas_vs_baseline"),
                    "R128 raw paired deltas",
                ),
                dtype=np.float64,
            )
            if raw_delta.shape != (128,) or not np.isfinite(raw_delta).all():
                raise ValueError("Attempt12 distillation R128 vector changed")
            summary = _delta_summary(raw_delta)
            stored = _mapping(
                rerank_row.get("paired_delta_vs_baseline"), "R128 summary"
            )
            for name, expected in summary.items():
                if not math.isclose(
                    _finite(stored.get(name), f"R128 {name}"),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError("Attempt12 distillation R128 summary changed")
            delta[index] = summary["mean"]
            tails[index] = (
                max(0.0, -summary["p05"]),
                max(0.0, -summary["p01"]),
                max(0.0, -summary["min"]),
            )
        if not np.allclose(delta[-1], 0.0, rtol=0.0, atol=1.0e-12) or not np.allclose(
            tails[-1], 0.0, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("Attempt12 distillation explicit baseline is not zero")

        relevance = np.zeros(group_size, dtype=np.int8)
        selected_position = expected_tokens.index(selected_token)
        relevance[selected_position] = 4
        safe = np.zeros(group_size, dtype=np.int8)
        evaluation = _mapping(teacher.get("evaluation"), "evaluation")
        evaluation_rows = _sequence(evaluation.get("actions"), "evaluation.actions")
        if fired:
            if evaluation.get("opened") is not True or len(evaluation_rows) != 2:
                raise ValueError("Attempt12 distillation fired row lacks E512")
            selected_evaluation = _mapping(
                evaluation_rows[0], "evaluation selected action"
            )
            if selected_evaluation.get("action_key") != selected_token:
                raise ValueError("Attempt12 distillation E512 action mapping changed")
            evaluation_raw = np.asarray(
                _sequence(
                    selected_evaluation.get("raw_paired_deltas_vs_baseline"),
                    "evaluation raw paired deltas",
                ),
                dtype=np.float64,
            )
            if evaluation_raw.shape != (512,) or not np.isfinite(evaluation_raw).all():
                raise ValueError("Attempt12 distillation fired row lacks E512")
            if float(np.mean(evaluation_raw)) > 0.0:
                safe[selected_position] = 1
        elif evaluation.get("opened") is not False or evaluation_rows:
            raise ValueError("Attempt12 distillation nonfire opened E512")

        states.append(
            Attempt12DistillationState(
                root_index=root_index,
                root_profile=profile,
                observation_fingerprint=fingerprint,
                sample=sample,
                baseline_index=baseline_index,
                proposal_indices=tuple(int(value) for value in proposal_indices),
                features=built.features[np.asarray(proposal_indices, dtype=np.int32)],
                relevance=relevance,
                delta=delta,
                safe=safe,
                tails=tails,
                selected_action_key=selected_token,
                teacher_override_fired=fired,
            )
        )
    return tuple(states)


def _validate_attempt12_public_teacher(
    observation: ActorObservation,
    *,
    baseline_action_key: str,
    teacher: Mapping[str, Any],
) -> None:
    """Invoke the producer-owned public artifact validator before labeling."""

    search = _mapping(teacher.get("search_config"), "teacher.search_config")
    config = Attempt12TeacherConfig(
        frozen_model_sha256=ATTEMPT12_FROZEN_MODEL_SHA256,
        hand_seed=_strict_int(search.get("hand_seed"), "teacher hand_seed"),
        rerank_seed=_strict_int(search.get("rerank_seed"), "teacher rerank_seed"),
        veto_seed=_strict_int(search.get("veto_seed"), "teacher veto_seed"),
        stress_seed=_strict_int(search.get("stress_seed"), "teacher stress_seed"),
        confirmation_seed=_strict_int(
            search.get("confirmation_seed"), "teacher confirmation_seed"
        ),
        evaluation_seed=_strict_int(
            search.get("evaluation_seed"), "teacher evaluation_seed"
        ),
        child_policy_seed=_strict_int(
            search.get("child_policy_seed"), "teacher child_policy_seed"
        ),
        run_id=str(search.get("run_id", "")),
        batch_child_selectors=search.get("batch_child_selectors"),
    )
    result = validate_attempt12_teacher_output(
        observation,
        baseline_action_key=baseline_action_key,
        payload=teacher,
        config=config,
    )
    decision = _mapping(teacher.get("decision"), "teacher.decision")
    if (
        result.get("selected_action_key")
        != decision.get("final_selected_action_key")
        or result.get("override_fired") is not decision.get("override_fired")
    ):
        raise ValueError("Attempt12 public teacher validation result changed")


def assign_attempt12_identity_folds(
    states: Sequence[Attempt12DistillationState], *, folds: int = ATTEMPT12_FOLDS
) -> np.ndarray:
    if folds != ATTEMPT12_FOLDS or not states:
        raise ValueError("Attempt12 distillation requires exactly five folds")
    if len({state.observation_fingerprint for state in states}) != len(states):
        raise ValueError("Attempt12 distillation fold identities repeat")
    result = np.full(len(states), -1, dtype=np.int8)
    for profile in ATTEMPT12_PROFILES:
        indices = [index for index, state in enumerate(states) if state.root_profile == profile]
        indices.sort(
            key=lambda index: hashlib.sha256(
                states[index].observation_fingerprint.encode("ascii")
            ).digest()
        )
        for rank, index in enumerate(indices):
            result[index] = rank % folds
    if np.any(result < 0):
        raise ValueError("Attempt12 distillation fold assignment found unknown profile")
    return result


def fit_attempt12_distilled_model(
    states: Sequence[Attempt12DistillationState],
    *,
    candidate_ranker: FrozenAttempt12LambdaRanker,
    config: Mapping[str, Any],
    smoke: bool = False,
    model_id: str = "hu-m43-attempt12-distilled-development200-oof",
) -> tuple[HuM43Attempt12DistilledModel, dict[str, Any]]:
    """Fit five train-four/hold-one models and freeze OOF tail cushions."""

    _validate_config(config)
    if len(states) < ATTEMPT12_FOLDS:
        raise ValueError("Attempt12 distillation has too few states")
    folds = assign_attempt12_identity_folds(states)
    training = _mapping(config.get("training"), "training")
    seed = _strict_int(training.get("random_seed"), "random_seed")
    predictors: list[Attempt12DistilledFoldPredictor] = []
    for fold_index in range(ATTEMPT12_FOLDS):
        train_states = [state for index, state in enumerate(states) if folds[index] != fold_index]
        predictors.append(
            _fit_fold(
                train_states,
                fold_index=fold_index,
                seed=seed + fold_index * 100,
                config=training,
                smoke=smoke,
            )
        )

    residuals: list[list[float]] = [[], [], []]
    oof_selected = 0
    for state_index, state in enumerate(states):
        predictor = predictors[int(folds[state_index])]
        output = predictor.predict(state.features)
        candidates = slice(0, state.group_size - 1)
        predicted_selected = int(np.argmax(output.rank_score))
        teacher_selected = int(np.argmax(state.relevance))
        oof_selected += int(predicted_selected == teacher_selected)
        for column, values in enumerate(
            (output.downside_p95, output.downside_p99, output.downside_max)
        ):
            residuals[column].extend(
                float(target - predicted)
                for target, predicted in zip(
                    state.tails[candidates, column], values[candidates], strict=True
                )
            )
    quantile = float(training.get("conformal_quantile"))
    cushions = tuple(
        max(0.0, float(np.quantile(values, quantile, method="linear")))
        if values
        else 0.0
        for values in residuals
    )
    runtime_gate = _mapping(config.get("runtime_gate"), "runtime_gate")
    model = HuM43Attempt12DistilledModel(
        candidate_generator=candidate_ranker.model,
        fold_predictors=tuple(predictors),
        conformal_cushions=cushions,
        conformal_quantile=quantile,
        safety_threshold=float(runtime_gate["safety_probability_threshold"]),
        tail_limits=tuple(float(value) for value in runtime_gate["tail_upper_limits"]),
        minimum_fold_votes=_strict_int(
            runtime_gate["minimum_agreeing_fold_votes"], "minimum fold votes"
        ),
        safety_enabled=False,
        winner_frozen=False,
        model_id=model_id,
        manifest={
            "training_data": "attempt12_development200_only",
            "audit50_fit_rows": 0,
            "threshold_sweep_performed": False,
            "identity_grouped_folds": ATTEMPT12_FOLDS,
            "fit_mode": "smoke" if smoke else "full",
            "effective_iterations": _strict_int(
                training["smoke_iterations" if smoke else "iterations"],
                "effective iterations",
            ),
            "training_states": len(states),
            "distillation_config_sha256": ATTEMPT12_DISTILLATION_CONFIG_SHA256,
        },
    )
    diagnostics = {
        "states": len(states),
        "rows": sum(state.group_size for state in states),
        "group_sizes": [state.group_size for state in states],
        "candidate_count_min": min(state.group_size - 1 for state in states),
        "candidate_count_max": max(state.group_size - 1 for state in states),
        "candidate_count_histogram": {
            str(candidate_count): sum(
                state.group_size - 1 == candidate_count for state in states
            )
            for candidate_count in range(ATTEMPT12_CANDIDATE_MAX + 1)
        },
        "fold_counts": {
            str(fold): int(np.sum(folds == fold)) for fold in range(ATTEMPT12_FOLDS)
        },
        "profile_counts": {
            profile: sum(state.root_profile == profile for state in states)
            for profile in ATTEMPT12_PROFILES
        },
        "teacher_fires": sum(state.teacher_override_fired for state in states),
        "safe_positive_rows": int(sum(np.sum(state.safe) for state in states)),
        "oof_policy_top1_accuracy_diagnostic_only": oof_selected / len(states),
        "conformal_cushions": list(cushions),
        "conformal_quantile": quantile,
        "audit50_fit_rows": 0,
        "threshold_sweep_performed": False,
    }
    return model, diagnostics


def train_attempt12_distilled_artifact(
    *,
    input_path: str | Path,
    candidate_model_path: str | Path,
    config_path: str | Path,
    development_decision_path: str | Path,
    development_selector_receipt_path: str | Path,
    development_pass_freeze_path: str | Path,
    output_model_path: str | Path,
    output_manifest_path: str | Path,
    runtime_source_root: str | Path,
    runtime_dependency_root: str | Path,
    output_runtime_source_archive_path: str | Path,
    output_runtime_source_manifest_path: str | Path,
    smoke: bool = False,
) -> dict[str, Any]:
    input_path = Path(input_path)
    config_path = Path(config_path)
    decision_path = Path(development_decision_path)
    selector_receipt_path = Path(development_selector_receipt_path)
    pass_freeze_path = Path(development_pass_freeze_path)
    input_sha = _file_sha256(input_path)
    config = _load_mapping(config_path, "distillation config")
    _validate_config(config)
    if _file_sha256(config_path) != ATTEMPT12_DISTILLATION_CONFIG_SHA256:
        raise ValueError("Attempt12 full distillation config SHA changed")
    decision = _load_mapping(decision_path, "development decision")
    if (
        decision.get("schema") != ATTEMPT12_DEVELOPMENT_DECISION_SCHEMA
        or decision.get("decision") != "go"
        or decision.get("search_freeze_authorized") is not True
    ):
        raise ValueError("Attempt12 distillation requires development Go")
    decision_source = _mapping(decision.get("source"), "development decision source")
    if decision_source.get("input_jsonl_sha256") != input_sha:
        raise ValueError("Attempt12 development decision does not bind training JSONL")
    selector_receipt = _load_mapping(
        selector_receipt_path, "development selector receipt"
    )
    pass_freeze = _load_mapping(pass_freeze_path, "development pass freeze")
    decision_sha = _file_sha256(decision_path)
    selector_receipt_sha = _file_sha256(selector_receipt_path)
    if (
        selector_receipt.get("schema")
        != ATTEMPT12_DEVELOPMENT_SELECTOR_RECEIPT_SCHEMA
        or selector_receipt.get("status")
        != "single_frozen_gate_evaluation_complete"
        or selector_receipt.get("decision_sha256") != decision_sha
        or selector_receipt.get("decision") != "go"
        or selector_receipt.get("search_freeze_authorized") is not True
        or selector_receipt.get("gate_evaluation_count") != 1
        or selector_receipt.get("selector_executed") is not True
        or selector_receipt.get("fit_performed") is not False
        or selector_receipt.get("threshold_selected") is not False
        or selector_receipt.get("current_profile_mutated") is not False
        or selector_receipt.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt12 development selector receipt is not fit authorization")
    if (
        pass_freeze.get("schema") != ATTEMPT12_DEVELOPMENT_PASS_FREEZE_SCHEMA
        or pass_freeze.get("status") != "go_freeze_attempt12_development"
        or pass_freeze.get("decision") != "go"
        or pass_freeze.get("future_audit_authorized") is not False
        or pass_freeze.get("fit_performed") is not False
        or pass_freeze.get("threshold_selected") is not False
        or pass_freeze.get("current_profile_mutated") is not False
        or pass_freeze.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt12 immutable development-pass freeze is not bound")
    freeze_bindings = _mapping(pass_freeze.get("bindings"), "freeze bindings")
    for name in (
        "plan_sha256",
        "manifest_sha256",
        "execution_authorization_sha256",
        "source_package_sha256",
        "schedule_sha256",
        "startup_sha256",
        "merged_input_sha256",
        "receive_receipt_sha256",
        "selector_decision_sha256",
        "selector_receipt_sha256",
        "selector_source_sha256",
        "root_identity_sha256",
    ):
        _require_sha256(
            freeze_bindings.get(name), f"development pass freeze bindings.{name}"
        )
    if (
        freeze_bindings["plan_sha256"] != M43_ATTEMPT12_PLAN_SHA256
        or freeze_bindings["merged_input_sha256"] != input_sha
        or freeze_bindings["selector_decision_sha256"] != decision_sha
        or freeze_bindings["selector_receipt_sha256"] != selector_receipt_sha
        or decision_source.get("plan_sha256") != M43_ATTEMPT12_PLAN_SHA256
        or decision_source.get("source_package_sha256")
        != freeze_bindings["source_package_sha256"]
        or decision_source.get("authorization_sha256")
        != freeze_bindings["execution_authorization_sha256"]
    ):
        raise ValueError("Attempt12 development source hash chain changed")

    runtime_source = build_distilled_runtime_source_freeze(
        source_root=runtime_source_root,
        output_archive=output_runtime_source_archive_path,
        output_manifest=output_runtime_source_manifest_path,
    )
    runtime_file_set = _mapping(runtime_source.get("file_set"), "runtime file set")
    runtime_semantic = _mapping(
        runtime_source.get("semantic_closure"), "runtime semantic closure"
    )
    runtime_external = _mapping(
        runtime_semantic.get("external_runtime"), "external runtime"
    )
    teacher_contract = _mapping(
        runtime_semantic.get("teacher_contract"), "runtime teacher contract"
    )
    if (
        runtime_semantic.get("distilled_source_closure_sha256")
        != runtime_file_set.get("sha256")
        or teacher_contract.get("plan_sha256") != M43_ATTEMPT12_PLAN_SHA256
        or teacher_contract.get("candidate_model_sha256")
        != ATTEMPT12_FROZEN_MODEL_SHA256
    ):
        raise ValueError("Attempt12 runtime source/teacher closure changed")
    runtime_dependencies = validate_distilled_runtime_dependencies(
        runtime_dependency_root
    )
    if _mapping(
        runtime_semantic.get("runtime_dependencies"), "runtime dependencies"
    ) != {
        "schema": runtime_dependencies["schema"],
        "source_model_manifest_sha256": runtime_dependencies[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": runtime_dependencies[
            "source_native_manifest_sha256"
        ],
        "model_count": runtime_dependencies["model_count"],
        "binary_count": runtime_dependencies["binary_count"],
    }:
        raise ValueError("Attempt12 runtime dependency semantic binding changed")

    ranker = FrozenAttempt12LambdaRanker.load(
        candidate_model_path, expected_sha256=ATTEMPT12_FROZEN_MODEL_SHA256
    )
    rows = read_attempt12_jsonl(input_path)
    states = prepare_attempt12_distillation_states(rows, ranker=ranker, expected_roots=200)
    model, diagnostics = fit_attempt12_distilled_model(
        states, candidate_ranker=ranker, config=config, smoke=smoke
    )
    model_sha = model.save(output_model_path)
    manifest = {
        "schema": ATTEMPT12_DISTILLED_TRAINING_MANIFEST_SCHEMA,
        "status": "fit_complete_runtime_disabled",
        "model_schema": HU_M43_ATTEMPT12_DISTILLED_MODEL_SCHEMA,
        "feature_schema": HU_M43_ATTEMPT12_DISTILLED_FEATURE_SCHEMA,
        "head_schema": HU_M43_ATTEMPT12_DISTILLED_HEAD_SCHEMA,
        "action_score_mode": HU_M43_ATTEMPT12_DISTILLED_ACTION_SCORE_MODE,
        "model_id": model.model_id,
        "model_sha256": model_sha,
        "source": {
            "development_jsonl_sha256": input_sha,
            "development_decision_sha256": _file_sha256(decision_path),
            "development_selector_receipt_sha256": selector_receipt_sha,
            "development_pass_freeze_sha256": _file_sha256(pass_freeze_path),
            "distillation_config_sha256": _file_sha256(config_path),
            "candidate_model_sha256": ATTEMPT12_FROZEN_MODEL_SHA256,
            "development_plan_sha256": freeze_bindings["plan_sha256"],
            "development_package_manifest_sha256": freeze_bindings[
                "manifest_sha256"
            ],
            "development_source_package_sha256": freeze_bindings[
                "source_package_sha256"
            ],
            "development_root_identity_sha256": freeze_bindings[
                "root_identity_sha256"
            ],
            "runtime_source_archive_sha256": runtime_source["archive"]["sha256"],
            "runtime_source_manifest_sha256": _file_sha256(
                output_runtime_source_manifest_path
            ),
            "runtime_source_closure_sha256": runtime_file_set["sha256"],
            "runtime_semantic_closure_sha256": runtime_semantic["sha256"],
            "runtime_requirements_sha256": runtime_external[
                "requirements_sha256"
            ],
            "runtime_fingerprint_sha256": runtime_external[
                "runtime_fingerprint_sha256"
            ],
            "source_model_manifest_sha256": runtime_dependencies[
                "source_model_manifest_sha256"
            ],
            "source_native_manifest_sha256": runtime_dependencies[
                "source_native_manifest_sha256"
            ],
            "runtime_dependency_closure_sha256": runtime_dependencies["sha256"],
        },
        "diagnostics": diagnostics,
        "fit_contract": {
            "fit_mode": "smoke" if smoke else "full",
            "effective_iterations": _strict_int(
                config["training"][
                    "smoke_iterations" if smoke else "iterations"
                ],
                "effective iterations",
            ),
            "states": int(diagnostics["states"]),
            "folds": ATTEMPT12_FOLDS,
        },
        "runtime": {
            "safety_enabled": False,
            "winner_frozen": False,
            "activation_allowed": False,
            "current_profile_mutated": False,
            "runtime_source_frozen": True,
            "runtime_requirements_sha256": ATTEMPT12_RUNTIME_REQUIREMENTS_SHA256,
            "runtime_fingerprint_sha256": (
                ATTEMPT12_EXPECTED_RUNTIME_FINGERPRINT_SHA256
            ),
        },
        "science_boundary": {
            "teacher_values_are_realized_match_ev": False,
            "audit50_fit_rows": 0,
            "threshold_sweep_performed": False,
            "top1_accuracy_is_acceptance_gate": False,
        },
    }
    _write_new_json(output_manifest_path, manifest)
    return manifest


def _fit_fold(
    states: Sequence[Attempt12DistillationState],
    *,
    fold_index: int,
    seed: int,
    config: Mapping[str, Any],
    smoke: bool,
) -> Attempt12DistilledFoldPredictor:
    if not states:
        raise ValueError("Attempt12 distilled fold has no training states")
    x = np.vstack([state.features for state in states]).astype(np.float32, copy=False)
    y_rank = np.concatenate([state.relevance for state in states])
    y_delta = np.concatenate([state.delta for state in states])
    y_safe = np.concatenate([state.safe for state in states])
    y_tail = np.vstack([state.tails for state in states])
    groups = [state.group_size for state in states]
    if any(not 1 <= group <= ATTEMPT12_PROPOSAL_ROWS_MAX for group in groups):
        raise ValueError("Attempt12 distilled group size is outside 1..27")
    if sum(groups) != x.shape[0]:
        raise ValueError("Attempt12 distilled LightGBM groups do not match row total")
    state_weight = _profile_state_weights(states)
    weight = np.concatenate(
        [
            np.full(group, value / group)
            for value, group in zip(state_weight, groups, strict=True)
        ]
    )
    iterations = _strict_int(
        config["smoke_iterations"] if smoke else config["iterations"], "iterations"
    )
    common = dict(
        n_estimators=iterations,
        learning_rate=float(config["learning_rate"]),
        num_leaves=_strict_int(config["num_leaves"], "num_leaves"),
        min_child_samples=_strict_int(config["min_child_samples"], "min_child_samples"),
        reg_lambda=float(config["reg_lambda"]),
        reg_alpha=float(config["reg_alpha"]),
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 2, 3, 7],
        random_state=seed,
        **common,
    ).fit(x, y_rank, group=groups, sample_weight=weight)
    delta_head: Any
    if np.allclose(y_delta, y_delta[0]):
        delta_head = _ConstantRegressor(float(y_delta[0]))
    else:
        delta_head = LGBMRegressor(
            objective="huber", random_state=seed + 1, **common
        ).fit(x, y_delta, sample_weight=weight)
    safe_head: Any
    if np.unique(y_safe).size == 1:
        safe_head = _ConstantProbability(float(y_safe[0]))
    else:
        safe_head = LGBMClassifier(
            objective="binary", random_state=seed + 2, **common
        ).fit(x, y_safe, sample_weight=weight * _binary_balance_weights(y_safe))
    tail_heads: list[Any] = []
    for column in range(3):
        if np.allclose(y_tail[:, column], y_tail[0, column]):
            tail_heads.append(_ConstantRegressor(float(y_tail[0, column])))
        else:
            tail_heads.append(
                LGBMRegressor(
                    objective="quantile",
                    alpha=float(config["tail_quantile"]),
                    random_state=seed + 10 + column,
                    **common,
                ).fit(x, y_tail[:, column], sample_weight=weight)
            )
    return Attempt12DistilledFoldPredictor(
        ranker=ranker,
        delta_head=delta_head,
        safe_head=safe_head,
        tail_p95_head=tail_heads[0],
        tail_p99_head=tail_heads[1],
        tail_max_head=tail_heads[2],
        fold_index=fold_index,
    )


def _profile_state_weights(states: Sequence[Attempt12DistillationState]) -> np.ndarray:
    counts = {profile: sum(state.root_profile == profile for state in states) for profile in ATTEMPT12_PROFILES}
    active = [value for value in counts.values() if value]
    result = np.asarray(
        [len(states) / (len(active) * counts[state.root_profile]) for state in states],
        dtype=np.float64,
    )
    return result


def _binary_balance_weights(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int8)
    result = np.ones(values.shape, dtype=np.float64)
    for label in (0, 1):
        count = int(np.sum(values == label))
        if count:
            result[values == label] = len(values) / (2.0 * count)
    return result


def _delta_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "standard_error": float(np.std(values, ddof=1) / math.sqrt(len(values))),
        "p05": float(np.quantile(values, 0.05, method="linear")),
        "p01": float(np.quantile(values, 0.01, method="linear")),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != ATTEMPT12_DISTILLATION_CONFIG_SCHEMA:
        raise ValueError("Attempt12 distillation config schema changed")
    schemas = _mapping(config.get("schemas"), "schemas")
    if (
        schemas.get("model") != HU_M43_ATTEMPT12_DISTILLED_MODEL_SCHEMA
        or schemas.get("feature") != HU_M43_ATTEMPT12_DISTILLED_FEATURE_SCHEMA
        or schemas.get("heads") != HU_M43_ATTEMPT12_DISTILLED_HEAD_SCHEMA
        or schemas.get("training_manifest")
        != ATTEMPT12_DISTILLED_TRAINING_MANIFEST_SCHEMA
    ):
        raise ValueError("Attempt12 distillation model schema changed")
    data = _mapping(config.get("data"), "data")
    if (
        data.get("source_search_plan_sha256") != M43_ATTEMPT12_PLAN_SHA256
        or data.get("development_roots") != ATTEMPT12_DISTILLATION_FULL_STATES
        or data.get("audit50_may_fit_or_calibrate") is not False
        or data.get("attempt11_rows_may_fit_or_calibrate") is not False
    ):
        raise ValueError("Attempt12 distillation data boundary changed")
    candidate = _mapping(config.get("candidate_generation"), "candidate_generation")
    if (
        candidate.get("source_model_sha256") != ATTEMPT12_FROZEN_MODEL_SHA256
        or candidate.get("nonbaseline_max") != ATTEMPT12_CANDIDATE_MAX
        or candidate.get("actual_count_range") != [0, ATTEMPT12_CANDIDATE_MAX]
        or candidate.get("padding_allowed") is not False
        or candidate.get("duplicate_action_keys_allowed") is not False
        or candidate.get("baseline_in_candidate_set") is not False
        or candidate.get("baseline_appended_exactly_once") is not True
        or candidate.get("lightgbm_group_size_range") != [1, ATTEMPT12_PROPOSAL_ROWS_MAX]
        or candidate.get("feature_dimension") != HU_M43_ATTEMPT12_DISTILLED_FEATURE_DIM
    ):
        raise ValueError("Attempt12 distillation candidate contract changed")
    gate = _mapping(config.get("runtime_gate"), "runtime_gate")
    if (
        gate.get("safety_probability_threshold") != 0.5
        or gate.get("tail_upper_limits") != [25.0, 40.0, 50.0]
        or gate.get("minimum_agreeing_fold_votes") != 4
        or gate.get("fold_count") != 5
        or gate.get("action_score_mode") != HU_M43_ATTEMPT12_DISTILLED_ACTION_SCORE_MODE
    ):
        raise ValueError("Attempt12 distillation runtime gate changed")
    training = _mapping(config.get("training"), "training")
    if training.get("threshold_sweep_allowed") is not False:
        raise ValueError("Attempt12 distillation threshold sweep is forbidden")
    authorization = _mapping(config.get("fit_authorization"), "fit_authorization")
    required_authorization = {
        "development_decision_go_required": True,
        "development_selector_receipt_required": True,
        "immutable_development_pass_freeze_required": True,
        "all_three_files_and_development_jsonl_sha256_bound": True,
        "audit50_rows_used_for_fit": False,
        "attempt11_rows_used_for_fit": False,
        "development_decision_schema": ATTEMPT12_DEVELOPMENT_DECISION_SCHEMA,
        "development_receipt_schema": ATTEMPT12_DEVELOPMENT_SELECTOR_RECEIPT_SCHEMA,
        "development_freeze_schema": ATTEMPT12_DEVELOPMENT_PASS_FREEZE_SCHEMA,
        "audit50_decision_schema": ATTEMPT12_AUDIT50_DECISION_SCHEMA,
        "audit50_receipt_schema": ATTEMPT12_AUDIT50_SELECTOR_RECEIPT_SCHEMA,
    }
    if authorization != required_authorization:
        raise ValueError("Attempt12 distillation fit authorization changed")


def _reject_hidden_information(value: Any, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {"opponent_private_discards", "opponent_discards"}:
                raise ValueError(f"Attempt12 distillation exposes hidden information at {path}")
            _reject_hidden_information(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_hidden_information(child, f"{path}[{index}]")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Attempt12 distillation {label} must be a mapping")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"Attempt12 distillation {label} must be a sequence")
    return list(value)


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Attempt12 distillation {label} must be an integer")
    return int(value)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Attempt12 distillation {label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"Attempt12 distillation {label} must be finite")
    return result


def _load_mapping(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return _mapping(value, label)


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Attempt12 distillation {label} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"Attempt12 distillation {label} is invalid")
    return normalized


def _write_new_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit frozen Attempt12 distillation heads")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--development-decision", required=True, type=Path)
    parser.add_argument(
        "--development-selector-receipt", required=True, type=Path
    )
    parser.add_argument("--development-pass-freeze", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--runtime-source-root", required=True, type=Path)
    parser.add_argument("--runtime-dependency-root", required=True, type=Path)
    parser.add_argument("--output-runtime-source-archive", required=True, type=Path)
    parser.add_argument("--output-runtime-source-manifest", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = train_attempt12_distilled_artifact(
        input_path=args.input,
        candidate_model_path=args.candidate_model,
        config_path=args.config,
        development_decision_path=args.development_decision,
        development_selector_receipt_path=args.development_selector_receipt,
        development_pass_freeze_path=args.development_pass_freeze,
        output_model_path=args.output_model,
        output_manifest_path=args.output_manifest,
        runtime_source_root=args.runtime_source_root,
        runtime_dependency_root=args.runtime_dependency_root,
        output_runtime_source_archive_path=args.output_runtime_source_archive,
        output_runtime_source_manifest_path=args.output_runtime_source_manifest,
        smoke=args.smoke,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT12_DISTILLATION_CONFIG_SHA256",
    "ATTEMPT12_DISTILLATION_FULL_ITERATIONS",
    "ATTEMPT12_DISTILLATION_FULL_STATES",
    "ATTEMPT12_DISTILLED_TRAINING_MANIFEST_SCHEMA",
    "ATTEMPT12_PROPOSAL_ROWS_MAX",
    "Attempt12DistillationState",
    "assign_attempt12_identity_folds",
    "fit_attempt12_distilled_model",
    "prepare_attempt12_distillation_states",
    "read_attempt12_jsonl",
    "train_attempt12_distilled_artifact",
]
