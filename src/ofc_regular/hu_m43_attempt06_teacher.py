"""Frozen Attempt06 T1-second top-8 search teacher.

This module is deliberately separate from the Attempt05 runtime policy.  The
frozen Attempt05 LambdaRank ensemble is used only to form an eight-action
non-baseline proposal set.  That set is fixed before any hidden-card particle
is sampled.  A common-random c8 pass then locks one action (the explicit
baseline is also eligible), and a disjoint common-random e128 pass evaluates
only the locked choice and explicit baseline without changing the lock.

All policy/model inputs are reconstructed from :class:`ActorObservation`.
Teacher scores are diagnostics, not realized match EV and not a runtime gate.
The bounded shard command consumes a pre-built canonical root; it never deals
or opens a fresh audit root itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    canonical_descending_indices,
    index_actions_by_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .ai_profiles import ModelPaths, build_policy, load_model_bundle
from .cards import ALL_CARDS
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation, validate_card_free_metadata
from .hu_m43_attempt06_contract import M43_ATTEMPT06_PLAN_SHA256
from .hu_m43_attempt05_model import HuM43Attempt05Model
from .hu_m4_t1_teacher import (
    M4T1TeacherConfig,
    _ActionScores,
    _ChildSelector,
    _paired_delta_summary,
    _score_actions,
    _score_actions_batched,
)
from .hu_m4_teacher_contract import (
    T1_SECOND_LIVE_SCHEDULE,
    require_disjoint_root_rng_keys,
    require_t1_second_root,
)
from .hu_turn3_model import hu_policy_sample


ATTEMPT06_TEACHER_SCHEMA = "hu_m43_attempt06_t1_second_top8_c8_e128_teacher_v3"
ATTEMPT06_ROOT_SCHEMA = "hu_m43_attempt06_t1_second_root_v1"
ATTEMPT06_SHARD_ROW_SCHEMA = "hu_m43_attempt06_t1_second_shard_row_v1"
ATTEMPT06_CHECKPOINT_SCHEMA = "hu_m43_attempt06_t1_second_checkpoint_v1"
ATTEMPT06_HEARTBEAT_SCHEMA = "hu_m43_attempt06_t1_second_heartbeat_v1"
ATTEMPT06_SHARD_SUMMARY_SCHEMA = "hu_m43_attempt06_t1_second_shard_summary_v1"
ATTEMPT06_SOLVER_ID = "attempt05_lambda_top8_m4_c8_locked_pair_e128_m3_mc1_v2"
ATTEMPT06_T2_POLICY_ID = "stage9f_p2"
ATTEMPT06_FROZEN_MODEL_SHA256 = (
    "e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3"
)
ATTEMPT06_FROZEN_MODEL_ID = "hu-m43-attempt05-lambda_rank-dev900-oof"
ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256 = (
    "e9b9d9748bb2b2f31a4baa0d928b60c05f254ab5d7915a7461fb8402e1e7ddb8"
)
ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256 = (
    "ec295d070ac7bb21a82aeb2f432b8f6f191e61b63027bc446e8bd5f39f51569f"
)
ATTEMPT06_PLAN_SHA256 = M43_ATTEMPT06_PLAN_SHA256
ATTEMPT06_ROOT_PROVENANCE_SCHEMA = (
    "hu_m43_attempt06_spot_root_provenance_v1"
)
ATTEMPT06_ROOT_GENERATION_POLICY = "explicit_root_population_live_t0_t1_first_v2"
ATTEMPT06_BASELINE_PROFILE = "stage18_p1"
ATTEMPT06_TOP_K = 8
ATTEMPT06_CANDIDATE_SAMPLES = 8
ATTEMPT06_EVALUATION_SAMPLES = 128
ATTEMPT06_NATIVE_BATCH_THREADS = 4
ATTEMPT06_ROOT_REMATERIALIZATION_MODE = (
    "same_seed_same_frozen_closure_after_matching_claim_and_absent_root_only"
)
ATTEMPT06_AUDIT_ROOTS = 50
ATTEMPT06_SEED_STRIDE = 1_000_003
ATTEMPT06_HAND_SEED_START = 17_306_071_901
ATTEMPT06_CANDIDATE_SEED_START = 23_306_071_901
ATTEMPT06_EVALUATION_SEED_START = 24_306_071_901
ATTEMPT06_CHILD_SEED_START = 25_306_071_901
ATTEMPT06_ROOT_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
_SHA256_HEX = frozenset("0123456789abcdef")
_CARD_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])[2-9TJQKA][hdcs](?![A-Za-z0-9])")
_ROOT_REQUIRED_KEYS = frozenset(
    {
        "schema",
        "root_index",
        "hand_seed",
        "root_profile",
        "policy_observation",
        "baseline_action_key",
        "provenance",
    }
)
_ROOT_PROVENANCE_KEYS = frozenset(
    {
        "schema",
        "run_name",
        "run_id",
        "root_index",
        "root_profile",
        "root_generation_policy",
        "root_policy_seed_base",
        "baseline_profile",
        "plan_sha256",
        "schedule_sha256",
        "candidate_model_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "source_sha256",
        "startup_sha256",
        "status_sha256",
        "source_closure_sha256",
        "native_batch_threads",
        "package_manifest_sha256",
        "global_consumption_marker_sha256",
        "root_consumption_claim_sha256",
        "profile_assignment",
        "current_profile_resolved",
        "opponent_private_discard_input_allowed",
        "teacher_value_status",
        "fresh_audit_retry_or_alternate_sample_allowed",
        "deterministic_claim_recovery_allowed",
        "deterministic_claim_recovery_mode",
    }
)
_TEACHER_PROVENANCE_KEYS = _ROOT_PROVENANCE_KEYS | frozenset(
    {"input_sha256", "config_sha256", "model_sha256"}
)


def _require_sha256(value: str, *, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in _SHA256_HEX for char in normalized):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return normalized


@dataclass(frozen=True)
class Attempt06TeacherConfig:
    """The fixed top8/c8/e128 science contract for one root."""

    frozen_model_sha256: str
    candidate_seed: int
    evaluation_seed: int
    child_policy_seed: int
    run_id: str
    candidate_top_k: int = ATTEMPT06_TOP_K
    candidate_samples: int = ATTEMPT06_CANDIDATE_SAMPLES
    evaluation_samples: int = ATTEMPT06_EVALUATION_SAMPLES
    t2_policy_id: str = ATTEMPT06_T2_POLICY_ID
    t3_candidate_samples: int = 1
    t3_evaluation_samples: int = 1
    t3_downstream_samples: int = 1
    t4_candidate_samples: int = 1
    t4_evaluation_samples: int = 1
    batch_child_selectors: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frozen_model_sha256",
            _require_sha256(self.frozen_model_sha256, name="frozen_model_sha256"),
        )
        if self.frozen_model_sha256 != ATTEMPT06_FROZEN_MODEL_SHA256:
            raise ValueError("Attempt06 frozen_model_sha256 is not the frozen Lambda artifact")
        fixed = {
            "candidate_top_k": ATTEMPT06_TOP_K,
            "candidate_samples": ATTEMPT06_CANDIDATE_SAMPLES,
            "evaluation_samples": ATTEMPT06_EVALUATION_SAMPLES,
            "t3_candidate_samples": 1,
            "t3_evaluation_samples": 1,
            "t3_downstream_samples": 1,
            "t4_candidate_samples": 1,
            "t4_evaluation_samples": 1,
        }
        for name, expected in fixed.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                raise ValueError(f"Attempt06 {name} is fixed at {expected}")
        for name in ("candidate_seed", "evaluation_seed", "child_policy_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if len({self.candidate_seed, self.evaluation_seed, self.child_policy_seed}) != 3:
            raise ValueError("Attempt06 candidate/evaluation/child seeds must be distinct")
        if self.t2_policy_id != ATTEMPT06_T2_POLICY_ID:
            raise ValueError("Attempt06 T2 policy is fixed at stage9f_p2")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("Attempt06 run_id must not be empty")
        if not isinstance(self.batch_child_selectors, bool):
            raise TypeError("batch_child_selectors must be a bool")

    def m4_config(self) -> M4T1TeacherConfig:
        return M4T1TeacherConfig(
            candidate_samples=self.candidate_samples,
            evaluation_samples=self.evaluation_samples,
            candidate_seed=self.candidate_seed,
            evaluation_seed=self.evaluation_seed,
            run_id=self.run_id,
            t2_policy_id=self.t2_policy_id,
            child_policy_seed=self.child_policy_seed,
            t3_candidate_samples=self.t3_candidate_samples,
            t3_evaluation_samples=self.t3_evaluation_samples,
            t3_downstream_samples=self.t3_downstream_samples,
            t4_candidate_samples=self.t4_candidate_samples,
            t4_evaluation_samples=self.t4_evaluation_samples,
            batch_child_selectors=self.batch_child_selectors,
        )


@dataclass(frozen=True)
class Attempt06RankScores:
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]

    def validate(self, action_count: int) -> None:
        if len(self.mean) != action_count or len(self.standard_deviation) != action_count:
            raise ValueError("Attempt06 rank output length disagrees with legal actions")
        if not all(math.isfinite(value) for value in (*self.mean, *self.standard_deviation)):
            raise ValueError("Attempt06 rank output contains non-finite values")
        if any(value < 0.0 for value in self.standard_deviation):
            raise ValueError("Attempt06 rank disagreement must be non-negative")


class Attempt06Ranker(Protocol):
    artifact_sha256: str
    model_id: str

    def score_actions(
        self,
        observation: ActorObservation,
        actions: Sequence[Action],
        *,
        baseline_index: int,
    ) -> Attempt06RankScores: ...


@dataclass(frozen=True)
class FrozenAttempt06LambdaRanker:
    """Hash-bound LambdaRank ensemble used only for candidate generation."""

    model: HuM43Attempt05Model
    artifact_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_sha256",
            _require_sha256(self.artifact_sha256, name="artifact_sha256"),
        )
        if self.model.family != "lambda_rank":
            raise ValueError("Attempt06 candidate generator must be LambdaRank")
        if self.model.model_id != ATTEMPT06_FROZEN_MODEL_ID:
            raise ValueError("Attempt06 candidate generator model_id changed")
        if self.model.runtime_enabled or self.model.winner_frozen:
            raise ValueError(
                "Attempt06 Lambda artifact must remain candidate-only and runtime-disabled"
            )

    @property
    def model_id(self) -> str:
        return self.model.model_id

    @classmethod
    def load(
        cls, path: str | Path, *, expected_sha256: str
    ) -> "FrozenAttempt06LambdaRanker":
        expected = _require_sha256(expected_sha256, name="expected_sha256")
        model = HuM43Attempt05Model.load(path, expected_sha256=expected)
        return cls(model=model, artifact_sha256=expected)

    def score_actions(
        self,
        observation: ActorObservation,
        actions: Sequence[Action],
        *,
        baseline_index: int,
    ) -> Attempt06RankScores:
        if observation.street != "T1" or observation.seat != "second":
            raise ValueError("Attempt06 ranker authorizes only T1-second")
        if not 0 <= baseline_index < len(actions):
            raise ValueError("Attempt06 baseline index is invalid")
        sample = hu_policy_sample(
            observation.hero_board,
            observation.dealt_cards,
            actions,
            opponent_board=observation.opponent_public_board,
            dead_cards=observation.legacy_dead_cards(),
            seat=observation.seat,
            to_act_order=observation.to_act_order,
        )
        sample["policy_observation"] = observation.to_dict()
        sample["baseline_action_row_index"] = baseline_index
        sample["baseline_action_key"] = action_key(actions[baseline_index]).to_token()
        fold_scores = []
        for predictor in sorted(
            self.model.fold_predictors, key=lambda item: item.fold_index
        ):
            output = predictor.predict(sample, baseline_index=baseline_index)
            values = np.asarray(output.rank_score, dtype=np.float64)
            if values.shape != (len(actions),) or not np.isfinite(values).all():
                raise ValueError("Attempt06 LambdaRank fold output is invalid")
            fold_scores.append(values)
        matrix = np.vstack(fold_scores)
        result = Attempt06RankScores(
            mean=tuple(float(value) for value in np.mean(matrix, axis=0)),
            standard_deviation=tuple(float(value) for value in np.std(matrix, axis=0)),
        )
        result.validate(len(actions))
        return result


def _require_stage9f_p2_policies(t2_policies: Mapping[str, object]) -> None:
    if set(t2_policies) != {"first", "second"}:
        raise ValueError("t2_policies must contain exactly first and second")
    for seat in ("first", "second"):
        policy = t2_policies[seat]
        if getattr(policy, "seat", None) != seat:
            raise ValueError(f"stage9f_p2 policy seat mismatch for {seat}")
        context = getattr(policy, "topk_context", None)
        if not isinstance(context, Mapping):
            raise ValueError("Attempt06 requires actual stage9f_p2 topk_context")
        if context.get("runtime_profile") != ATTEMPT06_T2_POLICY_ID:
            raise ValueError("Attempt06 T2 policy runtime_profile is not stage9f_p2")
        if context.get("runtime_status") != "p2_fixed":
            raise ValueError("Attempt06 T2 policy runtime_status is not p2_fixed")


def _require_concrete_stage9f_p2_policies(
    t2_policies: Mapping[str, object]
) -> None:
    """Reject metadata lookalikes at the production shard boundary."""

    _require_stage9f_p2_policies(t2_policies)
    from .evaluate_hu_turn2_stage8b_topk_mc_rerank import (
        HuTurn2Stage8bTopKMcRerankPolicy,
    )

    if any(
        type(t2_policies[seat]) is not HuTurn2Stage8bTopKMcRerankPolicy
        for seat in ("first", "second")
    ):
        raise TypeError("Attempt06 shard requires concrete stage9f_p2 policy objects")


def _reject_card_tokens(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_card_tokens(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_card_tokens(child, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and (
        value in ALL_CARDS or _CARD_TOKEN_PATTERN.search(value) is not None
    ):
        raise ValueError(f"Attempt06 provenance contains a card token at {path}")


def _validate_root_provenance(
    provenance: Mapping[str, Any],
    *,
    root_profile: str,
    root_index: int,
    teacher_output: bool = False,
) -> None:
    validate_card_free_metadata(provenance, path="root.provenance")
    _reject_card_tokens(provenance, path="root.provenance")
    expected_keys = (
        _TEACHER_PROVENANCE_KEYS if teacher_output else _ROOT_PROVENANCE_KEYS
    )
    if set(provenance) != expected_keys:
        raise ValueError("Attempt06 root provenance fields changed")
    required = {
        "schema": ATTEMPT06_ROOT_PROVENANCE_SCHEMA,
        "root_index": root_index,
        "root_profile": root_profile,
        "root_policy_seed_base": ATTEMPT06_HAND_SEED_START,
        "root_generation_policy": ATTEMPT06_ROOT_GENERATION_POLICY,
        "baseline_profile": ATTEMPT06_BASELINE_PROFILE,
        "plan_sha256": ATTEMPT06_PLAN_SHA256,
        "candidate_model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
        "source_model_manifest_sha256": ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256,
        "source_native_manifest_sha256": ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256,
        "native_batch_threads": ATTEMPT06_NATIVE_BATCH_THREADS,
        "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
        "current_profile_resolved": False,
        "opponent_private_discard_input_allowed": False,
        "teacher_value_status": "diagnostic_not_match_EV",
        "fresh_audit_retry_or_alternate_sample_allowed": False,
        "deterministic_claim_recovery_allowed": True,
        "deterministic_claim_recovery_mode": ATTEMPT06_ROOT_REMATERIALIZATION_MODE,
    }
    if any(provenance.get(key) != value for key, value in required.items()):
        raise ValueError("Attempt06 root provenance policy lineage mismatch")
    run_name = provenance.get("run_name")
    run_id = provenance.get("run_id")
    if (
        not isinstance(run_name, str)
        or not run_name
        or run_id != f"{run_name}:shard={root_index}"
    ):
        raise ValueError("Attempt06 root provenance run identity mismatch")
    for key in (
        "schedule_sha256",
        "package_manifest_sha256",
        "global_consumption_marker_sha256",
        "root_consumption_claim_sha256",
        "source_sha256",
        "startup_sha256",
        "status_sha256",
        "source_closure_sha256",
    ):
        _require_sha256(str(provenance.get(key, "")), name=f"provenance.{key}")
    if teacher_output:
        for key in ("input_sha256", "config_sha256", "model_sha256"):
            _require_sha256(
                str(provenance.get(key, "")), name=f"provenance.{key}"
            )
        if provenance.get("model_sha256") != ATTEMPT06_FROZEN_MODEL_SHA256:
            raise ValueError("Attempt06 teacher provenance model SHA-256 changed")


def _resolve_baseline(actions: Sequence[Action], token: str) -> tuple[int, ActionKey]:
    key = ActionKey.from_token(token)
    mapping = index_actions_by_key(actions)
    try:
        return mapping[key], key
    except KeyError as exc:
        raise ValueError("Attempt06 explicit baseline ActionKey is not legal") from exc


def _ordered_keys_digest(actions: Sequence[Action]) -> str:
    return hashlib.sha256(
        "\n".join(action_key(action).to_token() for action in actions).encode("ascii")
    ).hexdigest()


def _loss_summary(delta: Mapping[str, Any]) -> dict[str, float]:
    return {
        "p95": max(0.0, -float(delta["p05"])),
        "p99": max(0.0, -float(delta["p01"])),
        "max": max(0.0, -float(delta["min"])),
    }


def evaluate_attempt06_t1_second(
    observation: ActorObservation,
    *,
    baseline_action_key: str,
    ranker: Attempt06Ranker,
    t2_policies: Mapping[str, object],
    config: Attempt06TeacherConfig,
    library: Any | None = None,
) -> dict[str, Any]:
    """Run frozen top8 -> c8 lock -> independent e128 evaluation."""

    require_t1_second_root(observation)
    if ranker.artifact_sha256 != config.frozen_model_sha256:
        raise ValueError("Attempt06 ranker artifact hash disagrees with config")
    _require_stage9f_p2_policies(t2_policies)

    legal_actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    if not legal_actions:
        raise ValueError("Attempt06 root has no legal actions")
    baseline_index, baseline_key = _resolve_baseline(
        legal_actions, baseline_action_key
    )

    # Candidate generation is complete before either particle namespace opens.
    rank_scores = ranker.score_actions(
        observation, legal_actions, baseline_index=baseline_index
    )
    rank_scores.validate(len(legal_actions))
    nonbaseline_indices = [
        index for index in range(len(legal_actions)) if index != baseline_index
    ]
    if len(nonbaseline_indices) < config.candidate_top_k:
        raise ValueError("Attempt06 root has fewer than eight nonbaseline actions")
    nonbaseline_indices.sort(
        key=lambda index: (
            -rank_scores.mean[index],
            action_key(legal_actions[index]).sort_key(),
        )
    )
    top_indices = tuple(nonbaseline_indices[: config.candidate_top_k])
    candidate_indices = (*top_indices, baseline_index)
    candidate_actions = tuple(legal_actions[index] for index in candidate_indices)
    if len({action_key(action) for action in candidate_actions}) != 9:
        raise AssertionError("Attempt06 candidate set must be top8 plus baseline once")

    candidate_batch = sample_hidden_card_particles(
        observation,
        base_seed=config.candidate_seed,
        run_id=f"{config.run_id}:candidate_selection",
        sample_count=config.candidate_samples,
    )
    candidate_batch.validate_against(observation)
    candidate_keys = tuple(row.rng_key_digest for row in candidate_batch.particles)
    if len(candidate_keys) != ATTEMPT06_CANDIDATE_SAMPLES:
        raise ValueError("Attempt06 candidate particle count changed")

    m4_config = config.m4_config()
    selector = _ChildSelector(
        t2_policies=t2_policies,
        config=m4_config,
        library=library,
    )
    scorer = _score_actions_batched if config.batch_child_selectors else _score_actions
    candidate_scores = scorer(
        observation, candidate_actions, candidate_batch, selector
    )
    if len(candidate_scores) != len(candidate_actions):
        raise ValueError("Attempt06 c8 scorer returned the wrong action count")
    selection_values = tuple(row.mean for row in candidate_scores)
    selection_ranking = canonical_descending_indices(selection_values, candidate_actions)
    locked_candidate_position = selection_ranking[0]
    locked_original_index = candidate_indices[locked_candidate_position]
    locked_key = action_key(legal_actions[locked_original_index]).to_token()

    # Only after the semantic c8 lock exists may the independent e128
    # namespace be opened.  The e128 result cannot change ``locked_key``.
    evaluation_batch = sample_hidden_card_particles(
        observation,
        base_seed=config.evaluation_seed,
        run_id=f"{config.run_id}:locked_evaluation",
        sample_count=config.evaluation_samples,
    )
    evaluation_batch.validate_against(observation)
    evaluation_keys = tuple(row.rng_key_digest for row in evaluation_batch.particles)
    if len(evaluation_keys) != ATTEMPT06_EVALUATION_SAMPLES:
        raise ValueError("Attempt06 evaluation particle count changed")
    require_disjoint_root_rng_keys(candidate_keys, evaluation_keys)
    baseline_candidate_position = len(candidate_actions) - 1
    evaluation_candidate_positions = tuple(
        dict.fromkeys((locked_candidate_position, baseline_candidate_position))
    )
    evaluation_actions = tuple(
        candidate_actions[position] for position in evaluation_candidate_positions
    )
    evaluation_scores = scorer(
        observation, evaluation_actions, evaluation_batch, selector
    )
    if len(evaluation_scores) != len(evaluation_actions):
        raise ValueError("Attempt06 locked-pair e128 scorer returned the wrong action count")
    evaluation_by_candidate_position = dict(
        zip(evaluation_candidate_positions, evaluation_scores, strict=True)
    )
    baseline_evaluation = evaluation_by_candidate_position[
        baseline_candidate_position
    ]
    evaluation_values = tuple(row.mean for row in evaluation_scores)
    evaluation_ranking = canonical_descending_indices(
        evaluation_values, evaluation_actions
    )
    evaluation_best_candidate_position = evaluation_candidate_positions[
        evaluation_ranking[0]
    ]
    second_selection = selection_values[
        selection_ranking[min(1, len(selection_ranking) - 1)]
    ]

    scored_rows: list[dict[str, Any]] = []
    for candidate_position, original_index in enumerate(candidate_indices):
        action = legal_actions[original_index]
        evaluation_score = evaluation_by_candidate_position.get(candidate_position)
        delta = (
            _paired_delta_summary(evaluation_score, baseline_evaluation)
            if evaluation_score is not None
            else None
        )
        loss = _loss_summary(delta) if delta is not None else None
        scored_rows.append(
            {
                "candidate_position": candidate_position,
                "original_legal_index": original_index,
                "learned_nonbaseline_rank": (
                    candidate_position + 1
                    if candidate_position < config.candidate_top_k
                    else None
                ),
                "action_key": action_key(action).to_token(),
                "placements": [list(value) for value in action.placements],
                "discards": list(action.discards),
                "model_rank_score": rank_scores.mean[original_index],
                "model_rank_disagreement": rank_scores.standard_deviation[original_index],
                "candidate_mean": candidate_scores[candidate_position].mean,
                "candidate_standard_error": candidate_scores[
                    candidate_position
                ].standard_error,
                "evaluation_mean": (
                    evaluation_score.mean if evaluation_score is not None else None
                ),
                "evaluation_standard_error": (
                    evaluation_score.standard_error
                    if evaluation_score is not None
                    else None
                ),
                "paired_evaluation_delta_vs_baseline": delta,
                "paired_evaluation_override_loss": loss,
                "paired_evaluation_state_mean_loss_diagnostic": (
                    max(0.0, -float(delta["mean"]))
                    if delta is not None
                    else None
                ),
                "evaluation_scope": (
                    "locked_action_and_explicit_baseline"
                    if evaluation_score is not None
                    else "not_evaluated_after_c8_lock"
                ),
                "is_explicit_baseline": candidate_position
                == baseline_candidate_position,
                "selected_by_c8": candidate_position == locked_candidate_position,
                "evaluation_sample_best": (
                    candidate_position == evaluation_best_candidate_position
                    if evaluation_score is not None
                    else None
                ),
            }
        )

    legal_rows = []
    top_index_set = set(top_indices)
    for original_index, action in enumerate(legal_actions):
        legal_rows.append(
            {
                "original_legal_index": original_index,
                "action_key": action_key(action).to_token(),
                "model_rank_score": rank_scores.mean[original_index],
                "model_rank_disagreement": rank_scores.standard_deviation[
                    original_index
                ],
                "in_learned_top8": original_index in top_index_set,
                "is_explicit_baseline": original_index == baseline_index,
            }
        )

    locked_row = scored_rows[locked_candidate_position]
    locked_evaluation = evaluation_by_candidate_position[
        locked_candidate_position
    ]
    selected_paired_evaluation_deltas = [
        float(selected_value - baseline_value)
        for selected_value, baseline_value in zip(
            locked_evaluation.values,
            baseline_evaluation.values,
            strict=True,
        )
    ]
    if (
        len(selected_paired_evaluation_deltas)
        != ATTEMPT06_EVALUATION_SAMPLES
        or not all(
            math.isfinite(value)
            for value in selected_paired_evaluation_deltas
        )
    ):
        raise ValueError("Attempt06 selected paired e128 raw deltas changed")
    override_fired = locked_key != baseline_key.to_token()
    return {
        "status": "ok",
        "schema": ATTEMPT06_TEACHER_SCHEMA,
        "solver_id": ATTEMPT06_SOLVER_ID,
        "street": "T1",
        "seat": "second",
        "to_act_order": "second",
        "observation_fingerprint": observation.fingerprint(),
        "action_key_schema": ACTION_KEY_SCHEMA,
        "frozen_candidate_generator": {
            "family": "lambda_rank",
            "model_id": ranker.model_id,
            "artifact_sha256": ranker.artifact_sha256,
            "purpose": "candidate_generation_only",
            "runtime_authorized": False,
            "profile_runtime_feature": False,
        },
        "legal_action_count": len(legal_actions),
        "legal_action_set_digest": legal_action_set_digest(legal_actions),
        "legal_action_order_digest": ordered_action_mapping_digest(legal_actions),
        "legal_action_keys": [
            action_key(action).to_token() for action in legal_actions
        ],
        "legal_actions": legal_rows,
        "learned_top8_action_keys": [
            action_key(legal_actions[index]).to_token() for index in top_indices
        ],
        "learned_top8_order_digest": _ordered_keys_digest(
            tuple(legal_actions[index] for index in top_indices)
        ),
        "candidate_action_count": len(candidate_actions),
        "candidate_action_set_digest": legal_action_set_digest(candidate_actions),
        "candidate_action_order_digest": ordered_action_mapping_digest(
            candidate_actions
        ),
        "evaluation_action_count": len(evaluation_actions),
        "evaluation_action_keys": [
            action_key(action).to_token() for action in evaluation_actions
        ],
        "evaluation_action_order_digest": ordered_action_mapping_digest(
            evaluation_actions
        ),
        "evaluation_scope": "c8_locked_action_plus_explicit_baseline_only",
        "baseline_action_key": baseline_key.to_token(),
        "baseline_original_legal_index": baseline_index,
        "selected_action_key": locked_key,
        "selected_action_original_legal_index": locked_original_index,
        "selected_action_candidate_position": locked_candidate_position,
        "override_fired": override_fired,
        "selection_score_gap": selection_values[locked_candidate_position]
        - second_selection,
        "selected_action_evaluation_mean": locked_row["evaluation_mean"],
        "selected_action_evaluation_standard_error": locked_row[
            "evaluation_standard_error"
        ],
        "selected_action_paired_evaluation_delta_vs_baseline": locked_row[
            "paired_evaluation_delta_vs_baseline"
        ],
        "selected_action_paired_evaluation_deltas_vs_baseline": (
            selected_paired_evaluation_deltas
        ),
        "selected_action_paired_evaluation_deltas_sha256": hashlib.sha256(
            json.dumps(
                selected_paired_evaluation_deltas,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
        "selected_action_paired_evaluation_override_loss": locked_row[
            "paired_evaluation_override_loss"
        ],
        "selected_action_state_mean_loss_diagnostic": locked_row[
            "paired_evaluation_state_mean_loss_diagnostic"
        ],
        "audit_aggregation_contract": {
            "fire_definition": "selected_action_key_differs_from_explicit_baseline",
            "per_root_tail_source": (
                "selected_action_paired_evaluation_override_loss"
            ),
            "aggregate_over_fires": {
                "p95": "maximum_of_per_root_e128_p95",
                "p99": "maximum_of_per_root_e128_p99",
                "max": "maximum_of_per_root_e128_max",
            },
            "quantile_method": "numpy_linear",
            "state_mean_loss_is_non_gate_diagnostic": True,
        },
        "evaluation_sample_best_action_key": action_key(
            candidate_actions[evaluation_best_candidate_position]
        ).to_token(),
        "candidate_belief_digest": candidate_batch.digest(),
        "evaluation_belief_digest": evaluation_batch.digest(),
        "candidate_rng_key_digests": list(candidate_keys),
        "evaluation_rng_key_digests": list(evaluation_keys),
        "sample_independence": "disjoint_particle_rng_keys",
        "root_selection_lock": "top8_fixed_before_sampling_then_c8_locked_before_e128",
        "search_config": {
            "learned_nonbaseline_top_k": config.candidate_top_k,
            "baseline_added_exactly_once": True,
            "candidate_samples": config.candidate_samples,
            "evaluation_samples": config.evaluation_samples,
            "evaluation_action_scope": "locked_action_plus_explicit_baseline_only",
            "candidate_seed": config.candidate_seed,
            "evaluation_seed": config.evaluation_seed,
            "child_policy_seed": config.child_policy_seed,
            "run_id": config.run_id,
            "batch_child_selectors": config.batch_child_selectors,
            "candidate_tie_break": "ActionKey",
            "search_tie_break": "ActionKey",
        },
        "continuation_policy": {
            "t2_policy_id": ATTEMPT06_T2_POLICY_ID,
            "t2_resolution": "explicit_profile_never_current",
            "t2_runtime_status": "p2_fixed",
            "t3_selector": "m3_rust_evaluate_t3",
            "t3_candidate_samples": 1,
            "t3_evaluation_samples": 1,
            "t3_downstream_samples": 1,
            "hypothetical_t4_selector": "m3_rust_evaluate_t4",
            "hypothetical_t4_candidate_samples": 1,
            "hypothetical_t4_evaluation_samples": 1,
            "real_live_t4_exact_unchanged": True,
        },
        "live_schedule": [
            {
                "seat": step.seat,
                "street": step.street,
                "draw_offset": step.draw_offset,
            }
            for step in T1_SECOND_LIVE_SCHEDULE
        ],
        "actions": scored_rows,
        "child_information_set_count": len(selector.cache),
        "teacher_value_status": "diagnostic_not_match_EV",
        "runtime_gate_allowed": False,
    }


@dataclass(frozen=True)
class Attempt06Root:
    root_index: int
    hand_seed: int
    root_profile: str
    observation: ActorObservation
    baseline_action_key: str
    provenance: Mapping[str, Any]


def load_attempt06_roots(path: str | Path) -> tuple[Attempt06Root, ...]:
    """Load the bounded external root file; the audit uses one root/shard."""

    rows = []
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Attempt06 root line {line_number} is invalid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != _ROOT_REQUIRED_KEYS:
            raise ValueError("Attempt06 root has missing or extra fields")
        if payload.get("schema") != ATTEMPT06_ROOT_SCHEMA:
            raise ValueError("Attempt06 root schema mismatch")
        root_index = payload.get("root_index")
        hand_seed = payload.get("hand_seed")
        if isinstance(root_index, bool) or not isinstance(root_index, int):
            raise ValueError("Attempt06 root_index must be an integer")
        if not 0 <= root_index < ATTEMPT06_AUDIT_ROOTS:
            raise ValueError("Attempt06 root_index is outside the frozen audit")
        if isinstance(hand_seed, bool) or not isinstance(hand_seed, int):
            raise ValueError("Attempt06 hand_seed must be an integer")
        expected_hand_seed = ATTEMPT06_HAND_SEED_START + ATTEMPT06_SEED_STRIDE * root_index
        if hand_seed != expected_hand_seed:
            raise ValueError("Attempt06 hand_seed disagrees with frozen schedule")
        expected_profile = ATTEMPT06_ROOT_PROFILES[
            root_index % len(ATTEMPT06_ROOT_PROFILES)
        ]
        if payload.get("root_profile") != expected_profile:
            raise ValueError("Attempt06 root profile disagrees with root_index mod 5")
        raw_observation = payload.get("policy_observation")
        if not isinstance(raw_observation, Mapping):
            raise ValueError("Attempt06 root policy_observation is missing")
        observation = ActorObservation.from_dict(raw_observation)
        if dict(raw_observation) != observation.to_dict():
            raise ValueError("Attempt06 root policy_observation is non-canonical")
        require_t1_second_root(observation)
        baseline_token = payload.get("baseline_action_key")
        if not isinstance(baseline_token, str):
            raise ValueError("Attempt06 root baseline_action_key must be a string")
        actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
        _resolve_baseline(actions, baseline_token)
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Attempt06 root provenance must be a mapping")
        _validate_root_provenance(
            provenance, root_profile=expected_profile, root_index=root_index
        )
        rows.append(
            Attempt06Root(
                root_index=root_index,
                hand_seed=hand_seed,
                root_profile=expected_profile,
                observation=observation,
                baseline_action_key=baseline_token,
                provenance=dict(provenance),
            )
        )
    if len(rows) != 1:
        raise ValueError("Attempt06 audit contract requires exactly one root per shard")
    return tuple(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def attempt06_fixed_contract(
    *,
    root_index: int,
    input_sha256: str,
    model_sha256: str,
    source_model_manifest_sha256: str,
    source_native_manifest_sha256: str,
    run_id: str,
    batch_child_selectors: bool = True,
    native_batch_threads: int = ATTEMPT06_NATIVE_BATCH_THREADS,
) -> dict[str, Any]:
    """Return the complete hashable one-root execution contract."""

    if isinstance(root_index, bool) or not isinstance(root_index, int):
        raise TypeError("Attempt06 fixed-contract root_index must be an integer")
    if not 0 <= root_index < ATTEMPT06_AUDIT_ROOTS:
        raise ValueError("Attempt06 fixed-contract root_index is outside 0..49")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Attempt06 fixed-contract run_id is empty")
    if not isinstance(batch_child_selectors, bool):
        raise TypeError("Attempt06 fixed-contract batch flag must be bool")
    if (
        isinstance(native_batch_threads, bool)
        or not isinstance(native_batch_threads, int)
        or native_batch_threads != ATTEMPT06_NATIVE_BATCH_THREADS
    ):
        raise ValueError(
            "Attempt06 fixed-contract native batch threads must equal "
            f"{ATTEMPT06_NATIVE_BATCH_THREADS}"
        )
    input_hash = _require_sha256(input_sha256, name="input_sha256")
    model_hash = _require_sha256(model_sha256, name="model_sha256")
    source_model_hash = _require_sha256(
        source_model_manifest_sha256,
        name="source_model_manifest_sha256",
    )
    source_native_hash = _require_sha256(
        source_native_manifest_sha256,
        name="source_native_manifest_sha256",
    )
    if model_hash != ATTEMPT06_FROZEN_MODEL_SHA256:
        raise ValueError("Attempt06 fixed-contract model hash changed")
    if source_model_hash != ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256:
        raise ValueError("Attempt06 fixed-contract source-model hash changed")
    if source_native_hash != ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256:
        raise ValueError("Attempt06 fixed-contract source-native hash changed")
    return {
        "schema": ATTEMPT06_SHARD_SUMMARY_SCHEMA,
        "teacher_schema": ATTEMPT06_TEACHER_SCHEMA,
        "plan_sha256": ATTEMPT06_PLAN_SHA256,
        "input_sha256": input_hash,
        "model_sha256": model_hash,
        "source_model_manifest_sha256": source_model_hash,
        "source_native_manifest_sha256": source_native_hash,
        "root_index": root_index,
        "run_id": run_id,
        "batch_child_selectors": batch_child_selectors,
        "native_batch_threads": native_batch_threads,
        "top_k": ATTEMPT06_TOP_K,
        "candidate_samples": ATTEMPT06_CANDIDATE_SAMPLES,
        "evaluation_samples": ATTEMPT06_EVALUATION_SAMPLES,
        "evaluation_action_scope": (
            "locked_action_plus_explicit_baseline_only"
        ),
        "t2_policy_id": ATTEMPT06_T2_POLICY_ID,
        "t3_candidate_samples": 1,
        "t3_evaluation_samples": 1,
        "t3_downstream_samples": 1,
        "hypothetical_t4_candidate_samples": 1,
        "hypothetical_t4_evaluation_samples": 1,
        "real_live_t4_exact_unchanged": True,
    }


def attempt06_fixed_contract_sha256(**kwargs: Any) -> str:
    return _canonical_sha256(attempt06_fixed_contract(**kwargs))


def _canonical_path_identity(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _require_distinct_paths(paths: Mapping[str, Path]) -> None:
    identities: dict[str, str] = {}
    for name, path in paths.items():
        identity = _canonical_path_identity(path)
        if identity in identities:
            raise ValueError(
                f"Attempt06 paths must be distinct: {name} aliases {identities[identity]}"
            )
        identities[identity] = name
    items = tuple(paths.items())
    for left_index, (left_name, left_path) in enumerate(items):
        if not left_path.exists():
            continue
        for right_name, right_path in items[left_index + 1 :]:
            if not right_path.exists():
                continue
            try:
                same_file = os.path.samefile(left_path, right_path)
            except OSError as exc:
                raise ValueError(
                    "Attempt06 could not verify existing path identities"
                ) from exc
            if same_file:
                raise ValueError(
                    "Attempt06 paths must be distinct: "
                    f"{right_name} hard-links or aliases {left_name}"
                )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class _HeartbeatPump:
    """Write a live atomic heartbeat while one long root is evaluated."""

    def __init__(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        interval_seconds: float = 30.0,
    ) -> None:
        self.path = path
        self.payload = dict(payload)
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="attempt06-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        self._write()
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_seconds + 1.0))
        if self.thread.is_alive():
            raise RuntimeError("Attempt06 heartbeat thread did not stop")
        if self.error is not None:
            raise RuntimeError("Attempt06 live heartbeat write failed") from self.error

    def _run(self) -> None:
        try:
            while not self.stop_event.wait(self.interval_seconds):
                self._write()
        except BaseException as exc:  # surfaced synchronously by ``stop``
            self.error = exc
            self.stop_event.set()

    def _write(self) -> None:
        _atomic_json(
            self.path,
            {**self.payload, "updated_unix_seconds": time.time()},
        )


class _ShardFileLock:
    """Non-blocking process lock; the tiny lock file may safely persist."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any | None = None

    def __enter__(self) -> "_ShardFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("Attempt06 shard is already owned by another worker") from exc
        self.handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        handle = self.handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self.handle = None


def _resume_one_root(
    partial_path: Path,
    checkpoint_path: Path,
    *,
    config_sha256: str,
    input_sha256: str,
    model_sha256: str,
    source_model_manifest_sha256: str,
    source_native_manifest_sha256: str,
    expected_root_index: int,
) -> int:
    checkpoint: Mapping[str, Any] | None = None
    if checkpoint_path.exists():
        loaded = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, Mapping):
            raise ValueError("Attempt06 checkpoint must be a mapping")
        checkpoint = loaded
        expected_checkpoint = {
            "schema": ATTEMPT06_CHECKPOINT_SCHEMA,
            "config_sha256": config_sha256,
            "input_sha256": input_sha256,
            "model_sha256": model_sha256,
            "source_model_manifest_sha256": source_model_manifest_sha256,
            "source_native_manifest_sha256": source_native_manifest_sha256,
            "target_roots": 1,
            "root_index": expected_root_index,
        }
        if any(
            checkpoint.get(key) != value
            for key, value in expected_checkpoint.items()
        ):
            raise ValueError("Attempt06 checkpoint configuration or source mismatch")
        if checkpoint.get("completed_roots") not in (0, 1):
            raise ValueError("Attempt06 checkpoint completed_roots is invalid")

    if not partial_path.exists():
        if checkpoint is None:
            return 0
        if checkpoint.get("completed_roots") == 0 and checkpoint.get(
            "partial_sha256"
        ) == hashlib.sha256(b"").hexdigest():
            # Normal Spot preemption during one long root: no content was
            # committed, so the root is recomputed under the same closure.
            return 0
        raise ValueError("Attempt06 completed checkpoint has no partial output")

    raw = partial_path.read_bytes()
    if checkpoint is None:
        # This is the narrow crash window before the initial checkpoint in an
        # older worker.  Nothing is committed; discard only this shard partial
        # and recompute the one root.
        if raw:
            with partial_path.open("r+b") as handle:
                handle.truncate(0)
        return 0

    completed = int(checkpoint["completed_roots"])
    newline_offsets = [index + 1 for index, value in enumerate(raw) if value == 10]
    boundary = newline_offsets[completed - 1] if completed else 0
    if len(raw) != boundary:
        with partial_path.open("r+b") as handle:
            handle.truncate(boundary)
        raw = raw[:boundary]
    if hashlib.sha256(raw).hexdigest() != checkpoint.get("partial_sha256"):
        raise ValueError("Attempt06 partial hash disagrees with checkpoint")
    if completed:
        row = json.loads(raw.decode("utf-8").strip())
        if (
            row.get("schema") != ATTEMPT06_SHARD_ROW_SCHEMA
            or row.get("root_index") != expected_root_index
        ):
            raise ValueError("Attempt06 checkpointed row identity mismatch")
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Attempt06 checkpointed row provenance is missing")
        expected_provenance = {
            "config_sha256": config_sha256,
            "input_sha256": input_sha256,
            "model_sha256": model_sha256,
            "source_model_manifest_sha256": source_model_manifest_sha256,
            "source_native_manifest_sha256": source_native_manifest_sha256,
        }
        if any(
            provenance.get(key) != value
            for key, value in expected_provenance.items()
        ):
            raise ValueError("Attempt06 checkpointed row hash provenance mismatch")
    return int(completed)


def _run_attempt06_shard_locked(
    *,
    input_roots: str | Path,
    output: str | Path,
    checkpoint: str | Path,
    heartbeat: str | Path,
    model: str | Path,
    model_sha256: str,
    source_model_manifest_sha256: str,
    source_native_manifest_sha256: str,
    run_id: str,
    batch_child_selectors: bool = False,
    native_batch_threads: int = 4,
    paths: ModelPaths | None = None,
) -> dict[str, Any]:
    """Evaluate one pre-built root with fsync/checkpoint/resume semantics."""

    if not 1 <= native_batch_threads <= 64:
        raise ValueError("native_batch_threads must be between 1 and 64")
    if not run_id:
        raise ValueError("run_id must not be empty")
    input_path = Path(input_roots)
    output_path = Path(output)
    partial_path = output_path.with_name(output_path.name + ".partial")
    checkpoint_path = Path(checkpoint)
    heartbeat_path = Path(heartbeat)
    model_path = Path(model)
    lock_path = output_path.with_name(output_path.name + ".lock")
    _require_distinct_paths(
        {
            "input_roots": input_path,
            "output": output_path,
            "partial": partial_path,
            "checkpoint": checkpoint_path,
            "heartbeat": heartbeat_path,
            "model": model_path,
            "lock": lock_path,
        }
    )
    if output_path.exists():
        raise FileExistsError(f"Attempt06 shard already complete: {output_path}")
    roots = load_attempt06_roots(input_path)
    root = roots[0]
    if root.provenance.get("run_id") != run_id:
        raise ValueError("Attempt06 CLI run_id disagrees with immutable root provenance")
    expected_model_hash = _require_sha256(model_sha256, name="model_sha256")
    if expected_model_hash != ATTEMPT06_FROZEN_MODEL_SHA256:
        raise ValueError("Attempt06 model_sha256 is not the frozen Lambda artifact")
    source_model_hash = _require_sha256(
        source_model_manifest_sha256, name="source_model_manifest_sha256"
    )
    source_native_hash = _require_sha256(
        source_native_manifest_sha256, name="source_native_manifest_sha256"
    )
    if source_model_hash != ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256:
        raise ValueError("Attempt06 source model manifest hash changed")
    if source_native_hash != ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256:
        raise ValueError("Attempt06 source native manifest hash changed")
    if batch_child_selectors:
        os.environ["OFC_HU_M3_BATCH_THREADS"] = str(native_batch_threads)
    fixed_contract = attempt06_fixed_contract(
        root_index=root.root_index,
        input_sha256=_sha256_file(input_path),
        model_sha256=expected_model_hash,
        source_model_manifest_sha256=source_model_hash,
        source_native_manifest_sha256=source_native_hash,
        run_id=run_id,
        batch_child_selectors=batch_child_selectors,
        native_batch_threads=native_batch_threads,
    )
    config_sha256 = _canonical_sha256(fixed_contract)
    completed = _resume_one_root(
        partial_path,
        checkpoint_path,
        config_sha256=config_sha256,
        input_sha256=str(fixed_contract["input_sha256"]),
        model_sha256=expected_model_hash,
        source_model_manifest_sha256=source_model_hash,
        source_native_manifest_sha256=source_native_hash,
        expected_root_index=root.root_index,
    )
    started = time.perf_counter()
    if completed == 0:
        empty_sha256 = hashlib.sha256(b"").hexdigest()
        starting_common = {
            "config_sha256": config_sha256,
            "input_sha256": fixed_contract["input_sha256"],
            "model_sha256": expected_model_hash,
            "source_model_manifest_sha256": source_model_hash,
            "source_native_manifest_sha256": source_native_hash,
            "completed_roots": 0,
            "target_roots": 1,
            "root_index": root.root_index,
            "partial_sha256": empty_sha256,
        }
        _atomic_json(
            checkpoint_path,
            {
                "schema": ATTEMPT06_CHECKPOINT_SCHEMA,
                **starting_common,
                "updated_unix_seconds": time.time(),
            },
        )
        heartbeat_pump = _HeartbeatPump(
            heartbeat_path,
            {
                "schema": ATTEMPT06_HEARTBEAT_SCHEMA,
                "status": "running",
                **starting_common,
            },
        )
        heartbeat_pump.start()
        try:
            ranker = FrozenAttempt06LambdaRanker.load(
                model_path, expected_sha256=expected_model_hash
            )
            bundle = load_model_bundle(
                paths or ModelPaths(), profiles={ATTEMPT06_T2_POLICY_ID}
            )
            child_seed = (
                ATTEMPT06_CHILD_SEED_START
                + ATTEMPT06_SEED_STRIDE * root.root_index
            )
            t2_policies = {
                seat: build_policy(
                    ATTEMPT06_T2_POLICY_ID,
                    bundle,
                    seed=child_seed + (0 if seat == "first" else 1),
                    seat=seat,
                    opening_lookahead_samples=0,
                )
                for seat in ("first", "second")
            }
            _require_concrete_stage9f_p2_policies(t2_policies)
            root_run_id = (
                f"{run_id}:root={root.root_index}:seed={root.hand_seed}:"
                f"obs={root.observation.fingerprint()}"
            )
            teacher_config = Attempt06TeacherConfig(
                frozen_model_sha256=expected_model_hash,
                candidate_seed=ATTEMPT06_CANDIDATE_SEED_START
                + ATTEMPT06_SEED_STRIDE * root.root_index,
                evaluation_seed=ATTEMPT06_EVALUATION_SEED_START
                + ATTEMPT06_SEED_STRIDE * root.root_index,
                child_policy_seed=child_seed,
                run_id=root_run_id,
                batch_child_selectors=batch_child_selectors,
            )
            result = evaluate_attempt06_t1_second(
                root.observation,
                baseline_action_key=root.baseline_action_key,
                ranker=ranker,
                t2_policies=t2_policies,
                config=teacher_config,
            )
        finally:
            heartbeat_pump.stop()
        row = {
            "schema": ATTEMPT06_SHARD_ROW_SCHEMA,
            "root_index": root.root_index,
            "hand_seed": root.hand_seed,
            "root_profile": root.root_profile,
            "policy_observation": root.observation.to_dict(),
            "baseline_action_key": root.baseline_action_key,
            "provenance": {
                **dict(root.provenance),
                "input_sha256": fixed_contract["input_sha256"],
                "config_sha256": config_sha256,
                "model_sha256": expected_model_hash,
                "source_model_manifest_sha256": source_model_hash,
                "source_native_manifest_sha256": source_native_hash,
                "current_profile_resolved": False,
            },
            "teacher": result,
        }
        _validate_root_provenance(
            row["provenance"],
            root_profile=root.root_profile,
            root_index=root.root_index,
            teacher_output=True,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with partial_path.open("wb") as handle:
            encoded = (
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        common = {
            "config_sha256": config_sha256,
            "input_sha256": fixed_contract["input_sha256"],
            "model_sha256": expected_model_hash,
            "source_model_manifest_sha256": source_model_hash,
            "source_native_manifest_sha256": source_native_hash,
            "completed_roots": 1,
            "target_roots": 1,
            "root_index": root.root_index,
            "partial_sha256": _sha256_file(partial_path),
            "updated_unix_seconds": time.time(),
        }
        _atomic_json(
            checkpoint_path,
            {"schema": ATTEMPT06_CHECKPOINT_SCHEMA, **common},
        )
    try:
        os.link(partial_path, output_path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Attempt06 shard concurrently completed: {output_path}"
        ) from exc
    partial_path.unlink()
    summary = {
        "schema": ATTEMPT06_SHARD_SUMMARY_SCHEMA,
        "status": "complete",
        "root_index": root.root_index,
        "completed_roots": 1,
        "target_roots": 1,
        "config_sha256": config_sha256,
        "input_sha256": fixed_contract["input_sha256"],
        "model_sha256": expected_model_hash,
        "source_model_manifest_sha256": source_model_hash,
        "source_native_manifest_sha256": source_native_hash,
        "output_sha256": _sha256_file(output_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(
        heartbeat_path,
        {
            **summary,
            "schema": ATTEMPT06_HEARTBEAT_SCHEMA,
            "summary_schema": ATTEMPT06_SHARD_SUMMARY_SCHEMA,
        },
    )
    return summary


def run_attempt06_shard(
    *,
    input_roots: str | Path,
    output: str | Path,
    checkpoint: str | Path,
    heartbeat: str | Path,
    model: str | Path,
    model_sha256: str,
    source_model_manifest_sha256: str,
    source_native_manifest_sha256: str,
    run_id: str,
    batch_child_selectors: bool = False,
    native_batch_threads: int = 4,
    paths: ModelPaths | None = None,
) -> dict[str, Any]:
    """Evaluate one pre-built root under an exclusive resumable shard lock."""

    input_path = Path(input_roots)
    output_path = Path(output)
    partial_path = output_path.with_name(output_path.name + ".partial")
    checkpoint_path = Path(checkpoint)
    heartbeat_path = Path(heartbeat)
    model_path = Path(model)
    lock_path = output_path.with_name(output_path.name + ".lock")
    # This check must precede lock-file creation: an aliased checkpoint or
    # input must not be touched even by the one-byte process lock.
    _require_distinct_paths(
        {
            "input_roots": input_path,
            "output": output_path,
            "partial": partial_path,
            "checkpoint": checkpoint_path,
            "heartbeat": heartbeat_path,
            "model": model_path,
            "lock": lock_path,
        }
    )
    with _ShardFileLock(lock_path):
        return _run_attempt06_shard_locked(
            input_roots=input_roots,
            output=output,
            checkpoint=checkpoint,
            heartbeat=heartbeat,
            model=model,
            model_sha256=model_sha256,
            source_model_manifest_sha256=source_model_manifest_sha256,
            source_native_manifest_sha256=source_native_manifest_sha256,
            run_id=run_id,
            batch_child_selectors=batch_child_selectors,
            native_batch_threads=native_batch_threads,
            paths=paths,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser(
        "shard", help="evaluate one already-generated canonical Attempt06 root"
    )
    shard.add_argument("--input-roots", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--checkpoint", type=Path, required=True)
    shard.add_argument("--heartbeat", type=Path, required=True)
    shard.add_argument("--model", type=Path, required=True)
    shard.add_argument("--model-sha256", required=True)
    shard.add_argument("--source-model-manifest-sha256", required=True)
    shard.add_argument("--source-native-manifest-sha256", required=True)
    shard.add_argument("--run-id", required=True)
    shard.add_argument("--batch-child-selectors", action="store_true")
    shard.add_argument("--native-batch-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command != "shard":
        raise AssertionError("unhandled Attempt06 command")
    summary = run_attempt06_shard(
        input_roots=args.input_roots,
        output=args.output,
        checkpoint=args.checkpoint,
        heartbeat=args.heartbeat,
        model=args.model,
        model_sha256=args.model_sha256,
        source_model_manifest_sha256=args.source_model_manifest_sha256,
        source_native_manifest_sha256=args.source_native_manifest_sha256,
        run_id=args.run_id,
        batch_child_selectors=args.batch_child_selectors,
        native_batch_threads=args.native_batch_threads,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ATTEMPT06_AUDIT_ROOTS",
    "ATTEMPT06_CANDIDATE_SAMPLES",
    "ATTEMPT06_EVALUATION_SAMPLES",
    "ATTEMPT06_FROZEN_MODEL_SHA256",
    "ATTEMPT06_FROZEN_MODEL_ID",
    "ATTEMPT06_HAND_SEED_START",
    "ATTEMPT06_NATIVE_BATCH_THREADS",
    "ATTEMPT06_ROOT_REMATERIALIZATION_MODE",
    "ATTEMPT06_PLAN_SHA256",
    "ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256",
    "ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256",
    "ATTEMPT06_ROOT_PROFILES",
    "ATTEMPT06_ROOT_GENERATION_POLICY",
    "ATTEMPT06_ROOT_SCHEMA",
    "ATTEMPT06_ROOT_PROVENANCE_SCHEMA",
    "ATTEMPT06_TEACHER_SCHEMA",
    "ATTEMPT06_TOP_K",
    "Attempt06RankScores",
    "Attempt06Root",
    "Attempt06TeacherConfig",
    "FrozenAttempt06LambdaRanker",
    "attempt06_fixed_contract",
    "attempt06_fixed_contract_sha256",
    "evaluate_attempt06_t1_second",
    "load_attempt06_roots",
    "run_attempt06_shard",
]
