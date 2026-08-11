"""Profile-blind Attempt13 T1-second search teacher.

Attempt13 deliberately reuses the audited Attempt12 rollout machinery for the
all-legal R128 proposal, K=min(8,n) shortlist, common-random-future phases, and
locked E512 diagnostic.  This module owns the changed decision boundary:

* V256 keeps a candidate when its paired mean is positive and its normalized
  p05/p01 loss risk is at most 1.05;
* X1024 and C1024 receive every V survivor in the same frozen order and are
  concatenated into one P2048 vector;
* P2048 additionally gates on the 0.1-percent quantile and empirical ES1%; and
* eligible actions minimize the maximum of all four normalized tail risks.

Raw minima remain visible diagnostics only.  They never filter, rank, select,
or gate an action.  No opponent-private discard or particle payload is kept.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from . import hu_m43_attempt12_teacher as _attempt12
from .action_key import action_key
from .action_space import Action
from .hu_infoset import ActorObservation
from .hu_m43_attempt06_teacher import _require_sha256
from .hu_m4_t1_teacher import M4T1TeacherConfig
from .hu_m4_teacher_contract import require_t1_second_root


ATTEMPT13_TEACHER_SCHEMA = (
    "hu_m43_attempt13_t1_second_alllegal_r128_kmin8_v256_risk105_"
    "x1024_c1024_pooled2048_q001_es01_e512_v1"
)
ATTEMPT13_SOLVER_ID = (
    "alllegal_r128_head4_lambda_riskfill_kmin8_v256_risk105_"
    "parallel_x1024_c1024_pooled2048_four_tail_risk_lock_e512_m3_mc1_v1"
)
ATTEMPT13_VALIDATION_SCHEMA = "hu_m43_attempt13_teacher_validation_v1"

ATTEMPT13_FROZEN_MODEL_SHA256 = _attempt12.ATTEMPT12_FROZEN_MODEL_SHA256
ATTEMPT13_FROZEN_MODEL_ID = _attempt12.ATTEMPT12_FROZEN_MODEL_ID
ATTEMPT13_T2_POLICY_ID = _attempt12.ATTEMPT12_T2_POLICY_ID

ATTEMPT13_ALL_LEGAL_CANDIDATES = True
ATTEMPT13_RERANK_SAMPLES = 128
ATTEMPT13_RERANK_HEAD_MAX = 4
ATTEMPT13_SHORTLIST_MAX = 8
ATTEMPT13_VETO_SAMPLES = 256
ATTEMPT13_STRESS_SAMPLES = 1024
ATTEMPT13_CONFIRMATION_SAMPLES = 1024
ATTEMPT13_POOLED_SAMPLES = 2048
ATTEMPT13_EVALUATION_SAMPLES = 512

ATTEMPT13_MEAN_MIN = 0.0
ATTEMPT13_P05_LOSS_SCALE = 25.0
ATTEMPT13_P01_LOSS_SCALE = 40.0
ATTEMPT13_VETO_RISK_CAP = 1.05
ATTEMPT13_POOLED_RISK_CAP = 1.0
ATTEMPT13_Q001_QUANTILE = 0.001
ATTEMPT13_Q001_MIN = -50.0
ATTEMPT13_ES01_FRACTION = 0.01
ATTEMPT13_ES01_MIN = -40.0
ATTEMPT13_RAW_MIN_DIAGNOSTIC_REFERENCE = -50.0

ATTEMPT13_RNG_DOMAINS = _attempt12.ATTEMPT12_RNG_DOMAINS

Attempt13RankScores = _attempt12.Attempt12RankScores
FrozenAttempt13LambdaRanker = _attempt12.FrozenAttempt12LambdaRanker


class Attempt13Ranker(Protocol):
    artifact_sha256: str
    model_id: str

    def score_actions(
        self,
        observation: ActorObservation,
        actions: Sequence[Action],
        *,
        baseline_index: int,
    ) -> Attempt13RankScores: ...


@dataclass(frozen=True)
class Attempt13TeacherConfig:
    """Hard-locked Attempt13 configuration for one canonical root."""

    frozen_model_sha256: str
    hand_seed: int
    rerank_seed: int
    veto_seed: int
    stress_seed: int
    confirmation_seed: int
    evaluation_seed: int
    child_policy_seed: int
    run_id: str
    all_legal_candidates: bool = ATTEMPT13_ALL_LEGAL_CANDIDATES
    rerank_samples: int = ATTEMPT13_RERANK_SAMPLES
    rerank_head_max: int = ATTEMPT13_RERANK_HEAD_MAX
    shortlist_max: int = ATTEMPT13_SHORTLIST_MAX
    veto_samples: int = ATTEMPT13_VETO_SAMPLES
    stress_samples: int = ATTEMPT13_STRESS_SAMPLES
    confirmation_samples: int = ATTEMPT13_CONFIRMATION_SAMPLES
    evaluation_samples: int = ATTEMPT13_EVALUATION_SAMPLES
    veto_risk_cap: float = ATTEMPT13_VETO_RISK_CAP
    pooled_risk_cap: float = ATTEMPT13_POOLED_RISK_CAP
    p05_loss_scale: float = ATTEMPT13_P05_LOSS_SCALE
    p01_loss_scale: float = ATTEMPT13_P01_LOSS_SCALE
    q001_quantile: float = ATTEMPT13_Q001_QUANTILE
    q001_min: float = ATTEMPT13_Q001_MIN
    es01_fraction: float = ATTEMPT13_ES01_FRACTION
    es01_min: float = ATTEMPT13_ES01_MIN
    raw_min_diagnostic_reference: float = ATTEMPT13_RAW_MIN_DIAGNOSTIC_REFERENCE
    t2_policy_id: str = ATTEMPT13_T2_POLICY_ID
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
        if self.frozen_model_sha256 != ATTEMPT13_FROZEN_MODEL_SHA256:
            raise ValueError("Attempt13 requires the frozen Lambda raw-risk artifact")
        fixed_ints = {
            "rerank_samples": ATTEMPT13_RERANK_SAMPLES,
            "rerank_head_max": ATTEMPT13_RERANK_HEAD_MAX,
            "shortlist_max": ATTEMPT13_SHORTLIST_MAX,
            "veto_samples": ATTEMPT13_VETO_SAMPLES,
            "stress_samples": ATTEMPT13_STRESS_SAMPLES,
            "confirmation_samples": ATTEMPT13_CONFIRMATION_SAMPLES,
            "evaluation_samples": ATTEMPT13_EVALUATION_SAMPLES,
            "t3_candidate_samples": 1,
            "t3_evaluation_samples": 1,
            "t3_downstream_samples": 1,
            "t4_candidate_samples": 1,
            "t4_evaluation_samples": 1,
        }
        for name, expected in fixed_ints.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                raise ValueError(f"Attempt13 {name} is fixed at {expected}")
        fixed_floats = {
            "veto_risk_cap": ATTEMPT13_VETO_RISK_CAP,
            "pooled_risk_cap": ATTEMPT13_POOLED_RISK_CAP,
            "p05_loss_scale": ATTEMPT13_P05_LOSS_SCALE,
            "p01_loss_scale": ATTEMPT13_P01_LOSS_SCALE,
            "q001_quantile": ATTEMPT13_Q001_QUANTILE,
            "q001_min": ATTEMPT13_Q001_MIN,
            "es01_fraction": ATTEMPT13_ES01_FRACTION,
            "es01_min": ATTEMPT13_ES01_MIN,
            "raw_min_diagnostic_reference": ATTEMPT13_RAW_MIN_DIAGNOSTIC_REFERENCE,
        }
        for name, expected in fixed_floats.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != expected
            ):
                raise ValueError(f"Attempt13 {name} is fixed at {expected}")
        if self.all_legal_candidates is not True:
            raise ValueError("Attempt13 requires every unique legal nonbaseline action")
        seed_names = (
            "hand_seed",
            "rerank_seed",
            "veto_seed",
            "stress_seed",
            "confirmation_seed",
            "evaluation_seed",
            "child_policy_seed",
        )
        for name in seed_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Attempt13 {name} must be an integer")
        if len({getattr(self, name) for name in seed_names}) != len(seed_names):
            raise ValueError("Attempt13 hand/R/V/X/C/E/child seeds must all be distinct")
        if self.t2_policy_id != ATTEMPT13_T2_POLICY_ID:
            raise ValueError("Attempt13 T2 policy is fixed at stage9f_p2")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("Attempt13 run_id must not be empty")
        if not isinstance(self.batch_child_selectors, bool):
            raise TypeError("batch_child_selectors must be a bool")

    def m4_config(self) -> M4T1TeacherConfig:
        return M4T1TeacherConfig(
            candidate_samples=self.rerank_samples,
            evaluation_samples=self.veto_samples,
            candidate_seed=self.rerank_seed,
            evaluation_seed=self.veto_seed,
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


class _Attempt12CarrierConfig:
    """Duck-typed config used only to obtain the shared rollout phases.

    The deliberately unreachable pooled mean gate prevents Attempt12 from
    opening E512 before Attempt13 has made its own four-tail-risk decision.
    """

    def __init__(self, config: Attempt13TeacherConfig) -> None:
        for name in (
            "frozen_model_sha256",
            "hand_seed",
            "rerank_seed",
            "veto_seed",
            "stress_seed",
            "confirmation_seed",
            "evaluation_seed",
            "child_policy_seed",
            "run_id",
            "all_legal_candidates",
            "rerank_samples",
            "rerank_head_max",
            "shortlist_max",
            "veto_samples",
            "stress_samples",
            "confirmation_samples",
            "evaluation_samples",
            "t2_policy_id",
            "t3_candidate_samples",
            "t3_evaluation_samples",
            "t3_downstream_samples",
            "t4_candidate_samples",
            "t4_evaluation_samples",
            "batch_child_selectors",
        ):
            setattr(self, name, getattr(config, name))
        self.coarse_min_mean = ATTEMPT13_MEAN_MIN
        self.coarse_min_p05 = -ATTEMPT13_P05_LOSS_SCALE * ATTEMPT13_VETO_RISK_CAP
        self.coarse_min_p01 = -ATTEMPT13_P01_LOSS_SCALE * ATTEMPT13_VETO_RISK_CAP
        self.coarse_min_value = ATTEMPT13_RAW_MIN_DIAGNOSTIC_REFERENCE
        self.strict_min_mean = 1.0e300
        self.strict_min_p05 = -ATTEMPT13_P05_LOSS_SCALE
        self.strict_min_p01 = -ATTEMPT13_P01_LOSS_SCALE
        self.strict_min_value = ATTEMPT13_RAW_MIN_DIAGNOSTIC_REFERENCE
        self._source = config

    def m4_config(self) -> M4T1TeacherConfig:
        return self._source.m4_config()


def _loss_risk(
    summary: Mapping[str, Any], *, cap: float
) -> tuple[float, dict[str, float]]:
    del cap
    components = {
        "loss95_over_25": max(0.0, -float(summary["p05"]))
        / ATTEMPT13_P05_LOSS_SCALE,
        "loss99_over_40": max(0.0, -float(summary["p01"]))
        / ATTEMPT13_P01_LOSS_SCALE,
    }
    return max(components.values()), components


def _tail_metrics(raw_values: Sequence[float]) -> dict[str, Any]:
    raw = np.asarray(tuple(float(value) for value in raw_values), dtype=np.float64)
    if raw.ndim != 1 or raw.size != ATTEMPT13_POOLED_SAMPLES or not np.isfinite(raw).all():
        raise ValueError("Attempt13 P2048 raw paired vector is invalid")
    q001 = float(np.quantile(raw, ATTEMPT13_Q001_QUANTILE, method="linear"))
    q01 = float(np.quantile(raw, ATTEMPT13_ES01_FRACTION, method="linear"))
    tail_count = max(1, math.ceil(raw.size * ATTEMPT13_ES01_FRACTION))
    es01 = float(np.sort(raw, kind="stable")[:tail_count].mean())
    return {
        "quantile_method": "numpy_linear",
        "q001_quantile": ATTEMPT13_Q001_QUANTILE,
        "q001": q001,
        "q01_diagnostic": q01,
        "es01_fraction": ATTEMPT13_ES01_FRACTION,
        "es01_definition": "mean_lowest_ceil_fraction_count",
        "es01_included_count": tail_count,
        "es01": es01,
        "raw_min": float(raw.min()),
        "raw_min_can_filter_rank_select_or_gate": False,
    }


def _four_tail_risk(
    summary: Mapping[str, Any], tail: Mapping[str, Any]
) -> tuple[float, dict[str, float]]:
    _, p_components = _loss_risk(summary, cap=ATTEMPT13_POOLED_RISK_CAP)
    components = {
        **p_components,
        "loss_q001_over_50": max(0.0, -float(tail["q001"]))
        / abs(ATTEMPT13_Q001_MIN),
        "loss_es01_over_40": max(0.0, -float(tail["es01"]))
        / abs(ATTEMPT13_ES01_MIN),
    }
    return max(components.values()), components


def _veto_thresholds() -> dict[str, Any]:
    return {
        "mean_strictly_greater_than": ATTEMPT13_MEAN_MIN,
        "normalized_p95_p99_risk_at_most": ATTEMPT13_VETO_RISK_CAP,
        "loss_scales": {
            "p05": ATTEMPT13_P05_LOSS_SCALE,
            "p01": ATTEMPT13_P01_LOSS_SCALE,
        },
        "raw_min_reference_diagnostic_only": ATTEMPT13_RAW_MIN_DIAGNOSTIC_REFERENCE,
    }


def _pooled_thresholds() -> dict[str, Any]:
    return {
        "mean_strictly_greater_than": ATTEMPT13_MEAN_MIN,
        "normalized_p95_p99_risk_at_most": ATTEMPT13_POOLED_RISK_CAP,
        "loss_scales": {
            "p05": ATTEMPT13_P05_LOSS_SCALE,
            "p01": ATTEMPT13_P01_LOSS_SCALE,
        },
        "q001_quantile": ATTEMPT13_Q001_QUANTILE,
        "q001_at_least": ATTEMPT13_Q001_MIN,
        "es01_fraction": ATTEMPT13_ES01_FRACTION,
        "es01_at_least": ATTEMPT13_ES01_MIN,
        "es01_definition": "mean_lowest_ceil_fraction_count",
        "quantile_method": "numpy_linear",
        "raw_min_reference_diagnostic_only": ATTEMPT13_RAW_MIN_DIAGNOSTIC_REFERENCE,
    }


def _veto_contract(
    veto: Mapping[str, Any],
) -> tuple[list[dict[str, bool]], dict[str, dict[str, Any]], list[int]]:
    checks: list[dict[str, bool]] = []
    risks: dict[str, dict[str, Any]] = {}
    retained: list[int] = []
    for position, row in enumerate(veto["actions"][:-1]):
        score, components = _loss_risk(
            row["paired_delta_vs_baseline"], cap=ATTEMPT13_VETO_RISK_CAP
        )
        row_checks = {
            "mean_gt_0": float(row["paired_delta_vs_baseline"]["mean"])
            > ATTEMPT13_MEAN_MIN,
            "normalized_p95_p99_risk_at_most": score
            <= ATTEMPT13_VETO_RISK_CAP,
        }
        checks.append(row_checks)
        risks[str(position)] = {"score": score, "components": components}
        if all(row_checks.values()):
            retained.append(position)
    return checks, risks, retained


def _pooled_contract(
    pooled: Mapping[str, Any],
) -> tuple[
    list[dict[str, bool]],
    list[dict[str, Any]],
    list[int],
    dict[str, dict[str, Any]],
    int | None,
]:
    checks: list[dict[str, bool]] = []
    tails: list[dict[str, Any]] = []
    eligible: list[int] = []
    risks: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(pooled["actions"][:-1]):
        summary = row["paired_delta_vs_baseline"]
        tail = _tail_metrics(row["raw_paired_deltas_vs_baseline"])
        p_score, _ = _loss_risk(summary, cap=ATTEMPT13_POOLED_RISK_CAP)
        row_checks = {
            "mean_gt_0": float(summary["mean"]) > ATTEMPT13_MEAN_MIN,
            "normalized_p95_p99_risk_at_most": p_score
            <= ATTEMPT13_POOLED_RISK_CAP,
            "q001_at_least": float(tail["q001"]) >= ATTEMPT13_Q001_MIN,
            "es01_at_least": float(tail["es01"]) >= ATTEMPT13_ES01_MIN,
        }
        checks.append(row_checks)
        tails.append({"phase_position": position, "action_key": row["action_key"], **tail})
        if all(row_checks.values()):
            eligible.append(position)
            score, components = _four_tail_risk(summary, tail)
            risks[str(position)] = {"score": score, "components": components}
    selected = (
        min(
            eligible,
            key=lambda position: (
                float(risks[str(position)]["score"]),
                -float(
                    pooled["actions"][position]["paired_delta_vs_baseline"]["mean"]
                ),
                position,
                pooled["actions"][position]["action_key"],
            ),
        )
        if eligible
        else None
    )
    return checks, tails, eligible, risks, selected


def evaluate_attempt13_t1_second(
    observation: ActorObservation,
    *,
    baseline_action_key: str,
    ranker: Attempt13Ranker,
    t2_policies: Mapping[str, object],
    config: Attempt13TeacherConfig,
    library: Any | None = None,
) -> dict[str, Any]:
    """Evaluate the fixed Attempt13 search without activating a policy."""

    require_t1_second_root(observation)
    carrier = _Attempt12CarrierConfig(config)
    payload = _attempt12.evaluate_attempt12_t1_second(
        observation,
        baseline_action_key=baseline_action_key,
        ranker=ranker,
        t2_policies=t2_policies,
        config=carrier,  # type: ignore[arg-type]
        library=library,
    )
    payload["schema"] = ATTEMPT13_TEACHER_SCHEMA
    payload["solver_id"] = ATTEMPT13_SOLVER_ID

    veto = payload["veto"]
    veto_checks, veto_risks, veto_retained = _veto_contract(veto)
    expected_veto_keys = [veto["action_keys"][position] for position in veto_retained]
    if veto["retained_traversal_positions"] != veto_retained:
        raise AssertionError("Attempt13 carrier V256 boundary changed")
    veto["thresholds"] = _veto_thresholds()
    veto["checks_by_traversal_position"] = veto_checks
    veto["normalized_tail_risk_by_traversal_position"] = veto_risks
    veto["retained_action_keys"] = expected_veto_keys
    veto["selection_rule"] = "retain_all_mean_positive_risk105_in_frozen_R_order"

    pooled = payload["pooled"]
    selected_position: int | None = None
    if pooled["opened"]:
        checks, tails, eligible, risks, selected_position = _pooled_contract(pooled)
        pooled["thresholds"] = _pooled_thresholds()
        pooled["checks_by_position"] = checks
        pooled["tail_metrics_by_position"] = tails
        pooled["eligible_positions"] = eligible
        pooled["eligible_action_keys"] = [
            pooled["action_keys"][position] for position in eligible
        ]
        pooled["normalized_tail_risk_by_position"] = risks
        pooled["selected_position"] = selected_position
        pooled["selected_action_key"] = (
            pooled["action_keys"][selected_position]
            if selected_position is not None
            else None
        )
    else:
        pooled["thresholds"] = _pooled_thresholds()
        pooled["tail_metrics_by_position"] = []
    pooled["selection_rule"] = (
        "minimum_max_normalized_p05_p01_q001_es01_risk_then_pooled_mean_desc_"
        "then_frozen_R_order_then_ActionKey_else_baseline"
    )
    pooled["raw_min_can_filter_or_rank"] = False

    baseline_token = payload["baseline_action_key"]
    selected_token = (
        pooled["action_keys"][selected_position]
        if selected_position is not None
        else baseline_token
    )
    fired = selected_token != baseline_token
    fallback_reason = None
    if not veto_retained:
        fallback_reason = "no_v256_candidate_passed"
    elif not fired:
        fallback_reason = "no_pooled_x1024_c1024_candidate_passed"
    payload["decision"] = {
        "final_selected_action_key": selected_token,
        "override_fired": fired,
        "exact_baseline_fallback": not fired,
        "fallback_reason": fallback_reason,
        "candidate_fallback_after_V_or_pooled_allowed": True,
        "frozen_before_evaluation_namespace_open": True,
    }

    legal_actions = tuple(
        _attempt12.generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    legal_by_token = {action_key(action).to_token(): action for action in legal_actions}
    if fired:
        evaluation_actions = (
            legal_by_token[selected_token],
            legal_by_token[baseline_token],
        )
        prior_rng_keys = [
            tuple(values) for values in payload["rng_key_digests"].values()
        ]
        evaluation_result = _attempt12._score_phase(
            observation,
            evaluation_actions,
            phase="evaluation_e512",
            seed=config.evaluation_seed,
            sample_count=config.evaluation_samples,
            config=config,  # type: ignore[arg-type]
            t2_policies=t2_policies,
            prior_rng_keys=prior_rng_keys,
            library=library,
        )
        evaluation_rows = _attempt12._phase_actions(
            evaluation_actions, evaluation_result.scores, baseline_position=1
        )
        means = tuple(row.mean for row in evaluation_result.scores)
        best_mean = max(means)
        best_position = min(
            (position for position, value in enumerate(means) if value == best_mean),
            key=lambda position: action_key(evaluation_actions[position]).sort_key(),
        )
        payload["evaluation"] = {
            **_attempt12._mapping_payload(evaluation_actions),
            "opened": True,
            "sample_count": config.evaluation_samples,
            "configured_sample_count": config.evaluation_samples,
            "common_random_futures": True,
            "scope": "locked_final_nonbaseline_plus_explicit_baseline",
            "locked_final_action_key": selected_token,
            "sample_best_action_key": action_key(
                evaluation_actions[best_position]
            ).to_token(),
            "diagnostics_only": True,
            "can_rerank_or_gate": False,
            "decision_frozen_before_namespace_open": True,
            "retained_action_keys": [selected_token],
            "actions": evaluation_rows,
        }
        payload["belief_digests"]["evaluation_e512"] = evaluation_result.belief_digest
        payload["rng_key_digests"]["evaluation_e512"] = list(
            evaluation_result.rng_keys
        )
        payload["phase_child_information_set_counts"][
            "evaluation_e512"
        ] = evaluation_result.child_information_set_count
    else:
        evaluation = _attempt12._closed_phase(
            reason="not_opened_because_final_output_is_baseline",
            sample_count=config.evaluation_samples,
        )
        evaluation.update(
            {
                "locked_final_action_key": baseline_token,
                "sample_best_action_key": None,
                "diagnostics_only": True,
                "can_rerank_or_gate": False,
                "decision_frozen_before_namespace_open": True,
            }
        )
        payload["evaluation"] = evaluation

    payload["root_selection_lock"] = (
        "all_unique_legal_nonbaseline_before_R128_Kmin8_before_V256_risk105_"
        "then_identical_X1024_C1024_scopes_before_P2048_four_tail_risk_lock_"
        "then_final_before_diagnostic_E512"
    )
    payload["sample_independence"] = (
        "pairwise_disjoint_R128_V256_optional_X1024_optional_C1024_"
        "optional_locked_fire_E512_particle_rng_keys"
    )
    validate_attempt13_teacher_output(
        observation,
        baseline_action_key=baseline_token,
        payload=payload,
        config=config,
    )
    return payload


def _normalize_for_attempt12_carrier(
    payload: Mapping[str, Any], config: Attempt13TeacherConfig
) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    carrier = _Attempt12CarrierConfig(config)
    normalized["schema"] = _attempt12.ATTEMPT12_TEACHER_SCHEMA
    normalized["solver_id"] = _attempt12.ATTEMPT12_SOLVER_ID

    veto = normalized["veto"]
    veto["thresholds"] = {
        "mean_strictly_greater_than": carrier.coarse_min_mean,
        "p05_at_least": carrier.coarse_min_p05,
        "p01_at_least": carrier.coarse_min_p01,
        "raw_min_reference_diagnostic_only": carrier.coarse_min_value,
    }
    veto["checks_by_traversal_position"] = [
        _attempt12._checks(
            row["paired_delta_vs_baseline"],
            min_mean=carrier.coarse_min_mean,
            min_p05=carrier.coarse_min_p05,
            min_p01=carrier.coarse_min_p01,
        )
        for row in veto["actions"][:-1]
    ]

    pooled = normalized["pooled"]
    if pooled["opened"]:
        pooled["thresholds"] = {
            "mean_strictly_greater_than": carrier.strict_min_mean,
            "p05_at_least": carrier.strict_min_p05,
            "p01_at_least": carrier.strict_min_p01,
            "raw_min_reference_diagnostic_only": carrier.strict_min_value,
        }
        pooled["checks_by_position"] = [
            _attempt12._checks(
                row["paired_delta_vs_baseline"],
                min_mean=carrier.strict_min_mean,
                min_p05=carrier.strict_min_p05,
                min_p01=carrier.strict_min_p01,
            )
            for row in pooled["actions"][:-1]
        ]
        pooled["eligible_positions"] = []
        pooled["eligible_action_keys"] = []
        pooled["normalized_tail_risk_by_position"] = {}
        pooled["selected_position"] = None
        pooled["selected_action_key"] = None

    baseline = normalized["baseline_action_key"]
    has_veto_survivor = bool(veto["retained_traversal_positions"])
    normalized["decision"] = {
        "final_selected_action_key": baseline,
        "override_fired": False,
        "exact_baseline_fallback": True,
        "fallback_reason": (
            "no_pooled_x1024_c1024_candidate_passed"
            if has_veto_survivor
            else "no_v256_candidate_passed"
        ),
        "candidate_fallback_after_V_or_pooled_allowed": True,
        "frozen_before_evaluation_namespace_open": True,
    }
    evaluation = _attempt12._closed_phase(
        reason="not_opened_because_final_output_is_baseline",
        sample_count=config.evaluation_samples,
    )
    evaluation.update(
        {
            "locked_final_action_key": baseline,
            "sample_best_action_key": None,
            "diagnostics_only": True,
            "can_rerank_or_gate": False,
            "decision_frozen_before_namespace_open": True,
        }
    )
    normalized["evaluation"] = evaluation
    for field in (
        "belief_digests",
        "rng_key_digests",
        "phase_child_information_set_counts",
    ):
        normalized[field].pop("evaluation_e512", None)
    return normalized


def validate_attempt13_teacher_output(
    observation: ActorObservation,
    *,
    baseline_action_key: str,
    payload: Mapping[str, Any],
    config: Attempt13TeacherConfig,
) -> dict[str, Any]:
    """Recompute Attempt13 decisions and reject any structural drift."""

    require_t1_second_root(observation)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    if any(
        token in encoded
        for token in ("opponent_private_discard", "opponent_hidden", '"particles"')
    ):
        raise ValueError("Attempt13 output contains hidden opponent information")
    if (
        payload.get("schema") != ATTEMPT13_TEACHER_SCHEMA
        or payload.get("solver_id") != ATTEMPT13_SOLVER_ID
        or payload.get("status") != "ok"
        or payload.get("policy_observation") != observation.to_dict()
        or payload.get("observation_fingerprint") != observation.fingerprint()
        or payload.get("seat") != "second"
        or payload.get("street") != "T1"
        or payload.get("to_act_order") != "second"
    ):
        raise ValueError("Attempt13 top-level identity changed")
    carrier_payload = _normalize_for_attempt12_carrier(payload, config)
    try:
        _attempt12.validate_attempt12_teacher_output(
            observation,
            baseline_action_key=baseline_action_key,
            payload=carrier_payload,
            config=_Attempt12CarrierConfig(config),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Attempt13 inherited structural contract failed: {exc}") from exc

    veto = payload["veto"]
    veto_checks, veto_risks, retained = _veto_contract(veto)
    retained_keys = [veto["action_keys"][position] for position in retained]
    if (
        veto.get("thresholds") != _veto_thresholds()
        or veto.get("checks_by_traversal_position") != veto_checks
        or veto.get("normalized_tail_risk_by_traversal_position") != veto_risks
        or veto.get("retained_traversal_positions") != retained
        or veto.get("retained_action_keys") != retained_keys
        or veto.get("raw_min_can_filter_or_rank") is not False
    ):
        raise ValueError("Attempt13 V256 normalized-risk contract changed")

    pooled = payload["pooled"]
    if pooled.get("opened") is not bool(retained):
        raise ValueError("Attempt13 P2048 conditional-open boundary changed")
    selected_position: int | None = None
    if pooled["opened"]:
        checks, tails, eligible, risks, selected_position = _pooled_contract(pooled)
        eligible_keys = [pooled["action_keys"][position] for position in eligible]
        if (
            pooled.get("thresholds") != _pooled_thresholds()
            or pooled.get("checks_by_position") != checks
            or pooled.get("tail_metrics_by_position") != tails
            or pooled.get("eligible_positions") != eligible
            or pooled.get("eligible_action_keys") != eligible_keys
            or pooled.get("normalized_tail_risk_by_position") != risks
            or pooled.get("selected_position") != selected_position
            or pooled.get("selected_action_key")
            != (
                pooled["action_keys"][selected_position]
                if selected_position is not None
                else None
            )
            or pooled.get("raw_min_can_filter_or_rank") is not False
        ):
            raise ValueError("Attempt13 P2048 four-tail-risk contract changed")
    elif (
        pooled.get("thresholds") != _pooled_thresholds()
        or pooled.get("tail_metrics_by_position") != []
        or pooled.get("eligible_positions") != []
        or pooled.get("eligible_action_keys") != []
        or pooled.get("selected_position") is not None
        or pooled.get("selected_action_key") is not None
    ):
        raise ValueError("Attempt13 closed P2048 contract changed")

    selected = (
        pooled["action_keys"][selected_position]
        if selected_position is not None
        else baseline_action_key
    )
    fired = selected != baseline_action_key
    expected_reason = None
    if not retained:
        expected_reason = "no_v256_candidate_passed"
    elif not fired:
        expected_reason = "no_pooled_x1024_c1024_candidate_passed"
    decision = payload["decision"]
    if (
        decision.get("final_selected_action_key") != selected
        or decision.get("override_fired") is not fired
        or decision.get("exact_baseline_fallback") is fired
        or decision.get("fallback_reason") != expected_reason
        or decision.get("candidate_fallback_after_V_or_pooled_allowed") is not True
        or decision.get("frozen_before_evaluation_namespace_open") is not True
    ):
        raise ValueError("Attempt13 locked decision changed")

    legal_actions = tuple(
        _attempt12.generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    legal_by_token = {action_key(action).to_token(): action for action in legal_actions}
    evaluation = payload["evaluation"]
    if fired:
        expected_keys = [selected, baseline_action_key]
        _attempt12._phase_raw_contract(
            evaluation,
            sample_count=config.evaluation_samples,
            baseline_token=baseline_action_key,
            legal_by_token=legal_by_token,
            expected_action_keys=expected_keys,
        )
        _attempt12._open_phase_metadata(
            evaluation, sample_count=config.evaluation_samples
        )
        means = [float(row["mean"]) for row in evaluation["actions"]]
        best_mean = max(means)
        best_position = min(
            (position for position, value in enumerate(means) if value == best_mean),
            key=lambda position: action_key(
                legal_by_token[expected_keys[position]]
            ).sort_key(),
        )
        expected_best = expected_keys[best_position]
    else:
        _attempt12._closed_phase_contract(
            evaluation, configured_sample_count=config.evaluation_samples
        )
        expected_best = None
    if (
        evaluation.get("opened") is not fired
        or evaluation.get("locked_final_action_key") != selected
        or evaluation.get("sample_best_action_key") != expected_best
        or evaluation.get("diagnostics_only") is not True
        or evaluation.get("can_rerank_or_gate") is not False
        or evaluation.get("decision_frozen_before_namespace_open") is not True
    ):
        raise ValueError("Attempt13 E512 diagnostic lock changed")

    expected_phases = ["rerank_r128", "veto_v256"]
    if retained:
        expected_phases.extend(["stress_x1024", "confirmation_c1024"])
    if fired:
        expected_phases.append("evaluation_e512")
    def has_expected_phase_order(value: object) -> bool:
        """Accept generator order or the order imposed by canonical JSON.

        Attempt13 rows are validated once in memory and then persisted with
        ``sort_keys=True``.  The latter recursively sorts nested mapping keys,
        so a received canonical row cannot retain the generator insertion
        order.  No other permutation is part of the contract.
        """

        if not isinstance(value, Mapping):
            return False
        actual = list(value)
        return actual == expected_phases or actual == sorted(expected_phases)

    rng = payload.get("rng_key_digests")
    if not has_expected_phase_order(rng):
        raise ValueError("Attempt13 opened RNG phase order changed")
    key_sets = [set(rng[name]) for name in expected_phases]
    for index, left in enumerate(key_sets):
        for right in key_sets[index + 1 :]:
            if not left.isdisjoint(right):
                raise ValueError("Attempt13 phase RNG namespaces overlap")
    for field in ("belief_digests", "phase_child_information_set_counts"):
        if not has_expected_phase_order(payload.get(field)):
            raise ValueError("Attempt13 opened phase provenance changed")
    provenance = payload.get("seed_domain_provenance")
    expected_seed_values = {
        "hand_external": config.hand_seed,
        "rerank_r128": config.rerank_seed,
        "veto_v256": config.veto_seed,
        "stress_x1024": config.stress_seed,
        "confirmation_c1024": config.confirmation_seed,
        "evaluation_e512": config.evaluation_seed,
        "child_policy": config.child_policy_seed,
    }
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("domain_order") != list(ATTEMPT13_RNG_DOMAINS)
        or any(provenance.get(name) != value for name, value in expected_seed_values.items())
        or provenance.get("all_seven_base_seeds_pairwise_distinct") is not True
        or provenance.get("hand_sampled_inside_teacher") is not False
    ):
        raise ValueError("Attempt13 seven-domain seed provenance changed")
    return {
        "schema": ATTEMPT13_VALIDATION_SCHEMA,
        "selected_action_key": selected,
        "override_fired": fired,
        "exact_baseline_fallback": not fired,
        "opened_phases": expected_phases,
    }


__all__ = [
    "ATTEMPT13_ALL_LEGAL_CANDIDATES",
    "ATTEMPT13_CONFIRMATION_SAMPLES",
    "ATTEMPT13_ES01_FRACTION",
    "ATTEMPT13_ES01_MIN",
    "ATTEMPT13_EVALUATION_SAMPLES",
    "ATTEMPT13_FROZEN_MODEL_ID",
    "ATTEMPT13_FROZEN_MODEL_SHA256",
    "ATTEMPT13_P01_LOSS_SCALE",
    "ATTEMPT13_P05_LOSS_SCALE",
    "ATTEMPT13_POOLED_RISK_CAP",
    "ATTEMPT13_POOLED_SAMPLES",
    "ATTEMPT13_Q001_MIN",
    "ATTEMPT13_Q001_QUANTILE",
    "ATTEMPT13_RAW_MIN_DIAGNOSTIC_REFERENCE",
    "ATTEMPT13_RERANK_HEAD_MAX",
    "ATTEMPT13_RERANK_SAMPLES",
    "ATTEMPT13_RNG_DOMAINS",
    "ATTEMPT13_SHORTLIST_MAX",
    "ATTEMPT13_SOLVER_ID",
    "ATTEMPT13_STRESS_SAMPLES",
    "ATTEMPT13_T2_POLICY_ID",
    "ATTEMPT13_TEACHER_SCHEMA",
    "ATTEMPT13_VALIDATION_SCHEMA",
    "ATTEMPT13_VETO_RISK_CAP",
    "ATTEMPT13_VETO_SAMPLES",
    "Attempt13RankScores",
    "Attempt13Ranker",
    "Attempt13TeacherConfig",
    "FrozenAttempt13LambdaRanker",
    "evaluate_attempt13_t1_second",
    "validate_attempt13_teacher_output",
]
