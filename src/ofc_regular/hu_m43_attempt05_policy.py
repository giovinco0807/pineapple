"""Opt-in Attempt05 T1-second model -> shortlist-search -> gate wrapper.

The baseline policy is always called first.  Unless the wrapper is explicitly
enabled and its model binding is verified frozen, no model or search call is
made.  The model supplies only a nonbaseline top-4 ranking.  The Rust-backed
shortlist search proposes one exact semantic ``ActionKey`` and the model's
separate external-search gate decides whether that proposal may replace the
baseline.  Every failure and nonfire path returns the original baseline
action, so counterfactual cancellation is exact by ``ActionKey``.

This module is intentionally not imported by ``ai_profiles`` and registers no
profile.  Real-game T4 exact behavior remains owned by the existing baseline
profile; the approximate T4 calls here are hypothetical T1-search rollouts.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    index_actions_by_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .hu_infoset import ActorObservation
from .hu_m4_t1_shortlist_search import (
    M4_T1_SHORTLIST_SEARCH_SCHEMA,
    M4T1ShortlistSearchConfig,
    evaluate_t1_second_shortlist_search,
)
from .hu_turn3_model import hu_policy_sample


HU_M43_ATTEMPT05_POLICY_SCHEMA = "hu_m43_attempt05_t1_search_policy_v1"
HU_M43_ATTEMPT05_DECISION_SCHEMA = "hu_m43_attempt05_t1_search_decision_v1"
HU_M43_ATTEMPT05_TOP_K = 4


class HuM43Attempt05Policy:
    """Fail-closed composition wrapper; never registered by this module."""

    def __init__(
        self,
        baseline_policy: object,
        *,
        model: object | None,
        t2_policies: Mapping[str, object],
        search_config: M4T1ShortlistSearchConfig | None = None,
        enabled: bool = False,
        runtime_binding_verified: bool = False,
        search_backend: Callable[..., Mapping[str, Any]] = (
            evaluate_t1_second_shortlist_search
        ),
        library: Any | None = None,
        decision_log: list[dict[str, Any]] | None = None,
        policy_id: str = "hu-m43-attempt05-opt-in",
    ) -> None:
        chooser = getattr(baseline_policy, "choose_action_observation", None)
        if not callable(chooser):
            raise TypeError("baseline_policy must implement choose_action_observation")
        if set(t2_policies) != {"first", "second"}:
            raise ValueError("t2_policies must contain exactly first and second")
        if not callable(search_backend):
            raise TypeError("search_backend must be callable")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("policy_id must be non-empty")
        self.baseline_policy = baseline_policy
        self.model = model
        self.t2_policies = dict(t2_policies)
        self.search_config = search_config or M4T1ShortlistSearchConfig()
        self.enabled = bool(enabled)
        self.runtime_binding_verified = bool(runtime_binding_verified)
        self.search_backend = search_backend
        self.library = library
        self.decision_log = decision_log
        self.policy_id = policy_id

    def choose_action_observation(
        self,
        observation: ActorObservation,
        *,
        hand_id: str | int | None = None,
        game_id: str | int | None = None,
        decision_seed: int | None = None,
    ) -> Action:
        if not isinstance(observation, ActorObservation):
            raise TypeError("Attempt05 policy requires ActorObservation")
        started_at = time.perf_counter()
        baseline_action = self.baseline_policy.choose_action_observation(
            observation,
            hand_id=hand_id,
            game_id=game_id,
            decision_seed=decision_seed,
        )
        baseline_token = _safe_action_token(baseline_action)
        final_action = baseline_action
        final_token = baseline_token
        actions: list[Action] = []
        baseline_index: int | None = None
        shortlist_indices: tuple[int, ...] = ()
        shortlist_tokens: tuple[str, ...] = ()
        search_winner_index: int | None = None
        search_winner_token: str | None = None
        rank_source: str | None = None
        external_gate_reason: str | None = None
        model_called = False
        search_called = False
        external_gate_called = False
        override_fired = False
        nonfire_reason = ""
        failure_type: str | None = None
        search_candidate_margin: float | None = None
        search_evaluation_margin: float | None = None

        if observation.street != "T1":
            nonfire_reason = "street_not_t1"
        elif observation.seat != "second" or observation.to_act_order != "second":
            nonfire_reason = "first_or_nonsecond_delegated"
        elif not self.enabled:
            nonfire_reason = "attempt05_disabled"
        elif not self.runtime_binding_verified:
            nonfire_reason = "attempt05_model_unfrozen"
        elif self.model is None:
            nonfire_reason = "attempt05_model_unavailable"
        elif getattr(self.model, "runtime_enabled", False) is not True:
            nonfire_reason = "attempt05_model_runtime_disabled"
        elif getattr(self.model, "winner_frozen", False) is not True:
            nonfire_reason = "attempt05_model_unfrozen"
        elif not callable(getattr(self.model, "predict_heads_sample", None)):
            nonfire_reason = "attempt05_ranker_unavailable"
        elif not callable(
            getattr(self.model, "select_external_candidate_index", None)
        ):
            nonfire_reason = "attempt05_external_gate_unavailable"
        else:
            try:
                _validate_model_runtime_contract(self.model)
                actions = generate_turn_actions(
                    observation.hero_board, observation.dealt_cards
                )
                if not actions:
                    raise ValueError("T1 observation has no legal actions")
                baseline_key = action_key(baseline_action)
                baseline_index = index_actions_by_key(actions).get(baseline_key)
                if baseline_index is None:
                    raise LookupError("baseline ActionKey is not legal")
                baseline_token = baseline_key.to_token()

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
                sample["baseline_action_key"] = baseline_token
                model_called = True
                heads = self.model.predict_heads_sample(
                    sample, baseline_index=baseline_index
                )
                rank_scores, rank_source = _model_rank_scores(
                    heads, action_count=len(actions)
                )
                shortlist_indices = _top_nonbaseline_indices(
                    rank_scores,
                    actions,
                    baseline_index=baseline_index,
                    top_k=HU_M43_ATTEMPT05_TOP_K,
                )
                shortlist_tokens = tuple(
                    action_key(actions[index]).to_token()
                    for index in shortlist_indices
                )

                search_called = True
                search_result = self.search_backend(
                    observation,
                    t2_policies=self.t2_policies,
                    baseline_action_key=baseline_token,
                    learned_shortlist_action_keys=shortlist_tokens,
                    config=self.search_config,
                    library=self.library,
                )
                search_winner_index, search_winner_token, winner_row = (
                    _validate_search_result(
                        search_result,
                        observation=observation,
                        actions=actions,
                        baseline_token=baseline_token,
                        shortlist_tokens=shortlist_tokens,
                    )
                )
                if search_winner_index == baseline_index:
                    nonfire_reason = "search_winner_is_baseline"
                else:
                    search_candidate_margin = _finite_search_diagnostic(
                        winner_row.get("candidate_margin_vs_baseline"),
                        name="candidate margin",
                    )
                    search_evaluation_margin = _finite_search_diagnostic(
                        winner_row.get("evaluation_margin_vs_baseline"),
                        name="evaluation margin",
                    )
                    external_gate_called = True
                    gate_decision = self.model.select_external_candidate_index(
                        sample,
                        candidate_index=search_winner_index,
                        baseline_index=baseline_index,
                        candidate_action_key=search_winner_token,
                    )
                    gate_fired, external_gate_reason = _validate_external_gate_decision(
                        gate_decision,
                        baseline_index=baseline_index,
                        candidate_index=search_winner_index,
                    )
                    if gate_fired:
                        final_action = actions[search_winner_index]
                        final_token = search_winner_token
                        override_fired = True
                    else:
                        nonfire_reason = "external_gate_nonfire"
            except Exception as exc:
                final_action = baseline_action
                final_token = baseline_token
                override_fired = False
                nonfire_reason = "attempt05_fail_closed"
                failure_type = type(exc).__name__

        if not override_fired:
            # Do not reconstruct the fallback from an index.  Returning the
            # exact baseline object makes cancellation independent of action
            # ordering and downstream object equality semantics.
            final_action = baseline_action
            final_token = baseline_token
        if not nonfire_reason and not override_fired:
            nonfire_reason = "fallback_to_baseline"
        cancellation_exact = _safe_action_token(final_action) == baseline_token
        if not override_fired and not cancellation_exact:
            raise AssertionError("Attempt05 nonfire did not cancel to baseline ActionKey")

        if self.decision_log is not None:
            self.decision_log.append(
                {
                    "schema": HU_M43_ATTEMPT05_DECISION_SCHEMA,
                    "policy_schema": HU_M43_ATTEMPT05_POLICY_SCHEMA,
                    "policy_id": self.policy_id,
                    "observation_fingerprint": observation.fingerprint(),
                    "street": observation.street,
                    "seat": observation.seat,
                    "to_act_order": observation.to_act_order,
                    "baseline_called_first": True,
                    "enabled": self.enabled,
                    "runtime_binding_verified": self.runtime_binding_verified,
                    "model_called": model_called,
                    "search_called": search_called,
                    "external_gate_called": external_gate_called,
                    "action_key_schema": ACTION_KEY_SCHEMA,
                    "legal_action_count": len(actions),
                    "legal_action_set_digest": (
                        legal_action_set_digest(actions) if actions else None
                    ),
                    "legal_action_order_digest": (
                        ordered_action_mapping_digest(actions) if actions else None
                    ),
                    "baseline_action_index": baseline_index,
                    "baseline_action_key": baseline_token,
                    "model_rank_source": rank_source,
                    "shortlist_action_indices": list(shortlist_indices),
                    "shortlist_action_keys": list(shortlist_tokens),
                    "search_winner_index": search_winner_index,
                    "search_winner_action_key": search_winner_token,
                    "external_gate_reason": external_gate_reason,
                    "search_candidate_margin_vs_baseline": search_candidate_margin,
                    "search_evaluation_margin_vs_baseline": search_evaluation_margin,
                    "final_action_key": final_token,
                    "override_fired": override_fired,
                    "nonfire_reason": nonfire_reason,
                    "failure_type": failure_type,
                    "nonfire_counterfactual_cancellation_exact": (
                        cancellation_exact if not override_fired else None
                    ),
                    "search_value_used_as_direct_gate": False,
                    "runtime_latency_ms": (
                        time.perf_counter() - started_at
                    )
                    * 1000.0,
                }
            )
        return final_action


def _validate_model_runtime_contract(model: object) -> None:
    contract = getattr(model, "runtime_contract", None)
    if not isinstance(contract, Mapping):
        raise TypeError("Attempt05 model runtime contract is unavailable")
    if contract.get("authorization") != "T1-second-only":
        raise ValueError("Attempt05 model authorization changed")
    for field in ("runtime_teacher_ev", "runtime_teacher_lcb", "profile_runtime_feature"):
        if contract.get(field) is not False:
            raise ValueError(f"Attempt05 model {field} must be false")


def _validate_external_gate_decision(
    decision: object,
    *,
    baseline_index: int,
    candidate_index: int,
) -> tuple[bool, str]:
    if getattr(decision, "baseline_index", None) != baseline_index:
        raise ValueError("external gate baseline index changed")
    if getattr(decision, "proposal_index", None) != candidate_index:
        raise ValueError("external gate substituted a different proposal")
    if getattr(decision, "authorized", None) is not True:
        raise ValueError("external gate did not authorize this information set")
    fired = getattr(decision, "override_fired", None)
    if not isinstance(fired, bool):
        raise TypeError("external gate override flag must be a bool")
    selected_index = getattr(decision, "selected_index", None)
    expected = candidate_index if fired else baseline_index
    if selected_index != expected:
        raise ValueError("external gate selected index is inconsistent")
    reason = getattr(decision, "reason", None)
    if not isinstance(reason, str) or not reason:
        raise ValueError("external gate reason is unavailable")
    return fired, reason


def _finite_search_diagnostic(value: object, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"shortlist search {name} is non-finite")
    return result


def _model_rank_scores(
    heads: object, *, action_count: int
) -> tuple[np.ndarray, str]:
    for name in ("rank_score", "base_delta"):
        raw = getattr(heads, name, None)
        if raw is None:
            continue
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (action_count,):
            raise ValueError(f"model {name} shape disagrees with legal actions")
        if not np.isfinite(values).all():
            raise ValueError(f"model {name} contains non-finite values")
        return values, name
    raise TypeError("Attempt05 model heads lack rank_score/base_delta")


def _top_nonbaseline_indices(
    scores: np.ndarray,
    actions: Sequence[Action],
    *,
    baseline_index: int,
    top_k: int,
) -> tuple[int, ...]:
    candidates = [index for index in range(len(actions)) if index != baseline_index]
    if len(candidates) < top_k:
        raise ValueError("fewer than four nonbaseline legal actions")
    candidates.sort(
        key=lambda index: (
            -float(scores[index]),
            action_key(actions[index]).sort_key(),
        )
    )
    return tuple(candidates[:top_k])


def _validate_search_result(
    result: Mapping[str, Any],
    *,
    observation: ActorObservation,
    actions: Sequence[Action],
    baseline_token: str,
    shortlist_tokens: Sequence[str],
) -> tuple[int, str, Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        raise TypeError("shortlist search result must be a mapping")
    if result.get("schema") != M4_T1_SHORTLIST_SEARCH_SCHEMA:
        raise ValueError("shortlist search schema mismatch")
    if result.get("observation_fingerprint") != observation.fingerprint():
        raise ValueError("shortlist search observation fingerprint mismatch")
    if result.get("legal_action_set_digest") != legal_action_set_digest(actions):
        raise ValueError("shortlist search legal action set mismatch")
    if result.get("legal_action_order_digest") != ordered_action_mapping_digest(
        actions
    ):
        raise ValueError("shortlist search legal action order mismatch")
    if result.get("baseline_action_key") != baseline_token:
        raise ValueError("shortlist search baseline ActionKey mismatch")
    expected_fixed = [*shortlist_tokens, baseline_token]
    if result.get("fixed_search_action_keys") != expected_fixed:
        raise ValueError("shortlist search fixed ActionKeys changed")
    required_true = (
        "common_random_futures_across_fixed_actions",
        "candidate_evaluation_rng_disjoint",
        "fixed_candidates_before_sampling",
        "proposal_locked_before_evaluation",
    )
    if any(result.get(field) is not True for field in required_true):
        raise ValueError("shortlist search CRN/lock contract failed")
    if result.get("runtime_gate_applied") is not False:
        raise ValueError("shortlist search applied a direct runtime gate")
    if result.get("opponent_private_discard_input_used") is not False:
        raise ValueError("shortlist search used opponent private discard input")
    token = result.get("proposed_action_key")
    if not isinstance(token, str) or token not in expected_fixed:
        raise ValueError("shortlist search proposed an unbound ActionKey")
    by_key = {action_key(action).to_token(): index for index, action in enumerate(actions)}
    index = by_key[token]
    if result.get("proposed_original_index") != index:
        raise ValueError("shortlist search proposed index/ActionKey mismatch")
    rows = result.get("actions")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("shortlist search action rows are missing")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("action_key") == token]
    if len(matches) != 1 or matches[0].get("original_index") != index:
        raise ValueError("shortlist search winner row mapping mismatch")
    return index, token, matches[0]


def _safe_action_token(action: object) -> str | None:
    try:
        return action_key(action).to_token()  # type: ignore[arg-type]
    except Exception:
        return None


__all__ = [
    "HU_M43_ATTEMPT05_DECISION_SCHEMA",
    "HU_M43_ATTEMPT05_POLICY_SCHEMA",
    "HU_M43_ATTEMPT05_TOP_K",
    "HuM43Attempt05Policy",
]
