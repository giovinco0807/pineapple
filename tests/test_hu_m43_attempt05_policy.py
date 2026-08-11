from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ofc_regular.action_key import (
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt05_policy import HuM43Attempt05Policy
from ofc_regular.hu_m4_t1_shortlist_search import (
    M4_T1_SHORTLIST_SEARCH_SCHEMA,
    M4T1ShortlistSearchConfig,
)
from ofc_regular.state import Board


def _observation(*, seat: str = "second") -> ActorObservation:
    opponent = (
        Board.from_rows(
            top=("2h",),
            middle=("3h", "4h", "5h"),
            bottom=("6h", "7h", "8h"),
        )
        if seat == "second"
        else Board.from_rows(
            top=("2h",), middle=("3h", "4h"), bottom=("6h", "7h")
        )
    )
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("9h",), middle=("Th", "Jh"), bottom=("Qh", "Kh")
        ),
        opponent_public_board=opponent,
        dealt_cards=("Ah", "2d", "3d"),
        hero_private_discards=(),
        seat=seat,
        street="T1",
        to_act_order=seat,
    )


class _Baseline:
    def __init__(self, events: list[str], *, index: int = -1) -> None:
        self.events = events
        self.index = index
        self.returned = None

    def choose_action_observation(self, observation, **_kwargs):
        self.events.append("baseline")
        self.returned = generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )[self.index]
        return self.returned


@dataclass
class _Heads:
    base_delta: np.ndarray


@dataclass
class _GateDecision:
    baseline_index: int
    proposal_index: int
    selected_index: int
    override_fired: bool
    authorized: bool
    reason: str


class _Model:
    runtime_enabled = True
    winner_frozen = True
    runtime_contract = {
        "authorization": "T1-second-only",
        "runtime_teacher_ev": False,
        "runtime_teacher_lcb": False,
        "profile_runtime_feature": False,
    }

    def __init__(self, events: list[str], *, fire: bool = True) -> None:
        self.events = events
        self.fire = fire
        self.head_calls = []
        self.gate_calls = []

    def predict_heads_sample(self, sample, *, baseline_index):
        self.events.append("model_rank")
        actions = sample["actions"]
        scores = np.zeros(len(actions), dtype=np.float64)
        # Multiple equal maxima exercise the exact ActionKey tie break.
        scores[:] = np.linspace(-2.0, 2.0, len(actions))
        scores[baseline_index] = 1000.0  # Baseline must still be excluded.
        scores[1:6] = 50.0
        self.head_calls.append((sample, baseline_index, scores.copy()))
        return _Heads(scores)

    def select_external_candidate_index(
        self,
        sample,
        *,
        candidate_index,
        baseline_index,
        candidate_action_key,
    ):
        self.events.append("external_gate")
        self.gate_calls.append(
            (sample, candidate_index, baseline_index, candidate_action_key)
        )
        return _GateDecision(
            baseline_index=baseline_index,
            proposal_index=candidate_index,
            selected_index=candidate_index if self.fire else baseline_index,
            override_fired=self.fire,
            authorized=True,
            reason="external_candidate_override" if self.fire else "baseline_fallback",
        )


class _Search:
    def __init__(
        self,
        events: list[str],
        *,
        winner: str = "second_shortlist",
        poison: str | None = None,
        raises: bool = False,
    ) -> None:
        self.events = events
        self.winner = winner
        self.poison = poison
        self.raises = raises
        self.calls = []

    def __call__(
        self,
        observation,
        *,
        t2_policies,
        baseline_action_key,
        learned_shortlist_action_keys,
        config,
        library,
    ):
        self.events.append("search")
        self.calls.append(
            {
                "observation": observation,
                "baseline": baseline_action_key,
                "shortlist": tuple(learned_shortlist_action_keys),
                "config": config,
                "library": library,
                "t2_policies": t2_policies,
            }
        )
        if self.raises:
            raise RuntimeError("search failed")
        actions = generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )
        tokens = [action_key(value).to_token() for value in actions]
        winner = (
            baseline_action_key
            if self.winner == "baseline"
            else learned_shortlist_action_keys[1]
        )
        winner_index = tokens.index(winner)
        fixed = [*learned_shortlist_action_keys, baseline_action_key]
        result = {
            "schema": M4_T1_SHORTLIST_SEARCH_SCHEMA,
            "observation_fingerprint": observation.fingerprint(),
            "legal_action_set_digest": legal_action_set_digest(actions),
            "legal_action_order_digest": ordered_action_mapping_digest(actions),
            "baseline_action_key": baseline_action_key,
            "fixed_search_action_keys": fixed,
            "common_random_futures_across_fixed_actions": True,
            "candidate_evaluation_rng_disjoint": True,
            "fixed_candidates_before_sampling": True,
            "proposal_locked_before_evaluation": True,
            "runtime_gate_applied": False,
            "opponent_private_discard_input_used": False,
            "proposed_action_key": winner,
            "proposed_original_index": winner_index,
            "candidate_margin_vs_runner_up": 1.25,
            "actions": [
                {
                    "action_key": token,
                    "original_index": tokens.index(token),
                    "candidate_mean": 3.0,
                    "candidate_standard_error": 0.2,
                    "candidate_margin_vs_baseline": (
                        0.0 if token == baseline_action_key else 1.5
                    ),
                    "candidate_margin_standard_error": 0.3,
                    "evaluation_mean": 2.5,
                    "evaluation_standard_error": 0.25,
                    "evaluation_margin_vs_baseline": (
                        0.0 if token == baseline_action_key else 1.0
                    ),
                    "evaluation_margin_standard_error": 0.35,
                }
                for token in fixed
            ],
        }
        if self.poison == "winner_key":
            result["proposed_action_key"] = "not-legal"
        elif self.poison == "order_digest":
            result["legal_action_order_digest"] = "0" * 64
        elif self.poison == "direct_gate":
            result["runtime_gate_applied"] = True
        return result


def _policy(
    *,
    enabled: bool = True,
    frozen: bool = True,
    fire: bool = True,
    search: _Search | None = None,
    model=None,
):
    events: list[str] = []
    baseline = _Baseline(events)
    actual_model = model if model is not None else _Model(events, fire=fire)
    actual_search = search if search is not None else _Search(events)
    log: list[dict] = []
    policy = HuM43Attempt05Policy(
        baseline,
        model=actual_model,
        t2_policies={"first": object(), "second": object()},
        search_config=M4T1ShortlistSearchConfig(
            candidate_samples=4,
            evaluation_samples=4,
            candidate_seed=8101,
            evaluation_seed=8102,
            child_policy_seed=8103,
            run_id="attempt05-policy-test",
        ),
        enabled=enabled,
        runtime_binding_verified=frozen,
        search_backend=actual_search,
        decision_log=log,
    )
    return policy, baseline, actual_model, actual_search, events, log


@pytest.mark.parametrize(
    ("enabled", "frozen", "reason"),
    (
        (False, True, "attempt05_disabled"),
        (True, False, "attempt05_model_unfrozen"),
    ),
)
def test_disabled_or_unfrozen_skips_model_and_search_and_returns_exact_baseline(
    enabled, frozen, reason
):
    policy, baseline, model, search, events, log = _policy(
        enabled=enabled, frozen=frozen
    )
    selected = policy.choose_action_observation(_observation())

    assert selected is baseline.returned
    assert events == ["baseline"]
    assert model.head_calls == []
    assert search.calls == []
    assert log[0]["search_called"] is False
    assert log[0]["nonfire_reason"] == reason
    assert log[0]["nonfire_counterfactual_cancellation_exact"] is True


def test_first_seat_skips_model_and_search_after_baseline_first():
    policy, baseline, model, search, events, log = _policy()
    selected = policy.choose_action_observation(_observation(seat="first"))

    assert selected is baseline.returned
    assert events == ["baseline"]
    assert model.head_calls == []
    assert search.calls == []
    assert log[0]["nonfire_reason"] == "first_or_nonsecond_delegated"


def test_active_path_is_baseline_then_model_top4_then_search_then_external_gate():
    policy, baseline, model, search, events, log = _policy(fire=True)
    observation = _observation()
    selected = policy.choose_action_observation(observation)

    assert events == ["baseline", "model_rank", "search", "external_gate"]
    call = search.calls[0]
    assert len(call["shortlist"]) == 4
    assert call["baseline"] not in call["shortlist"]
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    baseline_token = action_key(baseline.returned).to_token()
    sample, baseline_index, scores = model.head_calls[0]
    assert sample["baseline_action_row_index"] == baseline_index
    assert sample["baseline_action_key"] == baseline_token
    expected = sorted(
        (index for index in range(len(actions)) if index != baseline_index),
        key=lambda index: (
            -float(scores[index]),
            action_key(actions[index]).sort_key(),
        ),
    )[:4]
    assert call["shortlist"] == tuple(
        action_key(actions[index]).to_token() for index in expected
    )
    winner = call["shortlist"][1]
    assert action_key(selected).to_token() == winner
    assert action_key(selected).to_token() != baseline_token
    assert len(model.gate_calls) == 1
    _gate_sample, candidate_index, gate_baseline_index, candidate_token = (
        model.gate_calls[0]
    )
    assert candidate_index == actions.index(selected)
    assert gate_baseline_index == baseline_index
    assert candidate_token == winner
    assert log[0]["override_fired"] is True
    assert log[0]["external_gate_reason"] == "external_candidate_override"
    assert log[0]["search_value_used_as_direct_gate"] is False


def test_external_gate_nonfire_returns_the_original_baseline_object():
    policy, baseline, model, search, events, log = _policy(fire=False)
    selected = policy.choose_action_observation(_observation())

    assert selected is baseline.returned
    assert events == ["baseline", "model_rank", "search", "external_gate"]
    assert log[0]["nonfire_reason"] == "external_gate_nonfire"
    assert log[0]["external_gate_reason"] == "baseline_fallback"
    assert log[0]["nonfire_counterfactual_cancellation_exact"] is True


def test_search_winner_baseline_skips_external_gate_and_cancels_exactly():
    events: list[str] = []
    search = _Search(events, winner="baseline")
    policy, baseline, model, _unused, policy_events, log = _policy(search=search)
    # _policy owns the event list used by its model/baseline; share it here.
    search.events = policy_events
    selected = policy.choose_action_observation(_observation())

    assert selected is baseline.returned
    assert policy_events == ["baseline", "model_rank", "search"]
    assert model.gate_calls == []
    assert log[0]["nonfire_reason"] == "search_winner_is_baseline"


@pytest.mark.parametrize("poison", ("winner_key", "order_digest", "direct_gate"))
def test_poisoned_search_contract_fails_closed_to_exact_baseline(poison):
    external_events: list[str] = []
    search = _Search(external_events, poison=poison)
    policy, baseline, model, _unused, events, log = _policy(search=search)
    search.events = events
    selected = policy.choose_action_observation(_observation())

    assert selected is baseline.returned
    assert events == ["baseline", "model_rank", "search"]
    assert model.gate_calls == []
    assert log[0]["nonfire_reason"] == "attempt05_fail_closed"
    assert log[0]["nonfire_counterfactual_cancellation_exact"] is True


def test_search_exception_and_external_gate_exception_both_fail_closed():
    search = _Search([], raises=True)
    policy, baseline, model, _unused, events, log = _policy(search=search)
    search.events = events
    assert policy.choose_action_observation(_observation()) is baseline.returned
    assert log[0]["failure_type"] == "RuntimeError"

    class ExplodingGate(_Model):
        def select_external_candidate_index(self, *_args, **_kwargs):
            self.events.append("external_gate")
            raise RuntimeError("gate failed")

    model_events: list[str] = []
    exploding = ExplodingGate(model_events)
    policy2, baseline2, _model2, search2, events2, log2 = _policy(model=exploding)
    exploding.events = events2
    search2.events = events2
    assert policy2.choose_action_observation(_observation()) is baseline2.returned
    assert events2 == ["baseline", "model_rank", "search", "external_gate"]
    assert log2[0]["failure_type"] == "RuntimeError"


@pytest.mark.parametrize("kind", ("disabled", "unfrozen", "missing"))
def test_unavailable_external_gate_skips_search(kind):
    events: list[str] = []
    if kind == "disabled":
        model = _Model(events)
        model.runtime_enabled = False
    elif kind == "unfrozen":
        model = _Model(events)
        model.winner_frozen = False
    else:
        model = _Model(events)
        model.select_external_candidate_index = None
    policy, baseline, _model, search, policy_events, log = _policy(model=model)
    if hasattr(model, "events"):
        model.events = policy_events
    selected = policy.choose_action_observation(_observation())

    assert selected is baseline.returned
    assert search.calls == []
    assert log[0]["search_called"] is False


def test_external_gate_cannot_substitute_a_different_action():
    class SubstitutingGate(_Model):
        def select_external_candidate_index(
            self,
            sample,
            *,
            candidate_index,
            baseline_index,
            candidate_action_key,
        ):
            self.events.append("external_gate")
            return _GateDecision(
                baseline_index=baseline_index,
                proposal_index=candidate_index + 1,
                selected_index=candidate_index + 1,
                override_fired=True,
                authorized=True,
                reason="substitution",
            )

    model = SubstitutingGate([])
    policy, baseline, _model, search, events, log = _policy(model=model)
    model.events = events
    search.events = events

    assert policy.choose_action_observation(_observation()) is baseline.returned
    assert events == ["baseline", "model_rank", "search", "external_gate"]
    assert log[0]["failure_type"] == "ValueError"
    assert log[0]["nonfire_counterfactual_cancellation_exact"] is True


def test_public_api_rejects_raw_hidden_discard_argument():
    policy, _baseline, _model, _search, _events, _log = _policy()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        policy.choose_action_observation(
            _observation(), opponent_private_discards=("As",)  # type: ignore[call-arg]
        )
