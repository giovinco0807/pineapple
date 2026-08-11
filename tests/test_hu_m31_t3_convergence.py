from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from ofc_regular.action_key import (
    action_key,
    canonical_descending_indices,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_m31_t3_runtime import T3SearchActionValue, T3SearchDecision
from ofc_regular.validate_hu_m31_t3_convergence import (
    CONVERGENCE_SCHEMA,
    ConvergenceConfig,
    SearchBudget,
    _write_json_atomic,
    run_convergence,
)


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class _FakeSolver:
    def __init__(self, budget: SearchBudget, *, unstable: bool = False) -> None:
        self.budget = budget
        self.unstable = unstable
        self.engine_version = "fake-convergence-engine"
        self.library_path = "fake-convergence-engine.dll"
        self.library_sha256 = "a" * 64

    def solve_many(self, observations):
        return [self._decision(observation) for observation in observations]

    def _decision(self, observation):
        actions = generate_turn_actions(
            observation.hero_board,
            observation.dealt_cards,
        )
        selection = [
            float(sum(action_key(action).masks) % 17) for action in actions
        ]
        evaluation = list(selection)
        ranking = canonical_descending_indices(selection, actions)
        if self.unstable and self.budget.label == "teacher_default_4_8_2":
            evaluation[ranking[-1]] = evaluation[ranking[0]] + 4.0
        rank_by_index = {index: rank for rank, index in enumerate(ranking)}
        best_evaluation = max(evaluation)
        values = tuple(
            sorted(
                (
                    T3SearchActionValue(
                        original_index=index,
                        rank=rank_by_index[index],
                        action_key=action_key(action).to_token(),
                        selection_ev=selection[index],
                        evaluation_ev=evaluation[index],
                        evaluation_regret=best_evaluation - evaluation[index],
                        action=action,
                    )
                    for index, action in enumerate(actions)
                ),
                key=lambda row: row.action_key,
            )
        )
        selected_index = ranking[0]
        selected_key = action_key(actions[selected_index]).to_token()
        second = selection[ranking[1]] if len(ranking) > 1 else selection[selected_index]
        fingerprint = observation.fingerprint()
        return T3SearchDecision(
            action=actions[selected_index],
            selected_action_key=selected_key,
            selected_selection_ev=selection[selected_index],
            selected_evaluation_ev=evaluation[selected_index],
            selection_gap=selection[selected_index] - second,
            evaluation_sample_regret=best_evaluation - evaluation[selected_index],
            action_values=values,
            seat=observation.seat,
            value_scope="fake-convergence-values",
            observation_fingerprint=fingerprint,
            legal_action_set_digest=legal_action_set_digest(actions),
            legal_action_order_digest=ordered_action_mapping_digest(actions),
            candidate_belief_digest=_digest([fingerprint, "candidate-belief"]),
            evaluation_belief_digest=_digest([fingerprint, "evaluation-belief"]),
            candidate_rng_digest=_digest([fingerprint, "candidate-rng"]),
            evaluation_rng_digest=_digest([fingerprint, "evaluation-rng"]),
            child_information_set_count=len(actions) * 2,
            candidate_samples=self.budget.candidate_samples,
            evaluation_samples=self.budget.evaluation_samples,
            downstream_t3_samples=self.budget.downstream_t3_samples,
            run_id="hu-m31-step3-local10-v1",
            continuation_seed=2026073100,
            candidate_seed=2026073101,
            evaluation_seed=2026073102,
            use_t4_action_cache=True,
            search_contract_digest=_digest(["fake-contract", self.budget.label]),
            solver_id="fake-convergence-solver",
            engine_version="fake-convergence-engine",
            native_library_sha256="a" * 64,
            native_latency_ms=1.0,
            validation_latency_ms=0.1,
            total_latency_ms=1.1,
            execution_mode="batch_amortized",
            batch_size=2,
            semantic_result_digest=_digest([fingerprint, self.budget.label]),
            result_digest=_digest([fingerprint, self.budget.label, "mapping"]),
        )


def _solvers(config: ConvergenceConfig, *, unstable: bool = False):
    return {
        budget.label: _FakeSolver(budget, unstable=unstable)
        for budget in config.budgets
    }


def test_convergence_pass_authorizes_only_local100_pilot():
    config = ConvergenceConfig()
    report = run_convergence(config=config, solvers=_solvers(config))

    assert report["schema"] == CONVERGENCE_SCHEMA
    assert report["status"] == "pass"
    assert report["all_gates_passed"] is True
    assert all(report["gates"].values())
    assert report["local100_authorization"]["authorized"] is True
    assert report["local100_authorization"]["budget"]["label"] == (
        "teacher_default_4_8_2"
    )
    assert "not_runtime_promotion" in report["local100_authorization"]["scope"]
    assert report["root_generation"]["seats"] == ["first", "second"]
    assert report["root_generation"]["current_profile_read"] is False
    assert report["engine"]["library_sha256"] == "a" * 64
    assert report["spot_vm_started"] is False


def test_reference_evaluation_regret_fails_closed():
    config = ConvergenceConfig()
    report = run_convergence(
        config=config,
        solvers=_solvers(config, unstable=True),
    )

    assert report["status"] == "no_go"
    assert report["all_gates_passed"] is False
    assert report["local100_authorization"]["authorized"] is False
    assert report["gates"]["reference_locked_evaluation_regret_lte_gate"] is False


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"root_count": 3}, "positive even"),
        ({"max_regret_under_reference": -0.1}, "non-negative"),
        (
            {
                "budgets": (
                    SearchBudget("a", 2, 2, 2),
                    SearchBudget("b", 1, 3, 2),
                    SearchBudget("c", 4, 8, 2),
                )
            },
            "non-decreasing",
        ),
    ),
)
def test_config_fails_closed(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        ConvergenceConfig(**kwargs)


def test_write_once_convergence_artifact(tmp_path):
    path = tmp_path / "convergence.json"
    _write_json_atomic(path, {"status": "pass"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "pass"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json_atomic(path, {"status": "changed"})
