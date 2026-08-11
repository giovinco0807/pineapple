from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m4_t1_policy import HU_M4_T1_DECISION_SCHEMA
from ofc_regular.state import Board
from ofc_regular.hu_m43_attempt13_contract import M43_ATTEMPT13_PLAN_SHA256
from ofc_regular.validate_hu_m43_attempt13_acceptance import (
    ATTEMPT13_OPPONENTS,
    ATTEMPT13_POPULATION_ACCEPTANCE_STATUS_SCHEMA,
    ATTEMPT13_POPULATION_NAMESPACE_BASES,
    ATTEMPT13_POPULATION_PREFLIGHT_SCHEMA,
)
from ofc_regular_promotion import attempt13_population as population
from ofc_regular_promotion import attempt13_profile as profile


_STAGE19_CONTEXT = {
    "runtime_profile": "stage19_p0",
    "runtime_status": "p0_fixed",
    "selective_override_only": True,
    "full_replacement_enabled": False,
    "fallback_policy": "stage18_p1",
    "t1_continuation": "stage18_p1",
    "t2_continuation": "stage9f_p2",
    "t3_continuation": "stage7_m5_r10",
}


class _Baseline:
    def __init__(self, seat: str = "second") -> None:
        self.seat = seat
        self.decision_context = dict(_STAGE19_CONTEXT)
        self.t3_continuation = "stage7_m5_r10"

    def choose_action_observation(self, *_: Any, **__: Any) -> object:
        return object()


class _SemanticBaseline(_Baseline):
    def __init__(self, seat: str = "first") -> None:
        super().__init__(seat)

    def choose_action_observation(self, observation: ActorObservation, **_: Any) -> object:
        return min(
            generate_turn_actions(observation.hero_board, observation.dealt_cards),
            key=lambda action: action_key(action).sort_key(),
        )


def _t1_first_observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=["Ah"], middle=["Kd"], bottom=["2s", "3s", "9s"]
        ),
        opponent_public_board=Board.from_rows(
            top=["Qh"], middle=["Jd"], bottom=["4s", "5s", "6s"]
        ),
        dealt_cards=("8c", "9c", "Tc"),
        hero_private_discards=(),
        seat="first",
        street="T1",
        to_act_order="first",
    )


def _t1_second_observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=["Ah"], middle=["Kd"], bottom=["2s", "3s", "9s"]
        ),
        opponent_public_board=Board.from_rows(
            top=["Qh"], middle=["Jd", "Td"], bottom=["4s", "5s", "6s", "7s"]
        ),
        dealt_cards=("8c", "9c", "Tc"),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )


def _complete_go(model_sha: str) -> dict[str, Any]:
    return {
        "schema": ATTEMPT13_POPULATION_ACCEPTANCE_STATUS_SCHEMA,
        "status": "complete_go",
        "passed_gates": 1,
        "total_gates": 1,
        "gates": [{"name": "all", "passed": True}],
        "profile_id": "stage20_m4_attempt13",
        "promotion_eligible": True,
        "explicit_opt_in_authorized": True,
        "automatic_activation_authorized": False,
        "population_plan_sha256": "1" * 64,
        "search_plan_sha256": M43_ATTEMPT13_PLAN_SHA256,
        "records_sha256": "2" * 64,
        "evaluation_sha256": "3" * 64,
        "merge_manifest_sha256": "4" * 64,
        "model_sha256": model_sha,
        "baseline_profile": "stage19_p0",
        "opponents": list(ATTEMPT13_OPPONENTS),
        "teacher_values_reported_as_realized_match_ev": False,
        "threshold_reselection_performed": False,
        "audit50_rows_used_for_fit": 0,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "full_replacement": False,
    }


def _write_status(tmp_path: Path, status: dict[str, Any]) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "complete_go.json"
    path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_attempt13_complete_go_is_hash_required_and_fail_closed(tmp_path: Path) -> None:
    model_sha = "a" * 64
    path, digest = _write_status(tmp_path, _complete_go(model_sha))
    loaded = profile.load_attempt13_complete_go(
        path,
        expected_sha256=digest,
        expected_model_sha256=model_sha,
    )
    assert loaded["profile_id"] == "stage20_m4_attempt13"
    with pytest.raises(ValueError, match="SHA mismatch"):
        profile.load_attempt13_complete_go(
            path,
            expected_sha256="0" * 64,
            expected_model_sha256=model_sha,
        )
    no_go = _complete_go(model_sha)
    no_go["status"] = "complete_no_go"
    no_go_path, no_go_digest = _write_status(tmp_path / "other", no_go)
    with pytest.raises(ValueError, match="not activation authority"):
        profile.load_attempt13_complete_go(
            no_go_path,
            expected_sha256=no_go_digest,
            expected_model_sha256=model_sha,
        )


def test_attempt13_profile_requires_bound_model_and_never_registers_current(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"attempt13")
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    go_path, go_sha = _write_status(tmp_path, _complete_go(model_sha))
    bound = SimpleNamespace(
        safety_enabled=True,
        winner_frozen=True,
        safety_threshold=0.5,
        minimum_fold_votes=4,
    )
    monkeypatch.setattr(profile, "is_bound_attempt13_distilled_model", lambda value: value is bound)
    monkeypatch.setattr(profile, "load_bound_attempt13_distilled_model", lambda *_, **__: bound)
    attested: list[dict[str, Any]] = []
    monkeypatch.setattr(
        profile,
        "validate_frozen_execution_modules",
        lambda **kwargs: attested.append(kwargs),
    )
    policy = profile.build_stage20_m4_attempt13(
        _Baseline(),
        complete_go_path=go_path,
        expected_complete_go_sha256=go_sha,
        model_path=model_path,
        expected_model_sha256=model_sha,
        runtime_freeze_path=tmp_path / "freeze.json",
        training_manifest_path=tmp_path / "training.json",
        runtime_source_manifest_path=tmp_path / "runtime.json",
        runtime_source_root=tmp_path,
        runtime_dependency_root=tmp_path,
    )
    assert policy.policy_id == "stage20_m4_attempt13"
    assert policy.allowed_seats == ("second",)
    assert policy.action_value_model is bound
    assert policy.safety_model is bound
    assert policy.runtime_binding_verified is True
    assert len(attested) == 1
    assert "ofc_regular_promotion.attempt13_profile" in attested[0]["module_names"]
    with pytest.raises(TypeError, match="Complete-Go capability"):
        profile.Attempt13SelectiveOverridePolicy(
            _Baseline(), bound_model=bound
        )


def test_attempt13_first_seat_is_exact_baseline_delegate_without_model_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Bound:
        safety_enabled = True
        winner_frozen = True
        safety_threshold = 0.5
        minimum_fold_votes = 4

        def predict_sample_with_baseline(self, *_: Any, **__: Any) -> np.ndarray:
            raise AssertionError("first-seat delegate must not call Attempt13")

    bound = Bound()
    monkeypatch.setattr(
        profile, "is_bound_attempt13_distilled_model", lambda value: value is bound
    )
    monkeypatch.setattr(
        profile, "validate_frozen_execution_modules", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        profile, "load_bound_attempt13_distilled_model", lambda *_, **__: bound
    )
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"attempt13")
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    go_path, go_sha = _write_status(tmp_path, _complete_go(model_sha))
    baseline = _SemanticBaseline("first")
    log: list[dict[str, Any]] = []
    policy = profile.build_stage20_m4_attempt13(
        baseline,
        complete_go_path=go_path,
        expected_complete_go_sha256=go_sha,
        model_path=model_path,
        expected_model_sha256=model_sha,
        runtime_freeze_path=tmp_path / "freeze.json",
        training_manifest_path=tmp_path / "training.json",
        runtime_source_manifest_path=tmp_path / "runtime.json",
        runtime_source_root=tmp_path,
        runtime_dependency_root=tmp_path,
        decision_log=log,
    )
    observation = _t1_first_observation()
    expected = baseline.choose_action_observation(observation)
    selected = policy.choose_action_observation(observation, decision_seed=7)
    assert action_key(selected) == action_key(expected)
    assert log[0]["override_fired"] is False
    assert log[0]["nonfire_reason"] == "first_seat_delegated"
    assert log[0]["runtime_binding_verified"] is True
    with pytest.raises(ValueError, match="seat mismatch"):
        policy.choose_action_observation(
            _t1_second_observation(),
            decision_seed=8,
        )
    baseline.decision_context["t2_continuation"] = "stage18_p1"
    with pytest.raises(ValueError, match="fixed baseline chain changed"):
        policy.choose_action_observation(observation, decision_seed=9)
    baseline.decision_context["t2_continuation"] = "stage9f_p2"
    policy.action_value_model = object()
    with pytest.raises(RuntimeError, match="runtime binding drifted"):
        policy.choose_action_observation(observation, decision_seed=10)


def test_attempt13_profile_rejects_ambient_source_drift_before_bound_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"attempt13")
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    go_path, go_sha = _write_status(tmp_path, _complete_go(model_sha))
    loader_called = False

    def fail_attestation(**_kwargs: Any) -> object:
        raise ValueError("execution module was imported outside frozen tree")

    def forbidden_loader(*_args: Any, **_kwargs: Any) -> object:
        nonlocal loader_called
        loader_called = True
        return object()

    monkeypatch.setattr(profile, "validate_frozen_execution_modules", fail_attestation)
    monkeypatch.setattr(profile, "load_bound_attempt13_distilled_model", forbidden_loader)
    with pytest.raises(ValueError, match="outside frozen tree"):
        profile.build_stage20_m4_attempt13(
            _Baseline(),
            complete_go_path=go_path,
            expected_complete_go_sha256=go_sha,
            model_path=model_path,
            expected_model_sha256=model_sha,
            runtime_freeze_path=tmp_path / "freeze.json",
            training_manifest_path=tmp_path / "training.json",
            runtime_source_manifest_path=tmp_path / "runtime.json",
            runtime_source_root=tmp_path,
            runtime_dependency_root=tmp_path,
        )
    assert loader_called is False


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("runtime_profile", "stage18_p1"),
        ("runtime_status", "validation_only"),
        ("fallback_policy", "random_exact_final"),
        ("t1_continuation", "stage9f_p2"),
        ("t2_continuation", "stage18_p1"),
        ("t3_continuation", "exact_t4"),
    ],
)
def test_attempt13_profile_rejects_fixed_baseline_chain_drift(
    tmp_path: Path, field: str, bad_value: str
) -> None:
    baseline = _Baseline()
    baseline.decision_context[field] = bad_value
    with pytest.raises(ValueError, match="fixed baseline chain changed"):
        profile.build_stage20_m4_attempt13(
            baseline,
            complete_go_path=tmp_path / "complete_go.json",
            expected_complete_go_sha256="a" * 64,
            model_path=tmp_path / "model.pkl",
            expected_model_sha256="b" * 64,
            runtime_freeze_path=tmp_path / "freeze.json",
            training_manifest_path=tmp_path / "training.json",
            runtime_source_manifest_path=tmp_path / "runtime.json",
            runtime_source_root=tmp_path,
            runtime_dependency_root=tmp_path,
        )


def test_attempt13_profile_rejects_invalid_baseline_seat(tmp_path: Path) -> None:
    baseline = _Baseline("observer")
    with pytest.raises(ValueError, match="baseline seat is invalid"):
        profile.build_stage20_m4_attempt13(
            baseline,
            complete_go_path=tmp_path / "complete_go.json",
            expected_complete_go_sha256="a" * 64,
            model_path=tmp_path / "model.pkl",
            expected_model_sha256="b" * 64,
            runtime_freeze_path=tmp_path / "freeze.json",
            training_manifest_path=tmp_path / "training.json",
            runtime_source_manifest_path=tmp_path / "runtime.json",
            runtime_source_root=tmp_path,
            runtime_dependency_root=tmp_path,
        )


def _preflight(bound: Any) -> dict[str, Any]:
    return {
        "schema": ATTEMPT13_POPULATION_PREFLIGHT_SCHEMA,
        "status": "pass",
        "profile_id": "stage20_m4_attempt13",
        "runtime_binding_verified": True,
        "model_sha256": "a" * 64,
        "model_id": bound.model_id,
        "opponents": list(ATTEMPT13_OPPONENTS),
        "baseline_profile": "stage19_p0",
        "population_namespace_bases": list(ATTEMPT13_POPULATION_NAMESPACE_BASES),
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
        "all_reserved_registry_overlap_count": 0,
        "audit50_one_shot_go_bound": True,
        "development200_full_fit_bound": True,
        "threshold_reselection_performed": False,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
        "runtime_freeze_sha256": "b" * 64,
        "training_manifest_sha256": "c" * 64,
        "runtime_source_manifest_sha256": "d" * 64,
        "runtime_source_closure_sha256": "e" * 64,
        "runtime_semantic_closure_sha256": "f" * 64,
        "runtime_fingerprint_sha256": "1" * 64,
        "source_model_manifest_sha256": "2" * 64,
        "source_native_manifest_sha256": "3" * 64,
        "runtime_dependency_closure_sha256": "4" * 64,
        "seed_registry_sha256": "5" * 64,
        "population_plan_file_sha256": "6" * 64,
    }


def test_attempt13_population_runner_locks_shard_and_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound = SimpleNamespace(
        model_id="attempt13-bound",
        schema="model",
        artifact_schema="artifact",
        feature_schema="features",
        head_schema="heads",
        action_score_mode="score",
    )
    monkeypatch.setattr(
        population, "is_bound_attempt13_distilled_model", lambda value: value is bound
    )
    observed: dict[str, Any] = {}

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        candidate = kwargs["candidate_policy_factory"](
            policy_seed=123, seat="second", decision_log=[]
        )
        assert candidate.allowed_seats == ("second",)
        assert candidate.action_value_model is bound
        Path(kwargs["records_output"]).write_text("{}\n", encoding="utf-8")
        return {"schema": "hu_m4_t1_population_evaluation_v1"}

    monkeypatch.setattr(population, "evaluate_hu_m4_population", fake_evaluate)

    def factory(**_: Any) -> _Baseline:
        return _Baseline()

    opponent_factories = {name: factory for name in ATTEMPT13_OPPONENTS}
    records = tmp_path / "records.jsonl"
    result = population.evaluate_attempt13_population_shard(
        bound_model=bound,
        preflight=_preflight(bound),
        baseline_policy_factory=factory,
        opponent_policy_factories=opponent_factories,
        shard_index=3,
        records_output=records,
    )
    assert records.read_text(encoding="utf-8") == "{}\n"
    assert observed["paired_seeds"] == 50
    assert observed["seed"] == 250108071901 + 150 * 1000003
    assert observed["seed_stride"] == 1000003
    assert observed["baseline_profile"] == "stage19_p0"
    assert result["runtime_config"]["current_profile_used"] is False
    assert result["runtime_config"]["population_namespace_bases"] == list(
        ATTEMPT13_POPULATION_NAMESPACE_BASES
    )
    with pytest.raises(FileExistsError, match="immutable"):
        population.evaluate_attempt13_population_shard(
            bound_model=bound,
            preflight=_preflight(bound),
            baseline_policy_factory=factory,
            opponent_policy_factories=opponent_factories,
            shard_index=3,
            records_output=records,
        )


def test_attempt13_population_runner_rejects_opponent_or_shard_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound = SimpleNamespace(model_id="attempt13-bound")
    monkeypatch.setattr(
        population, "is_bound_attempt13_distilled_model", lambda value: value is bound
    )

    def factory(**_: Any) -> _Baseline:
        return _Baseline()

    opponents = {name: factory for name in reversed(ATTEMPT13_OPPONENTS)}
    with pytest.raises(ValueError, match="opponent set/order"):
        population.evaluate_attempt13_population_shard(
            bound_model=bound,
            preflight=_preflight(bound),
            baseline_policy_factory=factory,
            opponent_policy_factories=opponents,
            shard_index=0,
            records_output=tmp_path / "records.jsonl",
        )
    with pytest.raises(ValueError, match="0..19"):
        population.evaluate_attempt13_population_shard(
            bound_model=bound,
            preflight=_preflight(bound),
            baseline_policy_factory=factory,
            opponent_policy_factories={name: factory for name in ATTEMPT13_OPPONENTS},
            shard_index=20,
            records_output=tmp_path / "records2.jsonl",
        )


def test_attempt13_population_nonfires_have_full_trajectory_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound = SimpleNamespace(
        model_id="attempt13-bound",
        schema="model",
        artifact_schema="artifact",
        feature_schema="features",
        head_schema="heads",
        action_score_mode="score",
    )
    monkeypatch.setattr(
        population, "is_bound_attempt13_distilled_model", lambda value: value is bound
    )

    def factory(*, policy_seed: int, seat: str, decision_log: Any) -> _SemanticBaseline:
        del policy_seed, decision_log
        return _SemanticBaseline(seat)

    def trace(*, seed: int, profile_p0: str, profile_p1: str, policy_p0: Any, policy_p1: Any) -> dict[str, Any]:
        candidate = policy_p0 if hasattr(policy_p0, "policy_id") else (
            policy_p1 if hasattr(policy_p1, "policy_id") else None
        )
        if candidate is not None:
            seat = candidate.baseline_policy.seat
            assert candidate.decision_log is not None
            candidate.decision_log.append(
                {
                    "schema": HU_M4_T1_DECISION_SCHEMA,
                    "street": "T1",
                    "seat": seat,
                    "override_fired": False,
                    "nonfire_reason": (
                        "first_seat_delegated" if seat == "first" else "same_as_baseline"
                    ),
                    "predicted_delta": 0.0,
                    "safety_probability": None,
                    "runtime_binding_verified": True,
                }
            )
        return {
            "seed": seed,
            "profiles": {"p0": profile_p0, "p1": profile_p1},
            "score_p0": 0.0,
            "turns": [
                {
                    "turn": "T1",
                    "profile": profile_p0,
                    "action_key": "baseline",
                    "board": {"top": ["Ah"], "middle": [], "bottom": []},
                }
            ],
            "final": {"p0": {}, "p1": {}},
            "board_scores": {"p0": {"busted": False}, "p1": {"busted": False}},
        }

    result = population.evaluate_attempt13_population_shard(
        bound_model=bound,
        preflight=_preflight(bound),
        baseline_policy_factory=factory,
        opponent_policy_factories={name: factory for name in ATTEMPT13_OPPONENTS},
        shard_index=0,
        records_output=tmp_path / "records.jsonl",
        trace_fn=trace,
        progress_every=0,
    )
    assert result["invalid_counterfactuals"] == 0
    assert result["nonfire_cancellation_mismatches"] == 0
    assert result["nonfire_cancellation_unknown"] == 0
    assert result["population"]["by_seat"]["first"]["overrides"] == 0
    assert result["population"]["by_seat"]["first"]["delta_ev_per_hand"]["mean"] == 0.0
    assert result["population"]["by_seat"]["second"]["overrides"] == 0
    assert result["population"]["by_seat"]["second"]["delta_ev_per_hand"]["mean"] == 0.0
