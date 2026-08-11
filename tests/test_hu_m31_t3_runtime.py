from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m3_rust import (
    HU_M3_BATCH_REQUEST_SCHEMA,
    HuM3RustError,
    build_native_engine,
    evaluate_batch,
    evaluate_request,
    repository_root,
    t3_request,
)
from ofc_regular.hu_m31_t3_runtime import (
    HU_M31_T3_RUNTIME_ID,
    HU_M31_T3_RUNTIME_SCHEMA,
    HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA,
    HuM31T3RuntimeConfig,
    HuM31T3RuntimeError,
    HuM31T3SearchSolver,
)
from ofc_regular.state import Board


def _constrained_board(cards: tuple[str, ...]) -> Board:
    """Keep integration tests small while preserving live T3 sequencing."""

    return Board.from_rows(
        top=cards[:3],
        middle=cards[3:8],
        bottom=cards[8:],
    )


def _t3_observation(to_act_order: str) -> ActorObservation:
    opponent_count = 9 if to_act_order == "first" else 11
    cursor = 0
    hero_cards = ALL_CARDS[cursor : cursor + 9]
    cursor += 9
    opponent_cards = ALL_CARDS[cursor : cursor + opponent_count]
    cursor += opponent_count
    dealt = ALL_CARDS[cursor : cursor + 3]
    cursor += 3
    hero_discards = ALL_CARDS[cursor : cursor + 2]
    return ActorObservation(
        hero_board=_constrained_board(hero_cards),
        opponent_public_board=_constrained_board(opponent_cards),
        dealt_cards=dealt,
        hero_private_discards=hero_discards,
        seat=to_act_order,  # type: ignore[arg-type]
        street="T3",
        to_act_order=to_act_order,  # type: ignore[arg-type]
    )


def _t4_second_observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("Qh", "Kc"),
            middle=("Ah", "Ac", "4s", "5s"),
            bottom=("Td", "7h", "7s", "7c", "Th"),
        ),
        opponent_public_board=Board.from_rows(
            top=("2h", "3h", "4h"),
            middle=("5c", "2s", "6h", "4c", "8c"),
            bottom=("7d", "Jd", "9d", "Qd", "Kd"),
        ),
        dealt_cards=("2c", "3c", "4d"),
        hero_private_discards=("6c", "8h", "9c"),
        seat="second",
        street="T4",
        to_act_order="second",
    )


@pytest.fixture(scope="session")
def m31_solver() -> HuM31T3SearchSolver:
    configured = os.environ.get("HU_M31_TEST_NATIVE_LIBRARY")
    if configured:
        library_path = Path(configured)
    else:
        accepted_m30 = (
            repository_root()
            / "target"
            / "m30_build"
            / "release"
            / ("ofc_hu_m3_engine.dll" if os.name == "nt" else "libofc_hu_m3_engine.so")
        )
        if accepted_m30.is_file():
            library_path = accepted_m30
        else:
            target_dir = repository_root() / "target" / "hu_m31_test"
            library_path = build_native_engine(
                release=True,
                target_dir=target_dir,
            ).library_path
    return HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            library_path=library_path,
            expected_library_sha256=hashlib.sha256(
                library_path.read_bytes()
            ).hexdigest(),
        )
    )


@pytest.fixture(scope="session")
def raw_first_result(m31_solver):
    return evaluate_request(
        t3_request(_t3_observation("first"), config=m31_solver.search_config),
        library=m31_solver.library,
    )


@pytest.fixture(scope="session")
def raw_batch_result(m31_solver):
    observations = [_t3_observation("first"), _t3_observation("second")]
    return evaluate_request(
        {
            "schema": HU_M3_BATCH_REQUEST_SCHEMA,
            "requests": [
                t3_request(observation, config=m31_solver.search_config)
                for observation in observations
            ],
        },
        library=m31_solver.library,
    )


def _semantic_values(decision) -> dict[str, tuple[float, float]]:
    return {
        row.action_key: (row.selection_ev, row.evaluation_ev)
        for row in decision.action_values
    }


def _complete_semantic_values(
    decision,
) -> dict[str, tuple[int, float, float, float]]:
    return {
        row.action_key: (
            row.rank,
            row.selection_ev,
            row.evaluation_ev,
            row.evaluation_regret,
        )
        for row in decision.action_values
    }


def test_both_seats_force_exact_t4_and_return_every_legal_action(m31_solver):
    decisions = [
        m31_solver.solve(_t3_observation("first")),
        m31_solver.solve(_t3_observation("second")),
    ]

    assert m31_solver.search_config.downstream_t4_samples == 0
    for decision in decisions:
        observation = _t3_observation(decision.seat)
        legal = generate_turn_actions(
            observation.hero_board,
            observation.dealt_cards,
        )
        payload = decision.to_dict()
        assert decision.action in legal
        assert decision.selected_action_key == action_key(decision.action).to_token()
        assert len(decision.action_values) == len(legal)
        assert {row.original_index for row in decision.action_values} == set(
            range(len(legal))
        )
        assert payload["runtime_id"] == HU_M31_T3_RUNTIME_ID
        assert payload["downstream_t4_samples"] == 0
        assert payload["downstream_t4_mode"] == "exact"
        assert payload["downstream_t4_native_semantics_id"] == (
            "m30_exact_t4_native_kernel_semantics_v1"
        )
        assert payload["run_id"] == m31_solver.config.run_id
        assert payload["continuation_seed"] == m31_solver.config.seed
        assert payload["candidate_seed"] == m31_solver.config.candidate_seed
        assert payload["evaluation_seed"] == m31_solver.config.evaluation_seed
        assert len(payload["search_contract_digest"]) == 64
        assert payload["teacher_value_status"] == "diagnostic_not_match_EV"
        assert decision.candidate_rng_digest != decision.evaluation_rng_digest


def test_scalar_batch_and_repeated_run_are_semantically_identical(m31_solver):
    observations = [_t3_observation("first"), _t3_observation("second")]
    scalar = [m31_solver.solve(observation) for observation in observations]
    batched = m31_solver.solve_many(observations)
    repeated = [m31_solver.solve(observation) for observation in observations]

    assert [row.selected_action_key for row in batched] == [
        row.selected_action_key for row in scalar
    ]
    assert [row.selected_evaluation_ev for row in batched] == [
        row.selected_evaluation_ev for row in scalar
    ]
    assert [_semantic_values(row) for row in batched] == [
        _semantic_values(row) for row in scalar
    ]
    assert [row.result_digest for row in repeated] == [
        row.result_digest for row in scalar
    ]
    assert all(row.execution_mode == "batch_amortized" for row in batched)


def test_all_t3_dealt_permutations_are_semantically_identical_for_both_seats(
    m31_solver,
):
    observations_by_seat = {
        seat: [
            replace(baseline, dealt_cards=tuple(cards))
            for cards in itertools.permutations(baseline.dealt_cards)
        ]
        for seat in ("first", "second")
        for baseline in (_t3_observation(seat),)
    }
    observations = [
        observation
        for seat in ("first", "second")
        for observation in observations_by_seat[seat]
    ]
    scalar = [m31_solver.solve(observation) for observation in observations]
    batched = m31_solver.solve_many(observations)

    assert [row.semantic_result_digest for row in batched] == [
        row.semantic_result_digest for row in scalar
    ]
    assert [row.result_digest for row in batched] == [
        row.result_digest for row in scalar
    ]

    offset = 0
    for seat in ("first", "second"):
        seat_observations = observations_by_seat[seat]
        seat_decisions = scalar[offset : offset + len(seat_observations)]
        offset += len(seat_observations)
        baseline = seat_decisions[0]

        assert {observation.fingerprint() for observation in seat_observations} == {
            baseline.observation_fingerprint
        }
        assert {row.semantic_result_digest for row in seat_decisions} == {
            baseline.semantic_result_digest
        }
        assert {row.selected_action_key for row in seat_decisions} == {
            baseline.selected_action_key
        }
        assert {row.selected_selection_ev for row in seat_decisions} == {
            baseline.selected_selection_ev
        }
        assert {row.selected_evaluation_ev for row in seat_decisions} == {
            baseline.selected_evaluation_ev
        }
        assert {row.selection_gap for row in seat_decisions} == {
            baseline.selection_gap
        }
        assert {row.evaluation_sample_regret for row in seat_decisions} == {
            baseline.evaluation_sample_regret
        }
        assert {
            json.dumps(_complete_semantic_values(row), sort_keys=True)
            for row in seat_decisions
        } == {
            json.dumps(_complete_semantic_values(baseline), sort_keys=True)
        }
        assert {row.legal_action_set_digest for row in seat_decisions} == {
            baseline.legal_action_set_digest
        }
        assert {row.candidate_belief_digest for row in seat_decisions} == {
            baseline.candidate_belief_digest
        }
        assert {row.evaluation_belief_digest for row in seat_decisions} == {
            baseline.evaluation_belief_digest
        }
        assert {row.candidate_rng_digest for row in seat_decisions} == {
            baseline.candidate_rng_digest
        }
        assert {row.evaluation_rng_digest for row in seat_decisions} == {
            baseline.evaluation_rng_digest
        }
        assert {row.child_information_set_count for row in seat_decisions} == {
            baseline.child_information_set_count
        }

        # Positional provenance deliberately remains permutation-bound.  Each
        # mapping must still resolve to the same semantic ActionKey locally.
        assert len({row.legal_action_order_digest for row in seat_decisions}) == 6
        assert len({row.result_digest for row in seat_decisions}) == 6
        for observation, decision in zip(
            seat_observations, seat_decisions, strict=True
        ):
            legal = generate_turn_actions(
                observation.hero_board,
                observation.dealt_cards,
            )
            for value in decision.action_values:
                assert action_key(legal[value.original_index]).to_token() == (
                    value.action_key
                )
            payload = decision.to_dict()
            assert payload["schema"] == HU_M31_T3_RUNTIME_SCHEMA
            assert payload["semantic_result_digest_schema"] == (
                HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA
            )
            assert payload["semantic_result_digest_scope"] == (
                "dealt_order_independent_action_value_result"
            )
            assert payload["result_digest_scope"] == (
                "ordered_action_mapping_bound"
            )


@pytest.mark.parametrize(
    "field",
    [
        "opponent_private_discards",
        "true_dead_cards",
        "remaining_deck",
        "world_state",
        "replay_truth",
    ],
)
def test_python_and_native_requests_reject_hidden_truth_aliases(
    field, m31_solver
):
    observation = _t3_observation("first")
    payload = observation.to_dict()
    payload[field] = ["5h"]
    with pytest.raises(ValueError, match="unknown fields"):
        ActorObservation.from_dict(payload)

    request = t3_request(observation, config=m31_solver.search_config)
    request["observation"][field] = ["5h"]
    with pytest.raises(HuM3RustError, match="unknown fields"):
        evaluate_request(request, library=m31_solver.library)
    with pytest.raises(HuM3RustError, match="unknown fields"):
        evaluate_batch([request], library=m31_solver.library)


def test_hash_version_release_and_missing_library_fail_closed(m31_solver, tmp_path):
    path = m31_solver.library_path
    with pytest.raises(HuM31T3RuntimeError, match="SHA-256 mismatch"):
        HuM31T3SearchSolver(
            HuM31T3RuntimeConfig(
                library_path=path,
                expected_library_sha256="0" * 64,
            )
        )
    with pytest.raises(HuM31T3RuntimeError, match="version mismatch"):
        HuM31T3SearchSolver(
            HuM31T3RuntimeConfig(
                library_path=path,
                expected_library_sha256=m31_solver.library_sha256,
                expected_engine_version="ofc_hu_m3_engine/invalid",
            )
        )
    with pytest.raises(HuM31T3RuntimeError, match="does not exist"):
        HuM31T3SearchSolver(
            HuM31T3RuntimeConfig(
                library_path=tmp_path / "missing" / "ofc_hu_m3_engine.dll",
                expected_library_sha256="0" * 64,
            )
        )


def test_runtime_rejects_legacy_or_wrong_street_inputs(m31_solver):
    with pytest.raises(TypeError, match="ActorObservation"):
        m31_solver.solve(object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="requires street T3"):
        m31_solver.solve(_t4_second_observation())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda result: result["continuation_policy"].__setitem__(
                "downstream_t4_samples", 1
            ),
            "downstream_t4_samples mismatch",
        ),
        (
            lambda result: result.__setitem__(
                "opponent_private_discards", ["5h"]
            ),
            "field mismatch",
        ),
        (
            lambda result: result["actions"][0].__setitem__(
                "score", float("nan")
            ),
            "must be finite",
        ),
        (
            lambda result: result["candidate_rng_key_digests"].__setitem__(
                0, result["evaluation_rng_key_digests"][0]
            ),
            "disagree with deterministic ActorObservation belief",
        ),
        (
            lambda result: result["candidate_rng_key_digests"].__setitem__(
                0, "a" * 64
            ),
            "disagree with deterministic ActorObservation belief",
        ),
        (
            lambda result: result["candidate_belief"]["particle_digests"].__setitem__(
                0, "b" * 64
            ),
            "particle digests disagree with deterministic belief",
        ),
        (
            lambda result: result["continuation_policy"].__setitem__(
                "downstream_t4_samples", False
            ),
            "expected integer 0",
        ),
        (
            lambda result: result["continuation_policy"].__setitem__(
                "downstream_t3_samples", True
            ),
            "expected integer 1",
        ),
        (
            lambda result: result["candidate_belief"].__setitem__(
                "sample_count", True
            ),
            "expected integer 1",
        ),
        (
            lambda result: result["candidate_belief"].__setitem__(
                "start_index", False
            ),
            "expected integer 0",
        ),
        (
            lambda result: result["actions"][0].__setitem__(
                "selection_future_count", True
            ),
            "expected integer 1",
        ),
        (
            lambda result: result.__setitem__(
                "selected_action_original_index", False
            ),
            "invalid selected_action_original_index",
        ),
        (
            lambda result: result["actions"][0].__setitem__(
                "selection_score", "1.0"
            ),
            "must be finite",
        ),
    ],
)
def test_native_result_corruption_never_falls_back(
    monkeypatch, m31_solver, raw_first_result, mutation, message
):
    from ofc_regular import hu_m31_t3_runtime as runtime

    corrupted = copy.deepcopy(raw_first_result)
    mutation(corrupted)
    monkeypatch.setattr(runtime, "evaluate_t3", lambda *args, **kwargs: corrupted)

    with pytest.raises(HuM31T3RuntimeError, match=message):
        m31_solver.solve(_t3_observation("first"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda response: response.__setitem__(
                "schema", "hu_m3_engine_batch_result_invalid"
            ),
            "schema mismatch",
        ),
        (
            lambda response: response.__setitem__(
                "engine_version", "ofc_hu_m3_engine/invalid"
            ),
            "engine_version mismatch",
        ),
        (
            lambda response: response.__setitem__("replay_truth", {}),
            "field mismatch",
        ),
    ],
)
def test_batch_envelope_corruption_never_falls_back(
    monkeypatch, m31_solver, raw_batch_result, mutation, message
):
    from ofc_regular import hu_m31_t3_runtime as runtime

    corrupted = copy.deepcopy(raw_batch_result)
    mutation(corrupted)
    monkeypatch.setattr(runtime, "evaluate_request", lambda *args, **kwargs: corrupted)

    with pytest.raises(HuM31T3RuntimeError, match=message):
        m31_solver.solve_many(
            [_t3_observation("first"), _t3_observation("second")]
        )


def test_runtime_config_requires_positive_samples_and_disjoint_seeds():
    with pytest.raises(ValueError, match="candidate_samples"):
        HuM31T3RuntimeConfig(
            expected_library_sha256="0" * 64,
            candidate_samples=0,
        )
    with pytest.raises(ValueError, match="must be distinct"):
        HuM31T3RuntimeConfig(
            expected_library_sha256="0" * 64,
            candidate_seed=7,
            evaluation_seed=7,
        )


def test_m31_config_pins_m30_binary_and_preserves_policy_registry():
    root = repository_root()
    config = json.loads(
        (root / "configs" / "hu_joint_policy_m31_t3_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    native = root / config["native_anchor"]["library"]
    policy_registry = root / "src" / "ofc_regular" / "ai_profiles.py"

    assert config["current_profile_changed"] is False
    assert config["named_profile_added"] is False
    assert config["runtime_policy_activated"] is False
    assert config["native_anchor"]["build_if_missing"] is False
    assert config["native_anchor"]["strict_no_fallback"] is True
    assert config["native_anchor"]["platform"] == "windows-x86_64"
    assert config["component"]["decision_schema"] == HU_M31_T3_RUNTIME_SCHEMA
    assert config["component"]["semantic_result_digest_schema"] == (
        HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA
    )
    assert config["search_smoke_contract"]["downstream_t4_samples"] == 0
    assert config["search_smoke_contract"]["root_t3_classification"] == (
        "monte_carlo_not_exact"
    )
    assert len(config["native_anchor"]["library_sha256"]) == 64
    if native.is_file():
        assert hashlib.sha256(native.read_bytes()).hexdigest() == config[
            "native_anchor"
        ]["library_sha256"]
    m30_config = root / config["m30_source_contract"]["runtime_config"]
    m30_audit = root / config["m30_source_contract"]["completion_audit"]
    assert hashlib.sha256(m30_config.read_bytes()).hexdigest() == config[
        "m30_source_contract"
    ]["runtime_config_sha256"]
    assert hashlib.sha256(m30_audit.read_bytes()).hexdigest() == config[
        "m30_source_contract"
    ]["completion_audit_sha256"]
    assert hashlib.sha256(policy_registry.read_bytes()).hexdigest() == config[
        "baseline_invariants"
    ]["policy_registry_expected_sha256"]
