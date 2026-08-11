from __future__ import annotations

import hashlib
import itertools
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m3_rust import (
    HuM3RustError,
    build_native_engine,
    evaluate_batch,
    evaluate_request,
    repository_root,
    t4_request,
)
from ofc_regular.hu_m3_t4_runtime import (
    HU_M30_T4_BELIEF_ID,
    HuM3T4ExactPolicy,
    HuM3T4ExactSolver,
    HuM3T4RuntimeConfig,
    HuM3T4RuntimeError,
)
from ofc_regular.state import Board
from ofc_regular.validate_hu_m30_t4_runtime import (
    generate_balanced_t4_observations,
    run_pilot,
)


def _t4_first_observation(
    dealt_cards: tuple[str, str, str] = ("4h", "2c", "Ah"),
) -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("Kh",),
            middle=("7d", "Jh", "Ac", "6s", "3c"),
            bottom=("Qs", "Qc", "Ks", "As", "Js"),
        ),
        opponent_public_board=Board.from_rows(
            top=("Ad",),
            middle=("3s", "8c", "4c", "7c", "6d"),
            bottom=("Td", "3d", "9d", "8d", "6h"),
        ),
        dealt_cards=dealt_cards,
        hero_private_discards=("Tc", "7h", "Th"),
        seat="first",
        street="T4",
        to_act_order="first",
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
def m30_solver() -> HuM3T4ExactSolver:
    configured = os.environ.get("HU_M30_TEST_TARGET_DIR")
    target_dir = (
        Path(configured)
        if configured
        else repository_root() / "target" / "hu_m30_test"
    )
    build = build_native_engine(release=True, target_dir=target_dir)
    return HuM3T4ExactSolver(
        HuM3T4RuntimeConfig(
            library_path=build.library_path,
            expected_library_sha256=hashlib.sha256(
                build.library_path.read_bytes()
            ).hexdigest(),
        )
    )


def _semantic_values(decision) -> dict[str, float]:
    return {row.action_key: row.ev for row in decision.action_values}


def test_first_runtime_returns_exact_action_and_every_legal_ev(m30_solver):
    observation = _t4_first_observation()
    decision = m30_solver.solve(observation)
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)

    assert decision.action in legal
    assert decision.selected_action_key == action_key(decision.action).to_token()
    assert decision.selected_ev == 6.0
    assert decision.future_count == 2024
    assert decision.belief_id == HU_M30_T4_BELIEF_ID
    assert decision.value_scope == (
        "exact_terminal_hu_ev_under_uniform_exchangeable_restart_belief"
    )
    assert len(decision.action_values) == len(legal)
    assert {row.original_index for row in decision.action_values} == set(range(len(legal)))
    assert decision.to_dict()["search_mode"] == "exhaustive"


def test_second_runtime_returns_terminal_exact_action_and_ev(m30_solver):
    observation = _t4_second_observation()
    decision = m30_solver.solve(observation)

    assert decision.selected_ev == 8.0
    assert decision.future_count == 1
    assert decision.belief_id is None
    assert decision.value_scope == "exact_terminal_hu_ev_against_complete_opponent_board"
    assert len(decision.action_values) == len(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )


def test_all_dealt_permutations_have_identical_semantic_action_values(m30_solver):
    decisions = [
        m30_solver.solve(_t4_first_observation(tuple(cards)))
        for cards in itertools.permutations(("4h", "2c", "Ah"))
    ]

    assert len({decision.selected_action_key for decision in decisions}) == 1
    assert len({decision.selected_ev for decision in decisions}) == 1
    assert all(
        _semantic_values(decision) == _semantic_values(decisions[0])
        for decision in decisions[1:]
    )


def test_scalar_and_native_batch_decisions_match(m30_solver):
    observations = [_t4_first_observation(), _t4_second_observation()]
    scalar = [m30_solver.solve(observation) for observation in observations]
    batched = m30_solver.solve_many(observations)

    assert [row.selected_action_key for row in batched] == [
        row.selected_action_key for row in scalar
    ]
    assert [row.selected_ev for row in batched] == [row.selected_ev for row in scalar]
    assert [_semantic_values(row) for row in batched] == [
        _semantic_values(row) for row in scalar
    ]
    assert all(row.execution_mode == "batch_amortized" for row in batched)
    assert all(
        row.total_latency_ms
        == pytest.approx(row.native_latency_ms + row.validation_latency_ms)
        for row in batched
    )


@pytest.mark.parametrize(
    "field",
    [
        "opponent_private_discards",
        "true_dead_cards",
        "remaining_deck",
        "world_state",
    ],
)
def test_python_and_native_observation_schemas_reject_hidden_truth_aliases(
    field, m30_solver
):
    observation = _t4_first_observation()
    payload = observation.to_dict()
    payload[field] = ["5h"]
    with pytest.raises(ValueError, match="unknown fields"):
        ActorObservation.from_dict(payload)

    request = t4_request(observation, config=m30_solver.search_config)
    request["observation"][field] = ["5h"]
    with pytest.raises(HuM3RustError, match="unknown fields"):
        evaluate_request(request, library=m30_solver.library)
    with pytest.raises(HuM3RustError, match="unknown fields"):
        evaluate_batch([request], library=m30_solver.library)


def test_native_request_and_config_reject_unknown_fields(m30_solver):
    observation = _t4_first_observation()
    request = t4_request(observation, config=m30_solver.search_config)
    request["replay_truth"] = {"remaining_deck": ["5h"]}
    with pytest.raises(HuM3RustError, match="unknown field"):
        evaluate_request(request, library=m30_solver.library)

    request = t4_request(observation, config=m30_solver.search_config)
    request["config"]["opponent_private_discards"] = ["5h"]
    with pytest.raises(HuM3RustError, match="unknown field"):
        evaluate_request(request, library=m30_solver.library)


def test_library_hash_and_engine_version_are_fail_closed(m30_solver):
    path = m30_solver.library_path
    with pytest.raises(HuM3T4RuntimeError, match="SHA-256 mismatch"):
        HuM3T4ExactSolver(
            HuM3T4RuntimeConfig(
                library_path=path,
                expected_library_sha256="0" * 64,
            )
        )
    with pytest.raises(HuM3T4RuntimeError, match="version mismatch"):
        HuM3T4ExactSolver(
            HuM3T4RuntimeConfig(
                library_path=path,
                expected_library_sha256=m30_solver.library_sha256,
                expected_engine_version="ofc_hu_m3_engine/invalid",
            )
        )


def test_wrapper_delegates_earlier_streets_and_uses_exact_t4(m30_solver):
    class BasePolicy:
        seat = "first"

        def __init__(self):
            self.calls = []

        def choose_action_observation(self, observation, **kwargs):
            self.calls.append((observation, kwargs))
            if observation.street == "T4":
                return generate_turn_actions(
                    observation.hero_board, observation.dealt_cards
                )[0]
            return "delegated"

        def choose_action(self, board, dealt_cards, **kwargs):
            return "legacy"

    base = BasePolicy()
    log = []
    wrapper = HuM3T4ExactPolicy(base, m30_solver, decision_log=log)
    non_t4 = SimpleNamespace(street="T3")

    assert wrapper.choose_action_observation(non_t4, hand_id=1) == "delegated"
    assert len(base.calls) == 1
    action = wrapper.choose_action_observation(_t4_first_observation(), hand_id=2)
    assert action_key(action).to_token() == wrapper.last_hu_t4_decision.selected_action_key
    assert len(log) == 1
    assert log[0]["selected_ev"] == wrapper.last_hu_t4_decision.selected_ev
    assert log[0]["baseline_action_key"] in {
        row.action_key for row in wrapper.last_hu_t4_decision.action_values
    }
    assert log[0]["final_action_key"] == wrapper.last_hu_t4_decision.selected_action_key
    assert log[0]["override_fired"] == (
        log[0]["baseline_action_key"] != log[0]["final_action_key"]
    )
    assert log[0]["exact_value_gain_vs_baseline"] >= 0.0
    assert log[0]["fallback_used"] is False


def test_wrapper_rejects_legacy_t4_direct_call(m30_solver):
    class BasePolicy:
        seat = "first"

        def choose_action(self, board, dealt_cards, **kwargs):
            return None

    wrapper = HuM3T4ExactPolicy(BasePolicy(), m30_solver)
    observation = _t4_first_observation()

    with pytest.raises(HuM3T4RuntimeError, match="ActorObservation"):
        wrapper.choose_action(observation.hero_board, observation.dealt_cards)


def test_wrapper_skips_legacy_t4_when_decision_logging_is_disabled(m30_solver):
    class BasePolicy:
        seat = "first"

        def __init__(self):
            self.calls = 0

        def choose_action_observation(self, observation, **kwargs):
            self.calls += 1
            raise AssertionError("legacy T4 must not run without diagnostic logging")

    base = BasePolicy()
    wrapper = HuM3T4ExactPolicy(base, m30_solver)

    action = wrapper.choose_action_observation(_t4_first_observation())

    assert base.calls == 0
    assert (
        action_key(action).to_token()
        == wrapper.last_hu_t4_decision.selected_action_key
    )


def test_wrapper_rejects_invalid_logged_legacy_baseline(m30_solver):
    class BasePolicy:
        seat = "first"

        def choose_action_observation(self, observation, **kwargs):
            return "not-an-action"

    wrapper = HuM3T4ExactPolicy(BasePolicy(), m30_solver, decision_log=[])

    with pytest.raises(HuM3T4RuntimeError, match="invalid Action"):
        wrapper.choose_action_observation(_t4_first_observation())


def test_native_result_corruption_never_falls_back(monkeypatch, m30_solver):
    observation = _t4_first_observation()
    from ofc_regular import hu_m3_t4_runtime as runtime

    real_evaluate = runtime.evaluate_t4

    def corrupt(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        result["actions"][0]["score"] = float("nan")
        return result

    monkeypatch.setattr(runtime, "evaluate_t4", corrupt)
    with pytest.raises(HuM3T4RuntimeError, match="must be finite"):
        m30_solver.solve(observation)


def test_nested_public_board_unknown_field_is_rejected():
    payload = _t4_first_observation().to_dict()
    payload["hero_board"]["opponent_private_discards"] = ["5h"]

    with pytest.raises(ValueError, match="public board contains unknown fields"):
        ActorObservation.from_dict(payload)


def test_matchup_opt_in_wraps_only_profile_a_and_records_both_seats(
    tmp_path, m30_solver
):
    from ofc_regular.evaluate_matchups import evaluate_matchup

    decision_output = tmp_path / "t4.jsonl"
    summary = evaluate_matchup(
        profile_a="random_exact_final",
        profile_b="random_exact_final",
        games=1,
        seed=2026071601,
        seed_stride=1009,
        bundle=SimpleNamespace(),
        opening_lookahead_samples=1,
        t4_solver_a=m30_solver,
        t4_solver_b=None,
        hu_t4_decision_output=decision_output,
    )

    assert summary["t4_mode_a"] == "m30_exact"
    assert summary["t4_mode_b"] == "legacy"
    assert summary["hu_t4_decisions_written"] == 2
    assert summary["hu_t4_runtime"]["first_decision_count"] == 1
    assert summary["hu_t4_runtime"]["second_decision_count"] == 1
    realized = summary["hu_t4_realized_pair"]
    assert realized["available"] is True
    assert realized["pure_t4_counterfactual"] is True
    assert realized["paired_count"] == 1
    assert realized["incomplete_pair_count"] == 0
    assert realized["override_count"] + realized["nonfire_count"] == 1
    if realized["nonfire_count"]:
        assert realized["nonfire_cancellation_evaluable"] is True
        assert realized["nonfire_exact_cancellation"] is True
        assert realized["nonfire_trajectory_mismatch_count"] == 0
    rows = decision_output.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert all('"baseline_action_key"' in row for row in rows)
    assert all('"final_action_key"' in row for row in rows)


def test_t4_pair_summary_counts_either_seat_fire_and_requires_trajectory_identity():
    from ofc_regular.evaluate_matchups import _summarize_t4_realized_pairs

    rows = [
        {
            "paired_index": 0,
            "policy_role": "a",
            "seat": "first",
            "realized_seat_score": 1.0,
            "override_fired": False,
            "hand_trajectory_digest": "same",
        },
        {
            "paired_index": 0,
            "policy_role": "a",
            "seat": "second",
            "realized_seat_score": 3.0,
            "override_fired": True,
            "hand_trajectory_digest": "different",
        },
        {
            "paired_index": 1,
            "policy_role": "a",
            "seat": "first",
            "realized_seat_score": -2.0,
            "override_fired": False,
            "hand_trajectory_digest": "left",
        },
        {
            "paired_index": 1,
            "policy_role": "a",
            "seat": "second",
            "realized_seat_score": 2.0,
            "override_fired": False,
            "hand_trajectory_digest": "right",
        },
    ]

    summary = _summarize_t4_realized_pairs(
        rows, pure_t4_counterfactual=True
    )

    assert summary["override_count"] == 1
    assert summary["second_seat_override_count"] == 1
    assert summary["realized_gain_per_override"] == 4.0
    assert summary["nonfire_count"] == 1
    assert summary["nonfire_nonzero_count"] == 0
    assert summary["nonfire_trajectory_mismatch_count"] == 1
    assert summary["nonfire_exact_cancellation"] is False

    nonpure = _summarize_t4_realized_pairs(
        rows, pure_t4_counterfactual=False
    )
    assert nonpure["available"] is False
    assert nonpure["override_count"] is None
    assert nonpure["realized_gain_per_override"] is None
    assert nonpure["nonfire_count"] is None
    assert nonpure["nonfire_exact_cancellation"] is False

    duplicate = _summarize_t4_realized_pairs(
        [rows[0], rows[0], rows[1]], pure_t4_counterfactual=True
    )
    assert duplicate["available"] is False
    assert duplicate["invalid_pair_count"] == 1
    assert duplicate["override_count"] is None


def test_balanced_pilot_roots_are_live_geometry_and_deterministic():
    first = generate_balanced_t4_observations(
        states=10,
        seed=2026071601,
        seed_stride=1009,
    )
    second = generate_balanced_t4_observations(
        states=10,
        seed=2026071601,
        seed_stride=1009,
    )

    assert [row.fingerprint() for row in first] == [row.fingerprint() for row in second]
    assert len({row.fingerprint() for row in first}) == 10
    assert sum(row.seat == "first" for row in first) == 5
    assert sum(row.seat == "second" for row in first) == 5


def test_bounded_pilot_reports_parity_determinism_and_latency(m30_solver):
    result = run_pilot(
        solver=m30_solver,
        states=4,
        seed=2026071601,
        seed_stride=1009,
        python_parity_roots=2,
        determinism_roots=2,
    )

    assert result["seat_counts"] == {"first": 2, "second": 2}
    assert result["gates"]["python_reference_parity"] is True
    assert result["gates"]["scalar_batch_parity"] is True
    assert result["gates"]["deterministic_rerun"] is True
    assert result["gates"]["all_first_exact_2024"] is True
    assert result["gates"]["all_second_terminal_exact"] is True
    assert result["gates"]["second_legacy_exact_action_parity"] is True
    assert result["by_seat"]["first"]["latency_p99_ms"] > 0.0
    assert result["by_seat"]["second"]["latency_p99_ms"] > 0.0
