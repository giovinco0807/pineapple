from __future__ import annotations

import pytest

import ofc_regular.hu_m4_t1_teacher as teacher
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_belief import sample_hidden_card_particles
from ofc_regular.hu_infoset import ActorObservation, WorldState
from ofc_regular.hu_m4_teacher_contract import (
    T1_SECOND_LIVE_SCHEDULE,
    child_policy_decision_seed,
    require_t1_second_root,
    split_t1_second_future,
)
from ofc_regular.state import Board


def _root() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("9h",), middle=("Th", "Jh"), bottom=("Qh", "Kh")
        ),
        opponent_public_board=Board.from_rows(
            top=("2h",),
            middle=("3h", "4h", "5h"),
            bottom=("6h", "7h", "8h"),
        ),
        dealt_cards=("Ah", "2d", "3d"),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )


class _FirstLegalPolicy:
    def __init__(self, seat: str, events: list | None = None) -> None:
        self.seat = seat
        self.events = events
        self.seeds: list[int] = []

    def choose_action_observation(
        self,
        observation: ActorObservation,
        *,
        hand_id=None,
        game_id=None,
        decision_seed=None,
    ):
        assert observation.seat == self.seat
        assert hand_id is None and game_id is None
        assert isinstance(decision_seed, int)
        self.seeds.append(decision_seed)
        if self.events is not None:
            self.events.append((observation.seat, observation.street, observation))
        return generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )[0]


def _config(**overrides) -> teacher.M4T1TeacherConfig:
    values = {
        "candidate_samples": 1,
        "evaluation_samples": 1,
        "candidate_seed": 101,
        "evaluation_seed": 102,
        "run_id": "m4-test",
        "t2_policy_id": "first-legal-v1",
        "child_policy_seed": 103,
        "t3_candidate_samples": 1,
        "t3_evaluation_samples": 1,
        "t3_downstream_samples": 1,
        "t4_candidate_samples": 1,
        "t4_evaluation_samples": 1,
    }
    values.update(overrides)
    return teacher.M4T1TeacherConfig(**values)


def _patch_first_legal_m3(monkeypatch, events: list | None = None) -> None:
    def choose(observation: ActorObservation, **_kwargs):
        if events is not None:
            events.append((observation.seat, observation.street, observation))
        action = generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )[0]
        return {"selected_action_key": action_key(action).to_token()}

    monkeypatch.setattr(teacher, "evaluate_t3", choose)
    monkeypatch.setattr(teacher, "evaluate_t4", choose)


def _patch_first_legal_m3_batch(monkeypatch) -> None:
    def choose_many(requests, **_kwargs):
        results = []
        for request in requests:
            observation = ActorObservation.from_dict(request["observation"])
            action = generate_turn_actions(
                observation.hero_board, observation.dealt_cards
            )[0]
            results.append({"selected_action_key": action_key(action).to_token()})
        return results

    monkeypatch.setattr(teacher, "evaluate_batch", choose_many)


def test_t1_second_live_schedule_is_the_real_six_decision_order() -> None:
    assert [
        (step.seat, step.street, step.draw_offset)
        for step in T1_SECOND_LIVE_SCHEDULE
    ] == [
        ("first", "T2", 0),
        ("second", "T2", 3),
        ("first", "T3", 6),
        ("second", "T3", 9),
        ("first", "T4", 12),
        ("second", "T4", 15),
    ]
    future = tuple(f"card-{index}" for index in range(18))
    assert split_t1_second_future(future)[-1] == (
        "card-15",
        "card-16",
        "card-17",
    )


def test_teacher_rejects_a_non_second_seat_root() -> None:
    first = ActorObservation(
        hero_board=_root().hero_board,
        opponent_public_board=Board.from_rows(
            top=("2h",), middle=("3h", "4h"), bottom=("5h", "6h")
        ),
        dealt_cards=_root().dealt_cards,
        hero_private_discards=(),
        seat="first",
        street="T1",
        to_act_order="first",
    )
    with pytest.raises(ValueError, match="T1/second"):
        require_t1_second_root(first)


def test_candidate_action_is_locked_before_independent_evaluation(monkeypatch) -> None:
    calls = 0

    def controlled_scores(_observation, actions, _batch, _selector):
        nonlocal calls
        calls += 1
        if calls == 1:
            values = [10.0, 9.0, *([0.0] * (len(actions) - 2))]
        else:
            values = [-10.0, 10.0, *([0.0] * (len(actions) - 2))]
        return tuple(teacher._ActionScores((value,)) for value in values)

    monkeypatch.setattr(teacher, "_score_actions", controlled_scores)
    result = teacher.evaluate_t1_second_actions(
        _root(),
        t2_policies={"first": object(), "second": object()},
        config=_config(),
    )
    actions = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    assert result["selected_action_key"] == action_key(actions[0]).to_token()
    assert result["selected_action_evaluation_score"] == -10.0
    assert result["evaluation_sample_best_score"] == 10.0
    assert result["evaluation_sample_regret_of_locked_selection"] == 20.0
    assert result["root_selection_lock"] == (
        "candidate_action_key_locked_before_evaluation"
    )


def test_paired_delta_summary_uses_aligned_common_future_scores() -> None:
    summary = teacher._paired_delta_summary(
        teacher._ActionScores((3.0, 6.0, -1.0, 2.0)),
        teacher._ActionScores((1.0, 2.0, -2.0, 2.0)),
    )

    # Aligned differences are (2, 4, 1, 0); marginal SE combination would be
    # a different and less efficient quantity.
    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(1.75)
    assert summary["standard_error"] == pytest.approx(1.707825127659933 / 2.0)
    assert summary["min"] == 0.0
    assert summary["p50"] == pytest.approx(1.5)
    assert summary["max"] == 4.0
    assert summary["lt0_rate"] == 0.0

    zero = teacher._paired_delta_summary(
        teacher._ActionScores((3.0, 6.0, -1.0, 2.0)),
        teacher._ActionScores((3.0, 6.0, -1.0, 2.0)),
    )
    assert zero["count"] == 4
    assert all(
        float(zero[key]) == 0.0
        for key in (
            "mean",
            "standard_error",
            "std",
            "min",
            "p01",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "max",
            "lt0_rate",
            "le_neg6_rate",
            "le_neg12_rate",
            "le_neg20_rate",
        )
    )


def test_full_fake_rollout_uses_observation_only_live_children(monkeypatch) -> None:
    events: list[tuple[str, str, ActorObservation]] = []
    _patch_first_legal_m3(monkeypatch, events)
    policies = {
        "first": _FirstLegalPolicy("first", events),
        "second": _FirstLegalPolicy("second", events),
    }
    result = teacher.evaluate_t1_second_actions(
        _root(), t2_policies=policies, config=_config()
    )
    assert [(seat, street) for seat, street, _ in events[:6]] == [
        ("first", "T2"),
        ("second", "T2"),
        ("first", "T3"),
        ("second", "T3"),
        ("first", "T4"),
        ("second", "T4"),
    ]
    for _seat, _street, observation in events:
        payload = observation.to_dict()
        assert "opponent_private_discards" not in payload
        assert "future_cards" not in payload
    assert set(result["candidate_rng_key_digests"]).isdisjoint(
        result["evaluation_rng_key_digests"]
    )
    assert result["selected_action_key"] == result["actions"][0]["action_key"]
    assert result["continuation_policy"]["outer_future_index_in_child_seed"] is False
    assert result["continuation_policy"]["outer_action_index_in_child_seed"] is False


def test_same_child_infoset_has_one_cached_action_and_seed() -> None:
    root = _root()
    config = _config()
    particle = sample_hidden_card_particles(
        root,
        base_seed=config.candidate_seed,
        run_id=f"{config.run_id}:candidate_selection",
        sample_count=1,
    ).particles[0]
    root_action = generate_turn_actions(root.hero_board, root.dealt_cards)[0]
    second_after = root.hero_board.place(root_action.placements)
    world = WorldState(
        boards=(root.opponent_public_board, second_after),
        private_discards=(particle.opponent_private_discards, root_action.discards),
        street="T2",
        next_player=0,
        scoring=root.scoring,
    )
    child = world.observe(0, particle.draw(3, offset=0))
    first_policy = _FirstLegalPolicy("first")
    selector = teacher._ChildSelector(
        t2_policies={
            "first": first_policy,
            "second": _FirstLegalPolicy("second"),
        },
        config=config,
        library=None,
    )
    assert selector.choose(child) == selector.choose(child)
    assert len(first_policy.seeds) == 1
    expected = child_policy_decision_seed(
        base_seed=config.child_policy_seed,
        policy_id=f"{config.t2_policy_id}:seat=first",
        observation=child,
    )
    assert first_policy.seeds == [expected]


def test_public_teacher_api_rejects_raw_hidden_truth_inputs() -> None:
    root = _root()
    config = _config()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        teacher.evaluate_t1_second_actions(
            root,
            t2_policies={"first": object(), "second": object()},
            config=config,
            opponent_private_discards=("As",),  # type: ignore[call-arg]
        )


def test_batched_child_execution_is_scalar_value_and_actionkey_identical(monkeypatch) -> None:
    _patch_first_legal_m3(monkeypatch)
    _patch_first_legal_m3_batch(monkeypatch)
    scalar_policies = {
        "first": _FirstLegalPolicy("first"),
        "second": _FirstLegalPolicy("second"),
    }
    batch_policies = {
        "first": _FirstLegalPolicy("first"),
        "second": _FirstLegalPolicy("second"),
    }
    baseline = generate_turn_actions(_root().hero_board, _root().dealt_cards)[-1]
    scalar = teacher.evaluate_t1_second_actions(
        _root(),
        t2_policies=scalar_policies,
        baseline_action=baseline,
        config=_config(batch_child_selectors=False),
    )
    batched = teacher.evaluate_t1_second_actions(
        _root(),
        t2_policies=batch_policies,
        baseline_action=baseline,
        config=_config(batch_child_selectors=True),
    )

    assert batched["selected_action_key"] == scalar["selected_action_key"]
    assert [row["action_key"] for row in batched["actions"]] == [
        row["action_key"] for row in scalar["actions"]
    ]
    assert [row["selection_score"] for row in batched["actions"]] == [
        row["selection_score"] for row in scalar["actions"]
    ]
    assert [row["evaluation_score"] for row in batched["actions"]] == [
        row["evaluation_score"] for row in scalar["actions"]
    ]
    assert [row["evaluation_delta_vs_baseline"] for row in batched["actions"]] == [
        row["evaluation_delta_vs_baseline"] for row in scalar["actions"]
    ]
    assert batched["paired_delta_baseline_action_key"] == action_key(
        baseline
    ).to_token()
    baseline_row = next(
        row
        for row in batched["actions"]
        if row["action_key"] == action_key(baseline).to_token()
    )
    assert baseline_row["evaluation_delta_vs_baseline"]["mean"] == 0.0
    assert baseline_row["evaluation_delta_vs_baseline"]["standard_error"] == 0.0
    assert batched["search_config"]["batch_child_selectors"] is True
    assert batched["continuation_policy"]["child_selector_execution"] == (
        "batched_infoset_locked_v1"
    )
