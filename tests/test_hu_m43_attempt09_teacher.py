from __future__ import annotations

import copy
import json

import pytest

import ofc_regular.hu_m43_attempt09_teacher as teacher
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m4_t1_teacher import _ActionScores
from ofc_regular.state import Board


MODEL_HASH = teacher.ATTEMPT09_FROZEN_MODEL_SHA256


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


class _Stage9fPolicy:
    def __init__(self, seat: str) -> None:
        self.seat = seat
        self.topk_context = {
            "runtime_profile": "stage9f_p2",
            "runtime_status": "p2_fixed",
        }


def _policies() -> dict[str, object]:
    return {
        "first": _Stage9fPolicy("first"),
        "second": _Stage9fPolicy("second"),
    }


class _Ranker:
    artifact_sha256 = MODEL_HASH
    model_id = teacher.ATTEMPT09_FROZEN_MODEL_ID

    def __init__(self, *, p95=None, p99=None, maximum=None) -> None:
        self.p95 = p95
        self.p99 = p99
        self.maximum = maximum

    def score_actions(self, observation, actions, *, baseline_index):
        assert observation == _root()
        assert 0 <= baseline_index < len(actions)
        return teacher.Attempt09RankScores(
            rank_mean=tuple(float(len(actions) - index) for index in range(len(actions))),
            rank_disagreement=tuple(0.25 for _ in actions),
            raw_downside_p95=tuple(self.p95 or (10.0 for _ in actions)),
            raw_downside_p99=tuple(self.p99 or (20.0 for _ in actions)),
            raw_downside_max=tuple(self.maximum or (30.0 for _ in actions)),
        )


def _config(**overrides) -> teacher.Attempt09TeacherConfig:
    values = {
        "frozen_model_sha256": MODEL_HASH,
        "hand_seed": 901,
        "rerank_seed": 902,
        "veto_seed": 903,
        "stress_seed": 904,
        "confirmation_seed": 905,
        "evaluation_seed": 906,
        "child_policy_seed": 907,
        "run_id": "attempt09-test",
    }
    values.update(overrides)
    return teacher.Attempt09TeacherConfig(**values)


def _baseline_token() -> str:
    actions = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    return action_key(actions[-1]).to_token()


def _constant_rows(means, sample_count: int) -> tuple[_ActionScores, ...]:
    return tuple(
        _ActionScores(tuple(float(mean) for _ in range(sample_count)))
        for mean in means
    )


def _scorer(*, veto=None, stress=None, confirmation=None, evaluation=None, events=None):
    def score(_observation, actions, batch, _selector):
        phase = batch.run_id.rsplit(":", 1)[-1]
        count = len(batch.particles)
        if events is not None:
            # Retain each object so CPython cannot recycle an id after a phase.
            events.append((phase, len(actions), count, batch))
        if phase == "rerank_r128":
            assert (len(actions), count) == (9, 128)
            spec = list(range(8, -1, -1))
        elif phase == "veto_v256":
            assert (len(actions), count) == (5, 256)
            spec = veto if veto is not None else [4, 3, 2, 1, 0]
        elif phase == "stress_x512":
            assert count == 512 and actions
            spec = stress if stress is not None else [1] * (len(actions) - 1) + [0]
        elif phase == "confirmation_c256":
            assert count == 256 and actions
            spec = (
                confirmation
                if confirmation is not None
                else [1] * (len(actions) - 1) + [0]
            )
        elif phase == "evaluation_e256":
            assert (len(actions), count) == (2, 256)
            spec = evaluation if evaluation is not None else [1, 0]
        else:  # pragma: no cover
            raise AssertionError(phase)
        rows = spec(actions, count) if callable(spec) else _constant_rows(spec, count)
        assert len(rows) == len(actions)
        return rows

    return score


def _evaluate(monkeypatch, *, scorer=None, ranker=None, config=None):
    monkeypatch.setattr(teacher, "_score_actions", scorer or _scorer())
    return teacher.evaluate_attempt09_t1_second(
        _root(),
        baseline_action_key=_baseline_token(),
        ranker=ranker or _Ranker(),
        t2_policies=_policies(),
        config=config or _config(),
    )


def test_config_locks_search_and_seven_seed_domains() -> None:
    config = _config()
    assert (
        config.rerank_samples,
        config.veto_samples,
        config.stress_samples,
        config.confirmation_samples,
        config.evaluation_samples,
    ) == (128, 256, 512, 256, 256)
    assert len(teacher.ATTEMPT09_RNG_DOMAINS) == 7
    with pytest.raises(ValueError, match="fixed at 512"):
        _config(stress_samples=1024)
    with pytest.raises(ValueError, match="all be distinct"):
        _config(confirmation_seed=904)
    with pytest.raises(ValueError, match="stage9f_p2"):
        _config(t2_policy_id="current")


def test_v_keeps_all_safe_candidates_and_x_c_preserve_frozen_r_order(monkeypatch) -> None:
    result = _evaluate(monkeypatch)
    assert result["veto"]["retained_traversal_positions"] == [0, 1, 2, 3]
    assert result["veto"]["retained_action_keys"] == result["k4"]["action_keys"]
    assert result["stress"]["action_keys"][:-1] == result["veto"]["retained_action_keys"]
    assert result["stress"]["retained_positions"] == [0, 1, 2, 3]
    assert (
        result["confirmation"]["action_keys"][:-1]
        == result["stress"]["retained_action_keys"]
    )
    assert result["confirmation"]["retained_positions"] == [0, 1, 2, 3]
    assert (
        result["confirmation"]["retained_action_keys"]
        == result["confirmation"]["action_keys"][:-1]
    )
    assert result["confirmation"]["first_passing_position"] == 0
    assert result["decision"]["final_selected_action_key"] == result["k4"]["action_keys"][0]


def test_later_candidate_can_survive_x_and_c_after_earlier_candidates_fail(monkeypatch) -> None:
    def stress(_actions, count):
        return (
            _ActionScores(tuple([1.0] * (count - 1) + [-60.0])),
            _ActionScores(tuple([2.0] * count)),
            _ActionScores(tuple([3.0] * count)),
            _ActionScores(tuple([4.0] * (count - 1) + [-60.0])),
            _ActionScores(tuple([0.0] * count)),
        )

    def confirmation(_actions, count):
        # X retained original candidates one and two. C rejects one and keeps two.
        return (
            _ActionScores(tuple([2.0] * (count - 1) + [-60.0])),
            _ActionScores(tuple([3.0] * count)),
            _ActionScores(tuple([0.0] * count)),
        )

    result = _evaluate(
        monkeypatch, scorer=_scorer(stress=stress, confirmation=confirmation)
    )
    assert result["stress"]["retained_positions"] == [1, 2]
    assert result["confirmation"]["first_passing_position"] == 1
    assert result["decision"]["final_selected_action_key"] == result["k4"]["action_keys"][2]
    assert result["decision"]["candidate_fallback_across_V_X_C_allowed"] is True


@pytest.mark.parametrize(
    ("scorer", "reason", "opened"),
    [
        (_scorer(veto=[-1, -2, -3, -4, 0]), "no_v256_candidate_passed", ["rerank_r128", "veto_v256"]),
        (
            _scorer(stress=lambda actions, count: _constant_rows([-60] * (len(actions) - 1) + [0], count)),
            "no_x512_candidate_passed",
            ["rerank_r128", "veto_v256", "stress_x512"],
        ),
        (
            _scorer(confirmation=lambda actions, count: _constant_rows([-60] * (len(actions) - 1) + [0], count)),
            "no_c256_candidate_passed",
            ["rerank_r128", "veto_v256", "stress_x512", "confirmation_c256"],
        ),
    ],
)
def test_exact_baseline_fallback_at_each_candidate_boundary(
    monkeypatch, scorer, reason, opened
) -> None:
    result = _evaluate(monkeypatch, scorer=scorer)
    assert result["decision"]["final_selected_action_key"] == result["baseline_action_key"]
    assert result["decision"]["exact_baseline_fallback"] is True
    assert result["decision"]["fallback_reason"] == reason
    assert list(result["rng_key_digests"]) == opened
    assert result["evaluation"]["opened"] is False


def test_e256_is_independent_diagnostic_and_cannot_change_final(monkeypatch) -> None:
    result = _evaluate(monkeypatch, scorer=_scorer(evaluation=[-1000, 0]))
    selected = result["confirmation"]["first_passing_action_key"]
    assert result["decision"]["final_selected_action_key"] == selected
    assert result["evaluation"]["locked_final_action_key"] == selected
    assert result["evaluation"]["sample_best_action_key"] == result["baseline_action_key"]
    assert result["evaluation"]["diagnostics_only"] is True
    assert result["evaluation"]["can_rerank_or_gate"] is False
    assert result["decision"]["override_fired"] is True


def test_crf_scopes_phase_independence_mapping_and_hidden_info(monkeypatch) -> None:
    events = []
    result = _evaluate(monkeypatch, scorer=_scorer(events=events))
    assert [(name, actions, samples) for name, actions, samples, _ in events] == [
        ("rerank_r128", 9, 128),
        ("veto_v256", 5, 256),
        ("stress_x512", 5, 512),
        ("confirmation_c256", 5, 256),
        ("evaluation_e256", 2, 256),
    ]
    assert len({id(batch) for *_, batch in events}) == len(events)
    key_sets = [set(values) for values in result["rng_key_digests"].values()]
    assert [len(values) for values in key_sets] == [128, 256, 512, 256, 256]
    for left_index, left in enumerate(key_sets):
        for right in key_sets[left_index + 1 :]:
            assert left.isdisjoint(right)
    legal = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    assert result["legal_action_mapping"]["action_keys"] == [
        action_key(action).to_token() for action in legal
    ]
    encoded = json.dumps(result, sort_keys=True)
    assert "opponent_private_discard" not in encoded
    assert '"particles"' not in encoded


def test_public_validator_rejects_action_and_final_lock_tampering(monkeypatch) -> None:
    result = _evaluate(monkeypatch)
    validated = teacher.validate_attempt09_teacher_output(
        _root(),
        baseline_action_key=_baseline_token(),
        payload=result,
        config=_config(),
    )
    assert validated["selected_action_key"] == result["decision"]["final_selected_action_key"]
    tampered = copy.deepcopy(result)
    tampered["legal_action_mapping"]["action_keys"].reverse()
    with pytest.raises(ValueError, match="legal action mapping"):
        teacher.validate_attempt09_teacher_output(
            _root(), baseline_action_key=_baseline_token(), payload=tampered, config=_config()
        )
    tampered = copy.deepcopy(result)
    tampered["decision"]["final_selected_action_key"] = result["baseline_action_key"]
    with pytest.raises(ValueError, match="locked final"):
        teacher.validate_attempt09_teacher_output(
            _root(), baseline_action_key=_baseline_token(), payload=tampered, config=_config()
        )
    tampered = copy.deepcopy(result)
    tampered["stress"]["action_keys"][0:2] = reversed(
        tampered["stress"]["action_keys"][0:2]
    )
    with pytest.raises(ValueError, match="subset/order"):
        teacher.validate_attempt09_teacher_output(
            _root(), baseline_action_key=_baseline_token(), payload=tampered, config=_config()
        )
    tampered = copy.deepcopy(result)
    tampered["confirmation"]["retained_action_keys"].pop()
    with pytest.raises(ValueError, match="first-safe"):
        teacher.validate_attempt09_teacher_output(
            _root(), baseline_action_key=_baseline_token(), payload=tampered, config=_config()
        )


def test_scalar_batch_parity_and_repeated_output_are_deterministic(monkeypatch) -> None:
    scorer = _scorer()
    monkeypatch.setattr(teacher, "_score_actions", scorer)
    monkeypatch.setattr(teacher, "_score_actions_batched", scorer)
    args = {
        "observation": _root(),
        "baseline_action_key": _baseline_token(),
        "ranker": _Ranker(),
        "t2_policies": _policies(),
    }
    scalar = teacher.evaluate_attempt09_t1_second(
        **args, config=_config(batch_child_selectors=False)
    )
    repeat = teacher.evaluate_attempt09_t1_second(
        **args, config=_config(batch_child_selectors=False)
    )
    batch = teacher.evaluate_attempt09_t1_second(
        **args, config=_config(batch_child_selectors=True)
    )
    assert scalar == repeat
    for key in (
        "legal_action_mapping",
        "rerank",
        "k4",
        "veto",
        "stress",
        "confirmation",
        "decision",
        "evaluation",
        "belief_digests",
        "rng_key_digests",
    ):
        assert scalar[key] == batch[key]
