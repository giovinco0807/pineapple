from __future__ import annotations

import copy
import json

import pytest

import ofc_regular.hu_m43_attempt10_teacher as teacher
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m4_t1_teacher import _ActionScores
from ofc_regular.state import Board


MODEL_HASH = teacher.ATTEMPT10_FROZEN_MODEL_SHA256


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
    model_id = teacher.ATTEMPT10_FROZEN_MODEL_ID

    def __init__(self, *, p95=None, p99=None, maximum=None) -> None:
        self.p95 = p95
        self.p99 = p99
        self.maximum = maximum

    def score_actions(self, observation, actions, *, baseline_index):
        assert observation == _root()
        assert 0 <= baseline_index < len(actions)
        size = len(actions)
        return teacher.Attempt10RankScores(
            rank_mean=tuple(float(size - index) for index in range(size)),
            rank_disagreement=tuple(0.25 for _ in actions),
            raw_downside_p95=tuple(self.p95 or (10.0 + index for index in range(size))),
            raw_downside_p99=tuple(self.p99 or (20.0 + index for index in range(size))),
            raw_downside_max=tuple(self.maximum or (30.0 + index for index in range(size))),
        )


def _config(**overrides) -> teacher.Attempt10TeacherConfig:
    values = {
        "frozen_model_sha256": MODEL_HASH,
        "hand_seed": 1001,
        "rerank_seed": 1002,
        "veto_seed": 1003,
        "stress_seed": 1004,
        "confirmation_seed": 1005,
        "evaluation_seed": 1006,
        "child_policy_seed": 1007,
        "run_id": "attempt10-test",
    }
    values.update(overrides)
    return teacher.Attempt10TeacherConfig(**values)


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
            events.append((phase, len(actions), count, batch))
        if phase == "rerank_r128":
            assert (len(actions), count) == (13, 128)
            spec = list(range(12, -1, -1))
        elif phase == "veto_v256":
            assert (len(actions), count) == (9, 256)
            spec = veto if veto is not None else list(range(8, -1, -1))
        elif phase == "stress_x1024":
            assert count == 1024 and actions
            spec = stress if stress is not None else list(range(len(actions) - 1, -1, -1))
        elif phase == "confirmation_c512":
            assert count == 512 and actions
            spec = (
                confirmation
                if confirmation is not None
                else list(range(len(actions) - 1, -1, -1))
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
    return teacher.evaluate_attempt10_t1_second(
        _root(),
        baseline_action_key=_baseline_token(),
        ranker=ranker or _Ranker(),
        t2_policies=_policies(),
        config=config or _config(),
    )


def _one_tail_row(count: int, *, mean_value: float, tail: float) -> _ActionScores:
    return _ActionScores(tuple([mean_value] * (count - 1) + [tail]))


def test_config_locks_attempt10_architecture_and_seven_seed_domains() -> None:
    config = _config()
    assert (
        config.candidate_top_k,
        config.rerank_samples,
        config.k8_size,
        config.veto_samples,
        config.stress_samples,
        config.confirmation_samples,
        config.evaluation_samples,
    ) == (12, 128, 8, 256, 1024, 512, 256)
    assert len(teacher.ATTEMPT10_RNG_DOMAINS) == 7
    with pytest.raises(ValueError, match="fixed at 1024"):
        _config(stress_samples=512)
    with pytest.raises(ValueError, match="all be distinct"):
        _config(confirmation_seed=1004)
    with pytest.raises(ValueError, match="stage9f_p2"):
        _config(t2_policy_id="current")


def test_top12_k8_and_all_survivor_phase_order_are_frozen(monkeypatch) -> None:
    result = _evaluate(monkeypatch)
    assert len(result["learned_top12_action_keys"]) == 12
    assert result["proposal_mapping"]["action_count"] == 13
    assert result["k8"]["action_count"] == 8
    assert len(result["k8"]["top4_rerank_positions"]) == 4
    assert len(result["k8"]["risk_reserve_rerank_positions"]) == 4
    assert set(result["k8"]["top4_rerank_positions"]).isdisjoint(
        result["k8"]["risk_reserve_rerank_positions"]
    )
    assert result["veto"]["retained_action_keys"] == result["k8"]["action_keys"]
    assert result["stress"]["action_keys"][:-1] == result["veto"]["retained_action_keys"]
    assert result["confirmation"]["action_keys"][:-1] == result["stress"]["retained_action_keys"]


def test_v_is_coarse_but_x_and_c_apply_all_strict_checks(monkeypatch) -> None:
    def veto(_actions, count):
        rows = [_constant_rows([2.0], count)[0] for _ in range(8)]
        rows[0] = _one_tail_row(count, mean_value=2.0, tail=-49.0)
        return (*rows, _constant_rows([0.0], count)[0])

    def stress(actions, count):
        # The first V survivor fails strict max >= -45; every later one survives.
        rows = [_constant_rows([2.0], count)[0] for _ in actions[:-1]]
        rows[0] = _one_tail_row(count, mean_value=2.0, tail=-49.0)
        return (*rows, _constant_rows([0.0], count)[0])

    result = _evaluate(monkeypatch, scorer=_scorer(veto=veto, stress=stress))
    assert result["veto"]["retained_traversal_positions"] == list(range(8))
    assert result["stress"]["retained_positions"] == list(range(1, 8))
    assert result["confirmation"]["action_count"] == 8
    assert result["decision"]["override_fired"] is True


@pytest.mark.parametrize(
    ("scorer", "reason", "opened"),
    [
        (
            _scorer(veto=[-1] * 8 + [0]),
            "no_v256_candidate_passed",
            ["rerank_r128", "veto_v256"],
        ),
        (
            _scorer(
                stress=lambda actions, count: _constant_rows(
                    [-60] * (len(actions) - 1) + [0], count
                )
            ),
            "no_x1024_candidate_passed",
            ["rerank_r128", "veto_v256", "stress_x1024"],
        ),
        (
            _scorer(
                confirmation=lambda actions, count: _constant_rows(
                    [-60] * (len(actions) - 1) + [0], count
                )
            ),
            "no_c512_candidate_passed",
            [
                "rerank_r128",
                "veto_v256",
                "stress_x1024",
                "confirmation_c512",
            ],
        ),
    ],
)
def test_exact_baseline_fallback_at_each_boundary(monkeypatch, scorer, reason, opened) -> None:
    result = _evaluate(monkeypatch, scorer=scorer)
    assert result["decision"]["final_selected_action_key"] == result["baseline_action_key"]
    assert result["decision"]["exact_baseline_fallback"] is True
    assert result["decision"]["fallback_reason"] == reason
    assert list(result["rng_key_digests"]) == opened
    assert result["evaluation"]["opened"] is False


def test_c512_locks_lowest_observed_tail_risk_before_mean(monkeypatch) -> None:
    def confirmation(actions, count):
        rows = [_constant_rows([1.0], count)[0] for _ in actions[:-1]]
        # Candidate zero has much higher mean but a nonzero observed tail risk.
        rows[0] = _one_tail_row(count, mean_value=10.0, tail=-40.0)
        # Candidate one has zero tail risk and therefore wins before mean tie-break.
        rows[1] = _constant_rows([2.0], count)[0]
        return (*rows, _constant_rows([0.0], count)[0])

    result = _evaluate(monkeypatch, scorer=_scorer(confirmation=confirmation))
    assert result["confirmation"]["selected_position"] == 1
    assert result["decision"]["final_selected_action_key"] == result["confirmation"][
        "action_keys"
    ][1]
    risk = result["confirmation"]["normalized_tail_risk_by_position"]
    assert risk["1"]["score"] < risk["0"]["score"]


def test_e256_is_independent_diagnostic_and_cannot_change_lock(monkeypatch) -> None:
    result = _evaluate(monkeypatch, scorer=_scorer(evaluation=[-1000, 0]))
    selected = result["confirmation"]["selected_action_key"]
    assert result["decision"]["final_selected_action_key"] == selected
    assert result["evaluation"]["locked_final_action_key"] == selected
    assert result["evaluation"]["sample_best_action_key"] == result["baseline_action_key"]
    assert result["evaluation"]["diagnostics_only"] is True
    assert result["evaluation"]["can_rerank_or_gate"] is False


def test_crf_rng_mapping_hidden_info_and_validator_tamper_guards(monkeypatch) -> None:
    events = []
    result = _evaluate(monkeypatch, scorer=_scorer(events=events))
    assert [(phase, actions, samples) for phase, actions, samples, _ in events] == [
        ("rerank_r128", 13, 128),
        ("veto_v256", 9, 256),
        ("stress_x1024", 9, 1024),
        ("confirmation_c512", 9, 512),
        ("evaluation_e256", 2, 256),
    ]
    key_sets = [set(values) for values in result["rng_key_digests"].values()]
    assert [len(values) for values in key_sets] == [128, 256, 1024, 512, 256]
    for left_index, left in enumerate(key_sets):
        for right in key_sets[left_index + 1 :]:
            assert left.isdisjoint(right)
    encoded = json.dumps(result, sort_keys=True)
    assert "opponent_private_discard" not in encoded
    assert '"particles"' not in encoded
    validated = teacher.validate_attempt10_teacher_output(
        _root(),
        baseline_action_key=_baseline_token(),
        payload=result,
        config=_config(),
    )
    assert validated["selected_action_key"] == result["decision"][
        "final_selected_action_key"
    ]

    tampered = copy.deepcopy(result)
    tampered["legal_action_mapping"]["action_keys"].reverse()
    with pytest.raises(ValueError, match="legal action mapping"):
        teacher.validate_attempt10_teacher_output(
            _root(),
            baseline_action_key=_baseline_token(),
            payload=tampered,
            config=_config(),
        )
    tampered = copy.deepcopy(result)
    tampered["decision"]["final_selected_action_key"] = result["baseline_action_key"]
    with pytest.raises(ValueError, match="locked final"):
        teacher.validate_attempt10_teacher_output(
            _root(),
            baseline_action_key=_baseline_token(),
            payload=tampered,
            config=_config(),
        )
    tampered = copy.deepcopy(result)
    tampered["confirmation"]["normalized_tail_risk_by_position"]["0"]["score"] += 1
    with pytest.raises(ValueError, match="risk-lock"):
        teacher.validate_attempt10_teacher_output(
            _root(),
            baseline_action_key=_baseline_token(),
            payload=tampered,
            config=_config(),
        )


def test_scalar_batch_parity_and_repeat_are_deterministic(monkeypatch) -> None:
    scorer = _scorer()
    monkeypatch.setattr(teacher, "_score_actions", scorer)
    monkeypatch.setattr(teacher, "_score_actions_batched", scorer)
    args = {
        "observation": _root(),
        "baseline_action_key": _baseline_token(),
        "ranker": _Ranker(),
        "t2_policies": _policies(),
    }
    scalar = teacher.evaluate_attempt10_t1_second(
        **args, config=_config(batch_child_selectors=False)
    )
    repeat = teacher.evaluate_attempt10_t1_second(
        **args, config=_config(batch_child_selectors=False)
    )
    batch = teacher.evaluate_attempt10_t1_second(
        **args, config=_config(batch_child_selectors=True)
    )
    assert scalar == repeat
    for key in (
        "legal_action_mapping",
        "rerank",
        "k8",
        "veto",
        "stress",
        "confirmation",
        "decision",
        "evaluation",
        "belief_digests",
        "rng_key_digests",
    ):
        assert scalar[key] == batch[key]
