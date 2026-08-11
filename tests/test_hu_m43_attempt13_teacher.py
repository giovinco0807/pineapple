from __future__ import annotations

import copy
import json
from typing import Callable

import pytest

import ofc_regular.hu_m43_attempt13_teacher as teacher
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m4_t1_teacher import _ActionScores
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
    artifact_sha256 = teacher.ATTEMPT13_FROZEN_MODEL_SHA256
    model_id = teacher.ATTEMPT13_FROZEN_MODEL_ID

    def score_actions(self, observation, actions, *, baseline_index):
        del observation, baseline_index
        size = len(actions)
        return teacher.Attempt13RankScores(
            rank_mean=tuple(float(size - index) for index in range(size)),
            rank_disagreement=tuple(0.25 for _ in actions),
            raw_downside_p95=tuple(10.0 + index for index in range(size)),
            raw_downside_p99=tuple(20.0 + index for index in range(size)),
            raw_downside_max=tuple(30.0 + index for index in range(size)),
        )


def _config(*, batch: bool = False, **overrides) -> teacher.Attempt13TeacherConfig:
    values = {
        "frozen_model_sha256": teacher.ATTEMPT13_FROZEN_MODEL_SHA256,
        "hand_seed": 2001,
        "rerank_seed": 2002,
        "veto_seed": 2003,
        "stress_seed": 2004,
        "confirmation_seed": 2005,
        "evaluation_seed": 2006,
        "child_policy_seed": 2007,
        "run_id": "attempt13-test",
        "batch_child_selectors": batch,
    }
    values.update(overrides)
    return teacher.Attempt13TeacherConfig(**values)


def _limited_actions(n: int):
    actions = tuple(generate_turn_actions(_root().hero_board, _root().dealt_cards))
    baseline = actions[-1]
    nonbaseline = [action for action in actions[:-1] if action != baseline]
    assert len(nonbaseline) >= n
    return tuple([*nonbaseline[:n], baseline])


def _constant_scores(_observation, actions, batch, _selector):
    phase = batch.run_id.rsplit(":", 1)[-1]
    output = []
    for position in range(len(actions)):
        if position == len(actions) - 1:
            value = 0.0
        elif phase == "evaluation_e512":
            value = -10.0
        else:
            value = float(len(actions) - position)
        output.append(_ActionScores((value,) * len(batch.particles)))
    return tuple(output)


def _retained_no_fire_scores(_observation, actions, batch, _selector):
    phase = batch.run_id.rsplit(":", 1)[-1]
    output = []
    for position in range(len(actions)):
        if position == len(actions) - 1:
            value = 0.0
        elif phase in {"stress_x1024", "confirmation_c1024"}:
            value = -1.0
        else:
            value = 1.0
        output.append(_ActionScores((value,) * len(batch.particles)))
    return tuple(output)


def _evaluate(
    monkeypatch: pytest.MonkeyPatch,
    n: int,
    *,
    scorer: Callable = _constant_scores,
    batch: bool = False,
):
    actions = _limited_actions(n)
    monkeypatch.setattr(
        teacher._attempt12, "generate_turn_actions", lambda *_: actions
    )
    monkeypatch.setattr(teacher._attempt12, "_score_actions", scorer)
    monkeypatch.setattr(teacher._attempt12, "_score_actions_batched", scorer)
    payload = teacher.evaluate_attempt13_t1_second(
        _root(),
        baseline_action_key=action_key(actions[-1]).to_token(),
        ranker=_Ranker(),
        t2_policies=_policies(),
        config=_config(batch=batch),
    )
    return actions, payload


def test_attempt13_configuration_is_hard_locked() -> None:
    assert _config().veto_risk_cap == 1.05
    with pytest.raises(ValueError, match="veto_risk_cap is fixed"):
        _config(veto_risk_cap=1.10)
    with pytest.raises(ValueError, match="seeds must all be distinct"):
        _config(evaluation_seed=2005)


def test_attempt13_alllegal_fresh_schema_and_locked_e512(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions, payload = _evaluate(monkeypatch, 10)
    baseline = action_key(actions[-1]).to_token()

    assert payload["schema"] == teacher.ATTEMPT13_TEACHER_SCHEMA
    assert payload["solver_id"] == teacher.ATTEMPT13_SOLVER_ID
    assert payload["candidate_nonbaseline_count"] == 10
    assert payload["rerank"]["action_count"] == 11
    assert payload["shortlist"]["nonbaseline_count"] == 8
    assert payload["veto"]["thresholds"] == teacher._veto_thresholds()
    assert payload["stress"]["action_keys"] == payload["confirmation"]["action_keys"]
    assert payload["stress"]["action_keys"] == payload["pooled"]["action_keys"]
    assert payload["decision"]["override_fired"] is True
    selected = payload["decision"]["final_selected_action_key"]
    assert selected != baseline
    assert payload["evaluation"]["locked_final_action_key"] == selected
    assert payload["evaluation"]["sample_best_action_key"] == baseline
    assert payload["evaluation"]["can_rerank_or_gate"] is False
    validation = teacher.validate_attempt13_teacher_output(
        _root(),
        baseline_action_key=baseline,
        payload=payload,
        config=_config(),
    )
    assert validation["schema"] == teacher.ATTEMPT13_VALIDATION_SCHEMA
    assert validation["opened_phases"] == [
        "rerank_r128",
        "veto_v256",
        "stress_x1024",
        "confirmation_c1024",
        "evaluation_e512",
    ]


def _veto_boundary_scores(_observation, actions, batch, _selector):
    phase = batch.run_id.rsplit(":", 1)[-1]
    count = len(batch.particles)
    output = []
    for position in range(len(actions)):
        if position == len(actions) - 1:
            raw = (0.0,) * count
        elif phase == "veto_v256":
            loss = -26.0 if position == 0 else -27.0
            raw = (loss,) * 14 + (4.0,) * (count - 14)
        elif phase == "evaluation_e512":
            raw = (-5.0,) * count
        else:
            raw = (2.0,) * count
        output.append(_ActionScores(tuple(raw)))
    return tuple(output)


def test_attempt13_v256_uses_normalized_risk105_not_raw_min(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, payload = _evaluate(monkeypatch, 2, scorer=_veto_boundary_scores)
    risks = payload["veto"]["normalized_tail_risk_by_traversal_position"]

    assert risks["0"]["score"] == pytest.approx(26.0 / 25.0)
    assert risks["1"]["score"] == pytest.approx(27.0 / 25.0)
    assert payload["veto"]["checks_by_traversal_position"] == [
        {"mean_gt_0": True, "normalized_p95_p99_risk_at_most": True},
        {"mean_gt_0": True, "normalized_p95_p99_risk_at_most": False},
    ]
    assert payload["veto"]["retained_traversal_positions"] == [0]
    assert payload["veto"]["raw_min_can_filter_or_rank"] is False


def _tail_gate_scores(_observation, actions, batch, _selector):
    phase = batch.run_id.rsplit(":", 1)[-1]
    count = len(batch.particles)
    output = []
    for position in range(len(actions)):
        if position == len(actions) - 1:
            raw = (0.0,) * count
        elif phase == "stress_x1024":
            if position == 1:
                raw = (-60.0,) * 2 + (2.0,) * (count - 2)
            elif position == 2:
                raw = (-100.0,) + (-40.0,) * 10 + (2.0,) * (count - 11)
            elif position == 3:
                raw = (-100.0,) + (2.0,) * (count - 1)
            else:
                raw = (1.0,) * count
        elif phase == "confirmation_c1024":
            if position == 1:
                raw = (-60.0,) + (2.0,) * (count - 1)
            elif position == 2:
                raw = (-100.0,) + (-40.0,) * 9 + (2.0,) * (count - 10)
            else:
                raw = (2.0,) * count
        elif phase == "evaluation_e512":
            raw = (-10.0,) * count
        else:
            raw = (float(len(actions) - position),) * count
        output.append(_ActionScores(tuple(raw)))
    return tuple(output)


def test_attempt13_p2048_q001_es01_and_raw_min_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, payload = _evaluate(monkeypatch, 4, scorer=_tail_gate_scores)
    pooled = payload["pooled"]

    assert pooled["checks_by_position"][0] == {
        "mean_gt_0": True,
        "normalized_p95_p99_risk_at_most": True,
        "q001_at_least": True,
        "es01_at_least": True,
    }
    assert pooled["checks_by_position"][1]["q001_at_least"] is False
    assert pooled["checks_by_position"][1]["es01_at_least"] is True
    assert pooled["checks_by_position"][2]["q001_at_least"] is True
    assert pooled["checks_by_position"][2]["es01_at_least"] is False
    assert pooled["checks_by_position"][3] == pooled["checks_by_position"][0]
    tails = pooled["tail_metrics_by_position"]
    assert tails[2]["es01_included_count"] == 21
    assert tails[2]["es01"] == pytest.approx((-200.0 - 760.0) / 21.0)
    assert tails[3]["raw_min"] == -100.0
    assert tails[3]["raw_min_can_filter_rank_select_or_gate"] is False
    assert pooled["eligible_positions"] == [0, 3]
    assert pooled["selected_position"] == 0
    assert set(pooled["normalized_tail_risk_by_position"]["3"]["components"]) == {
        "loss95_over_25",
        "loss99_over_40",
        "loss_q001_over_50",
        "loss_es01_over_40",
    }


def test_attempt13_zero_alternatives_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, payload = _evaluate(monkeypatch, 0)
    assert payload["stress"]["opened"] is False
    assert payload["confirmation"]["opened"] is False
    assert payload["pooled"]["opened"] is False
    assert payload["pooled"]["thresholds"] == teacher._pooled_thresholds()
    assert payload["decision"]["fallback_reason"] == "no_v256_candidate_passed"
    assert payload["evaluation"]["opened"] is False
    assert list(payload["rng_key_digests"]) == ["rerank_r128", "veto_v256"]


def test_attempt13_validator_rejects_hidden_tail_mapping_rng_and_e_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions, payload = _evaluate(monkeypatch, 4)
    baseline = action_key(actions[-1]).to_token()

    hidden = copy.deepcopy(payload)
    hidden["opponent_private_discard"] = ["As"]
    with pytest.raises(ValueError, match="hidden opponent"):
        teacher.validate_attempt13_teacher_output(
            _root(), baseline_action_key=baseline, payload=hidden, config=_config()
        )

    tail = copy.deepcopy(payload)
    tail["pooled"]["tail_metrics_by_position"][0]["q001"] -= 1.0
    with pytest.raises(ValueError, match="four-tail-risk"):
        teacher.validate_attempt13_teacher_output(
            _root(), baseline_action_key=baseline, payload=tail, config=_config()
        )

    mapping = copy.deepcopy(payload)
    mapping["confirmation"]["action_keys"][0] = mapping["confirmation"][
        "action_keys"
    ][1]
    with pytest.raises(ValueError, match="inherited structural contract"):
        teacher.validate_attempt13_teacher_output(
            _root(), baseline_action_key=baseline, payload=mapping, config=_config()
        )

    rng = copy.deepcopy(payload)
    rng["rng_key_digests"]["confirmation_c1024"] = list(
        rng["rng_key_digests"]["stress_x1024"]
    )
    with pytest.raises(ValueError, match="RNG namespaces overlap"):
        teacher.validate_attempt13_teacher_output(
            _root(), baseline_action_key=baseline, payload=rng, config=_config()
        )

    evaluation = copy.deepcopy(payload)
    evaluation["evaluation"]["locked_final_action_key"] = baseline
    with pytest.raises(ValueError, match="E512 diagnostic lock"):
        teacher.validate_attempt13_teacher_output(
            _root(), baseline_action_key=baseline, payload=evaluation, config=_config()
        )


def test_attempt13_scalar_batch_parity_and_no_hidden_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, scalar = _evaluate(monkeypatch, 3, batch=False)
    _, batch = _evaluate(monkeypatch, 3, batch=True)
    normalized = copy.deepcopy(batch)
    normalized["search_config"]["batch_child_selectors"] = False

    assert normalized == scalar
    encoded = json.dumps(scalar, sort_keys=True)
    assert "opponent_private_discard" not in encoded
    assert '"particles"' not in encoded


@pytest.mark.parametrize(
    ("candidate_count", "scorer"),
    (
        (0, _constant_scores),
        (3, _retained_no_fire_scores),
        (3, _constant_scores),
    ),
)
def test_attempt13_validator_accepts_canonical_json_phase_order(
    monkeypatch: pytest.MonkeyPatch,
    candidate_count: int,
    scorer: Callable,
) -> None:
    actions, payload = _evaluate(monkeypatch, candidate_count, scorer=scorer)
    baseline = action_key(actions[-1]).to_token()
    decoded = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))

    expected_phases = ["rerank_r128", "veto_v256"]
    if decoded["stress"]["opened"]:
        expected_phases.extend(["stress_x1024", "confirmation_c1024"])
    if decoded["decision"]["override_fired"]:
        expected_phases.append("evaluation_e512")
    for field in (
        "rng_key_digests",
        "belief_digests",
        "phase_child_information_set_counts",
    ):
        assert list(decoded[field]) == sorted(expected_phases)

    validation = teacher.validate_attempt13_teacher_output(
        _root(),
        baseline_action_key=baseline,
        payload=decoded,
        config=_config(),
    )
    assert validation["opened_phases"] == expected_phases


def test_attempt13_validator_rejects_noncanonical_phase_permutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions, payload = _evaluate(monkeypatch, 3)
    baseline = action_key(actions[-1]).to_token()
    keys = list(payload["rng_key_digests"])
    assert len(keys) >= 4
    arbitrary = [keys[1], keys[0], *keys[2:]]
    payload["rng_key_digests"] = {
        name: payload["rng_key_digests"][name] for name in arbitrary
    }

    with pytest.raises(ValueError, match="opened RNG phase order changed"):
        teacher.validate_attempt13_teacher_output(
            _root(),
            baseline_action_key=baseline,
            payload=payload,
            config=_config(),
        )
