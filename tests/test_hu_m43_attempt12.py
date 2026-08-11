from __future__ import annotations

import copy
import json
from typing import Callable

import pytest

import ofc_regular.hu_m43_attempt12_teacher as teacher
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
    artifact_sha256 = teacher.ATTEMPT12_FROZEN_MODEL_SHA256
    model_id = teacher.ATTEMPT12_FROZEN_MODEL_ID

    def score_actions(self, observation, actions, *, baseline_index):
        del observation, baseline_index
        size = len(actions)
        return teacher.Attempt12RankScores(
            rank_mean=tuple(float(size - index) for index in range(size)),
            rank_disagreement=tuple(0.25 for _ in actions),
            raw_downside_p95=tuple(10.0 + index for index in range(size)),
            raw_downside_p99=tuple(20.0 + index for index in range(size)),
            raw_downside_max=tuple(30.0 + index for index in range(size)),
        )


def _config(*, batch: bool = False) -> teacher.Attempt12TeacherConfig:
    return teacher.Attempt12TeacherConfig(
        frozen_model_sha256=teacher.ATTEMPT12_FROZEN_MODEL_SHA256,
        hand_seed=1001,
        rerank_seed=1002,
        veto_seed=1003,
        stress_seed=1004,
        confirmation_seed=1005,
        evaluation_seed=1006,
        child_policy_seed=1007,
        run_id="attempt12-test",
        batch_child_selectors=batch,
    )


def _constant_scores(_observation, actions, batch, _selector):
    phase = batch.run_id.rsplit(":", 1)[-1]
    expected = {
        "rerank_r128": 128,
        "veto_v256": 256,
        "stress_x1024": 1024,
        "confirmation_c1024": 1024,
        "evaluation_e512": 512,
    }
    assert len(batch.particles) == expected[phase]
    means = list(range(len(actions) - 1, -1, -1))
    return tuple(
        _ActionScores(tuple(float(mean) for _ in batch.particles))
        for mean in means
    )


def _limited_actions(n: int):
    actions = tuple(generate_turn_actions(_root().hero_board, _root().dealt_cards))
    baseline = actions[-1]
    nonbaseline = [action for action in actions[:-1] if action != baseline]
    assert len(nonbaseline) >= n
    return tuple([*nonbaseline[:n], baseline])


def _evaluate(
    monkeypatch: pytest.MonkeyPatch,
    n: int,
    *,
    scorer: Callable = _constant_scores,
    batch: bool = False,
):
    actions = _limited_actions(n)
    monkeypatch.setattr(teacher, "generate_turn_actions", lambda *_: actions)
    monkeypatch.setattr(teacher, "_score_actions", scorer)
    monkeypatch.setattr(teacher, "_score_actions_batched", scorer)
    payload = teacher.evaluate_attempt12_t1_second(
        _root(),
        baseline_action_key=action_key(actions[-1]).to_token(),
        ranker=_Ranker(),
        t2_policies=_policies(),
        config=_config(batch=batch),
    )
    return actions, payload


@pytest.mark.parametrize("n", [0, 1, 5, 8, 11, 16, 23, 26])
def test_attempt12_r128_contains_every_unique_legal_nonbaseline_action(
    monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    _, payload = _evaluate(monkeypatch, n)
    baseline = payload["baseline_action_key"]
    proposal = payload["proposal_mapping"]["action_keys"]
    expected_nonbaseline = sorted(proposal[:-1])

    assert payload["candidate_nonbaseline_count"] == n
    assert payload["all_legal_nonbaseline_action_keys"] == expected_nonbaseline
    assert proposal == [*expected_nonbaseline, baseline]
    assert len(set(proposal)) == n + 1
    assert payload["rerank"]["action_count"] == n + 1
    assert payload["shortlist"]["nonbaseline_count"] == min(8, n)
    assert len(payload["shortlist"]["head_rerank_positions"]) == min(4, n)
    assert len(payload["shortlist"]["risk_reserve_rerank_positions"]) == (
        min(8, n) - min(4, n)
    )


def test_attempt12_zero_alternatives_is_exact_baseline_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, payload = _evaluate(monkeypatch, 0)
    assert payload["rerank"]["action_count"] == 1
    assert payload["veto"]["action_count"] == 1
    assert list(payload["rng_key_digests"]) == ["rerank_r128", "veto_v256"]
    assert payload["stress"]["opened"] is False
    assert payload["confirmation"]["opened"] is False
    assert payload["pooled"]["opened"] is False
    assert payload["evaluation"]["opened"] is False
    assert payload["decision"]["final_selected_action_key"] == payload[
        "baseline_action_key"
    ]
    assert payload["decision"]["fallback_reason"] == "no_v256_candidate_passed"


def _veto_extreme_min_scores(_observation, actions, batch, _selector):
    phase = batch.run_id.rsplit(":", 1)[-1]
    count = len(batch.particles)
    output = []
    for position in range(len(actions)):
        if position == len(actions) - 1:
            raw = (0.0,) * count
        elif phase == "veto_v256":
            raw = (-100.0, *((1.0,) * (count - 1)))
        else:
            raw = (1.0,) * count
        output.append(_ActionScores(tuple(raw)))
    return tuple(output)


def test_attempt12_v256_raw_min_is_attested_but_cannot_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, payload = _evaluate(monkeypatch, 1, scorer=_veto_extreme_min_scores)
    check = payload["veto"]["checks_by_traversal_position"][0]
    diagnostic = payload["veto"]["raw_min_diagnostics_by_traversal_position"][0]

    assert check == {"mean_gt_0": True, "p05_at_least": True, "p01_at_least": True}
    assert diagnostic["raw_min"] == -100.0
    assert diagnostic["at_least_reference"] is False
    assert diagnostic["can_filter_or_rank"] is False
    assert payload["veto"]["raw_min_can_filter_or_rank"] is False
    assert payload["veto"]["retained_action_keys"]
    assert payload["decision"]["override_fired"] is True


def test_attempt12_x_and_c_receive_identical_v_scope_and_never_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, payload = _evaluate(monkeypatch, 8)
    veto_survivors = payload["veto"]["retained_action_keys"]
    expected = [*veto_survivors, payload["baseline_action_key"]]

    assert payload["stress"]["action_keys"] == expected
    assert payload["confirmation"]["action_keys"] == expected
    assert payload["pooled"]["action_keys"] == expected
    for phase in (payload["stress"], payload["confirmation"]):
        assert phase["filtering_allowed"] is False
        assert phase["reranking_allowed"] is False
        assert phase["selection_allowed"] is False
        assert phase["retained_action_keys"] == veto_survivors
    assert set(payload["rng_key_digests"]["stress_x1024"]).isdisjoint(
        payload["rng_key_digests"]["confirmation_c1024"]
    )


def _pooled_only_scores(_observation, actions, batch, _selector):
    phase = batch.run_id.rsplit(":", 1)[-1]
    count = len(batch.particles)
    output = []
    for position in range(len(actions)):
        if position == len(actions) - 1:
            mean = 0.0
        elif phase == "stress_x1024":
            mean = 2.0
        elif phase == "confirmation_c1024":
            mean = -1.0
        elif phase == "evaluation_e512":
            mean = -10.0
        else:
            mean = 1.0
        output.append(_ActionScores((mean,) * count))
    return tuple(output)


def test_attempt12_only_pooled_x_plus_c_can_decide_and_e512_cannot_change_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, payload = _evaluate(monkeypatch, 1, scorer=_pooled_only_scores)
    selected = payload["decision"]["final_selected_action_key"]

    assert payload["confirmation"]["actions"][0]["paired_delta_vs_baseline"][
        "mean"
    ] == -1.0
    assert payload["pooled"]["actions"][0]["paired_delta_vs_baseline"]["count"] == 2048
    assert payload["pooled"]["actions"][0]["paired_delta_vs_baseline"]["mean"] == 0.5
    assert payload["pooled"]["selected_action_key"] == selected
    assert set(
        payload["pooled"]["normalized_tail_risk_by_position"]["0"][
            "components"
        ]
    ) == {"loss95_over_22", "loss99_over_36"}
    assert payload["decision"]["override_fired"] is True
    assert payload["evaluation"]["locked_final_action_key"] == selected
    assert payload["evaluation"]["sample_best_action_key"] == payload[
        "baseline_action_key"
    ]
    assert payload["evaluation"]["can_rerank_or_gate"] is False


def _pooled_extreme_min_scores(_observation, actions, batch, _selector):
    phase = batch.run_id.rsplit(":", 1)[-1]
    count = len(batch.particles)
    output = []
    for position in range(len(actions)):
        if position == len(actions) - 1:
            raw = (0.0,) * count
        elif phase == "stress_x1024":
            raw = (-100.0, *((1.0,) * (count - 1)))
        else:
            raw = (1.0,) * count
        output.append(_ActionScores(tuple(raw)))
    return tuple(output)


def test_attempt12_pooled_raw_min_is_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, payload = _evaluate(monkeypatch, 1, scorer=_pooled_extreme_min_scores)
    diagnostic = payload["pooled"]["raw_min_diagnostics_by_position"][0]
    assert diagnostic["raw_min"] == -100.0
    assert diagnostic["at_least_reference"] is False
    assert payload["pooled"]["raw_min_can_filter_or_rank"] is False
    assert payload["pooled"]["eligible_positions"] == [0]
    assert payload["decision"]["override_fired"] is True


def test_attempt12_validator_rejects_scope_pool_mapping_rng_and_hidden_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions, payload = _evaluate(monkeypatch, 3)
    baseline = action_key(actions[-1]).to_token()

    assert teacher.validate_attempt12_teacher_output(
        _root(), baseline_action_key=baseline, payload=payload, config=_config()
    )["selected_action_key"] == payload["decision"]["final_selected_action_key"]

    hidden = copy.deepcopy(payload)
    hidden["opponent_private_discard"] = ["As"]
    with pytest.raises(ValueError, match="hidden opponent"):
        teacher.validate_attempt12_teacher_output(
            _root(), baseline_action_key=baseline, payload=hidden, config=_config()
        )

    scope = copy.deepcopy(payload)
    scope["confirmation"]["action_keys"][0] = scope["confirmation"]["action_keys"][1]
    with pytest.raises(ValueError, match="phase action subset/order|scopes differ"):
        teacher.validate_attempt12_teacher_output(
            _root(), baseline_action_key=baseline, payload=scope, config=_config()
        )

    pooled = copy.deepcopy(payload)
    pooled["pooled"]["actions"][0]["raw_paired_deltas_vs_baseline"][0] += 1.0
    with pytest.raises(ValueError, match="raw paired digest|concatenation"):
        teacher.validate_attempt12_teacher_output(
            _root(), baseline_action_key=baseline, payload=pooled, config=_config()
        )

    rng = copy.deepcopy(payload)
    rng["rng_key_digests"]["confirmation_c1024"] = list(
        rng["rng_key_digests"]["stress_x1024"]
    )
    with pytest.raises(ValueError, match="RNG namespaces overlap"):
        teacher.validate_attempt12_teacher_output(
            _root(), baseline_action_key=baseline, payload=rng, config=_config()
        )


def test_attempt12_scalar_batch_parity_and_no_hidden_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, scalar = _evaluate(monkeypatch, 5, batch=False)
    _, batch = _evaluate(monkeypatch, 5, batch=True)
    normalized = copy.deepcopy(batch)
    normalized["search_config"]["batch_child_selectors"] = False

    assert normalized == scalar
    encoded = json.dumps(scalar, sort_keys=True)
    assert "opponent_private_discard" not in encoded
    assert '"particles"' not in encoded


def test_attempt12_duplicate_legal_action_mapping_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _limited_actions(1)
    duplicate = (actions[0], actions[0], actions[-1])
    monkeypatch.setattr(teacher, "generate_turn_actions", lambda *_: duplicate)
    with pytest.raises(ValueError, match="not unique"):
        teacher.evaluate_attempt12_t1_second(
            _root(),
            baseline_action_key=action_key(actions[-1]).to_token(),
            ranker=_Ranker(),
            t2_policies=_policies(),
            config=_config(),
        )
