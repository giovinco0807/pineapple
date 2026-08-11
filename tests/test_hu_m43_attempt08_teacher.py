from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import ofc_regular.hu_m43_attempt08_teacher as teacher
from ofc_regular.action_key import (
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt05_model import (
    Attempt05FoldOutput,
    HuM43Attempt05Model,
)
from ofc_regular.hu_m4_t1_teacher import _ActionScores
from ofc_regular.state import Board


MODEL_HASH = teacher.ATTEMPT08_FROZEN_MODEL_SHA256


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
    model_id = teacher.ATTEMPT08_FROZEN_MODEL_ID

    def __init__(
        self,
        *,
        rank_mean=None,
        p95=None,
        p99=None,
        maximum=None,
        events: list[str] | None = None,
    ) -> None:
        self.rank_mean = rank_mean
        self.p95 = p95
        self.p99 = p99
        self.maximum = maximum
        self.events = events

    def score_actions(self, observation, actions, *, baseline_index):
        assert observation == _root()
        assert 0 <= baseline_index < len(actions)
        if self.events is not None:
            self.events.append("rank")
        rank = self.rank_mean
        if rank is None:
            rank = tuple(float(len(actions) - index) for index in range(len(actions)))
        p95 = self.p95 or tuple(10.0 for _ in actions)
        p99 = self.p99 or tuple(20.0 for _ in actions)
        maximum = self.maximum or tuple(30.0 for _ in actions)
        return teacher.Attempt08RankScores(
            rank_mean=tuple(rank),
            rank_disagreement=tuple(0.25 for _ in actions),
            raw_downside_p95=tuple(p95),
            raw_downside_p99=tuple(p99),
            raw_downside_max=tuple(maximum),
        )


def _config(**overrides) -> teacher.Attempt08TeacherConfig:
    values = {
        "frozen_model_sha256": MODEL_HASH,
        "hand_seed": 801,
        "rerank_seed": 802,
        "veto_seed": 803,
        "stress_seed": 804,
        "assessment_seed": 805,
        "child_policy_seed": 806,
        "run_id": "attempt08-test",
    }
    values.update(overrides)
    return teacher.Attempt08TeacherConfig(**values)


def _baseline_token() -> str:
    actions = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    return action_key(actions[-1]).to_token()


def _constant_rows(means, sample_count: int) -> tuple[_ActionScores, ...]:
    return tuple(
        _ActionScores(tuple(float(mean) for _ in range(sample_count)))
        for mean in means
    )


def _scorer(
    *,
    rerank=None,
    veto=None,
    stress=None,
    assessment=None,
    events: list[str] | None = None,
):
    def score(_observation, actions, batch, _selector):
        phase = batch.run_id.rsplit(":", 1)[-1]
        count = len(batch.particles)
        if phase == "rerank_r128":
            assert count == 128 and len(actions) == 9
            spec = rerank if rerank is not None else [8, 7, 6, 5, 4, 3, 2, 1, 0]
        elif phase == "veto_v256":
            assert count == 256 and len(actions) == 5
            spec = veto if veto is not None else [4, 3, 2, 1, 0]
        elif phase == "stress_x512":
            assert count == 512 and len(actions) == 2
            spec = stress if stress is not None else [1, 0]
        elif phase == "assessment_a256":
            assert count == 256 and len(actions) in (1, 2)
            spec = assessment
            if spec is None:
                spec = [1, 0] if len(actions) == 2 else [0]
        else:  # pragma: no cover - the assertion identifies schema drift
            raise AssertionError((phase, len(actions), count))
        if events is not None:
            events.append(f"score-{phase}")
        rows = spec(actions, count) if callable(spec) else _constant_rows(spec, count)
        assert len(rows) == len(actions)
        return rows

    return score


def _evaluate(monkeypatch, *, scorer=None, ranker=None, config=None):
    monkeypatch.setattr(teacher, "_score_actions", scorer or _scorer())
    return teacher.evaluate_attempt08_t1_second(
        _root(),
        baseline_action_key=_baseline_token(),
        ranker=ranker or _Ranker(),
        t2_policies=_policies(),
        config=config or _config(),
    )


def test_config_hard_locks_semantics_and_all_six_seed_domains() -> None:
    config = _config()
    assert (
        config.candidate_top_k,
        config.rerank_samples,
        config.rerank_top_k,
        config.k4_size,
        config.veto_samples,
        config.stress_samples,
        config.assessment_samples,
    ) == (8, 128, 3, 4, 256, 512, 256)
    with pytest.raises(ValueError, match="fixed at 128"):
        _config(rerank_samples=64)
    with pytest.raises(ValueError, match="fixed at -22.0"):
        _config(min_p05=-25.0)
    with pytest.raises(ValueError, match="all be distinct"):
        _config(stress_seed=803)
    with pytest.raises(ValueError, match="all be distinct"):
        _config(hand_seed=806)
    with pytest.raises(ValueError, match="stage9f_p2"):
        _config(t2_policy_id="current")
    with pytest.raises(ValueError, match="frozen candidate-only Lambda"):
        _config(frozen_model_sha256="a" * 64)


class _Fold:
    family = "lambda_rank"

    def __init__(self, fold_index: int) -> None:
        self.fold_index = fold_index

    def predict(self, runtime_sample, *, baseline_index):
        count = len(runtime_sample["actions"])
        base = np.arange(count, dtype=np.float64)
        shift = float(self.fold_index)
        return Attempt05FoldOutput(
            rank_score=base + shift,
            gain_probability=np.full(count, 0.5, dtype=np.float64),
            downside_p95=np.full(count, 10.0 + shift, dtype=np.float64),
            downside_p99=np.full(count, 20.0 + shift, dtype=np.float64),
            downside_max=np.full(count, 30.0 + shift, dtype=np.float64),
        )


def test_frozen_lambda_ranker_exposes_raw_fold_mean_tails_without_conformal() -> None:
    model = HuM43Attempt05Model(
        family="lambda_rank",
        fold_predictors=tuple(_Fold(index) for index in range(5)),
        conformal_cushions=(100.0, 200.0, 300.0),
        runtime_enabled=False,
        winner_frozen=False,
        model_id=teacher.ATTEMPT08_FROZEN_MODEL_ID,
    )
    ranker = teacher.FrozenAttempt08LambdaRanker(
        model=model, artifact_sha256=MODEL_HASH
    )
    actions = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    scores = ranker.score_actions(
        _root(), actions, baseline_index=len(actions) - 1
    )
    assert scores.fold_count == 5
    assert scores.rank_mean[3] == pytest.approx(5.0)
    assert scores.rank_disagreement[3] == pytest.approx(np.std(np.arange(5)))
    assert scores.raw_downside_p95[3] == pytest.approx(12.0)
    assert scores.raw_downside_p99[3] == pytest.approx(22.0)
    assert scores.raw_downside_max[3] == pytest.approx(32.0)
    assert scores.raw_downside_max[3] != pytest.approx(332.0)


def test_r128_k4_is_top3_plus_min_raw_risk_reserve_then_v_uses_r_order(
    monkeypatch,
) -> None:
    legal = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    count = len(legal)
    p95 = [20.0] * count
    p99 = [30.0] * count
    maximum = [44.0] * count
    # Proposal position six is R rank seven and uniquely safest in ranks 4..8.
    p95[6], p99[6], maximum[6] = 1.0, 2.0, 3.0
    ranker = _Ranker(p95=tuple(p95), p99=tuple(p99), maximum=tuple(maximum))
    result = _evaluate(monkeypatch, ranker=ranker)

    assert result["rerank"]["ordered_nonbaseline_proposal_positions"] == list(range(8))
    assert result["k4"]["top3_rerank_positions"] == [0, 1, 2]
    assert result["k4"]["risk_reserve_rerank_position"] == 6
    assert result["k4"]["risk_reserve_r_rank"] == 7
    assert result["k4"]["veto_traversal_rerank_positions"] == [0, 1, 2, 6]
    assert result["veto"]["traversal_action_keys"] == result["k4"]["action_keys"]


def test_risk_reserve_exact_score_tie_uses_actionkey_not_r_rank(monkeypatch) -> None:
    legal = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    count = len(legal)
    p95, p99, maximum = [20.0] * count, [30.0] * count, [44.0] * count
    for index in range(3, 8):
        p95[index], p99[index], maximum[index] = 2.0, 3.0, 4.0
    result = _evaluate(
        monkeypatch,
        ranker=_Ranker(
            p95=tuple(p95), p99=tuple(p99), maximum=tuple(maximum)
        ),
    )
    candidates = result["rerank"]["ordered_nonbaseline_action_keys"][3:]
    expected = min(
        candidates,
        key=lambda token: next(
            action_key(action).sort_key()
            for action in legal
            if action_key(action).to_token() == token
        ),
    )
    assert result["k4"]["risk_reserve_action_key"] == expected


def test_v256_promotes_first_safe_candidate_in_frozen_r_order(monkeypatch) -> None:
    def veto(_actions, count):
        # R-first is positive but has a catastrophe and fails. R-second passes.
        return (
            _ActionScores(tuple([1.0] * (count - 1) + [-100.0])),
            _ActionScores(tuple([2.0] * count)),
            _ActionScores(tuple([3.0] * count)),
            _ActionScores(tuple([4.0] * count)),
            _ActionScores(tuple([0.0] * count)),
        )

    result = _evaluate(monkeypatch, scorer=_scorer(veto=veto))
    assert result["veto"]["checks_by_traversal_position"][0]["min_ge_neg45"] is False
    assert result["veto"]["first_passing_traversal_position"] == 1
    assert result["decision"]["veto_selected_action_key"] == result["k4"]["action_keys"][1]
    assert result["stress"]["opened"] is True
    assert result["decision"]["final_selected_action_key"] == result["k4"]["action_keys"][1]
    assert result["decision"]["override_fired"] is True


def test_x512_is_cancel_only_and_never_promotes_later_v_candidate(monkeypatch) -> None:
    def stress(_actions, count):
        return (
            _ActionScores(tuple([1.0] * (count - 1) + [-60.0])),
            _ActionScores(tuple([0.0] * count)),
        )

    result = _evaluate(monkeypatch, scorer=_scorer(stress=stress))
    assert result["veto"]["first_passing_traversal_position"] == 0
    assert result["stress"]["opened"] is True
    assert result["stress"]["pass"] is False
    assert result["stress"]["cancelled"] is True
    assert result["stress"]["may_promote_or_rerank"] is False
    assert result["decision"]["final_selected_action_key"] == result["baseline_action_key"]
    assert result["decision"]["fallback_reason"] == "x512_catastrophe_cancel"
    assert result["decision"]["second_candidate_promotion_after_stress_cancel_allowed"] is False
    assert result["assessment"]["opened"] is False
    assert result["assessment"]["action_count"] == 0
    assert result["assessment"]["sample_count"] == 0
    assert result["assessment"]["raw_paired_deltas_vs_baseline"] == []
    assert "assessment_a256" not in result["rng_key_digests"]


def test_v_nonfire_never_opens_x_or_a_and_counterfactual_cancels_exactly(
    monkeypatch,
) -> None:
    events: list[str] = []
    original_sample = teacher.sample_hidden_card_particles

    def sample(*args, **kwargs):
        events.append(kwargs["run_id"].rsplit(":", 1)[-1])
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(teacher, "sample_hidden_card_particles", sample)
    result = _evaluate(
        monkeypatch,
        scorer=_scorer(veto=[-1, -2, -3, -4, 0], events=events),
    )
    assert events == [
        "rerank_r128",
        "score-rerank_r128",
        "veto_v256",
        "score-veto_v256",
    ]
    assert result["stress"]["opened"] is False
    assert result["stress"]["sample_count"] == 0
    assert "stress_x512" not in result["rng_key_digests"]
    assert result["decision"]["exact_baseline_fallback"] is True
    assert result["decision"]["fallback_reason"] == "no_v256_candidate_passed"
    assert result["assessment"]["opened"] is False
    assert result["assessment"]["action_count"] == 0
    assert result["assessment"]["sample_count"] == 0
    assert result["assessment"]["raw_paired_deltas_vs_baseline"] == []
    assert result["assessment"]["actions"] == []
    assert "assessment_a256" not in result["rng_key_digests"]
    assert set(result["rng_key_digests"]) == {"rerank_r128", "veto_v256"}


def test_a256_is_diagnostic_only_and_cannot_cancel_or_change_final(monkeypatch) -> None:
    def assessment(_actions, count):
        return (
            _ActionScores(tuple([-1000.0] * count)),
            _ActionScores(tuple([0.0] * count)),
        )

    result = _evaluate(monkeypatch, scorer=_scorer(assessment=assessment))
    selected_before = result["decision"]["veto_selected_action_key"]
    assert result["stress"]["pass"] is True
    assert result["decision"]["final_selected_action_key"] == selected_before
    assert result["assessment"]["sample_best_action_key"] == result["baseline_action_key"]
    assert result["assessment"]["can_rerank_or_gate"] is False
    assert result["assessment"]["diagnostics_only"] is True
    assert result["decision"]["override_fired"] is True


def test_rng_namespaces_are_disjoint_and_hidden_batches_are_not_retained(
    monkeypatch,
) -> None:
    result = _evaluate(monkeypatch)
    assert result["seed_domain_provenance"]["domain_order"] == list(
        teacher.ATTEMPT08_RNG_DOMAINS
    )
    assert result["seed_domain_provenance"]["hand_sampled_inside_teacher"] is False
    key_sets = [set(values) for values in result["rng_key_digests"].values()]
    assert [len(values) for values in key_sets] == [128, 256, 512, 256]
    for left_index, left in enumerate(key_sets):
        for right in key_sets[left_index + 1 :]:
            assert left.isdisjoint(right)
    assert result["memory_retention"] == {
        "retained_particle_batches": 0,
        "retained_hidden_particle_payload": False,
        "retained_child_selector_caches": 0,
        "output_contains_only_value_vectors_and_opaque_digests": True,
    }
    encoded = json.dumps(result, sort_keys=True)
    assert "opponent_private_discards" not in encoded
    assert '"particles"' not in encoded


def test_complete_action_mapping_masks_raw_digests_and_validator(monkeypatch) -> None:
    result = _evaluate(monkeypatch)
    legal = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    mapping = result["legal_action_mapping"]
    assert mapping["action_keys"] == [action_key(action).to_token() for action in legal]
    assert mapping["action_set_digest"] == legal_action_set_digest(legal)
    assert mapping["action_order_digest"] == ordered_action_mapping_digest(legal)
    assert result["legal_action_mask"] == [True] * len(legal)
    assert result["illegal_action_mask"] == [False] * len(legal)
    assert len(result["rerank"]["actions"][0]["raw_paired_deltas_vs_baseline"]) == 128
    assert len(result["veto"]["actions"][0]["raw_paired_deltas_vs_baseline"]) == 256
    assert len(result["stress"]["raw_paired_deltas_vs_baseline"]) == 512
    assert len(result["assessment"]["raw_paired_deltas_vs_baseline"]) == 256
    normalized = teacher.validate_attempt08_teacher_output(
        _root(),
        baseline_action_key=_baseline_token(),
        payload=result,
        config=_config(),
    )
    assert normalized["selected_action_key"] == result["decision"]["final_selected_action_key"]
    assert normalized["override_fired"] is True
    tampered = copy.deepcopy(result)
    tampered["assessment"]["raw_paired_deltas_vs_baseline"][0] += 1.0
    with pytest.raises(ValueError, match="A256 locked diagnostics"):
        teacher.validate_attempt08_teacher_output(
            _root(),
            baseline_action_key=_baseline_token(),
            payload=tampered,
            config=_config(),
        )


def test_public_validator_recomputes_every_selection_and_lock_boundary(
    monkeypatch,
) -> None:
    result = _evaluate(monkeypatch)

    def tampered(change):
        value = copy.deepcopy(result)
        change(value)
        return value

    cases = [
        (
            lambda value: value["legal_actions"][8].__setitem__(
                "model_rank_mean", 1_000_000.0
            ),
            "learned top8 selection",
        ),
        (
            lambda value: value["rerank"]["actions"][0][
                "raw_paired_deltas_vs_baseline"
            ].__setitem__(0, 99.0),
            "R128 raw paired digest",
        ),
        (
            lambda value: value["rerank"].__setitem__(
                "ordered_nonbaseline_proposal_positions",
                list(reversed(value["rerank"]["ordered_nonbaseline_proposal_positions"])),
            ),
            "R128 order",
        ),
        (
            lambda value: value["k4"].__setitem__(
                "risk_reserve_action_key", value["k4"]["top3_action_keys"][0]
            ),
            "K4 risk-reserve contract",
        ),
        (
            lambda value: value["veto"]["actions"][0][
                "raw_paired_deltas_vs_baseline"
            ].__setitem__(0, -99.0),
            "V256 raw paired digest",
        ),
        (
            lambda value: value["veto"].__setitem__(
                "first_passing_traversal_position", 3
            ),
            "V256 first-safe selection",
        ),
        (
            lambda value: value["stress"].__setitem__("pass", False),
            "X512 catastrophe decision",
        ),
        (
            lambda value: value["decision"].__setitem__(
                "final_selected_action_key", value["baseline_action_key"]
            ),
            "locked final decision",
        ),
        (
            lambda value: value["assessment"].__setitem__(
                "locked_final_action_key", value["baseline_action_key"]
            ),
            "A256 diagnostic lock",
        ),
        (
            lambda value: value["veto"].__setitem__(
                "action_order_digest", "0" * 64
            ),
            "V256 action_order_digest",
        ),
        (
            lambda value: value.__setitem__("root_profile", "stage19_p0"),
            "teacher top-level fields changed",
        ),
    ]
    for change, message in cases:
        with pytest.raises(ValueError, match=message):
            teacher.validate_attempt08_teacher_output(
                _root(),
                baseline_action_key=_baseline_token(),
                payload=tampered(change),
                config=_config(),
            )


def test_strict_schema_rejects_unknown_fields_at_every_trust_boundary(
    monkeypatch,
) -> None:
    result = _evaluate(monkeypatch)

    def add_extra(path):
        value = copy.deepcopy(result)
        target = value
        for key in path:
            target = target[key]
        target["unknown_extension"] = True
        return value

    paths = [
        (),
        ("frozen_candidate_generator",),
        ("legal_action_mapping",),
        ("legal_actions", 0),
        ("rerank",),
        ("rerank", "actions", 0),
        ("rerank", "actions", 0, "paired_delta_vs_baseline"),
        ("k4",),
        ("k4", "risk_reserve_raw_components"),
        ("veto",),
        ("veto", "thresholds"),
        ("veto", "checks_by_traversal_position", 0),
        ("stress",),
        ("decision",),
        ("assessment",),
        ("seed_domain_provenance",),
        ("search_config",),
        ("continuation_policy",),
        ("memory_retention",),
    ]
    for path in paths:
        with pytest.raises(ValueError, match="fields changed|mapping fields changed|memory-retention"):
            teacher.validate_attempt08_teacher_output(
                _root(),
                baseline_action_key=_baseline_token(),
                payload=add_extra(path),
                config=_config(),
            )


def test_phase_rows_label_unverifiable_marginal_digest_as_opaque(monkeypatch) -> None:
    result = _evaluate(monkeypatch)
    for phase in ("rerank", "veto", "stress", "assessment"):
        for row in result[phase]["actions"]:
            assert "opaque_action_values_sha256" in row
            assert "action_values_sha256" not in row


def test_scalar_batch_semantics_and_repeated_runs_are_deterministic(monkeypatch) -> None:
    scorer = _scorer()
    monkeypatch.setattr(teacher, "_score_actions", scorer)
    monkeypatch.setattr(teacher, "_score_actions_batched", scorer)
    scalar = teacher.evaluate_attempt08_t1_second(
        _root(),
        baseline_action_key=_baseline_token(),
        ranker=_Ranker(),
        t2_policies=_policies(),
        config=_config(batch_child_selectors=False),
    )
    scalar_repeat = teacher.evaluate_attempt08_t1_second(
        _root(),
        baseline_action_key=_baseline_token(),
        ranker=_Ranker(),
        t2_policies=_policies(),
        config=_config(batch_child_selectors=False),
    )
    batch = teacher.evaluate_attempt08_t1_second(
        _root(),
        baseline_action_key=_baseline_token(),
        ranker=_Ranker(),
        t2_policies=_policies(),
        config=_config(batch_child_selectors=True),
    )
    assert scalar == scalar_repeat
    for key in (
        "legal_action_mapping",
        "rerank",
        "k4",
        "veto",
        "stress",
        "decision",
        "assessment",
        "belief_digests",
        "rng_key_digests",
    ):
        assert scalar[key] == batch[key]


def test_baseline_and_stage9f_contracts_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(teacher, "_score_actions", _scorer())
    with pytest.raises(ValueError, match="not uniquely legal"):
        teacher.evaluate_attempt08_t1_second(
            _root(),
            baseline_action_key="ak1:0000000000000:0000000000000:0000000000000:0000000000000",
            ranker=_Ranker(),
            t2_policies=_policies(),
            config=_config(),
        )
    bad_policies = _policies()
    bad_policies["second"].topk_context["runtime_profile"] = "current"
    with pytest.raises(ValueError, match="stage9f_p2"):
        teacher.evaluate_attempt08_t1_second(
            _root(),
            baseline_action_key=_baseline_token(),
            ranker=_Ranker(),
            t2_policies=bad_policies,
            config=_config(),
        )
