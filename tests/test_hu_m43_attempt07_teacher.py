from __future__ import annotations

import json
from pathlib import Path

import pytest

import ofc_regular.hu_m43_attempt06_teacher as attempt06
import ofc_regular.hu_m43_attempt07_contract as contract
import ofc_regular.hu_m43_attempt07_teacher as teacher
from ofc_regular.action_key import (
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m4_t1_teacher import _ActionScores
from ofc_regular.state import Board


MODEL_HASH = attempt06.ATTEMPT06_FROZEN_MODEL_SHA256
ROOT = Path(__file__).resolve().parents[1]


def test_teacher_constants_are_cross_bound_to_frozen_attempt07_plan() -> None:
    plan = contract.load_and_validate_attempt07_plan(
        ROOT / "configs" / "hu_joint_policy_m43_attempt07.json"
    )
    search = plan["search_protocol"]
    assert search["candidate_generator_artifact_sha256"] == MODEL_HASH
    assert search["learned_nonbaseline_top_k"] == teacher.ATTEMPT07_TOP_K
    assert search["screen"]["samples"] == teacher.ATTEMPT07_SCREEN_SAMPLES
    assert search["shortlist"]["nonbaseline_actions"] == (
        teacher.ATTEMPT07_SHORTLIST_K
    )
    assert search["rerank"]["budgets"] == [32, teacher.ATTEMPT07_RERANK_SAMPLES]
    assert search["rerank"]["winner_tie_break"] == (
        "explicit_baseline_then_ActionKey"
    )
    assert search["veto"]["budgets"] == [64, teacher.ATTEMPT07_VETO_SAMPLES]
    assert search["veto"]["eligibility"] == {
        "paired_delta_mean_strictly_greater_than": teacher.ATTEMPT07_VETO_MIN_MEAN,
        "paired_delta_p05_min": teacher.ATTEMPT07_VETO_MIN_P05,
        "paired_delta_p01_min": teacher.ATTEMPT07_VETO_MIN_P01,
        "paired_delta_min_min": teacher.ATTEMPT07_VETO_MIN_VALUE,
    }
    assert search["assessment"]["samples"] == (
        teacher.ATTEMPT07_ASSESSMENT_SAMPLES
    )


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
    model_id = "fake-frozen-attempt06-lambda"

    def __init__(self, *, scores=None, events: list[str] | None = None) -> None:
        self.scores = scores
        self.events = events

    def score_actions(self, observation, actions, *, baseline_index):
        assert observation == _root()
        assert 0 <= baseline_index < len(actions)
        if self.events is not None:
            self.events.append("rank")
        scores = self.scores
        if scores is None:
            scores = tuple(float(len(actions) - index) for index in range(len(actions)))
        return attempt06.Attempt06RankScores(
            mean=tuple(scores),
            standard_deviation=tuple(0.25 for _ in actions),
        )


def _config(**overrides) -> teacher.Attempt07TeacherConfig:
    values = {
        "frozen_model_sha256": MODEL_HASH,
        "screen_seed": 701,
        "rerank_seed": 702,
        "veto_seed": 703,
        "assessment_seed": 704,
        "child_policy_seed": 705,
        "run_id": "attempt07-test",
    }
    values.update(overrides)
    return teacher.Attempt07TeacherConfig(**values)


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
    screen=None,
    rerank=None,
    veto=None,
    assessment=None,
    events: list[str] | None = None,
):
    def score(_observation, actions, batch, _selector):
        count = len(batch.particles)
        if count == 8:
            phase = "screen"
            spec = screen if screen is not None else [9, 8, 7, 6, 5, 4, 3, 2, 0]
        elif count == 64:
            phase = "rerank"
            spec = rerank if rerank is not None else [3, 2, 1, 0]
        elif count == 128 and len(actions) != 9:
            phase = "veto"
            spec = (
                veto
                if veto is not None
                else [*([1] * (len(actions) - 1)), 0]
            )
        elif count == 128 and len(actions) == 9:
            phase = "assessment"
            spec = assessment if assessment is not None else [9, 8, 7, 6, 5, 4, 3, 2, 0]
        else:  # pragma: no cover - contract failure is more useful than a fallback
            raise AssertionError((len(actions), count))
        if events is not None:
            events.append(f"score-{phase}")
        if callable(spec):
            rows = spec(actions, count)
        else:
            rows = _constant_rows(spec, count)
        assert len(rows) == len(actions)
        return rows

    return score


def _evaluate(monkeypatch, *, scorer=None, ranker=None):
    monkeypatch.setattr(teacher, "_score_actions", scorer or _scorer())
    return teacher.evaluate_attempt07_t1_second(
        _root(),
        baseline_action_key=_baseline_token(),
        ranker=ranker or _Ranker(),
        t2_policies=_policies(),
        config=_config(),
    )


def test_config_hard_locks_budgets_thresholds_and_distinct_seeds() -> None:
    config = _config()
    assert (
        config.candidate_top_k,
        config.screen_samples,
        config.shortlist_k,
        config.rerank_samples,
        config.veto_samples,
        config.assessment_samples,
    ) == (8, 8, 3, 64, 128, 128)
    with pytest.raises(ValueError, match="fixed at 64"):
        _config(rerank_samples=16)
    with pytest.raises(ValueError, match="fixed at -25.0"):
        _config(veto_min_p05=-20.0)
    with pytest.raises(ValueError, match="all be distinct"):
        _config(veto_seed=702)
    with pytest.raises(ValueError, match="stage9f_p2"):
        _config(t2_policy_id="stage3_baseline")
    with pytest.raises(ValueError, match="frozen Attempt06"):
        _config(frozen_model_sha256="a" * 64)


def test_top8_and_nonbaseline_shortlist_are_actionkey_tied_before_sampling(
    monkeypatch,
) -> None:
    events: list[str] = []
    original_sample = teacher.sample_hidden_card_particles

    def sample(*args, **kwargs):
        events.append(f"sample-{kwargs['sample_count']}")
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(teacher, "sample_hidden_card_particles", sample)
    monkeypatch.setattr(
        teacher,
        "_score_actions",
        _scorer(
            # Baseline is deliberately best but cannot consume a shortlist slot.
            screen=[1, 1, 1, 1, 1, 1, 1, 1, 100],
            events=events,
        ),
    )
    legal = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    baseline = legal[-1]
    ranker = _Ranker(scores=tuple(1.0 for _ in legal), events=events)
    result = teacher.evaluate_attempt07_t1_second(
        _root(),
        baseline_action_key=action_key(baseline).to_token(),
        ranker=ranker,
        t2_policies=_policies(),
        config=_config(),
    )

    expected_top8 = sorted(
        (action for action in legal if action_key(action) != action_key(baseline)),
        key=lambda action: action_key(action).sort_key(),
    )[:8]
    assert result["learned_top8_action_keys"] == [
        action_key(action).to_token() for action in expected_top8
    ]
    assert result["shortlist_action_keys"] == [
        action_key(action).to_token() for action in expected_top8[:3]
    ]
    assert result["baseline_action_key"] not in result["shortlist_action_keys"]
    assert events == [
        "rank",
        "sample-8",
        "score-screen",
        "sample-64",
        "score-rerank",
        "sample-128",
        "score-veto",
        "sample-128",
        "score-assessment",
    ]


def test_rerank_exact_ties_prefer_explicit_baseline_and_all_arms_fallback(
    monkeypatch,
) -> None:
    result = _evaluate(
        monkeypatch,
        scorer=_scorer(rerank=[0, 0, 0, 0], veto=[0]),
    )
    for prefix in ("R32", "R64"):
        assert result["rerank"]["prefixes"][prefix][
            "winner_action_key"
        ] == result["baseline_action_key"]
    for arm in result["arms"].values():
        assert arm["rerank_winner_is_baseline"] is True
        assert arm["override_fired"] is False
        assert arm["selected_action_key"] == result["baseline_action_key"]
        assert arm["fallback_reason"] == "rerank_winner_is_explicit_baseline"


def test_r32_v64_are_true_prefixes_of_r64_v128(monkeypatch) -> None:
    def rerank(_actions, count):
        assert count == 64
        return (
            _ActionScores(tuple([3.0] * 32 + [-3.0] * 32)),
            _ActionScores(tuple([1.0] * 64)),
            _ActionScores(tuple([0.5] * 64)),
            _ActionScores(tuple([0.0] * 64)),
        )

    def veto(_actions, count):
        assert count == 128
        return (
            _ActionScores(tuple([2.0] * 64 + [1.0] * 64)),
            _ActionScores(tuple([1.0] * 64 + [3.0] * 64)),
            _ActionScores(tuple([0.0] * 128)),
        )

    result = _evaluate(
        monkeypatch, scorer=_scorer(rerank=rerank, veto=veto)
    )
    r32 = result["rerank"]["prefixes"]["R32"]
    r64 = result["rerank"]["prefixes"]["R64"]
    assert r32["winner_position"] == 0
    assert r64["winner_position"] == 1
    assert r32["actions"][0]["raw_paired_deltas_vs_baseline"] == r64[
        "actions"
    ][0]["raw_paired_deltas_vs_baseline"][:32]
    v64 = result["veto"]["prefixes"]["V64"]["actions"]
    v128 = result["veto"]["prefixes"]["V128"]["actions"]
    assert result["veto"]["action_count"] == 3
    for position in range(3):
        assert v64[position]["raw_paired_deltas_vs_baseline"] == v128[position][
            "raw_paired_deltas_vs_baseline"
        ][:64]
    assert result["arms"]["R32_V64"]["rerank_winner_position"] == 0
    assert result["arms"]["R64_V128"]["rerank_winner_position"] == 1


def test_s_r_v_a_particle_namespaces_are_pairwise_disjoint(monkeypatch) -> None:
    result = _evaluate(monkeypatch)
    key_sets = [set(values) for values in result["rng_key_digests"].values()]
    assert [len(values) for values in key_sets] == [8, 64, 128, 128]
    for left_index, left in enumerate(key_sets):
        for right in key_sets[left_index + 1 :]:
            assert left.isdisjoint(right)
    assert result["sample_independence"].startswith("pairwise_disjoint")


def test_a128_is_diagnostic_and_cannot_rerank_or_change_any_arm(monkeypatch) -> None:
    result = _evaluate(
        monkeypatch,
        scorer=_scorer(assessment=[0, 0, 1000, 0, 0, 0, 0, 0, 0]),
    )
    assert result["assessment"]["sample_best_action_key"] == result[
        "learned_top8_action_keys"
    ][2]
    assert result["assessment"]["diagnostics_only"] is True
    assert result["assessment"]["can_rerank_or_gate"] is False
    assert result["assessment"]["decision_frozen_before_namespace_open"] is True
    for arm in result["arms"].values():
        assert arm["rerank_winner_position"] == 0
        assert arm["selected_action_key"] == result["shortlist_action_keys"][0]
        assert arm["override_fired"] is True


def test_tail_veto_falls_back_to_exact_baseline_never_safe_second_best(
    monkeypatch,
) -> None:
    def veto(_actions, count):
        # Winner 0 has positive mean and benign quantiles, but one -100 tail.
        # Action 1 is safe and positive; it still must never replace winner 0.
        return (
            _ActionScores(tuple([1.0] * (count - 1) + [-100.0])),
            _ActionScores(tuple([0.0] * count)),
        )

    result = _evaluate(monkeypatch, scorer=_scorer(veto=veto))
    for arm_name in ("R32_V128", "R64_V128"):
        arm = result["arms"][arm_name]
        assert arm["rerank_winner_position"] == 0
        assert arm["veto_checks"]["mean_gt_0"] is True
        assert arm["veto_checks"]["min_ge_neg50"] is False
        assert arm["veto_pass"] is False
        assert arm["selected_action_key"] == result["baseline_action_key"]
        assert arm["fallback_reason"] == "paired_safety_veto_failed"
        assert arm["second_best_promotion_allowed"] is False
        assert len(arm["veto_raw_paired_deltas_vs_baseline"]) == 128


def test_action_mappings_are_exact_and_output_is_hidden_discard_safe(
    monkeypatch,
) -> None:
    result = _evaluate(monkeypatch)
    legal = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    mapping = result["legal_action_mapping"]
    assert mapping["action_keys"] == [action_key(action).to_token() for action in legal]
    assert mapping["action_set_digest"] == legal_action_set_digest(legal)
    assert mapping["action_order_digest"] == ordered_action_mapping_digest(legal)
    assert set(result["proposal_mapping"]["action_keys"]) == {
        *result["learned_top8_action_keys"],
        result["baseline_action_key"],
    }
    assert set(result["veto"]["action_keys"]).issubset(
        result["rerank"]["action_keys"]
    )
    assert result["veto"]["action_keys"][-1] == result["baseline_action_key"]
    encoded = json.dumps(result, sort_keys=True)
    assert "opponent_private_discards" not in encoded
    assert result["policy_observation"] == _root().to_dict()
    assert result["teacher_value_status"] == "diagnostic_not_match_EV"
    assert result["runtime_gate_allowed"] is False
    assert result["profile_activation_allowed"] is False
    assert result["current_profile_resolved"] is False


def test_veto_scope_excludes_unselected_shortlist_action(monkeypatch) -> None:
    result = _evaluate(monkeypatch)
    # Both R32 and R64 lock shortlist position zero, so V scores only that
    # unique winner and the explicit baseline.  Positions one and two cannot
    # influence any safety decision.
    assert result["veto"]["action_keys"] == [
        result["shortlist_action_keys"][0],
        result["baseline_action_key"],
    ]
    assert result["shortlist_action_keys"][1] not in result["veto"]["action_keys"]
    assert result["shortlist_action_keys"][2] not in result["veto"]["action_keys"]
    assert result["veto"]["locked_nonbaseline_rerank_positions"] == [0]
