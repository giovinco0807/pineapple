from __future__ import annotations

import json

import pytest

import ofc_regular.hu_m4_t1_teacher as teacher
import ofc_regular.hu_m4_t1_shortlist_search as search
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m4_t1_shortlist_search import (
    M4_T1_SHORTLIST_SEARCH_SCHEMA,
    M4_T1_SHORTLIST_SOLVER_ID,
    M4T1ShortlistSearchConfig,
    evaluate_t1_second_shortlist_search,
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
    def __init__(self, seat: str) -> None:
        self.seat = seat

    def choose_action_observation(
        self,
        observation: ActorObservation,
        *,
        hand_id=None,
        game_id=None,
        decision_seed=None,
    ):
        assert observation.seat == self.seat
        assert isinstance(decision_seed, int)
        return generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )[0]


def _policies() -> dict[str, _FirstLegalPolicy]:
    return {
        "first": _FirstLegalPolicy("first"),
        "second": _FirstLegalPolicy("second"),
    }


def _keys() -> tuple[list[str], str]:
    actions = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    tokens = [action_key(value).to_token() for value in actions]
    return tokens[:4], tokens[-1]


def _config(*, batch: bool = True) -> M4T1ShortlistSearchConfig:
    return M4T1ShortlistSearchConfig(
        candidate_samples=4,
        evaluation_samples=4,
        candidate_seed=7101,
        evaluation_seed=7102,
        child_policy_seed=7103,
        run_id="attempt05-test",
        t2_policy_id="first-legal-v1",
        batch_child_selectors=batch,
    )


def _patch_first_legal_native(monkeypatch, captured: list[dict] | None = None) -> None:
    def scalar(observation: ActorObservation, *, config, **_kwargs):
        if captured is not None:
            captured.append(
                {
                    "street": observation.street,
                    "candidate_samples": config.candidate_samples,
                    "evaluation_samples": config.evaluation_samples,
                    "downstream_t4_samples": getattr(
                        config, "downstream_t4_samples", None
                    ),
                }
            )
        selected = generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )[0]
        return {"selected_action_key": action_key(selected).to_token()}

    def batched(requests, **_kwargs):
        results = []
        for request in requests:
            observation = ActorObservation.from_dict(request["observation"])
            config = request["config"]
            if captured is not None:
                captured.append(
                    {
                        "street": observation.street,
                        "candidate_samples": config["candidate_samples"],
                        "evaluation_samples": config["evaluation_samples"],
                        "downstream_t4_samples": config.get(
                            "downstream_t4_samples"
                        ),
                    }
                )
            selected = generate_turn_actions(
                observation.hero_board, observation.dealt_cards
            )[0]
            results.append(
                {"selected_action_key": action_key(selected).to_token()}
            )
        return results

    monkeypatch.setattr(teacher, "evaluate_t3", scalar)
    monkeypatch.setattr(teacher, "evaluate_t4", scalar)
    monkeypatch.setattr(teacher, "evaluate_batch", batched)


def _evaluate(*, config: M4T1ShortlistSearchConfig) -> dict:
    shortlist, baseline = _keys()
    return evaluate_t1_second_shortlist_search(
        _root(),
        t2_policies=_policies(),
        baseline_action_key=baseline,
        learned_shortlist_action_keys=shortlist,
        config=config,
    )


def test_config_freezes_top4_small_search_budgets_and_disjoint_seeds() -> None:
    assert M4T1ShortlistSearchConfig().top_k == 4
    assert M4T1ShortlistSearchConfig().t3_downstream_t4_samples == 1
    assert M4T1ShortlistSearchConfig().t4_candidate_samples == 1
    assert M4T1ShortlistSearchConfig().t4_evaluation_samples == 1
    assert M4T1ShortlistSearchConfig(t3_downstream_t4_samples=0).t3_downstream_t4_samples == 0
    with pytest.raises(ValueError, match="one of 4 or 8"):
        M4T1ShortlistSearchConfig(candidate_samples=2)
    with pytest.raises(ValueError, match="top_k is frozen at 4"):
        M4T1ShortlistSearchConfig(top_k=3)
    assert M4T1ShortlistSearchConfig(
        t4_candidate_samples=0, t4_evaluation_samples=0
    ).t4_candidate_samples == 0
    with pytest.raises(ValueError, match="must be 0 or 1"):
        M4T1ShortlistSearchConfig(t4_candidate_samples=2, t4_evaluation_samples=2)
    with pytest.raises(ValueError, match="budgets must match"):
        M4T1ShortlistSearchConfig(t4_candidate_samples=0, t4_evaluation_samples=1)
    with pytest.raises(ValueError, match="must be 0"):
        M4T1ShortlistSearchConfig(t3_downstream_t4_samples=2)
    with pytest.raises(ValueError, match="must be distinct"):
        M4T1ShortlistSearchConfig(candidate_seed=5, evaluation_seed=5)


def test_shortlist_is_canonical_baseline_inclusive_hidden_safe_and_gate_free(
    monkeypatch,
) -> None:
    _patch_first_legal_native(monkeypatch)
    result = _evaluate(config=_config())
    shortlist, baseline = _keys()

    assert result["schema"] == M4_T1_SHORTLIST_SEARCH_SCHEMA
    assert result["solver_id"] == M4_T1_SHORTLIST_SOLVER_ID
    assert result["learned_shortlist_action_keys"] == shortlist
    assert result["fixed_search_action_keys"] == [*shortlist, baseline]
    assert result["fixed_search_action_count"] == 5
    assert result["baseline_action_key"] == baseline
    assert result["baseline_original_index"] == len(
        result["ordered_legal_actions"]
    ) - 1
    assert [row["original_index"] for row in result["ordered_legal_actions"]] == list(
        range(result["legal_action_count"])
    )
    assert len({row["action_key"] for row in result["ordered_legal_actions"]}) == result[
        "legal_action_count"
    ]
    assert set(result["candidate_rng_key_digests"]).isdisjoint(
        result["evaluation_rng_key_digests"]
    )
    assert result["common_random_futures_across_fixed_actions"] is True
    assert result["fixed_candidates_before_sampling"] is True
    assert result["runtime_gate_applied"] is False
    assert result["opponent_policy_identity_used"] is False
    assert result["opponent_private_discard_input_used"] is False
    assert "lcb" not in json.dumps(result).lower()
    assert "opponent_private_discards" not in json.dumps(result)


def test_scalar_batch_reference_and_repeat_are_exactly_identical(monkeypatch) -> None:
    _patch_first_legal_native(monkeypatch)
    scalar = _evaluate(config=_config(batch=False))
    batch = _evaluate(config=_config(batch=True))
    repeated = _evaluate(config=_config(batch=True))

    ignored = {"child_selector_execution", "batch_child_selectors"}

    def normalized(value):
        if isinstance(value, dict):
            return {
                key: normalized(item)
                for key, item in value.items()
                if key not in ignored
            }
        if isinstance(value, list):
            return [normalized(item) for item in value]
        return value

    assert normalized(batch) == normalized(scalar)
    assert repeated == batch
    assert [row["action_key"] for row in batch["actions"]] == [
        row["action_key"] for row in scalar["actions"]
    ]
    assert [row["candidate_mean"] for row in batch["actions"]] == [
        row["candidate_mean"] for row in scalar["actions"]
    ]


def test_candidate_proposal_is_locked_before_independent_evaluation(monkeypatch) -> None:
    calls = 0

    def controlled(_observation, actions, _batch, _selector):
        nonlocal calls
        calls += 1
        if calls == 1:
            values = [10.0, 9.0, *([0.0] * (len(actions) - 2))]
        else:
            values = [-10.0, 10.0, *([0.0] * (len(actions) - 2))]
        return tuple(search._ActionScores((value,)) for value in values)

    monkeypatch.setattr(search, "_score_actions_batched", controlled)
    result = _evaluate(config=_config(batch=True))
    shortlist, _baseline = _keys()

    assert result["proposed_action_key"] == shortlist[0]
    assert result["proposal_locked_before_evaluation"] is True
    proposed = next(
        row
        for row in result["actions"]
        if row["action_key"] == result["proposed_action_key"]
    )
    assert proposed["candidate_mean"] == 10.0
    assert proposed["evaluation_mean"] == -10.0


def test_t1_search_t4_budgets_are_declared_and_outer_exact_is_untouched(
    monkeypatch,
) -> None:
    captured: list[dict] = []
    _patch_first_legal_native(monkeypatch, captured)
    result = _evaluate(config=_config(batch=True))

    assert any(row["street"] == "T3" for row in captured)
    assert any(row["street"] == "T4" for row in captured)
    assert all(
        row["downstream_t4_samples"] == 1
        for row in captured
        if row["street"] == "T3"
    )
    assert all(
        row["candidate_samples"] == 1 and row["evaluation_samples"] == 1
        for row in captured
        if row["street"] == "T4"
    )
    assert result["continuation_policy"]["outer_live_t4_exact_solver_unchanged"] is True
    assert result["continuation_policy"]["t1_search_direct_t4_exact"] is False
    assert result["continuation_policy"]["t1_search_direct_t4_mode"] == "counter_mc_1"
    assert result["continuation_policy"]["nested_t4_exact"] is False
    assert result["continuation_policy"]["nested_t4_mode"] == "counter_mc_1"


def test_shortlist_mapping_fails_closed_on_wrong_size_duplicate_or_illegal_key(
    monkeypatch,
) -> None:
    _patch_first_legal_native(monkeypatch)
    shortlist, baseline = _keys()
    kwargs = {
        "t2_policies": _policies(),
        "baseline_action_key": baseline,
        "config": _config(),
    }
    with pytest.raises(ValueError, match="exactly four"):
        evaluate_t1_second_shortlist_search(
            _root(), learned_shortlist_action_keys=shortlist[:3], **kwargs
        )
    with pytest.raises(ValueError, match="must be unique"):
        evaluate_t1_second_shortlist_search(
            _root(),
            learned_shortlist_action_keys=[shortlist[0]] * 4,
            **kwargs,
        )
    with pytest.raises(ValueError, match="illegal ActionKey"):
        evaluate_t1_second_shortlist_search(
            _root(),
            learned_shortlist_action_keys=[*shortlist[:3], "not-an-action"],
            **kwargs,
        )
    with pytest.raises(ValueError, match="baseline ActionKey"):
        evaluate_t1_second_shortlist_search(
            _root(),
            baseline_action_key="not-an-action",
            learned_shortlist_action_keys=shortlist,
            t2_policies=_policies(),
            config=_config(),
        )


def test_learned_order_does_not_change_fixed_set_search_semantics(monkeypatch) -> None:
    _patch_first_legal_native(monkeypatch)
    shortlist, baseline = _keys()
    first = evaluate_t1_second_shortlist_search(
        _root(),
        t2_policies=_policies(),
        baseline_action_key=baseline,
        learned_shortlist_action_keys=shortlist,
        config=_config(),
    )
    second = evaluate_t1_second_shortlist_search(
        _root(),
        t2_policies=_policies(),
        baseline_action_key=baseline,
        learned_shortlist_action_keys=list(reversed(shortlist)),
        config=_config(),
    )
    first_values = {
        row["action_key"]: (
            row["candidate_mean"],
            row["evaluation_mean"],
        )
        for row in first["actions"]
    }
    second_values = {
        row["action_key"]: (
            row["candidate_mean"],
            row["evaluation_mean"],
        )
        for row in second["actions"]
    }
    assert first_values == second_values
    assert first["proposed_action_key"] == second["proposed_action_key"]


def test_public_api_has_no_raw_hidden_discard_argument(monkeypatch) -> None:
    _patch_first_legal_native(monkeypatch)
    shortlist, baseline = _keys()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        evaluate_t1_second_shortlist_search(
            _root(),
            t2_policies=_policies(),
            baseline_action_key=baseline,
            learned_shortlist_action_keys=shortlist,
            config=_config(),
            opponent_private_discards=("As",),  # type: ignore[call-arg]
        )
