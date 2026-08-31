import json
from types import SimpleNamespace

import numpy as np
import torch
import pytest

import ai.tutor.hybrid_t1t2 as hybrid_t1t2
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board, Observation, encode_state
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.models.action_value_reranker import (
    BlendedActionValueReranker,
    CascadeSwitchActionValueReranker,
    ConditionalSwitchActionValueReranker,
    context_gate_accepts,
    encoded_state_gate_mask,
)
from ai.tutor.hybrid_t1t2 import (
    DummyPolicy,
    HybridConfig,
    _gate_accepts_override,
    _margin_accepts_override,
    _sync_refinement_candidates,
    _t1_bottom_sparse_rescue_candidate,
    _t1_blend_challenger_candidate,
    _t1_extra_refinement_candidates,
    _t1_low_risk_blend_challenger_candidate,
    _t1_model_rescue_candidate,
    _t1_refined_challenger_candidate,
    _t1_no_refine_fallback_candidate,
    _t2_middle_fill_bottom_shift_rescue_candidate,
    _t2_model_rank_rescue_candidate,
    _t2_model_top1_bust_rescue_candidate,
    _t2_model_top1_rescue_candidate,
    _t2_post_refine_policy_accepts,
    annotate_t2_forced_bust_candidates,
    build_shortlist,
    evaluate_hybrid_position,
    refine_t2_candidates_exact_partial,
)


class ScriptedActionValue(torch.nn.Module):
    def __init__(self, scores, bust=None, fl=None, fl_types=None):
        super().__init__()
        self.scores = list(scores)
        self.bust = list(bust) if bust is not None else [0.0] * len(self.scores)
        self.fl = list(fl) if fl is not None else [0.0] * len(self.scores)
        self.fl_types = fl_types

    def predict_components(self, batch, turn=None):
        n = batch.shape[0]
        scores = torch.tensor(self.scores[:n], dtype=torch.float32, device=batch.device)
        bust = torch.tensor(self.bust[:n], dtype=torch.float32, device=batch.device)
        fl = torch.tensor(self.fl[:n], dtype=torch.float32, device=batch.device)
        if self.fl_types is None:
            fl_types = torch.zeros((n, 4), dtype=torch.float32, device=batch.device)
        else:
            fl_types = torch.tensor(self.fl_types[:n], dtype=torch.float32, device=batch.device)
        return {
            "score": scores,
            "bust_prob": bust,
            "fl_prob": fl,
            "fl_type_probs": fl_types,
        }


def test_blended_action_value_reranker_blends_scores_and_probabilities():
    model_a = ScriptedActionValue(
        scores=[10.0, 0.0],
        bust=[0.2, 0.4],
        fl=[0.1, 0.3],
        fl_types=[[0.0, 0.2, 0.4, 0.6], [0.8, 0.6, 0.4, 0.2]],
    )
    model_b = ScriptedActionValue(
        scores=[2.0, 8.0],
        bust=[0.6, 0.8],
        fl=[0.5, 0.7],
        fl_types=[[1.0, 0.8, 0.6, 0.4], [0.2, 0.4, 0.6, 0.8]],
    )
    blended = BlendedActionValueReranker(model_a, model_b, model_b_weight=0.25)

    out = blended.predict_components(torch.zeros((2, 3)), turn=torch.tensor([3, 3]))

    assert torch.allclose(out["score"], torch.tensor([8.0, 2.0]))
    assert torch.allclose(out["bust_prob"], torch.tensor([0.3, 0.5]))
    assert torch.allclose(out["fl_prob"], torch.tensor([0.2, 0.4]))
    assert torch.allclose(out["fl_type_probs"][0], torch.tensor([0.25, 0.35, 0.45, 0.55]))
    assert torch.allclose(out["fl_aa"], out["fl_type_probs"][:, 2])


def test_opp_top_len_le_1_gate_accepts_context_and_encoded_state():
    obs = _obs(turn=2)

    assert context_gate_accepts(obs, "opp_top_len_le_1_context")
    assert context_gate_accepts(obs, "opp_top_len_le_1_dealt_max_rank_ge_j_context")
    state = torch.tensor(np.stack([encode_state(obs)]), dtype=torch.float32)
    assert bool(encoded_state_gate_mask(state, "opp_top_len_le_1_state")[0])

    obs.dealt_cards = ["9h", "8d", "3c"]
    assert not context_gate_accepts(obs, "opp_top_len_le_1_dealt_max_rank_ge_j_context")
    obs.dealt_cards = ["Ah", "Kd", "Qc"]
    obs.board_opponent.top.append("Qs")
    assert not context_gate_accepts(obs, "opp_top_len_le_1_context")
    assert not context_gate_accepts(obs, "opp_top_len_le_1_dealt_max_rank_ge_j_context")
    state = torch.tensor(np.stack([encode_state(obs)]), dtype=torch.float32)
    assert not bool(encoded_state_gate_mask(state, "opp_top_len_le_1_state")[0])


def test_composite_middle_gate_accepts_context_and_encoded_state():
    obs = _obs(turn=2)
    assert not context_gate_accepts(
        obs,
        "opp_top_len_le_1_dealt_max_rank_ge_j_own_middle_len_ge_3_context",
    )

    obs.board_self.middle.append("4d")
    assert context_gate_accepts(
        obs,
        "opp_top_len_le_1_dealt_max_rank_ge_j_own_middle_len_ge_3_context",
    )
    state = torch.tensor(np.stack([encode_state(obs)]), dtype=torch.float32)
    assert bool(
        encoded_state_gate_mask(
            state,
            "opp_top_len_le_1_dealt_max_rank_ge_j_own_middle_len_ge_3_state",
        )[0]
    )


def test_conditional_switch_action_value_reranker_uses_challenger_only_on_gate():
    base = ScriptedActionValue(scores=[1.0, 2.0])
    challenger = ScriptedActionValue(scores=[10.0, 20.0])
    switch = ConditionalSwitchActionValueReranker(base, challenger, gate="all")

    state = torch.zeros((2, 3), dtype=torch.float32)
    out = switch.predict_components_with_gate_mask(
        state,
        gate_mask=torch.tensor([True, False]),
        turn=torch.tensor([2, 2]),
    )

    assert torch.allclose(out["score"], torch.tensor([10.0, 2.0]))


def test_cascade_switch_action_value_reranker_applies_switches_in_order():
    base = ScriptedActionValue(scores=[1.0, 2.0, 3.0])
    first = ScriptedActionValue(scores=[10.0, 20.0, 30.0])
    second = ScriptedActionValue(scores=[100.0, 200.0, 300.0])
    cascade = CascadeSwitchActionValueReranker(
        base,
        [("first_context", first), ("second_context", second)],
    )

    out = cascade.predict_components_with_gate_masks(
        torch.zeros((3, 3), dtype=torch.float32),
        gate_masks={
            "first_context": torch.tensor([True, True, False]),
            "second_context": torch.tensor([False, True, False]),
        },
        turn=torch.tensor([2, 2, 2]),
    )

    assert torch.allclose(out["score"], torch.tensor([10.0, 100.0, 3.0]))


def test_parse_turn_model_blend_specs():
    parsed = hybrid_t1t2.parse_turn_model_blend_specs(["3=old.pt,new.pt,0.25"])

    assert parsed == {3: ("old.pt", "new.pt", 0.25)}

    with pytest.raises(ValueError):
        hybrid_t1t2.parse_turn_model_blend_specs(["3=old.pt,new.pt,1.5"])


def test_parse_turn_model_conditional_switch_ensemble_specs():
    parsed = hybrid_t1t2.parse_turn_model_conditional_switch_ensemble_specs(
        ["2=base_a.pt,base_b.pt;0.7,0.3;challenger_a.pt,challenger_b.pt;0.4,0.6;opp_top_len_le_1_context"]
    )

    assert parsed == {
        2: (
            ["base_a.pt", "base_b.pt"],
            [0.7, 0.3],
            ["challenger_a.pt", "challenger_b.pt"],
            [0.4, 0.6],
            "opp_top_len_le_1_context",
        )
    }

    with pytest.raises(ValueError):
        hybrid_t1t2.parse_turn_model_conditional_switch_ensemble_specs(["2=base.pt;1.0;challenger.pt;bad"])


def test_parse_turn_model_cascade_switch_ensemble_specs():
    parsed = hybrid_t1t2.parse_turn_model_cascade_switch_ensemble_specs(
        [
            "2=base_a.pt,base_b.pt;0.7,0.3;"
            "gate_a|challenger_a.pt|1.0;"
            "gate_b|challenger_b.pt,challenger_c.pt|0.4,0.6"
        ]
    )

    assert parsed == {
        2: (
            ["base_a.pt", "base_b.pt"],
            [0.7, 0.3],
            [
                ("gate_a", ["challenger_a.pt"], [1.0]),
                ("gate_b", ["challenger_b.pt", "challenger_c.pt"], [0.4, 0.6]),
            ],
        )
    }

    with pytest.raises(ValueError):
        hybrid_t1t2.parse_turn_model_cascade_switch_ensemble_specs(["2=base.pt;1.0;bad"])


def test_apply_hybrid_runtime_config_loads_default_mode(tmp_path):
    runtime_config = tmp_path / "runtime.json"
    runtime_config.write_text(
        json.dumps(
            {
                "default_mode": "k10",
                "models": {
                    "t1_action_value": "model.pt",
                    "t2_sync_selector": "t2_selector.json",
                    "t1_final_selector": "selector.json",
                    "t1_override_gate": "gate.json",
                },
                "k10": {
                    "sync_exact_k": 10,
                    "t1_refinement": "recursive_mc",
                    "t1_model_rescue_model_score_min": 15.0,
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        runtime_config=str(runtime_config),
        runtime_mode="",
        model="",
        sync_exact_k=3,
        t1_refinement="none",
        t2_sync_selector="",
        t1_final_selector="",
        t1_override_gate="",
        t1_model_rescue_model_score_min=-1.0,
    )

    mode_name, mode = hybrid_t1t2.apply_hybrid_runtime_config(args, provided_dests=set())

    assert mode_name == "k10"
    assert mode["sync_exact_k"] == 10
    assert args.model == "model.pt"
    assert args.sync_exact_k == 10
    assert args.t1_refinement == "recursive_mc"
    assert args.t2_sync_selector == "t2_selector.json"
    assert args.t1_final_selector == "selector.json"
    assert args.t1_override_gate == "gate.json"
    assert args.t1_model_rescue_model_score_min == 15.0


def test_apply_hybrid_runtime_config_accepts_generic_action_value_model(tmp_path):
    runtime_config = tmp_path / "runtime.json"
    runtime_config.write_text(
        json.dumps(
            {
                "default_mode": "t2",
                "models": {"action_value": "t2-model.pt"},
                "t2": {"t2_exact_backend": "t3_union"},
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        runtime_config=str(runtime_config),
        runtime_mode="",
        model="",
        t2_exact_backend="full_exact",
    )

    mode_name, mode = hybrid_t1t2.apply_hybrid_runtime_config(args, provided_dests=set())

    assert mode_name == "t2"
    assert mode["t2_exact_backend"] == "t3_union"
    assert args.model == "t2-model.pt"
    assert args.t2_exact_backend == "t3_union"


def test_apply_hybrid_runtime_config_inherits_mode_and_preserves_cli_overrides(tmp_path):
    runtime_config = tmp_path / "runtime.json"
    runtime_config.write_text(
        json.dumps(
            {
                "default_mode": "k10",
                "models": {"t1_action_value": "model.pt"},
                "k10": {
                    "sync_exact_k": 10,
                    "t1_refinement": "recursive_mc",
                },
                "optional_modes": {
                    "k12": {
                        "inherits": "k10",
                        "sync_exact_k": 12,
                        "use_case": "test",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        runtime_config=str(runtime_config),
        runtime_mode="k12",
        model="manual.pt",
        sync_exact_k=99,
        t1_refinement="none",
    )

    mode_name, mode = hybrid_t1t2.apply_hybrid_runtime_config(
        args,
        provided_dests={"model", "sync_exact_k"},
    )

    assert mode_name == "k12"
    assert mode["sync_exact_k"] == 12
    assert args.model == "manual.pt"
    assert args.sync_exact_k == 99
    assert args.t1_refinement == "recursive_mc"


def test_make_action_value_evaluator_turn_blend_overrides_turn_model(monkeypatch):
    base_model = ScriptedActionValue(scores=[0.0])
    single_turn_model = ScriptedActionValue(scores=[1.0])
    blended_turn_model = ScriptedActionValue(scores=[2.0])

    def fake_load(path, device="cpu"):
        assert path == "single.pt"
        return single_turn_model

    def fake_load_blend(model_a_path, model_b_path, model_b_weight, device="cpu"):
        assert (model_a_path, model_b_path, model_b_weight) == ("old.pt", "new.pt", 0.25)
        return blended_turn_model

    monkeypatch.setattr(hybrid_t1t2, "load_action_value_model", fake_load)
    monkeypatch.setattr(hybrid_t1t2, "load_blended_action_value_model", fake_load_blend)

    evaluator = hybrid_t1t2.make_action_value_evaluator(
        model=base_model,
        model_paths_by_turn={3: "single.pt"},
        model_blends_by_turn={3: ("old.pt", "new.pt", 0.25)},
        device="cpu",
    )

    assert evaluator._action_value_net_for_turn(3) is blended_turn_model


def test_make_action_value_evaluator_turn_switch_overrides_ensemble(monkeypatch):
    base_model = ScriptedActionValue(scores=[0.0])
    ensemble_turn_model = ScriptedActionValue(scores=[1.0])
    switch_turn_model = ScriptedActionValue(scores=[2.0])

    def fake_load(path, device="cpu"):
        assert path == "base.pt"
        return base_model

    def fake_load_ensemble(model_paths, weights, device="cpu"):
        assert model_paths == ["old.pt", "new.pt"]
        assert weights == [0.5, 0.5]
        return ensemble_turn_model

    def fake_load_switch(base_paths, base_weights, challenger_paths, challenger_weights, gate, device="cpu"):
        assert base_paths == ["base_a.pt"]
        assert base_weights == [1.0]
        assert challenger_paths == ["challenger_a.pt"]
        assert challenger_weights == [1.0]
        assert gate == "opp_top_len_le_1_context"
        return switch_turn_model

    monkeypatch.setattr(hybrid_t1t2, "load_action_value_model", fake_load)
    monkeypatch.setattr(hybrid_t1t2, "load_weighted_action_value_ensemble", fake_load_ensemble)
    monkeypatch.setattr(hybrid_t1t2, "load_conditional_switch_action_value_ensemble", fake_load_switch)

    evaluator = hybrid_t1t2.make_action_value_evaluator(
        model_path="base.pt",
        model_ensembles_by_turn={2: (["old.pt", "new.pt"], [0.5, 0.5])},
        model_conditional_switch_ensembles_by_turn={
            2: (["base_a.pt"], [1.0], ["challenger_a.pt"], [1.0], "opp_top_len_le_1_context")
        },
        device="cpu",
    )

    assert evaluator._action_value_net_for_turn(2) is switch_turn_model


def test_make_action_value_evaluator_turn_cascade_overrides_switch(monkeypatch):
    base_model = ScriptedActionValue(scores=[0.0])
    switch_turn_model = ScriptedActionValue(scores=[1.0])
    cascade_turn_model = ScriptedActionValue(scores=[2.0])

    def fake_load(path, device="cpu"):
        assert path == "base.pt"
        return base_model

    def fake_load_switch(base_paths, base_weights, challenger_paths, challenger_weights, gate, device="cpu"):
        return switch_turn_model

    def fake_load_cascade(base_paths, base_weights, switch_specs, device="cpu"):
        assert base_paths == ["base_a.pt"]
        assert base_weights == [1.0]
        assert switch_specs == [("gate_a", ["challenger_a.pt"], [1.0])]
        return cascade_turn_model

    monkeypatch.setattr(hybrid_t1t2, "load_action_value_model", fake_load)
    monkeypatch.setattr(hybrid_t1t2, "load_conditional_switch_action_value_ensemble", fake_load_switch)
    monkeypatch.setattr(hybrid_t1t2, "load_cascade_switch_action_value_ensemble", fake_load_cascade)

    evaluator = hybrid_t1t2.make_action_value_evaluator(
        model_path="base.pt",
        model_conditional_switch_ensembles_by_turn={
            2: (["switch_base.pt"], [1.0], ["switch_challenger.pt"], [1.0], "gate_b")
        },
        model_cascade_switch_ensembles_by_turn={
            2: (["base_a.pt"], [1.0], [("gate_a", ["challenger_a.pt"], [1.0])])
        },
        device="cpu",
    )

    assert evaluator._action_value_net_for_turn(2) is cascade_turn_model


def _obs(turn=2):
    return Observation(
        board_self=Board(
            top=["As"],
            middle=["2h", "3h"],
            bottom=["4h", "5h"],
        ),
        board_opponent=Board(
            top=["Ks"],
            middle=["6h", "7h"],
            bottom=["8h", "9h"],
        ),
        dealt_cards=["Ah", "Kd", "Qc"],
        known_discards_self=["2c"],
        turn=turn,
        is_btn=True,
    )


def _evaluator(model):
    return RolloutEvaluator(
        policy_net=DummyPolicy(),
        action_value_net=model,
        full_width=True,
        n_rollouts=0,
    )


def test_shortlist_keeps_model_top15_and_insurance_candidates():
    obs = _obs(turn=2)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    n = len(valid_actions)
    assert n == 27
    scores = list(range(n))
    bust = [0.01, 0.02, 0.03] + [0.5] * (n - 3)
    fl = [0.0] * n
    fl[3] = 0.95
    fl[4] = 0.90
    model = ScriptedActionValue(scores=scores, bust=bust, fl=fl)

    result = evaluate_hybrid_position(
        obs,
        evaluator=_evaluator(model),
        config=HybridConfig(enable_sync_refinement=False),
    )

    indices = {candidate["action_idx"] for candidate in result["candidates"]}
    assert result["candidate_pool_size"] == 20
    assert set(range(12, 27)).issubset(indices)
    assert {0, 1, 2, 3, 4}.issubset(indices)


def test_sync_keeps_high_bust_model_top_candidates_and_adds_safety_insurance():
    obs = _obs(turn=2)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    n = len(valid_actions)
    scores = [100.0 - i for i in range(n)]
    bust = [1.0, 1.0, 1.0] + [0.1] * (n - 3)
    model = ScriptedActionValue(scores=scores, bust=bust)

    result = evaluate_hybrid_position(
        obs,
        evaluator=_evaluator(model),
        config=HybridConfig(enable_sync_refinement=False),
    )

    assert result["sync_exact_action_indices"]
    assert {0, 1, 2}.issubset(set(result["sync_exact_action_indices"]))
    assert 3 in set(result["sync_exact_action_indices"])


def test_t2_sync_model_insurance_protects_model_top_with_tactical_insurance():
    candidates = []
    for i in range(12):
        candidates.append(
            {
                "action_idx": i,
                "model_score": float(100 - i),
                "model_rank": i + 1,
                "refinement_rank": 20 + i if i < 3 else i,
                "predicted_bust": 0.5,
                "predicted_fl": 0.0,
                "predicted_fl_types": {"qq": 0.0, "kk": 0.0, "aa": 0.0, "trips": 0.0},
            }
        )
    candidates[9]["insurance_reason"] = "low_bust"
    candidates[9]["predicted_bust"] = 0.01
    candidates[10]["insurance_reason"] = "low_bust"
    candidates[10]["predicted_bust"] = 0.02

    selected = _sync_refinement_candidates(
        candidates,
        HybridConfig(
            sync_exact_k=5,
            t2_sync_model_insurance_k=3,
            t2_tactical_insurance_k=2,
        ),
        turn=2,
    )

    selected_indices = {item["action_idx"] for item in selected}
    assert {0, 1, 2}.issubset(selected_indices)
    assert {9, 10}.issubset(selected_indices)
    assert len(selected) == 7
    assert [item["action_idx"] for item in selected[:5]] == [0, 9, 1, 10, 2]


def test_t2_sync_model_insurance_kk_gate_controls_interleave_order():
    candidates = []
    for i in range(12):
        candidates.append(
            {
                "action_idx": i,
                "model_score": float(100 - i),
                "model_rank": i + 1,
                "refinement_rank": 20 + i if i < 3 else i,
                "predicted_bust": 0.5,
                "predicted_fl": 0.0,
                "predicted_fl_types": {"qq": 0.0, "kk": 0.0, "aa": 0.0, "trips": 0.0},
            }
        )
    candidates[0]["predicted_fl_types"]["kk"] = 0.05
    candidates[9]["insurance_reason"] = "low_bust"
    candidates[9]["predicted_bust"] = 0.01
    candidates[10]["insurance_reason"] = "low_bust"
    candidates[10]["predicted_bust"] = 0.02

    enabled = _sync_refinement_candidates(
        candidates,
        HybridConfig(
            sync_exact_k=5,
            t2_sync_model_insurance_k=3,
            t2_sync_model_insurance_top1_kk_min=0.03,
            t2_tactical_insurance_k=2,
        ),
        turn=2,
    )
    disabled = _sync_refinement_candidates(
        candidates,
        HybridConfig(
            sync_exact_k=5,
            t2_sync_model_insurance_k=3,
            t2_sync_model_insurance_top1_kk_min=0.10,
            t2_tactical_insurance_k=2,
        ),
        turn=2,
    )

    assert [item["action_idx"] for item in enabled[:5]] == [0, 9, 1, 10, 2]
    assert 0 not in {item["action_idx"] for item in disabled}


def test_rollout_evaluator_details_include_components_and_adjusted_score():
    obs = _obs(turn=1)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    model = ScriptedActionValue(
        scores=[10.0] * len(valid_actions),
        bust=[0.5] * len(valid_actions),
        fl=[0.25] * len(valid_actions),
    )
    evaluator = RolloutEvaluator(
        policy_net=DummyPolicy(),
        action_value_net=model,
        action_value_bust_weight=2.0,
        action_value_fl_any_weight=4.0,
    )

    details = evaluator.score_candidates_action_value_details(obs, list(enumerate(valid_actions)))

    assert len(details["model_score"]) == len(valid_actions)
    assert details["raw_score"][0] == 10.0
    assert details["bust_prob"][0] == 0.5
    assert details["fl_prob"][0] == 0.25
    assert details["model_score"][0] == 10.0


def test_build_shortlist_respects_max_pool_even_with_large_insurance():
    candidates = [
        {
            "action_idx": i,
            "model_score": float(i),
            "predicted_bust": 0.0 if i < 10 else 0.5,
            "predicted_fl": 1.0 if 10 <= i < 20 else 0.0,
            "predicted_fl_types": {"qq": 0.0, "kk": 0.0, "aa": 0.0, "trips": 0.0},
        }
        for i in range(40)
    ]

    pool = build_shortlist(candidates, config=HybridConfig(insurance_k=20, max_pool=20))

    assert len(pool) == 20
    assert {candidate["action_idx"] for candidate in candidates[-15:]}.issubset(
        {candidate["action_idx"] for candidate in pool}
    )


def test_build_shortlist_can_add_t2_aux_shortlist_candidate():
    candidates = [
        {
            "action_idx": 0,
            "turn": 2,
            "model_score": 10.0,
            "aux_model_score": 1.0,
            "predicted_bust": 0.2,
            "predicted_fl": 0.0,
            "predicted_fl_types": {"qq": 0.0, "kk": 0.0, "aa": 0.0, "trips": 0.0},
        },
        {
            "action_idx": 1,
            "turn": 2,
            "model_score": 9.0,
            "aux_model_score": 2.0,
            "predicted_bust": 0.2,
            "predicted_fl": 0.0,
            "predicted_fl_types": {"qq": 0.0, "kk": 0.0, "aa": 0.0, "trips": 0.0},
        },
        {
            "action_idx": 2,
            "turn": 2,
            "model_score": 1.0,
            "aux_model_score": 100.0,
            "predicted_bust": 0.2,
            "predicted_fl": 0.0,
            "predicted_fl_types": {"qq": 0.0, "kk": 0.0, "aa": 0.0, "trips": 0.0},
        },
    ]

    pool = build_shortlist(
        candidates,
        config=HybridConfig(shortlist_k=1, insurance_k=0, max_pool=2, t2_aux_shortlist_k=1),
    )

    assert [candidate["action_idx"] for candidate in pool] == [0, 2]
    assert candidates[2]["aux_model_rank"] == 1
    assert candidates[2]["insurance_reason"] == "aux_shortlist"


def test_t2_forced_bust_candidates_are_marked_and_skipped_for_sync():
    obs = Observation(
        board_self=Board(
            top=["3c"],
            middle=["2d", "Js"],
            bottom=["6d", "7d", "8d", "2s"],
        ),
        board_opponent=Board(
            top=["X1"],
            middle=["X2", "Qd"],
            bottom=["5c", "5s", "7s", "5h"],
        ),
        dealt_cards=["9d", "8s", "7h"],
        known_discards_self=["6c"],
        turn=2,
        is_btn=True,
    )
    forced = {
        "action_idx": 1,
        "model_rank": 1,
        "refinement_rank": 1,
        "model_score": 10.0,
        "predicted_bust": 0.0,
        "board": {
            "top": ["3c"],
            "middle": ["2d", "Js", "8s"],
            "bottom": ["6d", "7d", "8d", "2s", "9d"],
        },
        "_action": SimpleNamespace(discard="7h"),
    }
    playable = {
        "action_idx": 2,
        "model_rank": 2,
        "refinement_rank": 2,
        "model_score": 9.0,
        "predicted_bust": 0.1,
        "board": {
            "top": ["3c"],
            "middle": ["2d", "Js", "7h"],
            "bottom": ["6d", "7d", "8d", "2s", "8s"],
        },
        "_action": SimpleNamespace(discard="9d"),
    }
    filler = {
        "action_idx": 3,
        "model_rank": 3,
        "refinement_rank": 3,
        "model_score": 8.0,
        "predicted_bust": 0.2,
        "board": playable["board"],
        "_action": SimpleNamespace(discard="9d"),
    }

    candidates = [forced, playable, filler]
    annotate_t2_forced_bust_candidates(obs, candidates)
    selected = _sync_refinement_candidates(
        candidates,
        HybridConfig(sync_exact_k=2),
        turn=2,
    )

    assert forced["forced_bust"] is True
    assert playable["forced_bust"] is False
    assert 1 not in {item["action_idx"] for item in selected}
    assert [item["action_idx"] for item in selected] == [2, 3]


def test_t2_sync_selector_can_reorder_sync_candidates():
    candidates = [
        {
            "action_idx": i,
            "model_rank": i + 1,
            "refinement_rank": i + 1,
            "model_score": float(10 - i),
            "predicted_bust": 0.1,
            "sync_selector_score": float(i),
        }
        for i in range(5)
    ]

    selected = _sync_refinement_candidates(
        candidates,
        HybridConfig(sync_exact_k=2, t2_sync_selection_policy="selector"),
        turn=2,
    )

    assert [item["action_idx"] for item in selected] == [4, 3]


def test_t2_adaptive_sync_expands_only_low_model_score_positions():
    obs = _obs(turn=2)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    n = len(valid_actions)
    low_confidence_model = ScriptedActionValue(scores=[1.0 - i * 0.1 for i in range(n)])

    expanded = evaluate_hybrid_position(
        obs,
        evaluator=_evaluator(low_confidence_model),
        config=HybridConfig(
            shortlist_k=15,
            insurance_k=5,
            max_pool=20,
            sync_exact_k=3,
            t2_sync_exact_k=15,
            t2_adaptive_shortlist_k=18,
            t2_adaptive_sync_exact_k=18,
            t2_adaptive_max_model_score=2.0,
            enable_sync_refinement=False,
        ),
    )

    assert expanded["t2_adaptive_sync_applied"]
    assert expanded["effective_shortlist_k"] == 18
    assert expanded["sync_refinement_candidate_count"] == 18

    high_confidence_model = ScriptedActionValue(scores=[10.0 - i * 0.1 for i in range(n)])
    unchanged = evaluate_hybrid_position(
        obs,
        evaluator=_evaluator(high_confidence_model),
        config=HybridConfig(
            shortlist_k=15,
            insurance_k=5,
            max_pool=20,
            sync_exact_k=3,
            t2_sync_exact_k=15,
            t2_adaptive_shortlist_k=18,
            t2_adaptive_sync_exact_k=18,
            t2_adaptive_max_model_score=2.0,
            enable_sync_refinement=False,
        ),
    )

    assert not unchanged["t2_adaptive_sync_applied"]
    assert unchanged["effective_shortlist_k"] == 15
    assert unchanged["sync_refinement_candidate_count"] == 15


def test_t2_post_refine_expands_extra_candidates_when_time_remains(monkeypatch):
    obs = _obs(turn=2)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    n = len(valid_actions)
    model = ScriptedActionValue(scores=[float(100 - i) for i in range(n)])
    calls = []

    def fake_refine(obs, candidates, *, config, rust_solver_path=None, started_at=None, t3_pooler=None):
        calls.append([int(candidate["action_idx"]) for candidate in candidates])
        for candidate in candidates:
            candidate["refined_score"] = float(1000 - int(candidate["model_rank"]))
            candidate["refinement_source"] = "exact_partial"
            candidate["refinement_backend"] = "t3_union"
            candidate["samples"] = int(candidate.get("samples", 0) or 0) + 1
        return {
            "exact_evaluated": len(candidates),
            "error": None,
            "t2_exact_backend": "t3_union",
            "samples_by_action": {str(candidate["action_idx"]): 1 for candidate in candidates},
        }

    monkeypatch.setattr(hybrid_t1t2, "refine_t2_candidates_exact_partial", fake_refine)

    result = evaluate_hybrid_position(
        obs,
        evaluator=_evaluator(model),
        config=HybridConfig(
            shortlist_k=6,
            insurance_k=0,
            max_pool=6,
            sync_exact_k=2,
            t2_sync_exact_k=2,
            t2_post_refine_sync_exact_k=4,
            t2_post_refine_min_remaining_ms=0,
            t2_refinement="exact_partial",
            t2_exact_backend="t3_union",
            time_budget_ms=5000,
        ),
    )

    assert len(calls) == 2
    assert len(calls[0]) == 2
    assert len(calls[1]) == 2
    assert result["exact_evaluated"] == 4
    assert result["t2_post_refine_expanded"] is True
    assert result["t2_post_refine_candidate_count"] == 2


def test_t2_post_refine_structured_top_policy_accepts_pair_or_joker():
    obs = _obs(turn=2)
    obs.board_self.top = ["Kc", "Kd"]
    accepted, details = _t2_post_refine_policy_accepts(
        obs,
        [{"model_score": 5.01}],
        HybridConfig(
            t2_post_refine_policy="structured_top_model_min",
            t2_post_refine_extra_model_score_min=5.0,
        ),
    )

    assert accepted is True
    assert details["own_top_structured"] is True

    obs.board_self.top = ["X1"]
    accepted, details = _t2_post_refine_policy_accepts(
        obs,
        [{"model_score": 5.01}],
        HybridConfig(
            t2_post_refine_policy="structured_top_model_min",
            t2_post_refine_extra_model_score_min=5.0,
        ),
    )

    assert accepted is True
    assert details["own_top_structured"] is True


def test_t2_post_refine_structured_top_policy_rejects_plain_or_low_score():
    obs = _obs(turn=2)
    obs.board_self.top = ["Kc"]
    accepted, details = _t2_post_refine_policy_accepts(
        obs,
        [{"model_score": 9.0}],
        HybridConfig(
            t2_post_refine_policy="structured_top_model_min",
            t2_post_refine_extra_model_score_min=5.0,
        ),
    )

    assert accepted is False
    assert details["reject_reason"] == "own_top_not_structured"

    obs.board_self.top = ["Kc", "Kd"]
    accepted, details = _t2_post_refine_policy_accepts(
        obs,
        [{"model_score": 4.99}],
        HybridConfig(
            t2_post_refine_policy="structured_top_model_min",
            t2_post_refine_extra_model_score_min=5.0,
        ),
    )

    assert accepted is False
    assert details["reject_reason"] == "extra_model_score_below_min"


def test_t2_selection_uses_structured_top_model_weight(monkeypatch):
    obs = _obs(turn=2)
    obs.board_self.top = ["Ts", "Td"]
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    n = len(valid_actions)
    model = ScriptedActionValue(scores=[10.0, 4.0] + [0.0] * max(0, n - 2))

    def fake_refine(obs, candidates, *, config, rust_solver_path=None, started_at=None, t3_pooler=None):
        for candidate in candidates:
            if int(candidate["model_rank"]) == 1:
                candidate["refined_score"] = 10.0
            elif int(candidate["model_rank"]) == 2:
                candidate["refined_score"] = 14.0
            else:
                candidate["refined_score"] = 0.0
            candidate["refinement_source"] = "exact_partial"
            candidate["samples"] = 1
        return {"exact_evaluated": len(candidates), "error": None}

    monkeypatch.setattr(hybrid_t1t2, "refine_t2_candidates_exact_partial", fake_refine)

    result = evaluate_hybrid_position(
        obs,
        evaluator=_evaluator(model),
        config=HybridConfig(
            shortlist_k=2,
            insurance_k=0,
            max_pool=2,
            sync_exact_k=2,
            t2_sync_exact_k=2,
            t2_refinement="exact_partial",
            t2_selection_policy="refined_plus_model",
            t2_selection_model_weight=0.5,
            t2_selection_structured_top_model_weight=1.0,
        ),
    )

    assert result["best"]["model_rank"] == 1
    assert result["best"]["t2_selection_model_weight_used"] == 1.0
    assert result["t2_own_top_structured"] is True


def test_t2_selection_can_use_final_selector(monkeypatch):
    obs = _obs(turn=2)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    n = len(valid_actions)
    model = ScriptedActionValue(scores=[10.0, 4.0] + [0.0] * max(0, n - 2))

    def fake_refine(obs, candidates, *, config, rust_solver_path=None, started_at=None, t3_pooler=None):
        for candidate in candidates:
            if int(candidate["model_rank"]) == 1:
                candidate["refined_score"] = 10.0
            elif int(candidate["model_rank"]) == 2:
                candidate["refined_score"] = 14.0
            else:
                candidate["refined_score"] = 0.0
            candidate["refinement_source"] = "exact_partial"
            candidate["samples"] = 1
        return {"exact_evaluated": len(candidates), "error": None}

    monkeypatch.setattr(hybrid_t1t2, "refine_t2_candidates_exact_partial", fake_refine)

    result = evaluate_hybrid_position(
        obs,
        evaluator=_evaluator(model),
        config=HybridConfig(
            shortlist_k=2,
            insurance_k=0,
            max_pool=2,
            sync_exact_k=2,
            t2_sync_exact_k=2,
            t2_refinement="exact_partial",
            t2_selection_policy="selector",
            t2_final_selector={
                "name": "prefer_model_score",
                "features": ["model_score"],
                "weights": [1.0],
                "intercept": 0.0,
                "means": {},
                "scales": {},
            },
        ),
    )

    assert result["best"]["model_rank"] == 1
    assert result["best"]["t2_final_selector_score"] == pytest.approx(10.0)
    assert result["t2_final_selector_name"] == "prefer_model_score"


def test_t2_exact_partial_uses_configured_initial_samples(monkeypatch):
    obs = _obs(turn=2)
    actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    candidates = [
        {
            "action_idx": i,
            "_action": actions[i],
            "model_rank": i + 1,
            "model_score": 10.0 - i,
            "predicted_bust": 0.1,
            "predicted_fl": 0.0,
            "samples": 0,
            "elapsed_ms": 0.0,
        }
        for i in range(2)
    ]
    batches = []

    def fake_sample_t3_draws(obs, action, action_idx, sample_count, *, sample_offset=0):
        return [["2d", "3d", "4d"] for _ in range(sample_count)]

    def fake_run(cmd, cwd=None, check=None, stdout=None, stderr=None, text=None, timeout=None):
        input_path = cmd[cmd.index("--input") + 1]
        output_path = cmd[cmd.index("--output") + 1]
        rows = [line for line in open(input_path, encoding="utf-8").read().splitlines() if line.strip()]
        batches.append(len(rows))
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, _row in enumerate(rows):
                f.write(
                    json.dumps(
                        {
                            "record_index": idx,
                            "turn": 3,
                            "legal_actions": 1,
                            "best": {"metrics": {"score": 5.0 + idx}},
                        }
                    )
                    + "\n"
                )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(hybrid_t1t2, "_sample_t3_draws", fake_sample_t3_draws)
    monkeypatch.setattr(hybrid_t1t2.subprocess, "run", fake_run)

    result = refine_t2_candidates_exact_partial(
        obs,
        candidates,
        config=HybridConfig(
            t2_initial_samples_per_candidate=3,
            t2_extra_samples_per_round=2,
            t2_max_samples_per_candidate=3,
            time_budget_ms=5000,
        ),
        rust_solver_path=__file__,
    )

    assert result["exact_evaluated"] == 6
    assert batches == [6]
    assert result["samples_by_action"] == {"0": 3, "1": 3}
    assert {candidate["samples"] for candidate in candidates} == {3}
    assert all(candidate["refinement_source"] == "exact_partial" for candidate in candidates)


def test_t2_exact_partial_ensures_model_rank_min_samples(monkeypatch):
    obs = _obs(turn=2)
    actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    candidates = [
        {
            "action_idx": i,
            "_action": actions[i],
            "model_rank": i + 1,
            "model_score": 10.0 - i,
            "predicted_bust": 0.1,
            "predicted_fl": 0.0,
            "samples": 0,
            "elapsed_ms": 0.0,
        }
        for i in range(4)
    ]
    batches = []

    def fake_sample_t3_draws(obs, action, action_idx, sample_count, *, sample_offset=0):
        return [["2d", "3d", "4d"] for _ in range(sample_count)]

    def fake_run(cmd, cwd=None, check=None, stdout=None, stderr=None, text=None, timeout=None):
        input_path = cmd[cmd.index("--input") + 1]
        output_path = cmd[cmd.index("--output") + 1]
        rows = [line for line in open(input_path, encoding="utf-8").read().splitlines() if line.strip()]
        batches.append(len(rows))
        with open(output_path, "w", encoding="utf-8") as f:
            for _row in rows:
                f.write(json.dumps({"best": {"metrics": {"score": 5.0}}}) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(hybrid_t1t2, "_sample_t3_draws", fake_sample_t3_draws)
    monkeypatch.setattr(hybrid_t1t2.subprocess, "run", fake_run)

    result = refine_t2_candidates_exact_partial(
        obs,
        candidates,
        config=HybridConfig(
            t2_initial_samples_per_candidate=2,
            t2_extra_samples_per_round=1,
            t2_extra_margin=-1.0,
            t2_max_samples_per_candidate=5,
            t2_close_candidate_limit=0,
            t2_baseline_min_samples=0,
            t2_model_rank_min_samples_k=2,
            t2_model_rank_min_samples=5,
            t2_model_rank_min_samples_min_remaining_ms=0,
            time_budget_ms=5000,
        ),
        rust_solver_path=__file__,
    )

    assert result["samples_by_action"] == {"0": 5, "1": 5, "2": 2, "3": 2}
    assert result["model_rank_min_sample_batches"] == 1
    assert batches == [8, 6]


def test_t2_exact_partial_can_use_t3_union_backend(monkeypatch):
    obs = _obs(turn=2)
    actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    candidates = [
        {
            "action_idx": i,
            "_action": actions[i],
            "model_rank": i + 1,
            "model_score": 10.0 - i,
            "predicted_bust": 0.1,
            "predicted_fl": 0.0,
            "samples": 0,
            "elapsed_ms": 0.0,
        }
        for i in range(2)
    ]
    pooler_configs = []
    calls = []

    def fake_sample_t3_draws(obs, action, action_idx, sample_count, *, sample_offset=0):
        return [["2d", "3d", "4d"] for _ in range(sample_count)]

    class FakeT3Pooler:
        def __init__(self, config_path, device="auto"):
            pooler_configs.append((str(config_path), device))

    def fake_evaluate_t3_position(row, *, pooler, per_source_top_k, rust_solver_path, rust_timeout_s, deadline_s=None):
        calls.append(
            {
                "row": row,
                "pooler": pooler,
                "per_source_top_k": per_source_top_k,
                "rust_solver_path": str(rust_solver_path),
                "rust_timeout_s": rust_timeout_s,
                "deadline_s": deadline_s,
            }
        )
        return {
            "best": {"metrics": {"score": 20.0 + len(calls)}},
            "elapsed_ms": 12.0,
            "candidate_pool_size": 18,
            "exact_evaluated": 18,
        }

    import ai.tutor.t3_runtime as t3_runtime

    monkeypatch.setattr(hybrid_t1t2, "_sample_t3_draws", fake_sample_t3_draws)
    monkeypatch.setattr(t3_runtime, "T3UnionCandidatePool", FakeT3Pooler)
    monkeypatch.setattr(t3_runtime, "evaluate_t3_position", fake_evaluate_t3_position)

    result = refine_t2_candidates_exact_partial(
        obs,
        candidates,
        config=HybridConfig(
            t2_exact_backend="t3_union",
            t2_initial_samples_per_candidate=1,
            t2_extra_samples_per_round=1,
            t2_max_samples_per_candidate=1,
            time_budget_ms=5000,
            t3_pool_config="ai/config/t3_ev_loss_fresh_pool_20260607.json",
            t3_pool_k=10,
        ),
        rust_solver_path=__file__,
    )

    assert result["exact_evaluated"] == 2
    assert result["t2_exact_backend"] == "t3_union"
    assert result["t3_union_summary"]["samples"] == 2
    assert result["t3_union_summary"]["pool_size_max"] == 18
    assert len(pooler_configs) == 1
    assert len(calls) == 2
    assert {call["per_source_top_k"] for call in calls} == {10}
    assert all(call["deadline_s"] is not None for call in calls)
    assert {candidate["samples"] for candidate in candidates} == {1}
    assert all(candidate["refinement_backend"] == "t3_union" for candidate in candidates)


def test_t2_exact_partial_reuses_passed_t3_pooler(monkeypatch):
    obs = _obs(turn=2)
    actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    candidates = [
        {
            "action_idx": 0,
            "_action": actions[0],
            "model_rank": 1,
            "model_score": 10.0,
            "predicted_bust": 0.1,
            "predicted_fl": 0.0,
            "samples": 0,
            "elapsed_ms": 0.0,
        }
    ]
    shared_pooler = object()
    constructed = []
    seen_poolers = []

    def fake_sample_t3_draws(obs, action, action_idx, sample_count, *, sample_offset=0):
        return [["2d", "3d", "4d"]]

    class FakeT3Pooler:
        def __init__(self, config_path, device="auto"):
            constructed.append((config_path, device))

    def fake_evaluate_t3_position(row, *, pooler, per_source_top_k, rust_solver_path, rust_timeout_s, deadline_s=None):
        seen_poolers.append(pooler)
        return {
            "best": {"metrics": {"score": 21.0}},
            "elapsed_ms": 12.0,
            "candidate_pool_size": 18,
            "exact_evaluated": 18,
        }

    import ai.tutor.t3_runtime as t3_runtime

    monkeypatch.setattr(hybrid_t1t2, "_sample_t3_draws", fake_sample_t3_draws)
    monkeypatch.setattr(t3_runtime, "T3UnionCandidatePool", FakeT3Pooler)
    monkeypatch.setattr(t3_runtime, "evaluate_t3_position", fake_evaluate_t3_position)

    result = refine_t2_candidates_exact_partial(
        obs,
        candidates,
        config=HybridConfig(
            t2_exact_backend="t3_union",
            t2_initial_samples_per_candidate=1,
            t2_max_samples_per_candidate=1,
            time_budget_ms=5000,
            t3_pool_config="ai/config/t3_ev_loss_fresh_pool_20260607.json",
            t3_pool_k=10,
        ),
        rust_solver_path=__file__,
        t3_pooler=shared_pooler,
    )

    assert result["exact_evaluated"] == 1
    assert constructed == []
    assert seen_poolers == [shared_pooler]


def test_t2_t3_union_timeout_preserves_completed_refinements(monkeypatch):
    obs = _obs(turn=2)
    actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    candidates = [
        {
            "action_idx": i,
            "_action": actions[i],
            "model_rank": i + 1,
            "model_score": 10.0 - i,
            "predicted_bust": 0.1,
            "predicted_fl": 0.0,
            "samples": 0,
            "elapsed_ms": 0.0,
        }
        for i in range(2)
    ]
    calls = []

    def fake_sample_t3_draws(obs, action, action_idx, sample_count, *, sample_offset=0):
        return [["2d", "3d", "4d"]]

    class FakeT3Pooler:
        def __init__(self, config_path, device="auto"):
            pass

    def fake_evaluate_t3_position(row, *, pooler, per_source_top_k, rust_solver_path, rust_timeout_s, deadline_s=None):
        calls.append(row)
        if len(calls) == 2:
            raise TimeoutError("t3_pool:set:slow:time_budget_exhausted")
        return {
            "best": {"metrics": {"score": 42.0}},
            "elapsed_ms": 12.0,
            "candidate_pool_size": 18,
            "exact_evaluated": 18,
        }

    import ai.tutor.t3_runtime as t3_runtime

    monkeypatch.setattr(hybrid_t1t2, "_sample_t3_draws", fake_sample_t3_draws)
    monkeypatch.setattr(t3_runtime, "T3UnionCandidatePool", FakeT3Pooler)
    monkeypatch.setattr(t3_runtime, "evaluate_t3_position", fake_evaluate_t3_position)

    result = refine_t2_candidates_exact_partial(
        obs,
        candidates,
        config=HybridConfig(
            t2_exact_backend="t3_union",
            t2_initial_samples_per_candidate=1,
            t2_max_samples_per_candidate=1,
            time_budget_ms=5000,
            t3_pool_config="ai/config/t3_ev_loss_fresh_pool_20260607.json",
            t3_pool_k=10,
        ),
        rust_solver_path=__file__,
    )

    assert result["exact_evaluated"] == 1
    assert result["error"].startswith("t3_union_time_budget_exhausted:")
    assert candidates[0]["refined_score"] == 42.0
    assert candidates[0]["refinement_backend"] == "t3_union"
    assert candidates[0]["samples"] == 1
    assert candidates[1].get("refinement_source", "none") == "none"


def test_t1_override_margin_supports_negative_guard_values():
    assert _margin_accepts_override(0.0, -999.0)
    assert _margin_accepts_override(-0.5, -0.5)
    assert _margin_accepts_override(-0.5, 0.0)
    assert not _margin_accepts_override(-0.5, -0.51)
    assert _margin_accepts_override(0.25, 0.25)
    assert not _margin_accepts_override(0.25, 0.24)


def test_t1_override_gate_supports_threshold_rules():
    gate = {
        "kind": "threshold_rule",
        "conditions": [
            {
                "feature": "refined_delta_per_rank_gap",
                "direction": "ge",
                "threshold": 0.1,
            },
            {
                "feature": "predicted_bust_delta",
                "direction": "le",
                "threshold": 0.2,
            },
        ],
    }

    accepted, probability = _gate_accepts_override(
        gate,
        {"refined_delta_per_rank_gap": 0.11, "predicted_bust_delta": 0.19},
        threshold=0.0,
    )
    assert accepted
    assert probability == 1.0

    accepted, probability = _gate_accepts_override(
        gate,
        {"refined_delta_per_rank_gap": 0.09, "predicted_bust_delta": 0.19},
        threshold=0.0,
    )
    assert not accepted
    assert probability == 0.0


def test_t1_override_gate_supports_reject_condition_groups():
    gate = {
        "kind": "threshold_rule",
        "conditions": [
            {
                "feature": "refined_delta_per_rank_gap",
                "direction": "ge",
                "threshold": 0.1,
            },
        ],
        "reject_condition_groups": [
            {
                "name": "small_low_bust_gain",
                "conditions": [
                    {
                        "feature": "predicted_bust_delta",
                        "direction": "le",
                        "threshold": -0.01,
                    },
                    {
                        "feature": "risk_adjusted_refined_delta",
                        "direction": "le",
                        "threshold": 1.34,
                    },
                ],
            }
        ],
    }

    accepted, probability = _gate_accepts_override(
        gate,
        {
            "refined_delta_per_rank_gap": 0.2,
            "predicted_bust_delta": -0.02,
            "risk_adjusted_refined_delta": 1.1,
        },
        threshold=0.0,
    )
    assert not accepted
    assert probability == 0.0

    accepted, probability = _gate_accepts_override(
        gate,
        {
            "refined_delta_per_rank_gap": 0.2,
            "predicted_bust_delta": -0.02,
            "risk_adjusted_refined_delta": 1.5,
        },
        threshold=0.0,
    )
    assert accepted
    assert probability == 1.0


def test_t1_extra_refinement_candidates_keep_close_refined_leaders():
    candidates = [
        {"action_idx": 1, "refined_score": 10.0, "model_score": 3.0, "model_rank": 2},
        {"action_idx": 2, "refined_score": 9.25, "model_score": 4.0, "model_rank": 1},
        {"action_idx": 3, "refined_score": 8.9, "model_score": 1.0, "model_rank": 3},
        {"action_idx": 4, "refined_score": 6.0, "model_score": 5.0, "model_rank": 4},
        {"action_idx": 5, "refined_score": None, "model_score": 9.0, "model_rank": 5},
    ]

    selected = _t1_extra_refinement_candidates(
        candidates,
        HybridConfig(t1_extra_refine_top_k=3, t1_extra_refine_margin=1.0, t1_extra_refine_sims=16),
    )

    assert [item["action_idx"] for item in selected] == [1, 2]
    assert _t1_extra_refinement_candidates(
        candidates,
        HybridConfig(t1_extra_refine_top_k=3, t1_extra_refine_margin=1.0, t1_extra_refine_sims=0),
    ) == []


def test_t1_final_selector_sees_refined_rank_before_scoring(monkeypatch):
    obs = _obs(turn=1)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    n = len(valid_actions)
    scores = [100.0, 99.0] + [0.0] * (n - 2)
    fl_types = [[0.0, 0.0, 0.0, 0.0] for _ in range(n)]
    fl_types[1] = [1.0, 0.0, 0.0, 0.0]
    model = ScriptedActionValue(scores=scores, fl_types=fl_types)

    def fake_refine(obs, candidates, *, config, prob_engine_path=None, started_at=None):
        for candidate in candidates:
            candidate["refined_score"] = 10.0 if candidate["action_idx"] == 0 else 9.0
            candidate["refinement_source"] = "recursive_mc"
            candidate["samples"] = 32
            candidate["elapsed_ms"] = 1.0
        return {"exact_evaluated": len(candidates), "error": None}

    monkeypatch.setattr(hybrid_t1t2, "refine_t1_candidates_recursive_mc", fake_refine)
    selector = {
        "name": "rank_test",
        "features": ["neg_refined_rank", "predicted_qq"],
        "weights": [100.0, 1.0],
        "means": {"neg_refined_rank": 0.0, "predicted_qq": 0.0},
        "scales": {"neg_refined_rank": 1.0, "predicted_qq": 1.0},
        "intercept": 0.0,
    }

    result = evaluate_hybrid_position(
        obs,
        evaluator=_evaluator(model),
        config=HybridConfig(
            shortlist_k=2,
            insurance_k=0,
            max_pool=2,
            sync_exact_k=2,
            t1_selection_policy="selector",
            t1_final_selector=selector,
        ),
    )

    assert result["best_action_idx"] == 0
    assert result["best"]["refined_rank"] == 1


def test_t1_refined_challenger_accepts_low_bust_close_refined_candidate():
    current = {
        "action_idx": 1,
        "refined_score": 10.0,
        "model_score": 5.0,
        "model_rank": 1,
        "predicted_bust": 0.2,
    }
    challenger = {
        "action_idx": 2,
        "refined_score": 10.1,
        "model_score": 4.0,
        "model_rank": 2,
        "predicted_bust": 0.03,
    }

    chosen, details = _t1_refined_challenger_candidate(
        current,
        [current, challenger],
        bust_max=0.038026779890060425,
        refined_delta_max=0.1458333333333337,
    )

    assert chosen["action_idx"] == 2
    assert details is not None
    assert details["accepted"]
    assert details["refined_score_delta"] == pytest.approx(0.1)


def test_t1_refined_challenger_rejects_high_bust_or_large_refined_gap():
    current = {
        "action_idx": 1,
        "refined_score": 10.0,
        "model_score": 5.0,
        "model_rank": 1,
        "predicted_bust": 0.2,
    }
    high_bust = {
        "action_idx": 2,
        "refined_score": 10.1,
        "model_score": 4.0,
        "model_rank": 2,
        "predicted_bust": 0.04,
    }
    large_gap = {
        "action_idx": 3,
        "refined_score": 10.3,
        "model_score": 3.0,
        "model_rank": 3,
        "predicted_bust": 0.01,
    }

    chosen, details = _t1_refined_challenger_candidate(
        current,
        [current, high_bust],
        bust_max=0.038026779890060425,
        refined_delta_max=0.1458333333333337,
    )
    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]

    chosen, details = _t1_refined_challenger_candidate(
        current,
        [current, large_gap],
        bust_max=0.038026779890060425,
        refined_delta_max=0.1458333333333337,
    )
    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]


def test_t1_blend_challenger_accepts_high_bust_premium_fl_candidate():
    current = {
        "action_idx": 1,
        "refined_score": 10.0,
        "model_score": 4.0,
        "model_rank": 1,
        "predicted_bust": 0.2,
        "predicted_aa": 0.01,
        "predicted_kk": 0.02,
        "predicted_trips": 0.01,
    }
    challenger = {
        "action_idx": 2,
        "refined_score": 10.2,
        "model_score": 4.5,
        "model_rank": 2,
        "predicted_bust": 0.5,
        "predicted_aa": 0.03,
        "predicted_kk": 0.04,
        "predicted_trips": 0.02,
    }

    chosen, details = _t1_blend_challenger_candidate(
        current,
        [current, challenger],
        weight=0.25,
        bust_min=0.38472288846969604,
        premium_fl_delta_min=0.043578820303082466,
    )

    assert chosen["action_idx"] == 2
    assert details is not None
    assert details["accepted"]
    assert details["premium_fl_delta"] == pytest.approx(0.05)


def test_t1_blend_challenger_rejects_low_bust_or_low_premium_fl_delta():
    current = {
        "action_idx": 1,
        "refined_score": 10.0,
        "model_score": 4.0,
        "model_rank": 1,
        "predicted_bust": 0.2,
        "predicted_aa": 0.01,
        "predicted_kk": 0.02,
        "predicted_trips": 0.01,
    }
    low_bust = {
        "action_idx": 2,
        "refined_score": 10.2,
        "model_score": 4.5,
        "model_rank": 2,
        "predicted_bust": 0.3,
        "predicted_aa": 0.03,
        "predicted_kk": 0.04,
        "predicted_trips": 0.02,
    }
    low_premium_fl_delta = {
        "action_idx": 3,
        "refined_score": 10.3,
        "model_score": 4.6,
        "model_rank": 3,
        "predicted_bust": 0.5,
        "predicted_aa": 0.015,
        "predicted_kk": 0.025,
        "predicted_trips": 0.015,
    }

    chosen, details = _t1_blend_challenger_candidate(
        current,
        [current, low_bust],
        weight=0.25,
        bust_min=0.38472288846969604,
        premium_fl_delta_min=0.043578820303082466,
    )
    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]

    chosen, details = _t1_blend_challenger_candidate(
        current,
        [current, low_premium_fl_delta],
        weight=0.25,
        bust_min=0.38472288846969604,
        premium_fl_delta_min=0.043578820303082466,
    )
    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]


def test_t1_low_risk_blend_challenger_accepts_low_fl_low_bust_candidate():
    current = {
        "action_idx": 1,
        "refined_score": 10.0,
        "model_score": 4.0,
        "model_rank": 1,
        "predicted_fl": 0.2,
        "predicted_bust": 0.2,
    }
    challenger = {
        "action_idx": 2,
        "refined_score": 10.1,
        "model_score": 4.5,
        "model_rank": 2,
        "predicted_fl": 0.00001,
        "predicted_bust": 0.01,
    }

    chosen, details = _t1_low_risk_blend_challenger_candidate(
        current,
        [current, challenger],
        weight=0.25,
        fl_max=8.5963918536435813e-05,
        bust_max=0.017788119614124298,
    )

    assert chosen["action_idx"] == 2
    assert details is not None
    assert details["accepted"]


def test_t1_low_risk_blend_challenger_rejects_high_fl_or_high_bust():
    current = {
        "action_idx": 1,
        "refined_score": 10.0,
        "model_score": 4.0,
        "model_rank": 1,
        "predicted_fl": 0.2,
        "predicted_bust": 0.2,
    }
    high_fl = {
        "action_idx": 2,
        "refined_score": 10.1,
        "model_score": 4.5,
        "model_rank": 2,
        "predicted_fl": 0.001,
        "predicted_bust": 0.01,
    }
    high_bust = {
        "action_idx": 3,
        "refined_score": 10.2,
        "model_score": 4.6,
        "model_rank": 3,
        "predicted_fl": 0.00001,
        "predicted_bust": 0.02,
    }

    chosen, details = _t1_low_risk_blend_challenger_candidate(
        current,
        [current, high_fl],
        weight=0.25,
        fl_max=8.5963918536435813e-05,
        bust_max=0.017788119614124298,
    )
    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]

    chosen, details = _t1_low_risk_blend_challenger_candidate(
        current,
        [current, high_bust],
        weight=0.25,
        fl_max=8.5963918536435813e-05,
        bust_max=0.017788119614124298,
    )
    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]


def test_t1_model_rescue_accepts_model_top1_with_fl_qq_delta():
    current = {
        "action_idx": 1,
        "predicted_fl": 0.1,
        "predicted_aa": 0.01,
        "predicted_kk": 0.02,
        "predicted_qq": 0.01,
        "predicted_trips": 0.0,
    }
    model_top1 = {
        "action_idx": 2,
        "predicted_fl": 0.32,
        "predicted_aa": 0.01,
        "predicted_kk": 0.02,
        "predicted_qq": 0.05,
        "predicted_trips": 0.0,
    }

    chosen, details = _t1_model_rescue_candidate(
        current,
        model_top1,
        qq_delta_min=0.03171762824058533,
        candidate_fl_delta_min=0.1830318570137024,
    )

    assert chosen["action_idx"] == 2
    assert details is not None
    assert details["accepted"]
    assert details["predicted_qq_delta"] == pytest.approx(0.04)


def test_t1_model_rescue_rejects_without_fl_or_qq_delta():
    current = {
        "action_idx": 1,
        "predicted_fl": 0.1,
        "predicted_qq": 0.01,
    }
    weak_qq = {
        "action_idx": 2,
        "predicted_fl": 0.32,
        "predicted_qq": 0.02,
    }
    weak_fl = {
        "action_idx": 3,
        "predicted_fl": 0.2,
        "predicted_qq": 0.05,
    }

    chosen, details = _t1_model_rescue_candidate(
        current,
        weak_qq,
        qq_delta_min=0.03171762824058533,
        candidate_fl_delta_min=0.1830318570137024,
    )
    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]

    chosen, details = _t1_model_rescue_candidate(
        current,
        weak_fl,
        qq_delta_min=0.03171762824058533,
        candidate_fl_delta_min=0.1830318570137024,
    )
    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]


def test_t1_model_rescue_accepts_model_top1_with_premium_fl_delta():
    current = {
        "action_idx": 1,
        "predicted_fl": 0.31,
        "predicted_aa": 0.28,
        "predicted_kk": 0.02,
        "predicted_qq": 0.02,
        "predicted_trips": 0.0,
    }
    model_top1 = {
        "action_idx": 2,
        "predicted_fl": 0.35,
        "predicted_aa": 0.33,
        "predicted_kk": 0.02,
        "predicted_qq": 0.01,
        "predicted_trips": 0.0,
    }

    chosen, details = _t1_model_rescue_candidate(
        current,
        model_top1,
        qq_delta_min=0.03171762824058533,
        premium_delta_min=0.03,
        candidate_fl_delta_min=0.03,
    )

    assert chosen["action_idx"] == 2
    assert details is not None
    assert details["accepted"]
    assert not details["qq_condition"]
    assert details["premium_condition"]
    assert details["premium_delta"] == pytest.approx(0.05)


def test_t1_model_rescue_keeps_premium_rescue_disabled_by_default():
    current = {
        "action_idx": 1,
        "predicted_fl": 0.31,
        "predicted_aa": 0.28,
        "predicted_kk": 0.02,
        "predicted_qq": 0.02,
        "predicted_trips": 0.0,
    }
    model_top1 = {
        "action_idx": 2,
        "predicted_fl": 0.35,
        "predicted_aa": 0.33,
        "predicted_kk": 0.02,
        "predicted_qq": 0.01,
        "predicted_trips": 0.0,
    }

    chosen, details = _t1_model_rescue_candidate(
        current,
        model_top1,
        qq_delta_min=0.03171762824058533,
        candidate_fl_delta_min=0.03,
    )

    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]
    assert not details["premium_condition"]


def test_t1_model_rescue_accepts_with_conservative_quality_gates():
    current = {
        "action_idx": 1,
        "model_score": 13.6,
        "refined_score": 41.7,
        "predicted_bust": 0.07,
        "predicted_fl": 0.27,
        "predicted_aa": 0.31,
        "predicted_kk": 0.01,
        "predicted_qq": 0.02,
        "predicted_trips": 0.0,
    }
    model_top1 = {
        "action_idx": 2,
        "model_score": 15.1,
        "refined_score": 40.9,
        "predicted_bust": 0.08,
        "predicted_fl": 0.30,
        "predicted_aa": 0.35,
        "predicted_kk": 0.01,
        "predicted_qq": 0.01,
        "predicted_trips": 0.0,
    }

    chosen, details = _t1_model_rescue_candidate(
        current,
        model_top1,
        qq_delta_min=0.03171762824058533,
        premium_delta_min=0.03,
        candidate_fl_delta_min=0.02,
        model_score_min=15.0,
        model_bust_max=0.25,
        refined_delta_max=1.25,
    )

    assert chosen["action_idx"] == 2
    assert details is not None
    assert details["accepted"]
    assert details["model_score_condition"]
    assert details["model_bust_condition"]
    assert details["refined_delta_condition"]
    assert details["refined_delta"] == pytest.approx(0.8)


def test_t1_model_rescue_rejects_high_bust_or_large_refined_gap():
    current = {
        "action_idx": 1,
        "model_score": 12.7,
        "refined_score": 17.8,
        "predicted_bust": 0.53,
        "predicted_fl": 0.25,
        "predicted_aa": 0.38,
        "predicted_kk": 0.01,
        "predicted_qq": 0.01,
        "predicted_trips": 0.0,
    }
    high_bust_model_top1 = {
        "action_idx": 2,
        "model_score": 14.1,
        "refined_score": 15.45,
        "predicted_bust": 0.58,
        "predicted_fl": 0.30,
        "predicted_aa": 0.42,
        "predicted_kk": 0.01,
        "predicted_qq": 0.01,
        "predicted_trips": 0.0,
    }

    chosen, details = _t1_model_rescue_candidate(
        current,
        high_bust_model_top1,
        qq_delta_min=0.03171762824058533,
        premium_delta_min=0.03,
        candidate_fl_delta_min=0.02,
        model_score_min=15.0,
        model_bust_max=0.25,
        refined_delta_max=1.25,
    )

    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]
    assert not details["model_score_condition"]
    assert not details["model_bust_condition"]
    assert not details["refined_delta_condition"]


def test_t2_model_top1_rescue_accepts_high_fl_close_refined_gap():
    current = {
        "action_idx": 1,
        "refined_score": 26.4,
        "predicted_fl": 0.08,
    }
    model_top1 = {
        "action_idx": 2,
        "refined_score": 24.9,
        "predicted_fl": 0.45,
    }

    chosen, details = _t2_model_top1_rescue_candidate(
        current,
        model_top1,
        fl_min=0.4,
        refined_delta_max=2.0,
    )

    assert chosen["action_idx"] == 2
    assert details is not None
    assert details["accepted"]
    assert details["challenger_predicted_fl"] == pytest.approx(0.45)
    assert details["refined_delta"] == pytest.approx(1.5)


def test_t2_model_top1_rescue_rejects_low_fl_or_large_refined_gap():
    current = {
        "action_idx": 1,
        "refined_score": 26.4,
        "predicted_fl": 0.08,
    }
    low_fl_model_top1 = {
        "action_idx": 2,
        "refined_score": 24.9,
        "predicted_fl": 0.20,
    }
    large_gap_model_top1 = {
        "action_idx": 3,
        "refined_score": 23.8,
        "predicted_fl": 0.45,
    }

    chosen, details = _t2_model_top1_rescue_candidate(
        current,
        low_fl_model_top1,
        fl_min=0.4,
        refined_delta_max=2.0,
    )
    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]
    assert not details["fl_condition"]

    chosen, details = _t2_model_top1_rescue_candidate(
        current,
        large_gap_model_top1,
        fl_min=0.4,
        refined_delta_max=2.0,
    )
    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]
    assert not details["refined_delta_condition"]


def test_t2_model_top1_rescue_rejects_high_bust_when_configured():
    current = {
        "action_idx": 1,
        "refined_score": 26.4,
        "predicted_fl": 0.08,
    }
    high_bust_model_top1 = {
        "action_idx": 2,
        "refined_score": 25.2,
        "predicted_fl": 0.45,
        "predicted_bust": 0.78,
    }

    chosen, details = _t2_model_top1_rescue_candidate(
        current,
        high_bust_model_top1,
        fl_min=0.1,
        bust_max=0.7,
        refined_delta_max=2.0,
    )

    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]
    assert not details["bust_condition"]


def test_t2_model_top1_rescue_rejects_when_selected_model_rank_is_too_good():
    current = {
        "action_idx": 1,
        "model_rank": 2,
        "refined_score": 13.8,
    }
    model_top1 = {
        "action_idx": 2,
        "model_rank": 1,
        "refined_score": 12.8,
        "predicted_fl": 0.45,
        "predicted_bust": 0.08,
    }

    chosen, details = _t2_model_top1_rescue_candidate(
        current,
        model_top1,
        fl_min=0.1,
        bust_max=0.75,
        selected_model_rank_min=4,
        refined_delta_max=3.0,
    )

    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]
    assert not details["primary_rank_condition"]
    assert details["selected_model_rank_min"] == 4


def test_t2_model_top1_rescue_secondary_accepts_ranked_low_fl_band():
    current = {
        "action_idx": 1,
        "model_rank": 7,
        "refined_score": 29.0,
        "predicted_fl": 0.19,
    }
    model_top1 = {
        "action_idx": 2,
        "model_rank": 1,
        "refined_score": 22.0,
        "predicted_fl": 0.16,
        "predicted_bust": 0.65,
    }

    chosen, details = _t2_model_top1_rescue_candidate(
        current,
        model_top1,
        fl_min=0.4,
        refined_delta_max=2.0,
        secondary_fl_min=0.10,
        secondary_fl_max=0.20,
        secondary_bust_max=0.70,
        secondary_selected_model_rank_min=5,
        secondary_refined_delta_max=20.0,
    )

    assert chosen["action_idx"] == 2
    assert details is not None
    assert details["accepted"]
    assert details["accepted_gate"] == "secondary"
    assert details["secondary_fl_condition"]
    assert details["secondary_bust_condition"]
    assert details["secondary_rank_condition"]
    assert details["secondary_delta_condition"]


def test_t2_model_top1_rescue_secondary_rejects_high_fl_and_low_selected_rank():
    model_top1 = {
        "action_idx": 2,
        "model_rank": 1,
        "refined_score": 22.0,
        "predicted_fl": 0.35,
        "predicted_bust": 0.65,
    }
    current = {
        "action_idx": 1,
        "model_rank": 3,
        "refined_score": 26.0,
    }

    chosen, details = _t2_model_top1_rescue_candidate(
        current,
        model_top1,
        fl_min=0.4,
        refined_delta_max=2.0,
        secondary_fl_min=0.10,
        secondary_fl_max=0.20,
        secondary_bust_max=0.70,
        secondary_selected_model_rank_min=5,
        secondary_refined_delta_max=20.0,
    )

    assert chosen["action_idx"] == 1
    assert details is not None
    assert not details["accepted"]
    assert not details["secondary_fl_condition"]
    assert not details["secondary_rank_condition"]


def test_t2_model_top1_bust_rescue_accepts_lower_bust_higher_fl_top1():
    current = {
        "action_idx": 13,
        "model_rank": 7,
        "model_score": 11.1,
        "refined_score": 28.7,
        "predicted_bust": 0.83,
        "predicted_fl": 0.20,
    }
    model_top1 = {
        "action_idx": 22,
        "model_rank": 1,
        "model_score": 14.3,
        "refined_score": 21.4,
        "predicted_bust": 0.68,
        "predicted_fl": 0.32,
    }

    chosen, details = _t2_model_top1_bust_rescue_candidate(
        current,
        model_top1,
        selected_model_rank_min=5,
        refined_delta_max=12.0,
        model_delta_min=2.0,
        bust_delta_min=0.08,
        top_bust_max=0.75,
        top_fl_min=0.15,
        current_fl_min=0.10,
        fl_delta_min=0.01,
    )

    assert chosen["action_idx"] == 22
    assert details is not None
    assert details["accepted"]
    assert details["model_delta"] == pytest.approx(3.2)
    assert details["bust_delta"] == pytest.approx(0.15)
    assert details["fl_delta"] == pytest.approx(0.12)


def test_t2_model_top1_bust_rescue_rejects_small_fl_gain_and_low_current_fl():
    current = {
        "action_idx": 5,
        "model_rank": 8,
        "model_score": 10.7,
        "refined_score": 17.8,
        "predicted_bust": 0.33,
        "predicted_fl": 0.315,
    }
    small_fl_gain_top1 = {
        "action_idx": 7,
        "model_rank": 1,
        "model_score": 13.9,
        "refined_score": 10.7,
        "predicted_bust": 0.17,
        "predicted_fl": 0.318,
    }

    chosen, details = _t2_model_top1_bust_rescue_candidate(
        current,
        small_fl_gain_top1,
        selected_model_rank_min=5,
        refined_delta_max=12.0,
        model_delta_min=2.0,
        bust_delta_min=0.08,
        top_bust_max=0.75,
        top_fl_min=0.15,
        current_fl_min=0.10,
        fl_delta_min=0.01,
    )

    assert chosen["action_idx"] == 5
    assert details is not None
    assert not details["accepted"]
    assert not details["fl_delta_condition"]

    low_fl_current = dict(current, action_idx=6, predicted_fl=0.05)
    high_fl_top1 = dict(small_fl_gain_top1, predicted_fl=0.30)
    chosen, details = _t2_model_top1_bust_rescue_candidate(
        low_fl_current,
        high_fl_top1,
        selected_model_rank_min=5,
        refined_delta_max=12.0,
        model_delta_min=2.0,
        bust_delta_min=0.08,
        top_bust_max=0.75,
        top_fl_min=0.15,
        current_fl_min=0.10,
        fl_delta_min=0.01,
    )

    assert chosen["action_idx"] == 6
    assert details is not None
    assert not details["accepted"]
    assert not details["current_fl_condition"]


def test_t2_middle_fill_bottom_shift_rescue_accepts_structural_rank4_move():
    obs = Observation(
        board_self=Board(
            top=["Qs"],
            middle=["2c", "3c", "6h", "6s"],
            bottom=["9c", "Td"],
        ),
        board_opponent=Board(),
        dealt_cards=["As", "2d", "3d"],
        known_discards_self=[],
        turn=2,
        is_btn=False,
    )
    current = {
        "action_idx": 12,
        "model_rank": 2,
        "model_score": 4.91,
        "refined_score": 11.76,
        "predicted_bust": 0.56,
        "predicted_fl": 0.187,
        "action": {"placements": [["2d", "middle"], ["As", "top"]], "discard": "3d"},
    }
    challenger = {
        "action_idx": 15,
        "model_rank": 4,
        "model_score": 4.10,
        "refined_score": 0.04,
        "predicted_bust": 0.51,
        "predicted_fl": 0.214,
        "action": {"placements": [["2d", "bottom"], ["As", "top"]], "discard": "3d"},
    }

    chosen, details = _t2_middle_fill_bottom_shift_rescue_candidate(
        obs,
        current,
        [current, challenger],
        selected_model_rank_max=3,
        challenger_model_rank_max=4,
        model_gap_max=1.0,
        bust_delta_min=0.04,
        fl_delta_min=0.02,
        challenger_bust_max=0.55,
        challenger_fl_min=0.20,
        selected_bust_min=0.50,
    )

    assert chosen["action_idx"] == 15
    assert details is not None
    assert details["accepted"]
    assert details["moved_card"] == "2d"
    assert details["model_gap"] == pytest.approx(0.81)
    assert details["bust_delta"] == pytest.approx(0.05)
    assert details["fl_delta"] == pytest.approx(0.027)


def test_t2_middle_fill_bottom_shift_rescue_rejects_nonstructural_or_weak_delta():
    obs = Observation(
        board_self=Board(
            top=["Qs"],
            middle=["2c", "3c", "6h", "6s"],
            bottom=["9c", "Td"],
        ),
        board_opponent=Board(),
        dealt_cards=["As", "2d", "3d"],
        known_discards_self=[],
        turn=2,
        is_btn=False,
    )
    current = {
        "action_idx": 12,
        "model_rank": 2,
        "model_score": 4.91,
        "predicted_bust": 0.56,
        "predicted_fl": 0.187,
        "action": {"placements": [["2d", "middle"], ["As", "top"]], "discard": "3d"},
    }
    weak = {
        "action_idx": 15,
        "model_rank": 4,
        "model_score": 4.10,
        "predicted_bust": 0.51,
        "predicted_fl": 0.19,
        "action": {"placements": [["2d", "bottom"], ["As", "top"]], "discard": "3d"},
    }
    different_discard = {
        "action_idx": 16,
        "model_rank": 4,
        "model_score": 4.10,
        "predicted_bust": 0.51,
        "predicted_fl": 0.214,
        "action": {"placements": [["2d", "bottom"], ["As", "top"]], "discard": "7d"},
    }

    chosen, details = _t2_middle_fill_bottom_shift_rescue_candidate(
        obs,
        current,
        [current, weak, different_discard],
        selected_model_rank_max=3,
        challenger_model_rank_max=4,
        model_gap_max=1.0,
        bust_delta_min=0.04,
        fl_delta_min=0.02,
        challenger_bust_max=0.55,
        challenger_fl_min=0.20,
        selected_bust_min=0.50,
    )

    assert chosen["action_idx"] == 12
    assert details is not None
    assert not details["accepted"]
    assert details["structural_count"] == 1


def test_t2_model_rank_rescue_accepts_close_refined_better_model_rank():
    current = {
        "action_idx": 10,
        "model_rank": 5,
        "model_score": 3.1,
        "refined_score": 12.0,
    }
    challenger = {
        "action_idx": 11,
        "model_rank": 2,
        "model_score": 3.8,
        "refined_score": 11.4,
    }
    weak = {
        "action_idx": 12,
        "model_rank": 1,
        "model_score": 3.3,
        "refined_score": 11.9,
    }

    chosen, details = _t2_model_rank_rescue_candidate(
        current,
        [current, challenger, weak],
        rank_k=4,
        selected_model_rank_min=5,
        refined_delta_max=0.8,
        model_delta_min=0.4,
        min_refined_score=1.0,
    )

    assert chosen["action_idx"] == 11
    assert details is not None
    assert details["accepted"]
    assert details["current_model_rank"] == 5
    assert details["challenger_model_rank"] == 2
    assert details["refined_delta"] == pytest.approx(0.6)
    assert details["model_delta"] == pytest.approx(0.7)


def test_t2_model_rank_rescue_rejects_when_selected_model_rank_is_too_good():
    current = {
        "action_idx": 10,
        "model_rank": 3,
        "model_score": 3.1,
        "refined_score": 12.0,
    }
    challenger = {
        "action_idx": 11,
        "model_rank": 2,
        "model_score": 3.8,
        "refined_score": 11.4,
    }

    chosen, details = _t2_model_rank_rescue_candidate(
        current,
        [current, challenger],
        rank_k=4,
        selected_model_rank_min=5,
        refined_delta_max=0.8,
        model_delta_min=0.4,
        min_refined_score=1.0,
    )

    assert chosen["action_idx"] == 10
    assert details is not None
    assert not details["accepted"]
    assert details["reject_reason"] == "current_rank_or_refined"


def test_t1_tactical_sync_insurance_promotes_top_trips_completion():
    candidates = []
    for i in range(12):
        candidates.append(
            {
                "action_idx": i,
                "action": {"placements": [["2d", "bottom"], ["3d", "middle"]], "discard": "4d"},
                "board": {"top": ["9c", "9h"], "middle": ["3d"], "bottom": ["2d"]},
                "model_score": float(100 - i),
                "model_rank": i + 1,
                "refinement_rank": i + 1,
                "predicted_bust": 0.1,
            }
        )
    candidates[11].update(
        {
            "action": {"placements": [["9s", "top"], ["3d", "middle"]], "discard": "2d"},
            "board": {"top": ["9c", "9h", "9s"], "middle": ["3d"], "bottom": []},
            "model_score": 1.0,
        }
    )

    base = _sync_refinement_candidates(
        candidates,
        HybridConfig(sync_exact_k=3),
        turn=1,
    )
    insured = _sync_refinement_candidates(
        candidates,
        HybridConfig(sync_exact_k=3, t1_sync_tactical_insurance_k=1),
        turn=1,
    )

    assert [item["action_idx"] for item in base] == [0, 1, 2]
    assert 11 in {item["action_idx"] for item in insured}
    assert len(insured) == 3


def test_t1_tactical_sync_insurance_prefers_sparse_bottom_over_risky_joker_trip():
    risky_joker_trip = {
        "action_idx": 1,
        "action": {"placements": [["Kh", "top"], ["Jc", "bottom"]], "discard": "6c"},
        "board": {"top": ["Kd", "X1", "Kh"], "middle": ["6h", "6s"], "bottom": ["2c", "Jc"]},
        "model_score": 20.0,
        "model_rank": 1,
        "refinement_rank": 1,
        "predicted_bust": 0.63,
    }
    sparse_bottom = {
        "action_idx": 11,
        "action": {"placements": [["6c", "bottom"], ["Jc", "bottom"]], "discard": "Kh"},
        "board": {"top": ["Kd", "X1"], "middle": ["6h", "6s"], "bottom": ["2c", "6c", "Jc"]},
        "model_score": 10.0,
        "model_rank": 11,
        "refinement_rank": 11,
        "predicted_bust": 0.10,
    }
    filler = [
        {
            "action_idx": i + 2,
            "action": {"placements": [["6c", "middle"], ["Jc", "bottom"]], "discard": "Kh"},
            "board": {"top": ["Kd", "X1"], "middle": ["6h", "6s", "6c"], "bottom": ["2c", "Jc"]},
            "model_score": float(19 - i),
            "model_rank": i + 2,
            "refinement_rank": i + 2,
            "predicted_bust": 0.2,
        }
        for i in range(9)
    ]
    candidates = [risky_joker_trip, *filler, sparse_bottom]

    insured = _sync_refinement_candidates(
        candidates,
        HybridConfig(sync_exact_k=3, t1_sync_tactical_insurance_k=1),
        turn=1,
    )

    assert insured[0]["action_idx"] == 11
    assert 1 in {item["action_idx"] for item in insured}


def test_t1_bottom_sparse_rescue_respects_refined_margin():
    current = {
        "action_idx": 1,
        "action": {"placements": [["2d", "middle"], ["7h", "bottom"]], "discard": "8h"},
        "board": {"top": ["6s", "Ks", "X2"], "middle": ["5h", "2d"], "bottom": ["Jh", "7h"]},
        "refined_score": 18.0,
        "model_score": 6.8,
        "model_rank": 3,
    }
    sparse = {
        "action_idx": 8,
        "action": {"placements": [["7h", "bottom"], ["8h", "bottom"]], "discard": "2d"},
        "board": {"top": ["6s", "Ks", "X2"], "middle": ["5h"], "bottom": ["Jh", "7h", "8h"]},
        "refined_score": 17.7,
        "model_score": 6.5,
        "model_rank": 5,
    }

    rescued, delta = _t1_bottom_sparse_rescue_candidate(current, [current, sparse], margin=0.5)
    assert rescued is sparse
    assert delta == pytest.approx(0.3)

    rescued, delta = _t1_bottom_sparse_rescue_candidate(current, [current, sparse], margin=0.1)
    assert rescued is None
    assert delta == pytest.approx(0.3)


def test_t1_no_refine_fallback_can_choose_fl_safe_sync_candidate():
    model_top = {
        "action_idx": 1,
        "model_rank": 1,
        "model_score": 10.0,
        "predicted_fl": 0.10,
        "predicted_bust": 0.05,
    }
    fl_safe = {
        "action_idx": 2,
        "model_rank": 2,
        "model_score": 9.0,
        "predicted_fl": 0.55,
        "predicted_bust": 0.04,
    }
    risky_fl = {
        "action_idx": 3,
        "model_rank": 3,
        "model_score": 8.0,
        "predicted_fl": 0.60,
        "predicted_bust": 0.45,
    }

    selected = _t1_no_refine_fallback_candidate(
        [model_top, fl_safe, risky_fl],
        [model_top, fl_safe, risky_fl],
        policy="fl_safe",
    )
    assert selected is fl_safe

    selected = _t1_no_refine_fallback_candidate(
        [model_top, fl_safe, risky_fl],
        [model_top, fl_safe, risky_fl],
        policy="model",
    )
    assert selected is model_top
