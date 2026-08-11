import argparse
from pathlib import Path

from ofc_regular import evaluate_model_set_matchup
from ofc_regular.evaluate_model_set_matchup import load_policy_parts


def test_load_policy_parts_stage7_failure_uses_reference_stage3(monkeypatch):
    reference_model = object()

    def fake_action_loader(path):
        return f"action:{path}"

    def fake_hu_loader(path):
        if path == Path("stage7_missing.pt"):
            raise OSError("missing stage7")
        if path == Path("stage3_reference.pt"):
            return reference_model
        raise AssertionError(path)

    monkeypatch.setattr(evaluate_model_set_matchup, "load_action_value_model", fake_action_loader)
    monkeypatch.setattr(evaluate_model_set_matchup, "load_hu_action_value_model", fake_hu_loader)

    args = argparse.Namespace(
        opening_a=Path("opening.pt"),
        turn1_a=Path("turn1.pt"),
        turn2_a=Path("turn2.pkl"),
        turn3_a=Path("turn3.pkl"),
        hu_turn3_a=Path("stage7_missing.pt"),
        hu_turn3_reference_a=Path("stage3_reference.pt"),
        hu_turn3_support_a=None,
        hu_turn3_gate_a=None,
        hu_turn3_min_margin_a=5.0,
        hu_turn3_reference_min_margin_a=10.0,
        hu_turn3_min_support_margin_a=0.0,
        hu_turn3_min_model_score_a=None,
        hu_turn3_allowed_seats_a="",
        hu_turn3_min_gate_probability_a=0.0,
        hu_turn3_max_self_regret_a=None,
        hu_turn3_decision_log_a=None,
        disable_hu_turn3_stage7_a=False,
        opening_lookahead_samples=64,
    )

    parts = load_policy_parts(args, "a")

    assert parts["hu_turn3_model"] is reference_model
    assert parts["hu_turn3_reference_model"] is None
    assert parts["hu_turn3_min_margin"] == 10.0
    assert parts["hu_turn3_reference_min_margin"] == 0.0
