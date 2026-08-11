import numpy as np

from ofc_regular.prepare_hu_turn1_oof_selector_targets import build_target_row, stable_fold
from ofc_regular.train_hu_turn1_safe_override_selector import label_for_row


def _action(score: float, index: int) -> dict:
    return {
        "placements": [["As", "top"], ["2c", "bottom"]],
        "discards": ["3d"],
        "score": score,
        "original_index": index,
    }


def test_build_target_row_uses_oof_candidate_and_teacher_delta():
    row = {
        "state_key": "state-a",
        "best_action": 0,
        "baseline_action_row_index": 1,
        "actions": [_action(5.0, 9), _action(1.5, 3), _action(-2.0, 7)],
    }
    target = build_target_row(
        row,
        candidate_row_index=0,
        predictions=np.asarray([2.0, 0.5, -1.0]),
        fold=2,
    )

    assert target["candidate_action_index"] == 9
    assert target["fallback_action_index"] == 3
    assert target["stage10_mc32_candidate_delta"] == 3.5
    assert target["hu_turn1_predicted_margin"] == 1.5
    assert target["teacher_candidate_regret"] == 0.0
    assert target["oof_prediction"] is True


def test_stable_fold_is_deterministic():
    row = {"state_key": "same-state"}

    assert stable_fold(row, folds=5, seed=17) == stable_fold(row, folds=5, seed=17)


def test_teacher_delta_thresholded_label_uses_mc_teacher_delta():
    label, weight, delta, _regret, gray = label_for_row(
        {"stage10_mc32_candidate_delta": 0.75},
        label_mode="teacher_delta_thresholded",
        teacher_accept_regret=0.25,
        teacher_gray_regret=1.0,
        accept_delta=0.5,
        hard_negative_delta=-0.5,
        gray_weight=0.1,
    )

    assert label == 1
    assert weight == 1.0
    assert delta == 0.75
    assert gray is False
