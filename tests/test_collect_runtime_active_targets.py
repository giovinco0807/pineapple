import json

from ai.tutor.collect_runtime_active_targets import load_excluded_target_keys, selected_reasons
from ai.tutor.weak_groups_to_active_targets import target_key


def test_selected_reasons_collects_good_override_when_final_beats_model():
    result = {
        "model_top1_overridden": True,
        "model_teacher_score": 1.0,
        "final_teacher_score": 2.25,
    }

    reasons = selected_reasons(
        result,
        {"good_override"},
        min_regret=0.0,
        min_override_delta=1.0,
    )

    assert reasons == ["runtime_good_override"]


def test_selected_reasons_collects_bad_override_when_model_beats_final():
    result = {
        "model_top1_overridden": True,
        "model_teacher_score": 2.25,
        "final_teacher_score": 1.0,
    }

    reasons = selected_reasons(
        result,
        {"bad_override"},
        min_regret=0.0,
        min_override_delta=1.0,
    )

    assert reasons == ["runtime_bad_override"]


def test_load_excluded_target_keys_uses_stable_target_key(tmp_path):
    target = {
        "turn": 1,
        "board": {"top": ["Ah"], "mid": [], "bot": []},
        "opponent_board": {"top": [], "mid": [], "bot": []},
        "dealt": ["2c", "3d", "4s"],
        "known_discards": [],
        "exclude": [],
        "is_btn": False,
        "runtime_result_line": 17,
    }
    path = tmp_path / "targets.jsonl"
    path.write_text(json.dumps(target) + "\n", encoding="utf-8")

    excluded = load_excluded_target_keys([path])

    assert excluded == {target_key(target)}
