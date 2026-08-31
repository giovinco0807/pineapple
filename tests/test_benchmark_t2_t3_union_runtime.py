import pytest

from ai.tutor.benchmark_t2_t3_union_runtime import (
    build_arg_parser,
    build_config,
    compare_teacher,
    summarize,
    teacher_scores_from_record,
)
from ai.tutor.exact_late import action_key


def test_teacher_scores_from_record_reads_mc_avg_score():
    record = {
        "candidates": [
            {
                "placements": [["As", "top"], ["Td", "bottom"]],
                "discard": "4c",
                "mc": {"avg_score": 7.5},
            },
            {
                "placements": [["As", "middle"], ["Td", "bottom"]],
                "discard": "4c",
                "mc": {"avg_score": 4.0},
            },
        ]
    }

    scores = teacher_scores_from_record(record)

    assert scores[action_key({"placements": [["As", "top"], ["Td", "bottom"]], "discard": "4c"})] == 7.5
    assert len(scores) == 2


def test_compare_teacher_reports_chosen_ev_loss():
    best_action = {"placements": [["As", "top"], ["Td", "bottom"]], "discard": "4c"}
    weak_action = {"placements": [["As", "middle"], ["Td", "bottom"]], "discard": "4c"}
    record = {
        "candidates": [
            {"placements": best_action["placements"], "discard": "4c", "mc": {"avg_score": 7.5}},
            {"placements": weak_action["placements"], "discard": "4c", "mc": {"avg_score": 4.0}},
        ]
    }
    result = {
        "best": {"action": weak_action},
        "candidates": [
            {"model_rank": 1, "action": weak_action},
            {"model_rank": 2, "action": best_action},
        ],
    }

    comparison = compare_teacher(record, result)

    assert comparison is not None
    assert comparison["chosen_ev_loss"] == pytest.approx(3.5)
    assert comparison["model_top1_ev_loss"] == pytest.approx(3.5)
    assert comparison["chosen_hit"] is False


def test_compare_teacher_marks_degenerate_scores_invalid():
    action_a = {"placements": [["As", "top"], ["Td", "bottom"]], "discard": "4c"}
    action_b = {"placements": [["As", "middle"], ["Td", "bottom"]], "discard": "4c"}
    record = {
        "candidates": [
            {"action": action_a, "source_score": 0.0},
            {"action": action_b, "source_score": 0.0},
        ]
    }
    result = {
        "best": {"action": action_a},
        "candidates": [
            {"model_rank": 1, "action": action_a},
            {"model_rank": 2, "action": action_b},
        ],
    }

    comparison = compare_teacher(record, result)

    assert comparison is not None
    assert comparison["valid"] is False
    assert comparison["invalid_reason"] == "degenerate_teacher_scores"
    assert comparison["chosen_ev_loss"] is None


def test_summarize_reports_latency_and_teacher_tail():
    rows = [
        {
            "elapsed_ms": 1000.0,
            "candidate_pool_size": 20,
            "model_top1_overridden": False,
            "teacher": {"chosen_hit": True, "chosen_ev_loss": 0.0, "model_top1_ev_loss": 0.0},
        },
        {
            "elapsed_ms": 6000.0,
            "candidate_pool_size": 18,
            "model_top1_overridden": True,
            "teacher": {"chosen_hit": False, "chosen_ev_loss": 0.5, "model_top1_ev_loss": 1.0},
        },
    ]

    summary = summarize(rows, time_budget_ms=5000)

    assert summary["latency_ms"]["under_budget_count"] == 1
    assert summary["model_top1_overridden_count"] == 1
    assert summary["chosen_vs_teacher"]["loss_ge_0_5"] == 1
    assert summary["chosen_vs_teacher"]["ev_loss_max"] == pytest.approx(0.5)


def test_summarize_excludes_invalid_teacher_rows():
    rows = [
        {
            "elapsed_ms": 1000.0,
            "candidate_pool_size": 20,
            "model_top1_overridden": False,
            "teacher": {"valid": False, "invalid_reason": "degenerate_teacher_scores"},
        },
        {
            "elapsed_ms": 2000.0,
            "candidate_pool_size": 18,
            "model_top1_overridden": False,
            "teacher": {"valid": True, "chosen_hit": False, "chosen_ev_loss": 0.25, "model_top1_ev_loss": 0.5},
        },
    ]

    summary = summarize(rows, time_budget_ms=5000)

    assert summary["teacher_compared_count"] == 1
    assert summary["teacher_invalid_count"] == 1
    assert summary["teacher_invalid_reasons"] == {"degenerate_teacher_scores": 1}
    assert summary["chosen_vs_teacher"]["ev_loss_mean"] == pytest.approx(0.25)


def test_build_config_loads_t2_sync_selector(tmp_path):
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        '{"name":"selector-test","features":[],"weights":[],"intercept":0.0}',
        encoding="utf-8",
    )
    args = build_arg_parser().parse_args(
        [
            "--t2-sync-selection-policy",
            "selector",
            "--t2-sync-selector",
            str(selector_path),
        ]
    )

    config = build_config({}, args)

    assert config.t2_sync_selection_policy == "selector"
    assert config.t2_sync_selector is not None
    assert config.t2_sync_selector["name"] == "selector-test"


def test_build_config_t2_sync_selector_implies_selector_policy(tmp_path):
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        '{"name":"selector-test","features":[],"weights":[],"intercept":0.0}',
        encoding="utf-8",
    )
    args = build_arg_parser().parse_args(["--t2-sync-selector", str(selector_path)])

    config = build_config({}, args)

    assert config.t2_sync_selection_policy == "selector"


def test_build_config_accepts_t2_selection_model_weight_override():
    args = build_arg_parser().parse_args(["--t2-selection-model-weight", "0.75"])

    config = build_config({"t2_selection_model_weight": 0.5}, args)

    assert config.t2_selection_model_weight == pytest.approx(0.75)


def test_build_config_accepts_t2_top1_rescue_rank_override():
    args = build_arg_parser().parse_args(["--t2-model-top1-rescue-selected-model-rank-min", "4"])

    config = build_config({"t2_model_top1_rescue_selected_model_rank_min": 0}, args)

    assert config.t2_model_top1_rescue_selected_model_rank_min == 4


def test_build_config_accepts_t2_model_rank_rescue_overrides():
    args = build_arg_parser().parse_args(
        [
            "--t2-model-rank-rescue-k",
            "4",
            "--t2-model-rank-rescue-selected-model-rank-min",
            "5",
            "--t2-model-rank-rescue-refined-delta-max",
            "0.8",
            "--t2-model-rank-rescue-model-delta-min",
            "0.4",
            "--t2-model-rank-rescue-min-refined-score",
            "1.0",
        ]
    )

    config = build_config({}, args)

    assert config.t2_model_rank_rescue_k == 4
    assert config.t2_model_rank_rescue_selected_model_rank_min == 5
    assert config.t2_model_rank_rescue_refined_delta_max == pytest.approx(0.8)
    assert config.t2_model_rank_rescue_model_delta_min == pytest.approx(0.4)
    assert config.t2_model_rank_rescue_min_refined_score == pytest.approx(1.0)


def test_build_config_accepts_t2_model_top1_bust_rescue_overrides():
    args = build_arg_parser().parse_args(
        [
            "--t2-model-top1-bust-rescue-selected-model-rank-min",
            "5",
            "--t2-model-top1-bust-rescue-refined-delta-max",
            "12.0",
            "--t2-model-top1-bust-rescue-model-delta-min",
            "2.0",
            "--t2-model-top1-bust-rescue-bust-delta-min",
            "0.08",
            "--t2-model-top1-bust-rescue-top-bust-max",
            "0.75",
            "--t2-model-top1-bust-rescue-top-fl-min",
            "0.15",
            "--t2-model-top1-bust-rescue-current-fl-min",
            "0.10",
            "--t2-model-top1-bust-rescue-fl-delta-min",
            "0.01",
        ]
    )

    config = build_config({}, args)

    assert config.t2_model_top1_bust_rescue_selected_model_rank_min == 5
    assert config.t2_model_top1_bust_rescue_refined_delta_max == pytest.approx(12.0)
    assert config.t2_model_top1_bust_rescue_model_delta_min == pytest.approx(2.0)
    assert config.t2_model_top1_bust_rescue_bust_delta_min == pytest.approx(0.08)
    assert config.t2_model_top1_bust_rescue_top_bust_max == pytest.approx(0.75)
    assert config.t2_model_top1_bust_rescue_top_fl_min == pytest.approx(0.15)
    assert config.t2_model_top1_bust_rescue_current_fl_min == pytest.approx(0.10)
    assert config.t2_model_top1_bust_rescue_fl_delta_min == pytest.approx(0.01)


def test_build_config_accepts_t2_middle_fill_bottom_shift_rescue_overrides():
    args = build_arg_parser().parse_args(
        [
            "--t2-middle-fill-bottom-shift-rescue-selected-model-rank-max",
            "3",
            "--t2-middle-fill-bottom-shift-rescue-challenger-model-rank-max",
            "4",
            "--t2-middle-fill-bottom-shift-rescue-model-gap-max",
            "1.0",
            "--t2-middle-fill-bottom-shift-rescue-bust-delta-min",
            "0.04",
            "--t2-middle-fill-bottom-shift-rescue-fl-delta-min",
            "0.02",
            "--t2-middle-fill-bottom-shift-rescue-challenger-bust-max",
            "0.55",
            "--t2-middle-fill-bottom-shift-rescue-challenger-fl-min",
            "0.20",
            "--t2-middle-fill-bottom-shift-rescue-selected-bust-min",
            "0.50",
        ]
    )

    config = build_config({}, args)

    assert config.t2_middle_fill_bottom_shift_rescue_selected_model_rank_max == 3
    assert config.t2_middle_fill_bottom_shift_rescue_challenger_model_rank_max == 4
    assert config.t2_middle_fill_bottom_shift_rescue_model_gap_max == pytest.approx(1.0)
    assert config.t2_middle_fill_bottom_shift_rescue_bust_delta_min == pytest.approx(0.04)
    assert config.t2_middle_fill_bottom_shift_rescue_fl_delta_min == pytest.approx(0.02)
    assert config.t2_middle_fill_bottom_shift_rescue_challenger_bust_max == pytest.approx(0.55)
    assert config.t2_middle_fill_bottom_shift_rescue_challenger_fl_min == pytest.approx(0.20)
    assert config.t2_middle_fill_bottom_shift_rescue_selected_bust_min == pytest.approx(0.50)


def test_build_config_t2_final_selector_implies_selector_policy(tmp_path):
    selector_path = tmp_path / "t2_final_selector.json"
    selector_path.write_text(
        '{"name":"t2-final-test","features":["model_score"],"weights":[1.0],"intercept":0.0}',
        encoding="utf-8",
    )
    args = build_arg_parser().parse_args(["--t2-final-selector", str(selector_path)])

    config = build_config({}, args)

    assert config.t2_selection_policy == "selector"
    assert config.t2_final_selector is not None
    assert config.t2_final_selector["name"] == "t2-final-test"
