from ofc_regular.compare_hu_turn1_t2_continuations import summarize_profile_comparison


def _row(first_discard, second_discard, first_score, second_score):
    return {
        "hand_seed": 1,
        "player": 0,
        "seat": "first",
        "board": {"top": ["As"], "middle": [], "bottom": []},
        "opponent_board": {"top": ["Kd"], "middle": [], "bottom": []},
        "dealt": ["2c", "3d", "4h"],
        "visible_dead_cards": ["9s"],
        "actions": [
            {
                "placements": [["2c", "top"], ["3d", "middle"]],
                "discards": [first_discard],
                "score": first_score,
            },
            {
                "placements": [["2c", "top"], ["3d", "middle"]],
                "discards": [second_discard],
                "score": second_score,
            },
        ],
    }


def test_summarize_profile_comparison_reports_state_and_best_action_overlap():
    rows = {
        "stage9f_p2": [_row("4h", "5h", 10.0, 8.0)],
        "stage9f_fast_t2_t1_teacher": [_row("5h", "4h", 9.0, 7.0)],
    }
    summaries = {
        "stage9f_p2": {
            "seconds_per_sample": 10.0,
            "mean_action_count": 24.0,
            "topk_decisions": 2,
            "topk_overrides": 1,
            "profile_stats": {"choose_action_T2_seconds": 8.0},
        },
        "stage9f_fast_t2_t1_teacher": {
            "seconds_per_sample": 2.0,
            "mean_action_count": 24.0,
            "profile_stats": {"choose_action_T2_seconds": 0.5},
        },
    }

    summary = summarize_profile_comparison(rows, summaries)

    fast = summary["comparisons"]["stage9f_fast_t2_t1_teacher"]
    assert fast["state_match_rate"] == 1.0
    assert fast["comparable_state_count"] == 1
    assert fast["best_action_match_rate"] == 0.0
    assert fast["profile_best_in_baseline_top3_rate"] == 1.0
    assert fast["profile_best_baseline_rank_mean"] == 2.0
    assert fast["profile_best_baseline_regret_mean"] == 2.0
    assert fast["baseline_best_profile_rank_mean"] == 2.0
    assert fast["baseline_best_profile_regret_mean"] == 2.0
    assert fast["t2_choose_action_seconds"] == 0.5
