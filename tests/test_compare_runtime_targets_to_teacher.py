from ai.tutor.compare_runtime_targets_to_teacher import align_records, apply_runtime_results, summarize


def test_align_records_matches_parallel_teacher_output_by_source_line():
    targets = [
        {"source": "a.jsonl", "source_line": 1, "turn": 1, "payload": "target-a"},
        {"source": "b.jsonl", "source_line": 2, "turn": 1, "payload": "target-b"},
    ]
    teachers = [
        {"source": "b.jsonl", "source_line": 2, "turn": 1, "payload": "teacher-b"},
        {"source": "a.jsonl", "source_line": 1, "turn": 1, "payload": "teacher-a"},
    ]

    pairs = align_records(targets, teachers)

    assert [teacher["payload"] for _, teacher in pairs] == ["teacher-a", "teacher-b"]


def test_apply_runtime_results_overrides_model_and_final_action_indices():
    targets = [
        {
            "runtime_result_line": 7,
            "runtime_model_action_idx": 1,
            "runtime_final_action_idx": 2,
        }
    ]
    runtime_results = {
        7: {
            "model_top1_action_idx": 3,
            "best_action_idx": 4,
            "model_teacher_score": 1.25,
            "final_teacher_score": 2.5,
            "teacher_best_score": 2.5,
        }
    }

    updated = apply_runtime_results(targets, runtime_results)

    assert updated[0]["runtime_model_action_idx"] == 3
    assert updated[0]["runtime_final_action_idx"] == 4
    assert updated[0]["runtime_model_teacher_score"] == 1.25
    assert updated[0]["runtime_final_teacher_score"] == 2.5


def test_summarize_counts_zero_regret_separately_from_exact_best_action():
    rows = [
        {
            "relabel_model_found": True,
            "relabel_final_found": True,
            "relabel_model_is_best": False,
            "relabel_final_is_best": False,
            "relabel_model_zero_regret": False,
            "relabel_final_zero_regret": True,
            "relabel_model_minus_final": -1.5,
            "relabel_model_regret": 1.5,
            "relabel_final_regret": 0.0,
        },
        {
            "relabel_model_found": True,
            "relabel_final_found": True,
            "relabel_model_is_best": True,
            "relabel_final_is_best": True,
            "relabel_model_zero_regret": True,
            "relabel_final_zero_regret": True,
            "relabel_model_minus_final": 0.0,
            "relabel_model_regret": 0.0,
            "relabel_final_regret": 0.0,
        },
    ]

    summary = summarize(rows)

    assert summary["model_is_best"] == 1
    assert summary["final_is_best"] == 1
    assert summary["model_zero_regret"] == 1
    assert summary["final_zero_regret"] == 2
    assert summary["max_model_regret"] == 1.5
    assert summary["max_final_regret"] == 0.0
