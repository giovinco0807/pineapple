import json

from ai.training.compare_exact_rule_migration import classify_input, summarize


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate(card, score, bust=0.0):
    return {
        "action": {"placements": [[card, "middle"]], "discard": None},
        "metrics": {
            "score": score,
            "raw_score": score,
            "royalty": 0.0,
            "bust_rate": bust,
            "fl_rate": 0.0,
        },
    }


def test_classification_marks_unseen_future_joker_as_reachable():
    info = classify_input(
        {
            "turn": 3,
            "board": {"top": ["2h"], "middle": [], "bottom": []},
            "dealt": ["3h", "4h", "5h"],
            "exclude": [],
        }
    )
    assert info["visible_joker"] is False
    assert info["future_joker_reachable"] is True
    assert info["joker_relevant"] is True

    terminal = classify_input(
        {
            "turn": 4,
            "board": {"top": ["2h"], "middle": [], "bottom": []},
            "dealt": ["3h", "4h", "5h"],
        }
    )
    assert terminal["joker_relevant"] is False


def test_summary_detects_teacher_action_and_metric_changes(tmp_path):
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    input_path = tmp_path / "input.jsonl"

    old_a = _candidate("2h", 1.0, bust=0.4)
    old_b = _candidate("3h", 0.0, bust=0.1)
    new_a = _candidate("2h", 0.5, bust=0.2)
    new_b = _candidate("3h", 2.0, bust=0.0)
    _write_jsonl(
        baseline_path,
        [{"record_index": 7, "best": old_a, "candidates": [old_a, old_b]}],
    )
    _write_jsonl(
        candidate_path,
        [{"record_index": 7, "best": new_b, "candidates": [new_b, new_a]}],
    )
    _write_jsonl(
        input_path,
        [
            {
                "record_index": 7,
                "turn": 3,
                "board": {"top": [], "middle": [], "bottom": []},
                "dealt": ["X1", "4h", "5h"],
            }
        ],
    )

    summary, rows = summarize(
        baseline_path,
        candidate_path,
        input_path=input_path,
    )

    assert summary["matched_records"] == 1
    assert summary["best_action_changed"] == 1
    assert summary["old_policy_suboptimal_under_candidate"] == 1
    assert summary["candidate_policy_suboptimal_vs_baseline"] == 0
    assert summary["records_with_any_metric_change"] == 1
    assert summary["visible_joker"]["records"] == 1
    assert rows[0]["old_policy_regret_under_candidate"] == 1.5
