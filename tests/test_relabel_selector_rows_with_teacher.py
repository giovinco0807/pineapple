from ai.tutor.evaluate_hybrid_refinement_teacher import teacher_index
from ai.tutor.relabel_selector_rows_with_teacher import relabel_row


def test_relabel_row_updates_teacher_score_rank_and_best_flag():
    teacher_record = {
        "eval_mode": "mc1000",
        "candidates": [
            {
                "placements": [["Ah", "top"], ["2c", "middle"]],
                "discard": "3d",
                "mc": {"avg_score": 7.5, "simulations": 1000},
            },
            {
                "placements": [["Ah", "middle"], ["2c", "middle"]],
                "discard": "3d",
                "mc": {"avg_score": 5.0, "simulations": 1000},
            },
        ],
    }
    selector_row = {
        "action": {"placements": [["2c", "middle"], ["Ah", "top"]], "discard": "3d"},
        "teacher_score": -1.0,
        "teacher_rank": 99,
        "is_teacher_best": False,
    }

    out = relabel_row(selector_row, {"index": teacher_index(teacher_record), "eval_mode": "mc1000"})

    assert out["teacher_score"] == 7.5
    assert out["teacher_rank"] == 1
    assert out["teacher_best_score"] == 7.5
    assert out["teacher_margin"] == 2.5
    assert out["teacher_samples"] == 1000
    assert out["teacher_eval_mode"] == "mc1000"
    assert out["is_teacher_best"] is True
