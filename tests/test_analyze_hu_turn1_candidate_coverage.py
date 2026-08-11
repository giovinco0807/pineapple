from __future__ import annotations

from typing import Any

from ofc_regular.analyze_hu_turn1_candidate_coverage import (
    CandidateModelSpec,
    _read_jsonl,
    analyze_rows,
    analyze_rows_union,
    infer_artifact_stage,
    summarize_details_by_seat,
)


class DummyCandidateModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def predict_sample(self, sample: dict[str, Any]) -> list[float]:
        assert len(sample["actions"]) == len(self.scores)
        return list(self.scores)


def test_analyze_rows_maps_predictions_by_original_action_order() -> None:
    row = {
        "sample_id": 7,
        "hand_seed": 123,
        "seat": "first",
        "board": {"top": ["As"], "middle": ["2c"], "bottom": ["3c", "4c", "5c"]},
        "opponent_board": {"top": ["Kh"], "middle": ["7h"], "bottom": ["8h", "9h", "Th"]},
        "dealt": ["6d", "7d", "8d"],
        "visible_dead_cards": ["Kh", "7h", "8h", "9h", "Th"],
        "best_action": 0,
        "score_gap": 3.0,
        "actions": [
            {
                "placements": [["6d", "middle"], ["7d", "bottom"]],
                "discards": ["8d"],
                "original_index": 5,
                "ev": 10.0,
            },
            {
                "placements": [["6d", "top"], ["7d", "middle"]],
                "discards": ["8d"],
                "original_index": 2,
                "ev": 7.0,
            },
            {
                "placements": [["6d", "bottom"], ["7d", "bottom"]],
                "discards": ["8d"],
                "original_index": 9,
                "ev": 1.0,
            },
        ],
    }

    # Scores are for actions sorted by original_index: 2, 5, 9.
    summary, details = analyze_rows([row], DummyCandidateModel([3.0, 2.0, 1.0]), [1, 2])

    assert summary[0]["k"] == 1
    assert summary[0]["hit"] == 0
    assert summary[0]["avg_topk_regret"] == 3.0
    assert summary[1]["k"] == 2
    assert summary[1]["hit"] == 1
    assert summary[1]["avg_topk_regret"] == 0.0
    assert details[0]["top1_regret"] == 3.0
    assert details[0]["top2_hit"] is True
    seat_summary = summarize_details_by_seat(details, [1, 2])
    assert seat_summary[0] == {
        "seat": "first",
        "k": 1,
        "records": 1,
        "hit": 0,
        "miss": 1,
        "recall": 0.0,
        "avg_topk_regret": 3.0,
        "max_topk_regret": 3.0,
    }


def test_analyze_rows_union_preserves_original_action_mapping() -> None:
    row = {
        "sample_id": 8,
        "hand_seed": 124,
        "seat": "second",
        "board": {"top": ["As"], "middle": ["2c"], "bottom": ["3c", "4c", "5c"]},
        "opponent_board": {"top": ["Kh"], "middle": ["7h"], "bottom": ["8h", "9h", "Th"]},
        "dealt": ["6d", "7d", "8d"],
        "visible_dead_cards": ["Kh", "7h", "8h", "9h", "Th"],
        "best_action": 0,
        "score_gap": 4.0,
        "actions": [
            {
                "placements": [["6d", "middle"], ["7d", "bottom"]],
                "discards": ["8d"],
                "original_index": 5,
                "ev": 10.0,
            },
            {
                "placements": [["6d", "top"], ["7d", "middle"]],
                "discards": ["8d"],
                "original_index": 2,
                "ev": 6.0,
            },
            {
                "placements": [["6d", "bottom"], ["7d", "bottom"]],
                "discards": ["8d"],
                "original_index": 9,
                "ev": 1.0,
            },
        ],
    }

    # Scores are for actions sorted by original_index: 2, 5, 9.
    summary, details = analyze_rows_union(
        [row],
        [
            CandidateModelSpec("miss", DummyCandidateModel([3.0, 0.0, 1.0]), 1),
            CandidateModelSpec("hit", DummyCandidateModel([0.0, 3.0, 1.0]), 1),
        ],
        [1, 2],
    )

    assert summary[0]["k"] == 1
    assert summary[0]["hit"] == 0
    assert summary[0]["avg_topk_regret"] == 4.0
    assert summary[1]["k"] == 2
    assert summary[1]["hit"] == 1
    assert summary[1]["avg_topk_regret"] == 0.0
    assert details[0]["candidate_union_size"] == 2
    assert details[0]["candidate_model_count"] == 2


def test_analyze_rows_union_rank_sum_prefers_consensus_actions() -> None:
    row = {
        "sample_id": 9,
        "hand_seed": 125,
        "seat": "first",
        "board": {"top": ["As"], "middle": ["2c"], "bottom": ["3c", "4c", "5c"]},
        "opponent_board": {"top": ["Kh"], "middle": ["7h"], "bottom": ["8h", "9h", "Th"]},
        "dealt": ["6d", "7d", "8d"],
        "visible_dead_cards": ["Kh", "7h", "8h", "9h", "Th"],
        "best_action": 1,
        "score_gap": 3.0,
        "actions": [
            {
                "placements": [["6d", "middle"], ["7d", "bottom"]],
                "discards": ["8d"],
                "original_index": 0,
                "ev": 6.0,
            },
            {
                "placements": [["6d", "top"], ["7d", "middle"]],
                "discards": ["8d"],
                "original_index": 1,
                "ev": 10.0,
            },
            {
                "placements": [["6d", "bottom"], ["7d", "bottom"]],
                "discards": ["8d"],
                "original_index": 2,
                "ev": 1.0,
            },
            {
                "placements": [["6d", "bottom"], ["7d", "middle"]],
                "discards": ["8d"],
                "original_index": 3,
                "ev": 0.0,
            },
        ],
    }

    model_specs = [
        CandidateModelSpec("a", DummyCandidateModel([4.0, 3.0, 2.0, 1.0]), 3),
        CandidateModelSpec("b", DummyCandidateModel([1.0, 4.0, 3.0, 2.0]), 3),
    ]

    min_rank_summary, _details = analyze_rows_union([row], model_specs, [1], union_mode="min_rank")
    rank_sum_summary, details = analyze_rows_union([row], model_specs, [1], union_mode="rank_sum")

    assert min_rank_summary[0]["hit"] == 0
    assert min_rank_summary[0]["avg_topk_regret"] == 4.0
    assert rank_sum_summary[0]["hit"] == 1
    assert rank_sum_summary[0]["avg_topk_regret"] == 0.0
    assert details[0]["candidate_union_mode"] == "rank_sum"


def test_read_jsonl_accepts_utf8_sig(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('\ufeff{"sample_id": 1}\n{"sample_id": 2}\n', encoding="utf-8")

    rows = _read_jsonl(path)

    assert [row["sample_id"] for row in rows] == [1, 2]


def test_infer_artifact_stage_recognizes_turn0_rows() -> None:
    assert infer_artifact_stage([{"phase": "hu_turn0_0card"}]) == "hu_turn0"
    assert infer_artifact_stage([{"phase": "hu_turn1_5card"}]) == "hu_turn1"
