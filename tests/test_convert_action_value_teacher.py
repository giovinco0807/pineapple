import argparse
import json
from pathlib import Path

import numpy as np

from ai.training.convert_action_value_teacher import convert, count_samples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def candidate(placements: list[list[str]], discard: str, ev: float) -> dict:
    return {
        "placements": placements,
        "discard": discard,
        "ev": ev,
        "bust_prob": 0.0,
        "fl_rate": 0.0,
        "fl_type_rates": {},
    }


def convert_args(input_path: Path, output_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        turns="2",
        state_dim=520,
        t0_max_candidates=0,
        regular_max_candidates=10,
        keep_t0_fl_shaped=False,
        fl_positive_weight=2.0,
        high_route_weight=1.0,
        fl_shape_but_foul_weight=1.5,
        teacher_best_weight=0.5,
        gap_weight=0.5,
        clean_fl_weight=1.0,
        t0_safe_fl_min=0.20,
        t0_safe_bust_max=0.35,
        t0_risky_fl_min=0.20,
        t0_risky_bust_min=0.35,
        t0_dead_fl_max=0.05,
        t0_dead_bust_min=0.25,
        t0_safe_route_weight=0.0,
        t0_risky_route_weight=0.0,
        t0_dead_route_weight=0.0,
        trips_fl_weight=0.0,
        t0_safe_route_score_bonus=0.0,
        t0_risky_route_score_penalty=0.0,
        t0_dead_route_score_penalty=0.0,
        trips_fl_score_bonus=0.0,
        max_sample_weight=8.0,
        limit_records=0,
    )


def test_convert_skips_candidate_that_does_not_match_dealt(tmp_path: Path) -> None:
    input_path = tmp_path / "teacher.jsonl"
    output_path = tmp_path / "dataset"
    rows = [
        {
            "turn": 2,
            "eval_mode": "exact_t2_capped",
            "board": {
                "top": ["3c"],
                "middle": ["4c", "5c", "6c"],
                "bottom": ["7c", "8c", "9c"],
            },
            "opponent_board": {
                "top": ["Ts", "Js"],
                "middle": ["Qs", "Ks", "As"],
                "bottom": ["2h", "3h", "4h", "5h"],
            },
            "dealt": ["Ah", "Kd", "2c"],
            "is_btn": True,
            "position": "btn",
            "candidates": [
                candidate([["Ah", "top"], ["Kd", "middle"]], "2c", 2.0),
                candidate([["7h", "top"], ["Kd", "middle"]], "2c", 1.0),
            ],
        }
    ]
    write_jsonl(input_path, rows)
    args = convert_args(input_path, output_path)

    n_total, meta = count_samples(input_path, {2}, args)
    assert n_total == 1
    assert meta["invalid_selected_candidates"] == 1

    convert(args)

    metadata = json.loads((output_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_allocated"] == 1
    assert metadata["n_samples"] == 1
    assert metadata["invalid_selected_candidates"] == 1
    assert metadata["count_invalid_selected_candidates"] == 1
    assert np.load(output_path / "scores.npy").shape == (1,)
