import argparse
import json
from pathlib import Path

from ai.training.build_t2_action_value_teacher_from_exact import build


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def source_row(dealt: list[str]) -> dict:
    return {
        "source": "test",
        "turn": 2,
        "board": {"top": [], "middle": [], "bottom": []},
        "opponent_board": {"top": [], "middle": [], "bottom": []},
        "dealt": dealt,
        "is_btn": True,
        "position": "btn",
    }


def exact_row(record_index: int, dealt: list[str]) -> dict:
    placements = [[dealt[0], "top"], [dealt[1], "middle"]]
    return {
        "record_index": record_index,
        "turn": 2,
        "elapsed_ms": 1.0,
        "candidates": [
            {
                "action": {"placements": placements, "discard": dealt[2]},
                "metrics": {
                    "score": 1.0,
                    "raw_score": 1.0,
                    "royalty": 1.0,
                    "bust_rate": 0.0,
                    "fl_rate": 0.0,
                    "fl_type_rates": {},
                    "samples": 50,
                    "source": "exact_t2_capped",
                },
            }
        ],
    }


def run_build(tmp_path: Path, source_rows: list[dict], exact_rows: list[dict]) -> dict:
    source = tmp_path / "source.jsonl"
    exact = tmp_path / "exact.jsonl"
    output = tmp_path / "teacher.jsonl"
    write_jsonl(source, source_rows)
    write_jsonl(exact, exact_rows)
    return build(
        argparse.Namespace(
            source=str(source),
            exact=str(exact),
            output=str(output),
            eval_mode="exact_t2_capped",
            cap="50",
            source_candidate_top_k=10,
            prefer_row_order=False,
        )
    )


def test_duplicate_record_indices_auto_join_by_row_order(tmp_path: Path) -> None:
    source_rows = [
        source_row(["Ah", "Kd", "2c"]),
        source_row(["3c", "4d", "5h"]),
        source_row(["6s", "7s", "8s"]),
        source_row(["9h", "Th", "Jh"]),
    ]
    exact_rows = [
        exact_row(0, ["Ah", "Kd", "2c"]),
        exact_row(1, ["3c", "4d", "5h"]),
        exact_row(0, ["6s", "7s", "8s"]),
        exact_row(1, ["9h", "Th", "Jh"]),
    ]

    summary = run_build(tmp_path, source_rows, exact_rows)

    assert summary["join_mode"] == "row_order"
    assert summary["written"] == 4
    assert summary["invalid_rows"] == 0
    assert summary["invalid_candidates"] == 0


def test_unique_record_indices_join_by_record_index(tmp_path: Path) -> None:
    source_rows = [
        source_row(["Ah", "Kd", "2c"]),
        source_row(["3c", "4d", "5h"]),
    ]
    exact_rows = [
        exact_row(1, ["3c", "4d", "5h"]),
        exact_row(0, ["Ah", "Kd", "2c"]),
    ]

    summary = run_build(tmp_path, source_rows, exact_rows)

    assert summary["join_mode"] == "record_index"
    assert summary["written"] == 2
    assert summary["invalid_rows"] == 0
    assert summary["invalid_candidates"] == 0
