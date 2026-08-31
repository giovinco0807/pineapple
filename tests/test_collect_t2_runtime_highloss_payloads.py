import argparse
import json
from pathlib import Path

from ai.tutor.collect_t2_runtime_highloss_payloads import collect


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def runtime_row(source_line: int, *, loss, source_record: int = 100, chosen_key: str = "chosen") -> dict:
    return {
        "source_line": source_line,
        "payload": {
            "source": "fixture",
            "source_line": source_record,
            "turn": 2,
            "board": {"top": [], "middle": [], "bottom": []},
            "dealt": ["As", "Kd", "2c"],
            "candidates": [],
        },
        "result": {
            "mode": "fast",
            "best": {"action_idx": 3},
            "model_top1_action_idx": 1,
            "elapsed_ms": 123.0,
            "candidate_pool_size": 20,
            "exact_evaluated": 50,
        },
        "teacher": {
            "chosen_ev_loss": loss,
            "chosen_action_key": chosen_key,
            "teacher_best_action_key": "best",
            "model_top1_action_key": "top1",
            "model_top1_ev_loss": 2.0,
        },
    }


def test_collect_filters_high_loss_payloads_and_adds_runtime_metadata(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline" / "results.jsonl"
    duplicate = tmp_path / "duplicate" / "results.jsonl"
    write_jsonl(
        baseline,
        [
            runtime_row(1, loss=0.1, source_record=1),
            runtime_row(2, loss=0.75, source_record=2, chosen_key="selected"),
            runtime_row(3, loss=None, source_record=3),
        ],
    )
    write_jsonl(
        duplicate,
        [
            runtime_row(1, loss=0.8, source_record=2, chosen_key="selected"),
            runtime_row(2, loss=1.25, source_record=4, chosen_key="other"),
        ],
    )

    output = tmp_path / "out.jsonl"
    summary = collect(
        argparse.Namespace(
            set=[f"base={baseline}", f"dup={duplicate.parent}"],
            threshold=0.5,
            output=str(output),
            summary="",
            allow_duplicates=False,
        )
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["counts"]["rows_seen"] == 5
    assert summary["counts"]["selected"] == 2
    assert summary["counts"]["duplicate_skipped"] == 1
    assert summary["counts"]["unknown_loss"] == 1
    assert [row["runtime_chosen_ev_loss"] for row in rows] == [0.75, 1.25]
    assert rows[0]["runtime_eval_set"] == "base"
    assert rows[0]["runtime_selected_action_idx"] == 3
    assert rows[0]["runtime_model_top1_action_idx"] == 1
    assert rows[1]["runtime_source_line"] == 2
