import json
from types import SimpleNamespace

from ai.tutor import run_oracle_first_batch
from ai.tutor.run_oracle_first_batch import run, run_t2_cap


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _args(input_path, out_dir, **overrides):
    args = {
        "input": str(input_path),
        "out_dir": str(out_dir),
        "dry_run": True,
        "skip_existing": False,
        "record_indices": "",
        "record_offset": 0,
        "record_limit": 0,
        "progress_every": 100,
        "print_errors": False,
        "skip_late": True,
        "late_turns": "3,4",
        "late_workers": 1,
        "late_max_records": 0,
        "skip_t2": False,
        "t2_caps": "100,200",
        "t2_limit": 10,
        "t2_top_n": 24,
        "t2_manifest": "ai/rust_solver/t3_exact_solver/Cargo.toml",
        "t2_binary": "",
        "fl_config": "ai/config/fl_ev.json",
        "miss_top_k": 5,
        "stable_weak_cap": -1,
        "stable_strong_cap": -1,
        "stable_max_records": 0,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_dry_run_filters_t2_window_and_reports_cap_outputs(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [
            {"turn": 2, "board": {"top": []}},
            {"turn": 3, "board": {"top": []}},
            {"turn": 2, "board": {"top": ["Ah"]}},
            {"turn": 2, "board": {"top": ["Kh"]}},
        ],
    )

    summary = run(_args(source, tmp_path / "out", record_offset=1, record_limit=1))

    assert summary["dry_run"] is True
    assert summary["late_exact"] is None
    assert summary["t2"]["targets"] == 1
    assert [cap["cap"] for cap in summary["t2"]["caps"]] == [100, 200]
    assert all(cap["status"] == "dry_run" for cap in summary["t2"]["caps"])

    t2_input = summary["t2"]["input"]
    with open(t2_input, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["board"]["top"] == ["Ah"]
    assert rows[0]["oracle_first_original_index"] == 2


def test_dry_run_can_select_specific_t2_indices(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [
            {"turn": 2, "board": {"top": ["2h"]}},
            {"turn": 2, "board": {"top": ["3h"]}},
            {"turn": 3, "board": {"top": ["4h"]}},
            {"turn": 2, "board": {"top": ["5h"]}},
        ],
    )

    summary = run(_args(source, tmp_path / "out", record_indices="0,2", record_offset=99))

    assert summary["t2"]["targets"] == 2
    with open(summary["t2"]["input"], "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert [row["board"]["top"] for row in rows] == [["2h"], ["5h"]]


def test_t2_cap_passes_complete_exact_runner_namespace(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"turn": 2, "board": {"top": []}}])
    captured = {}

    def fake_run_solver(args, output):
        captured["args"] = args
        output.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        run_oracle_first_batch.run_t2_exact_oracle,
        "run_solver",
        fake_run_solver,
    )
    monkeypatch.setattr(
        run_oracle_first_batch.run_t2_exact_oracle,
        "compare",
        lambda *_args, **_kwargs: {"records": 1, "misses": []},
    )
    monkeypatch.setattr(
        run_oracle_first_batch.run_t2_exact_oracle,
        "write_markdown",
        lambda *_args, **_kwargs: None,
    )

    result = run_t2_cap(
        _args(source, tmp_path / "out", dry_run=False),
        source,
        tmp_path / "out",
        100,
    )

    assert result["status"] == "ran"
    assert captured["args"].skip == 0
    assert captured["args"].source_candidate_top_k == 0
