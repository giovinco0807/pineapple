import json

import pytest

from ai.tutor import evaluate_t3_runtime_pool_teacher as evaluator


ACTION_A = {"placements": [["Jc", "top"], ["Qd", "middle"]], "discard": "Kh"}
ACTION_B = {"placements": [["Jc", "middle"], ["Qd", "bottom"]], "discard": "Kh"}
ACTION_C = {"placements": [["Jc", "middle"], ["Qd", "middle"]], "discard": "Kh"}
ACTION_UNKNOWN = {"placements": [["Jc", "bottom"], ["Qd", "middle"]], "discard": "Kh"}


def teacher_row(position: str, candidates: list[tuple[dict, float]]) -> dict:
    opponent_board = {
        "top": ["2h", "3h"],
        "middle": ["4h", "5h", "6d"] + (["Jh"] if position == "btn" else []),
        "bottom": ["7d", "8h", "9d", "Td"] + (["Qh"] if position == "btn" else []),
    }
    return {
        "source": "unit",
        "source_line": 1,
        "turn": 3,
        "position": position,
        "is_btn": position == "btn",
        "board": {
            "top": ["2c", "3d"],
            "middle": ["4c", "5d", "6h"],
            "bottom": ["7c", "8d", "9h", "Ts"],
        },
        "opponent_board": opponent_board,
        "dealt": ["Jc", "Qd", "Kh"],
        "exclude": [],
        "candidates": [
            {"placements": action["placements"], "discard": action["discard"], "ev": ev}
            for action, ev in candidates
        ],
    }


class FakePooler:
    def __init__(self, pools: dict[str, list[dict]]):
        self.pools = pools

    def build_pool(self, _obs, *, position: str, **_kwargs):
        candidates = self.pools[position]
        return {
            "position": position,
            "pool_policy": "fast",
            "per_source_top_k": 10,
            "legal_actions": 21,
            "candidate_pool_size": len(candidates),
            "sources": [{"name": "fake", "kind": "set"}],
            "candidates": [
                {
                    "action": action,
                    "best_source_rank": rank,
                    "source_ranks": {"set:fake": rank},
                }
                for rank, action in enumerate(candidates, start=1)
            ],
        }


def test_tie_aware_hit_ev_loss_unmatched_and_by_position():
    rows = [
        teacher_row("bb", [(ACTION_A, 5.0), (ACTION_B, 5.0), (ACTION_C, 3.0)]),
        teacher_row("btn", [(ACTION_A, 7.0), (ACTION_B, 4.0)]),
    ]
    pooler = FakePooler(
        {
            "bb": [ACTION_B, ACTION_C, ACTION_UNKNOWN],
            "btn": [ACTION_B],
        }
    )

    summary, details = evaluator.evaluate_teacher_rows(rows, pooler)

    assert details[0]["exact_best_hit"] is True
    assert details[0]["teacher_best_action_keys"] != []
    assert len(details[0]["teacher_best_action_keys"]) == 2
    assert details[0]["ev_loss"] == pytest.approx(0.0)
    assert details[0]["unmatched_pool_actions"] == 1
    assert details[1]["exact_best_hit"] is False
    assert details[1]["ev_loss"] == pytest.approx(3.0)

    assert summary["exact_best_recall"] == pytest.approx(0.5)
    assert summary["zero_ev_loss_rate"] == pytest.approx(0.5)
    assert summary["ev_loss"]["mean"] == pytest.approx(1.5)
    assert summary["pool_size"]["mean"] == pytest.approx(2.0)
    assert summary["unmatched_pool_actions"]["total"] == 1
    assert summary["by_position"]["bb"]["exact_best_recall"] == pytest.approx(1.0)
    assert summary["by_position"]["btn"]["ev_loss"]["max"] == pytest.approx(3.0)


def test_all_unmatched_pool_reports_missing_ev_loss():
    row = teacher_row("bb", [(ACTION_A, 5.0), (ACTION_B, 4.0)])
    summary, details = evaluator.evaluate_teacher_rows(
        [row],
        FakePooler({"bb": [ACTION_UNKNOWN]}),
    )

    assert details[0]["pool_best_ev"] is None
    assert details[0]["ev_loss"] is None
    assert details[0]["exact_best_hit"] is False
    assert summary["evaluable_rows"] == 0
    assert summary["ev_loss_missing_rows"] == 1
    assert summary["unmatched_pool_actions"]["rate"] == pytest.approx(1.0)


def test_pool_best_equal_ev_uses_exact_teacher_quality_order():
    row = teacher_row("bb", [(ACTION_A, 5.0), (ACTION_B, 5.0)])

    _summary, details = evaluator.evaluate_teacher_rows(
        [row],
        FakePooler({"bb": [ACTION_B, ACTION_A]}),
    )

    assert details[0]["pool_best_action"] == ACTION_A
    assert [candidate["teacher_rank"] for candidate in details[0]["pool_candidates"]] == [2, 1]


def test_summary_reports_pool_latency_percentiles():
    rows = [
        {
            "position": "bb",
            "ev_loss": 0.0,
            "candidate_pool_size": 10,
            "unique_pool_actions": 10,
            "unmatched_pool_actions": 0,
            "exact_best_hit": True,
            "zero_ev_loss": True,
            "pool_elapsed_ms": 5.0,
        },
        {
            "position": "bb",
            "ev_loss": 1.0,
            "candidate_pool_size": 20,
            "unique_pool_actions": 20,
            "unmatched_pool_actions": 0,
            "exact_best_hit": False,
            "zero_ev_loss": False,
            "pool_elapsed_ms": 15.0,
        },
    ]

    summary = evaluator.summarize_bucket(rows)

    assert summary["pool_elapsed_ms"] == {
        "mean": pytest.approx(10.0),
        "p50": pytest.approx(10.0),
        "p95": pytest.approx(14.5),
        "p99": pytest.approx(14.9),
        "max": pytest.approx(15.0),
    }


def test_cli_writes_summary_and_rows_jsonl(tmp_path, monkeypatch, capsys):
    teacher_path = tmp_path / "teacher.jsonl"
    config_path = tmp_path / "pool.json"
    summary_path = tmp_path / "summary.json"
    rows_path = tmp_path / "rows.jsonl"
    row = teacher_row("bb", [(ACTION_A, 5.0), (ACTION_B, 4.0)])
    teacher_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config_path.write_text("{}\n", encoding="utf-8")

    class CliPooler(FakePooler):
        def __init__(self, config_path, device, pool_policy):
            assert str(config_path).endswith("pool.json")
            assert device == "cpu"
            assert pool_policy == "fast"
            super().__init__({"bb": [ACTION_A]})

    monkeypatch.setattr(evaluator, "T3UnionCandidatePool", CliPooler)

    summary = evaluator.main(
        [
            "--teacher",
            str(teacher_path),
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--output",
            str(summary_path),
            "--rows-output",
            str(rows_path),
        ]
    )

    assert summary["rust_exact_calls"] == 0
    assert summary["exact_best_recall"] == pytest.approx(1.0)
    assert summary["pool_elapsed_ms"]["p95"] >= 0.0
    assert json.loads(summary_path.read_text(encoding="utf-8"))["zero_ev_loss_rate"] == pytest.approx(1.0)
    written_rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    assert len(written_rows) == 1
    assert written_rows[0]["pool_best_action"] == ACTION_A
    assert "rust_exact_calls" in capsys.readouterr().out
