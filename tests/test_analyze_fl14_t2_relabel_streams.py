import json

from ai.tutor.analyze_fl14_t2_relabel_streams import analyze


def _action(key: str, value: float, draws: int = 96) -> dict:
    return {"action_key": key, "value": value, "t3_draws": draws}


def _record(root_id: str, actions: list[dict], opponents: int = 120) -> dict:
    return {"id": root_id, "opponents": opponents, "actions": actions}


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_common_stream_offsets_cancel_and_pilot_passes(tmp_path):
    expected = tmp_path / "roots.jsonl"
    stream_a = tmp_path / "a.jsonl"
    stream_b = tmp_path / "b.jsonl"
    _write_jsonl(expected, [{"id": "r1"}, {"id": "r2"}])
    _write_jsonl(
        stream_a,
        [
            _record("r1", [_action("a", 10), _action("b", 8), _action("c", 5)]),
            _record("r2", [_action("a", 4), _action("b", 3), _action("c", 2)]),
        ],
    )
    _write_jsonl(
        stream_b,
        [
            _record("r1", [_action("a", 12), _action("b", 10), _action("c", 7)]),
            _record("r2", [_action("a", 5), _action("b", 4), _action("c", 3)]),
        ],
    )

    report = analyze(
        stream_a,
        stream_b,
        expected_roots_path=expected,
        expected_opponents=120,
        expected_t3_draws=96,
    )

    assert report["passed"] is True
    assert report["metrics"]["best_agreement_rate"] == 1.0
    assert report["metrics"]["root_centered_action_difference_rmse"] == 0.0
    assert report["metrics"]["top5_pair_gap_difference_rmse"] == 0.0
    assert report["stable_ids"] == ["r1", "r2"]
    assert report["adjudicate_ids"] == []


def test_disagreement_or_small_margin_requires_adjudication(tmp_path):
    stream_a = tmp_path / "a.jsonl"
    stream_b = tmp_path / "b.jsonl"
    _write_jsonl(
        stream_a,
        [
            _record("r1", [_action("a", 2), _action("b", 1)]),
            _record("r2", [_action("a", 2.2), _action("b", 2.0)]),
        ],
    )
    _write_jsonl(
        stream_b,
        [
            _record("r1", [_action("a", 1), _action("b", 2)]),
            _record("r2", [_action("a", 2.2), _action("b", 2.0)]),
        ],
    )

    report = analyze(
        stream_a,
        stream_b,
        stable_margin=0.5,
        min_best_agreement_rate=0.5,
        max_root_centered_rmse=10.0,
        max_top5_pair_gap_rmse=10.0,
    )

    assert report["metrics"]["best_agreement_count"] == 1
    assert report["stable_ids"] == []
    assert report["adjudicate_ids"] == ["r1", "r2"]


def test_missing_action_mismatch_and_short_draws_fail_integrity(tmp_path):
    expected = tmp_path / "roots.jsonl"
    stream_a = tmp_path / "a.jsonl"
    stream_b = tmp_path / "b.jsonl"
    _write_jsonl(expected, [{"id": "r1"}, {"id": "r2"}, {"id": "r3"}])
    _write_jsonl(
        stream_a,
        [
            _record("r1", [_action("a", 2), _action("b", 1)]),
            _record("r2", [_action("a", 2, draws=95), _action("b", 1, draws=95)]),
            _record("r3", [_action("a", 2), _action("b", 1)]),
        ],
    )
    _write_jsonl(
        stream_b,
        [
            _record("r1", [_action("a", 2), _action("c", 1)]),
            _record("r2", [_action("a", 2), _action("b", 1)]),
        ],
    )

    report = analyze(
        stream_a,
        stream_b,
        expected_roots_path=expected,
        expected_opponents=120,
        expected_t3_draws=96,
    )

    assert report["passed"] is False
    assert report["checks"]["integrity"] is False
    assert report["integrity"]["missing_ids_b"] == ["r3"]
    assert report["integrity"]["action_set_mismatch_ids"] == ["r1"]
    assert report["integrity"]["short_or_setting_mismatch_ids"] == ["r2"]
