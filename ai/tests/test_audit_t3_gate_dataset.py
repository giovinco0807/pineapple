import json

from ai.tutor.audit_t3_gate_dataset import audit_dataset, decision_signature


def _record(root: int, *, position: str = "bb", dealt=None):
    return {
        "turn": 3,
        "position": position,
        "board": {"top": ["As"], "middle": ["2c"], "bottom": ["3d"]},
        "opponent_board": {"top": ["Kh"], "middle": [], "bottom": []},
        "dealt": dealt or ["4s", "5s", "6s"],
        "known_discards": ["7h"],
        "exclude": ["Kh", "7h"],
        "branch": {"root_index": root},
    }


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_signature_ignores_card_order_and_row_aliases():
    first = _record(0)
    second = _record(9)
    second["board"] = {"top": ["As"], "mid": ["2c"], "bot": ["3d"]}
    second["dealt"] = list(reversed(second["dealt"]))

    assert decision_signature(first) == decision_signature(second)


def test_audit_reports_duplicates_overlap_positions_and_jokers(tmp_path):
    duplicate = _record(1)
    btn_joker = _record(2, position="btn", dealt=["X1", "8c", "9c"])
    input_path = tmp_path / "input.jsonl"
    compare_path = tmp_path / "compare.jsonl"
    _write_jsonl(input_path, [_record(0), duplicate, btn_joker])
    _write_jsonl(compare_path, [_record(99)])

    result = audit_dataset(input_path, [compare_path])

    assert result["records"] == 3
    assert result["unique_decisions"] == 2
    assert result["duplicate_occurrences"] == 1
    assert result["positions"] == {"bb": 2, "btn": 1}
    assert result["jokers"]["visible_records"] == 1
    assert result["comparisons"][0]["overlap_decisions"] == 1
    assert result["prior_union_overlap_decisions"] == 1
    assert result["gate_checks"] == {
        "nonempty": True,
        "all_decisions_unique": False,
        "both_positions_present": True,
        "no_prior_overlap": False,
    }
