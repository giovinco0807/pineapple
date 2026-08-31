import json

from ai.tutor.evaluate_hybrid_refinement_teacher import _iter_jsonl


def test_iter_jsonl_supports_skip_and_limit(tmp_path):
    path = tmp_path / "records.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"i": i}) + "\n")

    rows = list(_iter_jsonl(path, limit=2, skip=2))

    assert [line_no for line_no, _record in rows] == [3, 4]
    assert [record["i"] for _line_no, record in rows] == [2, 3]
