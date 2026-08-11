import csv
import json

from ofc_regular.analyze_hu_turn2_stage8c_candidate_generators import (
    candidate_metrics,
    overlap_metrics,
    write_markdown,
)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _eval_dir(tmp_path, name, fired_rows):
    path = tmp_path / name
    _write_csv(path / "seed_breakdown.csv", [{"config_id": "cfg", "paired_seeds": "10", "ev_per_hand": "0.1"}])
    decisions = []
    for row in fired_rows:
        decisions.append(
            {
                "config_id": "cfg",
                "hand_id": row["hand_id"],
                "seat_swap": "ab",
                "street": "T2",
                "seat": "first",
                "override_fired": True,
                "realized_delta_valid": True,
                "realized_candidate_seat_delta": row["delta"],
                "final_action": row["action"],
            }
        )
    decisions.append(
        {
            "config_id": "cfg",
            "hand_id": 999,
            "seat_swap": "ab",
            "street": "T2",
            "seat": "first",
            "override_fired": False,
            "realized_delta_valid": True,
            "realized_candidate_seat_delta": 0.0,
            "no_override_reason": "topk_empty",
        }
    )
    _write_jsonl(path / "runtime_decisions.jsonl", decisions)
    _write_csv(
        path / "cancellation_audit.csv",
        [
            {
                "config_id": "cfg",
                "valid_realized_delta_count": str(len(decisions)),
                "non_fired_count": "1",
                "non_fired_nonzero_count": "0",
                "non_fired_delta_sum": "0",
                "non_fired_delta_max_abs": "0",
                "fired_count": str(len(fired_rows)),
                "fired_delta_sum": str(sum(row["delta"] for row in fired_rows)),
            }
        ],
    )
    return path


def _eval_dir_without_cancellation(tmp_path, name, fired_rows):
    path = _eval_dir(tmp_path, name, fired_rows)
    (path / "cancellation_audit.csv").unlink()
    return path


def test_candidate_generator_comparison_reports_overlap_and_metrics(tmp_path):
    base = _eval_dir(
        tmp_path,
        "base",
        [
            {"hand_id": 1, "delta": 2.0, "action": {"placements": [["Ah", "top"]]}},
            {"hand_id": 2, "delta": 4.0, "action": {"placements": [["Kh", "middle"]]}},
        ],
    )
    plus = _eval_dir(
        tmp_path,
        "plus",
        [
            {"hand_id": 1, "delta": 6.0, "action": {"placements": [["Ah", "top"]]}},
            {"hand_id": 3, "delta": -1.0, "action": {"placements": [["Qh", "bottom"]]}},
        ],
    )

    groups = {"base": [base], "plus": [plus]}
    metrics = candidate_metrics(groups)
    overlaps = overlap_metrics(groups)

    assert [row["group"] for row in metrics] == ["base", "plus"]
    assert metrics[0]["fires"] == 2
    assert metrics[0]["non_fired_nonzero_count"] == 0
    assert metrics[0]["cancellation_audit_present"] is True
    assert metrics[0]["cancellation_clean"] is True
    assert metrics[0]["primary_metric_valid"] is True
    assert metrics[0]["metric_exclusion_reason"] == ""
    assert len(overlaps) == 1
    assert overlaps[0]["overlap"] == 1
    assert overlaps[0]["union"] == 3
    assert overlaps[0]["same_action_overlap"] == 1


def test_candidate_generator_comparison_marks_missing_cancellation_audit_no_go(tmp_path):
    missing = _eval_dir_without_cancellation(
        tmp_path,
        "missing",
        [{"hand_id": 1, "delta": 2.0, "action": {"placements": [["Ah", "top"]]}}],
    )
    clean = _eval_dir(
        tmp_path,
        "clean",
        [{"hand_id": 2, "delta": 4.0, "action": {"placements": [["Kh", "middle"]]}}],
    )

    metrics = candidate_metrics({"clean": [clean], "missing": [missing]})
    by_group = {row["group"]: row for row in metrics}

    assert by_group["missing"]["cancellation_audit_present"] is False
    assert by_group["missing"]["cancellation_clean"] is False
    assert by_group["missing"]["primary_metric_valid"] is False
    assert by_group["missing"]["metric_exclusion_reason"] == "missing_cancellation_audit"
    assert by_group["clean"]["primary_metric_valid"] is True


def test_candidate_generator_markdown_is_validation_only(tmp_path):
    path = tmp_path / "summary.md"
    write_markdown(
        path,
        [
            {
                "group": "base",
                "paired_seeds": 10,
                "fires": 2,
                "override_rate": 0.2,
                "estimated_ev_per_hand": 0.3,
                "per_fire_delta_mean": 1.5,
                "per_fire_delta_ci95_low": 0.1,
                "per_fire_delta_ci95_high": 2.9,
                "realized_loss_count": 0,
                "max_loss": 0.0,
                "primary_metric_valid": True,
                "non_fired_nonzero_count": 0,
                "metric_exclusion_reason": "",
            }
        ],
        [
            {
                "left_group": "base",
                "right_group": "plus",
                "overlap": 1,
                "union": 3,
                "jaccard": 1 / 3,
                "left_only": 1,
                "right_only": 1,
                "same_action_overlap": 1,
            }
        ],
    )

    text = path.read_text(encoding="utf-8")
    assert "production / P2 fixed: `No-Go`" in text
    assert "base" in text
    assert "primary valid" in text
    assert "Jaccard" in text
