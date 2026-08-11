import csv
import json

from ofc_regular.analyze_hu_turn2_stage8c_ranker_replay import (
    analyze,
    breakdown,
    enriched_rows,
)


def _summary_row(row_index, delta, **overrides):
    row = {
        "row_index": str(row_index),
        "status": "ok",
        "action_mapping_status": "ok",
        "delta_for_label": str(delta),
        "delta_standard_error_for_label": "0.5",
        "safe_lcb196_label": "gray" if delta > 0 else "negative",
        "safe_lcb164_label": "gray" if delta > 0 else "negative",
        "hard_negative_label": "1" if delta < 0 else "0",
    }
    row.update(overrides)
    return row


def _teacher_row(ranker, delta, **source_overrides):
    source = {
        "ranker": ranker,
        "ranker_rank": 1,
        "ranker_score": 2.0,
        "hand_seed": 123,
        "seat": "first",
        "candidate_source": "below_confirm_se",
        "recommended_training_use": "topk_confirm_rejected",
        "candidate_ev_rank": 2,
        "fire_probability": 0.8,
        "policy_delta_prediction": 1.0,
        "predicted_delta": 0.5,
        "confirm_delta": 2.0,
        "confirm_delta_se": 0.4,
    }
    source.update(source_overrides)
    return {
        "hand_seed": source["hand_seed"],
        "topk_hard_negative_source": source,
        "topk_hard_negative_replay": {
            "action_mapping_status": "ok",
            "delta_for_label": delta,
            "delta_standard_error_for_label": 0.5,
            "safe_lcb196_label": "gray" if delta > 0 else "negative",
            "safe_lcb164_label": "gray" if delta > 0 else "negative",
            "hard_negative_label": int(delta < 0),
            "replay_delta_lcb196": delta - 0.98,
            "replay_delta_lcb164": delta - 0.82,
        },
    }


def test_enriched_rows_and_breakdown_by_ranker():
    summary = [_summary_row(0, 2.0), _summary_row(1, -1.0)]
    teachers = [_teacher_row("a", 2.0), _teacher_row("b", -1.0)]

    rows = enriched_rows(summary, teachers)
    grouped = breakdown(rows)
    ranker_rows = {row["group_value"]: row for row in grouped if row["group_field"] == "ranker"}

    assert len(rows) == 2
    assert ranker_rows["a"]["mean_delta"] == 2.0
    assert ranker_rows["b"]["hard_negative_rows"] == 1
    assert grouped[0]["rows"] == 2
    assert grouped[0]["primary_metric_source"] == "independent_replay_delta_for_label"
    assert grouped[0]["confirm_delta_metric_role"] == "source_gate_diagnostic_only"
    assert grouped[0]["confirm_delta_performance_claim_allowed"] is False


def test_analyze_writes_ranker_outputs(tmp_path):
    summary_path = tmp_path / "summary.csv"
    teacher_path = tmp_path / "teacher.jsonl"
    output_dir = tmp_path / "analysis"

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_summary_row(0, 2.0).keys()))
        writer.writeheader()
        writer.writerow(_summary_row(0, 2.0))
        writer.writerow(_summary_row(1, -1.0))
    teacher_path.write_text(
        "\n".join(json.dumps(row) for row in [_teacher_row("a", 2.0), _teacher_row("b", -1.0)]) + "\n",
        encoding="utf-8",
    )

    manifest = analyze(
        replay_summary_csv=summary_path,
        replay_teacher_jsonl=teacher_path,
        output_dir=output_dir,
        top_loss_count=1,
    )

    assert manifest["rows"] == 2
    assert manifest["primary_metric_source"] == "independent_replay_delta_for_label"
    assert manifest["confirm_delta_metric_role"] == "source_gate_diagnostic_only"
    assert manifest["confirm_delta_performance_claim_allowed"] is False
    assert (output_dir / "ranker_replay_summary.md").exists()
    assert (output_dir / "ranker_replay_breakdown.csv").exists()
    text = (output_dir / "ranker_replay_summary.md").read_text(encoding="utf-8")
    assert "runtime gate: `No-Go`" in text
    assert "`mean_confirm_delta` is `source_gate_diagnostic_only`" in text
