import csv

from ofc_regular.analyze_hu_turn3_stage9_tail_veto_thresholds import (
    read_rows,
    threshold_rows,
    write_csv,
)


def test_tail_veto_threshold_vetoes_high_hard_negative_probability(tmp_path):
    rows = [
        {
            "split": "test",
            "tail_label": "safe_positive",
            "delta_vs_baseline": "3.0",
            "hard_negative_probability": "0.1",
        },
        {
            "split": "test",
            "tail_label": "hard_negative",
            "delta_vs_baseline": "-2.0",
            "hard_negative_probability": "0.8",
        },
        {
            "split": "train",
            "tail_label": "hard_negative",
            "delta_vs_baseline": "-1.0",
            "hard_negative_probability": "0.4",
        },
    ]

    result = threshold_rows(rows, thresholds=[0.5])
    all_row = next(row for row in result if row["split"] == "all")
    test_row = next(row for row in result if row["split"] == "test")

    assert all_row["kept_rows"] == 2
    assert all_row["kept_hard_negative_count"] == 1
    assert all_row["vetoed_hard_negative_count"] == 1
    assert all_row["vetoed_safe_positive_count"] == 0
    assert test_row["kept_rows"] == 1
    assert test_row["kept_delta_sum"] == 3.0

    path = tmp_path / "sweep.csv"
    write_csv(path, result)
    written = list(csv.DictReader(path.open("r", encoding="utf-8")))
    assert written[0]["threshold"] == "0.5"


def test_tail_veto_threshold_read_rows_utf8_sig(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text(
        "\ufeffsplit,tail_label,delta_vs_baseline,hard_negative_probability\n"
        "test,hard_negative,-1.5,0.9\n",
        encoding="utf-8",
    )

    rows = read_rows(path)

    assert rows[0]["split"] == "test"
    assert rows[0]["tail_label"] == "hard_negative"
