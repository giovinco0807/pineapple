import csv

from ofc_regular.select_hu_turn0_safe_selector_candidate import select_candidate


def _candidate(tmp_path, name, *, lcb, hard_rate=0.0, fires=20, positives=15):
    path = tmp_path / f"{name}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "threshold",
                "fires",
                "safe_positive_count",
                "hard_negative_rate",
                "mc_delta_ci95_low",
                "p95_mc_loss",
                "precision_safe_positive",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "extra_trees",
                "threshold": 0.9,
                "fires": fires,
                "safe_positive_count": positives,
                "hard_negative_rate": hard_rate,
                "mc_delta_ci95_low": lcb,
                "p95_mc_loss": 2.0,
                "precision_safe_positive": positives / fires,
            }
        )
    return {
        "run": name,
        "best_model": "extra_trees",
        "best_oof_average_precision": 0.5,
        "model": f"{name}.pkl",
        "threshold_sweep": str(path),
    }


def test_selector_uses_fixed_constraints_and_highest_delta_lcb(tmp_path):
    candidates = [
        _candidate(tmp_path, "lower", lcb=0.2),
        _candidate(tmp_path, "higher", lcb=0.5),
        _candidate(tmp_path, "unsafe", lcb=2.0, hard_rate=0.5),
    ]

    result = select_candidate(
        candidates,
        min_fires=10,
        min_safe_positives=5,
        max_hard_negative_rate=0.1,
        min_mc_delta_ci95_low=0.0,
        max_p95_mc_loss=10.0,
    )

    assert result["decision"] == "Go"
    assert result["selected"]["run"] == "higher"
    assert result["selected"]["model"] == "higher.pkl"
    assert result["selected"]["estimator_name"] == "extra_trees"
    assert result["eligible_count"] == 2


def test_selector_returns_no_go_when_no_candidate_passes(tmp_path):
    result = select_candidate(
        [_candidate(tmp_path, "weak", lcb=-0.1)],
        min_fires=10,
        min_safe_positives=5,
        max_hard_negative_rate=0.1,
        min_mc_delta_ci95_low=0.0,
        max_p95_mc_loss=10.0,
    )

    assert result["decision"] == "No-Go"
    assert result["selected"] is None
