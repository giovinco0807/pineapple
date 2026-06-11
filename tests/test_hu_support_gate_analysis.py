from ofc_regular.analyze_hu_support_gate_trace import (
    analyze_support_thresholds,
    parse_thresholds,
)


def test_parse_support_thresholds_accepts_comma_separated_values():
    assert parse_thresholds("0, 2,4") == [0.0, 2.0, 4.0]


def test_analyze_support_thresholds_filters_by_support_margin():
    rows = [
        {
            "support_margin": 4.0,
            "counterfactual_delta_vs_baseline": 10.0,
            "support_best_is_chosen": True,
            "support_best_is_baseline": False,
        },
        {
            "support_margin": 1.0,
            "counterfactual_delta_vs_baseline": -2.0,
            "support_best_is_chosen": False,
            "support_best_is_baseline": True,
        },
        {
            "support_margin": 6.0,
            "counterfactual_delta_vs_baseline": 0.0,
            "support_best_is_chosen": False,
            "support_best_is_baseline": False,
        },
    ]

    results = analyze_support_thresholds(rows, thresholds=[0.0, 4.0], paired_seeds=10)

    assert results[0]["kept_overrides"] == 3.0
    assert results[0]["approx_ev_per_hand"] == 0.4
    assert results[0]["wins"] == 1.0
    assert results[0]["losses"] == 1.0
    assert results[1]["kept_overrides"] == 2.0
    assert results[1]["approx_ev_per_hand"] == 0.5
    assert results[1]["support_best_is_chosen"] == 1.0
