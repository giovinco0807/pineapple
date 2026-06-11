from ofc_regular.analyze_hu_override_trace import analyze_thresholds, parse_thresholds


def test_parse_thresholds_accepts_comma_separated_values():
    assert parse_thresholds("8, 10,12") == [8.0, 10.0, 12.0]


def test_analyze_thresholds_sweeps_predicted_margin():
    rows = [
        {
            "predicted_margin": 8.0,
            "counterfactual_delta_vs_baseline": 10.0,
            "candidate_terminal_score": 7.0,
        },
        {
            "predicted_margin": 11.0,
            "counterfactual_delta_vs_baseline": -2.0,
            "candidate_terminal_score": -1.0,
        },
        {
            "predicted_margin": 15.0,
            "counterfactual_delta_vs_baseline": 0.0,
            "candidate_terminal_score": 0.0,
        },
    ]

    results = analyze_thresholds(rows, thresholds=[8.0, 12.0], paired_seeds=10)

    assert results[0]["overrides"] == 3.0
    assert results[0]["wins"] == 1.0
    assert results[0]["losses"] == 1.0
    assert results[0]["ties"] == 1.0
    assert results[0]["approx_ev_per_hand"] == 0.4
    assert results[1]["overrides"] == 1.0
    assert results[1]["delta_avg"] == 0.0
