import pytest

from ai.tutor.compute_fl_ev import COUNTS, FlEvInputs, solve


def _base_inputs(**overrides):
    inputs = FlEvInputs(
        immediate_vs_normal={14: 10.0, 15: 15.0, 16: 20.0, 17: 25.0},
        stay={14: 0.35, 15: 0.50, 16: 0.65, 17: 0.78},
    )
    for key, value in overrides.items():
        setattr(inputs, key, value)
    return inputs


def test_closed_form_geometric_compounding_without_entries():
    """No opponent entries: F(n) = e / (1 - stay) exactly (count carryover)."""
    inputs = _base_inputs()
    result = solve(inputs)
    for n in COUNTS:
        expected = inputs.immediate_vs_normal[n] / (1.0 - inputs.stay[n])
        assert result["fl_ev"][str(n)] == pytest.approx(expected, rel=1e-8)
    # 17-card chain compounds ~4.5x, the number the rules correction predicts.
    assert result["fl_ev"]["17"] == pytest.approx(25.0 / 0.22, rel=1e-8)


def test_equal_count_ff_is_zero_and_antisymmetry_holds():
    inputs = _base_inputs(
        immediate_ff={(17, 14): 8.0},
        stay_both={(17, 14): 0.25},
    )
    result = solve(inputs)
    assert result["fl_vs_fl"]["14v17"] == pytest.approx(
        -_g_from(result, 17, 14), rel=1e-8
    )


def _g_from(result, n, m):
    key = f"{min(n,m)}v{max(n,m)}"
    value = result["fl_vs_fl"][key]
    return value if n < m else -value


def test_opponent_entries_reduce_the_chain():
    """If the opponent enters a big FL whenever the hero fails to stay, the
    hero's FL value must drop below the no-entry closed form."""
    no_entry = solve(_base_inputs())
    with_entry = solve(
        _base_inputs(
            entry_given_no_stay={
                n: {0: 0.7, 17: 0.3} for n in COUNTS
            }
        )
    )
    for n in COUNTS:
        assert with_entry["fl_ev"][str(n)] < no_entry["fl_ev"][str(n)]


def test_dependent_joint_stay_matters():
    """Joint stay below independence weakens the both-stay branch; with a
    positive ff edge for the hero that must move the value."""
    base = _base_inputs(
        entry_given_stay={17: {0: 0.5, 14: 0.5}},
        immediate_ff={(17, 14): 8.0},
    )
    independent = solve(base)
    dependent = solve(
        _base_inputs(
            entry_given_stay={17: {0: 0.5, 14: 0.5}},
            immediate_ff={(17, 14): 8.0},
            stay_both={(17, 14): 0.10},  # below 0.78*0.35=0.273
        )
    )
    assert dependent["fl_ev"]["17"] != pytest.approx(
        independent["fl_ev"]["17"], rel=1e-6
    )


def test_convergence_reported():
    result = solve(_base_inputs())
    assert result["iterations"] < 10_000
    assert result["count_carryover"] is True
