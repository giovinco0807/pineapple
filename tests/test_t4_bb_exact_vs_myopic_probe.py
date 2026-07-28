import pytest

import ai.tutor.exact_late as exact_late_module
import ai.tutor.t4_bb_exact_vs_myopic_probe as probe
from ai.engine.encoding import ALL_CARDS


def test_random_root_is_a_seeded_54_card_partition():
    root = probe.sample_random_root(seed=42)
    again = probe.sample_random_root(seed=42)
    assert root == again

    bb_cards = [card for row in root["bb_board"] for card in row]
    btn_cards = [card for row in root["btn_board"] for card in row]
    used = bb_cards + list(root["bb_discards"]) + btn_cards + list(root["draw"])
    assert len(used) == 11 + 3 + 11 + 3
    assert len(set(used)) == len(used)
    assert set(used) <= set(ALL_CARDS)

    assert tuple(len(row) for row in root["bb_board"]) in probe.TWO_OPEN_SHAPES
    assert tuple(len(row) for row in root["btn_board"]) in probe.TWO_OPEN_SHAPES

    other = probe.sample_random_root(seed=43)
    assert other != root


def test_fl_live_root_reuses_sensitivity_fixture():
    root = probe.fl_live_root(("Ts", "9d", "5s"))
    assert root["generator"] == "fl_live"
    assert [len(row) for row in root["bb_board"]] == [2, 5, 4]
    assert [len(row) for row in root["btn_board"]] == [2, 5, 4]
    assert root["draw"] == ["Ts", "9d", "5s"]


def test_evaluate_root_regret_is_exact_table_gap(monkeypatch):
    root = probe.sample_random_root(seed=7)

    def stub_distribution(final_board, opponent_board, exclude=None):
        # Deterministic fake EV that varies by the discarded card so the
        # exact argmax is well-defined without the expensive enumeration.
        discard = sorted(set(exclude) - set(root["bb_discards"]))[0]
        return {"score": float(ord(discard[0]) + ord(discard[1]) / 100.0)}

    with monkeypatch.context() as patch:
        patch.setattr(
            exact_late_module,
            "exact_t4_opponent_response_distribution",
            stub_distribution,
        )
        row = probe._evaluate_root(root)

    assert row["legal_action_count"] >= 3
    assert row["myopic_regret"] == pytest.approx(
        row["exact_ev"] - row["myopic_ev"]
    )
    assert row["myopic_regret"] >= 0.0
    assert row["fired"] == (row["exact_action"] != row["myopic_action"])


def test_aggregate_summary_fields():
    rows = [
        {"fired": False, "myopic_regret": 0.0},
        {"fired": True, "myopic_regret": 2.0},
        {"fired": True, "myopic_regret": 4.0},
    ]
    summary = probe._aggregate(rows)
    assert summary["roots"] == 3
    assert summary["fired"] == 2
    assert summary["fire_rate"] == pytest.approx(2 / 3)
    assert summary["mean_regret"] == pytest.approx(2.0)
    assert summary["mean_regret_when_fired"] == pytest.approx(3.0)
    assert summary["max_regret"] == 4.0
    assert summary["min_regret"] == 0.0
