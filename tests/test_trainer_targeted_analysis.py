"""Grading one street of one hand, and folding it into what was already there.

A hand is ten decisions and the deep rungs cost minutes each, so the rung a
spot deserves is not one anybody wants to pay ten times.  Targeting a turn is
what makes those rungs usable -- and the hazard it introduces is the one these
tests pin: a partial result must not overwrite a whole-hand analysis, because
the hand list reads ``hero_total_ev_loss`` straight out of it and would then
report one street's loss as the hand's.

No engine needed: the filter, the merge and the persistence are all reachable
without grading anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from trainer import handlog
from trainer.store import TrainerStore


def _hand():
    """A complete five-street hand, hero acting first, no Fantasyland."""
    deals = {
        0: (["As", "Kd", "Qc", "7h", "2s"], ["Ac", "Kh", "Qd", "7s", "2c"]),
        1: (["3d", "4d", "5d"], ["3c", "4c", "5c"]),
        2: (["6h", "8h", "9h"], ["6s", "8s", "9s"]),
        3: (["Th", "Jh", "2h"], ["Ts", "Js", "2d"]),
        4: (["3h", "4h", "5h"], ["3s", "4s", "5s"]),
    }
    rows = ["top", "middle", "bottom", "bottom", "bottom"]
    streets = []
    for turn in range(5):
        street = {"turn": turn}
        for seat, cards in zip(("hero", "opp"), deals[turn]):
            # Placements are [card, row], the order validate_hand normalises to.
            if turn == 0:
                placements = [[card, rows[i]] for i, card in enumerate(cards)]
                discard = None
            else:
                placements = [[cards[0], "bottom"], [cards[1], "middle"]]
                discard = cards[2]
            street[seat] = {
                "dealt": list(cards),
                "placements": placements,
                "discard": discard,
            }
        streets.append(street)
    return {
        "hero_position": "first",
        "fl": {"hero": False, "opp": False},
        "streets": streets,
        "label": "targeted-test",
    }


def test_turns_filter_keeps_only_the_named_street():
    hand = _hand()
    every = handlog.decision_contexts(hand)
    one = handlog.decision_contexts(hand, turns=[2])

    assert {c["turn"] for c in every} == {0, 1, 2, 3, 4}
    assert {c["turn"] for c in one} == {2}
    assert len(one) == 1


def test_targeted_context_carries_the_state_that_faced_that_street():
    """Filtering must drop gradings, not skip the replay that builds them."""
    hand = _hand()
    full = [c for c in handlog.decision_contexts(hand) if c["turn"] == 3][0]
    targeted = handlog.decision_contexts(hand, turns=[3])[0]

    assert targeted == full
    # T3 acting first: three streets are behind it, so the board is not empty
    # and the two earlier discards are dead.
    assert sum(len(v) for v in targeted["hero_board"].values()) == 9
    assert len(targeted["dead"]) == 2


def test_opponent_streets_still_need_include_opp():
    hand = _hand()
    hero_only = handlog.decision_contexts(hand, turns=[2])
    both = handlog.decision_contexts(hand, turns=[2], include_opp=True)

    assert {c["seat"] for c in hero_only} == {"hero"}
    assert {c["seat"] for c in both} == {"hero", "opp"}


def test_merge_replaces_only_the_targeted_slot_and_recomputes_totals():
    base = {
        "precision": "fast",
        "method": "model",
        "decisions": [
            {"seat": "hero", "turn": 0, "ev_loss": 1.0, "is_best": False,
             "precision": "fast", "method": "model"},
            {"seat": "hero", "turn": 1, "ev_loss": 2.0, "is_best": False,
             "precision": "fast", "method": "model"},
            {"seat": "hero", "turn": 2, "ev_loss": 0.0, "is_best": True,
             "precision": "fast", "method": "model"},
        ],
        "hero_total_ev_loss": 3.0,
        "hero_graded": 3,
        "hero_best": 1,
    }
    update = {
        "precision": "deep",
        "method": "teacher",
        "turns": [1],
        "decisions": [
            {"seat": "hero", "turn": 1, "ev_loss": 0.5, "is_best": False,
             "precision": "deep", "method": "teacher"},
        ],
        "hero_total_ev_loss": 0.5,
        "hero_graded": 1,
        "hero_best": 0,
    }

    merged = handlog.merge_analysis(base, update)

    assert [d["turn"] for d in merged["decisions"]] == [0, 1, 2]
    assert merged["decisions"][1]["ev_loss"] == 0.5
    assert merged["decisions"][1]["precision"] == "deep"
    # The untouched rows keep the rung that actually graded them.
    assert merged["decisions"][0]["precision"] == "fast"
    assert merged["hero_total_ev_loss"] == pytest.approx(1.5)
    assert merged["hero_graded"] == 3
    assert merged["hero_best"] == 1
    assert merged["mixed_precision"] is True
    assert merged["precisions"] == ["deep", "fast"]
    # Merged output describes the hand again, so the targeted marker is gone.
    assert "turns" not in merged


def test_merge_onto_nothing_keeps_the_partial_marker():
    update = {"precision": "deep", "turns": [2], "decisions": [
        {"seat": "hero", "turn": 2, "ev_loss": 0.5, "precision": "deep"}]}
    assert handlog.merge_analysis(None, update) is update
    assert handlog.merge_analysis({"decisions": []}, update) is update


def test_analysis_marks_a_targeted_run(monkeypatch):
    monkeypatch.setattr(
        handlog,
        "grade_decision",
        lambda ctx, precision, method: {
            "seat": ctx["seat"], "turn": ctx["turn"], "ev_loss": 0.25,
            "is_best": False,
        },
    )
    analysis = handlog.analyze_hand(_hand(), precision="deep", turns=[2])

    assert analysis["turns"] == [2]
    assert [d["turn"] for d in analysis["decisions"]] == [2]
    assert analysis["decisions"][0]["precision"] == "deep"


def test_targeted_persistence_does_not_shrink_the_stored_hand(tmp_path):
    store = TrainerStore(tmp_path / "trainer.db")
    account_id = store.get_or_create_account("tester")["id"]
    hand = _hand()
    hand_id = store.add_hand_log(
        account_id=account_id, label="targeted", note="", position="first", hand=hand
    )

    whole = {
        "precision": "fast",
        "decisions": [
            {"seat": "hero", "turn": turn, "ev_loss": 1.0, "is_best": False,
             "precision": "fast"}
            for turn in range(5)
        ],
        "hero_total_ev_loss": 5.0,
        "hero_graded": 5,
        "hero_best": 0,
    }
    assert store.set_hand_log_analysis(hand_id, account_id, whole, expect_hand=hand)

    targeted = {
        "precision": "deep",
        "turns": [2],
        "decisions": [
            {"seat": "hero", "turn": 2, "ev_loss": 0.0, "is_best": True,
             "precision": "deep"},
        ],
        "hero_total_ev_loss": 0.0,
        "hero_graded": 1,
        "hero_best": 1,
    }
    assert store.update_hand_log_analysis(
        hand_id,
        account_id,
        lambda existing: handlog.merge_analysis(existing, targeted),
        expect_hand=hand,
    )

    record = store.get_hand_log(hand_id, account_id)
    analysis = record["analysis"]
    assert len(analysis["decisions"]) == 5
    assert analysis["hero_total_ev_loss"] == pytest.approx(4.0)
    # The list column follows the merged total, not the one street's.
    assert record["total_ev_loss"] == pytest.approx(4.0)


def test_stale_hand_rejects_a_targeted_result(tmp_path):
    store = TrainerStore(tmp_path / "trainer.db")
    account_id = store.get_or_create_account("tester")["id"]
    hand = _hand()
    hand_id = store.add_hand_log(
        account_id=account_id, label="targeted", note="", position="first", hand=hand
    )

    corrected = _hand()
    corrected["streets"][2]["hero"]["dealt"] = ["6h", "8h", "9d"]
    corrected["streets"][2]["hero"]["discard"] = "9d"
    store.update_hand_log(
        hand_id, account_id, label="typo fixed", note="", position="first", hand=corrected
    )

    called = []
    assert not store.update_hand_log_analysis(
        hand_id,
        account_id,
        lambda existing: called.append(existing) or {"decisions": []},
        expect_hand=hand,
    )
    assert called == []
