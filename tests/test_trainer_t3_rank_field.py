"""T3 is ranked by `ev`, and that is why its verdict stops moving with the seed.

The engine orders T3 by its candidate batch.  That statistic does not converge:
measured on one root, raising `evaluation_samples` 1,024 -> 8,192 -> 32,768
moved the top-two `selection_score` gap +2.875 -> +7.325 -> +6.950 and raising
`candidate_samples` 8 -> 512 moved it +7.325 -> +4.438, while `ev` settled
within 0.2 under both.  A user meets that instability as a T3 grade that
changes when nothing about the position did.

Two properties are pinned here:

  * ranking by `rank_score` reproduces the engine's own order wherever the two
    already agreed (T0/T1/T2), so the change is confined to T3;
  * T3's ranking is stable across seeds, which the old field was not.

The seed-stability test needs the engine DLL and its pinned weights and skips
without them; the ordering test is pure and always runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pytest

from trainer import engine_eval


HERO = {"top": ["Kc", "Ah"], "middle": ["5c", "9c", "6c"], "bottom": ["7d", "8d", "Jd", "Kd"]}


def _row(index, action_key, placements, *, ev, selection, candidate=None):
    row = {
        "sorted_index": index,
        "action_key": action_key,
        "placements": placements,
        "discards": [],
        "score": ev,
        "selection_score": selection,
    }
    if candidate is not None:
        row["candidate_score"] = candidate
    return row


def test_t3_ranks_by_ev_not_the_candidate_batch():
    """The two fields disagree; the ranking must follow ev."""
    rows = [
        # The engine's order (sorted_index) follows selection_score here.
        _row(0, "a", [["Th", "top"]], ev=-5.0, selection=-1.0),
        _row(1, "b", [["7s", "top"]], ev=-2.0, selection=-3.0),
    ]
    ranked = engine_eval._normalize_rows(rows, HERO, turn=3)

    assert [c["metrics"]["ranked_by"] for c in ranked] == ["ev", "ev"]
    # ev prefers 'b' (-2.0 beats -5.0) even though the engine listed 'a' first.
    assert ranked[0]["metrics"]["ev"] == -2.0
    assert ranked[0]["metrics"]["rank_score"] == -2.0
    assert ranked[1]["metrics"]["ev"] == -5.0


@pytest.mark.parametrize("turn", [1, 2])
def test_the_resort_is_a_no_op_where_the_engine_already_agreed(turn):
    """T1/T2 rank on ev already, so their order must not move."""
    rows = [
        _row(0, "a", [["Th", "top"]], ev=-1.0, selection=-9.0),
        _row(1, "b", [["7s", "top"]], ev=-2.0, selection=-8.0),
        _row(2, "c", [["4c", "middle"]], ev=-3.0, selection=-7.0),
    ]
    ranked = engine_eval._normalize_rows(rows, HERO, turn=turn)
    assert [c["metrics"]["ev"] for c in ranked] == [-1.0, -2.0, -3.0]


def test_t0_still_ranks_on_the_candidate_batch():
    """T0's staged search prices pruned rows on fewer particles, so its `ev`
    is not comparable across rows and the candidate batch stays the ranker."""
    rows = [
        _row(0, "a", [["Th", "top"]], ev=-5.0, selection=-1.0, candidate=-1.0),
        _row(1, "b", [["7s", "top"]], ev=-2.0, selection=-3.0, candidate=-3.0),
    ]
    ranked = engine_eval._normalize_rows(rows, HERO, turn=0)
    assert [c["metrics"]["ranked_by"] for c in ranked] == ["candidate_score"] * 2
    assert ranked[0]["metrics"]["candidate_score"] == -1.0


def test_ties_come_back_in_a_fixed_order():
    """Equal rows must not depend on whatever order the engine emitted."""
    rows = [
        _row(0, "a", [["Th", "top"]], ev=-4.0, selection=-1.0),
        _row(1, "b", [["7s", "top"]], ev=-4.0, selection=-2.0),
    ]
    first = engine_eval._normalize_rows(rows, HERO, turn=3)
    second = engine_eval._normalize_rows(list(reversed(rows)), HERO, turn=3)
    assert [c["key"] for c in first] == [c["key"] for c in second]


@pytest.mark.skipif(
    not engine_eval.available(), reason="m3 engine DLL or pinned weights unavailable"
)
def test_t3_ranking_is_stable_across_seeds():
    """The property the change exists for, against the real engine.

    The trainer derives its seed from the observation fingerprint, so the same
    position always draws the same particles.  This calls the engine directly
    on one position at several sample counts: if the ranking is read off a
    converging field, the winner does not move.
    """
    position = dict(
        hero_board=HERO,
        opp_board={"top": ["Ac", "Td"], "middle": ["Ts", "Js", "6s", "4s"],
                   "bottom": ["Kh", "Qh", "Jh", "2h", "3h"]},
        dealt=["Th", "4c", "7s"],
        dead=["Ks", "3s"],
        turn=3,
        position="second",
    )
    winners = set()
    for precision in ("standard", "high", "deep"):
        result = engine_eval.evaluate_with_engine(**position, precision=precision)
        best = result["candidates"][0]
        winners.add(best["key"])
        assert best["metrics"]["ranked_by"] == "ev"
        # The list is ordered by the number it is priced with.
        scores = [c["metrics"]["rank_score"] for c in result["candidates"]]
        assert scores == sorted(scores, reverse=True)
    assert len(winners) == 1, f"T3 winner moved with the sample count: {winners}"
