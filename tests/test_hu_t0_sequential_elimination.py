"""The elimination rule's arithmetic, checked where it can be checked exactly.

Every one of these is a property the twenty-opening run depended on and that a
plausible rewrite could quietly break: paired comparison over shared batches,
particle weighting, and the fact that a candidate joining late is judged on its
own evidence rather than the leader's.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ofc_regular.hu_t0_sequential_elimination_v1 import (
    GAP_SD_AT_REFERENCE,
    REFERENCE_SAMPLES,
    Batch,
    load_batch,
    paired_gap,
    particles_needed,
    particles_spent,
    pooled_means,
    read_batches,
    standard_error,
    survivors,
    verdict,
)


def batch(samples: int, **scores: float) -> Batch:
    return Batch(samples=samples, scores=dict(scores))


def test_pooled_mean_weights_by_particles() -> None:
    """A 4,096-particle batch counts four times a 1,024-particle one."""

    means = pooled_means([batch(1024, a=0.0), batch(4096, a=1.0)])
    assert means["a"] == pytest.approx(0.8)   # (1*0 + 4*1) / 5


def test_pooled_mean_ignores_batches_that_skipped_an_action() -> None:
    means = pooled_means([batch(1024, a=1.0, b=0.0), batch(1024, a=3.0)])
    assert means["a"] == pytest.approx(2.0)
    assert means["b"] == pytest.approx(0.0)


def test_paired_gap_uses_only_shared_batches() -> None:
    """The newcomer is judged on the two batches it was in, not the leader's four."""

    batches = [
        batch(1024, leader=1.0),
        batch(1024, leader=1.0),
        batch(1024, leader=1.0, late=0.5),
        batch(1024, leader=1.0, late=0.5),
    ]
    gap, particles = paired_gap(batches, "leader", "late")
    assert gap == pytest.approx(0.5)
    assert particles == 2048


def test_paired_gap_reports_nothing_for_an_unmeasured_pair() -> None:
    gap, particles = paired_gap([batch(1024, a=1.0), batch(1024, b=2.0)], "a", "b")
    assert particles == 0
    assert gap != gap   # NaN


def test_standard_error_halves_when_particles_quadruple() -> None:
    assert standard_error(REFERENCE_SAMPLES) == pytest.approx(GAP_SD_AT_REFERENCE)
    assert standard_error(4 * REFERENCE_SAMPLES) == pytest.approx(
        GAP_SD_AT_REFERENCE / 2)


def test_a_candidate_far_behind_on_one_batch_still_survives() -> None:
    """One batch resolves 1.90 points; anything closer than that has to stay.

    This is the property the fixed top-ten violated. A candidate the sieve's
    single batch put eleventh could be there on draw luck, and eliminating it
    unmeasured is the one mistake no later stage can undo.
    """

    alive, leader, _ = survivors([batch(1024, best=1.0, near=-0.8, far=-3.0)])
    assert leader == "best"
    assert set(alive) == {"best", "near"}          # 1.8 < 1.90, 4.0 > 1.90


def test_the_same_candidate_falls_once_the_particles_arrive() -> None:
    """Eight identical batches shrink the bar to 0.67 and 1.8 no longer fits."""

    alive, _, _ = survivors([batch(1024, best=1.0, near=-0.8)] * 8)
    assert alive == ["best"]


def test_a_late_entrant_is_not_eliminated_on_the_leaders_precision() -> None:
    """Twelve batches for the leader, one shared with the newcomer.

    Pooling all twelve against the newcomer's one would compare a 0.9 gap to the
    twelve-batch bar of 0.55 and drop it. On the one batch they share, the bar is
    1.90 and it survives -- which is right, because nothing else is known.
    """

    batches = [batch(1024, leader=1.0)] * 11 + [batch(1024, leader=1.0, late=0.1)]
    alive, _, _ = survivors(batches)
    assert set(alive) == {"leader", "late"}


def test_particles_spent_reports_the_best_measured_action() -> None:
    batches = [batch(1024, a=0.0, b=0.0), batch(4096, a=0.0)]
    assert particles_spent(batches) == pytest.approx(5.0)


def test_particles_needed_is_quadratic_in_the_gap() -> None:
    assert particles_needed(1.0) == pytest.approx((4 * GAP_SD_AT_REFERENCE) ** 2)
    assert particles_needed(0.5) == pytest.approx(4 * particles_needed(1.0))
    assert particles_needed(0.0) == float("inf")


def test_verdict_converged_budget_and_open() -> None:
    assert verdict([batch(1024, a=5.0, b=0.0)])["state"] == "converged"

    close = [batch(1024, a=0.10, b=0.0)] * 3
    assert verdict(close, budget=120.0)["state"] == "open"
    assert verdict(close, budget=3.0)["state"] == "budget"


def test_verdict_names_the_leader_and_prices_the_rest() -> None:
    result = verdict([batch(1024, a=1.0, b=0.4, c=-9.0)])
    assert result["best"] == "a"
    assert result["gap"] == pytest.approx(0.6)
    assert "c" not in result["alive"]
    assert result["particles_needed"] == pytest.approx(particles_needed(0.6))


def test_load_batch_accepts_both_stored_shapes() -> None:
    worker = {"samples": 4096, "runs": [{"seed_trial": 0, "scores": {"a": 1.5}}]}
    duel = {"samples": 1024, "scores": {"a": 1.5}}
    assert load_batch(worker)["samples"] == 4096
    assert load_batch(duel)["scores"] == {"a": 1.5}


def test_read_batches_skips_what_it_cannot_parse(tmp_path: pathlib.Path) -> None:
    (tmp_path / "b00.json").write_text(
        json.dumps({"samples": 1024, "scores": {"a": 1.0}}), encoding="utf-8")
    (tmp_path / "b01.json").write_text("{ truncated", encoding="utf-8")
    (tmp_path / "b02.json").write_text(
        json.dumps({"samples": 2048, "scores": {"a": 2.0}}), encoding="utf-8")

    batches = read_batches([(tmp_path, "b*.json")])
    assert [b["samples"] for b in batches] == [1024, 2048]


def test_read_batches_tolerates_a_missing_directory(tmp_path: pathlib.Path) -> None:
    assert read_batches([(tmp_path / "absent", "*.json")]) == []
