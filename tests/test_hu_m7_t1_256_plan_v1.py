"""Pins for the 256-particle T1 plan pair.

The reused fixture package and hash-audit behaviour are already pinned by
`test_hu_m7_t1_1024_plan_v1`; this file pins what the 2026-08-16 sizing
decision changed, and the two properties that decision must not break:

  * the coarse T2 pair is present in FULL alongside the full-precision pins --
    the worker refuses a partial set, and it refuses a T1 plan with no full
    pins at all;
  * the fresh hand block does not touch the superseded 1024p plans' block,
    because a partial overlap between two label plans makes duplicate
    positions that are invisible in the output.
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

from ofc_regular import hu_m7_t1_256_plan_v1 as subject
from ofc_regular import hu_m7_t1_1024_plan_v1 as superseded
from ofc_regular.hu_m31_label_gen_worker_v1 import load_plan as load_worker_plan

from test_hu_m7_t1_1024_plan_v1 import _fixture_package, _sha  # noqa: E402


def test_the_sizing_decision_is_pinned():
    assert subject.SAMPLES == 256
    assert subject.POSITION_COUNT == 18_000
    assert subject.SHARD_COUNT == 116
    assert subject.FL_EV_VALUE == 9.6


def test_builds_both_seats_with_the_coarse_t2_pair(tmp_path: Path):
    package, identity = _fixture_package(tmp_path)
    plans, _ = subject.build_plan_pair(package, expected_identity=identity)

    for seat, plan in plans.items():
        assert plan["street"] == "T1"
        assert plan["samples"] == 256
        assert sum(row["count"] for row in plan["shards"]) == 18_000
        assert len(plan["shards"]) == 116
        # Coarse replies, and the full pair kept beside them.
        assert plan["fast_t2_first_model"] == "weights/fast_t2_first_v1.bin"
        assert plan["fast_t2_second_model"] == "weights/fast_t2_second_v1.bin"
        assert plan["t2_first_model"] == "weights/t2first_model_v2.bin"
        assert plan["t2_second_model"] == "weights/t2_model_v2.bin"

    assert "t1_second_model" in plans["first"]
    assert "t1_second_model" not in plans["second"]


def test_the_hand_block_does_not_touch_the_superseded_plans():
    old_lo = superseded.HAND_SEED_BASE
    old_hi = old_lo + superseded.POSITION_COUNT
    new_lo = subject.HAND_SEED_BASE
    new_hi = new_lo + subject.POSITION_COUNT
    assert new_hi <= old_lo or new_lo >= old_hi
    assert subject.EVAL_SEED_BASE != superseded.EVAL_SEED_BASE


def test_a_partial_coarse_pair_is_refused(tmp_path: Path):
    package, identity = _fixture_package(tmp_path)
    plans, _ = subject.build_plan_pair(package, expected_identity=identity)
    del plans["first"]["fast_t2_second_model"]
    with pytest.raises(subject.PlanValidationError, match="fast_t2_second_model"):
        subject.validate_plan_pair(plans)


def test_dropping_the_full_t2_pins_is_refused(tmp_path: Path):
    """The engine loads both; a plan with only the coarse pair is unreadable."""
    package, identity = _fixture_package(tmp_path)
    plans, _ = subject.build_plan_pair(package, expected_identity=identity)
    del plans["second"]["t2_first_model"]
    with pytest.raises(subject.PlanValidationError, match="t2_first_model"):
        subject.validate_plan_pair(plans)


def test_write_is_write_once(tmp_path: Path):
    package, identity = _fixture_package(tmp_path)
    output = tmp_path / "plans"
    manifest = subject.write_plan_pair_once(package, output,
                                            expected_identity=identity)
    first_raw = (output / subject.FIRST_FILENAME).read_bytes()
    assert manifest["plans"]["first"]["sha256"] == _sha(first_raw)
    assert manifest["paired_contract"]["coarse_t2_replies"] is True
    with pytest.raises(subject.PlanValidationError, match="already exists"):
        subject.write_plan_pair_once(package, output, expected_identity=identity)
    assert (output / subject.FIRST_FILENAME).read_bytes() == first_raw


def test_the_real_worker_accepts_both_plans(tmp_path: Path):
    package, identity = _fixture_package(tmp_path)
    output = tmp_path / "plans"
    subject.write_plan_pair_once(package, output, expected_identity=identity)

    first = load_worker_plan(output / subject.FIRST_FILENAME)
    second = load_worker_plan(output / subject.SECOND_FILENAME)
    assert (first["street"], first["seat"], first["samples"]) == ("T1", "first", 256)
    assert (second["street"], second["seat"], second["samples"]) == ("T1", "second", 256)
