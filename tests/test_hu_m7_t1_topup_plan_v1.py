"""Pins for the second-seat +7,000-root top-up plan.

The fixture package and the hash-audit behaviour are already pinned by
`test_hu_m7_t1_1024_plan_v1`, and the 256-particle settings by
`test_hu_m7_t1_256_plan_v1`.  This file pins the two things a top-up is:

  * **the same corpus** -- every model pin and every label setting equals the
    18k second-seat plan's, checked against a plan built from the same
    package rather than against retyped constants;
  * **different seeds** -- proved by materialising the actual seed sets from
    the worker's own formulas and intersecting them, because a shared seed
    makes a duplicate position that is invisible in the merged output.

The evaluation seeds are the case worth reading twice.  The worker strides
them by seven, so the 18k block runs from 12,000,000 to 12,125,993 and the
top-up's base of 12,100,000 sits INSIDE that range.  The sets are still
disjoint -- different residues mod 7 -- and the test below asserts the sets,
not the ranges, so it stays honest about which property actually holds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pytest

from ofc_regular import hu_m7_t1_topup_plan_v1 as subject
from ofc_regular import hu_m7_t1_256_plan_v1 as extended
from ofc_regular import hu_m7_t1_1024_plan_v1 as superseded
from ofc_regular import hu_m7_t2_2048_plan_v1 as t2_corpus
from ofc_regular.hu_m31_label_gen_worker_v1 import load_plan as load_worker_plan

from test_hu_m7_t1_1024_plan_v1 import _fixture_package, _sha  # noqa: E402


def _hand_seeds(base: int, positions: int) -> set[int]:
    """`hu_m31_label_gen_worker_v1.make_root`: hand_seed_base + offset."""

    return {base + offset for offset in range(positions)}


def _behavior_seeds(base: int, offset_base: int, positions: int) -> set[int]:
    """Same function: hand_seed_base + behavior_seed_offset + offset."""

    return {base + offset_base + offset for offset in range(positions)}


def _eval_seeds(base: int, positions: int) -> set[int]:
    """`hu_m31_label_gen_worker_v1`: eval_seed_base + offset * 7 (trial 0)."""

    return {base + offset * 7 for offset in range(positions)}


def test_the_topup_decision_is_pinned():
    assert subject.SEAT == "second"
    assert subject.POSITION_COUNT == 7_000
    assert subject.SHARD_COUNT == 46
    assert subject.SAMPLES == 256
    assert subject.SEEDS_PER_POSITION == 1
    assert subject.FL_EV_CARDS == 14
    assert subject.FL_EV_VALUE == 9.6
    assert subject.HAND_SEED_BASE == 952_100_000
    assert subject.EVAL_SEED_BASE == 12_100_000
    # 18,000 + 7,000 is the count the incumbent was trained on, which is the
    # whole reason this plan exists.
    assert subject.MERGED_POSITION_COUNT == 25_000
    assert subject.JOB_ID == "m7v6-t1second-topup7k-256p"


def test_the_run_name_is_new_and_fleet_legal():
    """`hu_m31_label_gen_gcp_plan_v1` takes lowercase alphanumerics and dashes."""

    assert set(subject.JOB_ID) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")
    taken = {
        extended.FIRST_JOB_ID,
        extended.SECOND_JOB_ID,
        superseded.FIRST_JOB_ID,
        superseded.SECOND_JOB_ID,
        t2_corpus.FIRST_JOB_ID,
        t2_corpus.SECOND_JOB_ID,
    }
    assert subject.JOB_ID not in taken
    assert subject.PLAN_FILENAME not in {
        extended.FIRST_FILENAME, extended.SECOND_FILENAME,
        superseded.FIRST_FILENAME, superseded.SECOND_FILENAME,
    }
    assert subject.MANIFEST_FILENAME not in {
        extended.MANIFEST_FILENAME, superseded.MANIFEST_FILENAME,
    }


def test_shards_split_seven_thousand_exactly():
    shards = subject.make_shards(position_count=7_000, shard_count=46)
    assert len(shards) == 46
    counts = [row["count"] for row in shards]
    assert counts.count(152) == 38
    assert counts.count(153) == 8
    assert sum(counts) == 7_000
    assert [row["shard_id"] for row in shards] == [f"{i:02d}" for i in range(46)]
    start = 0
    for row in shards:
        assert row["start"] == start
        start += row["count"]
    assert start == 7_000


def test_builds_one_second_seat_plan(tmp_path: Path):
    package, identity = _fixture_package(tmp_path)
    plan, _ = subject.build_plan(package, expected_identity=identity)

    assert plan["street"] == "T1"
    assert plan["seat"] == "second"
    assert plan["samples"] == 256
    assert plan["seeds_per_position"] == 1
    assert plan["hand_seed_base"] == 952_100_000
    assert plan["eval_seed_base"] == 12_100_000
    assert len(plan["shards"]) == 46
    assert sum(row["count"] for row in plan["shards"]) == 7_000
    # Coarse replies, and the full pair kept beside them, exactly as the 18k
    # second-seat plan carries them.
    assert plan["fast_t2_first_model"] == "weights/fast_t2_first_v1.bin"
    assert plan["fast_t2_second_model"] == "weights/fast_t2_second_v1.bin"
    assert plan["t2_first_model"] == "weights/t2first_model_v2.bin"
    assert plan["t2_second_model"] == "weights/t2_model_v2.bin"
    assert plan["t3_first_model"] == "weights/t3first_model_v2.bin"
    assert plan["t3_second_model"] == "weights/t3_model_v3.bin"
    assert plan["t4_model"] == "weights/t4_model_v6.bin"
    # The second seat has no T1 reply ahead of it and no narrowing anywhere.
    assert "t1_second_model" not in plan
    for field in ("prefilter_samples", "prefilter_keep", "prefilter_margin",
                  "audit_full_every", "race_schedule", "race_lcb_z", "probe"):
        assert field not in plan


def test_every_pin_matches_the_corpus_it_joins(tmp_path: Path):
    """The merge property: same package, same teacher, same settings."""

    package, identity = _fixture_package(tmp_path)
    plan, audit = subject.build_plan(package, expected_identity=identity)
    corpus_plan = extended._base_plan(audit, seat="second")

    assert set(plan) == set(corpus_plan)
    for field in sorted(set(plan) - set(subject._MUST_DIFFER)):
        assert plan[field] == corpus_plan[field], field
    # And the pins specifically, named so a future reader sees the list.
    for field in (
        "engine_library_sha256", "feature_encoder_library_sha256",
        "t4_model_sha256", "t3_first_model_sha256", "t3_second_model_sha256",
        "t2_first_model_sha256", "t2_second_model_sha256",
        "fast_t2_first_model_sha256", "fast_t2_second_model_sha256",
    ):
        assert plan[field] == corpus_plan[field]
    assert plan["job_id"] != corpus_plan["job_id"]
    assert plan["shards"] != corpus_plan["shards"]


def test_a_drifted_pin_is_refused(tmp_path: Path):
    package, identity = _fixture_package(tmp_path)
    plan, audit = subject.build_plan(package, expected_identity=identity)
    corpus_plan = extended._base_plan(audit, seat="second")
    plan["t2_second_model"] = "weights/t2_model_v1.bin"
    with pytest.raises(subject.PlanValidationError, match="t2_second_model differs"):
        subject.validate_merges_with_extended_corpus(plan, corpus_plan)


def test_a_reused_seed_block_is_refused(tmp_path: Path):
    """The defect this plan exists to prevent, at the size that hides it."""

    package, identity = _fixture_package(tmp_path)
    plan, audit = subject.build_plan(package, expected_identity=identity)
    corpus_plan = extended._base_plan(audit, seat="second")
    plan["hand_seed_base"] = corpus_plan["hand_seed_base"]
    with pytest.raises(subject.PlanValidationError, match="hand_seed_base is identical"):
        subject.validate_merges_with_extended_corpus(plan, corpus_plan)

    # A one-position overlap, which no output would show.
    plan["hand_seed_base"] = (
        corpus_plan["hand_seed_base"] + extended.POSITION_COUNT - 1
    )
    with pytest.raises(subject.PlanValidationError, match="overlaps"):
        subject.validate_merges_with_extended_corpus(plan, corpus_plan)


def test_hand_and_behavior_seed_sets_are_disjoint_from_every_known_block():
    """Materialise the seeds the worker would draw and intersect them."""

    mine_hand = _hand_seeds(subject.HAND_SEED_BASE, subject.POSITION_COUNT)
    mine_behavior = _behavior_seeds(
        subject.HAND_SEED_BASE, subject.BEHAVIOR_SEED_OFFSET, subject.POSITION_COUNT
    )
    assert (min(mine_hand), max(mine_hand)) == (952_100_000, 952_106_999)
    assert (min(mine_behavior), max(mine_behavior)) == (952_600_000, 952_606_999)
    assert mine_hand.isdisjoint(mine_behavior)

    # The three live plan generators, read from source rather than retyped.
    for module in (extended, superseded, t2_corpus):
        other_hand = _hand_seeds(module.HAND_SEED_BASE, module.POSITION_COUNT)
        other_behavior = _behavior_seeds(
            module.HAND_SEED_BASE, module.BEHAVIOR_SEED_OFFSET, module.POSITION_COUNT
        )
        for mine in (mine_hand, mine_behavior):
            assert mine.isdisjoint(other_hand), module.__name__
            assert mine.isdisjoint(other_behavior), module.__name__

    # And the blocks the cascade ledger records, which have no module.
    for label, lo, hi in subject.reserved_seed_intervals():
        for mine in (mine_hand, mine_behavior):
            assert max(mine) < lo or min(mine) >= hi, label


def test_the_four_intervals_of_the_merged_corpus_are_pairwise_disjoint():
    """18k hand/behavior and top-up hand/behavior, the actual values."""

    blocks = {
        "18k hand": (952_000_000, 952_018_000),
        "18k behavior": (952_500_000, 952_518_000),
        "topup hand": (952_100_000, 952_107_000),
        "topup behavior": (952_600_000, 952_607_000),
    }
    assert blocks["18k hand"] == (
        extended.HAND_SEED_BASE, extended.HAND_SEED_BASE + extended.POSITION_COUNT
    )
    assert blocks["topup hand"] == (
        subject.HAND_SEED_BASE, subject.HAND_SEED_BASE + subject.POSITION_COUNT
    )
    seeds = {name: set(range(lo, hi)) for name, (lo, hi) in blocks.items()}
    names = sorted(seeds)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            assert seeds[left].isdisjoint(seeds[right]), (left, right)


def test_eval_seed_sets_are_disjoint_although_their_ranges_overlap():
    """The stride-7 trap: 100,000 of headroom does not clear an 18,000 block."""

    mine = _eval_seeds(subject.EVAL_SEED_BASE, subject.POSITION_COUNT)
    theirs = _eval_seeds(extended.EVAL_SEED_BASE, extended.POSITION_COUNT)
    assert (min(theirs), max(theirs)) == (12_000_000, 12_125_993)
    assert (min(mine), max(mine)) == (12_100_000, 12_148_993)
    # The ranges DO overlap ...
    assert min(mine) < max(theirs)
    # ... and the seed sets still do not, by residue.
    assert mine.isdisjoint(theirs)
    assert extended.EVAL_SEED_BASE % 7 == 5
    assert subject.EVAL_SEED_BASE % 7 == 3

    for module in (superseded, t2_corpus):
        assert mine.isdisjoint(_eval_seeds(module.EVAL_SEED_BASE, module.POSITION_COUNT))
    # The doc-recorded bases have no written extent; the new one sits above all.
    assert subject.EVAL_SEED_BASE > max(subject._DOC_RECORDED_EVAL_BASES)


def test_the_progression_check_is_not_merely_a_range_check():
    """It accepts a clean interleave and rejects a real collision."""

    assert subject.eval_progressions_disjoint(12_100_000, 7_000, 12_000_000, 18_000)
    # Same residue and overlapping span: an actual shared evaluation seed.
    assert not subject.eval_progressions_disjoint(12_000_007, 7_000, 12_000_000, 18_000)
    assert subject.eval_seed_span(12_100_000, 7_000) == (12_100_000, 12_148_994)


def test_a_partial_coarse_pair_is_refused(tmp_path: Path):
    package, identity = _fixture_package(tmp_path)
    plan, _ = subject.build_plan(package, expected_identity=identity)
    del plan["fast_t2_second_model"]
    with pytest.raises(subject.PlanValidationError, match="fast_t2_second_model"):
        subject.validate_plan(plan)


def test_dropping_the_full_t2_pins_is_refused(tmp_path: Path):
    """The engine loads both; a plan with only the coarse pair is unreadable."""

    package, identity = _fixture_package(tmp_path)
    plan, _ = subject.build_plan(package, expected_identity=identity)
    del plan["t2_first_model"]
    with pytest.raises(subject.PlanValidationError, match="t2_first_model"):
        subject.validate_plan(plan)


def test_a_narrowing_field_is_refused(tmp_path: Path):
    """All 27 actions, as in the corpus this joins."""

    package, identity = _fixture_package(tmp_path)
    plan, _ = subject.build_plan(package, expected_identity=identity)
    plan["prefilter_keep"] = 10
    with pytest.raises(subject.PlanValidationError, match="prefilter_keep"):
        subject.validate_plan(plan)


def test_a_second_seat_t1_reply_pin_is_refused(tmp_path: Path):
    package, identity = _fixture_package(tmp_path)
    plan, _ = subject.build_plan(package, expected_identity=identity)
    plan["t1_second_model"] = "weights/t1_model_v1.bin"
    with pytest.raises(subject.PlanValidationError, match="t1_second_model"):
        subject.validate_plan(plan)


def test_write_is_deterministic_and_write_once(tmp_path: Path):
    package, identity = _fixture_package(tmp_path)
    output = tmp_path / "plans"
    manifest = subject.write_plan_once(package, output, expected_identity=identity)
    raw = (output / subject.PLAN_FILENAME).read_bytes()
    stored = json.loads((output / subject.MANIFEST_FILENAME).read_text())

    assert stored == manifest
    assert stored["plans"]["second"]["sha256"] == _sha(raw)
    contract = stored["topup_contract"]
    assert contract["seat"] == "second"
    assert contract["extends_job_id"] == extended.SECOND_JOB_ID
    assert contract["merged_positions"] == 25_000
    assert contract["coarse_t2_replies"] is True
    assert contract["first_seat_topup"] is False
    assert contract["current_profile_changed"] is False

    with pytest.raises(subject.PlanValidationError, match="already exists"):
        subject.write_plan_once(package, output, expected_identity=identity)
    assert (output / subject.PLAN_FILENAME).read_bytes() == raw


def test_the_real_worker_accepts_the_plan(tmp_path: Path):
    package, identity = _fixture_package(tmp_path)
    output = tmp_path / "plans"
    subject.write_plan_once(package, output, expected_identity=identity)

    plan = load_worker_plan(output / subject.PLAN_FILENAME)
    assert (plan["street"], plan["seat"], plan["samples"]) == ("T1", "second", 256)
    assert plan["job_id"] == subject.JOB_ID
    assert sum(row["count"] for row in plan["shards"]) == 7_000
