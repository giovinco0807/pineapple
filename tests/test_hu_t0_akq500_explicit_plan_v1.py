"""Pins for the AKQ-500 T0 first-seat explicit plan.

The package-audit machinery is already pinned by `test_hu_m7_t1_1024_plan_v1`
and reused wholesale, so this file pins the things that are this plan's own:

  * **the measurement it is** -- a wide sieve, a uniform 1,024-particle stage
    two, and no race, because the race is the mechanism under test and a plan
    that raced would be measuring itself;
  * **the hands it names** -- read from a file, in the file's order, digest
    recorded, so offset n of the corpus is hand n of the file and a substituted
    file cannot pass unnoticed;
  * **the worker accepts it** -- checked against `load_plan` itself rather than
    against this module's idea of what the worker wants;
  * **no seed chooses a position**, and the one seed block it does spend is
    proved untouched by comparing seed SETS, not ranges.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pytest

from ofc_regular import hu_t0_akq500_explicit_plan_v1 as subject
from ofc_regular import hu_m7_t1_256_plan_v1 as t1_corpus
from ofc_regular import hu_m7_t1_topup_plan_v1 as t1_topup
from ofc_regular.hu_m31_label_gen_worker_v1 import (
    explicit_root_observation,
    load_plan as load_worker_plan,
)

from test_hu_m7_t1_1024_plan_v1 import _fixture_package  # noqa: E402

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _hands(count: int = subject.POSITION_COUNT) -> list[list[str]]:
    """`count` distinct five-card openings, each holding a broadway card.

    Deterministic and generated rather than copied: the real file lives on a
    drive this suite cannot depend on, and what the plan generator has to be
    right about is the shape, not the particular five hundred.
    """

    import random

    deck = [rank + suit for suit in _SUITS for rank in _RANKS]
    rng = random.Random(20260819)
    seen: set[tuple[str, ...]] = set()
    hands: list[list[str]] = []
    while len(hands) < count:
        hand = rng.sample(deck, 5)
        if not any(card[0] in "AKQ" for card in hand):
            continue
        key = tuple(sorted(hand))
        if key in seen:
            continue
        seen.add(key)
        hands.append(hand)
    assert len({tuple(hand) for hand in hands}) == count
    return hands


def _hands_file(tmp_path: Path, hands: list[list[str]] | None = None) -> Path:
    path = tmp_path / "hands.json"
    path.write_text(json.dumps({
        "seed": 20260819,
        "count": len(hands if hands is not None else _hands()),
        "filter": "contains at least one of A/K/Q",
        "dedup": "suit-canonical",
        "attempts": 136,
        "hands": hands if hands is not None else _hands(),
    }), encoding="utf-8")
    return path


def _built(tmp_path: Path, hands: list[list[str]] | None = None):
    package, identity = _fixture_package(tmp_path)
    return subject.build_plan(
        package,
        hands_file=_hands_file(tmp_path, hands),
        expected_identity=identity,
    )


# --------------------------------------------------------------------------
# The measurement this plan is
# --------------------------------------------------------------------------

def test_the_measurement_is_pinned() -> None:
    assert subject.STREET == "T0"
    assert subject.SEAT == "first"
    assert subject.POSITION_COUNT == 500
    assert subject.SHARD_COUNT == 63
    assert subject.SAMPLES == 1_024
    assert subject.SEEDS_PER_POSITION == 1
    assert subject.PREFILTER_SAMPLES == 64
    assert subject.PREFILTER_KEEP == 16
    assert subject.PREFILTER_MARGIN == 2.4
    assert subject.FL_EV_CARDS == 14
    assert subject.FL_EV_VALUE == 9.6
    assert subject.EVAL_SEED_BASE == 20_000_000
    assert subject.JOB_ID == "t0first-akq500-1024p"


def test_the_run_name_is_new_and_fleet_legal() -> None:
    """`hu_m31_label_gen_gcp_plan_v1` takes lowercase alphanumerics and dashes."""

    assert set(subject.JOB_ID) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")
    assert subject.JOB_ID not in {
        t1_corpus.FIRST_JOB_ID, t1_corpus.SECOND_JOB_ID, t1_topup.JOB_ID,
    }
    assert subject.PLAN_FILENAME not in {
        t1_corpus.FIRST_FILENAME, t1_corpus.SECOND_FILENAME,
        t1_topup.PLAN_FILENAME,
    }


def test_the_plan_does_not_race(tmp_path: Path) -> None:
    """The race is what is being measured, so the measurement must not use it."""

    plan, _ = _built(tmp_path)
    assert "race_schedule" not in plan
    assert "race_lcb_z" not in plan
    assert plan["provenance"]["root_schedule"]["race"] is None


def test_the_sieve_is_wider_than_production(tmp_path: Path) -> None:
    plan, _ = _built(tmp_path)
    assert plan["prefilter_samples"] == 64
    assert plan["prefilter_keep"] == 16
    assert plan["prefilter_margin"] == 2.4
    assert plan["samples"] == 1_024


def test_no_distilled_reply_is_pinned(tmp_path: Path) -> None:
    """A coarse continuation would fold a second approximation into the number."""

    plan, _ = _built(tmp_path)
    assert not [field for field in plan if field.startswith("fast_")]


def test_all_eight_learned_evaluators_are_pinned(tmp_path: Path) -> None:
    """The T0 first seat is the kind with the whole ladder ahead of it."""

    plan, _ = _built(tmp_path)
    for field in ("t4_model", "t3_second_model", "t3_first_model",
                  "t2_second_model", "t2_first_model", "t1_second_model",
                  "t1_first_model", "t0_second_model"):
        assert plan[field].startswith("weights/")
        assert len(plan[f"{field}_sha256"]) == 64


# --------------------------------------------------------------------------
# The hands it names
# --------------------------------------------------------------------------

def test_roots_are_the_hands_in_the_files_order(tmp_path: Path) -> None:
    hands = _hands()
    plan, _ = _built(tmp_path, hands)
    assert plan["root_source"] == "explicit"
    assert len(plan["roots"]) == 500
    assert [root["dealt_cards"] for root in plan["roots"]] == hands
    # Nothing but the cards: at T0 acting first that is the whole position.
    assert {key for root in plan["roots"] for key in root} == {"dealt_cards"}


def test_the_hand_file_digest_is_recorded(tmp_path: Path) -> None:
    """A substituted file must not pass as the one the plan was written for."""

    import hashlib

    hands_file = _hands_file(tmp_path)
    package, identity = _fixture_package(tmp_path)
    plan, _ = subject.build_plan(
        package, hands_file=hands_file, expected_identity=identity
    )
    source = plan["provenance"]["hand_source"]
    assert source["sha256"] == hashlib.sha256(hands_file.read_bytes()).hexdigest()
    assert source["count"] == 500
    assert source["seed"] == 20260819
    assert source["filter"] == "contains at least one of A/K/Q"


def test_a_hand_file_of_the_wrong_length_is_refused(tmp_path: Path) -> None:
    with pytest.raises(subject.PlanValidationError, match="499 hands"):
        _built(tmp_path, _hands(499))


def test_a_hand_of_the_wrong_size_is_refused(tmp_path: Path) -> None:
    hands = _hands()
    hands[7] = hands[7][:4]
    with pytest.raises(subject.PlanValidationError, match="hand 7"):
        _built(tmp_path, hands)


def test_a_repeated_hand_is_refused(tmp_path: Path) -> None:
    """Two identical roots are one measurement counted twice."""

    hands = _hands()
    hands[42] = list(hands[41])
    with pytest.raises(subject.PlanValidationError, match="duplicates"):
        _built(tmp_path, hands)


def test_a_missing_hand_file_is_refused(tmp_path: Path) -> None:
    package, identity = _fixture_package(tmp_path)
    with pytest.raises(subject.PlanValidationError, match="hand file is missing"):
        subject.build_plan(
            package,
            hands_file=tmp_path / "absent.json",
            expected_identity=identity,
        )


# --------------------------------------------------------------------------
# The worker accepts it, and rebuilds the positions the plan names
# --------------------------------------------------------------------------

def test_the_real_worker_loader_accepts_the_plan(tmp_path: Path) -> None:
    plan, _ = _built(tmp_path)
    path = tmp_path / "worker_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded = load_worker_plan(path)
    assert loaded["root_source"] == "explicit"
    assert len(loaded["roots"]) == 500


def test_every_root_rebuilds_as_a_t0_first_seat_opening(tmp_path: Path) -> None:
    hands = _hands()
    plan, _ = _built(tmp_path, hands)
    for index, hand in enumerate(hands):
        root = explicit_root_observation(plan, index)
        assert root.street == "T0"
        assert root.seat == "first"
        assert root.dealt_cards == tuple(hand)
        assert root.hero_board.card_count() == 0
        assert root.opponent_public_board.card_count() == 0
        assert root.hero_private_discards == ()


# --------------------------------------------------------------------------
# Seeds: none chooses a position, and the one that measures is untouched
# --------------------------------------------------------------------------

def test_no_seed_block_chooses_a_position(tmp_path: Path) -> None:
    plan, _ = _built(tmp_path)
    assert "hand_seed_base" not in plan
    assert "behavior_seed_offset" not in plan
    assert plan["eval_seed_base"] == subject.EVAL_SEED_BASE


def test_the_evaluation_progression_is_unspent() -> None:
    audit = subject.check_seed_allocation()
    assert audit["eval_seed_base"] == 20_000_000
    # Stride seven, so five hundred positions reach 20,003,493 -- the block is
    # seven times wider than its position count, which is the arithmetic a
    # base alone hides.
    assert audit["eval_seed_span"] == [20_000_000, 20_003_493]
    assert audit["hand_seed_block_spent"] is None


def test_seed_disjointness_is_decided_by_sets_not_ranges() -> None:
    """Two stride-seven blocks can overlap in range and share no seed.

    The T1 top-up is the live example: base 12,100,000 sits inside the 18k
    corpus's range and the two share nothing, because their residues mod 7
    differ. A range test would have refused a legal allocation, and -- worse --
    could accept an illegal one.
    """

    assert subject.eval_progressions_disjoint(0, 10, 7, 10) is False
    assert subject.eval_progressions_disjoint(0, 10, 3, 10) is True
    assert subject.eval_progressions_disjoint(
        12_000_000, 18_000, 12_100_000, 7_000
    ) is True


def test_the_block_is_clear_of_the_restricted_evaluation_experiments() -> None:
    """2026-08-19 spent 61M through 97M by hand; this block sits below them."""

    _low, high = subject.eval_seed_span(
        subject.EVAL_SEED_BASE, subject.POSITION_COUNT
    )
    assert high < 61_000_000


# --------------------------------------------------------------------------
# Write-once
# --------------------------------------------------------------------------

def test_write_once_refuses_to_overwrite(tmp_path: Path) -> None:
    package, identity = _fixture_package(tmp_path)
    hands_file = _hands_file(tmp_path)
    out = tmp_path / "plans"
    manifest = subject.write_plan_once(
        package, out, hands_file=hands_file, expected_identity=identity
    )
    assert manifest["schema"] == "hu_t0_akq500_explicit_plan_set_v1"
    assert (out / subject.PLAN_FILENAME).is_file()
    assert (out / subject.MANIFEST_FILENAME).is_file()
    assert manifest["measurement_contract"]["raced"] is False
    assert manifest["measurement_contract"]["hand_seed_block_spent"] is False

    with pytest.raises(subject.PlanValidationError, match="write-once"):
        subject.write_plan_once(
            package, out, hands_file=hands_file, expected_identity=identity
        )


def test_the_written_plan_is_what_the_worker_loads(tmp_path: Path) -> None:
    """The bytes on disk, not the dict in memory, are what the fleet runs."""

    package, identity = _fixture_package(tmp_path)
    out = tmp_path / "plans"
    subject.write_plan_once(
        package, out, hands_file=_hands_file(tmp_path), expected_identity=identity
    )
    loaded = load_worker_plan(out / subject.PLAN_FILENAME)
    assert loaded["job_id"] == subject.JOB_ID
    assert loaded["samples"] == 1_024
    assert sum(shard["count"] for shard in loaded["shards"]) == 500


def test_shards_tile_the_five_hundred_roots(tmp_path: Path) -> None:
    """Fifty-nine shards of eight and four of seven: one root a worker on the
    intended eight-worker instance, so a shard's wall time is one root's."""

    plan, _ = _built(tmp_path)
    shards = plan["shards"]
    assert len(shards) == 63
    counts = [shard["count"] for shard in shards]
    assert counts.count(8) == 59 and counts.count(7) == 4
    start = 0
    for shard in shards:
        assert shard["start"] == start
        start += shard["count"]
    assert start == 500
