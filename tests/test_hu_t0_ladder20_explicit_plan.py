"""The ladder plan's shape, checked without a package on disk.

``build_plan`` needs the 314 MB m7v7 archive to pin hashes against, which no
test should require, so these exercise the two pieces that carry the
measurement's meaning and can be built from the hand list alone: the root
construction and the validator.  What they defend against is a later edit that
still audits clean but measures something else -- twelve repeats quietly become
eleven, the interleave becomes a block, a race creeps back into the schedule.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ofc_regular.hu_m7_t1_256_plan_v1 import (
    PLAN_SCHEMA,
    PlanValidationError,
    make_shards,
)
from ofc_regular.hu_t0_ladder20_explicit_plan_v1 import (
    EVAL_SEED_BASE,
    HAND_COUNT,
    POSITION_COUNT,
    PREFILTER_KEEP,
    PREFILTER_MARGIN,
    PREFILTER_SAMPLES,
    REPEATS,
    SAMPLES,
    SEEDS_PER_POSITION,
    SHARD_COUNT,
    _MODEL_PINS,
    build_roots,
    check_seed_allocation,
    load_hands,
    validate_plan,
)

RANKS = "23456789TJQKA"
SUITS = "shdc"
DECK = [f"{rank}{suit}" for suit in SUITS for rank in RANKS]


def _hands(count: int = HAND_COUNT) -> list[list[str]]:
    """``count`` distinct five-card hands.

    Hand *i* is the five cards at deck offsets 5i..5i+4, wrapped.  Five is
    coprime with 52, so 5i ≡ 5j only when i ≡ j and all 52 available hands are
    distinct.  These stand in for the akq500 openings: nothing here reads their
    ranks.
    """

    assert count <= len(DECK)
    return [
        [DECK[(index * 5 + offset) % len(DECK)] for offset in range(5)]
        for index in range(count)
    ]


def _plan(hands: list[list[str]] | None = None, **overrides: object) -> dict:
    """A plan of the right shape, with the package pins faked."""

    roots = build_roots(hands if hands is not None else _hands())
    plan = {
        "schema": PLAN_SCHEMA,
        "job_id": "t0first-ladder20-1024p-x12",
        "street": "T0",
        "seat": "first",
        "samples": SAMPLES,
        "seeds_per_position": SEEDS_PER_POSITION,
        "eval_seed_base": EVAL_SEED_BASE,
        "root_source": "explicit",
        "roots": roots,
        "prefilter_samples": PREFILTER_SAMPLES,
        "prefilter_keep": PREFILTER_KEEP,
        "prefilter_margin": PREFILTER_MARGIN,
        "fl_ev_cards": 14,
        "fl_ev_value": 9.6,
        "shards": make_shards(
            position_count=POSITION_COUNT, shard_count=SHARD_COUNT
        ),
    }
    for field, relative in _MODEL_PINS:
        plan[field] = relative
        plan[f"{field}_sha256"] = "0" * 64
    plan.update(overrides)
    return plan


def test_a_well_formed_plan_validates() -> None:
    validate_plan(_plan())


def test_roots_are_twelve_interleaved_passes() -> None:
    roots = build_roots(_hands())
    assert len(roots) == POSITION_COUNT == HAND_COUNT * REPEATS
    # Interleaved, not blocked: the first twenty roots are twenty distinct
    # hands, which is what puts eight different hands in every shard.
    first_pass = [tuple(root["dealt_cards"]) for root in roots[:HAND_COUNT]]
    assert len(set(first_pass)) == HAND_COUNT
    for index in range(POSITION_COUNT):
        assert roots[index]["dealt_cards"] == roots[index % HAND_COUNT]["dealt_cards"]


def test_a_blocked_layout_is_refused() -> None:
    """Twelve of hand 0, then twelve of hand 1 -- the shard-loss hazard."""

    hands = _hands()
    blocked = [
        {"dealt_cards": list(hands[index // REPEATS])}
        for index in range(POSITION_COUNT)
    ]
    with pytest.raises(PlanValidationError, match="interleave"):
        validate_plan(_plan(**{"roots": blocked}))


def test_a_missing_repeat_is_refused() -> None:
    roots = build_roots(_hands())[: POSITION_COUNT - 1]
    with pytest.raises(PlanValidationError, match="240"):
        validate_plan(_plan(**{"roots": roots}))


def test_repeats_as_trials_are_refused() -> None:
    """The layout this plan exists to avoid: twenty roots, twelve trials each.

    The worker strides its per-VM workers over roots, so this shape runs one
    worker for twelve batches in series and idles seven.
    """

    with pytest.raises(PlanValidationError, match="seeds per position"):
        validate_plan(_plan(seeds_per_position=REPEATS))


def test_a_raced_plan_is_refused() -> None:
    with pytest.raises(PlanValidationError, match="race"):
        validate_plan(_plan(race_schedule=[32, 64, 128, 256]))


@pytest.mark.parametrize(
    "field, value",
    (
        ("samples", 2048),
        ("prefilter_samples", 32),
        ("prefilter_keep", 8),
        ("prefilter_margin", 0.0),
        ("fl_ev_value", 9.0),
        ("eval_seed_base", 20_000_000),
    ),
)
def test_schedule_drift_is_refused(field: str, value: object) -> None:
    """The schedule must stay the one the local runs used, or it checks nothing."""

    with pytest.raises(PlanValidationError):
        validate_plan(_plan(**{field: value}))


def test_an_unpinned_model_is_refused() -> None:
    field = _MODEL_PINS[0][0]
    with pytest.raises(PlanValidationError, match="unpinned"):
        validate_plan(_plan(**{f"{field}_sha256": ""}))


def test_shards_must_tile_the_roots() -> None:
    shards = make_shards(position_count=POSITION_COUNT, shard_count=SHARD_COUNT)
    shards[3]["start"] += 1  # a one-root gap, sums unchanged downstream
    with pytest.raises(PlanValidationError, match="starts at"):
        validate_plan(_plan(shards=shards))


def test_eval_seeds_are_disjoint_from_everything_spent() -> None:
    report = check_seed_allocation()
    low, high = report["eval_seed_span"]
    assert low == EVAL_SEED_BASE
    assert high == EVAL_SEED_BASE + (POSITION_COUNT - 1) * 7
    # Repeats are roots, so the three-million-wide trial term is never spent.
    assert high - low == 1673
    assert "t0first-akq500-1024p plan" in report["compared_against"]


def test_load_hands_takes_the_first_twenty(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "hands.json"
    payload = {"hands": _hands(40) , "seed": 20260819}
    path.write_text(json.dumps(payload), encoding="utf-8")
    hands, source = load_hands(path)
    assert len(hands) == HAND_COUNT
    assert hands == payload["hands"][:HAND_COUNT]
    assert source["count"] == 40 and source["taken"] == HAND_COUNT
    assert source["seed"] == 20260819
    assert len(source["sha256"]) == 64


def test_load_hands_refuses_a_short_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "hands.json"
    path.write_text(json.dumps({"hands": _hands(19)}), encoding="utf-8")
    with pytest.raises(PlanValidationError, match="at least"):
        load_hands(path)


def test_load_hands_refuses_a_repeated_hand(tmp_path: pathlib.Path) -> None:
    hands = _hands()
    hands[7] = list(hands[3])
    path = tmp_path / "hands.json"
    path.write_text(json.dumps({"hands": hands}), encoding="utf-8")
    with pytest.raises(PlanValidationError, match="duplicates"):
        load_hands(path)
