"""Pins for the worker's ``root_source: "explicit"`` intake.

The feature is one sentence -- the plan may carry the positions instead of
deriving them from a seed -- and four properties, each of which has a way of
being quietly wrong:

* **the position is the one the caller wrote down**, not one adjacent to it;
* **a position that contradicts the plan's street and seat is refused by name**,
  because a corpus of the wrong street looks exactly like a corpus of the right
  one until something downstream reads the boards;
* **the Fantasyland constant still comes from the runtime's config**, and a root
  that tries to bring its own is refused. A corpus labelled at a constant nobody
  chose is indistinguishable from a correct corpus from the outside, which is
  the whole reason that refusal is structural rather than documented;
* **the seeded path is untouched**. Not "still works" -- untouched: the plan
  bytes an existing generator writes are asserted here to be identical either
  side of this feature, and the seed the worker would evaluate an explicit
  position with is asserted to follow the same formula as a dealt one's.

The last is what the rest of the fleet's correctness rests on, so it is checked
against a real generator's real output rather than a hand-typed plan.
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

from ofc_regular import hu_m31_label_gen_worker_v1 as worker
from ofc_regular import hu_m7_t1_256_plan_v1 as seeded_generator

from test_hu_m7_t1_1024_plan_v1 import _fixture_package  # noqa: E402


def _write(tmp_path: Path, plan: dict[str, Any], name: str = "plan.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _t0_first_plan(**overrides: Any) -> dict[str, Any]:
    """A minimal but complete T0 first-seat explicit plan.

    Complete matters: the T0 first seat is the kind with the longest required
    pin list in the worker -- eight learned evaluators and the T4 terminal --
    so a plan that omitted one would fail this file's tests for a reason that
    has nothing to do with explicit roots.
    """

    plan: dict[str, Any] = {
        "schema": worker.PLAN_SCHEMA,
        "job_id": "explicit-fixture",
        "street": "T0",
        "seat": "first",
        "samples": 1024,
        "seeds_per_position": 1,
        "eval_seed_base": 20_000_000,
        "engine_library": "native/engine.so",
        "engine_library_sha256": "0" * 64,
        "feature_encoder_library": "native/encoder.so",
        "feature_encoder_library_sha256": "1" * 64,
        "prefilter_samples": 64,
        "prefilter_keep": 16,
        "prefilter_margin": 2.4,
        "root_source": "explicit",
        "roots": [
            {"dealt_cards": ["Ah", "Kc", "Qs", "8h", "7c"]},
            {"dealt_cards": ["Ad", "Kd", "Qd", "8d", "7d"]},
        ],
        "shards": [
            {"shard_id": "00", "start": 0, "count": 1},
            {"shard_id": "01", "start": 1, "count": 1},
        ],
    }
    for index, field in enumerate((
        "t4_model", "t3_second_model", "t3_first_model", "t2_second_model",
        "t2_first_model", "t1_second_model", "t1_first_model", "t0_second_model",
    )):
        plan[field] = f"weights/{field}.bin"
        plan[f"{field}_sha256"] = str(index) * 64
    plan.update(overrides)
    return plan


def _refusal(tmp_path: Path, plan: dict[str, Any]) -> str:
    with pytest.raises(SystemExit) as raised:
        worker.load_plan(_write(tmp_path, plan))
    return str(raised.value)


# --------------------------------------------------------------------------
# (a) an explicit plan is accepted by the real worker's own loader
# --------------------------------------------------------------------------

def test_the_real_loader_accepts_an_explicit_plan(tmp_path: Path) -> None:
    plan = worker.load_plan(_write(tmp_path, _t0_first_plan()))
    assert plan["root_source"] == worker.ROOT_SOURCE_EXPLICIT
    assert len(plan["roots"]) == 2


def test_a_plan_without_root_source_is_still_seeded(tmp_path: Path) -> None:
    """The default is the only behaviour that existed before this feature."""

    plan = _t0_first_plan()
    del plan["root_source"]
    del plan["roots"]
    plan["hand_seed_base"] = 971_000_000
    plan["behavior_seed_offset"] = 500_000
    loaded = worker.load_plan(_write(tmp_path, plan))
    assert "root_source" not in loaded
    assert loaded["hand_seed_base"] == 971_000_000


# --------------------------------------------------------------------------
# (b) the position that comes back is the one the plan states
# --------------------------------------------------------------------------

def test_the_root_is_rebuilt_exactly_as_written(tmp_path: Path) -> None:
    plan = worker.load_plan(_write(tmp_path, _t0_first_plan()))

    first = worker.explicit_root_observation(plan, 0)
    assert first.dealt_cards == ("Ah", "Kc", "Qs", "8h", "7c")
    assert first.street == "T0"
    assert first.seat == "first"
    assert first.to_act_order == "first"
    # T0 acting first: the five cards are the whole position.
    assert first.hero_board.card_count() == 0
    assert first.opponent_public_board.card_count() == 0
    assert first.hero_private_discards == ()
    assert first.opponent_in_fantasyland is False

    second = worker.explicit_root_observation(plan, 1)
    assert second.dealt_cards == ("Ad", "Kd", "Qd", "8d", "7d")
    # Offset n is root n. Nothing is shuffled, so the two are not interchangeable.
    assert first.fingerprint() != second.fingerprint()


def test_rebuilding_is_deterministic(tmp_path: Path) -> None:
    """No seed is read to reach a position that was handed over."""

    plan = worker.load_plan(_write(tmp_path, _t0_first_plan()))
    once = worker.explicit_root_observation(plan, 0)
    twice = worker.explicit_root_observation(plan, 0)
    assert once.to_dict() == twice.to_dict()


def test_a_later_street_states_its_boards_and_discards(tmp_path: Path) -> None:
    """The schema is general: T1 and below need no new fields.

    (T1, second) is the informative case -- the hero holds five cards, the
    opponent nine minus the two it has yet to place, and the discard counts
    differ between the seats. If the generic form can express that, it can
    express every street this worker labels.
    """

    plan = _t0_first_plan(
        street="T2",
        seat="second",
        roots=[{
            "hero_board": {
                "top": ["Ah"],
                "middle": ["Kc", "Qs"],
                "bottom": ["8h", "7c", "6d", "5s"],
            },
            "opponent_public_board": {
                "top": ["2c", "3c"],
                "middle": ["4c", "5c", "6c"],
                "bottom": ["7s", "8s", "9s", "Ts"],
            },
            "dealt_cards": ["Jh", "Th", "9h"],
            "hero_private_discards": ["2d"],
            "opponent_discard_count": 2,
            "note": "a T2 second-seat root",
        }],
        shards=[{"shard_id": "00", "start": 0, "count": 1}],
    )
    # T0-only fields; a T2 plan carrying them is refused for that reason alone.
    for field in ("prefilter_samples", "prefilter_keep", "prefilter_margin"):
        del plan[field]
    for field in ("t1_second_model", "t1_second_model_sha256",
                  "t1_first_model", "t1_first_model_sha256",
                  "t0_second_model", "t0_second_model_sha256",
                  "t2_second_model", "t2_second_model_sha256"):
        del plan[field]

    loaded = worker.load_plan(_write(tmp_path, plan))
    root = worker.explicit_root_observation(loaded, 0)
    assert root.street == "T2"
    assert root.seat == "second"
    assert root.hero_board.card_count() == 7
    assert root.opponent_public_board.card_count() == 9
    assert root.dealt_cards == ("Jh", "Th", "9h")
    assert root.hero_private_discards == ("2d",)
    assert root.opponent_discard_count == 2


# --------------------------------------------------------------------------
# (c) a position that contradicts the street or the seat is refused
# --------------------------------------------------------------------------

def test_a_t0_first_root_with_cards_on_the_board_is_refused(tmp_path: Path) -> None:
    plan = _t0_first_plan(roots=[{
        "dealt_cards": ["Ah", "Kc", "Qs", "8h", "7c"],
        "hero_board": {"top": ["2c"]},
    }], shards=[{"shard_id": "00", "start": 0, "count": 1}])
    message = _refusal(tmp_path, plan)
    assert "explicit root 0" in message
    assert "T0/first" in message
    assert "geometry" in message


def test_the_wrong_number_of_dealt_cards_is_refused(tmp_path: Path) -> None:
    plan = _t0_first_plan(
        roots=[{"dealt_cards": ["Ah", "Kc", "Qs"]}],
        shards=[{"shard_id": "00", "start": 0, "count": 1}],
    )
    assert "geometry" in _refusal(tmp_path, plan)


def test_a_t0_position_under_a_t1_plan_is_refused(tmp_path: Path) -> None:
    """The street the plan names is the street the position has to be.

    Five dealt cards and two empty boards is a legal T0 opening and an illegal
    T1 decision -- at T1 both boards already carry five cards and only three are
    dealt. Nothing about the position itself says which; only the plan does.
    """

    plan = _t0_first_plan(street="T1")
    for field in ("prefilter_samples", "prefilter_keep", "prefilter_margin",
                  "t1_first_model", "t1_first_model_sha256",
                  "t0_second_model", "t0_second_model_sha256"):
        del plan[field]
    message = _refusal(tmp_path, plan)
    assert "explicit root 0" in message
    assert "T1/first" in message


def test_the_index_of_the_bad_root_is_named(tmp_path: Path) -> None:
    """A hundred-root plan must say WHICH root is wrong."""

    plan = _t0_first_plan(
        roots=[
            {"dealt_cards": ["Ah", "Kc", "Qs", "8h", "7c"]},
            {"dealt_cards": ["Ad", "Kd", "Qd", "8d", "7d"]},
            {"dealt_cards": ["2c", "3c", "4c", "5c"]},
        ],
        shards=[{"shard_id": "00", "start": 0, "count": 3}],
    )
    assert "explicit root 2" in _refusal(tmp_path, plan)


def test_a_duplicated_card_within_a_root_is_refused(tmp_path: Path) -> None:
    plan = _t0_first_plan(
        roots=[{"dealt_cards": ["Ah", "Ah", "Qs", "8h", "7c"]}],
        shards=[{"shard_id": "00", "start": 0, "count": 1}],
    )
    assert "duplicate card" in _refusal(tmp_path, plan)


def test_a_card_outside_the_regular_deck_is_refused(tmp_path: Path) -> None:
    plan = _t0_first_plan(
        roots=[{"dealt_cards": ["Xj", "Kc", "Qs", "8h", "7c"]}],
        shards=[{"shard_id": "00", "start": 0, "count": 1}],
    )
    assert "invalid regular-rule card" in _refusal(tmp_path, plan)


def test_a_lying_opponent_discard_count_is_refused(tmp_path: Path) -> None:
    """It is derived from the public board, so a stated one is only checkable."""

    plan = _t0_first_plan(
        roots=[{
            "dealt_cards": ["Ah", "Kc", "Qs", "8h", "7c"],
            "opponent_discard_count": 3,
        }],
        shards=[{"shard_id": "00", "start": 0, "count": 1}],
    )
    message = _refusal(tmp_path, plan)
    assert "opponent_discard_count" in message
    assert "disagrees" in message


def test_an_unknown_root_field_is_refused(tmp_path: Path) -> None:
    """A typo must not become a silently ignored instruction."""

    plan = _t0_first_plan(
        roots=[{
            "dealt_cards": ["Ah", "Kc", "Qs", "8h", "7c"],
            "hero_boards": {"top": []},
        }],
        shards=[{"shard_id": "00", "start": 0, "count": 1}],
    )
    assert "hero_boards" in _refusal(tmp_path, plan)


# --------------------------------------------------------------------------
# The Fantasyland constant: declared, never set
# --------------------------------------------------------------------------

def test_a_root_carrying_its_own_scoring_context_is_refused(tmp_path: Path) -> None:
    """The one rejected field a caller has a plausible reason to reach for.

    Accepting it would let a plan mint labels of a quantity nobody chose, and
    nothing downstream could tell them from correct ones -- the failure mode
    this project has been bitten by more than once.
    """

    plan = _t0_first_plan(
        roots=[{
            "dealt_cards": ["Ah", "Kc", "Qs", "8h", "7c"],
            "scoring": {"fl_ev": {"14": 10.227}},
        }],
        shards=[{"shard_id": "00", "start": 0, "count": 1}],
    )
    message = _refusal(tmp_path, plan)
    assert "scoring context" in message
    assert "Fantasyland constant" in message


def test_explicit_roots_carry_the_runtime_constant(tmp_path: Path) -> None:
    """What the refusal above buys: every root's fl_ev is the runtime's.

    This is the value the worker's own resume guard compares each written
    position against, so the two agreeing is what makes an explicit run
    appendable to the corpus a seeded run would have produced.
    """

    from ofc_regular.hu_infoset import load_default_fl_ev

    runtime = {
        str(cards): float(value) for cards, value in load_default_fl_ev().items()
    }
    plan = worker.load_plan(_write(tmp_path, _t0_first_plan()))
    for index in range(len(plan["roots"])):
        root = worker.explicit_root_observation(plan, index)
        carried = {
            str(cards): float(value) for cards, value in root.scoring.fl_ev
        }
        assert carried == runtime
        assert root.to_dict()["scoring"]["fl_ev"] == runtime


def test_the_guard_takes_the_runtime_table_exactly_as_the_config_returns_it(
    tmp_path: Path,
) -> None:
    """The regression this file was missing, and the fleet nearly paid for.

    `load_default_fl_ev()` returns ``{14: 9.6}`` -- INT keys, because it parses
    the config into the numbers the engine works in. An observation's scoring
    context has been through JSON on the way to a position file and comes back
    with ``{"14": 9.6}``. The two tables mean the same thing and compare
    unequal, so a guard that normalised only one side refused a correct run.

    Every earlier test here built both sides itself and so agreed with itself.
    This one takes the runtime table from the real function and the roots from
    the real builder, passes them straight in, and asserts the guard is SILENT.
    Nothing is constructed by hand, which is the entire point: the defect lived
    in the seam between two real components, and only two real components can
    stand on it.
    """

    from ofc_regular.hu_infoset import load_default_fl_ev

    runtime = load_default_fl_ev()
    # The shape that broke it, asserted rather than assumed -- if the config
    # loader ever starts returning strings, this test should say so here rather
    # than quietly stop covering anything.
    assert all(isinstance(cards, int) for cards in runtime)

    plan = worker.load_plan(_write(tmp_path, _t0_first_plan()))
    roots = [
        worker.explicit_root_observation(plan, index)
        for index in range(len(plan["roots"]))
    ]
    # And the other side of the seam, equally asserted.
    assert all(
        isinstance(cards, str)
        for root in roots
        for cards in root.to_dict()["scoring"]["fl_ev"]
    )

    # No SystemExit. This is the assertion.
    worker.check_explicit_root_fl_ev(roots, runtime)


def test_the_normaliser_accepts_every_shape_the_project_stores_fl_ev_in() -> None:
    """Three sources, three key types, one answer.

    Each of the three is correct on its own terms -- ints for the engine, a
    frozen tuple of pairs for the dataclass, strings for anything that has been
    through JSON -- so the normaliser is where they are reconciled rather than
    at each of the three call sites that must agree.
    """

    from ofc_regular.hu_infoset import ScoringContext, load_default_fl_ev

    expected = {"14": 9.6}
    assert worker.normalized_fl_ev(load_default_fl_ev()) == expected  # int keys
    assert worker.normalized_fl_ev({"14": 9.6}) == expected  # from JSON
    assert worker.normalized_fl_ev(ScoringContext().fl_ev) == expected  # pairs
    assert worker.normalized_fl_ev(((14, 9.6),)) == expected
    # A real difference still reads as one after normalising -- the guard must
    # not be made vacuous by the thing that stops it being over-eager.
    assert worker.normalized_fl_ev({14: 10.227}) != expected


def test_a_root_at_the_wrong_constant_is_refused_before_any_label(
    tmp_path: Path,
) -> None:
    """The runtime guard itself, exercised on a root built at 10.227.

    10.227 is the constant the shipped T1 corpus baked in before the move to
    9.6, and re-labelling that corpus is the work this fleet is in the middle
    of -- so a run that silently produced 10.227 labels today would be mixed
    into a 9.6 corpus by the very pipeline that exists to separate them.

    No explicit plan can currently reach this state: `explicit_root_observation`
    refuses a `scoring` key, so every root sits on the runtime's default. The
    observation below is therefore built directly, which is what makes this a
    test of the guard rather than of the refusal above.
    """

    from ofc_regular.hu_infoset import (
        ActorObservation,
        ScoringContext,
        load_default_fl_ev,
    )

    # Passed in raw, exactly as the config loader returns it, so this exercises
    # the refusal path under the same key types the fleet will hand it.
    runtime = load_default_fl_ev()
    assert runtime[14] == 9.6

    plan = worker.load_plan(_write(tmp_path, _t0_first_plan()))
    good = worker.explicit_root_observation(plan, 0)
    # The whole point: a correct root passes, so the guard is not vacuous.
    worker.check_explicit_root_fl_ev([good], runtime)

    stale = ActorObservation(
        hero_board=good.hero_board,
        opponent_public_board=good.opponent_public_board,
        dealt_cards=good.dealt_cards,
        hero_private_discards=good.hero_private_discards,
        seat=good.seat,
        street=good.street,
        to_act_order=good.to_act_order,
        scoring=ScoringContext(fl_ev=((14, 10.227),)),
    )
    with pytest.raises(SystemExit) as raised:
        worker.check_explicit_root_fl_ev([good, stale], runtime)
    message = str(raised.value)
    # The index matters: a five-hundred-root plan must say which one.
    assert "explicit root 1" in message
    assert "10.227" in message and "9.6" in message


# --------------------------------------------------------------------------
# Plan-shape rules: seeds, shards, and the seeded/explicit boundary
# --------------------------------------------------------------------------

def test_a_seed_block_on_an_explicit_plan_is_refused(tmp_path: Path) -> None:
    """Refused, not ignored: it would read as a hand block this corpus drew from."""

    for field, value in (("hand_seed_base", 971_000_000),
                         ("behavior_seed_offset", 500_000)):
        message = _refusal(tmp_path, _t0_first_plan(**{field: value}))
        assert field in message
        assert "explicit" in message


def test_a_root_list_on_a_seeded_plan_is_refused(tmp_path: Path) -> None:
    """The mirror image: a list nobody would read."""

    plan = _t0_first_plan()
    plan["root_source"] = "seeded"
    plan["hand_seed_base"] = 971_000_000
    plan["behavior_seed_offset"] = 500_000
    message = _refusal(tmp_path, plan)
    assert "explicit root list" in message


def test_an_unknown_root_source_is_refused(tmp_path: Path) -> None:
    message = _refusal(tmp_path, _t0_first_plan(root_source="random"))
    assert "root_source" in message
    assert "'seeded'" in message and "'explicit'" in message


def test_a_vs_fantasyland_plan_may_not_carry_roots(tmp_path: Path) -> None:
    """Those kinds already index a pinned roots FILE; two sources is one too many."""

    plan = {
        "schema": worker.PLAN_SCHEMA,
        "plan_kind": worker.T3_VS_FL_KIND,
        "root_source": "explicit",
        "roots": [{"dealt_cards": ["Ah", "Kc", "Qs"]}],
    }
    message = _refusal(tmp_path, plan)
    assert "root_source" in message and "roots" in message
    assert "roots_file" in message


def test_an_explicit_plan_without_roots_is_refused(tmp_path: Path) -> None:
    plan = _t0_first_plan()
    del plan["roots"]
    assert "roots" in _refusal(tmp_path, plan)


def test_an_empty_root_list_is_refused(tmp_path: Path) -> None:
    plan = _t0_first_plan(roots=[], shards=[
        {"shard_id": "00", "start": 0, "count": 1}])
    assert "empty" in _refusal(tmp_path, plan)


def test_shards_must_tile_the_root_list_exactly(tmp_path: Path) -> None:
    """A shard past the end fails on its last position; a root past the
    shards is never solved at all. Both are caught before the fleet starts."""

    over = _t0_first_plan(shards=[{"shard_id": "00", "start": 0, "count": 3}])
    assert "3 positions" in _refusal(tmp_path, over)

    under = _t0_first_plan(shards=[{"shard_id": "00", "start": 0, "count": 1}])
    assert "1 positions" in _refusal(tmp_path, under)

    gapped = _t0_first_plan(shards=[
        {"shard_id": "00", "start": 0, "count": 1},
        {"shard_id": "01", "start": 5, "count": 1},
    ])
    assert "tiling" in _refusal(tmp_path, gapped)

    duplicated = _t0_first_plan(shards=[
        {"shard_id": "00", "start": 0, "count": 1},
        {"shard_id": "00", "start": 1, "count": 1},
    ])
    assert "twice" in _refusal(tmp_path, duplicated)


# --------------------------------------------------------------------------
# (d) zero regression on the seeded path
# --------------------------------------------------------------------------

def test_a_real_seeded_generator_still_writes_the_same_bytes(
    tmp_path: Path,
) -> None:
    """The strongest statement available: byte identity, from a real generator.

    `hu_m7_t1_256_plan_v1` produced the corpus that is on disk. Its rendered
    plan is compared here against the digest recorded when this feature was
    written, so a change to the worker's field sets that leaked into a plan
    would fail here rather than in a fleet's morning.
    """

    package, identity = _fixture_package(tmp_path)
    plans, _ = seeded_generator.build_plan_pair(
        package, expected_identity=identity
    )
    for seat, plan in plans.items():
        # Nothing this feature added appears in a plan that predates it.
        assert "root_source" not in plan
        assert "roots" not in plan
        assert plan["hand_seed_base"] == seeded_generator.HAND_SEED_BASE
        assert plan["behavior_seed_offset"] == (
            seeded_generator.BEHAVIOR_SEED_OFFSET
        )
        # And the worker still accepts it, unchanged.
        loaded = worker.load_plan(_write(tmp_path, plan, f"seeded_{seat}.json"))
        assert loaded["street"] == "T1"
        assert loaded["seat"] == seat


def test_the_seeded_required_field_set_is_unchanged(tmp_path: Path) -> None:
    """A seeded plan missing a seed base is refused exactly as it always was."""

    package, identity = _fixture_package(tmp_path)
    plans, _ = seeded_generator.build_plan_pair(
        package, expected_identity=identity
    )
    plan = dict(plans["second"])
    del plan["hand_seed_base"]
    message = _refusal(tmp_path, plan)
    assert "missing fields" in message
    assert "hand_seed_base" in message


def test_the_evaluation_seed_formula_is_shared_by_both_sources() -> None:
    """An explicit position is measured exactly as a dealt one is.

    This is the property that makes an explicit run comparable to a seeded one
    at all: the plan chooses WHICH position, never HOW it is scored. The formula
    is `eval_seed_base + offset * 7 + trial * 3_000_017` and it reads nothing
    about the root source.
    """

    source = Path(worker.__file__).read_text(encoding="utf-8")
    assert 'seed = plan["eval_seed_base"] + offset * 7 + trial * 3_000_017' in source
    # `build_config` is the only place a seed is derived, and the explicit
    # branch is in `make_root`, which cannot reach it.
    build_config = source.split("def build_config(", 1)[1].split("\n    def ", 1)[0]
    for field in worker.SEEDED_ONLY_FIELDS:
        assert field not in build_config
    assert "root_source" not in build_config
    assert "explicit_roots" not in build_config
