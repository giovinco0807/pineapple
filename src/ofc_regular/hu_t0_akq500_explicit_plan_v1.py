"""Build the immutable T0 first-seat plan for the AKQ-500 hand set.

The first plan written against ``root_source: "explicit"``, and the reason that
worker feature exists.  Every label plan before it drew its positions from
``hand_seed_base + offset``, which is the right shape for a corpus of
independent random hands and the wrong shape for a *measurement*: a seed cannot
be asked for five hundred openings that each hold an ace, a king or a queen.
This plan carries those five hundred openings itself, and the fleet solves the
list.

## What is being measured

``docs/t0_restricted_eval_20260819.md`` established, on four hands worked by
hand, that the production T0 root schedule -- 256 particles behind a two-stage
prefilter with the stage-two race -- can put a candidate first that a uniform
high-particle re-score places near the bottom.  On ``As Kd 7c 7h 2s`` the race's
winner finished 2.3 points behind the true best family, its trajectory sinking
monotonically as particles were added: the classic regression of a candidate
that led on early noise.  Four hands is an anecdote.  Five hundred, drawn under
a stated filter and solved under one schedule, is a rate.

So this plan is deliberately NOT the production schedule.  It is the settled
half of the two-step the same document argues for -- narrow with a sieve, then
re-score what survives uniformly:

* ``prefilter_samples: 64`` over the whole 232-action root fan.  Double the
  production 32, and bought on structure rather than on a measured miss.  That
  document's section 9 reported the top-32 sieve dropping a hand's true best
  action, and its section 11 **retracted** that finding on six 5,096-particle
  batches: the sieve's original first place was right all along.  So no
  surviving case shows the sieve losing a winner.  What remains true is that it
  is the only stage whose mistakes are irreversible -- an action it drops is
  never scored at all, whereas anything stage two misranks is at least
  measured -- and at roughly a sixth of a root's cost, doubling it is the
  cheapest insurance on the schedule.
* ``prefilter_keep: 16`` with ``prefilter_margin: 2.4``.  The margin carries
  through anything the coarse stage cannot separate across the keep boundary,
  which is what let a -0.72 sieve score reach the top three at 1,024 particles.
* ``samples: 1024``, spent uniformly.  **No race.**  The race is the mechanism
  under measurement; a plan that raced would be measuring itself.

One label run a position (``seeds_per_position: 1``).  The document's own
discipline -- a claim about one hand needs four-plus batches at 5,096 particles
before it is more than a lean -- is a rule for adjudicating a SINGLE hand, and
this run is not adjudicating any of them.  It is one uniform 1,024-particle
observation of each of five hundred, which is what a distribution is made of.

## The hands

``D:/ofc_data/t0_rung/akq500/hands.json``, generated 2026-08-19 under seed
20260819: 500 five-card openings, each containing at least one ace, king or
queen, deduplicated suit-canonically.  The file's own digest is pinned in the
plan's provenance, and the roots are written in the file's order, so offset *n*
of the output corpus is hand *n* of that file with nothing in between.

The plan states each root as ``dealt_cards`` alone.  At T0 acting first the
position IS the five cards -- both boards empty, nothing discarded -- so every
other field of the explicit-root schema takes its default, and the worker's
``ActorObservation`` check refuses anything that is not that shape.

## What this plan may not do

It cannot set the Fantasyland constant.  ``fl_ev_cards``/``fl_ev_value`` are a
declaration checked against the runtime's config, exactly as on every seeded
plan; the value reaches the engine through the observation's scoring context
from the one config ``hu_infoset`` reads.  A corpus labelled at a constant
nobody chose looks exactly like a correct one from the outside, which is why
that check is the worker's and not this generator's to relax.

It also carries no hand or behavior seed block, and the worker refuses one on an
explicit plan.  There is no seed collision to audit here, because there is no
seed that chooses a position -- the only seed block this plan spends is the
evaluation progression, which is what MEASURES the fixed positions and is
allocated clear of every base the cascade ledger records.

Output is write-once.  Existing plans are evidence and are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any, Mapping, Sequence

# The consumer's own names, imported rather than retyped so this generator
# cannot drift from the worker that has to accept what it writes.
from ofc_regular.hu_m31_label_gen_worker_v1 import (
    PRUNING_SAFETY_FIELDS,
    RACE_FIELDS,
    ROOT_SOURCE_EXPLICIT,
    SEEDED_ONLY_FIELDS,
)
from ofc_regular.hu_m7_t1_256_plan_v1 import (  # noqa: F401
    DEFAULT_PACKAGE_IDENTITY,
    EXPECTED_LEDGER_SHA256,
    EXPECTED_RUNTIME_SHA256,
    EXPECTED_WHEELHOUSE_SHA256,
    PLAN_SCHEMA,
    ExpectedPackageIdentity,
    PackageAudit,
    PlanValidationError,
    _package_pin,
    _require,
    audit_package,
    make_shards,
    rendered_json_bytes,
    sha256_bytes,
)

PROVENANCE_SCHEMA = "hu_t0_akq500_explicit_plan_provenance_v1"
PLAN_SET_SCHEMA = "hu_t0_akq500_explicit_plan_set_v1"
HANDS_SCHEMA_KEYS = ("hands",)

STREET = "T0"
SEAT = "first"
POSITION_COUNT = 500
# Eight roots a shard, near enough: 500 over 63 shards is fifty-nine of eight
# and four of seven.  The intended machine is a c4-standard-8 running eight
# strided workers, so each worker takes exactly one root and a shard's wall time
# IS one root's -- about three hours at the measured 10,573 core-s, half the
# six-hour watchdog.  See the sizing note in the provenance below.
SHARD_COUNT = 63
SAMPLES = 1_024
SEEDS_PER_POSITION = 1

# The sieve.  Wider than production at both ends, for the reasons in the module
# docstring: stage one is the only stage that can drop the true best action, and
# this run exists to measure how often the production schedule's answer is wrong.
PREFILTER_SAMPLES = 64
PREFILTER_KEEP = 16
PREFILTER_MARGIN = 2.4

FL_EV_CARDS = 14
FL_EV_VALUE = 9.6

# The one seed block this plan spends.  The worker derives a position's
# evaluation seed as `eval_seed_base + offset * 7 + trial * 3_000_017`, so with
# one label run a position the progression runs 20,000,000..20,003,493.
#
# Clear of everything: the label corpora hold eval bases 5/6/8/9/10/11/12M and
# 12.1M (whose progression reaches 12,148,993 at most), and the T0 restricted-
# evaluation experiments of 2026-08-19 spent 61M through 97M.  Twenty million
# sits in the gap between those two families with more than seven million of
# headroom on either side.
EVAL_SEED_BASE = 20_000_000
EVAL_SEED_STRIDE = 7

JOB_ID = "t0first-akq500-1024p"
PLAN_FILENAME = "worker_plan_t0first_akq500_1024p.json"
MANIFEST_FILENAME = "plan_set_t0first_akq500_1024p.json"

DEFAULT_HANDS_FILE = pathlib.Path("D:/ofc_data/t0_rung/akq500/hands.json")

# Every eval-seed progression already spent, as (label, base, positions), so the
# check below compares seed SETS rather than bases.  A base alone says nothing:
# the stride is seven, so a block is seven times wider than its position count.
_RECORDED_EVAL_PROGRESSIONS: tuple[tuple[str, int, int], ...] = (
    ("m7v6 T1 256p corpora", 12_000_000, 18_000),
    ("m7v6 T1 second-seat top-up", 12_100_000, 7_000),
    ("m7v6 T1 1024p plans (written, never run)", 11_000_000, 25_000),
    ("m7v5 T2 2048p corpora", 10_000_000, 25_000),
)
# Bases the cascade ledger records without a position count, and the T0
# restricted-evaluation experiment seeds of 2026-08-19.  A base nobody sized is
# reserved this wide so arithmetic cannot walk into it.
_UNRECORDED_EXTENT = 1_000_000
_DOC_RECORDED_EVAL_BASES: tuple[int, ...] = (
    5_000_000, 6_000_000, 8_000_000, 9_000_000,
    61_000_000, 62_000_000, 63_000_000, 64_000_000, 65_000_000,
    66_000_000, 67_000_000, 71_000_000, 72_000_000, 73_000_000,
    74_000_000, 75_000_000, 81_000_000, 82_000_000, 91_000_000,
    92_000_000, 96_000_000, 97_000_000,
)

# The full continuation ladder a T0 first-seat teacher plays through.  Eight
# learned evaluators and the T4 terminal: the worker refuses the kind without
# every one of them, because from the opening street every later decision --
# both seats at T1 and T2, the opponent's T3 first-seat reply, the T4 terminal,
# and the opponent's own T0 reply -- is still ahead of the teacher.
#
# All full precision.  No `fast_*` twin is pinned: this run measures how well a
# root schedule ranks openings, and answering the replies from distilled
# networks would fold a second approximation into the number being read.
_MODEL_PINS: tuple[tuple[str, str], ...] = (
    ("t4_model", "weights/t4_model_v6.bin"),
    ("t3_second_model", "weights/t3_model_v3.bin"),
    ("t3_first_model", "weights/t3first_model_v2.bin"),
    ("t2_second_model", "weights/t2_model_v2.bin"),
    ("t2_first_model", "weights/t2first_model_v2.bin"),
    ("t1_second_model", "weights/t1_model_v1.bin"),
    ("t1_first_model", "weights/t1first_model_v1.bin"),
    ("t0_second_model", "weights/t0_model_v1.bin"),
)


def eval_seed_span(base: int, positions: int) -> tuple[int, int]:
    """The closed span the worker's stride-seven progression actually touches."""

    _require(positions > 0, "positions must be positive")
    return base, base + (positions - 1) * EVAL_SEED_STRIDE


def eval_progressions_disjoint(
    base_a: int, positions_a: int, base_b: int, positions_b: int
) -> bool:
    """Whether two stride-seven progressions share a single seed.

    Compared as sets, not as ranges.  Two blocks of the same stride overlap in
    range all the time and still share nothing when their residues differ, and
    the property that matters is the seed, not the interval.
    """

    left = {base_a + index * EVAL_SEED_STRIDE for index in range(positions_a)}
    right = {base_b + index * EVAL_SEED_STRIDE for index in range(positions_b)}
    return not (left & right)


def check_seed_allocation() -> dict[str, Any]:
    """Prove this plan's evaluation seeds are untouched, at build time."""

    conflicts: list[str] = []
    for label, base, positions in _RECORDED_EVAL_PROGRESSIONS:
        if not eval_progressions_disjoint(
            EVAL_SEED_BASE, POSITION_COUNT, base, positions
        ):
            conflicts.append(f"{label} (base {base:,})")
    low, high = eval_seed_span(EVAL_SEED_BASE, POSITION_COUNT)
    for base in _DOC_RECORDED_EVAL_BASES:
        if base <= high and low <= base + _UNRECORDED_EXTENT:
            conflicts.append(f"unsized base {base:,}")
    _require(
        not conflicts,
        "evaluation seed block collides with " + ", ".join(conflicts),
    )
    return {
        "eval_seed_base": EVAL_SEED_BASE,
        "eval_seed_span": [low, high],
        "eval_seed_stride": EVAL_SEED_STRIDE,
        "compared_against": [label for label, _, _ in _RECORDED_EVAL_PROGRESSIONS],
        "unsized_bases_reserved": list(_DOC_RECORDED_EVAL_BASES),
        "hand_seed_block_spent": None,
        "note": (
            "An explicit plan spends no hand or behavior seed block: no seed "
            "chooses a position, so there is no duplicate-position hazard to "
            "audit. The evaluation progression is the only one allocated, and "
            "it is compared as a seed SET rather than a range because the "
            "stride is seven and blocks of one stride overlap in range while "
            "sharing nothing."
        ),
    }


def load_hands(path: pathlib.Path) -> tuple[list[list[str]], dict[str, Any]]:
    """Read the hand file and return its hands plus its own provenance.

    Validated hard.  A hand list is the entire specification of what this run
    measures, so a file that is the wrong length, or holds a hand of the wrong
    size, or repeats one, is a different measurement wearing this plan's name.
    """

    _require(path.is_file(), f"hand file is missing: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PlanValidationError(f"hand file is not UTF-8 JSON: {path}") from error
    _require(isinstance(payload, Mapping), "hand file must be a JSON object")
    for key in HANDS_SCHEMA_KEYS:
        _require(key in payload, f"hand file is missing {key!r}")
    hands = payload["hands"]
    _require(isinstance(hands, list), "hands must be a list")
    _require(
        len(hands) == POSITION_COUNT,
        f"hand file holds {len(hands)} hands; this plan is written for "
        f"{POSITION_COUNT}",
    )
    seen: set[tuple[str, ...]] = set()
    for index, hand in enumerate(hands):
        _require(
            isinstance(hand, list) and len(hand) == 5,
            f"hand {index} is not a list of five cards: {hand!r}",
        )
        for card in hand:
            _require(
                isinstance(card, str),
                f"hand {index} holds {card!r}, which is not a card string",
            )
        _require(
            len(set(hand)) == 5, f"hand {index} repeats a card: {hand!r}"
        )
        key = tuple(hand)
        _require(key not in seen, f"hand {index} duplicates an earlier hand")
        seen.add(key)
    # The card alphabet and the T0 geometry are NOT re-derived here. The worker
    # builds an `ActorObservation` from every root when it loads the plan, and
    # that is the single copy of those rules; a second one in this file would be
    # free to drift from the engine's.
    source = {
        "path": str(path),
        "sha256": digest,
        "bytes": len(raw),
        "count": len(hands),
    }
    for key in ("seed", "filter", "dedup", "attempts"):
        if key in payload:
            source[key] = payload[key]
    return [list(hand) for hand in hands], source


def build_roots(hands: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    """One explicit root a hand, in the hand file's order.

    ``dealt_cards`` alone.  At T0 acting first both boards are empty and nothing
    has been discarded, so the five cards are the whole position and every other
    field of the explicit-root schema is at its default. Writing the defaults
    out would say the same thing in more bytes, and would give a later reader
    the impression that some other value had been possible here.
    """

    return [{"dealt_cards": list(hand)} for hand in hands]


def _plan(audit: PackageAudit, roots: Sequence[Mapping[str, Any]],
          source: Mapping[str, Any]) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "job_id": JOB_ID,
        "street": STREET,
        "seat": SEAT,
        "samples": SAMPLES,
        "seeds_per_position": SEEDS_PER_POSITION,
        "eval_seed_base": EVAL_SEED_BASE,
        "root_source": ROOT_SOURCE_EXPLICIT,
        "roots": [dict(root) for root in roots],
        "engine_library": "native/libofc_hu_m3_engine.so",
        "engine_library_sha256": _package_pin(
            audit, "native/libofc_hu_m3_engine.so"
        ),
        "feature_encoder_library": "native/libofc_stage3_feature_encoder.so",
        "feature_encoder_library_sha256": _package_pin(
            audit, "native/libofc_stage3_feature_encoder.so"
        ),
        "prefilter_samples": PREFILTER_SAMPLES,
        "prefilter_keep": PREFILTER_KEEP,
        "prefilter_margin": PREFILTER_MARGIN,
        "fl_ev_cards": FL_EV_CARDS,
        "fl_ev_value": FL_EV_VALUE,
        "shards": make_shards(
            position_count=POSITION_COUNT, shard_count=SHARD_COUNT
        ),
    }
    for field, relative in _MODEL_PINS:
        plan[field] = relative
        plan[f"{field}_sha256"] = _package_pin(audit, relative)

    plan["provenance"] = {
        "schema": PROVENANCE_SCHEMA,
        "generation": "m7v6",
        "package_audit": audit.provenance(),
        "hand_source": dict(source),
        "purpose": (
            "Measure how often the production T0 root schedule's answer "
            "survives a uniform high-particle re-score, on five hundred openings "
            "chosen to contain a broadway card rather than on four worked by "
            "hand. docs/t0_restricted_eval_20260819.md is the four-hand "
            "predecessor: on As Kd 7c 7h 2s the raced 256-particle winner "
            "finished 2.3 points behind the true best family, its score sinking "
            "monotonically as particles were added."
        ),
        "root_schedule": {
            "prefilter_samples": PREFILTER_SAMPLES,
            "prefilter_keep": PREFILTER_KEEP,
            "prefilter_margin": PREFILTER_MARGIN,
            "evaluation_samples": SAMPLES,
            "race": None,
            "why_no_race": (
                "The race is the mechanism under measurement. Stage two is "
                "spent uniformly so that every surviving action is scored at "
                "the resolution the label claims, which is the only footing on "
                "which the production schedule's ranking can be called right "
                "or wrong."
            ),
            "why_a_wider_sieve": (
                "64 rather than the production 32, and keep 16 with margin "
                "2.4. Bought on structure, not on a measured miss: section 9 "
                "of the 2026-08-19 document reported a top-32 sieve dropping a "
                "hand's true best action and section 11 retracted it on six "
                "5,096-particle batches, so no surviving case shows the sieve "
                "losing a winner. Stage one remains the only stage whose "
                "mistakes are irreversible -- a dropped action is never scored "
                "at all -- and at about a sixth of a root's cost it is the "
                "cheapest place on this schedule to buy margin."
            ),
        },
        "continuation": {
            field: relative for field, relative in _MODEL_PINS
        },
        "continuation_precision": (
            "Full precision throughout: no fast_* twin is pinned. A distilled "
            "reply would fold a second approximation into the very number this "
            "run reads."
        ),
        "seed_audit": check_seed_allocation(),
        "fantasyland": {
            "cards": FL_EV_CARDS,
            "value": FL_EV_VALUE,
            "config": "configs/fl_ev_regular_v4_selfplay.json",
            "config_sha256": _package_pin(
                audit, "configs/fl_ev_regular_v4_selfplay.json"
            ),
            "declared_not_set": (
                "A plan cannot set the constant. It reaches the engine through "
                "the observation's scoring context from the one config "
                "hu_infoset reads, for explicit roots exactly as for dealt "
                "ones. These fields are a declaration the worker checks against "
                "the runtime, and it checks a second time that every explicit "
                "root will carry that same constant into the engine."
            ),
        },
        "sharding": {
            "shards": SHARD_COUNT,
            "roots_per_shard": POSITION_COUNT // SHARD_COUNT,
            "concurrency_is_not_authorized_by_this_plan": True,
            "intended_machine_type": "c4-standard-8",
            "intended_workers_per_vm": 8,
            "measured_core_seconds_per_root": 10_573,
            "estimated_core_hours": round(POSITION_COUNT * 10_573 / 3600),
            "waves_at_the_regional_ceiling": 2,
            "sizing_reason": (
                "A root is MEASURED at 10,573 core-s under this exact schedule "
                "-- sieve 64 over the 232-action fan, keep 16 with margin 2.4, "
                "which in practice carries about 32 actions into a uniform "
                "1,024-particle stage two. Five hundred roots is therefore "
                "about 1,470 core-hours. Eight roots a shard across eight "
                "strided workers puts one root on each, so a shard's wall time "
                "is one root's: about 2.9 hours, half the six-hour watchdog, on "
                "a number that is measured rather than extrapolated. The "
                "binding constraint is the region's 248-vCPU ceiling, not the "
                "shard count: 63 shards of c4-standard-8 is two waves and "
                "about six hours of fleet time however they are laid out."
            ),
        },
        "explicit_roots": {
            "root_source": ROOT_SOURCE_EXPLICIT,
            "count": len(roots),
            "order": "the hand file's order; offset n is hand n",
            "fields_stated": ["dealt_cards"],
            "why_only_dealt_cards": (
                "At T0 acting first both boards are empty and nothing has been "
                "discarded, so the five cards are the whole position. The "
                "worker's ActorObservation check refuses anything that is not "
                "that shape."
            ),
            "seed_blocks_spent": ["eval_seed_base"],
        },
    }
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Everything this plan has to be, checked against nothing but itself."""

    _require(plan.get("schema") == PLAN_SCHEMA, "plan schema drifted")
    _require(plan.get("job_id") == JOB_ID, "plan job id drifted")
    _require(plan.get("street") == STREET, f"plan is not {STREET}")
    _require(plan.get("seat") == SEAT, f"plan is not the {SEAT} seat")
    _require(plan.get("samples") == SAMPLES, f"plan is not {SAMPLES}p")
    _require(
        plan.get("seeds_per_position") == SEEDS_PER_POSITION,
        "plan does not have exactly one label run",
    )
    _require(plan.get("eval_seed_base") == EVAL_SEED_BASE, "eval seed base drifted")
    _require(
        plan.get("root_source") == ROOT_SOURCE_EXPLICIT,
        "plan does not carry its own roots",
    )
    # No seed chooses a position here, and the worker refuses a plan that
    # implies one does.
    for field in SEEDED_ONLY_FIELDS:
        _require(field not in plan, f"explicit plan carries the seeded field {field}")

    roots = plan.get("roots")
    _require(
        isinstance(roots, list) and len(roots) == POSITION_COUNT,
        f"plan does not carry {POSITION_COUNT} roots",
    )
    seen: set[tuple[str, ...]] = set()
    for index, root in enumerate(roots):
        _require(isinstance(root, Mapping), f"root {index} is not an object")
        _require(
            set(root) == {"dealt_cards"},
            f"root {index} states fields other than dealt_cards: {sorted(root)}",
        )
        cards = root["dealt_cards"]
        _require(
            isinstance(cards, list) and len(cards) == 5,
            f"root {index} is not a five-card T0 opening",
        )
        key = tuple(cards)
        _require(key not in seen, f"root {index} duplicates an earlier root")
        seen.add(key)

    shards = plan.get("shards")
    _require(
        isinstance(shards, list) and len(shards) == SHARD_COUNT,
        f"plan does not have {SHARD_COUNT} shards",
    )
    expected_start = 0
    shard_ids: set[str] = set()
    for shard in shards:
        shard_id = shard.get("shard_id")
        _require(
            isinstance(shard_id, str) and shard_id not in shard_ids,
            "shard id is missing or duplicated",
        )
        shard_ids.add(shard_id)
        _require(
            shard.get("start") == expected_start,
            f"shard {shard_id} is not contiguous",
        )
        count = shard.get("count")
        _require(isinstance(count, int) and count > 0, "bad shard count")
        expected_start += count
    _require(expected_start == POSITION_COUNT, f"plan does not cover {POSITION_COUNT}")

    _require(plan.get("prefilter_samples") == PREFILTER_SAMPLES, "sieve width drifted")
    _require(plan.get("prefilter_keep") == PREFILTER_KEEP, "sieve keep drifted")
    _require(plan.get("prefilter_margin") == PREFILTER_MARGIN, "sieve margin drifted")
    _require("audit_full_every" not in plan, "plan carries an audit cadence")
    # The race is the mechanism under measurement; a plan that raced would be
    # measuring itself.
    for field in RACE_FIELDS:
        _require(field not in plan, f"plan carries the race field {field}")
    _require(
        set(PRUNING_SAFETY_FIELDS) & set(plan) == {"prefilter_margin"},
        "the only pruning-safety field this plan states is prefilter_margin",
    )
    # A T0 first-seat teacher plays through all eight learned evaluators; the
    # worker refuses the kind without every one of them.
    for field, _relative in _MODEL_PINS:
        _require(field in plan, f"plan is missing {field}")
        _require(f"{field}_sha256" in plan, f"plan is missing {field}_sha256")
    for field in plan:
        _require(
            not field.startswith("fast_"),
            f"plan pins the distilled {field}; this run measures a schedule and "
            "must not fold a second approximation into it",
        )
    _require(
        plan.get("fl_ev_cards") == FL_EV_CARDS
        and plan.get("fl_ev_value") == FL_EV_VALUE,
        "plan FL contract drifted",
    )
    _require(
        plan.get("provenance", {}).get("schema") == PROVENANCE_SCHEMA,
        "plan provenance schema drifted",
    )


def build_plan(
    package_dir: pathlib.Path,
    *,
    hands_file: pathlib.Path = DEFAULT_HANDS_FILE,
    expected_identity: ExpectedPackageIdentity = DEFAULT_PACKAGE_IDENTITY,
) -> tuple[dict[str, Any], PackageAudit]:
    audit = audit_package(package_dir, expected_identity=expected_identity)
    hands, source = load_hands(pathlib.Path(hands_file))
    plan = _plan(audit, build_roots(hands), source)
    validate_plan(plan)
    return plan, audit


def write_plan_once(
    package_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    hands_file: pathlib.Path = DEFAULT_HANDS_FILE,
    expected_identity: ExpectedPackageIdentity = DEFAULT_PACKAGE_IDENTITY,
) -> dict[str, Any]:
    plan, audit = build_plan(
        package_dir, hands_file=hands_file, expected_identity=expected_identity
    )
    rendered = {PLAN_FILENAME: rendered_json_bytes(plan)}
    manifest = {
        "schema": PLAN_SET_SCHEMA,
        "generator": "ofc_regular.hu_t0_akq500_explicit_plan_v1",
        "package_audit": audit.provenance(),
        "plans": {
            "first": {
                "filename": PLAN_FILENAME,
                "job_id": JOB_ID,
                "sha256": sha256_bytes(rendered[PLAN_FILENAME]),
            }
        },
        "measurement_contract": {
            "street": STREET,
            "seat": SEAT,
            "root_source": ROOT_SOURCE_EXPLICIT,
            "positions": POSITION_COUNT,
            "samples": SAMPLES,
            "seeds_per_position": SEEDS_PER_POSITION,
            "shards": SHARD_COUNT,
            "prefilter_samples": PREFILTER_SAMPLES,
            "prefilter_keep": PREFILTER_KEEP,
            "prefilter_margin": PREFILTER_MARGIN,
            "raced": False,
            "hand_source": plan["provenance"]["hand_source"],
            "eval_seed_base": EVAL_SEED_BASE,
            "eval_seed_span": list(eval_seed_span(EVAL_SEED_BASE, POSITION_COUNT)),
            "hand_seed_block_spent": False,
            "distilled_replies": False,
        },
    }
    rendered[MANIFEST_FILENAME] = rendered_json_bytes(manifest)

    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: output_dir / name for name in rendered}
    collisions = sorted(str(path) for path in targets.values() if path.exists())
    _require(not collisions, f"write-once output already exists: {collisions}")
    for name in (PLAN_FILENAME, MANIFEST_FILENAME):
        try:
            with targets[name].open("xb") as stream:
                stream.write(rendered[name])
                stream.flush()
        except FileExistsError as error:
            raise PlanValidationError(
                f"write-once race: output appeared while writing {targets[name]}"
            ) from error
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "write"), required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--hands-file", default=str(DEFAULT_HANDS_FILE))
    parser.add_argument("--out-dir", default="")
    # The package a plan is written against is part of its identity, and the
    # default is the m7v6 image these constants were first pinned to. m7v7
    # carries the same weights and the same engine but a runtime tree that
    # understands `root_source: explicit`; m7v6's worker does not, and a fleet
    # launched against it dies on every shard with "plan is missing fields:
    # ['behavior_seed_offset', 'hand_seed_base']" -- the plan was right and the
    # shipped code was a generation behind it. Passing the digests explicitly
    # is how a caller says which image it means, rather than editing a constant
    # and losing the record of what the earlier plans ran on.
    parser.add_argument("--expect-ledger-sha256", default="")
    parser.add_argument("--expect-runtime-sha256", default="")
    parser.add_argument("--expect-wheelhouse-sha256", default="")
    args = parser.parse_args(argv)

    package_dir = pathlib.Path(args.package_dir)
    hands_file = pathlib.Path(args.hands_file)
    overrides = (args.expect_ledger_sha256, args.expect_runtime_sha256,
                 args.expect_wheelhouse_sha256)
    if any(overrides) and not all(overrides):
        raise SystemExit(
            "--expect-*-sha256 are the three parts of one identity; give all "
            "three or none, because a partial override would check some of the "
            "package against the caller and the rest against a different image"
        )
    identity = (ExpectedPackageIdentity(*overrides) if all(overrides)
                else DEFAULT_PACKAGE_IDENTITY)
    if args.mode == "audit":
        if args.out_dir:
            raise SystemExit("--out-dir is only valid in write mode")
        plan, audit = build_plan(package_dir, hands_file=hands_file,
                                 expected_identity=identity)
        print(json.dumps({
            "schema": "hu_t0_akq500_explicit_plan_audit_summary_v1",
            "package_audit": audit.provenance(),
            "planned": {
                "street": STREET,
                "seat": SEAT,
                "root_source": ROOT_SOURCE_EXPLICIT,
                "positions": POSITION_COUNT,
                "samples": SAMPLES,
                "shards": SHARD_COUNT,
                "prefilter": [PREFILTER_SAMPLES, PREFILTER_KEEP, PREFILTER_MARGIN],
                "raced": False,
            },
            "hand_source": plan["provenance"]["hand_source"],
            "seed_audit": plan["provenance"]["seed_audit"],
            "writes_performed": False,
        }, indent=2, sort_keys=True))
        return 0

    if not args.out_dir:
        raise SystemExit("write mode requires --out-dir")
    print(json.dumps(
        write_plan_once(package_dir, pathlib.Path(args.out_dir),
                        hands_file=hands_file, expected_identity=identity),
        indent=2, sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HANDS_FILE",
    "DEFAULT_PACKAGE_IDENTITY",
    "EVAL_SEED_BASE",
    "EVAL_SEED_STRIDE",
    "ExpectedPackageIdentity",
    "FL_EV_VALUE",
    "JOB_ID",
    "MANIFEST_FILENAME",
    "PLAN_FILENAME",
    "POSITION_COUNT",
    "PREFILTER_KEEP",
    "PREFILTER_MARGIN",
    "PREFILTER_SAMPLES",
    "PackageAudit",
    "PlanValidationError",
    "SAMPLES",
    "SEAT",
    "SHARD_COUNT",
    "STREET",
    "audit_package",
    "build_plan",
    "build_roots",
    "check_seed_allocation",
    "eval_progressions_disjoint",
    "eval_seed_span",
    "load_hands",
    "make_shards",
    "validate_plan",
    "write_plan_once",
]
