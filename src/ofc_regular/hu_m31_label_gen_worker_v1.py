"""Portable label-generation worker for the learned-evaluator program.

One worker, one shard of positions, one plan. The plan pins everything that
decides what the labels mean -- engine library digest, model weight digests,
particle count, seed bases -- and the worker refuses to run against anything
whose digest is not the pinned one. The previous cloud workstream lost five
runs to defects that only surfaced on the VM; this worker is written to run
identically on a laptop, under WSL, or on a VM, so that a local run of the
exact artifact is the preflight.

Output is one create-only JSON file per position. Resume is therefore free:
a position whose file exists and parses is done, anything else is redone.
Preemption can at worst lose the position that was in flight.

Free resume has one edge the create-only rule does not cover. A shard directory
is resumed by scanning it, so a resume that lands on positions written under a
different Fantasyland constant would append labels of one quantity to a corpus
of another, and nothing downstream could tell which row was which. Every
position carries its own scoring context, so the worker compares it against the
constant this run would write and refuses the shard when they differ, by the
same rule the vs-Fantasyland kinds apply to ``opponent_mode``. A plan may also
state ``fl_ev_cards``/``fl_ev_value``; that is a declaration checked against the
runtime config, never a second source competing with it, because a joint-exact
plan cannot set the constant -- it reaches the engine through the observation.

That resume rule reads the local directory, which is the whole truth only while
the disk survives. A Spot instance that is RECREATED rather than restarted comes
back with an empty disk even though the previous attempt's positions are safely
in the object store, so the local-only scan would regenerate every one of them
and the create-only uploads would then collide. ``--existing-manifest`` closes
that gap: it names positions the caller already knows are durable elsewhere, and
they count as done exactly as a local file would. Without the flag the worker
behaves as it always has.

A T0 plan may additionally pin the fast T1 pair. Those weights are not a faster
route to the same label: the engine answers both T1 replies with a coarse
distilled model, and says so by reporting those two evaluators as
``learned_fast``. The worker therefore treats them exactly as it treats every
other pinned weight -- all four fields or none, digests recomputed before the
engine opens, and a provenance gate that demands the engine confirm it used the
coarse pair rather than quietly falling back to the full-precision one. A plan
without those fields runs precisely as it did before they existed.

A T1 or T0 plan, either seat, may pin the same kind of pair one street down.
``fast_t2_first_model``/``fast_t2_second_model`` and their digests replace both
T2 replies with distilled twins on exactly the kinds whose rollouts still have
both of them ahead: all four fields or none, T1 and T0 plans only, digests
recomputed before the engine opens, and a provenance gate that then expects
``learned_fast`` from both T2 evaluators. A (T2, first) plan plays one of the
two and would apply half the pair, so it is refused rather than half-honoured.

A T0 FIRST-seat plan may pin one more such pair. Acting first on the opening
street the opponent's own T0 second-seat reply is still ahead of the teacher and
opens every rollout, so that reply has a full-precision model of its own -- the
eighth and last learned evaluator -- and, optionally, a coarse distilled twin
answered through in its place. ``fast_t0_second_model`` and its digest are that
twin, validated exactly as the fast T1 pair is: both fields or neither, T0 plans
only, digest recomputed before the engine opens, and the same provenance gate,
which then expects ``learned_fast`` from ``t0_second_evaluator`` rather than
``learned``. The T0 SECOND seat has no T0 reply ahead of it at all, so a plan of
that kind carrying the pair is refused rather than silently ignored.

The joint-exact family covers both T3 seats. (T3, second) is the one kind whose
teacher has NO learned street model ahead of it but the T4 one: acting second at
T3 the opponent's T3 turn is already behind, so the engine's ``rollout_t3_second``
reaches the terminal through two T4 decisions and never consults
``t3_second_model``. That plan therefore does not pin it, and is refused if it
does. The distinction is not pedantry -- the engine reports an evaluator as
``learned`` when its weights were LOADED rather than when they were reached, so a
pinned-but-unreachable model would have the provenance gate confirming a
dependency the label does not have, and would make a later change to those
weights read as invalidating a corpus it cannot affect.

A plan may declare ``plan_kind: "t3_vs_fl"`` instead of the joint-exact path
above. That kind labels a different game: the hero at T3 against an opponent who
is already in Fantasyland, so there is no opponent decision left to model and no
learned evaluator on the path at all. It is scored by the ``fl_solver_regular``
binary rather than by the M3 engine, and it pins a different set of artifacts --
the solver binary, the FL EV config whose value the labels bake in, and the
roots file the shard indexes into. Those three replace the engine library, the
feature encoder and the five-to-eight model weights, which is why the required
field set is chosen by kind rather than shared: demanding an engine digest from
a plan that never opens the engine would be a gate that proves nothing.

Everything else is deliberately the same. Create-only one file per position,
``--existing-manifest`` counted exactly as a local file, stride splitting by
offset, the SHARD_DONE marker, and a provenance gate that reads what the solver
reported and refuses to write when it is not what the plan pinned. A plan
without ``plan_kind`` takes the joint-exact path and produces byte-identical
requests and position files to one written before this kind existed.

A T0 plan may also carry either or both of the two pruning-safety fields.
``prefilter_margin`` widens the root prefilter's keep boundary wherever the
coarse stage cannot separate the actions across it; ``audit_full_every`` makes
one position in every N pay for the single-stage answer as well, and the worker
stores the engine's audit verdict verbatim in that position's file. Both are
optional and independent, and a plan carrying neither produces the requests and
the position files it produced before they existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time
from typing import Any, Mapping

PLAN_SCHEMA = "hu_m31_label_gen_plan_v1"
POSITION_SCHEMA = "hu_m31_label_gen_position_v1"
DONE_SCHEMA = "hu_m31_label_gen_shard_done_v1"

# The T3-vs-Fantasyland kind. Its positions carry a different payload -- a
# candidate list scored against solved Fantasyland boards, not an action-key to
# engine-score map -- so they get a schema of their own rather than reusing the
# joint-exact one under a different meaning.
T3_VS_FL_KIND = "t3_vs_fl"
T3_VS_FL_POSITION_SCHEMA = "hu_m31_label_gen_t3_vs_fl_position_v1"

# The T2 kind. Same artifacts and same gate; it differs only in which
# street it labels and in carrying two sampled draw counts instead of one
# exhaustive one, because C(unseen, 3) at both T3 and T4 is what makes
# that tree explode.
T2_VS_FL_KIND = "t2_vs_fl"
T1_VS_FL_KIND = "t1_vs_fl"
T1_VS_FL_POSITION_SCHEMA = "hu_m31_label_gen_t1_vs_fl_position_v1"

# The T0 first-seat kind. One street deeper again, and the first whose fan is
# worth cutting: 232 openings each carry a whole line beneath them, and unlike
# T1 -- where the shared draw tree dominates and narrowing bought only 1.3x --
# every T0 opening leaves a different board, so its Fantasyland solving is its
# own. The narrowing model is therefore part of the label's identity too: a
# label that saw twenty of 232 openings is not the label that saw all of them.
T0_VS_FL_KIND = "t0_first_vs_fl"
T0_VS_FL_POSITION_SCHEMA = "hu_m31_label_gen_t0_first_vs_fl_position_v1"
T2_VS_FL_POSITION_SCHEMA = "hu_m31_label_gen_t2_vs_fl_position_v1"

# What a t3_vs_fl plan must pin. The three artifacts are the whole provenance of
# a label of this kind: the solver that computed it, the Fantasyland EV it was
# computed against, and the roots file the offsets index into. Each is a path
# and a digest, all-or-nothing, for the same reason every other pinned pair in
# this worker is.
T3_VS_FL_REQUIRED = (
    "schema", "plan_kind", "job_id", "street", "seat", "samples",
    "hero_deals", "root_policy",
    "fl_solver_binary", "fl_solver_binary_sha256",
    "fl_ev_config", "fl_ev_config_sha256", "fl_ev_cards", "fl_ev_value",
    "roots_file", "roots_file_sha256",
    "opponent_seed_base", "hero_deal_seed_base",
    "solver_version", "objective",
    "shards",
)

# The optional fast T1 pair, named once so the all-or-nothing check and the
# pinning below cannot drift apart. Both seats, because acting first at T1 both
# boards carry five cards where the second seat sees seven -- the same reason
# the full-precision pair is two models rather than one.
FAST_T1_FIELDS = (
    "fast_t1_first_model",
    "fast_t1_first_model_sha256",
    "fast_t1_second_model",
    "fast_t1_second_model_sha256",
)

# The optional fast T2 pair, same contract one street down. Both seats, for the
# same reason the full-precision pair is two models: acting first at T2 the
# opponent's board carries seven cards where the second seat sees nine.
#
# Accepted on T1 and T0 plans, either seat. Those are exactly the kinds whose
# rollouts still have BOTH T2 replies ahead of them -- the same condition the
# full-precision `learned_t2_first` pin is gated on. A (T2, first) plan plays
# only the second-seat reply, so half this pair would apply and half would be
# ignored, which the all-or-nothing rule below is there to prevent.
FAST_T2_FIELDS = (
    "fast_t2_first_model",
    "fast_t2_first_model_sha256",
    "fast_t2_second_model",
    "fast_t2_second_model_sha256",
)

# The optional fast T0 second-seat model, same contract one street up. One seat,
# not two: only a T0 FIRST-seat teacher has a T0 reply ahead of it, so there is
# no first-seat twin to pair it with. It is still a pair of FIELDS -- a path and
# its digest -- and still all-or-nothing, because a path without its digest is
# the same defect the fast T1 pair's check exists to prevent.
FAST_T0_SECOND_FIELDS = (
    "fast_t0_second_model",
    "fast_t0_second_model_sha256",
)

# The two pruning-safety mechanisms, each optional and each independent of the
# other. Named here so the T0-only check and the pass-through below cannot
# drift apart. Absent means today's behaviour exactly: a fixed-rank keep
# boundary and no audit.
PRUNING_SAFETY_FIELDS = ("prefilter_margin", "audit_full_every")

# The optional stage-two race. Stage two is where a T0 root's surviving actions
# are scored at full resolution, and uniformly: every survivor spends the whole
# evaluation batch even after the leader has beaten it decisively. A schedule
# spends the batch in cumulative instalments and drops candidates the leader has
# separated by more than `race_lcb_z` standard errors of their paired
# difference.
#
# All-or-nothing, and for a sharper reason than the model pairs above. A
# schedule with the z left at zero races and eliminates on no evidence at all;
# a z with no schedule is refused by the engine outright. So one field without
# the other is either a silent change of meaning or a stop on the first
# position, and neither is something a plan should be able to say by omission.
#
# T0-only, like the two pruning-safety fields: stage two exists only where there
# is a prefilter, and only the T0 evaluator has one.
RACE_FIELDS = ("race_schedule", "race_lcb_z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plan(path: pathlib.Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise SystemExit(f"unsupported plan schema: {plan.get('schema')!r}")
    # The kind decides which artifacts are provenance, so it decides which
    # fields are required. A plan without the key is the joint-exact kind and
    # takes exactly the path it took before this branch existed.
    kind = plan.get("plan_kind")
    vs_fl_kinds = (T3_VS_FL_KIND, T2_VS_FL_KIND, T1_VS_FL_KIND, T0_VS_FL_KIND)
    if kind is not None and kind not in vs_fl_kinds:
        raise SystemExit(
            f"this worker version generates plan_kind one of "
            f"{list(vs_fl_kinds)!r} or an absent plan_kind; the plan asks for "
            f"{kind!r}"
        )
    if kind in vs_fl_kinds:
        return load_t3_vs_fl_plan(plan)
    required = {
        "schema", "job_id", "street", "seat", "samples", "seeds_per_position",
        "hand_seed_base", "behavior_seed_offset", "eval_seed_base",
        "engine_library", "engine_library_sha256",
        "feature_encoder_library", "feature_encoder_library_sha256",
        "t4_model", "t4_model_sha256",
        "shards",
    }
    # The T3 second-seat model is on every joint-exact path except one. Acting
    # SECOND at T3 the opponent's T3 turn is already behind the teacher, so
    # `rollout_t3_second` reaches the terminal through two T4 decisions and
    # never consults it. Requiring it there would pin a model the label does
    # not depend on: the engine reports an evaluator as `learned` when the
    # weights were LOADED, not when they were used, so the provenance gate
    # would confirm a dependency that does not exist -- and a later change to
    # those weights would read as invalidating a corpus it cannot affect.
    if (plan.get("street"), plan.get("seat")) != ("T3", "second"):
        required |= {"t3_second_model", "t3_second_model_sha256"}
    missing = required - set(plan)
    if missing:
        raise SystemExit(f"plan is missing fields: {sorted(missing)}")
    supported = {("T3", "second"), ("T3", "first"),
                 ("T2", "second"), ("T2", "first"),
                 ("T1", "second"), ("T1", "first"), ("T0", "second"),
                 ("T0", "first")}
    if (plan["street"], plan["seat"]) not in supported:
        raise SystemExit(
            f"this worker version generates {sorted(supported)}; the plan asks "
            f"for {plan['street']}/{plan['seat']}"
        )
    if (plan["street"], plan["seat"]) == ("T3", "second"):
        # Refused rather than ignored, for the reason above and by the same
        # rule every other misplaced field in this worker is refused by: a plan
        # that pins weights this kind never opens reads as a label that depends
        # on them.
        stray_t3_second = sorted(
            field for field in ("t3_second_model", "t3_second_model_sha256")
            if field in plan
        )
        if stray_t3_second:
            raise SystemExit(
                f"{stray_t3_second} are wired for the joint-exact kinds whose "
                "teacher still has the opponent's T3 second-seat reply ahead "
                "of it; this plan is T3/second, whose rollouts reach the "
                "terminal through two T4 decisions and never consult that "
                "model"
            )
    if plan["street"] == "T2":
        for field in ("t3_first_model", "t3_first_model_sha256"):
            if field not in plan:
                raise SystemExit(
                    f"T2 plans must pin {field}: the T2 teacher plays the "
                    "opponent's T3 first-seat reply through that model"
                )
    if (plan["street"], plan["seat"]) == ("T2", "first"):
        for field in ("t2_second_model", "t2_second_model_sha256"):
            if field not in plan:
                raise SystemExit(
                    f"T2 first-seat plans must pin {field}: the teacher plays "
                    "the opponent's T2 second-seat reply through that model"
                )
    if plan["street"] == "T1":
        # From a T1 second-seat decision the teacher still has to play both
        # seats' T2 replies and the opponent's T3 first-seat reply, so every
        # learned evaluator in the ladder below T1 is on the path and all five
        # have to be pinned before a single label is written.
        for field in ("t3_first_model", "t3_first_model_sha256",
                      "t2_second_model", "t2_second_model_sha256",
                      "t2_first_model", "t2_first_model_sha256"):
            if field not in plan:
                raise SystemExit(
                    f"T1 {plan['seat']}-seat plans must pin {field}: the T1 "
                    "teacher plays the rest of the hand through all five "
                    "learned evaluators below this street"
                )
    if (plan["street"], plan["seat"]) == ("T1", "first"):
        # Acting first on T1 the opponent's own T1 second-seat reply is still
        # ahead of the teacher and opens every rollout, so a sixth learned
        # evaluator joins the five. The second seat never needs it: by the time
        # it acts the opponent's T1 turn is already behind it.
        for field in ("t1_second_model", "t1_second_model_sha256"):
            if field not in plan:
                raise SystemExit(
                    f"T1 first-seat plans must pin {field}: the teacher plays "
                    "the opponent's T1 second-seat reply through that model"
                )
    if plan["street"] == "T0":
        # The opening street. Acting second the teacher plays the whole hand
        # out through every learned evaluator below it -- both T1 decisions,
        # both T2 decisions, the opponent's T3 first-seat reply and the T4
        # terminal -- so all seven have to be pinned before a single label is
        # written. The seventh, t1_first_model, is new here: no other kind this
        # worker supports has a T1 first-seat decision ahead of the teacher.
        for field in ("t3_first_model", "t3_first_model_sha256",
                      "t2_second_model", "t2_second_model_sha256",
                      "t2_first_model", "t2_first_model_sha256",
                      "t1_second_model", "t1_second_model_sha256",
                      "t1_first_model", "t1_first_model_sha256"):
            if field not in plan:
                raise SystemExit(
                    f"T0 {plan['seat']}-seat plans must pin {field}: the T0 "
                    "teacher plays the rest of the hand through all seven "
                    "learned evaluators below this street"
                )
        if plan["seat"] == "first":
            # Acting FIRST on the opening street nothing at all has happened
            # yet, so the opponent's T0 second-seat reply is still ahead of the
            # teacher and opens every rollout: the eighth learned evaluator, and
            # the last one there is. The second seat never needs it -- by the
            # time it acts the opponent's T0 turn is already behind it -- which
            # is the same asymmetry every street below shows between its seats.
            for field in ("t0_second_model", "t0_second_model_sha256"):
                if field not in plan:
                    raise SystemExit(
                        f"T0 first-seat plans must pin {field}: the teacher "
                        "plays the opponent's T0 second-seat reply through that "
                        "model, which makes eight learned evaluators on this "
                        "kind's path"
                    )
        # The T0 root fan is ~232 actions where every other street this worker
        # labels faces tens. Scoring all of them at the plan's full particle
        # count is not something a plan should get by omission, so the two-stage
        # schedule is REQUIRED here rather than defaulted: a plan that wants the
        # single-stage exact-comparison path has to say so somewhere other than
        # this worker.
        for field in ("prefilter_samples", "prefilter_keep"):
            value = plan.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SystemExit(
                    f"T0 plans must set {field} to a positive integer; got "
                    f"{value!r}. The 232-action root fan makes single-stage "
                    "evaluation unaffordable, so both stages of the schedule "
                    "are chosen by the plan, never defaulted"
                )
        # The two pruning-safety mechanisms, in contrast, ARE optional: a plan
        # that omits them runs exactly as T0 plans have run. Validated when
        # present so a malformed one fails here rather than on the first
        # position, and independently of one another because widening the beam
        # and auditing the result answer different questions.
        if "prefilter_margin" in plan:
            margin = plan["prefilter_margin"]
            if isinstance(margin, bool) or not isinstance(margin, (int, float)):
                raise SystemExit(
                    f"prefilter_margin must be a real number; got {margin!r}"
                )
            # NaN and the infinities are the ones worth naming. `json.loads`
            # accepts all three by default, NaN compares false against every
            # gap, and a run whose plan asked for a margin the engine could
            # never apply would report a margin it never applied.
            if not math.isfinite(margin) or margin < 0:
                raise SystemExit(
                    f"prefilter_margin must be a finite non-negative number; "
                    f"got {margin!r}. It is a stage-one score difference, so "
                    "neither a negative one nor a NaN has any reading"
                )
        if "audit_full_every" in plan:
            cadence = plan["audit_full_every"]
            if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 0:
                raise SystemExit(
                    f"audit_full_every must be a non-negative integer; got "
                    f"{cadence!r}"
                )
        # The stage-two race, optional in the same way and validated here for
        # the same reason: a malformed schedule should stop the plan rather
        # than the first position, and every rule below is one the engine also
        # enforces. Restating them is not duplication -- the engine's copy
        # protects the label, this one protects the fleet's morning.
        racing = sorted(field for field in RACE_FIELDS if field in plan)
        if racing and len(racing) != len(RACE_FIELDS):
            raise SystemExit(
                f"race fields are all-or-nothing; this plan carries {racing} "
                f"but is missing {sorted(set(RACE_FIELDS) - set(racing))}. A "
                "schedule with no threshold eliminates on no evidence, and a "
                "threshold with no schedule is refused by the engine"
            )
        if racing:
            schedule = plan["race_schedule"]
            if isinstance(schedule, (str, bytes)) or not isinstance(schedule, list):
                raise SystemExit(
                    f"race_schedule must be a list of cumulative particle "
                    f"counts; got {schedule!r}"
                )
            if not schedule:
                raise SystemExit(
                    "race_schedule is empty, which is the uniform stage two "
                    "wearing the race's name; omit both fields instead"
                )
            for checkpoint in schedule:
                if isinstance(checkpoint, bool) or not isinstance(checkpoint, int):
                    raise SystemExit(
                        f"race_schedule checkpoints must be integers; got "
                        f"{checkpoint!r}"
                    )
            if schedule[0] <= 0 or any(later <= earlier for earlier, later
                                       in zip(schedule, schedule[1:])):
                raise SystemExit(
                    f"race_schedule must be strictly increasing positive "
                    f"cumulative particle counts; got {schedule!r}"
                )
            # The selected action has to have been measured at the resolution
            # the label claims it was. A schedule ending short of `samples`
            # would be a cheaper evaluation under the full one's name.
            if schedule[-1] != plan["samples"]:
                raise SystemExit(
                    f"the last race_schedule checkpoint must equal samples "
                    f"({plan['samples']}); the schedule ends at {schedule[-1]}"
                )
            z = plan["race_lcb_z"]
            if isinstance(z, bool) or not isinstance(z, (int, float)):
                raise SystemExit(f"race_lcb_z must be a real number; got {z!r}")
            if not math.isfinite(z) or z < 0:
                raise SystemExit(
                    f"race_lcb_z must be a finite non-negative number of "
                    f"standard errors; got {z!r}"
                )
    else:
        # Only the T0 evaluator has a prefilter, so only a T0 plan has anywhere
        # to put either of these. Ignoring them silently on another kind would
        # produce unwidened, unaudited labels under a plan that reads as
        # carrying both safeguards -- the same defect the fast-pair check below
        # exists to prevent.
        stray = sorted(field for field in PRUNING_SAFETY_FIELDS + RACE_FIELDS
                       if field in plan)
        if stray:
            raise SystemExit(
                f"{stray} are wired for T0 plans only; this plan is "
                f"{plan['street']}/{plan['seat']}, which would ignore them"
            )
    # The fast T1 pair is optional even on T0, but it is not optional by halves.
    # Pinning one seat's coarse model and not the other's would answer one T1
    # reply from a distilled network and the other exactly, which is a label
    # nobody chose and which nothing downstream could tell from either pure run.
    # A path without its digest is the same defect one level down.
    present = sorted(field for field in FAST_T1_FIELDS if field in plan)
    if present and len(present) != len(FAST_T1_FIELDS):
        raise SystemExit(
            f"fast T1 model fields are all-or-nothing; this plan carries "
            f"{present} but is missing "
            f"{sorted(set(FAST_T1_FIELDS) - set(present))}. A partial set "
            "would run one T1 seat coarse and the other exact"
        )
    if present and plan["street"] != "T0":
        # Only the T0 teacher plays both T1 seats, so only a T0 plan has
        # anywhere to put these. Ignoring them silently on another kind would
        # produce a full-precision label under a plan that reads as coarse.
        raise SystemExit(
            f"fast T1 model fields are wired for T0 plans only; this plan is "
            f"{plan['street']}/{plan['seat']}, which would ignore them"
        )
    # The fast T2 pair, held to the same standard for the same reasons, and with
    # a wider gate: a T1 teacher plays both T2 replies just as a T0 teacher does.
    t2_fast = sorted(field for field in FAST_T2_FIELDS if field in plan)
    if t2_fast and len(t2_fast) != len(FAST_T2_FIELDS):
        raise SystemExit(
            f"fast T2 model fields are all-or-nothing; this plan carries "
            f"{t2_fast} but is missing "
            f"{sorted(set(FAST_T2_FIELDS) - set(t2_fast))}. A partial set "
            "would run one T2 seat coarse and the other exact"
        )
    if t2_fast and plan["street"] not in ("T1", "T0"):
        # A (T2, first) plan plays the second-seat reply but not the first-seat
        # one, so the pair cannot apply whole; every other kind plays neither.
        # Ignoring them silently would produce a full-precision label under a
        # plan that reads as coarse.
        raise SystemExit(
            f"fast T2 model fields are wired for T1 and T0 plans only; this "
            f"plan is {plan['street']}/{plan['seat']}, whose rollouts do not "
            "have both T2 replies ahead of them"
        )
    # The fast T0 second-seat model, held to the same standard for the same
    # reasons. A path without its digest would pin the label to weights nobody
    # measured; a digest without its path would read as coarse and run exact.
    t0_fast = sorted(field for field in FAST_T0_SECOND_FIELDS if field in plan)
    if t0_fast and len(t0_fast) != len(FAST_T0_SECOND_FIELDS):
        raise SystemExit(
            f"fast T0 second-seat model fields are all-or-nothing; this plan "
            f"carries {t0_fast} but is missing "
            f"{sorted(set(FAST_T0_SECOND_FIELDS) - set(t0_fast))}. A path "
            "without its digest pins weights nobody measured, and a digest "
            "without its path reads as coarse and runs exact"
        )
    if t0_fast and plan["street"] != "T0":
        raise SystemExit(
            f"fast T0 second-seat model fields are wired for T0 plans only; "
            f"this plan is {plan['street']}/{plan['seat']}, which would ignore "
            "them"
        )
    # The Fantasyland constant the labels bake in. Optional on the joint-exact
    # kinds, because every plan written before it existed omits it and must go
    # on meaning exactly what it meant. Where it IS stated it is checked against
    # the runtime rather than trusted, and it makes the corpus-mixing guard say
    # what the corpus should have been rather than only that two shards
    # disagree.
    #
    # Note what a joint-exact plan canNOT do: it cannot SET the constant. The
    # value reaches the engine through the observation's scoring context, which
    # comes from the one config `hu_infoset` reads. Pinning it here is therefore
    # a declaration checked against that single source of truth, never a second
    # one competing with it -- the whole point of the v3 consolidation.
    fl_ev_fields = sorted(
        field for field in ("fl_ev_cards", "fl_ev_value") if field in plan
    )
    if fl_ev_fields and len(fl_ev_fields) != 2:
        raise SystemExit(
            f"fl_ev fields are all-or-nothing; this plan carries {fl_ev_fields} "
            f"but is missing {sorted({'fl_ev_cards', 'fl_ev_value'} - set(fl_ev_fields))}. "
            "A value with no card count does not say which entry it prices"
        )
    if fl_ev_fields:
        cards = plan["fl_ev_cards"]
        if isinstance(cards, bool) or not isinstance(cards, int) or cards <= 0:
            raise SystemExit(
                f"fl_ev_cards must be a positive integer; got {cards!r}"
            )
        value = plan["fl_ev_value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(
                f"fl_ev_value must be a real number; got {value!r}"
            )
        if not math.isfinite(value):
            raise SystemExit(
                f"fl_ev_value must be finite; got {value!r}"
            )
    if t0_fast and plan["seat"] != "first":
        # Narrower than the T1 pair's gate, because the thing it displaces is
        # narrower: a T0 SECOND-seat teacher has no T0 reply ahead of it, so
        # there is nothing for a coarse T0 second-seat model to answer. Left
        # unchecked, such a plan would read as coarse and produce the ordinary
        # seven-evaluator label.
        raise SystemExit(
            f"fast T0 second-seat model fields are wired for the T0 FIRST seat "
            f"only; this plan is {plan['street']}/{plan['seat']}, whose teacher "
            "has no T0 reply ahead of it to answer coarsely"
        )
    return plan


def load_t3_vs_fl_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate a T3-vs-Fantasyland plan.

    Deliberately strict about the fields the joint-exact kind pins and this one
    does not: an engine digest on a plan that never opens the engine is a gate
    that proves nothing, and leaving it accepted would let a copy-pasted plan
    look validated while pinning the wrong things.
    """

    kind = plan["plan_kind"]
    required = set(T3_VS_FL_REQUIRED)
    if kind == T2_VS_FL_KIND:
        # T2 samples both downstream draws instead of enumerating one.
        required = (required - {"hero_deals"}) | {"t3_draws", "t4_draws"}
    if kind == T1_VS_FL_KIND:
        # T1 samples three draw levels, and -- unlike every kind before it --
        # its labels depend on two learned continuation policies. Those models
        # are part of the label's identity exactly as opponent_mode is, so the
        # plan must pin them and the gate below must see the solver echo them.
        required = (required - {"hero_deals"}) | {
            "t2_draws", "t3_draws", "t4_draws",
            "t2_model", "t2_model_sha256",
            "t3_model", "t3_model_sha256",
            "engine_features_rev",
        }
    if kind == T0_VS_FL_KIND:
        # T0 samples four draw levels and continues through all three rankers.
        # `narrow_keep` and `narrow_model` are optional together: a plan may
        # evaluate the whole fan, but it may not narrow without saying by whose
        # ranking, because that decides which openings a label ever saw.
        required = (required - {"hero_deals"}) | {
            "t1_draws", "t2_draws", "t3_draws", "t4_draws",
            "t1_model", "t1_model_sha256",
            "t2_model", "t2_model_sha256",
            "t3_model", "t3_model_sha256",
            "engine_features_rev",
        }
        if "narrow_keep" in plan or "narrow_model" in plan:
            required |= {"narrow_keep", "narrow_model", "narrow_model_sha256"}
        # The pre-solved Fantasyland pool, when the run draws its opponents from
        # one instead of solving at every leaf. It decides which opponents were
        # averaged, so it is part of the label's identity exactly as the
        # continuation models are, and it is pinned the same way.
        if "fl_library" in plan or "fl_library_sha256" in plan:
            required |= {"fl_library", "fl_library_sha256"}
    missing = required - set(plan)
    if missing:
        raise SystemExit(f"{kind} plan is missing fields: {sorted(missing)}")
    forbidden = sorted(
        field
        for field in (
            "engine_library", "engine_library_sha256",
            "feature_encoder_library", "feature_encoder_library_sha256",
            "t4_model", "t3_second_model", "t3_first_model",
            "t2_second_model", "t2_first_model",
            "t1_second_model", "t1_first_model", "t0_second_model",
        )
        if field in plan
    )
    if forbidden:
        raise SystemExit(
            f"t3_vs_fl plan pins joint-exact artifacts it never uses: {forbidden}; "
            "labels of this kind are produced by the solver binary alone"
        )
    expected_street = {
        T3_VS_FL_KIND: "T3", T2_VS_FL_KIND: "T2", T1_VS_FL_KIND: "T1",
        T0_VS_FL_KIND: "T0",
    }[kind]
    if (plan["street"], plan["seat"]) != (expected_street, "first"):
        raise SystemExit(
            f"{kind} plans label the hero own {expected_street} decision, so "
            f"street/seat must be {expected_street}/first; the plan asks for "
            f"{plan['street']}/{plan['seat']}"
        )
    if kind == T3_VS_FL_KIND and plan["hero_deals"] != 0:
        raise SystemExit(
            "t3_vs_fl plans must set hero_deals to 0, which enumerates every "
            "hand the hero can be dealt; a sampled outer expectation was "
            "measured to be both slower to converge and no cheaper"
        )
    if kind == T2_VS_FL_KIND:
        for field in ("t3_draws", "t4_draws"):
            if not isinstance(plan[field], int) or plan[field] <= 0:
                raise SystemExit(f"t2_vs_fl {field} must be a positive integer")
    if kind == T1_VS_FL_KIND:
        for field in ("t2_draws", "t3_draws", "t4_draws"):
            if not isinstance(plan[field], int) or plan[field] <= 0:
                raise SystemExit(f"t1_vs_fl {field} must be a positive integer")
        for field in ("t2_model_sha256", "t3_model_sha256"):
            value = plan[field]
            if not isinstance(value, str) or len(value) != 64:
                raise SystemExit(
                    f"t1_vs_fl {field} must be a 64-character sha256; the plan "
                    f"pins {value!r}"
                )
        if plan["t2_model_sha256"] == plan["t3_model_sha256"]:
            raise SystemExit(
                "t1_vs_fl pins the same digest for both continuation models; "
                "the T2 and T3 rankers are different artifacts and a plan that "
                "names one twice would silently continue both streets with it"
            )
        # The T1 teacher only builds adaptive-opponent labels, and its
        # continuation models were themselves distilled from adaptive labels.
        # A static_v1 T1 plan would be rejected by the runtime gate anyway;
        # refusing it here names the reason instead of failing mid-shard.
        if plan.get("opponent_mode", "static_v1") != "adaptive_v1":
            raise SystemExit(
                "t1_vs_fl plans must set opponent_mode to adaptive_v1; the "
                "teacher builds no other kind and its continuation models were "
                "distilled from adaptive labels"
            )
    if kind == T0_VS_FL_KIND:
        for field in ("t1_draws", "t2_draws", "t3_draws", "t4_draws"):
            if not isinstance(plan[field], int) or plan[field] <= 0:
                raise SystemExit(f"t0_first_vs_fl {field} must be a positive integer")
        digests = {}
        for field in ("t1_model_sha256", "t2_model_sha256", "t3_model_sha256"):
            value = plan[field]
            if not isinstance(value, str) or len(value) != 64:
                raise SystemExit(
                    f"t0_first_vs_fl {field} must be a 64-character sha256; the "
                    f"plan pins {value!r}"
                )
            digests[field] = value
        if len(set(digests.values())) != 3:
            raise SystemExit(
                "t0_first_vs_fl pins the same digest for two continuation "
                "models; the T1, T2 and T3 rankers are different artifacts and "
                "a plan naming one twice would silently continue two streets "
                "with it"
            )
        if plan.get("opponent_mode", "static_v1") != "adaptive_v1":
            raise SystemExit(
                "t0_first_vs_fl plans must set opponent_mode to adaptive_v1; "
                "the teacher builds no other kind"
            )
        keep = plan.get("narrow_keep")
        if keep is not None:
            if not isinstance(keep, int) or not 0 < keep < 232:
                raise SystemExit(
                    "t0_first_vs_fl narrow_keep must be an integer in 1..231; "
                    f"the plan asks for {keep!r}. A plan that means to score "
                    "the whole fan omits the field rather than naming 232."
                )
            value = plan["narrow_model_sha256"]
            if not isinstance(value, str) or len(value) != 64:
                raise SystemExit(
                    "t0_first_vs_fl narrow_model_sha256 must be a 64-character "
                    f"sha256; the plan pins {value!r}"
                )
    if not isinstance(plan["samples"], int) or plan["samples"] <= 0:
        raise SystemExit("t3_vs_fl samples must be a positive integer")
    # Absent means static_v1: that is what every plan written before the
    # adaptive opponent existed in fact was, and those labels must stay
    # replayable.
    plan.setdefault("opponent_mode", "static_v1")
    if plan["opponent_mode"] not in ("static_v1", "adaptive_v1"):
        raise SystemExit(
            f"{kind} opponent_mode must be static_v1 or adaptive_v1; the "
            f"plan asks for {plan['opponent_mode']!r}"
        )
    if plan["fl_ev_cards"] != 14:
        raise SystemExit(
            "regular-rule Fantasyland is always 14 cards; "
            f"the plan pins fl_ev_cards={plan['fl_ev_cards']!r}"
        )
    for entry in plan["shards"]:
        for field in ("shard_id", "start", "count"):
            if field not in entry:
                raise SystemExit(f"t3_vs_fl shard entry is missing {field}")
    return plan


def pinned(runtime: pathlib.Path, relative: str, expected: str, label: str) -> pathlib.Path:
    path = (runtime / relative).resolve()
    if not str(path).startswith(str(runtime.resolve())):
        raise SystemExit(f"{label} escapes the runtime root: {relative}")
    if not path.is_file():
        raise SystemExit(f"{label} is missing at {path}")
    actual = sha256_of(path)
    if actual != expected:
        raise SystemExit(
            f"{label} digest {actual} does not match the plan's {expected}; "
            "refusing to generate labels whose provenance is not the pinned one"
        )
    return path


def load_existing_manifest(path_text: str | None) -> frozenset[str]:
    """Names of position files a previous attempt already published elsewhere.

    One filename per line, blank lines ignored. A missing manifest is an error
    rather than an empty set: the caller asked the worker to trust a list, and
    silently trusting nothing is how a recreated fleet regenerates a whole
    shard.
    """

    if path_text is None:
        return frozenset()
    path = pathlib.Path(path_text)
    if not path.is_file():
        raise SystemExit(f"existing manifest is missing at {path}")
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        if "/" in name or "\\" in name:
            raise SystemExit(
                f"existing manifest entry {name!r} is a path; entries are bare "
                "file names such as position_00000123.json"
            )
        names.add(name)
    return frozenset(names)


def shard_entry(plan: Mapping[str, Any], shard_id: str) -> Mapping[str, Any]:
    for entry in plan["shards"]:
        if entry.get("shard_id") == shard_id:
            return entry
    raise SystemExit(f"shard {shard_id!r} is not in the plan")


def stride_offsets(shard: Mapping[str, Any], args: argparse.Namespace) -> list[int]:
    if not 0 <= args.stride_index < args.stride_count:
        raise SystemExit("stride index must lie inside the stride count")
    offsets = range(shard["start"], shard["start"] + shard["count"])
    return [o for o in offsets
            if (o - shard["start"]) % args.stride_count == args.stride_index]


def run_t3_vs_fl(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    plan_sha: str,
    shard: Mapping[str, Any],
    out: pathlib.Path,
    existing_manifest: frozenset[str],
) -> int:
    """Generate one shard of T3-vs-Fantasyland labels.

    The solver is a batch tool -- most of its cost is solving the shared pool of
    opponent Fantasyland hands -- so positions are computed a chunk at a time
    and then split into the same create-only one-file-per-position layout every
    other kind writes. Chunking is what resume granularity costs: a preempted
    instance loses at most the chunk in flight, not the shard.
    """

    import subprocess
    import tempfile

    runtime = pathlib.Path(args.runtime_root).resolve()
    solver = pinned(runtime, plan["fl_solver_binary"],
                    plan["fl_solver_binary_sha256"], "FL solver binary")
    fl_ev_config = pinned(runtime, plan["fl_ev_config"],
                          plan["fl_ev_config_sha256"], "FL EV config")
    roots_path = pinned(runtime, plan["roots_file"],
                        plan["roots_file_sha256"], "roots file")

    # The value the labels bake in has to be the one the pinned config holds,
    # not merely the one the plan claims. Read it here rather than trusting the
    # plan's copy of it.
    fl_ev_payload = json.loads(fl_ev_config.read_text(encoding="utf-8"))
    on_disk = float(fl_ev_payload["fl_ev"][str(plan["fl_ev_cards"])])
    if abs(on_disk - float(plan["fl_ev_value"])) > 1e-12:
        raise SystemExit(
            f"pinned FL EV config holds {on_disk} for {plan['fl_ev_cards']} "
            f"cards; the plan states {plan['fl_ev_value']}. Refusing to generate "
            "labels whose Fantasyland value is not the pinned one"
        )

    kind = plan["plan_kind"]
    position_schema = {
        T3_VS_FL_KIND: T3_VS_FL_POSITION_SCHEMA,
        T2_VS_FL_KIND: T2_VS_FL_POSITION_SCHEMA,
        T1_VS_FL_KIND: T1_VS_FL_POSITION_SCHEMA,
        T0_VS_FL_KIND: T0_VS_FL_POSITION_SCHEMA,
    }[kind]
    # Resolved through the same pinning helper as every other package member,
    # so a continuation model whose bytes drifted fails here rather than after
    # a shard of labels has been written against it.
    t1_t2_model = t1_t3_model = None
    t0_t1_model = t0_narrow_model = None
    if kind in (T1_VS_FL_KIND, T0_VS_FL_KIND):
        t1_t2_model = pinned(
            runtime, plan["t2_model"], plan["t2_model_sha256"], "t2 continuation model")
        t1_t3_model = pinned(
            runtime, plan["t3_model"], plan["t3_model_sha256"], "t3 continuation model")
    t0_library = None
    if kind == T0_VS_FL_KIND:
        t0_t1_model = pinned(
            runtime, plan["t1_model"], plan["t1_model_sha256"], "t1 continuation model")
        if plan.get("narrow_keep"):
            t0_narrow_model = pinned(
                runtime, plan["narrow_model"], plan["narrow_model_sha256"],
                "t0 narrowing model")
        if plan.get("fl_library"):
            t0_library = pinned(
                runtime, plan["fl_library"], plan["fl_library_sha256"],
                "pre-solved Fantasyland pool")
    roots = [json.loads(line) for line in
             roots_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_index = {int(record["root_index"]): record for record in roots}

    def position_path(offset: int) -> pathlib.Path:
        return out / f"position_{offset:08d}.json"

    def already_done(offset: int) -> bool:
        path = position_path(offset)
        if path.name in existing_manifest:
            return True
        if not path.is_file():
            return False
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            path.unlink()
            return False
        return record.get("schema") == position_schema

    mine = stride_offsets(shard, args)
    for offset in mine:
        if offset not in by_index:
            raise SystemExit(
                f"the pinned roots file has no root_index {offset}; the plan's "
                "shard math and the roots file disagree"
            )
    remaining = [o for o in mine if not already_done(o)]
    print(f"shard {args.shard_id} stride {args.stride_index}/{args.stride_count}: "
          f"{len(remaining)}/{len(mine)} positions to generate", flush=True)

    began = time.perf_counter()
    produced = 0
    checked = False
    chunk_size = max(1, args.solver_chunk)
    for start in range(0, len(remaining), chunk_size):
        if args.max_new_positions >= 0 and produced >= args.max_new_positions:
            break
        chunk = remaining[start:start + chunk_size]
        with tempfile.TemporaryDirectory() as scratch:
            scratch_root = pathlib.Path(scratch)
            chunk_roots = scratch_root / "roots.jsonl"
            with chunk_roots.open("w", encoding="utf-8") as stream:
                for offset in chunk:
                    stream.write(json.dumps(by_index[offset]) + "\n")
            chunk_out = scratch_root / "labels.jsonl"
            subcommand = {
                T3_VS_FL_KIND: "label-t3",
                T2_VS_FL_KIND: "label-t2",
                T1_VS_FL_KIND: "label-t1",
                T0_VS_FL_KIND: "label-t0",
            }[kind]
            command = [
                str(solver), subcommand,
                "--roots-file", str(chunk_roots),
                "--limit", str(len(chunk)),
                "--samples", str(plan["samples"]),
                "--opponent-seed", str(plan["opponent_seed_base"]),
                "--hero-deal-seed", str(plan["hero_deal_seed_base"]),
                "--root-policy", str(plan["root_policy"]),
                "--fl-ev-config", str(fl_ev_config),
                "--threads", str(args.solver_threads),
                "--out", str(chunk_out),
            ]
            if kind == T3_VS_FL_KIND:
                command += ["--hero-deals", str(plan["hero_deals"])]
            else:
                command += [
                    "--t3-draws", str(plan["t3_draws"]),
                    "--t4-draws", str(plan["t4_draws"]),
                ]
            if kind in (T1_VS_FL_KIND, T0_VS_FL_KIND):
                # Continuation models travel in the package and are handed to
                # the solver by digest, so the solver refuses an image that is
                # not the one this plan names.
                command += [
                    "--t2-draws", str(plan["t2_draws"]),
                    "--t2-model", str(t1_t2_model),
                    "--t2-model-sha256", str(plan["t2_model_sha256"]),
                    "--t3-model", str(t1_t3_model),
                    "--t3-model-sha256", str(plan["t3_model_sha256"]),
                ]
            if kind == T0_VS_FL_KIND:
                command += [
                    "--t1-draws", str(plan["t1_draws"]),
                    "--t1-model", str(t0_t1_model),
                    "--t1-model-sha256", str(plan["t1_model_sha256"]),
                ]
                if t0_narrow_model is not None:
                    command += [
                        "--narrow-keep", str(plan["narrow_keep"]),
                        "--narrow-model", str(t0_narrow_model),
                        "--narrow-model-sha256", str(plan["narrow_model_sha256"]),
                    ]
                if t0_library is not None:
                    command += [
                        "--fl-library", str(t0_library),
                        "--fl-library-sha256", str(plan["fl_library_sha256"]),
                    ]
            if plan["opponent_mode"] == "adaptive_v1":
                # The Fantasyland side best-responds to the hero's finished
                # board. A label built the other way is a different quantity,
                # which is why the mode is pinned and gated rather than assumed.
                command += ["--opponent-mode", "adaptive"]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                raise SystemExit(
                    "FL solver failed: "
                    + (completed.stderr.strip() or completed.stdout.strip())[:800]
                )
            labelled = [json.loads(line) for line in
                        chunk_out.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
        if len(labelled) != len(chunk):
            raise SystemExit(
                f"solver returned {len(labelled)} labels for {len(chunk)} roots"
            )
        for record in labelled:
            provenance = record.get("provenance", {})
            if not checked:
                # The provenance gate. Same standard as the engine kinds: what
                # the tool reports about itself has to be what the plan pinned,
                # in both directions, before a single position is written.
                expectations = {
                    "solver_version": plan["solver_version"],
                    "objective": plan["objective"],
                    "fl_ev_value": plan["fl_ev_value"],
                    "fl_ev_cards": plan["fl_ev_cards"],
                    "behavior_policy": plan["root_policy"],
                }
                for field, expected in expectations.items():
                    actual = provenance.get(field)
                    if actual != expected:
                        raise SystemExit(
                            f"solver reports {field}={actual!r}, expected "
                            f"{expected!r}; refusing to write labels whose "
                            "provenance is not what the plan pinned"
                        )
                if record.get("opponent_mode") != plan["opponent_mode"]:
                    raise SystemExit(
                        f"solver reports opponent_mode="
                        f"{record.get('opponent_mode')!r}, the plan pinned "
                        f"{plan['opponent_mode']!r}"
                    )
                if kind == T3_VS_FL_KIND and record.get("deal_mode") != "exhaustive":
                    raise SystemExit(
                        f"solver reports deal_mode={record.get('deal_mode')!r}, "
                        "expected 'exhaustive'"
                    )
                if kind == T2_VS_FL_KIND:
                    for field in ("t3_draws", "t4_draws"):
                        if record.get(field) != plan[field]:
                            raise SystemExit(
                                f"solver used {field}={record.get(field)}, the "
                                f"plan pinned {plan[field]}"
                            )
                if kind == T1_VS_FL_KIND:
                    for field in ("t2_draws", "t3_draws", "t4_draws"):
                        if record.get(field) != plan[field]:
                            raise SystemExit(
                                f"solver used {field}={record.get(field)}, the "
                                f"plan pinned {plan[field]}"
                            )
                    # The continuation policies are part of what the label MEANS.
                    # Relabelling with a different pair of rankers produces a
                    # different quantity, exactly as a different opponent_mode
                    # does, so the solver has to echo both digests and the engine
                    # generation that computed their features.
                    continuation = record.get("continuation") or {}
                    for field, expected in (
                        ("t2_model_sha256", plan["t2_model_sha256"]),
                        ("t3_model_sha256", plan["t3_model_sha256"]),
                        ("engine_features_rev", plan["engine_features_rev"]),
                    ):
                        actual = continuation.get(field)
                        if actual != expected:
                            raise SystemExit(
                                f"solver reports continuation {field}={actual!r}, "
                                f"the plan pinned {expected!r}; refusing to write "
                                "labels whose continuation policy is not the one "
                                "the plan names"
                            )
                    if continuation.get("mode") != "model_guided_v1":
                        raise SystemExit(
                            f"solver reports continuation mode="
                            f"{continuation.get('mode')!r}, expected "
                            "'model_guided_v1'"
                        )
                if record.get("opponent_samples") != plan["samples"]:
                    raise SystemExit(
                        f"solver used {record.get('opponent_samples')} opponent "
                        f"samples, the plan pinned {plan['samples']}"
                    )
                checked = True
            offset = int(record["root_index"])
            payload = {
                "schema": position_schema,
                "plan_sha256": plan_sha,
                "offset": offset,
                "samples": plan["samples"],
                "label": record,
            }
            try:
                with position_path(offset).open("xb") as stream:
                    stream.write(canonical_bytes(payload))
                    stream.flush()
            except FileExistsError:
                # A sibling stride or a resumed attempt got there first; the
                # create-only write is the arbiter, exactly as elsewhere.
                continue
            produced += 1
            if produced % args.report_every == 0:
                rate = produced / (time.perf_counter() - began)
                print(f"  {produced}/{len(remaining)} new  {rate:.2f} pos/s",
                      flush=True)

    offsets = range(shard["start"], shard["start"] + shard["count"])
    done = all(already_done(o) for o in offsets)
    if done:
        marker = out / "SHARD_DONE.json"
        payload = {
            "schema": DONE_SCHEMA,
            "plan_sha256": plan_sha,
            "shard_id": args.shard_id,
            "positions": shard["count"],
        }
        try:
            with marker.open("xb") as stream:
                stream.write(canonical_bytes(payload))
        except FileExistsError:
            pass
    print(f"shard {args.shard_id}: produced {produced} new positions, "
          f"complete={done}", flush=True)
    return 0


def run(args: argparse.Namespace) -> int:
    runtime = pathlib.Path(args.runtime_root).resolve()
    plan_path = pathlib.Path(args.plan).resolve()
    plan = load_plan(plan_path)
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    shard = shard_entry(plan, args.shard_id)
    out = pathlib.Path(args.shard_directory)
    out.mkdir(parents=True, exist_ok=True)
    # Read before the engine and the weights: a manifest the caller expected to
    # exist and does not is worth failing on in the first second, not after a
    # minute of digest checks.
    existing_manifest = load_existing_manifest(args.existing_manifest)
    if args.existing_manifest is not None:
        print(f"existing manifest {args.existing_manifest}: "
              f"{len(existing_manifest)} positions already published", flush=True)

    if plan.get("plan_kind") in (T3_VS_FL_KIND, T2_VS_FL_KIND, T1_VS_FL_KIND,
                                T0_VS_FL_KIND):
        return run_t3_vs_fl(args, plan, plan_sha, shard, out, existing_manifest)

    engine = pinned(runtime, plan["engine_library"],
                    plan["engine_library_sha256"], "engine library")
    encoder = pinned(runtime, plan["feature_encoder_library"],
                     plan["feature_encoder_library_sha256"], "feature encoder")
    t4_weights = pinned(runtime, plan["t4_model"],
                        plan["t4_model_sha256"], "T4 model")
    street = plan["street"]
    seat = plan["seat"]
    # Absent on exactly one kind -- see the carve-out in `load_plan`, which has
    # already established that a (T3, second) plan does not carry these fields
    # and that every other kind does.
    t3_weights = None
    if (street, seat) != ("T3", "second"):
        t3_weights = pinned(runtime, plan["t3_second_model"],
                            plan["t3_second_model_sha256"], "T3 second model")
    t3_first_weights = None
    if street in ("T2", "T1", "T0"):
        t3_first_weights = pinned(runtime, plan["t3_first_model"],
                                  plan["t3_first_model_sha256"], "T3 first model")
    t2_second_weights = None
    if (street, seat) == ("T2", "first") or street in ("T1", "T0"):
        t2_second_weights = pinned(runtime, plan["t2_second_model"],
                                   plan["t2_second_model_sha256"],
                                   "T2 second model")
    t2_first_weights = None
    if street in ("T1", "T0"):
        t2_first_weights = pinned(runtime, plan["t2_first_model"],
                                  plan["t2_first_model_sha256"],
                                  "T2 first model")
    t1_second_weights = None
    if (street, seat) == ("T1", "first") or street == "T0":
        t1_second_weights = pinned(runtime, plan["t1_second_model"],
                                   plan["t1_second_model_sha256"],
                                   "T1 second model")
    t1_first_weights = None
    if street == "T0":
        t1_first_weights = pinned(runtime, plan["t1_first_model"],
                                  plan["t1_first_model_sha256"],
                                  "T1 first model")
    t0_second_weights = None
    if (street, seat) == ("T0", "first"):
        t0_second_weights = pinned(runtime, plan["t0_second_model"],
                                   plan["t0_second_model_sha256"],
                                   "T0 second model")
    # Optional, and T0-only; load_plan has already established that the plan
    # carries all four fields or none. Pinned exactly like the seven above:
    # a coarse model is still a model the label depends on, so a stale image
    # has to fail here rather than after a shard has been written against it.
    fast_t1_first_weights = None
    fast_t1_second_weights = None
    if "fast_t1_first_model" in plan:
        fast_t1_first_weights = pinned(runtime, plan["fast_t1_first_model"],
                                       plan["fast_t1_first_model_sha256"],
                                       "fast T1 first model")
        fast_t1_second_weights = pinned(runtime, plan["fast_t1_second_model"],
                                        plan["fast_t1_second_model_sha256"],
                                        "fast T1 second model")
    # Optional, and T1/T0-only; load_plan has already established that the plan
    # carries all four fields or none and that this kind plays both replies.
    fast_t2_first_weights = None
    fast_t2_second_weights = None
    if "fast_t2_first_model" in plan:
        fast_t2_first_weights = pinned(runtime, plan["fast_t2_first_model"],
                                       plan["fast_t2_first_model_sha256"],
                                       "fast T2 first model")
        fast_t2_second_weights = pinned(runtime, plan["fast_t2_second_model"],
                                        plan["fast_t2_second_model_sha256"],
                                        "fast T2 second model")
    # Optional, and (T0, first)-only; load_plan has already established that the
    # plan carries both fields or neither and that this kind is the one that can
    # use them.
    fast_t0_second_weights = None
    if "fast_t0_second_model" in plan:
        fast_t0_second_weights = pinned(runtime, plan["fast_t0_second_model"],
                                        plan["fast_t0_second_model_sha256"],
                                        "fast T0 second model")

    from . import hu_m3_rust
    from .ai_profiles import ModelPaths, load_model_bundle
    from .hu_m31_t0_behavior_roots import generate_behavior_t0_roots
    from .hu_m31_t1_behavior_roots import generate_behavior_t1_roots
    from .hu_m31_t2_behavior_roots import generate_behavior_t2_roots
    from .hu_m31_t3_behavior_roots import (
        behavior_profile_for_index,
        generate_behavior_t3_roots,
    )
    from .hu_turn3_joint_exact_teacher import JointExactConfig
    from .hu_turn3_stage3_feature_rust import pinned_feature_encoder_library

    library = hu_m3_rust.load_native_engine(path=engine, build_if_missing=False)
    bundle = load_model_bundle(
        ModelPaths(), {behavior_profile_for_index(i) for i in range(5)}
    )

    def build_config(offset: int, trial: int) -> JointExactConfig:
        seed = plan["eval_seed_base"] + offset * 7 + trial * 3_000_017
        extra = {}
        if street in ("T2", "T1", "T0"):
            extra = {
                "learned_t3_first_model_path": str(t3_first_weights),
                "learned_t3_first_model_sha256": plan["t3_first_model_sha256"],
            }
            if (street, seat) == ("T2", "first") or street in ("T1", "T0"):
                # The first seat acts before the second on T2, so the teacher
                # needs the opponent's own T2 reply learned as well; the engine
                # only accepts a first-seat T2 root once all four are pinned.
                # Any T1 root is upstream of the same T2 second-seat reply, so
                # it needs that model for the identical reason.
                extra["learned_t2_second_model_path"] = str(t2_second_weights)
                extra["learned_t2_second_model_sha256"] = (
                    plan["t2_second_model_sha256"]
                )
            if street in ("T1", "T0"):
                # T1 is a whole street above T2, so both T2 decisions are still
                # ahead of the teacher: the opponent's first-seat T2 reply is the
                # fifth learned evaluator on the path, for either T1 seat. T0 is
                # a street above that again, so it inherits the same need.
                extra["learned_t2_first_model_path"] = str(t2_first_weights)
                extra["learned_t2_first_model_sha256"] = (
                    plan["t2_first_model_sha256"]
                )
                if fast_t2_first_weights is not None:
                    # Passed ALONGSIDE the full-precision pair, not instead of
                    # it: the engine loads both and reaches its T2 replies
                    # through the coarse one. Written here rather than in the
                    # T0-only block below because a T1 teacher plays both T2
                    # replies too, which is the condition this branch tests.
                    extra["fast_t2_first_model_path"] = str(fast_t2_first_weights)
                    extra["fast_t2_first_model_sha256"] = (
                        plan["fast_t2_first_model_sha256"]
                    )
                    extra["fast_t2_second_model_path"] = str(fast_t2_second_weights)
                    extra["fast_t2_second_model_sha256"] = (
                        plan["fast_t2_second_model_sha256"]
                    )
            if (street, seat) == ("T1", "first") or street == "T0":
                # Acting first on T1 nothing on this street has happened yet, so
                # the opponent's T1 second-seat reply opens every rollout: the
                # sixth learned evaluator. From T0 the whole T1 street is still
                # ahead, so both T1 seats' models are on the path.
                extra["learned_t1_second_model_path"] = str(t1_second_weights)
                extra["learned_t1_second_model_sha256"] = (
                    plan["t1_second_model_sha256"]
                )
            if street == "T0":
                # The seventh and last: from the opening street the T1 first-seat
                # decision is played on both seats' rollouts, and acting first at
                # T1 both boards carry five cards where the second seat sees
                # seven, so that seat has its own model. No other kind this
                # worker supports has ever needed it.
                extra["learned_t1_first_model_path"] = str(t1_first_weights)
                extra["learned_t1_first_model_sha256"] = (
                    plan["t1_first_model_sha256"]
                )
                if seat == "first":
                    # The eighth, and the last learned evaluator the ladder has:
                    # acting first on the opening street the opponent's T0
                    # second-seat reply opens every rollout. The engine refuses a
                    # T0 first-seat root outright without it.
                    extra["learned_t0_second_model_path"] = str(t0_second_weights)
                    extra["learned_t0_second_model_sha256"] = (
                        plan["t0_second_model_sha256"]
                    )
                    if fast_t0_second_weights is not None:
                        # Alongside the full-precision eighth model, exactly as
                        # the coarse T1 pair sits alongside the full-precision
                        # T1 pair: the engine loads both and reaches that one
                        # reply through the coarse one, which is why the
                        # provenance gate below expects "learned_fast" from
                        # t0_second_evaluator.
                        extra["fast_t0_second_model_path"] = str(
                            fast_t0_second_weights
                        )
                        extra["fast_t0_second_model_sha256"] = (
                            plan["fast_t0_second_model_sha256"]
                        )
                # The two-stage root schedule, which load_plan required this kind
                # to state. Passing it is the whole point of requiring it: the
                # engine's default is single-stage over all ~232 root actions.
                extra["prefilter_samples"] = plan["prefilter_samples"]
                extra["prefilter_keep"] = plan["prefilter_keep"]
                # And the two optional safeguards on that schedule, passed
                # through exactly as the plan states them. Omitted from the
                # config when the plan omits them, so the request the engine
                # sees is byte-identical to one built before they existed.
                for field in PRUNING_SAFETY_FIELDS:
                    if field in plan:
                        extra[field] = plan[field]
                # And the stage-two race, passed whole or not at all --
                # load_plan has already established the plan carries both
                # fields or neither. Omitted from the config when the plan
                # omits them, so the request the engine sees is byte-identical
                # to one built before racing existed, which is what lets the
                # pinned engine on a running fleet keep answering old plans.
                if "race_schedule" in plan:
                    extra["race_schedule"] = tuple(plan["race_schedule"])
                    extra["race_lcb_z"] = plan["race_lcb_z"]
                if fast_t1_first_weights is not None:
                    # Passed ALONGSIDE the full-precision pair, not instead of
                    # it: the engine loads both and reaches its T1 replies
                    # through the coarse one. That is why the provenance gate
                    # below expects "learned_fast" for exactly these two
                    # evaluators while the other five still report "learned".
                    extra["fast_t1_first_model_path"] = str(fast_t1_first_weights)
                    extra["fast_t1_first_model_sha256"] = (
                        plan["fast_t1_first_model_sha256"]
                    )
                    extra["fast_t1_second_model_path"] = str(fast_t1_second_weights)
                    extra["fast_t1_second_model_sha256"] = (
                        plan["fast_t1_second_model_sha256"]
                    )
        if t3_weights is not None:
            # Omitted entirely rather than passed as None on the one kind that
            # does not use it, so the request the engine sees carries exactly
            # the models the label depends on.
            extra["learned_t3_second_model_path"] = str(t3_weights)
            extra["learned_t3_second_model_sha256"] = (
                plan["t3_second_model_sha256"]
            )
        return JointExactConfig(
            candidate_samples=8,
            evaluation_samples=plan["samples"],
            downstream_t3_samples=4,
            downstream_t4_samples=0,
            seed=seed,
            candidate_seed=seed,
            evaluation_seed=seed,
            run_id=f"{plan['job_id']}-{args.shard_id}-{offset}-{trial}",
            seat=plan["seat"],
            to_act_order=plan["seat"],
            learned_t4_model_path=str(t4_weights),
            learned_t4_model_sha256=plan["t4_model_sha256"],
            **extra,
        )

    def make_root(offset: int):
        """The observation this street labels: T3 exposes the first seat's
        decision, T2, T1 and T0 either seat's, chosen by the plan."""

        arguments = {
            "hand_seed": plan["hand_seed_base"] + offset,
            "behavior_seed": plan["hand_seed_base"]
            + plan["behavior_seed_offset"] + offset,
            "profile": behavior_profile_for_index(offset),
            "bundle": bundle,
        }
        if street == "T3":
            # (first, second), the same convention every other street's pair
            # follows. The second element used to be discarded because no plan
            # could ask for it.
            first, second = generate_behavior_t3_roots(**arguments)
            return first if seat == "first" else second
        if street == "T0":
            # (first, second), same convention as every other street's pair. The
            # opening street makes the two elements trivially distinguishable:
            # the first seat sees an empty table (0/0), the second sees the
            # opponent's five placed cards (0/5). Both are dealt five with
            # nothing discarded, so the board geometry is the whole
            # discriminator between the two kinds.
            first, second = generate_behavior_t0_roots(**arguments)
            return first if seat == "first" else second
        if street == "T1":
            first, second = generate_behavior_t1_roots(**arguments)
            return first if seat == "first" else second
        first, second = generate_behavior_t2_roots(**arguments)
        return first if seat == "first" else second

    evaluate = {
        "T3": hu_m3_rust.evaluate_t3,
        "T2": hu_m3_rust.evaluate_t2,
        "T1": hu_m3_rust.evaluate_t1,
        "T0": hu_m3_rust.evaluate_t0,
    }[street]
    # Evaluator field -> the value the engine has to report for it. Every
    # learned evaluator on this kind's path reports "learned"; the two the fast
    # pair displaces report "learned_fast" instead. Both directions matter: a
    # plan that pinned the coarse pair must not accept a full-precision answer,
    # and a plan that did not must not accept a coarse one.
    provenance_expected = {
        "t4_first_evaluator": "learned",
    }
    if t3_weights is not None:
        # Every kind but (T3, second). The engine reports this evaluator when
        # the weights are loaded, so demanding it on the one kind that does not
        # load them would fail a correct run, and demanding it on a run that
        # loaded them but never reached them would prove nothing.
        provenance_expected["t3_second_evaluator"] = "learned"
    if street in ("T2", "T1", "T0"):
        provenance_expected["t3_first_evaluator"] = "learned"
    if (street, seat) == ("T2", "first") or street in ("T1", "T0"):
        provenance_expected["t2_second_evaluator"] = "learned"
    if street in ("T1", "T0"):
        provenance_expected["t2_first_evaluator"] = "learned"
    if (street, seat) == ("T1", "first") or street == "T0":
        provenance_expected["t1_second_evaluator"] = "learned"
    if street == "T0":
        provenance_expected["t1_first_evaluator"] = "learned"
    if (street, seat) == ("T0", "first"):
        provenance_expected["t0_second_evaluator"] = "learned"
    # Digest fields the engine reports only once the coarse pair is loaded.
    # "learned_fast" alone would be satisfied by any fast model; these say
    # which one, which is the same standard the pinned digests hold the
    # full-precision weights to.
    provenance_digests: dict[str, str] = {}
    if fast_t1_first_weights is not None:
        provenance_expected["t1_second_evaluator"] = "learned_fast"
        provenance_expected["t1_first_evaluator"] = "learned_fast"
        provenance_digests["fast_t1_second_model_sha256"] = (
            plan["fast_t1_second_model_sha256"]
        )
        provenance_digests["fast_t1_first_model_sha256"] = (
            plan["fast_t1_first_model_sha256"]
        )
    if fast_t2_first_weights is not None:
        provenance_expected["t2_second_evaluator"] = "learned_fast"
        provenance_expected["t2_first_evaluator"] = "learned_fast"
        provenance_digests["fast_t2_second_model_sha256"] = (
            plan["fast_t2_second_model_sha256"]
        )
        provenance_digests["fast_t2_first_model_sha256"] = (
            plan["fast_t2_first_model_sha256"]
        )
    if fast_t0_second_weights is not None:
        provenance_expected["t0_second_evaluator"] = "learned_fast"
        provenance_digests["fast_t0_second_model_sha256"] = (
            plan["fast_t0_second_model_sha256"]
        )

    # The Fantasyland constant this run will bake into every label it writes.
    # Read from the same single source of truth the observations will carry, so
    # it is what the labels will MEAN rather than what the plan hoped they would.
    from .hu_infoset import load_default_fl_ev

    runtime_fl_ev = {
        str(cards): float(value) for cards, value in load_default_fl_ev().items()
    }
    if "fl_ev_value" in plan:
        cards = str(plan["fl_ev_cards"])
        stated = float(plan["fl_ev_value"])
        if runtime_fl_ev.get(cards) != stated:
            raise SystemExit(
                f"plan pins fl_ev[{cards}]={stated!r} but this runtime's config "
                f"reads {runtime_fl_ev.get(cards)!r}. The plan cannot set the "
                "constant -- it reaches the engine through the observation's "
                "scoring context from the one config hu_infoset reads -- so a "
                "disagreement means the package carries a different config from "
                "the one the plan was written against, and every label this "
                "shard produced would be a quantity nobody asked for"
            )

    def position_path(offset: int) -> pathlib.Path:
        return out / f"position_{offset:08d}.json"

    def already_done(offset: int) -> bool:
        path = position_path(offset)
        if path.name in existing_manifest:
            # Durable somewhere the caller can see even if this disk cannot:
            # regenerating it would only produce an upload collision. Nothing
            # local to compare an fl_ev against, so the mixing guard below
            # cannot see these -- the manifest's own producer is where that
            # check belongs for them.
            return True
        if not path.is_file():
            return False
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A resume copy truncated by preemption: redo under a fresh name is
            # impossible with create-only semantics, so remove the provably
            # broken file and regenerate.
            path.unlink()
            return False
        if record.get("schema") != POSITION_SCHEMA:
            return False
        # The corpus-mixing guard. A shard directory is resumed by scanning it,
        # so a resume that lands on positions written under a DIFFERENT
        # Fantasyland constant would append labels of one quantity to a corpus
        # of another and leave nothing downstream able to tell which row was
        # which. This is the same standard the vs-Fantasyland kinds hold
        # `opponent_mode` to: a label built one way is a different quantity from
        # a label built the other way, and mixing them silently is exactly the
        # failure the gate exists to prevent.
        recorded = record.get("observation", {}).get("scoring", {}).get("fl_ev")
        if isinstance(recorded, Mapping):
            recorded_fl_ev = {
                str(cards): float(value) for cards, value in recorded.items()
            }
            if recorded_fl_ev != runtime_fl_ev:
                raise SystemExit(
                    f"{path.name} carries fl_ev={recorded_fl_ev} but this run "
                    f"would write fl_ev={runtime_fl_ev}. Refusing to resume: "
                    "appending to this shard would mix two different quantities "
                    "into one corpus. Relabel the shard into an empty directory, "
                    "or point this run at the config the existing positions were "
                    "written under"
                )
        return True

    began = time.perf_counter()
    produced = 0
    checked = False
    offsets = range(shard["start"], shard["start"] + shard["count"])
    # Several processes may share one shard directory. The split is by offset
    # stride, so no two processes ever compute the same position and the
    # create-only files never collide.
    if not 0 <= args.stride_index < args.stride_count:
        raise SystemExit("stride index must lie inside the stride count")
    mine = [o for o in offsets
            if (o - shard["start"]) % args.stride_count == args.stride_index]
    remaining = [o for o in mine if not already_done(o)]
    print(f"shard {args.shard_id} stride {args.stride_index}/{args.stride_count}: "
          f"{len(remaining)}/{len(mine)} positions to generate", flush=True)

    with pinned_feature_encoder_library(
        encoder, expected_sha256=plan["feature_encoder_library_sha256"]
    ):
        for offset in remaining:
            if args.max_new_positions >= 0 and produced >= args.max_new_positions:
                break
            root = make_root(offset)
            runs = []
            for trial in range(plan["seeds_per_position"]):
                result = evaluate(
                    root, config=build_config(offset, trial), library=library
                )
                if not checked:
                    policy = result["continuation_policy"]
                    for field, expected in provenance_expected.items():
                        if policy.get(field) != expected:
                            raise SystemExit(
                                f"engine reports {field}={policy.get(field)!r}, "
                                f"expected {expected!r}; refusing to write "
                                "labels whose provenance is not what the plan "
                                "pinned"
                            )
                    for field, expected in provenance_digests.items():
                        if policy.get(field) != expected:
                            raise SystemExit(
                                f"engine reports {field}={policy.get(field)!r}, "
                                f"expected the plan's {expected!r}; refusing to "
                                "write labels whose provenance is not what the "
                                "plan pinned"
                            )
                    # The race, echoed at the top level of the result rather
                    # than in the continuation policy, because it describes how
                    # the ROOT was scored rather than how the hand was played
                    # out. Checked in both directions like every evaluator
                    # above: a plan that asked to race must not accept a
                    # uniform stage two, and a plan that did not must not
                    # discover it raced. An engine too old to know the fields
                    # accepts the request and reports nothing, which this
                    # catches on the first position rather than at the end of
                    # the shard.
                    for field in RACE_FIELDS:
                        want = plan.get(field)
                        got = result.get(field)
                        if field == "race_schedule" and got is not None:
                            got = list(got)
                        if want != got:
                            raise SystemExit(
                                f"engine reports {field}={got!r}, the plan asks "
                                f"for {want!r}; refusing to write labels whose "
                                "root schedule is not what the plan chose"
                            )
                    checked = True
                if street == "T0":
                    # T0 is the one street whose root is scored by a two-stage
                    # schedule, so a bare number no longer says what it is: the
                    # engine reports, per action, which stage measured it and
                    # what the cheap first-stage pass thought. Keeping the row
                    # verbatim keeps that -- and anything else the engine adds
                    # later -- instead of discarding it at the moment it is
                    # cheapest to record.
                    scores = {row["action_key"]: row
                              for row in result["actions"]}
                else:
                    scores = {row["action_key"]: row["score"]
                              for row in result["actions"]}
                run_record = {"seed_trial": trial, "scores": scores}
                # The standing audit's verdict, when this position was one the
                # cadence selected. Stored verbatim and only when present: the
                # engine writes the key on audited positions alone, and the
                # measurement the audit exists to accumulate is worthless if it
                # is summarised here rather than carried. A run without it is
                # byte-identical to one written before the audit existed.
                audit = result.get("audit")
                if audit is not None:
                    run_record["audit"] = audit
                runs.append(run_record)
            record = {
                "schema": POSITION_SCHEMA,
                "plan_sha256": plan_sha,
                "offset": offset,
                "skeleton": hashlib.sha256(
                    ("".join(sorted(root.hero_board.all_cards())) + "|"
                     + "".join(sorted(root.opponent_public_board.all_cards()))
                     ).encode()
                ).hexdigest()[:2],
                "observation": root.to_dict(),
                "samples": plan["samples"],
                "runs": runs,
            }
            with position_path(offset).open("xb") as stream:
                stream.write(canonical_bytes(record))
                stream.flush()
            produced += 1
            if produced % args.report_every == 0:
                rate = produced / (time.perf_counter() - began)
                print(f"  {produced}/{len(remaining)} new  {rate:.2f} pos/s",
                      flush=True)

    done = all(already_done(o) for o in offsets)
    if done:
        marker = out / "SHARD_DONE.json"
        payload = {
            "schema": DONE_SCHEMA,
            "plan_sha256": plan_sha,
            "shard_id": args.shard_id,
            "positions": shard["count"],
        }
        try:
            with marker.open("xb") as stream:
                stream.write(canonical_bytes(payload))
        except FileExistsError:
            pass  # a sibling stride finished the same check first
    print(f"shard {args.shard_id}: produced {produced} new positions, "
          f"complete={done}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="generate one shard")
    runner.add_argument("--plan", required=True)
    runner.add_argument("--shard-id", required=True)
    runner.add_argument("--shard-directory", required=True)
    runner.add_argument("--runtime-root", required=True)
    runner.add_argument("--max-new-positions", type=int, default=-1,
                        help="-1 runs to completion; N stops after N new")
    runner.add_argument("--stride-index", type=int, default=0)
    runner.add_argument("--stride-count", type=int, default=1,
                        help="split one shard across K sibling processes by "
                             "offset stride; no two ever touch the same position")
    runner.add_argument("--report-every", type=int, default=100)
    runner.add_argument("--solver-threads", type=int, default=1,
                        help="t3_vs_fl only: worker threads inside the FL "
                             "solver. One process with N threads beats N "
                             "single-threaded strides here, because the "
                             "opponent pool is solved once per position")
    runner.add_argument("--solver-chunk", type=int, default=32,
                        help="t3_vs_fl only: positions per solver invocation. "
                             "Larger amortises process startup; smaller is what "
                             "a preemption can lose")
    runner.add_argument("--existing-manifest", default=None,
                        help="file of position file names, one per line, that "
                             "are already published elsewhere; they count as "
                             "done exactly as a local file would, which is how "
                             "a recreated instance resumes from an empty disk")
    args = parser.parse_args()
    if args.command == "run":
        return run(args)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
