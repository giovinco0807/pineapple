"""How good does the policy at an interior node have to be?

The 3-max cascade is exact only where the tree has no opponent decision left in
it: T4 everywhere, and T3 for the BTN because it closes the street.  Every rung
above that has an interior opponent node, and the whole plan rests on filling
those nodes with a distilled model instead of a solver -- which is affordable
only if the label above the node does not move when you do it.

(T3, BB) is the cheapest place that question can be asked: exactly one interior
node, the BTN's T3, and an exact T4 round underneath it.

    python scripts/probe_three_max_interior.py samples --roots 40
    python scripts/probe_three_max_interior.py arms --roots 200 --workers 14

``samples``
    The reference arm resolves the interior node with the exact solver, whose
    cost is linear in how many T4 draws the BTN averages over.  Its CHOICE is
    much more stable than its values, so this measures the cheapest sample
    count that still picks the reference's action, and the ``arms`` run uses it.

``arms``
    Labels the same roots under every interior policy on shared strata, and
    scores each arm's pick against the reference's labels.  The reference's own
    seed noise is measured the same way -- a second exact arm on a disjoint
    stream -- because an arm is only "as good as exact" relative to how much
    exact disagrees with itself.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# One worker per core, so every worker must stay single-threaded.  Set before
# torch or a BLAS is imported anywhere, including in a spawned child.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ofc_regular.action_space import generate_actions  # noqa: E402
from ofc_regular.cards import create_deck  # noqa: E402
from ofc_regular.three_max.exact import evaluate_t3  # noqa: E402
from ofc_regular.three_max.interior import (  # noqa: E402
    action_token,
    evaluate_t3_middle,
    exact_interior,
    mc_interior,
    model_interior,
    random_interior,
    sample_middle_draws,
)
from ofc_regular.three_max.mc import mc_policy  # noqa: E402
from ofc_regular.three_max.world import WorldState3  # noqa: E402

SCHEMA = "regular_ofc_3max_interior_probe_v1"
REFERENCE_SALT = 0
FLOOR_SALT = 0x5DEECE66D
SAMPLE_LADDER = (4, 8, 16, 32, 64)
REFERENCE_SAMPLES_FOR_LADDER = 512


def _root_policy(name: str):
    if name == "hu":
        from ofc_regular.three_max.hu_bridge import hu_policy

        return hu_policy()
    return mc_policy(sims=4)


def _root_at(seat: str, seed: int, root_policy: str) -> WorldState3:
    """Play a hand under ``root_policy`` and stop at ``seat``'s T3 decision."""
    policy = _root_policy(root_policy)
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(seed)))
    while True:
        slot = world.current_slot()
        if slot.street == "T3" and slot.seat == seat:
            return world
        world = world.apply(policy(world.observe(), seed * 1_000_003 + slot.decision_index))


# --- samples: how stable is the interior CHOICE in its sample count? ---------


def _sample_ladder_root(args: tuple) -> dict:
    seed, root_policy, fl_ev_14 = args
    observation = _root_at("btn", seed, root_policy).observe()
    reference = evaluate_t3(
        observation,
        samples=REFERENCE_SAMPLES_FOR_LADDER,
        seed=seed ^ 0xABCDEF,
        fl_ev_per_pair={14: fl_ev_14},
    )
    truth = {action_token(candidate.action): candidate.ev for candidate in reference}
    best = max(truth.values())
    spread = best - min(truth.values())

    row: dict[str, float] = {"spread": spread, "width": len(truth)}
    for samples in SAMPLE_LADDER:
        ranked = evaluate_t3(
            observation,
            samples=samples,
            seed=seed,
            fl_ev_per_pair={14: fl_ev_14},
        )
        picked = truth[action_token(ranked[0].action)]
        row[f"regret_{samples}"] = best - picked
        row[f"top1_{samples}"] = float(picked == best)
    return row


def run_samples(args: argparse.Namespace) -> dict:
    payload = [
        (args.base_seed + index, args.root_policy, args.fl_ev_14)
        for index in range(args.roots)
    ]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_sample_ladder_root, payload))

    spread = statistics.fmean(row["spread"] for row in rows) or 1.0
    ladder = [
        {
            "samples": samples,
            "regret": statistics.fmean(row[f"regret_{samples}"] for row in rows),
            "regret_normalised": statistics.fmean(
                row[f"regret_{samples}"] for row in rows
            )
            / spread,
            "agrees_with_reference": statistics.fmean(
                row[f"top1_{samples}"] for row in rows
            ),
        }
        for samples in SAMPLE_LADDER
    ]
    return {
        "schema": SCHEMA,
        "mode": "samples",
        "question": (
            "the interior arm's cost is linear in evaluate_t3's samples; how few "
            "can it use and still pick the reference's action"
        ),
        "roots": len(rows),
        "root_policy": args.root_policy,
        "reference_samples": REFERENCE_SAMPLES_FOR_LADDER,
        "mean_within_root_spread": spread,
        "mean_fan_width": statistics.fmean(row["width"] for row in rows),
        "ladder": ladder,
        "wall_seconds": round(time.time() - started, 1),
    }


# --- arms: swap the interior policy, watch the label above it ----------------


def _arms_for(args: argparse.Namespace) -> list[tuple[str, dict]]:
    """Arm name -> constructor spec.  The control is first, deliberately."""
    arms: list[tuple[str, dict]] = [
        ("random", {"kind": "random"}),
        ("mc4", {"kind": "mc", "sims": 4}),
        ("mc32", {"kind": "mc", "sims": 32}),
    ]
    if args.model:
        arms.append(("model", {"kind": "model", "path": str(args.model)}))
    if args.hu:
        arms.append(("hu", {"kind": "hu"}))
    # Two floors, and they measure different things.  ``exact_floor`` re-runs
    # the reference's own interior solver on a disjoint seed stream over the
    # SAME strata: that is how much the interior node's sampling alone moves
    # the pick.  ``exact_other_draws`` re-runs it on a different set of strata:
    # that is the yardstick's own noise, the floor no arm can be judged below.
    arms.append(
        ("exact_floor", {"kind": "exact", "samples": args.interior_samples,
                         "salt": FLOOR_SALT})
    )
    arms.append(
        ("exact_other_draws", {"kind": "exact", "samples": args.interior_samples,
                               "salt": REFERENCE_SALT, "draws": "alt"})
    )
    keep = {name.strip() for name in args.only_arms.split(",") if name.strip()}
    if keep:
        keep |= {"exact_floor", "exact_other_draws"}
        arms = [(name, spec) for name, spec in arms if name in keep]
    return arms


def _build_arm(spec: dict):
    kind = spec["kind"]
    if kind == "random":
        return random_interior()
    if kind == "mc":
        return mc_interior(sims=spec["sims"])
    if kind == "model":
        return model_interior(spec["path"])
    if kind == "exact":
        return exact_interior(samples=spec["samples"], seed_salt=spec["salt"])
    if kind == "hu":
        from ofc_regular.three_max.hu_bridge import hu_policy

        return hu_policy()
    raise ValueError(f"unknown interior arm: {kind!r}")


def _arms_root(payload: tuple) -> dict | None:
    seed, config = payload
    observation = _root_at("bb", seed, config["root_policy"]).observe()
    draw_sets = {
        "main": sample_middle_draws(
            observation.unknown_cards(),
            strata=config["strata"],
            per_stratum=config["per_stratum"],
            seed=seed ^ 0x51ED,
        ),
        "alt": sample_middle_draws(
            observation.unknown_cards(),
            strata=config["strata"],
            per_stratum=config["per_stratum"],
            seed=seed ^ 0xA17E,
        ),
    }
    fl = {14: config["fl_ev_14"]}

    def label(policy, which: str = "main") -> dict[str, float]:
        ranked = evaluate_t3_middle(
            observation,
            interior_policy=policy,
            draws=draw_sets[which],
            seed=seed,
            fl_ev_per_pair=fl,
        )
        return {action_token(item.action): item.ev for item in ranked}

    reference = label(
        exact_interior(samples=config["interior_samples"], seed_salt=REFERENCE_SALT)
    )
    best = max(reference.values())
    spread = best - min(reference.values())

    # A root is "Fantasyland live" when at least one legal move finishes the
    # hero's top row at QQ+.  The gen2 report flagged FL-live roots as where
    # the T3-BTN model loses but could only put 28 roots in the control bucket,
    # so the split travels with every row from here on.
    from ofc_regular.rules import check_fl_entry

    fl_actions = 0
    for action in generate_actions(observation.hero_board, observation.dealt_cards):
        top = observation.hero_board.place(action.placements).top
        fl_actions += len(top) == 3 and check_fl_entry(top).qualifies

    row: dict[str, float] = {
        "seed": seed,
        "spread": spread,
        "width": len(reference),
        "reference_best": best,
        "fl_actions": fl_actions,
        "top_slots_open": 3 - len(observation.hero_board.top),
    }
    for name, spec in config["arms"]:
        values = label(_build_arm(spec), spec.get("draws", "main"))
        pick = max(values, key=lambda token: values[token])
        row[f"regret_{name}"] = best - reference[pick]
        row[f"top1_{name}"] = float(reference[pick] == best)
        # How far the arm's own numbers sit from the reference's, action by
        # action.  A shifted label can still rank correctly; a scrambled one
        # cannot.
        row[f"mae_{name}"] = statistics.fmean(
            abs(values[token] - reference[token]) for token in reference
        )
    return row


def _subgroups(rows: list[dict], arms: list[tuple[str, dict]]) -> dict:
    """The same paired comparison inside and outside the Fantasyland race.

    An aggregate null can hide two effects that cancel, and this is the split
    most likely to carry one: FL-live roots are where the decisions are hardest
    and where the previous generation's model lost most.
    """
    buckets = {
        "fl_live": [row for row in rows if row["fl_actions"] > 0],
        "no_fl": [row for row in rows if row["fl_actions"] == 0],
    }
    report = {}
    for name, bucket in buckets.items():
        if not bucket:
            report[name] = {"roots": 0}
            continue
        spread = statistics.fmean(row["spread"] for row in bucket) or 1.0
        report[name] = {
            "roots": len(bucket),
            "mean_within_root_spread": spread,
            "arms": {
                arm: {
                    "regret": statistics.fmean(row[f"regret_{arm}"] for row in bucket),
                    "vs_floor_paired": statistics.fmean(
                        row[f"regret_{arm}"] - row["regret_exact_floor"]
                        for row in bucket
                    ),
                    "vs_floor_paired_stderr": (
                        statistics.stdev(
                            [
                                row[f"regret_{arm}"] - row["regret_exact_floor"]
                                for row in bucket
                            ]
                        )
                        / len(bucket) ** 0.5
                        if len(bucket) > 1
                        else 0.0
                    ),
                }
                for arm, _spec in arms
            },
        }
    return report


def run_arms(args: argparse.Namespace) -> dict:
    arms = _arms_for(args)
    config = {
        "root_policy": args.root_policy,
        "strata": args.strata,
        "per_stratum": args.per_stratum,
        "interior_samples": args.interior_samples,
        "fl_ev_14": args.fl_ev_14,
        "arms": arms,
    }
    payload = [(args.base_seed + index, config) for index in range(args.roots)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for row in pool.map(_arms_root, payload) if row is not None]

    spread = statistics.fmean(row["spread"] for row in rows) or 1.0
    results = []
    for name, spec in arms:
        regrets = [row[f"regret_{name}"] for row in rows]
        # Every arm ran on the same roots and the same strata, so the honest
        # comparison is paired.  The reference disagreeing with its own seed
        # stream is the bar: an arm at or below ``exact_floor`` is, on this
        # measurement, indistinguishable from resolving the node exactly.
        paired = [
            row[f"regret_{name}"] - row["regret_exact_floor"] for row in rows
        ]
        results.append(
            {
                "arm": name,
                "spec": spec,
                "regret": statistics.fmean(regrets),
                "regret_normalised": statistics.fmean(regrets) / spread,
                "regret_stderr": (
                    statistics.stdev(regrets) / len(regrets) ** 0.5
                    if len(regrets) > 1
                    else 0.0
                ),
                "vs_floor_paired": statistics.fmean(paired),
                "vs_floor_paired_stderr": (
                    statistics.stdev(paired) / len(paired) ** 0.5
                    if len(paired) > 1
                    else 0.0
                ),
                "agrees_with_reference": statistics.fmean(
                    row[f"top1_{name}"] for row in rows
                ),
                "label_mae": statistics.fmean(row[f"mae_{name}"] for row in rows),
            }
        )
    if args.rows_output:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        args.rows_output.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    return {
        "schema": SCHEMA,
        "mode": "arms",
        "question": (
            "does the label at (T3, BB) move when the one interior node -- the "
            "BTN's T3 -- is resolved by a distilled model instead of a solver"
        ),
        "roots": len(rows),
        "subgroups": _subgroups(rows, arms),
        "root_policy": args.root_policy,
        "draws": {"strata": args.strata, "per_stratum": args.per_stratum,
                  "total": args.strata * args.per_stratum},
        "reference": f"exact_interior(samples={args.interior_samples})",
        "mean_within_root_spread": spread,
        "mean_fan_width": statistics.fmean(row["width"] for row in rows),
        "arms": results,
        "wall_seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("samples", "arms"))
    parser.add_argument("--roots", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=8_300_000)
    parser.add_argument("--root-policy", choices=("hu", "mc"), default="hu")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--fl-ev-14", type=float, default=9.6)
    parser.add_argument("--strata", type=int, default=8)
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument("--interior-samples", type=int, default=16)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--hu", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rows-output", type=Path, default=None)
    parser.add_argument(
        "--only-arms",
        default="",
        help="comma-separated arm names to keep; the two floors are always kept",
    )
    args = parser.parse_args()

    report = run_samples(args) if args.mode == "samples" else run_arms(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
