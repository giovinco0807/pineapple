"""T3 teacher-data generation for the 3-max BTN seat.

The BTN seat at T3 is the first rung of the 3-max cascade for the same reason
the second seat was in heads-up: both opponents have already finished their own
T3, so the continuation is a single T4 round with no interior opponent decision
and ``exact.evaluate_t3`` can solve it outright.

One record is one root: the observation, every legal action, and each action's
value under a shared draw set.  The winner also carries ``ev_holdout``, its
value re-scored on draws the ranking never saw -- the ranked list is what a
distilled model imitates, but anything that consumes the winner's NUMBER must
read the held-out one, because the in-sample maximum of a shared-draw estimate
is biased upward by selection.

    python -m ofc_regular.three_max.teacher --count 200 --output out.jsonl

Roots come from played hands rather than from randomly filled boards: a random
11-card board is usually already fouled or unreachable, and a teacher trained
on states its own policy never visits is teaching the wrong distribution.  The
generating policy is recorded in the manifest, because it IS part of the label
distribution and a later generation will want to know which one produced this
corpus.

Which policy plays those eleven decisions turns out to matter enormously.
Measured over 180 finished boards:

    root policy          foul     royalty   Fantasyland entry
    mc_policy(sims=4)     8.9%      0.99          1.1%
    mc_policy(sims=32)    3.3%      1.45          1.1%
    hu_policy (m7v5)     22.8%      5.25         29.4%

The Monte-Carlo referee reaches Fantasyland once in ninety hands.  Heads-up
production measures 24.5%, and the frozen heads-up models reproduce that here.
A corpus built on referee roots therefore teaches T3 almost nothing about the
Fantasyland race, which is a large part of what the street is deciding -- so
``hu`` is the default, and ``mc`` is kept only for the runs that used it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..action_key import action_key
from ..cards import create_deck
from .exact import _HOLDOUT_SEED_SALT, evaluate_t3
from .hu_bridge import hu_policy
from .interior import (
    evaluate_t3_middle,
    exact_interior,
    mc_interior,
    model_interior,
    sample_middle_draws,
)
from .mc import mc_policy
from .scoring import DEFAULT_FL_EV_PER_PAIR
from .seating import ACT_ORDER, SEAT_BB, SEAT_BTN, Seat3
from .world import ThreeMaxObservation, WorldState3

TEACHER_SCHEMA = "regular_ofc_3max_t3_teacher_v1"
MANIFEST_SCHEMA = "regular_ofc_3max_teacher_manifest_v1"
RUN_ID = "regular_ofc_3max_t3_teacher"


@dataclass(frozen=True)
class TeacherConfig:
    samples: int = 64
    holdout_samples: int = 64
    rollout_policy_sims: int = 4
    root_policy: str = "hu"
    fl_ev_per_pair: Mapping[int, float] = None  # type: ignore[assignment]
    # The BB seat only.  Its tree has one interior opponent node -- the BTN's
    # T3 -- so its label is a function of who stands there, and the answer has
    # to travel with the corpus: two BB corpora built with different interior
    # policies are different datasets, not two samples of one.
    seat: Seat3 = SEAT_BTN
    strata: int = 8
    per_stratum: int = 8
    # Measured 2026-08-13: every interior policy down to uniform random yields
    # a label the (T3, BB) probe cannot tell from the exact solver's, and the
    # model is NOT the cheap one -- its encoder costs 78 ms per interior
    # decision against the solver's 157 ms, six times the referee's.  So the
    # default is the referee, not the model.  See
    # docs/three_max_interior_probe_20260813.md.
    interior: str = "mc"
    interior_model: str = ""
    interior_samples: int = 32
    interior_sims: int = 4

    def __post_init__(self) -> None:
        if self.root_policy not in ("hu", "mc"):
            raise ValueError(f"unknown root policy: {self.root_policy!r}")
        if self.seat not in (SEAT_BTN, SEAT_BB):
            raise ValueError(f"no teacher for seat {self.seat!r}")
        if self.fl_ev_per_pair is None:
            object.__setattr__(
                self, "fl_ev_per_pair", dict(DEFAULT_FL_EV_PER_PAIR)
            )
        if self.samples <= 0 or self.holdout_samples <= 0:
            raise ValueError("sample counts must be positive")
        if self.seat == SEAT_BB:
            if self.strata <= 0 or self.per_stratum <= 0:
                raise ValueError("strata and per_stratum must be positive")
            if self.interior not in ("model", "exact", "mc"):
                raise ValueError(f"unknown interior policy: {self.interior!r}")
            if self.interior == "model" and not self.interior_model:
                raise ValueError("interior='model' needs --interior-model")

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "samples": self.samples,
                "holdout_samples": self.holdout_samples,
                "rollout_policy_sims": self.rollout_policy_sims,
                "root_policy": self.root_policy,
                "fl_ev_per_pair": {
                    str(k): v for k, v in sorted(self.fl_ev_per_pair.items())
                },
                **(
                    {}
                    if self.seat == SEAT_BTN
                    else {
                        "seat": self.seat,
                        "strata": self.strata,
                        "per_stratum": self.per_stratum,
                        "interior": self.interior,
                        "interior_model": self.interior_model,
                        "interior_samples": self.interior_samples,
                        "interior_sims": self.interior_sims,
                    }
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def interior_policy(self):
        """The policy that fills the BB tree's one interior node."""
        if self.interior == "exact":
            return exact_interior(samples=self.interior_samples)
        if self.interior == "mc":
            return mc_interior(sims=self.interior_sims)
        return model_interior(self.interior_model)


def root_policy_for(config: TeacherConfig):
    if config.root_policy == "hu":
        return hu_policy()
    return mc_policy(sims=config.rollout_policy_sims)


def sample_root(seed: int, *, config: TeacherConfig) -> WorldState3:
    """Play a hand up to ``config.seat``'s T3 decision and stop there."""
    policy = root_policy_for(config)
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(seed)))
    while True:
        slot = world.current_slot()
        if slot.street == "T3" and slot.seat == config.seat:
            return world
        world = world.apply(
            policy(world.observe(), seed * 1_000_003 + slot.decision_index)
        )


def _board_payload(board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


def _rank_middle(
    observation: ThreeMaxObservation, *, seed: int, config: TeacherConfig
) -> list:
    """Rank the BB seat's T3 actions, and hold the winner's value out.

    The winner's ``ev`` is the max of estimates sharing one set of strata and
    so is biased upward by selection, exactly as it is for the BTN.  The
    holdout re-scores it on strata the ranking never saw -- including a fresh
    interior deal, because the interior node is part of what is being sampled.
    """
    policy = config.interior_policy()
    unseen = observation.unknown_cards()
    ranked = evaluate_t3_middle(
        observation,
        interior_policy=policy,
        draws=sample_middle_draws(
            unseen,
            strata=config.strata,
            per_stratum=config.per_stratum,
            seed=seed,
        ),
        seed=seed,
        fl_ev_per_pair=config.fl_ev_per_pair,
    )
    holdout = evaluate_t3_middle(
        observation,
        interior_policy=policy,
        draws=sample_middle_draws(
            unseen,
            strata=config.strata,
            per_stratum=config.per_stratum,
            seed=seed ^ _HOLDOUT_SEED_SALT,
        ),
        seed=seed ^ _HOLDOUT_SEED_SALT,
        fl_ev_per_pair=config.fl_ev_per_pair,
    )
    rescored = {candidate.action: candidate.ev for candidate in holdout}
    winner = ranked[0]
    ranked[0] = replace(winner, ev_holdout=rescored[winner.action])
    return ranked


def label_root(
    observation: ThreeMaxObservation,
    *,
    seed: int,
    config: TeacherConfig,
) -> dict[str, Any]:
    """Solve one root and return its JSONL record."""
    if config.seat == SEAT_BB:
        ranked = _rank_middle(observation, seed=seed, config=config)
    else:
        ranked = evaluate_t3(
            observation,
            samples=config.samples,
            seed=seed,
            fl_ev_per_pair=config.fl_ev_per_pair,
            holdout_samples=config.holdout_samples,
        )
    actions = [
        {
            "placements": [[card, row] for card, row in candidate.action.placements],
            "discards": list(candidate.action.discards),
            "key": action_key(candidate.action).to_token(),
            "ev": candidate.ev,
            "board": _board_payload(candidate.board),
        }
        for candidate in ranked
    ]
    gap = ranked[0].ev - ranked[1].ev if len(ranked) > 1 else 0.0
    return {
        "schema": TEACHER_SCHEMA,
        "seat": observation.seat,
        "street": observation.street,
        "hero_board": _board_payload(observation.hero_board),
        "opponent_boards": [
            _board_payload(board) for board in observation.opponent_boards
        ],
        "opponent_seats": list(observation.opponent_seats),
        "dealt": list(observation.dealt_cards),
        "hero_private_discards": list(observation.hero_private_discards),
        "unseen_count": observation.unknown_card_count(),
        "seed": seed,
        "samples": ranked[0].samples,
        "best_action": 0,
        "score_gap": gap,
        "ev_holdout": ranked[0].ev_holdout,
        "actions": actions,
    }


def generate(
    *,
    count: int,
    base_seed: int,
    config: TeacherConfig,
    progress_every: int = 0,
) -> Iterator[dict[str, Any]]:
    """Yield ``count`` labelled roots, one per seed in a contiguous block."""
    if count <= 0:
        raise ValueError("count must be positive")
    for index in range(count):
        seed = base_seed + index
        world = sample_root(seed, config=config)
        yield label_root(world.observe(), seed=seed, config=config)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"  {index + 1}/{count} roots", flush=True)


def write_corpus(
    *,
    output: Path,
    count: int,
    base_seed: int,
    config: TeacherConfig,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Write a JSONL corpus plus a sibling ``.manifest.json``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    digest = hashlib.sha256()
    rows = 0
    actions = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in generate(
            count=count,
            base_seed=base_seed,
            config=config,
            progress_every=progress_every,
        ):
            line = json.dumps(record, sort_keys=True, separators=(",", ":"))
            handle.write(line + "\n")
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
            rows += 1
            actions += len(record["actions"])

    elapsed = time.time() - started
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "run_id": RUN_ID,
        "seat": config.seat,
        "street": "T3",
        "rows": rows,
        "actions": actions,
        "base_seed": base_seed,
        "seed_block": [base_seed, base_seed + count - 1],
        "config": {
            "samples": config.samples,
            "holdout_samples": config.holdout_samples,
            "rollout_policy_sims": config.rollout_policy_sims,
            "root_policy": config.root_policy,
            "fl_ev_per_pair": {
                str(k): v for k, v in sorted(config.fl_ev_per_pair.items())
            },
            **(
                {}
                if config.seat == SEAT_BTN
                else {
                    "seat": config.seat,
                    "strata": config.strata,
                    "per_stratum": config.per_stratum,
                    "interior": config.interior,
                    "interior_model": config.interior_model,
                    "interior_samples": config.interior_samples,
                    "interior_sims": config.interior_sims,
                }
            ),
        },
        "config_fingerprint": config.fingerprint(),
        "root_generator": (
            "played hand under the frozen heads-up m7v5 models (one opponent "
            f"shown per decision) up to the {config.seat.upper()} T3 decision"
            if config.root_policy == "hu"
            else f"played hand under mc_policy(sims={config.rollout_policy_sims}) "
            f"up to the {config.seat.upper()} T3 decision"
        ),
        "labeler": (
            "three_max.exact.evaluate_t3 (exact T4 round, backward induction)"
            if config.seat == SEAT_BTN
            else (
                "three_max.interior.evaluate_t3_middle (exact T4 round; the "
                f"BTN's interior T3 played by {config.interior})"
            )
        ),
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_root": round(elapsed / max(rows, 1), 4),
        "content_sha256": digest.hexdigest(),
        "caveats": [
            "Root distribution carries the generating policy's biases; a later "
            "generation should re-sample roots from its own chain.",
            "actions[].ev is an in-sample value under a shared draw set; only "
            "ev_holdout is unbiased for the winner.",
        ]
        + (
            []
            if config.seat == SEAT_BTN
            else [
                "The BB label is exact only BELOW the interior node; the BTN's "
                "T3 inside the tree is played by a policy, so the label "
                "inherits that policy's error. See the interior probe report."
            ]
        ),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=8_100_000)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--holdout-samples", type=int, default=64)
    parser.add_argument("--rollout-policy-sims", type=int, default=4)
    parser.add_argument("--root-policy", choices=("hu", "mc"), default="hu")
    parser.add_argument("--seat", choices=(SEAT_BTN, SEAT_BB), default=SEAT_BTN)
    parser.add_argument("--strata", type=int, default=8)
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument("--interior", choices=("model", "exact", "mc"), default="mc")
    parser.add_argument("--interior-model", default="")
    parser.add_argument("--interior-samples", type=int, default=32)
    parser.add_argument("--interior-sims", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()

    config = TeacherConfig(
        samples=args.samples,
        holdout_samples=args.holdout_samples,
        rollout_policy_sims=args.rollout_policy_sims,
        root_policy=args.root_policy,
        seat=args.seat,
        strata=args.strata,
        per_stratum=args.per_stratum,
        interior=args.interior,
        interior_model=args.interior_model,
        interior_samples=args.interior_samples,
        interior_sims=args.interior_sims,
    )
    manifest = write_corpus(
        output=args.output,
        count=args.count,
        base_seed=args.base_seed,
        config=config,
        progress_every=args.progress_every,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
