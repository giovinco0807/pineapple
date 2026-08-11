"""Run a bounded local correctness pilot for the M2 sequential T3 teacher.

The pilot deliberately does not load an AI profile or model artifact.  It
creates fresh first- and second-seat T3 information sets from deterministic
deck seeds, using a documented canonical legal action only to advance the
public game.  Teacher values in this report are correctness diagnostics, not
realized heads-up match EV and not policy-promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .action_key import action_key
from .action_space import Action, generate_actions
from .cards import create_deck
from .hu_belief import HiddenCardParticleBatch, sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_turn3_joint_exact_teacher import (
    JointExactConfig,
    evaluate_t3_joint_exact_actions,
)
from .state import Board


M2_PILOT_SCHEMA = "hu_m2_teacher_correctness_pilot_v1"
ROOT_GENERATION_POLICY = "canonical_min_action_key_v1"
TEACHER_VALUE_STATUS = "diagnostic_not_match_EV"
MAX_HANDS = 4
MAX_SAMPLE_COUNT = 8


@dataclass(frozen=True)
class M2PilotConfig:
    hands: int = 4
    seed_start: int = 2026071301
    seed_stride: int = 1_000_003
    teacher_seed: int = 20260713
    candidate_seed: int = 2026071301
    evaluation_seed: int = 2026071302
    candidate_samples: int = 1
    evaluation_samples: int = 1
    downstream_t3_samples: int = 1
    downstream_t4_samples: int = 1
    run_id: str = "hu-m2-correctness-pilot"

    def __post_init__(self) -> None:
        if isinstance(self.hands, bool) or not isinstance(self.hands, int):
            raise TypeError("hands must be an integer")
        if not 1 <= self.hands <= MAX_HANDS:
            raise ValueError(f"hands must be between 1 and {MAX_HANDS}")
        for name in (
            "seed_start",
            "seed_stride",
            "teacher_seed",
            "candidate_seed",
            "evaluation_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive")
        for name in (
            "candidate_samples",
            "evaluation_samples",
            "downstream_t3_samples",
            "downstream_t4_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 1 <= value <= MAX_SAMPLE_COUNT:
                raise ValueError(
                    f"{name} must be between 1 and {MAX_SAMPLE_COUNT} in the bounded pilot"
                )
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must not be empty")

    def hand_seed(self, hand_index: int) -> int:
        if not 0 <= hand_index < self.hands:
            raise IndexError("hand_index is outside the configured pilot")
        return self.seed_start + hand_index * self.seed_stride

    def to_dict(self) -> dict[str, Any]:
        return {
            "hands": self.hands,
            "root_count": self.hands * 2,
            "seed_start": self.seed_start,
            "seed_stride": self.seed_stride,
            "teacher_seed": self.teacher_seed,
            "candidate_seed": self.candidate_seed,
            "evaluation_seed": self.evaluation_seed,
            "candidate_samples": self.candidate_samples,
            "evaluation_samples": self.evaluation_samples,
            "downstream_t3_samples": self.downstream_t3_samples,
            "downstream_t4_samples": self.downstream_t4_samples,
            "run_id": self.run_id,
        }


def generate_fresh_t3_roots(hand_seed: int) -> tuple[ActorObservation, ActorObservation]:
    """Generate the live T3 first/second roots for one deterministic fresh hand."""

    if isinstance(hand_seed, bool) or not isinstance(hand_seed, int):
        raise TypeError("hand_seed must be an integer")
    deck = create_deck(shuffle=True, rng=random.Random(hand_seed))
    cursor = 0
    boards = [Board.from_rows(), Board.from_rows()]
    private_discards: list[list[str]] = [[], []]

    def deal(count: int) -> tuple[str, ...]:
        nonlocal cursor
        cards = tuple(deck[cursor : cursor + count])
        cursor += count
        if len(cards) != count:
            raise RuntimeError("fresh pilot deck was exhausted")
        return cards

    def advance(player: int, dealt: tuple[str, ...]) -> None:
        action = _canonical_generation_action(boards[player], dealt)
        boards[player] = boards[player].place(action.placements)
        private_discards[player].extend(action.discards)

    for player in (0, 1):
        advance(player, deal(5))
    for _street in ("T1", "T2"):
        for player in (0, 1):
            advance(player, deal(3))

    first_deal = deal(3)
    first = ActorObservation(
        hero_board=boards[0],
        opponent_public_board=boards[1],
        dealt_cards=first_deal,
        hero_private_discards=tuple(private_discards[0]),
        seat="first",
        street="T3",
        to_act_order="first",
    )
    advance(0, first_deal)
    second = ActorObservation(
        hero_board=boards[1],
        opponent_public_board=boards[0],
        dealt_cards=deal(3),
        hero_private_discards=tuple(private_discards[1]),
        seat="second",
        street="T3",
        to_act_order="second",
    )
    return first, second


def run_m2_correctness_pilot(
    config: M2PilotConfig,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the bounded pilot and atomically persist its JSON report."""

    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"M2 pilot output already exists: {output_path}")
    started = time.perf_counter()
    roots: list[tuple[int, int, ActorObservation]] = []
    hand_seeds: list[int] = []
    for hand_index in range(config.hands):
        hand_seed = config.hand_seed(hand_index)
        hand_seeds.append(hand_seed)
        roots.extend(
            (hand_index, hand_seed, observation)
            for observation in generate_fresh_t3_roots(hand_seed)
        )

    root_records: list[dict[str, Any]] = []
    root_seconds: list[float] = []
    candidate_rng_keys: set[str] = set()
    evaluation_rng_keys: set[str] = set()
    for hand_index, hand_seed, observation in roots:
        evaluated, elapsed, overlap = _evaluate_root(
            config, hand_index, hand_seed, observation
        )
        root_seconds.append(elapsed)
        candidate_rng_keys.update(evaluated["candidate_rng_keys"])
        evaluation_rng_keys.update(evaluated["evaluation_rng_keys"])
        root_records.append(
            {
                "hand_index": hand_index,
                "hand_seed": hand_seed,
                "seat": observation.seat,
                "root_fingerprint": observation.fingerprint(),
                "seconds": elapsed,
                "candidate_evaluation_rng_overlap_count": overlap,
                "candidate_belief_digest": evaluated["candidate_belief_digest"],
                "evaluation_belief_digest": evaluated["evaluation_belief_digest"],
                "selected_action_key": evaluated["result"]["selected_action_key"],
                "selected_action_evaluation_score": evaluated["result"][
                    "selected_action_evaluation_score"
                ],
                "teacher_value_status": TEACHER_VALUE_STATUS,
                "teacher_result_digest": _digest(evaluated["result"]),
            }
        )

    rerun_root = roots[0][2]
    first_rerun, _elapsed, _overlap = _evaluate_root(
        config, roots[0][0], roots[0][1], rerun_root
    )
    second_rerun, _elapsed, _overlap = _evaluate_root(
        config, roots[0][0], roots[0][1], rerun_root
    )
    first_digest = _digest(first_rerun["result"])
    second_digest = _digest(second_rerun["result"])
    rerun_match = first_digest == second_digest

    fingerprints = [observation.fingerprint() for _index, _seed, observation in roots]
    unique_fingerprints = sorted(set(fingerprints))
    seats = Counter(observation.seat for _index, _seed, observation in roots)
    per_root_overlap = sum(
        int(record["candidate_evaluation_rng_overlap_count"])
        for record in root_records
    )
    global_overlap = len(candidate_rng_keys & evaluation_rng_keys)
    gates = {
        "balanced_first_second_roots": seats == Counter(
            {"first": config.hands, "second": config.hands}
        ),
        "unique_root_fingerprints": len(unique_fingerprints) == len(roots),
        "candidate_evaluation_rng_overlap_zero": per_root_overlap == 0
        and global_overlap == 0,
        "deterministic_rerun_match": rerun_match,
        "teacher_values_labeled_diagnostic": all(
            record["teacher_value_status"] == TEACHER_VALUE_STATUS
            for record in root_records
        ),
        "bounded_local_scope": config.hands <= MAX_HANDS
        and max(
            config.candidate_samples,
            config.evaluation_samples,
            config.downstream_t3_samples,
            config.downstream_t4_samples,
        )
        <= MAX_SAMPLE_COUNT,
    }
    report = {
        "schema": M2_PILOT_SCHEMA,
        "status": "smoke_pass" if all(gates.values()) else "smoke_fail",
        "completion_gate_aggregate": False,
        "gate_scope": "fresh_root_rng_determinism_and_speed_only",
        "phase": "M2_correctness_pilot",
        "teacher_values": TEACHER_VALUE_STATUS,
        "promotion_evidence": False,
        "match_ev_reported": False,
        "configuration": config.to_dict(),
        "hand_seeds": hand_seeds,
        "root_generation": {
            "policy": ROOT_GENERATION_POLICY,
            "profile_selection": "none",
            "current_profile_read": False,
            "model_artifacts_loaded": [],
        },
        "compute_scope": {
            "local_bounded_pilot": True,
            "cloud_actions_performed": False,
            "max_hands": MAX_HANDS,
            "max_sample_count_per_dimension": MAX_SAMPLE_COUNT,
        },
        "root_count": len(roots),
        "seat_counts": {"first": seats["first"], "second": seats["second"]},
        "unique_root_fingerprint_count": len(unique_fingerprints),
        "unique_root_fingerprints": unique_fingerprints,
        "candidate_evaluation_rng_overlap_count": per_root_overlap,
        "global_candidate_evaluation_rng_overlap_count": global_overlap,
        "deterministic_rerun": {
            "checked_root_fingerprint": rerun_root.fingerprint(),
            "first_result_digest": first_digest,
            "second_result_digest": second_digest,
            "match": rerun_match,
        },
        "seconds_per_root": {
            "count": len(root_seconds),
            "p50": _percentile(root_seconds, 0.50),
            "p95": _percentile(root_seconds, 0.95),
            "max": max(root_seconds),
            "percentile_method": "linear_between_order_statistics",
        },
        "total_seconds": time.perf_counter() - started,
        "gates": gates,
        "roots": root_records,
    }
    _atomic_write_json(output_path, report, overwrite=overwrite)
    return report


def _evaluate_root(
    config: M2PilotConfig,
    hand_index: int,
    hand_seed: int,
    observation: ActorObservation,
) -> tuple[dict[str, Any], float, int]:
    root_run_id = (
        f"{config.run_id}:hand={hand_index}:seed={hand_seed}:"
        f"root={observation.fingerprint()}"
    )
    candidate = sample_hidden_card_particles(
        observation,
        base_seed=config.candidate_seed,
        run_id=f"{root_run_id}:candidate_selection",
        sample_count=config.candidate_samples,
    )
    evaluation = sample_hidden_card_particles(
        observation,
        base_seed=config.evaluation_seed,
        run_id=f"{root_run_id}:locked_evaluation",
        sample_count=config.evaluation_samples,
    )
    candidate_keys = _rng_keys(candidate)
    evaluation_keys = _rng_keys(evaluation)
    overlap = len(candidate_keys & evaluation_keys)
    teacher_config = JointExactConfig(
        candidate_samples=config.candidate_samples,
        evaluation_samples=config.evaluation_samples,
        downstream_t3_samples=config.downstream_t3_samples,
        downstream_t4_samples=config.downstream_t4_samples,
        seed=config.teacher_seed,
        candidate_seed=config.candidate_seed,
        evaluation_seed=config.evaluation_seed,
        run_id=root_run_id,
        seat=observation.seat,
        to_act_order=observation.to_act_order,
    )
    started = time.perf_counter()
    result = evaluate_t3_joint_exact_actions(
        observation=observation,
        config=teacher_config,
        candidate_belief_batch=candidate,
        evaluation_belief_batch=evaluation,
    )
    elapsed = time.perf_counter() - started
    return (
        {
            "result": result,
            "candidate_belief_digest": candidate.digest(),
            "evaluation_belief_digest": evaluation.digest(),
            "candidate_rng_keys": candidate_keys,
            "evaluation_rng_keys": evaluation_keys,
        },
        elapsed,
        overlap,
    )


def _canonical_generation_action(board: Board, dealt: tuple[str, ...]) -> Action:
    actions = generate_actions(board, dealt)
    if not actions:
        raise RuntimeError("fresh root generator found no legal action")
    return min(actions, key=lambda action: action_key(action).sort_key())


def _rng_keys(batch: HiddenCardParticleBatch) -> set[str]:
    return {particle.rng_key_digest for particle in batch.particles}


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in ordered):
        raise ValueError("timings must be finite and non-negative")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def _atomic_write_json(
    output: Path, payload: dict[str, Any], *, overwrite: bool
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"M2 pilot output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=2026071301)
    parser.add_argument("--seed-stride", type=int, default=1_000_003)
    parser.add_argument("--teacher-seed", type=int, default=20260713)
    parser.add_argument("--candidate-seed", type=int, default=2026071301)
    parser.add_argument("--evaluation-seed", type=int, default=2026071302)
    parser.add_argument("--candidate-samples", type=int, default=1)
    parser.add_argument("--evaluation-samples", type=int, default=1)
    parser.add_argument("--downstream-t3-samples", type=int, default=1)
    parser.add_argument("--downstream-t4-samples", type=int, default=1)
    parser.add_argument("--run-id", default="hu-m2-correctness-pilot")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = M2PilotConfig(
        hands=args.hands,
        seed_start=args.seed_start,
        seed_stride=args.seed_stride,
        teacher_seed=args.teacher_seed,
        candidate_seed=args.candidate_seed,
        evaluation_seed=args.evaluation_seed,
        candidate_samples=args.candidate_samples,
        evaluation_samples=args.evaluation_samples,
        downstream_t3_samples=args.downstream_t3_samples,
        downstream_t4_samples=args.downstream_t4_samples,
        run_id=args.run_id,
    )
    report = run_m2_correctness_pilot(
        config, args.output, overwrite=bool(args.force)
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "output": str(args.output),
                "root_count": report["root_count"],
                "teacher_values": report["teacher_values"],
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
