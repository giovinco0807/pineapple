"""Run the bounded M3.0 exact-T4 correctness and latency pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .action_key import action_key
from .action_space import generate_actions, generate_turn_actions
from .cards import create_deck
from .hu_infoset import ActorObservation, WorldState
from .hu_late_street_teacher import evaluate_t4_sequential_actions
from .hu_m3_t4_runtime import (
    HuM3T4ExactSolver,
    HuM3T4RuntimeConfig,
    T4ExactDecision,
)
from .state import Board
from .teacher import DEFAULT_FL_EV, evaluate_turn_actions


PILOT_SCHEMA = "hu_m30_t4_runtime_pilot_v1"
DEFAULT_SEED = 2026071601
DEFAULT_SEED_STRIDE = 1_000_003
FIRST_P95_GATE_MS = 75.0
FIRST_P99_GATE_MS = 100.0
SECOND_P99_GATE_MS = 2.0
_FORBIDDEN_OBSERVATION_FIELDS = {
    "opponent_private_discards",
    "true_dead_cards",
    "remaining_deck",
    "world_state",
    "replay_truth",
}


def generate_balanced_t4_observations(
    *,
    states: int,
    seed: int,
    seed_stride: int,
) -> list[ActorObservation]:
    """Generate live-order T4 roots without passing hidden truth to a policy."""

    if states <= 0 or states % 2:
        raise ValueError("states must be a positive even number")
    if seed_stride <= 0:
        raise ValueError("seed_stride must be positive")
    observations: list[ActorObservation] = []
    for hand_index in range(states // 2):
        hand_seed = seed + hand_index * seed_stride
        rng = random.Random(hand_seed)
        action_rng = random.Random(hand_seed ^ 0x5A17_4D30)
        deck = create_deck(shuffle=True, rng=rng)
        cursor = 0
        boards = [Board.from_rows(), Board.from_rows()]
        private_discards: list[list[str]] = [[], []]

        for player in (0, 1):
            dealt = deck[cursor : cursor + 5]
            cursor += 5
            actions = generate_actions(boards[player], dealt)
            action = action_rng.choice(actions)
            boards[player] = boards[player].place(action.placements)
            private_discards[player].extend(action.discards)

        for round_index in range(1, 4):
            for player in (0, 1):
                dealt = deck[cursor : cursor + 3]
                cursor += 3
                actions = generate_turn_actions(boards[player], dealt)
                action = action_rng.choice(actions)
                boards[player] = boards[player].place(action.placements)
                private_discards[player].extend(action.discards)

        first_dealt = deck[cursor : cursor + 3]
        cursor += 3
        first_world = WorldState(
            boards=(boards[0], boards[1]),
            private_discards=(tuple(private_discards[0]), tuple(private_discards[1])),
            street="T4",
            next_player=0,
        )
        first_observation = first_world.observe(0, first_dealt)
        observations.append(first_observation)
        first_action = _legacy_t4_action(first_observation)
        boards[0] = boards[0].place(first_action.placements)
        private_discards[0].extend(first_action.discards)

        second_dealt = deck[cursor : cursor + 3]
        second_world = WorldState(
            boards=(boards[0], boards[1]),
            private_discards=(tuple(private_discards[0]), tuple(private_discards[1])),
            street="T4",
            next_player=1,
        )
        observations.append(second_world.observe(1, second_dealt))
    return observations


def run_pilot(
    *,
    solver: HuM3T4ExactSolver,
    states: int,
    seed: int,
    seed_stride: int,
    python_parity_roots: int,
    determinism_roots: int,
) -> dict[str, Any]:
    observations = generate_balanced_t4_observations(
        states=states,
        seed=seed,
        seed_stride=seed_stride,
    )
    started = time.perf_counter()
    decisions: list[T4ExactDecision] = []
    rows: list[dict[str, Any]] = []
    for root_index, observation in enumerate(observations):
        decision = solver.solve(observation)
        decisions.append(decision)
        legacy_action = _legacy_t4_action(observation)
        legacy_key = action_key(legacy_action).to_token()
        exact_values = {row.action_key: row.ev for row in decision.action_values}
        if legacy_key not in exact_values:
            raise AssertionError("legacy T4 action is absent from exact legal values")
        observation_payload = observation.to_dict()
        if _FORBIDDEN_OBSERVATION_FIELDS.intersection(observation_payload):
            raise AssertionError("pilot observation contains a forbidden hidden-truth field")
        rows.append(
            {
                "root_index": root_index,
                "source_seed": seed + (root_index // 2) * seed_stride,
                "seat": observation.seat,
                "observation": observation_payload,
                "observation_fingerprint": observation.fingerprint(),
                "selected_action_key": decision.selected_action_key,
                "selected_ev": decision.selected_ev,
                "hero_legal_action_count": len(decision.action_values),
                "opponent_response_action_count": _opponent_response_action_count(
                    observation
                ),
                "action_values": {
                    row.action_key: row.ev for row in decision.action_values
                },
                "legacy_action_key": legacy_key,
                "legacy_action_ev_under_exact_value": exact_values[legacy_key],
                "legacy_regret_under_exact_value": (
                    decision.selected_ev - exact_values[legacy_key]
                ),
                "action_changed_from_legacy": legacy_key
                != decision.selected_action_key,
                "total_latency_ms": decision.total_latency_ms,
                "native_latency_ms": decision.native_latency_ms,
                "result_digest": decision.result_digest,
            }
        )
    scalar_seconds = time.perf_counter() - started

    batch_started = time.perf_counter()
    batch_decisions = solver.solve_many(observations)
    batch_seconds = time.perf_counter() - batch_started
    scalar_batch_parity = all(
        scalar.result_digest == batched.result_digest
        for scalar, batched in zip(decisions, batch_decisions, strict=True)
    )

    deterministic_count = min(max(determinism_roots, 0), len(observations))
    deterministic_rerun = [
        solver.solve(observation) for observation in observations[:deterministic_count]
    ]
    deterministic = all(
        decisions[index].result_digest == deterministic_rerun[index].result_digest
        for index in range(deterministic_count)
    )

    parity_count = min(max(python_parity_roots, 0), len(observations))
    python_parity = True
    python_parity_details: list[dict[str, Any]] = []
    for index in range(parity_count):
        observation = observations[index]
        expected = evaluate_t4_sequential_actions(observation)
        expected_values = {
            str(row["action_key"]): float(row["score"])
            for row in expected["actions"]
        }
        actual_values = {
            row.action_key: row.ev for row in decisions[index].action_values
        }
        matches = (
            expected["selected_action_key"] == decisions[index].selected_action_key
            and expected_values.keys() == actual_values.keys()
            and all(
                math.isclose(expected_values[key], actual_values[key], rel_tol=0.0, abs_tol=1e-12)
                for key in expected_values
            )
        )
        python_parity &= matches
        python_parity_details.append(
            {
                "root_index": index,
                "seat": observation.seat,
                "matches": matches,
            }
        )

    by_seat = {
        seat: _seat_summary([row for row in rows if row["seat"] == seat])
        for seat in ("first", "second")
    }
    fingerprints = [row["observation_fingerprint"] for row in rows]
    gates = {
        "balanced_seats": by_seat["first"]["count"]
        == by_seat["second"]["count"],
        "unique_observation_fingerprints": len(set(fingerprints)) == len(fingerprints),
        "all_first_exact_2024": all(
            decision.future_count == 2024
            for decision in decisions
            if decision.seat == "first"
        ),
        "all_second_terminal_exact": all(
            decision.future_count == 1
            for decision in decisions
            if decision.seat == "second"
        ),
        "python_reference_parity": python_parity,
        "scalar_batch_parity": scalar_batch_parity,
        "deterministic_rerun": deterministic,
        "second_legacy_exact_action_parity": all(
            not row["action_changed_from_legacy"]
            for row in rows
            if row["seat"] == "second"
        ),
        "legacy_regret_nonnegative": all(
            float(row["legacy_regret_under_exact_value"]) >= -1e-12
            for row in rows
        ),
        "first_p95_latency": by_seat["first"]["latency_p95_ms"]
        <= FIRST_P95_GATE_MS,
        "first_p99_latency": by_seat["first"]["latency_p99_ms"]
        <= FIRST_P99_GATE_MS,
        "second_p99_latency": by_seat["second"]["latency_p99_ms"]
        <= SECOND_P99_GATE_MS,
    }
    return {
        "schema": PILOT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "states": states,
        "seed": seed,
        "seed_stride": seed_stride,
        "seat_counts": {
            "first": by_seat["first"]["count"],
            "second": by_seat["second"]["count"],
        },
        "engine_version": solver.engine_version,
        "native_library": str(solver.library_path),
        "native_library_sha256": solver.library_sha256,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scalar_seconds": scalar_seconds,
        "scalar_roots_per_second": states / scalar_seconds,
        "batch_seconds": batch_seconds,
        "batch_roots_per_second": states / batch_seconds,
        "python_parity_roots": parity_count,
        "python_parity_details": python_parity_details,
        "determinism_roots": deterministic_count,
        "by_seat": by_seat,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "teacher_value_status": "diagnostic_not_match_EV",
        "legacy_regret_scope": "belief_EV_diagnostic_not_realized_match_EV",
        "rows_digest": _digest(rows),
        "rows": rows,
    }


def _legacy_t4_action(observation: ActorObservation):
    opponent = (
        observation.opponent_public_board
        if observation.opponent_public_board.is_complete()
        else None
    )
    ranked = evaluate_turn_actions(
        observation.hero_board,
        observation.dealt_cards,
        opponent_board=opponent,
        fl_ev=DEFAULT_FL_EV,
    )
    if not ranked:
        raise RuntimeError("legacy T4 produced no legal action")
    return ranked[0].action


def _opponent_response_action_count(observation: ActorObservation) -> int:
    if observation.to_act_order == "second":
        return 0
    known = set(observation.known_unavailable_cards())
    future = tuple(card for card in create_deck(shuffle=False) if card not in known)[:3]
    return len(generate_turn_actions(observation.opponent_public_board, future))


def _seat_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(float(row["total_latency_ms"]) for row in rows)
    regrets = [float(row["legacy_regret_under_exact_value"]) for row in rows]
    changed = [row for row in rows if row["action_changed_from_legacy"]]
    geometry: dict[str, int] = {}
    for row in rows:
        key = (
            f"hero{row['hero_legal_action_count']}_"
            f"response{row['opponent_response_action_count']}"
        )
        geometry[key] = geometry.get(key, 0) + 1
    return {
        "count": len(rows),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "latency_p99_ms": _percentile(latencies, 0.99),
        "latency_max_ms": max(latencies) if latencies else None,
        "changed_from_legacy_count": len(changed),
        "changed_from_legacy_rate": len(changed) / len(rows) if rows else 0.0,
        "mean_legacy_regret_under_exact_value": (
            statistics.fmean(regrets) if regrets else 0.0
        ),
        "mean_changed_legacy_regret_under_exact_value": (
            statistics.fmean(
                float(row["legacy_regret_under_exact_value"]) for row in changed
            )
            if changed
            else 0.0
        ),
        "max_legacy_regret_under_exact_value": max(regrets) if regrets else 0.0,
        "geometry_counts": dict(sorted(geometry.items())),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, math.ceil(len(values) * fraction) - 1))
    return float(values[index])


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--states", type=int, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seed-stride", type=int, default=DEFAULT_SEED_STRIDE)
    parser.add_argument("--python-parity-roots", type=int, default=4)
    parser.add_argument("--determinism-roots", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    solver = HuM3T4ExactSolver(
        HuM3T4RuntimeConfig(
            library_path=args.library,
            expected_library_sha256=args.library_sha256,
        )
    )
    result = run_pilot(
        solver=solver,
        states=args.states,
        seed=args.seed,
        seed_stride=args.seed_stride,
        python_parity_roots=args.python_parity_roots,
        determinism_roots=args.determinism_roots,
    )
    _write_json_atomic(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    if not result["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
