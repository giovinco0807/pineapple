"""Validate the bounded M3.1 T3 permutation and execution-mode contract.

This is a correctness smoke, not a strength pilot.  It uses deliberately
constrained legal geometry, the pinned native engine, and the fixed 1/1/1
root CRN budget with exact T4 children.  Normal T3 chance remains Monte Carlo.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .action_key import action_key
from .action_space import generate_turn_actions
from .cards import ALL_CARDS
from .hu_infoset import ActorObservation
from .hu_m31_t3_runtime import (
    HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA,
    HuM31T3RuntimeConfig,
    HuM31T3SearchSolver,
    T3SearchDecision,
)
from .hu_turn3_joint_exact_teacher import evaluate_t3_joint_exact_actions
from .state import Board


CORRECTNESS_SCHEMA = "hu_m31_t3_step2_correctness_v1"


def constrained_t3_observation(to_act_order: str) -> ActorObservation:
    """Create a live-shape T3 root with three legal hero actions."""

    if to_act_order not in {"first", "second"}:
        raise ValueError("to_act_order must be first or second")
    opponent_count = 9 if to_act_order == "first" else 11
    cursor = 0
    hero_cards = ALL_CARDS[cursor : cursor + 9]
    cursor += 9
    opponent_cards = ALL_CARDS[cursor : cursor + opponent_count]
    cursor += opponent_count
    dealt = ALL_CARDS[cursor : cursor + 3]
    cursor += 3
    hero_discards = ALL_CARDS[cursor : cursor + 2]
    return ActorObservation(
        hero_board=_constrained_board(hero_cards),
        opponent_public_board=_constrained_board(opponent_cards),
        dealt_cards=dealt,
        hero_private_discards=hero_discards,
        seat=to_act_order,  # type: ignore[arg-type]
        street="T3",
        to_act_order=to_act_order,  # type: ignore[arg-type]
    )


def run_correctness_smoke(*, solver: HuM31T3SearchSolver) -> dict[str, Any]:
    """Run all dealt permutations in scalar, native-batch, and repeat modes."""

    observations_by_seat = {
        seat: [
            replace(base, dealt_cards=tuple(cards))
            for cards in itertools.permutations(base.dealt_cards)
        ]
        for seat in ("first", "second")
        for base in (constrained_t3_observation(seat),)
    }
    observations = [
        observation
        for seat in ("first", "second")
        for observation in observations_by_seat[seat]
    ]

    scalar_started = time.perf_counter()
    scalar = [solver.solve(observation) for observation in observations]
    scalar_seconds = time.perf_counter() - scalar_started

    batch_started = time.perf_counter()
    batched = solver.solve_many(observations)
    batch_seconds = time.perf_counter() - batch_started

    repeated = {
        seat: solver.solve(observations_by_seat[seat][0])
        for seat in ("first", "second")
    }
    scalar_by_seat = {
        "first": scalar[:6],
        "second": scalar[6:],
    }
    batch_by_seat = {
        "first": batched[:6],
        "second": batched[6:],
    }

    seat_summaries: dict[str, Any] = {}
    for seat in ("first", "second"):
        seat_summaries[seat] = _seat_summary(
            observations=observations_by_seat[seat],
            scalar=scalar_by_seat[seat],
            batched=batch_by_seat[seat],
            repeated=repeated[seat],
        )

    gates = {
        "both_seats_present": set(seat_summaries) == {"first", "second"},
        "six_permutations_per_seat": all(
            row["permutation_count"] == 6 for row in seat_summaries.values()
        ),
        "observation_fingerprint_invariant": all(
            row["observation_fingerprint_unique_count"] == 1
            for row in seat_summaries.values()
        ),
        "legal_action_set_invariant": all(
            row["legal_action_set_digest_unique_count"] == 1
            for row in seat_summaries.values()
        ),
        "semantic_result_invariant": all(
            row["semantic_result_digest_unique_count"] == 1
            for row in seat_summaries.values()
        ),
        "complete_action_value_map_invariant": all(
            row["action_value_map_unique_count"] == 1
            for row in seat_summaries.values()
        ),
        "selected_action_invariant": all(
            row["selected_action_key_unique_count"] == 1
            for row in seat_summaries.values()
        ),
        "belief_and_rng_invariant": all(
            row["belief_and_rng_unique_count"] == 1
            for row in seat_summaries.values()
        ),
        "child_information_set_count_invariant": all(
            row["child_information_set_count_unique_count"] == 1
            for row in seat_summaries.values()
        ),
        "all_local_index_key_mappings_valid": all(
            row["all_local_index_key_mappings_valid"]
            for row in seat_summaries.values()
        ),
        "scalar_batch_semantic_and_mapping_parity": all(
            row["scalar_batch_semantic_and_mapping_parity"]
            for row in seat_summaries.values()
        ),
        "deterministic_repeat_both_digest_scopes": all(
            row["deterministic_repeat_both_digest_scopes"]
            for row in seat_summaries.values()
        ),
        "ordered_mapping_provenance_retained": all(
            row["legal_action_order_digest_unique_count"] == 6
            and row["mapping_bound_result_digest_unique_count"] == 6
            for row in seat_summaries.values()
        ),
        "candidate_evaluation_rng_disjoint": all(
            row["candidate_evaluation_rng_disjoint"]
            for row in seat_summaries.values()
        ),
        "root_t3_is_mc": (
            solver.config.candidate_samples > 0
            and solver.config.evaluation_samples > 0
            and solver.config.downstream_t3_samples > 0
        ),
        "downstream_t4_is_exact": solver.search_config.downstream_t4_samples == 0,
    }
    return {
        "schema": CORRECTNESS_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "bounded_semantic_correctness_not_strength_or_match_ev",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "engine_version": solver.engine_version,
        "native_library": str(solver.library_path),
        "native_library_sha256": solver.library_sha256,
        "semantic_result_digest_schema": (
            HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA
        ),
        "search_contract": {
            "candidate_samples": solver.config.candidate_samples,
            "evaluation_samples": solver.config.evaluation_samples,
            "downstream_t3_samples": solver.config.downstream_t3_samples,
            "downstream_t4_samples": 0,
            "root_method": "common_random_monte_carlo",
            "candidate_evaluation_domains": "disjoint",
            "downstream_t4_method": "exact_native_enumeration",
            "teacher_value_status": "diagnostic_not_match_EV",
        },
        "exactness_matrix": [
            {
                "scope": "T4 first",
                "chance": "all C(24,3)=2024 opponent deals",
                "actions": "all legal hero and opponent placements",
                "classification": "exact under uniform exchangeable restart belief",
                "full_game_optimality_claimed": False,
            },
            {
                "scope": "T4 second",
                "chance": "terminal state",
                "actions": "all legal hero placements",
                "classification": "terminal exact",
                "full_game_optimality_claimed": False,
            },
            {
                "scope": "normal full-deck T3 runtime",
                "chance": "candidate/evaluation CRN particles",
                "actions": "all legal root actions; exact T4 children",
                "classification": "Monte Carlo T3 with exact T4 continuation",
                "full_t3_exact_claimed": False,
            },
            {
                "scope": "declared finite-support T3 oracle",
                "chance": "all caller-declared positive normalized worlds",
                "actions": "all legal root actions under fixed infoset continuation",
                "classification": "exact only over declared finite support",
                "full_52_card_tree_claimed": False,
            },
        ],
        "scalar_seconds": scalar_seconds,
        "batch_seconds": batch_seconds,
        "seat_summaries": seat_summaries,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "current_profile_changed": False,
        "policy_or_profile_activated": False,
        "spot_vm_started": False,
    }


def run_python_exact_t4_parity(
    *, solver: HuM31T3SearchSolver
) -> dict[str, Any]:
    """Compare Python and pinned Rust T3 paths when every child T4 is exact.

    This deliberately slow check proves implementation parity for the
    composition.  It does not make root T3 chance exact: both implementations
    still use the same finite candidate/evaluation CRN particle batches.
    """

    rows: list[dict[str, Any]] = []
    for seat in ("first", "second"):
        observation = constrained_t3_observation(seat)
        python_started = time.perf_counter()
        expected = evaluate_t3_joint_exact_actions(
            observation=observation,
            config=solver.search_config,
        )
        python_seconds = time.perf_counter() - python_started

        rust_started = time.perf_counter()
        actual = solver.solve(observation)
        rust_seconds = time.perf_counter() - rust_started

        expected_rows = {
            str(row["action_key"]): row for row in expected["actions"]
        }
        actual_rows = {row.action_key: row for row in actual.action_values}
        keys_match = expected_rows.keys() == actual_rows.keys()
        numeric_differences: list[float] = []
        mapping_match = keys_match
        if keys_match:
            for key, expected_row in expected_rows.items():
                actual_row = actual_rows[key]
                mapping_match &= (
                    int(expected_row["original_index"])
                    == actual_row.original_index
                    and int(expected_row["sorted_index"]) == actual_row.rank
                )
                numeric_differences.extend(
                    (
                        abs(
                            float(expected_row["selection_score"])
                            - actual_row.selection_ev
                        ),
                        abs(
                            float(expected_row["score"])
                            - actual_row.evaluation_ev
                        ),
                        abs(
                            float(
                                expected_row[
                                    "evaluation_regret_vs_sample_best"
                                ]
                            )
                            - actual_row.evaluation_regret
                        ),
                    )
                )
        max_abs_difference = (
            max(numeric_differences) if numeric_differences else None
        )
        envelope_match = (
            expected["observation_fingerprint"]
            == actual.observation_fingerprint
            and expected["selected_action_key"] == actual.selected_action_key
            and int(expected["selected_action_original_index"])
            == next(
                row.original_index
                for row in actual.action_values
                if row.action_key == actual.selected_action_key
            )
            and math.isclose(
                float(expected["selected_action_evaluation_score"]),
                actual.selected_evaluation_ev,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(expected["selection_score_gap"]),
                actual.selection_gap,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        passed = (
            keys_match
            and mapping_match
            and max_abs_difference is not None
            and max_abs_difference <= 1e-12
            and envelope_match
        )
        rows.append(
            {
                "seat": seat,
                "passed": passed,
                "python_seconds": python_seconds,
                "rust_seconds": rust_seconds,
                "speedup": (
                    python_seconds / rust_seconds if rust_seconds > 0.0 else None
                ),
                "legal_action_count": len(actual_rows),
                "keys_match": keys_match,
                "mapping_match": mapping_match,
                "envelope_match": envelope_match,
                "max_abs_action_value_difference": max_abs_difference,
                "selected_action_key": actual.selected_action_key,
                "semantic_result_digest": actual.semantic_result_digest,
            }
        )
    return {
        "status": "pass" if all(row["passed"] for row in rows) else "fail",
        "classification": "T3_MC_root_with_exact_T4_children",
        "full_t3_exact_claimed": False,
        "absolute_tolerance": 1e-12,
        "rows": rows,
        "all_rows_passed": all(row["passed"] for row in rows),
    }


def _seat_summary(
    *,
    observations: list[ActorObservation],
    scalar: list[T3SearchDecision],
    batched: list[T3SearchDecision],
    repeated: T3SearchDecision,
) -> dict[str, Any]:
    baseline = scalar[0]
    local_mapping_valid = True
    for observation, decision in zip(observations, scalar, strict=True):
        legal = generate_turn_actions(
            observation.hero_board,
            observation.dealt_cards,
        )
        local_mapping_valid &= all(
            action_key(legal[row.original_index]).to_token() == row.action_key
            for row in decision.action_values
        )

    scalar_batch_parity = all(
        left.semantic_result_digest == right.semantic_result_digest
        and left.result_digest == right.result_digest
        for left, right in zip(scalar, batched, strict=True)
    )
    rows = [
        {
            "permutation": list(observation.dealt_cards),
            "observation_fingerprint": decision.observation_fingerprint,
            "selected_action_key": decision.selected_action_key,
            "legal_action_set_digest": decision.legal_action_set_digest,
            "legal_action_order_digest": decision.legal_action_order_digest,
            "semantic_result_digest": decision.semantic_result_digest,
            "mapping_bound_result_digest": decision.result_digest,
            "child_information_set_count": decision.child_information_set_count,
        }
        for observation, decision in zip(observations, scalar, strict=True)
    ]
    return {
        "permutation_count": len(observations),
        "observation_fingerprint_unique_count": _unique_count(
            row.observation_fingerprint for row in scalar
        ),
        "legal_action_set_digest_unique_count": _unique_count(
            row.legal_action_set_digest for row in scalar
        ),
        "legal_action_order_digest_unique_count": _unique_count(
            row.legal_action_order_digest for row in scalar
        ),
        "semantic_result_digest_unique_count": _unique_count(
            row.semantic_result_digest for row in scalar
        ),
        "mapping_bound_result_digest_unique_count": _unique_count(
            row.result_digest for row in scalar
        ),
        "selected_action_key_unique_count": _unique_count(
            row.selected_action_key for row in scalar
        ),
        "action_value_map_unique_count": _unique_count(
            _action_value_projection(row) for row in scalar
        ),
        "belief_and_rng_unique_count": _unique_count(
            (
                row.candidate_belief_digest,
                row.evaluation_belief_digest,
                row.candidate_rng_digest,
                row.evaluation_rng_digest,
            )
            for row in scalar
        ),
        "child_information_set_count_unique_count": _unique_count(
            row.child_information_set_count for row in scalar
        ),
        "all_local_index_key_mappings_valid": local_mapping_valid,
        "scalar_batch_semantic_and_mapping_parity": scalar_batch_parity,
        "deterministic_repeat_both_digest_scopes": (
            repeated.semantic_result_digest == baseline.semantic_result_digest
            and repeated.result_digest == baseline.result_digest
        ),
        "candidate_evaluation_rng_disjoint": all(
            row.candidate_rng_digest != row.evaluation_rng_digest for row in scalar
        ),
        "selected_action_key": baseline.selected_action_key,
        "semantic_result_digest": baseline.semantic_result_digest,
        "rows": rows,
    }


def _action_value_projection(decision: T3SearchDecision) -> tuple[Any, ...]:
    return tuple(
        (
            row.action_key,
            row.rank,
            row.selection_ev,
            row.evaluation_ev,
            row.evaluation_regret,
        )
        for row in decision.action_values
    )


def _unique_count(values: Iterable[Any]) -> int:
    return len({json.dumps(value, sort_keys=True) for value in values})


def _constrained_board(cards: tuple[str, ...]) -> Board:
    return Board.from_rows(
        top=cards[:3],
        middle=cards[3:8],
        bottom=cards[8:],
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite correctness artifact: {path}")
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
    parser.add_argument(
        "--python-exact-t4-parity",
        action="store_true",
        help="run the deliberately slow both-seat Python/Rust composition check",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    solver = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            library_path=args.library,
            expected_library_sha256=args.library_sha256,
        )
    )
    result = run_correctness_smoke(solver=solver)
    if args.python_exact_t4_parity:
        parity = run_python_exact_t4_parity(solver=solver)
        result["python_exact_t4_parity"] = parity
        result["gates"]["python_rust_exact_t4_composition_parity"] = parity[
            "all_rows_passed"
        ]
        result["all_gates_passed"] = all(result["gates"].values())
        result["status"] = "pass" if result["all_gates_passed"] else "fail"
    _write_json_atomic(args.output, result)
    summary = {
        key: value for key, value in result.items() if key != "seat_summaries"
    }
    print(json.dumps(summary, indent=2))
    if not result["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
