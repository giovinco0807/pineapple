"""Run the local M3.1 Step 3 general-board T3 profiling preflight.

The preflight is deliberately capped at ten balanced roots.  It measures the
pinned 1/1/1 root-CRN search with exact T4 children, verifies scalar/batch and
permutation semantics, records process high-water memory, and runs a two-root
2/3/2 diagnostic ladder.  It does not load a policy profile, report match EV,
start cloud compute, or authorize the 100-root pilot by itself.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import hashlib
import itertools
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .action_key import action_key
from .action_space import generate_actions, generate_turn_actions
from .cards import create_deck
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_m31_t3_runtime import (
    HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA,
    HuM31T3RuntimeConfig,
    HuM31T3SearchSolver,
    T3SearchDecision,
)
from .state import Board


PROFILE_SCHEMA = "hu_m31_t3_step3_local10_profile_v1"
ROOT_GENERATION_POLICY = "deterministic_uniform_legal_simulation_t0_t2_v1"
DEFAULT_ROOT_COUNT = 10
DEFAULT_SEED_START = 2026073201
DEFAULT_SEED_STRIDE = 1_000_003
DEFAULT_RUN_ID = "hu-m31-step3-local10-v1"
DEFAULT_CONTINUATION_SEED = 2026073100
DEFAULT_CANDIDATE_SEED = 2026073101
DEFAULT_EVALUATION_SEED = 2026073102
MAX_PROJECTED_LOCAL100_SECONDS = 3_600.0
MAX_PEAK_RSS_BYTES = 16 * 1024**3
_GENERATION_RNG_XOR = 0x5A17_4D31
_FORBIDDEN_OBSERVATION_FIELDS = frozenset(
    {
        "opponent_private_discards",
        "true_dead_cards",
        "remaining_deck",
        "world_state",
        "replay_truth",
        "draw_pile",
        "future_cards",
    }
)


@dataclass(frozen=True)
class T3ProfileConfig:
    root_count: int = DEFAULT_ROOT_COUNT
    seed_start: int = DEFAULT_SEED_START
    seed_stride: int = DEFAULT_SEED_STRIDE
    run_id: str = DEFAULT_RUN_ID
    continuation_seed: int = DEFAULT_CONTINUATION_SEED
    candidate_seed: int = DEFAULT_CANDIDATE_SEED
    evaluation_seed: int = DEFAULT_EVALUATION_SEED
    candidate_samples: int = 1
    evaluation_samples: int = 1
    downstream_t3_samples: int = 1
    confirmation_candidate_samples: int = 2
    confirmation_evaluation_samples: int = 3
    confirmation_downstream_t3_samples: int = 2
    deterministic_roots: int = 4
    permutation_roots: int = 2
    ladder_roots: int = 2
    max_projected_local100_seconds: float = MAX_PROJECTED_LOCAL100_SECONDS
    max_peak_rss_bytes: int = MAX_PEAK_RSS_BYTES

    def __post_init__(self) -> None:
        for name in (
            "root_count",
            "seed_start",
            "seed_stride",
            "continuation_seed",
            "candidate_seed",
            "evaluation_seed",
            "candidate_samples",
            "evaluation_samples",
            "downstream_t3_samples",
            "confirmation_candidate_samples",
            "confirmation_evaluation_samples",
            "confirmation_downstream_t3_samples",
            "deterministic_roots",
            "permutation_roots",
            "ladder_roots",
            "max_peak_rss_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.root_count <= 0 or self.root_count % 2:
            raise ValueError("root_count must be a positive even integer")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive")
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.candidate_seed == self.evaluation_seed:
            raise ValueError("candidate and evaluation seeds must be distinct")
        for name in (
            "candidate_samples",
            "evaluation_samples",
            "downstream_t3_samples",
            "confirmation_candidate_samples",
            "confirmation_evaluation_samples",
            "confirmation_downstream_t3_samples",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for base, confirmation in (
            (self.candidate_samples, self.confirmation_candidate_samples),
            (self.evaluation_samples, self.confirmation_evaluation_samples),
            (
                self.downstream_t3_samples,
                self.confirmation_downstream_t3_samples,
            ),
        ):
            if confirmation < base:
                raise ValueError("confirmation budgets must not be smaller than base")
        for name in ("deterministic_roots", "permutation_roots", "ladder_roots"):
            value = getattr(self, name)
            if not 1 <= value <= self.root_count:
                raise ValueError(f"{name} must be between 1 and root_count")
        if self.permutation_roots % 2 or self.ladder_roots % 2:
            raise ValueError("permutation_roots and ladder_roots must be even")
        if (
            not math.isfinite(self.max_projected_local100_seconds)
            or self.max_projected_local100_seconds <= 0.0
        ):
            raise ValueError("max_projected_local100_seconds must be positive")
        if self.max_peak_rss_bytes <= 0:
            raise ValueError("max_peak_rss_bytes must be positive")

    def hand_seed(self, hand_index: int) -> int:
        if not 0 <= hand_index < self.root_count // 2:
            raise IndexError("hand_index is outside the configured profile")
        return self.seed_start + hand_index * self.seed_stride

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_count": self.root_count,
            "hand_count": self.root_count // 2,
            "seed_start": self.seed_start,
            "seed_stride": self.seed_stride,
            "run_id": self.run_id,
            "continuation_seed": self.continuation_seed,
            "candidate_seed": self.candidate_seed,
            "evaluation_seed": self.evaluation_seed,
            "candidate_samples": self.candidate_samples,
            "evaluation_samples": self.evaluation_samples,
            "downstream_t3_samples": self.downstream_t3_samples,
            "downstream_t4_samples": 0,
            "confirmation_candidate_samples": (
                self.confirmation_candidate_samples
            ),
            "confirmation_evaluation_samples": (
                self.confirmation_evaluation_samples
            ),
            "confirmation_downstream_t3_samples": (
                self.confirmation_downstream_t3_samples
            ),
            "deterministic_roots": self.deterministic_roots,
            "permutation_roots": self.permutation_roots,
            "ladder_roots": self.ladder_roots,
            "max_projected_local100_seconds": (
                self.max_projected_local100_seconds
            ),
            "max_peak_rss_bytes": self.max_peak_rss_bytes,
        }


@dataclass(frozen=True)
class ProfileRoot:
    root_index: int
    hand_index: int
    hand_seed: int
    observation: ActorObservation


def generate_general_t3_roots(config: T3ProfileConfig) -> list[ProfileRoot]:
    """Generate balanced live-shape T3 roots with varied legal geometry."""

    roots: list[ProfileRoot] = []
    for hand_index in range(config.root_count // 2):
        hand_seed = config.hand_seed(hand_index)
        for observation in _generate_hand_t3_roots(hand_seed):
            roots.append(
                ProfileRoot(
                    root_index=len(roots),
                    hand_index=hand_index,
                    hand_seed=hand_seed,
                    observation=observation,
                )
            )
    return roots


def run_profile(
    *,
    config: T3ProfileConfig,
    primary_solver: HuM31T3SearchSolver,
    confirmation_solver: HuM31T3SearchSolver,
    memory_sampler: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the ten-root correctness, latency, memory, and budget preflight."""

    sample_memory = memory_sampler or process_memory_snapshot
    roots = generate_general_t3_roots(config)
    observations = [root.observation for root in roots]
    memory: dict[str, dict[str, Any]] = {"start": sample_memory()}

    scalar_decisions: list[T3SearchDecision] = []
    scalar_wall_seconds: list[float] = []
    scalar_started = time.perf_counter()
    for observation in observations:
        started = time.perf_counter()
        scalar_decisions.append(primary_solver.solve(observation))
        scalar_wall_seconds.append(time.perf_counter() - started)
    scalar_seconds = time.perf_counter() - scalar_started
    memory["after_scalar"] = sample_memory()

    batch_started = time.perf_counter()
    batch_decisions = primary_solver.solve_many(observations)
    batch_seconds = time.perf_counter() - batch_started
    memory["after_batch"] = sample_memory()

    repeat_started = time.perf_counter()
    repeated = [
        primary_solver.solve(observation)
        for observation in observations[: config.deterministic_roots]
    ]
    repeat_seconds = time.perf_counter() - repeat_started
    memory["after_determinism"] = sample_memory()

    permutation_result = _run_permutation_probe(
        primary_solver,
        observations[: config.permutation_roots],
    )
    memory["after_permutations"] = sample_memory()

    ladder_result = _run_budget_ladder(
        config=config,
        roots=roots[: config.ladder_roots],
        low_decisions=scalar_decisions[: config.ladder_roots],
        confirmation_solver=confirmation_solver,
    )
    memory["after_ladder"] = sample_memory()

    integrity = _integrity_report(
        config=config,
        roots=roots,
        scalar=scalar_decisions,
        batched=batch_decisions,
        repeated=repeated,
    )
    projections = _project_local100_workflow(
        config=config,
        scalar_seconds=scalar_seconds,
        batch_seconds=batch_seconds,
        repeat_seconds=repeat_seconds,
        permutation_seconds=float(permutation_result["seconds"]),
        ladder_seconds=float(ladder_result["seconds"]),
    )
    peak_rss = max(
        (
            int(snapshot["peak_rss_bytes"])
            for snapshot in memory.values()
            if snapshot.get("peak_rss_bytes") is not None
        ),
        default=None,
    )
    memory_supported = all(
        snapshot.get("supported") is True for snapshot in memory.values()
    )
    gates = {
        **integrity["gates"],
        "general_board_action_geometry_not_single_bucket": len(
            {len(decision.action_values) for decision in scalar_decisions}
        )
        >= 2,
        "permutation_semantic_parity": permutation_result["passed"],
        "budget_ladder_integrity": ladder_result["passed"],
        "memory_measurement_supported": memory_supported,
        "peak_rss_within_16_gib": (
            peak_rss is not None and peak_rss <= config.max_peak_rss_bytes
        ),
        "projected_local100_workflow_within_60_minutes": (
            projections["projected_total_seconds"]
            <= config.max_projected_local100_seconds
        ),
        "no_profile_current_or_cloud_access": True,
        "teacher_values_diagnostic_only": True,
    }
    by_seat = {
        seat: _seat_latency_summary(
            [
                (decision, elapsed)
                for decision, elapsed in zip(
                    scalar_decisions, scalar_wall_seconds, strict=True
                )
                if decision.seat == seat
            ]
        )
        for seat in ("first", "second")
    }
    rows = [
        _root_record(root, decision, elapsed)
        for root, decision, elapsed in zip(
            roots, scalar_decisions, scalar_wall_seconds, strict=True
        )
    ]
    return {
        "schema": PROFILE_SCHEMA,
        "status": "pass" if all(gates.values()) else "no_go",
        "scope": "local10_profile_not_strength_match_ev_or_promotion",
        "configuration": config.to_dict(),
        "root_generation": {
            "policy": ROOT_GENERATION_POLICY,
            "policy_input": "simulator_world_only",
            "solver_input": "ActorObservation_only",
            "generation_action_sampling": "uniform_over_all_legal_actions",
            "seed_formula": "seed_start + hand_index * seed_stride",
            "profile_selection": "none",
            "current_profile_read": False,
        },
        "engine": {
            "version": primary_solver.engine_version,
            "library": str(primary_solver.library_path),
            "library_sha256": primary_solver.library_sha256,
            "build_or_fallback": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "search_classification": {
            "root_t3": "common_random_monte_carlo",
            "downstream_t4": "exact_native_enumeration",
            "full_t3_exact_claimed": False,
            "teacher_value_status": "diagnostic_not_match_EV",
        },
        "performance": {
            "scalar_seconds": scalar_seconds,
            "batch_seconds": batch_seconds,
            "batch_speedup": scalar_seconds / max(batch_seconds, 1e-12),
            "deterministic_rerun_seconds": repeat_seconds,
            "root_count": len(roots),
            "scalar_roots_per_second": len(roots) / scalar_seconds,
            "batch_roots_per_second": len(roots) / batch_seconds,
            "by_seat": by_seat,
        },
        "integrity": integrity,
        "permutation_probe": permutation_result,
        "sample_budget_ladder": ladder_result,
        "memory": {
            "snapshots": memory,
            "peak_rss_bytes": peak_rss,
            "gate_bytes": config.max_peak_rss_bytes,
        },
        "local100_projection": projections,
        "semantic_result_digest_schema": (
            HU_M31_T3_SEMANTIC_RESULT_DIGEST_SCHEMA
        ),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "current_profile_changed": False,
        "policy_or_profile_activated": False,
        "spot_vm_started": False,
        "rows": rows,
    }


def process_memory_snapshot() -> dict[str, Any]:
    """Return current and high-water process RSS without optional packages."""

    if sys.platform == "win32":
        return _windows_process_memory_snapshot()
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        scale = 1024 if sys.platform != "darwin" else 1
        peak = int(usage.ru_maxrss) * scale
        return {
            "supported": True,
            "source": "resource.getrusage",
            "rss_bytes": None,
            "peak_rss_bytes": peak,
            "private_bytes": None,
        }
    except (ImportError, OSError, ValueError) as exc:
        return {
            "supported": False,
            "source": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "rss_bytes": None,
            "peak_rss_bytes": None,
            "private_bytes": None,
        }


def _generate_hand_t3_roots(
    hand_seed: int,
) -> tuple[ActorObservation, ActorObservation]:
    deck = create_deck(shuffle=True, rng=random.Random(hand_seed))
    action_rng = random.Random(hand_seed ^ _GENERATION_RNG_XOR)
    cursor = 0
    boards = [Board.from_rows(), Board.from_rows()]
    private_discards: list[list[str]] = [[], []]

    def deal(count: int) -> tuple[str, ...]:
        nonlocal cursor
        cards = tuple(deck[cursor : cursor + count])
        cursor += count
        if len(cards) != count:
            raise RuntimeError("Step 3 profile deck was exhausted")
        return cards

    def advance(player: int, dealt: tuple[str, ...]) -> None:
        actions = (
            generate_actions(boards[player], dealt)
            if boards[player].card_count() == 0
            else generate_turn_actions(boards[player], dealt)
        )
        if not actions:
            raise RuntimeError("Step 3 generation state has no legal action")
        selected = action_rng.choice(actions)
        boards[player] = boards[player].place(selected.placements)
        private_discards[player].extend(selected.discards)

    for player in (0, 1):
        advance(player, deal(5))
    for _street in ("T1", "T2"):
        for player in (0, 1):
            advance(player, deal(3))

    first_dealt = deal(3)
    first = ActorObservation(
        hero_board=boards[0],
        opponent_public_board=boards[1],
        dealt_cards=first_dealt,
        hero_private_discards=tuple(private_discards[0]),
        seat="first",
        street="T3",
        to_act_order="first",
    )
    advance(0, first_dealt)
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


def _integrity_report(
    *,
    config: T3ProfileConfig,
    roots: Sequence[ProfileRoot],
    scalar: Sequence[T3SearchDecision],
    batched: Sequence[T3SearchDecision],
    repeated: Sequence[T3SearchDecision],
) -> dict[str, Any]:
    fingerprints = [root.observation.fingerprint() for root in roots]
    safe_observations = all(
        not _FORBIDDEN_OBSERVATION_FIELDS.intersection(root.observation.to_dict())
        for root in roots
    )
    complete_legal_mapping = True
    all_values_finite = True
    exact_t4 = True
    for root, decision in zip(roots, scalar, strict=True):
        legal = generate_turn_actions(
            root.observation.hero_board,
            root.observation.dealt_cards,
        )
        complete_legal_mapping &= len(decision.action_values) == len(legal)
        complete_legal_mapping &= all(
            0 <= row.original_index < len(legal)
            and action_key(legal[row.original_index]).to_token() == row.action_key
            for row in decision.action_values
        )
        all_values_finite &= all(
            math.isfinite(value)
            for row in decision.action_values
            for value in (
                row.selection_ev,
                row.evaluation_ev,
                row.evaluation_regret,
            )
        )
        exact_t4 &= decision.to_dict()["downstream_t4_samples"] == 0

    scalar_batch = all(
        left.semantic_result_digest == right.semantic_result_digest
        and left.result_digest == right.result_digest
        for left, right in zip(scalar, batched, strict=True)
    )
    deterministic = all(
        original.semantic_result_digest == rerun.semantic_result_digest
        and original.result_digest == rerun.result_digest
        for original, rerun in zip(scalar, repeated, strict=False)
    ) and len(repeated) == config.deterministic_roots

    candidate_keys: set[str] = set()
    evaluation_keys: set[str] = set()
    for root in roots:
        observation = root.observation
        candidate = sample_hidden_card_particles(
            observation,
            base_seed=config.candidate_seed,
            run_id=config.run_id,
            sample_count=config.candidate_samples,
        )
        evaluation = sample_hidden_card_particles(
            observation,
            base_seed=config.evaluation_seed,
            run_id=config.run_id,
            sample_count=config.evaluation_samples,
        )
        candidate_keys.update(row.rng_key_digest for row in candidate.particles)
        evaluation_keys.update(row.rng_key_digest for row in evaluation.particles)

    seats = [root.observation.seat for root in roots]
    gates = {
        "exactly_configured_root_count": len(roots) == config.root_count,
        "balanced_first_second": seats.count("first") == seats.count("second"),
        "unique_observation_fingerprints": len(set(fingerprints)) == len(roots),
        "actor_observation_contains_no_forbidden_truth_fields": safe_observations,
        "all_legal_action_key_index_mappings_complete": complete_legal_mapping,
        "all_action_values_finite": all_values_finite,
        "all_downstream_t4_exact": exact_t4,
        "scalar_batch_both_digest_scopes_match": scalar_batch,
        "deterministic_subset_both_digest_scopes_match": deterministic,
        "candidate_evaluation_rng_globally_disjoint": not (
            candidate_keys & evaluation_keys
        ),
        "candidate_rng_keys_globally_unique": (
            len(candidate_keys) == config.root_count * config.candidate_samples
        ),
        "evaluation_rng_keys_globally_unique": (
            len(evaluation_keys) == config.root_count * config.evaluation_samples
        ),
    }
    return {
        "passed": all(gates.values()),
        "fingerprint_count": len(fingerprints),
        "unique_fingerprint_count": len(set(fingerprints)),
        "candidate_rng_key_count": len(candidate_keys),
        "evaluation_rng_key_count": len(evaluation_keys),
        "candidate_evaluation_rng_overlap_count": len(
            candidate_keys & evaluation_keys
        ),
        "gates": gates,
    }


def _run_permutation_probe(
    solver: HuM31T3SearchSolver,
    observations: Sequence[ActorObservation],
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    passed = True
    for observation in observations:
        permutations = [
            replace(observation, dealt_cards=tuple(cards))
            for cards in itertools.permutations(observation.dealt_cards)
        ]
        decisions = solver.solve_many(permutations)
        semantic_digests = {row.semantic_result_digest for row in decisions}
        selected_keys = {row.selected_action_key for row in decisions}
        set_digests = {row.legal_action_set_digest for row in decisions}
        local_mapping_valid = True
        for permuted, decision in zip(permutations, decisions, strict=True):
            legal = generate_turn_actions(
                permuted.hero_board,
                permuted.dealt_cards,
            )
            local_mapping_valid &= all(
                action_key(legal[value.original_index]).to_token()
                == value.action_key
                for value in decision.action_values
            )
        root_passed = (
            len(semantic_digests) == 1
            and len(selected_keys) == 1
            and len(set_digests) == 1
            and local_mapping_valid
        )
        passed &= root_passed
        rows.append(
            {
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "permutation_count": len(permutations),
                "semantic_result_digest_unique_count": len(semantic_digests),
                "mapping_bound_result_digest_unique_count": len(
                    {row.result_digest for row in decisions}
                ),
                "legal_action_order_digest_unique_count": len(
                    {row.legal_action_order_digest for row in decisions}
                ),
                "selected_action_key_unique_count": len(selected_keys),
                "legal_action_set_digest_unique_count": len(set_digests),
                "local_mapping_valid": local_mapping_valid,
                "passed": root_passed,
            }
        )
    return {
        "passed": passed and len(rows) == len(observations),
        "seconds": time.perf_counter() - started,
        "root_count": len(observations),
        "rows": rows,
    }


def _run_budget_ladder(
    *,
    config: T3ProfileConfig,
    roots: Sequence[ProfileRoot],
    low_decisions: Sequence[T3SearchDecision],
    confirmation_solver: HuM31T3SearchSolver,
) -> dict[str, Any]:
    started = time.perf_counter()
    confirmation = [
        confirmation_solver.solve(root.observation) for root in roots
    ]
    seconds = time.perf_counter() - started
    rows: list[dict[str, Any]] = []
    passed = len(confirmation) == len(low_decisions) == len(roots)
    for root, low, higher in zip(
        roots, low_decisions, confirmation, strict=True
    ):
        low_values = {
            row.action_key: (row.selection_ev, row.evaluation_ev)
            for row in low.action_values
        }
        higher_values = {
            row.action_key: (row.selection_ev, row.evaluation_ev)
            for row in higher.action_values
        }
        action_sets_match = low_values.keys() == higher_values.keys()
        candidate_prefix = _particle_prefix_matches(
            root.observation,
            base_seed=config.candidate_seed,
            run_id=config.run_id,
            low_count=config.candidate_samples,
            high_count=config.confirmation_candidate_samples,
        )
        evaluation_prefix = _particle_prefix_matches(
            root.observation,
            base_seed=config.evaluation_seed,
            run_id=config.run_id,
            low_count=config.evaluation_samples,
            high_count=config.confirmation_evaluation_samples,
        )
        finite = all(
            math.isfinite(value)
            for values in higher_values.values()
            for value in values
        )
        root_passed = action_sets_match and candidate_prefix and evaluation_prefix and finite
        passed &= root_passed
        selection_diffs = [
            abs(low_values[key][0] - higher_values[key][0])
            for key in low_values.keys() & higher_values.keys()
        ]
        evaluation_diffs = [
            abs(low_values[key][1] - higher_values[key][1])
            for key in low_values.keys() & higher_values.keys()
        ]
        higher_evaluation_by_key = {
            row.action_key: row.evaluation_ev for row in higher.action_values
        }
        higher_best_evaluation = max(higher_evaluation_by_key.values())
        low_selected_under_higher = higher_evaluation_by_key.get(
            low.selected_action_key
        )
        rows.append(
            {
                "root_index": root.root_index,
                "seat": root.observation.seat,
                "observation_fingerprint": root.observation.fingerprint(),
                "legal_action_count": len(low_values),
                "low_selected_action_key": low.selected_action_key,
                "confirmation_selected_action_key": higher.selected_action_key,
                "selected_action_agreement": (
                    low.selected_action_key == higher.selected_action_key
                ),
                "action_sets_match": action_sets_match,
                "candidate_particle_prefix_match": candidate_prefix,
                "evaluation_particle_prefix_match": evaluation_prefix,
                "max_abs_selection_ev_change": max(selection_diffs, default=0.0),
                "max_abs_evaluation_ev_change": max(
                    evaluation_diffs, default=0.0
                ),
                "low_selected_evaluation_regret_under_confirmation_batch": (
                    None
                    if low_selected_under_higher is None
                    else higher_best_evaluation - low_selected_under_higher
                ),
                "passed": root_passed,
            }
        )
    return {
        "passed": passed,
        "classification": "diagnostic_nested_sample_budget_not_strength_gate",
        "seconds": seconds,
        "root_count": len(roots),
        "base_budget": {
            "candidate": config.candidate_samples,
            "evaluation": config.evaluation_samples,
            "downstream_t3": config.downstream_t3_samples,
            "downstream_t4": 0,
        },
        "confirmation_budget": {
            "candidate": config.confirmation_candidate_samples,
            "evaluation": config.confirmation_evaluation_samples,
            "downstream_t3": config.confirmation_downstream_t3_samples,
            "downstream_t4": 0,
        },
        "selected_action_agreement_count": sum(
            bool(row["selected_action_agreement"]) for row in rows
        ),
        "rows": rows,
    }


def _particle_prefix_matches(
    observation: ActorObservation,
    *,
    base_seed: int,
    run_id: str,
    low_count: int,
    high_count: int,
) -> bool:
    low = sample_hidden_card_particles(
        observation,
        base_seed=base_seed,
        run_id=run_id,
        sample_count=low_count,
    )
    high = sample_hidden_card_particles(
        observation,
        base_seed=base_seed,
        run_id=run_id,
        sample_count=high_count,
    )
    return [row.digest() for row in low.particles] == [
        row.digest() for row in high.particles[:low_count]
    ]


def _project_local100_workflow(
    *,
    config: T3ProfileConfig,
    scalar_seconds: float,
    batch_seconds: float,
    repeat_seconds: float,
    permutation_seconds: float,
    ladder_seconds: float,
) -> dict[str, Any]:
    scalar_100 = scalar_seconds * (100 / config.root_count)
    batch_100 = batch_seconds * (100 / config.root_count)
    deterministic_20 = repeat_seconds * (20 / config.deterministic_roots)
    permutations_10 = permutation_seconds * (10 / config.permutation_roots)
    ladder_10 = ladder_seconds * (10 / config.ladder_roots)
    total = (
        scalar_100
        + batch_100
        + deterministic_20
        + permutations_10
        + ladder_10
    )
    return {
        "formula": (
            "scalar100 + batch100 + deterministic20 + "
            "six_permutations10 + confirmation_budget10"
        ),
        "projected_scalar_100_seconds": scalar_100,
        "projected_batch_100_seconds": batch_100,
        "projected_deterministic_20_seconds": deterministic_20,
        "projected_six_permutations_10_seconds": permutations_10,
        "projected_confirmation_budget_10_seconds": ladder_10,
        "projected_total_seconds": total,
        "gate_seconds": config.max_projected_local100_seconds,
        "projection_is_linear_diagnostic": True,
    }


def _seat_latency_summary(
    rows: Sequence[tuple[T3SearchDecision, float]],
) -> dict[str, Any]:
    latencies = sorted(elapsed for _decision, elapsed in rows)
    actions = [len(decision.action_values) for decision, _elapsed in rows]
    child_counts = [
        decision.child_information_set_count for decision, _elapsed in rows
    ]
    return {
        "count": len(rows),
        "mean_seconds": statistics.fmean(latencies),
        "p50_seconds": _percentile(latencies, 0.50),
        "p95_seconds": _percentile(latencies, 0.95),
        "p99_seconds": _percentile(latencies, 0.99),
        "max_seconds": max(latencies),
        "legal_action_count_min": min(actions),
        "legal_action_count_max": max(actions),
        "child_information_set_count_min": min(child_counts),
        "child_information_set_count_max": max(child_counts),
    }


def _root_record(
    root: ProfileRoot,
    decision: T3SearchDecision,
    elapsed: float,
) -> dict[str, Any]:
    board = root.observation.hero_board
    opponent = root.observation.opponent_public_board
    return {
        "root_index": root.root_index,
        "hand_index": root.hand_index,
        "hand_seed": root.hand_seed,
        "seat": root.observation.seat,
        "observation_fingerprint": root.observation.fingerprint(),
        "hero_row_counts": {
            "top": len(board.top),
            "middle": len(board.middle),
            "bottom": len(board.bottom),
        },
        "opponent_public_row_counts": {
            "top": len(opponent.top),
            "middle": len(opponent.middle),
            "bottom": len(opponent.bottom),
        },
        "legal_action_count": len(decision.action_values),
        "child_information_set_count": decision.child_information_set_count,
        "wall_seconds": elapsed,
        "native_latency_ms": decision.native_latency_ms,
        "validation_latency_ms": decision.validation_latency_ms,
        "selected_action_key": decision.selected_action_key,
        "selected_selection_ev": decision.selected_selection_ev,
        "selected_evaluation_ev": decision.selected_evaluation_ev,
        "semantic_result_digest": decision.semantic_result_digest,
        "mapping_bound_result_digest": decision.result_digest,
        "teacher_value_status": "diagnostic_not_match_EV",
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, math.ceil(len(values) * fraction) - 1))
    return float(values[index])


def _windows_process_memory_snapshot() -> dict[str, Any]:
    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PageFaultCount", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            ctypes.wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            error = ctypes.get_last_error()
            raise OSError(error, os.strerror(error))
        return {
            "supported": True,
            "source": "GetProcessMemoryInfo",
            "rss_bytes": int(counters.WorkingSetSize),
            "peak_rss_bytes": int(counters.PeakWorkingSetSize),
            "private_bytes": int(counters.PrivateUsage),
        }
    except (AttributeError, OSError, ValueError) as exc:
        return {
            "supported": False,
            "source": "GetProcessMemoryInfo",
            "error": f"{type(exc).__name__}: {exc}",
            "rss_bytes": None,
            "peak_rss_bytes": None,
            "private_bytes": None,
        }


def _write_json_atomic(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Step 3 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=DEFAULT_ROOT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-stride", type=int, default=DEFAULT_SEED_STRIDE)
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    config = T3ProfileConfig(
        root_count=args.roots,
        seed_start=args.seed,
        seed_stride=args.seed_stride,
    )
    common = {
        "library_path": args.library,
        "expected_library_sha256": args.library_sha256,
        "run_id": config.run_id,
        "seed": config.continuation_seed,
        "candidate_seed": config.candidate_seed,
        "evaluation_seed": config.evaluation_seed,
    }
    primary = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            **common,
            candidate_samples=config.candidate_samples,
            evaluation_samples=config.evaluation_samples,
            downstream_t3_samples=config.downstream_t3_samples,
        )
    )
    confirmation = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            **common,
            candidate_samples=config.confirmation_candidate_samples,
            evaluation_samples=config.confirmation_evaluation_samples,
            downstream_t3_samples=config.confirmation_downstream_t3_samples,
        )
    )
    report = run_profile(
        config=config,
        primary_solver=primary,
        confirmation_solver=confirmation,
    )
    report["source_hashes"] = {
        "src/ofc_regular/validate_hu_m31_t3_profile.py": _sha256_file(
            Path(__file__)
        ),
        "src/ofc_regular/hu_m31_t3_runtime.py": _sha256_file(
            Path(__file__).with_name("hu_m31_t3_runtime.py")
        ),
    }
    _write_json_atomic(args.output, report)
    summary = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not report["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
