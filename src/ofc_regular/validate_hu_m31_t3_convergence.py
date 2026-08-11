"""Freeze the M3.1 T3 sample budget before the 100-root local pilot.

This preflight recomputes the same balanced first/second-seat roots at nested
1/1/1, 2/3/2, and 4/8/2 budgets.  The final gate is deliberately based on
regret under the higher, independent evaluation batch rather than action
agreement alone.  Passing authorizes only a local 100-root correctness and
performance pilot; it is not match EV, a strength result, or runtime promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .action_key import (
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import generate_turn_actions
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_m31_t3_runtime import (
    HuM31T3RuntimeConfig,
    HuM31T3SearchSolver,
    T3SearchDecision,
)
from .validate_hu_m31_t3_profile import (
    DEFAULT_CANDIDATE_SEED,
    DEFAULT_CONTINUATION_SEED,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_RUN_ID,
    DEFAULT_SEED_START,
    DEFAULT_SEED_STRIDE,
    T3ProfileConfig,
    generate_general_t3_roots,
)


CONVERGENCE_SCHEMA = "hu_m31_t3_step3_budget_convergence_v1"
DEFAULT_OUTPUT = Path(
    "outputs/hu_joint_policy/m31_t3_step3/convergence_2root.json"
)
DEFAULT_MAX_REGRET = 1.0
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
class SearchBudget:
    label: str
    candidate_samples: int
    evaluation_samples: int
    downstream_t3_samples: int

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("budget label must not be empty")
        for name in (
            "candidate_samples",
            "evaluation_samples",
            "downstream_t3_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "candidate_samples": self.candidate_samples,
            "evaluation_samples": self.evaluation_samples,
            "downstream_t3_samples": self.downstream_t3_samples,
            "downstream_t4_samples": 0,
        }


BASELINE_BUDGET = SearchBudget("baseline_1_1_1", 1, 1, 1)
CONFIRMATION_BUDGET = SearchBudget("confirmation_2_3_2", 2, 3, 2)
REFERENCE_BUDGET = SearchBudget("teacher_default_4_8_2", 4, 8, 2)


@dataclass(frozen=True)
class ConvergenceConfig:
    root_count: int = 2
    seed_start: int = DEFAULT_SEED_START
    seed_stride: int = DEFAULT_SEED_STRIDE
    run_id: str = DEFAULT_RUN_ID
    continuation_seed: int = DEFAULT_CONTINUATION_SEED
    candidate_seed: int = DEFAULT_CANDIDATE_SEED
    evaluation_seed: int = DEFAULT_EVALUATION_SEED
    max_regret_under_reference: float = DEFAULT_MAX_REGRET
    budgets: tuple[SearchBudget, ...] = (
        BASELINE_BUDGET,
        CONFIRMATION_BUDGET,
        REFERENCE_BUDGET,
    )

    def __post_init__(self) -> None:
        if isinstance(self.root_count, bool) or not isinstance(self.root_count, int):
            raise TypeError("root_count must be an integer")
        if self.root_count <= 0 or self.root_count % 2:
            raise ValueError("root_count must be a positive even integer")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive")
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.candidate_seed == self.evaluation_seed:
            raise ValueError("candidate and evaluation seeds must be distinct")
        if (
            not math.isfinite(self.max_regret_under_reference)
            or self.max_regret_under_reference < 0.0
        ):
            raise ValueError("max_regret_under_reference must be finite and non-negative")
        if len(self.budgets) != 3:
            raise ValueError("exactly three ordered budgets are required")
        if len({budget.label for budget in self.budgets}) != len(self.budgets):
            raise ValueError("budget labels must be unique")
        for lower, higher in zip(
            self.budgets[:-1], self.budgets[1:], strict=True
        ):
            if (
                higher.candidate_samples < lower.candidate_samples
                or higher.evaluation_samples < lower.evaluation_samples
                or higher.downstream_t3_samples < lower.downstream_t3_samples
            ):
                raise ValueError("budgets must be component-wise non-decreasing")

    def profile_config(self) -> T3ProfileConfig:
        return T3ProfileConfig(
            root_count=self.root_count,
            seed_start=self.seed_start,
            seed_stride=self.seed_stride,
            run_id=self.run_id,
            continuation_seed=self.continuation_seed,
            candidate_seed=self.candidate_seed,
            evaluation_seed=self.evaluation_seed,
            candidate_samples=BASELINE_BUDGET.candidate_samples,
            evaluation_samples=BASELINE_BUDGET.evaluation_samples,
            downstream_t3_samples=BASELINE_BUDGET.downstream_t3_samples,
            confirmation_candidate_samples=CONFIRMATION_BUDGET.candidate_samples,
            confirmation_evaluation_samples=CONFIRMATION_BUDGET.evaluation_samples,
            confirmation_downstream_t3_samples=(
                CONFIRMATION_BUDGET.downstream_t3_samples
            ),
            deterministic_roots=self.root_count,
            permutation_roots=self.root_count,
            ladder_roots=self.root_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_count": self.root_count,
            "seed_start": self.seed_start,
            "seed_stride": self.seed_stride,
            "run_id": self.run_id,
            "continuation_seed": self.continuation_seed,
            "candidate_seed": self.candidate_seed,
            "evaluation_seed": self.evaluation_seed,
            "max_regret_under_reference": self.max_regret_under_reference,
            "budgets": [budget.to_dict() for budget in self.budgets],
        }


class _BatchSolver(Protocol):
    def solve_many(
        self, observations: Sequence[ActorObservation]
    ) -> list[T3SearchDecision]: ...


def run_convergence(
    *,
    config: ConvergenceConfig,
    solvers: Mapping[str, _BatchSolver],
) -> dict[str, Any]:
    """Run nested-budget convergence and return a write-ready audit record."""

    expected_labels = [budget.label for budget in config.budgets]
    if set(solvers) != set(expected_labels):
        raise ValueError("solvers must match the configured budget labels exactly")
    engine_rows = {
        label: {
            "engine_version": str(getattr(solver, "engine_version")),
            "library": str(getattr(solver, "library_path")),
            "library_sha256": str(getattr(solver, "library_sha256")),
        }
        for label, solver in solvers.items()
    }
    engine_metadata = list(engine_rows.values())
    if any(row != engine_metadata[0] for row in engine_metadata[1:]):
        raise ValueError("all convergence solvers must use the same native engine")

    roots = generate_general_t3_roots(config.profile_config())
    observations = [root.observation for root in roots]
    hidden_fields_absent = all(
        not (_FORBIDDEN_OBSERVATION_FIELDS & root.observation.to_dict().keys())
        for root in roots
    )
    decisions_by_label: dict[str, list[T3SearchDecision]] = {}
    budget_runs: list[dict[str, Any]] = []
    all_decision_integrity = True

    for budget in config.budgets:
        started = time.perf_counter()
        decisions = solvers[budget.label].solve_many(observations)
        elapsed = time.perf_counter() - started
        if len(decisions) != len(roots):
            raise RuntimeError(
                f"{budget.label} returned {len(decisions)} decisions for "
                f"{len(roots)} roots"
            )
        decisions_by_label[budget.label] = decisions
        rows: list[dict[str, Any]] = []
        for root, decision in zip(roots, decisions, strict=True):
            integrity = _decision_integrity(root.observation, decision, budget)
            all_decision_integrity &= all(integrity.values())
            rows.append(
                {
                    "root_index": root.root_index,
                    "seat": root.observation.seat,
                    "observation_fingerprint": root.observation.fingerprint(),
                    "selected_action_key": decision.selected_action_key,
                    "selected_selection_ev": decision.selected_selection_ev,
                    "selected_evaluation_ev": decision.selected_evaluation_ev,
                    "selection_gap": decision.selection_gap,
                    "locked_evaluation_regret": decision.evaluation_sample_regret,
                    "legal_action_count": len(decision.action_values),
                    "child_information_set_count": (
                        decision.child_information_set_count
                    ),
                    "semantic_result_digest": decision.semantic_result_digest,
                    "mapping_bound_result_digest": decision.result_digest,
                    "decision_integrity": integrity,
                    "action_values": [
                        {
                            "original_index": value.original_index,
                            "rank": value.rank,
                            "action_key": value.action_key,
                            "selection_ev": value.selection_ev,
                            "evaluation_ev": value.evaluation_ev,
                            "evaluation_regret": value.evaluation_regret,
                        }
                        for value in decision.action_values
                    ],
                }
            )
        budget_runs.append(
            {
                "budget": budget.to_dict(),
                "batch_wall_seconds": elapsed,
                "rows": rows,
            }
        )

    pairwise: list[dict[str, Any]] = []
    all_nested_prefixes = True
    all_pair_structure = True
    for pair_index, (lower_budget, higher_budget) in enumerate(
        zip(config.budgets[:-1], config.budgets[1:], strict=True)
    ):
        final_pair = pair_index == len(config.budgets) - 2
        comparison = _compare_budget_pair(
            config=config,
            roots=observations,
            lower_budget=lower_budget,
            higher_budget=higher_budget,
            lower_decisions=decisions_by_label[lower_budget.label],
            higher_decisions=decisions_by_label[higher_budget.label],
            enforce_regret_gate=final_pair,
        )
        pairwise.append(comparison)
        all_nested_prefixes &= all(
            row["candidate_particle_prefix_match"]
            and row["evaluation_particle_prefix_match"]
            for row in comparison["rows"]
        )
        all_pair_structure &= all(
            row["action_sets_match"] and row["finite_values"]
            for row in comparison["rows"]
        )

    final_rows = pairwise[-1]["rows"]
    previous_selection_regret_pass = all(
        row["lower_selected_regret_under_higher_evaluation"]
        <= config.max_regret_under_reference
        for row in final_rows
    )
    reference_locked_regret_pass = all(
        row["higher_locked_evaluation_regret"]
        <= config.max_regret_under_reference
        for row in final_rows
    )
    seats = [root.observation.seat for root in roots]
    balanced_both_seats = (
        seats.count("first") == seats.count("second") == len(seats) // 2
    )
    gates = {
        "balanced_both_seats": balanced_both_seats,
        "public_information_only": hidden_fields_absent,
        "all_decision_integrity": all_decision_integrity,
        "all_action_sets_and_values_valid": all_pair_structure,
        "candidate_and_evaluation_particle_prefixes_nested": (
            all_nested_prefixes
        ),
        "confirmation_selection_regret_under_reference_lte_gate": (
            previous_selection_regret_pass
        ),
        "reference_locked_evaluation_regret_lte_gate": (
            reference_locked_regret_pass
        ),
        "single_pinned_native_engine_for_all_budgets": True,
    }
    authorized = all(gates.values())
    reference = config.budgets[-1]
    return {
        "schema": CONVERGENCE_SCHEMA,
        "status": "pass" if authorized else "no_go",
        "config": config.to_dict(),
        "gate_definition_frozen_before_real_run": True,
        "gate_interpretation": (
            "The confirmation-selected action and the reference-selected action "
            "must each be within the pre-registered regret limit under the "
            "reference budget's independent evaluation batch for both seats."
        ),
        "root_generation": {
            "root_count": len(roots),
            "seats": seats,
            "observation_fingerprints": [
                root.observation.fingerprint() for root in roots
            ],
            "current_profile_read": False,
        },
        "engine": {
            **engine_metadata[0],
            "build_or_fallback": False,
        },
        "budget_runs": budget_runs,
        "pairwise_comparisons": pairwise,
        "gates": gates,
        "all_gates_passed": authorized,
        "local100_authorization": {
            "authorized": authorized,
            "budget": reference.to_dict() if authorized else None,
            "scope": (
                "local_100_root_correctness_and_performance_pilot_only; "
                "not_runtime_promotion_not_match_ev_not_strength_proof"
            ),
            "reason": (
                "pre_registered_nested_budget_regret_gate_passed"
                if authorized
                else "pre_registered_nested_budget_regret_gate_failed"
            ),
        },
        "teacher_value_status": "diagnostic_not_match_EV",
        "current_profile_changed": False,
        "spot_vm_started": False,
    }


def _decision_integrity(
    observation: ActorObservation,
    decision: T3SearchDecision,
    budget: SearchBudget,
) -> dict[str, bool]:
    legal_actions = generate_turn_actions(
        observation.hero_board,
        observation.dealt_cards,
    )
    expected_by_index = {
        index: action_key(action).to_token()
        for index, action in enumerate(legal_actions)
    }
    values_by_index = {
        value.original_index: value.action_key for value in decision.action_values
    }
    evaluation_by_key = {
        value.action_key: value.evaluation_ev for value in decision.action_values
    }
    finite = all(
        math.isfinite(number)
        for value in decision.action_values
        for number in (
            value.selection_ev,
            value.evaluation_ev,
            value.evaluation_regret,
        )
    )
    best_evaluation = max(evaluation_by_key.values(), default=-math.inf)
    selected_evaluation = evaluation_by_key.get(decision.selected_action_key)
    selected_regret_consistent = (
        selected_evaluation is not None
        and math.isclose(
            decision.evaluation_sample_regret,
            best_evaluation - selected_evaluation,
            abs_tol=1e-9,
        )
    )
    return {
        "observation_fingerprint": (
            decision.observation_fingerprint == observation.fingerprint()
        ),
        "seat": decision.seat == observation.seat,
        "budget": (
            decision.candidate_samples == budget.candidate_samples
            and decision.evaluation_samples == budget.evaluation_samples
            and decision.downstream_t3_samples == budget.downstream_t3_samples
        ),
        "exact_t4": decision.use_t4_action_cache,
        "finite_values": finite,
        "unique_action_keys": (
            len(evaluation_by_key) == len(decision.action_values)
        ),
        "original_index_mapping": values_by_index == expected_by_index,
        "selected_action_mapping": (
            decision.selected_action_key == action_key(decision.action).to_token()
        ),
        "legal_action_set_digest": (
            decision.legal_action_set_digest
            == legal_action_set_digest(legal_actions)
        ),
        "legal_action_order_digest": (
            decision.legal_action_order_digest
            == ordered_action_mapping_digest(legal_actions)
        ),
        "selected_regret_consistent": selected_regret_consistent,
        "candidate_evaluation_rng_domains_distinct": (
            decision.candidate_rng_digest != decision.evaluation_rng_digest
        ),
    }


def _compare_budget_pair(
    *,
    config: ConvergenceConfig,
    roots: Sequence[ActorObservation],
    lower_budget: SearchBudget,
    higher_budget: SearchBudget,
    lower_decisions: Sequence[T3SearchDecision],
    higher_decisions: Sequence[T3SearchDecision],
    enforce_regret_gate: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, (observation, lower, higher) in enumerate(
        zip(roots, lower_decisions, higher_decisions, strict=True)
    ):
        lower_values = {value.action_key: value for value in lower.action_values}
        higher_values = {value.action_key: value for value in higher.action_values}
        common_keys = lower_values.keys() & higher_values.keys()
        action_sets_match = lower_values.keys() == higher_values.keys()
        finite = all(
            math.isfinite(number)
            for value in higher_values.values()
            for number in (value.selection_ev, value.evaluation_ev)
        )
        higher_best = max(
            (value.evaluation_ev for value in higher_values.values()),
            default=math.inf,
        )
        lower_selected_value = higher_values.get(lower.selected_action_key)
        lower_selected_regret = (
            math.inf
            if lower_selected_value is None
            else higher_best - lower_selected_value.evaluation_ev
        )
        candidate_prefix = _particle_prefix_matches(
            observation,
            base_seed=config.candidate_seed,
            run_id=config.run_id,
            lower_count=lower_budget.candidate_samples,
            higher_count=higher_budget.candidate_samples,
        )
        evaluation_prefix = _particle_prefix_matches(
            observation,
            base_seed=config.evaluation_seed,
            run_id=config.run_id,
            lower_count=lower_budget.evaluation_samples,
            higher_count=higher_budget.evaluation_samples,
        )
        structural = action_sets_match and finite and candidate_prefix and evaluation_prefix
        regret_gate = (
            lower_selected_regret <= config.max_regret_under_reference
            and higher.evaluation_sample_regret
            <= config.max_regret_under_reference
        )
        rows.append(
            {
                "root_index": index,
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "lower_selected_action_key": lower.selected_action_key,
                "higher_selected_action_key": higher.selected_action_key,
                "selected_action_agreement": (
                    lower.selected_action_key == higher.selected_action_key
                ),
                "action_sets_match": action_sets_match,
                "finite_values": finite,
                "candidate_particle_prefix_match": candidate_prefix,
                "evaluation_particle_prefix_match": evaluation_prefix,
                "max_abs_selection_ev_change": max(
                    (
                        abs(
                            lower_values[key].selection_ev
                            - higher_values[key].selection_ev
                        )
                        for key in common_keys
                    ),
                    default=0.0,
                ),
                "max_abs_evaluation_ev_change": max(
                    (
                        abs(
                            lower_values[key].evaluation_ev
                            - higher_values[key].evaluation_ev
                        )
                        for key in common_keys
                    ),
                    default=0.0,
                ),
                "lower_selected_regret_under_higher_evaluation": (
                    lower_selected_regret
                ),
                "higher_locked_evaluation_regret": (
                    higher.evaluation_sample_regret
                ),
                "regret_gate_enforced": enforce_regret_gate,
                "passed": structural and (regret_gate if enforce_regret_gate else True),
            }
        )
    return {
        "lower_budget": lower_budget.to_dict(),
        "higher_budget": higher_budget.to_dict(),
        "classification": (
            "authorization_gate" if enforce_regret_gate else "diagnostic_only"
        ),
        "selected_action_agreement_count": sum(
            row["selected_action_agreement"] for row in rows
        ),
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
    }


def _particle_prefix_matches(
    observation: ActorObservation,
    *,
    base_seed: int,
    run_id: str,
    lower_count: int,
    higher_count: int,
) -> bool:
    lower = sample_hidden_card_particles(
        observation,
        base_seed=base_seed,
        run_id=run_id,
        sample_count=lower_count,
    )
    higher = sample_hidden_card_particles(
        observation,
        base_seed=base_seed,
        run_id=run_id,
        sample_count=higher_count,
    )
    return [particle.digest() for particle in lower.particles] == [
        particle.digest() for particle in higher.particles[:lower_count]
    ]


def _write_json_atomic(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite convergence artifact: {path}")
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roots", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-stride", type=int, default=DEFAULT_SEED_STRIDE)
    parser.add_argument(
        "--max-regret-under-reference",
        type=float,
        default=DEFAULT_MAX_REGRET,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    config = ConvergenceConfig(
        root_count=args.roots,
        seed_start=args.seed,
        seed_stride=args.seed_stride,
        max_regret_under_reference=args.max_regret_under_reference,
    )
    common = {
        "library_path": args.library,
        "expected_library_sha256": args.library_sha256,
        "run_id": config.run_id,
        "seed": config.continuation_seed,
        "candidate_seed": config.candidate_seed,
        "evaluation_seed": config.evaluation_seed,
    }
    solvers = {
        budget.label: HuM31T3SearchSolver(
            HuM31T3RuntimeConfig(
                **common,
                candidate_samples=budget.candidate_samples,
                evaluation_samples=budget.evaluation_samples,
                downstream_t3_samples=budget.downstream_t3_samples,
            )
        )
        for budget in config.budgets
    }
    report = run_convergence(config=config, solvers=solvers)
    report["source_hashes"] = {
        "src/ofc_regular/validate_hu_m31_t3_convergence.py": _sha256_file(
            Path(__file__)
        ),
        "src/ofc_regular/validate_hu_m31_t3_profile.py": _sha256_file(
            Path(__file__).with_name("validate_hu_m31_t3_profile.py")
        ),
        "src/ofc_regular/hu_m31_t3_runtime.py": _sha256_file(
            Path(__file__).with_name("hu_m31_t3_runtime.py")
        ),
    }
    _write_json_atomic(args.output, report)
    summary = {key: value for key, value in report.items() if key != "budget_runs"}
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not report["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
