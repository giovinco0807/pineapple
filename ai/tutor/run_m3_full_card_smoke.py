"""Reproduce the six-stratum non-promoting M3 full-card smoke artifact.

The runner uses the four frozen T1/T2 position specialists plus a frozen T3
BB ranking prior.  All five checkpoints remain explicitly non-promotable.  A
successful run proves physical range/solver wiring only; the gate always keeps
``m3_promotion_passed`` and ``full_card_policy_promoted`` false.
"""
from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from ai.tutor.frozen_behavior_torch import (
    DEFAULT_QUANTIZATION_DENOMINATOR,
    KNOWN_HU_POLICY_VALUE_ASSETS,
    TorchPolicyValueBehaviorModel,
    TurnActorBehaviorDispatch,
)
from ai.tutor.promotion_gate_m3_full_card_smoke import (
    FullCardSmokeStratumInput,
    run_validate_and_write_six_strata_smoke,
)
from ai.tutor.t3_hu_full_card_range import (
    FrozenBehaviorModel,
    build_history_weighted_full_card_range,
)
from ai.tutor.t3_hu_reduced_fixtures import compile_canonical_reduced_fixture


T3_BB_PRIOR_RELATIVE_PATH = (
    "ai/data/t3_oracle_rust_discard_sensitive_ft/t3_policyvalue_v2_best.pt"
)
T3_BB_PRIOR_CHECKPOINT_SHA256 = (
    "1939170943333d7f136299d03391ba31bf4017192661cf7bf7d9e773c7629c00"
)


def build_smoke_behavior_prior(
    workspace_root: str | Path,
    *,
    temperature: Fraction | int | str = Fraction(1, 1),
    quantization_denominator: int = DEFAULT_QUANTIZATION_DENOMINATOR,
) -> TurnActorBehaviorDispatch:
    """Load the five exact-hash ranking priors required by both T3 roots."""

    root = Path(workspace_root).resolve()
    routes: dict[tuple[int, str], FrozenBehaviorModel] = {}
    for (turn, actor), asset in sorted(KNOWN_HU_POLICY_VALUE_ASSETS.items()):
        routes[(turn, actor)] = TorchPolicyValueBehaviorModel(
            root / str(asset["relative_path"]),
            model_id=f"hu_t{turn}_{actor}_2m_policyvalue_prior",
            supported_turns=(turn,),
            supported_actors=(actor,),
            training_scope_id=f"hu_t{turn}_{actor}_visible_opponent_board_2m_v1",
            temperature=temperature,
            quantization_denominator=quantization_denominator,
            expected_checkpoint_sha256=str(asset["checkpoint_sha256"]),
        )
    routes[(3, "bb")] = TorchPolicyValueBehaviorModel(
        root / T3_BB_PRIOR_RELATIVE_PATH,
        model_id="hu_t3_bb_discard_sensitive_policyvalue_prior",
        supported_turns=(3,),
        supported_actors=("bb",),
        training_scope_id=(
            "hu_t3_bb_discard_sensitive_teacher_ev_ranking_prior_v1"
        ),
        temperature=temperature,
        quantization_denominator=quantization_denominator,
        expected_checkpoint_sha256=T3_BB_PRIOR_CHECKPOINT_SHA256,
    )
    return TurnActorBehaviorDispatch(
        routes,
        model_id="known_hu_t1_t2_plus_t3_bb_policyvalue_prior_dispatch_v1",
    )


def build_six_strata_inputs(
    behavior_model: FrozenBehaviorModel,
    *,
    base_seed: int,
    max_particles: int,
    epsilon: Fraction | int | str,
    rust_solver_path: str | Path | None = None,
    rust_timeout_s: float = 30.0,
) -> tuple[dict[str, FullCardSmokeStratumInput], list[dict[str, Any]]]:
    """Build six independent real-card observations and posterior ranges."""

    strata: dict[str, FullCardSmokeStratumInput] = {}
    audit: list[dict[str, Any]] = []
    for actor in ("bb", "btn"):
        for joker_count in (0, 1, 2):
            name = f"{actor}_joker{joker_count}"
            started = time.perf_counter()
            fixture = compile_canonical_reduced_fixture(
                actor,
                joker_count,
                rust_solver_path=rust_solver_path,
                rust_timeout_s=rust_timeout_s,
            )
            observation = fixture.root.branches[0].child.infoset_key
            root_range = build_history_weighted_full_card_range(
                observation,
                behavior_model,
                epsilon=epsilon,
                max_particles=max_particles,
                seed=base_seed + len(strata),
            )
            strata[name] = FullCardSmokeStratumInput(observation, root_range)
            audit.append(
                {
                    "stratum": name,
                    "fixture_manifest_sha256": fixture.fixture_manifest_sha256,
                    "observation_digest": observation.digest(),
                    "range_content_sha256": root_range.range_content_sha256,
                    "range_build_sha256": root_range.range_build_sha256,
                    "particle_count": root_range.particle_count,
                    "effective_sample_size_exact": str(
                        root_range.effective_sample_size
                    ),
                    "behavior_distribution_source_counts": dict(
                        root_range.metadata[
                            "behavior_distribution_source_counts"
                        ]
                    ),
                    "build_wall_seconds": time.perf_counter() - started,
                }
            )
    return strata, audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the six-stratum non-promoting M3 full-card smoke gate"
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai/reports/m3_full_card_smoke_20260713"),
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260713)
    parser.add_argument("--max-particles", type=int, default=4)
    parser.add_argument("--max-infosets", type=int, default=100_000)
    parser.add_argument("--epsilon", default="0/1")
    parser.add_argument("--temperature", default="1/1")
    parser.add_argument("--rust-solver-path", type=Path, default=None)
    parser.add_argument("--rust-timeout-s", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace_root = args.workspace_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else workspace_root / args.output_dir
    )
    behavior = build_smoke_behavior_prior(
        workspace_root,
        temperature=Fraction(args.temperature),
    )
    strata, range_audit = build_six_strata_inputs(
        behavior,
        base_seed=args.base_seed,
        max_particles=args.max_particles,
        epsilon=Fraction(args.epsilon),
        rust_solver_path=args.rust_solver_path,
        rust_timeout_s=args.rust_timeout_s,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence, result = run_validate_and_write_six_strata_smoke(
        strata,
        evidence_path=output_dir / "evidence.json",
        result_path=output_dir / "result.json",
        iterations=args.iterations,
        base_seed=args.base_seed,
        max_infosets=args.max_infosets,
        linear_averaging=True,
    )
    summary = {
        "behavior_model_id": behavior.model_id,
        "behavior_model_sha256": behavior.model_sha256,
        "behavior_promotion_eligible": behavior.model_manifest[
            "promotion_eligible"
        ],
        "range_audit": range_audit,
        "evidence_artifact_sha256": evidence["artifact_sha256"],
        "result_sha256": result["result_sha256"],
        "passed": result["passed"],
        "execution_smoke_passed": result["execution_smoke_passed"],
        "m3_promotion_passed": result["m3_promotion_passed"],
        "full_card_policy_promoted": result["full_card_policy_promoted"],
        "promotion_blockers": result["promotion_blockers"],
        "output_dir": str(output_dir.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
