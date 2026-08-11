"""Run one immutable Attempt13 population shard from explicit artifacts."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ofc_regular.ai_profiles import ModelPaths, build_policy, load_model_bundle, required_profiles
from ofc_regular.evaluate_hu_m4_population import evaluate_hu_m4_population
from ofc_regular.hu_m43_attempt13_distilled_model import is_bound_attempt13_distilled_model
from ofc_regular.hu_m43_attempt13_distilled_runtime import (
    ATTEMPT13_BOUND_EXECUTION_MODULES,
    ATTEMPT13_RUNTIME_REQUIREMENTS_SHA256,
    validate_frozen_execution_modules,
)
from ofc_regular.hu_m4_t1_policy import HuM4T1SelectiveOverridePolicy
from ofc_regular.validate_hu_m43_attempt13_acceptance import (
    ATTEMPT13_BASELINE_PROFILE,
    ATTEMPT13_OPPONENTS,
    ATTEMPT13_POPULATION_NAMESPACE_BASES,
    ATTEMPT13_POPULATION_PREFLIGHT_SCHEMA,
    ATTEMPT13_POPULATION_SEED,
    ATTEMPT13_POPULATION_SEEDS_PER_SHARD,
    ATTEMPT13_POPULATION_SEED_STRIDE,
    ATTEMPT13_POPULATION_SHARDS,
    ATTEMPT13_PROFILE_ID,
    build_attempt13_population_preflight,
    load_bound_attempt13_distilled_model,
)


PolicyFactory = Callable[..., Any]


def evaluate_attempt13_population_shard(
    *,
    bound_model: object,
    preflight: Mapping[str, Any],
    baseline_policy_factory: PolicyFactory,
    opponent_policy_factories: Mapping[str, PolicyFactory],
    shard_index: int,
    records_output: str | Path,
    trace_fn: Callable[..., Mapping[str, Any]] | None = None,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Evaluate a fixed 50-seed shard; records publish only after completion."""

    if not is_bound_attempt13_distilled_model(bound_model):
        raise TypeError("Attempt13 population requires a bound runtime model")
    if (
        preflight.get("schema") != ATTEMPT13_POPULATION_PREFLIGHT_SCHEMA
        or preflight.get("status") != "pass"
        or preflight.get("profile_id") != ATTEMPT13_PROFILE_ID
        or preflight.get("runtime_binding_verified") is not True
        or preflight.get("model_sha256") is None
        or preflight.get("model_id") != bound_model.model_id
        or preflight.get("opponents") != list(ATTEMPT13_OPPONENTS)
        or preflight.get("baseline_profile") != ATTEMPT13_BASELINE_PROFILE
        or preflight.get("population_namespace_bases")
        != list(ATTEMPT13_POPULATION_NAMESPACE_BASES)
        or preflight.get("teacher_overlap_count") != 0
        or preflight.get("prior_population_overlap_count") != 0
        or preflight.get("all_reserved_registry_overlap_count") != 0
        or preflight.get("audit50_one_shot_go_bound") is not True
        or preflight.get("development200_full_fit_bound") is not True
        or preflight.get("threshold_reselection_performed") is not False
        or preflight.get("current_profile_mutated") is not False
        or preflight.get("no_runtime_activation") is not True
    ):
        raise ValueError("Attempt13 population preflight is incomplete")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise TypeError("Attempt13 shard index must be an integer")
    if not 0 <= shard_index < ATTEMPT13_POPULATION_SHARDS:
        raise ValueError("Attempt13 shard index must be in 0..19")
    if tuple(opponent_policy_factories) != ATTEMPT13_OPPONENTS:
        raise ValueError("Attempt13 population opponent set/order changed")

    def candidate_factory(
        *, policy_seed: int, seat: str, decision_log: list[dict[str, Any]] | None
    ) -> Any:
        baseline = baseline_policy_factory(
            policy_seed=policy_seed,
            seat=seat,
            decision_log=None,
        )
        return HuM4T1SelectiveOverridePolicy(
            baseline,
            action_value_model=bound_model,
            safety_model=bound_model,
            safety_probability_threshold=0.5,
            enabled=True,
            allowed_seats=("second",),
            decision_log=decision_log,
            policy_id="attempt13_population_evaluation_only",
            runtime_binding_verified=True,
        )

    destination = Path(records_output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Attempt13 population records are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shard_seed = (
        ATTEMPT13_POPULATION_SEED
        + shard_index
        * ATTEMPT13_POPULATION_SEEDS_PER_SHARD
        * ATTEMPT13_POPULATION_SEED_STRIDE
    )
    kwargs: dict[str, Any] = {}
    if trace_fn is not None:
        kwargs["trace_fn"] = trace_fn
    try:
        result = evaluate_hu_m4_population(
            candidate_policy_factory=candidate_factory,
            baseline_policy_factory=baseline_policy_factory,
            opponent_policy_factories=opponent_policy_factories,
            paired_seeds=ATTEMPT13_POPULATION_SEEDS_PER_SHARD,
            seed=shard_seed,
            seed_stride=ATTEMPT13_POPULATION_SEED_STRIDE,
            records_output=temporary,
            progress_every=progress_every,
            baseline_profile=ATTEMPT13_BASELINE_PROFILE,
            **kwargs,
        )
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    result["records_output"] = str(destination)
    result["runtime_config"] = {
        "profile_id": ATTEMPT13_PROFILE_ID,
        "baseline_profile": ATTEMPT13_BASELINE_PROFILE,
        "opponents": list(ATTEMPT13_OPPONENTS),
        "candidate_model_sha256": preflight["model_sha256"],
        "safety_model_sha256": preflight["model_sha256"],
        "model_id": preflight["model_id"],
        "model_schema": bound_model.schema,
        "artifact_schema": bound_model.artifact_schema,
        "feature_schema": bound_model.feature_schema,
        "head_schema": bound_model.head_schema,
        "action_score_mode": bound_model.action_score_mode,
        "runtime_binding_verified": True,
        "freeze_manifest_sha256": preflight["runtime_freeze_sha256"],
        "training_manifest_sha256": preflight["training_manifest_sha256"],
        "runtime_source_manifest_sha256": preflight[
            "runtime_source_manifest_sha256"
        ],
        "runtime_source_closure_sha256": preflight[
            "runtime_source_closure_sha256"
        ],
        "runtime_semantic_closure_sha256": preflight[
            "runtime_semantic_closure_sha256"
        ],
        "runtime_requirements_sha256": ATTEMPT13_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": preflight["runtime_fingerprint_sha256"],
        "source_model_manifest_sha256": preflight[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": preflight[
            "source_native_manifest_sha256"
        ],
        "runtime_dependency_closure_sha256": preflight[
            "runtime_dependency_closure_sha256"
        ],
        "seed_registry_sha256": preflight["seed_registry_sha256"],
        "population_namespace_bases": list(ATTEMPT13_POPULATION_NAMESPACE_BASES),
        "population_plan_sha256": preflight["population_plan_file_sha256"],
        "current_profile_used": False,
        "promotion_artifact_contract": True,
        "diagnostic_legacy": False,
        "safety_enabled": True,
        "safety_threshold": 0.5,
        "sharded_evaluation": True,
        "shard_count": ATTEMPT13_POPULATION_SHARDS,
        "final_metrics_recomputed_from_merged_records": False,
    }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--runtime-freeze", required=True, type=Path)
    parser.add_argument("--population-plan", required=True, type=Path)
    parser.add_argument("--runtime-source-archive", required=True, type=Path)
    parser.add_argument("--runtime-source-manifest", required=True, type=Path)
    parser.add_argument("--runtime-source-root", required=True, type=Path)
    parser.add_argument("--runtime-dependency-root", required=True, type=Path)
    parser.add_argument("--development-decision", required=True, type=Path)
    parser.add_argument("--development-selector-receipt", required=True, type=Path)
    parser.add_argument("--development-pass-freeze", required=True, type=Path)
    parser.add_argument("--audit-decision", required=True, type=Path)
    parser.add_argument("--audit-selector-receipt", required=True, type=Path)
    parser.add_argument("--opening-lookahead-samples", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--records-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Attempt13 population shard summary is immutable")
    preflight = build_attempt13_population_preflight(
        model_path=args.model,
        training_manifest_path=args.training_manifest,
        runtime_freeze_path=args.runtime_freeze,
        population_plan_path=args.population_plan,
        runtime_source_archive_path=args.runtime_source_archive,
        runtime_source_manifest_path=args.runtime_source_manifest,
        runtime_source_root=args.runtime_source_root,
        runtime_dependency_root=args.runtime_dependency_root,
        development_decision_path=args.development_decision,
        development_selector_receipt_path=args.development_selector_receipt,
        development_pass_freeze_path=args.development_pass_freeze,
        audit_decision_path=args.audit_decision,
        audit_selector_receipt_path=args.audit_selector_receipt,
    )
    # Population play imports these explicit dispatch modules in addition to
    # the bound model closure; verify they also came from the extracted tree.
    validate_frozen_execution_modules(
        extracted_root=args.runtime_source_root,
        manifest=args.runtime_source_manifest,
        module_names=tuple(ATTEMPT13_BOUND_EXECUTION_MODULES)
        + (
            "ofc_regular.ai_profiles",
            "ofc_regular.evaluate_matchups",
            "ofc_regular_promotion.attempt13_population",
        ),
    )
    bound_model = load_bound_attempt13_distilled_model(
        args.model,
        expected_sha256=args.expected_model_sha256,
        runtime_freeze=args.runtime_freeze,
        training_manifest_path=args.training_manifest,
        runtime_source_manifest_path=args.runtime_source_manifest,
        runtime_source_root=args.runtime_source_root,
        runtime_dependency_root=args.runtime_dependency_root,
    )
    needed: set[str] = set()
    for opponent in ATTEMPT13_OPPONENTS:
        needed.update(required_profiles(ATTEMPT13_BASELINE_PROFILE, opponent))
    bundle = load_model_bundle(ModelPaths(), needed)

    def baseline_factory(
        *, policy_seed: int, seat: str, decision_log: list[dict[str, Any]] | None
    ) -> Any:
        del decision_log
        return build_policy(
            ATTEMPT13_BASELINE_PROFILE,
            bundle,
            seed=policy_seed,
            seat=seat,
            opening_lookahead_samples=args.opening_lookahead_samples,
        )

    opponents: dict[str, PolicyFactory] = {}
    for opponent_name in ATTEMPT13_OPPONENTS:

        def opponent_factory(
            *,
            policy_seed: int,
            seat: str,
            decision_log: list[dict[str, Any]] | None,
            _profile: str = opponent_name,
        ) -> Any:
            del decision_log
            return build_policy(
                _profile,
                bundle,
                seed=policy_seed,
                seat=seat,
                opening_lookahead_samples=args.opening_lookahead_samples,
            )

        opponents[opponent_name] = opponent_factory
    result = evaluate_attempt13_population_shard(
        bound_model=bound_model,
        preflight=preflight,
        baseline_policy_factory=baseline_factory,
        opponent_policy_factories=opponents,
        shard_index=args.shard_index,
        records_output=args.records_output,
        progress_every=args.progress_every,
    )
    _write_new_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_attempt13_population_shard", "main"]
