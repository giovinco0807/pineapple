"""Production-only factory wiring for the M3.1 locked promotion plan.

This module is deliberately separate from the generic scientific runner.  It
constructs every named policy from a validated, extracted runtime closure,
wraps all policies with the pinned exact-T4 solver, and builds the
StreetPolicyNetV1 evaluation-only candidate from the plan-bound evidence.

No factory resolves ``current``.  The CLI only prepares and preflights a
dormant execution directory.  A runnable ``LockedPromotionRunner`` is returned
by the API only after all three independently frozen ABR bindings are supplied;
until then the receipt says that locked execution is not ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import ai_profiles
from . import hu_m31_t3_locked_promotion_runner_v1 as runner_module
from . import hu_m31_t3_promotion_runtime_closure_v1 as closure
from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from . import hu_m31_t3_street_policy_runtime_v1 as policy_runtime
from .hu_m3_t4_runtime import (
    HuM3T4ExactPolicy,
    HuM3T4ExactSolver,
    HuM3T4RuntimeConfig,
)


PRODUCTION_FACTORY_SCHEMA = "hu_m31_t3_locked_promotion_production_v1"
PREPARE_RECEIPT_SCHEMA = (
    "hu_m31_t3_locked_promotion_production_prepare_receipt_v1"
)
_SHA_CHARS = frozenset("0123456789abcdef")


class LockedPromotionProductionError(RuntimeError):
    """Raised when production wiring differs from the frozen closure."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA_CHARS
    )


def _read_canonical(path: str | Path, label: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise LockedPromotionProductionError(
            f"{label} must be a regular non-symlink file"
        )
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockedPromotionProductionError(
            f"{label} is not canonical JSON"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise LockedPromotionProductionError(
            f"{label} is not a canonical JSON object"
        )
    return value, raw


def _write_once(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("production prepare receipt is create-only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes(value)
    with destination.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _host_os() -> str:
    value = platform.system().casefold()
    if value == "darwin":
        return "macos"
    if value in {"linux", "windows"}:
        return value
    raise LockedPromotionProductionError(
        f"unsupported locked-evaluation host OS: {value}"
    )


def _extracted_path(root: Path, archive_name: str) -> Path:
    relative = PurePosixPath(archive_name)
    return root.joinpath(*relative.parts)


@dataclass(frozen=True)
class PreparedLockedPromotion:
    """Validated paths for one dormant, immutable execution closure."""

    plan: Mapping[str, Any]
    plan_path: Path
    plan_file_sha256: str
    closure_manifest: Mapping[str, Any]
    closure_package_path: Path
    closure_package_sha256: str
    extraction_root: Path
    source_replay_root: Path
    compatibility_threshold_lock_path: Path
    policy_registry_path: Path
    candidate_checkpoint_bundle_path: Path
    training_threshold_lock_path: Path
    exact_t4_native_library_path: Path


@dataclass(frozen=True)
class FrozenNamedProfileFactory:
    """Callable exact named-profile factory; never accepts ``current``."""

    profile_id: str
    bundle: ai_profiles.ModelBundle
    exact_t4_solver: HuM3T4ExactSolver
    factory_id: str

    def __call__(self, *, policy_seed: int, seat: str) -> object:
        if self.profile_id not in closure.POPULATION_PROFILE_IDS:
            raise LockedPromotionProductionError(
                "named policy factory escaped the frozen population"
            )
        if seat not in promotion.SEATS:
            raise ValueError("named policy seat must be first or second")
        policy = ai_profiles.build_policy(
            self.profile_id,
            self.bundle,
            seed=policy_seed,
            seat=seat,
            opening_lookahead_samples=closure.OPENING_LOOKAHEAD_SAMPLES,
        )
        wrapped = HuM3T4ExactPolicy(policy, self.exact_t4_solver)
        setattr(wrapped, "locked_policy_factory_id", self.factory_id)
        setattr(wrapped, "current_profile_resolved", False)
        setattr(wrapped, "opponent_private_discards_used", False)
        return wrapped


@dataclass(frozen=True)
class LockedEvaluationCandidateFactory:
    """Callable plan-bound StreetPolicyNetV1 evaluation-only factory."""

    baseline_factory: FrozenNamedProfileFactory
    template_runtime: policy_runtime.HuM31T3StreetPolicyRuntime
    prepared: PreparedLockedPromotion
    factory_id: str = closure.CANDIDATE_FACTORY_ID

    def __call__(self, *, policy_seed: int, seat: str) -> object:
        baseline = self.baseline_factory(
            policy_seed=policy_seed,
            seat=seat,
        )
        candidate = self.template_runtime.spawn_with_baseline(
            baseline,
            baseline_profile_id=policy_runtime.BASELINE_PROFILE,
        )
        if (
            candidate.runtime_scope
            != policy_runtime.RUNTIME_SCOPE_EVALUATION_ONLY
            or candidate.evaluation_only is not True
            or candidate.profile_candidate
            != policy_runtime.EVALUATION_ONLY_CANDIDATE
        ):
            raise LockedPromotionProductionError(
                "candidate factory crossed the evaluation-only boundary"
            )
        return candidate


@dataclass(frozen=True)
class ProductionPolicyFactories:
    prepared: PreparedLockedPromotion
    candidate: LockedEvaluationCandidateFactory
    baseline: FrozenNamedProfileFactory
    population: Mapping[str, FrozenNamedProfileFactory]


def prepare_locked_execution_plan(
    *,
    plan_path: str | Path,
    closure_package_path: str | Path,
    expected_closure_package_sha256: str,
    extraction_root: str | Path,
    source_replay_root: str | Path,
    compatibility_threshold_lock_path: str | Path,
    policy_registry_path: str | Path,
    require_host_target: bool = True,
) -> PreparedLockedPromotion:
    """Extract and source-replay the exact execution closure for a plan."""

    if not _is_sha256(expected_closure_package_sha256):
        raise ValueError("closure package SHA-256 must be pinned")
    plan_value, plan_raw = _read_canonical(
        plan_path, "locked promotion plan"
    )
    try:
        plan = promotion.validate_locked_promotion_plan(plan_value)
    except (TypeError, ValueError) as exc:
        raise LockedPromotionProductionError(
            "locked promotion plan failed source validation"
        ) from exc
    closure_binding = plan["artifact_binding"]["evaluation_runtime_closure"]
    if (
        closure_binding["sha256"] != expected_closure_package_sha256
        or closure_binding["filename"]
        != Path(closure_package_path).name
    ):
        raise LockedPromotionProductionError(
            "plan is bound to another runtime closure package"
        )
    source_root = Path(source_replay_root).resolve()
    package = Path(closure_package_path).resolve()
    extraction = Path(extraction_root).resolve()
    closure.extract_runtime_closure_package(
        package,
        expected_sha256=expected_closure_package_sha256,
        output_directory=extraction,
        source_replay_root=source_root,
    )
    manifest = closure.validate_runtime_closure_package(
        package,
        expected_sha256=expected_closure_package_sha256,
        source_replay_root=source_root,
    )
    if require_host_target and manifest["exact_t4_native"]["target_os"] != _host_os():
        raise LockedPromotionProductionError(
            "runtime closure exact-T4 library targets another OS"
        )
    candidate_manifest = _extracted_path(
        extraction,
        manifest["candidate_artifacts"]["checkpoint_manifest_path"],
    )
    candidate_bundle = candidate_manifest.parent
    rich_threshold = _extracted_path(
        extraction,
        manifest["candidate_artifacts"]["training_threshold_lock_path"],
    )
    exact_t4 = _extracted_path(
        extraction, manifest["exact_t4_native"]["path"]
    )
    compatibility = Path(compatibility_threshold_lock_path).resolve()
    registry = Path(policy_registry_path).resolve()
    try:
        promotion.validate_artifact_files(
            plan,
            model_path=candidate_manifest,
            threshold_lock_path=compatibility,
            policy_registry_path=registry,
            evaluation_runtime_closure_path=package,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise LockedPromotionProductionError(
            "plan artifacts differ from the extracted execution closure"
        ) from exc
    if (
        closure.sha256_file(rich_threshold)
        != manifest["candidate_artifacts"][
            "training_threshold_lock_sha256"
        ]
        or closure.sha256_file(exact_t4)
        != manifest["exact_t4_native"]["sha256"]
        or closure.sha256_file(
            _extracted_path(
                extraction, "src/ofc_regular/ai_profiles.py"
            )
        )
        != plan["artifact_binding"]["policy_registry"]["sha256"]
        or manifest["candidate_artifacts"]["checkpoint_manifest_sha256"]
        != plan["artifact_binding"]["model"]["sha256"]
    ):
        raise LockedPromotionProductionError(
            "extracted execution artifact binding changed"
        )
    return PreparedLockedPromotion(
        plan=plan,
        plan_path=Path(plan_path).resolve(),
        plan_file_sha256=_sha256(plan_raw),
        closure_manifest=manifest,
        closure_package_path=package,
        closure_package_sha256=expected_closure_package_sha256,
        extraction_root=extraction,
        source_replay_root=source_root,
        compatibility_threshold_lock_path=compatibility,
        policy_registry_path=registry,
        candidate_checkpoint_bundle_path=candidate_bundle,
        training_threshold_lock_path=rich_threshold,
        exact_t4_native_library_path=exact_t4,
    )


def _model_paths(prepared: PreparedLockedPromotion) -> ai_profiles.ModelPaths:
    bindings = prepared.closure_manifest["legacy_model_bindings"]
    values = {
        field: _extracted_path(prepared.extraction_root, relative)
        for field, relative in bindings.items()
    }
    return ai_profiles.ModelPaths(**values)


def build_production_policy_factories(
    prepared: PreparedLockedPromotion,
    *,
    torch: Any,
) -> ProductionPolicyFactories:
    """Load pinned models and return only explicit plan factories."""

    manifest = prepared.closure_manifest
    if manifest["exact_t4_native"]["target_os"] != _host_os():
        raise LockedPromotionProductionError(
            "cannot execute a cross-OS exact-T4 native closure"
        )
    profiles = set(closure.POPULATION_PROFILE_IDS)
    if "current" in profiles:
        raise AssertionError("production profile grid must never contain current")
    bundle = ai_profiles.load_model_bundle(_model_paths(prepared), profiles)
    missing_models = [
        field
        for field in prepared.closure_manifest["legacy_model_bindings"]
        if getattr(bundle, field, None) is None
    ]
    if missing_models:
        raise LockedPromotionProductionError(
            "pinned legacy model failed semantic load: "
            + ", ".join(sorted(missing_models))
        )
    solver = HuM3T4ExactSolver(
        HuM3T4RuntimeConfig(
            expected_library_sha256=manifest["exact_t4_native"]["sha256"],
            library_path=prepared.exact_t4_native_library_path,
        )
    )
    population: dict[str, FrozenNamedProfileFactory] = {}
    for record in manifest["factory_contract"]["population"]:
        profile_id = str(record["profile_id"])
        population[profile_id] = FrozenNamedProfileFactory(
            profile_id=profile_id,
            bundle=bundle,
            exact_t4_solver=solver,
            factory_id=str(record["factory_id"]),
        )
    if tuple(population) != closure.POPULATION_PROFILE_IDS:
        raise LockedPromotionProductionError(
            "production population factory order changed"
        )
    baseline = population[closure.BASELINE_PROFILE_ID]
    candidate_manifest, _ = _read_canonical(
        prepared.candidate_checkpoint_bundle_path / "manifest.json",
        "extracted candidate checkpoint manifest",
    )
    training_config = closure._training_config_from_manifest(  # type: ignore[attr-defined]
        candidate_manifest
    )
    template_runtime = policy_runtime.build_locked_evaluation_t3_policy_candidate(
        baseline_policy=baseline(policy_seed=0, seat="first"),
        baseline_profile_id=policy_runtime.BASELINE_PROFILE,
        torch=torch,
        checkpoint_bundle_path=prepared.candidate_checkpoint_bundle_path,
        training_config=training_config,
        expected_dataset_identity_sha256=str(
            candidate_manifest["training_view_identity_sha256"]
        ),
        expected_bundle_identity_sha256=str(
            candidate_manifest["bundle_identity_sha256"]
        ),
        threshold_lock_path=prepared.training_threshold_lock_path,
        expected_threshold_lock_file_sha256=str(
            manifest["candidate_artifacts"][
                "training_threshold_lock_sha256"
            ]
        ),
        evaluation_evidence=policy_runtime.LockedEvaluationEvidencePaths(
            plan_path=prepared.plan_path,
            expected_plan_file_sha256=prepared.plan_file_sha256,
            compatibility_threshold_lock_path=(
                prepared.compatibility_threshold_lock_path
            ),
            policy_registry_path=prepared.policy_registry_path,
            evaluation_runtime_closure_path=(
                prepared.closure_package_path
            ),
        ),
    )
    candidate = LockedEvaluationCandidateFactory(
        baseline_factory=baseline,
        template_runtime=template_runtime,
        prepared=prepared,
    )
    if (
        candidate.factory_id
        != manifest["factory_contract"]["candidate"]["factory_id"]
        or baseline.factory_id
        != manifest["factory_contract"]["baseline"]["factory_id"]
    ):
        raise LockedPromotionProductionError(
            "production candidate/baseline factory ID changed"
        )
    return ProductionPolicyFactories(
        prepared=prepared,
        candidate=candidate,
        baseline=baseline,
        population=population,
    )


def build_production_locked_promotion_runner(
    factories: ProductionPolicyFactories,
    *,
    abr_policy_bindings: Mapping[
        str, runner_module.AbrPolicyBinding
    ],
) -> runner_module.LockedPromotionRunner:
    """Build the runner only after all exact ABR factory IDs are present."""

    expected_abr = factories.prepared.closure_manifest[
        "factory_contract"
    ]["abr"]
    if tuple(abr_policy_bindings) != tuple(
        record["response_id"] for record in expected_abr
    ):
        raise LockedPromotionProductionError(
            "production ABR binding order/coverage changed"
        )
    for record in expected_abr:
        binding = abr_policy_bindings[record["response_id"]]
        factory_id = getattr(binding.policy_factory, "factory_id", None)
        if factory_id != record["factory_id"]:
            raise LockedPromotionProductionError(
                f"ABR factory ID changed for {record['response_id']}"
            )
    readiness = prepare_receipt(
        factories.prepared,
        abr_policy_bindings=abr_policy_bindings,
    )
    if readiness["locked_execution_ready"] is not True:
        raise LockedPromotionProductionError(
            "all frozen ABR bindings are required before runner creation"
        )
    return runner_module.LockedPromotionRunner(
        plan=factories.prepared.plan,
        candidate_policy_factory=factories.candidate,
        baseline_policy_factory=factories.baseline,
        opponent_policy_factories=factories.population,
        abr_policy_bindings=abr_policy_bindings,
    )


def prepare_receipt(
    prepared: PreparedLockedPromotion,
    *,
    abr_policy_bindings: Mapping[
        str, runner_module.AbrPolicyBinding
    ]
    | None = None,
) -> dict[str, Any]:
    expected_abr = tuple(closure.ABR_FACTORY_IDS)
    observed: tuple[str, ...] = ()
    if abr_policy_bindings is not None:
        if tuple(abr_policy_bindings) != expected_abr:
            raise LockedPromotionProductionError(
                "prepare receipt ABR binding order/coverage changed"
            )
        for response_id, binding in abr_policy_bindings.items():
            expected_factory_id = closure.ABR_FACTORY_IDS[response_id]
            if (
                getattr(binding.policy_factory, "factory_id", None)
                != expected_factory_id
            ):
                raise LockedPromotionProductionError(
                    f"prepare receipt ABR factory ID changed: {response_id}"
                )
            runner_module.validate_abr_policy_binding(
                binding, plan=prepared.plan
            )
        observed = tuple(abr_policy_bindings)
    missing = [value for value in expected_abr if value not in observed]
    return {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "status": (
            "ready_for_factory_load"
            if not missing
            else "dormant_waiting_for_frozen_abr_bindings"
        ),
        "plan_file_sha256": prepared.plan_file_sha256,
        "plan_sha256": promotion.canonical_sha256(prepared.plan),
        "closure_package_sha256": prepared.closure_package_sha256,
        "closure_manifest_identity_sha256": prepared.closure_manifest[
            "manifest_identity_sha256"
        ],
        "candidate_checkpoint_bundle_identity_sha256": (
            prepared.closure_manifest["candidate_artifacts"][
                "checkpoint_bundle_identity_sha256"
            ]
        ),
        "population_factory_ids": [
            record["factory_id"]
            for record in prepared.closure_manifest["factory_contract"][
                "population"
            ]
        ],
        "candidate_factory_id": closure.CANDIDATE_FACTORY_ID,
        "abr_factory_ids": dict(closure.ABR_FACTORY_IDS),
        "missing_abr_bindings": missing,
        "locked_execution_ready": not missing,
        "cloud_execution_started": False,
        "promotion_authorized": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the M3.1 locked-promotion production factories"
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--closure-package", required=True)
    parser.add_argument("--expected-closure-sha256", required=True)
    parser.add_argument("--extraction-root", required=True)
    parser.add_argument("--source-replay-root", required=True)
    parser.add_argument("--compatibility-threshold-lock", required=True)
    parser.add_argument("--policy-registry", required=True)
    parser.add_argument("--output-receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prepared = prepare_locked_execution_plan(
            plan_path=args.plan,
            closure_package_path=args.closure_package,
            expected_closure_package_sha256=(
                args.expected_closure_sha256
            ),
            extraction_root=args.extraction_root,
            source_replay_root=args.source_replay_root,
            compatibility_threshold_lock_path=(
                args.compatibility_threshold_lock
            ),
            policy_registry_path=args.policy_registry,
        )
        receipt = prepare_receipt(prepared)
        _write_once(args.output_receipt, receipt)
    except Exception as exc:  # pragma: no cover - subprocess boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FrozenNamedProfileFactory",
    "LockedEvaluationCandidateFactory",
    "LockedPromotionProductionError",
    "PREPARE_RECEIPT_SCHEMA",
    "PRODUCTION_FACTORY_SCHEMA",
    "PreparedLockedPromotion",
    "ProductionPolicyFactories",
    "build_production_locked_promotion_runner",
    "build_production_policy_factories",
    "main",
    "prepare_locked_execution_plan",
    "prepare_receipt",
]
