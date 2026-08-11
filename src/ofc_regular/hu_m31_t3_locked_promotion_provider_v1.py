"""Production provider for the frozen M3.1 260-item evaluation grid.

Nothing in this module runs at import time.  The provider requires the
content-addressed runtime closure, the accepted StreetPolicyNetV1 candidate,
all three independently frozen ABR checkpoints, and the exact execution plan
before it can construct a runner.  It never resolves ``current`` and never
registers or activates a profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_abr_v1 as abr
from . import hu_m31_t3_abr_cli_v1 as abr_cli
from . import hu_m31_t3_locked_promotion_execution_v1 as execution
from . import hu_m31_t3_locked_promotion_production_v1 as production
from . import hu_m31_t3_locked_promotion_runner_v1 as runner_module
from . import hu_m31_t3_promotion_runtime_closure_v1 as closure
from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion


PROVIDER_SCHEMA = "hu_m31_t3_locked_promotion_provider_v1"
PROVIDER_RECEIPT_SCHEMA = (
    "hu_m31_t3_locked_promotion_provider_work_item_receipt_v1"
)
_SHA_CHARS = frozenset("0123456789abcdef")


class LockedPromotionProviderError(RuntimeError):
    """Raised when production inputs do not form one immutable runner."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA_CHARS
    )


def _read_canonical(
    path: str | Path, label: str
) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise LockedPromotionProviderError(
            f"{label} must be a regular non-symlink file"
        )
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockedPromotionProviderError(
            f"{label} is not canonical JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or raw != promotion.canonical_bytes(value)
    ):
        raise LockedPromotionProviderError(
            f"{label} is not a canonical JSON object"
        )
    return value, raw


def _write_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    raw = promotion.canonical_bytes(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != raw
        ):
            raise FileExistsError(
                "immutable provider receipt already exists with other bytes"
            ) from None
    return destination.resolve()


def load_abr_policy_bindings(
    *,
    promotion_plan: Mapping[str, Any],
    candidate_bundle_directory: str | Path,
    abr_bundle_directory: str | Path,
    expected_abr_bundle_file_sha256: str,
    torch: Any,
    legacy_policy_factory: Any,
    exact_t4_solver: Any | None = None,
) -> dict[str, runner_module.AbrPolicyBinding]:
    """Load three T3-only ABRs over the pinned legacy early-street chain."""

    plan = promotion.validate_locked_promotion_plan(promotion_plan)
    if not _is_sha256(expected_abr_bundle_file_sha256):
        raise ValueError("ABR bundle.json SHA-256 must be pinned")
    if (
        not callable(legacy_policy_factory)
        or getattr(legacy_policy_factory, "profile_id", None)
        != abr.ABR_LEGACY_PROFILE_BY_STREET["T0"]
        or getattr(legacy_policy_factory, "factory_id", None)
        != (
            f"{closure.NAMED_PROFILE_FACTORY_PREFIX}:"
            f"{abr.ABR_LEGACY_PROFILE_BY_STREET['T0']}"
        )
    ):
        raise LockedPromotionProviderError(
            "ABR early-street legacy factory must be explicit stage19_p0"
        )
    if exact_t4_solver is None:
        raise LockedPromotionProviderError(
            "production ABR bindings require the pinned exact-T4 solver"
        )
    root = Path(abr_bundle_directory)
    if root.is_symlink() or not root.is_dir():
        raise LockedPromotionProviderError(
            "ABR bundle directory is missing or unsafe"
        )
    bundle_path = root / "bundle.json"
    bundle, bundle_raw = _read_canonical(
        bundle_path, "ABR policy bundle"
    )
    if (
        hashlib.sha256(bundle_raw).hexdigest()
        != expected_abr_bundle_file_sha256
    ):
        raise LockedPromotionProviderError(
            "pinned ABR bundle.json SHA-256 changed"
        )
    validated = abr.validate_policy_bundle(
        bundle,
        directory=root,
        promotion_plan=plan,
        candidate_bundle_directory=candidate_bundle_directory,
        torch=torch,
    )
    bindings: dict[str, runner_module.AbrPolicyBinding] = {}
    for response_id, record in zip(
        abr.RESPONSE_IDS,
        validated["families"],
        strict=True,
    ):
        if record["response_id"] != response_id:
            raise LockedPromotionProviderError(
                "ABR family order changed after validation"
            )
        manifest_path = root / str(record["manifest_filename"])
        checkpoint_path = root / str(record["checkpoint_filename"])
        factory = abr.load_artifact_bound_policy_factory(
            response_id=response_id,
            manifest_path=manifest_path,
            expected_manifest_file_sha256=str(
                record["manifest_sha256"]
            ),
            checkpoint_path=checkpoint_path,
            expected_checkpoint_file_sha256=str(
                record["checkpoint_sha256"]
            ),
            promotion_plan=plan,
            candidate_bundle_directory=candidate_bundle_directory,
            torch=torch,
            legacy_policy_factory=legacy_policy_factory,
            exact_t4_solver=exact_t4_solver,
        )
        if (
            factory.legacy_profile_by_street
            != abr.ABR_LEGACY_PROFILE_BY_STREET
            or factory.learned_streets != (abr.ABR_LEARNED_STREET,)
            or factory.exact_streets != ("T4",)
            or factory.runtime_street_composition_sha256
            != abr.ABR_RUNTIME_STREET_COMPOSITION_SHA256
            or factory.scientific_role
            != abr.ABR_RESPONSE_SCIENTIFIC_ROLE[response_id]
        ):
            raise LockedPromotionProviderError(
                "ABR runtime street composition or scientific role changed"
            )
        binding = runner_module.AbrPolicyBinding(
            response_id=response_id,
            policy_factory=factory,
            manifest_path=manifest_path.resolve(),
            expected_manifest_file_sha256=str(
                record["manifest_sha256"]
            ),
            checkpoint_path=checkpoint_path.resolve(),
            expected_checkpoint_file_sha256=str(
                record["checkpoint_sha256"]
            ),
        )
        runner_module.validate_abr_policy_binding(binding, plan=plan)
        bindings[response_id] = binding
    if tuple(bindings) != abr.RESPONSE_IDS:
        raise LockedPromotionProviderError(
            "production ABR binding coverage changed"
        )
    return bindings


def build_bound_runner(
    *,
    prepared: production.PreparedLockedPromotion,
    abr_bundle_directory: str | Path,
    expected_abr_bundle_file_sha256: str,
    expected_abr_production_build_receipt_sha256: str,
    torch: Any,
) -> runner_module.LockedPromotionRunner:
    """Construct the production runner only after all bindings replay."""

    root = Path(abr_bundle_directory)
    bundle, bundle_raw = _read_canonical(
        root / "bundle.json", "ABR policy bundle"
    )
    build_receipt, build_receipt_raw = _read_canonical(
        root / abr_cli.ABR_PRODUCTION_BUILD_RECEIPT_FILE,
        "ABR production build receipt",
    )
    if (
        not _is_sha256(expected_abr_production_build_receipt_sha256)
        or hashlib.sha256(build_receipt_raw).hexdigest()
        != expected_abr_production_build_receipt_sha256
    ):
        raise LockedPromotionProviderError(
            "pinned ABR production build receipt SHA-256 changed"
        )
    abr_cli.validate_production_build_receipt(
        build_receipt,
        bundle_file_sha256=hashlib.sha256(bundle_raw).hexdigest(),
        bundle_identity_sha256=str(bundle["bundle_identity_sha256"]),
    )
    factories = production.build_production_policy_factories(
        prepared,
        torch=torch,
    )
    bindings = load_abr_policy_bindings(
        promotion_plan=prepared.plan,
        candidate_bundle_directory=(
            prepared.candidate_checkpoint_bundle_path
        ),
        abr_bundle_directory=abr_bundle_directory,
        expected_abr_bundle_file_sha256=(
            expected_abr_bundle_file_sha256
        ),
        torch=torch,
        legacy_policy_factory=factories.population["stage19_p0"],
        exact_t4_solver=factories.baseline.exact_t4_solver,
    )
    return production.build_production_locked_promotion_runner(
        factories,
        abr_policy_bindings=bindings,
    )


def run_provider_work_item(
    *,
    plan_path: str | Path,
    closure_package_path: str | Path,
    expected_closure_package_sha256: str,
    extraction_root: str | Path,
    source_replay_root: str | Path,
    compatibility_threshold_lock_path: str | Path,
    policy_registry_path: str | Path,
    abr_bundle_directory: str | Path,
    expected_abr_bundle_file_sha256: str,
    expected_abr_production_build_receipt_sha256: str,
    execution_plan_path: str | Path,
    work_id: str,
    shard_directory: str | Path,
    torch: Any,
    require_host_target: bool = True,
) -> dict[str, Any]:
    """Run exactly one item from the source-replayed 260-item grid."""

    prepared = production.prepare_locked_execution_plan(
        plan_path=plan_path,
        closure_package_path=closure_package_path,
        expected_closure_package_sha256=(
            expected_closure_package_sha256
        ),
        extraction_root=extraction_root,
        source_replay_root=source_replay_root,
        compatibility_threshold_lock_path=(
            compatibility_threshold_lock_path
        ),
        policy_registry_path=policy_registry_path,
        require_host_target=require_host_target,
    )
    execution_plan, execution_raw = _read_canonical(
        execution_plan_path, "locked-promotion execution plan"
    )
    validated_execution = execution.validate_execution_plan(
        execution_plan,
        promotion_plan=prepared.plan,
    )
    item = execution.get_work_item(
        execution_plan=validated_execution,
        promotion_plan=prepared.plan,
        work_id=work_id,
    )
    runner = build_bound_runner(
        prepared=prepared,
        abr_bundle_directory=abr_bundle_directory,
        expected_abr_bundle_file_sha256=(
            expected_abr_bundle_file_sha256
        ),
        expected_abr_production_build_receipt_sha256=(
            expected_abr_production_build_receipt_sha256
        ),
        torch=torch,
    )
    shard = execution.run_work_item(
        runner=runner,
        promotion_plan=prepared.plan,
        execution_plan=validated_execution,
        work_id=work_id,
        shard_directory=shard_directory,
    )
    core = {
        "schema": PROVIDER_RECEIPT_SCHEMA,
        "provider_schema": PROVIDER_SCHEMA,
        "status": "complete_one_frozen_work_item",
        "work_id": work_id,
        "work_ordinal": item["ordinal"],
        "schedule": item["schedule"],
        "entity_id": item["entity_id"],
        "promotion_plan_sha256": promotion.canonical_sha256(
            prepared.plan
        ),
        "execution_plan_file_sha256": hashlib.sha256(
            execution_raw
        ).hexdigest(),
        "execution_plan_sha256": promotion.canonical_sha256(
            validated_execution
        ),
        "abr_bundle_file_sha256": expected_abr_bundle_file_sha256,
        "abr_production_build_receipt_file_sha256": (
            expected_abr_production_build_receipt_sha256
        ),
        "abr_runtime_street_composition_sha256": (
            abr.ABR_RUNTIME_STREET_COMPOSITION_SHA256
        ),
        "shard_sha256": promotion.canonical_sha256(shard),
        "row_count": shard["row_count"],
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
    }
    return {
        **core,
        "receipt_sha256": promotion.canonical_sha256(core),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one immutable item from the M3.1 locked 260-item "
            "population/ABR grid."
        )
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--closure-package", required=True)
    parser.add_argument("--expected-closure-sha256", required=True)
    parser.add_argument("--extraction-root", required=True)
    parser.add_argument("--source-replay-root", required=True)
    parser.add_argument("--compatibility-threshold-lock", required=True)
    parser.add_argument("--policy-registry", required=True)
    parser.add_argument("--abr-bundle-directory", required=True)
    parser.add_argument("--expected-abr-bundle-sha256", required=True)
    parser.add_argument(
        "--expected-abr-production-build-receipt-sha256",
        required=True,
    )
    parser.add_argument("--execution-plan", required=True)
    parser.add_argument("--work-id", required=True)
    parser.add_argument("--shard-directory", required=True)
    parser.add_argument("--output-receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import torch

        receipt = run_provider_work_item(
            plan_path=args.plan,
            closure_package_path=args.closure_package,
            expected_closure_package_sha256=args.expected_closure_sha256,
            extraction_root=args.extraction_root,
            source_replay_root=args.source_replay_root,
            compatibility_threshold_lock_path=(
                args.compatibility_threshold_lock
            ),
            policy_registry_path=args.policy_registry,
            abr_bundle_directory=args.abr_bundle_directory,
            expected_abr_bundle_file_sha256=(
                args.expected_abr_bundle_sha256
            ),
            expected_abr_production_build_receipt_sha256=(
                args.expected_abr_production_build_receipt_sha256
            ),
            execution_plan_path=args.execution_plan,
            work_id=args.work_id,
            shard_directory=args.shard_directory,
            torch=torch,
            require_host_target=True,
        )
        if args.output_receipt:
            _write_once(args.output_receipt, receipt)
    except Exception as exc:  # pragma: no cover - subprocess boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LockedPromotionProviderError",
    "PROVIDER_RECEIPT_SCHEMA",
    "PROVIDER_SCHEMA",
    "build_bound_runner",
    "load_abr_policy_bindings",
    "main",
    "run_provider_work_item",
]
