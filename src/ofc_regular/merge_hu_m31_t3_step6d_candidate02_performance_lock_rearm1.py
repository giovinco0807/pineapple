"""Scientifically merge the fresh Candidate02 performance-lock rearm1.

The audited v1 scientific merger remains the implementation for DONE parsing,
candidate/reference isolation, exact hand coverage, root pairing, integrity,
performance gates, write-once output, and independent replay.  This adapter
activates the rearm1 plan/open/Spot producers only for one call and restores
all v1 globals in ``finally``.

Rearm1 has a wider open-input identity than v1 because it must bind the
terminal startup-failure receipt and the immutable old-v1 controls.  The
adapter therefore serializes those paths into the merge, reconstructs the
exact rearm1 input type during validation, and requires the fresh recovery-v2
contract plus zero old-v1 root/seed/fingerprint overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import hu_m31_t3_step6d_candidate02_performance_lock_rearm1_plan as lock_plan
from . import hu_m31_t3_step6d_performance_lock_rearm1_open as lock_open
from . import hu_m31_t3_step6d_performance_lock_rearm1_spot as lock_spot
from . import hu_m31_t3_step6d_performance_lock_spot_v1 as spot_v1
from . import merge_hu_m31_t3_step6d_candidate02_performance_lock as v1
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


MERGE_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_rearm1_scientific_merge_v1"
)
VALIDATION_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_rearm1_scientific_validation_v1"
)
SCOPE = "candidate02_one_shot_performance_lock_rearm1_fresh_roots"
PASS_DECISION = "performance_lock_rearm1_pass_authorize_fresh_quality_pilot_only"
NO_GO_DECISION = (
    "performance_lock_rearm1_no_go_candidate_finalized_no_rerun_or_reseed"
)

PerformanceLockReceiveInputs = v1.PerformanceLockReceiveInputs

_OPEN_INPUT_KEYS = frozenset(
    {
        "repository_root",
        "plan_path",
        "lock_output_directory",
        "candidate_library",
        "reference_library",
        "feature_encoder",
        "startup_source",
        "incident_receipt_path",
        "old_v1_root_directory",
        "development_summary_path",
        "development_validation_path",
        "development_root_directory",
        "old_v1_global_claim_path",
        "global_claim_path",
    }
)
_CONTEXT_LOCK = threading.RLock()


class _RunnerAdapter:
    """Delegate to the shared runner with the rearm recovery variant selected."""

    CANDIDATE02_PERFORMANCE_LOCK_VARIANT = (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
    )

    def __getattr__(self, name: str) -> Any:
        return getattr(runner, name)


class _SpotAdapter:
    """Expose rearm receive replay with the shared immutable receipt schemas."""

    DEFAULT_PROJECT = lock_spot.DEFAULT_PROJECT
    DEFAULT_BUCKET = lock_spot.DEFAULT_BUCKET
    RECEIVE_SCHEMA = spot_v1.RECEIVE_SCHEMA
    RESULT_OPEN_CLAIM_NAME = spot_v1.RESULT_OPEN_CLAIM_NAME

    @staticmethod
    def validate_received_directory(**kwargs: Any) -> dict[str, Any]:
        return lock_spot.validate_received_directory(**kwargs)


_RUNNER_ADAPTER = _RunnerAdapter()
_SPOT_ADAPTER = _SpotAdapter()


def _require_rearm_inputs(
    inputs: Any,
) -> lock_open.PerformanceLockRearm1Inputs:
    if not isinstance(inputs, lock_open.PerformanceLockRearm1Inputs):
        raise TypeError(
            "rearm1 scientific merge requires PerformanceLockRearm1Inputs; "
            "legacy v1 inputs are forbidden"
        )
    return inputs


def _require_exact_done_inputs(
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
) -> None:
    by_role = {
        "candidate": [Path(path).resolve() for path in candidate_done_paths],
        "reference": [Path(path).resolve() for path in reference_done_paths],
    }
    for role, paths in by_role.items():
        if (
            len(paths) != lock_plan.SHARD_COUNT_PER_ROLE
            or len(set(paths)) != lock_plan.SHARD_COUNT_PER_ROLE
        ):
            raise ValueError(
                f"performance-lock rearm1 {role} DONE inputs must be exactly "
                f"{lock_plan.SHARD_COUNT_PER_ROLE} unique claimed shards"
            )
    if set(by_role["candidate"]) & set(by_role["reference"]):
        raise ValueError(
            "performance-lock rearm1 candidate/reference DONE inputs must be isolated"
        )


def _serialize_open_inputs(inputs: Any) -> dict[str, str]:
    value = _require_rearm_inputs(inputs)
    return {
        "repository_root": str(Path(value.repository_root).resolve()),
        "plan_path": str(Path(value.plan_path).resolve()),
        "lock_output_directory": os.path.normcase(
            os.path.abspath(os.fspath(value.lock_output_directory))
        ),
        "candidate_library": str(Path(value.candidate_library).resolve()),
        "reference_library": str(Path(value.reference_library).resolve()),
        "feature_encoder": str(Path(value.feature_encoder).resolve()),
        "startup_source": str(Path(value.startup_source).resolve()),
        "incident_receipt_path": str(Path(value.incident_receipt_path).resolve()),
        "old_v1_root_directory": os.path.normcase(
            os.path.abspath(os.fspath(value.old_v1_root_directory))
        ),
        "development_summary_path": str(
            Path(value.development_summary_path).resolve()
        ),
        "development_validation_path": str(
            Path(value.development_validation_path).resolve()
        ),
        "development_root_directory": str(
            Path(value.development_root_directory).resolve()
        ),
        "old_v1_global_claim_path": os.path.normcase(
            os.path.abspath(os.fspath(value.old_v1_global_claim_path))
        ),
        "global_claim_path": os.path.normcase(
            os.path.abspath(os.fspath(value.global_claim_path))
        ),
    }


def _deserialize_open_inputs(raw: Any) -> lock_open.PerformanceLockRearm1Inputs:
    value = v1._mapping(raw, "performance-lock rearm1 open inputs")
    if set(value) != _OPEN_INPUT_KEYS or any(
        not isinstance(value[key], str) or not value[key] for key in _OPEN_INPUT_KEYS
    ):
        raise ValueError("performance-lock rearm1 open input identity changed")
    return lock_open.PerformanceLockRearm1Inputs(
        repository_root=Path(value["repository_root"]),
        plan_path=Path(value["plan_path"]),
        lock_output_directory=Path(value["lock_output_directory"]),
        candidate_library=Path(value["candidate_library"]),
        reference_library=Path(value["reference_library"]),
        feature_encoder=Path(value["feature_encoder"]),
        startup_source=Path(value["startup_source"]),
        incident_receipt_path=Path(value["incident_receipt_path"]),
        old_v1_root_directory=Path(value["old_v1_root_directory"]),
        development_summary_path=Path(value["development_summary_path"]),
        development_validation_path=Path(value["development_validation_path"]),
        development_root_directory=Path(value["development_root_directory"]),
        old_v1_global_claim_path=Path(value["old_v1_global_claim_path"]),
        global_claim_path=Path(value["global_claim_path"]),
    )


def _validate_claim_plan_lineage(
    *,
    inputs: Any,
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> None:
    value = _require_rearm_inputs(inputs)
    plan_path = Path(value.plan_path).resolve()
    precontent = v1._mapping(
        claim.get("precontent_plan"), "rearm1 open-claim precontent plan"
    )
    accepted = v1._mapping(
        claim.get("accepted_binaries"), "rearm1 open-claim accepted binaries"
    )
    source_identity = v1._mapping(
        plan.get("source_identity"), "rearm1 plan source identity"
    )
    old = v1._mapping(
        claim.get("old_v1_lock_evidence"), "rearm1 old-v1 lock evidence"
    )
    old_claim = v1._mapping(old.get("global_claim"), "rearm1 old-v1 global claim")
    old_seal = v1._mapping(old.get("seal"), "rearm1 old-v1 seal")
    guards = v1._mapping(claim.get("rearm_guards"), "rearm1 claim guards")
    restrictions = v1._mapping(
        claim.get("restrictions"), "rearm1 claim restrictions"
    )
    root_contract = v1._mapping(
        plan.get("root_contract"), "rearm1 plan root contract"
    )
    accepted_hashes = {
        role: v1._mapping(accepted.get(role), f"rearm1 accepted {role} binary").get(
            "sha256"
        )
        for role in ("candidate", "reference", "feature_encoder")
    }
    expected_hashes = {
        "candidate": source_identity.get("candidate_library_sha256"),
        "reference": source_identity.get("reference_library_sha256"),
        "feature_encoder": source_identity.get("feature_encoder_sha256"),
    }
    if (
        plan.get("schema") != lock_plan.PLAN_SCHEMA
        or plan.get("scope") != lock_plan.PLAN_SCOPE
        or plan.get("candidate_variant")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
        or plan.get("run_contract_digest") != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or claim.get("schema") != lock_open.CLAIM_SCHEMA
        or claim.get("status") != lock_open.CLAIM_STATUS
        or Path(str(precontent.get("path"))).resolve() != plan_path
        or precontent.get("sha256") != lock_open.sha256_file(plan_path)
        or precontent.get("canonical_sha256") != lock_plan.canonical_sha256(plan)
        or precontent.get("schema") != lock_plan.PLAN_SCHEMA
        or claim.get("global_claim_path")
        != lock_open._lexical_absolute(value.global_claim_path)
        or claim.get("lock_output_directory")
        != lock_open._lexical_absolute(value.lock_output_directory)
        or claim.get("lock_output_directory")
        == lock_open._lexical_absolute(value.old_v1_root_directory)
        or claim.get("lock_run_contract") != plan.get("run_contract")
        or claim.get("lock_run_contract_digest")
        != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or claim.get("image") != plan.get("image")
        or claim.get("allocation") != plan.get("allocation")
        or claim.get("seed_contract") != root_contract.get("seed_contract")
        or accepted_hashes != expected_hashes
        or v1._mapping(
            claim.get("ai_profiles_current"), "rearm1 current registry"
        ).get("sha256")
        != source_identity.get("current_profile_registry_sha256")
        or old_claim.get("sha256") != lock_open.OLD_V1_GLOBAL_CLAIM_SHA256
        or old_seal.get("sha256") != lock_open.OLD_V1_SEAL_SHA256
        or old.get("attempt1_used") is not False
        or old.get("root_reuse_authorized") is not False
        or guards.get("new_global_claim") is not True
        or guards.get("fresh_700_series_seed_schedule") is not True
        or guards.get("old_v1_attempt1_used") is not False
        or guards.get("old_v1_attempt1_reused") is not False
        or guards.get("old_v1_root_reused") is not False
        or guards.get("old_v1_package_reused") is not False
        or restrictions.get("old_v1_attempt1_allowed") is not False
        or restrictions.get("old_v1_root_reuse_allowed") is not False
        or restrictions.get("alternate_seed_allowed") is not False
        or restrictions.get("reseed_allowed") is not False
        or restrictions.get("post_claim_reseed_allowed") is not False
        or restrictions.get("cloud_authorized") is not False
        or restrictions.get("training_authorized") is not False
        or restrictions.get("quality_authorized") is not False
        or restrictions.get("promotion_authorized") is not False
        or restrictions.get("opponent_private_discards_allowed") is not False
    ):
        raise ValueError("performance-lock rearm1 claim/plan lineage changed")


def _validate_materialization_and_seal(
    *,
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
    materialization: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> None:
    root_hashes = v1._array(
        seal.get("root_artifact_sha256"), "rearm1 root-seal artifact hashes"
    )
    development = v1._mapping(
        seal.get("development_comparison"), "rearm1 development comparison"
    )
    old = v1._mapping(
        seal.get("old_performance_lock_comparison"),
        "rearm1 old performance-lock comparison",
    )
    guards = v1._mapping(seal.get("rearm_guards"), "rearm1 seal guards")
    if (
        set(materialization) != lock_open._MATERIALIZATION_KEYS
        or materialization.get("schema") != lock_open.MATERIALIZATION_SCHEMA
        or materialization.get("status") != lock_open.MATERIALIZATION_STATUS
        or materialization.get("global_claim_sha256")
        != lock_open.canonical_sha256(claim)
        or materialization.get("plan_sha256")
        != v1._mapping(claim.get("precontent_plan"), "rearm1 claim plan").get(
            "sha256"
        )
        or materialization.get("run_contract_digest")
        != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or materialization.get("hand_indices")
        != list(runner.CONTRACT_HAND_INDICES)
        or materialization.get("root_count") != 100
        or materialization.get("same_identity_resume_only") is not True
        or materialization.get("fresh_recovery_seed_schedule") is not True
        or materialization.get("old_v1_attempt1_reused") is not False
        or materialization.get("old_v1_root_reused") is not False
        or materialization.get("reseeded") is not False
        or materialization.get("training_eligible") is not False
        or materialization.get("current_profile_changed") is not False
        or set(seal) != lock_open._SEAL_KEYS
        or seal.get("schema") != lock_open.SEAL_SCHEMA
        or seal.get("status") != lock_open.SEAL_STATUS
        or seal.get("global_claim_sha256") != lock_open.canonical_sha256(claim)
        or seal.get("materialization_sha256")
        != lock_open.canonical_sha256(materialization)
        or seal.get("plan_sha256") != materialization.get("plan_sha256")
        or seal.get("run_contract_digest") != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or seal.get("hand_indices") != list(runner.CONTRACT_HAND_INDICES)
        or seal.get("root_count") != 100
        or seal.get("observation_count") != 200
        or seal.get("profile_counts")
        != {profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES}
        or seal.get("seat_counts") != {"first": 100, "second": 100}
        or root_hashes != materialization.get("root_artifact_sha256")
        or len(root_hashes) != 100
        or len(set(root_hashes)) != 100
        or any(not v1._is_sha256(value) for value in root_hashes)
        or seal.get("aggregate_root_sha256")
        != lock_open.canonical_sha256(root_hashes)
        or seal.get("aggregate_root_sha256")
        != materialization.get("aggregate_root_sha256")
        or seal.get("root_artifact_unique") is not True
        or seal.get("observation_fingerprint_unique") is not True
        or development.get("development_root_count") != 100
        or any(
            development.get(field) != 0
            for field in (
                "lock_fingerprint_overlap_count",
                "lock_root_hash_overlap_count",
                "lock_seed_overlap_count",
            )
        )
        or old.get("old_v1_global_claim_sha256")
        != lock_open.OLD_V1_GLOBAL_CLAIM_SHA256
        or old.get("old_v1_seal_sha256") != lock_open.OLD_V1_SEAL_SHA256
        or old.get("old_v1_root_count") != 100
        or any(
            old.get(field) != 0
            for field in (
                "rearm1_fingerprint_overlap_count",
                "rearm1_root_hash_overlap_count",
                "rearm1_seed_overlap_count",
            )
        )
        or old.get("old_v1_attempt1_reused") is not False
        or old.get("old_v1_root_reused") is not False
        or guards.get("incident_receipt_sha256")
        != lock_open.STARTUP_FAILURE_RECEIPT_SHA256
        or guards.get("fresh_700_series_seed_schedule") is not True
        or guards.get("same_identity_resume_only") is not True
        or guards.get("old_v1_attempt1_reused") is not False
        or guards.get("old_v1_root_reused") is not False
        or guards.get("post_claim_reseeded") is not False
        or v1._mapping(seal.get("visibility"), "rearm1 visibility")
        != {
            "runner_validator_replayed_all_roots": True,
            "first_observation_count": 100,
            "second_observation_count": 100,
            "opponent_private_discards_used": False,
            "current_profile_resolved": False,
        }
        or v1._mapping(seal.get("selection_inputs"), "rearm1 selection inputs")
        != {
            "timing_used": False,
            "q_used": False,
            "ev_used": False,
            "all_100_preregistered_hands_used": True,
        }
        or any(
            seal.get(field) is not False
            for field in (
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError(
            "performance-lock rearm1 materialization/root-seal lineage changed"
        )
    for field in (
        "root_topology_sha256",
        "observation_fingerprint_sha256",
    ):
        v1._require_sha256(seal.get(field), f"rearm1 root-seal {field}")
    v1._require_sha256(
        development.get("development_all100_root_sha256"),
        "rearm1 sealed development root SHA-256",
    )


@contextmanager
def _rearm1_merge_context() -> Iterator[None]:
    """Temporarily bind the v1 scientific merger to the rearm1 producers."""

    replacements = {
        "lock_plan": lock_plan,
        "lock_open": lock_open,
        "lock_spot": _SPOT_ADAPTER,
        "runner": _RUNNER_ADAPTER,
        "MERGE_SCHEMA": MERGE_SCHEMA,
        "VALIDATION_SCHEMA": VALIDATION_SCHEMA,
        "SCOPE": SCOPE,
        "PASS_DECISION": PASS_DECISION,
        "NO_GO_DECISION": NO_GO_DECISION,
        "_serialize_open_inputs": _serialize_open_inputs,
        "_deserialize_open_inputs": _deserialize_open_inputs,
        "_validate_claim_plan_lineage": _validate_claim_plan_lineage,
        "_validate_materialization_and_seal": _validate_materialization_and_seal,
    }
    with _CONTEXT_LOCK:
        previous = {name: getattr(v1, name) for name in replacements}
        for name, value in replacements.items():
            setattr(v1, name, value)
        try:
            yield
        finally:
            for name, value in previous.items():
                setattr(v1, name, value)


def merge_candidate02_performance_lock_rearm1(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    lock_inputs: lock_open.PerformanceLockRearm1Inputs,
    receive_inputs: PerformanceLockReceiveInputs,
) -> dict[str, Any]:
    """Replay and merge only the fresh rearm1 one-shot receive identity."""

    _require_rearm_inputs(lock_inputs)
    _require_exact_done_inputs(candidate_done_paths, reference_done_paths)
    with _rearm1_merge_context():
        return v1.merge_candidate02_performance_lock(
            candidate_done_paths=candidate_done_paths,
            reference_done_paths=reference_done_paths,
            lock_inputs=lock_inputs,
            receive_inputs=receive_inputs,
        )


def validate_candidate02_performance_lock_rearm1_merge(
    *, summary_path: Path, output_path: Path | None = None
) -> dict[str, Any]:
    """Independently replay a stored rearm1 scientific merge."""

    with _rearm1_merge_context():
        return v1.validate_candidate02_performance_lock_merge(
            summary_path=summary_path,
            output_path=output_path,
        )


def merge_and_validate_candidate02_performance_lock_rearm1(
    *,
    candidate_done_paths: Sequence[Path],
    reference_done_paths: Sequence[Path],
    lock_inputs: lock_open.PerformanceLockRearm1Inputs,
    receive_inputs: PerformanceLockReceiveInputs,
    summary_output_path: Path,
    validation_output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write once and independently replay the rearm1 scientific merge."""

    _require_rearm_inputs(lock_inputs)
    _require_exact_done_inputs(candidate_done_paths, reference_done_paths)
    with _rearm1_merge_context():
        return v1.merge_and_validate_candidate02_performance_lock(
            candidate_done_paths=candidate_done_paths,
            reference_done_paths=reference_done_paths,
            lock_inputs=lock_inputs,
            receive_inputs=receive_inputs,
            summary_output_path=summary_output_path,
            validation_output_path=validation_output_path,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-done", type=Path, action="append", required=True)
    parser.add_argument("--reference-done", type=Path, action="append", required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--receive-dir", type=Path, required=True)
    parser.add_argument("--project", default=lock_spot.DEFAULT_PROJECT)
    parser.add_argument("--bucket", default=lock_spot.DEFAULT_BUCKET)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=lock_open.DEFAULT_REPOSITORY_ROOT,
    )
    parser.add_argument(
        "--plan", type=Path, default=lock_open.DEFAULT_PRECONTENT_PLAN_PATH
    )
    parser.add_argument("--lock-output", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--reference-library", type=Path, required=True)
    parser.add_argument("--feature-encoder", type=Path, required=True)
    parser.add_argument("--startup-source", type=Path, required=True)
    parser.add_argument("--incident-receipt", type=Path, required=True)
    parser.add_argument("--old-v1-roots", type=Path, required=True)
    parser.add_argument(
        "--development-summary",
        type=Path,
        default=lock_open.DEFAULT_DEVELOPMENT_MERGE_DIR / "summary.json",
    )
    parser.add_argument(
        "--development-validation",
        type=Path,
        default=lock_open.DEFAULT_DEVELOPMENT_MERGE_DIR / "validation.json",
    )
    parser.add_argument(
        "--development-roots",
        type=Path,
        default=lock_open.DEFAULT_DEVELOPMENT_ROOT_DIR,
    )
    parser.add_argument(
        "--old-v1-global-claim",
        type=Path,
        default=lock_open.DEFAULT_OLD_V1_GLOBAL_CLAIM_PATH,
    )
    parser.add_argument(
        "--global-claim",
        type=Path,
        default=lock_open.DEFAULT_GLOBAL_CLAIM_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = lock_open.PerformanceLockRearm1Inputs(
        repository_root=args.repository_root,
        plan_path=args.plan,
        lock_output_directory=args.lock_output,
        candidate_library=args.candidate_library,
        reference_library=args.reference_library,
        feature_encoder=args.feature_encoder,
        startup_source=args.startup_source,
        incident_receipt_path=args.incident_receipt,
        old_v1_root_directory=args.old_v1_roots,
        development_summary_path=args.development_summary,
        development_validation_path=args.development_validation,
        development_root_directory=args.development_roots,
        old_v1_global_claim_path=args.old_v1_global_claim,
        global_claim_path=args.global_claim,
    )
    receive_inputs = PerformanceLockReceiveInputs(
        run_dir=args.run_dir,
        receive_dir=args.receive_dir,
        project=args.project,
        bucket=args.bucket,
    )
    summary, _validation = (
        merge_and_validate_candidate02_performance_lock_rearm1(
            candidate_done_paths=args.candidate_done,
            reference_done_paths=args.reference_done,
            lock_inputs=inputs,
            receive_inputs=receive_inputs,
            summary_output_path=args.summary_output,
            validation_output_path=args.validation_output,
        )
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MERGE_SCHEMA",
    "NO_GO_DECISION",
    "PASS_DECISION",
    "PerformanceLockReceiveInputs",
    "SCOPE",
    "VALIDATION_SCHEMA",
    "main",
    "merge_and_validate_candidate02_performance_lock_rearm1",
    "merge_candidate02_performance_lock_rearm1",
    "validate_candidate02_performance_lock_rearm1_merge",
]
