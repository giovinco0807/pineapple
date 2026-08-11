"""Bind the immutable Stage3 encoder probe to the rearm2 lock chain.

The original probe generator remains byte-for-byte unchanged.  Probe creation
and cross-platform comparison are delegated to it; only the lock-chain
validator is replaced for one serialized call.  The resulting receipt remains
compatible with the audited v1 package validator while carrying an explicit
rearm2 attestation and zero-reuse proof for both v1 and rearm1.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (  # noqa: E402
    verify_hu_m31_t3_feature_encoder_platform_parity as probe_v1,
)

from . import hu_m31_t3_step6d_performance_lock_rearm2_open as rearm_open


VALIDATOR_SCHEMA = "hu_m31_t3_feature_encoder_rearm2_chain_validator_v1"
VALIDATOR_STATUS = (
    "verified_rearm2_chain_and_zero_v1_rearm1_reuse_before_packaging"
)
_VALIDATOR_LOCK = threading.RLock()


def _require_false(
    value: Mapping[str, Any],
    fields: Sequence[str],
    *,
    label: str,
) -> None:
    if any(value.get(field) is not False for field in fields):
        raise ValueError(f"{label} changed")


def _validate_rearm2_chain(
    claim: Mapping[str, Any],
    materialization: Mapping[str, Any],
    seal: Mapping[str, Any],
    *,
    linux_probe: Mapping[str, Any],
    ai_profiles_path: str | Path,
) -> dict[str, Any]:
    if (
        set(claim) != rearm_open._CLAIM_KEYS
        or claim.get("schema") != rearm_open.CLAIM_SCHEMA
        or claim.get("status") != rearm_open.CLAIM_STATUS
        or claim.get("scope")
        != "candidate02_performance_lock_rearm2_fresh_roots_only"
    ):
        raise ValueError("rearm2 global claim contract changed")
    if (
        set(materialization) != rearm_open._MATERIALIZATION_KEYS
        or materialization.get("schema") != rearm_open.MATERIALIZATION_SCHEMA
        or materialization.get("status") != rearm_open.MATERIALIZATION_STATUS
    ):
        raise ValueError("rearm2 materialization contract changed")
    if (
        set(seal) != rearm_open._SEAL_KEYS
        or seal.get("schema") != rearm_open.SEAL_SCHEMA
        or seal.get("status") != rearm_open.SEAL_STATUS
    ):
        raise ValueError("rearm2 seal contract changed")

    claim_sha = probe_v1.canonical_sha256(claim)
    materialization_sha = probe_v1.canonical_sha256(materialization)
    seal_sha = probe_v1.canonical_sha256(seal)
    if (
        materialization.get("global_claim_sha256") != claim_sha
        or seal.get("global_claim_sha256") != claim_sha
        or seal.get("materialization_sha256") != materialization_sha
    ):
        raise ValueError("rearm2 claim/materialization/seal hash chain changed")

    accepted = claim.get("accepted_binaries")
    if not isinstance(accepted, dict) or set(accepted) != {
        "candidate",
        "reference",
        "feature_encoder",
    }:
        raise ValueError("rearm2 accepted binary set changed")
    accepted_feature = probe_v1._validate_file_record_shape(
        accepted.get("feature_encoder"),
        "rearm2 accepted Linux feature encoder",
        expected_suffix=".so",
    )
    probe_v1._same_file_content(
        accepted_feature,
        linux_probe["library"],
        "rearm2 accepted Linux feature encoder",
    )
    if accepted_feature["sha256"] != rearm_open.FEATURE_ENCODER_SHA256:
        raise ValueError("rearm2 accepted Linux feature encoder hash changed")

    restrictions = claim.get("restrictions")
    required_false = (
        "timing_used_for_root_selection",
        "q_used_for_root_selection",
        "ev_used_for_root_selection",
        "old_v1_root_reuse_allowed",
        "rearm1_attempt1_allowed",
        "rearm1_package_reuse_allowed",
        "rearm1_root_reuse_allowed",
        "rearm1_seed_reuse_allowed",
        "rearm1_claim_reuse_allowed",
        "alternate_seed_allowed",
        "reseed_allowed",
        "post_claim_reseed_allowed",
        "authorization_before_actual_package_smoke_allowed",
        "actual_package_smoke_receipt_sha_bound_at_open",
        "cloud_authorized",
        "training_authorized",
        "quality_authorized",
        "promotion_authorized",
        "current_profile_resolution_allowed",
        "runtime_activation_allowed",
        "opponent_private_discards_allowed",
    )
    if not isinstance(restrictions, Mapping):
        raise ValueError("rearm2 restrictions are missing")
    _require_false(restrictions, required_false, label="rearm2 restrictions")
    if (
        restrictions.get(
            "authorization_requires_exhaustive_actual_package_smoke"
        )
        is not True
        or restrictions.get(
            "actual_package_smoke_receipt_sha_must_bind_spot_claim"
        )
        is not True
        or restrictions.get(
            "actual_package_smoke_receipt_sha_must_bind_authorization"
        )
        is not True
    ):
        raise ValueError("rearm2 smoke authorization restrictions changed")

    claim_guards = claim.get("rearm_guards")
    reuse_fields = (
        "old_v1_root_reused",
        "rearm1_attempt1_reused",
        "rearm1_package_reused",
        "rearm1_root_reused",
        "rearm1_seed_reused",
        "rearm1_claim_reused",
    )
    if (
        not isinstance(claim_guards, Mapping)
        or claim_guards.get("new_global_claim") is not True
        or claim_guards.get("claim_before_new_root_path_touch") is not True
        or claim_guards.get("crash_consumes_claim") is not True
        or claim_guards.get("deterministic_exact_identity_resume_only")
        is not True
        or claim_guards.get("fresh_710_series_seed_schedule") is not True
    ):
        raise ValueError("rearm2 claim guards changed")
    _require_false(claim_guards, reuse_fields, label="rearm2 claim reuse guards")

    seed_contract = claim.get("seed_contract")
    if (
        not isinstance(seed_contract, Mapping)
        or seed_contract.get("seed_set_sha256")
        != rearm_open.runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SEED_SET_SHA256
        or seed_contract.get("seed_count") != 600
        or seed_contract.get("seed_min") != 710_108_071_901
        or seed_contract.get("seed_max") != 715_207_072_198
        or seed_contract.get("performance_lock_v1_overlap_count") != 0
        or seed_contract.get("performance_lock_rearm1_overlap_count") != 0
        or seed_contract.get("existing_step6d_union_overlap_count") != 0
        or seed_contract.get("all_values_unique") is not True
    ):
        raise ValueError("rearm2 fresh seed proof changed")

    hand_indices = list(range(100))
    root_hashes = materialization.get("root_artifact_sha256")
    if (
        materialization.get("root_count") != 100
        or materialization.get("hand_indices") != hand_indices
        or not isinstance(root_hashes, list)
        or len(root_hashes) != 100
        or len(set(root_hashes)) != 100
        or any(
            probe_v1._validate_hash(value, "rearm2 root hash") != value
            for value in root_hashes
        )
        or materialization.get("aggregate_root_sha256")
        != probe_v1.canonical_sha256(root_hashes)
        or materialization.get("same_identity_resume_only") is not True
        or materialization.get("fresh_recovery_v3_seed_schedule") is not True
        or materialization.get("reseeded") is not False
        or materialization.get("training_eligible") is not False
        or materialization.get("current_profile_changed") is not False
    ):
        raise ValueError("rearm2 materialization invariants changed")
    _require_false(
        materialization,
        reuse_fields,
        label="rearm2 materialization reuse guards",
    )

    matching_fields = (
        "plan_sha256",
        "run_contract_digest",
        "hand_indices",
        "root_count",
        "root_artifact_sha256",
        "aggregate_root_sha256",
    )
    if any(seal.get(field) != materialization.get(field) for field in matching_fields):
        raise ValueError("rearm2 materialization/seal root identity changed")
    if (
        seal.get("observation_count") != 200
        or seal.get("seat_counts") != {"first": 100, "second": 100}
        or seal.get("root_artifact_unique") is not True
        or seal.get("observation_fingerprint_unique") is not True
        or seal.get("training_eligible") is not False
        or seal.get("quality_evidence") is not False
        or seal.get("promotion_evidence") is not False
        or seal.get("current_profile_changed") is not False
        or seal.get("named_profile_added") is not False
        or seal.get("runtime_policy_activated") is not False
    ):
        raise ValueError("rearm2 seal invariants changed")

    visibility = seal.get("visibility")
    selection = seal.get("selection_inputs")
    if (
        not isinstance(visibility, Mapping)
        or visibility.get("first_observation_count") != 100
        or visibility.get("second_observation_count") != 100
        or visibility.get("opponent_private_discards_used") is not False
        or visibility.get("current_profile_resolved") is not False
        or not isinstance(selection, Mapping)
        or selection.get("all_100_preregistered_hands_used") is not True
        or selection.get("timing_used") is not False
        or selection.get("q_used") is not False
        or selection.get("ev_used") is not False
    ):
        raise ValueError("rearm2 visibility/selection invariants changed")

    development = seal.get("development_comparison")
    prior = seal.get("prior_lock_comparison")
    seal_guards = seal.get("rearm_guards")
    overlap_fields = tuple(
        f"{name}_{kind}_overlap_count"
        for name in ("old_v1", "rearm1")
        for kind in ("root_hash", "fingerprint", "seed")
    )
    if (
        not isinstance(development, Mapping)
        or any(
            development.get(field) != 0
            for field in (
                "lock_fingerprint_overlap_count",
                "lock_root_hash_overlap_count",
                "lock_seed_overlap_count",
            )
        )
        or not isinstance(prior, Mapping)
        or any(prior.get(field) != 0 for field in overlap_fields)
        or prior.get("old_v1_global_claim_sha256")
        != rearm_open.OLD_V1_GLOBAL_CLAIM_SHA256
        or prior.get("old_v1_seal_sha256") != rearm_open.OLD_V1_SEAL_SHA256
        or prior.get("rearm1_global_claim_sha256")
        != rearm_open.REARM1_GLOBAL_CLAIM_SHA256
        or prior.get("rearm1_seal_sha256") != rearm_open.REARM1_SEAL_SHA256
        or not isinstance(seal_guards, Mapping)
        or seal_guards.get("incident_receipt_sha256")
        != rearm_open.REARM1_STARTUP_FAILURE_RECEIPT_SHA256
        or seal_guards.get("fresh_710_series_seed_schedule") is not True
        or seal_guards.get("same_identity_resume_only") is not True
        or seal_guards.get("post_claim_reseeded") is not False
    ):
        raise ValueError("rearm2 v1/rearm1 no-reuse proof changed")
    _require_false(
        prior,
        reuse_fields[1:],
        label="rearm2 prior comparison reuse guards",
    )
    _require_false(
        seal_guards,
        reuse_fields,
        label="rearm2 seal reuse guards",
    )

    smoke = seal.get("actual_package_smoke_requirement")
    if (
        smoke != claim.get("actual_package_smoke_requirement")
        or smoke != rearm_open._expected_smoke_requirement()
        or smoke.get("verified_job_count") != 20
        or smoke.get("startup_invocation_count") != 20
        or smoke.get("root_read") is not False
        or smoke.get("cloud_mutated") is not False
        or smoke.get("global_spot_claim_and_authorization_bind_receipt_sha256")
        is not True
    ):
        raise ValueError("rearm2 actual-package smoke contract changed")

    ai_profiles = probe_v1._validate_ai_profiles(ai_profiles_path, claim)
    validator = probe_v1._file_record(Path(__file__), "rearm2 chain validator")
    return {
        "contract": "performance_lock_rearm2_v1",
        "validator_schema": VALIDATOR_SCHEMA,
        "validator_status": VALIDATOR_STATUS,
        "validator_bytes": validator["bytes"],
        "validator_sha256": validator["sha256"],
        "global_claim_sha256": claim_sha,
        "materialization_sha256": materialization_sha,
        "seal_sha256": seal_sha,
        "accepted_linux_feature_encoder_sha256": accepted_feature["sha256"],
        "ai_profiles_sha256": ai_profiles["sha256"],
        "root_count": 100,
        "reseeded": False,
        "current_profile_changed": False,
        "old_v1_global_claim_sha256": rearm_open.OLD_V1_GLOBAL_CLAIM_SHA256,
        "old_v1_seal_sha256": rearm_open.OLD_V1_SEAL_SHA256,
        "rearm1_global_claim_sha256": rearm_open.REARM1_GLOBAL_CLAIM_SHA256,
        "rearm1_seal_sha256": rearm_open.REARM1_SEAL_SHA256,
        "rearm1_startup_failure_receipt_sha256": (
            rearm_open.REARM1_STARTUP_FAILURE_RECEIPT_SHA256
        ),
        "v1_reused": False,
        "rearm1_reused": False,
        "fresh_recovery_v3_seed_schedule": True,
        "exhaustive_actual_package_smoke_required": True,
    }


@contextlib.contextmanager
def _rearm2_chain_validator() -> Any:
    """Install the rearm2 validator for one comparison and always restore v1."""

    with _VALIDATOR_LOCK:
        original = probe_v1._validate_lock_chain
        probe_v1._validate_lock_chain = _validate_rearm2_chain
        try:
            yield
        finally:
            probe_v1._validate_lock_chain = original


def compare_probes(
    *,
    windows_probe_path: str | Path,
    linux_probe_path: str | Path,
    global_claim_path: str | Path,
    materialization_path: str | Path,
    seal_path: str | Path,
    rust_source_path: str | Path,
    cargo_lock_path: str | Path,
    ai_profiles_path: str | Path,
) -> dict[str, Any]:
    with _rearm2_chain_validator():
        return probe_v1.compare_probes(
            windows_probe_path=windows_probe_path,
            linux_probe_path=linux_probe_path,
            global_claim_path=global_claim_path,
            materialization_path=materialization_path,
            seal_path=seal_path,
            rust_source_path=rust_source_path,
            cargo_lock_path=cargo_lock_path,
            ai_profiles_path=ai_profiles_path,
        )


def write_comparison_receipt(
    *,
    windows_probe_path: str | Path,
    linux_probe_path: str | Path,
    global_claim_path: str | Path,
    materialization_path: str | Path,
    seal_path: str | Path,
    rust_source_path: str | Path,
    cargo_lock_path: str | Path,
    ai_profiles_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    if Path(output_path).name != probe_v1.PARITY_RECEIPT_NAME:
        raise ValueError(
            f"parity receipt must be named {probe_v1.PARITY_RECEIPT_NAME}"
        )
    with _rearm2_chain_validator():
        return probe_v1.write_comparison_receipt(
            windows_probe_path=windows_probe_path,
            linux_probe_path=linux_probe_path,
            global_claim_path=global_claim_path,
            materialization_path=materialization_path,
            seal_path=seal_path,
            rust_source_path=rust_source_path,
            cargo_lock_path=cargo_lock_path,
            ai_profiles_path=ai_profiles_path,
            output_path=output_path,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--library", type=Path, required=True)
    probe.add_argument(
        "--platform-label",
        choices=("windows", "linux"),
        required=True,
    )
    probe.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--windows-probe", type=Path, required=True)
    compare.add_argument("--linux-probe", type=Path, required=True)
    compare.add_argument("--global-claim", type=Path, required=True)
    compare.add_argument("--materialization", type=Path, required=True)
    compare.add_argument("--seal", type=Path, required=True)
    compare.add_argument("--rust-source", type=Path, required=True)
    compare.add_argument("--cargo-lock", type=Path, required=True)
    compare.add_argument("--ai-profiles", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "probe":
        value = probe_v1.write_probe(
            args.library,
            args.platform_label,
            args.output,
        )
    else:
        value = write_comparison_receipt(
            windows_probe_path=args.windows_probe,
            linux_probe_path=args.linux_probe,
            global_claim_path=args.global_claim,
            materialization_path=args.materialization,
            seal_path=args.seal,
            rust_source_path=args.rust_source,
            cargo_lock_path=args.cargo_lock,
            ai_profiles_path=args.ai_profiles,
            output_path=args.output,
        )
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "VALIDATOR_SCHEMA",
    "VALIDATOR_STATUS",
    "compare_probes",
    "main",
    "write_comparison_receipt",
]
