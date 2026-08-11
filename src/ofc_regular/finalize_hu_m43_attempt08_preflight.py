"""Finalize Attempt08 preflight and authorize exactly development roots 0..199."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .aggregate_hu_m43_attempt08_preflight import (
    ATTEMPT08_PREFLIGHT_AGGREGATE_SCHEMA,
    load_and_validate_preflight_aggregate,
    validate_preflight_aggregate_with_proofs,
)
from .hu_m43_attempt08_contract import (
    M43_ATTEMPT08_PLAN_SCHEMA,
    M43_ATTEMPT08_PLAN_SHA256,
    M43_ATTEMPT08_PROFILES,
    load_and_validate_attempt08_plan,
)
from .run_hu_m43_attempt07_preflight import _sha256_value, canonical_json_bytes
from .run_hu_m43_attempt08_preflight import (
    ATTEMPT08_PREFLIGHT_PLAN_SCHEMA,
    ATTEMPT08_PREFLIGHT_PLAN_SHA256,
    ATTEMPT08_PREFLIGHT_SLOTS,
    DEFAULT_ATTEMPT08_PLAN_PATH,
    DEFAULT_PREFLIGHT_PLAN_PATH,
    DEFAULT_SOURCE_PATH,
    load_preflight_plan,
)
from .hu_m43_attempt08_runtime_anchor import ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256
from .hu_m43_attempt08_runtime_anchor_contract import (
    ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
)
from .hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT08_GCP_IMAGE_ID,
    ATTEMPT08_GCP_IMAGE_NAME,
)


ATTEMPT08_PREFLIGHT_FINALIZATION_SCHEMA = (
    "hu_m43_attempt08_preflight_finalization_v1"
)
ATTEMPT08_DEVELOPMENT_OPEN_AUTHORIZATION_SCHEMA = (
    "hu_m43_attempt08_development_open_authorization_v1"
)
ATTEMPT08_PREFLIGHT_GO_STATUS = (
    "pass_correctness_preflight_and_authorize_development200_only"
)
ATTEMPT08_PREFLIGHT_NO_GO_STATUS = (
    "complete_preflight_no_go_development_not_authorized"
)
ATTEMPT08_PREFLIGHT_EXECUTION_EVIDENCE_SCHEMA = (
    "hu_m43_attempt08_preflight_execution_evidence_v1"
)

_AUTH_TOP_KEYS = {
    "schema",
    "status",
    "milestone",
    "target_plan",
    "preflight_plan",
    "preflight_result",
    "evidence",
    "development_population",
    "authorization",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Attempt08 finalizer {label} must be a mapping")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_canonical(path: str | Path, *, label: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise ValueError(f"Attempt08 {label} is not canonical JSON")
    return payload


def validate_preflight_execution_evidence(payload: Mapping[str, Any]) -> None:
    """Validate the value-free Spot execution attestation bound to a Go."""

    expected_slots = set(ATTEMPT08_PREFLIGHT_SLOTS)
    if set(payload) != {
        "schema", "status", "run_name", "manifest_sha256",
        "launch_authorization_sha256", "local_evidence_sha256", "done_sha256",
        "support_sha256", "runtime", "all_five_done_before_payloads_opened",
        "proof_payloads_opened_after_all_done", "current_profile_mutated",
        "runtime_policy_activated",
    }:
        raise ValueError("Attempt08 preflight execution evidence fields changed")
    done = _mapping(payload.get("done_sha256"), "execution DONE hashes")
    support = _mapping(payload.get("support_sha256"), "execution support hashes")
    runtime = _mapping(payload.get("runtime"), "execution runtime")
    if (
        payload.get("schema") != ATTEMPT08_PREFLIGHT_EXECUTION_EVIDENCE_SCHEMA
        or payload.get("status") != "all_five_spot_jobs_received_and_validated"
        or not isinstance(payload.get("run_name"), str)
        or not payload["run_name"]
        or any(
            not _is_sha256(payload.get(key))
            for key in (
                "manifest_sha256", "launch_authorization_sha256",
                "local_evidence_sha256",
            )
        )
        or set(done) != expected_slots
        or not all(_is_sha256(value) for value in done.values())
        or set(support) != expected_slots
        or runtime
        != {
            "runtime_semantic_anchor_sha256": (
                ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
            ),
            "runtime_source_closure_sha256": (
                ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
            ),
            "runtime_fingerprint_sha256": (
                ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
            ),
            "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
        }
        or payload.get("all_five_done_before_payloads_opened") is not True
        or payload.get("proof_payloads_opened_after_all_done") is not True
        or payload.get("current_profile_mutated") is not False
        or payload.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt08 preflight execution evidence boundary changed")
    expected_support_keys = {
        "proof", "checkpoint", "heartbeat", "summary", "run_log", "boot_image"
    }
    for slot, row in support.items():
        item = _mapping(row, f"execution support {slot}")
        if set(item) != expected_support_keys or not all(
            _is_sha256(value) for value in item.values()
        ):
            raise ValueError("Attempt08 preflight execution support hashes changed")


def _authorization_payload(
    *, aggregate: Mapping[str, Any], aggregate_sha256: str
) -> dict[str, Any]:
    proof_hashes = dict(_mapping(aggregate.get("proof_file_sha256"), "proof hashes"))
    proof_gates = dict(_mapping(aggregate.get("proof_gates"), "proof gates"))
    operational_gates = dict(
        _mapping(aggregate.get("operational_gates"), "operational gates")
    )
    return {
        "schema": ATTEMPT08_DEVELOPMENT_OPEN_AUTHORIZATION_SCHEMA,
        "status": "authorized_for_attempt08_development200_only",
        "milestone": "M4.3-attempt08",
        "target_plan": {
            "path": "configs/hu_joint_policy_m43_attempt08.json",
            "schema": M43_ATTEMPT08_PLAN_SCHEMA,
            "sha256": M43_ATTEMPT08_PLAN_SHA256,
        },
        "preflight_plan": {
            "path": "configs/hu_joint_policy_m43_attempt08_preflight.json",
            "schema": ATTEMPT08_PREFLIGHT_PLAN_SCHEMA,
            "sha256": ATTEMPT08_PREFLIGHT_PLAN_SHA256,
        },
        "preflight_result": {
            "schema": ATTEMPT08_PREFLIGHT_AGGREGATE_SCHEMA,
            "sha256": aggregate_sha256,
            "decision": "go",
            "execution_evidence_sha256": aggregate[
                "spot_operational_evidence_sha256"
            ],
        },
        "evidence": {
            "proof_file_sha256": proof_hashes,
            "proof_evidence_sha256": aggregate["proof_evidence_sha256"],
            "proof_gates_sha256": _sha256_value(proof_gates),
            "operational_gates_sha256": _sha256_value(operational_gates),
            "runtime_semantic_anchor_sha256": (
                ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
            ),
            "runtime_source_closure_sha256": (
                ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
            ),
            "runtime_fingerprint_sha256": aggregate["runtime_fingerprint_sha256"],
            "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
            "preflight_execution_evidence_sha256": aggregate[
                "spot_operational_evidence_sha256"
            ],
        },
        "development_population": {
            "classification": "new_balanced_development_only",
            "roots": 200,
            "root_index_first": 0,
            "root_index_last": 199,
            "profiles": list(M43_ATTEMPT08_PROFILES),
            "roots_per_profile": 40,
            "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
        },
        "authorization": {
            "scope": "exact_development_roots_0_through_199_only",
            "development_generation_authorized": True,
            "future_audit_authorized": False,
            "fit_allowed": False,
            "threshold_selection_allowed": False,
            "runtime_activation_allowed": False,
            "current_profile_change_allowed": False,
            "full_replacement_enabled": False,
        },
    }


def validate_development_open_authorization(
    payload: Mapping[str, Any],
    *,
    attempt08_plan_sha256: str = M43_ATTEMPT08_PLAN_SHA256,
    preflight_plan_sha256: str = ATTEMPT08_PREFLIGHT_PLAN_SHA256,
) -> None:
    """Validate the immutable development-only authorization shape."""

    if set(payload) != _AUTH_TOP_KEYS:
        raise ValueError("Attempt08 development authorization fields changed")
    target = _mapping(payload.get("target_plan"), "authorization target")
    preflight = _mapping(payload.get("preflight_plan"), "authorization preflight")
    result = _mapping(payload.get("preflight_result"), "authorization result")
    evidence = _mapping(payload.get("evidence"), "authorization evidence")
    population = _mapping(
        payload.get("development_population"), "authorization population"
    )
    guards = _mapping(payload.get("authorization"), "authorization guards")
    proof_hashes = _mapping(evidence.get("proof_file_sha256"), "proof evidence")
    expected_population = {
        "classification": "new_balanced_development_only",
        "roots": 200,
        "root_index_first": 0,
        "root_index_last": 199,
        "profiles": list(M43_ATTEMPT08_PROFILES),
        "roots_per_profile": 40,
        "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
    }
    expected_guards = {
        "scope": "exact_development_roots_0_through_199_only",
        "development_generation_authorized": True,
        "future_audit_authorized": False,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
        "current_profile_change_allowed": False,
        "full_replacement_enabled": False,
    }
    if (
        payload.get("schema") != ATTEMPT08_DEVELOPMENT_OPEN_AUTHORIZATION_SCHEMA
        or payload.get("status") != "authorized_for_attempt08_development200_only"
        or payload.get("milestone") != "M4.3-attempt08"
        or target
        != {
            "path": "configs/hu_joint_policy_m43_attempt08.json",
            "schema": M43_ATTEMPT08_PLAN_SCHEMA,
            "sha256": attempt08_plan_sha256,
        }
        or preflight
        != {
            "path": "configs/hu_joint_policy_m43_attempt08_preflight.json",
            "schema": ATTEMPT08_PREFLIGHT_PLAN_SCHEMA,
            "sha256": preflight_plan_sha256,
        }
        or set(result)
        != {"schema", "sha256", "decision", "execution_evidence_sha256"}
        or result.get("schema") != ATTEMPT08_PREFLIGHT_AGGREGATE_SCHEMA
        or result.get("decision") != "go"
        or not _is_sha256(result.get("sha256"))
        or not _is_sha256(result.get("execution_evidence_sha256"))
        or set(evidence)
        != {
            "proof_file_sha256",
            "proof_evidence_sha256",
            "proof_gates_sha256",
            "operational_gates_sha256",
            "runtime_semantic_anchor_sha256",
            "runtime_source_closure_sha256",
            "runtime_fingerprint_sha256",
            "runtime_requirements_sha256",
            "gcp_image_name",
            "gcp_image_id",
            "preflight_execution_evidence_sha256",
        }
        or set(proof_hashes) != set(ATTEMPT08_PREFLIGHT_SLOTS)
        or not all(_is_sha256(value) for value in proof_hashes.values())
        or evidence.get("proof_evidence_sha256")
        != _sha256_value(dict(proof_hashes))
        or not _is_sha256(evidence.get("proof_gates_sha256"))
        or not _is_sha256(evidence.get("operational_gates_sha256"))
        or evidence.get("runtime_semantic_anchor_sha256")
        != ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
        or evidence.get("runtime_source_closure_sha256")
        != ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
        or evidence.get("runtime_fingerprint_sha256")
        != ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        or evidence.get("runtime_requirements_sha256")
        != ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256
        or evidence.get("gcp_image_name") != ATTEMPT08_GCP_IMAGE_NAME
        or evidence.get("gcp_image_id") != ATTEMPT08_GCP_IMAGE_ID
        or evidence.get("preflight_execution_evidence_sha256")
        != result.get("execution_evidence_sha256")
        or dict(population) != expected_population
        or dict(guards) != expected_guards
    ):
        raise ValueError("Attempt08 development authorization boundary changed")
    encoded = json.dumps(payload, sort_keys=True)
    for forbidden in (
        '"opaque_teacher_sha256"',
        '"semantic_parity_sha256"',
        '"selected_action_key"',
        '"raw_paired_deltas',
        '"paired_delta_vs_baseline"',
    ):
        if forbidden in encoded:
            raise ValueError("Attempt08 authorization leaked teacher details")


def load_and_validate_development_open_authorization(
    path: str | Path,
    *,
    attempt08_plan_path: str | Path = DEFAULT_ATTEMPT08_PLAN_PATH,
    preflight_plan_path: str | Path = DEFAULT_PREFLIGHT_PLAN_PATH,
) -> dict[str, Any]:
    """Load canonical authorization and bind it to the current frozen plans."""

    load_and_validate_attempt08_plan(attempt08_plan_path)
    load_preflight_plan(preflight_plan_path)
    attempt08_sha = _sha256_file(attempt08_plan_path)
    preflight_sha = _sha256_file(preflight_plan_path)
    if attempt08_sha != M43_ATTEMPT08_PLAN_SHA256:
        raise ValueError("Attempt08 target plan SHA-256 changed")
    if preflight_sha != ATTEMPT08_PREFLIGHT_PLAN_SHA256:
        raise ValueError("Attempt08 preflight plan SHA-256 changed")
    payload = _load_canonical(path, label="development authorization")
    validate_development_open_authorization(
        payload,
        attempt08_plan_sha256=attempt08_sha,
        preflight_plan_sha256=preflight_sha,
    )
    return payload


def _publish_atomic_files(files: Mapping[Path, bytes]) -> None:
    """Publish one or two files as a best-effort atomic no-clobber group."""

    identities = [os.path.normcase(str(path.resolve(strict=False))) for path in files]
    if len(identities) != len(set(identities)):
        raise ValueError("Attempt08 finalizer output paths must be distinct")
    for path in files:
        if path.exists():
            raise FileExistsError(f"Attempt08 finalizer output exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    temporaries: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for target, payload in files.items():
            handle = tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            )
            temporary = Path(handle.name)
            with handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporaries[target] = temporary
        for target, temporary in temporaries.items():
            os.link(temporary, target)
            published.append(target)
    except BaseException:
        for target in published:
            target.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def finalize_preflight(
    *,
    aggregate: str | Path,
    output: str | Path,
    authorization_output: str | Path | None = None,
    attempt08_plan: str | Path = DEFAULT_ATTEMPT08_PLAN_PATH,
    preflight_plan: str | Path = DEFAULT_PREFLIGHT_PLAN_PATH,
    proof_paths: Mapping[str, str | Path] | None = None,
    source: str | Path = DEFAULT_SOURCE_PATH,
    execution_evidence: str | Path | None = None,
    spot_run_dir: str | Path | None = None,
    spot_authorization: str | Path | None = None,
    spot_local_evidence: str | Path | None = None,
    spot_jobs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write finalization and, only on Go, a development200 authorization."""

    load_and_validate_attempt08_plan(attempt08_plan)
    load_preflight_plan(preflight_plan)
    if _sha256_file(attempt08_plan) != M43_ATTEMPT08_PLAN_SHA256:
        raise ValueError("Attempt08 target plan SHA-256 changed")
    if _sha256_file(preflight_plan) != ATTEMPT08_PREFLIGHT_PLAN_SHA256:
        raise ValueError("Attempt08 preflight plan SHA-256 changed")
    aggregate_payload = load_and_validate_preflight_aggregate(aggregate)
    aggregate_sha = _sha256_file(aggregate)
    decision = str(aggregate_payload["decision"])
    is_go = decision == "go"
    output_path = Path(output)
    auth_path = Path(authorization_output) if authorization_output is not None else None
    inputs = {
        os.path.normcase(str(Path(value).resolve(strict=False)))
        for value in (aggregate, attempt08_plan, preflight_plan, source)
    }
    if proof_paths is not None:
        inputs.update(
            os.path.normcase(str(Path(value).resolve(strict=False)))
            for value in proof_paths.values()
        )
    if execution_evidence is not None:
        inputs.add(
            os.path.normcase(str(Path(execution_evidence).resolve(strict=False)))
        )
    spot_bundle_paths = (
        spot_run_dir, spot_authorization, spot_local_evidence, spot_jobs_root
    )
    if any(value is not None for value in spot_bundle_paths):
        if not all(value is not None for value in spot_bundle_paths):
            raise ValueError("Attempt08 Spot execution bundle paths must be all-or-none")
        inputs.update(
            os.path.normcase(str(Path(value).resolve(strict=False)))
            for value in spot_bundle_paths
            if value is not None
        )
    outputs = {os.path.normcase(str(output_path.resolve(strict=False)))}
    if auth_path is not None:
        outputs.add(os.path.normcase(str(auth_path.resolve(strict=False))))
    if inputs.intersection(outputs) or len(outputs) != (2 if auth_path else 1):
        raise ValueError("Attempt08 finalizer inputs and outputs must be distinct")
    if spot_run_dir is not None and spot_jobs_root is not None:
        protected_roots = (Path(spot_run_dir).resolve(), Path(spot_jobs_root).resolve())
        if any(
            Path(output).resolve(strict=False).is_relative_to(protected)
            for output in (output_path, *(tuple([auth_path]) if auth_path else ()))
            for protected in protected_roots
        ):
            raise ValueError("Attempt08 finalizer outputs must be outside Spot inputs")
    if is_go and auth_path is None:
        raise ValueError("Attempt08 preflight Go requires authorization_output")
    if is_go:
        if (
            proof_paths is None
            or set(proof_paths) != set(ATTEMPT08_PREFLIGHT_SLOTS)
        ):
            raise ValueError("Attempt08 preflight Go requires exact five proof paths")
        if execution_evidence is None:
            raise ValueError("Attempt08 preflight Go requires Spot execution evidence")
        if not all(value is not None for value in spot_bundle_paths):
            raise ValueError("Attempt08 preflight Go requires actual Spot execution bundle")
        execution_payload = _load_canonical(
            execution_evidence, label="preflight execution evidence"
        )
        validate_preflight_execution_evidence(execution_payload)
        if (
            _sha256_file(execution_evidence)
            != aggregate_payload.get("spot_operational_evidence_sha256")
        ):
            raise ValueError("Attempt08 Spot execution evidence hash changed")
        # Lazy import avoids the module cycle: the Spot lifecycle imports this
        # finalizer, while Go finalization must re-open the lifecycle artifacts.
        from .hu_m43_attempt08_preflight_spot import (
            validate_preflight_receive_bundle,
        )

        assert spot_run_dir is not None
        assert spot_authorization is not None
        assert spot_local_evidence is not None
        assert spot_jobs_root is not None
        received = validate_preflight_receive_bundle(
            run_dir=spot_run_dir,
            authorization_path=spot_authorization,
            local_evidence_path=spot_local_evidence,
            jobs_root=spot_jobs_root,
            execution_evidence_path=execution_evidence,
        )
        expected_proof_paths = {
            slot: os.path.normcase(str(Path(path).resolve(strict=True)))
            for slot, path in received["proof_paths"].items()
        }
        supplied_proof_paths = {
            slot: os.path.normcase(str(Path(path).resolve(strict=True)))
            for slot, path in proof_paths.items()
        }
        if supplied_proof_paths != expected_proof_paths:
            raise ValueError("Attempt08 proof paths are not the received Spot proofs")
        validate_preflight_aggregate_with_proofs(
            aggregate_payload, proof_paths=proof_paths, source=source
        )
    if not is_go and auth_path is not None and auth_path.exists():
        raise FileExistsError("Attempt08 No-Go refuses an existing authorization path")

    authorization: dict[str, Any] | None = None
    authorization_bytes: bytes | None = None
    authorization_sha: str | None = None
    if is_go:
        authorization = _authorization_payload(
            aggregate=aggregate_payload, aggregate_sha256=aggregate_sha
        )
        validate_development_open_authorization(authorization)
        authorization_bytes = canonical_json_bytes(authorization)
        authorization_sha = hashlib.sha256(authorization_bytes).hexdigest()
    status = ATTEMPT08_PREFLIGHT_GO_STATUS if is_go else ATTEMPT08_PREFLIGHT_NO_GO_STATUS
    report = {
        "schema": ATTEMPT08_PREFLIGHT_FINALIZATION_SCHEMA,
        "milestone": "M4.3-attempt08",
        "status": status,
        "decision": (
            "authorize_exact_development_roots_0_through_199_only"
            if is_go
            else "no_go_no_development_authorization"
        ),
        "inputs": {
            "attempt08_plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
            "preflight_plan_sha256": ATTEMPT08_PREFLIGHT_PLAN_SHA256,
            "preflight_result_schema": ATTEMPT08_PREFLIGHT_AGGREGATE_SCHEMA,
            "preflight_result_sha256": aggregate_sha,
            "proof_evidence_sha256": aggregate_payload["proof_evidence_sha256"],
            "runtime_semantic_anchor_sha256": (
                ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
            ),
            "runtime_source_closure_sha256": (
                ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
            ),
            "runtime_fingerprint_sha256": aggregate_payload[
                "runtime_fingerprint_sha256"
            ],
            "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
            "preflight_execution_evidence_sha256": aggregate_payload[
                "spot_operational_evidence_sha256"
            ],
        },
        "gate_digests": {
            "proof_gates_sha256": _sha256_value(aggregate_payload["proof_gates"]),
            "operational_gates_sha256": _sha256_value(
                aggregate_payload["operational_gates"]
            ),
        },
        "authorization": {
            "emitted": is_go,
            "schema": (
                ATTEMPT08_DEVELOPMENT_OPEN_AUTHORIZATION_SCHEMA if is_go else None
            ),
            "sha256": authorization_sha,
        },
        "science_boundary": {
            "future_audit_authorized": False,
            "fit_started": False,
            "threshold_selection_started": False,
            "runtime_policy_activated": False,
            "current_profile_changed": False,
            "full_replacement_enabled": False,
        },
        "next": (
            "development_runner_and_spot_package_must_consume_exact_authorization_sha"
            if is_go
            else "close_preflight_no_go_without_authorization"
        ),
    }
    report_bytes = canonical_json_bytes(report)
    files = {output_path: report_bytes}
    if is_go:
        assert auth_path is not None and authorization_bytes is not None
        files[auth_path] = authorization_bytes
    _publish_atomic_files(files)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorization-output")
    parser.add_argument("--attempt08-plan", default=str(DEFAULT_ATTEMPT08_PLAN_PATH))
    parser.add_argument("--preflight-plan", default=str(DEFAULT_PREFLIGHT_PLAN_PATH))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--execution-evidence")
    parser.add_argument("--spot-run-dir")
    parser.add_argument("--spot-authorization")
    parser.add_argument("--spot-local-evidence")
    parser.add_argument("--spot-jobs-root")
    for slot in ATTEMPT08_PREFLIGHT_SLOTS:
        parser.add_argument(f"--{slot.replace('_', '-')}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    proof_paths = {
        slot: getattr(args, slot) for slot in ATTEMPT08_PREFLIGHT_SLOTS
    }
    if not any(proof_paths.values()):
        proof_paths = None
    elif not all(proof_paths.values()):
        raise ValueError("Attempt08 finalizer requires all five proof paths")
    finalize_preflight(
        aggregate=args.aggregate,
        output=args.output,
        authorization_output=args.authorization_output,
        attempt08_plan=args.attempt08_plan,
        preflight_plan=args.preflight_plan,
        proof_paths=proof_paths,
        source=args.source,
        execution_evidence=args.execution_evidence,
        spot_run_dir=args.spot_run_dir,
        spot_authorization=args.spot_authorization,
        spot_local_evidence=args.spot_local_evidence,
        spot_jobs_root=args.spot_jobs_root,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTEMPT08_DEVELOPMENT_OPEN_AUTHORIZATION_SCHEMA",
    "ATTEMPT08_PREFLIGHT_FINALIZATION_SCHEMA",
    "ATTEMPT08_PREFLIGHT_GO_STATUS",
    "ATTEMPT08_PREFLIGHT_NO_GO_STATUS",
    "ATTEMPT08_PREFLIGHT_EXECUTION_EVIDENCE_SCHEMA",
    "finalize_preflight",
    "load_and_validate_development_open_authorization",
    "validate_development_open_authorization",
    "validate_preflight_execution_evidence",
]
