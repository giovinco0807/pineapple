"""Create-only, source-replayed M3.1 T3 opt-in registration.

This module is deliberately outside ``ai_profiles.py``.  It can create a
dormant registration manifest only after the post-dataset authorization and
the exact locked population/ABR gate replay as a pass.  Loading a candidate
also requires the caller to provide:

* the manifest path and its exact file SHA-256;
* the unchanged ``stage7_m5_r10`` baseline policy object;
* the torch module; and
* the repository root used to replay the runtime closure.

There is no default path, environment-variable fallback, ``current`` lookup,
profile registry mutation, or full-replacement path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_post_dataset_controller_v1 as post_dataset
from . import hu_m31_t3_promotion_runtime_closure_v1 as runtime_closure
from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from . import hu_m31_t3_street_policy_runtime_v1 as policy_runtime
from . import hu_m31_t3_street_policy_training_cli_v1 as training_cli
from . import hu_m31_t3_street_policy_training_v1 as training


REGISTRATION_MANIFEST_SCHEMA = "hu_m31_t3_opt_in_registration_manifest_v1"
REGISTRATION_PATCH_SPEC_SCHEMA = "hu_m31_t3_opt_in_registry_patch_spec_v1"
REGISTRATION_TRANSITION_RECEIPT_SCHEMA = (
    "hu_m31_t3_opt_in_registry_transition_receipt_v1"
)
REGISTRATION_STATUS = (
    "qualified_create_only_dormant_explicit_opt_in_not_registered"
)
PROFILE_ID = policy_runtime.OPT_IN_PROFILE_CANDIDATE
BASELINE_PROFILE_ID = policy_runtime.BASELINE_PROFILE
FACTORY_MODULE = "ofc_regular.hu_m31_t3_street_policy_runtime_v1"
FACTORY_SYMBOL = "build_opt_in_t3_policy_candidate"
RESOLVER_MODULE = "ofc_regular.hu_m31_t3_opt_in_registration_v1"
RESOLVER_SYMBOL = "resolve_explicit_opt_in_t3_policy"
PINNED_POLICY_REGISTRY_SHA256 = runtime_closure.FROZEN_POLICY_REGISTRY_SHA256
_SHA_CHARS = frozenset("0123456789abcdef")


class OptInRegistrationError(RuntimeError):
    """Raised when registration evidence cannot be replayed exactly."""


def canonical_bytes(value: Any) -> bytes:
    return promotion.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return promotion.canonical_sha256(value)


def sha256_file(path: str | Path) -> str:
    return promotion.sha256_file(path)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA_CHARS
    )


def _require_sha(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _safe_absolute_file(path: str | Path, label: str) -> Path:
    source = Path(path)
    if not source.is_absolute():
        raise ValueError(f"{label} path must be absolute and explicit")
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return source.resolve()


def _safe_absolute_directory(path: str | Path, label: str) -> Path:
    source = Path(path)
    if not source.is_absolute():
        raise ValueError(f"{label} path must be absolute and explicit")
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"{label} must be a regular non-symlink directory")
    return source.resolve()


def _read_canonical_file(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], bytes, Path]:
    source = _safe_absolute_file(path, label)
    expected = _require_sha(expected_sha256, f"{label} file")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise OptInRegistrationError(f"{label} file SHA-256 changed")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OptInRegistrationError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise OptInRegistrationError(
            f"{label} is not a canonical JSON object"
        )
    return value, raw, source


def _write_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    if not target.is_absolute():
        raise ValueError("output path must be absolute and explicit")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"create-only output already exists: {target}")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _file_record(
    path: Path,
    *,
    schema: str,
    internal_identity_sha256: str,
) -> dict[str, Any]:
    return {
        "absolute_path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "schema": schema,
        "internal_identity_sha256": _require_sha(
            internal_identity_sha256, f"{schema} internal identity"
        ),
    }


def _authorization_identity(value: Mapping[str, Any]) -> str:
    identity = dict(value)
    declared = identity.pop("authorization_sha256", None)
    computed = canonical_sha256(identity)
    if declared != computed:
        raise OptInRegistrationError(
            "post-dataset authorization internal identity changed"
        )
    return computed


def _training_config_from_checkpoint(
    checkpoint_manifest: Mapping[str, Any],
) -> training.StreetPolicyTrainingConfig:
    raw = checkpoint_manifest.get("training_config")
    if not isinstance(raw, Mapping):
        raise OptInRegistrationError(
            "checkpoint bundle lacks its training config"
        )
    values = dict(raw)
    if values.pop("schema", None) != training.TRAINING_CONFIG_SCHEMA:
        raise OptInRegistrationError("checkpoint training config schema changed")
    try:
        config = training.StreetPolicyTrainingConfig(**values)
    except (TypeError, ValueError) as exc:
        raise OptInRegistrationError(
            "checkpoint training config is invalid"
        ) from exc
    if (
        config.to_dict() != dict(raw)
        or config.identity_sha256
        != checkpoint_manifest.get("training_config_sha256")
    ):
        raise OptInRegistrationError(
            "checkpoint training config identity changed"
        )
    return config


def _replay_authorization(
    *,
    authorization: Mapping[str, Any],
    promotion_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    shard_directory: Path,
    merge: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        replayed = post_dataset.build_opt_in_registration_authorization(
            promotion_plan=promotion_plan,
            execution_plan=execution_plan,
            shard_directory=shard_directory,
            merge=merge,
            gate=gate,
        )
    except (OSError, TypeError, ValueError, PermissionError) as exc:
        raise OptInRegistrationError(
            "post-dataset authorization failed exact source replay"
        ) from exc
    if dict(authorization) != replayed:
        raise OptInRegistrationError(
            "post-dataset authorization differs from source replay"
        )
    _authorization_identity(replayed)
    return replayed


def _require_passing_gate(gate: Mapping[str, Any]) -> None:
    gates = gate.get("gates")
    if (
        gate.get("status") != "pass"
        or gate.get("all_gates_passed") is not True
        or gate.get("scientific_promotion_passed") is not True
        or gate.get("separate_opt_in_profile_candidate_authorized") is not True
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or gate.get("named_profile_added") is not False
        or gate.get("current_profile_changed") is not False
        or gate.get("runtime_activated") is not False
        or gate.get("full_replacement_enabled") is not False
    ):
        raise PermissionError(
            "registration requires an exact fully passing promotion gate"
        )


def build_registration_manifest(
    *,
    torch: Any,
    authorization_path: str | Path,
    expected_authorization_file_sha256: str,
    promotion_plan_path: str | Path,
    expected_promotion_plan_file_sha256: str,
    execution_plan_path: str | Path,
    expected_execution_plan_file_sha256: str,
    evaluation_shard_directory: str | Path,
    promotion_merge_path: str | Path,
    expected_promotion_merge_file_sha256: str,
    promotion_gate_path: str | Path,
    expected_promotion_gate_file_sha256: str,
    checkpoint_bundle_directory: str | Path,
    expected_checkpoint_manifest_file_sha256: str,
    training_run_config_path: str | Path,
    expected_training_run_config_file_sha256: str,
    training_threshold_lock_path: str | Path,
    expected_training_threshold_lock_file_sha256: str,
    compatibility_threshold_lock_path: str | Path,
    expected_compatibility_threshold_lock_file_sha256: str,
    evaluation_runtime_closure_path: str | Path,
    expected_evaluation_runtime_closure_file_sha256: str,
    policy_registry_path: str | Path,
    expected_policy_registry_file_sha256: str,
    source_replay_root: str | Path,
) -> dict[str, Any]:
    """Replay every qualification input and build a dormant manifest."""

    source_root = _safe_absolute_directory(
        source_replay_root, "source replay root"
    )
    authorization, _authorization_raw, authorization_file = (
        _read_canonical_file(
            authorization_path,
            expected_sha256=expected_authorization_file_sha256,
            label="post-dataset authorization",
        )
    )
    plan, _plan_raw, plan_file = _read_canonical_file(
        promotion_plan_path,
        expected_sha256=expected_promotion_plan_file_sha256,
        label="locked promotion plan",
    )
    execution_plan, _execution_raw, execution_file = _read_canonical_file(
        execution_plan_path,
        expected_sha256=expected_execution_plan_file_sha256,
        label="locked promotion execution plan",
    )
    shard_directory = _safe_absolute_directory(
        evaluation_shard_directory, "locked evaluation shard directory"
    )
    merge, _merge_raw, merge_file = _read_canonical_file(
        promotion_merge_path,
        expected_sha256=expected_promotion_merge_file_sha256,
        label="locked promotion merge",
    )
    gate, _gate_raw, gate_file = _read_canonical_file(
        promotion_gate_path,
        expected_sha256=expected_promotion_gate_file_sha256,
        label="locked promotion gate",
    )
    run_config_file = _safe_absolute_file(
        training_run_config_path, "StreetPolicyNetV1 run config"
    )
    if (
        sha256_file(run_config_file)
        != _require_sha(
            expected_training_run_config_file_sha256,
            "StreetPolicyNetV1 run config file",
        )
    ):
        raise OptInRegistrationError(
            "StreetPolicyNetV1 run config file SHA-256 changed"
        )
    run_config = training_cli.load_run_config(
        run_config_file,
        expected_file_sha256=expected_training_run_config_file_sha256,
    )
    bundle = _safe_absolute_directory(
        checkpoint_bundle_directory, "StreetPolicyNetV1 checkpoint bundle"
    )
    checkpoint_manifest_path = _safe_absolute_file(
        bundle / "manifest.json", "StreetPolicyNetV1 checkpoint manifest"
    )
    if (
        sha256_file(checkpoint_manifest_path)
        != _require_sha(
            expected_checkpoint_manifest_file_sha256,
            "StreetPolicyNetV1 checkpoint manifest file",
        )
    ):
        raise OptInRegistrationError(
            "StreetPolicyNetV1 checkpoint manifest file SHA-256 changed"
        )
    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_bytes().decode("ascii")
    )
    if (
        not isinstance(checkpoint_manifest, dict)
        or checkpoint_manifest_path.read_bytes()
        != canonical_bytes(checkpoint_manifest)
    ):
        raise OptInRegistrationError(
            "StreetPolicyNetV1 checkpoint manifest is not canonical"
        )
    checkpoint_config = _training_config_from_checkpoint(checkpoint_manifest)
    if checkpoint_config != run_config.training_config:
        raise OptInRegistrationError(
            "run config and checkpoint training config differ"
        )
    try:
        replayed_models, replayed_checkpoint_manifest = (
            training.load_ensemble_checkpoint_bundle(
                bundle,
                torch=torch,
                expected_dataset_identity_sha256=str(
                    checkpoint_manifest["training_view_identity_sha256"]
                ),
                expected_training_config=run_config.training_config,
                expected_stage="risk",
                expected_bundle_identity_sha256=str(
                    checkpoint_manifest["bundle_identity_sha256"]
                ),
            )
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise OptInRegistrationError(
            "StreetPolicyNetV1 checkpoint bundle failed source replay"
        ) from exc
    if (
        replayed_checkpoint_manifest != checkpoint_manifest
        or checkpoint_manifest["completed_epoch"]
        != run_config.training_config.risk_epochs
        or any(
            getattr(model, "config", None) != run_config.model_config
            for model in replayed_models
        )
    ):
        raise OptInRegistrationError(
            "only the completed risk-stage checkpoint may be registered"
        )

    rich_threshold, rich_threshold_raw, rich_threshold_file = (
        _read_canonical_file(
            training_threshold_lock_path,
            expected_sha256=expected_training_threshold_lock_file_sha256,
            label="StreetPolicyNetV1 training threshold lock",
        )
    )
    model_hashes = [
        str(record["model_state_sha256"])
        for record in checkpoint_manifest["models"]
    ]
    try:
        rich_threshold = training._validate_threshold_lock(  # type: ignore[attr-defined]
            rich_threshold,
            training_config=run_config.training_config,
            expected_dataset_identity_sha256=str(
                checkpoint_manifest["training_view_identity_sha256"]
            ),
            expected_model_hashes=model_hashes,
        )
    except (TypeError, ValueError) as exc:
        raise OptInRegistrationError(
            "StreetPolicyNetV1 training threshold lock failed source replay"
        ) from exc
    compatibility_lock, _compatibility_raw, compatibility_file = (
        _read_canonical_file(
            compatibility_threshold_lock_path,
            expected_sha256=(
                expected_compatibility_threshold_lock_file_sha256
            ),
            label="locked-promotion compatibility threshold lock",
        )
    )
    policy_registry = _safe_absolute_file(
        policy_registry_path, "frozen policy registry"
    )
    policy_registry_sha = sha256_file(policy_registry)
    if (
        policy_registry_sha
        != _require_sha(
            expected_policy_registry_file_sha256,
            "frozen policy registry file",
        )
        or policy_registry_sha != PINNED_POLICY_REGISTRY_SHA256
    ):
        raise OptInRegistrationError(
            "policy registry differs from the frozen pre-registration source"
        )
    closure = _safe_absolute_file(
        evaluation_runtime_closure_path, "evaluation runtime closure"
    )
    closure_sha = sha256_file(closure)
    if closure_sha != _require_sha(
        expected_evaluation_runtime_closure_file_sha256,
        "evaluation runtime closure file",
    ):
        raise OptInRegistrationError(
            "evaluation runtime closure file SHA-256 changed"
        )

    try:
        validated_plan = promotion.validate_locked_promotion_plan(plan)
        validated_merge = promotion.validate_locked_promotion_merge(
            merge, plan=validated_plan, replay_sources=True
        )
        validated_gate = promotion.validate_locked_promotion_gate(
            gate,
            plan=validated_plan,
            merge=validated_merge,
            replay_sources=True,
        )
        promotion.validate_artifact_files(
            validated_plan,
            model_path=checkpoint_manifest_path,
            threshold_lock_path=compatibility_file,
            policy_registry_path=policy_registry,
            evaluation_runtime_closure_path=closure,
        )
        compatibility_lock = promotion._validate_threshold_lock(
            compatibility_lock,
            expected_model_sha256=sha256_file(checkpoint_manifest_path),
        )
        seat_thresholds, seat_enabled = policy_runtime._validate_compatibility_lock(
            compatibility_lock,
            training_threshold_lock=rich_threshold,
            training_threshold_lock_file_sha256=hashlib.sha256(
                rich_threshold_raw
            ).hexdigest(),
            checkpoint_bundle_identity_sha256=str(
                checkpoint_manifest["bundle_identity_sha256"]
            ),
        )
    except (OSError, TypeError, ValueError, PermissionError) as exc:
        raise OptInRegistrationError(
            "promotion plan, merge, gate, or bound artifact failed source replay"
        ) from exc
    _require_passing_gate(validated_gate)
    if seat_enabled != {"first": True, "second": True}:
        raise PermissionError(
            "registration requires enabled locked thresholds for both seats"
        )

    closure_manifest = runtime_closure.validate_runtime_closure_package(
        closure,
        expected_sha256=closure_sha,
        source_replay_root=source_root,
    )
    candidate_binding = closure_manifest["candidate_artifacts"]
    if (
        candidate_binding["checkpoint_manifest_sha256"]
        != sha256_file(checkpoint_manifest_path)
        or candidate_binding["checkpoint_bundle_identity_sha256"]
        != checkpoint_manifest["bundle_identity_sha256"]
        or candidate_binding["training_config_sha256"]
        != run_config.training_config.identity_sha256
        or candidate_binding["training_threshold_lock_sha256"]
        != sha256_file(rich_threshold_file)
        or closure_manifest["entries"]["src/ofc_regular/ai_profiles.py"][
            "sha256"
        ]
        != policy_registry_sha
        or closure_manifest["factory_contract"]["current_profile_allowed"]
        is not False
        or closure_manifest["factory_contract"][
            "implicit_profile_resolution_allowed"
        ]
        is not False
    ):
        raise OptInRegistrationError(
            "runtime closure is bound to another candidate or registry"
        )

    replayed_authorization = _replay_authorization(
        authorization=authorization,
        promotion_plan=validated_plan,
        execution_plan=execution_plan,
        shard_directory=shard_directory,
        merge=validated_merge,
        gate=validated_gate,
    )
    if (
        replayed_authorization["profile_id"] != PROFILE_ID
        or replayed_authorization["baseline_profile_id"]
        != BASELINE_PROFILE_ID
        or replayed_authorization["runtime_factory_module"] != FACTORY_MODULE
        or replayed_authorization["runtime_factory_symbol"] != FACTORY_SYMBOL
        or replayed_authorization["registration_authorized"] is not True
        or replayed_authorization["registration_applied"] is not False
        or replayed_authorization["named_profile_added"] is not False
        or replayed_authorization["current_profile_changed"] is not False
        or replayed_authorization["runtime_activated"] is not False
        or replayed_authorization["full_replacement_enabled"] is not False
    ):
        raise OptInRegistrationError(
            "post-dataset authorization crossed the dormant opt-in boundary"
        )

    model_members = []
    for record in checkpoint_manifest["models"]:
        member = _safe_absolute_file(
            bundle / str(record["path"]),
            f"checkpoint model {record['model_index']}",
        )
        model_members.append(
            {
                "absolute_path": str(member),
                "bytes": member.stat().st_size,
                "file_sha256": str(record["sha256"]),
                "model_index": int(record["model_index"]),
                "model_state_sha256": str(record["model_state_sha256"]),
                "checkpoint_identity_sha256": str(
                    record["checkpoint_identity_sha256"]
                ),
            }
        )
    payload = {
        "schema": REGISTRATION_MANIFEST_SCHEMA,
        "status": REGISTRATION_STATUS,
        "profile_id": PROFILE_ID,
        "baseline_profile_id": BASELINE_PROFILE_ID,
        "factory": {
            "module": FACTORY_MODULE,
            "symbol": FACTORY_SYMBOL,
            "resolver_module": RESOLVER_MODULE,
            "resolver_symbol": RESOLVER_SYMBOL,
            "requires_explicit_manifest_path": True,
            "requires_explicit_manifest_file_sha256": True,
            "requires_explicit_baseline_policy_object": True,
            "default_path_allowed": False,
            "environment_fallback_allowed": False,
        },
        "artifacts": {
            "post_dataset_authorization": _file_record(
                authorization_file,
                schema=post_dataset.OPT_IN_AUTHORIZATION_SCHEMA,
                internal_identity_sha256=_authorization_identity(
                    replayed_authorization
                ),
            ),
            "promotion_plan": _file_record(
                plan_file,
                schema=promotion.PLAN_SCHEMA,
                internal_identity_sha256=canonical_sha256(validated_plan),
            ),
            "evaluation_execution_plan": _file_record(
                execution_file,
                schema=str(execution_plan["schema"]),
                internal_identity_sha256=canonical_sha256(execution_plan),
            ),
            "evaluation_shards": {
                "absolute_path": str(shard_directory),
                "work_item_count": 260,
                "row_count": 13_000,
                "source_shard_aggregate_sha256": validated_merge[
                    "source_shard_aggregate_sha256"
                ],
            },
            "promotion_merge": _file_record(
                merge_file,
                schema=promotion.MERGE_SCHEMA,
                internal_identity_sha256=canonical_sha256(validated_merge),
            ),
            "promotion_gate": _file_record(
                gate_file,
                schema=promotion.GATE_SCHEMA,
                internal_identity_sha256=canonical_sha256(validated_gate),
            ),
            "checkpoint_bundle": {
                "absolute_path": str(bundle),
                "manifest_absolute_path": str(checkpoint_manifest_path),
                "manifest_file_sha256": sha256_file(
                    checkpoint_manifest_path
                ),
                "bundle_identity_sha256": checkpoint_manifest[
                    "bundle_identity_sha256"
                ],
                "training_view_identity_sha256": checkpoint_manifest[
                    "training_view_identity_sha256"
                ],
                "members": model_members,
            },
            "training_run_config": _file_record(
                run_config_file,
                schema=training_cli.RUN_CONFIG_SCHEMA,
                internal_identity_sha256=run_config.identity_sha256,
            ),
            "training_threshold_lock": _file_record(
                rich_threshold_file,
                schema=training.THRESHOLD_LOCK_SCHEMA,
                internal_identity_sha256=str(
                    rich_threshold["threshold_lock_sha256"]
                ),
            ),
            "compatibility_threshold_lock": _file_record(
                compatibility_file,
                schema=promotion.THRESHOLD_LOCK_SCHEMA,
                internal_identity_sha256=canonical_sha256(compatibility_lock),
            ),
            "evaluation_runtime_closure": _file_record(
                closure,
                schema=runtime_closure.CLOSURE_SCHEMA,
                internal_identity_sha256=closure_manifest[
                    "manifest_identity_sha256"
                ],
            ),
            "policy_registry_before": _file_record(
                policy_registry,
                schema="python_source_policy_registry_v1",
                internal_identity_sha256=policy_registry_sha,
            ),
        },
        "runtime_binding": {
            "training_view_identity_sha256": checkpoint_manifest[
                "training_view_identity_sha256"
            ],
            "training_config_sha256": run_config.training_config.identity_sha256,
            "checkpoint_bundle_identity_sha256": checkpoint_manifest[
                "bundle_identity_sha256"
            ],
            "training_threshold_lock_file_sha256": sha256_file(
                rich_threshold_file
            ),
            "compatibility_threshold_lock_file_sha256": sha256_file(
                compatibility_file
            ),
            "promotion_gate_file_sha256": sha256_file(gate_file),
            "seat_safe_probability_thresholds": seat_thresholds,
            "seat_enabled": seat_enabled,
        },
        "registration_contract": {
            "scope": "one_explicit_named_profile_only",
            "both_seats_required": True,
            "baseline_action_object_returned_unchanged_on_nonfire": True,
            "policy_rng_advanced_on_nonfire": False,
            "future_deck_or_board_mutated_on_nonfire": False,
            "implicit_current_resolution_allowed": False,
            "full_replacement_allowed": False,
            "teacher_ev_lcb_runtime_gate": False,
            "registration_applied": False,
            "named_profile_added": False,
            "current_profile_changed": False,
            "runtime_activated": False,
        },
    }
    result = dict(payload)
    result["manifest_identity_sha256"] = canonical_sha256(payload)
    return result


def write_registration_manifest(
    *,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    manifest = build_registration_manifest(**kwargs)
    target = _write_once(output_path, manifest)
    stored, raw, _ = _read_canonical_file(
        target,
        expected_sha256=sha256_file(target),
        label="stored opt-in registration manifest",
    )
    if stored != manifest:
        raise OptInRegistrationError("stored registration manifest changed")
    return {
        "schema": REGISTRATION_MANIFEST_SCHEMA,
        "status": REGISTRATION_STATUS,
        "absolute_path": str(target),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "registration_applied": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }


def _manifest_replay_kwargs(
    manifest: Mapping[str, Any],
    *,
    torch: Any,
    source_replay_root: str | Path,
) -> dict[str, Any]:
    artifacts = manifest["artifacts"]
    checkpoint = artifacts["checkpoint_bundle"]
    return {
        "torch": torch,
        "authorization_path": artifacts["post_dataset_authorization"][
            "absolute_path"
        ],
        "expected_authorization_file_sha256": artifacts[
            "post_dataset_authorization"
        ]["file_sha256"],
        "promotion_plan_path": artifacts["promotion_plan"]["absolute_path"],
        "expected_promotion_plan_file_sha256": artifacts["promotion_plan"][
            "file_sha256"
        ],
        "execution_plan_path": artifacts["evaluation_execution_plan"][
            "absolute_path"
        ],
        "expected_execution_plan_file_sha256": artifacts[
            "evaluation_execution_plan"
        ]["file_sha256"],
        "evaluation_shard_directory": artifacts["evaluation_shards"][
            "absolute_path"
        ],
        "promotion_merge_path": artifacts["promotion_merge"]["absolute_path"],
        "expected_promotion_merge_file_sha256": artifacts["promotion_merge"][
            "file_sha256"
        ],
        "promotion_gate_path": artifacts["promotion_gate"]["absolute_path"],
        "expected_promotion_gate_file_sha256": artifacts["promotion_gate"][
            "file_sha256"
        ],
        "checkpoint_bundle_directory": checkpoint["absolute_path"],
        "expected_checkpoint_manifest_file_sha256": checkpoint[
            "manifest_file_sha256"
        ],
        "training_run_config_path": artifacts["training_run_config"][
            "absolute_path"
        ],
        "expected_training_run_config_file_sha256": artifacts[
            "training_run_config"
        ]["file_sha256"],
        "training_threshold_lock_path": artifacts["training_threshold_lock"][
            "absolute_path"
        ],
        "expected_training_threshold_lock_file_sha256": artifacts[
            "training_threshold_lock"
        ]["file_sha256"],
        "compatibility_threshold_lock_path": artifacts[
            "compatibility_threshold_lock"
        ]["absolute_path"],
        "expected_compatibility_threshold_lock_file_sha256": artifacts[
            "compatibility_threshold_lock"
        ]["file_sha256"],
        "evaluation_runtime_closure_path": artifacts[
            "evaluation_runtime_closure"
        ]["absolute_path"],
        "expected_evaluation_runtime_closure_file_sha256": artifacts[
            "evaluation_runtime_closure"
        ]["file_sha256"],
        "policy_registry_path": artifacts["policy_registry_before"][
            "absolute_path"
        ],
        "expected_policy_registry_file_sha256": artifacts[
            "policy_registry_before"
        ]["file_sha256"],
        "source_replay_root": source_replay_root,
    }


def validate_registration_manifest(
    manifest_path: str | Path,
    *,
    expected_manifest_file_sha256: str,
    torch: Any,
    source_replay_root: str | Path,
) -> dict[str, Any]:
    """Rebuild the entire manifest from its absolute source bindings."""

    manifest, _raw, _path = _read_canonical_file(
        manifest_path,
        expected_sha256=expected_manifest_file_sha256,
        label="opt-in registration manifest",
    )
    identity = dict(manifest)
    declared = identity.pop("manifest_identity_sha256", None)
    if (
        manifest.get("schema") != REGISTRATION_MANIFEST_SCHEMA
        or manifest.get("status") != REGISTRATION_STATUS
        or manifest.get("profile_id") != PROFILE_ID
        or manifest.get("baseline_profile_id") != BASELINE_PROFILE_ID
        or declared != canonical_sha256(identity)
    ):
        raise OptInRegistrationError(
            "opt-in registration manifest identity or boundary changed"
        )
    try:
        rebuilt = build_registration_manifest(
            **_manifest_replay_kwargs(
                manifest,
                torch=torch,
                source_replay_root=source_replay_root,
            )
        )
    except (KeyError, TypeError, ValueError, PermissionError) as exc:
        raise OptInRegistrationError(
            "opt-in registration manifest source replay failed"
        ) from exc
    if rebuilt != manifest:
        raise OptInRegistrationError(
            "opt-in registration manifest differs from source replay"
        )
    return rebuilt


def resolve_explicit_opt_in_t3_policy(
    *,
    registration_manifest_path: str | Path,
    expected_registration_manifest_file_sha256: str,
    baseline_policy: object,
    baseline_profile_id: str,
    torch: Any,
    source_replay_root: str | Path,
) -> policy_runtime.HuM31T3StreetPolicyRuntime:
    """Resolve the candidate only from an explicit manifest and baseline."""

    if baseline_policy is None:
        raise ValueError("an explicit baseline policy object is required")
    if baseline_profile_id != BASELINE_PROFILE_ID:
        raise ValueError(
            "baseline must be explicit stage7_m5_r10; current is forbidden"
        )
    manifest = validate_registration_manifest(
        registration_manifest_path,
        expected_manifest_file_sha256=(
            expected_registration_manifest_file_sha256
        ),
        torch=torch,
        source_replay_root=source_replay_root,
    )
    artifacts = manifest["artifacts"]
    run_config_record = artifacts["training_run_config"]
    run_config = training_cli.load_run_config(
        run_config_record["absolute_path"],
        expected_file_sha256=run_config_record["file_sha256"],
    )
    checkpoint = artifacts["checkpoint_bundle"]
    evidence = policy_runtime.PromotionEvidencePaths(
        plan_path=Path(artifacts["promotion_plan"]["absolute_path"]),
        merge_path=Path(artifacts["promotion_merge"]["absolute_path"]),
        gate_path=Path(artifacts["promotion_gate"]["absolute_path"]),
        expected_gate_file_sha256=artifacts["promotion_gate"][
            "file_sha256"
        ],
        compatibility_threshold_lock_path=Path(
            artifacts["compatibility_threshold_lock"]["absolute_path"]
        ),
        policy_registry_path=Path(
            artifacts["policy_registry_before"]["absolute_path"]
        ),
        evaluation_runtime_closure_path=Path(
            artifacts["evaluation_runtime_closure"]["absolute_path"]
        ),
    )
    candidate = policy_runtime.build_opt_in_t3_policy_candidate(
        baseline_policy=baseline_policy,
        baseline_profile_id=baseline_profile_id,
        torch=torch,
        checkpoint_bundle_path=checkpoint["absolute_path"],
        expected_dataset_identity_sha256=checkpoint[
            "training_view_identity_sha256"
        ],
        training_config=run_config.training_config,
        expected_bundle_identity_sha256=checkpoint[
            "bundle_identity_sha256"
        ],
        threshold_lock_path=artifacts["training_threshold_lock"][
            "absolute_path"
        ],
        expected_threshold_lock_file_sha256=artifacts[
            "training_threshold_lock"
        ]["file_sha256"],
        promotion_evidence=evidence,
    )
    if (
        getattr(candidate, "_baseline_policy", None) is not baseline_policy
        or candidate.runtime_scope
        != policy_runtime.RUNTIME_SCOPE_QUALIFIED_OPT_IN
    ):
        raise OptInRegistrationError(
            "runtime factory returned another baseline or runtime scope"
        )
    return candidate


def _ast_node_sha256(node: ast.AST) -> str:
    return hashlib.sha256(
        ast.dump(node, annotate_fields=True, include_attributes=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _find_profile_branch(tree: ast.Module, profile_id: str) -> ast.If | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "profile"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == profile_id
        ):
            return node
    return None


def _find_build_policy(tree: ast.Module) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_policy"
    ]
    if len(matches) != 1:
        raise OptInRegistrationError(
            "policy registry must define build_policy exactly once"
        )
    return matches[0]


def _profile_literal_count(tree: ast.Module, profile_id: str) -> int:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else node.targets[0] if len(node.targets) == 1 else None
        )
        if not isinstance(target, ast.Name) or target.id != "ProfileName":
            continue
        annotation = (
            node.annotation if isinstance(node, ast.AnnAssign) else node.value
        )
        values = [
            child.value
            for child in ast.walk(annotation)
            if isinstance(child, ast.Constant)
            and isinstance(child.value, str)
        ]
        return values.count(profile_id)
    raise OptInRegistrationError("policy registry lacks ProfileName")


def build_ai_profiles_patch_spec(
    *,
    registration_manifest_path: str | Path,
    expected_registration_manifest_file_sha256: str,
    policy_registry_patch_target_path: str | Path,
    torch: Any,
    source_replay_root: str | Path,
) -> dict[str, Any]:
    """Describe the only later ``ai_profiles.py`` change; do not apply it."""

    manifest = validate_registration_manifest(
        registration_manifest_path,
        expected_manifest_file_sha256=(
            expected_registration_manifest_file_sha256
        ),
        torch=torch,
        source_replay_root=source_replay_root,
    )
    registry_record = manifest["artifacts"]["policy_registry_before"]
    frozen_registry = _safe_absolute_file(
        registry_record["absolute_path"],
        "frozen pre-registration policy registry evidence",
    )
    registry = _safe_absolute_file(
        policy_registry_patch_target_path,
        "live policy registry patch target",
    )
    if (
        sha256_file(registry) != PINNED_POLICY_REGISTRY_SHA256
        or sha256_file(frozen_registry) != PINNED_POLICY_REGISTRY_SHA256
        or registry_record["file_sha256"] != PINNED_POLICY_REGISTRY_SHA256
        or registry == frozen_registry
    ):
        raise OptInRegistrationError(
            "patch target and immutable pre-registration evidence must be "
            "distinct pinned registry files"
        )
    tree = ast.parse(registry.read_text(encoding="utf-8"))
    current_branch = _find_profile_branch(tree, "current")
    if (
        current_branch is None
        or _find_profile_branch(tree, PROFILE_ID)
        or _profile_literal_count(tree, PROFILE_ID) != 0
    ):
        raise OptInRegistrationError(
            "current branch is missing or opt-in profile already exists"
        )
    payload = {
        "schema": REGISTRATION_PATCH_SPEC_SCHEMA,
        "status": "review_only_minimal_patch_not_applied",
        "registration_manifest_file_sha256": (
            expected_registration_manifest_file_sha256
        ),
        "registration_manifest_identity_sha256": manifest[
            "manifest_identity_sha256"
        ],
        "target_absolute_path": str(registry),
        "frozen_evidence_absolute_path": str(frozen_registry),
        "policy_registry_sha256_before": PINNED_POLICY_REGISTRY_SHA256,
        "profile_id_to_add": PROFILE_ID,
        "baseline_profile_id": BASELINE_PROFILE_ID,
        "resolver_module": RESOLVER_MODULE,
        "resolver_symbol": RESOLVER_SYMBOL,
        "required_build_policy_keyword_arguments": [
            "m31_registration_manifest_path",
            "m31_registration_manifest_file_sha256",
            "m31_torch",
            "m31_source_replay_root",
        ],
        "operations": [
            "add_profile_literal_exactly_once",
            "add_four_optional_keyword_parameters_with_none_defaults",
            "add_one_profile_branch_requiring_all_four_explicit_values",
            "build_stage7_m5_r10_baseline_with_existing_seed_seat_and_lookahead",
            "call_resolver_with_explicit_manifest_hash_baseline_torch_and_root",
        ],
        "forbidden_changes": [
            "change_current_branch_ast",
            "change_existing_stage7_m5_r10_semantics",
            "add_default_manifest_path",
            "read_registration_path_from_environment",
            "enable_full_replacement",
            "activate_profile_implicitly",
        ],
        "current_branch_ast_sha256_before": _ast_node_sha256(current_branch),
        "patch_applied": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }
    result = dict(payload)
    result["patch_spec_identity_sha256"] = canonical_sha256(payload)
    return result


def write_ai_profiles_patch_spec(
    *,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    value = build_ai_profiles_patch_spec(**kwargs)
    target = _write_once(output_path, value)
    return {
        "schema": REGISTRATION_PATCH_SPEC_SCHEMA,
        "status": value["status"],
        "absolute_path": str(target),
        "file_sha256": sha256_file(target),
        "patch_spec_identity_sha256": value[
            "patch_spec_identity_sha256"
        ],
        "patch_applied": False,
        "current_profile_changed": False,
    }


def _validate_patch_spec(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    spec = deepcopy(dict(value))
    identity = dict(spec)
    declared = identity.pop("patch_spec_identity_sha256", None)
    if (
        spec.get("schema") != REGISTRATION_PATCH_SPEC_SCHEMA
        or spec.get("status") != "review_only_minimal_patch_not_applied"
        or spec.get("registration_manifest_identity_sha256")
        != manifest["manifest_identity_sha256"]
        or spec.get("profile_id_to_add") != PROFILE_ID
        or spec.get("baseline_profile_id") != BASELINE_PROFILE_ID
        or spec.get("policy_registry_sha256_before")
        != PINNED_POLICY_REGISTRY_SHA256
        or spec.get("patch_applied") is not False
        or spec.get("current_profile_changed") is not False
        or declared != canonical_sha256(identity)
    ):
        raise OptInRegistrationError("registry patch specification changed")
    return spec


def _read_transition_receipt(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    value, _raw, _ = _read_canonical_file(
        path,
        expected_sha256=expected_sha256,
        label="registration transition receipt",
    )
    identity = dict(value)
    declared = identity.pop("receipt_identity_sha256", None)
    if (
        value.get("schema") != REGISTRATION_TRANSITION_RECEIPT_SCHEMA
        or declared != canonical_sha256(identity)
    ):
        raise OptInRegistrationError(
            "registration transition receipt identity changed"
        )
    return value


def build_before_registration_receipt(
    *,
    registration_manifest_path: str | Path,
    expected_registration_manifest_file_sha256: str,
    patch_spec_path: str | Path,
    expected_patch_spec_file_sha256: str,
    torch: Any,
    source_replay_root: str | Path,
) -> dict[str, Any]:
    manifest = validate_registration_manifest(
        registration_manifest_path,
        expected_manifest_file_sha256=(
            expected_registration_manifest_file_sha256
        ),
        torch=torch,
        source_replay_root=source_replay_root,
    )
    spec, _raw, _ = _read_canonical_file(
        patch_spec_path,
        expected_sha256=expected_patch_spec_file_sha256,
        label="registry patch specification",
    )
    spec = _validate_patch_spec(spec, manifest=manifest)
    registry = _safe_absolute_file(
        spec["target_absolute_path"], "pre-registration policy registry"
    )
    if sha256_file(registry) != PINNED_POLICY_REGISTRY_SHA256:
        raise OptInRegistrationError(
            "pre-registration policy registry SHA-256 changed"
        )
    tree = ast.parse(registry.read_text(encoding="utf-8"))
    if _find_profile_branch(tree, PROFILE_ID) is not None:
        raise OptInRegistrationError("opt-in profile was already registered")
    payload = {
        "schema": REGISTRATION_TRANSITION_RECEIPT_SCHEMA,
        "phase": "before",
        "status": "before_patch_verified_not_applied",
        "registration_manifest_file_sha256": (
            expected_registration_manifest_file_sha256
        ),
        "registration_manifest_identity_sha256": manifest[
            "manifest_identity_sha256"
        ],
        "patch_spec_file_sha256": expected_patch_spec_file_sha256,
        "patch_spec_identity_sha256": spec["patch_spec_identity_sha256"],
        "policy_registry_absolute_path": str(registry),
        "policy_registry_sha256_before": PINNED_POLICY_REGISTRY_SHA256,
        "policy_registry_sha256_after": None,
        "profile_id": PROFILE_ID,
        "registration_applied": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
    }
    result = dict(payload)
    result["receipt_identity_sha256"] = canonical_sha256(payload)
    return result


def _validate_after_registry(
    source: str,
    *,
    spec: Mapping[str, Any],
) -> None:
    if "os.environ" in source or "getenv(" in source:
        raise OptInRegistrationError(
            "after registry contains an environment fallback"
        )
    tree = ast.parse(source)
    profile_branch = _find_profile_branch(tree, PROFILE_ID)
    current_branch = _find_profile_branch(tree, "current")
    build_policy = _find_build_policy(tree)
    if profile_branch is None or current_branch is None:
        raise OptInRegistrationError(
            "after registry lacks current or explicit opt-in branch"
        )
    if (
        _ast_node_sha256(current_branch)
        != spec["current_branch_ast_sha256_before"]
    ):
        raise OptInRegistrationError("current profile branch changed")
    if _profile_literal_count(tree, PROFILE_ID) != 1:
        raise OptInRegistrationError(
            "opt-in profile literal must be added exactly once"
        )
    required = set(spec["required_build_policy_keyword_arguments"])
    keyword_arguments = {
        argument.arg: default
        for argument, default in zip(
            build_policy.args.kwonlyargs,
            build_policy.args.kw_defaults,
            strict=True,
        )
    }
    if not required <= set(keyword_arguments) or any(
        not isinstance(keyword_arguments[name], ast.Constant)
        or keyword_arguments[name].value is not None
        for name in required
    ):
        raise OptInRegistrationError(
            "opt-in build_policy arguments are not explicit None-only inputs"
        )
    calls = [
        node
        for node in ast.walk(profile_branch)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id == RESOLVER_SYMBOL
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == RESOLVER_SYMBOL
            )
        )
    ]
    if len(calls) != 1:
        raise OptInRegistrationError(
            "opt-in profile branch must call the resolver exactly once"
        )
    resolver_call = calls[0]
    keyword_values = {
        keyword.arg: keyword.value
        for keyword in resolver_call.keywords
        if keyword.arg is not None
    }
    expected_runtime_keywords = {
        "registration_manifest_path": "m31_registration_manifest_path",
        "expected_registration_manifest_file_sha256": (
            "m31_registration_manifest_file_sha256"
        ),
        "torch": "m31_torch",
        "source_replay_root": "m31_source_replay_root",
    }
    if any(
        not isinstance(keyword_values.get(keyword), ast.Name)
        or keyword_values[keyword].id != argument
        for keyword, argument in expected_runtime_keywords.items()
    ):
        raise OptInRegistrationError(
            "resolver call does not forward the four explicit inputs"
        )
    baseline_keyword = keyword_values.get("baseline_policy")
    baseline_profile_keyword = keyword_values.get("baseline_profile_id")
    baseline_builds = [
        node
        for node in ast.walk(profile_branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_stage7_policy"
    ]
    if (
        len(baseline_builds) != 1
        or not isinstance(baseline_keyword, ast.Name)
        or baseline_keyword.id != "baseline"
        or not isinstance(baseline_profile_keyword, ast.Constant)
        or baseline_profile_keyword.value != BASELINE_PROFILE_ID
    ):
        raise OptInRegistrationError(
            "opt-in branch does not construct the explicit frozen baseline"
        )


def build_after_registration_receipt(
    *,
    registration_manifest_path: str | Path,
    expected_registration_manifest_file_sha256: str,
    patch_spec_path: str | Path,
    expected_patch_spec_file_sha256: str,
    before_receipt_path: str | Path,
    expected_before_receipt_file_sha256: str,
    policy_registry_after_path: str | Path,
    expected_policy_registry_after_file_sha256: str,
    torch: Any,
    source_replay_root: str | Path,
) -> dict[str, Any]:
    manifest = validate_registration_manifest(
        registration_manifest_path,
        expected_manifest_file_sha256=(
            expected_registration_manifest_file_sha256
        ),
        torch=torch,
        source_replay_root=source_replay_root,
    )
    spec, _raw, _ = _read_canonical_file(
        patch_spec_path,
        expected_sha256=expected_patch_spec_file_sha256,
        label="registry patch specification",
    )
    spec = _validate_patch_spec(spec, manifest=manifest)
    before = _read_transition_receipt(
        before_receipt_path,
        expected_sha256=expected_before_receipt_file_sha256,
    )
    if (
        before.get("phase") != "before"
        or before.get("status") != "before_patch_verified_not_applied"
        or before.get("registration_manifest_identity_sha256")
        != manifest["manifest_identity_sha256"]
        or before.get("patch_spec_identity_sha256")
        != spec["patch_spec_identity_sha256"]
        or before.get("policy_registry_sha256_before")
        != PINNED_POLICY_REGISTRY_SHA256
        or before.get("registration_applied") is not False
    ):
        raise OptInRegistrationError("before receipt is not eligible")
    registry_after = _safe_absolute_file(
        policy_registry_after_path, "after-registration policy registry"
    )
    if registry_after != Path(spec["target_absolute_path"]).resolve():
        raise OptInRegistrationError(
            "after receipt must inspect the exact reviewed patch target"
        )
    expected_after = _require_sha(
        expected_policy_registry_after_file_sha256,
        "after-registration policy registry",
    )
    if (
        sha256_file(registry_after) != expected_after
        or expected_after == PINNED_POLICY_REGISTRY_SHA256
    ):
        raise OptInRegistrationError(
            "after-registration policy registry SHA-256 is invalid"
        )
    _validate_after_registry(
        registry_after.read_text(encoding="utf-8"), spec=spec
    )
    payload = {
        "schema": REGISTRATION_TRANSITION_RECEIPT_SCHEMA,
        "phase": "after",
        "status": "explicit_named_profile_registered_not_activated",
        "registration_manifest_file_sha256": (
            expected_registration_manifest_file_sha256
        ),
        "registration_manifest_identity_sha256": manifest[
            "manifest_identity_sha256"
        ],
        "patch_spec_file_sha256": expected_patch_spec_file_sha256,
        "patch_spec_identity_sha256": spec["patch_spec_identity_sha256"],
        "before_receipt_file_sha256": expected_before_receipt_file_sha256,
        "before_receipt_identity_sha256": before[
            "receipt_identity_sha256"
        ],
        "policy_registry_absolute_path": str(registry_after),
        "policy_registry_sha256_before": PINNED_POLICY_REGISTRY_SHA256,
        "policy_registry_sha256_after": expected_after,
        "profile_id": PROFILE_ID,
        "registration_applied": True,
        "named_profile_added": True,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
    }
    result = dict(payload)
    result["receipt_identity_sha256"] = canonical_sha256(payload)
    return result


def _add_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--promotion-plan", required=True)
    parser.add_argument("--promotion-plan-sha256", required=True)
    parser.add_argument("--execution-plan", required=True)
    parser.add_argument("--execution-plan-sha256", required=True)
    parser.add_argument("--evaluation-shards", required=True)
    parser.add_argument("--promotion-merge", required=True)
    parser.add_argument("--promotion-merge-sha256", required=True)
    parser.add_argument("--promotion-gate", required=True)
    parser.add_argument("--promotion-gate-sha256", required=True)
    parser.add_argument("--checkpoint-bundle", required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--training-run-config", required=True)
    parser.add_argument("--training-run-config-sha256", required=True)
    parser.add_argument("--training-threshold-lock", required=True)
    parser.add_argument("--training-threshold-lock-sha256", required=True)
    parser.add_argument("--compatibility-threshold-lock", required=True)
    parser.add_argument("--compatibility-threshold-lock-sha256", required=True)
    parser.add_argument("--runtime-closure", required=True)
    parser.add_argument("--runtime-closure-sha256", required=True)
    parser.add_argument("--policy-registry", required=True)
    parser.add_argument("--policy-registry-sha256", required=True)
    parser.add_argument("--source-replay-root", required=True)


def _manifest_kwargs(args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    return {
        "torch": torch,
        "authorization_path": args.authorization,
        "expected_authorization_file_sha256": args.authorization_sha256,
        "promotion_plan_path": args.promotion_plan,
        "expected_promotion_plan_file_sha256": args.promotion_plan_sha256,
        "execution_plan_path": args.execution_plan,
        "expected_execution_plan_file_sha256": args.execution_plan_sha256,
        "evaluation_shard_directory": args.evaluation_shards,
        "promotion_merge_path": args.promotion_merge,
        "expected_promotion_merge_file_sha256": args.promotion_merge_sha256,
        "promotion_gate_path": args.promotion_gate,
        "expected_promotion_gate_file_sha256": args.promotion_gate_sha256,
        "checkpoint_bundle_directory": args.checkpoint_bundle,
        "expected_checkpoint_manifest_file_sha256": (
            args.checkpoint_manifest_sha256
        ),
        "training_run_config_path": args.training_run_config,
        "expected_training_run_config_file_sha256": (
            args.training_run_config_sha256
        ),
        "training_threshold_lock_path": args.training_threshold_lock,
        "expected_training_threshold_lock_file_sha256": (
            args.training_threshold_lock_sha256
        ),
        "compatibility_threshold_lock_path": (
            args.compatibility_threshold_lock
        ),
        "expected_compatibility_threshold_lock_file_sha256": (
            args.compatibility_threshold_lock_sha256
        ),
        "evaluation_runtime_closure_path": args.runtime_closure,
        "expected_evaluation_runtime_closure_file_sha256": (
            args.runtime_closure_sha256
        ),
        "policy_registry_path": args.policy_registry,
        "expected_policy_registry_file_sha256": args.policy_registry_sha256,
        "source_replay_root": args.source_replay_root,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3.1 create-only explicit opt-in registration"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-manifest")
    _add_manifest_arguments(create)
    create.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--manifest-sha256", required=True)
    validate.add_argument("--source-replay-root", required=True)
    patch = subparsers.add_parser("write-patch-spec")
    patch.add_argument("--manifest", required=True)
    patch.add_argument("--manifest-sha256", required=True)
    patch.add_argument("--source-replay-root", required=True)
    patch.add_argument("--policy-registry-patch-target", required=True)
    patch.add_argument("--output", required=True)
    before = subparsers.add_parser("write-before-receipt")
    before.add_argument("--manifest", required=True)
    before.add_argument("--manifest-sha256", required=True)
    before.add_argument("--patch-spec", required=True)
    before.add_argument("--patch-spec-sha256", required=True)
    before.add_argument("--source-replay-root", required=True)
    before.add_argument("--output", required=True)
    after = subparsers.add_parser("write-after-receipt")
    after.add_argument("--manifest", required=True)
    after.add_argument("--manifest-sha256", required=True)
    after.add_argument("--patch-spec", required=True)
    after.add_argument("--patch-spec-sha256", required=True)
    after.add_argument("--before-receipt", required=True)
    after.add_argument("--before-receipt-sha256", required=True)
    after.add_argument("--policy-registry-after", required=True)
    after.add_argument("--policy-registry-after-sha256", required=True)
    after.add_argument("--source-replay-root", required=True)
    after.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import torch

        if args.command == "create-manifest":
            result = write_registration_manifest(
                output_path=args.output,
                **_manifest_kwargs(args, torch),
            )
        elif args.command == "validate-manifest":
            manifest = validate_registration_manifest(
                args.manifest,
                expected_manifest_file_sha256=args.manifest_sha256,
                torch=torch,
                source_replay_root=args.source_replay_root,
            )
            result = {
                "schema": REGISTRATION_MANIFEST_SCHEMA,
                "status": "validated_source_replayed_dormant_opt_in",
                "manifest_file_sha256": args.manifest_sha256,
                "manifest_identity_sha256": manifest[
                    "manifest_identity_sha256"
                ],
                "registration_applied": False,
                "current_profile_changed": False,
                "runtime_activated": False,
            }
        elif args.command == "write-patch-spec":
            result = write_ai_profiles_patch_spec(
                output_path=args.output,
                registration_manifest_path=args.manifest,
                expected_registration_manifest_file_sha256=(
                    args.manifest_sha256
                ),
                policy_registry_patch_target_path=(
                    args.policy_registry_patch_target
                ),
                torch=torch,
                source_replay_root=args.source_replay_root,
            )
        elif args.command == "write-before-receipt":
            value = build_before_registration_receipt(
                registration_manifest_path=args.manifest,
                expected_registration_manifest_file_sha256=(
                    args.manifest_sha256
                ),
                patch_spec_path=args.patch_spec,
                expected_patch_spec_file_sha256=args.patch_spec_sha256,
                torch=torch,
                source_replay_root=args.source_replay_root,
            )
            target = _write_once(args.output, value)
            result = {
                "status": value["status"],
                "absolute_path": str(target),
                "file_sha256": sha256_file(target),
                "receipt_identity_sha256": value[
                    "receipt_identity_sha256"
                ],
            }
        else:
            value = build_after_registration_receipt(
                registration_manifest_path=args.manifest,
                expected_registration_manifest_file_sha256=(
                    args.manifest_sha256
                ),
                patch_spec_path=args.patch_spec,
                expected_patch_spec_file_sha256=args.patch_spec_sha256,
                before_receipt_path=args.before_receipt,
                expected_before_receipt_file_sha256=(
                    args.before_receipt_sha256
                ),
                policy_registry_after_path=args.policy_registry_after,
                expected_policy_registry_after_file_sha256=(
                    args.policy_registry_after_sha256
                ),
                torch=torch,
                source_replay_root=args.source_replay_root,
            )
            target = _write_once(args.output, value)
            result = {
                "status": value["status"],
                "absolute_path": str(target),
                "file_sha256": sha256_file(target),
                "receipt_identity_sha256": value[
                    "receipt_identity_sha256"
                ],
                "current_profile_changed": False,
                "runtime_activated": False,
            }
    except Exception as exc:  # pragma: no cover - subprocess boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_PROFILE_ID",
    "OptInRegistrationError",
    "PINNED_POLICY_REGISTRY_SHA256",
    "PROFILE_ID",
    "REGISTRATION_MANIFEST_SCHEMA",
    "REGISTRATION_PATCH_SPEC_SCHEMA",
    "REGISTRATION_TRANSITION_RECEIPT_SCHEMA",
    "build_after_registration_receipt",
    "build_ai_profiles_patch_spec",
    "build_before_registration_receipt",
    "build_registration_manifest",
    "canonical_bytes",
    "canonical_sha256",
    "main",
    "resolve_explicit_opt_in_t3_policy",
    "sha256_file",
    "validate_registration_manifest",
    "write_ai_profiles_patch_spec",
    "write_registration_manifest",
]
