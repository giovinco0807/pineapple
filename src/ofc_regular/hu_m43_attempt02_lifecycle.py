"""Fail-closed lifecycle validation for the M4.3 Attempt02 v4 artifact.

This module deliberately has no locked-JSONL reader.  Freeze validation may
inspect the sealed inherited-lock metadata, but opening that file belongs only
to the one-shot evaluator after it has claimed the canonical global marker.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m43_attempt02_contract import (
    M43_ATTEMPT02_CALIBRATION_PARTITION_SCHEMA,
    M43_ATTEMPT02_DATA_CONTRACT_SCHEMA,
    M43_ATTEMPT02_TEACHER_SHARDS_SCHEMA,
    _partition_calibration,
)
from .hu_m43_joint_model_v4 import (
    HU_M43_V4_CROSSFIT_SCHEMA,
    HU_M43_V4_MODEL_SCHEMA,
    HU_M43_V4_PRECAL_GATE_SCHEMA,
    HU_M43_V4_SAFETY_FIT_SCHEMA,
    HU_M43_V4_THRESHOLD_SCHEMA,
    HU_M43_V4_TRAINING_MANIFEST_SCHEMA,
)
from .hu_m43_pilot_contract import (
    _identity_digest,
    _ordered_shard_binding,
    _read_jsonl,
    _row_identity,
    canonical_manifest_sha256,
)


M43_ATTEMPT02_TRAINING_BINDING_SCHEMA = (
    "hu_m43_attempt02_training_data_binding_v1"
)
M43_ATTEMPT02_FREEZE_SCHEMA = "hu_m43_attempt02_model_threshold_freeze_v1"
M43_ATTEMPT02_FREEZE_STATUS = (
    "attempt02_v4_model_and_threshold_frozen_inherited_locked_unopened"
)
M43_ATTEMPT02_MARKER_SCHEMA = (
    "hu_m43_attempt02_locked_holdout_consumption_marker_v1"
)
M43_ATTEMPT02_LOCKED_RECEIPT_SCHEMA = (
    "hu_m43_attempt02_locked_holdout_receipt_v1"
)

MAX_FALSE_POSITIVE_RATE = 0.30
MAX_P95_LOSS = 25.0
MAX_P99_LOSS = 40.0
MAX_MAX_LOSS = 50.0

_FRESH_SPLITS = ("train", "calibration")
_ROLES = ("safety_fit", "threshold_lock")


def read_json_mapping(path: str | Path, location: str) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON mapping")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def self_digest(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return canonical_manifest_sha256(unsigned)


def write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Publish one lifecycle artifact with create-exclusive semantics."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partially written lifecycle artifact is intentionally not removed:
        # its path has been consumed and must be audited, not silently retried.
        raise


def resolve_contract_path(repo_root: str | Path, token: Any, location: str) -> Path:
    if not isinstance(token, str) or not token:
        raise ValueError(f"{location} path is missing")
    root = Path(repo_root).resolve()
    candidate = Path(token)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def build_fresh_bindings(
    *,
    repo_root: str | Path,
    train: Sequence[str | Path],
    calibration: Sequence[str | Path],
) -> dict[str, dict[str, Any]]:
    root = Path(repo_root).resolve()
    paths = {
        "train": tuple(Path(path).resolve() for path in train),
        "calibration": tuple(Path(path).resolve() for path in calibration),
    }
    if not paths["train"] or not paths["calibration"]:
        raise ValueError("Attempt02 freeze requires train and calibration shards")
    return {
        split: _ordered_shard_binding(
            paths[split], repo_root=root, expected_split=split
        )
        for split in _FRESH_SPLITS
    }


def build_fresh_identity_summaries(
    *,
    train: Sequence[str | Path],
    calibration: Sequence[str | Path],
) -> dict[str, dict[str, Any]]:
    paths = {"train": train, "calibration": calibration}
    result: dict[str, dict[str, Any]] = {}
    for split in _FRESH_SPLITS:
        identities: list[tuple[int, str]] = []
        profile_counts: dict[str, int] = {}
        for raw_path in paths[split]:
            source = Path(raw_path)
            for row in _read_jsonl(source):
                identities.append(_row_identity(row, str(source)))
                provenance = _mapping(row.get("provenance"), "fresh provenance")
                profile = str(provenance.get("root_profile", ""))
                if not profile:
                    raise ValueError("Attempt02 fresh row has no root profile")
                profile_counts[profile] = profile_counts.get(profile, 0) + 1
        if len(identities) != len(set(identities)):
            raise ValueError(f"Attempt02 {split} contains duplicate identities")
        result[split] = {
            "records": len(identities),
            "identity_sha256": _identity_digest(identities),
            "profile_counts": dict(sorted(profile_counts.items())),
        }
    return result


def build_calibration_role_binding(
    *,
    plan: Mapping[str, Any],
    calibration: Sequence[str | Path],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw_path in calibration:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Attempt02 calibration row is not a mapping: {path}:{line_number}"
                    )
                rows.append(row)
    return _partition_calibration(rows, plan)


def expected_training_data_binding(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    shards = _mapping(contract.get("teacher_shards"), "teacher_shards")
    split_bindings = _mapping(shards.get("splits"), "teacher_shards.splits")
    partition = _mapping(
        contract.get("calibration_partition"), "calibration_partition"
    )
    inherited = _mapping(contract.get("inherited_locked"), "inherited_locked")
    return {
        "schema": M43_ATTEMPT02_TRAINING_BINDING_SCHEMA,
        "contract_sha256": _sha256_text(
            contract.get("contract_sha256"), "contract_sha256"
        ),
        "plan_sha256": _sha256_text(contract.get("plan_sha256"), "plan_sha256"),
        "teacher_shards_all_fresh_splits_sha256": _sha256_text(
            contract.get("teacher_shards_all_fresh_splits_sha256"),
            "teacher_shards_all_fresh_splits_sha256",
        ),
        "fresh_splits": {
            split: {
                "records": _integer(split_bindings[split].get("records"), minimum=1),
                "ordered_shards_sha256": _sha256_text(
                    split_bindings[split].get("ordered_shards_sha256"),
                    f"{split}.ordered_shards_sha256",
                ),
                "identity_sha256": _sha256_text(
                    _mapping(contract["fresh_splits"][split], split).get(
                        "identity_sha256"
                    ),
                    f"{split}.identity_sha256",
                ),
            }
            for split in _FRESH_SPLITS
        },
        "calibration_roles": {
            role: {
                "records": _integer(partition[role].get("records"), minimum=1),
                "identity_sha256": _sha256_text(
                    partition[role].get("identity_sha256"),
                    f"calibration_partition.{role}.identity_sha256",
                ),
            }
            for role in _ROLES
        },
        "inherited_locked": {
            "identity_sha256": _sha256_text(
                inherited.get("identity_sha256"), "inherited_locked.identity_sha256"
            ),
            "content_opened": False,
            "used_for_training": False,
            "used_for_calibration": False,
        },
    }


def validate_attempt02_data_contract(
    contract: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    fresh_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    fresh_identity_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    calibration_role_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the sealed fresh-data contract without opening locked JSONL."""

    if contract.get("schema") != M43_ATTEMPT02_DATA_CONTRACT_SCHEMA:
        raise ValueError("unsupported Attempt02 data contract schema")
    if contract.get("status") != (
        "pass_fresh_train_calibration_sealed_inherited_locked_unopened"
    ):
        raise ValueError("Attempt02 data contract is not sealed/unopened")
    if contract.get("contract_sha256") != self_digest(contract, "contract_sha256"):
        raise ValueError("Attempt02 data contract self digest mismatch")
    expected_plan_sha = _sha256_text(plan_sha256, "plan SHA-256")
    if contract.get("plan_sha256") != expected_plan_sha:
        raise ValueError("Attempt02 data contract plan SHA mismatch")

    budget = _mapping(plan.get("budget"), "plan.budget")
    plan_splits = _mapping(plan.get("fresh_splits"), "plan.fresh_splits")
    fresh = _mapping(contract.get("fresh_splits"), "fresh_splits")
    if set(fresh) != set(_FRESH_SPLITS):
        raise ValueError("Attempt02 data contract fresh split set changed")
    for split in _FRESH_SPLITS:
        report = _mapping(fresh[split], f"fresh_splits.{split}")
        roots = _integer(plan_splits[split].get("roots"), minimum=1)
        if report.get("records") != roots:
            raise ValueError(f"Attempt02 {split} record count mismatch")
        expected_shards = roots // _integer(budget.get("roots_per_shard"), minimum=1)
        if report.get("shards") != expected_shards:
            raise ValueError(f"Attempt02 {split} shard count mismatch")
        if fresh_identity_summaries is not None:
            summary = _mapping(
                fresh_identity_summaries.get(split), f"actual {split} identities"
            )
            for field in ("records", "identity_sha256", "profile_counts"):
                if report.get(field) != summary.get(field):
                    raise ValueError(
                        f"Attempt02 {split} fresh identity binding mismatch: {field}"
                    )

    shards = _mapping(contract.get("teacher_shards"), "teacher_shards")
    if shards.get("schema") != M43_ATTEMPT02_TEACHER_SHARDS_SCHEMA:
        raise ValueError("Attempt02 teacher-shards schema mismatch")
    split_bindings = _mapping(shards.get("splits"), "teacher_shards.splits")
    if set(split_bindings) != set(_FRESH_SPLITS):
        raise ValueError("Attempt02 teacher-shards split set changed")
    unsigned_shards = {"schema": shards["schema"], "splits": split_bindings}
    all_fresh_sha = canonical_manifest_sha256(unsigned_shards)
    if shards.get("all_fresh_splits_sha256") != all_fresh_sha:
        raise ValueError("Attempt02 teacher-shards aggregate digest mismatch")
    if contract.get("teacher_shards_all_fresh_splits_sha256") != all_fresh_sha:
        raise ValueError("Attempt02 contract teacher-shards digest mismatch")
    for split in _FRESH_SPLITS:
        declared = _mapping(split_bindings[split], f"teacher_shards.{split}")
        if declared.get("records") != fresh[split].get("records"):
            raise ValueError(f"Attempt02 {split} binding count mismatch")
        if len(declared.get("ordered_shards", ())) != fresh[split].get("shards"):
            raise ValueError(f"Attempt02 {split} ordered shard count mismatch")
        if fresh_bindings is not None and dict(declared) != dict(
            _mapping(fresh_bindings.get(split), f"actual {split} binding")
        ):
            raise ValueError(f"Attempt02 {split} shard binding mismatch")

    partition = _mapping(
        contract.get("calibration_partition"), "calibration_partition"
    )
    if partition.get("schema") != M43_ATTEMPT02_CALIBRATION_PARTITION_SCHEMA:
        raise ValueError("Attempt02 calibration role schema mismatch")
    if partition.get("method") != "profile_stratified_identity_hash_v1":
        raise ValueError("Attempt02 calibration role method mismatch")
    if partition.get("overlap") != 0 or partition.get("inherited_locked_used") is not False:
        raise ValueError("Attempt02 calibration roles overlap or use locked data")
    role_total = 0
    role_identities: set[tuple[int, str]] = set()
    for role in _ROLES:
        declared = _mapping(partition.get(role), f"calibration_partition.{role}")
        expected_records = _integer(
            plan["calibration_partition"].get(f"{role}_roots"), minimum=1
        )
        if declared.get("records") != expected_records:
            raise ValueError(f"Attempt02 calibration {role} count mismatch")
        identities = declared.get("identities")
        if not isinstance(identities, Sequence) or isinstance(identities, (str, bytes)):
            raise ValueError(f"Attempt02 calibration {role} identities missing")
        normalized: list[tuple[int, str]] = []
        for raw in identities:
            row = _mapping(raw, f"calibration_partition.{role}.identity")
            normalized.append(
                (
                    _integer(row.get("hand_seed"), minimum=0),
                    _sha256_text(
                        row.get("observation_fingerprint"),
                        f"calibration_partition.{role}.fingerprint",
                    ),
                )
            )
        if len(normalized) != expected_records or len(set(normalized)) != expected_records:
            raise ValueError(f"Attempt02 calibration {role} identities are invalid")
        if declared.get("identity_sha256") != _identity_digest(normalized):
            raise ValueError(f"Attempt02 calibration {role} identity digest mismatch")
        if role_identities & set(normalized):
            raise ValueError("Attempt02 calibration role identities overlap")
        role_identities.update(normalized)
        role_total += len(normalized)
    if role_total != fresh["calibration"].get("records"):
        raise ValueError("Attempt02 calibration roles do not cover calibration")
    if calibration_role_binding is not None and dict(partition) != dict(
        calibration_role_binding
    ):
        raise ValueError("Attempt02 calibration role binding mismatch")

    freshness = _mapping(contract.get("freshness"), "freshness")
    if any(value != 0 for value in freshness.values()):
        raise ValueError("Attempt02 data freshness overlap is nonzero")
    inherited = _mapping(contract.get("inherited_locked"), "inherited_locked")
    plan_locked = _mapping(plan.get("inherited_locked"), "plan.inherited_locked")
    for key in (
        "classification",
        "path",
        "records",
        "bytes",
        "file_sha256",
        "canonical_rows_sha256",
        "identity_sha256",
        "ordered_shards_sha256",
    ):
        if inherited.get(key) != plan_locked.get(key):
            raise ValueError(f"Attempt02 inherited locked metadata mismatch: {key}")
    if inherited.get("classification") != "inherited_unopened":
        raise ValueError("Attempt02 locked classification is not unopened")
    if inherited.get("content_parse_count") != 0 or inherited.get(
        "model_evaluation_count"
    ) != 0:
        raise ValueError("Attempt02 locked content was already opened")
    global_lock = _mapping(
        contract.get("global_locked_consumption"), "global_locked_consumption"
    )
    if (
        global_lock.get("identity_sha256") != inherited.get("identity_sha256")
        or global_lock.get("canonical_marker_path")
        != plan_locked.get("global_consumption_marker")
        or global_lock.get("matching_marker_count") != 0
        or global_lock.get("status") != "unconsumed_preflight"
        or global_lock.get("claim_before_content_open_required") is not True
        or global_lock.get("claim_is_consuming_even_on_crash") is not True
    ):
        raise ValueError("Attempt02 global locked-consumption binding is invalid")
    runtime = _mapping(contract.get("runtime"), "runtime")
    if any(value is not False for value in runtime.values()):
        raise ValueError("Attempt02 contract changed a runtime/default guard")
    return expected_training_data_binding(contract)


def validate_v4_training_manifest(
    manifest: Mapping[str, Any],
    *,
    model_sha256: str,
    expected_binding: Mapping[str, Any],
    data_contract_file_sha256: str,
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> float:
    if manifest.get("schema") != HU_M43_V4_TRAINING_MANIFEST_SCHEMA:
        raise ValueError("unsupported Attempt02 v4 training manifest schema")
    if manifest.get("manifest_sha256") != self_digest(manifest, "manifest_sha256"):
        raise ValueError("Attempt02 v4 training manifest self digest mismatch")
    if manifest.get("model_schema") != HU_M43_V4_MODEL_SCHEMA:
        raise ValueError("Attempt02 training manifest model schema mismatch")
    if manifest.get("model_sha256") != _sha256_text(
        model_sha256, "model SHA-256"
    ):
        raise ValueError("Attempt02 training manifest model SHA mismatch")
    if manifest.get("promotion_status") != "candidate_ready_for_freeze":
        raise ValueError("Attempt02 v4 model is not a freeze candidate")
    if manifest.get("baseline_training_rows_included") is not False:
        raise ValueError("Attempt02 v4 training included baseline rows")
    if manifest.get("baseline_runtime_score_exact_zero") is not True:
        raise ValueError("Attempt02 v4 baseline runtime score is not exact zero")
    if manifest.get("all_action_state_balanced_oof_safety") is not True:
        raise ValueError("Attempt02 v4 all-action OOF safety is missing")
    if manifest.get("runtime_teacher_inputs") is not False:
        raise ValueError("Attempt02 v4 runtime uses teacher inputs")
    declared_contract = _mapping(
        manifest.get("attempt02_data_contract"), "training.attempt02_data_contract"
    )
    if dict(declared_contract) != {
        "schema": M43_ATTEMPT02_DATA_CONTRACT_SCHEMA,
        "file_sha256": _sha256_text(
            data_contract_file_sha256, "data contract file SHA-256"
        ),
        "contract_sha256": expected_binding["contract_sha256"],
    }:
        raise ValueError("Attempt02 v4 training data-contract binding mismatch")
    partition = _mapping(
        contract.get("calibration_partition"), "calibration_partition"
    )
    expected_calibration_binding = {
        "schema": "hu_m43_attempt02_v4_calibration_binding_v1",
        "data_contract_file_sha256": declared_contract["file_sha256"],
        "data_contract_sha256": expected_binding["contract_sha256"],
        "teacher_shards_all_fresh_splits_sha256": expected_binding[
            "teacher_shards_all_fresh_splits_sha256"
        ],
        "roles": {
            role: {
                "records": expected_binding["calibration_roles"][role]["records"],
                "identity_sha256": expected_binding["calibration_roles"][role][
                    "identity_sha256"
                ],
                "profile_counts": dict(
                    _mapping(partition[role], f"calibration_partition.{role}").get(
                        "profile_counts", {}
                    )
                ),
            }
            for role in _ROLES
        },
        "train_calibration_identity_overlap": 0,
        "inherited_holdout_input_accepted": False,
        "inherited_holdout_content_opened": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if manifest.get("calibration_binding") != expected_calibration_binding:
        raise ValueError("Attempt02 v4 training shard/role binding mismatch")

    crossfit = _mapping(manifest.get("crossfit"), "training.crossfit")
    if crossfit.get("schema") != HU_M43_V4_CROSSFIT_SCHEMA:
        raise ValueError("Attempt02 v4 crossfit schema mismatch")
    if crossfit.get("status") != "pass":
        raise ValueError("Attempt02 v4 crossfit did not pass")
    if crossfit.get("states") != plan["fresh_splits"]["train"]["roots"]:
        raise ValueError("Attempt02 v4 crossfit train-state count mismatch")
    if crossfit.get("folds") != 5 or crossfit.get(
        "exact_outer_inner_job_grid"
    ) is not True:
        raise ValueError("Attempt02 v4 nested crossfit grid mismatch")
    if crossfit.get("calibration_opened") is not False:
        raise ValueError("Attempt02 v4 opened calibration before precalibration gate")
    precal = _mapping(
        crossfit.get("precalibration_gate"), "training.precalibration_gate"
    )
    if (
        precal.get("schema") != HU_M43_V4_PRECAL_GATE_SCHEMA
        or precal.get("status") != "go"
        or precal.get("calibration_opened") is not False
        or precal.get("runtime_teacher_inputs") is not False
    ):
        raise ValueError("Attempt02 v4 precalibration gate is not Go")
    gates = _mapping(precal.get("gates"), "precalibration.gates")
    if not gates or any(value is not True for value in gates.values()):
        raise ValueError("Attempt02 v4 precalibration gate evidence is incomplete")

    safety = _mapping(manifest.get("safety_fit"), "training.safety_fit")
    if (
        safety.get("schema") != HU_M43_V4_SAFETY_FIT_SCHEMA
        or safety.get("status") != "fit_threshold_unselected"
        or safety.get("threshold_selected") is not False
        or safety.get("safety_enabled") is not False
        or safety.get("runtime_teacher_inputs") is not False
    ):
        raise ValueError("Attempt02 v4 safety-fit boundary is invalid")
    safety_sources = _mapping(safety.get("sources"), "safety_fit.sources")
    if safety_sources.get("locked_holdout") != {"used": False, "labels_opened": False}:
        raise ValueError("Attempt02 v4 safety fit used locked holdout")
    if safety_sources.get("calibration.threshold_lock") != {
        "used": False,
        "labels_opened": False,
    }:
        raise ValueError("Attempt02 v4 safety fit used threshold-lock labels")
    if _mapping(safety_sources.get("train_oof"), "train_oof").get("states") != (
        plan["fresh_splits"]["train"]["roots"]
    ):
        raise ValueError("Attempt02 v4 train-OOF safety source count mismatch")
    if _mapping(
        safety_sources.get("calibration.safety_fit"), "calibration.safety_fit"
    ).get("states") != plan["calibration_partition"]["safety_fit_roots"]:
        raise ValueError("Attempt02 v4 safety-fit role count mismatch")

    threshold = _mapping(
        manifest.get("threshold_selection"), "training.threshold_selection"
    )
    if (
        threshold.get("schema") != HU_M43_V4_THRESHOLD_SCHEMA
        or threshold.get("status") != "go"
        or threshold.get("source") != "fresh_calibration.threshold_lock_only"
        or threshold.get("safety_enabled") is not True
        or threshold.get("threshold_adaptation_after_selection") is not False
        or threshold.get("runtime_teacher_inputs") is not False
        or threshold.get("locked_holdout") != {"used": False, "labels_opened": False}
    ):
        raise ValueError("Attempt02 v4 threshold-lock boundary is invalid")
    if threshold.get("states") != plan["calibration_partition"][
        "threshold_lock_roots"
    ]:
        raise ValueError("Attempt02 v4 threshold-lock role count mismatch")
    if threshold.get("proposal_rows") != threshold.get("states"):
        raise ValueError("Attempt02 v4 threshold-lock proposal coverage mismatch")
    if threshold.get("identity_overlap_with_fit") != {
        "seed_values": 0,
        "observation_fingerprints": 0,
    }:
        raise ValueError("Attempt02 v4 threshold-lock identities overlap fit")
    constraints = _mapping(threshold.get("constraints"), "threshold.constraints")
    required_minimum = _integer(
        plan["calibration_partition"].get(
            "minimum_threshold_lock_fires_for_positive_pilot_signal"
        ),
        minimum=10,
    )
    expected_constraints = {
        "minimum_fires": required_minimum,
        "maximum_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
        "maximum_p95_loss": MAX_P95_LOSS,
        "maximum_p99_loss": MAX_P99_LOSS,
        "maximum_max_loss": MAX_MAX_LOSS,
    }
    for key, expected in expected_constraints.items():
        if not _same_number(constraints.get(key), expected):
            raise ValueError(f"Attempt02 v4 threshold constraint changed: {key}")
    selected = _mapping(threshold.get("selected_metrics"), "selected_metrics")
    if _integer(selected.get("fires"), minimum=0) < required_minimum:
        raise ValueError("Attempt02 v4 threshold has fewer than 10 fires")
    checks = (
        ("false_positive_rate", MAX_FALSE_POSITIVE_RATE),
        ("p95_loss", MAX_P95_LOSS),
        ("p99_loss", MAX_P99_LOSS),
        ("max_loss", MAX_MAX_LOSS),
    )
    for field, maximum in checks:
        value = _finite_number(selected.get(field), f"selected_metrics.{field}")
        if value < 0.0 or value > maximum:
            raise ValueError(f"Attempt02 v4 selected {field} exceeds its gate")
    if _finite_number(
        selected.get("teacher_mean_delta_per_fire"),
        "selected_metrics.teacher_mean_delta_per_fire",
    ) <= 0.0:
        raise ValueError("Attempt02 v4 selected threshold has non-positive delta")
    frozen_threshold = _finite_number(
        threshold.get("selected_threshold"), "selected_threshold"
    )
    if not 0.0 <= frozen_threshold <= 1.0 or not _same_number(
        selected.get("threshold"), frozen_threshold
    ):
        raise ValueError("Attempt02 v4 selected threshold is invalid")

    if manifest.get("locked_holdout") != {
        "status": "not_evaluated_pre_freeze",
        "labels_opened": False,
    }:
        raise ValueError("Attempt02 v4 training opened locked holdout")
    model_artifact = _mapping(
        manifest.get("model_artifact"), "training.model_artifact"
    )
    if (
        model_artifact.get("sha256") != model_sha256
        or model_artifact.get("model_id") != manifest.get("model_id")
        or model_artifact.get("safety_enabled") is not True
        or not _same_number(model_artifact.get("safety_threshold"), frozen_threshold)
    ):
        raise ValueError("Attempt02 v4 model-artifact binding mismatch")
    for field in (
        "inherited_holdout_input_accepted",
        "inherited_holdout_content_opened",
        "current_profile_mutated",
        "runtime_policy_activated",
        "full_replacement",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"Attempt02 v4 unsafe training flag: {field}")
    if manifest.get("calibration_opened_after_precalibration_go") is not True:
        raise ValueError("Attempt02 v4 calibration lifecycle declaration is missing")
    runtime = _mapping(manifest.get("runtime"), "training.runtime")
    if runtime != {
        "current_profile_mutated": False,
        "policy_activated": False,
        "full_replacement": False,
    }:
        raise ValueError("Attempt02 v4 training changed runtime/default state")
    return frozen_threshold


def validate_attempt02_freeze(
    freeze: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
) -> None:
    if freeze.get("schema") != M43_ATTEMPT02_FREEZE_SCHEMA:
        raise ValueError("unsupported Attempt02 freeze schema")
    if freeze.get("status") != M43_ATTEMPT02_FREEZE_STATUS:
        raise ValueError("Attempt02 freeze is not in frozen/unopened state")
    if freeze.get("freeze_sha256") != self_digest(freeze, "freeze_sha256"):
        raise ValueError("Attempt02 freeze self digest mismatch")
    for field in (
        "plan_sha256",
        "population_plan_file_sha256",
        "data_contract_sha256",
        "data_contract_file_sha256",
        "training_manifest_sha256",
        "model_sha256",
    ):
        _sha256_text(freeze.get(field), f"freeze.{field}")
    if not isinstance(freeze.get("population_plan_resolved_path"), str) or not freeze.get(
        "population_plan_resolved_path"
    ):
        raise ValueError("Attempt02 freeze population plan path is missing")
    threshold = _finite_number(freeze.get("frozen_threshold"), "frozen_threshold")
    if not 0.0 <= threshold <= 1.0 or freeze.get("safety_enabled") is not True:
        raise ValueError("Attempt02 freeze safety gate is not enabled")
    if not isinstance(freeze.get("model_id"), str) or not freeze.get("model_id"):
        raise ValueError("Attempt02 freeze model_id is missing")
    if freeze.get("model_schema") != HU_M43_V4_MODEL_SCHEMA:
        raise ValueError("Attempt02 freeze model schema mismatch")
    if freeze.get("precalibration_status") != "go" or freeze.get(
        "threshold_lock_status"
    ) != "go":
        raise ValueError("Attempt02 freeze did not preserve both Go gates")
    if freeze.get("minimum_threshold_lock_fires", 0) < 10:
        raise ValueError("Attempt02 freeze minimum-fire gate is too weak")
    if freeze.get("locked_content_access_count_at_freeze") != 0:
        raise ValueError("Attempt02 freeze reports locked content access")
    if freeze.get("locked_status") != "inherited_unopened":
        raise ValueError("Attempt02 freeze locked status changed")
    guards = _mapping(freeze.get("activation_guards"), "freeze.activation_guards")
    if any(value is not False for value in guards.values()):
        raise ValueError("Attempt02 freeze changed a runtime/default guard")
    if contract is not None:
        if freeze.get("data_contract_sha256") != contract.get("contract_sha256"):
            raise ValueError("Attempt02 freeze/contract digest mismatch")
        inherited = _mapping(contract.get("inherited_locked"), "inherited_locked")
        locked_binding = _mapping(
            freeze.get("inherited_locked_binding"), "freeze.inherited_locked_binding"
        )
        for field in (
            "path",
            "records",
            "bytes",
            "file_sha256",
            "canonical_rows_sha256",
            "identity_sha256",
            "ordered_shards_sha256",
        ):
            if locked_binding.get(field) != inherited.get(field):
                raise ValueError(f"Attempt02 freeze locked binding mismatch: {field}")
        global_lock = _mapping(
            contract.get("global_locked_consumption"), "global_locked_consumption"
        )
        if freeze.get("canonical_consumption_marker_resolved_path") is None:
            raise ValueError("Attempt02 freeze canonical marker path is missing")
        if freeze.get("canonical_consumption_marker_relative_path") != global_lock.get(
            "canonical_marker_path"
        ):
            raise ValueError("Attempt02 freeze canonical marker binding mismatch")


def validate_attempt02_locked_receipt(
    receipt: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    contract: Mapping[str, Any],
    marker: Mapping[str, Any],
) -> None:
    validate_attempt02_freeze(freeze, contract=contract)
    if receipt.get("schema") != M43_ATTEMPT02_LOCKED_RECEIPT_SCHEMA:
        raise ValueError("unsupported Attempt02 locked receipt schema")
    if receipt.get("status") != "evaluated_once_diagnostic_only_no_activation":
        raise ValueError("Attempt02 locked receipt status is invalid")
    if receipt.get("receipt_sha256") != self_digest(receipt, "receipt_sha256"):
        raise ValueError("Attempt02 locked receipt self digest mismatch")
    if marker.get("schema") != M43_ATTEMPT02_MARKER_SCHEMA:
        raise ValueError("Attempt02 consumption marker schema mismatch")
    if marker.get("status") != "claimed_before_inherited_locked_content_read":
        raise ValueError("Attempt02 consumption marker claim status mismatch")
    if marker.get("marker_sha256") != self_digest(marker, "marker_sha256"):
        raise ValueError("Attempt02 consumption marker self digest mismatch")
    inherited = _mapping(contract.get("inherited_locked"), "inherited_locked")
    expected = {
        "freeze_sha256": freeze.get("freeze_sha256"),
        "data_contract_sha256": contract.get("contract_sha256"),
        "population_plan_file_sha256": freeze.get(
            "population_plan_file_sha256"
        ),
        "model_sha256": freeze.get("model_sha256"),
        "locked_identity_sha256": inherited.get("identity_sha256"),
        "locked_file_sha256": inherited.get("file_sha256"),
        "locked_bytes": inherited.get("bytes"),
        "locked_ordered_shards_sha256": inherited.get("ordered_shards_sha256"),
        "evaluation_pass_count": 1,
    }
    for field, value in expected.items():
        if marker.get(field) != value or receipt.get(field) != value:
            raise ValueError(f"Attempt02 locked lifecycle binding mismatch: {field}")
    if not _same_number(
        receipt.get("frozen_threshold"), freeze.get("frozen_threshold")
    ):
        raise ValueError("Attempt02 locked receipt threshold mismatch")
    for field in (
        "threshold_search_performed",
        "threshold_reselection_performed",
        "model_selection_performed",
        "feature_selection_performed",
        "current_profile_resolved",
        "current_profile_changed",
        "runtime_policy_activated",
        "policy_promoted",
    ):
        if receipt.get(field) is not False:
            raise ValueError(f"Attempt02 locked receipt unsafe flag: {field}")
    if receipt.get("requires_fresh_population_acceptance") is not True:
        raise ValueError("Attempt02 locked receipt bypasses population acceptance")
    if _integer(receipt.get("minimum_population_valid_overrides"), minimum=300) < 300:
        raise ValueError("Attempt02 locked receipt population minimum is too small")
    diagnostic = _mapping(receipt.get("locked_diagnostic"), "locked_diagnostic")
    if diagnostic.get("threshold_search_performed") is not False:
        raise ValueError("Attempt02 locked diagnostic searched a threshold")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _integer(value: Any, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("expected an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"integer must be at least {minimum}")
    return value


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


def _same_number(value: Any, expected: float | int) -> bool:
    try:
        return math.isclose(
            _finite_number(value, "numeric value"),
            float(expected),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    except ValueError:
        return False


def _sha256_text(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{location} must be a SHA-256 string")
    return normalized


__all__ = [
    "M43_ATTEMPT02_FREEZE_SCHEMA",
    "M43_ATTEMPT02_FREEZE_STATUS",
    "M43_ATTEMPT02_LOCKED_RECEIPT_SCHEMA",
    "M43_ATTEMPT02_MARKER_SCHEMA",
    "M43_ATTEMPT02_TRAINING_BINDING_SCHEMA",
    "build_calibration_role_binding",
    "build_fresh_bindings",
    "build_fresh_identity_summaries",
    "expected_training_data_binding",
    "file_sha256",
    "read_json_mapping",
    "resolve_contract_path",
    "self_digest",
    "validate_attempt02_data_contract",
    "validate_attempt02_freeze",
    "validate_attempt02_locked_receipt",
    "validate_v4_training_manifest",
    "write_immutable_json",
]
