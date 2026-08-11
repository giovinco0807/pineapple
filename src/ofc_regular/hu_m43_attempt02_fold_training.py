"""Train-only immutable fold artifacts for M4.3 Attempt02 v4.

This module is a deliberately narrower successor to the v3 distributed fold
path.  A cloud worker accepts only the 200 fresh train rows and a redacted
contract derived from those rows.  Fresh safety-fit/threshold-selection data,
the Attempt02 sealed data contract, and the inherited holdout are local-only
inputs and are not arguments to any function in the worker path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn

from .hu_m43_fold_training import (
    M43_FROZEN_DEPENDENCIES,
    M43_FROZEN_PROCESS_ENVIRONMENT,
)
from .hu_m43_joint_model_v4 import (
    HU_M43_V4_MODEL_SCHEMA,
    HU_M43_V4_PROPOSAL_SCHEMA,
    V4FoldWorkerConfig,
    V4PrecalibrationGateConfig,
    fit_v4_fold_worker_compatible,
)
from .hu_m43_pilot_contract import canonical_manifest_sha256
from .hu_m4_joint_model import PairedDeltaRiskFoldEstimator
from .train_hu_m4_joint_model import (
    M43FoldJobDefinition,
    M43FoldJobSpec,
    M43FoldTrainingPlan,
    PreparedTeacherSample,
    _m43_sample_identity_sha256,
    _normalize_input_paths,
    _sample_membership_hash,
    build_m43_fold_training_plan,
    prepare_teacher_samples,
    read_teacher_jsonl,
)


M43_ATTEMPT02_FOLD_CLOUD_CONTRACT_SCHEMA = (
    "hu_m43_attempt02_v4_fold_cloud_contract_v1"
)
M43_ATTEMPT02_FOLD_ESTIMATOR_SCHEMA = (
    "hu_m43_attempt02_v4_fold_estimator_artifact_v1"
)
M43_ATTEMPT02_FOLD_JOB_MANIFEST_SCHEMA = (
    "hu_m43_attempt02_v4_fold_job_manifest_v1"
)
M43_ATTEMPT02_FOLD_DONE_SCHEMA = "hu_m43_attempt02_v4_fold_done_v1"
M43_ATTEMPT02_FOLD_ASSEMBLY_SCHEMA = "hu_m43_attempt02_v4_fold_assembly_v1"
M43_ATTEMPT02_TRAINING_CONFIG_SCHEMA = "hu_m43_attempt02_v4_training_config_v1"

M43_ATTEMPT02_TRAIN_ROWS = 200
M43_ATTEMPT02_TRAIN_SHARDS = 20
M43_ATTEMPT02_ROWS_PER_SHARD = 10
M43_ATTEMPT02_FOLDS = 5
M43_ATTEMPT02_FOLD_JOBS = M43_ATTEMPT02_FOLDS * (M43_ATTEMPT02_FOLDS + 1)
M43_ATTEMPT02_FIXED_THRESHOLDS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.975,
    0.99,
    1.0,
)

_FORBIDDEN_WORKER_KEYS = frozenset(
    {
        "calibration",
        "calibration_path",
        "calibration_paths",
        "data_contract",
        "data_contract_path",
        "inherited_locked",
        "locked_holdout",
        "locked_holdout_path",
    }
)
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class Attempt02V4TrainingConfig:
    """Configuration frozen before any threshold-selection labels are read."""

    model_id: str = "hu-m43-t1-v4-attempt02"
    fold_seed: int = 2026071801
    worker: V4FoldWorkerConfig = field(default_factory=V4FoldWorkerConfig)
    precalibration_gate: V4PrecalibrationGateConfig = field(
        default_factory=V4PrecalibrationGateConfig
    )
    thresholds: tuple[float, ...] = M43_ATTEMPT02_FIXED_THRESHOLDS
    safety_calibrator_c: float = 0.25
    safety_seed: int = 2026071805
    minimum_fires: int = 10
    maximum_false_positive_rate: float = 0.30
    maximum_p95_loss: float = 25.0
    maximum_p99_loss: float = 40.0
    maximum_max_loss: float = 50.0
    schema: str = M43_ATTEMPT02_TRAINING_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != M43_ATTEMPT02_TRAINING_CONFIG_SCHEMA:
            raise ValueError("unsupported Attempt02 v4 training config schema")
        if not self.model_id:
            raise ValueError("Attempt02 v4 model_id must be non-empty")
        if self.fold_seed != 2026071801:
            raise ValueError("Attempt02 v4 fold seed changed")
        if self.precalibration_gate.to_manifest() != (
            V4PrecalibrationGateConfig().to_manifest()
        ):
            raise ValueError("Attempt02 v4 pre-calibration gate is not frozen")
        normalized = tuple(sorted({float(value) for value in self.thresholds}))
        if normalized != M43_ATTEMPT02_FIXED_THRESHOLDS:
            raise ValueError("Attempt02 v4 threshold grid changed")
        if (
            not math.isfinite(self.safety_calibrator_c)
            or self.safety_calibrator_c <= 0.0
        ):
            raise ValueError("Attempt02 v4 safety_calibrator_c must be positive")
        if isinstance(self.safety_seed, bool) or self.safety_seed < 0:
            raise ValueError("Attempt02 v4 safety seed is invalid")
        if self.minimum_fires != 10:
            raise ValueError("Attempt02 v4 minimum threshold-lock fires changed")
        expected_limits = (0.30, 25.0, 40.0, 50.0)
        actual_limits = (
            float(self.maximum_false_positive_rate),
            float(self.maximum_p95_loss),
            float(self.maximum_p99_loss),
            float(self.maximum_max_loss),
        )
        if actual_limits != expected_limits:
            raise ValueError("Attempt02 v4 fixed FP/tail constraints changed")

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "model_schema": HU_M43_V4_MODEL_SCHEMA,
            "proposal_schema": HU_M43_V4_PROPOSAL_SCHEMA,
            "model_id": self.model_id,
            "fold_seed": int(self.fold_seed),
            "train_states": M43_ATTEMPT02_TRAIN_ROWS,
            "cross_fit_folds": M43_ATTEMPT02_FOLDS,
            "fold_jobs": M43_ATTEMPT02_FOLD_JOBS,
            "worker": self.worker.to_manifest(),
            "precalibration_gate": self.precalibration_gate.to_manifest(),
            "precalibration_gate_sha256": canonical_manifest_sha256(
                self.precalibration_gate.to_manifest()
            ),
            "safety_fit": {
                "estimator": "standardized_l2_logistic_low_capacity",
                "c": float(self.safety_calibrator_c),
                "seed": int(self.safety_seed),
                "sources": ["train_oof", "fresh_safety_fit_50"],
            },
            "threshold_selection": {
                "source": "fresh_threshold_selection_50_only",
                "thresholds": [float(value) for value in self.thresholds],
                "minimum_fires": int(self.minimum_fires),
                "maximum_false_positive_rate": float(
                    self.maximum_false_positive_rate
                ),
                "maximum_p95_loss": float(self.maximum_p95_loss),
                "maximum_p99_loss": float(self.maximum_p99_loss),
                "maximum_max_loss": float(self.maximum_max_loss),
            },
            "runtime": {
                "current_profile_mutated": False,
                "policy_activated": False,
                "full_replacement": False,
            },
        }
        return payload

    @classmethod
    def from_manifest(cls, value: Any) -> "Attempt02V4TrainingConfig":
        payload = _mapping(value, "training_config")
        expected_keys = set(cls().to_manifest())
        if set(payload) != expected_keys:
            raise ValueError("Attempt02 v4 training config key set changed")
        worker_payload = _mapping(payload.get("worker"), "training_config.worker")
        worker_keys = {
            "schema",
            "paired_se_floor",
            "huber_alpha",
            "iterations",
            "max_leaf_nodes",
            "learning_rate",
            "baseline_training_rows_included",
            "proposal_score",
            "downside_objective",
        }
        if set(worker_payload) != worker_keys:
            raise ValueError("Attempt02 v4 worker config key set changed")
        worker = V4FoldWorkerConfig(
            paired_se_floor=float(worker_payload["paired_se_floor"]),
            huber_alpha=float(worker_payload["huber_alpha"]),
            iterations=_integer(worker_payload["iterations"], "iterations", minimum=1),
            max_leaf_nodes=_integer(
                worker_payload["max_leaf_nodes"], "max_leaf_nodes", minimum=2
            ),
            learning_rate=float(worker_payload["learning_rate"]),
            schema=str(worker_payload["schema"]),
        )
        if dict(worker_payload) != worker.to_manifest():
            raise ValueError("Attempt02 v4 worker config semantic fields changed")
        gate_payload = dict(
            _mapping(
                payload.get("precalibration_gate"),
                "training_config.precalibration_gate",
            )
        )
        gate = V4PrecalibrationGateConfig(**gate_payload)
        safety = _mapping(payload.get("safety_fit"), "training_config.safety_fit")
        threshold = _mapping(
            payload.get("threshold_selection"),
            "training_config.threshold_selection",
        )
        result = cls(
            model_id=str(payload.get("model_id", "")),
            fold_seed=_integer(payload.get("fold_seed"), "fold_seed", minimum=0),
            worker=worker,
            precalibration_gate=gate,
            thresholds=tuple(float(value) for value in threshold.get("thresholds", ())),
            safety_calibrator_c=float(safety.get("c")),
            safety_seed=_integer(safety.get("seed"), "safety seed", minimum=0),
            minimum_fires=_integer(
                threshold.get("minimum_fires"), "minimum_fires", minimum=1
            ),
            maximum_false_positive_rate=float(
                threshold.get("maximum_false_positive_rate")
            ),
            maximum_p95_loss=float(threshold.get("maximum_p95_loss")),
            maximum_p99_loss=float(threshold.get("maximum_p99_loss")),
            maximum_max_loss=float(threshold.get("maximum_max_loss")),
            schema=str(payload.get("schema", "")),
        )
        if dict(payload) != result.to_manifest():
            raise ValueError("Attempt02 v4 training config changed")
        return result


def build_attempt02_fold_cloud_contract(
    *,
    train_path: str | Path | Sequence[str | Path],
    config: Attempt02V4TrainingConfig = Attempt02V4TrainingConfig(),
) -> dict[str, Any]:
    """Build a redacted contract containing fresh train provenance only."""

    paths, rows_by_path, samples, plan = _read_train_and_plan(
        train_path, config=config
    )
    inputs = [
        _input_entry(path, rows, index=index)
        for index, (path, rows) in enumerate(
            zip(paths, rows_by_path, strict=True)
        )
    ]
    fold_plan = {
        "outer_folds": M43_ATTEMPT02_FOLDS,
        "inner_folds_per_outer": M43_ATTEMPT02_FOLDS,
        "total_jobs": M43_ATTEMPT02_FOLD_JOBS,
        "train_identity_sha256": _m43_sample_identity_sha256(
            plan.ordered_samples
        ),
        "jobs": [
            {**job.spec.to_manifest(), "job_spec_sha256": job.spec.sha256}
            for job in plan.jobs
        ],
    }
    fold_plan["fold_plan_sha256"] = canonical_manifest_sha256(fold_plan)
    training_config = config.to_manifest()
    unsigned = {
        "schema": M43_ATTEMPT02_FOLD_CLOUD_CONTRACT_SCHEMA,
        "status": "frozen_train_only",
        "inputs": {"train": inputs},
        "input_bundle_sha256": canonical_manifest_sha256({"train": inputs}),
        "fold_plan": fold_plan,
        "training_config": training_config,
        "training_config_sha256": canonical_manifest_sha256(training_config),
        "dependencies": dict(M43_FROZEN_DEPENDENCIES),
        "process_environment": dict(M43_FROZEN_PROCESS_ENVIRONMENT),
        "worker_input_boundary": {
            "fresh_train_only": True,
            "train_rows": len(samples),
            "nontrain_input_count": 0,
            "sealed_contract_content_count": 0,
        },
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _reject_worker_sensitive(unsigned)
    return {**unsigned, "contract_sha256": canonical_manifest_sha256(unsigned)}


def write_attempt02_fold_cloud_contract(
    path: str | Path,
    *,
    train_path: str | Path | Sequence[str | Path],
    config: Attempt02V4TrainingConfig = Attempt02V4TrainingConfig(),
) -> dict[str, Any]:
    payload = build_attempt02_fold_cloud_contract(
        train_path=train_path, config=config
    )
    _write_json_exclusive(Path(path), payload)
    return payload


def load_attempt02_fold_cloud_contract(
    path: str | Path, *, verify_process_environment: bool = False
) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Attempt02 fold cloud contract must be an object")
    expected_keys = {
        "schema",
        "status",
        "inputs",
        "input_bundle_sha256",
        "fold_plan",
        "training_config",
        "training_config_sha256",
        "dependencies",
        "process_environment",
        "worker_input_boundary",
        "current_profile_mutated",
        "runtime_policy_activated",
        "contract_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("Attempt02 fold cloud contract key set changed")
    unsigned = dict(payload)
    declared = _require_sha256(unsigned.pop("contract_sha256", None), "contract")
    if canonical_manifest_sha256(unsigned) != declared:
        raise ValueError("Attempt02 fold cloud contract digest mismatch")
    if (
        payload.get("schema") != M43_ATTEMPT02_FOLD_CLOUD_CONTRACT_SCHEMA
        or payload.get("status") != "frozen_train_only"
        or payload.get("current_profile_mutated") is not False
        or payload.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt02 fold cloud lifecycle flags changed")
    _reject_worker_sensitive(payload)
    inputs = _mapping(payload.get("inputs"), "inputs")
    if set(inputs) != {"train"}:
        raise ValueError("Attempt02 worker contract accepts fresh train only")
    entries = _sequence(inputs.get("train"), "inputs.train")
    if len(entries) != M43_ATTEMPT02_TRAIN_SHARDS:
        raise ValueError("Attempt02 worker requires exact train shard coverage")
    for index, raw in enumerate(entries):
        entry = _mapping(raw, f"inputs.train[{index}]")
        if set(entry) != {"index", "sha256", "bytes", "rows"}:
            raise ValueError("Attempt02 train input entry key set changed")
        if (
            _integer(entry.get("index"), "train input index") != index
            or _integer(entry.get("rows"), "train input rows")
            != M43_ATTEMPT02_ROWS_PER_SHARD
            or _integer(entry.get("bytes"), "train input bytes", minimum=1) < 1
        ):
            raise ValueError("Attempt02 train shard shape changed")
        _require_sha256(entry.get("sha256"), "train input SHA-256")
    if payload.get("input_bundle_sha256") != canonical_manifest_sha256(
        {"train": list(entries)}
    ):
        raise ValueError("Attempt02 train input bundle digest mismatch")
    config = Attempt02V4TrainingConfig.from_manifest(
        payload.get("training_config")
    )
    if payload.get("training_config_sha256") != canonical_manifest_sha256(
        config.to_manifest()
    ):
        raise ValueError("Attempt02 training config digest mismatch")
    _validate_fold_plan(payload.get("fold_plan"))
    if dict(_mapping(payload.get("dependencies"), "dependencies")) != dict(
        M43_FROZEN_DEPENDENCIES
    ):
        raise ValueError("Attempt02 frozen dependencies changed")
    actual_dependencies = {
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if actual_dependencies != M43_FROZEN_DEPENDENCIES:
        raise RuntimeError("Attempt02 runtime dependency versions changed")
    if dict(
        _mapping(payload.get("process_environment"), "process_environment")
    ) != dict(M43_FROZEN_PROCESS_ENVIRONMENT):
        raise ValueError("Attempt02 frozen process environment changed")
    boundary = _mapping(payload.get("worker_input_boundary"), "worker boundary")
    if boundary != {
        "fresh_train_only": True,
        "train_rows": M43_ATTEMPT02_TRAIN_ROWS,
        "nontrain_input_count": 0,
        "sealed_contract_content_count": 0,
    }:
        raise ValueError("Attempt02 worker input boundary changed")
    if verify_process_environment:
        _verify_process_environment()
    return payload


def run_attempt02_fold_job(
    *,
    job_index: int,
    train_path: str | Path | Sequence[str | Path],
    fold_cloud_contract_path: str | Path,
    output_dir: str | Path,
    run_name: str,
    source_sha256: str,
    run_manifest_sha256: str,
    input_bundle_sha256: str,
    job_spec_sha256: str,
) -> dict[str, Any]:
    """Fit one v4 fold from fresh train only and publish DONE last."""

    index = _integer(job_index, "job_index", minimum=0)
    if index >= M43_ATTEMPT02_FOLD_JOBS:
        raise ValueError("Attempt02 job index is outside exact 30-job grid")
    _validate_run_name(run_name)
    source_sha = _require_sha256(source_sha256, "source_sha256")
    run_manifest_sha = _require_sha256(
        run_manifest_sha256, "run_manifest_sha256"
    )
    input_bundle_sha = _require_sha256(
        input_bundle_sha256, "input_bundle_sha256"
    )
    expected_spec_sha = _require_sha256(job_spec_sha256, "job_spec_sha256")
    contract_path = Path(fold_cloud_contract_path).resolve()
    contract = load_attempt02_fold_cloud_contract(
        contract_path, verify_process_environment=True
    )
    contract_file_sha = _file_sha256(contract_path)
    if input_bundle_sha != contract.get("input_bundle_sha256"):
        raise ValueError("Attempt02 runner input bundle disagrees with contract")
    config = Attempt02V4TrainingConfig.from_manifest(
        contract.get("training_config")
    )
    samples, plan = _load_train_plan(
        contract, train_path=train_path, config=config
    )
    definition = plan.jobs[index]
    if definition.spec.sha256 != expected_spec_sha:
        raise ValueError("Attempt02 runner job spec digest changed")
    destination = Path(output_dir)
    artifact_path, manifest_path, done_path = _job_files(destination)
    if done_path.exists():
        return _validate_existing_done(
            destination,
            definition=definition,
            contract_file_sha256=contract_file_sha,
            training_config_sha256=str(contract["training_config_sha256"]),
            run_name=run_name,
            source_sha256=source_sha,
            run_manifest_sha256=run_manifest_sha,
            input_bundle_sha256=input_bundle_sha,
        )
    if artifact_path.exists() or manifest_path.exists():
        raise FileExistsError("Attempt02 partial fold output exists without DONE")
    estimator = fit_v4_fold_worker_compatible(
        definition.spec,
        definition.fit_samples,
        config=config.worker,
    )
    artifact_payload = {
        "schema": M43_ATTEMPT02_FOLD_ESTIMATOR_SCHEMA,
        "model_schema": HU_M43_V4_MODEL_SCHEMA,
        "proposal_schema": HU_M43_V4_PROPOSAL_SCHEMA,
        "training_config_sha256": contract["training_config_sha256"],
        "job_spec_sha256": definition.spec.sha256,
        "estimator": estimator,
    }
    _write_pickle_exclusive(artifact_path, artifact_payload)
    artifact_sha = _file_sha256(artifact_path)
    manifest = _job_manifest(
        definition.spec,
        run_name=run_name,
        source_sha256=source_sha,
        run_manifest_sha256=run_manifest_sha,
        contract_file_sha256=contract_file_sha,
        input_bundle_sha256=input_bundle_sha,
        training_config_sha256=str(contract["training_config_sha256"]),
        artifact_sha256=artifact_sha,
    )
    _reject_worker_sensitive(manifest)
    _write_json_exclusive(manifest_path, manifest)
    done = {
        key: value
        for key, value in manifest.items()
        if key not in {"job_spec", "schema", "status"}
    }
    done.update(
        {
            "schema": M43_ATTEMPT02_FOLD_DONE_SCHEMA,
            "status": "complete",
            "job_manifest_sha256": _file_sha256(manifest_path),
        }
    )
    _reject_worker_sensitive(done)
    _write_json_exclusive(done_path, done)
    return done


@dataclass
class Attempt02FoldArtifactProvider:
    estimators: dict[int, PairedDeltaRiskFoldEstimator]
    expected_specs: dict[int, M43FoldJobSpec]
    training_samples: tuple[PreparedTeacherSample, ...]
    assembly_receipt: dict[str, Any]
    consumed: set[int]

    def __call__(
        self,
        spec: M43FoldJobSpec,
        fit_samples: Sequence[PreparedTeacherSample],
    ) -> PairedDeltaRiskFoldEstimator:
        if spec.job_index in self.consumed:
            raise ValueError(f"Attempt02 fold consumed twice: {spec.job_index}")
        expected = self.expected_specs.get(spec.job_index)
        if expected is None or expected.to_manifest() != spec.to_manifest():
            raise ValueError("Attempt02 assembler job spec changed")
        if _m43_sample_identity_sha256(fit_samples) != spec.fit_identity_sha256:
            raise ValueError("Attempt02 assembler fit identities changed")
        estimator = self.estimators.get(spec.job_index)
        if estimator is None:
            raise ValueError(f"Attempt02 fold estimator missing: {spec.job_index}")
        self.consumed.add(spec.job_index)
        return estimator

    def assert_complete(self) -> None:
        expected = set(range(M43_ATTEMPT02_FOLD_JOBS))
        if self.consumed != expected:
            raise ValueError(
                "Attempt02 assembler did not consume exact 30-job coverage"
            )


def load_attempt02_fold_artifact_provider(
    *,
    artifacts_dir: str | Path,
    fold_cloud_contract_path: str | Path,
    train_path: str | Path | Sequence[str | Path],
    expected_run_name: str,
    expected_source_sha256: str,
    expected_run_manifest_sha256: str,
) -> Attempt02FoldArtifactProvider:
    """Load and bind all 30 artifacts without accepting non-train inputs."""

    _validate_run_name(expected_run_name)
    expected_source = _require_sha256(
        expected_source_sha256, "expected_source_sha256"
    )
    expected_run_manifest = _require_sha256(
        expected_run_manifest_sha256, "expected_run_manifest_sha256"
    )
    contract_path = Path(fold_cloud_contract_path).resolve()
    contract = load_attempt02_fold_cloud_contract(contract_path)
    config = Attempt02V4TrainingConfig.from_manifest(
        contract.get("training_config")
    )
    samples, plan = _load_train_plan(
        contract, train_path=train_path, config=config
    )
    root = Path(artifacts_dir).resolve()
    expected_names = {
        f"job-{index:02d}" for index in range(M43_ATTEMPT02_FOLD_JOBS)
    }
    actual_names = {path.name for path in root.glob("job-*") if path.is_dir()}
    if actual_names != expected_names:
        raise ValueError("Attempt02 fold artifact directory coverage mismatch")
    contract_file_sha = _file_sha256(contract_path)
    estimators: dict[int, PairedDeltaRiskFoldEstimator] = {}
    receipts: list[dict[str, Any]] = []
    for definition in plan.jobs:
        estimator, receipt = _load_job_artifact(
            root / f"job-{definition.spec.job_index:02d}",
            definition=definition,
            contract_file_sha256=contract_file_sha,
            input_bundle_sha256=str(contract["input_bundle_sha256"]),
            training_config_sha256=str(contract["training_config_sha256"]),
        )
        if (
            receipt["run_name"] != expected_run_name
            or receipt["source_sha256"] != expected_source
            or receipt["run_manifest_sha256"] != expected_run_manifest
        ):
            raise ValueError("Attempt02 fold artifact immutable run lineage mismatch")
        estimators[definition.spec.job_index] = estimator
        receipts.append(receipt)
    assembly = {
        "schema": M43_ATTEMPT02_FOLD_ASSEMBLY_SCHEMA,
        "status": "verified_unconsumed",
        "job_count": len(receipts),
        "run_name": expected_run_name,
        "source_sha256": expected_source,
        "run_manifest_sha256": expected_run_manifest,
        "cloud_contract_file_sha256": contract_file_sha,
        "cloud_contract_sha256": contract["contract_sha256"],
        "training_config_sha256": contract["training_config_sha256"],
        "input_bundle_sha256": contract["input_bundle_sha256"],
        "jobs": receipts,
        "exact_outer_inner_coverage": True,
        "fresh_train_only": True,
        "nontrain_input_count": 0,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    return Attempt02FoldArtifactProvider(
        estimators=estimators,
        expected_specs={job.spec.job_index: job.spec for job in plan.jobs},
        training_samples=tuple(samples),
        assembly_receipt=assembly,
        consumed=set(),
    )


def _read_train_and_plan(
    train_path: str | Path | Sequence[str | Path],
    *,
    config: Attempt02V4TrainingConfig,
) -> tuple[
    tuple[Path, ...],
    list[list[dict[str, Any]]],
    list[PreparedTeacherSample],
    M43FoldTrainingPlan,
]:
    paths = _normalize_input_paths(train_path, split="train")
    if len(paths) != M43_ATTEMPT02_TRAIN_SHARDS:
        raise ValueError("Attempt02 v4 requires exactly 20 fresh train shards")
    rows_by_path = [read_teacher_jsonl(path) for path in paths]
    if any(len(rows) != M43_ATTEMPT02_ROWS_PER_SHARD for rows in rows_by_path):
        raise ValueError("Attempt02 v4 fresh train shards must contain 10 rows")
    rows = [row for shard in rows_by_path for row in shard]
    if len(rows) != M43_ATTEMPT02_TRAIN_ROWS:
        raise ValueError("Attempt02 v4 requires exactly 200 fresh train rows")
    if any(row.get("split") not in (None, "train") for row in rows):
        raise ValueError("Attempt02 v4 worker received a non-train row")
    samples = prepare_teacher_samples(rows)
    memberships = [_sample_membership_hash(sample) for sample in samples]
    if len(set(memberships)) != len(samples):
        raise ValueError("Attempt02 v4 fresh train membership is duplicated")
    if len({sample.observation_fingerprint for sample in samples}) != len(samples):
        raise ValueError("Attempt02 v4 fresh train fingerprint is duplicated")
    plan = build_m43_fold_training_plan(
        samples,
        cross_fit_folds=M43_ATTEMPT02_FOLDS,
        seed=config.fold_seed,
    )
    return paths, rows_by_path, samples, plan


def _load_train_plan(
    contract: Mapping[str, Any],
    *,
    train_path: str | Path | Sequence[str | Path],
    config: Attempt02V4TrainingConfig,
) -> tuple[list[PreparedTeacherSample], M43FoldTrainingPlan]:
    paths, rows_by_path, samples, plan = _read_train_and_plan(
        train_path, config=config
    )
    actual_inputs = [
        _input_entry(path, rows, index=index)
        for index, (path, rows) in enumerate(zip(paths, rows_by_path, strict=True))
    ]
    declared_inputs = _mapping(contract.get("inputs"), "inputs").get("train")
    if actual_inputs != declared_inputs:
        raise ValueError("Attempt02 fresh train bytes/rows disagree with contract")
    if canonical_manifest_sha256({"train": actual_inputs}) != contract.get(
        "input_bundle_sha256"
    ):
        raise ValueError("Attempt02 fresh train bundle digest changed")
    expected_plan = _mapping(contract.get("fold_plan"), "fold_plan")
    actual_jobs = [
        {**job.spec.to_manifest(), "job_spec_sha256": job.spec.sha256}
        for job in plan.jobs
    ]
    if (
        expected_plan.get("train_identity_sha256")
        != _m43_sample_identity_sha256(plan.ordered_samples)
        or expected_plan.get("jobs") != actual_jobs
    ):
        raise ValueError("Attempt02 recomputed train fold plan changed")
    return samples, plan


def _load_job_artifact(
    job_dir: Path,
    *,
    definition: M43FoldJobDefinition,
    contract_file_sha256: str,
    input_bundle_sha256: str,
    training_config_sha256: str,
) -> tuple[PairedDeltaRiskFoldEstimator, dict[str, Any]]:
    artifact_path, manifest_path, done_path = _job_files(job_dir)
    if not all(path.is_file() for path in (artifact_path, manifest_path, done_path)):
        raise ValueError("Attempt02 fold artifact set is incomplete")
    manifest = _read_mapping(manifest_path, "job manifest")
    done = _read_mapping(done_path, "job DONE")
    spec = definition.spec
    artifact_sha = _file_sha256(artifact_path)
    expected_manifest = _job_manifest(
        spec,
        run_name=str(manifest.get("run_name", "")),
        source_sha256=str(manifest.get("source_sha256", "")),
        run_manifest_sha256=str(manifest.get("run_manifest_sha256", "")),
        contract_file_sha256=contract_file_sha256,
        input_bundle_sha256=input_bundle_sha256,
        training_config_sha256=training_config_sha256,
        artifact_sha256=artifact_sha,
    )
    if manifest != expected_manifest:
        raise ValueError(f"Attempt02 fold job hash/spec chain invalid: {spec.job_index}")
    expected_done = {
        key: value
        for key, value in expected_manifest.items()
        if key not in {"job_spec", "schema", "status"}
    }
    expected_done.update(
        {
            "schema": M43_ATTEMPT02_FOLD_DONE_SCHEMA,
            "status": "complete",
            "job_manifest_sha256": _file_sha256(manifest_path),
        }
    )
    if done != expected_done:
        raise ValueError(f"Attempt02 fold DONE hash chain invalid: {spec.job_index}")
    payload = pickle.loads(artifact_path.read_bytes())
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "model_schema",
        "proposal_schema",
        "training_config_sha256",
        "job_spec_sha256",
        "estimator",
    }:
        raise ValueError("Attempt02 fold estimator artifact header changed")
    if (
        payload.get("schema") != M43_ATTEMPT02_FOLD_ESTIMATOR_SCHEMA
        or payload.get("model_schema") != HU_M43_V4_MODEL_SCHEMA
        or payload.get("proposal_schema") != HU_M43_V4_PROPOSAL_SCHEMA
        or payload.get("training_config_sha256") != training_config_sha256
        or payload.get("job_spec_sha256") != spec.sha256
    ):
        raise ValueError("Attempt02 fold estimator artifact binding changed")
    estimator = payload.get("estimator")
    if not isinstance(estimator, PairedDeltaRiskFoldEstimator):
        raise TypeError("Attempt02 fold estimator type mismatch")
    estimator.__post_init__()
    if estimator.fold_index != spec.estimator_fold_index:
        raise ValueError("Attempt02 fold estimator index mismatch")
    return estimator, {
        "job_index": spec.job_index,
        "artifact_sha256": artifact_sha,
        "job_manifest_sha256": _file_sha256(manifest_path),
        "job_spec_sha256": spec.sha256,
        "run_name": manifest["run_name"],
        "source_sha256": manifest["source_sha256"],
        "run_manifest_sha256": manifest["run_manifest_sha256"],
    }


def _validate_existing_done(
    output_dir: Path,
    *,
    definition: M43FoldJobDefinition,
    contract_file_sha256: str,
    training_config_sha256: str,
    run_name: str,
    source_sha256: str,
    run_manifest_sha256: str,
    input_bundle_sha256: str,
) -> dict[str, Any]:
    artifact_path, manifest_path, done_path = _job_files(output_dir)
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise ValueError("Attempt02 completed fold is incomplete")
    artifact_sha = _file_sha256(artifact_path)
    expected_manifest = _job_manifest(
        definition.spec,
        run_name=run_name,
        source_sha256=source_sha256,
        run_manifest_sha256=run_manifest_sha256,
        contract_file_sha256=contract_file_sha256,
        input_bundle_sha256=input_bundle_sha256,
        training_config_sha256=training_config_sha256,
        artifact_sha256=artifact_sha,
    )
    if _read_mapping(manifest_path, "job manifest") != expected_manifest:
        raise ValueError("Attempt02 existing fold manifest lineage changed")
    done = _read_mapping(done_path, "job DONE")
    expected_done = {
        key: value
        for key, value in expected_manifest.items()
        if key not in {"job_spec", "schema", "status"}
    }
    expected_done.update(
        {
            "schema": M43_ATTEMPT02_FOLD_DONE_SCHEMA,
            "status": "complete",
            "job_manifest_sha256": _file_sha256(manifest_path),
        }
    )
    if done != expected_done:
        raise ValueError("Attempt02 existing fold DONE lineage changed")
    return done


def _job_manifest(
    spec: M43FoldJobSpec,
    *,
    run_name: str,
    source_sha256: str,
    run_manifest_sha256: str,
    contract_file_sha256: str,
    input_bundle_sha256: str,
    training_config_sha256: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    _validate_run_name(run_name)
    values = {
        "source_sha256": source_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "cloud_contract_file_sha256": contract_file_sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "training_config_sha256": training_config_sha256,
        "artifact_sha256": artifact_sha256,
    }
    for label, value in values.items():
        values[label] = _require_sha256(value, label)
    return {
        "schema": M43_ATTEMPT02_FOLD_JOB_MANIFEST_SCHEMA,
        "status": "pass",
        "run_name": run_name,
        "job_index": spec.job_index,
        "job_kind": spec.kind,
        "outer_fold": spec.outer_fold,
        "inner_fold": spec.inner_fold,
        "job_spec": spec.to_manifest(),
        "job_spec_sha256": spec.sha256,
        **values,
        "fresh_train_only": True,
        "nontrain_input_count": 0,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def _validate_fold_plan(value: Any) -> None:
    plan = _mapping(value, "fold_plan")
    if set(plan) != {
        "outer_folds",
        "inner_folds_per_outer",
        "total_jobs",
        "train_identity_sha256",
        "jobs",
        "fold_plan_sha256",
    }:
        raise ValueError("Attempt02 fold plan key set changed")
    unsigned = dict(plan)
    declared = _require_sha256(
        unsigned.pop("fold_plan_sha256", None), "fold plan SHA-256"
    )
    if canonical_manifest_sha256(unsigned) != declared:
        raise ValueError("Attempt02 fold plan digest mismatch")
    if (
        _integer(plan.get("outer_folds"), "outer_folds") != M43_ATTEMPT02_FOLDS
        or _integer(plan.get("inner_folds_per_outer"), "inner_folds")
        != M43_ATTEMPT02_FOLDS
        or _integer(plan.get("total_jobs"), "total_jobs")
        != M43_ATTEMPT02_FOLD_JOBS
    ):
        raise ValueError("Attempt02 fold plan is not exact 5x(1+5)")
    _require_sha256(plan.get("train_identity_sha256"), "train identity SHA-256")
    jobs = _sequence(plan.get("jobs"), "fold_plan.jobs")
    if len(jobs) != M43_ATTEMPT02_FOLD_JOBS:
        raise ValueError("Attempt02 fold plan does not contain exact 30 jobs")
    for index, raw in enumerate(jobs):
        job = _mapping(raw, f"fold job {index}")
        spec_payload = dict(job)
        digest = _require_sha256(
            spec_payload.pop("job_spec_sha256", None), "job spec SHA-256"
        )
        if canonical_manifest_sha256(spec_payload) != digest:
            raise ValueError(f"Attempt02 fold job digest invalid: {index}")
        if _integer(spec_payload.get("job_index"), "job index") != index:
            raise ValueError("Attempt02 fold job order changed")


def _input_entry(
    path: Path, rows: Sequence[Mapping[str, Any]], *, index: int
) -> dict[str, Any]:
    return {
        "index": int(index),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
    }


def _verify_process_environment() -> None:
    actual = {key: os.environ.get(key) for key in M43_FROZEN_PROCESS_ENVIRONMENT}
    if actual != M43_FROZEN_PROCESS_ENVIRONMENT:
        raise RuntimeError("Attempt02 worker process environment is not frozen")


def _reject_worker_sensitive(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_WORKER_KEYS:
                raise ValueError("Attempt02 worker contract contains a forbidden key")
            _reject_worker_sensitive(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_worker_sensitive(child)


def _validate_run_name(value: str) -> None:
    if not value or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in value
    ):
        raise ValueError("Attempt02 run_name contains unsafe characters")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _job_files(root: Path) -> tuple[Path, Path, Path]:
    return root / "estimator.pkl", root / "job_manifest.json", root / "DONE.json"


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_pickle_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        pickle.dump(dict(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "Attempt02FoldArtifactProvider",
    "Attempt02V4TrainingConfig",
    "M43_ATTEMPT02_FIXED_THRESHOLDS",
    "M43_ATTEMPT02_FOLD_ASSEMBLY_SCHEMA",
    "M43_ATTEMPT02_FOLD_CLOUD_CONTRACT_SCHEMA",
    "M43_ATTEMPT02_FOLD_DONE_SCHEMA",
    "M43_ATTEMPT02_FOLD_ESTIMATOR_SCHEMA",
    "M43_ATTEMPT02_FOLD_JOB_MANIFEST_SCHEMA",
    "M43_ATTEMPT02_FOLD_JOBS",
    "build_attempt02_fold_cloud_contract",
    "load_attempt02_fold_artifact_provider",
    "load_attempt02_fold_cloud_contract",
    "run_attempt02_fold_job",
    "write_attempt02_fold_cloud_contract",
]
