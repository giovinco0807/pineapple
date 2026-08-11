from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ofc_regular import assemble_hu_m43_attempt02_model as assembly
from ofc_regular.hu_m43_attempt02_contract import (
    M43_ATTEMPT02_CALIBRATION_PARTITION_SCHEMA,
    M43_ATTEMPT02_DATA_CONTRACT_SCHEMA,
    M43_ATTEMPT02_TEACHER_SHARDS_SCHEMA,
)
from ofc_regular.hu_m43_attempt02_fold_training import (
    Attempt02V4TrainingConfig,
)
from ofc_regular.hu_m43_joint_model_v4 import (
    HuM43JointModelV4,
    V4SafetyFitResult,
    V4ThresholdSelectionResult,
)
from ofc_regular.hu_m4_joint_model import (
    ConstantProbabilityEstimator,
    PairedDeltaRiskFoldEstimator,
)
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM


SHA_A = "a" * 64
SHA_B = "b" * 64


def _fingerprint(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_shards(tmp_path: Path):
    profiles = list(assembly._PROFILES)
    rows = {"train": [], "calibration": []}
    for index in range(200):
        rows["train"].append(
            {
                "split": "train",
                "hand_seed": 10_000 + index,
                "observation_fingerprint": _fingerprint(f"train-{index}"),
                "provenance": {"root_profile": profiles[(index // 40) % 5]},
            }
        )
    for profile_index, profile in enumerate(profiles):
        for within_profile in range(20):
            index = profile_index * 20 + within_profile
            rows["calibration"].append(
                {
                    "split": "calibration",
                    "hand_seed": 20_000 + index,
                    "observation_fingerprint": _fingerprint(
                        f"calibration-{index}"
                    ),
                    "provenance": {"root_profile": profile},
                }
            )
    paths = {"train": [], "calibration": []}
    for split, count in (("train", 20), ("calibration", 10)):
        for shard in range(count):
            path = tmp_path / f"{split}-{shard:02d}.jsonl"
            shard_rows = rows[split][shard * 10 : (shard + 1) * 10]
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in shard_rows),
                encoding="utf-8",
            )
            paths[split].append(path)
    return rows, paths


def _data_contract(tmp_path: Path, rows, paths) -> dict:
    teacher_shards = {
        "schema": M43_ATTEMPT02_TEACHER_SHARDS_SCHEMA,
        "splits": {
            split: assembly._ordered_shard_binding(
                paths[split],
                [rows[split][index * 10 : (index + 1) * 10] for index in range(len(paths[split]))],
                root=tmp_path,
                split=split,
            )
            for split in ("train", "calibration")
        },
    }
    teacher_shards["all_fresh_splits_sha256"] = (
        assembly.canonical_manifest_sha256(teacher_shards)
    )
    safety = []
    threshold = []
    for profile in assembly._PROFILES:
        profile_rows = [
            row
            for row in rows["calibration"]
            if row["provenance"]["root_profile"] == profile
        ]
        safety.extend(assembly._row_identity(row, "safety") for row in profile_rows[:10])
        threshold.extend(
            assembly._row_identity(row, "threshold") for row in profile_rows[10:]
        )

    def role(identities):
        ordered = sorted(identities)
        return {
            "records": 50,
            "identity_sha256": assembly._identity_digest(ordered),
            "profile_counts": {profile: 10 for profile in assembly._PROFILES},
            "identities": [
                {"hand_seed": seed, "observation_fingerprint": fingerprint}
                for seed, fingerprint in ordered
            ],
        }

    train_ids = {
        assembly._row_identity(row, "train") for row in rows["train"]
    }
    calibration_ids = {
        assembly._row_identity(row, "calibration")
        for row in rows["calibration"]
    }
    unsigned = {
        "schema": M43_ATTEMPT02_DATA_CONTRACT_SCHEMA,
        "status": "pass_fresh_train_calibration_sealed_inherited_locked_unopened",
        "plan_sha256": SHA_A,
        "preflight": {},
        "fresh_generation": {},
        "exclusion_union": {},
        "fresh_splits": {
            "train": {
                "records": 200,
                "shards": 20,
                "identity_sha256": assembly._identity_digest(train_ids),
                "profile_counts": {
                    profile: 40 for profile in assembly._PROFILES
                },
                "audited_paired_delta_records": 200,
            },
            "calibration": {
                "records": 100,
                "shards": 10,
                "identity_sha256": assembly._identity_digest(calibration_ids),
                "profile_counts": {
                    profile: 20 for profile in assembly._PROFILES
                },
                "audited_paired_delta_records": 100,
            },
        },
        "calibration_partition": {
            "schema": M43_ATTEMPT02_CALIBRATION_PARTITION_SCHEMA,
            "method": "profile_stratified_identity_hash_v1",
            "safety_fit": role(safety),
            "threshold_lock": role(threshold),
            "overlap": 0,
            "inherited_locked_used": False,
        },
        "teacher_shards": teacher_shards,
        "teacher_shards_all_fresh_splits_sha256": teacher_shards[
            "all_fresh_splits_sha256"
        ],
        "inherited_locked": {
            "classification": "inherited_unopened",
            "content_parse_count": 0,
            "model_evaluation_count": 0,
            "structural_byte_hash_audit_only": True,
        },
        "global_locked_consumption": {
            "matching_marker_count": 0,
            "status": "unconsumed_preflight",
        },
        "freshness": {
            "train_calibration_identity_overlap": 0,
            "exclusion_hand_seed_overlap": 0,
            "exclusion_observation_fingerprint_overlap": 0,
            "inherited_locked_hand_seed_overlap": 0,
            "inherited_locked_observation_fingerprint_overlap": 0,
        },
        "runtime": {
            "current_profile_resolved": False,
            "current_profile_changed": False,
            "policy_activated": False,
            "full_replacement": False,
            "large_scale_authorized": False,
        },
    }
    return {
        **unsigned,
        "contract_sha256": assembly.canonical_manifest_sha256(unsigned),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def test_calibration_roles_strictly_follow_attempt02_contract_50_50(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, paths = _write_shards(tmp_path)
    contract = _data_contract(tmp_path, rows, paths)
    contract_path = tmp_path / "data-contract.json"
    _write_json(contract_path, contract)
    monkeypatch.setattr(
        assembly,
        "prepare_teacher_samples",
        lambda raw: [
            SimpleNamespace(observation_fingerprint=row["observation_fingerprint"])
            for row in raw
        ],
    )

    roles = assembly.load_attempt02_calibration_roles(
        data_contract_path=contract_path,
        train_path=paths["train"],
        calibration_path=paths["calibration"],
        repo_root=tmp_path,
    )
    assert len(roles.safety_fit) == 50
    assert len(roles.threshold_lock) == 50
    assert roles.audit["inherited_holdout_input_accepted"] is False
    assert roles.audit["inherited_holdout_content_opened"] is False
    assert set(roles.audit["roles"]) == {"safety_fit", "threshold_lock"}

    tampered = deepcopy(contract)
    duplicate = tampered["calibration_partition"]["safety_fit"]["identities"][0]
    tampered["calibration_partition"]["safety_fit"]["identities"][1] = duplicate
    unsigned = dict(tampered)
    unsigned.pop("contract_sha256")
    tampered["contract_sha256"] = assembly.canonical_manifest_sha256(unsigned)
    _write_json(contract_path, tampered)
    with pytest.raises(ValueError, match="safety_fit identity binding"):
        assembly.load_attempt02_calibration_roles(
            data_contract_path=contract_path,
            train_path=paths["train"],
            calibration_path=paths["calibration"],
            repo_root=tmp_path,
        )


class _Provider:
    def __init__(self) -> None:
        self.training_samples = ()
        self.assembly_receipt = {
            "schema": "test-fold-assembly",
            "status": "verified_unconsumed",
        }
        self.complete = False

    def assert_complete(self) -> None:
        self.complete = True


def _cloud_contract() -> dict:
    config = Attempt02V4TrainingConfig()
    return {
        "contract_sha256": SHA_A,
        "training_config_sha256": SHA_B,
        "training_config": config.to_manifest(),
    }


def _crossfit_reports(status: str):
    config = Attempt02V4TrainingConfig()
    precalibration = {
        "status": status,
        "config_sha256": assembly.canonical_manifest_sha256(
            config.precalibration_gate.to_manifest()
        ),
        "metrics": {"states": 200},
        "calibration_opened": False,
    }
    report = {
        "schema": "test-crossfit",
        "status": "pass" if status == "go" else "no_go_precalibration",
        "states": 200,
        "folds": 5,
        "fold_jobs": 30,
        "exact_outer_inner_job_grid": True,
        "baseline_training_rows_included": False,
        "calibration_opened": False,
    }
    return precalibration, report


def test_precalibration_no_go_never_opens_contract_or_fresh_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _Provider()
    monkeypatch.setattr(
        assembly,
        "load_attempt02_fold_cloud_contract",
        lambda *_args, **_kwargs: _cloud_contract(),
    )
    monkeypatch.setattr(
        assembly,
        "load_attempt02_fold_artifact_provider",
        lambda **_kwargs: provider,
    )
    monkeypatch.setattr(
        assembly,
        "fit_v4_nested_crossfit",
        lambda *_args, **_kwargs: SimpleNamespace(
            precalibration_report=_crossfit_reports("no_go")[0],
            report=_crossfit_reports("no_go")[1],
        ),
    )
    model = tmp_path / "model.pkl"
    manifest = tmp_path / "decision.json"
    missing_calibration = tmp_path / "MUST_NOT_OPEN_CALIBRATION.jsonl"
    missing_contract = tmp_path / "MUST_NOT_OPEN_DATA_CONTRACT.json"

    result = assembly.assemble_attempt02_v4_from_fold_artifacts(
        fold_artifacts_dir=tmp_path / "folds",
        train_path=tmp_path / "train.jsonl",
        calibration_path=missing_calibration,
        attempt02_data_contract_path=missing_contract,
        repo_root=tmp_path,
        fold_cloud_contract_path=tmp_path / "cloud.json",
        output_model=model,
        manifest_output=manifest,
        run_name="attempt02",
        source_sha256=SHA_A,
        run_manifest_sha256=SHA_B,
    )
    assert provider.complete
    assert result["status"] == "no_go_precalibration"
    assert result["calibration"] == {
        "data_contract_opened": False,
        "fresh_rows_opened": False,
        "safety_fit_performed": False,
        "threshold_selection_performed": False,
    }
    assert result["model_written"] is False
    assert not model.exists()
    assert manifest.is_file()
    serialized = manifest.read_text(encoding="utf-8")
    assert str(missing_calibration) not in serialized
    assert str(missing_contract) not in serialized


def test_precalibration_go_opens_roles_then_runs_fixed_safety_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _Provider()
    precalibration, report = _crossfit_reports("go")
    crossfit = SimpleNamespace(
        precalibration_report=precalibration,
        report=report,
        model=object(),
        safety_dataset=object(),
    )
    calls = []
    monkeypatch.setattr(
        assembly,
        "load_attempt02_fold_cloud_contract",
        lambda *_args, **_kwargs: _cloud_contract(),
    )
    monkeypatch.setattr(
        assembly,
        "load_attempt02_fold_artifact_provider",
        lambda **_kwargs: provider,
    )
    monkeypatch.setattr(
        assembly, "fit_v4_nested_crossfit", lambda *_args, **_kwargs: crossfit
    )
    roles = assembly.Attempt02CalibrationRoles(
        safety_fit=(object(),), threshold_lock=(object(),), audit={"schema": "audit"}
    )

    def load_roles(**_kwargs):
        calls.append("roles")
        return roles

    def fit_safety(*_args, **_kwargs):
        calls.append("safety")
        return "safety-result"

    def select_threshold(*_args, **_kwargs):
        calls.append("threshold")
        return "threshold-result"

    def publish(**kwargs):
        calls.append("publish")
        assert kwargs["safety_fit"] == "safety-result"
        assert kwargs["threshold"] == "threshold-result"
        return {"status": "published-test"}

    monkeypatch.setattr(assembly, "load_attempt02_calibration_roles", load_roles)
    monkeypatch.setattr(assembly, "fit_v4_safety_calibrator", fit_safety)
    monkeypatch.setattr(assembly, "select_v4_threshold", select_threshold)
    monkeypatch.setattr(assembly, "_publish_model_and_manifest", publish)
    result = assembly.assemble_attempt02_v4_from_fold_artifacts(
        fold_artifacts_dir=tmp_path / "folds",
        train_path=tmp_path / "train.jsonl",
        calibration_path=tmp_path / "calibration.jsonl",
        attempt02_data_contract_path=tmp_path / "data-contract.json",
        repo_root=tmp_path,
        fold_cloud_contract_path=tmp_path / "cloud.json",
        output_model=tmp_path / "model.pkl",
        manifest_output=tmp_path / "manifest.json",
        run_name="attempt02",
        source_sha256=SHA_A,
        run_manifest_sha256=SHA_B,
    )
    assert result == {"status": "published-test"}
    assert calls == ["roles", "safety", "threshold", "publish"]


def test_assembler_cli_and_function_accept_no_inherited_holdout_argument() -> None:
    parameters = inspect.signature(
        assembly.assemble_attempt02_v4_from_fold_artifacts
    ).parameters
    assert "locked_holdout" not in parameters
    assert "inherited_locked" not in parameters
    valid = [
        "--fold-artifacts-dir",
        "folds",
        "--train",
        "train.jsonl",
        "--calibration",
        "calibration.jsonl",
        "--attempt02-data-contract",
        "contract.json",
        "--repo-root",
        ".",
        "--fold-cloud-contract",
        "cloud.json",
        "--output-model",
        "model.pkl",
        "--manifest-output",
        "manifest.json",
        "--run-name",
        "attempt02",
        "--source-sha256",
        SHA_A,
        "--run-manifest-sha256",
        SHA_B,
    ]
    args = assembly.parse_args(valid)
    assert not hasattr(args, "locked_holdout")
    assert not hasattr(args, "inherited_locked")
    with pytest.raises(SystemExit):
        assembly.parse_args([*valid, "--locked-holdout", "forbidden.jsonl"])


def test_publish_writes_hash_bound_opt_in_model_and_manifest(tmp_path: Path) -> None:
    head = ConstantProbabilityEstimator(0.5)
    folds = tuple(
        PairedDeltaRiskFoldEstimator(
            delta_estimator=head,
            positive_gain_estimator=head,
            downside_p95_estimator=head,
            downside_p99_estimator=head,
            downside_max_estimator=head,
            paired_feature_dim=4 * HU_FEATURE_DIM,
            fold_index=index,
        )
        for index in range(5)
    )
    base_model = HuM43JointModelV4(
        paired_fold_estimators=folds,
        model_id="attempt02-publish-test",
    )
    final_model = base_model.with_frozen_safety(
        ConstantProbabilityEstimator(0.8), threshold=0.5, enabled=True
    )
    safety_fit = V4SafetyFitResult(
        model=base_model.with_frozen_safety(
            ConstantProbabilityEstimator(0.8), threshold=1.0, enabled=False
        ),
        report={"status": "fit_threshold_unselected"},
        fit_seed_values=frozenset(),
        fit_observation_fingerprints=frozenset(),
    )
    threshold = V4ThresholdSelectionResult(
        model=final_model,
        report={"status": "go", "selected_threshold": 0.5},
    )
    model_path = tmp_path / "candidate.pkl"
    manifest_path = tmp_path / "training_manifest.json"
    result = assembly._publish_model_and_manifest(
        model_path=model_path,
        manifest_path=manifest_path,
        config=Attempt02V4TrainingConfig(),
        cloud_contract={
            "contract_sha256": SHA_A,
            "training_config_sha256": SHA_B,
        },
        assembly={"schema": "test-assembly", "status": "pass"},
        crossfit=SimpleNamespace(report={"schema": "test-crossfit", "status": "pass"}),
        safety_fit=safety_fit,
        threshold=threshold,
        calibration_audit={
            "schema": assembly.M43_ATTEMPT02_CALIBRATION_BINDING_SCHEMA,
            "data_contract_file_sha256": SHA_A,
            "data_contract_sha256": SHA_B,
        },
    )
    assert model_path.is_file()
    assert manifest_path.is_file()
    assert result["promotion_status"] == "candidate_ready_for_freeze"
    assert result["model_artifact"]["sha256"] == assembly._file_sha256(model_path)
    loaded = HuM43JointModelV4.load(
        model_path, expected_sha256=result["model_artifact"]["sha256"]
    )
    assert loaded.safety_enabled is True
    assert loaded.safety_threshold == 0.5
    unsigned = dict(result)
    declared = unsigned.pop("manifest_sha256")
    assert declared == assembly.canonical_manifest_sha256(unsigned)
    assert result["current_profile_mutated"] is False
    assert result["runtime_policy_activated"] is False
