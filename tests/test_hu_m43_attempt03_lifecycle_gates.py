from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from ofc_regular import assemble_hu_m43_attempt03_model as assembly
from ofc_regular import hu_m43_attempt03_training as training
from ofc_regular.hu_m43_pilot_contract import (
    build_ordered_teacher_shard_binding,
    canonical_manifest_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _precal_receive_receipt(tmp_path: Path) -> tuple[Path, dict]:
    authorization = tmp_path / "authorization.json"
    claim = tmp_path / "open-claim.json"
    _write(authorization, b"authorization")
    _write(claim, b"claim")
    payload = {
        "schema": "hu_m43_attempt03_teacher_precal_receive_receipt_v1",
        "status": "verified_structural_precal_open_after_frozen_claim",
        "verified_shards": 20,
        "verified_roots": 200,
        "manifest_sha256": "1" * 64,
        "schedule_sha256": "2" * 64,
        "model_freeze_file_sha256": "3" * 64,
        "training_freeze_file_sha256": "4" * 64,
        "precal_open_authorization_path": str(authorization),
        "precal_open_authorization_file_sha256": _sha(authorization),
        "precal_open_authorization_sha256": "5" * 64,
        "candidate_model_sha256": "6" * 64,
        "fit_bundle_sha256": "7" * 64,
        "fit_manifest_file_sha256": "8" * 64,
        "fold_cloud_contract_file_sha256": "9" * 64,
        "open_claim_path": str(claim),
        "open_claim_file_sha256": _sha(claim),
        "data_contract_sha256": "a" * 64,
        "sealed_calibration_opened": False,
        "inherited_locked_opened": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    payload["receipt_sha256"] = canonical_manifest_sha256(payload)
    path = tmp_path / "one-shot-receipt.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, payload


def _precal_data_binding(root: Path) -> tuple[dict, dict, Path]:
    shard_paths: list[Path] = []
    for shard in range(20):
        path = root / "shards" / f"teacher-{shard:02d}.jsonl"
        rows = [
            {
                "split": "train",
                "hand_seed": shard * 10 + offset,
                "observation_fingerprint": f"{shard * 10 + offset + 1:064x}",
            }
            for offset in range(10)
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        shard_paths.append(path)
    binding = build_ordered_teacher_shard_binding(
        shard_paths, repo_root=root, expected_split="train"
    )
    teacher_shards = {
        "schema": "hu_m43_attempt03_ordered_teacher_shards_v1",
        "roles": {
            "train.fit": deepcopy(binding),
            "train.precal_holdout": deepcopy(binding),
        },
    }
    teacher_shards["all_fresh_roles_sha256"] = canonical_manifest_sha256(
        teacher_shards
    )
    contract = {
        "teacher_shards": teacher_shards,
        "teacher_shards_all_fresh_roles_sha256": teacher_shards[
            "all_fresh_roles_sha256"
        ],
        "precal_holdout": {
            "records": 200,
            "shards": 20,
            "identity_sha256": "b" * 64,
        },
    }
    merged = root / "fresh_precal_holdout.jsonl"
    merged.write_bytes(b"".join(path.read_bytes() for path in shard_paths))
    receive = {
        "fresh_precal_holdout": {
            "path": str(merged),
            "rows": 200,
            "sha256": _sha(merged),
            "shards": [
                {
                    "shard": 50 + index,
                    "logical_split": "train.precal_holdout",
                    "split_shard": index,
                    "rows": 10,
                    "sha256": binding["ordered_shards"][index][
                        "file_sha256"
                    ],
                    "path": str(path),
                }
                for index, path in enumerate(shard_paths)
            ],
        }
    }
    return receive, contract, merged


def test_precal_receive_receipt_self_hash_rejects_nested_tamper(
    tmp_path: Path,
) -> None:
    path, payload = _precal_receive_receipt(tmp_path)
    assert assembly._load_precal_receive_receipt(path)["receipt_sha256"] == (
        payload["receipt_sha256"]
    )
    payload["candidate_model_sha256"] = "f" * 64
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt digest changed"):
        assembly._load_precal_receive_receipt(path)


def test_precal_receive_rejects_shard_and_resigned_merged_content_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    receive, contract, merged = _precal_data_binding(root)
    raw_path, raw_sha, shard_paths, declared = (
        assembly._validate_precal_receive_data_binding(
            receive=receive, contract=contract, repo_root=root
        )
    )
    rows = assembly._validate_precal_content_binding(
        raw_path=raw_path,
        raw_sha256=raw_sha,
        shard_paths=shard_paths,
        declared_shard_binding=declared,
        repo_root=root,
    )
    assert len(rows) == 200

    shard_tamper = deepcopy(receive)
    shard_tamper["fresh_precal_holdout"]["shards"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="shard 0 binding changed"):
        assembly._validate_precal_receive_data_binding(
            receive=shard_tamper, contract=contract, repo_root=root
        )

    merged_rows = merged.read_text(encoding="utf-8").splitlines()
    first = json.loads(merged_rows[0])
    first["hand_seed"] += 1_000_000
    merged_rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    merged.write_text("\n".join(merged_rows) + "\n", encoding="utf-8")
    resigned = deepcopy(receive)
    resigned["fresh_precal_holdout"]["sha256"] = _sha(merged)
    raw_path, raw_sha, shard_paths, declared = (
        assembly._validate_precal_receive_data_binding(
            receive=resigned, contract=contract, repo_root=root
        )
    )
    with pytest.raises(ValueError, match="merged pre-cal canonical rows changed"):
        assembly._validate_precal_content_binding(
            raw_path=raw_path,
            raw_sha256=raw_sha,
            shard_paths=shard_paths,
            declared_shard_binding=declared,
            repo_root=root,
        )


def test_science_correction_is_hash_bound_through_runtime_population_chain() -> None:
    correction_path = ROOT / training.M43_ATTEMPT03_SCIENCE_CORRECTION_PATH
    correction = training.load_attempt03_training_science_correction(
        correction_path, repo_root=ROOT
    )
    assert correction["schema"] == training.M43_ATTEMPT03_SCIENCE_CORRECTION_SCHEMA
    assert training.M43_ATTEMPT03_SCIENCE_CORRECTION_PATH in (
        training.M43_ATTEMPT03_TRAINING_SOURCE_PATHS
    )

    model_start = (ROOT / "scripts/Start-GcpHuM43Attempt03ModelRun.ps1").read_text(
        encoding="utf-8"
    )
    model_receive = (
        ROOT / "scripts/Receive-GcpHuM43Attempt03ModelRun.ps1"
    ).read_text(encoding="utf-8")
    population_start = (
        ROOT / "scripts/Start-GcpHuM43Attempt03PopulationRun.ps1"
    ).read_text(encoding="utf-8")
    population_validator = (
        ROOT / "src/ofc_regular/validate_hu_m43_attempt03_population.py"
    ).read_text(encoding="utf-8")
    runtime = (ROOT / "src/ofc_regular/hu_m43_attempt03_runtime.py").read_text(
        encoding="utf-8"
    )
    for text in (model_start, model_receive):
        assert "training_freeze_file_sha256" in text
    assert '"--training-freeze", $resolvedTrainingFreeze' in model_start
    assert '"--training-freeze", $resolvedTrainingFreeze' in model_receive
    assert "training_pipeline_freeze" in runtime
    assert "training_freeze_file_sha256" in runtime
    assert "load_attempt03_training_freeze(" in population_validator
    assert "training_freeze_sha256" in population_start


def test_calibration_cli_requires_explicit_consumption_marker() -> None:
    base = [
        "calibrate",
        "--candidate-model",
        "candidate.pkl",
        "--fit-bundle",
        "bundle.pkl",
        "--fit-manifest",
        "fit.json",
        "--precalibration-receipt",
        "precal.json",
        "--attempt02-data-contract",
        "contract.json",
        "--attempt02-train",
        "train.jsonl",
        "--sealed-calibration",
        "MUST_NOT_TOUCH.jsonl",
        "--model-freeze",
        "model-freeze.json",
        "--training-freeze",
        "training-freeze.json",
        "--repo-root",
        ".",
        "--output-dir",
        "out",
    ]
    with pytest.raises(SystemExit):
        assembly.parse_args(base)
    args = assembly.parse_args(
        base
        + [
            "--calibration-consumption-marker",
            "M43_ATTEMPT03_SEALED_CALIBRATION_CONSUMED.json",
        ]
    )
    assert args.calibration_consumption_marker.endswith("CONSUMED.json")


def test_calibration_claim_precedes_any_sealed_loader_and_retry_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    candidate = root / "fit" / "candidate.pkl"
    bundle = root / "fit" / "bundle.pkl"
    fit_manifest_path = root / "fit" / "fit.json"
    model_freeze = root / "model-freeze.json"
    training_freeze = root / "training-freeze.json"
    for path, content in (
        (candidate, b"candidate"),
        (bundle, b"bundle"),
        (fit_manifest_path, b"fit-manifest"),
        (model_freeze, b"model-freeze"),
        (training_freeze, b"training-freeze"),
    ):
        _write(path, content)

    fit_manifest = {
        "model_sha256": _sha(candidate),
        "fit_bundle_sha256": _sha(bundle),
        "manifest_sha256": "1" * 64,
        "model_freeze_file_sha256": _sha(model_freeze),
        "training_freeze_file_sha256": _sha(training_freeze),
    }
    precal = {
        "schema": assembly.M43_ATTEMPT03_PRECAL_RECEIPT_SCHEMA,
        "status": "go_precalibration",
        "promotion_status": "eligible_to_open_sealed_calibration",
        "sealed_calibration_open_allowed": True,
        "sealed_calibration_opened": False,
        "inherited_locked_opened": False,
        "candidate_model_sha256": _sha(candidate),
        "fit_manifest_sha256": _sha(fit_manifest_path),
    }
    precal["receipt_sha256"] = canonical_manifest_sha256(precal)
    precal_path = root / "precal.json"
    precal_path.write_text(json.dumps(precal), encoding="utf-8")

    identity = "d" * 64
    declaration = {
        "data_contract_file_sha256": "2" * 64,
        "data_contract_sha256": "3" * 64,
        "identity_sha256": identity,
        "roles": {
            "safety_fit": {"records": 50, "identity_sha256": "4" * 64},
            "threshold_lock": {"records": 50, "identity_sha256": "5" * 64},
        },
    }
    marker = assembly.canonical_attempt03_calibration_marker_path(
        repo_root=root, calibration_identity_sha256=identity
    )
    sealed = root / "MUST_NOT_EXIST_OR_BE_TOUCHED.jsonl"

    monkeypatch.setattr(assembly, "load_attempt03_model_freeze", lambda *a, **k: {})
    monkeypatch.setattr(
        assembly, "load_attempt03_training_freeze", lambda *a, **k: {}
    )
    monkeypatch.setattr(assembly, "_load_fit_manifest", lambda _path: fit_manifest)
    monkeypatch.setattr(
        assembly.HuM43JointModelV5,
        "load",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(
        assembly, "load_attempt03_fit_bundle", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        assembly,
        "_load_attempt02_sealed_calibration_declaration",
        lambda _path: declaration,
    )

    loader_calls = 0

    def sealed_loader(**_kwargs):
        nonlocal loader_calls
        loader_calls += 1
        assert marker.exists(), "sealed loader ran before durable claim"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload["status"] == (
            "consumed_before_any_sealed_calibration_stat_hash_or_read"
        )
        assert payload["candidate_model_sha256"] == _sha(candidate)
        assert payload["fit_manifest_file_sha256"] == _sha(fit_manifest_path)
        assert payload["precalibration_receipt_sha256"] == precal["receipt_sha256"]
        assert payload["sealed_calibration_identity_sha256"] == identity
        assert payload["claim_is_consuming_even_on_crash"] is True
        raise RuntimeError("synthetic crash immediately after claim")

    monkeypatch.setattr(assembly, "load_attempt02_calibration_roles", sealed_loader)
    kwargs = {
        "candidate_model_path": candidate,
        "fit_bundle_path": bundle,
        "fit_manifest_path": fit_manifest_path,
        "precalibration_receipt_path": precal_path,
        "attempt02_data_contract_path": root / "contract.json",
        "attempt02_train_path": root / "train.jsonl",
        "sealed_calibration_path": sealed,
        "calibration_consumption_marker_path": marker,
        "model_freeze_path": model_freeze,
        "training_freeze_path": training_freeze,
        "repo_root": root,
        "output_dir": root / "calibrated",
    }
    with pytest.raises(RuntimeError, match="synthetic crash"):
        assembly.calibrate_attempt03_after_precal_go(**kwargs)
    assert loader_calls == 1
    assert marker.exists()
    assert not sealed.exists()

    # The crash is consuming.  A retry fails at CreateNew before the sealed
    # loader can be invoked for a second time.
    with pytest.raises(FileExistsError):
        assembly.calibrate_attempt03_after_precal_go(**kwargs)
    assert loader_calls == 1
    assert not sealed.exists()


def test_calibration_marker_rejects_alternate_path_for_same_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="canonical identity-bound path"):
        assembly.claim_attempt03_sealed_calibration(
            tmp_path / "alternate.json",
            repo_root=tmp_path,
            candidate_model_sha256="1" * 64,
            fit_bundle_sha256="2" * 64,
            fit_manifest_file_sha256="3" * 64,
            fit_manifest_canonical_sha256="4" * 64,
            precalibration_receipt_file_sha256="5" * 64,
            precalibration_receipt_sha256="6" * 64,
            attempt02_data_contract_file_sha256="7" * 64,
            attempt02_data_contract_sha256="8" * 64,
            calibration_identity_sha256="9" * 64,
            calibration_roles={},
            sealed_calibration_lexical_path="MUST_NOT_TOUCH.jsonl",
        )
