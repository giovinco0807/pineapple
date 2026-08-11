"""Production orchestration for the M3.1 StreetPolicyNetV1 trainer.

The numerical trainer deliberately exposes small, composable functions.  This
module closes the operational contract around them:

* the 360-shard dataset and an externally frozen run config are replayed;
* CPU or CUDA execution is selected explicitly and frozen in a run contract;
* every epoch is a write-once checkpoint plus fit receipt, atomically renamed;
* interrupted runs resume only through a complete, source-replayed chain;
* calibration and diagnostic holdout are immutable, non-promoting artifacts.

Nothing in this module registers or changes an AI profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_dataset_contract_v1 as dataset_contract
from . import hu_m31_t3_street_policy_training_v1 as training
from .street_policy_net_v1 import (
    FEATURE_SCHEMA_HASH,
    LOSS_SCHEMA_HASH,
    StreetPolicyNetV1Config,
    model_state_sha256,
)


RUN_CONFIG_SCHEMA = "hu_m31_t3_street_policy_run_config_v1"
RUN_CONTRACT_SCHEMA = "hu_m31_t3_street_policy_run_contract_v1"
EPOCH_RECEIPT_SCHEMA = "hu_m31_t3_street_policy_epoch_receipt_v1"
CALIBRATION_RECEIPT_SCHEMA = "hu_m31_t3_street_policy_calibration_receipt_v1"
FINAL_RECEIPT_SCHEMA = "hu_m31_t3_street_policy_run_receipt_v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _read_canonical(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or _is_link_or_junction(path):
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise ValueError(f"{label} is not a canonical JSON object")
    return value, raw


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"write-once artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_directory(
    destination: Path,
    writer: Any,
) -> None:
    if destination.exists() or _is_link_or_junction(destination):
        raise FileExistsError(f"write-once directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists() or _is_link_or_junction(temporary):
        raise FileExistsError(f"stale temporary directory exists: {temporary}")
    temporary.mkdir()
    try:
        writer(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@dataclass(frozen=True)
class StreetPolicyRunConfig:
    training_config: training.StreetPolicyTrainingConfig
    model_config: StreetPolicyNetV1Config

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": RUN_CONFIG_SCHEMA,
            "training_config": self.training_config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "loss_schema_hash": LOSS_SCHEMA_HASH,
            "teacher_values_are_realized_match_ev": False,
            "profile_activation_authorized": False,
            "current_profile_changed": False,
        }
        return payload

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def default_run_config() -> StreetPolicyRunConfig:
    return StreetPolicyRunConfig(
        training_config=training.StreetPolicyTrainingConfig(),
        model_config=StreetPolicyNetV1Config(),
    )


def write_default_run_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    config = default_run_config()
    value = config.to_dict()
    _write_once(target, value)
    return {
        "path": str(target.resolve()),
        "file_sha256": _sha256_bytes(target.read_bytes()),
        "config_identity_sha256": config.identity_sha256,
    }


def load_run_config(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> StreetPolicyRunConfig:
    if not _is_sha256(expected_file_sha256):
        raise ValueError("expected config SHA-256 must be a lowercase digest")
    value, raw = _read_canonical(Path(path), "StreetPolicyNetV1 run config")
    if _sha256_bytes(raw) != expected_file_sha256:
        raise ValueError("StreetPolicyNetV1 run config file SHA-256 changed")
    required = {
        "schema",
        "training_config",
        "model_config",
        "feature_schema_hash",
        "loss_schema_hash",
        "teacher_values_are_realized_match_ev",
        "profile_activation_authorized",
        "current_profile_changed",
    }
    if set(value) != required:
        raise ValueError("StreetPolicyNetV1 run config fields changed")
    raw_training = value["training_config"]
    raw_model = value["model_config"]
    if not isinstance(raw_training, dict) or set(raw_training) != {
        "schema",
        *asdict(training.StreetPolicyTrainingConfig()).keys(),
    }:
        raise ValueError("StreetPolicyNetV1 training config fields changed")
    if raw_training["schema"] != training.TRAINING_CONFIG_SCHEMA:
        raise ValueError("StreetPolicyNetV1 training config schema changed")
    if not isinstance(raw_model, dict) or set(raw_model) != set(
        asdict(StreetPolicyNetV1Config())
    ):
        raise ValueError("StreetPolicyNetV1 model config fields changed")
    config = StreetPolicyRunConfig(
        training_config=training.StreetPolicyTrainingConfig(
            **{
                key: raw_training[key]
                for key in asdict(training.StreetPolicyTrainingConfig())
            }
        ),
        model_config=StreetPolicyNetV1Config(**raw_model),
    )
    if (
        value != config.to_dict()
        or value["feature_schema_hash"] != FEATURE_SCHEMA_HASH
        or value["loss_schema_hash"] != LOSS_SCHEMA_HASH
        or value["teacher_values_are_realized_match_ev"] is not False
        or value["profile_activation_authorized"] is not False
        or value["current_profile_changed"] is not False
    ):
        raise ValueError("StreetPolicyNetV1 run config contract changed")
    return config


def _resolve_device(torch: Any, requested: str) -> tuple[Any, dict[str, Any]]:
    if requested == "auto":
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        requested_device = requested
    try:
        device = torch.device(requested_device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("invalid StreetPolicyNetV1 device") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("StreetPolicyNetV1 device must be cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError("requested CUDA device index is unavailable")
        device = torch.device(f"cuda:{index}")
        capability = list(torch.cuda.get_device_capability(index))
        device_name = str(torch.cuda.get_device_name(index))
    else:
        device = torch.device("cpu")
        capability = None
        device_name = "cpu"
    # PyTorch requires this before the first cuBLAS operation whenever
    # deterministic algorithms are enabled.  Set it even for CPU training:
    # the CLI's deterministic flag is process-global and a later CUDA
    # validation in the same process must remain reproducible as well.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    contract = {
        "requested": requested,
        "resolved": str(device),
        "device_type": device.type,
        "device_name": device_name,
        "cuda_capability": capability,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            str(torch.version.cuda) if torch.version.cuda is not None else None
        ),
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    return device, contract


def _validate_fit_receipt(
    receipt: Mapping[str, Any],
    *,
    dataset: training.PolicyTrainingDataset,
    config: training.StreetPolicyTrainingConfig,
    stage: str,
    epoch: int,
    before_hashes: Sequence[str],
    after_hashes: Sequence[str],
) -> None:
    identity = dict(receipt)
    declared = identity.pop("receipt_sha256", None)
    split = "train" if stage == "core" else "safety-fit"
    if (
        set(receipt)
        != {
            "schema",
            "split_role",
            "update_scope",
            "training_view_identity_sha256",
            "training_config_sha256",
            "start_epoch",
            "end_epoch",
            "example_count",
            "model_state_sha256_before",
            "model_state_sha256_after",
            "losses",
            "unauthorized_parameter_change_count",
            "teacher_values_are_realized_match_ev",
            "current_profile_changed",
            "receipt_sha256",
        }
        or receipt["schema"] != training.FIT_RECEIPT_SCHEMA
        or declared != _canonical_sha256(identity)
        or receipt["split_role"] != split
        or receipt["update_scope"] != stage
        or receipt["training_view_identity_sha256"] != dataset.identity_sha256
        or receipt["training_config_sha256"] != config.identity_sha256
        or receipt["start_epoch"] != epoch - 1
        or receipt["end_epoch"] != epoch
        or receipt["example_count"] != len(dataset.for_split(split))
        or receipt["model_state_sha256_before"] != list(before_hashes)
        or receipt["model_state_sha256_after"] != list(after_hashes)
        or receipt["unauthorized_parameter_change_count"] != 0
        or receipt["teacher_values_are_realized_match_ev"] is not False
        or receipt["current_profile_changed"] is not False
    ):
        raise ValueError(f"{stage} epoch {epoch} fit receipt changed")


def _epoch_receipt(
    *,
    stage: str,
    epoch: int,
    dataset: training.PolicyTrainingDataset,
    run_config: StreetPolicyRunConfig,
    run_contract_sha256: str,
    fit_receipt: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema": EPOCH_RECEIPT_SCHEMA,
        "stage": stage,
        "completed_epoch": epoch,
        "training_view_identity_sha256": dataset.identity_sha256,
        "run_config_identity_sha256": run_config.identity_sha256,
        "run_contract_sha256": run_contract_sha256,
        "fit_receipt_sha256": fit_receipt["receipt_sha256"],
        "checkpoint_bundle_identity_sha256": bundle["bundle_identity_sha256"],
        "model_state_sha256_before": fit_receipt["model_state_sha256_before"],
        "model_state_sha256_after": fit_receipt["model_state_sha256_after"],
        "teacher_values_are_realized_match_ev": False,
        "profile_activation_authorized": False,
        "current_profile_changed": False,
    }
    result = dict(identity)
    result["epoch_receipt_sha256"] = _canonical_sha256(identity)
    return result


def _validate_epoch_directory(
    path: Path,
    *,
    torch: Any,
    device: Any,
    dataset: training.PolicyTrainingDataset,
    run_config: StreetPolicyRunConfig,
    run_contract_sha256: str,
    stage: str,
    epoch: int,
    expected_before_hashes: Sequence[str],
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    expected_children = {"bundle", "fit_receipt.json", "epoch_receipt.json"}
    if (
        not path.is_dir()
        or _is_link_or_junction(path)
        or {child.name for child in path.iterdir()} != expected_children
        or any(_is_link_or_junction(child) for child in path.iterdir())
    ):
        raise ValueError(f"{stage} epoch {epoch} directory changed")
    epoch_receipt, _ = _read_canonical(
        path / "epoch_receipt.json", f"{stage} epoch receipt"
    )
    required = {
        "schema",
        "stage",
        "completed_epoch",
        "training_view_identity_sha256",
        "run_config_identity_sha256",
        "run_contract_sha256",
        "fit_receipt_sha256",
        "checkpoint_bundle_identity_sha256",
        "model_state_sha256_before",
        "model_state_sha256_after",
        "teacher_values_are_realized_match_ev",
        "profile_activation_authorized",
        "current_profile_changed",
        "epoch_receipt_sha256",
    }
    identity = dict(epoch_receipt)
    declared = identity.pop("epoch_receipt_sha256", None)
    if (
        set(epoch_receipt) != required
        or epoch_receipt["schema"] != EPOCH_RECEIPT_SCHEMA
        or declared != _canonical_sha256(identity)
        or epoch_receipt["stage"] != stage
        or epoch_receipt["completed_epoch"] != epoch
        or epoch_receipt["training_view_identity_sha256"] != dataset.identity_sha256
        or epoch_receipt["run_config_identity_sha256"] != run_config.identity_sha256
        or epoch_receipt["run_contract_sha256"] != run_contract_sha256
        or epoch_receipt["model_state_sha256_before"] != list(expected_before_hashes)
        or epoch_receipt["teacher_values_are_realized_match_ev"] is not False
        or epoch_receipt["profile_activation_authorized"] is not False
        or epoch_receipt["current_profile_changed"] is not False
    ):
        raise ValueError(f"{stage} epoch {epoch} receipt changed")
    models, bundle = training.load_ensemble_checkpoint_bundle(
        path / "bundle",
        torch=torch,
        expected_dataset_identity_sha256=dataset.identity_sha256,
        expected_training_config=run_config.training_config,
        expected_stage=stage,
        expected_bundle_identity_sha256=epoch_receipt[
            "checkpoint_bundle_identity_sha256"
        ],
        map_location=device,
    )
    after_hashes = [model_state_sha256(model) for model in models]
    if (
        bundle["completed_epoch"] != epoch
        or epoch_receipt["model_state_sha256_after"] != after_hashes
    ):
        raise ValueError(f"{stage} epoch {epoch} checkpoint changed")
    fit_receipt, _ = _read_canonical(
        path / "fit_receipt.json", f"{stage} epoch fit receipt"
    )
    if fit_receipt.get("receipt_sha256") != epoch_receipt["fit_receipt_sha256"]:
        raise ValueError(f"{stage} epoch {epoch} fit binding changed")
    _validate_fit_receipt(
        fit_receipt,
        dataset=dataset,
        config=run_config.training_config,
        stage=stage,
        epoch=epoch,
        before_hashes=expected_before_hashes,
        after_hashes=after_hashes,
    )
    return models, bundle, epoch_receipt


def _write_epoch(
    output_root: Path,
    *,
    torch: Any,
    models: Sequence[Any],
    dataset: training.PolicyTrainingDataset,
    run_config: StreetPolicyRunConfig,
    run_contract_sha256: str,
    stage: str,
    epoch: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before_hashes = [model_state_sha256(model) for model in models]
    if stage == "core":
        fit_receipt = training.fit_core_from_train(
            torch,
            models,
            dataset,
            training_config=run_config.training_config,
            start_epoch=epoch - 1,
            end_epoch=epoch,
        )
    else:
        fit_receipt = training.fit_risk_from_safety(
            torch,
            models,
            dataset,
            training_config=run_config.training_config,
            start_epoch=epoch - 1,
            end_epoch=epoch,
        )
    after_hashes = [model_state_sha256(model) for model in models]
    _validate_fit_receipt(
        fit_receipt,
        dataset=dataset,
        config=run_config.training_config,
        stage=stage,
        epoch=epoch,
        before_hashes=before_hashes,
        after_hashes=after_hashes,
    )
    epoch_path = output_root / f"{stage}_epoch_{epoch:03d}"
    captured: dict[str, Any] = {}

    def writer(temporary: Path) -> None:
        bundle = training.write_ensemble_checkpoint_bundle(
            temporary / "bundle",
            models,
            dataset=dataset,
            training_config=run_config.training_config,
            stage=stage,
            completed_epoch=epoch,
        )
        receipt = _epoch_receipt(
            stage=stage,
            epoch=epoch,
            dataset=dataset,
            run_config=run_config,
            run_contract_sha256=run_contract_sha256,
            fit_receipt=fit_receipt,
            bundle=bundle,
        )
        _write_once(temporary / "fit_receipt.json", fit_receipt)
        _write_once(temporary / "epoch_receipt.json", receipt)
        captured["bundle"] = bundle
        captured["receipt"] = receipt

    _atomic_directory(epoch_path, writer)
    return captured["bundle"], captured["receipt"]


def _validate_root_names(
    root: Path,
    *,
    core_epochs: int,
    risk_epochs: int,
) -> None:
    allowed = {"run_contract.json", "calibration", "run_receipt.json"}
    allowed.update(f"core_epoch_{epoch:03d}" for epoch in range(1, core_epochs + 1))
    allowed.update(f"risk_epoch_{epoch:03d}" for epoch in range(1, risk_epochs + 1))
    unknown = {child.name for child in root.iterdir()} - allowed
    if unknown:
        raise ValueError(f"training output contains unknown entries: {sorted(unknown)}")
    if any(_is_link_or_junction(child) for child in root.iterdir()):
        raise ValueError("training output contains a link or junction")


def _source_replay_chain(
    root: Path,
    *,
    torch: Any,
    device: Any,
    dataset: training.PolicyTrainingDataset,
    run_config: StreetPolicyRunConfig,
    run_contract_sha256: str,
) -> tuple[list[Any], str, int, dict[str, Any] | None]:
    config = run_config.training_config
    initial = training.create_deterministic_ensemble(
        torch,
        training_config=config,
        model_config=run_config.model_config,
        device=device,
    )
    models = initial
    hashes = [model_state_sha256(model) for model in models]
    last_stage = "initial"
    last_epoch = 0
    last_bundle: dict[str, Any] | None = None
    missing_core = False
    for epoch in range(1, config.core_epochs + 1):
        path = root / f"core_epoch_{epoch:03d}"
        if not path.exists():
            missing_core = True
            if any(
                (root / f"core_epoch_{later:03d}").exists()
                for later in range(epoch + 1, config.core_epochs + 1)
            ):
                raise ValueError("core checkpoint chain has a gap")
            break
        models, last_bundle, _ = _validate_epoch_directory(
            path,
            torch=torch,
            device=device,
            dataset=dataset,
            run_config=run_config,
            run_contract_sha256=run_contract_sha256,
            stage="core",
            epoch=epoch,
            expected_before_hashes=hashes,
        )
        hashes = [model_state_sha256(model) for model in models]
        last_stage, last_epoch = "core", epoch
    risk_paths = [
        root / f"risk_epoch_{epoch:03d}" for epoch in range(1, config.risk_epochs + 1)
    ]
    if missing_core and any(path.exists() for path in risk_paths):
        raise ValueError("risk checkpoints exist before core is complete")
    if not missing_core:
        for epoch, path in enumerate(risk_paths, start=1):
            if not path.exists():
                if any(later.exists() for later in risk_paths[epoch:]):
                    raise ValueError("risk checkpoint chain has a gap")
                break
            models, last_bundle, _ = _validate_epoch_directory(
                path,
                torch=torch,
                device=device,
                dataset=dataset,
                run_config=run_config,
                run_contract_sha256=run_contract_sha256,
                stage="risk",
                epoch=epoch,
                expected_before_hashes=hashes,
            )
            hashes = [model_state_sha256(model) for model in models]
            last_stage, last_epoch = "risk", epoch
    return models, last_stage, last_epoch, last_bundle


def _write_calibration(
    path: Path,
    *,
    torch: Any,
    models: Sequence[Any],
    dataset: training.PolicyTrainingDataset,
    run_config: StreetPolicyRunConfig,
    final_bundle: Mapping[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def writer(temporary: Path) -> None:
        before = [model_state_sha256(model) for model in models]
        threshold = training.lock_seat_thresholds(
            torch,
            models,
            dataset,
            training_config=run_config.training_config,
        )
        diagnostic = training.report_diagnostic_holdout(
            torch,
            models,
            dataset,
            training_config=run_config.training_config,
            threshold_lock=threshold,
        )
        after = [model_state_sha256(model) for model in models]
        if after != before:
            raise RuntimeError("calibration changed StreetPolicyNetV1 weights")
        _write_once(temporary / "threshold_lock.json", threshold)
        _write_once(temporary / "diagnostic_report.json", diagnostic)
        identity = {
            "schema": CALIBRATION_RECEIPT_SCHEMA,
            "training_view_identity_sha256": dataset.identity_sha256,
            "run_config_identity_sha256": run_config.identity_sha256,
            "run_contract_sha256": run_contract_sha256,
            "checkpoint_bundle_identity_sha256": final_bundle["bundle_identity_sha256"],
            "threshold_lock_sha256": threshold["threshold_lock_sha256"],
            "threshold_lock_file_sha256": _sha256_bytes(
                (temporary / "threshold_lock.json").read_bytes()
            ),
            "diagnostic_report_sha256": diagnostic["report_sha256"],
            "diagnostic_report_file_sha256": _sha256_bytes(
                (temporary / "diagnostic_report.json").read_bytes()
            ),
            "model_state_sha256": before,
            "weight_update_count": 0,
            "threshold_research_on_diagnostic_holdout": False,
            "promotion_authorized": False,
            "teacher_values_are_realized_match_ev": False,
            "current_profile_changed": False,
        }
        receipt = dict(identity)
        receipt["calibration_receipt_sha256"] = _canonical_sha256(identity)
        _write_once(temporary / "calibration_receipt.json", receipt)
        captured["receipt"] = receipt

    _atomic_directory(path, writer)
    return captured["receipt"]


def _validate_calibration(
    path: Path,
    *,
    dataset: training.PolicyTrainingDataset,
    run_config: StreetPolicyRunConfig,
    final_bundle: Mapping[str, Any],
    run_contract_sha256: str,
    model_hashes: Sequence[str],
) -> dict[str, Any]:
    expected = {
        "threshold_lock.json",
        "diagnostic_report.json",
        "calibration_receipt.json",
    }
    if (
        not path.is_dir()
        or _is_link_or_junction(path)
        or {child.name for child in path.iterdir()} != expected
        or any(_is_link_or_junction(child) for child in path.iterdir())
    ):
        raise ValueError("calibration artifact directory changed")
    threshold, threshold_raw = _read_canonical(
        path / "threshold_lock.json", "training threshold lock"
    )
    diagnostic, diagnostic_raw = _read_canonical(
        path / "diagnostic_report.json", "diagnostic report"
    )
    receipt, _ = _read_canonical(
        path / "calibration_receipt.json", "calibration receipt"
    )
    identity = dict(receipt)
    declared = identity.pop("calibration_receipt_sha256", None)
    required = {
        "schema",
        "training_view_identity_sha256",
        "run_config_identity_sha256",
        "run_contract_sha256",
        "checkpoint_bundle_identity_sha256",
        "threshold_lock_sha256",
        "threshold_lock_file_sha256",
        "diagnostic_report_sha256",
        "diagnostic_report_file_sha256",
        "model_state_sha256",
        "weight_update_count",
        "threshold_research_on_diagnostic_holdout",
        "promotion_authorized",
        "teacher_values_are_realized_match_ev",
        "current_profile_changed",
        "calibration_receipt_sha256",
    }
    diagnostic_identity = dict(diagnostic)
    diagnostic_declared = diagnostic_identity.pop("report_sha256", None)
    if (
        set(receipt) != required
        or receipt["schema"] != CALIBRATION_RECEIPT_SCHEMA
        or declared != _canonical_sha256(identity)
        or receipt["training_view_identity_sha256"] != dataset.identity_sha256
        or receipt["run_config_identity_sha256"] != run_config.identity_sha256
        or receipt["run_contract_sha256"] != run_contract_sha256
        or receipt["checkpoint_bundle_identity_sha256"]
        != final_bundle["bundle_identity_sha256"]
        or receipt["threshold_lock_sha256"] != threshold.get("threshold_lock_sha256")
        or receipt["threshold_lock_file_sha256"] != _sha256_bytes(threshold_raw)
        or receipt["diagnostic_report_sha256"] != diagnostic_declared
        or diagnostic_declared != _canonical_sha256(diagnostic_identity)
        or receipt["diagnostic_report_file_sha256"] != _sha256_bytes(diagnostic_raw)
        or receipt["model_state_sha256"] != list(model_hashes)
        or receipt["weight_update_count"] != 0
        or receipt["threshold_research_on_diagnostic_holdout"] is not False
        or receipt["promotion_authorized"] is not False
        or receipt["teacher_values_are_realized_match_ev"] is not False
        or receipt["current_profile_changed"] is not False
        or threshold.get("training_view_identity_sha256") != dataset.identity_sha256
        or threshold.get("training_config_sha256")
        != run_config.training_config.identity_sha256
        or diagnostic.get("training_view_identity_sha256") != dataset.identity_sha256
        or diagnostic.get("training_config_sha256")
        != run_config.training_config.identity_sha256
        or diagnostic.get("diagnostic_only") is not True
        or diagnostic.get("promotion_authorized") is not False
        or diagnostic.get("threshold_research_performed") is not False
    ):
        raise ValueError("calibration artifact contract changed")
    # Reuse the trainer's complete semantic validator for the rich lock.
    training._validate_threshold_lock(  # type: ignore[attr-defined]
        threshold,
        training_config=run_config.training_config,
        expected_dataset_identity_sha256=dataset.identity_sha256,
        expected_model_hashes=model_hashes,
    )
    return receipt


def execute_training_run(
    *,
    torch: Any,
    dataset: training.PolicyTrainingDataset,
    run_config: StreetPolicyRunConfig,
    output_root: str | Path,
    requested_device: str,
    source_dataset_receipt: Mapping[str, Any],
    allow_synthetic_cpu_smoke: bool = False,
) -> dict[str, Any]:
    """Run or resume an immutable StreetPolicyNetV1 training directory."""

    dataset.validate()
    source_type = dataset.manifest["source_type"]
    if source_type != "verified_m31_dataset":
        if not (
            allow_synthetic_cpu_smoke
            and source_type == "synthetic_cpu_smoke"
            and requested_device == "cpu"
        ):
            raise PermissionError(
                "production trainer requires a source-replayed M3.1 dataset"
            )
    if source_dataset_receipt.get("dataset_identity_sha256") != (
        dataset.manifest["source_dataset_identity_sha256"]
    ):
        raise ValueError("source dataset receipt does not bind the training view")
    if source_type == "verified_m31_dataset" and (
        source_dataset_receipt.get("source_replayed") is not True
        or source_dataset_receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("production source dataset receipt is not qualified")
    device, device_contract = _resolve_device(torch, requested_device)
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("training output root must be absolute")
    if root.exists() and (not root.is_dir() or _is_link_or_junction(root)):
        raise ValueError("training output root must be a non-link directory")
    existing_ancestors = (parent for parent in root.parents if parent.exists())
    if any(_is_link_or_junction(parent) for parent in existing_ancestors):
        raise ValueError("training output root ancestry contains a link or junction")
    root.mkdir(parents=True, exist_ok=True)
    contract_identity = {
        "schema": RUN_CONTRACT_SCHEMA,
        "training_view_identity_sha256": dataset.identity_sha256,
        "source_dataset_identity_sha256": dataset.manifest[
            "source_dataset_identity_sha256"
        ],
        "source_dataset_receipt_sha256": _canonical_sha256(
            dict(source_dataset_receipt)
        ),
        "run_config": run_config.to_dict(),
        "run_config_identity_sha256": run_config.identity_sha256,
        "device": device_contract,
        "resume_unit": "one_epoch_atomic_checkpoint_plus_fit_receipt",
        "teacher_values_are_realized_match_ev": False,
        "profile_activation_authorized": False,
        "current_profile_changed": False,
    }
    contract = dict(contract_identity)
    contract["run_contract_sha256"] = _canonical_sha256(contract_identity)
    contract_path = root / "run_contract.json"
    if contract_path.exists():
        observed, _ = _read_canonical(contract_path, "training run contract")
        if observed != contract:
            raise ValueError("training run contract changed on resume")
    else:
        if any(root.iterdir()):
            raise ValueError("new training output root is not empty")
        _write_once(contract_path, contract)
    config = run_config.training_config
    _validate_root_names(
        root, core_epochs=config.core_epochs, risk_epochs=config.risk_epochs
    )
    models, stage, completed, bundle = _source_replay_chain(
        root,
        torch=torch,
        device=device,
        dataset=dataset,
        run_config=run_config,
        run_contract_sha256=contract["run_contract_sha256"],
    )
    start_core = completed + 1 if stage == "core" else 1
    if stage == "initial":
        start_core = 1
    if stage in {"initial", "core"} and not (
        stage == "core" and completed == config.core_epochs
    ):
        for epoch in range(start_core, config.core_epochs + 1):
            bundle, _ = _write_epoch(
                root,
                torch=torch,
                models=models,
                dataset=dataset,
                run_config=run_config,
                run_contract_sha256=contract["run_contract_sha256"],
                stage="core",
                epoch=epoch,
            )
        stage, completed = "core", config.core_epochs
    if stage == "core":
        risk_start = 1
    elif stage == "risk":
        risk_start = completed + 1
    else:
        raise RuntimeError("training checkpoint state is invalid")
    for epoch in range(risk_start, config.risk_epochs + 1):
        bundle, _ = _write_epoch(
            root,
            torch=torch,
            models=models,
            dataset=dataset,
            run_config=run_config,
            run_contract_sha256=contract["run_contract_sha256"],
            stage="risk",
            epoch=epoch,
        )
    if bundle is None:
        raise RuntimeError("training completed without a checkpoint bundle")
    final_hashes = [model_state_sha256(model) for model in models]
    calibration_path = root / "calibration"
    if calibration_path.exists():
        calibration = _validate_calibration(
            calibration_path,
            dataset=dataset,
            run_config=run_config,
            final_bundle=bundle,
            run_contract_sha256=contract["run_contract_sha256"],
            model_hashes=final_hashes,
        )
    else:
        calibration = _write_calibration(
            calibration_path,
            torch=torch,
            models=models,
            dataset=dataset,
            run_config=run_config,
            final_bundle=bundle,
            run_contract_sha256=contract["run_contract_sha256"],
        )
    identity = {
        "schema": FINAL_RECEIPT_SCHEMA,
        "complete": True,
        "training_view_identity_sha256": dataset.identity_sha256,
        "source_dataset_identity_sha256": dataset.manifest[
            "source_dataset_identity_sha256"
        ],
        "run_config_identity_sha256": run_config.identity_sha256,
        "run_contract_sha256": contract["run_contract_sha256"],
        "device": device_contract,
        "core_epochs_completed": config.core_epochs,
        "risk_epochs_completed": config.risk_epochs,
        "final_checkpoint_stage": "risk",
        "final_checkpoint_bundle_identity_sha256": bundle["bundle_identity_sha256"],
        "final_checkpoint_manifest_file_sha256": _sha256_bytes(
            (
                root
                / f"risk_epoch_{config.risk_epochs:03d}"
                / "bundle"
                / "manifest.json"
            ).read_bytes()
        ),
        "final_model_state_sha256": final_hashes,
        "calibration_receipt_sha256": calibration["calibration_receipt_sha256"],
        "promotion_authorized": False,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    result = dict(identity)
    result["run_receipt_sha256"] = _canonical_sha256(identity)
    final_path = root / "run_receipt.json"
    if final_path.exists():
        observed, _ = _read_canonical(final_path, "training final receipt")
        if observed != result:
            raise ValueError("training final receipt changed on resume")
    else:
        _write_once(final_path, result)
    _validate_root_names(
        root, core_epochs=config.core_epochs, risk_epochs=config.risk_epochs
    )
    return result


def _load_shard_map(path: str | Path) -> dict[str, Path]:
    value, _ = _read_canonical(Path(path), "M3.1 shard map")
    expected_keys = {
        str(row["shard_id"]) for row in dataset_contract.build_dataset_plan()["shards"]
    }
    if set(value) != expected_keys or any(
        not isinstance(key, str)
        or not isinstance(item, str)
        or not Path(item).is_absolute()
        for key, item in value.items()
    ):
        raise ValueError(
            "M3.1 shard map must be the exact absolute 360-shard map "
            "consumed by merge-dataset"
        )
    return {key: Path(item) for key, item in value.items()}


def _command_write_default(args: argparse.Namespace) -> int:
    print(json.dumps(write_default_run_config(args.output), sort_keys=True))
    return 0


def _command_train(args: argparse.Namespace) -> int:
    import torch

    run_config = load_run_config(
        args.config,
        expected_file_sha256=args.expected_config_sha256,
    )
    verified = training.verify_immutable_dataset_manifest(
        plan_path=args.plan,
        merge_manifest_path=args.merge,
        shard_directories=_load_shard_map(args.shard_map),
        expected_merge_file_sha256=args.expected_merge_sha256,
    )
    dataset = training.load_verified_policy_training_dataset(
        verified,
        teacher_policy_temperature=(
            run_config.training_config.teacher_policy_temperature
        ),
    )
    result = execute_training_run(
        torch=torch,
        dataset=dataset,
        run_config=run_config,
        output_root=args.output_root,
        requested_device=args.device,
        source_dataset_receipt=verified.receipt,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3.1 StreetPolicyNetV1 production training"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write-default-config")
    write.add_argument("--output", required=True)
    write.set_defaults(handler=_command_write_default)
    train = subparsers.add_parser("train")
    train.add_argument("--plan", required=True)
    train.add_argument("--merge", required=True)
    train.add_argument("--expected-merge-sha256", required=True)
    train.add_argument("--shard-map", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--expected-config-sha256", required=True)
    train.add_argument("--output-root", required=True)
    train.add_argument("--device", default="auto")
    train.set_defaults(handler=_command_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:  # pragma: no cover - exercised through subprocess
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_RECEIPT_SCHEMA",
    "EPOCH_RECEIPT_SCHEMA",
    "FINAL_RECEIPT_SCHEMA",
    "RUN_CONFIG_SCHEMA",
    "RUN_CONTRACT_SCHEMA",
    "StreetPolicyRunConfig",
    "default_run_config",
    "execute_training_run",
    "load_run_config",
    "main",
    "write_default_run_config",
]
