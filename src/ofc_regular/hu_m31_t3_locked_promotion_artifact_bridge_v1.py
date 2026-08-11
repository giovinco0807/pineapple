"""Bind StreetPolicyNetV1 training outputs to the locked-promotion contract.

Training emits a directory of ensemble checkpoints and a rich, seat-specific
threshold-lock object.  Locked promotion binds the bundle's canonical
``manifest.json`` as its immutable model file and consumes a compact runtime
lock.  This bridge fully replays the directory, then creates that write-once
compatibility lock.  Runtime still replays every model file named by the
manifest; no directory hash or lossy single-model export is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import ACTION_KEY_SCHEMA
from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from . import hu_m31_t3_street_policy_training_v1 as training
from .street_policy_net_v1 import FEATURE_SCHEMA_HASH


def canonical_bytes(value: Any) -> bytes:
    return promotion.canonical_bytes(value)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_canonical(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical LF JSON")
    return value, source.resolve()


def _write_once(path: Path, raw: bytes) -> None:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite immutable artifact: {target}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_locked_promotion_artifacts(
    *,
    checkpoint_bundle_directory: str | Path,
    training_threshold_lock_path: str | Path,
    model_artifact_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay training outputs and return manifest binding plus runtime lock."""

    bundle = Path(checkpoint_bundle_directory)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("checkpoint bundle must be a safe directory")
    manifest, manifest_path = _read_canonical(
        bundle / "manifest.json", "StreetPolicyNetV1 bundle manifest"
    )
    config_raw = manifest.get("training_config")
    if not isinstance(config_raw, Mapping):
        raise ValueError("StreetPolicyNetV1 training config is missing")
    config_value = dict(config_raw)
    if config_value.pop("schema", None) != training.TRAINING_CONFIG_SCHEMA:
        raise ValueError("StreetPolicyNetV1 training config schema changed")
    config = training.StreetPolicyTrainingConfig(**config_value)
    if (
        manifest.get("schema") != training.CHECKPOINT_BUNDLE_SCHEMA
        or manifest.get("stage") != "risk"
        or manifest.get("completed_epoch") != config.risk_epochs
        or manifest.get("feature_schema_hash") != FEATURE_SCHEMA_HASH
    ):
        raise ValueError("only a completed risk-stage ensemble may be promoted")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("promotion artifact replay requires torch") from exc
    _models, replayed_manifest = training.load_ensemble_checkpoint_bundle(
        bundle,
        torch=torch,
        expected_dataset_identity_sha256=str(
            manifest["training_view_identity_sha256"]
        ),
        expected_training_config=config,
        expected_stage="risk",
        expected_bundle_identity_sha256=str(manifest["bundle_identity_sha256"]),
    )
    if replayed_manifest != manifest:
        raise ValueError("StreetPolicyNetV1 bundle differs from source replay")

    source_lock, source_lock_path = _read_canonical(
        training_threshold_lock_path, "StreetPolicyNetV1 threshold lock"
    )
    model_hashes = [
        str(record["model_state_sha256"]) for record in manifest["models"]
    ]
    validated_lock = training._validate_threshold_lock(
        source_lock,
        training_config=config,
        expected_dataset_identity_sha256=str(
            manifest["training_view_identity_sha256"]
        ),
        expected_model_hashes=model_hashes,
    )
    manifest_sha = _sha256(manifest_path.read_bytes())
    seats = validated_lock["seat_thresholds"]
    compact_lock = {
        "schema": promotion.THRESHOLD_LOCK_SCHEMA,
        "status": "locked_no_holdout_reselection",
        "model_artifact_id": model_artifact_id,
        "model_sha256": manifest_sha,
        "state_action_input_schema_sha256": FEATURE_SCHEMA_HASH,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "seat_thresholds": {
            seat: float(seats[seat]["safe_probability_threshold"])
            for seat in promotion.SEATS
        },
        "seat_enabled": {
            seat: bool(seats[seat]["enabled"]) for seat in promotion.SEATS
        },
        "source_training_threshold_lock_sha256": _sha256(
            source_lock_path.read_bytes()
        ),
        "source_checkpoint_bundle_identity_sha256": str(
            manifest["bundle_identity_sha256"]
        ),
        "locked_on_threshold_holdout_only": True,
        "model_change_requires_new_lock": True,
        "teacher_ev_lcb_is_runtime_gate": False,
        "top1_accuracy_is_promotion_gate": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }
    promotion._validate_threshold_lock(
        compact_lock, expected_model_sha256=manifest_sha
    )
    binding = {
        "model_artifact_id": model_artifact_id,
        "model_path": str(manifest_path),
        "model_sha256": manifest_sha,
        "checkpoint_bundle_directory": str(bundle.resolve()),
        "checkpoint_bundle_identity_sha256": str(
            manifest["bundle_identity_sha256"]
        ),
    }
    return binding, compact_lock


def write_locked_promotion_artifacts(
    *,
    checkpoint_bundle_directory: str | Path,
    training_threshold_lock_path: str | Path,
    model_artifact_id: str,
    output_threshold_lock_path: str | Path,
) -> dict[str, Any]:
    """Create once the compact lock consumed by ``build_locked_promotion_plan``."""

    lock_output = Path(output_threshold_lock_path)
    if lock_output.exists() or lock_output.is_symlink():
        raise FileExistsError("locked-promotion bridge outputs are write-once")
    binding, compact_lock = build_locked_promotion_artifacts(
        checkpoint_bundle_directory=checkpoint_bundle_directory,
        training_threshold_lock_path=training_threshold_lock_path,
        model_artifact_id=model_artifact_id,
    )
    _write_once(lock_output, canonical_bytes(compact_lock))
    if (
        promotion.sha256_file(binding["model_path"])
        != compact_lock["model_sha256"]
        or _read_canonical(lock_output, "stored promotion threshold lock")[0]
        != compact_lock
    ):
        raise ValueError("stored locked-promotion artifacts changed")
    return {
        **binding,
        "threshold_lock_path": str(lock_output.resolve()),
        "threshold_lock_sha256": promotion.sha256_file(lock_output),
        "source_checkpoint_bundle_identity_sha256": compact_lock[
            "source_checkpoint_bundle_identity_sha256"
        ],
        "source_training_threshold_lock_sha256": compact_lock[
            "source_training_threshold_lock_sha256"
        ],
        "current_profile_changed": False,
        "runtime_activated": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind a completed StreetPolicyNetV1 risk bundle and frozen "
            "training threshold lock to the M3.1 locked-promotion contract."
        )
    )
    parser.add_argument("--checkpoint-bundle-directory", required=True)
    parser.add_argument("--training-threshold-lock", required=True)
    parser.add_argument("--model-artifact-id", required=True)
    parser.add_argument("--output-threshold-lock", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = write_locked_promotion_artifacts(
            checkpoint_bundle_directory=args.checkpoint_bundle_directory,
            training_threshold_lock_path=args.training_threshold_lock,
            model_artifact_id=args.model_artifact_id,
            output_threshold_lock_path=args.output_threshold_lock,
        )
    except Exception as exc:  # pragma: no cover - exercised through subprocess
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_locked_promotion_artifacts",
    "canonical_bytes",
    "main",
    "write_locked_promotion_artifacts",
]
