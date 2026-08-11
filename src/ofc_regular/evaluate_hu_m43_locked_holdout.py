"""Run the single diagnostic M4.3 locked-holdout pass after model freeze.

This command never trains, calibrates, searches a threshold, promotes a model,
or activates a runtime profile.  It atomically consumes a caller-specified
marker before reading the locked shard, so failures after that point also
consume the holdout and cannot be retried with the same marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m43_pilot_contract import (
    M43_LOCKED_RECEIPT_SCHEMA,
    canonical_manifest_sha256,
    validate_data_contract_binding,
    validate_locked_holdout_receipt,
    validate_model_threshold_freeze,
)
from .hu_m4_joint_model import (
    PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
    HuM4JointActionModel,
)
from .train_hu_m4_joint_model import (
    evaluate_model,
    prepare_teacher_samples,
    read_teacher_jsonl,
)


M43_CONSUMPTION_MARKER_SCHEMA = "hu_m43_locked_holdout_consumption_marker_v1"


def evaluate_locked_holdout_once(
    *,
    model_path: str | Path,
    freeze_manifest_path: str | Path,
    data_contract_path: str | Path,
    plan_path: str | Path,
    repo_root: str | Path,
    locked_holdout: Sequence[str | Path],
    receipt_path: str | Path,
    consumption_marker_path: str | Path,
) -> dict[str, Any]:
    """Evaluate frozen M4.3 artifact exactly once and write its receipt."""

    model_source = Path(model_path).resolve()
    freeze_source = Path(freeze_manifest_path).resolve()
    contract_source = Path(data_contract_path).resolve()
    plan_source = Path(plan_path).resolve()
    locked_paths = tuple(Path(path).resolve() for path in locked_holdout)
    receipt_output = Path(receipt_path).resolve()
    marker_output = Path(consumption_marker_path).resolve()
    all_paths = {
        model_source,
        freeze_source,
        contract_source,
        plan_source,
        *locked_paths,
        receipt_output,
        marker_output,
    }
    expected_count = 6 + len(locked_paths)
    if len(all_paths) != expected_count:
        raise ValueError("M4.3 one-shot inputs, marker, and receipt must be distinct")
    if not locked_paths:
        raise ValueError("M4.3 one-shot evaluator requires locked shards")
    if receipt_output.exists():
        raise FileExistsError(f"M4.3 locked receipt already exists: {receipt_output}")
    if marker_output.exists():
        raise FileExistsError(
            f"M4.3 locked holdout was already consumed: {marker_output}"
        )

    freeze = _read_mapping(freeze_source)
    data_contract = _read_mapping(contract_source)
    validate_model_threshold_freeze(freeze, data_contract=data_contract)
    if str(contract_source) != freeze.get("data_contract_resolved_path"):
        raise ValueError("M4.3 one-shot data-contract path disagrees with freeze")
    if str(marker_output) != freeze.get("locked_consumption_marker_resolved_path"):
        raise ValueError("M4.3 one-shot consumption-marker path disagrees with freeze")
    model_sha = _sha256(model_source)
    if model_sha != freeze.get("model_sha256"):
        raise ValueError("M4.3 one-shot model SHA disagrees with freeze")
    model = HuM4JointActionModel.load(model_source)
    if model.action_score_mode != PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE:
        raise ValueError("M4.3 one-shot artifact mode is not paired-delta risk")
    if model.model_id != freeze.get("model_id"):
        raise ValueError("M4.3 one-shot model_id disagrees with freeze")
    if float(model.safety_threshold) != float(freeze.get("frozen_threshold")):
        raise ValueError("M4.3 one-shot threshold disagrees with freeze")
    if bool(model.safety_enabled) != bool(freeze.get("safety_enabled")):
        raise ValueError("M4.3 one-shot safety state disagrees with freeze")

    marker = {
        "schema": M43_CONSUMPTION_MARKER_SCHEMA,
        "status": "claimed_before_locked_content_read",
        "freeze_manifest_sha256": canonical_manifest_sha256(freeze),
        "data_contract_sha256": data_contract["contract_sha256"],
        "model_sha256": model_sha,
        "locked_identity_sha256": data_contract["splits"]["locked_holdout"][
            "identity_sha256"
        ],
        "locked_teacher_shards_sha256": data_contract["teacher_shards"][
            "splits"
        ]["locked_holdout"]["ordered_shards_sha256"],
        "evaluation_pass_count": 1,
    }
    _claim_marker(marker_output, marker)

    binding = validate_data_contract_binding(
        data_contract,
        plan_path=plan_source,
        repo_root=repo_root,
        locked_holdout=locked_paths,
    )
    rows = [
        row
        for path in locked_paths
        for row in read_teacher_jsonl(path)
    ]
    samples = prepare_teacher_samples(rows)
    diagnostic = evaluate_model(model, samples, split="locked_holdout")
    receipt: dict[str, Any] = {
        "schema": M43_LOCKED_RECEIPT_SCHEMA,
        "status": "evaluated_once_diagnostic_only_no_activation",
        "freeze_manifest_sha256": canonical_manifest_sha256(freeze),
        "data_contract_sha256": data_contract["contract_sha256"],
        "model_sha256": model_sha,
        "frozen_threshold": float(model.safety_threshold),
        "locked_identity_sha256": data_contract["splits"]["locked_holdout"][
            "identity_sha256"
        ],
        "locked_teacher_shards_sha256": binding["verified_splits"][
            "locked_holdout"
        ]["ordered_shards_sha256"],
        "consumption_marker_sha256": _sha256(marker_output),
        "evaluation_pass_count": 1,
        "threshold_search_performed": False,
        "model_selection_performed": False,
        "feature_selection_performed": False,
        "current_profile_resolved": False,
        "runtime_policy_activated": False,
        "policy_promoted": False,
        "requires_fresh_population_acceptance": True,
        "minimum_population_valid_overrides": int(
            freeze["minimum_population_valid_overrides"]
        ),
        "teacher_value_status": "diagnostic_not_realized_match_ev",
        "locked_diagnostic": diagnostic,
    }
    validate_locked_holdout_receipt(
        receipt, freeze_manifest=freeze, data_contract=data_contract
    )
    _atomic_json(receipt_output, receipt)
    return receipt


def _claim_marker(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A created marker means the holdout was claimed even if writing or a
        # later evaluation failed.  Deliberately do not remove it.
        raise


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"M4.3 JSON artifact must be a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--locked-holdout", type=Path, action="append", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--consumption-marker", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = evaluate_locked_holdout_once(
        model_path=args.model,
        freeze_manifest_path=args.freeze_manifest,
        data_contract_path=args.data_contract,
        plan_path=args.plan,
        repo_root=args.repo_root,
        locked_holdout=args.locked_holdout,
        receipt_path=args.receipt,
        consumption_marker_path=args.consumption_marker,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["M43_CONSUMPTION_MARKER_SCHEMA", "evaluate_locked_holdout_once"]
