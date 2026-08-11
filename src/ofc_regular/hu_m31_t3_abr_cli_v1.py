"""Operational CLI for the three frozen M3.1 approximate best responses.

The numerical ABR implementation intentionally exposes only library
functions.  This module supplies the missing production boundary:

* the search-teacher examples are an externally hash-pinned canonical object;
* the normalized development dataset is rebuilt before every use;
* all three checkpoints are built in a sibling staging directory and become
  visible together through one atomic rename;
* a completed bundle is source-replayed on resume instead of retrained.

No locked-population/ABR evaluation seed is accepted for training, and this
module never registers a profile or resolves ``current``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_abr_v1 as abr
from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion


RAW_EXAMPLES_SCHEMA = "hu_m31_t3_abr_raw_examples_v1"
ABR_OPERATION_RECEIPT_SCHEMA = "hu_m31_t3_abr_operation_receipt_v1"
ABR_PRODUCTION_BUILD_RECEIPT_SCHEMA = (
    "hu_m31_t3_abr_production_build_receipt_v1"
)
ABR_PRODUCTION_BUILD_RECEIPT_FILE = "production_build_receipt.json"


def _canonical_bytes(value: Any) -> bytes:
    return promotion.canonical_bytes(value)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_canonical(
    path: str | Path, label: str
) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise ValueError(f"{label} is not a canonical JSON object")
    return value, raw


def _read_pinned(
    path: str | Path, expected_sha256: str, label: str
) -> tuple[dict[str, Any], bytes]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} expected SHA-256 is not lowercase hex")
    value, raw = _read_canonical(path, label)
    if _sha256(raw) != expected_sha256:
        raise ValueError(f"{label} file SHA-256 changed")
    return value, raw


def _write_once_or_replay(path: Path, value: Mapping[str, Any]) -> bool:
    """Write canonical bytes once; return True when an identical file existed."""

    raw = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return False
    except FileExistsError:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != raw
        ):
            raise FileExistsError(
                f"immutable ABR artifact already exists with other bytes: {path}"
            ) from None
        return True


def build_raw_examples_document(
    raw_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze a search-teacher example list before candidate-specific replay."""

    examples = [dict(value) for value in raw_examples]
    if not examples:
        raise ValueError("ABR raw example document cannot be empty")
    identity = {
        "schema": RAW_EXAMPLES_SCHEMA,
        "status": "frozen_search_teacher_development_only",
        "source_schedule": "abr_development",
        "locked_evaluation_schedule": promotion.LOCKED_ABR,
        "locked_seed_training_allowed": False,
        "example_count": len(examples),
        "examples": examples,
        "example_aggregate_sha256": promotion.canonical_sha256(examples),
        "teacher_values_are_realized_locked_match_ev": False,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
        "current_profile_resolved": False,
    }
    return {
        **identity,
        "raw_examples_identity_sha256": promotion.canonical_sha256(identity),
    }


def validate_raw_examples_document(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    document = dict(value)
    required = {
        "schema",
        "status",
        "source_schedule",
        "locked_evaluation_schedule",
        "locked_seed_training_allowed",
        "example_count",
        "examples",
        "example_aggregate_sha256",
        "teacher_values_are_realized_locked_match_ev",
        "opponent_private_discards_used",
        "realized_deck_tail_used",
        "current_profile_resolved",
        "raw_examples_identity_sha256",
    }
    examples = document.get("examples")
    if (
        set(document) != required
        or document.get("schema") != RAW_EXAMPLES_SCHEMA
        or document.get("status")
        != "frozen_search_teacher_development_only"
        or document.get("source_schedule") != "abr_development"
        or document.get("locked_evaluation_schedule") != promotion.LOCKED_ABR
        or document.get("locked_seed_training_allowed") is not False
        or not isinstance(examples, list)
        or not examples
        or not all(isinstance(item, Mapping) for item in examples)
        or document.get("example_count") != len(examples)
        or document.get("example_aggregate_sha256")
        != promotion.canonical_sha256(examples)
        or document.get("teacher_values_are_realized_locked_match_ev")
        is not False
        or document.get("opponent_private_discards_used") is not False
        or document.get("realized_deck_tail_used") is not False
        or document.get("current_profile_resolved") is not False
    ):
        raise ValueError("ABR raw example document contract changed")
    identity = dict(document)
    declared = identity.pop("raw_examples_identity_sha256")
    if declared != promotion.canonical_sha256(identity):
        raise ValueError("ABR raw example document identity changed")
    return document


def write_raw_examples_document(
    *,
    raw_examples: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    document = build_raw_examples_document(raw_examples)
    _write_once_or_replay(Path(output_path), document)
    return validate_raw_examples_document(document)


def _load_candidate_manifest(
    *,
    candidate_bundle_directory: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    manifest, _raw = _read_pinned(
        Path(candidate_bundle_directory) / "manifest.json",
        expected_manifest_sha256,
        "candidate checkpoint manifest",
    )
    return manifest


def freeze_development_dataset(
    *,
    promotion_plan_path: str | Path,
    expected_promotion_plan_sha256: str,
    candidate_bundle_directory: str | Path,
    expected_candidate_manifest_sha256: str,
    raw_examples_path: str | Path,
    expected_raw_examples_sha256: str,
    teacher_receipt_path: str | Path,
    expected_teacher_receipt_sha256: str,
    output_path: str | Path,
) -> dict[str, Any]:
    plan, _ = _read_pinned(
        promotion_plan_path,
        expected_promotion_plan_sha256,
        "locked promotion plan",
    )
    plan = promotion.validate_locked_promotion_plan(plan)
    candidate = _load_candidate_manifest(
        candidate_bundle_directory=candidate_bundle_directory,
        expected_manifest_sha256=expected_candidate_manifest_sha256,
    )
    raw_document, _ = _read_pinned(
        raw_examples_path,
        expected_raw_examples_sha256,
        "ABR raw examples",
    )
    raw_document = validate_raw_examples_document(raw_document)
    teacher_receipt, _ = _read_pinned(
        teacher_receipt_path,
        expected_teacher_receipt_sha256,
        "ABR real-teacher receipt",
    )
    from . import hu_m31_t3_abr_teacher_v1 as abr_teacher

    teacher_receipt = abr_teacher.require_production_teacher_coverage(
        teacher_receipt,
        raw_document=raw_document,
        raw_file_sha256=expected_raw_examples_sha256,
    )
    dataset = abr.build_development_dataset(
        promotion_plan=plan,
        candidate_bundle_manifest=candidate,
        raw_examples=raw_document["examples"],
    )
    resumed = _write_once_or_replay(Path(output_path), dataset)
    # Replay from the just-frozen source object even on resume.
    observed, _ = _read_canonical(output_path, "ABR development dataset")
    validated = abr.validate_development_dataset(
        observed,
        promotion_plan=plan,
        candidate_bundle_manifest=candidate,
    )
    if validated != dataset:
        raise ValueError("stored ABR development dataset changed")
    return {
        "schema": ABR_OPERATION_RECEIPT_SCHEMA,
        "operation": "freeze_development_dataset",
        "status": "resume_complete" if resumed else "complete",
        "promotion_plan_file_sha256": expected_promotion_plan_sha256,
        "candidate_manifest_file_sha256": expected_candidate_manifest_sha256,
        "raw_examples_file_sha256": expected_raw_examples_sha256,
        "teacher_receipt_file_sha256": expected_teacher_receipt_sha256,
        "teacher_receipt_identity_sha256": teacher_receipt[
            "teacher_receipt_identity_sha256"
        ],
        "development_dataset_file_sha256": _sha256(
            Path(output_path).read_bytes()
        ),
        "development_dataset_identity_sha256": validated[
            "dataset_identity_sha256"
        ],
        "locked_seed_training_allowed": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }


def build_production_build_receipt(
    *,
    teacher_receipt: Mapping[str, Any],
    teacher_receipt_file_sha256: str,
    raw_examples_file_sha256: str,
    development_dataset: Mapping[str, Any],
    development_dataset_file_sha256: str,
    bundle: Mapping[str, Any],
    bundle_file_sha256: str,
) -> dict[str, Any]:
    """Bind exact real-teacher coverage through the three trained policies."""

    from . import hu_m31_t3_abr_teacher_v1 as abr_teacher

    if (
        teacher_receipt.get("production_training_authorized") is not True
        or teacher_receipt.get("pair_count")
        != abr_teacher.MAX_DEVELOPMENT_PAIRS
        or teacher_receipt.get("example_count")
        != abr_teacher.MAX_DEVELOPMENT_PAIRS * 2
        or teacher_receipt.get("paired_count_per_response")
        != {
            response_id: abr_teacher.MAX_DEVELOPMENT_PAIRS
            for response_id in abr.RESPONSE_IDS
        }
        or teacher_receipt.get("root_count_per_response")
        != {
            response_id: abr_teacher.MAX_DEVELOPMENT_PAIRS * 2
            for response_id in abr.RESPONSE_IDS
        }
        or development_dataset.get("example_count")
        != abr_teacher.MAX_DEVELOPMENT_PAIRS * 2
        or tuple(development_dataset.get("response_ids", ()))
        != abr.RESPONSE_IDS
        or bundle.get("family_count") != len(abr.RESPONSE_IDS)
        or [row.get("response_id") for row in bundle.get("families", ())]
        != list(abr.RESPONSE_IDS)
    ):
        raise ValueError("ABR production build lacks exact development coverage")
    identity = {
        "schema": ABR_PRODUCTION_BUILD_RECEIPT_SCHEMA,
        "status": "complete_exact_250_pair_real_teacher_lineage",
        "teacher_receipt_file_sha256": teacher_receipt_file_sha256,
        "teacher_receipt_identity_sha256": teacher_receipt[
            "teacher_receipt_identity_sha256"
        ],
        "raw_examples_file_sha256": raw_examples_file_sha256,
        "development_dataset_file_sha256": (
            development_dataset_file_sha256
        ),
        "development_dataset_identity_sha256": development_dataset[
            "dataset_identity_sha256"
        ],
        "abr_bundle_file_sha256": bundle_file_sha256,
        "abr_bundle_identity_sha256": bundle["bundle_identity_sha256"],
        "pair_count": abr_teacher.MAX_DEVELOPMENT_PAIRS,
        "example_count": abr_teacher.MAX_DEVELOPMENT_PAIRS * 2,
        "seat_counts": {
            "first": abr_teacher.MAX_DEVELOPMENT_PAIRS,
            "second": abr_teacher.MAX_DEVELOPMENT_PAIRS,
        },
        "response_ids": list(abr.RESPONSE_IDS),
        "family_count": len(abr.RESPONSE_IDS),
        "paired_count_per_response": {
            response_id: abr_teacher.MAX_DEVELOPMENT_PAIRS
            for response_id in abr.RESPONSE_IDS
        },
        "root_count_per_response": {
            response_id: abr_teacher.MAX_DEVELOPMENT_PAIRS * 2
            for response_id in abr.RESPONSE_IDS
        },
        "family_reward_sha256": abr_teacher.FAMILY_REWARD_SHA256,
        "training_street": abr.ABR_LEARNED_STREET,
        "runtime_street_composition": dict(
            abr.ABR_RUNTIME_STREET_COMPOSITION
        ),
        "runtime_street_composition_sha256": (
            abr.ABR_RUNTIME_STREET_COMPOSITION_SHA256
        ),
        "response_scientific_roles": dict(
            abr.ABR_RESPONSE_SCIENTIFIC_ROLE
        ),
        "legacy_q_bit_exact_root_count": (
            abr_teacher.MAX_DEVELOPMENT_PAIRS * 2
        ),
        "synthetic_values_used": False,
        "locked_seed_training_allowed": False,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
        "frozen_before_locked_evaluation": True,
        "current_profile_resolved": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }
    return {
        **identity,
        "production_build_receipt_identity_sha256": (
            promotion.canonical_sha256(identity)
        ),
    }


def validate_production_build_receipt(
    value: Mapping[str, Any],
    *,
    bundle_file_sha256: str,
    bundle_identity_sha256: str,
    expected_teacher_receipt_file_sha256: str | None = None,
    expected_dataset_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed before training resume or locked ABR evaluation."""

    from . import hu_m31_t3_abr_teacher_v1 as abr_teacher

    receipt = dict(value)
    identity = dict(receipt)
    declared = identity.pop(
        "production_build_receipt_identity_sha256", None
    )
    if (
        receipt.get("schema") != ABR_PRODUCTION_BUILD_RECEIPT_SCHEMA
        or receipt.get("status")
        != "complete_exact_250_pair_real_teacher_lineage"
        or receipt.get("abr_bundle_file_sha256") != bundle_file_sha256
        or receipt.get("abr_bundle_identity_sha256")
        != bundle_identity_sha256
        or (
            expected_teacher_receipt_file_sha256 is not None
            and receipt.get("teacher_receipt_file_sha256")
            != expected_teacher_receipt_file_sha256
        )
        or (
            expected_dataset_file_sha256 is not None
            and receipt.get("development_dataset_file_sha256")
            != expected_dataset_file_sha256
        )
        or receipt.get("pair_count")
        != abr_teacher.MAX_DEVELOPMENT_PAIRS
        or receipt.get("example_count")
        != abr_teacher.MAX_DEVELOPMENT_PAIRS * 2
        or receipt.get("seat_counts")
        != {
            "first": abr_teacher.MAX_DEVELOPMENT_PAIRS,
            "second": abr_teacher.MAX_DEVELOPMENT_PAIRS,
        }
        or tuple(receipt.get("response_ids", ())) != abr.RESPONSE_IDS
        or receipt.get("family_count") != len(abr.RESPONSE_IDS)
        or receipt.get("paired_count_per_response")
        != {
            response_id: abr_teacher.MAX_DEVELOPMENT_PAIRS
            for response_id in abr.RESPONSE_IDS
        }
        or receipt.get("root_count_per_response")
        != {
            response_id: abr_teacher.MAX_DEVELOPMENT_PAIRS * 2
            for response_id in abr.RESPONSE_IDS
        }
        or receipt.get("family_reward_sha256")
        != abr_teacher.FAMILY_REWARD_SHA256
        or receipt.get("training_street") != abr.ABR_LEARNED_STREET
        or receipt.get("runtime_street_composition")
        != abr.ABR_RUNTIME_STREET_COMPOSITION
        or receipt.get("runtime_street_composition_sha256")
        != abr.ABR_RUNTIME_STREET_COMPOSITION_SHA256
        or receipt.get("response_scientific_roles")
        != abr.ABR_RESPONSE_SCIENTIFIC_ROLE
        or receipt.get("legacy_q_bit_exact_root_count")
        != abr_teacher.MAX_DEVELOPMENT_PAIRS * 2
        or receipt.get("synthetic_values_used") is not False
        or receipt.get("locked_seed_training_allowed") is not False
        or receipt.get("opponent_private_discards_used") is not False
        or receipt.get("realized_deck_tail_used") is not False
        or receipt.get("frozen_before_locked_evaluation") is not True
        or receipt.get("current_profile_resolved") is not False
        or receipt.get("current_profile_changed") is not False
        or receipt.get("runtime_activated") is not False
        or not _is_sha256(declared)
        or declared != promotion.canonical_sha256(identity)
    ):
        raise ValueError("ABR production build receipt changed")
    return receipt


def train_three_policy_bundle(
    *,
    promotion_plan_path: str | Path,
    expected_promotion_plan_sha256: str,
    candidate_bundle_directory: str | Path,
    expected_candidate_manifest_sha256: str,
    development_dataset_path: str | Path,
    expected_development_dataset_sha256: str,
    raw_examples_path: str | Path | None = None,
    expected_raw_examples_sha256: str | None = None,
    teacher_receipt_path: str | Path | None = None,
    expected_teacher_receipt_sha256: str | None = None,
    production_provenance_required: bool = True,
    output_directory: str | Path,
    torch: Any,
) -> dict[str, Any]:
    """Train atomically, or source-replay an already complete bundle."""

    plan, _ = _read_pinned(
        promotion_plan_path,
        expected_promotion_plan_sha256,
        "locked promotion plan",
    )
    plan = promotion.validate_locked_promotion_plan(plan)
    candidate = _load_candidate_manifest(
        candidate_bundle_directory=candidate_bundle_directory,
        expected_manifest_sha256=expected_candidate_manifest_sha256,
    )
    dataset, _ = _read_pinned(
        development_dataset_path,
        expected_development_dataset_sha256,
        "ABR development dataset",
    )
    dataset = abr.validate_development_dataset(
        dataset,
        promotion_plan=plan,
        candidate_bundle_manifest=candidate,
    )
    teacher_receipt: dict[str, Any] | None = None
    if production_provenance_required:
        if (
            raw_examples_path is None
            or expected_raw_examples_sha256 is None
            or teacher_receipt_path is None
            or expected_teacher_receipt_sha256 is None
        ):
            raise ValueError(
                "ABR production training requires raw examples and teacher receipt"
            )
        raw_document, _ = _read_pinned(
            raw_examples_path,
            expected_raw_examples_sha256,
            "ABR raw examples",
        )
        raw_document = validate_raw_examples_document(raw_document)
        teacher_receipt, _ = _read_pinned(
            teacher_receipt_path,
            expected_teacher_receipt_sha256,
            "ABR real-teacher receipt",
        )
        from . import hu_m31_t3_abr_teacher_v1 as abr_teacher

        teacher_receipt = abr_teacher.require_production_teacher_coverage(
            teacher_receipt,
            raw_document=raw_document,
            raw_file_sha256=expected_raw_examples_sha256,
        )
        replayed_dataset = abr.build_development_dataset(
            promotion_plan=plan,
            candidate_bundle_manifest=candidate,
            raw_examples=raw_document["examples"],
        )
        if dataset != replayed_dataset:
            raise ValueError(
                "ABR production dataset does not replay from the real teacher receipt"
            )
    destination = Path(output_directory)
    resumed = destination.exists()
    if resumed:
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("ABR output directory is unsafe")
        bundle, bundle_raw = _read_canonical(
            destination / "bundle.json", "ABR policy bundle"
        )
        bundle = abr.validate_policy_bundle(
            bundle,
            directory=destination,
            promotion_plan=plan,
            candidate_bundle_directory=candidate_bundle_directory,
            torch=torch,
        )
        if production_provenance_required:
            build_receipt, build_receipt_raw = _read_canonical(
                destination / ABR_PRODUCTION_BUILD_RECEIPT_FILE,
                "ABR production build receipt",
            )
            validate_production_build_receipt(
                build_receipt,
                bundle_file_sha256=_sha256(bundle_raw),
                bundle_identity_sha256=bundle["bundle_identity_sha256"],
                expected_teacher_receipt_file_sha256=(
                    expected_teacher_receipt_sha256
                ),
                expected_dataset_file_sha256=(
                    expected_development_dataset_sha256
                ),
            )
        else:
            build_receipt = None
            build_receipt_raw = b""
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(
            f".{destination.name}.build-{os.getpid()}"
        )
        if staging.exists() or staging.is_symlink():
            raise FileExistsError(
                f"ABR staging directory already exists: {staging}"
            )
        # Deliberately retain a failed staging directory for forensic replay.
        bundle = abr.write_policy_bundle(
            promotion_plan=plan,
            candidate_bundle_directory=candidate_bundle_directory,
            development_dataset_path=development_dataset_path,
            output_directory=staging,
            torch=torch,
        )
        bundle_raw = (staging / "bundle.json").read_bytes()
        if production_provenance_required:
            assert teacher_receipt is not None
            assert expected_teacher_receipt_sha256 is not None
            assert expected_raw_examples_sha256 is not None
            build_receipt = build_production_build_receipt(
                teacher_receipt=teacher_receipt,
                teacher_receipt_file_sha256=(
                    expected_teacher_receipt_sha256
                ),
                raw_examples_file_sha256=expected_raw_examples_sha256,
                development_dataset=dataset,
                development_dataset_file_sha256=(
                    expected_development_dataset_sha256
                ),
                bundle=bundle,
                bundle_file_sha256=_sha256(bundle_raw),
            )
            _write_once_or_replay(
                staging / ABR_PRODUCTION_BUILD_RECEIPT_FILE,
                build_receipt,
            )
            build_receipt_raw = (
                staging / ABR_PRODUCTION_BUILD_RECEIPT_FILE
            ).read_bytes()
        else:
            build_receipt = None
            build_receipt_raw = b""
        os.replace(staging, destination)
        stored, bundle_raw = _read_canonical(
            destination / "bundle.json", "ABR policy bundle"
        )
        if stored != bundle:
            raise ValueError("atomically published ABR bundle changed")
        bundle = abr.validate_policy_bundle(
            stored,
            directory=destination,
            promotion_plan=plan,
            candidate_bundle_directory=candidate_bundle_directory,
            torch=torch,
        )
        if production_provenance_required:
            assert build_receipt is not None
            validate_production_build_receipt(
                build_receipt,
                bundle_file_sha256=_sha256(bundle_raw),
                bundle_identity_sha256=bundle["bundle_identity_sha256"],
                expected_teacher_receipt_file_sha256=(
                    expected_teacher_receipt_sha256
                ),
                expected_dataset_file_sha256=(
                    expected_development_dataset_sha256
                ),
            )
    result = {
        "schema": ABR_OPERATION_RECEIPT_SCHEMA,
        "operation": "train_three_policy_bundle",
        "status": "resume_complete" if resumed else "complete",
        "promotion_plan_file_sha256": expected_promotion_plan_sha256,
        "candidate_manifest_file_sha256": expected_candidate_manifest_sha256,
        "development_dataset_file_sha256": (
            expected_development_dataset_sha256
        ),
        "abr_bundle_file_sha256": _sha256(bundle_raw),
        "abr_bundle_identity_sha256": bundle["bundle_identity_sha256"],
        "family_count": bundle["family_count"],
        "response_ids": [
            record["response_id"] for record in bundle["families"]
        ],
        "frozen_before_locked_evaluation": True,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }
    if production_provenance_required:
        assert teacher_receipt is not None
        assert build_receipt is not None
        result.update(
            {
                "teacher_receipt_file_sha256": (
                    expected_teacher_receipt_sha256
                ),
                "teacher_receipt_identity_sha256": teacher_receipt[
                    "teacher_receipt_identity_sha256"
                ],
                "raw_examples_file_sha256": expected_raw_examples_sha256,
                "production_build_receipt_file_sha256": _sha256(
                    build_receipt_raw
                ),
                "production_build_receipt_identity_sha256": build_receipt[
                    "production_build_receipt_identity_sha256"
                ],
            }
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3.1 three-family ABR artifact operations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset = subparsers.add_parser("freeze-dataset")
    bundle = subparsers.add_parser("train-bundle")
    for command in (dataset, bundle):
        command.add_argument("--promotion-plan", required=True)
        command.add_argument("--expected-promotion-plan-sha256", required=True)
        command.add_argument("--candidate-bundle-directory", required=True)
        command.add_argument(
            "--expected-candidate-manifest-sha256", required=True
        )
    dataset.add_argument("--raw-examples", required=True)
    dataset.add_argument("--expected-raw-examples-sha256", required=True)
    dataset.add_argument("--teacher-receipt", required=True)
    dataset.add_argument("--expected-teacher-receipt-sha256", required=True)
    dataset.add_argument("--output", required=True)
    bundle.add_argument("--development-dataset", required=True)
    bundle.add_argument("--expected-development-dataset-sha256", required=True)
    bundle.add_argument("--raw-examples", required=True)
    bundle.add_argument("--expected-raw-examples-sha256", required=True)
    bundle.add_argument("--teacher-receipt", required=True)
    bundle.add_argument("--expected-teacher-receipt-sha256", required=True)
    bundle.add_argument("--output-directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze-dataset":
            result = freeze_development_dataset(
                promotion_plan_path=args.promotion_plan,
                expected_promotion_plan_sha256=(
                    args.expected_promotion_plan_sha256
                ),
                candidate_bundle_directory=args.candidate_bundle_directory,
                expected_candidate_manifest_sha256=(
                    args.expected_candidate_manifest_sha256
                ),
                raw_examples_path=args.raw_examples,
                expected_raw_examples_sha256=(
                    args.expected_raw_examples_sha256
                ),
                teacher_receipt_path=args.teacher_receipt,
                expected_teacher_receipt_sha256=(
                    args.expected_teacher_receipt_sha256
                ),
                output_path=args.output,
            )
        else:
            import torch

            result = train_three_policy_bundle(
                promotion_plan_path=args.promotion_plan,
                expected_promotion_plan_sha256=(
                    args.expected_promotion_plan_sha256
                ),
                candidate_bundle_directory=args.candidate_bundle_directory,
                expected_candidate_manifest_sha256=(
                    args.expected_candidate_manifest_sha256
                ),
                development_dataset_path=args.development_dataset,
                expected_development_dataset_sha256=(
                    args.expected_development_dataset_sha256
                ),
                raw_examples_path=args.raw_examples,
                expected_raw_examples_sha256=(
                    args.expected_raw_examples_sha256
                ),
                teacher_receipt_path=args.teacher_receipt,
                expected_teacher_receipt_sha256=(
                    args.expected_teacher_receipt_sha256
                ),
                output_directory=args.output_directory,
                torch=torch,
            )
    except Exception as exc:  # pragma: no cover - subprocess boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABR_OPERATION_RECEIPT_SCHEMA",
    "ABR_PRODUCTION_BUILD_RECEIPT_FILE",
    "ABR_PRODUCTION_BUILD_RECEIPT_SCHEMA",
    "RAW_EXAMPLES_SCHEMA",
    "build_production_build_receipt",
    "build_raw_examples_document",
    "freeze_development_dataset",
    "main",
    "train_three_policy_bundle",
    "validate_production_build_receipt",
    "validate_raw_examples_document",
    "write_raw_examples_document",
]
