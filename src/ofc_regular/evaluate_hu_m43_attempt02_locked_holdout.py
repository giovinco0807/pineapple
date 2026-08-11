"""Consume and evaluate the inherited M4.3 Attempt02 locked holdout once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .hu_m43_attempt02_contract import (
    _find_consumption_markers,
    load_and_validate_attempt02_plan,
)
from .hu_m43_attempt02_lifecycle import (
    M43_ATTEMPT02_LOCKED_RECEIPT_SCHEMA,
    M43_ATTEMPT02_MARKER_SCHEMA,
    file_sha256,
    read_json_mapping,
    resolve_contract_path,
    self_digest,
    validate_attempt02_data_contract,
    validate_attempt02_freeze,
    validate_attempt02_locked_receipt,
    write_immutable_json,
)
from .hu_m43_joint_model_loader import load_hu_m43_joint_action_model
from .hu_m43_joint_model_v4 import HuM43JointModelV4
from .hu_m43_pilot_contract import _ordered_shard_binding
from .train_hu_m4_joint_model import prepare_teacher_samples, read_teacher_jsonl


def evaluate_attempt02_locked_holdout_once(
    *,
    model_path: str | Path,
    freeze_manifest_path: str | Path,
    data_contract_path: str | Path,
    plan_path: str | Path,
    population_plan_path: str | Path,
    repo_root: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Claim the canonical marker, then run one fixed-threshold diagnostic."""

    root = Path(repo_root).resolve()
    model_source = Path(model_path).resolve()
    freeze_source = Path(freeze_manifest_path).resolve()
    contract_source = Path(data_contract_path).resolve()
    plan_source = Path(plan_path).resolve()
    population_plan_source = Path(population_plan_path).resolve()
    receipt_output = Path(receipt_path).resolve()
    fixed_inputs = {
        model_source,
        freeze_source,
        contract_source,
        plan_source,
        population_plan_source,
    }
    if len(fixed_inputs) != 5 or receipt_output in fixed_inputs:
        raise ValueError("Attempt02 one-shot inputs and receipt must be distinct")
    if receipt_output.exists():
        raise FileExistsError(f"Attempt02 locked receipt already exists: {receipt_output}")

    plan = load_and_validate_attempt02_plan(plan_source)
    plan_sha = file_sha256(plan_source)
    from .validate_hu_m43_attempt02_acceptance import (
        load_and_validate_attempt02_population_plan,
    )

    population_plan = load_and_validate_attempt02_population_plan(
        population_plan_source
    )
    population_plan_sha = file_sha256(population_plan_source)
    if population_plan["attempt02_training_plan"]["file_sha256"] != plan_sha:
        raise ValueError("Attempt02 population plan targets another training plan")
    contract = read_json_mapping(contract_source, "Attempt02 data contract")
    validate_attempt02_data_contract(
        contract,
        plan=plan,
        plan_sha256=plan_sha,
    )
    freeze = read_json_mapping(freeze_source, "Attempt02 freeze")
    validate_attempt02_freeze(freeze, contract=contract)
    if freeze.get("plan_sha256") != plan_sha:
        raise ValueError("Attempt02 one-shot plan SHA disagrees with freeze")
    if (
        freeze.get("population_plan_file_sha256") != population_plan_sha
        or freeze.get("population_plan_resolved_path")
        != str(population_plan_source)
    ):
        raise ValueError("Attempt02 one-shot population plan disagrees with freeze")
    if freeze.get("data_contract_file_sha256") != file_sha256(contract_source):
        raise ValueError("Attempt02 one-shot contract bytes disagree with freeze")
    if freeze.get("data_contract_resolved_path") != str(contract_source):
        raise ValueError("Attempt02 one-shot contract path disagrees with freeze")
    training_manifest_path = Path(
        str(freeze.get("training_manifest_resolved_path", ""))
    ).resolve()
    if file_sha256(training_manifest_path) != freeze.get("training_manifest_sha256"):
        raise ValueError("Attempt02 one-shot training manifest disagrees with freeze")
    model = load_hu_m43_joint_action_model(
        model_source,
        expected_sha256=str(freeze["model_sha256"]),
        freeze_manifest=freeze,
        training_manifest_path=training_manifest_path,
    )
    if not isinstance(model, HuM43JointModelV4):
        raise TypeError("Attempt02 one-shot requires a v4 model")

    locked_path = resolve_contract_path(
        root,
        contract["inherited_locked"]["path"],
        "Attempt02 inherited locked holdout",
    )
    plan_locked = resolve_contract_path(
        root,
        plan["inherited_locked"]["path"],
        "Attempt02 plan inherited locked holdout",
    )
    if locked_path != plan_locked:
        raise ValueError("Attempt02 inherited locked path disagrees with plan")
    marker_output = resolve_contract_path(
        root,
        contract["global_locked_consumption"]["canonical_marker_path"],
        "Attempt02 canonical consumption marker",
    )
    if str(marker_output) != freeze.get(
        "canonical_consumption_marker_resolved_path"
    ):
        raise ValueError("Attempt02 canonical marker path disagrees with freeze")
    if marker_output in fixed_inputs or marker_output == receipt_output:
        raise ValueError("Attempt02 canonical marker aliases another lifecycle path")
    marker_hits = _find_consumption_markers(
        root, str(contract["inherited_locked"]["identity_sha256"])
    )
    if marker_hits:
        raise FileExistsError(
            "Attempt02 inherited locked identity already has a global marker"
        )

    # No locked file stat/hash/open/JSON parse occurs above this point.  The
    # exclusive marker is the irreversible boundary; any later crash consumes
    # this inherited holdout and forces a fresh one for another attempt.
    marker: dict[str, Any] = {
        "schema": M43_ATTEMPT02_MARKER_SCHEMA,
        "status": "claimed_before_inherited_locked_content_read",
        "freeze_sha256": freeze["freeze_sha256"],
        "freeze_file_sha256": file_sha256(freeze_source),
        "data_contract_sha256": contract["contract_sha256"],
        "data_contract_file_sha256": file_sha256(contract_source),
        "population_plan_file_sha256": population_plan_sha,
        "model_sha256": freeze["model_sha256"],
        "locked_identity_sha256": contract["inherited_locked"][
            "identity_sha256"
        ],
        "locked_file_sha256": contract["inherited_locked"]["file_sha256"],
        "locked_bytes": contract["inherited_locked"]["bytes"],
        "locked_ordered_shards_sha256": contract["inherited_locked"][
            "ordered_shards_sha256"
        ],
        "evaluation_pass_count": 1,
        "claim_is_consuming_even_on_crash": True,
    }
    marker["marker_sha256"] = self_digest(marker, "marker_sha256")
    write_immutable_json(marker_output, marker)

    rows, audited_binding = _audit_and_read_locked_rows(
        locked_path,
        repo_root=root,
        expected=contract["inherited_locked"],
    )
    samples = prepare_teacher_samples(rows)
    diagnostic = _evaluate_v4_fixed_threshold(model, samples)
    receipt: dict[str, Any] = {
        "schema": M43_ATTEMPT02_LOCKED_RECEIPT_SCHEMA,
        "status": "evaluated_once_diagnostic_only_no_activation",
        "freeze_sha256": freeze["freeze_sha256"],
        "freeze_file_sha256": file_sha256(freeze_source),
        "data_contract_sha256": contract["contract_sha256"],
        "data_contract_file_sha256": file_sha256(contract_source),
        "population_plan_file_sha256": population_plan_sha,
        "model_sha256": freeze["model_sha256"],
        "frozen_threshold": float(freeze["frozen_threshold"]),
        "locked_identity_sha256": contract["inherited_locked"][
            "identity_sha256"
        ],
        "locked_file_sha256": audited_binding["ordered_shards"][0][
            "file_sha256"
        ],
        "locked_bytes": audited_binding["ordered_shards"][0]["bytes"],
        "locked_ordered_shards_sha256": audited_binding[
            "ordered_shards_sha256"
        ],
        "consumption_marker_resolved_path": str(marker_output),
        "consumption_marker_file_sha256": file_sha256(marker_output),
        "consumption_marker_canonical_sha256": marker["marker_sha256"],
        "evaluation_pass_count": 1,
        "threshold_search_performed": False,
        "threshold_reselection_performed": False,
        "model_selection_performed": False,
        "feature_selection_performed": False,
        "current_profile_resolved": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "policy_promoted": False,
        "requires_fresh_population_acceptance": True,
        "minimum_population_valid_overrides": int(
            freeze["minimum_population_valid_overrides"]
        ),
        "teacher_value_status": (
            "diagnostic_only_not_realized_match_ev_not_runtime_gate"
        ),
        "locked_diagnostic": diagnostic,
    }
    receipt["receipt_sha256"] = self_digest(receipt, "receipt_sha256")
    validate_attempt02_locked_receipt(
        receipt,
        freeze=freeze,
        contract=contract,
        marker=marker,
    )
    write_immutable_json(receipt_output, receipt)
    return receipt


def _audit_and_read_locked_rows(
    path: Path,
    *,
    repo_root: Path,
    expected: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open locked bytes only after the caller has claimed the marker."""

    if not path.is_file():
        raise ValueError("Attempt02 inherited locked file is missing after claim")
    if path.stat().st_size != expected.get("bytes"):
        raise ValueError("Attempt02 inherited locked byte size mismatch after claim")
    if file_sha256(path) != expected.get("file_sha256"):
        raise ValueError("Attempt02 inherited locked file SHA mismatch after claim")
    binding = _ordered_shard_binding(
        (path,), repo_root=repo_root, expected_split="locked_holdout"
    )
    first = binding["ordered_shards"][0]
    comparisons = {
        "records": binding["records"],
        "canonical_rows_sha256": binding["canonical_rows_sha256"],
        "ordered_shards_sha256": binding["ordered_shards_sha256"],
        "file_sha256": first["file_sha256"],
        "bytes": first["bytes"],
        "identity_sha256": first["identity_sha256"],
    }
    for field, actual in comparisons.items():
        if actual != expected.get(field):
            raise ValueError(
                f"Attempt02 inherited locked hash-chain mismatch after claim: {field}"
            )
    rows = read_teacher_jsonl(path)
    if len(rows) != expected.get("records"):
        raise ValueError("Attempt02 inherited locked parsed row count mismatch")
    return rows, binding


def _evaluate_v4_fixed_threshold(
    model: HuM43JointModelV4,
    samples: Sequence[Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    top1 = 0
    for sample in samples:
        baseline = sample.baseline_index
        heads = model.predict_heads_sample(
            sample.policy_sample, baseline_index=baseline
        )
        proposal = model.top_nonbaseline_index(
            sample.policy_sample,
            baseline_index=baseline,
            predictions=heads,
        )
        probability = model.predict_safety_probability(
            sample.policy_sample,
            candidate_index=proposal,
            baseline_index=baseline,
        )
        fired = bool(
            model.safety_enabled
            and probability >= float(model.safety_threshold)
        )
        paired = np.asarray(sample.teacher_paired_delta_mean, dtype=np.float64)
        true_best = float(np.max(paired))
        top1 += int(float(paired[proposal]) >= true_best - 1.0e-9)
        rows.append(
            {
                "seat": sample.seat,
                "proposal_index": proposal,
                "baseline_index": baseline,
                "safety_probability": float(probability),
                "fired": fired,
                "teacher_delta": float(paired[proposal]),
                "downside_loss_p95": float(sample.downside_loss_p95[proposal]),
                "downside_loss_p99": float(sample.downside_loss_p99[proposal]),
                "downside_loss_max": float(sample.downside_loss_max[proposal]),
            }
        )
    return {
        "split": "inherited_locked_holdout",
        "samples": len(samples),
        "proposal_top1_accuracy": float(top1 / len(samples)) if samples else 0.0,
        "frozen_threshold": float(model.safety_threshold),
        "safety_enabled": bool(model.safety_enabled),
        "override_metrics": _fixed_threshold_metrics(rows),
        "seat_override_metrics": {
            seat: _fixed_threshold_metrics(
                [row for row in rows if row["seat"] == seat]
            )
            for seat in ("first", "second")
        },
        "threshold_search_performed": False,
        "teacher_value_status": (
            "diagnostic_only_not_realized_match_ev_not_runtime_gate"
        ),
    }


def _fixed_threshold_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fired = [row for row in rows if row.get("fired") is True]
    deltas = np.asarray(
        [float(row["teacher_delta"]) for row in fired], dtype=np.float64
    )
    losses = deltas[deltas <= 0.0]
    return {
        "states": len(rows),
        "fires": len(fired),
        "fire_rate": float(len(fired) / len(rows)) if rows else 0.0,
        "false_positives": int(losses.size),
        "false_positive_rate": (
            float(losses.size / len(fired)) if fired else 0.0
        ),
        "teacher_delta_sum": float(np.sum(deltas)) if deltas.size else 0.0,
        "teacher_mean_delta_per_fire": (
            float(np.mean(deltas)) if deltas.size else 0.0
        ),
        "teacher_p05_delta": (
            float(np.quantile(deltas, 0.05)) if deltas.size else 0.0
        ),
        "teacher_min_delta": float(np.min(deltas)) if deltas.size else 0.0,
        "p95_loss": (
            max(float(row["downside_loss_p95"]) for row in fired)
            if fired
            else 0.0
        ),
        "p99_loss": (
            max(float(row["downside_loss_p99"]) for row in fired)
            if fired
            else 0.0
        ),
        "max_loss": (
            max(float(row["downside_loss_max"]) for row in fired)
            if fired
            else 0.0
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--population-plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = evaluate_attempt02_locked_holdout_once(
        model_path=args.model,
        freeze_manifest_path=args.freeze_manifest,
        data_contract_path=args.data_contract,
        plan_path=args.plan,
        population_plan_path=args.population_plan,
        repo_root=args.repo_root,
        receipt_path=args.receipt,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["evaluate_attempt02_locked_holdout_once"]
