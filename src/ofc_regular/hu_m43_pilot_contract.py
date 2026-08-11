"""Fail-closed data and lifecycle contract for the bounded M4.3 pilot.

M4.3 is a pre-promotion experiment.  It may select a model and a safety
threshold using fresh train/calibration data, but it must not inspect the
locked holdout until both are frozen.  Even a positive locked result is not a
replacement for the separately seeded population evaluation required by the
M4 acceptance gate.

The module deliberately does not train or activate a policy.  It validates the
frozen plan, excludes every available M4/M4.1/M4.2 teacher identity, derives a
deterministic safety-fit/threshold-lock partition, and checks the paired
candidate-versus-baseline risk targets consumed by the M4.3 learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .audit_hu_m4_t1_data import audit_locked_splits
from .hu_m4_t1_teacher import M4_PAIRED_DELTA_SUMMARY_SCHEMA


M43_PILOT_PLAN_SCHEMA = "hu_m43_bounded_pilot_plan_v1"
M43_DATA_CONTRACT_SCHEMA = "hu_m43_bounded_pilot_data_contract_v1"
M43_FREEZE_SCHEMA = "hu_m43_model_threshold_freeze_v1"
M43_LOCKED_RECEIPT_SCHEMA = "hu_m43_locked_holdout_one_shot_receipt_v1"
M43_CALIBRATION_PARTITION_SCHEMA = "hu_m43_calibration_partition_v1"
M43_TEACHER_SHARD_BINDING_SCHEMA = "hu_m43_ordered_teacher_shards_v1"

_SPLITS = ("train", "calibration", "locked_holdout")
_HEX = frozenset("0123456789abcdef")


def load_and_validate_plan(path: str | Path) -> dict[str, Any]:
    """Load the frozen plan and reject any weakened M4.3 boundary."""

    source = Path(path)
    plan = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict):
        raise ValueError("M4.3 plan must be a mapping")
    _validate_plan(plan)
    return plan


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != M43_PILOT_PLAN_SCHEMA:
        raise ValueError("unsupported M4.3 pilot plan schema")
    if plan.get("status") != "frozen_pre_generation":
        raise ValueError("M4.3 plan must be frozen before teacher generation")
    if plan.get("milestone") != "M4.3":
        raise ValueError("M4.3 plan milestone mismatch")

    budget = _mapping(plan.get("budget"), "budget")
    if budget.get("kind") != "bounded_pre_promotion_pilot":
        raise ValueError("M4.3 budget must remain a pre-promotion pilot")
    if _integer(budget.get("roots"), "budget.roots", minimum=1) != 200:
        raise ValueError("M4.3 frozen pilot must contain exactly 200 roots")
    if _integer(budget.get("max_roots"), "budget.max_roots", minimum=1) > 1000:
        raise ValueError("M4.3 bounded pilot may not exceed 1,000 roots")
    roots_per_shard = _integer(
        budget.get("roots_per_shard"), "budget.roots_per_shard", minimum=1
    )
    if roots_per_shard != 10:
        raise ValueError("M4.3 frozen shard size must remain 10 roots")
    if _integer(budget.get("shards"), "budget.shards", minimum=1) != 20:
        raise ValueError("M4.3 frozen pilot must contain 20 resumable shards")

    split_plan = _mapping(plan.get("splits"), "splits")
    if set(split_plan) != set(_SPLITS):
        raise ValueError("M4.3 plan requires train/calibration/locked_holdout")
    expected_roots = {"train": 100, "calibration": 60, "locked_holdout": 40}
    all_planned_seeds: set[int] = set()
    for split, expected in expected_roots.items():
        spec = _mapping(split_plan.get(split), f"splits.{split}")
        roots = _integer(spec.get("roots"), f"splits.{split}.roots", minimum=1)
        if roots != expected or roots % roots_per_shard:
            raise ValueError(f"M4.3 {split} root/shard allocation changed")
        seed_start = _integer(
            spec.get("seed_start"), f"splits.{split}.seed_start", minimum=0
        )
        seed_stride = _integer(
            spec.get("seed_stride"), f"splits.{split}.seed_stride", minimum=1
        )
        seeds = {seed_start + seed_stride * index for index in range(roots)}
        if all_planned_seeds & seeds:
            raise ValueError("M4.3 planned split hand seeds overlap")
        all_planned_seeds.update(seeds)
        allowed_use = spec.get("allowed_use")
        expected_use = {
            "train": "ranker_fit_and_nested_oof_only",
            "calibration": "safety_fit_and_threshold_lock_only",
            "locked_holdout": "one_shot_after_model_and_threshold_freeze",
        }[split]
        if allowed_use != expected_use:
            raise ValueError(f"M4.3 {split} allowed-use boundary changed")

    calibration = _mapping(plan.get("calibration_partition"), "calibration_partition")
    if calibration.get("schema") != M43_CALIBRATION_PARTITION_SCHEMA:
        raise ValueError("M4.3 calibration partition schema mismatch")
    if calibration.get("method") != "profile_stratified_identity_hash_v1":
        raise ValueError("M4.3 calibration partition method changed")
    if _integer(calibration.get("safety_fit_roots"), "safety_fit_roots") != 30:
        raise ValueError("M4.3 safety-fit allocation must remain 30")
    if _integer(calibration.get("threshold_lock_roots"), "threshold_lock_roots") != 30:
        raise ValueError("M4.3 threshold-lock allocation must remain 30")
    if calibration.get("threshold_selection_source") != "calibration.threshold_lock":
        raise ValueError("thresholds may be selected only on threshold-lock")
    if calibration.get("locked_holdout_used") is not False:
        raise ValueError("locked holdout may not participate in calibration")
    if _integer(
        calibration.get("minimum_threshold_lock_fires_for_positive_pilot_signal"),
        "minimum_threshold_lock_fires_for_positive_pilot_signal",
    ) != 10:
        raise ValueError("M4.3 positive pilot signal requires 10 threshold-lock fires")
    if calibration.get("insufficient_fires_decision") != (
        "no_go_insufficient_pilot_signal"
    ):
        raise ValueError("M4.3 insufficient calibration fires must remain No-Go")

    search = _mapping(plan.get("teacher_search"), "teacher_search")
    if _integer(search.get("candidate_samples"), "candidate_samples") != 2:
        raise ValueError("M4.3 candidate sample count changed")
    if _integer(search.get("evaluation_samples"), "evaluation_samples") != 16:
        raise ValueError("M4.3 paired-label evaluation sample count changed")
    if search.get("common_random_futures") is not True:
        raise ValueError("M4.3 requires common random evaluation futures")
    if search.get("candidate_evaluation_rng_disjoint") is not True:
        raise ValueError("M4.3 candidate/evaluation RNGs must be disjoint")

    profiles = plan.get("root_population")
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)):
        raise ValueError("M4.3 root population missing")
    expected_profiles = {
        "stage19_p0",
        "stage9f_p2",
        "stage7_m5_r10",
        "stage3_baseline",
        "random_exact_final",
    }
    observed_profiles: set[str] = set()
    split_totals = Counter()
    for row in profiles:
        profile = _mapping(row, "root_population entry")
        name = str(profile.get("profile", ""))
        if name in observed_profiles or name not in expected_profiles:
            raise ValueError("M4.3 root population is missing, duplicated, or unsafe")
        observed_profiles.add(name)
        allocation = _mapping(profile.get("roots"), f"root_population.{name}.roots")
        expected_allocation = {"train": 20, "calibration": 12, "locked_holdout": 8}
        if dict(allocation) != expected_allocation:
            raise ValueError(f"M4.3 profile allocation changed for {name}")
        split_totals.update({key: int(value) for key, value in allocation.items()})
    if observed_profiles != expected_profiles:
        raise ValueError("M4.3 root population must contain the five frozen profiles")
    if dict(split_totals) != expected_roots:
        raise ValueError("M4.3 root population totals disagree with split budget")

    labels = _mapping(plan.get("paired_label_contract"), "paired_label_contract")
    expected_targets = {
        "paired_delta_mean": "paired_delta_vs_baseline.mean",
        "paired_delta_se": "paired_delta_vs_baseline.standard_error",
        "downside_loss_p95": "max(0,-paired_delta_vs_baseline.p05)",
        "downside_loss_p99": "max(0,-paired_delta_vs_baseline.p01)",
        "downside_loss_max": "max(0,-paired_delta_vs_baseline.min)",
    }
    if labels.get("targets") != expected_targets:
        raise ValueError("M4.3 paired mean/SE/tail target mapping changed")
    if labels.get("baseline_action_score_exact_zero") is not True:
        raise ValueError("M4.3 baseline action score must be exactly zero")
    if labels.get("teacher_values_runtime_allowed") is not False:
        raise ValueError("teacher values must not enter the runtime gate")
    if labels.get("teacher_lcb_runtime_allowed") is not False:
        raise ValueError("teacher LCB must not enter the runtime gate")
    if labels.get("teacher_metrics_are_realized_match_ev") is not False:
        raise ValueError("teacher metrics must not be labeled realized match EV")

    model = _mapping(plan.get("model_contract"), "model_contract")
    expected_model_contract = {
        "name": "baseline_paired_delta_risk_ensemble",
        "ranker_training_sources": ["train"],
        "ranker_crossfit_source": "train",
        "runtime_predictor": "crossfit_fold_ensemble",
        "full_refit_distribution_switch_allowed": False,
        "safety_calibrator_sources": ["train_oof", "calibration.safety_fit"],
        "threshold_selection_source": "calibration.threshold_lock",
        "model_selection_sources": ["train"],
        "locked_holdout_access_before_freeze_allowed": False,
    }
    if dict(model) != expected_model_contract:
        raise ValueError("M4.3 model/source/freeze contract changed")

    locked_protocol = _mapping(
        plan.get("locked_holdout_protocol"), "locked_holdout_protocol"
    )
    expected_locked_protocol = {
        "structural_identity_audit_allowed": True,
        "training_or_threshold_label_access_before_freeze_allowed": False,
        "required_freeze_schema": M43_FREEZE_SCHEMA,
        "evaluation_pass_count": 1,
        "threshold_or_model_reselection_after_result_allowed": False,
        "fresh_locked_data_required_after_any_reselection": True,
    }
    if dict(locked_protocol) != expected_locked_protocol:
        raise ValueError("M4.3 one-shot locked-holdout protocol changed")

    guards = _mapping(plan.get("activation_guards"), "activation_guards")
    for key in (
        "current_profile_changed",
        "runtime_policy_activated",
        "full_replacement_enabled",
        "large_scale_authorized",
        "m5_authorized",
    ):
        if guards.get(key) is not False:
            raise ValueError(f"M4.3 activation guard must remain false: {key}")

    acceptance = _mapping(plan.get("promotion_boundary"), "promotion_boundary")
    if acceptance.get("pilot_can_promote_policy") is not False:
        raise ValueError("the bounded M4.3 pilot cannot promote a policy")
    if _integer(
        acceptance.get("minimum_population_valid_overrides"),
        "minimum_population_valid_overrides",
    ) < 300:
        raise ValueError("M4 population override gate may not be lowered")
    if acceptance.get("fresh_separately_seeded_population_evaluation") is not True:
        raise ValueError("M4.3 requires a fresh population acceptance evaluation")
    if acceptance.get("teacher_metrics_are_diagnostic_only") is not True:
        raise ValueError("teacher metrics may not be reported as realized match EV")
    if acceptance.get("acceptance_gates_may_be_lowered") is not False:
        raise ValueError("M4 acceptance gates may not be lowered")

    freshness = _mapping(plan.get("freshness_exclusions"), "freshness_exclusions")
    if freshness.get("reject_hand_seed_or_fingerprint_overlap") is not True:
        raise ValueError("M4.3 freshness must reject seed or fingerprint reuse")
    sources = freshness.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise ValueError("M4.3 prior-data exclusion sources missing")
    milestones = {str(_mapping(row, "exclusion source").get("milestone")) for row in sources}
    if milestones != {"M4", "M4.1", "M4.2"}:
        raise ValueError("M4.3 exclusions must cover M4, M4.1, and M4.2")


def audit_frozen_exclusions(
    plan: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    """Verify frozen prior shard hashes and return their identity union."""

    _validate_plan(plan)
    root = Path(repo_root)
    sources = _mapping(plan["freshness_exclusions"], "freshness_exclusions")[
        "sources"
    ]
    identities: set[tuple[int, str]] = set()
    seeds: set[int] = set()
    fingerprints: set[str] = set()
    reports: list[dict[str, Any]] = []
    for raw_source in sources:
        source = _mapping(raw_source, "exclusion source")
        relative = str(source.get("path", ""))
        path = (root / relative).resolve()
        if not path.is_file():
            raise ValueError(f"M4.3 exclusion source missing: {relative}")
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_sha = _sha256_text(source.get("sha256"), f"{relative}.sha256")
        if actual_sha != expected_sha:
            raise ValueError(f"M4.3 exclusion source hash mismatch: {relative}")
        rows = _read_jsonl(path)
        expected_records = _integer(source.get("records"), f"{relative}.records")
        if len(rows) != expected_records:
            raise ValueError(f"M4.3 exclusion source record count mismatch: {relative}")
        expected_split = str(source.get("split", ""))
        local_identities = set()
        for row in rows:
            if row.get("split") != expected_split:
                raise ValueError(f"M4.3 exclusion source split mismatch: {relative}")
            identity = _row_identity(row, relative)
            local_identities.add(identity)
            seeds.add(identity[0])
            fingerprints.add(identity[1])
        if len(local_identities) != len(rows):
            raise ValueError(f"M4.3 exclusion source has duplicate identities: {relative}")
        identities.update(local_identities)
        reports.append(
            {
                "milestone": source["milestone"],
                "path": relative,
                "records": len(rows),
                "sha256": actual_sha,
            }
        )
    return {
        "sources": reports,
        "source_records": sum(row["records"] for row in reports),
        "unique_identities": len(identities),
        "unique_hand_seeds": len(seeds),
        "unique_observation_fingerprints": len(fingerprints),
        "identity_sha256": _identity_digest(identities),
        "identities": identities,
        "hand_seeds": seeds,
        "observation_fingerprints": fingerprints,
    }


def paired_risk_targets(action: Mapping[str, Any]) -> dict[str, float]:
    """Validate and expose the M4.3 candidate-minus-baseline targets."""

    summary = _mapping(action.get("paired_delta_vs_baseline"), "paired delta")
    if summary.get("schema") != M4_PAIRED_DELTA_SUMMARY_SCHEMA:
        raise ValueError("M4.3 action lacks a paired-delta summary")
    values = {
        key: _finite(summary.get(key), f"paired_delta_vs_baseline.{key}")
        for key in ("mean", "standard_error", "min", "p01", "p05")
    }
    if values["standard_error"] < 0.0:
        raise ValueError("M4.3 paired-delta standard error is negative")
    if not values["min"] <= values["p01"] <= values["p05"]:
        raise ValueError("M4.3 paired downside quantiles are not monotone")
    flat_mean = _finite(action.get("delta_vs_baseline"), "delta_vs_baseline")
    flat_se = _finite(action.get("delta_se_vs_baseline"), "delta_se_vs_baseline")
    if not math.isclose(flat_mean, values["mean"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("M4.3 paired-delta mean field mismatch")
    if not math.isclose(flat_se, values["standard_error"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("M4.3 paired-delta SE field mismatch")
    return {
        "paired_delta_mean": values["mean"],
        "paired_delta_se": values["standard_error"],
        "downside_loss_p95": max(0.0, -values["p05"]),
        "downside_loss_p99": max(0.0, -values["p01"]),
        "downside_loss_max": max(0.0, -values["min"]),
    }


def audit_pilot_data(
    *,
    plan_path: str | Path,
    repo_root: str | Path,
    train: Sequence[str | Path],
    calibration: Sequence[str | Path],
    locked_holdout: Sequence[str | Path],
) -> dict[str, Any]:
    """Audit fresh M4.3 shards and create the immutable role assignment.

    This is an independent structural audit.  Its access to the locked file is
    not a model/threshold evaluation; downstream training must consume only the
    emitted train and calibration identities.  The one-shot evaluator is
    separately bound by :func:`validate_model_threshold_freeze`.
    """

    plan_source = Path(plan_path)
    plan = load_and_validate_plan(plan_source)
    prior = audit_frozen_exclusions(plan, repo_root=repo_root)
    base_audit = audit_locked_splits(
        train=train,
        calibration=calibration,
        locked_holdout=locked_holdout,
        require_paired_delta=True,
    )
    paths = {
        "train": tuple(Path(path) for path in train),
        "calibration": tuple(Path(path) for path in calibration),
        "locked_holdout": tuple(Path(path) for path in locked_holdout),
    }
    rows = {split: _read_many_jsonl(split_paths) for split, split_paths in paths.items()}
    teacher_shards = {
        "schema": M43_TEACHER_SHARD_BINDING_SCHEMA,
        "splits": {
            split: _ordered_shard_binding(
                paths[split], repo_root=Path(repo_root), expected_split=split
            )
            for split in _SPLITS
        },
    }
    teacher_shards["all_splits_sha256"] = _canonical_sha256(teacher_shards)
    expected_counts = {"train": 100, "calibration": 60, "locked_holdout": 40}
    identities: dict[str, set[tuple[int, str]]] = {}
    profile_counts: dict[str, Counter[str]] = {}
    target_counts: dict[str, int] = {}
    for split in _SPLITS:
        if len(rows[split]) != expected_counts[split]:
            raise ValueError(f"M4.3 {split} record count disagrees with frozen plan")
        identities[split] = set()
        profile_counts[split] = Counter()
        target_count = 0
        spec = _mapping(plan["splits"][split], f"split {split}")
        planned_seeds = {
            int(spec["seed_start"]) + int(spec["seed_stride"]) * index
            for index in range(int(spec["roots"]))
        }
        observed_seeds = set()
        for row in rows[split]:
            identity = _row_identity(row, split)
            identities[split].add(identity)
            observed_seeds.add(identity[0])
            provenance = _mapping(row.get("provenance"), f"{split} provenance")
            if provenance.get("current_profile_resolved") is not False:
                raise ValueError("M4.3 data resolved the current profile")
            profile_counts[split][str(provenance.get("root_profile", ""))] += 1
            search = _mapping(row.get("search_config"), f"{split} search_config")
            if search.get("candidate_samples") != 2 or search.get("evaluation_samples") != 16:
                raise ValueError("M4.3 data does not use frozen c2/e16 search")
            actions = row.get("actions")
            if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
                raise ValueError("M4.3 action rows missing")
            baseline_key = str(row.get("baseline_action_key", ""))
            baseline_count = 0
            for action in actions:
                target = paired_risk_targets(_mapping(action, "action"))
                target_count += 1
                if action.get("action_key") == baseline_key:
                    baseline_count += 1
                    if any(value != 0.0 for value in target.values()):
                        raise ValueError("M4.3 baseline targets are not exactly zero")
            if baseline_count != 1:
                raise ValueError("M4.3 baseline action mapping is not unique")
        if len(identities[split]) != len(rows[split]):
            raise ValueError(f"M4.3 {split} contains duplicate identities")
        if observed_seeds != planned_seeds:
            raise ValueError(f"M4.3 {split} hand seeds disagree with frozen schedule")
        target_counts[split] = target_count

    for left_index, left in enumerate(_SPLITS):
        for right in _SPLITS[left_index + 1 :]:
            if identities[left] & identities[right]:
                raise ValueError(f"M4.3 identity overlap: {left} vs {right}")
    current_seeds = {seed for values in identities.values() for seed, _ in values}
    current_fingerprints = {
        fingerprint for values in identities.values() for _, fingerprint in values
    }
    prior_seed_overlap = current_seeds & prior["hand_seeds"]
    prior_fingerprint_overlap = current_fingerprints & prior["observation_fingerprints"]
    if prior_seed_overlap or prior_fingerprint_overlap:
        raise ValueError(
            "M4.3 reuses an M4/M4.1/M4.2 hand seed or observation fingerprint"
        )

    expected_profile_counts = {
        split: {
            str(profile["profile"]): int(profile["roots"][split])
            for profile in plan["root_population"]
        }
        for split in _SPLITS
    }
    for split in _SPLITS:
        if dict(profile_counts[split]) != expected_profile_counts[split]:
            raise ValueError(f"M4.3 {split} root-profile quota mismatch")

    calibration_partition = _partition_calibration(rows["calibration"], plan)
    clean_prior = {key: value for key, value in prior.items() if not isinstance(value, set)}
    clean_prior["audit_sha256"] = _canonical_sha256(clean_prior)
    base_audit_sha256 = _canonical_sha256(base_audit)
    report: dict[str, Any] = {
        "schema": M43_DATA_CONTRACT_SCHEMA,
        "status": "pass_fresh_data_sealed_for_model_freeze",
        "plan_sha256": hashlib.sha256(plan_source.read_bytes()).hexdigest(),
        "base_audit": base_audit,
        "base_audit_sha256": base_audit_sha256,
        "prior_exclusions": clean_prior,
        "teacher_shards": teacher_shards,
        "teacher_shards_all_splits_sha256": teacher_shards[
            "all_splits_sha256"
        ],
        "splits": {
            split: {
                "records": len(rows[split]),
                "paired_action_targets": target_counts[split],
                "identity_sha256": _identity_digest(identities[split]),
                "profile_counts": dict(sorted(profile_counts[split].items())),
            }
            for split in _SPLITS
        },
        "calibration_partition": calibration_partition,
        "freshness": {
            "cross_split_identity_overlap": 0,
            "prior_hand_seed_overlap": 0,
            "prior_observation_fingerprint_overlap": 0,
        },
        "locked_holdout": {
            "structural_audit_allowed": True,
            "model_or_threshold_label_access_count": 0,
            "allowed_next_access": "one_shot_after_model_and_threshold_freeze",
        },
        "runtime": {
            "current_profile_resolved": False,
            "policy_activated": False,
            "full_replacement": False,
        },
    }
    report["contract_sha256"] = _canonical_sha256(report)
    return report


def validate_data_contract_binding(
    data_contract: Mapping[str, Any],
    *,
    plan_path: str | Path,
    repo_root: str | Path,
    train: Sequence[str | Path] | None = None,
    calibration: Sequence[str | Path] | None = None,
    locked_holdout: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Verify a sealed contract against its plan, exclusions, and shard bytes.

    Callers may verify only the split paths they are authorized to consume.
    The trainer supplies train/calibration and never receives the locked path;
    the one-shot evaluator supplies only locked_holdout.  Every supplied split
    is matched in exact order and by file SHA plus canonical-row digest.
    """

    _validate_data_contract_digest(data_contract)
    plan_source = Path(plan_path)
    plan = load_and_validate_plan(plan_source)
    plan_sha = hashlib.sha256(plan_source.read_bytes()).hexdigest()
    if data_contract.get("plan_sha256") != plan_sha:
        raise ValueError("M4.3 data contract frozen-plan SHA mismatch")

    declared_prior = _mapping(
        data_contract.get("prior_exclusions"), "prior_exclusions"
    )
    declared_prior_sha = _sha256_text(
        declared_prior.get("audit_sha256"), "prior_exclusions.audit_sha256"
    )
    unsigned_prior = dict(declared_prior)
    unsigned_prior.pop("audit_sha256", None)
    if _canonical_sha256(unsigned_prior) != declared_prior_sha:
        raise ValueError("M4.3 prior-freshness audit digest mismatch")
    actual_prior = audit_frozen_exclusions(plan, repo_root=repo_root)
    clean_actual_prior = {
        key: value for key, value in actual_prior.items() if not isinstance(value, set)
    }
    if _canonical_sha256(clean_actual_prior) != declared_prior_sha:
        raise ValueError("M4.3 prior-freshness audit no longer matches sources")

    base_audit = _mapping(data_contract.get("base_audit"), "base_audit")
    if base_audit.get("schema") != "hu_m4_t1_second_data_audit_v1":
        raise ValueError("M4.3 embedded base audit schema mismatch")
    if base_audit.get("status") != "pass":
        raise ValueError("M4.3 embedded base audit did not pass")
    if _canonical_sha256(base_audit) != data_contract.get("base_audit_sha256"):
        raise ValueError("M4.3 embedded base audit digest mismatch")
    gates = _mapping(base_audit.get("gates"), "base_audit.gates")
    for gate in (
        "hidden_discard_safe",
        "all_legal_actions_mapped",
        "candidate_evaluation_rng_disjoint",
        "paired_common_future_delta_contract",
        "seed_ranges_disjoint",
        "fingerprints_disjoint",
    ):
        if gates.get(gate) is not True:
            raise ValueError(f"M4.3 embedded base audit gate failed: {gate}")
    if gates.get("current_profile_resolved") is not False:
        raise ValueError("M4.3 embedded base audit resolved current")
    if gates.get("holdout_threshold_search_allowed") is not False:
        raise ValueError("M4.3 embedded base audit allows holdout threshold search")

    teacher_shards = _mapping(data_contract.get("teacher_shards"), "teacher_shards")
    if teacher_shards.get("schema") != M43_TEACHER_SHARD_BINDING_SCHEMA:
        raise ValueError("M4.3 teacher shard binding schema mismatch")
    declared_all_sha = _sha256_text(
        teacher_shards.get("all_splits_sha256"), "teacher_shards.all_splits_sha256"
    )
    unsigned_teacher = dict(teacher_shards)
    unsigned_teacher.pop("all_splits_sha256", None)
    if _canonical_sha256(unsigned_teacher) != declared_all_sha:
        raise ValueError("M4.3 all-split teacher binding digest mismatch")
    declared_splits = _mapping(teacher_shards.get("splits"), "teacher_shards.splits")

    requested = {
        "train": train,
        "calibration": calibration,
        "locked_holdout": locked_holdout,
    }
    verified: dict[str, Any] = {}
    for split, split_paths in requested.items():
        if split_paths is None:
            continue
        actual = _ordered_shard_binding(
            tuple(Path(path) for path in split_paths),
            repo_root=Path(repo_root),
            expected_split=split,
        )
        declared = _mapping(declared_splits.get(split), f"teacher_shards.{split}")
        if actual != declared:
            raise ValueError(
                f"M4.3 {split} ordered path/content binding mismatch"
            )
        verified[split] = {
            "ordered_shards_sha256": actual["ordered_shards_sha256"],
            "canonical_rows_sha256": actual["canonical_rows_sha256"],
            "records": actual["records"],
        }
    if not verified:
        raise ValueError("M4.3 data contract validation requires an authorized split")
    return {
        "schema": "hu_m43_data_contract_binding_validation_v1",
        "status": "pass",
        "contract_sha256": data_contract["contract_sha256"],
        "plan_sha256": plan_sha,
        "prior_freshness_audit_sha256": declared_prior_sha,
        "base_audit_sha256": data_contract["base_audit_sha256"],
        "teacher_shards_all_splits_sha256": declared_all_sha,
        "verified_splits": verified,
    }


def _partition_calibration(
    rows: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]
) -> dict[str, Any]:
    spec = _mapping(plan["calibration_partition"], "calibration_partition")
    salt = str(spec.get("identity_hash_salt", ""))
    if not salt:
        raise ValueError("M4.3 calibration identity hash salt is empty")
    by_profile: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for row in rows:
        seed, fingerprint = _row_identity(row, "calibration")
        provenance = _mapping(row.get("provenance"), "calibration provenance")
        profile = str(provenance.get("root_profile", ""))
        token = hashlib.sha256(
            f"{M43_CALIBRATION_PARTITION_SCHEMA}\0{salt}\0{seed}\0{fingerprint}".encode()
        ).hexdigest()
        by_profile[profile].append((token, seed, fingerprint))
    safety_fit: list[tuple[int, str]] = []
    threshold_lock: list[tuple[int, str]] = []
    safety_profile_counts: Counter[str] = Counter()
    threshold_profile_counts: Counter[str] = Counter()
    for profile, values in sorted(by_profile.items()):
        ordered = sorted(values)
        if len(ordered) != 12:
            raise ValueError(f"M4.3 calibration profile quota is not 12: {profile}")
        safety_fit.extend((seed, fingerprint) for _, seed, fingerprint in ordered[:6])
        threshold_lock.extend((seed, fingerprint) for _, seed, fingerprint in ordered[6:])
        safety_profile_counts[profile] += 6
        threshold_profile_counts[profile] += 6
    if len(safety_fit) != 30 or len(threshold_lock) != 30:
        raise ValueError("M4.3 calibration role totals disagree with frozen plan")
    if set(safety_fit) & set(threshold_lock):
        raise ValueError("M4.3 safety-fit and threshold-lock identities overlap")
    return {
        "schema": M43_CALIBRATION_PARTITION_SCHEMA,
        "method": "profile_stratified_identity_hash_v1",
        "safety_fit": {
            "records": 30,
            "identity_sha256": _identity_digest(safety_fit),
            "profile_counts": dict(sorted(safety_profile_counts.items())),
            "identities": [_identity_json(value) for value in sorted(safety_fit)],
        },
        "threshold_lock": {
            "records": 30,
            "identity_sha256": _identity_digest(threshold_lock),
            "profile_counts": dict(sorted(threshold_profile_counts.items())),
            "identities": [_identity_json(value) for value in sorted(threshold_lock)],
        },
        "overlap": 0,
    }


def partition_calibration_rows(
    rows: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Public deterministic calibration partition used by the trainer."""

    _validate_plan(plan)
    return _partition_calibration(rows, plan)


def validate_model_threshold_freeze(
    manifest: Mapping[str, Any], *, data_contract: Mapping[str, Any]
) -> None:
    """Require model/calibrator/threshold freeze before holdout access."""

    if manifest.get("schema") != M43_FREEZE_SCHEMA:
        raise ValueError("M4.3 model-threshold freeze schema mismatch")
    _validate_data_contract_digest(data_contract)
    if manifest.get("status") != "model_and_threshold_frozen_locked_unopened":
        raise ValueError("M4.3 model/threshold freeze status mismatch")
    if manifest.get("data_contract_sha256") != data_contract.get("contract_sha256"):
        raise ValueError("M4.3 freeze does not bind the audited data contract")
    if not str(manifest.get("data_contract_resolved_path", "")):
        raise ValueError("M4.3 freeze data-contract path binding is empty")
    marker_path = str(manifest.get("locked_consumption_marker_resolved_path", ""))
    if not marker_path or Path(marker_path).name != "M43_LOCKED_CONSUMED.json":
        raise ValueError("M4.3 freeze consumption-marker path binding is invalid")
    for key in (
        "plan_sha256",
        "base_audit_sha256",
        "teacher_shards_all_splits_sha256",
    ):
        _sha256_text(manifest.get(key), key)
        if manifest.get(key) != data_contract.get(key):
            raise ValueError(f"M4.3 freeze/data-contract binding mismatch: {key}")
    declared_prior = _mapping(
        data_contract.get("prior_exclusions"), "prior_exclusions"
    )
    if manifest.get("prior_freshness_audit_sha256") != declared_prior.get(
        "audit_sha256"
    ):
        raise ValueError("M4.3 freeze prior-freshness binding mismatch")
    _sha256_text(manifest.get("training_manifest_sha256"), "training_manifest_sha256")
    _sha256_text(manifest.get("model_sha256"), "model_sha256")
    if not str(manifest.get("model_id", "")):
        raise ValueError("M4.3 frozen model_id is empty")
    threshold = _finite(manifest.get("frozen_threshold"), "frozen_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("M4.3 frozen threshold is outside [0, 1]")
    expected = {
        "ranker_training_sources": ["train"],
        "ranker_crossfit_source": "train",
        "safety_calibrator_sources": ["train_oof", "calibration.safety_fit"],
        "threshold_selection_source": "calibration.threshold_lock",
        "model_selection_sources": ["train"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"M4.3 unsafe freeze source declaration: {key}")
    if manifest.get("locked_holdout_used_for_training_selection_or_threshold") is not False:
        raise ValueError("M4.3 locked holdout was used before freeze")
    if manifest.get("model_or_threshold_locked_label_access_count_at_freeze") != 0:
        raise ValueError(
            "M4.3 locked holdout labels reached model or threshold selection "
            "before freeze"
        )
    for key in ("current_profile_resolved", "runtime_policy_activated"):
        if manifest.get(key) is not False:
            raise ValueError(f"M4.3 freeze violates runtime guard: {key}")
    if manifest.get("pilot_can_promote_policy") is not False:
        raise ValueError("M4.3 freeze may not promote the bounded pilot")
    if manifest.get("fresh_population_acceptance_required") is not True:
        raise ValueError("M4.3 freeze must require fresh population acceptance")
    if _integer(
        manifest.get("minimum_population_valid_overrides"),
        "minimum_population_valid_overrides",
    ) < 300:
        raise ValueError("M4.3 freeze lowers the population override gate")


def validate_locked_holdout_receipt(
    receipt: Mapping[str, Any],
    *,
    freeze_manifest: Mapping[str, Any],
    data_contract: Mapping[str, Any],
) -> None:
    """Validate the only permitted M4.3 locked-holdout evaluation receipt."""

    validate_model_threshold_freeze(freeze_manifest, data_contract=data_contract)
    if receipt.get("schema") != M43_LOCKED_RECEIPT_SCHEMA:
        raise ValueError("M4.3 locked receipt schema mismatch")
    if receipt.get("status") != "evaluated_once_diagnostic_only_no_activation":
        raise ValueError("M4.3 locked receipt status mismatch")
    if receipt.get("freeze_manifest_sha256") != _canonical_sha256(freeze_manifest):
        raise ValueError("M4.3 locked receipt is not bound to the freeze manifest")
    if receipt.get("data_contract_sha256") != data_contract.get("contract_sha256"):
        raise ValueError("M4.3 locked receipt data-contract binding mismatch")
    if receipt.get("model_sha256") != freeze_manifest.get("model_sha256"):
        raise ValueError("M4.3 locked receipt model binding mismatch")
    if receipt.get("frozen_threshold") != freeze_manifest.get("frozen_threshold"):
        raise ValueError("M4.3 locked receipt threshold binding mismatch")
    expected_locked_digest = data_contract["splits"]["locked_holdout"][
        "identity_sha256"
    ]
    if receipt.get("locked_identity_sha256") != expected_locked_digest:
        raise ValueError("M4.3 locked receipt identity digest mismatch")
    expected_shard_digest = data_contract["teacher_shards"]["splits"][
        "locked_holdout"
    ]["ordered_shards_sha256"]
    if receipt.get("locked_teacher_shards_sha256") != expected_shard_digest:
        raise ValueError("M4.3 locked receipt teacher-shard binding mismatch")
    _sha256_text(receipt.get("consumption_marker_sha256"), "consumption_marker_sha256")
    if receipt.get("evaluation_pass_count") != 1:
        raise ValueError("M4.3 locked holdout is strictly one-shot")
    for key in (
        "threshold_search_performed",
        "model_selection_performed",
        "feature_selection_performed",
        "current_profile_resolved",
        "runtime_policy_activated",
        "policy_promoted",
    ):
        if receipt.get(key) is not False:
            raise ValueError(f"M4.3 locked receipt violates guard: {key}")
    if receipt.get("requires_fresh_population_acceptance") is not True:
        raise ValueError("M4.3 locked receipt bypasses population acceptance")
    if _integer(
        receipt.get("minimum_population_valid_overrides"),
        "minimum_population_valid_overrides",
    ) < 300:
        raise ValueError("M4.3 locked receipt lowers the population fire gate")
    if receipt.get("teacher_value_status") != "diagnostic_not_realized_match_ev":
        raise ValueError("M4.3 locked receipt mislabels teacher diagnostics")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _integer(value: Any, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _finite(value: Any, location: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{location} must be finite")
    return number


def _sha256_text(value: Any, location: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return text


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a mapping")
            rows.append(row)
    return rows


def _read_many_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [row for path in paths for row in _read_jsonl(path)]


def _ordered_shard_binding(
    paths: Sequence[Path], *, repo_root: Path, expected_split: str
) -> dict[str, Any]:
    ordered_shards: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        source = path.resolve()
        if not source.is_file():
            raise ValueError(f"M4.3 teacher shard missing: {path}")
        rows = _read_jsonl(source)
        if any(row.get("split") != expected_split for row in rows):
            raise ValueError(f"M4.3 teacher shard split mismatch: {path}")
        canonical_rows_sha = _canonical_rows_sha256(rows)
        identities = [_row_identity(row, str(path)) for row in rows]
        ordered_shards.append(
            {
                "index": index,
                "path": _path_token(source, repo_root),
                "bytes": source.stat().st_size,
                "records": len(rows),
                "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "canonical_rows_sha256": canonical_rows_sha,
                "identity_sha256": _identity_digest(identities),
            }
        )
        all_rows.extend(rows)
    if not ordered_shards:
        raise ValueError(f"M4.3 {expected_split} shard list is empty")
    payload: dict[str, Any] = {
        "ordered_shards": ordered_shards,
        "records": len(all_rows),
        "canonical_rows_sha256": _canonical_rows_sha256(all_rows),
    }
    payload["ordered_shards_sha256"] = _canonical_sha256(payload)
    return payload


def build_ordered_teacher_shard_binding(
    paths: Sequence[str | Path], *, repo_root: str | Path, expected_split: str
) -> dict[str, Any]:
    """Build the exact ordered byte/content binding used by the data contract."""

    if expected_split not in _SPLITS:
        raise ValueError(f"unsupported M4.3 split: {expected_split!r}")
    return _ordered_shard_binding(
        tuple(Path(path) for path in paths),
        repo_root=Path(repo_root),
        expected_split=expected_split,
    )


def _path_token(path: Path, repo_root: Path) -> str:
    resolved_root = repo_root.resolve()
    try:
        return path.resolve().relative_to(resolved_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _canonical_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_data_contract_digest(data_contract: Mapping[str, Any]) -> None:
    if data_contract.get("schema") != M43_DATA_CONTRACT_SCHEMA:
        raise ValueError("M4.3 data contract schema mismatch")
    if data_contract.get("status") != "pass_fresh_data_sealed_for_model_freeze":
        raise ValueError("M4.3 data contract is not sealed for model freeze")
    unsigned_contract = dict(data_contract)
    declared_contract_sha = _sha256_text(
        unsigned_contract.pop("contract_sha256", None), "contract_sha256"
    )
    if _canonical_sha256(unsigned_contract) != declared_contract_sha:
        raise ValueError("M4.3 data contract digest mismatch")


def _row_identity(row: Mapping[str, Any], location: str) -> tuple[int, str]:
    seed = _integer(row.get("hand_seed"), f"{location}.hand_seed")
    fingerprint = _sha256_text(
        row.get("observation_fingerprint"), f"{location}.observation_fingerprint"
    )
    return seed, fingerprint


def _identity_digest(values: Iterable[tuple[int, str]]) -> str:
    encoded = "".join(f"{seed}\t{fingerprint}\n" for seed, fingerprint in sorted(values))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _identity_json(value: tuple[int, str]) -> dict[str, Any]:
    return {"hand_seed": value[0], "observation_fingerprint": value[1]}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_manifest_sha256(value: Mapping[str, Any]) -> str:
    """Digest a lifecycle manifest using the contract's canonical JSON form."""

    return _canonical_sha256(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    parser.add_argument("--locked-holdout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = audit_pilot_data(
        plan_path=args.plan,
        repo_root=args.repo_root,
        train=args.train,
        calibration=args.calibration,
        locked_holdout=args.locked_holdout,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "M43_CALIBRATION_PARTITION_SCHEMA",
    "M43_DATA_CONTRACT_SCHEMA",
    "M43_FREEZE_SCHEMA",
    "M43_LOCKED_RECEIPT_SCHEMA",
    "M43_PILOT_PLAN_SCHEMA",
    "M43_TEACHER_SHARD_BINDING_SCHEMA",
    "audit_frozen_exclusions",
    "audit_pilot_data",
    "build_ordered_teacher_shard_binding",
    "canonical_manifest_sha256",
    "load_and_validate_plan",
    "paired_risk_targets",
    "partition_calibration_rows",
    "validate_data_contract_binding",
    "validate_locked_holdout_receipt",
    "validate_model_threshold_freeze",
]
