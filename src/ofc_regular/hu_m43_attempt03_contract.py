"""Immutable data/experiment contract for the M4.3 Attempt03 pilot.

Attempt02's 200 train rows are inherited as fit data.  Its 100 calibration
rows and the inherited locked40 stay opaque: this module verifies their bytes
and obtains their already-sealed identities from prior contracts, but never
parses either JSONL file.  Fresh ``train.fit`` and one-shot
``train.precal_holdout`` shards are sealed by :func:`finalize_fresh_data`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit_hu_m4_t1_data import read_and_audit_shard
from .hu_m43_attempt02_contract import (
    INHERITED_LOCKED_IDENTITY_SHA256,
    _audit_preflight as _audit_attempt02_preflight,
    _file_sha256,
    _find_consumption_markers,
    _mapping,
    _path_token,
    _read_mapping,
    _require_bound_path,
    _self_digest,
    _validate_embedded_digest,
    load_and_validate_attempt02_plan,
)
from .hu_m43_pilot_contract import (
    _identity_digest,
    _identity_json,
    _ordered_shard_binding,
    _read_jsonl,
    _row_identity,
    canonical_manifest_sha256,
)


M43_ATTEMPT03_PLAN_SCHEMA = "hu_m43_attempt03_plan_v1"
M43_ATTEMPT03_PREFLIGHT_SCHEMA = "hu_m43_attempt03_preflight_receipt_v1"
M43_ATTEMPT03_DATA_CONTRACT_SCHEMA = "hu_m43_attempt03_data_contract_v1"
M43_ATTEMPT03_TEACHER_SHARDS_SCHEMA = "hu_m43_attempt03_ordered_teacher_shards_v1"

_ROLES = ("train.fit", "train.precal_holdout")
_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
_EXPECTED_SPLITS = {
    "train.fit": {
        "roots": 500,
        "seed_start": 9_106_071_901,
        "candidate_seed_start": 9_906_071_901,
        "evaluation_seed_start": 10_706_071_901,
        "child_policy_seed_start": 11_506_071_901,
        "allowed_use": "ranker_fit_and_grouped_crossfit_only",
    },
    "train.precal_holdout": {
        "roots": 200,
        "seed_start": 9_506_071_901,
        "candidate_seed_start": 10_306_071_901,
        "evaluation_seed_start": 11_106_071_901,
        "child_policy_seed_start": 11_906_071_901,
        "allowed_use": "one_shot_precalibration_go_no_go_only",
    },
}
_SEED_STRIDE = 1_000_003
_ROOTS_PER_SHARD = 10


def load_and_validate_attempt03_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict):
        raise ValueError("M4.3 Attempt03 plan must be a mapping")
    _validate_plan(plan)
    return plan


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != M43_ATTEMPT03_PLAN_SCHEMA:
        raise ValueError("unsupported M4.3 Attempt03 plan schema")
    if plan.get("milestone") != "M4.3-attempt03":
        raise ValueError("M4.3 Attempt03 milestone mismatch")
    if plan.get("status") != "frozen_pre_generation":
        raise ValueError("Attempt03 plan must be frozen before generation")
    if plan.get("fixed_baseline_profile") != "stage18_p1":
        raise ValueError("Attempt03 baseline profile changed")
    continuation = _mapping(plan.get("fixed_continuation"), "fixed_continuation")
    if continuation.get("t2_profile") != "stage9f_p2":
        raise ValueError("Attempt03 T2 continuation changed")

    budget = _mapping(plan.get("budget"), "budget")
    expected_budget = {
        "kind": "bounded_pre_go_pilot",
        "fresh_roots": 700,
        "max_fresh_roots": 1000,
        "roots_per_shard": _ROOTS_PER_SHARD,
        "fresh_shards": 70,
        "inherited_train_fit_roots": 200,
        "sealed_calibration_roots": 100,
        "inherited_locked_roots": 40,
        "large_scale_allowed": False,
    }
    for key, expected in expected_budget.items():
        if budget.get(key) != expected:
            raise ValueError(f"Attempt03 frozen budget changed: {key}")
    if budget.get("spot_allowed_after_preflight") is not True:
        raise ValueError("Attempt03 Spot use must remain preflight-gated")

    splits = _mapping(plan.get("fresh_splits"), "fresh_splits")
    if set(splits) != set(_ROLES):
        raise ValueError("Attempt03 fresh roles must be fit and one-shot pre-cal")
    schedules: list[tuple[str, set[int]]] = []
    for role in _ROLES:
        spec = _mapping(splits.get(role), f"fresh_splits.{role}")
        expected = _EXPECTED_SPLITS[role]
        for key, value in expected.items():
            if spec.get(key) != value:
                raise ValueError(f"Attempt03 {role} frozen field changed: {key}")
        if spec.get("record_split") != "train":
            raise ValueError(f"Attempt03 {role} records must retain split=train")
        if spec.get("seed_stride") != _SEED_STRIDE:
            raise ValueError(f"Attempt03 {role} seed stride changed")
        if spec.get("phase_seed_stride_scope") != "split_shard":
            raise ValueError(f"Attempt03 {role} phase seeds must advance per shard")
        roots = int(spec["roots"])
        for field in (
            "seed_start",
            "candidate_seed_start",
            "evaluation_seed_start",
            "child_policy_seed_start",
        ):
            count = roots if field == "seed_start" else roots // _ROOTS_PER_SHARD
            schedules.append(
                (
                    f"{role}.{field}",
                    {int(spec[field]) + _SEED_STRIDE * index for index in range(count)},
                )
            )
    _reject_schedule_overlap(schedules, "Attempt03")

    search = _mapping(plan.get("teacher_search"), "teacher_search")
    expected_search = {
        "candidate_samples": 2,
        "evaluation_samples": 64,
        "common_random_futures": True,
        "candidate_evaluation_rng_disjoint": True,
        "candidate_seed_namespace": "hu-m43-attempt03-candidate-v1",
        "evaluation_seed_namespace": "hu-m43-attempt03-evaluation-v1",
        "child_policy_seed_namespace": "hu-m43-attempt03-child-v1",
        "namespace_domain_separation": "split_shard",
        "batch_child_selectors": True,
        "native_batch_threads": 4,
    }
    if dict(search) != expected_search:
        raise ValueError("Attempt03 c2/e64 RNG namespace contract changed")

    population = plan.get("root_population")
    if not isinstance(population, Sequence) or isinstance(population, (str, bytes)):
        raise ValueError("Attempt03 root population missing")
    expected_quota = {
        "inherited_train_fit": 40,
        "train.fit": 100,
        "train.precal_holdout": 40,
        "sealed_calibration": 20,
        "inherited_locked": 8,
    }
    seen: set[str] = set()
    for raw in population:
        row = _mapping(raw, "root_population entry")
        profile = str(row.get("profile", ""))
        if profile not in _PROFILES or profile in seen:
            raise ValueError("Attempt03 root population changed or duplicated")
        seen.add(profile)
        if dict(_mapping(row.get("roots"), f"roots.{profile}")) != expected_quota:
            raise ValueError(f"Attempt03 profile quota changed: {profile}")
    if seen != set(_PROFILES):
        raise ValueError("Attempt03 must retain all five profiles")

    provenance = _mapping(plan.get("attempt02_provenance"), "attempt02_provenance")
    expected_paths = {
        "plan": "configs/hu_joint_policy_m43_attempt02.json",
        "data_contract": (
            "outputs/hu_joint_policy/m43_attempt02_teacher/"
            "regular-hu-m43-attempt02-c2e64-pilot300-20260713-2201/"
            "data_contract.json"
        ),
        "teacher_receipt": (
            "outputs/hu_joint_policy/m43_attempt02_teacher/"
            "regular-hu-m43-attempt02-c2e64-pilot300-20260713-2201/receipt.json"
        ),
        "model_receipt": (
            "outputs/hu_joint_policy/m43_attempt02_model_runs/"
            "regular-hu-m43-attempt02-v4-model-r2-20260714-0117/receipt.json"
        ),
        "training_manifest": (
            "outputs/hu_joint_policy/m43_attempt02_model_runs/"
            "regular-hu-m43-attempt02-v4-model-r2-20260714-0117/"
            "training_manifest.json"
        ),
        "train_fit": (
            "outputs/hu_joint_policy/m43_attempt02_teacher/"
            "regular-hu-m43-attempt02-c2e64-pilot300-20260713-2201/train.jsonl"
        ),
        "sealed_calibration": (
            "outputs/hu_joint_policy/m43_attempt02_teacher/"
            "regular-hu-m43-attempt02-c2e64-pilot300-20260713-2201/"
            "calibration.jsonl"
        ),
    }
    for name, expected in expected_paths.items():
        if _mapping(provenance.get(name), name).get("path") != expected:
            raise ValueError(f"Attempt03 Attempt02 {name} path changed")
    expected_bindings = {
        "plan": "7b321604b14bc4c448a8140a772c5365f44e94172153524a42c1bcb91f78e025",
        "data_contract": "bcff8d6b0616dbc395d6c897488c01631353aec4e3fd6be5e6e5f617d0601664",
        "teacher_receipt": "c2025e4cf8ec2919156f96f2d716b2ab28dde5d6bc7d2455bc3207793ac27415",
        "model_receipt": "6d04af784850fe7c8fa38087e93d96c584bc1091132eae41eee4d13083d22654",
        "training_manifest": "6b061a690006ec21d4f52305bc21a2fd7c8ec0cd374fe47ad22dced1d7ce84ae",
    }
    for name, digest in expected_bindings.items():
        if _mapping(provenance.get(name), name).get("file_sha256") != digest:
            raise ValueError(f"Attempt03 Attempt02 {name} binding changed")
    train = _mapping(provenance.get("train_fit"), "train_fit")
    calibration = _mapping(provenance.get("sealed_calibration"), "sealed_calibration")
    expected_train = {
        "records": 200,
        "bytes": 6_474_349,
        "file_sha256": "0a55541b26305c5dfe50b9122263bb92ca1ee60a5f085d2020b38b31a4cdb996",
        "identity_sha256": "623112de0b7a8af1357782f6f80a09de8dfc370bcf8faadb3e5c23ec03d80372",
        "canonical_rows_sha256": "ce8d41f383d1c591c3f8ae5c2fdc218bdfaa1a55e40f49b87ab9361ad067943d",
        "ordered_shards_sha256": "28784143a257afab0bd39c593a8f8957566cf25f9bf9f0257ac5aad0fbcd1b4a",
        "allowed_use": "ranker_fit_and_grouped_crossfit_only",
    }
    if any(train.get(key) != expected for key, expected in expected_train.items()):
        raise ValueError("Attempt03 inherited Attempt02 train200 binding changed")
    expected_calibration = {
        "records": 100,
        "bytes": 3_308_076,
        "file_sha256": "907df8ae301a2c8d4e791cf8d4529ee6017f4085e587b8a18d884d418eaab2f6",
        "identity_sha256": "d037408b7cdc1ccc70832dd50914edcb0d8b2596caa02e9f79824c1ce29f9c2d",
        "canonical_rows_sha256": "e4a63daa428104a0c2fd82765b0bf2b93068bfd6166da030d638dab3e63cecf8",
        "ordered_shards_sha256": "8cbad463009d7e15dbf620f166095423f39254342385a4dcc06d89b9019ccdc2",
    }
    if any(
        calibration.get(key) != expected
        for key, expected in expected_calibration.items()
    ):
        raise ValueError("Attempt03 sealed calibration hash binding changed")
    if (
        calibration.get("classification"),
        calibration.get("records"),
        calibration.get("identity_sha256"),
        calibration.get("content_open_before_precalibration_go_allowed"),
        calibration.get("structural_byte_hash_audit_only"),
    ) != (
        "inherited_unopened",
        100,
        "d037408b7cdc1ccc70832dd50914edcb0d8b2596caa02e9f79824c1ce29f9c2d",
        False,
        True,
    ):
        raise ValueError("Attempt03 unopened calibration boundary changed")
    roles = _mapping(calibration.get("roles"), "sealed_calibration.roles")
    expected_roles = {
        "safety_fit": (50, "b468dd1079cbbf5e958ee03a539309bb0d65b236e8a68fdcf17b586bfeeff696"),
        "threshold_lock": (50, "8874b690a757a946c2c2f87db9407b5292b1e186f3130848aa84cb15bb8e18c0"),
    }
    if set(roles) != set(expected_roles):
        raise ValueError("Attempt03 calibration roles changed")
    for role, (records, digest) in expected_roles.items():
        spec = _mapping(roles[role], role)
        if (spec.get("records"), spec.get("identity_sha256")) != (records, digest):
            raise ValueError(f"Attempt03 calibration role binding changed: {role}")
    prior = _mapping(provenance.get("prior_exclusion_union"), "prior exclusion")
    if dict(prior) != {
        "records": 412,
        "unique_identities": 412,
        "identity_sha256": "c7598ba0f74528562b79966e53dd843fe2225f23033b791377327f6ea6ab3b3e",
    }:
        raise ValueError("Attempt03 prior exclusion union binding changed")

    _validate_model_and_gates(plan)
    locked = _mapping(plan.get("inherited_locked"), "inherited_locked")
    if (
        locked.get("classification"),
        locked.get("records"),
        locked.get("identity_sha256"),
        locked.get("content_open_before_frozen_model_and_threshold_allowed"),
    ) != ("inherited_unopened", 40, INHERITED_LOCKED_IDENTITY_SHA256, False):
        raise ValueError("Attempt03 inherited locked40 boundary changed")
    expected_locked = {
        "path": (
            "outputs/hu_joint_policy/m43_spot/"
            "regular-hu-m43-c2e16-pilot200-20260713-1810/locked_holdout.jsonl"
        ),
        "bytes": 1_144_214,
        "file_sha256": "0a88e1cc8b079906e0ca14ddf4a8c3df7b108a09d0f363c7cb665cab89b40a1e",
        "canonical_rows_sha256": "fa99408bc4be031a7b4f2a8dfe05f4af07d90ebb7450e1eb03ba1fa596a87890",
        "ordered_shards_sha256": "6c351acd69cf0ce546fa3ff74e397d0091f4e0cd2ffc5b2ce95fc22a418bcaf2",
        "structural_byte_hash_audit_only": True,
        "global_consumption_marker": (
            "outputs/hu_joint_policy/m43_locked_consumption/"
            f"{INHERITED_LOCKED_IDENTITY_SHA256}/M43_LOCKED_CONSUMED.json"
        ),
    }
    if any(locked.get(key) != expected for key, expected in expected_locked.items()):
        raise ValueError("Attempt03 inherited locked40 hash binding changed")
    freshness = _mapping(plan.get("freshness"), "freshness")
    if any(value is not True for value in freshness.values()):
        raise ValueError("Attempt03 freshness rejection was weakened")
    lifecycle = _mapping(plan.get("holdout_lifecycle"), "holdout_lifecycle")
    if any(value is not True for value in lifecycle.values()):
        raise ValueError("Attempt03 holdout lifecycle was weakened")
    guards = _mapping(plan.get("activation_guards"), "activation_guards")
    if any(value is not False for value in guards.values()):
        raise ValueError("Attempt03 activation guard must remain false")


def _validate_model_and_gates(plan: Mapping[str, Any]) -> None:
    model = _mapping(plan.get("model_contract"), "model_contract")
    if model.get("name") != "hu_m43_t1_joint_model_v5":
        raise ValueError("Attempt03 model name changed")
    if model.get("state_grouped_folds") != 5:
        raise ValueError("Attempt03 must retain five state-grouped folds")
    if model.get("base_heads") != [
        "delta_huber",
        "positive",
        "downside_p95_huber",
        "downside_p99_huber",
        "downside_max_huber",
        "disagreement",
    ]:
        raise ValueError("Attempt03 robust base heads changed")
    proposal = _mapping(model.get("proposal_head"), "proposal_head")
    expected_proposal = {
        "features": "stage18_aware_22",
        "implementation": "lightgbm",
        "objective": "huber",
        "trees": 180,
        "learning_rate": 0.035,
        "num_leaves": 7,
        "min_child_samples": 50,
        "l2": 8.0,
        "l1": 0.5,
        "state_balanced": True,
    }
    if dict(proposal) != expected_proposal:
        raise ValueError("Attempt03 frozen proposal head changed")
    if model.get("hard_eligibility") != (
        "base_centered_delta_positive_vote_at_least_3_of_5"
    ):
        raise ValueError("Attempt03 hard eligibility changed")
    safety = _mapping(model.get("safety_calibrator"), "safety_calibrator")
    if dict(safety) != {
        "kind": "l2_logistic",
        "c": 0.25,
        "fit_after_hard_eligibility_only": True,
    }:
        raise ValueError("Attempt03 safety calibrator changed")
    if model.get("algorithm_change_after_precalibration_allowed") is not False:
        raise ValueError("Attempt03 permits post-precal algorithm change")
    expected_model_boundaries = {
        "fit_sources": ["attempt02.train", "train.fit"],
        "model_selection_sources": ["attempt02.train", "train.fit"],
        "precalibration_source": "train.precal_holdout",
        "calibration_sources_after_precalibration_go_only": [
            "calibration.safety_fit",
            "calibration.threshold_lock",
        ],
        "locked_holdout_access_before_model_and_threshold_freeze_allowed": False,
    }
    for key, expected in expected_model_boundaries.items():
        if model.get(key) != expected:
            raise ValueError(f"Attempt03 model data boundary changed: {key}")
    precal = _mapping(plan.get("precalibration_gate"), "precalibration_gate")
    required = {
        "one_shot": True,
        "roots": 200,
        "roots_per_profile": 40,
        "raw_proposal_positive_rate_min": 0.40,
        "eligible_fires_min": 30,
        "eligible_mean_delta_per_fire_strictly_positive": True,
        "eligible_delta_per_state_strictly_positive": True,
        "false_positive_delta_le_zero_max": 0.35,
        "cluster_lcb_confidence": 0.90,
        "cluster_lcb_mean_delta_per_fire_strictly_positive": True,
        "mapping_baseline_tie_overlap_gates_required": True,
        "sealed_calibration_opened": False,
        "failure_decision": "no_go_no_algorithm_or_gate_change",
    }
    for key, expected in required.items():
        if precal.get(key) != expected:
            raise ValueError(f"Attempt03 pre-calibration gate changed: {key}")
    tails = _mapping(precal.get("selected_action_downside_max"), "precal tails")
    if dict(tails) != {"p95": 25.0, "p99": 40.0, "max": 50.0}:
        raise ValueError("Attempt03 pre-calibration tail limits changed")
    calibration = _mapping(plan.get("calibration_gate"), "calibration_gate")
    expected_calibration_gate = {
        "open_only_after_precalibration_go": True,
        "safety_fit_records": 50,
        "threshold_lock_records": 50,
        "threshold_lock_fires_min": 10,
        "false_positive_delta_le_zero_max": 0.35,
        "delta_per_state_strictly_positive": True,
        "threshold_research_on_holdout_allowed": False,
    }
    for key, expected in expected_calibration_gate.items():
        if calibration.get(key) != expected:
            raise ValueError(f"Attempt03 calibration gate changed: {key}")
    if calibration.get("threshold_grid") != [
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    ]:
        raise ValueError("Attempt03 calibration threshold grid changed")
    tails = _mapping(
        calibration.get("selected_action_downside_max"), "calibration tails"
    )
    if dict(tails) != {"p95": 25.0, "p99": 40.0, "max": 50.0}:
        raise ValueError("Attempt03 calibration tail limits changed")


def build_preflight_receipt(
    *, plan_path: str | Path, repo_root: str | Path
) -> dict[str, Any]:
    report, _ = _audit_preflight(plan_path=plan_path, repo_root=repo_root)
    return report


def _audit_preflight(
    *, plan_path: str | Path, repo_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repo_root).resolve()
    plan_source = Path(plan_path).resolve()
    plan = load_and_validate_attempt03_plan(plan_source)
    provenance = _mapping(plan["attempt02_provenance"], "attempt02_provenance")

    attempt02_plan_path = _bound_declared(root, provenance, "plan")
    attempt02_plan = load_and_validate_attempt02_plan(attempt02_plan_path)
    old_provenance = _mapping(
        attempt02_plan["attempt01_provenance"], "attempt01_provenance"
    )
    old_sources = {
        str(row["split"]): row
        for row in attempt02_plan["freshness_exclusions"]["attempt01_sources"]
    }
    attempt02_preflight, old_context = _audit_attempt02_preflight(
        plan_path=attempt02_plan_path,
        repo_root=root,
        attempt01_data_contract=root / old_provenance["data_contract"]["path"],
        attempt01_training_manifest=root / old_provenance["training_manifest"]["path"],
        attempt01_train=root / old_sources["train"]["path"],
        attempt01_calibration=root / old_sources["calibration"]["path"],
        inherited_locked=root / attempt02_plan["inherited_locked"]["path"],
    )

    data_contract_path = _bound_declared(root, provenance, "data_contract")
    data_contract = _read_mapping(data_contract_path, "Attempt02 data contract")
    _validate_embedded_digest(
        data_contract,
        field="contract_sha256",
        expected=provenance["data_contract"]["canonical_sha256"],
        location="Attempt02 data contract",
    )
    if (
        data_contract.get("schema") != "hu_m43_attempt02_data_contract_v1"
        or data_contract.get("status")
        != "pass_fresh_train_calibration_sealed_inherited_locked_unopened"
    ):
        raise ValueError("Attempt02 data contract is not a sealed success")
    if data_contract.get("plan_sha256") != provenance["plan"]["file_sha256"]:
        raise ValueError("Attempt02 data contract plan binding mismatch")
    if data_contract.get("exclusion_union") != attempt02_preflight["exclusion_union"]:
        raise ValueError("Attempt02 exclusion union changed")

    train_path = _bound_declared(root, provenance, "train_fit")
    train_spec = _mapping(provenance["train_fit"], "train_fit")
    _verify_opaque_file(train_path, train_spec, "Attempt02 train")
    train_rows = _read_jsonl(train_path)
    train_ids = {_row_identity(row, "Attempt02 train") for row in train_rows}
    if len(train_rows) != train_spec["records"] or len(train_ids) != len(train_rows):
        raise ValueError("Attempt02 train200 count or uniqueness changed")
    if _identity_digest(train_ids) != train_spec["identity_sha256"]:
        raise ValueError("Attempt02 train200 identity digest changed")
    _verify_profile_counts(train_rows, {profile: 40 for profile in _PROFILES}, "Attempt02 train")
    sealed_train = _mapping(data_contract["fresh_splits"]["train"], "sealed train")
    if (
        sealed_train.get("records"),
        sealed_train.get("identity_sha256"),
        sealed_train.get("profile_counts"),
    ) != (
        train_spec["records"],
        train_spec["identity_sha256"],
        {profile: 40 for profile in sorted(_PROFILES)},
    ):
        raise ValueError("Attempt02 data contract train200 binding changed")
    train_binding = _mapping(
        data_contract["teacher_shards"]["splits"]["train"],
        "Attempt02 train shard binding",
    )
    for key in ("canonical_rows_sha256", "ordered_shards_sha256"):
        if train_binding.get(key) != train_spec[key]:
            raise ValueError(f"Attempt02 train sealed hash changed: {key}")

    calibration_path = _bound_declared(root, provenance, "sealed_calibration")
    calibration_spec = _mapping(provenance["sealed_calibration"], "sealed_calibration")
    _verify_opaque_file(calibration_path, calibration_spec, "Attempt02 calibration")
    calibration_ids = _sealed_calibration_identities(data_contract, calibration_spec)

    teacher_receipt_path = _bound_declared(root, provenance, "teacher_receipt")
    teacher_receipt = _read_mapping(teacher_receipt_path, "Attempt02 teacher receipt")
    _validate_teacher_receipt(teacher_receipt, provenance, data_contract)
    model_receipt_path = _bound_declared(root, provenance, "model_receipt")
    model_receipt = _read_mapping(model_receipt_path, "Attempt02 model receipt")
    training_manifest_path = _bound_declared(root, provenance, "training_manifest")
    training_manifest = _read_mapping(
        training_manifest_path, "Attempt02 training manifest"
    )
    _validate_attempt02_no_go(model_receipt, training_manifest, provenance)

    locked_path = root / plan["inherited_locked"]["path"]
    _verify_opaque_file(locked_path, plan["inherited_locked"], "inherited locked")
    locked_ids = set(old_context["locked_identities"])
    prior_ids = set(old_context["exclusion_identities"])
    named_prior = {
        "prior exclusion union": prior_ids,
        "Attempt02 train": train_ids,
        "Attempt02 sealed calibration": calibration_ids,
        "inherited locked": locked_ids,
    }
    _reject_identity_overlap(named_prior)
    if sum(len(values) for values in named_prior.values()) != 752:
        raise ValueError("Attempt03 frozen prior identity union must equal 752")

    new_schedules = _planned_seed_sets(plan)
    attempt02_schedules = _planned_attempt02_seed_sets(attempt02_plan)
    prior_data_seeds = {seed for values in named_prior.values() for seed, _ in values}
    for name, values in new_schedules.items():
        if values & prior_data_seeds:
            raise ValueError(f"Attempt03 planned seed overlaps prior data: {name}")
        for old_name, old_values in attempt02_schedules.items():
            if values & old_values:
                raise ValueError(
                    f"Attempt03 planned seed overlaps Attempt02 schedule: {name} vs {old_name}"
                )

    report: dict[str, Any] = {
        "schema": M43_ATTEMPT03_PREFLIGHT_SCHEMA,
        "status": "pass_frozen_before_fresh_generation",
        "plan": {
            "path": _path_token(plan_source, root),
            "file_sha256": _file_sha256(plan_source),
            "schema": plan["schema"],
        },
        "fresh_generation": _fresh_generation_projection(plan),
        "attempt02": {
            "plan_file_sha256": provenance["plan"]["file_sha256"],
            "data_contract_file_sha256": provenance["data_contract"]["file_sha256"],
            "data_contract_canonical_sha256": provenance["data_contract"]["canonical_sha256"],
            "teacher_receipt_file_sha256": provenance["teacher_receipt"]["file_sha256"],
            "model_receipt_file_sha256": provenance["model_receipt"]["file_sha256"],
            "training_manifest_file_sha256": provenance["training_manifest"]["file_sha256"],
            "terminal_status": "no_go_precalibration",
        },
        "fit_source": {
            "classification": "inherited_attempt02_train_fit",
            "records": len(train_ids),
            "identity_sha256": _identity_digest(train_ids),
            "content_parse_count": 1,
            "allowed_use": train_spec["allowed_use"],
        },
        "sealed_calibration": {
            "classification": "inherited_unopened",
            "records": len(calibration_ids),
            "identity_sha256": _identity_digest(calibration_ids),
            "roles": {
                role: {
                    "records": int(spec["records"]),
                    "identity_sha256": str(spec["identity_sha256"]),
                }
                for role, spec in calibration_spec["roles"].items()
            },
            "jsonl_content_parse_count": 0,
            "model_evaluation_count": 0,
            "structural_byte_hash_audit_only": True,
        },
        "inherited_locked": {
            "classification": "inherited_unopened",
            "records": len(locked_ids),
            "identity_sha256": _identity_digest(locked_ids),
            "jsonl_content_parse_count": 0,
            "model_evaluation_count": 0,
            "global_consumption_status": "unconsumed_preflight",
        },
        "freshness_exclusions": {
            "prior_exclusion_records": len(prior_ids),
            "attempt02_train_records": len(train_ids),
            "attempt02_calibration_records": len(calibration_ids),
            "inherited_locked_records": len(locked_ids),
            "unique_records": 752,
            "identity_sha256": _identity_digest(
                prior_ids | train_ids | calibration_ids | locked_ids
            ),
            "planned_seed_overlap": 0,
        },
        "experiment_freeze": {
            "model_contract_sha256": canonical_manifest_sha256(plan["model_contract"]),
            "precalibration_gate_sha256": canonical_manifest_sha256(
                plan["precalibration_gate"]
            ),
            "calibration_gate_sha256": canonical_manifest_sha256(
                plan["calibration_gate"]
            ),
            "post_precalibration_algorithm_change_allowed": False,
        },
        "runtime": {
            "current_profile_resolved": False,
            "current_profile_changed": False,
            "policy_activated": False,
            "full_replacement": False,
            "large_scale_authorized": False,
        },
    }
    report["receipt_sha256"] = _self_digest(report, "receipt_sha256")
    return report, {
        "root": root,
        "plan": plan,
        "prior_identities": prior_ids,
        "attempt02_train_identities": train_ids,
        "calibration_identities": calibration_ids,
        "locked_identities": locked_ids,
    }


def _bound_declared(root: Path, provenance: Mapping[str, Any], name: str) -> Path:
    spec = _mapping(provenance.get(name), name)
    path = _require_bound_path(root / str(spec["path"]), root, str(spec["path"]), name)
    if _file_sha256(path) != spec.get("file_sha256"):
        raise ValueError(f"Attempt03 bound file SHA mismatch: {name}")
    return path


def _verify_opaque_file(path: Path, spec: Mapping[str, Any], location: str) -> None:
    if not path.is_file():
        raise ValueError(f"{location} file is missing")
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"{location} byte size mismatch")
    if _file_sha256(path) != spec["file_sha256"]:
        raise ValueError(f"{location} file SHA mismatch")


def _sealed_calibration_identities(
    data_contract: Mapping[str, Any], expected: Mapping[str, Any]
) -> set[tuple[int, str]]:
    partition = _mapping(data_contract.get("calibration_partition"), "calibration_partition")
    identities: dict[str, set[tuple[int, str]]] = {}
    for role in ("safety_fit", "threshold_lock"):
        actual = _mapping(partition.get(role), f"calibration_partition.{role}")
        spec = _mapping(expected["roles"][role], f"expected roles.{role}")
        raw = actual.get("identities")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError(f"Attempt02 sealed calibration role identities missing: {role}")
        values = {
            (int(_mapping(item, role)["hand_seed"]), str(_mapping(item, role)["observation_fingerprint"]))
            for item in raw
        }
        if len(values) != int(spec["records"]):
            raise ValueError(f"Attempt02 calibration role count changed: {role}")
        if _identity_digest(values) != spec["identity_sha256"]:
            raise ValueError(f"Attempt02 calibration role identity digest changed: {role}")
        if actual.get("profile_counts") != {profile: 10 for profile in sorted(_PROFILES)}:
            raise ValueError(f"Attempt02 calibration role profile balance changed: {role}")
        identities[role] = values
    if identities["safety_fit"] & identities["threshold_lock"]:
        raise ValueError("Attempt02 sealed calibration roles overlap")
    result = identities["safety_fit"] | identities["threshold_lock"]
    if len(result) != int(expected["records"]):
        raise ValueError("Attempt02 sealed calibration role union count changed")
    if _identity_digest(result) != expected["identity_sha256"]:
        raise ValueError("Attempt02 sealed calibration identity digest changed")
    sealed = _mapping(data_contract["fresh_splits"]["calibration"], "sealed calibration")
    if (sealed.get("records"), sealed.get("identity_sha256")) != (
        expected["records"],
        expected["identity_sha256"],
    ):
        raise ValueError("Attempt02 data contract calibration binding changed")
    shard_binding = _mapping(
        data_contract["teacher_shards"]["splits"]["calibration"],
        "calibration shard binding",
    )
    for key in ("canonical_rows_sha256", "ordered_shards_sha256"):
        if shard_binding.get(key) != expected[key]:
            raise ValueError(f"Attempt02 calibration sealed hash changed: {key}")
    return result


def _validate_teacher_receipt(
    receipt: Mapping[str, Any],
    provenance: Mapping[str, Any],
    data_contract: Mapping[str, Any],
) -> None:
    expected = _mapping(provenance["teacher_receipt"], "teacher_receipt")
    if receipt.get("schema") != "hu_m43_attempt02_teacher_receive_receipt_v1":
        raise ValueError("Attempt02 teacher receipt schema mismatch")
    if receipt.get("status") != expected["required_status"]:
        raise ValueError("Attempt02 teacher receipt status changed")
    if receipt.get("verified_roots") != 300 or receipt.get("no_runtime_activation") is not True:
        raise ValueError("Attempt02 teacher receipt completeness changed")
    if receipt.get("data_contract_sha256") != data_contract["contract_sha256"]:
        raise ValueError("Attempt02 teacher receipt contract hash mismatch")
    merged = _mapping(receipt.get("merged"), "teacher receipt merged")
    source_specs = provenance
    for split, source_name in (("train", "train_fit"), ("calibration", "sealed_calibration")):
        spec = _mapping(source_specs[source_name], source_name)
        actual = _mapping(merged.get(split), f"merged.{split}")
        if (actual.get("rows"), actual.get("sha256")) != (
            spec["records"],
            spec["file_sha256"],
        ):
            raise ValueError(f"Attempt02 teacher merged {split} binding changed")
    if receipt.get("current_profile_mutated") is not False:
        raise ValueError("Attempt02 teacher receipt mutated current")


def _validate_attempt02_no_go(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    expected_receipt = _mapping(provenance["model_receipt"], "model_receipt")
    expected_manifest = _mapping(provenance["training_manifest"], "training_manifest")
    if receipt.get("schema") != "hu_m43_attempt02_v4_model_receive_receipt_v1":
        raise ValueError("Attempt02 model receipt schema mismatch")
    if receipt.get("status") != expected_receipt["required_status"]:
        raise ValueError("Attempt02 model receipt status changed")
    if receipt.get("training_manifest_sha256") != expected_manifest["file_sha256"]:
        raise ValueError("Attempt02 model receipt manifest hash mismatch")
    receipt_required = {
        "model_sha256": None,
        "calibration_and_contract_passed_to_frozen_assembler_only": True,
        "precalibration_no_go_opens_local_calibration_or_contract": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    for key, expected in receipt_required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"Attempt02 no-Go receipt evidence changed: {key}")
    if manifest.get("schema") != "hu_m43_attempt02_v4_assembly_decision_v1":
        raise ValueError("Attempt02 training manifest schema mismatch")
    for key in ("status", "promotion_status"):
        if manifest.get(key) != expected_manifest["required_status"]:
            raise ValueError(f"Attempt02 training manifest {key} changed")
    for key in (
        "model_written",
        "inherited_holdout_content_opened",
        "inherited_holdout_input_accepted",
        "current_profile_mutated",
        "runtime_policy_activated",
        "full_replacement",
    ):
        if manifest.get(key) is not False:
            raise ValueError(f"Attempt02 no-Go lifecycle evidence changed: {key}")
    calibration = _mapping(manifest.get("calibration"), "manifest.calibration")
    for key in (
        "data_contract_opened",
        "fresh_rows_opened",
        "safety_fit_performed",
        "threshold_selection_performed",
    ):
        if calibration.get(key) is not False:
            raise ValueError(f"Attempt02 calibration was opened or used: {key}")


def _verify_profile_counts(
    rows: Sequence[Mapping[str, Any]], expected: Mapping[str, int], location: str
) -> None:
    counts = Counter(
        str(_mapping(row.get("provenance"), f"{location} provenance").get("root_profile"))
        for row in rows
    )
    if dict(counts) != dict(expected):
        raise ValueError(f"{location} root profile counts changed")
    for row in rows:
        provenance = _mapping(row.get("provenance"), f"{location} provenance")
        if provenance.get("current_profile_resolved") is not False:
            raise ValueError(f"{location} resolved current profile")


def _reject_identity_overlap(named: Mapping[str, set[tuple[int, str]]]) -> None:
    items = list(named.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left & right:
                raise ValueError(f"identity overlap: {left_name} vs {right_name}")
            if {seed for seed, _ in left} & {seed for seed, _ in right}:
                raise ValueError(f"hand seed overlap: {left_name} vs {right_name}")
            if {fingerprint for _, fingerprint in left} & {
                fingerprint for _, fingerprint in right
            }:
                raise ValueError(f"fingerprint overlap: {left_name} vs {right_name}")


def _reject_schedule_overlap(
    schedules: Sequence[tuple[str, set[int]]], location: str
) -> None:
    for index, (left_name, left) in enumerate(schedules):
        for right_name, right in schedules[index + 1 :]:
            if left & right:
                raise ValueError(f"{location} seed schedules overlap: {left_name} vs {right_name}")


def _planned_seed_sets(plan: Mapping[str, Any]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for role in _ROLES:
        spec = plan["fresh_splits"][role]
        roots = int(spec["roots"])
        for field in (
            "seed_start",
            "candidate_seed_start",
            "evaluation_seed_start",
            "child_policy_seed_start",
        ):
            count = roots if field == "seed_start" else roots // _ROOTS_PER_SHARD
            result[f"{role}.{field}"] = {
                int(spec[field]) + _SEED_STRIDE * index for index in range(count)
            }
    return result


def _planned_attempt02_seed_sets(plan: Mapping[str, Any]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    roots_per_shard = int(plan["budget"]["roots_per_shard"])
    for split in ("train", "calibration"):
        spec = plan["fresh_splits"][split]
        roots = int(spec["roots"])
        stride = int(spec["seed_stride"])
        for field in (
            "seed_start",
            "candidate_seed_start",
            "evaluation_seed_start",
            "child_policy_seed_start",
        ):
            count = roots if field == "seed_start" else roots // roots_per_shard
            result[f"{split}.{field}"] = {
                int(spec[field]) + stride * index for index in range(count)
            }
    return result


def _fresh_generation_projection(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fresh_roots": 700,
        "roots_per_shard": _ROOTS_PER_SHARD,
        "fresh_shards": 70,
        "roles": {
            role: {
                "roots": int(plan["fresh_splits"][role]["roots"]),
                "shards": int(plan["fresh_splits"][role]["roots"]) // _ROOTS_PER_SHARD,
                "record_split": "train",
                "seed_start": int(plan["fresh_splits"][role]["seed_start"]),
                "seed_stride": _SEED_STRIDE,
                "candidate_seed_start": int(
                    plan["fresh_splits"][role]["candidate_seed_start"]
                ),
                "evaluation_seed_start": int(
                    plan["fresh_splits"][role]["evaluation_seed_start"]
                ),
                "child_policy_seed_start": int(
                    plan["fresh_splits"][role]["child_policy_seed_start"]
                ),
                "profile_roots": 100 if role == "train.fit" else 40,
            }
            for role in _ROLES
        },
        "teacher_search": dict(plan["teacher_search"]),
    }


def finalize_fresh_data(
    *,
    plan_path: str | Path,
    repo_root: str | Path,
    preflight_receipt: str | Path,
    train_fit: Sequence[str | Path],
    train_precal_holdout: Sequence[str | Path],
) -> dict[str, Any]:
    """Seal fresh fit/pre-cal shards without opening calibration or locked JSONL."""

    receipt_path = Path(preflight_receipt).resolve()
    declared = _read_mapping(receipt_path, "Attempt03 preflight receipt")
    _validate_embedded_digest(
        declared,
        field="receipt_sha256",
        expected=None,
        location="Attempt03 preflight receipt",
    )
    if declared.get("schema") != M43_ATTEMPT03_PREFLIGHT_SCHEMA:
        raise ValueError("Attempt03 preflight receipt schema mismatch")
    if declared.get("status") != "pass_frozen_before_fresh_generation":
        raise ValueError("Attempt03 preflight did not pass")
    actual, context = _audit_preflight(plan_path=plan_path, repo_root=repo_root)
    if declared != actual:
        raise ValueError("Attempt03 preflight no longer matches frozen sources")

    root: Path = context["root"]
    plan: Mapping[str, Any] = context["plan"]
    paths = {
        "train.fit": tuple(Path(path).resolve() for path in train_fit),
        "train.precal_holdout": tuple(
            Path(path).resolve() for path in train_precal_holdout
        ),
    }
    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    identities_by_role: dict[str, set[tuple[int, str]]] = {}
    profile_counts: dict[str, Counter[str]] = {}
    shard_audits: dict[str, list[dict[str, Any]]] = {}
    for role in _ROLES:
        role_paths = paths[role]
        spec = _mapping(plan["fresh_splits"][role], f"fresh_splits.{role}")
        expected_shards = int(spec["roots"]) // _ROOTS_PER_SHARD
        if len(role_paths) != expected_shards:
            raise ValueError(f"Attempt03 {role} shard count disagrees with plan")
        normalized = [os.path.normcase(str(path)) for path in role_paths]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Attempt03 {role} contains duplicate shard paths")
        audits: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for shard_index, path in enumerate(role_paths):
            audit = read_and_audit_shard(
                path,
                expected_split=str(spec["record_split"]),
                require_paired_delta=True,
            )
            if int(audit["records"]) != _ROOTS_PER_SHARD:
                raise ValueError(f"Attempt03 {role} shard size is not 10")
            shard_rows = _read_jsonl(path)
            if len(shard_rows) != _ROOTS_PER_SHARD:
                raise ValueError(f"Attempt03 {role} parsed shard size is not 10")
            expected_seeds = {
                int(spec["seed_start"])
                + _SEED_STRIDE * (shard_index * _ROOTS_PER_SHARD + offset)
                for offset in range(_ROOTS_PER_SHARD)
            }
            observed_seeds = {int(row["hand_seed"]) for row in shard_rows}
            if observed_seeds != expected_seeds:
                raise ValueError(f"Attempt03 {role} shard seed schedule mismatch")
            audits.append(audit)
            rows.extend(shard_rows)
        if len(rows) != int(spec["roots"]):
            raise ValueError(f"Attempt03 {role} root count disagrees with plan")
        identities = {_row_identity(row, role) for row in rows}
        if len(identities) != len(rows):
            raise ValueError(f"Attempt03 {role} contains duplicate identities")
        counts = _validate_fresh_rows(rows, role=role, plan=plan)
        rows_by_role[role] = rows
        identities_by_role[role] = identities
        profile_counts[role] = counts
        shard_audits[role] = audits

    prior_sets = {
        "prior exclusion union": set(context["prior_identities"]),
        "Attempt02 train": set(context["attempt02_train_identities"]),
        "Attempt02 sealed calibration": set(context["calibration_identities"]),
        "inherited locked": set(context["locked_identities"]),
        "fresh train.fit": identities_by_role["train.fit"],
        "fresh train.precal_holdout": identities_by_role["train.precal_holdout"],
    }
    _reject_identity_overlap(prior_sets)

    combined_fit = (
        set(context["attempt02_train_identities"])
        | identities_by_role["train.fit"]
    )
    if len(combined_fit) != 700:
        raise ValueError("Attempt03 combined fit set must contain exactly 700 states")
    expected_combined_profiles = {profile: 140 for profile in _PROFILES}
    combined_profiles = Counter({profile: 40 for profile in _PROFILES})
    combined_profiles.update(profile_counts["train.fit"])
    if dict(combined_profiles) != expected_combined_profiles:
        raise ValueError("Attempt03 combined fit profile balance changed")

    precal_ids = identities_by_role["train.precal_holdout"]
    precal_digest = _identity_digest(precal_ids)
    precal_marker_relative = (
        "outputs/hu_joint_policy/m43_attempt03_precal_consumption/"
        f"{precal_digest}/M43_ATTEMPT03_PRECAL_CONSUMED.json"
    )
    marker_hits = _find_precal_markers(root, precal_digest)
    if (root / precal_marker_relative).exists() or marker_hits:
        raise ValueError("Attempt03 pre-calibration identity is already consumed")

    teacher_shards: dict[str, Any] = {
        "schema": M43_ATTEMPT03_TEACHER_SHARDS_SCHEMA,
        "roles": {
            role: _ordered_shard_binding(
                paths[role], repo_root=root, expected_split="train"
            )
            for role in _ROLES
        },
    }
    teacher_shards["all_fresh_roles_sha256"] = canonical_manifest_sha256(
        teacher_shards
    )
    report: dict[str, Any] = {
        "schema": M43_ATTEMPT03_DATA_CONTRACT_SCHEMA,
        "status": (
            "pass_fresh_fit_and_one_shot_precal_sealed_"
            "calibration_locked_unopened"
        ),
        "plan_sha256": actual["plan"]["file_sha256"],
        "preflight": {
            "receipt_sha256": actual["receipt_sha256"],
            "file_sha256": _file_sha256(receipt_path),
        },
        "fresh_generation": actual["fresh_generation"],
        "attempt02": actual["attempt02"],
        "fit": {
            "records": 700,
            "identity_sha256": _identity_digest(combined_fit),
            "profile_counts": dict(sorted(combined_profiles.items())),
            "sources": {
                "attempt02.train": {
                    "records": 200,
                    "identity_sha256": actual["fit_source"]["identity_sha256"],
                },
                "train.fit": _fresh_role_manifest(
                    identities_by_role["train.fit"],
                    profile_counts["train.fit"],
                    shard_audits["train.fit"],
                ),
            },
            "allowed_use": "ranker_fit_and_grouped_crossfit_only",
        },
        "precal_holdout": {
            **_fresh_role_manifest(
                precal_ids,
                profile_counts["train.precal_holdout"],
                shard_audits["train.precal_holdout"],
            ),
            "classification": "fresh_one_shot_precalibration_holdout",
            "contract_structural_audit_count": 1,
            "model_evaluation_count": 0,
            "hyperparameter_or_algorithm_selection_allowed": False,
            "consumption_marker": precal_marker_relative,
            "consumption_status": "unconsumed_sealed",
        },
        "sealed_calibration": actual["sealed_calibration"],
        "inherited_locked": actual["inherited_locked"],
        "freshness": {
            "all_prior_and_fresh_identity_overlap": 0,
            "all_prior_and_fresh_hand_seed_overlap": 0,
            "all_prior_and_fresh_fingerprint_overlap": 0,
            "unique_prior_and_fresh_records": sum(
                len(values) for values in prior_sets.values()
            ),
        },
        "teacher_shards": teacher_shards,
        "teacher_shards_all_fresh_roles_sha256": teacher_shards[
            "all_fresh_roles_sha256"
        ],
        "experiment_freeze": actual["experiment_freeze"],
        "runtime": actual["runtime"],
    }
    report["contract_sha256"] = _self_digest(report, "contract_sha256")
    return report


def _validate_fresh_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    plan: Mapping[str, Any],
) -> Counter[str]:
    spec = _mapping(plan["fresh_splits"][role], f"fresh_splits.{role}")
    expected_per_profile = 100 if role == "train.fit" else 40
    counts: Counter[str] = Counter()
    seed_to_index = {
        int(spec["seed_start"]) + _SEED_STRIDE * index: index
        for index in range(int(spec["roots"]))
    }
    for row in rows:
        if row.get("split") != "train":
            raise ValueError(f"Attempt03 {role} row split changed")
        provenance = _mapping(row.get("provenance"), f"{role} provenance")
        if provenance.get("current_profile_resolved") is not False:
            raise ValueError(f"Attempt03 {role} resolved current profile")
        profile = str(provenance.get("root_profile", ""))
        if profile not in _PROFILES:
            raise ValueError(f"Attempt03 {role} contains unknown profile")
        counts[profile] += 1
        search = _mapping(row.get("search_config"), f"{role} search_config")
        if (
            search.get("candidate_samples") != 2
            or search.get("evaluation_samples") != 64
        ):
            raise ValueError(f"Attempt03 {role} did not use frozen c2/e64")
        seed = int(row["hand_seed"])
        if seed not in seed_to_index:
            raise ValueError(f"Attempt03 {role} contains unplanned hand seed")
        shard_index = seed_to_index[seed] // _ROOTS_PER_SHARD
        for field in ("candidate_seed", "evaluation_seed", "child_policy_seed"):
            expected = int(spec[f"{field}_start"]) + _SEED_STRIDE * shard_index
            if search.get(field) != expected:
                raise ValueError(f"Attempt03 {role} {field} schedule mismatch")
    expected = {profile: expected_per_profile for profile in _PROFILES}
    if dict(counts) != expected:
        raise ValueError(f"Attempt03 {role} root-profile quota mismatch")
    return counts


def _fresh_role_manifest(
    identities: set[tuple[int, str]],
    profiles: Counter[str],
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "records": len(identities),
        "shards": len(audits),
        "identity_sha256": _identity_digest(identities),
        "profile_counts": dict(sorted(profiles.items())),
        "audited_paired_delta_records": sum(
            int(audit["paired_delta_records"]) for audit in audits
        ),
        "identities": [_identity_json(value) for value in sorted(identities)],
    }


def _find_precal_markers(root: Path, identity_sha256: str) -> list[str]:
    matches: list[str] = []
    for marker in root.rglob("M43_ATTEMPT03_PRECAL_CONSUMED.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Attempt03 pre-cal marker: {marker}") from exc
        if identity_sha256 in json.dumps(payload, sort_keys=True):
            matches.append(_path_token(marker.resolve(), root))
    return sorted(matches)


def _write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--plan", required=True)
    preflight.add_argument("--repo-root", default=".")
    preflight.add_argument("--output", required=True)
    finalize = subparsers.add_parser("finalize-fresh")
    finalize.add_argument("--plan", required=True)
    finalize.add_argument("--repo-root", default=".")
    finalize.add_argument("--preflight-receipt", required=True)
    finalize.add_argument("--train-fit", action="append", required=True)
    finalize.add_argument(
        "--train-precal-holdout", action="append", required=True
    )
    finalize.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "preflight":
        payload = build_preflight_receipt(
            plan_path=args.plan, repo_root=args.repo_root
        )
    else:
        payload = finalize_fresh_data(
            plan_path=args.plan,
            repo_root=args.repo_root,
            preflight_receipt=args.preflight_receipt,
            train_fit=args.train_fit,
            train_precal_holdout=args.train_precal_holdout,
        )
    _write_immutable_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
