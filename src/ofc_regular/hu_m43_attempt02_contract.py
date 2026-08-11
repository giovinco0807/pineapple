"""Fail-closed data contract for the M4.3 Attempt02 pilot.

Attempt01 train/calibration identities are permanently excluded.  Its locked
holdout may be inherited only while it remains globally unconsumed, and this
module treats that file as opaque bytes until a separately frozen model and
threshold claim the one-shot consumption marker.

The preflight command runs before any fresh teacher generation.  The finalize
command validates only the fresh train/calibration JSONL shards and seals their
deterministic roles.  Neither command parses the inherited locked JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .audit_hu_m4_t1_data import read_and_audit_shard
from .hu_m43_pilot_contract import (
    _identity_digest,
    _identity_json,
    _ordered_shard_binding,
    _read_jsonl,
    _row_identity,
    audit_frozen_exclusions,
    canonical_manifest_sha256,
    load_and_validate_plan as load_attempt01_plan,
)


M43_ATTEMPT02_PLAN_SCHEMA = "hu_m43_attempt02_plan_v1"
M43_ATTEMPT02_PREFLIGHT_SCHEMA = "hu_m43_attempt02_preflight_receipt_v1"
M43_ATTEMPT02_DATA_CONTRACT_SCHEMA = "hu_m43_attempt02_data_contract_v1"
M43_ATTEMPT02_CALIBRATION_PARTITION_SCHEMA = (
    "hu_m43_attempt02_calibration_partition_v1"
)
M43_ATTEMPT02_TEACHER_SHARDS_SCHEMA = "hu_m43_attempt02_ordered_teacher_shards_v1"

_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
_FRESH_SPLITS = ("train", "calibration")
_HEX = frozenset("0123456789abcdef")

# Frozen audit fixtures.  Changing any value creates a new lifecycle, not an
# in-place revision of Attempt02.
ATTEMPT01_CONTRACT_CANONICAL_SHA256 = (
    "f612ed36f9a371854e990c4214e0728d78b0d144937106527d83de38440391cf"
)
ATTEMPT01_CONTRACT_FILE_SHA256 = (
    "1c79da05b85564d4c72314cc7dbbcc353155f7b963f70a19fcc7fbba583c371a"
)
ATTEMPT01_TRAIN_FILE_SHA256 = (
    "59b00b12ace2639f1e0cd11abdcfcdb2eaf74506094f17b55fd21efc54c16e6a"
)
ATTEMPT01_CALIBRATION_FILE_SHA256 = (
    "40a16164b710853132b53c7bc55790f1c591bf96876ec523938b74817091eb45"
)
ATTEMPT01_TRAIN_IDENTITY_SHA256 = (
    "c992fbaeea0752db79fd481fe7da7e025c11e7d44123a740a7ee25c87343cc61"
)
ATTEMPT01_CALIBRATION_IDENTITY_SHA256 = (
    "df32b384e43e8b4acd8d551c0a5f97d057211c6f901b5e79fb87aedc99dc0929"
)
ATTEMPT01_TRAIN_CAL_IDENTITY_SHA256 = (
    "4c6a0ffc1d46e9ecde75d37ca471bdd80fb78b627c50a7cca909ab153a1089c8"
)
ATTEMPT01_TRAINING_MANIFEST_FILE_SHA256 = (
    "70698a4ec2adb096616c4368c74c04b7f92b303cc11643f5a17028f9a2d5a3ff"
)
EXCLUSION_UNION_RECORDS = 412
EXCLUSION_UNION_IDENTITY_SHA256 = (
    "c7598ba0f74528562b79966e53dd843fe2225f23033b791377327f6ea6ab3b3e"
)
INHERITED_LOCKED_RECORDS = 40
INHERITED_LOCKED_BYTES = 1_144_214
INHERITED_LOCKED_FILE_SHA256 = (
    "0a88e1cc8b079906e0ca14ddf4a8c3df7b108a09d0f363c7cb665cab89b40a1e"
)
INHERITED_LOCKED_CANONICAL_ROWS_SHA256 = (
    "fa99408bc4be031a7b4f2a8dfe05f4af07d90ebb7450e1eb03ba1fa596a87890"
)
INHERITED_LOCKED_IDENTITY_SHA256 = (
    "3946c110de49831359316439fa230d9096fc41046b0cb07a92506eb931e0e629"
)
INHERITED_LOCKED_ORDERED_SHARDS_SHA256 = (
    "6c351acd69cf0ce546fa3ff74e397d0091f4e0cd2ffc5b2ce95fc22a418bcaf2"
)


def load_and_validate_attempt02_plan(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    plan = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict):
        raise ValueError("M4.3 Attempt02 plan must be a mapping")
    _validate_plan(plan)
    return plan


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != M43_ATTEMPT02_PLAN_SCHEMA:
        raise ValueError("unsupported M4.3 Attempt02 plan schema")
    if plan.get("milestone") != "M4.3-attempt02":
        raise ValueError("M4.3 Attempt02 milestone mismatch")
    if plan.get("status") != "frozen_pre_generation":
        raise ValueError("Attempt02 plan must be frozen before fresh generation")
    if plan.get("fixed_baseline_profile") != "stage18_p1":
        raise ValueError("Attempt02 baseline profile changed")
    continuation = _mapping(plan.get("fixed_continuation"), "fixed_continuation")
    if continuation.get("t2_profile") != "stage9f_p2":
        raise ValueError("Attempt02 T2 continuation changed")

    budget = _mapping(plan.get("budget"), "budget")
    if budget.get("kind") != "bounded_pre_go_pilot":
        raise ValueError("Attempt02 budget is not a bounded pre-Go pilot")
    roots_per_shard = _integer(
        budget.get("roots_per_shard"), "budget.roots_per_shard", minimum=1
    )
    if roots_per_shard != 10:
        raise ValueError("Attempt02 resumable shard size must remain 10")
    max_roots = _integer(
        budget.get("max_fresh_roots"), "budget.max_fresh_roots", minimum=1
    )
    if max_roots > 1000:
        raise ValueError("Attempt02 pre-Go budget may not exceed 1,000 fresh roots")

    split_specs = _mapping(plan.get("fresh_splits"), "fresh_splits")
    if set(split_specs) != set(_FRESH_SPLITS):
        raise ValueError("Attempt02 permits fresh train and calibration only")
    total_roots = 0
    total_shards = 0
    all_seed_schedules: list[tuple[str, set[int]]] = []
    for split in _FRESH_SPLITS:
        spec = _mapping(split_specs.get(split), f"fresh_splits.{split}")
        roots = _integer(spec.get("roots"), f"{split}.roots", minimum=1)
        if roots % roots_per_shard:
            raise ValueError(f"Attempt02 {split} roots do not fill exact shards")
        total_roots += roots
        total_shards += roots // roots_per_shard
        stride = _integer(spec.get("seed_stride"), f"{split}.seed_stride", minimum=1)
        if spec.get("phase_seed_stride_scope") != "split_shard":
            raise ValueError(f"Attempt02 {split} phase seeds must advance per shard")
        for field in (
            "seed_start",
            "candidate_seed_start",
            "evaluation_seed_start",
            "child_policy_seed_start",
        ):
            start = _integer(spec.get(field), f"{split}.{field}", minimum=0)
            schedule_count = roots if field == "seed_start" else roots // roots_per_shard
            all_seed_schedules.append(
                (
                    f"{split}.{field}",
                    {start + stride * index for index in range(schedule_count)},
                )
            )
        expected_use = {
            "train": "ranker_fit_and_nested_oof_only",
            "calibration": "safety_fit_and_threshold_lock_only",
        }[split]
        if spec.get("allowed_use") != expected_use:
            raise ValueError(f"Attempt02 {split} allowed-use boundary changed")
    if total_roots != _integer(budget.get("fresh_roots"), "budget.fresh_roots"):
        raise ValueError("Attempt02 split roots disagree with fresh budget")
    if total_roots > max_roots:
        raise ValueError("Attempt02 fresh roots exceed bounded budget")
    if total_shards != _integer(budget.get("fresh_shards"), "budget.fresh_shards"):
        raise ValueError("Attempt02 split shards disagree with fresh budget")
    if _integer(budget.get("inherited_locked_roots"), "inherited_locked_roots") != 40:
        raise ValueError("Attempt02 must inherit exactly the unopened locked40")
    for index, (left_name, left) in enumerate(all_seed_schedules):
        for right_name, right in all_seed_schedules[index + 1 :]:
            if left & right:
                raise ValueError(
                    f"Attempt02 seed schedules overlap: {left_name} vs {right_name}"
                )

    search = _mapping(plan.get("teacher_search"), "teacher_search")
    if search.get("candidate_samples") != 2 or search.get("evaluation_samples") != 64:
        raise ValueError("Attempt02 fresh teacher search must remain c2/e64")
    if search.get("common_random_futures") is not True:
        raise ValueError("Attempt02 requires common random futures")
    if search.get("candidate_evaluation_rng_disjoint") is not True:
        raise ValueError("Attempt02 candidate/evaluation RNGs must be disjoint")

    population = plan.get("root_population")
    if not isinstance(population, Sequence) or isinstance(population, (str, bytes)):
        raise ValueError("Attempt02 root population missing")
    profile_totals = Counter()
    observed_profiles: set[str] = set()
    profile_allocations: dict[str, dict[str, int]] = {}
    for raw in population:
        row = _mapping(raw, "root_population entry")
        profile = str(row.get("profile", ""))
        if profile not in _PROFILES or profile in observed_profiles:
            raise ValueError("Attempt02 root population changed or duplicated")
        observed_profiles.add(profile)
        roots = _mapping(row.get("roots"), f"root_population.{profile}.roots")
        if set(roots) != {"train", "calibration", "inherited_locked"}:
            raise ValueError("Attempt02 profile split allocation changed")
        allocation = {
            key: _integer(value, f"root_population.{profile}.{key}")
            for key, value in roots.items()
        }
        profile_allocations[profile] = allocation
        profile_totals.update(allocation)
    if observed_profiles != set(_PROFILES):
        raise ValueError("Attempt02 must retain all five root profiles")
    expected_totals = {
        "train": int(split_specs["train"]["roots"]),
        "calibration": int(split_specs["calibration"]["roots"]),
        "inherited_locked": int(budget["inherited_locked_roots"]),
    }
    if dict(profile_totals) != expected_totals:
        raise ValueError("Attempt02 profile quotas disagree with split totals")

    partition = _mapping(plan.get("calibration_partition"), "calibration_partition")
    if partition.get("schema") != M43_ATTEMPT02_CALIBRATION_PARTITION_SCHEMA:
        raise ValueError("Attempt02 calibration partition schema mismatch")
    if partition.get("method") != "profile_stratified_identity_hash_v1":
        raise ValueError("Attempt02 calibration partition method changed")
    per_role = _integer(
        partition.get("roots_per_profile_per_role"), "roots_per_profile_per_role"
    )
    safety = _integer(partition.get("safety_fit_roots"), "safety_fit_roots")
    threshold = _integer(
        partition.get("threshold_lock_roots"), "threshold_lock_roots"
    )
    if safety + threshold != expected_totals["calibration"]:
        raise ValueError("Attempt02 calibration roles do not cover calibration")
    if safety != per_role * len(_PROFILES) or threshold != per_role * len(_PROFILES):
        raise ValueError("Attempt02 calibration role quotas are not profile-balanced")
    for profile, allocation in profile_allocations.items():
        if allocation["calibration"] != 2 * per_role:
            raise ValueError(f"Attempt02 calibration quota changed for {profile}")
    if partition.get("threshold_selection_source") != "calibration.threshold_lock":
        raise ValueError("Attempt02 thresholds must use threshold-lock only")
    if partition.get("inherited_locked_used") is not False:
        raise ValueError("Attempt02 inherited locked may not calibrate the model")

    exclusions = _mapping(plan.get("freshness_exclusions"), "freshness_exclusions")
    if exclusions.get("reject_hand_seed_or_fingerprint_overlap") is not True:
        raise ValueError("Attempt02 freshness overlap rejection was disabled")
    expected_union = _mapping(exclusions.get("expected_union"), "expected_union")
    for field in (
        "source_records",
        "unique_identities",
        "unique_hand_seeds",
        "unique_observation_fingerprints",
    ):
        if expected_union.get(field) != EXCLUSION_UNION_RECORDS:
            raise ValueError("Attempt02 exclusion union must remain exactly 412")
    if expected_union.get("identity_sha256") != EXCLUSION_UNION_IDENTITY_SHA256:
        raise ValueError("Attempt02 exclusion union digest changed")
    sources = exclusions.get("attempt01_sources")
    if not isinstance(sources, Sequence) or len(sources) != 2:
        raise ValueError("Attempt02 must exclude Attempt01 train and calibration")

    provenance = _mapping(plan.get("attempt01_provenance"), "attempt01_provenance")
    contract = _mapping(provenance.get("data_contract"), "data_contract")
    if contract.get("file_sha256") != ATTEMPT01_CONTRACT_FILE_SHA256:
        raise ValueError("Attempt01 contract file fixture changed")
    if contract.get("canonical_sha256") != ATTEMPT01_CONTRACT_CANONICAL_SHA256:
        raise ValueError("Attempt01 canonical contract fixture changed")
    manifest = _mapping(provenance.get("training_manifest"), "training_manifest")
    if manifest.get("file_sha256") != ATTEMPT01_TRAINING_MANIFEST_FILE_SHA256:
        raise ValueError("Attempt01 training manifest fixture changed")

    inherited = _mapping(plan.get("inherited_locked"), "inherited_locked")
    expected_locked = {
        "classification": "inherited_unopened",
        "records": INHERITED_LOCKED_RECORDS,
        "bytes": INHERITED_LOCKED_BYTES,
        "file_sha256": INHERITED_LOCKED_FILE_SHA256,
        "canonical_rows_sha256": INHERITED_LOCKED_CANONICAL_ROWS_SHA256,
        "identity_sha256": INHERITED_LOCKED_IDENTITY_SHA256,
        "ordered_shards_sha256": INHERITED_LOCKED_ORDERED_SHARDS_SHA256,
        "source_data_contract_canonical_sha256": ATTEMPT01_CONTRACT_CANONICAL_SHA256,
        "fresh_search_budget_mismatch": (
            "allowed_immutable_provenance_c2e16_locked_vs_c2e64_fresh"
        ),
        "content_open_before_frozen_model_and_threshold_allowed": False,
        "structural_byte_hash_audit_only": True,
    }
    for key, expected in expected_locked.items():
        if inherited.get(key) != expected:
            raise ValueError(f"Attempt02 inherited locked fixture changed: {key}")
    if dict(_mapping(inherited.get("source_search"), "source_search")) != {
        "candidate_samples": 2,
        "evaluation_samples": 16,
    }:
        raise ValueError("Attempt02 inherited locked c2/e16 provenance changed")
    marker = str(inherited.get("global_consumption_marker", ""))
    if marker != (
        "outputs/hu_joint_policy/m43_locked_consumption/"
        f"{INHERITED_LOCKED_IDENTITY_SHA256}/M43_LOCKED_CONSUMED.json"
    ):
        raise ValueError("Attempt02 global locked marker path changed")

    lifecycle = _mapping(plan.get("locked_lifecycle"), "locked_lifecycle")
    for key in (
        "one_shot_only",
        "claim_marker_before_content_open",
        "marker_is_consumed_even_if_evaluation_crashes",
        "fresh_locked_required_after_any_model_threshold_or_feature_reselection",
        "may_inherit_again_after_calibration_no_go_if_unconsumed",
    ):
        if lifecycle.get(key) is not True:
            raise ValueError(f"Attempt02 locked lifecycle weakened: {key}")
    guards = _mapping(plan.get("activation_guards"), "activation_guards")
    for key in (
        "current_profile_changed",
        "runtime_policy_activated",
        "full_replacement_enabled",
        "large_scale_authorized",
        "m5_authorized",
    ):
        if guards.get(key) is not False:
            raise ValueError(f"Attempt02 activation guard must remain false: {key}")


def build_preflight_receipt(
    *,
    plan_path: str | Path,
    repo_root: str | Path,
    attempt01_data_contract: str | Path,
    attempt01_training_manifest: str | Path,
    attempt01_train: str | Path,
    attempt01_calibration: str | Path,
    inherited_locked: str | Path,
) -> dict[str, Any]:
    report, _ = _audit_preflight(
        plan_path=plan_path,
        repo_root=repo_root,
        attempt01_data_contract=attempt01_data_contract,
        attempt01_training_manifest=attempt01_training_manifest,
        attempt01_train=attempt01_train,
        attempt01_calibration=attempt01_calibration,
        inherited_locked=inherited_locked,
    )
    return report


def _audit_preflight(
    *,
    plan_path: str | Path,
    repo_root: str | Path,
    attempt01_data_contract: str | Path,
    attempt01_training_manifest: str | Path,
    attempt01_train: str | Path,
    attempt01_calibration: str | Path,
    inherited_locked: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repo_root).resolve()
    plan_source = Path(plan_path).resolve()
    plan = load_and_validate_attempt02_plan(plan_source)
    plan_sha = _file_sha256(plan_source)
    exclusions = _mapping(plan["freshness_exclusions"], "freshness_exclusions")
    provenance = _mapping(plan["attempt01_provenance"], "attempt01_provenance")

    contract_spec = _mapping(provenance["data_contract"], "data_contract")
    contract_path = _require_bound_path(
        attempt01_data_contract, root, str(contract_spec["path"]), "Attempt01 contract"
    )
    if _file_sha256(contract_path) != contract_spec["file_sha256"]:
        raise ValueError("Attempt01 data contract file SHA mismatch")
    attempt_contract = _read_mapping(contract_path, "Attempt01 data contract")
    _validate_embedded_digest(
        attempt_contract,
        field="contract_sha256",
        expected=ATTEMPT01_CONTRACT_CANONICAL_SHA256,
        location="Attempt01 data contract",
    )
    if attempt_contract.get("schema") != "hu_m43_bounded_pilot_data_contract_v1":
        raise ValueError("Attempt01 data contract schema mismatch")
    if attempt_contract.get("status") != "pass_fresh_data_sealed_for_model_freeze":
        raise ValueError("Attempt01 data contract was not sealed")

    attempt_source_specs = {
        str(_mapping(row, "Attempt01 exclusion source")["split"]): _mapping(
            row, "Attempt01 exclusion source"
        )
        for row in exclusions["attempt01_sources"]
    }
    supplied_attempt_paths = {
        "train": attempt01_train,
        "calibration": attempt01_calibration,
    }
    attempt_identities: set[tuple[int, str]] = set()
    attempt_reports: list[dict[str, Any]] = []
    for split in _FRESH_SPLITS:
        spec = attempt_source_specs[split]
        source = _require_bound_path(
            supplied_attempt_paths[split], root, str(spec["path"]), f"Attempt01 {split}"
        )
        actual_sha = _file_sha256(source)
        if actual_sha != spec["file_sha256"]:
            raise ValueError(f"Attempt01 {split} file SHA mismatch")
        rows = _read_jsonl(source)
        if len(rows) != spec["records"]:
            raise ValueError(f"Attempt01 {split} record count mismatch")
        identities = {_row_identity(row, f"Attempt01 {split}") for row in rows}
        if len(identities) != len(rows):
            raise ValueError(f"Attempt01 {split} contains duplicate identities")
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"Attempt01 {split} split tag mismatch")
        identity_sha = _identity_digest(identities)
        if identity_sha != spec["identity_sha256"]:
            raise ValueError(f"Attempt01 {split} identity digest mismatch")
        sealed_split = _mapping(attempt_contract["splits"][split], f"sealed {split}")
        if sealed_split.get("records") != len(rows) or sealed_split.get(
            "identity_sha256"
        ) != identity_sha:
            raise ValueError(f"Attempt01 {split} no longer matches sealed contract")
        attempt_identities.update(identities)
        attempt_reports.append(
            {
                "split": split,
                "path": _path_token(source, root),
                "records": len(rows),
                "file_sha256": actual_sha,
                "identity_sha256": identity_sha,
            }
        )
    if _identity_digest(attempt_identities) != ATTEMPT01_TRAIN_CAL_IDENTITY_SHA256:
        raise ValueError("Attempt01 train/calibration union digest mismatch")

    base_spec = _mapping(exclusions["base_plan"], "base_plan")
    base_plan_path = (root / str(base_spec["path"])).resolve()
    if _file_sha256(base_plan_path) != base_spec["file_sha256"]:
        raise ValueError("M4/M4.1/M4.2 base exclusion plan SHA mismatch")
    base_plan = load_attempt01_plan(base_plan_path)
    base = audit_frozen_exclusions(base_plan, repo_root=root)
    if base["source_records"] != base_spec["records"]:
        raise ValueError("base exclusion count changed")
    if base["identity_sha256"] != base_spec["identity_sha256"]:
        raise ValueError("base exclusion identity digest changed")
    union_identities = set(base["identities"]) | attempt_identities
    union_seeds = {seed for seed, _ in union_identities}
    union_fingerprints = {fingerprint for _, fingerprint in union_identities}
    expected_union = _mapping(exclusions["expected_union"], "expected_union")
    actual_union = {
        "source_records": base["source_records"] + len(attempt_identities),
        "unique_identities": len(union_identities),
        "unique_hand_seeds": len(union_seeds),
        "unique_observation_fingerprints": len(union_fingerprints),
        "identity_sha256": _identity_digest(union_identities),
    }
    if actual_union != dict(expected_union):
        raise ValueError("Attempt02 frozen exclusion union no longer equals 412")

    locked_spec = _mapping(plan["inherited_locked"], "inherited_locked")
    locked_path = _require_bound_path(
        inherited_locked, root, str(locked_spec["path"]), "inherited locked"
    )
    if locked_path.stat().st_size != locked_spec["bytes"]:
        raise ValueError("inherited locked byte size mismatch")
    if _file_sha256(locked_path) != locked_spec["file_sha256"]:
        raise ValueError("inherited locked file SHA mismatch")
    # Identity metadata is read from the sealed contract, never from locked JSONL.
    locked_binding = _mapping(
        attempt_contract["teacher_shards"]["splits"]["locked_holdout"],
        "sealed locked shard binding",
    )
    if locked_binding.get("records") != locked_spec["records"]:
        raise ValueError("sealed locked record count mismatch")
    if locked_binding.get("canonical_rows_sha256") != locked_spec[
        "canonical_rows_sha256"
    ]:
        raise ValueError("sealed locked canonical-row hash mismatch")
    if locked_binding.get("ordered_shards_sha256") != locked_spec[
        "ordered_shards_sha256"
    ]:
        raise ValueError("sealed locked ordered-shard hash mismatch")
    ordered_locked = locked_binding.get("ordered_shards")
    if not isinstance(ordered_locked, Sequence) or len(ordered_locked) != 1:
        raise ValueError("sealed inherited locked binding must contain one shard")
    sealed_file = _mapping(ordered_locked[0], "sealed locked file")
    for key, expected in (
        ("bytes", locked_spec["bytes"]),
        ("file_sha256", locked_spec["file_sha256"]),
        ("canonical_rows_sha256", locked_spec["canonical_rows_sha256"]),
        ("identity_sha256", locked_spec["identity_sha256"]),
        ("records", locked_spec["records"]),
    ):
        if sealed_file.get(key) != expected:
            raise ValueError(f"sealed inherited locked hash-chain mismatch: {key}")
    locked_identities = _sealed_locked_identities(attempt_contract)
    if _identity_digest(locked_identities) != locked_spec["identity_sha256"]:
        raise ValueError("sealed inherited locked identity digest mismatch")
    if union_identities & locked_identities:
        raise ValueError("inherited locked identities appear in exclusion train/cal union")

    manifest_spec = _mapping(provenance["training_manifest"], "training_manifest")
    manifest_path = _require_bound_path(
        attempt01_training_manifest,
        root,
        str(manifest_spec["path"]),
        "Attempt01 training manifest",
    )
    if _file_sha256(manifest_path) != manifest_spec["file_sha256"]:
        raise ValueError("Attempt01 training manifest SHA mismatch")
    training_manifest = _read_mapping(manifest_path, "Attempt01 training manifest")
    no_go_evidence = _audit_attempt01_no_go(training_manifest, manifest_spec)

    marker_relative = str(locked_spec["global_consumption_marker"])
    marker_path = (root / marker_relative).resolve()
    marker_hits = _find_consumption_markers(root, INHERITED_LOCKED_IDENTITY_SHA256)
    if marker_path.exists() or marker_hits:
        raise ValueError("inherited locked identity is already globally consumed")

    planned_seed_sets = _planned_seed_sets(plan)
    forbidden_seed_values = union_seeds | {seed for seed, _ in locked_identities}
    for name, values in planned_seed_sets.items():
        overlap = values & forbidden_seed_values
        if overlap:
            raise ValueError(f"Attempt02 planned {name} seed overlaps frozen data")

    fresh_generation = _fresh_generation_projection(plan)
    report: dict[str, Any] = {
        "schema": M43_ATTEMPT02_PREFLIGHT_SCHEMA,
        "status": "pass_frozen_before_fresh_generation",
        "plan": {
            "path": _path_token(plan_source, root),
            "file_sha256": plan_sha,
            "schema": plan["schema"],
        },
        "fresh_generation": fresh_generation,
        "exclusion_union": {
            **actual_union,
            "base_source_records": base["source_records"],
            "attempt01_train_calibration_records": len(attempt_identities),
            "attempt01_train_calibration_identity_sha256": (
                ATTEMPT01_TRAIN_CAL_IDENTITY_SHA256
            ),
            "attempt01_sources": attempt_reports,
        },
        "attempt01": {
            "data_contract": {
                "path": _path_token(contract_path, root),
                "file_sha256": _file_sha256(contract_path),
                "canonical_sha256": ATTEMPT01_CONTRACT_CANONICAL_SHA256,
            },
            "training_manifest": {
                "path": _path_token(manifest_path, root),
                "file_sha256": _file_sha256(manifest_path),
            },
            "no_go_evidence": no_go_evidence,
        },
        "inherited_locked": {
            "classification": "inherited_unopened",
            "path": _path_token(locked_path, root),
            "records": locked_spec["records"],
            "bytes": locked_spec["bytes"],
            "file_sha256": locked_spec["file_sha256"],
            "canonical_rows_sha256": locked_spec["canonical_rows_sha256"],
            "identity_sha256": locked_spec["identity_sha256"],
            "ordered_shards_sha256": locked_spec["ordered_shards_sha256"],
            "source_data_contract_canonical_sha256": (
                ATTEMPT01_CONTRACT_CANONICAL_SHA256
            ),
            "content_parse_count": 0,
            "model_evaluation_count": 0,
            "structural_byte_hash_audit_only": True,
            "source_search": {"candidate_samples": 2, "evaluation_samples": 16},
            "fresh_search": {"candidate_samples": 2, "evaluation_samples": 64},
            "search_budget_mismatch": locked_spec["fresh_search_budget_mismatch"],
        },
        "global_locked_consumption": {
            "identity_sha256": INHERITED_LOCKED_IDENTITY_SHA256,
            "canonical_marker_path": marker_relative,
            "matching_marker_count": 0,
            "status": "unconsumed_preflight",
            "claim_before_content_open_required": True,
            "claim_is_consuming_even_on_crash": True,
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
    context = {
        "plan": plan,
        "root": root,
        "exclusion_identities": union_identities,
        "exclusion_seeds": union_seeds,
        "exclusion_fingerprints": union_fingerprints,
        "locked_identities": locked_identities,
        "locked_seeds": {seed for seed, _ in locked_identities},
        "locked_fingerprints": {fingerprint for _, fingerprint in locked_identities},
    }
    return report, context


def finalize_fresh_data(
    *,
    plan_path: str | Path,
    repo_root: str | Path,
    attempt01_data_contract: str | Path,
    attempt01_training_manifest: str | Path,
    attempt01_train: str | Path,
    attempt01_calibration: str | Path,
    inherited_locked: str | Path,
    preflight_receipt: str | Path,
    train: Sequence[str | Path],
    calibration: Sequence[str | Path],
) -> dict[str, Any]:
    declared_preflight = _read_mapping(Path(preflight_receipt), "Attempt02 preflight")
    _validate_embedded_digest(
        declared_preflight,
        field="receipt_sha256",
        expected=None,
        location="Attempt02 preflight",
    )
    if declared_preflight.get("schema") != M43_ATTEMPT02_PREFLIGHT_SCHEMA:
        raise ValueError("Attempt02 preflight schema mismatch")
    if declared_preflight.get("status") != "pass_frozen_before_fresh_generation":
        raise ValueError("Attempt02 preflight did not pass")
    actual_preflight, context = _audit_preflight(
        plan_path=plan_path,
        repo_root=repo_root,
        attempt01_data_contract=attempt01_data_contract,
        attempt01_training_manifest=attempt01_training_manifest,
        attempt01_train=attempt01_train,
        attempt01_calibration=attempt01_calibration,
        inherited_locked=inherited_locked,
    )
    if declared_preflight != actual_preflight:
        raise ValueError("Attempt02 preflight no longer matches frozen sources")

    plan = context["plan"]
    root = context["root"]
    paths = {
        "train": tuple(Path(path).resolve() for path in train),
        "calibration": tuple(Path(path).resolve() for path in calibration),
    }
    if any(not split_paths for split_paths in paths.values()):
        raise ValueError("Attempt02 finalize requires explicit train and calibration shards")
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    identities_by_split: dict[str, set[tuple[int, str]]] = {}
    profile_counts: dict[str, Counter[str]] = {}
    shard_audits: dict[str, list[dict[str, Any]]] = {}
    roots_per_shard = int(plan["budget"]["roots_per_shard"])
    for split in _FRESH_SPLITS:
        spec = _mapping(plan["fresh_splits"][split], f"fresh_splits.{split}")
        expected_records = int(spec["roots"])
        expected_shards = expected_records // roots_per_shard
        if len(paths[split]) != expected_shards:
            raise ValueError(f"Attempt02 {split} shard count disagrees with plan")
        normalized = [os.path.normcase(str(path)) for path in paths[split]]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Attempt02 {split} contains duplicate shard paths")
        audits = [
            read_and_audit_shard(path, expected_split=split, require_paired_delta=True)
            for path in paths[split]
        ]
        if any(audit["records"] != roots_per_shard for audit in audits):
            raise ValueError(f"Attempt02 {split} shard is not exactly roots-per-shard")
        shard_audits[split] = audits
        rows = [row for path in paths[split] for row in _read_jsonl(path)]
        if len(rows) != expected_records:
            raise ValueError(f"Attempt02 {split} record count disagrees with plan")
        identities = {_row_identity(row, f"Attempt02 {split}") for row in rows}
        if len(identities) != len(rows):
            raise ValueError(f"Attempt02 {split} contains duplicate identities")
        observed_seeds = {seed for seed, _ in identities}
        expected_seeds = _root_seed_schedule(spec)
        if observed_seeds != expected_seeds:
            raise ValueError(f"Attempt02 {split} hand seeds disagree with plan")
        counts: Counter[str] = Counter()
        seed_to_index = {
            int(spec["seed_start"]) + int(spec["seed_stride"]) * index: index
            for index in range(expected_records)
        }
        for row in rows:
            provenance = _mapping(row.get("provenance"), f"{split} provenance")
            if provenance.get("current_profile_resolved") is not False:
                raise ValueError("Attempt02 fresh data resolved current profile")
            profile = str(provenance.get("root_profile", ""))
            if profile not in _PROFILES:
                raise ValueError("Attempt02 fresh data contains unknown root profile")
            counts[profile] += 1
            search = _mapping(row.get("search_config"), f"{split} search_config")
            if search.get("candidate_samples") != plan["teacher_search"][
                "candidate_samples"
            ] or search.get("evaluation_samples") != plan["teacher_search"][
                "evaluation_samples"
            ]:
                raise ValueError("Attempt02 fresh data does not use frozen c2/e64")
            seed = int(row["hand_seed"])
            index = seed_to_index[seed]
            stride = int(spec["seed_stride"])
            for field in (
                "candidate_seed",
                "evaluation_seed",
                "child_policy_seed",
            ):
                start_field = f"{field}_start"
                expected = int(spec[start_field]) + stride * (
                    index // roots_per_shard
                )
                if search.get(field) != expected:
                    raise ValueError(f"Attempt02 {split} {field} schedule mismatch")
        expected_profile_counts = {
            str(row["profile"]): int(row["roots"][split])
            for row in plan["root_population"]
        }
        if dict(counts) != expected_profile_counts:
            raise ValueError(f"Attempt02 {split} root-profile quota mismatch")
        rows_by_split[split] = rows
        identities_by_split[split] = identities
        profile_counts[split] = counts

    train_ids = identities_by_split["train"]
    calibration_ids = identities_by_split["calibration"]
    if train_ids & calibration_ids:
        raise ValueError("Attempt02 fresh train/calibration identities overlap")
    fresh_seeds = {seed for seed, _ in train_ids | calibration_ids}
    fresh_fingerprints = {fingerprint for _, fingerprint in train_ids | calibration_ids}
    if fresh_seeds & context["exclusion_seeds"]:
        raise ValueError("Attempt02 fresh hand seed overlaps exclusion union")
    if fresh_fingerprints & context["exclusion_fingerprints"]:
        raise ValueError("Attempt02 fresh fingerprint overlaps exclusion union")
    if fresh_seeds & context["locked_seeds"]:
        raise ValueError("Attempt02 fresh hand seed overlaps inherited locked")
    if fresh_fingerprints & context["locked_fingerprints"]:
        raise ValueError("Attempt02 fresh fingerprint overlaps inherited locked")

    teacher_shards: dict[str, Any] = {
        "schema": M43_ATTEMPT02_TEACHER_SHARDS_SCHEMA,
        "splits": {
            split: _ordered_shard_binding(
                paths[split], repo_root=root, expected_split=split
            )
            for split in _FRESH_SPLITS
        },
    }
    teacher_shards["all_fresh_splits_sha256"] = canonical_manifest_sha256(
        teacher_shards
    )
    partition = _partition_calibration(rows_by_split["calibration"], plan)
    report: dict[str, Any] = {
        "schema": M43_ATTEMPT02_DATA_CONTRACT_SCHEMA,
        "status": "pass_fresh_train_calibration_sealed_inherited_locked_unopened",
        "plan_sha256": actual_preflight["plan"]["file_sha256"],
        "preflight": {
            "receipt_sha256": actual_preflight["receipt_sha256"],
            "file_sha256": _file_sha256(Path(preflight_receipt)),
        },
        "fresh_generation": actual_preflight["fresh_generation"],
        "exclusion_union": actual_preflight["exclusion_union"],
        "fresh_splits": {
            split: {
                "records": len(rows_by_split[split]),
                "shards": len(paths[split]),
                "identity_sha256": _identity_digest(identities_by_split[split]),
                "profile_counts": dict(sorted(profile_counts[split].items())),
                "audited_paired_delta_records": sum(
                    int(row["paired_delta_records"]) for row in shard_audits[split]
                ),
            }
            for split in _FRESH_SPLITS
        },
        "calibration_partition": partition,
        "teacher_shards": teacher_shards,
        "teacher_shards_all_fresh_splits_sha256": teacher_shards[
            "all_fresh_splits_sha256"
        ],
        "inherited_locked": actual_preflight["inherited_locked"],
        "global_locked_consumption": actual_preflight["global_locked_consumption"],
        "freshness": {
            "train_calibration_identity_overlap": 0,
            "exclusion_hand_seed_overlap": 0,
            "exclusion_observation_fingerprint_overlap": 0,
            "inherited_locked_hand_seed_overlap": 0,
            "inherited_locked_observation_fingerprint_overlap": 0,
        },
        "runtime": actual_preflight["runtime"],
    }
    report["contract_sha256"] = _self_digest(report, "contract_sha256")
    return report


def _fresh_generation_projection(plan: Mapping[str, Any]) -> dict[str, Any]:
    budget = _mapping(plan["budget"], "budget")
    splits: dict[str, Any] = {}
    for split in _FRESH_SPLITS:
        spec = _mapping(plan["fresh_splits"][split], f"fresh_splits.{split}")
        splits[split] = {
            "roots": int(spec["roots"]),
            "shards": int(spec["roots"]) // int(budget["roots_per_shard"]),
            "seed_start": int(spec["seed_start"]),
            "seed_stride": int(spec["seed_stride"]),
            "candidate_seed_start": int(spec["candidate_seed_start"]),
            "evaluation_seed_start": int(spec["evaluation_seed_start"]),
            "child_policy_seed_start": int(spec["child_policy_seed_start"]),
            "hand_seed_stride_scope": "root",
            "phase_seed_stride_scope": str(spec["phase_seed_stride_scope"]),
        }
    return {
        "fresh_roots": int(budget["fresh_roots"]),
        "roots_per_shard": int(budget["roots_per_shard"]),
        "fresh_shards": int(budget["fresh_shards"]),
        "splits": splits,
        "teacher_search": {
            key: plan["teacher_search"][key]
            for key in (
                "candidate_samples",
                "evaluation_samples",
                "common_random_futures",
                "candidate_evaluation_rng_disjoint",
                "batch_child_selectors",
                "native_batch_threads",
            )
        },
        "root_population": [
            {
                "profile": row["profile"],
                "train_roots": int(row["roots"]["train"]),
                "calibration_roots": int(row["roots"]["calibration"]),
            }
            for row in plan["root_population"]
        ],
        "fixed_baseline_profile": plan["fixed_baseline_profile"],
        "fixed_t2_profile": plan["fixed_continuation"]["t2_profile"],
    }


def _planned_seed_sets(plan: Mapping[str, Any]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for split in _FRESH_SPLITS:
        spec = plan["fresh_splits"][split]
        roots = int(spec["roots"])
        roots_per_shard = int(plan["budget"]["roots_per_shard"])
        stride = int(spec["seed_stride"])
        for field in (
            "seed_start",
            "candidate_seed_start",
            "evaluation_seed_start",
            "child_policy_seed_start",
        ):
            schedule_count = roots if field == "seed_start" else roots // roots_per_shard
            result[f"{split}.{field}"] = {
                int(spec[field]) + stride * index for index in range(schedule_count)
            }
    return result


def _root_seed_schedule(spec: Mapping[str, Any]) -> set[int]:
    return {
        int(spec["seed_start"]) + int(spec["seed_stride"]) * index
        for index in range(int(spec["roots"]))
    }


def _partition_calibration(
    rows: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]
) -> dict[str, Any]:
    spec = _mapping(plan["calibration_partition"], "calibration_partition")
    salt = str(spec.get("identity_hash_salt", ""))
    if not salt:
        raise ValueError("Attempt02 calibration partition salt is empty")
    per_role = int(spec["roots_per_profile_per_role"])
    by_profile: dict[str, list[tuple[str, tuple[int, str]]]] = defaultdict(list)
    for row in rows:
        identity = _row_identity(row, "Attempt02 calibration")
        profile = str(_mapping(row.get("provenance"), "provenance").get("root_profile"))
        token = hashlib.sha256(
            (
                f"{M43_ATTEMPT02_CALIBRATION_PARTITION_SCHEMA}\0{salt}\0"
                f"{identity[0]}\0{identity[1]}"
            ).encode("utf-8")
        ).hexdigest()
        by_profile[profile].append((token, identity))
    roles: dict[str, list[tuple[int, str]]] = {
        "safety_fit": [],
        "threshold_lock": [],
    }
    role_profile_counts = {role: Counter() for role in roles}
    for profile in _PROFILES:
        ordered = sorted(by_profile.get(profile, ()))
        if len(ordered) != 2 * per_role:
            raise ValueError(f"Attempt02 calibration profile quota mismatch: {profile}")
        for _, identity in ordered[:per_role]:
            roles["safety_fit"].append(identity)
            role_profile_counts["safety_fit"][profile] += 1
        for _, identity in ordered[per_role:]:
            roles["threshold_lock"].append(identity)
            role_profile_counts["threshold_lock"][profile] += 1
    if set(roles["safety_fit"]) & set(roles["threshold_lock"]):
        raise ValueError("Attempt02 calibration roles overlap")
    return {
        "schema": M43_ATTEMPT02_CALIBRATION_PARTITION_SCHEMA,
        "method": "profile_stratified_identity_hash_v1",
        "safety_fit": _role_manifest(
            roles["safety_fit"], role_profile_counts["safety_fit"]
        ),
        "threshold_lock": _role_manifest(
            roles["threshold_lock"], role_profile_counts["threshold_lock"]
        ),
        "overlap": 0,
        "inherited_locked_used": False,
    }


def _role_manifest(
    identities: Iterable[tuple[int, str]], profile_counts: Counter[str]
) -> dict[str, Any]:
    values = sorted(identities)
    return {
        "records": len(values),
        "identity_sha256": _identity_digest(values),
        "profile_counts": dict(sorted(profile_counts.items())),
        "identities": [_identity_json(value) for value in values],
    }


def _sealed_locked_identities(
    attempt_contract: Mapping[str, Any],
) -> set[tuple[int, str]]:
    split = _mapping(
        attempt_contract["base_audit"]["splits"]["locked_holdout"],
        "sealed base audit locked split",
    )
    shards = split.get("shards")
    if not isinstance(shards, Sequence) or not shards:
        raise ValueError("sealed base audit lacks locked identity metadata")
    identities: set[tuple[int, str]] = set()
    total = 0
    for raw in shards:
        shard = _mapping(raw, "sealed locked audit shard")
        seeds = shard.get("hand_seeds")
        fingerprints = shard.get("observation_fingerprints")
        if not isinstance(seeds, Sequence) or not isinstance(fingerprints, Sequence):
            raise ValueError("sealed locked audit identity lists missing")
        if len(seeds) != len(fingerprints):
            raise ValueError("sealed locked audit identity lists differ in length")
        total += len(seeds)
        for seed, fingerprint in zip(seeds, fingerprints):
            identities.add((int(seed), _sha256_text(fingerprint, "locked fingerprint")))
    if total != INHERITED_LOCKED_RECORDS or len(identities) != total:
        raise ValueError("sealed locked identity metadata is not unique locked40")
    return identities


def _audit_attempt01_no_go(
    manifest: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    if manifest.get("schema") != "hu_m4_t1_joint_training_manifest_v2":
        raise ValueError("Attempt01 training manifest schema mismatch")
    if manifest.get("promotion_status") != expected["required_promotion_status"]:
        raise ValueError("Attempt01 was not calibration No-Go")
    calibration = _mapping(manifest.get("calibration"), "Attempt01 calibration")
    if calibration.get("status") != expected["required_calibration_status"]:
        raise ValueError("Attempt01 calibration status changed")
    zero = int(expected["required_candidate_overrides"])
    candidate_counts = {
        "calibration": calibration.get("candidate_overrides"),
        "safety_fit": _mapping(calibration.get("safety_fit"), "safety_fit").get(
            "candidate_overrides"
        ),
        "threshold_lock": _mapping(
            calibration.get("threshold_lock"), "threshold_lock"
        ).get("candidate_overrides"),
        "selected_fires": _mapping(
            calibration.get("selected_metrics"), "selected_metrics"
        ).get("fires"),
    }
    if any(value != zero for value in candidate_counts.values()):
        raise ValueError("Attempt01 No-Go zero-fire evidence changed")
    sweep = calibration.get("threshold_sweep")
    if not isinstance(sweep, Sequence) or not sweep:
        raise ValueError("Attempt01 threshold sweep evidence missing")
    if any(_mapping(row, "threshold sweep row").get("fires") != 0 for row in sweep):
        raise ValueError("Attempt01 threshold sweep unexpectedly fired")
    if manifest.get("locked_holdout") != {
        "status": expected["required_locked_status"]
    }:
        raise ValueError("Attempt01 locked status changed")
    if manifest.get("locked_holdout_used_for_threshold_or_training") is not False:
        raise ValueError("Attempt01 trainer used locked holdout")
    if calibration.get("locked_holdout_used") is not False:
        raise ValueError("Attempt01 calibration used locked holdout")
    if manifest.get("threshold_adaptation_after_calibration") is not False:
        raise ValueError("Attempt01 adapted its threshold after calibration")
    return {
        "promotion_status": manifest["promotion_status"],
        "calibration_status": calibration["status"],
        "candidate_overrides": candidate_counts,
        "threshold_sweep_entries": len(sweep),
        "threshold_sweep_fires": 0,
        "locked_holdout_status": manifest["locked_holdout"]["status"],
        "locked_holdout_used_for_threshold_or_training": False,
        "threshold_adaptation_after_calibration": False,
    }


def _find_consumption_markers(root: Path, identity_sha256: str) -> list[str]:
    matches: list[str] = []
    for marker in root.rglob("M43_LOCKED_CONSUMED.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid global locked consumption marker: {marker}") from exc
        if identity_sha256 in json.dumps(payload, sort_keys=True):
            matches.append(_path_token(marker.resolve(), root))
    return sorted(matches)


def _require_bound_path(
    supplied: str | Path,
    root: Path,
    declared_relative: str,
    location: str,
) -> Path:
    actual = Path(supplied).resolve()
    declared = (root / declared_relative).resolve()
    if os.path.normcase(str(actual)) != os.path.normcase(str(declared)):
        raise ValueError(f"{location} path does not match frozen plan")
    if not actual.is_file():
        raise ValueError(f"{location} file is missing")
    return actual


def _path_token(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path, location: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return value


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _integer(value: Any, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _sha256_text(value: Any, location: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return text


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return canonical_manifest_sha256(unsigned)


def _validate_embedded_digest(
    value: Mapping[str, Any],
    *,
    field: str,
    expected: str | None,
    location: str,
) -> None:
    declared = _sha256_text(value.get(field), f"{location}.{field}")
    actual = _self_digest(value, field)
    if declared != actual:
        raise ValueError(f"{location} embedded digest mismatch")
    if expected is not None and declared != expected:
        raise ValueError(f"{location} frozen digest mismatch")


def _write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "finalize-fresh"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--plan", required=True)
        sub.add_argument("--repo-root", default=".")
        sub.add_argument("--attempt01-data-contract", required=True)
        sub.add_argument("--attempt01-training-manifest", required=True)
        sub.add_argument("--attempt01-train", required=True)
        sub.add_argument("--attempt01-calibration", required=True)
        sub.add_argument("--inherited-locked", required=True)
        sub.add_argument("--output", required=True)
        if command == "finalize-fresh":
            sub.add_argument("--preflight-receipt", required=True)
            sub.add_argument("--train", action="append", required=True)
            sub.add_argument("--calibration", action="append", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    common = {
        "plan_path": args.plan,
        "repo_root": args.repo_root,
        "attempt01_data_contract": args.attempt01_data_contract,
        "attempt01_training_manifest": args.attempt01_training_manifest,
        "attempt01_train": args.attempt01_train,
        "attempt01_calibration": args.attempt01_calibration,
        "inherited_locked": args.inherited_locked,
    }
    if args.command == "preflight":
        payload = build_preflight_receipt(**common)
    else:
        payload = finalize_fresh_data(
            **common,
            preflight_receipt=args.preflight_receipt,
            train=args.train,
            calibration=args.calibration,
        )
    _write_immutable_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
