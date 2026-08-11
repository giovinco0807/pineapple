"""Fail-closed validation for the M3.1 T3 Step 6d contract freeze.

Passing this validator freezes an experiment ladder and disjoint seed
namespaces only.  It does not run search, launch cloud work, generate rows,
train a model, add a profile, resolve ``current``, or activate a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m31_t3_step6d_contract import (
    EXPECTED_STEP6D_CONTRACT_BYTE_SHA256,
    EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
    HISTORICAL_CONFIG_SEED_MAX,
    PLANNED_SEED_COUNT,
    PLANNED_SEED_MAX,
    PLANNED_SEED_MIN,
    PLANNED_SEED_SET_SHA256,
    SEED_STRIDE,
    STEP6D_CONTRACT_SCHEMA,
    STEP6D_SCHEDULE_SCHEMA,
    STEP6D_VALIDATION_SCHEMA,
    canonical_bytes,
    canonical_sha256,
    require_exact_keys,
    seed_schedule_payload,
    validate_seed_schedule,
)


_TOP_LEVEL_KEYS = (
    "schema",
    "status_date",
    "status",
    "decision",
    "scope",
    "anchors",
    "step6c_fixed_outcome",
    "immutable_boundaries",
    "seed_contract",
    "substeps",
    "reuse_matrix",
    "activation_guards",
    "forbidden",
    "next_step",
)

_ANCHOR_SPECS: dict[str, tuple[str, str, str | None]] = {
    "step5_contract": (
        "configs/hu_joint_policy_m31_t3_step5_contract.json",
        "5f9fab4d844f1a7411a99f9dada2fe289314d4dea578d0c0f6a6d14918263c93",
        "04c4298feaed78f327f7fb6601f70a6821c3a91dcd2996cc34becbc4bb38e0b4",
    ),
    "step6c_contract": (
        "configs/hu_joint_policy_m31_t3_step6c_contract.json",
        "6d9cc9f0bb79423ead58eea24b84474085071f9dd0dbda8cfc5acf89648fb8ce",
        "0a745fb746766423424f71a00292dac86e6102d15187ef763cc0e7d30ed49123",
    ),
    "step6c_quality": (
        "outputs/hu_joint_policy/m31_t3_step6c/quality100_validation_v1.json",
        "66ddae312ee9561a0101ba94deb5e5e867d407dda78af00b03da10fc96739731",
        None,
    ),
    "step6c_completion_audit": (
        "docs/hu_joint_policy_m31_t3_step6c_completion_audit.md",
        "148b6839572d29e065d5bce7118eb8a5c9d245594363d92f689c2734d6937b85",
        None,
    ),
    "policy_registry": (
        "src/ofc_regular/ai_profiles.py",
        "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3",
        None,
    ),
    "accepted_step6c_native": (
        "outputs/gcp_runs/regular-hu-m31-step6c-quality-20260717-003/"
        "package_src/native/release/libofc_hu_m3_engine.so",
        "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0",
        None,
    ),
    "accepted_feature_encoder": (
        "outputs/gcp_runs/regular-hu-m31-step6c-quality-20260717-003/"
        "package_src/target/release/libofc_stage3_feature_encoder.so",
        "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411",
        None,
    ),
}

# These section hashes are deliberately independent of the whole-contract
# hash.  Tests and callers may disable only the outer hash in order to prove
# that every gate family still fails closed on its own.
_SECTION_HASHES = {
    "anchors": "7b47bafeef3aab80cdd7a9c445ee91b5573f2b5fadfffe611806c6affa1e30c3",
    "step6c_fixed_outcome": (
        "0dffdf1721412d2a8dadc9b83d93e1178379dc35b4a7856f11c7998fc7fff0dd"
    ),
    "immutable_boundaries": (
        "2a61032006d193d4ed899938f6acbecb6179cb32d91bd5e4f9b3b3362e9d53b2"
    ),
    "seed_contract": (
        "efc4a27558e3a540f7776603968b4a72bb9655034278200b2f29d32d3e9335a7"
    ),
    "substeps": "35b65f78193197ce064ca867588e795d527a85567da6d1dc0da4fd5f090cecbf",
    "reuse_matrix": (
        "b67a95ee1863165500c32a7645dd61aed05ec8283a8a6afc606328ab89437391"
    ),
    "activation_guards": (
        "07c86d126264b82458903eb266b461e26127e591a6e2c1396d9333d406a23074"
    ),
    "forbidden": "ef2a8bbfd8e613528ee94b07d51cd220045146ada525ff0c581418f0a1f03452",
}

_SUBSTEP_HASHES = {
    "contract_freeze": (
        "73944ecfc3fe370e600863733c672a5dec320d5145deeb516360da630bf124b0"
    ),
    "performance_development": (
        "c98dbeddd26907071327c05d2a0f6ebef88258d85844e16dfb9407022a9dc47e"
    ),
    "performance_lock": (
        "cc385ca51400c22096680a28c581c7760321dfd360cc97a5b2273f611ad25042"
    ),
    "quality_pilot": (
        "ea4313d71535bcb6f9718fbbc314f5bf965ae9da903ad22b5c909396a72168ac"
    ),
    "artifact_rebuild": (
        "f6e576d0017bb4557f7ef8a199c531bd5e422344f1934bf2b13dd742f2beab9d"
    ),
    "model_and_threshold_freeze": (
        "bd7afc91c7fcddb44913d0d32948d8144ed3994f517ef3ed558858faf9c3377d"
    ),
    "development_population": (
        "513bdb59f1c48c705824bbdaf5e3aaba99dafe6d4f11a1c1e838bb80a9b75381"
    ),
    "locked_promotion": (
        "95f57975d6a22794803d4e4042b442f9844efc24486fb114d3fd871d650f218d"
    ),
}

_REUSE_ROWS = (
    "step5_contract",
    "step5_seed_ranges",
    "step6a_step6b_rows",
    "step6c_rows",
    "performance_development",
    "performance_lock",
    "quality_pilot",
    "train",
    "safety_fit",
    "threshold_lock",
    "diagnostic_teacher_holdout",
    "development_population",
    "abr_development",
    "locked_population",
    "locked_abr",
    "legacy_t3_models",
    "m30_exact_t4",
)
_REUSE_ROW_KEYS = (
    "allowed_use",
    "training_eligible",
    "threshold_eligible",
    "quality_evidence",
    "promotion_evidence",
)
_ACTIVATION_KEYS = (
    "performance_execution_authorized",
    "quality_pilot_authorized",
    "artifact_fanout_authorized",
    "training_authorized",
    "named_profile_added",
    "current_profile_changed",
    "runtime_policy_activated",
    "full_replacement_enabled",
    "m31_complete",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _require_section_hash(name: str, value: Any) -> None:
    if canonical_sha256(value) != _SECTION_HASHES[name]:
        raise ValueError(f"Step 6d {name} gates changed")


def _validate_anchor_payloads(
    anchors: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    require_exact_keys(anchors, tuple(_ANCHOR_SPECS), "Step 6d anchors")
    observed: dict[str, Any] = {}
    for name, (relative_path, expected_byte_sha, expected_canonical_sha) in (
        _ANCHOR_SPECS.items()
    ):
        entry = _mapping(anchors.get(name), f"Step 6d anchor {name}")
        expected_keys = ("path", "byte_sha256")
        if expected_canonical_sha is not None:
            expected_keys += ("canonical_sha256",)
        require_exact_keys(entry, expected_keys, f"Step 6d anchor {name}")
        if (
            entry.get("path") != relative_path
            or entry.get("byte_sha256") != expected_byte_sha
            or (
                expected_canonical_sha is not None
                and entry.get("canonical_sha256") != expected_canonical_sha
            )
        ):
            raise ValueError(f"Step 6d anchor declaration changed: {name}")
        path = repo_root / relative_path
        if not path.is_file() or _sha256(path) != expected_byte_sha:
            raise ValueError(f"Step 6d anchor file/hash mismatch: {name}")
        if expected_canonical_sha is not None:
            payload = _load_json(path)
            if canonical_sha256(payload) != expected_canonical_sha:
                raise ValueError(f"Step 6d anchor canonical mismatch: {name}")
        observed[name] = {
            "path": relative_path,
            "byte_sha256": expected_byte_sha,
            "canonical_sha256": expected_canonical_sha,
        }

    quality = _load_json(repo_root / _ANCHOR_SPECS["step6c_quality"][0])
    gates = _mapping(quality.get("gates"), "Step 6c fixed gates")
    false_gates = {name for name, passed in gates.items() if passed is False}
    primary_by_seat = _mapping(
        _mapping(quality.get("performance"), "Step 6c performance").get(
            "primary_latency_by_seat"
        ),
        "Step 6c primary latency",
    )
    first = _mapping(primary_by_seat.get("first"), "Step 6c first latency")
    if (
        quality.get("schema") != "hu_m31_t3_step6c_quality_validation_v1"
        or quality.get("status") != "no_go"
        or quality.get("decision")
        != "production_label_quality_pilot_no_go_no_same_data_reselection"
        or quality.get("all_gates_passed") is not False
        or false_gates
        != {"all_shard_summaries_pass", "first_primary_p95_within_180_seconds"}
        or first.get("p95_seconds") != 326.2574110039998
        or quality.get("training_eligible") is not False
        or quality.get("production_fanout_authorized") is not False
        or quality.get("current_profile_changed") is not False
        or quality.get("named_profile_added") is not False
        or quality.get("runtime_policy_activated") is not False
        or quality.get("m31_complete") is not False
        or quality.get("teacher_values_are_realized_match_ev") is not False
    ):
        raise ValueError("Step 6c fixed No-Go outcome changed")
    return observed


def _validate_fixed_outcome(contract: Mapping[str, Any]) -> None:
    outcome = _mapping(contract.get("step6c_fixed_outcome"), "Step 6c outcome")
    _require_section_hash("step6c_fixed_outcome", outcome)
    if (
        outcome.get("status") != "complete_no_go"
        or outcome.get("observed_first_p95_seconds") != 326.2574110039998
        or outcome.get("frozen_first_p95_limit_seconds") != 180.0
        or outcome.get("confirmation_and_integrity_gates_passed") is not True
        or any(
            outcome.get(name) is not False
            for name in (
                "reopened",
                "reseed_allowed",
                "extension_allowed",
                "threshold_reselection_allowed",
                "rows_training_eligible",
                "quality_evidence_reused_by_step6d",
            )
        )
    ):
        raise ValueError("Step 6c fixed No-Go boundary changed")


def _validate_immutable_boundaries(contract: Mapping[str, Any]) -> None:
    immutable = _mapping(contract.get("immutable_boundaries"), "boundaries")
    _require_section_hash("immutable_boundaries", immutable)
    if (
        any(
            immutable.get(name) is not True
            for name in (
                "step5_numeric_gates_reused",
                "step5_seed_ranges_retired_from_step6d",
                "step6c_no_go_preserved",
                "m30_exact_t4_semantics_preserved",
            )
        )
        or any(
            immutable.get(name) is not False
            for name in (
                "hidden_opponent_discards_allowed",
                "sample_budget_reduction_allowed",
                "legal_action_pruning_allowed",
                "approximate_t4_allowed",
                "teacher_values_are_realized_match_ev",
                "current_profile_resolution_allowed",
                "named_profile_addition_allowed",
                "runtime_activation_allowed",
                "production_fanout_authorized",
                "training_authorized",
            )
        )
    ):
        raise ValueError("Step 6d immutable boundary changed")


def _validate_seed_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    seeds = _mapping(contract.get("seed_contract"), "Step 6d seed contract")
    _require_section_hash("seed_contract", seeds)
    require_exact_keys(
        seeds,
        (
            "schema",
            "seed_stride",
            "formula",
            "historical_config_seed_max_at_freeze",
            "planned_seed_min",
            "planned_seed_max",
            "planned_seed_count",
            "planned_seed_set_sha256",
            "all_namespaces_globally_disjoint",
            "all_roles_globally_disjoint",
            "alternate_seed_after_results_allowed",
            "posthoc_extension_allowed",
            "step5_seed_values_reused",
            "schedules",
        ),
        "Step 6d seed contract",
    )
    schedules = _array(seeds.get("schedules"), "Step 6d schedules")
    if (
        seeds.get("schema") != STEP6D_SCHEDULE_SCHEMA
        or seeds.get("seed_stride") != SEED_STRIDE
        or seeds.get("formula") != "namespace_seed_base + seed_stride * index"
        or seeds.get("historical_config_seed_max_at_freeze")
        != HISTORICAL_CONFIG_SEED_MAX
        or seeds.get("planned_seed_min") != PLANNED_SEED_MIN
        or seeds.get("planned_seed_max") != PLANNED_SEED_MAX
        or seeds.get("planned_seed_count") != PLANNED_SEED_COUNT
        or seeds.get("planned_seed_set_sha256") != PLANNED_SEED_SET_SHA256
        or seeds.get("all_namespaces_globally_disjoint") is not True
        or seeds.get("all_roles_globally_disjoint") is not True
        or seeds.get("alternate_seed_after_results_allowed") is not False
        or seeds.get("posthoc_extension_allowed") is not False
        or seeds.get("step5_seed_values_reused") is not False
        or schedules != seed_schedule_payload()
    ):
        raise ValueError("Step 6d seed split or namespace changed")
    return validate_seed_schedule()


def _validate_substeps(contract: Mapping[str, Any]) -> None:
    substeps = _mapping(contract.get("substeps"), "Step 6d substeps")
    _require_section_hash("substeps", substeps)
    require_exact_keys(substeps, tuple(_SUBSTEP_HASHES), "Step 6d substeps")
    for name, expected_hash in _SUBSTEP_HASHES.items():
        substep = _mapping(substeps.get(name), f"Step 6d substep {name}")
        if canonical_sha256(substep) != expected_hash:
            raise ValueError(f"Step 6d {name} split or gates changed")

    performance = _mapping(substeps["performance_lock"], "performance lock")
    quality = _mapping(substeps["quality_pilot"], "quality pilot")
    artifacts = _mapping(substeps["artifact_rebuild"], "artifact rebuild")
    model = _mapping(substeps["model_and_threshold_freeze"], "model freeze")
    population = _mapping(substeps["locked_promotion"], "locked promotion")
    if (
        performance.get("one_shot") is not True
        or performance.get("rerun_after_content_read_allowed") is not False
        or quality.get("training_eligible") is not False
        or quality.get("reseed_extension_or_threshold_change_allowed") is not False
        or artifacts.get("row_drop_or_replacement_allowed") is not False
        or model.get("top1_accuracy_is_promotion_gate") is not False
        or model.get("teacher_ev_lcb_is_runtime_gate") is not False
        or model.get("current_profile_changed") is not False
        or population.get("threshold_model_response_or_seed_change_after_open_allowed")
        is not False
        or population.get("posthoc_extension_allowed") is not False
        or population.get("current_profile_changed") is not False
    ):
        raise ValueError("Step 6d substep authorization boundary changed")


def _validate_reuse_matrix(contract: Mapping[str, Any]) -> None:
    reuse = _mapping(contract.get("reuse_matrix"), "Step 6d reuse matrix")
    _require_section_hash("reuse_matrix", reuse)
    require_exact_keys(reuse, _REUSE_ROWS, "Step 6d reuse matrix")
    for name in _REUSE_ROWS:
        require_exact_keys(
            _mapping(reuse.get(name), f"Step 6d reuse row {name}"),
            _REUSE_ROW_KEYS,
            f"Step 6d reuse row {name}",
        )
    true_by_flag = {
        flag: {name for name in _REUSE_ROWS if reuse[name].get(flag) is True}
        for flag in _REUSE_ROW_KEYS[1:]
    }
    if true_by_flag != {
        "training_eligible": {"train", "safety_fit"},
        "threshold_eligible": {"threshold_lock"},
        "quality_evidence": {"quality_pilot"},
        "promotion_evidence": {"locked_population", "locked_abr"},
    }:
        raise ValueError("Step 6d reuse eligibility changed")
    if any(not isinstance(reuse[name].get("allowed_use"), str) for name in _REUSE_ROWS):
        raise ValueError("Step 6d reuse purpose changed")


def _validate_activation_guards(contract: Mapping[str, Any]) -> None:
    guards = _mapping(contract.get("activation_guards"), "activation guards")
    _require_section_hash("activation_guards", guards)
    require_exact_keys(guards, _ACTIVATION_KEYS, "Step 6d activation guards")
    if any(value is not False for value in guards.values()):
        raise ValueError("Step 6d execution/training/profile activation enabled")


def validate_contract_payload(
    contract: Mapping[str, Any],
    *,
    repo_root: Path,
    verify_anchors: bool = True,
    verify_contract_hash: bool = True,
) -> dict[str, Any]:
    """Validate one in-memory contract without authorizing any execution."""

    require_exact_keys(contract, _TOP_LEVEL_KEYS, "Step 6d contract")
    if contract.get("schema") != STEP6D_CONTRACT_SCHEMA:
        raise ValueError("Step 6d contract schema changed")
    observed_canonical = canonical_sha256(contract)
    if (
        verify_contract_hash
        and observed_canonical != EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
    ):
        raise ValueError("Step 6d canonical contract SHA-256 changed")
    if (
        contract.get("status_date") != "2026-07-17"
        or contract.get("status") != "frozen_before_performance_development"
        or contract.get("decision")
        != (
            "freeze_disjoint_performance_repair_quality_artifact_model_and_"
            "promotion_ladder_without_reopening_step6c"
        )
        or contract.get("scope")
        != "m31_t3_step6d_contract_only_no_search_cloud_training_profile_or_activation"
        or contract.get("next_step")
        != (
            "validate this contract and then authorize only repeatable "
            "performance-development instrumentation on its disjoint "
            "100-paired-hand set"
        )
    ):
        raise ValueError("Step 6d identity/status/scope changed")

    anchors_payload = _mapping(contract.get("anchors"), "Step 6d anchors")
    _require_section_hash("anchors", anchors_payload)
    _validate_fixed_outcome(contract)
    _validate_immutable_boundaries(contract)
    seed_audit = _validate_seed_contract(contract)
    _validate_substeps(contract)
    _validate_reuse_matrix(contract)
    _validate_activation_guards(contract)
    forbidden = _array(contract.get("forbidden"), "Step 6d forbidden list")
    _require_section_hash("forbidden", forbidden)
    if len(forbidden) != 10 or not all(isinstance(item, str) for item in forbidden):
        raise ValueError("Step 6d forbidden boundary changed")

    anchors = (
        _validate_anchor_payloads(anchors_payload, repo_root)
        if verify_anchors
        else None
    )
    return {
        "schema": STEP6D_VALIDATION_SCHEMA,
        "status": "pass",
        "decision": (
            "step6d_contract_frozen_performance_development_execution_requires_"
            "separate_authorization"
        ),
        "scope": "contract_only_no_search_cloud_training_profile_or_activation",
        "contract_canonical_sha256": observed_canonical,
        "contract_byte_sha256": EXPECTED_STEP6D_CONTRACT_BYTE_SHA256,
        "checks": {
            "step6c_complete_no_go_anchored": verify_anchors,
            "policy_registry_and_current_resolution_anchored": verify_anchors,
            "accepted_step6c_binaries_anchored": verify_anchors,
            "all_67500_seed_values_unique_and_disjoint": True,
            "all_splits_and_seed_namespaces_frozen": True,
            "all_substep_gates_frozen": True,
            "reuse_matrix_frozen": True,
            "locked_sets_one_shot_and_no_posthoc_extension": True,
            "hidden_truth_approximate_t4_and_action_pruning_forbidden": True,
            "execution_training_profile_and_activation_forbidden": True,
        },
        "seed_audit": dict(seed_audit),
        "anchors": anchors,
        "performance_execution_authorized": False,
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "full_replacement_enabled": False,
        "m31_complete": False,
    }


def validate_contract_file(
    contract_path: Path,
    *,
    repo_root: Path,
    verify_anchors: bool = True,
    verify_contract_hash: bool = True,
) -> dict[str, Any]:
    if (
        verify_contract_hash
        and _sha256(contract_path) != EXPECTED_STEP6D_CONTRACT_BYTE_SHA256
    ):
        raise ValueError("Step 6d contract byte SHA-256 changed")
    return validate_contract_payload(
        _load_json(contract_path),
        repo_root=repo_root,
        verify_anchors=verify_anchors,
        verify_contract_hash=verify_contract_hash,
    )


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Step 6d validation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/hu_joint_policy_m31_t3_step6d_contract.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_contract_file(
        args.contract.resolve(), repo_root=args.repo_root.resolve()
    )
    if args.output is not None:
        _write_once(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "validate_contract_file", "validate_contract_payload"]
