"""Deterministic, fail-closed validation for the Attempt07 No-Go closeout.

This module does not create a winner freeze, authorize audit50, fit a model,
select a threshold, or activate runtime policy.  It validates the one frozen
development100 run and its one canonical selector result, then reports that
Attempt07 is closed as a No-Go.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ATTEMPT07_CLOSEOUT_SCHEMA = "hu_m43_attempt07_no_go_closeout_v1"
ATTEMPT07_CLOSEOUT_VALIDATION_SCHEMA = (
    "hu_m43_attempt07_no_go_closeout_validation_v1"
)

_RUN_NAME = "regular-hu-m43-attempt07-development100-20260714-150046"
_CLOSEOUT_PATH = "configs/hu_joint_policy_m43_attempt07_closeout.json"
_DEVELOPMENT_DIR = (
    "outputs/hu_joint_policy/m43_attempt07_development/" + _RUN_NAME
)
_PACKAGE_DIR = "outputs/gcp_runs/" + _RUN_NAME
_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
_ARMS = ("r32_v128", "r32_v64", "r64_v128", "r64_v64")

_ARTIFACTS: dict[str, dict[str, str]] = {
    "selection": {
        "path": _DEVELOPMENT_DIR + "/merged/development_arm_selection.json",
        "sha256": "53b797756747cbb71f89251020e7fdc86dabce75cd298507dc59c0764ed6373a",
    },
    "merged_input": {
        "path": _DEVELOPMENT_DIR + "/merged/teacher.jsonl",
        "sha256": "c15e018fee5e86cbcba8fadcfdaa5a13b25866efa1092c7b74177814049761a5",
    },
    "receive_receipt": {
        "path": _DEVELOPMENT_DIR + "/merged/receive_receipt.json",
        "sha256": "256f8d40541b81205e028e6a3d99aeb8bb15212807b10e476e0a64fed7c0ae1e",
    },
    "merge_receipt": {
        "path": _DEVELOPMENT_DIR + "/merged/merge_receipt.json",
        "sha256": "d41fa8928abe9330f613462bc995a56828ab2de33b041e0e92b5eddf22a42a8c",
    },
    "consumption_claim": {
        "path": (
            "outputs/hu_joint_policy/m43_attempt07_development/"
            + _RUN_NAME
            + "_OUTPUT_CONSUMED.json"
        ),
        "sha256": "3c79d75b1c1913939b0a0b739749b855a8ffdb5fc76dbf7ce226a2e1c0855699",
    },
    "canonical_plan": {
        "path": "configs/hu_joint_policy_m43_attempt07.json",
        "sha256": "8bf15f8d109a2e441e1c240e533c56df42a7e65d2eefd3b2750d1bd63758b189",
    },
    "packaged_plan": {
        "path": _PACKAGE_DIR + "/hu_joint_policy_m43_attempt07.json",
        "sha256": "8bf15f8d109a2e441e1c240e533c56df42a7e65d2eefd3b2750d1bd63758b189",
    },
    "selector_source": {
        "path": "src/ofc_regular/select_hu_m43_attempt07_development_arm.py",
        "sha256": "ec61f2d02364ea2b535e4f593e27eb0da0f8ac0955112d34868af7120bf3c0fc",
    },
    "package_manifest": {
        "path": _PACKAGE_DIR + "/manifest.json",
        "sha256": "2f42d41849eec7e5d94da756b088acc5c762d0df3bab2f6525457615da92476d",
    },
    "spot_authorization": {
        "path": _PACKAGE_DIR + "/development_spot_authorization.json",
        "sha256": "aa6c6a38ee13ad12bffb1e70383db31046dc93ab194b53e7fc4a68cf40c62f9a",
    },
    "shard_schedule": {
        "path": _PACKAGE_DIR + "/shards_manifest.jsonl",
        "sha256": "6a8f80670e79cd0336f968719f921be033d1e810657c722f590bb4a4fe0eae34",
    },
    "source_zip": {
        "path": _PACKAGE_DIR
        + "/ofc_regular_hu_m43_attempt07_development_source.zip",
        "sha256": "1ad4f8730e2731bf92768b5d4b407f4537f1dce6c4d7fd24504d497366ffe9e7",
    },
    "startup": {
        "path": _PACKAGE_DIR + "/startup_hu_m43_attempt07_development.sh",
        "sha256": "c285d1aa46d7c0dee6a5a8137ea5cbd07ce67739e1caccba9025bdfc54ea8ed6",
    },
    "ai_profiles": {
        "path": "src/ofc_regular/ai_profiles.py",
        "sha256": "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3",
    },
}

_SELECTION_INTEGRITY = {
    "action_mapping_violation_count": 0,
    "hidden_information_violation_count": 0,
    "nonfire_complete_trajectory_acceptance_deferred": True,
    "nonfire_complete_trajectory_cancellation_verified": False,
    "nonfire_exact_baseline_action_fallback_verified": True,
    "rng_domain_violation_count": 0,
}
_SELECTION_SCIENCE_BOUNDARY = {
    "assessment_source": "disjoint_A128_raw_selected_action_vs_baseline_only",
    "current_profile_mutated": False,
    "fit_performed": False,
    "full_replacement_enabled": False,
    "future_audit_authorized": False,
    "future_audit_opened": False,
    "gate_reselected": False,
    "runtime_policy_activated": False,
    "runtime_trajectory_cancellation_claimed": False,
    "teacher_values_are_realized_match_ev": False,
    "threshold_selected": False,
    "veto_values_used_as_development_quality_metrics": False,
}
_FORBIDDEN_ACTIONS = {
    "audit50_authorized": False,
    "audit50_opened": False,
    "fit_allowed": False,
    "fit_performed": False,
    "threshold_selection_allowed": False,
    "threshold_selected": False,
    "runtime_activation_allowed": False,
    "runtime_policy_activated": False,
    "current_profile_mutation_allowed": False,
    "current_profile_mutated": False,
    "full_replacement_enabled": False,
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    return _mapping(value, label)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} changed: expected {expected!r}, got {actual!r}")


def _artifact_path(repo_root: Path, identity: Mapping[str, str], label: str) -> Path:
    path = repo_root / identity["path"]
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    digest = _sha256(path)
    _require(digest, identity["sha256"], f"{label} SHA-256")
    return path


def _validate_selection(selection_path: Path) -> Mapping[str, Any]:
    encoded = selection_path.read_bytes()
    selection = _load_json(selection_path, "Attempt07 selection")
    canonical = (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if encoded != canonical:
        raise ValueError("Attempt07 selection is not the canonical selector encoding")

    _require(
        selection.get("schema"),
        "hu_m43_attempt07_development_arm_selection_v1",
        "selection schema",
    )
    _require(
        selection.get("status"),
        "no_go_close_attempt07_development",
        "selection status",
    )
    _require(selection.get("decision"), "no_go", "selection decision")
    _require(
        selection.get("decision_scope"),
        "close_attempt07_development_without_fit_threshold_or_audit",
        "selection decision scope",
    )
    _require(selection.get("selected_arm"), None, "selection selected_arm")
    _require(selection.get("winner"), None, "selection winner")

    source = _mapping(selection.get("source"), "selection source")
    _require(
        source.get("input_jsonl_sha256"),
        _ARTIFACTS["merged_input"]["sha256"],
        "selection input SHA-256",
    )
    _require(
        source.get("plan_sha256"),
        _ARTIFACTS["canonical_plan"]["sha256"],
        "selection plan SHA-256",
    )
    _require(
        source.get("selector_source_sha256"),
        _ARTIFACTS["selector_source"]["sha256"],
        "selection selector-source SHA-256",
    )

    population = _mapping(
        selection.get("development_population"), "development population"
    )
    _require(population.get("roots"), 100, "development roots")
    _require(population.get("profiles"), list(_PROFILES), "development profiles")
    _require(
        population.get("profile_counts"),
        {profile: 20 for profile in _PROFILES},
        "development profile counts",
    )
    _require(
        population.get("unique_hand_seeds"), 100, "development unique hand seeds"
    )
    _require(
        population.get("unique_observation_fingerprints"),
        100,
        "development unique observation fingerprints",
    )

    _require(selection.get("integrity"), _SELECTION_INTEGRITY, "selection integrity")
    _require(
        selection.get("science_boundary"),
        _SELECTION_SCIENCE_BOUNDARY,
        "selection science boundary",
    )
    selection_contract = _mapping(
        selection.get("selection_contract"), "selection contract"
    )
    _require(
        selection_contract.get("threshold_reselection_performed"),
        False,
        "selection threshold reselection",
    )

    arms = _mapping(selection.get("arms"), "selection arms")
    _require(set(arms), set(_ARMS), "selection arm names")
    for arm_name in _ARMS:
        arm = _mapping(arms[arm_name], f"selection arm {arm_name}")
        _require(arm.get("arm_name"), arm_name, f"{arm_name} arm name")
        _require(arm.get("eligible"), False, f"{arm_name} eligibility")
        gates = arm.get("gates")
        if not isinstance(gates, list) or not gates:
            raise ValueError(f"{arm_name} gates must be a non-empty list")
        by_name = {
            str(_mapping(gate, f"{arm_name} gate").get("name")): gate
            for gate in gates
        }
        if not any(gate.get("passed") is False for gate in by_name.values()):
            raise ValueError(f"{arm_name} must have at least one failed quality gate")
        for gate_name in (
            "action_mapping_violation_count",
            "rng_domain_violation_count",
            "hidden_information_violation_count",
        ):
            gate = _mapping(by_name.get(gate_name), f"{arm_name} {gate_name}")
            _require(gate.get("observed"), 0, f"{arm_name} {gate_name} observed")
            _require(gate.get("passed"), True, f"{arm_name} {gate_name} passed")
        fallback = _mapping(
            by_name.get("nonfire_exact_baseline_action_fallback"),
            f"{arm_name} nonfire fallback gate",
        )
        _require(fallback.get("observed"), True, f"{arm_name} nonfire fallback")
        _require(fallback.get("passed"), True, f"{arm_name} nonfire fallback passed")
    return selection


def _validate_run_chain(paths: Mapping[str, Path]) -> None:
    manifest = _load_json(paths["package_manifest"], "Attempt07 manifest")
    authorization = _load_json(
        paths["spot_authorization"], "Attempt07 Spot authorization"
    )
    claim = _load_json(paths["consumption_claim"], "Attempt07 consumption claim")
    receive = _load_json(paths["receive_receipt"], "Attempt07 receive receipt")
    merge = _load_json(paths["merge_receipt"], "Attempt07 merge receipt")

    manifest_sha = _ARTIFACTS["package_manifest"]["sha256"]
    authorization_sha = _ARTIFACTS["spot_authorization"]["sha256"]
    claim_sha = _ARTIFACTS["consumption_claim"]["sha256"]
    schedule_sha = _ARTIFACTS["shard_schedule"]["sha256"]
    merged_sha = _ARTIFACTS["merged_input"]["sha256"]
    plan_sha = _ARTIFACTS["canonical_plan"]["sha256"]

    _require(
        manifest.get("schema"),
        "hu_m43_attempt07_development_spot_package_v1",
        "manifest schema",
    )
    _require(manifest.get("run_name"), _RUN_NAME, "manifest run name")
    _require(
        manifest.get("status"),
        "frozen_package_only_no_root_opened",
        "manifest status",
    )
    _require(manifest.get("plan_sha256"), plan_sha, "manifest plan SHA-256")
    _require(manifest.get("schedule_sha256"), schedule_sha, "manifest schedule SHA-256")
    _require(manifest.get("total_roots"), 100, "manifest total roots")
    _require(manifest.get("total_shards"), 100, "manifest total shards")
    _require(manifest.get("roots_per_shard"), 1, "manifest roots per shard")
    _require(
        manifest.get("source_zip_sha256"),
        _ARTIFACTS["source_zip"]["sha256"],
        "manifest source ZIP SHA-256",
    )
    _require(
        manifest.get("startup_sha256"),
        _ARTIFACTS["startup"]["sha256"],
        "manifest startup SHA-256",
    )
    _require(
        manifest.get("ai_profiles_sha256"),
        _ARTIFACTS["ai_profiles"]["sha256"],
        "manifest ai_profiles SHA-256",
    )
    for field in (
        "fresh_root_opened",
        "gcloud_invoked",
        "instances_created",
        "teacher_executed",
        "current_profile_mutated",
        "runtime_policy_activated",
    ):
        _require(manifest.get(field), False, f"manifest {field}")

    _require(
        authorization.get("schema"),
        "hu_m43_attempt07_development_spot_authorization_v1",
        "authorization schema",
    )
    _require(
        authorization.get("development_run_name"),
        _RUN_NAME,
        "authorization run name",
    )
    _require(
        authorization.get("status"),
        "authorized_after_attempt07_preflight",
        "authorization status",
    )
    _require(
        authorization.get("development_manifest_sha256"),
        manifest_sha,
        "authorization manifest SHA-256",
    )
    _require(
        authorization.get("development_schedule_sha256"),
        schedule_sha,
        "authorization schedule SHA-256",
    )
    _require(
        authorization.get("attempt07_plan_sha256"),
        plan_sha,
        "authorization plan SHA-256",
    )
    _require(
        authorization.get("development_total_roots"),
        100,
        "authorization roots",
    )
    _require(
        authorization.get("development_total_shards"),
        100,
        "authorization shards",
    )
    _require(
        authorization.get("all_gates_passed"), True, "authorization gates"
    )
    _require(
        authorization.get("spot_authorized"), True, "authorization Spot flag"
    )
    _require(
        authorization.get("development_started"),
        False,
        "authorization pre-launch state",
    )
    _require(
        authorization.get("current_profile_mutated"),
        False,
        "authorization current mutation",
    )
    _require(
        authorization.get("runtime_policy_activated"),
        False,
        "authorization runtime activation",
    )

    _require(
        claim.get("schema"),
        "hu_m43_attempt07_development_output_consumption_v1",
        "claim schema",
    )
    _require(claim.get("run_name"), _RUN_NAME, "claim run name")
    _require(
        claim.get("status"),
        "claimed_after_all_done_before_any_teacher_read",
        "claim status",
    )
    _require(claim.get("manifest_sha256"), manifest_sha, "claim manifest SHA-256")
    _require(claim.get("authorization_sha256"), authorization_sha, "claim authorization SHA-256")
    _require(claim.get("schedule_sha256"), schedule_sha, "claim schedule SHA-256")
    _require(claim.get("expected_roots"), 100, "claim roots")
    _require(claim.get("expected_shards"), 100, "claim shards")
    _require(claim.get("all_done_markers_verified"), True, "claim DONE verification")
    _require(
        claim.get("result_objects_addressed_when_claimed"),
        False,
        "claim read-before-claim boundary",
    )
    done = _mapping(claim.get("done_sha256"), "claim DONE identities")
    _require(set(done), {f"{index:03d}" for index in range(100)}, "claim DONE shard set")
    if not all(
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
        for value in done.values()
    ):
        raise ValueError("claim DONE identities contain an invalid SHA-256")
    for field in (
        "selector_executed",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    ):
        _require(claim.get(field), False, f"claim {field}")

    _require(
        receive.get("schema"),
        "hu_m43_attempt07_development_receive_receipt_v1",
        "receive schema",
    )
    _require(receive.get("run_name"), _RUN_NAME, "receive run name")
    _require(
        receive.get("status"),
        "verified_exact_100_without_selection",
        "receive status",
    )
    _require(receive.get("manifest_sha256"), manifest_sha, "receive manifest SHA-256")
    _require(
        receive.get("authorization_sha256"),
        authorization_sha,
        "receive authorization SHA-256",
    )
    _require(receive.get("consumption_claim_sha256"), claim_sha, "receive claim SHA-256")
    _require(receive.get("merged_sha256"), merged_sha, "receive merged SHA-256")
    _require(receive.get("roots"), 100, "receive roots")
    for field in (
        "selector_executed",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    ):
        _require(receive.get(field), False, f"receive {field}")

    _require(
        merge.get("schema"),
        "hu_m43_attempt07_development_receive_merge_v1",
        "merge schema",
    )
    _require(merge.get("status"), "complete_no_selection", "merge status")
    _require(merge.get("manifest_sha256"), manifest_sha, "merge manifest SHA-256")
    _require(merge.get("consumption_claim_sha256"), claim_sha, "merge claim SHA-256")
    _require(merge.get("schedule_sha256"), schedule_sha, "merge schedule SHA-256")
    _require(merge.get("merged_sha256"), merged_sha, "merge merged SHA-256")
    _require(merge.get("roots"), 100, "merge roots")
    _require(merge.get("root_indices"), "0..99", "merge root indices")
    _require(merge.get("profiles"), {profile: 20 for profile in _PROFILES}, "merge profiles")
    audits = merge.get("received_audit_sha256")
    if not isinstance(audits, list) or len(audits) != 100 or len(set(audits)) != 100:
        raise ValueError("merge must bind 100 unique received-audit identities")
    if not all(
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
        for value in audits
    ):
        raise ValueError("merge contains an invalid received-audit SHA-256")
    for field in (
        "selector_executed",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    ):
        _require(merge.get(field), False, f"merge {field}")

    schedule_rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        paths["shard_schedule"].read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            schedule_rows.append(_mapping(json.loads(line), f"schedule line {line_number}"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid schedule JSON at line {line_number}") from exc
    _require(len(schedule_rows), 100, "schedule row count")
    _require(
        [row.get("root_index") for row in schedule_rows],
        list(range(100)),
        "schedule root indices",
    )
    _require(
        [row.get("shard") for row in schedule_rows],
        list(range(100)),
        "schedule shard indices",
    )
    _require(
        [row.get("root_profile") for row in schedule_rows],
        [_PROFILES[index % len(_PROFILES)] for index in range(100)],
        "schedule profile assignment",
    )
    if any(row.get("roots") != 1 for row in schedule_rows):
        raise ValueError("schedule must contain exactly one root per shard")


def validate_attempt07_no_go_closeout(
    repo_root: str | Path,
    *,
    closeout_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the exact Attempt07 No-Go evidence chain without writing files."""

    root = Path(repo_root).resolve()
    requested = Path(closeout_path) if closeout_path is not None else Path(_CLOSEOUT_PATH)
    if not requested.is_absolute():
        requested = root / requested
    requested = requested.resolve()
    canonical_closeout = (root / _CLOSEOUT_PATH).resolve()
    _require(requested, canonical_closeout, "closeout path")
    if not requested.is_file() or requested.is_symlink():
        raise ValueError(f"closeout must be a regular non-symlink file: {requested}")
    closeout = _load_json(requested, "Attempt07 closeout")

    _require(closeout.get("schema"), ATTEMPT07_CLOSEOUT_SCHEMA, "closeout schema")
    _require(closeout.get("milestone"), "M4.3-attempt07", "closeout milestone")
    _require(closeout.get("run_name"), _RUN_NAME, "closeout run name")
    _require(closeout.get("status"), "complete_no_go_development", "closeout status")
    _require(
        closeout.get("decision"),
        "close_attempt07_without_audit50_fit_threshold_or_runtime",
        "closeout decision",
    )
    _require(closeout.get("immutable_evidence"), _ARTIFACTS, "closeout immutable evidence")
    _require(
        closeout.get("development_population"),
        {
            "roots": 100,
            "profiles": list(_PROFILES),
            "states_per_profile": 20,
        },
        "closeout development population",
    )
    _require(
        closeout.get("ineligible_arms"), list(_ARMS), "closeout ineligible arms"
    )
    _require(
        closeout.get("passed_integrity_gates"),
        _SELECTION_INTEGRITY,
        "closeout integrity gates",
    )
    _require(
        closeout.get("selection_science_boundary"),
        _SELECTION_SCIENCE_BOUNDARY,
        "closeout selection science boundary",
    )
    _require(closeout.get("forbidden_actions"), _FORBIDDEN_ACTIONS, "closeout forbidden actions")
    _require(
        closeout.get("baseline_hash_audit"),
        {
            "policy_registry_path": _ARTIFACTS["ai_profiles"]["path"],
            "policy_registry_sha256": _ARTIFACTS["ai_profiles"]["sha256"],
            "current_mapping_changed": False,
        },
        "closeout baseline hash audit",
    )
    _require(
        closeout.get("next"),
        {
            "attempt07_closed": True,
            "attempt07_data_use": "development_postmortem_only",
            "audit50_must_not_run": True,
            "new_attempt_and_new_disjoint_evidence_required": True,
        },
        "closeout next boundary",
    )

    paths = {
        label: _artifact_path(root, identity, label)
        for label, identity in _ARTIFACTS.items()
    }
    _validate_selection(paths["selection"])
    _validate_run_chain(paths)

    return {
        "schema": ATTEMPT07_CLOSEOUT_VALIDATION_SCHEMA,
        "status": "validated_complete_no_go_development",
        "run_name": _RUN_NAME,
        "decision": "no_go",
        "selected_arm": None,
        "winner": None,
        "validated_artifacts": len(paths),
        "selection_sha256": _ARTIFACTS["selection"]["sha256"],
        "merged_input_sha256": _ARTIFACTS["merged_input"]["sha256"],
        "audit50_authorized": False,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
        "current_profile_mutation_allowed": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the exact immutable Attempt07 No-Go closeout"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--closeout", type=Path, default=Path(_CLOSEOUT_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = validate_attempt07_no_go_closeout(
        args.repo_root, closeout_path=args.closeout
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
