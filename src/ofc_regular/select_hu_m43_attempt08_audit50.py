"""One-shot semantic selector for the disjoint Attempt08 audit50 holdout.

The public CLI is deliberately disabled.  The immutable receive lifecycle in
``hu_m43_attempt08_audit50_spot`` owns the sole selector claim and calls the
private lifecycle entry point exactly once.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # package import in tests/repository
    from .hu_m43_attempt08_audit50_contract import (
        AI_PROFILES_SHA256,
        AUDIT50_PLAN_SHA256,
        EXPECTED_GATES,
        LAMBDA_MODEL_SHA256,
        PROFILES,
        ROOT_FIRST,
        ROOT_LAST,
        SEED_DOMAINS,
        SOURCE_PLAN_SHA256,
        TOTAL_SHARDS,
        canonical_json_bytes,
        load_and_validate_audit50_plan,
        require_sha256,
        sha256_file,
        write_canonical_json,
    )
except ImportError:  # direct execution from the frozen audit overlay
    from hu_m43_attempt08_audit50_contract import (  # type: ignore
        AI_PROFILES_SHA256,
        AUDIT50_PLAN_SHA256,
        EXPECTED_GATES,
        LAMBDA_MODEL_SHA256,
        PROFILES,
        ROOT_FIRST,
        ROOT_LAST,
        SEED_DOMAINS,
        SOURCE_PLAN_SHA256,
        TOTAL_SHARDS,
        canonical_json_bytes,
        load_and_validate_audit50_plan,
        require_sha256,
        sha256_file,
        write_canonical_json,
    )


AUDIT50_DECISION_SCHEMA = "hu_m43_attempt08_audit50_go_no_go_v1"
AUDIT50_ROW_SCHEMA = "hu_m43_attempt08_future_audit_shard_row_v1"
AUDIT50_PROVENANCE_SCHEMA = "hu_m43_attempt08_future_audit_provenance_v1"
AUDIT50_CORE_AUTH_SCHEMA = "hu_m43_attempt08_future_audit_open_authorization_v1"
_SEARCH_CORE = (
    "same_attempt08_t1_second_root_generation_lambda_candidate_and_teacher_core"
)
_ROW_KEYS = frozenset(
    {
        "schema",
        "population",
        "root_index",
        "audit_local_index",
        "hand_seed",
        "root_profile",
        "policy_observation",
        "baseline_action_key",
        "provenance",
        "teacher",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "schema",
        "population",
        "run_id",
        "root_index",
        "root_profile",
        "seeds",
        "audit_open_authorization_sha256",
        "development_pass_freeze_sha256",
        "package_manifest_sha256",
        "source_closure_sha256",
        "source_zip_sha256",
        "runtime_semantic_anchor_sha256",
        "plan_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "search_core",
        "baseline_profile",
        "continuation_profile",
        "current_profile_resolved",
        "opponent_private_discard_input_allowed",
        "teacher_values_are_realized_match_ev",
        "future_audit_only",
        "fit_allowed",
        "threshold_selection_allowed",
        "runtime_activation_allowed",
    }
)
_LIFECYCLE_TOKEN = object()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _reject_hidden_fields(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "opponent_private_discards":
                raise ValueError(f"Attempt08 hidden-information violation at {path}.{key}")
            _reject_hidden_fields(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_hidden_fields(child, path=f"{path}[{index}]")


def _flatten_digests(value: Any, *, label: str) -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in sorted(value):
            result.extend(_flatten_digests(value[key], label=f"{label}.{key}"))
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        result = []
        for index, child in enumerate(value):
            result.extend(_flatten_digests(child, label=f"{label}[{index}]"))
        return result
    return [require_sha256(value, label)]


def _assessment_metrics(raw_value: Any, *, root_index: int, fired: bool) -> dict[str, Any]:
    raw = _sequence(raw_value, f"root[{root_index}].assessment A256")
    if not fired:
        if raw:
            raise ValueError(f"Attempt08 nonfire root {root_index} opened A256")
        return {"mean": 0.0, "loss": {"p95": 0.0, "p99": 0.0, "max": 0.0}}
    if len(raw) != 256:
        raise ValueError(f"Attempt08 root {root_index} A256 count changed")
    values = np.asarray(
        [_finite(value, f"root[{root_index}].assessment") for value in raw],
        dtype=np.float64,
    )
    return {
        "mean": float(np.mean(values)),
        "loss": {
            "p95": max(0.0, -float(np.quantile(values, 0.05, method="linear"))),
            "p99": max(0.0, -float(np.quantile(values, 0.01, method="linear"))),
            "max": max(0.0, -float(np.min(values))),
        },
    }


def _validate_row(
    row: Mapping[str, Any],
    *,
    root_index: int,
    expected_seeds: Mapping[str, int],
    audit_run_name: str,
    core_authorization: Mapping[str, Any],
    core_authorization_sha256: str,
) -> dict[str, Any]:
    # These imports must resolve from the frozen development package when this
    # file is executed from the standalone audit overlay on a worker.
    from ofc_regular.action_key import ActionKey, resolve_action_key
    from ofc_regular.action_space import generate_turn_actions
    from ofc_regular.hu_infoset import ActorObservation
    from ofc_regular.hu_m43_attempt08_teacher import (
        Attempt08TeacherConfig,
        validate_attempt08_teacher_output,
    )

    local_index = root_index - ROOT_FIRST
    if set(row) != _ROW_KEYS:
        raise ValueError(f"Attempt08 audit row fields changed at root {root_index}")
    profile = PROFILES[root_index % len(PROFILES)]
    if (
        row.get("schema") != AUDIT50_ROW_SCHEMA
        or row.get("population") != "future_audit"
        or _strict_int(row.get("root_index"), "root_index") != root_index
        or _strict_int(row.get("audit_local_index"), "audit_local_index")
        != local_index
        or row.get("root_profile") != profile
        or _strict_int(row.get("hand_seed"), "hand_seed") != expected_seeds["hand"]
    ):
        raise ValueError(f"Attempt08 audit row identity changed at root {root_index}")
    _reject_hidden_fields(row, path=f"row[{root_index}]")
    observation_payload = _mapping(row.get("policy_observation"), "observation")
    observation = ActorObservation.from_dict(observation_payload)
    if (
        observation.to_dict() != observation_payload
        or observation.street != "T1"
        or observation.seat != "second"
        or observation.to_act_order != "second"
    ):
        raise ValueError(f"Attempt08 audit observation changed at root {root_index}")
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    baseline_token = str(row.get("baseline_action_key"))
    resolve_action_key(actions, ActionKey.from_token(baseline_token))

    provenance = _mapping(row.get("provenance"), "provenance")
    expected_run_id = f"{audit_run_name}:shard={local_index}"
    if set(provenance) != _PROVENANCE_KEYS:
        raise ValueError(f"Attempt08 audit provenance fields changed at root {root_index}")
    expected_provenance = {
        "schema": AUDIT50_PROVENANCE_SCHEMA,
        "population": "future_audit",
        "run_id": expected_run_id,
        "root_index": root_index,
        "root_profile": profile,
        "seeds": dict(expected_seeds),
        "audit_open_authorization_sha256": core_authorization_sha256,
        "development_pass_freeze_sha256": core_authorization[
            "development_pass_freeze_sha256"
        ],
        "package_manifest_sha256": core_authorization["package_manifest_sha256"],
        "source_closure_sha256": core_authorization["source_closure_sha256"],
        "source_zip_sha256": core_authorization["source_zip_sha256"],
        "runtime_semantic_anchor_sha256": core_authorization[
            "runtime_semantic_anchor_sha256"
        ],
        "plan_sha256": SOURCE_PLAN_SHA256,
        "model_sha256": LAMBDA_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "search_core": _SEARCH_CORE,
        "baseline_profile": "stage18_p1",
        "continuation_profile": "stage9f_p2",
        "current_profile_resolved": False,
        "opponent_private_discard_input_allowed": False,
        "teacher_values_are_realized_match_ev": False,
        "future_audit_only": True,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
    }
    if provenance != expected_provenance:
        raise ValueError(f"Attempt08 audit provenance changed at root {root_index}")

    config = Attempt08TeacherConfig(
        frozen_model_sha256=LAMBDA_MODEL_SHA256,
        hand_seed=expected_seeds["hand"],
        rerank_seed=expected_seeds["rerank"],
        veto_seed=expected_seeds["veto"],
        stress_seed=expected_seeds["stress"],
        assessment_seed=expected_seeds["assessment"],
        child_policy_seed=expected_seeds["child"],
        run_id=(
            f"{expected_run_id}:root={root_index}:seed={expected_seeds['hand']}:"
            f"obs={observation.fingerprint()}"
        ),
        batch_child_selectors=True,
    )
    normalized = _mapping(
        validate_attempt08_teacher_output(
            observation,
            baseline_action_key=baseline_token,
            payload=_mapping(row.get("teacher"), "teacher"),
            config=config,
        ),
        "normalized teacher",
    )
    expected_normalized = {
        "selected_action_key",
        "override_fired",
        "exact_baseline_fallback",
        "assessment_raw_paired_deltas_vs_baseline",
        "rng_key_digests",
        "belief_digests",
    }
    if set(normalized) != expected_normalized:
        raise ValueError(f"Attempt08 audit normalized teacher changed at root {root_index}")
    selected_token = str(normalized["selected_action_key"])
    resolve_action_key(actions, ActionKey.from_token(selected_token))
    fired = normalized["override_fired"]
    fallback = normalized["exact_baseline_fallback"]
    if type(fired) is not bool or type(fallback) is not bool:
        raise ValueError("Attempt08 audit decision flags changed")
    if fired and (selected_token == baseline_token or fallback):
        raise ValueError(f"Attempt08 audit fired baseline at root {root_index}")
    if not fired and (selected_token != baseline_token or not fallback):
        raise ValueError(f"Attempt08 audit nonfire fallback changed at root {root_index}")
    assessment = _assessment_metrics(
        normalized["assessment_raw_paired_deltas_vs_baseline"],
        root_index=root_index,
        fired=fired,
    )
    rng = _flatten_digests(normalized["rng_key_digests"], label="rng")
    beliefs = _flatten_digests(normalized["belief_digests"], label="belief")
    if not rng or len(rng) != len(set(rng)) or not beliefs or len(beliefs) != len(set(beliefs)):
        raise ValueError(f"Attempt08 audit digest domains overlap at root {root_index}")
    return {
        "root_index": root_index,
        "profile": profile,
        "hand_seed": expected_seeds["hand"],
        "observation_fingerprint": observation.fingerprint(),
        "baseline_action_key": baseline_token,
        "selected_action_key": selected_token,
        "fired": fired,
        "assessment_mean": assessment["mean"],
        "assessment_loss": assessment["loss"],
        "rng_digests": rng,
        "belief_digests": beliefs,
    }


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fired = [row for row in rows if row["fired"]]
    fired_means = [float(row["assessment_mean"]) for row in fired]
    states = len(rows)
    return {
        "states": states,
        "fires": len(fired),
        "mean_delta_per_state": (
            sum(float(row["assessment_mean"]) for row in rows) / states
            if states
            else None
        ),
        "mean_delta_per_fire": (
            sum(fired_means) / len(fired_means) if fired_means else None
        ),
        "false_positive_fires": sum(value <= 0.0 for value in fired_means),
        "false_positive_rate_per_fire": (
            sum(value <= 0.0 for value in fired_means) / len(fired_means)
            if fired_means
            else None
        ),
        "maximum_per_fired_root_loss": {
            name: max(float(row["assessment_loss"][name]) for row in fired)
            if fired
            else None
            for name in ("p95", "p99", "max")
        },
    }


def _gate(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def evaluate_audit50_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    audit_plan_path: str | Path,
    source_plan_path: str | Path,
    source_input_sha256: str,
    audit_run_name: str,
    core_authorization: Mapping[str, Any],
    core_authorization_sha256: str,
    outer_authorization_sha256: str,
    merge_receipt_sha256: str,
) -> dict[str, Any]:
    from ofc_regular.hu_m43_attempt08_contract import (
        enumerate_attempt08_seed_schedules,
        load_and_validate_attempt08_plan,
    )

    audit_plan = load_and_validate_audit50_plan(audit_plan_path)
    if sha256_file(source_plan_path) != SOURCE_PLAN_SHA256:
        raise ValueError("Attempt08 audit source plan changed")
    source_plan = load_and_validate_attempt08_plan(source_plan_path)
    for digest, label in (
        (source_input_sha256, "input"),
        (core_authorization_sha256, "core authorization"),
        (outer_authorization_sha256, "outer authorization"),
        (merge_receipt_sha256, "merge receipt"),
    ):
        require_sha256(digest, label)
    if len(rows) != TOTAL_SHARDS:
        raise ValueError("Attempt08 audit selector requires exactly 50 rows")
    by_root: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        root = _strict_int(_mapping(row, "row").get("root_index"), "root_index")
        if root in by_root:
            raise ValueError(f"Attempt08 audit duplicate root: {root}")
        by_root[root] = row
    if set(by_root) != set(range(ROOT_FIRST, ROOT_LAST + 1)):
        raise ValueError("Attempt08 audit roots must be exactly 200..249")
    schedules = enumerate_attempt08_seed_schedules(source_plan, population="future_audit")
    if set(schedules) != set(SEED_DOMAINS):
        raise ValueError("Attempt08 audit seed domains changed")
    validated: list[dict[str, Any]] = []
    for root in range(ROOT_FIRST, ROOT_LAST + 1):
        local = root - ROOT_FIRST
        seeds = {domain: int(schedules[domain][local]) for domain in SEED_DOMAINS}
        if len(set(seeds.values())) != len(seeds):
            raise ValueError(f"Attempt08 audit seed domains overlap at root {root}")
        validated.append(
            _validate_row(
                by_root[root],
                root_index=root,
                expected_seeds=seeds,
                audit_run_name=audit_run_name,
                core_authorization=core_authorization,
                core_authorization_sha256=core_authorization_sha256,
            )
        )
    all_rng = [digest for row in validated for digest in row["rng_digests"]]
    all_beliefs = [digest for row in validated for digest in row["belief_digests"]]
    if len(all_rng) != len(set(all_rng)) or len(all_beliefs) != len(set(all_beliefs)):
        raise ValueError("Attempt08 audit cross-root RNG/belief domains overlap")
    profile_counts = Counter(row["profile"] for row in validated)
    if profile_counts != Counter({profile: 10 for profile in PROFILES}):
        raise ValueError("Attempt08 audit profile balance changed")
    overall = _metrics(validated)
    by_profile = {
        profile: _metrics([row for row in validated if row["profile"] == profile])
        for profile in PROFILES
    }
    gates_plan = _mapping(audit_plan["audit_go_no_go"], "audit gates")
    tails = _mapping(overall["maximum_per_fired_root_loss"], "tails")
    gates = [
        _gate("fires_total", overall["fires"] >= 10, overall["fires"], ">= 10"),
        _gate(
            "fires_each_profile",
            all(by_profile[p]["fires"] >= 1 for p in PROFILES),
            {p: by_profile[p]["fires"] for p in PROFILES},
            "each >= 1",
        ),
        _gate(
            "mean_delta_per_state",
            float(overall["mean_delta_per_state"]) > 0.0,
            overall["mean_delta_per_state"],
            "> 0.0",
        ),
        _gate(
            "mean_delta_per_fire",
            overall["mean_delta_per_fire"] is not None
            and float(overall["mean_delta_per_fire"]) > 0.0,
            overall["mean_delta_per_fire"],
            "> 0.0",
        ),
        _gate(
            "false_positive_rate_per_fire",
            overall["false_positive_rate_per_fire"] is not None
            and float(overall["false_positive_rate_per_fire"]) <= 0.4,
            overall["false_positive_rate_per_fire"],
            "<= 0.4",
        ),
        _gate("maximum_per_fired_root_p95_loss", tails["p95"] is not None and float(tails["p95"]) <= 25.0, tails["p95"], "<= 25.0"),
        _gate("maximum_per_fired_root_p99_loss", tails["p99"] is not None and float(tails["p99"]) <= 40.0, tails["p99"], "<= 40.0"),
        _gate("maximum_per_fired_root_max_loss", tails["max"] is not None and float(tails["max"]) <= 50.0, tails["max"], "<= 50.0"),
        _gate("action_mapping_violation_count", True, 0, "= 0"),
        _gate("rng_domain_violation_count", True, 0, "= 0"),
        _gate("hidden_information_violation_count", True, 0, "= 0"),
        _gate("risk_reserve_contract_violation_count", True, 0, "= 0"),
        _gate("locked_action_change_violation_count", True, 0, "= 0"),
        _gate(
            "nonfire_exact_baseline_action_fallback",
            all(row["fired"] or row["selected_action_key"] == row["baseline_action_key"] for row in validated),
            sum(not row["fired"] for row in validated),
            "required for every nonfire",
        ),
    ]
    # Make the literal gate values above subordinate to the hash-frozen plan.
    for key, expected in EXPECTED_GATES.items():
        if gates_plan.get(key) != expected:
            raise AssertionError(f"audit plan gate drift after validation: {key}")
    passed = all(gate["passed"] for gate in gates)
    return {
        "schema": AUDIT50_DECISION_SCHEMA,
        "status": (
            "go_write_separate_distillation_freeze_only"
            if passed
            else "no_go_close_attempt08_without_fit_threshold_or_runtime"
        ),
        "decision": "go" if passed else "no_go",
        "source": {
            "audit_run_name": audit_run_name,
            "input_jsonl_sha256": source_input_sha256,
            "audit_plan_sha256": AUDIT50_PLAN_SHA256,
            "source_plan_sha256": SOURCE_PLAN_SHA256,
            "core_authorization_sha256": core_authorization_sha256,
            "outer_authorization_sha256": outer_authorization_sha256,
            "merge_receipt_sha256": merge_receipt_sha256,
            "roots": TOTAL_SHARDS,
            "root_index_first": ROOT_FIRST,
            "root_index_last": ROOT_LAST,
        },
        "overall": overall,
        "by_profile": by_profile,
        "gates": gates,
        "decision_contract": {
            "gate_evaluation_count": 1,
            "all_gates_required": True,
            "threshold_search_on_audit_performed": False,
            "fit_on_audit_performed": False,
            "profile_exclusion_performed": False,
            "distillation_source": "development200_only",
            "audit_rows_used_for_fit": False,
            "realized_population_acceptance_still_required": True,
        },
        "science_boundary": {
            "teacher_values_reported_as_realized_match_ev": False,
            "nonfire_exact_baseline_action_fallback_verified": True,
            "nonfire_complete_trajectory_cancellation_verified": False,
            "runtime_policy_activated": False,
            "current_profile_mutated": False,
            "full_replacement_enabled": False,
        },
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def _read_canonical_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    raw = Path(path).read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise ValueError("Attempt08 audit merged input must use canonical LF JSONL")
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(raw.splitlines()):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Attempt08 audit row {index} is invalid JSON") from exc
        row = _mapping(value, f"row[{index}]")
        if line + b"\n" != canonical_json_bytes(row):
            raise ValueError(f"Attempt08 audit row {index} is not canonical")
        rows.append(row)
    return rows


def _write_from_lifecycle(
    *,
    token: object,
    input_jsonl: str | Path,
    output: str | Path,
    audit_plan_path: str | Path,
    source_plan_path: str | Path,
    audit_run_name: str,
    core_authorization: Mapping[str, Any],
    core_authorization_sha256: str,
    outer_authorization_sha256: str,
    merge_receipt_sha256: str,
) -> dict[str, Any]:
    if token is not _LIFECYCLE_TOKEN:
        raise PermissionError("Attempt08 audit selector requires the claimed lifecycle")
    decision = evaluate_audit50_rows(
        _read_canonical_jsonl(input_jsonl),
        audit_plan_path=audit_plan_path,
        source_plan_path=source_plan_path,
        source_input_sha256=sha256_file(input_jsonl),
        audit_run_name=audit_run_name,
        core_authorization=core_authorization,
        core_authorization_sha256=core_authorization_sha256,
        outer_authorization_sha256=outer_authorization_sha256,
        merge_receipt_sha256=merge_receipt_sha256,
    )
    write_canonical_json(output, decision)
    return decision


def main() -> int:
    raise SystemExit(
        "direct audit50 selector execution is disabled; use the claimed receive lifecycle"
    )


if __name__ == "__main__":
    main()


__all__ = ["AUDIT50_DECISION_SCHEMA", "evaluate_audit50_rows"]
