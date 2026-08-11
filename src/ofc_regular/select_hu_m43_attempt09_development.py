"""One-shot Attempt09 development200 Go/No-Go selector.

Only the locked final action's disjoint E256 vector is used for quality gates.
A Go permits a separate search freeze and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .action_key import ActionKey, resolve_action_key
from .action_space import generate_turn_actions
from .generate_hu_m4_t1_data import _profile_policy_seed
from .hu_infoset import ActorObservation
from .hu_m43_attempt09_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT09_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT09_PLAN_SHA256,
    M43_ATTEMPT09_PROFILES,
    enumerate_attempt09_seed_schedules,
    load_and_validate_attempt09_plan,
    validate_attempt09_plan,
)
from .hu_m43_attempt09_teacher import (
    Attempt09TeacherConfig,
    validate_attempt09_teacher_output,
)
from .run_hu_m43_attempt09 import (
    ATTEMPT09_BASELINE_PROFILE,
    ATTEMPT09_CONTINUATION_PROFILE,
    ATTEMPT09_ROW_SCHEMA,
    _authorization,
    _canonical_sha256,
)


ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA = "hu_m43_attempt09_development_go_no_go_v1"
ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA = "hu_m43_attempt09_development_selector_receipt_v1"
ROOT_CONTRACT_SCHEMA = "hu_m43_attempt09_root_contract_v1"
TEACHER_VALIDATION_SCHEMA = "hu_m43_attempt09_teacher_validation_v1"
NO_GO_STATUS = "no_go_close_attempt09_development"
EVALUATION_SAMPLE_COUNT = 256
SELECTOR_SOURCE_PATH = Path(__file__)
_ROOTS = 200
POPULATION = "development"
ROOT_INDEX_FIRST = 0
ROOTS_PER_PROFILE = 40
GATE_SECTION = "development_go_no_go"
POPULATION_REPORT_KEY = "development_population"
GO_STATUS = "go_write_separate_search_freeze_only"
_DOMAINS = ("hand", "rerank", "veto", "stress", "confirmation", "evaluation", "child")
_ROW_KEYS = {
    "schema", "root_index", "hand_seed", "root_profile", "policy_observation",
    "baseline_action_key", "provenance", "teacher",
}
_SAFE_RUN = re.compile(r"^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$")
_HEX = frozenset("0123456789abcdef")
_GATES = {
    "fires_total_min": 40,
    "fires_each_profile_min": 3,
    "mean_delta_per_state_strictly_greater_than": 0.0,
    "mean_delta_per_fire_strictly_greater_than": 0.0,
    "false_positive_rate_per_fire_max": 0.4,
    "override_loss_p95_max": 25.0,
    "override_loss_p99_max": 40.0,
    "override_loss_max": 50.0,
    "action_mapping_violation_count_max": 0,
    "rng_domain_violation_count_max": 0,
    "hidden_information_violation_count_max": 0,
    "risk_reserve_contract_violation_count_max": 0,
    "retained_order_violation_count_max": 0,
    "phase_filter_violation_count_max": 0,
    "locked_action_change_violation_count_max": 0,
    "nonfire_exact_baseline_action_fallback_required": True,
    "nonfire_complete_trajectory_acceptance_deferred": True,
    "teacher_values_reported_as_realized_match_ev": False,
}


def _expected_search_config(
    config: Attempt09TeacherConfig, seeds: Mapping[str, int]
) -> dict[str, Any]:
    return {
        "learned_nonbaseline_top_k": 8,
        "baseline_added_exactly_once": True,
        "rerank_samples": 128,
        "rerank_top_k": 3,
        "risk_reserve_count": 1,
        "k4_size": 4,
        "veto_samples": 256,
        "stress_samples": 512,
        "confirmation_samples": 256,
        "evaluation_samples": 256,
        "hand_seed": seeds["hand"],
        "rerank_seed": seeds["rerank"],
        "veto_seed": seeds["veto"],
        "stress_seed": seeds["stress"],
        "confirmation_seed": seeds["confirmation"],
        "evaluation_seed": seeds["evaluation"],
        "child_policy_seed": seeds["child"],
        "run_id": config.run_id,
        "batch_child_selectors": True,
    }


def _expected_seed_domain_provenance(seeds: Mapping[str, int]) -> dict[str, Any]:
    return {
        "domain_order": [
            "hand_external",
            "rerank_r128",
            "veto_v256",
            "stress_x512",
            "confirmation_c256",
            "evaluation_e256",
            "child_policy",
        ],
        "hand_external": seeds["hand"],
        "rerank_r128": seeds["rerank"],
        "veto_v256": seeds["veto"],
        "stress_x512": seeds["stress"],
        "confirmation_c256": seeds["confirmation"],
        "evaluation_e256": seeds["evaluation"],
        "child_policy": seeds["child"],
        "all_seven_base_seeds_pairwise_distinct": True,
        "hand_sampled_inside_teacher": False,
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _exact(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (isinstance(actual, Mapping) and set(actual) == set(expected)
                and all(_exact(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_exact(left, right) for left, right in zip(actual, expected, strict=True)))
    return type(actual) is type(expected) and actual == expected


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _read_rows(raw: bytes) -> list[Mapping[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Attempt09 input must be UTF-8") from exc
    if not text.endswith("\n") or "\r" in text:
        raise ValueError("Attempt09 input must be canonical LF JSONL")
    lines = text.splitlines()
    if len(lines) != _ROOTS or any(not line for line in lines):
        raise ValueError(f"Attempt09 {POPULATION} requires exactly {_ROOTS} rows")
    try:
        rows = [_mapping(json.loads(line), f"row[{index}]") for index, line in enumerate(lines)]
    except json.JSONDecodeError as exc:
        raise ValueError("Attempt09 input contains invalid JSON") from exc
    if raw != b"".join(_canonical(row) for row in rows):
        raise ValueError("Attempt09 input is not canonical JSONL")
    return rows


def _reject_hidden(value: Any, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            if name in {"opponent_private_discards", "opponent_private_discard", "opponent_hidden", "particles"}:
                raise ValueError(f"Attempt09 hidden information at {path}.{key}")
            _reject_hidden(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_hidden(child, f"{path}[{index}]")


def _teacher_config(seeds: Mapping[str, int], run_id: str, root: int, obs: ActorObservation) -> Attempt09TeacherConfig:
    return Attempt09TeacherConfig(
        frozen_model_sha256=ATTEMPT09_LAMBDA_MODEL_SHA256,
        hand_seed=seeds["hand"], rerank_seed=seeds["rerank"], veto_seed=seeds["veto"],
        stress_seed=seeds["stress"], confirmation_seed=seeds["confirmation"],
        evaluation_seed=seeds["evaluation"], child_policy_seed=seeds["child"],
        run_id=f"{run_id}:root={root}:seed={seeds['hand']}:obs={obs.fingerprint()}",
        batch_child_selectors=True,
    )


def _validate_row(row: Mapping[str, Any], root: int, seeds: Mapping[str, int], *, run_name: str,
                  authorization_sha256: str, source_package_sha256: str) -> dict[str, Any]:
    if set(row) != _ROW_KEYS or row.get("schema") != ATTEMPT09_ROW_SCHEMA:
        raise ValueError(f"Attempt09 row schema changed at root {root}")
    if type(row.get("root_index")) is not int or row["root_index"] != root:
        raise ValueError(f"Attempt09 root order changed at root {root}")
    profile = M43_ATTEMPT09_PROFILES[root % len(M43_ATTEMPT09_PROFILES)]
    if row.get("root_profile") != profile or row.get("hand_seed") != seeds["hand"]:
        raise ValueError(f"Attempt09 root/profile/hand identity changed at root {root}")
    _reject_hidden(row, f"row[{root}]")
    obs_payload = _mapping(row.get("policy_observation"), "policy_observation")
    obs = ActorObservation.from_dict(obs_payload)
    if obs.to_dict() != obs_payload or (obs.street, obs.seat, obs.to_act_order) != ("T1", "second", "second"):
        raise ValueError(f"Attempt09 observation changed at root {root}")
    actions = generate_turn_actions(obs.hero_board, obs.dealt_cards)
    baseline = str(row.get("baseline_action_key"))
    baseline_key = ActionKey.from_token(baseline)
    if baseline_key.to_token() != baseline:
        raise ValueError(f"Attempt09 baseline ActionKey is not canonical at root {root}")
    resolve_action_key(actions, baseline_key)

    run_id = f"{run_name}:{POPULATION}:root={root}:search"
    fixed = {
        "schema": ROOT_CONTRACT_SCHEMA, "mode": POPULATION,
        "root_index": root, "root_profile": profile, "run_id": run_id, "seeds": dict(seeds),
        "plan_sha256": M43_ATTEMPT09_PLAN_SHA256, "ai_profiles_sha256": AI_PROFILES_SHA256,
        "model_sha256": ATTEMPT09_LAMBDA_MODEL_SHA256,
        "authorization_sha256": authorization_sha256, "source_package_sha256": source_package_sha256,
        "batch_child_selectors": True, "native_batch_threads": 4,
    }
    expected_provenance = {
        **fixed, "config_sha256": _canonical_sha256(fixed),
        "root_policy_profiles": {"first": profile, "second": profile},
        "root_policy_seeds": {seat: _profile_policy_seed(seeds["hand"], profile, seat) for seat in ("first", "second")},
        "baseline_profile": ATTEMPT09_BASELINE_PROFILE,
        "baseline_policy_seed": _profile_policy_seed(seeds["hand"], ATTEMPT09_BASELINE_PROFILE, "second"),
        "continuation_profile": ATTEMPT09_CONTINUATION_PROFILE,
        "continuation_policy_seeds": {"first": seeds["child"], "second": seeds["child"] + 1},
        "explicit_loaded_profiles": sorted({profile, ATTEMPT09_BASELINE_PROFILE, ATTEMPT09_CONTINUATION_PROFILE}),
        "opponent_private_discard_input_allowed": False, "opponent_profile_runtime_feature_allowed": False,
        "teacher_values_are_realized_match_ev": False, "teacher_ev_or_lcb_runtime_gate_allowed": False,
        "development_only": True, "fit_allowed": False, "threshold_selection_allowed": False,
        "runtime_activation_allowed": False, "current_profile_resolved": False,
    }
    provenance = _mapping(row.get("provenance"), "provenance")
    if set(provenance) != {*expected_provenance, "elapsed_seconds", "peak_rss_bytes"}:
        raise ValueError(f"Attempt09 provenance fields changed at root {root}")
    if any(provenance.get(key) != value or type(provenance.get(key)) is not type(value) for key, value in expected_provenance.items()):
        raise ValueError(f"Attempt09 provenance changed at root {root}")
    elapsed, rss = provenance["elapsed_seconds"], provenance["peak_rss_bytes"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or elapsed < 0:
        raise ValueError(f"Attempt09 elapsed metric changed at root {root}")
    if type(rss) is not int or rss < 0:
        raise ValueError(f"Attempt09 RSS metric changed at root {root}")

    config = _teacher_config(seeds, run_id, root, obs)
    teacher = _mapping(row.get("teacher"), "teacher")
    expected_search = _expected_search_config(config, seeds)
    if not _exact(teacher.get("search_config"), expected_search):
        raise ValueError(f"Attempt09 teacher search config changed at root {root}")
    expected_seed_provenance = _expected_seed_domain_provenance(seeds)
    if not _exact(teacher.get("seed_domain_provenance"), expected_seed_provenance):
        raise ValueError(f"Attempt09 teacher seed provenance changed at root {root}")
    normalized = _mapping(validate_attempt09_teacher_output(obs, baseline_action_key=baseline, payload=teacher, config=config), "teacher validation")
    if set(normalized) != {"schema", "selected_action_key", "override_fired", "exact_baseline_fallback", "opened_phases"} or normalized.get("schema") != TEACHER_VALIDATION_SCHEMA:
        raise ValueError(f"Attempt09 strict teacher validation schema changed at root {root}")
    selected = str(normalized["selected_action_key"])
    selected_key = ActionKey.from_token(selected)
    if selected_key.to_token() != selected:
        raise ValueError(f"Attempt09 selected ActionKey is not canonical at root {root}")
    resolve_action_key(actions, selected_key)
    fired, fallback = normalized["override_fired"], normalized["exact_baseline_fallback"]
    if type(fired) is not bool or type(fallback) is not bool or (fired and (selected == baseline or fallback)) or (not fired and (selected != baseline or not fallback)):
        raise ValueError(f"Attempt09 exact baseline fallback changed at root {root}")
    evaluation = _mapping(teacher.get("evaluation"), "evaluation")
    if not fired:
        if evaluation.get("opened") is not False or evaluation.get("actions") != []:
            raise ValueError(f"Attempt09 nonfire opened E256 at root {root}")
        values: list[float] = []
    else:
        rows = evaluation.get("actions")
        if evaluation.get("opened") is not True or not isinstance(rows, list) or len(rows) != 2 or rows[0].get("action_key") != selected:
            raise ValueError(f"Attempt09 fired E256 mapping changed at root {root}")
        raw = rows[0].get("raw_paired_deltas_vs_baseline")
        if not isinstance(raw, list) or len(raw) != EVALUATION_SAMPLE_COUNT:
            raise ValueError(f"Attempt09 E256 sample count changed at root {root}")
        values = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Attempt09 E256 contains non-finite data at root {root}")
            values.append(float(value))
    rng = _mapping(teacher.get("rng_key_digests"), "rng_key_digests")
    digests: list[str] = []
    for phase, phase_values in rng.items():
        if not isinstance(phase_values, list) or any(not _sha(value) for value in phase_values):
            raise ValueError(f"Attempt09 RNG mapping changed at root {root}:{phase}")
        digests.extend(phase_values)
    if not digests or len(digests) != len(set(digests)):
        raise ValueError(f"Attempt09 RNG domains overlap at root {root}")
    beliefs = _mapping(teacher.get("belief_digests"), "belief_digests")
    if set(beliefs) != set(rng) or any(not _sha(value) for value in beliefs.values()):
        raise ValueError(f"Attempt09 belief/RNG phase mapping changed at root {root}")
    belief_digests = list(beliefs.values())
    if len(belief_digests) != len(set(belief_digests)):
        raise ValueError(f"Attempt09 belief digests overlap at root {root}")
    if normalized["opened_phases"] != list(rng):
        raise ValueError(f"Attempt09 opened phase mapping changed at root {root}")
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array)) if fired else 0.0
    loss = ({"p95": max(0.0, -float(np.quantile(array, .05, method="linear"))),
             "p99": max(0.0, -float(np.quantile(array, .01, method="linear"))),
             "max": max(0.0, -float(np.min(array)))} if fired else {"p95": 0.0, "p99": 0.0, "max": 0.0})
    return {"root_index": root, "profile": profile, "hand_seed": seeds["hand"],
            "observation_fingerprint": obs.fingerprint(), "baseline_action_key": baseline,
            "selected_action_key": selected, "fired": fired, "mean": mean, "loss": loss,
            "rng_digests": digests, "belief_digests": belief_digests,
            "config_sha256": expected_provenance["config_sha256"]}


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fired = [row for row in rows if row["fired"]]
    means = [float(row["mean"]) for row in fired]
    false = sum(value <= 0 for value in means)
    return {"states": len(rows), "fires": len(fired),
            "mean_delta_per_state": sum(float(row["mean"]) for row in rows) / len(rows),
            "mean_delta_per_fire": sum(means) / len(means) if means else None,
            "false_positive_fires": false,
            "false_positive_rate_per_fire": false / len(means) if means else None,
            "maximum_per_fired_root_loss": {name: max((float(row["loss"][name]) for row in fired), default=None) for name in ("p95", "p99", "max")}}


def _gate(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "requirement": requirement}


def aggregate_attempt09_development_rows(rows: Sequence[Mapping[str, Any]], *, plan: Mapping[str, Any],
        source_input_sha256: str, source_plan_sha256: str, authorization_sha256: str,
        source_package_sha256: str, run_name: str) -> dict[str, Any]:
    """Validate and evaluate the frozen development population exactly once."""
    validate_attempt09_plan(plan)
    declared = _mapping(plan.get(GATE_SECTION), GATE_SECTION)
    if any(declared.get(key) != value or type(declared.get(key)) is not type(value) for key, value in _GATES.items()):
        raise ValueError("Attempt09 frozen development gates changed")
    if declared.get("assessment_source") != "disjoint_E256_locked_final_nonbaseline_output_vs_explicit_baseline" or declared.get("quantile_method") != "numpy_linear" or declared.get("all_gates_required") is not True:
        raise ValueError("Attempt09 E256 gate semantics changed")
    if source_plan_sha256 != M43_ATTEMPT09_PLAN_SHA256 or not all(_sha(value) for value in (source_input_sha256, authorization_sha256, source_package_sha256)):
        raise ValueError("Attempt09 source hashes changed")
    if not _SAFE_RUN.fullmatch(run_name) or len(rows) != _ROOTS:
        raise ValueError("Attempt09 run identity or root count changed")
    schedules = enumerate_attempt09_seed_schedules(plan, population=POPULATION)
    if tuple(schedules) != _DOMAINS:
        raise ValueError("Attempt09 seed domain order changed")
    validated = []
    for offset, row in enumerate(rows):
        root = ROOT_INDEX_FIRST + offset
        seeds = {name: int(schedules[name][offset]) for name in _DOMAINS}
        validated.append(_validate_row(_mapping(row, f"row[{root}]"), root, seeds, run_name=run_name,
                                       authorization_sha256=authorization_sha256,
                                       source_package_sha256=source_package_sha256))
    profiles = Counter(str(row["profile"]) for row in validated)
    if profiles != Counter({profile: ROOTS_PER_PROFILE for profile in M43_ATTEMPT09_PROFILES}):
        raise ValueError(f"Attempt09 {POPULATION} population is not {ROOTS_PER_PROFILE}/profile")
    if len({row["hand_seed"] for row in validated}) != _ROOTS or len({row["observation_fingerprint"] for row in validated}) != _ROOTS:
        raise ValueError("Attempt09 development roots repeat")
    all_rng = [digest for row in validated for digest in row["rng_digests"]]
    if len(all_rng) != len(set(all_rng)):
        raise ValueError("Attempt09 RNG domains overlap across roots")
    all_beliefs = [digest for row in validated for digest in row["belief_digests"]]
    if len(all_beliefs) != len(set(all_beliefs)):
        raise ValueError("Attempt09 belief digests overlap across roots")
    overall = _metrics(validated)
    by_profile = {profile: _metrics([row for row in validated if row["profile"] == profile]) for profile in M43_ATTEMPT09_PROFILES}
    fires_by_profile = {profile: by_profile[profile]["fires"] for profile in M43_ATTEMPT09_PROFILES}
    tails = overall["maximum_per_fired_root_loss"]
    gates = [
        _gate("fires_total", overall["fires"] >= int(declared["fires_total_min"]), overall["fires"], f">= {declared['fires_total_min']}"),
        _gate("fires_each_profile", all(value >= int(declared["fires_each_profile_min"]) for value in fires_by_profile.values()), fires_by_profile, f"each >= {declared['fires_each_profile_min']}"),
        _gate("mean_delta_per_state", overall["mean_delta_per_state"] > 0, overall["mean_delta_per_state"], "> 0"),
        _gate("mean_delta_per_fire", overall["mean_delta_per_fire"] is not None and overall["mean_delta_per_fire"] > 0, overall["mean_delta_per_fire"], "> 0"),
        _gate("false_positive_rate_per_fire", overall["false_positive_rate_per_fire"] is not None and overall["false_positive_rate_per_fire"] <= .4, overall["false_positive_rate_per_fire"], "<= 0.4"),
        _gate("maximum_per_fired_root_p95_loss", tails["p95"] is not None and tails["p95"] <= 25, tails["p95"], "<= 25"),
        _gate("maximum_per_fired_root_p99_loss", tails["p99"] is not None and tails["p99"] <= 40, tails["p99"], "<= 40"),
        _gate("maximum_per_fired_root_max_loss", tails["max"] is not None and tails["max"] <= 50, tails["max"], "<= 50"),
    ]
    for name in ("action_mapping", "rng_domain", "hidden_information", "risk_reserve_contract", "retained_order", "phase_filter", "locked_action_change"):
        gates.append(_gate(f"{name}_violation_count", True, 0, "= 0"))
    gates.append(_gate("nonfire_exact_baseline_action_fallback", True, True, "required"))
    passed = all(gate["passed"] for gate in gates)
    identities = [{key: row[key] for key in ("root_index", "profile", "hand_seed", "observation_fingerprint", "baseline_action_key", "config_sha256")} for row in validated]
    return {
        "schema": ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA,
        "status": GO_STATUS if passed else NO_GO_STATUS,
        "decision": "go" if passed else "no_go", "search_freeze_authorized": passed,
        "selected_arm": None, "selected_threshold": None,
        "source": {"input_jsonl_sha256": source_input_sha256, "plan_sha256": source_plan_sha256,
                   "authorization_sha256": authorization_sha256, "source_package_sha256": source_package_sha256,
                   "run_name": run_name, "selector_source_sha256": hashlib.sha256(Path(SELECTOR_SOURCE_PATH).read_bytes()).hexdigest(),
                   "root_identity_sha256": _canonical_sha256({"roots": identities})},
        POPULATION_REPORT_KEY: {"roots": _ROOTS, "profile_counts": dict(profiles)},
        "metrics": {"overall": overall, "by_profile": by_profile,
                    "fired_root_diagnostics": [{"root_index": row["root_index"], "profile": row["profile"],
                        "selected_action_key": row["selected_action_key"], "e256_paired_delta_mean": row["mean"],
                        "e256_per_root_loss": row["loss"], "false_positive": row["mean"] <= 0} for row in validated if row["fired"]]},
        "gates": gates,
        "decision_contract": {"single_frozen_search_architecture": True, "arm_selection_performed": False,
                              "threshold_selection_performed": False, "gate_evaluation_count": 1, "all_gates_required": True},
        "integrity": {"action_mapping_violation_count": 0, "rng_domain_violation_count": 0,
                      "hidden_information_violation_count": 0, "risk_reserve_contract_violation_count": 0,
                      "retained_order_violation_count": 0, "phase_filter_violation_count": 0,
                      "locked_action_change_violation_count": 0, "nonfire_exact_baseline_action_fallback_verified": True},
        "science_boundary": {"assessment_source": declared["assessment_source"],
                             "teacher_values_are_realized_match_ev": False, "future_audit_authorized": False,
                             "fit_performed": False, "threshold_selected": False, "runtime_policy_activated": False,
                             "current_profile_mutated": False, "full_replacement_enabled": False},
    }


def select_attempt09_development(*, input_path: str | Path, plan_path: str | Path,
        authorization_path: str | Path, source_package_sha256: str, run_name: str) -> dict[str, Any]:
    plan_bytes = Path(plan_path).read_bytes()
    plan = load_and_validate_attempt09_plan(plan_path)
    _, auth_sha = _authorization(authorization_path, mode=POPULATION, source_package_sha256=source_package_sha256)
    if auth_sha is None:
        raise ValueError("Attempt09 development authorization is missing")
    input_bytes = Path(input_path).read_bytes()
    return aggregate_attempt09_development_rows(_read_rows(input_bytes), plan=plan,
        source_input_sha256=hashlib.sha256(input_bytes).hexdigest(),
        source_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), authorization_sha256=auth_sha,
        source_package_sha256=source_package_sha256, run_name=run_name)


def execute_attempt09_development_selector(*, input_path: str | Path, plan_path: str | Path,
        authorization_path: str | Path, source_package_sha256: str, run_name: str,
        output_dir: str | Path) -> dict[str, Any]:
    """Evaluate once and atomically publish decision.json plus its receipt."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Attempt09 selector output already exists: {destination}")
    report = select_attempt09_development(input_path=input_path, plan_path=plan_path,
        authorization_path=authorization_path, source_package_sha256=source_package_sha256, run_name=run_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir()
    try:
        decision_bytes = _canonical(report)
        receipt = {"schema": ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA,
            "status": "single_frozen_gate_evaluation_complete", "run_name": run_name,
            "decision_sha256": hashlib.sha256(decision_bytes).hexdigest(), "decision": report["decision"],
            "search_freeze_authorized": report["search_freeze_authorized"], "gate_evaluation_count": 1,
            "selector_executed": True, "future_audit_authorized": False, "fit_performed": False,
            "threshold_selected": False, "runtime_policy_activated": False, "current_profile_mutated": False}
        for name, data in (("decision.json", decision_bytes), ("decision_receipt.json", _canonical(receipt))):
            with (staging / name).open("xb") as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(f"Attempt09 selector output already exists: {destination}")
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply frozen Attempt09 development gates once")
    parser.add_argument("--input", required=True, type=Path); parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path); parser.add_argument("--source-package-sha256", required=True)
    parser.add_argument("--run-name", required=True); parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = execute_attempt09_development_selector(input_path=args.input, plan_path=args.plan,
        authorization_path=args.authorization, source_package_sha256=args.source_package_sha256,
        run_name=args.run_name, output_dir=args.output_dir)
    print(json.dumps(receipt, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
