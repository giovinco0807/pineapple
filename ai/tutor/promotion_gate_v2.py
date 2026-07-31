"""Content-bound validator for the M2 reduced public-tree promotion gate.

Unlike ``promotion_gate_v1``, v2 does not accept threshold-shaped assertions
as evidence.  Every published metric is re-derived from the embedded raw run
records, and every artifact hash is recomputed from an embedded manifest.
The gate can promote only the finite reduced reference; it can never promote
the full-card or serving policy.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "ai" / "config" / "promotion_gate_v2.json"
PROMOTION_GATE_V2_CONFIG_SHA256 = (
    "ad1f657dc098f5d35ebd593550e250c2f27f16e2315cf74721465e8c03e92fb2"
)
EVIDENCE_SCHEMA = "ofc_promotion_gate_evidence/v2"
RESULT_SCHEMA = "ofc_promotion_gate_result/v2"
PASS_STATUS = "m2_reduced_reference_ready"
FAIL_STATUS = "m2_in_progress"


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nearest_rank_p95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("p95 requires at least one value")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _weighted_p95(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights) or not values:
        raise ValueError("weighted p95 requires equally sized non-empty inputs")
    pairs = sorted((float(value), max(0.0, float(weight))) for value, weight in zip(values, weights))
    total = math.fsum(weight for _value, weight in pairs)
    if total <= 0:
        return _nearest_rank_p95(values)
    target = 0.95 * total
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= target:
            return value
    return pairs[-1][0]


def _strategy_tv(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if set(left) != set(right):
        return 1.0
    total = 0.0
    for action in left:
        try:
            total += abs(float.fromhex(str(left[action])) - float.fromhex(str(right[action])))
        except (TypeError, ValueError):
            return math.inf
    return 0.5 * total


def derive_stratum_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Derive every thresholded per-stratum metric from raw measurements."""
    audit = raw["audit"]
    runs = list(raw["runs"])
    if not runs:
        raise ValueError("stratum needs at least one run")

    def rate(observed: Any, expected: Any) -> float:
        expected_number = float(expected)
        return float(observed) / expected_number if expected_number > 0 else 0.0

    parity = list(audit["python_rust_leaf_metrics"])
    parity_fields = ("score", "raw_score", "royalty", "bust_rate", "fl_rate")
    max_parity_error = 0.0
    for record in parity:
        for field in parity_fields:
            max_parity_error = max(
                max_parity_error,
                abs(float(record["rust"][field]) - float(record["python"][field])),
            )

    primary = runs[0]
    metrics = primary["metrics"]
    unilateral = max(
        0.0,
        float(metrics["bb_best_response"]) - float(metrics["value_bb"]),
        float(metrics["value_bb"]) - float(metrics["btn_best_response"]),
    )
    trace = list(primary["exploitability_trace"])

    pair_tvs: list[float] = []
    pair_weights: list[float] = []
    value_drifts: list[float] = []
    exploitability_drifts: list[float] = []
    pairwise = 0
    for left, right in itertools.combinations(runs, 2):
        pairwise += 1
        left_strategy = left["strategy"]
        right_strategy = right["strategy"]
        all_keys = sorted(set(left_strategy) | set(right_strategy))
        for digest in all_keys:
            if digest not in left_strategy or digest not in right_strategy:
                pair_tvs.append(1.0)
                pair_weights.append(1.0)
                continue
            pair_tvs.append(
                _strategy_tv(
                    left_strategy[digest]["actions"],
                    right_strategy[digest]["actions"],
                )
            )
            pair_weights.append(
                0.5
                * (
                    float(left["infoset_reach"].get(digest, 0.0))
                    + float(right["infoset_reach"].get(digest, 0.0))
                )
            )
        value_drifts.append(
            abs(float(left["metrics"]["value_bb"]) - float(right["metrics"]["value_bb"]))
        )
        exploitability_drifts.append(
            abs(
                float(left["metrics"]["exploitability"])
                - float(right["metrics"]["exploitability"])
            )
        )

    weight_total = math.fsum(pair_weights)
    weighted_mean = (
        math.fsum(value * weight for value, weight in zip(pair_tvs, pair_weights))
        / weight_total
        if weight_total > 0
        else (math.fsum(pair_tvs) / len(pair_tvs) if pair_tvs else 0.0)
    )
    hidden_tvs = [float(value) for value in audit["hidden_only_mutation_policy_tvs"]]

    return {
        "legal_action_rate_eq": rate(audit["legal_action_matches"], audit["legal_action_checks"]),
        "candidate_coverage_rate_eq": rate(audit["candidate_actions_observed"], audit["candidate_actions_expected"]),
        "declared_reduced_transition_coverage_rate_eq": rate(audit["transition_matches"], audit["transition_checks"]),
        "chance_probability_mass_abs_error_lte": max(
            [abs(float(value)) for value in audit["chance_mass_errors"]] or [0.0]
        ),
        "python_rust_leaf_metric_abs_error_lte": max_parity_error,
        "duplicate_or_impossible_physical_card_failures_eq": int(audit["physical_card_failures"]),
        "selection_before_infoset_aggregation_failures_eq": int(audit["preselection_failures"]),
        "hidden_only_mutation_pairs_min": int(audit["hidden_only_mutation_pairs"]),
        "hidden_only_mutation_policy_tv_lte": max(hidden_tvs or [math.inf]),
        "forbidden_policy_input_failures_eq": int(audit["forbidden_policy_input_failures"]),
        "best_response_numerical_residual_lte": float(audit["best_response_numerical_residual"]),
        "max_unilateral_improvement_score_lte": unilateral,
        "nash_conv_score_lte": float(metrics["nash_conv"]),
        "exploitability_score_lte": float(metrics["exploitability"]),
        "final_trace_not_above_initial": bool(trace) and float(trace[-1][1]) <= float(trace[0][1]),
        "deterministic_order_replays_min": len(runs),
        "pairwise_comparisons_min": pairwise,
        "reach_weighted_mean_total_variation_lte": weighted_mean,
        "reach_weighted_p95_total_variation_lte": _weighted_p95(pair_tvs, pair_weights) if pair_tvs else 0.0,
        "max_total_variation_lte": max(pair_tvs or [0.0]),
        "max_game_value_drift_score_lte": max(value_drifts or [0.0]),
        "max_exploitability_drift_score_lte": max(exploitability_drifts or [0.0]),
        "x1_x2_identity_preservation_failures_eq": int(audit["x1_x2_identity_failures"]),
        "visible_joker_stratum_match": bool(audit["visible_joker_stratum_match"]),
        "position_contract_mismatches_eq": int(audit["position_contract_mismatches"]),
        "actor_sequence_mismatches_eq": int(audit["actor_sequence_mismatches"]),
        "decision_board_shape_mismatches_eq": int(audit["decision_board_shape_mismatches"]),
        "public_history_order_mismatches_eq": int(audit["public_history_order_mismatches"]),
    }


def derive_global_metrics(raw: Mapping[str, Any], provenance_ok: bool) -> dict[str, Any]:
    runtime = raw["runtime"]
    measured_by_stratum = runtime["recursive_measured_ms_by_stratum"]
    all_measured = [
        float(value)
        for name in sorted(measured_by_stratum)
        for value in measured_by_stratum[name]
    ]
    calibration = [float(value) for value in runtime["calibration_measured_ms"]]
    return {
        "pimc_counterexample_goldens_per_actor_min": min(
            int(value) for value in raw["pimc_counterexample_goldens_by_actor"].values()
        ),
        "solution_snapshot_round_trip_mismatches_eq": int(raw["solution_snapshot_round_trip_mismatches"]),
        "reduced_bluff_20000_iterations_wall_ms_max_lte": max(calibration or [math.inf]),
        "runtime_warmup_runs_min": min(
            len(values) for values in runtime["recursive_warmup_ms_by_stratum"].values()
        ),
        "runtime_measured_runs_min": min(len(values) for values in measured_by_stratum.values()),
        "canonical_recursive_solve_wall_ms_p95_lte": _nearest_rank_p95(all_measured),
        "canonical_recursive_solve_wall_ms_max_lte": max(all_measured or [math.inf]),
        "complete_gate_wall_s_lte": float(runtime["complete_gate_wall_s"]),
        "peak_rss_mb_gt": float(runtime["peak_rss_mb"]),
        "all_provenance_hashes_bound": bool(provenance_ok),
        "artifact_status_matches_result": True,
    }


def _compare(path: str, key: str, actual: Any, expected: Any, errors: list[str]) -> None:
    if isinstance(expected, bool):
        passed = actual is expected
    elif isinstance(expected, (int, float)):
        if not _finite(actual):
            errors.append(f"{path}: must be finite")
            return
        if key.endswith("_lte"):
            passed = float(actual) <= float(expected)
        elif key.endswith("_min") or key.endswith("_gte"):
            passed = float(actual) >= float(expected)
        elif key.endswith("_gt"):
            passed = float(actual) > float(expected)
        else:
            passed = float(actual) == float(expected)
    else:
        passed = actual == expected
    if not passed:
        errors.append(f"{path}: requirement {expected!r}, observed {actual!r}")


def _load_config(path: Path) -> tuple[dict[str, Any] | None, str, list[str]]:
    errors: list[str] = []
    try:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        config = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, "", [f"config: {exc}"]
    if digest != PROMOTION_GATE_V2_CONFIG_SHA256:
        errors.append(
            "config: locked SHA256 mismatch "
            f"({digest} != {PROMOTION_GATE_V2_CONFIG_SHA256})"
        )
    return config, digest, errors


def _validate_provenance(
    evidence: Mapping[str, Any], config: Mapping[str, Any], errors: list[str]
) -> bool:
    provenance = evidence.get("provenance")
    artifact = evidence.get("artifact")
    if not isinstance(provenance, Mapping) or not isinstance(artifact, Mapping):
        errors.append("provenance/artifact: both must be objects")
        return False
    manifests = provenance.get("hash_manifests")
    if not isinstance(manifests, Mapping):
        errors.append("provenance.hash_manifests: must be an object")
        return False
    required = config["provenance"]["required_hash_manifests"]
    if set(manifests) != set(required):
        errors.append("provenance.hash_manifests: required manifest set mismatch")
        return False
    fields = {
        "action_contract": "action_contract_sha256",
        "infoset_contract": "infoset_contract_sha256",
        "reduced_fixtures": "reduced_fixture_sha256",
        "chance_model": "chance_model_sha256",
        "range_model": "range_model_sha256",
        "solver_config": "solver_config_sha256",
        "solver_code": "solver_code_sha256",
        "source_tree": "source_tree_manifest_sha256",
    }
    ok = True
    for name, field in fields.items():
        actual = canonical_sha256(manifests[name])
        if artifact.get(field) != actual:
            errors.append(f"artifact.{field}: not bound to provenance.{name}")
            ok = False
    raw = evidence.get("raw_measurements")
    if artifact.get("raw_measurements_sha256") != canonical_sha256(raw):
        errors.append("artifact.raw_measurements_sha256: raw measurements hash mismatch")
        ok = False
    artifact_without_hash = dict(artifact)
    claimed = artifact_without_hash.pop("artifact_sha256", None)
    if claimed != canonical_sha256(artifact_without_hash):
        errors.append("artifact.artifact_sha256: canonical self-hash mismatch")
        ok = False
    rust_hash = provenance.get("rust_executable_sha256")
    if not isinstance(rust_hash, str) or len(rust_hash) != 64:
        errors.append("provenance.rust_executable_sha256: missing executable hash")
        ok = False
    return ok


def validate_promotion_evidence_v2(
    evidence: Any, *, config_path: str | Path = DEFAULT_CONFIG_PATH
) -> dict[str, Any]:
    config, config_digest, errors = _load_config(Path(config_path))
    evidence_digest: str | None = None
    try:
        evidence_digest = canonical_sha256(evidence)
    except (TypeError, ValueError):
        errors.append("evidence: not finite canonical JSON")
    if not isinstance(evidence, Mapping) or config is None:
        return {
            "schema": RESULT_SCHEMA,
            "gate_id": "promotion_gate_v2",
            "passed": False,
            "status": FAIL_STATUS,
            "promotion_result": None,
            "full_card_policy_promoted": False,
            "gate_config_sha256": config_digest or None,
            "evidence_sha256": evidence_digest,
            "failures": errors or ["evidence: must be an object"],
        }

    if evidence.get("schema") != EVIDENCE_SCHEMA:
        errors.append(f"evidence.schema: must equal {EVIDENCE_SCHEMA}")
    if evidence.get("gate_id") != config["gate_id"]:
        errors.append("evidence.gate_id: mismatch")
    if evidence.get("gate_config_sha256") != config_digest:
        errors.append("evidence.gate_config_sha256: mismatch")

    artifact = evidence.get("artifact")
    if not isinstance(artifact, Mapping):
        errors.append("artifact: must be an object")
        artifact = {}
    expected_artifact = {
        "schema": "ofc_m2_reduced_artifact/v2",
        "gate_id": config["gate_id"],
        "scope": config["scope"]["name"],
        "position_contract_version": config["contracts"]["position_contract_version"],
        "rules_version": config["contracts"]["rules_version"],
        "fl_ev_sha256": config["contracts"]["fl_ev_sha256"],
        "range_model": config["contracts"]["range_model"],
        "information_model": config["contracts"]["information_model"],
        "method": config["contracts"]["solver_method"],
        "best_response_method": config["contracts"]["best_response_method"],
        "strategy_fusion": False,
        "equilibrium_approx": True,
        "reduced_tree_enumeration_exact": True,
        "hu_exact": False,
        "full_card_policy_promoted": False,
        "explicit_full_deck_chance_enumeration": False,
        "solver_rng_used": False,
    }
    for field, expected in expected_artifact.items():
        if artifact.get(field) != expected:
            errors.append(f"artifact.{field}: must equal {expected!r}")
    for field in (
        "created_at_utc",
        "iterations",
        "replay_ids",
        "runtime_environment",
        "metrics",
        "tests",
        "artifact_sha256",
        "raw_measurements_sha256",
    ):
        if field not in artifact:
            errors.append(f"artifact.{field}: required")

    provenance_errors_before = len(errors)
    provenance_ok = _validate_provenance(evidence, config, errors)
    if len(errors) != provenance_errors_before:
        provenance_ok = False

    raw = evidence.get("raw_measurements")
    strata = evidence.get("strata")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("strata"), Mapping):
        errors.append("raw_measurements.strata: required")
        raw_strata: Mapping[str, Any] = {}
    else:
        raw_strata = raw["strata"]
    required_strata = set(config["strata"]["required"])
    if not isinstance(strata, Mapping) or set(strata) != required_strata:
        errors.append("strata: must contain exactly the six locked strata")
        strata = {}
    if set(raw_strata) != required_strata:
        errors.append("raw_measurements.strata: must contain exactly the six locked strata")

    for name in sorted(required_strata & set(strata) & set(raw_strata)):
        published = strata[name]
        raw_item = raw_strata[name]
        actor, joker_text = name.split("_joker")
        if published.get("actor") != actor or published.get("visible_joker_count") != int(joker_text):
            errors.append(f"strata.{name}: identity mismatch")
        try:
            derived = derive_stratum_metrics(raw_item)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"strata.{name}: cannot derive metrics: {exc}")
            continue
        if published.get("metrics") != derived:
            errors.append(f"strata.{name}.metrics: not equal to raw-derived metrics")
        for key, expected in config["per_stratum_thresholds"].items():
            _compare(
                f"strata.{name}.metrics.{key}",
                key,
                derived.get(key),
                expected,
                errors,
            )

    published_global = evidence.get("global_metrics")
    if isinstance(raw, Mapping):
        try:
            derived_global = derive_global_metrics(raw, provenance_ok)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"global_metrics: cannot derive metrics: {exc}")
            derived_global = {}
    else:
        derived_global = {}
    if published_global != derived_global:
        errors.append("global_metrics: not equal to raw-derived metrics")
    for key, expected in config["global_thresholds"].items():
        _compare(f"global_metrics.{key}", key, derived_global.get(key), expected, errors)

    # Status is checked only after all substantive failures are known.
    substantive_errors = list(errors)
    expected_status = PASS_STATUS if not substantive_errors else FAIL_STATUS
    if artifact.get("status") != expected_status:
        errors.append(
            f"artifact.status: must be {expected_status!r} for the observed evidence"
        )

    passed = not errors
    return {
        "schema": RESULT_SCHEMA,
        "gate_id": config["gate_id"],
        "passed": passed,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "promotion_result": PASS_STATUS if passed else None,
        "scope": config["scope"]["name"],
        "full_card_policy_promoted": False,
        "gate_config_sha256": config_digest,
        "evidence_sha256": evidence_digest,
        "evaluated_strata": sorted(set(strata)),
        "failures": errors,
    }


def finalize_artifact_hash(artifact: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(artifact)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = canonical_sha256(result)
    return result


def write_gate_result(
    evidence_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    try:
        evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        evidence = {"load_error": str(exc)}
    result = validate_promotion_evidence_v2(evidence, config_path=config_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return result
