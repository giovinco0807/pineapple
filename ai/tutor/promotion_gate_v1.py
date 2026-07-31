"""Fail-closed validator for the locked M2 reduced-reference promotion gate.

This module deliberately does not construct fixtures, run a solver, or infer
missing measurements.  It only validates a complete evidence document against
the byte-locked ``promotion_gate_v1`` configuration.  Consequently synthetic
documents are useful for testing this validator, but are never runtime/solver
evidence and must not be published as a real gate result.

Evidence schema (``ofc_promotion_gate_evidence/v1``):

* ``gate_config_sha256`` commits to the exact locked configuration bytes;
* ``artifact`` contains every metadata field required by the configuration;
* ``strata`` contains exactly BB/BTN x visible Joker 0/1/2;
* every stratum contains its own complete copy of all required gate metrics.

Threshold suffixes are interpreted mechanically: ``_eq``, ``_lte``,
``_gte``/``_min``, and ``_gt``.  Other scalar requirements must equal the
configured value.  Missing and non-finite values fail closed.  A successful
result can only mark ``m2_reduced_reference_ready``; this validator has no
full-card promotion state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "ai" / "config" / "promotion_gate_v1.json"

# Byte hash of ai/config/promotion_gate_v1.json when v1 was locked.  Editing
# thresholds in-place must fail until a new gate version is created.
PROMOTION_GATE_V1_CONFIG_SHA256 = (
    "adcaee5cb1f532748d2aa9528ff3f690e5253379aec690bb0bc5f8469af2624a"
)

EVIDENCE_SCHEMA = "ofc_promotion_gate_evidence/v1"
RESULT_SCHEMA = "ofc_promotion_gate_result/v1"
PASS_STATUS = "m2_reduced_reference_ready"
FAIL_STATUS = "m2_in_progress"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_NON_METRIC_CONFIG_KEYS = {
    "required",
    "basis",
    "provenance",
    "required_fields",
    "sha256_fields",
    "required_values",
    "forbidden_policy_inputs",
    "visible_joker_counts",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _add_error(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def _load_locked_config(config_path: Path) -> tuple[dict[str, Any] | None, str, list[str]]:
    errors: list[str] = []
    try:
        raw = config_path.read_bytes()
    except OSError as exc:
        return None, "", [f"config: cannot read locked config: {exc}"]

    digest = _sha256_bytes(raw)
    if digest != PROMOTION_GATE_V1_CONFIG_SHA256:
        _add_error(
            errors,
            "config",
            "locked SHA256 mismatch "
            f"(expected {PROMOTION_GATE_V1_CONFIG_SHA256}, got {digest})",
        )
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, digest, errors + [f"config: invalid JSON: {exc}"]

    required_guards = {
        "schema": "ofc_promotion_gate/v1",
        "gate_id": "promotion_gate_v1",
        "milestone": "M2",
    }
    for key, expected in required_guards.items():
        if config.get(key) != expected:
            _add_error(errors, f"config.{key}", f"must equal {expected!r}")

    scope = config.get("scope")
    if not isinstance(scope, Mapping):
        _add_error(errors, "config.scope", "must be an object")
    else:
        if scope.get("promotion_result") != PASS_STATUS:
            _add_error(
                errors,
                "config.scope.promotion_result",
                f"must equal {PASS_STATUS!r}",
            )
        false_claims = scope.get("required_false_claims")
        if false_claims != {"hu_exact": False, "full_card_policy_promoted": False}:
            _add_error(errors, "config.scope.required_false_claims", "unsafe scope")

    decision = config.get("decision_rule")
    if not isinstance(decision, Mapping):
        _add_error(errors, "config.decision_rule", "must be an object")
    else:
        for key in (
            "all_required_gates_must_pass",
            "all_required_strata_must_pass",
            "missing_or_non_finite_metric_is_failure",
            "pooled_metric_cannot_override_stratum_failure",
            "threshold_changes_require_new_gate_version",
        ):
            if decision.get(key) is not True:
                _add_error(errors, f"config.decision_rule.{key}", "must be true")

    return config, digest, errors


def _compare_requirement(
    *, path: str, key: str, observed: Any, expected: Any, errors: list[str]
) -> None:
    """Compare one configured scalar requirement deterministically."""

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not _is_finite_number(observed):
            _add_error(errors, path, "must be a finite number")
            return
        actual = float(observed)
        limit = float(expected)
        if key.endswith("_lte"):
            passed = actual <= limit
            relation = "<="
        elif key.endswith("_gte") or key.endswith("_min"):
            passed = actual >= limit
            relation = ">="
        elif key.endswith("_gt"):
            passed = actual > limit
            relation = ">"
        elif key.endswith("_eq"):
            passed = actual == limit
            relation = "=="
        else:
            passed = actual == limit
            relation = "=="
        if not passed:
            _add_error(errors, path, f"must be {relation} {expected!r}; got {observed!r}")
        return

    if observed != expected:
        _add_error(errors, path, f"must equal {expected!r}; got {observed!r}")


def _validate_artifact(
    artifact: Any, config: Mapping[str, Any], errors: list[str]
) -> None:
    path = "evidence.artifact"
    if not isinstance(artifact, Mapping):
        _add_error(errors, path, "must be an object")
        return

    metadata_gate = config["gates"]["artifact_metadata"]
    for field in metadata_gate["required_fields"]:
        if field not in artifact or artifact[field] is None:
            _add_error(errors, f"{path}.{field}", "required field is missing")

    for field in metadata_gate["sha256_fields"]:
        value = artifact.get(field)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            _add_error(errors, f"{path}.{field}", "must be a lowercase 64-hex SHA256")

    for field, expected in metadata_gate["required_values"].items():
        if artifact.get(field) != expected:
            _add_error(
                errors,
                f"{path}.{field}",
                f"must equal locked value {expected!r}; got {artifact.get(field)!r}",
            )

    expected_identity = {
        "schema": config["schema"],
        "gate_id": config["gate_id"],
        "scope": config["scope"]["name"],
        "fl_ev_sha256": config["contracts"]["fl_ev_sha256"],
    }
    for field, expected in expected_identity.items():
        if artifact.get(field) != expected:
            _add_error(errors, f"{path}.{field}", f"must equal {expected!r}")

    status = artifact.get("status")
    if status not in {FAIL_STATUS, PASS_STATUS}:
        _add_error(
            errors,
            f"{path}.status",
            f"must be {FAIL_STATUS!r} or {PASS_STATUS!r}; full-card claims are forbidden",
        )

    created_at = artifact.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at.endswith("Z"):
        _add_error(errors, f"{path}.created_at_utc", "must be an ISO-8601 UTC string ending Z")
    else:
        try:
            datetime.fromisoformat(created_at[:-1] + "+00:00")
        except ValueError:
            _add_error(errors, f"{path}.created_at_utc", "invalid ISO-8601 UTC timestamp")

    for field in ("method", "information_model", "exploitability_method"):
        value = artifact.get(field)
        if not isinstance(value, str) or not value.strip():
            _add_error(errors, f"{path}.{field}", "must be a non-empty string")

    iterations = artifact.get("iterations")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        _add_error(errors, f"{path}.iterations", "must be a positive integer")

    seeds = artifact.get("seeds")
    minimum_runs = config["gates"]["strategy_drift"]["independent_runs_min"]
    if not isinstance(seeds, list) or len(seeds) < minimum_runs:
        _add_error(errors, f"{path}.seeds", f"must contain at least {minimum_runs} seeds")
    else:
        seed_keys: list[tuple[type[Any], Any]] = []
        invalid_seed = False
        for index, seed in enumerate(seeds):
            if isinstance(seed, bool) or not isinstance(seed, (int, str)):
                _add_error(
                    errors,
                    f"{path}.seeds[{index}]",
                    "must be an integer or non-empty string",
                )
                invalid_seed = True
            elif isinstance(seed, str) and not seed:
                _add_error(errors, f"{path}.seeds[{index}]", "must not be empty")
                invalid_seed = True
            else:
                seed_keys.append((type(seed), seed))
        if not invalid_seed and len(set(seed_keys)) != len(seed_keys):
            _add_error(errors, f"{path}.seeds", "seeds must be unique")

    particle_count = artifact.get("range_particle_count")
    ess = artifact.get("range_effective_sample_size")
    minimum_particles = metadata_gate["range_particle_count_min"]
    if (
        not isinstance(particle_count, int)
        or isinstance(particle_count, bool)
        or particle_count < minimum_particles
    ):
        _add_error(
            errors,
            f"{path}.range_particle_count",
            f"must be an integer >= {minimum_particles}",
        )
    if not _is_finite_number(ess) or float(ess) <= metadata_gate["range_effective_sample_size_gt"]:
        _add_error(errors, f"{path}.range_effective_sample_size", "must be finite and > 0")
    elif isinstance(particle_count, int) and float(ess) > particle_count:
        _add_error(
            errors,
            f"{path}.range_effective_sample_size",
            "must not exceed range_particle_count",
        )

    runtime_environment = artifact.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping):
        _add_error(errors, f"{path}.runtime_environment", "must be an object")
    else:
        for field in ("machine", "cpu", "python", "git_commit", "git_dirty"):
            if field not in runtime_environment:
                _add_error(errors, f"{path}.runtime_environment.{field}", "required field is missing")
        if "git_dirty" in runtime_environment and not isinstance(
            runtime_environment["git_dirty"], bool
        ):
            _add_error(errors, f"{path}.runtime_environment.git_dirty", "must be boolean")

    for field in ("metrics", "tests"):
        value = artifact.get(field)
        if not isinstance(value, (Mapping, list)) or not value:
            _add_error(errors, f"{path}.{field}", "must be a non-empty object or list")


def _validate_stratum(
    name: str,
    value: Any,
    config: Mapping[str, Any],
    errors: list[str],
) -> None:
    path = f"evidence.strata.{name}"
    if not isinstance(value, Mapping):
        _add_error(errors, path, "must be an object")
        return

    match = re.fullmatch(r"(bb|btn)_joker([012])", name)
    if match is None:  # protected by the required-strata equality check
        _add_error(errors, path, "invalid stratum name")
        return
    expected_actor, joker_text = match.groups()
    expected_joker = int(joker_text)
    if value.get("actor") != expected_actor:
        _add_error(errors, f"{path}.actor", f"must equal {expected_actor!r}")
    if value.get("visible_joker_count") != expected_joker:
        _add_error(
            errors,
            f"{path}.visible_joker_count",
            f"must equal {expected_joker}",
        )

    fixture_count = value.get("canonical_reduced_fixture_count")
    fixture_minimum = config["strata"]["minimum_canonical_reduced_fixtures_per_stratum"]
    if (
        not isinstance(fixture_count, int)
        or isinstance(fixture_count, bool)
        or fixture_count < fixture_minimum
    ):
        _add_error(
            errors,
            f"{path}.canonical_reduced_fixture_count",
            f"must be an integer >= {fixture_minimum}",
        )

    observed_gates = value.get("gates")
    if not isinstance(observed_gates, Mapping):
        _add_error(errors, f"{path}.gates", "must be an object")
        return
    required_gate_names = {
        gate_name
        for gate_name, gate in config["gates"].items()
        if gate.get("required") is True
    }
    actual_gate_names = set(observed_gates)
    if actual_gate_names != required_gate_names:
        missing = sorted(required_gate_names - actual_gate_names)
        extra = sorted(actual_gate_names - required_gate_names)
        if missing:
            _add_error(errors, f"{path}.gates", f"missing required gates: {missing}")
        if extra:
            _add_error(errors, f"{path}.gates", f"unexpected gates: {extra}")

    for gate_name in sorted(required_gate_names & actual_gate_names):
        gate_path = f"{path}.gates.{gate_name}"
        observed = observed_gates[gate_name]
        if not isinstance(observed, Mapping):
            _add_error(errors, gate_path, "must be an object")
            continue
        gate_config = config["gates"][gate_name]
        for metric_name, expected in gate_config.items():
            if metric_name in _NON_METRIC_CONFIG_KEYS:
                continue
            if isinstance(expected, (Mapping, list)):
                continue
            metric_path = f"{gate_path}.{metric_name}"
            if metric_name not in observed:
                _add_error(errors, metric_path, "required metric is missing")
                continue
            _compare_requirement(
                path=metric_path,
                key=metric_name,
                observed=observed[metric_name],
                expected=expected,
                errors=errors,
            )

        if gate_name == "information_leakage":
            checked = observed.get("forbidden_policy_inputs_checked")
            expected_checked = gate_config["forbidden_policy_inputs"]
            if checked != expected_checked:
                _add_error(
                    errors,
                    f"{gate_path}.forbidden_policy_inputs_checked",
                    "must exactly list every locked forbidden input",
                )
            if observed.get("forbidden_policy_inputs_found") != []:
                _add_error(
                    errors,
                    f"{gate_path}.forbidden_policy_inputs_found",
                    "must be an explicit empty list",
                )

        if gate_name == "joker":
            if expected_joker not in gate_config["visible_joker_counts"]:
                _add_error(errors, gate_path, "stratum Joker count is not configured")

        if gate_name == "artifact_metadata":
            particle_count = observed.get("range_particle_count")
            ess = observed.get("range_effective_sample_size")
            if _is_finite_number(particle_count) and _is_finite_number(ess):
                if float(ess) > float(particle_count):
                    _add_error(
                        errors,
                        f"{gate_path}.range_effective_sample_size",
                        "must not exceed range_particle_count",
                    )


def validate_promotion_evidence(
    evidence: Any,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Validate one complete evidence payload and return an honest M2 result."""

    config, config_digest, errors = _load_locked_config(Path(config_path))
    evidence_digest: str | None = None
    try:
        evidence_digest = _canonical_sha256(evidence)
    except (TypeError, ValueError):
        _add_error(errors, "evidence", "must be finite, canonical JSON data")

    if not isinstance(evidence, Mapping):
        _add_error(errors, "evidence", "must be an object")
    elif config is not None:
        if evidence.get("schema") != EVIDENCE_SCHEMA:
            _add_error(errors, "evidence.schema", f"must equal {EVIDENCE_SCHEMA!r}")
        if evidence.get("gate_id") != config["gate_id"]:
            _add_error(errors, "evidence.gate_id", f"must equal {config['gate_id']!r}")
        if evidence.get("gate_config_sha256") != config_digest:
            _add_error(
                errors,
                "evidence.gate_config_sha256",
                f"must commit to locked config {config_digest}",
            )

        _validate_artifact(evidence.get("artifact"), config, errors)

        strata = evidence.get("strata")
        required_strata = list(config["strata"]["required"])
        if not isinstance(strata, Mapping):
            _add_error(
                errors,
                "evidence.strata",
                "must contain complete per-stratum evidence; pooled evidence is insufficient",
            )
        else:
            actual = set(strata)
            required = set(required_strata)
            if actual != required:
                missing = sorted(required - actual)
                extra = sorted(actual - required)
                if missing:
                    _add_error(
                        errors,
                        "evidence.strata",
                        f"missing required strata: {missing}; pooled evidence cannot replace them",
                    )
                if extra:
                    _add_error(errors, "evidence.strata", f"unexpected strata: {extra}")
            for name in sorted(required & actual):
                _validate_stratum(name, strata[name], config, errors)

    passed = not errors
    return {
        "schema": RESULT_SCHEMA,
        "gate_id": "promotion_gate_v1",
        "passed": passed,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "promotion_result": PASS_STATUS if passed else None,
        "scope": "reduced_t3_t4_public_tree",
        "full_card_policy_promoted": False,
        "gate_config_sha256": config_digest or None,
        "evidence_sha256": evidence_digest,
        "evaluated_strata": (
            sorted(evidence.get("strata", {}))
            if isinstance(evidence, Mapping) and isinstance(evidence.get("strata"), Mapping)
            else []
        ),
        "failures": errors,
    }


def run_promotion_gate(
    evidence_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load evidence, validate it, and write the result as deterministic JSON."""

    evidence_file = Path(evidence_path)
    try:
        evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        evidence = {"_load_failure": str(exc)}

    result = validate_promotion_evidence(evidence, config_path=config_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    result = run_promotion_gate(args.evidence, args.output, config_path=args.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
