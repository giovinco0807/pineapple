"""Fail-closed evidence gate for the endogenous T3-BB likelihood route.

The BTN ``t3_second`` posterior consumes the immediately preceding BB T3
action.  That likelihood cannot be supplied by the frozen ranking prior used
by the execution smoke: it must be regenerated from the candidate policy and
iterated with the physical-card range until both policy and posterior weights
are stable.  This module makes that fixed-point claim independently
re-derivable from raw rows.

The gate deliberately does not run the solver.  It validates a locked root and
seed partition, checks every BB/BTN x Joker0/1/2 stratum for every independent
seed and round, re-derives total-variation distances, verifies round-to-round
chains, and requires consecutive converged rounds.  Only
``build_t3_bb_likelihood_binding`` can turn a verified passing result into the
compact binding consumed by the M3 strength schema.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

from ai.tutor.promotion_gate_m3_full_card_strength import (
    INFORMATION_MODEL,
    POSITION_CONTRACT_VERSION,
    REQUIRED_EXCLUDED_PARTITIONS,
    REQUIRED_STRATA,
    RULESET,
    SOLVER_METHOD,
    T3_BB_LIKELIHOOD_METHOD,
    T3_BB_LIKELIHOOD_SCHEMA,
    root_identity_commitment_sha256,
    verify_t3_bb_likelihood_binding,
)


CONFIG_SCHEMA = "ofc_m3_t3_bb_fixed_point_gate_config/v1"
EVIDENCE_SCHEMA = "ofc_m3_t3_bb_fixed_point_evidence/v1"
RESULT_SCHEMA = "ofc_m3_t3_bb_fixed_point_gate_result/v1"
ROOT_MANIFEST_SCHEMA = "ofc_m3_t3_bb_fixed_point_roots/v1"
ROOT_COMMITMENT_SCHEMA = "ofc_m3_t3_bb_fixed_point_root_commitment/v1"
PARTITION_SCHEMA = "ofc_m3_t3_bb_fixed_point_excluded_partition/v1"
GATE_ID = "promotion_gate_m3_t3_bb_fixed_point"
SCOPE = "t3_bb_candidate_policy_btn_posterior_fixed_point"
EVIDENCE_KIND = "raw_root_seed_iteration_rows"
PASS_STATUS = "t3_bb_fixed_point_ready"
FAIL_STATUS = "t3_bb_fixed_point_blocked"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_TOL = 1e-12

THRESHOLD_KEYS = frozenset(
    {
        "min_independent_seeds",
        "min_roots_per_stratum",
        "min_consecutive_converged_rounds",
        "max_policy_tv",
        "max_btn_posterior_weight_tv",
    }
)
CONFIG_KEYS = frozenset(
    {
        "schema",
        "gate_id",
        "scope",
        "locked_before_evaluation",
        "ruleset",
        "position_contract_version",
        "information_model",
        "exact_exploitability_computed",
        "candidate_policy_method",
        "approved_solver_manifest_sha256",
        "approved_range_builder_source_sha256",
        "approved_root_manifest_sha256",
        "approved_excluded_partition_sha256",
        "thresholds",
        "gate_config_sha256",
    }
)
ROOT_KEYS = frozenset(
    {
        "root_id",
        "stratum",
        "actor",
        "visible_joker_count",
        "root_identity_commitment_sha256",
        "root_commitment_sha256",
    }
)
ROOT_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "purpose",
        "locked_before_evaluation",
        "ruleset",
        "position_contract_version",
        "evaluation_seeds",
        "roots",
        "manifest_sha256",
    }
)
PARTITION_KEYS = frozenset(
    {
        "schema",
        "purpose",
        "root_identity_commitments",
        "solver_seeds",
        "manifest_sha256",
    }
)
ROW_KEYS = frozenset(
    {
        "round_index",
        "solver_seed",
        "root_id",
        "root_commitment_sha256",
        "root_identity_commitment_sha256",
        "stratum",
        "actor",
        "visible_joker_count",
        "previous_candidate_policy_artifact_sha256",
        "previous_candidate_policy_checkpoint_sha256",
        "candidate_policy_artifact_sha256",
        "candidate_policy_checkpoint_sha256",
        "candidate_policy_method",
        "solver_manifest_sha256",
        "range_builder_source_sha256",
        "exact_exploitability_computed",
        "previous_policy_distribution",
        "current_policy_distribution",
        "previous_btn_posterior_weights",
        "current_btn_posterior_weights",
    }
)
EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "gate_id",
        "scope",
        "evidence_kind",
        "gate_config_sha256",
        "exact_exploitability_computed",
        "candidate_policy_method",
        "solver_manifest_sha256",
        "range_builder_source_sha256",
        "root_manifest",
        "excluded_partitions",
        "raw_iteration_rows",
        "published_summary",
        "artifact_sha256",
    }
)


def canonical_json(value: Any) -> str:
    """Return finite canonical JSON for all fixed-point artifacts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def self_hash(value: Mapping[str, Any], field: str) -> str:
    content = dict(value)
    content.pop(field, None)
    return canonical_sha256(content)


def root_commitment_sha256(root: Mapping[str, Any]) -> str:
    content = dict(root)
    content.pop("root_commitment_sha256", None)
    return canonical_sha256({"schema": ROOT_COMMITMENT_SCHEMA, "root": content})


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _hash_claim_or_none(value: Any) -> str | None:
    """Keep malformed claims out of the self-hashed failure result."""

    return value if isinstance(value, str) else None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str, failures: list[str]
) -> None:
    if set(value) != expected:
        failures.append(
            f"{label}: exact versioned fields required; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _check_self_hash(
    value: Mapping[str, Any], field: str, label: str, failures: list[str]
) -> str | None:
    try:
        computed = self_hash(value, field)
    except (TypeError, ValueError) as exc:
        failures.append(f"{label}: canonical hash failed: {exc}")
        return None
    if not _valid_sha256(value.get(field)):
        failures.append(f"{label}.{field}: lowercase SHA256 required")
    elif value.get(field) != computed:
        failures.append(f"{label}.{field}: self-hash mismatch")
    return computed


def _expected_identity(stratum: str) -> tuple[str, int]:
    actor, joker = stratum.split("_joker", 1)
    return actor, int(joker)


def build_locked_t3_bb_fixed_point_gate_config(
    *,
    approved_solver_manifest_sha256: str,
    approved_range_builder_source_sha256: str,
    approved_root_manifest_sha256: str,
    approved_excluded_partition_sha256: Mapping[str, str],
    max_policy_tv: float,
    max_btn_posterior_weight_tv: float,
    min_independent_seeds: int = 2,
    min_roots_per_stratum: int = 1,
    min_consecutive_converged_rounds: int = 2,
) -> dict[str, Any]:
    """Build and validate the immutable gate configuration."""

    config: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "locked_before_evaluation": True,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "information_model": INFORMATION_MODEL,
        "exact_exploitability_computed": False,
        "candidate_policy_method": SOLVER_METHOD,
        "approved_solver_manifest_sha256": approved_solver_manifest_sha256,
        "approved_range_builder_source_sha256": (
            approved_range_builder_source_sha256
        ),
        "approved_root_manifest_sha256": approved_root_manifest_sha256,
        "approved_excluded_partition_sha256": dict(
            approved_excluded_partition_sha256
        ),
        "thresholds": {
            "min_independent_seeds": min_independent_seeds,
            "min_roots_per_stratum": min_roots_per_stratum,
            "min_consecutive_converged_rounds": (
                min_consecutive_converged_rounds
            ),
            "max_policy_tv": float(max_policy_tv),
            "max_btn_posterior_weight_tv": float(
                max_btn_posterior_weight_tv
            ),
        },
    }
    config["gate_config_sha256"] = self_hash(config, "gate_config_sha256")
    failures = _validate_config(config)
    if failures:
        raise ValueError("; ".join(failures))
    return config


def _validate_config(config: Any) -> list[str]:
    failures: list[str] = []
    if not isinstance(config, Mapping):
        return ["config: object required"]
    _exact_fields(config, CONFIG_KEYS, "config", failures)
    _check_self_hash(config, "gate_config_sha256", "config", failures)
    expected = {
        "schema": CONFIG_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "locked_before_evaluation": True,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "information_model": INFORMATION_MODEL,
        "exact_exploitability_computed": False,
        "candidate_policy_method": SOLVER_METHOD,
    }
    for field, wanted in expected.items():
        if config.get(field) != wanted:
            failures.append(f"config.{field}: must equal {wanted!r}")
    for field in (
        "approved_solver_manifest_sha256",
        "approved_range_builder_source_sha256",
        "approved_root_manifest_sha256",
    ):
        if not _valid_sha256(config.get(field)):
            failures.append(f"config.{field}: lowercase SHA256 required")
    partitions = config.get("approved_excluded_partition_sha256")
    if not isinstance(partitions, Mapping) or set(partitions) != set(
        REQUIRED_EXCLUDED_PARTITIONS
    ):
        failures.append(
            "config.approved_excluded_partition_sha256: exact "
            "training/calibration/smoke mapping required"
        )
    elif any(not _valid_sha256(value) for value in partitions.values()):
        failures.append(
            "config.approved_excluded_partition_sha256: lowercase SHA256 values required"
        )
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, Mapping):
        failures.append("config.thresholds: object required")
        return failures
    _exact_fields(thresholds, THRESHOLD_KEYS, "config.thresholds", failures)
    for field, lower in (
        ("min_independent_seeds", 2),
        ("min_roots_per_stratum", 1),
        ("min_consecutive_converged_rounds", 2),
    ):
        value = thresholds.get(field)
        if not _is_int(value) or value < lower:
            failures.append(f"config.thresholds.{field}: integer >= {lower} required")
    for field in ("max_policy_tv", "max_btn_posterior_weight_tv"):
        value = thresholds.get(field)
        if not _finite(value) or not 0.0 <= float(value) <= 1.0:
            failures.append(f"config.thresholds.{field}: finite value in [0,1] required")
    return failures


def verify_locked_t3_bb_fixed_point_gate_config(config: Any) -> dict[str, Any]:
    failures = _validate_config(config)
    if failures:
        raise ValueError("; ".join(failures))
    return dict(config)


def _validate_root_manifest(value: Any, failures: list[str]) -> dict[str, Any]:
    roots_by_id: dict[str, Mapping[str, Any]] = {}
    if not isinstance(value, Mapping):
        failures.append("evidence.root_manifest: object required")
        return {
            "roots_by_id": roots_by_id,
            "seeds": [],
            "identities": set(),
            "stratum_counts": {},
        }
    _exact_fields(value, ROOT_MANIFEST_KEYS, "evidence.root_manifest", failures)
    _check_self_hash(value, "manifest_sha256", "evidence.root_manifest", failures)
    expected = {
        "schema": ROOT_MANIFEST_SCHEMA,
        "purpose": "independent_fixed_point_holdout",
        "locked_before_evaluation": True,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            failures.append(
                f"evidence.root_manifest.{field}: must equal {wanted!r}"
            )
    seeds = value.get("evaluation_seeds")
    if (
        not isinstance(seeds, list)
        or any(not _is_int(seed) for seed in seeds)
        or seeds != sorted(set(seeds))
    ):
        failures.append(
            "evidence.root_manifest.evaluation_seeds: sorted unique integer list required"
        )
        seeds = []
    roots = value.get("roots")
    if not isinstance(roots, list):
        failures.append("evidence.root_manifest.roots: list required")
        roots = []
    identity_seen: set[str] = set()
    commitment_seen: set[str] = set()
    stratum_counts: defaultdict[str, int] = defaultdict(int)
    for index, root in enumerate(roots):
        label = f"evidence.root_manifest.roots[{index}]"
        if not isinstance(root, Mapping):
            failures.append(f"{label}: object required")
            continue
        _exact_fields(root, ROOT_KEYS, label, failures)
        root_id = root.get("root_id")
        if not isinstance(root_id, str) or not root_id:
            failures.append(f"{label}.root_id: non-empty string required")
            continue
        if root_id in roots_by_id:
            failures.append(f"{label}.root_id: duplicate root")
        roots_by_id[root_id] = root
        stratum = root.get("stratum")
        if stratum not in REQUIRED_STRATA:
            failures.append(f"{label}.stratum: required six-stratum value")
        else:
            actor, joker = _expected_identity(stratum)
            if root.get("actor") != actor:
                failures.append(f"{label}.actor: stratum mismatch")
            if root.get("visible_joker_count") != joker:
                failures.append(f"{label}.visible_joker_count: stratum mismatch")
            stratum_counts[stratum] += 1
        try:
            identity = root_identity_commitment_sha256(root_id)
        except ValueError:
            identity = None
        if root.get("root_identity_commitment_sha256") != identity:
            failures.append(f"{label}.root_identity_commitment_sha256: mismatch")
        if identity in identity_seen:
            failures.append(f"{label}.root_identity_commitment_sha256: duplicate")
        if identity is not None:
            identity_seen.add(identity)
        try:
            commitment = root_commitment_sha256(root)
        except (TypeError, ValueError) as exc:
            failures.append(f"{label}.root_commitment_sha256: canonical hash failed: {exc}")
            commitment = None
        if root.get("root_commitment_sha256") != commitment:
            failures.append(f"{label}.root_commitment_sha256: mismatch")
        if commitment is not None and commitment in commitment_seen:
            failures.append(f"{label}.root_commitment_sha256: duplicate")
        if commitment is not None:
            commitment_seen.add(commitment)
    return {
        "roots_by_id": roots_by_id,
        "seeds": list(seeds),
        "identities": identity_seen,
        "stratum_counts": dict(stratum_counts),
    }


def _validate_partitions(
    value: Any,
    *,
    holdout_identities: set[str],
    evaluation_seeds: Sequence[int],
    failures: list[str],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not isinstance(value, Mapping) or set(value) != set(
        REQUIRED_EXCLUDED_PARTITIONS
    ):
        failures.append(
            "evidence.excluded_partitions: exact training/calibration/smoke mapping required"
        )
        return hashes
    for purpose in REQUIRED_EXCLUDED_PARTITIONS:
        partition = value[purpose]
        label = f"evidence.excluded_partitions.{purpose}"
        if not isinstance(partition, Mapping):
            failures.append(f"{label}: object required")
            continue
        _exact_fields(partition, PARTITION_KEYS, label, failures)
        computed = _check_self_hash(partition, "manifest_sha256", label, failures)
        if computed is not None:
            hashes[purpose] = computed
        if partition.get("schema") != PARTITION_SCHEMA:
            failures.append(f"{label}.schema: must equal {PARTITION_SCHEMA!r}")
        if partition.get("purpose") != purpose:
            failures.append(f"{label}.purpose: must equal {purpose!r}")
        identities = partition.get("root_identity_commitments")
        if (
            not isinstance(identities, list)
            or any(not _valid_sha256(item) for item in identities)
            or identities != sorted(set(identities))
        ):
            failures.append(
                f"{label}.root_identity_commitments: sorted unique SHA256 list required"
            )
            identities = []
        overlap = holdout_identities & set(identities)
        if overlap:
            failures.append(
                f"evidence.root_manifest: root overlap with {purpose} partition"
            )
        seeds = partition.get("solver_seeds")
        if (
            not isinstance(seeds, list)
            or any(not _is_int(seed) for seed in seeds)
            or seeds != sorted(set(seeds))
        ):
            failures.append(f"{label}.solver_seeds: sorted unique integer list required")
            seeds = []
        if set(evaluation_seeds) & set(seeds):
            failures.append(
                f"evidence.root_manifest: seed overlap with {purpose} partition"
            )
    return hashes


def _validate_distribution(
    value: Any, label: str, failures: list[str]
) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or not value:
        failures.append(f"{label}: non-empty action-probability object required")
        return None
    converted: dict[str, float] = {}
    for key, probability in value.items():
        if not isinstance(key, str) or not key:
            failures.append(f"{label}: non-empty string action keys required")
            return None
        if not _finite(probability) or float(probability) < 0.0:
            failures.append(f"{label}.{key}: finite nonnegative probability required")
            return None
        converted[key] = float(probability)
    try:
        total = math.fsum(converted.values())
    except OverflowError:
        failures.append(f"{label}: probability sum overflow")
        return None
    if not math.isclose(total, 1.0, abs_tol=_FLOAT_TOL):
        failures.append(f"{label}: probabilities must sum to 1")
        return None
    return converted


def _validate_weights(
    value: Any, label: str, failures: list[str]
) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or not value:
        failures.append(f"{label}: non-empty raw posterior-weight object required")
        return None
    converted: dict[str, float] = {}
    for key, weight in value.items():
        if not isinstance(key, str) or not key:
            failures.append(f"{label}: non-empty string state keys required")
            return None
        if not _finite(weight) or float(weight) < 0.0:
            failures.append(f"{label}.{key}: finite nonnegative weight required")
            return None
        converted[key] = float(weight)
    try:
        total = math.fsum(converted.values())
    except OverflowError:
        failures.append(f"{label}: raw weight sum overflow")
        return None
    if total <= 0.0:
        failures.append(f"{label}: positive total raw weight required")
        return None
    return converted


def _distribution_tv(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if set(left) != set(right):
        raise ValueError("support mismatch")
    return 0.5 * math.fsum(abs(left[key] - right[key]) for key in sorted(left))


def _weight_tv(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if set(left) != set(right):
        raise ValueError("support mismatch")
    left_total = math.fsum(left.values())
    right_total = math.fsum(right.values())
    return 0.5 * math.fsum(
        abs(left[key] / left_total - right[key] / right_total)
        for key in sorted(left)
    )


def _validate_rows(
    rows: Any,
    *,
    roots_by_id: Mapping[str, Mapping[str, Any]],
    evaluation_seeds: Sequence[int],
    solver_manifest_sha256: Any,
    range_builder_source_sha256: Any,
    failures: list[str],
) -> list[dict[str, Any]]:
    valid_rows: list[dict[str, Any]] = []
    if not isinstance(rows, list) or not rows:
        failures.append("evidence.raw_iteration_rows: non-empty list required")
        return valid_rows
    seen: set[tuple[int, int, str]] = set()
    for index, row in enumerate(rows):
        label = f"evidence.raw_iteration_rows[{index}]"
        if not isinstance(row, Mapping):
            failures.append(f"{label}: object required")
            continue
        _exact_fields(row, ROW_KEYS, label, failures)
        round_index = row.get("round_index")
        seed = row.get("solver_seed")
        root_id = row.get("root_id")
        if not _is_int(round_index) or round_index < 1:
            failures.append(f"{label}.round_index: positive integer required")
        if not _is_int(seed):
            failures.append(f"{label}.solver_seed: integer required")
        elif seed not in evaluation_seeds:
            failures.append(f"{label}.solver_seed: not in locked evaluation seeds")
        if not isinstance(root_id, str) or root_id not in roots_by_id:
            failures.append(f"{label}.root_id: not in locked root manifest")
            root = None
        else:
            root = roots_by_id[root_id]
            for field in (
                "root_commitment_sha256",
                "root_identity_commitment_sha256",
                "stratum",
                "actor",
                "visible_joker_count",
            ):
                if row.get(field) != root.get(field):
                    failures.append(f"{label}.{field}: locked root mismatch")
        if _is_int(round_index) and _is_int(seed) and isinstance(root_id, str):
            key = (round_index, seed, root_id)
            if key in seen:
                failures.append(f"{label}: duplicate round/seed/root row")
            seen.add(key)
        for field in (
            "previous_candidate_policy_artifact_sha256",
            "previous_candidate_policy_checkpoint_sha256",
            "candidate_policy_artifact_sha256",
            "candidate_policy_checkpoint_sha256",
            "solver_manifest_sha256",
            "range_builder_source_sha256",
        ):
            if not _valid_sha256(row.get(field)):
                failures.append(f"{label}.{field}: lowercase SHA256 required")
        if row.get("solver_manifest_sha256") != solver_manifest_sha256:
            failures.append(f"{label}.solver_manifest_sha256: evidence binding mismatch")
        if row.get("range_builder_source_sha256") != range_builder_source_sha256:
            failures.append(
                f"{label}.range_builder_source_sha256: evidence binding mismatch"
            )
        if row.get("exact_exploitability_computed") is not False:
            failures.append(f"{label}.exact_exploitability_computed: must be false")
        if row.get("candidate_policy_method") != SOLVER_METHOD:
            failures.append(
                f"{label}.candidate_policy_method: endogenous solver candidate required"
            )
        previous_policy = _validate_distribution(
            row.get("previous_policy_distribution"),
            f"{label}.previous_policy_distribution",
            failures,
        )
        current_policy = _validate_distribution(
            row.get("current_policy_distribution"),
            f"{label}.current_policy_distribution",
            failures,
        )
        policy_tv: float | None = None
        if previous_policy is not None and current_policy is not None:
            try:
                policy_tv = _distribution_tv(previous_policy, current_policy)
            except ValueError:
                failures.append(f"{label}: policy action support mismatch")
        actor = row.get("actor")
        previous_weights: dict[str, float] | None = None
        current_weights: dict[str, float] | None = None
        posterior_tv: float | None = None
        if actor == "btn":
            previous_weights = _validate_weights(
                row.get("previous_btn_posterior_weights"),
                f"{label}.previous_btn_posterior_weights",
                failures,
            )
            current_weights = _validate_weights(
                row.get("current_btn_posterior_weights"),
                f"{label}.current_btn_posterior_weights",
                failures,
            )
            if previous_weights is not None and current_weights is not None:
                try:
                    posterior_tv = _weight_tv(previous_weights, current_weights)
                except ValueError:
                    failures.append(f"{label}: BTN posterior support mismatch")
        else:
            if row.get("previous_btn_posterior_weights") is not None:
                failures.append(
                    f"{label}.previous_btn_posterior_weights: must be null for BB"
                )
            if row.get("current_btn_posterior_weights") is not None:
                failures.append(
                    f"{label}.current_btn_posterior_weights: must be null for BB"
                )
        valid_rows.append(
            {
                "raw": row,
                "round_index": round_index,
                "solver_seed": seed,
                "root_id": root_id,
                "policy_tv": policy_tv,
                "posterior_tv": posterior_tv,
                "previous_policy": previous_policy,
                "current_policy": current_policy,
                "previous_weights": previous_weights,
                "current_weights": current_weights,
            }
        )
    return valid_rows


def _derive_metrics_from_valid_rows(
    valid_rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    rows_by_round: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        if _is_int(row.get("round_index")):
            rows_by_round[int(row["round_index"])].append(row)
    per_round: dict[str, Any] = {}
    converged_flags: list[bool] = []
    for round_index in sorted(rows_by_round):
        rows = rows_by_round[round_index]
        policy_values = [
            float(row["policy_tv"])
            for row in rows
            if _finite(row.get("policy_tv"))
        ]
        posterior_values = [
            float(row["posterior_tv"])
            for row in rows
            if _finite(row.get("posterior_tv"))
        ]
        max_policy = max(policy_values) if policy_values else None
        max_posterior = max(posterior_values) if posterior_values else None
        converged = (
            max_policy is not None
            and max_posterior is not None
            and max_policy <= float(thresholds["max_policy_tv"])
            and max_posterior
            <= float(thresholds["max_btn_posterior_weight_tv"])
        )
        converged_flags.append(converged)
        per_round[str(round_index)] = {
            "row_count": len(rows),
            "policy_comparison_count": len(policy_values),
            "btn_posterior_comparison_count": len(posterior_values),
            "max_policy_tv": max_policy,
            "mean_policy_tv": (
                math.fsum(policy_values) / len(policy_values)
                if policy_values
                else None
            ),
            "max_btn_posterior_weight_tv": max_posterior,
            "mean_btn_posterior_weight_tv": (
                math.fsum(posterior_values) / len(posterior_values)
                if posterior_values
                else None
            ),
            "converged": converged,
        }
    trailing = 0
    for converged in reversed(converged_flags):
        if not converged:
            break
        trailing += 1
    final_round = max(rows_by_round) if rows_by_round else None
    final_rows = rows_by_round.get(final_round, []) if final_round is not None else []
    final_artifacts = sorted(
        {
            row["raw"].get("candidate_policy_artifact_sha256")
            for row in final_rows
            if _valid_sha256(row["raw"].get("candidate_policy_artifact_sha256"))
        }
    )
    final_checkpoints = sorted(
        {
            row["raw"].get("candidate_policy_checkpoint_sha256")
            for row in final_rows
            if _valid_sha256(row["raw"].get("candidate_policy_checkpoint_sha256"))
        }
    )
    return {
        "round_indices": sorted(rows_by_round),
        "round_count": len(rows_by_round),
        "row_count": len(valid_rows),
        "independent_seeds": sorted(
            {
                int(row["solver_seed"])
                for row in valid_rows
                if _is_int(row.get("solver_seed"))
            }
        ),
        "per_round": per_round,
        "final_consecutive_converged_rounds": trailing,
        "final_round_index": final_round,
        "final_candidate_policy_artifact_sha256": (
            final_artifacts[0] if len(final_artifacts) == 1 else None
        ),
        "final_candidate_policy_checkpoint_sha256": (
            final_checkpoints[0] if len(final_checkpoints) == 1 else None
        ),
        "exact_exploitability_computed": False,
    }


def derive_t3_bb_fixed_point_metrics(
    evidence: Mapping[str, Any], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-derive the publishable summary from otherwise-valid raw rows.

    This helper is intended for artifact producers.  Full acceptance still
    requires :func:`validate_t3_bb_fixed_point_evidence`.
    """

    config_failures = _validate_config(config)
    if config_failures:
        raise ValueError("; ".join(config_failures))
    failures: list[str] = []
    root_state = _validate_root_manifest(evidence.get("root_manifest"), failures)
    valid_rows = _validate_rows(
        evidence.get("raw_iteration_rows"),
        roots_by_id=root_state["roots_by_id"],
        evaluation_seeds=root_state["seeds"],
        solver_manifest_sha256=evidence.get("solver_manifest_sha256"),
        range_builder_source_sha256=evidence.get("range_builder_source_sha256"),
        failures=failures,
    )
    if failures:
        raise ValueError("; ".join(failures))
    return _derive_metrics_from_valid_rows(valid_rows, config["thresholds"])


def _result(
    *,
    evidence: Any,
    config: Any,
    failures: Sequence[str],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    passed = not failures
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "passed": passed,
        "promotion_eligible": passed,
        "exact_exploitability_computed": False,
        "gate_config_sha256": (
            _hash_claim_or_none(config.get("gate_config_sha256"))
            if isinstance(config, Mapping)
            else None
        ),
        "evidence_sha256": (
            _hash_claim_or_none(evidence.get("artifact_sha256"))
            if isinstance(evidence, Mapping)
            else None
        ),
        "solver_manifest_sha256": (
            _hash_claim_or_none(evidence.get("solver_manifest_sha256"))
            if isinstance(evidence, Mapping)
            else None
        ),
        "range_builder_source_sha256": (
            _hash_claim_or_none(evidence.get("range_builder_source_sha256"))
            if isinstance(evidence, Mapping)
            else None
        ),
        "candidate_policy_artifact_sha256": metrics.get(
            "final_candidate_policy_artifact_sha256"
        ),
        "candidate_policy_checkpoint_sha256": metrics.get(
            "final_candidate_policy_checkpoint_sha256"
        ),
        "derived_metrics": dict(metrics),
        "failures": list(failures),
    }
    result["gate_result_sha256"] = self_hash(result, "gate_result_sha256")
    return result


def validate_t3_bb_fixed_point_evidence(
    evidence: Any, *, config: Any
) -> dict[str, Any]:
    """Validate raw fixed-point evidence and return a self-hashed gate result."""

    failures = _validate_config(config)
    metrics: dict[str, Any] = {
        "round_indices": [],
        "round_count": 0,
        "row_count": 0,
        "independent_seeds": [],
        "per_round": {},
        "final_consecutive_converged_rounds": 0,
        "final_round_index": None,
        "final_candidate_policy_artifact_sha256": None,
        "final_candidate_policy_checkpoint_sha256": None,
        "exact_exploitability_computed": False,
    }
    if not isinstance(evidence, Mapping):
        failures.append("evidence: object required")
        return _result(evidence=evidence, config=config, failures=failures, metrics=metrics)
    _exact_fields(evidence, EVIDENCE_KEYS, "evidence", failures)
    _check_self_hash(evidence, "artifact_sha256", "evidence", failures)
    expected = {
        "schema": EVIDENCE_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "evidence_kind": EVIDENCE_KIND,
        "exact_exploitability_computed": False,
        "candidate_policy_method": SOLVER_METHOD,
    }
    for field, wanted in expected.items():
        if evidence.get(field) != wanted:
            failures.append(f"evidence.{field}: must equal {wanted!r}")
    if isinstance(config, Mapping):
        if evidence.get("gate_config_sha256") != config.get("gate_config_sha256"):
            failures.append("evidence.gate_config_sha256: locked config mismatch")
        for evidence_field, config_field in (
            ("solver_manifest_sha256", "approved_solver_manifest_sha256"),
            (
                "range_builder_source_sha256",
                "approved_range_builder_source_sha256",
            ),
        ):
            if evidence.get(evidence_field) != config.get(config_field):
                failures.append(f"evidence.{evidence_field}: locked config mismatch")
    for field in ("solver_manifest_sha256", "range_builder_source_sha256"):
        if not _valid_sha256(evidence.get(field)):
            failures.append(f"evidence.{field}: lowercase SHA256 required")

    root_state = _validate_root_manifest(evidence.get("root_manifest"), failures)
    if isinstance(config, Mapping):
        manifest = evidence.get("root_manifest")
        claimed = manifest.get("manifest_sha256") if isinstance(manifest, Mapping) else None
        if claimed != config.get("approved_root_manifest_sha256"):
            failures.append("evidence.root_manifest: locked config hash mismatch")
    partition_hashes = _validate_partitions(
        evidence.get("excluded_partitions"),
        holdout_identities=root_state["identities"],
        evaluation_seeds=root_state["seeds"],
        failures=failures,
    )
    if isinstance(config, Mapping):
        expected_partitions = config.get("approved_excluded_partition_sha256")
        if isinstance(expected_partitions, Mapping):
            for purpose in REQUIRED_EXCLUDED_PARTITIONS:
                if partition_hashes.get(purpose) != expected_partitions.get(purpose):
                    failures.append(
                        f"evidence.excluded_partitions.{purpose}: locked config hash mismatch"
                    )

    thresholds = config.get("thresholds", {}) if isinstance(config, Mapping) else {}
    if isinstance(thresholds, Mapping):
        min_seeds = thresholds.get("min_independent_seeds")
        if _is_int(min_seeds) and len(root_state["seeds"]) < min_seeds:
            failures.append(
                "evidence.root_manifest.evaluation_seeds: insufficient independent seeds"
            )
        min_roots = thresholds.get("min_roots_per_stratum")
        if _is_int(min_roots):
            for stratum in REQUIRED_STRATA:
                if root_state["stratum_counts"].get(stratum, 0) < min_roots:
                    failures.append(
                        f"evidence.root_manifest: {stratum} needs at least {min_roots} roots"
                    )

    valid_rows = _validate_rows(
        evidence.get("raw_iteration_rows"),
        roots_by_id=root_state["roots_by_id"],
        evaluation_seeds=root_state["seeds"],
        solver_manifest_sha256=evidence.get("solver_manifest_sha256"),
        range_builder_source_sha256=evidence.get("range_builder_source_sha256"),
        failures=failures,
    )
    if isinstance(thresholds, Mapping) and set(thresholds) == THRESHOLD_KEYS:
        try:
            metrics = _derive_metrics_from_valid_rows(valid_rows, thresholds)
        except (TypeError, ValueError, OverflowError) as exc:
            failures.append(f"evidence.raw_iteration_rows: metric derivation failed: {exc}")
    round_indices = metrics["round_indices"]
    if round_indices and round_indices != list(range(1, max(round_indices) + 1)):
        failures.append("evidence.raw_iteration_rows: round indices must be consecutive from 1")

    # Every round must contain exactly one row for every locked root/seed pair.
    expected_pairs = {
        (seed, root_id)
        for seed in root_state["seeds"]
        for root_id in root_state["roots_by_id"]
    }
    rows_by_round: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        if _is_int(row.get("round_index")):
            rows_by_round[int(row["round_index"])].append(row)
    for round_index in round_indices:
        actual = {
            (row.get("solver_seed"), row.get("root_id"))
            for row in rows_by_round[round_index]
            if _is_int(row.get("solver_seed"))
            and isinstance(row.get("root_id"), str)
        }
        if actual != expected_pairs or len(rows_by_round[round_index]) != len(expected_pairs):
            failures.append(
                f"evidence.raw_iteration_rows: round {round_index} lacks exact root/seed coverage"
            )

    # Candidate/source bindings are round-wide and form an unbroken chain.
    round_binding: dict[int, tuple[Any, Any, Any, Any]] = {}
    by_key: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for round_index in round_indices:
        bindings = {
            (
                _hash_claim_or_none(
                    row["raw"].get("previous_candidate_policy_artifact_sha256")
                ),
                _hash_claim_or_none(
                    row["raw"].get("previous_candidate_policy_checkpoint_sha256")
                ),
                _hash_claim_or_none(
                    row["raw"].get("candidate_policy_artifact_sha256")
                ),
                _hash_claim_or_none(
                    row["raw"].get("candidate_policy_checkpoint_sha256")
                ),
            )
            for row in rows_by_round[round_index]
        }
        if len(bindings) != 1:
            failures.append(
                f"evidence.raw_iteration_rows: round {round_index} candidate bindings drift"
            )
        else:
            round_binding[round_index] = next(iter(bindings))
        for row in rows_by_round[round_index]:
            if _is_int(row.get("solver_seed")) and isinstance(
                row.get("root_id"), str
            ):
                by_key[(round_index, row["solver_seed"], row["root_id"])] = row
    for round_index in round_indices[1:]:
        previous = round_binding.get(round_index - 1)
        current = round_binding.get(round_index)
        if (
            previous is not None
            and current is not None
            and current[:2] != previous[2:]
        ):
            failures.append(
                f"evidence.raw_iteration_rows: round {round_index} candidate artifact chain mismatch"
            )
        for seed, root_id in expected_pairs:
            prior_row = by_key.get((round_index - 1, seed, root_id))
            row = by_key.get((round_index, seed, root_id))
            if prior_row is None or row is None:
                continue
            if row["previous_policy"] != prior_row["current_policy"]:
                failures.append(
                    "evidence.raw_iteration_rows: "
                    f"round {round_index} policy chain mismatch "
                    f"for seed={seed}, root={root_id}"
                )
            if (
                row["raw"].get("actor") == "btn"
                and row["previous_weights"] != prior_row["current_weights"]
            ):
                failures.append(
                    "evidence.raw_iteration_rows: "
                    f"round {round_index} posterior chain mismatch "
                    f"for seed={seed}, root={root_id}"
                )

    if isinstance(thresholds, Mapping):
        required_rounds = thresholds.get("min_consecutive_converged_rounds")
        if _is_int(required_rounds) and metrics[
            "final_consecutive_converged_rounds"
        ] < required_rounds:
            failures.append(
                "evidence.raw_iteration_rows: insufficient consecutive converged rounds"
            )
    if metrics["final_candidate_policy_artifact_sha256"] is None:
        failures.append("evidence.raw_iteration_rows: final candidate policy artifact not unique")
    if metrics["final_candidate_policy_checkpoint_sha256"] is None:
        failures.append("evidence.raw_iteration_rows: final candidate checkpoint not unique")

    if evidence.get("published_summary") != metrics:
        failures.append("evidence.published_summary: raw-derived mismatch")
    return _result(
        evidence=evidence,
        config=config,
        failures=failures,
        metrics=metrics,
    )


def verify_t3_bb_fixed_point_gate_result(
    evidence: Any, *, config: Any, gate_result: Any
) -> dict[str, Any]:
    """Re-run the gate and require an exact, self-hashed result artifact."""

    if not isinstance(gate_result, Mapping):
        raise ValueError("gate_result: object required")
    if gate_result.get("gate_result_sha256") != self_hash(
        gate_result, "gate_result_sha256"
    ):
        raise ValueError("gate_result.gate_result_sha256: self-hash mismatch")
    expected = validate_t3_bb_fixed_point_evidence(evidence, config=config)
    if dict(gate_result) != expected:
        raise ValueError("gate_result: does not match raw evidence and locked config")
    return dict(gate_result)


def build_t3_bb_likelihood_binding(
    evidence: Any, *, config: Any, gate_result: Any
) -> dict[str, Any]:
    """Emit the M3 likelihood binding only from a reverified passing gate."""

    verified = verify_t3_bb_fixed_point_gate_result(
        evidence, config=config, gate_result=gate_result
    )
    if verified.get("passed") is not True or verified.get("promotion_eligible") is not True:
        raise ValueError("t3_bb fixed-point gate did not pass")
    if verified.get("exact_exploitability_computed") is not False:
        raise ValueError("exact exploitability must remain false")
    binding: dict[str, Any] = {
        "schema": T3_BB_LIKELIHOOD_SCHEMA,
        "route": "t3_bb",
        "consumer_root_actor": "btn",
        "consumer_phase": "t3_second",
        "method": T3_BB_LIKELIHOOD_METHOD,
        "promotion_eligible": True,
        "fixed_point_converged": True,
        "candidate_policy_artifact_sha256": verified[
            "candidate_policy_artifact_sha256"
        ],
        "candidate_policy_checkpoint_sha256": verified[
            "candidate_policy_checkpoint_sha256"
        ],
        "fixed_point_evidence_sha256": verified["evidence_sha256"],
        "fixed_point_gate_result_sha256": verified["gate_result_sha256"],
        "fixed_point_config_sha256": verified["gate_config_sha256"],
        "range_builder_source_sha256": verified["range_builder_source_sha256"],
        "solver_manifest_sha256": verified["solver_manifest_sha256"],
    }
    binding["binding_sha256"] = self_hash(binding, "binding_sha256")
    return verify_t3_bb_likelihood_binding(
        binding,
        solver_manifest_sha256=verified["solver_manifest_sha256"],
        range_builder_source_sha256=verified["range_builder_source_sha256"],
    )


__all__ = [
    "CONFIG_SCHEMA",
    "EVIDENCE_KIND",
    "EVIDENCE_SCHEMA",
    "GATE_ID",
    "PARTITION_SCHEMA",
    "RESULT_SCHEMA",
    "ROOT_MANIFEST_SCHEMA",
    "SCOPE",
    "build_locked_t3_bb_fixed_point_gate_config",
    "build_t3_bb_likelihood_binding",
    "canonical_json",
    "canonical_sha256",
    "derive_t3_bb_fixed_point_metrics",
    "root_commitment_sha256",
    "self_hash",
    "validate_t3_bb_fixed_point_evidence",
    "verify_locked_t3_bb_fixed_point_gate_config",
    "verify_t3_bb_fixed_point_gate_result",
]
