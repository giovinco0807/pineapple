"""Six-stratum full-card T3->T4 dynamic MCCFR smoke runner and gate.

This gate has one deliberately narrow meaning: it proves that a small
full-card dynamic solve completed for each BB/BTN x visible-Joker stratum and
that the produced manifests are internally bound.  Passing this smoke gate is
*not* an M3 policy promotion.  In particular, the current PolicyValue prior is
allowed to carry ``promotion_eligible == false`` so execution can be audited
without laundering that prior into a promoted policy.

The pure validator re-derives range/content/model hashes, role/Joker binding,
behavior-query aggregates, solver safety metadata, and every published
summary field.  The runner additionally calls the physical
``verify_full_card_range`` verifier before constructing the dynamic adapter.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.promotion_gate_m3_range import REQUIRED_STRATA
from ai.tutor.t3_hu_full_card_mccfr import (
    FullCardGenerativeAdapter,
    solve_full_card_external_sampling_mccfr,
)
from ai.tutor.t3_hu_full_card_range import (
    BUILD_SCHEMA as RANGE_BUILD_SCHEMA,
    CONTENT_SCHEMA as RANGE_CONTENT_SCHEMA,
    RANGE_MODEL,
    FullCardRange,
    verify_full_card_range,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey


EVIDENCE_SCHEMA = "ofc_m3_full_card_smoke_evidence/v1"
RESULT_SCHEMA = "ofc_m3_full_card_smoke_gate_result/v1"
RANGE_MANIFEST_SCHEMA = "ofc_m3_full_card_smoke_range_manifest/v1"
SOLVER_MANIFEST_SCHEMA = "ofc_m3_full_card_smoke_solver_manifest/v1"
GATE_ID = "promotion_gate_m3_full_card_smoke"
PASS_STATUS = "m3_full_card_smoke_ready_nonpromoted"
FAIL_STATUS = "m3_full_card_smoke_failed"
PASS_SCOPE = "six_stratum_execution_smoke_only"
BEHAVIOR_MODEL_SCHEMA = "ofc_frozen_behavior_model/v1"
SOLVER_METHOD = "full_card_dynamic_external_sampling_mccfr_plus_v1"
SOLVER_ADAPTER = "full_card_generative_t3_t4_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRODUCTION_BEHAVIOR_SOURCES = frozenset({"model", "table"})


def _json_compatible(value: Any) -> Any:
    """Recursively snapshot read-only Mapping/tuple containers as JSON data."""

    if isinstance(value, Mapping):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return the canonical finite JSON representation committed by the gate."""

    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _snapshot(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _finalize_manifest(value: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(hash_field, None)
    result[hash_field] = canonical_sha256(result)
    return result


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: Any) -> bool:
    return _is_int(value) and value > 0


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _failure_payload(exc: BaseException) -> dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def _expected_identity(name: str) -> tuple[str, int, str]:
    actor, joker_text = name.split("_joker")
    return actor, int(joker_text), "t3_first" if actor == "bb" else "t3_second"


@dataclass(frozen=True)
class FullCardSmokeStratumInput:
    """One independently built public observation and posterior range."""

    observation: InfoSetKey
    root_range: FullCardRange


def _range_manifest(
    name: str,
    observation: InfoSetKey,
    root_range: FullCardRange,
) -> dict[str, Any]:
    actor, joker_count, phase = _expected_identity(name)
    if observation.actor != actor:
        raise ValueError(
            f"{name}: observation actor {observation.actor!r} does not match {actor!r}"
        )
    if observation.turn != 3 or observation.phase != phase:
        raise ValueError(
            f"{name}: observation must be T3/{phase}, got "
            f"T{observation.turn}/{observation.phase}"
        )
    observed_jokers = sum(card in ("X1", "X2") for card in observation.current_draw)
    if observed_jokers != joker_count:
        raise ValueError(
            f"{name}: current draw has {observed_jokers} visible Jokers, "
            f"expected {joker_count}"
        )

    verification = _snapshot(verify_full_card_range(observation, root_range))
    if verification.get("verified") is not True:
        raise ValueError(f"{name}: physical range verifier did not return verified=true")
    metadata = _snapshot(root_range.metadata)
    model = metadata.get("behavior_model_manifest")
    if not isinstance(model, Mapping):
        raise ValueError(f"{name}: range metadata has no behavior model manifest")
    promotion_eligible = model.get("promotion_eligible")
    if not isinstance(promotion_eligible, bool):
        raise ValueError(f"{name}: behavior promotion_eligible must be boolean")

    manifest = {
        "schema": RANGE_MANIFEST_SCHEMA,
        "actor": actor,
        "visible_joker_count": joker_count,
        "observation_digest": observation.digest(),
        "range_sha256": root_range.range_sha256,
        "range_content_sha256": root_range.range_content_sha256,
        "range_build_sha256": root_range.range_build_sha256,
        "behavior_model_id": root_range.behavior_model_id,
        "behavior_model_sha256": root_range.behavior_model_sha256,
        "behavior_promotion_eligible": promotion_eligible,
        "metadata": metadata,
        "physical_verification": verification,
    }
    return _finalize_manifest(manifest, "range_manifest_sha256")


def _strategy_audit(result: Any, observation: InfoSetKey) -> dict[str, bool]:
    average = result.average_strategy
    current = result.current_strategy
    regrets = result.cumulative_regret_plus
    key_sets_match = set(average) == set(current) == set(regrets)
    action_sets_match = key_sets_match
    probabilities_valid = key_sets_match
    regrets_valid = key_sets_match
    if key_sets_match:
        for key in average:
            action_ids = set(average[key])
            if action_ids != set(current[key]) or action_ids != set(regrets[key]):
                action_sets_match = False
                probabilities_valid = False
                regrets_valid = False
                continue
            for distribution in (average[key], current[key]):
                values = tuple(distribution.values())
                if (
                    not values
                    or any(not _finite(value) or float(value) < 0 for value in values)
                    or not math.isclose(
                        math.fsum(float(value) for value in values),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    probabilities_valid = False
            if any(
                not _finite(value) or float(value) < 0
                for value in regrets[key].values()
            ):
                regrets_valid = False
    return {
        "average_strategy_sha256_matches_json": (
            hashlib.sha256(result.average_strategy_json.encode("utf-8")).hexdigest()
            == result.average_strategy_sha256
        ),
        "current_strategy_sha256_matches_json": (
            hashlib.sha256(result.current_strategy_json.encode("utf-8")).hexdigest()
            == result.current_strategy_sha256
        ),
        "encountered_infoset_count_matches": len(average) == result.encountered_infosets,
        "strategy_key_sets_match": key_sets_match,
        "strategy_action_sets_match": action_sets_match,
        "strategy_probabilities_valid": probabilities_valid,
        "cumulative_regret_plus_valid": regrets_valid,
        "root_information_set_encountered": observation in average,
    }


def _solver_manifest(
    name: str,
    observation: InfoSetKey,
    range_manifest: Mapping[str, Any],
    *,
    result: Any,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    actor, joker_count, _phase = _expected_identity(name)
    manifest = {
        "schema": SOLVER_MANIFEST_SCHEMA,
        "completed": True,
        "actor": actor,
        "visible_joker_count": joker_count,
        "observation_digest": observation.digest(),
        "seed": result.seed,
        "expected_seed": seed,
        "iterations": result.iterations,
        "traversals": result.traversals,
        "encountered_infosets": result.encountered_infosets,
        "average_strategy_sha256": result.average_strategy_sha256,
        "current_strategy_sha256": result.current_strategy_sha256,
        "sampling_stats": _snapshot(result.sampling_stats),
        "metadata": _snapshot(result.metadata),
        "strategy_audit": _strategy_audit(result, observation),
        "expected_iterations": iterations,
        "range_content_sha256": range_manifest["range_content_sha256"],
        "range_build_sha256": range_manifest["range_build_sha256"],
        "behavior_model_sha256": range_manifest["behavior_model_sha256"],
    }
    return _finalize_manifest(manifest, "solver_manifest_sha256")


def _failed_range_manifest(name: str, exc: BaseException) -> dict[str, Any]:
    actor, joker_count, _phase = _expected_identity(name)
    return _finalize_manifest(
        {
            "schema": RANGE_MANIFEST_SCHEMA,
            "actor": actor,
            "visible_joker_count": joker_count,
            "verified": False,
            "failure": _failure_payload(exc),
            "behavior_promotion_eligible": False,
        },
        "range_manifest_sha256",
    )


def _failed_solver_manifest(
    name: str,
    *,
    seed: int,
    iterations: int,
    exc: BaseException,
) -> dict[str, Any]:
    actor, joker_count, _phase = _expected_identity(name)
    return _finalize_manifest(
        {
            "schema": SOLVER_MANIFEST_SCHEMA,
            "completed": False,
            "actor": actor,
            "visible_joker_count": joker_count,
            "seed": seed,
            "iterations": iterations,
            "failure": _failure_payload(exc),
        },
        "solver_manifest_sha256",
    )


def run_six_strata_full_card_smoke(
    strata: Mapping[str, FullCardSmokeStratumInput],
    *,
    iterations: int = 1,
    base_seed: int = 20260713,
    max_infosets: int = 100_000,
    linear_averaging: bool = True,
) -> dict[str, Any]:
    """Execute one small dynamic solve per locked stratum and return evidence.

    Per-stratum range/solver exceptions are captured into failed manifests so a
    partial run can never be mistaken for a complete six-stratum pass.
    """

    if not isinstance(strata, Mapping):
        raise TypeError("strata must be a mapping")
    if not _positive_int(iterations):
        raise ValueError("iterations must be a positive integer")
    if not _is_int(base_seed):
        raise TypeError("base_seed must be an integer")
    if not _positive_int(max_infosets):
        raise ValueError("max_infosets must be a positive integer")
    if not isinstance(linear_averaging, bool):
        raise TypeError("linear_averaging must be boolean")

    required = set(REQUIRED_STRATA)
    actual = {key for key in strata if isinstance(key, str)}
    runner_failures: list[str] = []
    if any(not isinstance(key, str) for key in strata):
        runner_failures.append("strata keys must all be strings")
    if actual != required:
        runner_failures.append(
            "exact six-stratum input required; "
            f"missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )

    seeds = {name: base_seed + index for index, name in enumerate(REQUIRED_STRATA)}
    evidence_strata: dict[str, Any] = {}
    for name in REQUIRED_STRATA:
        seed = seeds[name]
        item = strata.get(name)
        if not isinstance(item, FullCardSmokeStratumInput):
            exc = TypeError(f"{name}: missing or invalid FullCardSmokeStratumInput")
            evidence_strata[name] = {
                "actor": _expected_identity(name)[0],
                "visible_joker_count": _expected_identity(name)[1],
                "range": _failed_range_manifest(name, exc),
                "solver": _failed_solver_manifest(
                    name, seed=seed, iterations=iterations, exc=exc
                ),
            }
            continue

        try:
            range_manifest = _range_manifest(name, item.observation, item.root_range)
        except Exception as exc:  # fail-closed evidence for a bad physical artifact
            range_manifest = _failed_range_manifest(name, exc)
            solver_manifest = _failed_solver_manifest(
                name, seed=seed, iterations=iterations, exc=exc
            )
        else:
            try:
                adapter = FullCardGenerativeAdapter(item.observation, item.root_range)
                result = solve_full_card_external_sampling_mccfr(
                    adapter,
                    iterations=iterations,
                    seed=seed,
                    max_infosets=max_infosets,
                    linear_averaging=linear_averaging,
                )
                solver_manifest = _solver_manifest(
                    name,
                    item.observation,
                    range_manifest,
                    result=result,
                    seed=seed,
                    iterations=iterations,
                )
            except Exception as exc:  # preserve evidence while preventing a false pass
                solver_manifest = _failed_solver_manifest(
                    name, seed=seed, iterations=iterations, exc=exc
                )

        actor, joker_count, _phase = _expected_identity(name)
        evidence_strata[name] = {
            "actor": actor,
            "visible_joker_count": joker_count,
            "range": range_manifest,
            "solver": solver_manifest,
        }

    completed = [
        name
        for name in REQUIRED_STRATA
        if evidence_strata[name]["solver"].get("completed") is True
    ]
    behavior_eligible = [
        name
        for name in REQUIRED_STRATA
        if evidence_strata[name]["range"].get("behavior_promotion_eligible") is True
    ]
    summary = {
        "execution_completed_strata": completed,
        "execution_smoke_passed": not runner_failures and len(completed) == 6,
        "behavior_promotion_eligible_strata": behavior_eligible,
        "behavior_prior_promotion_eligible": len(behavior_eligible) == 6,
        # This runner is intentionally incapable of authorizing promotion.
        "full_card_policy_promoted": False,
        "m3_promotion_passed": False,
    }
    artifact = {
        "schema": EVIDENCE_SCHEMA,
        "gate_id": GATE_ID,
        "pass_scope": PASS_SCOPE,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "required_strata": list(REQUIRED_STRATA),
        "run_config": {
            "iterations_per_stratum": iterations,
            "base_seed": base_seed,
            "seed_derivation": "base_seed_plus_required_stratum_index_v1",
            "seeds_by_stratum": seeds,
            "max_infosets": max_infosets,
            "linear_averaging": linear_averaging,
        },
        "runner_failures": runner_failures,
        "strata": evidence_strata,
        "summary": summary,
    }
    return _finalize_manifest(artifact, "artifact_sha256")


def _validate_self_hash(
    value: Any,
    *,
    field: str,
    label: str,
    failures: list[str],
) -> None:
    if not isinstance(value, Mapping):
        failures.append(f"{label}: must be an object")
        return
    claimed = value.get(field)
    without_hash = dict(value)
    without_hash.pop(field, None)
    try:
        computed = canonical_sha256(without_hash)
    except (TypeError, ValueError, OverflowError):
        failures.append(f"{label}: must be finite canonical JSON")
        return
    if claimed != computed:
        failures.append(f"{label}.{field}: canonical self-hash mismatch")


def _validate_behavior_audit(
    build: Mapping[str, Any], *, label: str, failures: list[str]
) -> None:
    rows = build.get("behavior_query_audit")
    if not isinstance(rows, list) or not rows:
        failures.append(f"{label}.behavior_query_audit: non-empty array required")
        return
    seen: set[str] = set()
    source_counts: Counter[str] = Counter()
    fallback_events = 0
    fallback_unique = 0
    for index, row in enumerate(rows):
        row_label = f"{label}.behavior_query_audit[{index}]"
        if not isinstance(row, Mapping):
            failures.append(f"{row_label}: must be an object")
            continue
        digest = row.get("information_digest")
        distribution_sha = row.get("distribution_sha256")
        source = row.get("source")
        used_fallback = row.get("used_fallback")
        count = row.get("evaluation_count")
        if not _valid_sha256(digest):
            failures.append(f"{row_label}.information_digest: invalid SHA256")
        elif digest in seen:
            failures.append(f"{row_label}.information_digest: duplicate")
        else:
            seen.add(digest)
        if not _valid_sha256(distribution_sha):
            failures.append(f"{row_label}.distribution_sha256: invalid SHA256")
        if source not in _PRODUCTION_BEHAVIOR_SOURCES:
            failures.append(f"{row_label}.source: requires model/table without fallback")
        if used_fallback is not False:
            failures.append(f"{row_label}.used_fallback: must be false")
        if not _positive_int(count):
            failures.append(f"{row_label}.evaluation_count: positive integer required")
            count = 0
        source_counts[str(source)] += count
        if used_fallback is True:
            fallback_events += count
            fallback_unique += 1

    query_count = sum(source_counts.values())
    expected_counts = dict(sorted(source_counts.items()))
    expected_fields = {
        "behavior_query_count": query_count,
        "behavior_unique_query_count": len(seen),
        "behavior_model_evaluation_count": len(seen),
        "behavior_distribution_source_counts": expected_counts,
        "behavior_uniform_fallback_count": fallback_events,
        "behavior_uniform_fallback_unique_count": fallback_unique,
        "behavior_model_hit_rate_exact": "1/1",
        "behavior_distribution_validation_failures": 0,
    }
    for field, expected in expected_fields.items():
        if build.get(field) != expected:
            failures.append(f"{label}.{field}: expected raw-derived {expected!r}")
    hit_rate = build.get("behavior_model_hit_rate")
    if not _finite(hit_rate) or float(hit_rate) != 1.0:
        failures.append(f"{label}.behavior_model_hit_rate: must equal 1.0")


def _validate_range_manifest(
    value: Any,
    *,
    name: str,
) -> tuple[list[str], bool, dict[str, Any]]:
    failures: list[str] = []
    actor, joker_count, phase = _expected_identity(name)
    label = f"strata.{name}.range"
    _validate_self_hash(
        value, field="range_manifest_sha256", label=label, failures=failures
    )
    if not isinstance(value, Mapping):
        return failures, False, {}
    if value.get("schema") != RANGE_MANIFEST_SCHEMA:
        failures.append(f"{label}.schema: must equal {RANGE_MANIFEST_SCHEMA!r}")
    if value.get("actor") != actor:
        failures.append(f"{label}.actor: must equal {actor!r}")
    if value.get("visible_joker_count") != joker_count:
        failures.append(f"{label}.visible_joker_count: must equal {joker_count}")

    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        failures.append(f"{label}.metadata: required object")
        return failures, False, {}
    content = metadata.get("content_manifest")
    build = metadata.get("build_manifest")
    model = metadata.get("behavior_model_manifest")
    if not all(isinstance(item, Mapping) for item in (content, build, model)):
        failures.append(f"{label}.metadata: content/build/model manifests required")
        return failures, False, {}
    assert isinstance(content, Mapping)
    assert isinstance(build, Mapping)
    assert isinstance(model, Mapping)

    try:
        content_sha = canonical_sha256(content)
        build_sha = canonical_sha256(build)
        model_sha = canonical_sha256(model)
    except (TypeError, ValueError, OverflowError):
        failures.append(f"{label}.metadata: manifests must be finite canonical JSON")
        return failures, False, {}

    expected_bindings = {
        "range_sha256": content_sha,
        "range_content_sha256": content_sha,
        "range_build_sha256": build_sha,
        "behavior_model_sha256": model_sha,
    }
    for field, expected in expected_bindings.items():
        if value.get(field) != expected:
            failures.append(f"{label}.{field}: manifest hash binding mismatch")
        if metadata.get(field) != expected:
            failures.append(f"{label}.metadata.{field}: manifest hash binding mismatch")
    if build.get("range_content_sha256") != content_sha:
        failures.append(f"{label}.build_manifest: content hash binding mismatch")

    expected_content = {
        "schema": RANGE_CONTENT_SCHEMA,
        "range_model": RANGE_MODEL,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "deck_size": 54,
        "physical_joker_ids": ["X1", "X2"],
        "actor": actor,
        "turn": 3,
        "phase": phase,
        "visible_joker_count": joker_count,
    }
    for field, expected in expected_content.items():
        if content.get(field) != expected:
            failures.append(f"{label}.content_manifest.{field}: must equal {expected!r}")
    observation_digest = content.get("observation_digest")
    if not _valid_sha256(observation_digest):
        failures.append(f"{label}.content_manifest.observation_digest: invalid SHA256")
    if value.get("observation_digest") != observation_digest:
        failures.append(f"{label}.observation_digest: content binding mismatch")
    if build.get("schema") != RANGE_BUILD_SCHEMA:
        failures.append(f"{label}.build_manifest.schema: unsupported")
    if build.get("range_model") != RANGE_MODEL:
        failures.append(f"{label}.build_manifest.range_model: unsupported")

    if model.get("schema") != BEHAVIOR_MODEL_SCHEMA:
        failures.append(f"{label}.behavior_model_manifest.schema: unsupported")
    if model.get("position_contract_version") != POSITION_CONTRACT_VERSION:
        failures.append(f"{label}.behavior_model_manifest: requires bb_first_v1")
    model_id = model.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        failures.append(f"{label}.behavior_model_manifest.model_id: required")
    for source, field, expected in (
        (content, "behavior_model_id", model_id),
        (content, "behavior_model_sha256", model_sha),
        (metadata, "behavior_model_id", model_id),
        (metadata, "behavior_model_sha256", model_sha),
        (value, "behavior_model_id", model_id),
    ):
        if source.get(field) != expected:
            failures.append(f"{label}.{field}: behavior model binding mismatch")

    promotion_eligible = model.get("promotion_eligible")
    if not isinstance(promotion_eligible, bool):
        failures.append(
            f"{label}.behavior_model_manifest.promotion_eligible: boolean required"
        )
        promotion_eligible = False
    if value.get("behavior_promotion_eligible") is not promotion_eligible:
        failures.append(f"{label}.behavior_promotion_eligible: model binding mismatch")

    _validate_behavior_audit(build, label=f"{label}.build_manifest", failures=failures)
    verification = value.get("physical_verification")
    if not isinstance(verification, Mapping) or verification.get("verified") is not True:
        failures.append(f"{label}.physical_verification.verified: must be true")
    else:
        if verification.get("range_content_sha256") != content_sha:
            failures.append(f"{label}.physical_verification: content hash mismatch")
        if verification.get("range_build_sha256") != build_sha:
            failures.append(f"{label}.physical_verification: build hash mismatch")
        if verification.get("behavior_uniform_fallback_count") != 0:
            failures.append(f"{label}.physical_verification: fallback count must be zero")
        if not _positive_int(verification.get("particle_count")):
            failures.append(f"{label}.physical_verification.particle_count: must be positive")

    metrics = {
        "range_content_sha256": content_sha,
        "range_build_sha256": build_sha,
        "behavior_model_sha256": model_sha,
        "behavior_model_id": model_id,
        "behavior_promotion_eligible": promotion_eligible,
        "observation_digest": observation_digest,
    }
    return failures, promotion_eligible, metrics


def _validate_solver_manifest(
    value: Any,
    *,
    name: str,
    run_config: Mapping[str, Any],
    range_metrics: Mapping[str, Any],
) -> tuple[list[str], bool]:
    failures: list[str] = []
    actor, joker_count, _phase = _expected_identity(name)
    label = f"strata.{name}.solver"
    _validate_self_hash(
        value, field="solver_manifest_sha256", label=label, failures=failures
    )
    if not isinstance(value, Mapping):
        return failures, False
    if value.get("schema") != SOLVER_MANIFEST_SCHEMA:
        failures.append(f"{label}.schema: must equal {SOLVER_MANIFEST_SCHEMA!r}")
    if value.get("completed") is not True:
        failures.append(f"{label}.completed: must be true")
        return failures, False
    if value.get("actor") != actor:
        failures.append(f"{label}.actor: must equal {actor!r}")
    if value.get("visible_joker_count") != joker_count:
        failures.append(f"{label}.visible_joker_count: must equal {joker_count}")
    if value.get("observation_digest") != range_metrics.get("observation_digest"):
        failures.append(f"{label}.observation_digest: range binding mismatch")

    iterations = run_config.get("iterations_per_stratum")
    seeds = run_config.get("seeds_by_stratum")
    expected_seed = seeds.get(name) if isinstance(seeds, Mapping) else None
    if value.get("seed") != expected_seed or value.get("expected_seed") != expected_seed:
        failures.append(f"{label}.seed: run-config binding mismatch")
    if value.get("iterations") != iterations or value.get("expected_iterations") != iterations:
        failures.append(f"{label}.iterations: run-config binding mismatch")
    expected_traversals = 2 * iterations if _positive_int(iterations) else None
    if value.get("traversals") != expected_traversals:
        failures.append(f"{label}.traversals: must equal 2 * iterations")
    if not _positive_int(value.get("encountered_infosets")):
        failures.append(f"{label}.encountered_infosets: must be positive")
    for field in (
        "average_strategy_sha256",
        "current_strategy_sha256",
    ):
        if not _valid_sha256(value.get(field)):
            failures.append(f"{label}.{field}: invalid SHA256")

    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        failures.append(f"{label}.metadata: required object")
        metadata = {}
    required_metadata = {
        "method": SOLVER_METHOD,
        "adapter": SOLVER_ADAPTER,
        "sampling_scheme": "external_sampling",
        "traverser_schedule": "bb_then_btn_each_iteration",
        "alternating_updates": True,
        "regret_matching_plus": True,
        "encountered_infoset_tables": True,
        "opponent_sample_cached_per_infoset": True,
        "future_chance_sampled_per_physical_state": True,
        "root_posterior_sampled_once_per_traversal": True,
        "posterior_probability_multiplied_after_sampling": False,
        "chance_probability_multiplied_after_sampling": False,
        "joint_particle_weight_used_after_sampling": False,
        "table_key_contains_particle_commitment": False,
        "table_key_contains_remaining_cards": False,
        "table_key_contains_particle_weight": False,
        "artifact_contains_raw_particle_world": False,
        "full_card": True,
        # A smoke run must preserve these non-promotion boundaries.
        "full_card_policy_promoted": False,
        "hu_exact": False,
        "runtime_integrated": False,
        "exact_exploitability_computed": False,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "linear_averaging": run_config.get("linear_averaging"),
        "max_infosets": run_config.get("max_infosets"),
    }
    for field, expected in required_metadata.items():
        if metadata.get(field) != expected:
            failures.append(f"{label}.metadata.{field}: must equal {expected!r}")
    binding_fields = (
        "range_content_sha256",
        "range_build_sha256",
        "behavior_model_sha256",
    )
    for field in binding_fields:
        expected = range_metrics.get(field)
        if value.get(field) != expected or metadata.get(field) != expected:
            failures.append(f"{label}.{field}: range/solver binding mismatch")
    expected_model_id = range_metrics.get("behavior_model_id")
    if metadata.get("behavior_model_id") != expected_model_id:
        failures.append(f"{label}.metadata.behavior_model_id: range binding mismatch")
    for field in ("average_strategy_sha256", "current_strategy_sha256"):
        if metadata.get(field) != value.get(field):
            failures.append(f"{label}.metadata.{field}: strategy hash mismatch")

    stats = value.get("sampling_stats")
    if not isinstance(stats, Mapping):
        failures.append(f"{label}.sampling_stats: required object")
        stats = {}
    if stats.get("traversals") != expected_traversals:
        failures.append(f"{label}.sampling_stats.traversals: mismatch")
    if stats.get("root_posterior_samples") != expected_traversals:
        failures.append(f"{label}.sampling_stats.root_posterior_samples: mismatch")
    if stats.get("traversals_by_actor") != {"bb": iterations, "btn": iterations}:
        failures.append(f"{label}.sampling_stats.traversals_by_actor: mismatch")
    for field in (
        "future_draw_samples",
        "decision_visits",
        "terminal_visits",
        "traverser_actions_expanded",
        "opponent_action_samples",
        "strategy_sum_updates",
        "infosets_created",
    ):
        if not _positive_int(stats.get(field)):
            failures.append(f"{label}.sampling_stats.{field}: must be positive")

    audit = value.get("strategy_audit")
    required_audits = (
        "average_strategy_sha256_matches_json",
        "current_strategy_sha256_matches_json",
        "encountered_infoset_count_matches",
        "strategy_key_sets_match",
        "strategy_action_sets_match",
        "strategy_probabilities_valid",
        "cumulative_regret_plus_valid",
        "root_information_set_encountered",
    )
    if not isinstance(audit, Mapping):
        failures.append(f"{label}.strategy_audit: required object")
    else:
        for field in required_audits:
            if audit.get(field) is not True:
                failures.append(f"{label}.strategy_audit.{field}: must be true")
    return failures, not failures


def validate_m3_full_card_smoke_evidence(evidence: Any) -> dict[str, Any]:
    """Validate a six-stratum smoke artifact without authorizing promotion."""

    failures: list[str] = []
    evidence_sha: str | None = None
    try:
        evidence_sha = canonical_sha256(evidence)
    except (TypeError, ValueError, OverflowError):
        failures.append("evidence: must be finite canonical JSON")
    if not isinstance(evidence, Mapping):
        artifact: Mapping[str, Any] = {}
        failures.append("evidence: must be an object")
    else:
        wrapped = evidence.get("artifact")
        artifact = wrapped if isinstance(wrapped, Mapping) else evidence

    _validate_self_hash(
        artifact,
        field="artifact_sha256",
        label="artifact",
        failures=failures,
    )
    expected_header = {
        "schema": EVIDENCE_SCHEMA,
        "gate_id": GATE_ID,
        "pass_scope": PASS_SCOPE,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "required_strata": list(REQUIRED_STRATA),
    }
    for field, expected in expected_header.items():
        if artifact.get(field) != expected:
            failures.append(f"artifact.{field}: must equal {expected!r}")

    run_config = artifact.get("run_config")
    if not isinstance(run_config, Mapping):
        failures.append("artifact.run_config: required object")
        run_config = {}
    iterations = run_config.get("iterations_per_stratum")
    base_seed = run_config.get("base_seed")
    max_infosets = run_config.get("max_infosets")
    linear_averaging = run_config.get("linear_averaging")
    if not _positive_int(iterations):
        failures.append("artifact.run_config.iterations_per_stratum: must be positive")
    if not _is_int(base_seed):
        failures.append("artifact.run_config.base_seed: must be integer")
    if not _positive_int(max_infosets):
        failures.append("artifact.run_config.max_infosets: must be positive")
    if not isinstance(linear_averaging, bool):
        failures.append("artifact.run_config.linear_averaging: must be boolean")
    if run_config.get("seed_derivation") != "base_seed_plus_required_stratum_index_v1":
        failures.append("artifact.run_config.seed_derivation: unsupported")
    expected_seeds = (
        {name: base_seed + index for index, name in enumerate(REQUIRED_STRATA)}
        if _is_int(base_seed)
        else {}
    )
    if run_config.get("seeds_by_stratum") != expected_seeds:
        failures.append("artifact.run_config.seeds_by_stratum: derivation mismatch")

    runner_failures = artifact.get("runner_failures")
    if not isinstance(runner_failures, list) or any(
        not isinstance(item, str) for item in (runner_failures or [])
    ):
        failures.append("artifact.runner_failures: must be an array of strings")
        runner_failures = ["invalid runner failure payload"]
    if runner_failures:
        failures.extend(f"runner: {item}" for item in runner_failures)

    raw_strata = artifact.get("strata")
    required = set(REQUIRED_STRATA)
    if not isinstance(raw_strata, Mapping):
        failures.append("artifact.strata: required object")
        raw_strata = {}
    string_strata = {
        key: value for key, value in raw_strata.items() if isinstance(key, str)
    }
    if len(string_strata) != len(raw_strata):
        failures.append("artifact.strata: all keys must be strings")
    actual = set(string_strata)
    if actual != required:
        failures.append(
            "artifact.strata: exact six-stratum set required; "
            f"missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )

    stratum_results: dict[str, Any] = {}
    completed: list[str] = []
    behavior_eligible: list[str] = []
    seen_content: dict[str, str] = {}
    seen_observations: dict[str, str] = {}
    for name in REQUIRED_STRATA:
        item = string_strata.get(name)
        item_failures: list[str] = []
        actor, joker_count, _phase = _expected_identity(name)
        if not isinstance(item, Mapping):
            item_failures.append(f"strata.{name}: required object")
            item = {}
        if item.get("actor") != actor:
            item_failures.append(f"strata.{name}.actor: must equal {actor!r}")
        if item.get("visible_joker_count") != joker_count:
            item_failures.append(
                f"strata.{name}.visible_joker_count: must equal {joker_count}"
            )
        range_failures, eligible, range_metrics = _validate_range_manifest(
            item.get("range"), name=name
        )
        item_failures.extend(range_failures)
        solver_failures, solver_completed = _validate_solver_manifest(
            item.get("solver"),
            name=name,
            run_config=run_config,
            range_metrics=range_metrics,
        )
        item_failures.extend(solver_failures)

        content_sha = range_metrics.get("range_content_sha256")
        if _valid_sha256(content_sha):
            if content_sha in seen_content:
                item_failures.append(
                    f"strata.{name}: range content reused from {seen_content[content_sha]}"
                )
            else:
                seen_content[content_sha] = name
        observation_digest = range_metrics.get("observation_digest")
        if _valid_sha256(observation_digest):
            if observation_digest in seen_observations:
                item_failures.append(
                    f"strata.{name}: observation reused from "
                    f"{seen_observations[observation_digest]}"
                )
            else:
                seen_observations[observation_digest] = name

        if eligible:
            behavior_eligible.append(name)
        if solver_completed and not item_failures:
            completed.append(name)
        failures.extend(item_failures)
        stratum_results[name] = {
            "execution_passed": solver_completed and not item_failures,
            "behavior_promotion_eligible": eligible,
            "failures": item_failures,
            "range_content_sha256": content_sha,
            "solver_manifest_sha256": (
                item.get("solver", {}).get("solver_manifest_sha256")
                if isinstance(item.get("solver"), Mapping)
                else None
            ),
        }

    structural_failures_before_summary = len(failures)
    expected_summary = {
        "execution_completed_strata": completed,
        "execution_smoke_passed": (
            not runner_failures
            and structural_failures_before_summary == 0
            and len(completed) == 6
        ),
        "behavior_promotion_eligible_strata": behavior_eligible,
        "behavior_prior_promotion_eligible": len(behavior_eligible) == 6,
        "full_card_policy_promoted": False,
        "m3_promotion_passed": False,
    }
    if artifact.get("summary") != expected_summary:
        failures.append("artifact.summary: not equal to independently derived summary")

    smoke_passed = not failures and len(completed) == 6
    behavior_prior_eligible = len(behavior_eligible) == 6
    promotion_blockers = ["smoke_only_no_strength_or_exploitability_gate"]
    if not behavior_prior_eligible:
        promotion_blockers.insert(0, "behavior_prior_not_promotion_eligible")
    # Even an eligible behavior prior would not turn this smoke into promotion.
    promotion_blockers.append("solver_contract_explicitly_nonpromoted")

    result = {
        "schema": RESULT_SCHEMA,
        "gate_id": GATE_ID,
        "passed": smoke_passed,
        "pass_scope": PASS_SCOPE,
        "status": PASS_STATUS if smoke_passed else FAIL_STATUS,
        "execution_smoke_passed": smoke_passed,
        "behavior_prior_promotion_eligible": behavior_prior_eligible,
        "m3_promotion_passed": False,
        "full_card_policy_promoted": False,
        "promotion_blockers": promotion_blockers,
        "required_strata": list(REQUIRED_STRATA),
        "evidence_sha256": evidence_sha,
        "artifact_sha256": artifact.get("artifact_sha256"),
        "strata": stratum_results,
        "failures": failures,
    }
    return _finalize_manifest(result, "result_sha256")


def write_canonical_manifest(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically write sorted finite JSON while retaining canonical hashes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def run_validate_and_write_six_strata_smoke(
    strata: Mapping[str, FullCardSmokeStratumInput],
    *,
    evidence_path: str | Path,
    result_path: str | Path,
    iterations: int = 1,
    base_seed: int = 20260713,
    max_infosets: int = 100_000,
    linear_averaging: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = run_six_strata_full_card_smoke(
        strata,
        iterations=iterations,
        base_seed=base_seed,
        max_infosets=max_infosets,
        linear_averaging=linear_averaging,
    )
    result = validate_m3_full_card_smoke_evidence(evidence)
    write_canonical_manifest(evidence_path, evidence)
    write_canonical_manifest(result_path, result)
    return evidence, result


__all__ = [
    "EVIDENCE_SCHEMA",
    "FAIL_STATUS",
    "FullCardSmokeStratumInput",
    "GATE_ID",
    "PASS_SCOPE",
    "PASS_STATUS",
    "RANGE_MANIFEST_SCHEMA",
    "REQUIRED_STRATA",
    "RESULT_SCHEMA",
    "SOLVER_MANIFEST_SCHEMA",
    "canonical_json",
    "canonical_sha256",
    "run_six_strata_full_card_smoke",
    "run_validate_and_write_six_strata_smoke",
    "validate_m3_full_card_smoke_evidence",
    "write_canonical_manifest",
]
