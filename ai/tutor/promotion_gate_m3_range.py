"""Fail-closed promotion verifier for M3 full-card range evidence.

The verifier is deliberately pure: callers provide JSON-compatible evidence
and a JSON-compatible gate configuration, and receive a JSON-compatible result.
It does not read files, mutate artifacts, or reuse the locked M2 gate.

Every thresholded behavior-policy metric is re-derived from the embedded
``behavior_query_audit`` rows.  Published aggregate counters are accepted only
when they exactly match those raw rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from fractions import Fraction
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "ofc_m3_range_gate_config/v1"
EVIDENCE_SCHEMA = "ofc_m3_range_promotion_evidence/v1"
RESULT_SCHEMA = "ofc_m3_range_gate_result/v1"
GATE_ID = "promotion_gate_m3_range"
PASS_STATUS = "m3_range_ready"
FAIL_STATUS = "m3_range_blocked"

CONTENT_SCHEMA = "ofc_full_card_range_content/v1"
BUILD_SCHEMA = "ofc_full_card_range_build/v1"
MODEL_SCHEMA = "ofc_frozen_behavior_model/v1"
POSITION_CONTRACT_VERSION = "bb_first_v1"

REQUIRED_STRATA = (
    "bb_joker0",
    "bb_joker1",
    "bb_joker2",
    "btn_joker0",
    "btn_joker1",
    "btn_joker2",
)
PRODUCTION_SOURCES = frozenset({"model", "table"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value: Any) -> str:
    """Return the canonical JSON SHA-256 used by range artifacts."""

    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _config_snapshot(
    config: Any, errors: list[str]
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(config, Mapping):
        errors.append("config: must be an object")
        return None, None
    try:
        digest = canonical_sha256(config)
    except (TypeError, ValueError, OverflowError):
        errors.append("config: must be finite canonical JSON")
        return None, None

    if config.get("schema") != CONFIG_SCHEMA:
        errors.append(f"config.schema: must equal {CONFIG_SCHEMA!r}")
    if config.get("gate_id") != GATE_ID:
        errors.append(f"config.gate_id: must equal {GATE_ID!r}")

    approved_sha = config.get("approved_behavior_model_sha256")
    if not _valid_sha256(approved_sha):
        errors.append(
            "config.approved_behavior_model_sha256: must be a lowercase SHA256"
        )

    approved_type = config.get("approved_behavior_model_type")
    if not isinstance(approved_type, str) or not approved_type.strip():
        errors.append(
            "config.approved_behavior_model_type: must be a non-empty string"
        )
    elif approved_type == "uniform_legal":
        errors.append(
            "config.approved_behavior_model_type: uniform_legal is forbidden"
        )

    raw_sources = config.get("approved_distribution_sources")
    sources: tuple[str, ...] = ()
    if (
        not isinstance(raw_sources, Sequence)
        or isinstance(raw_sources, (str, bytes))
        or not raw_sources
    ):
        errors.append(
            "config.approved_distribution_sources: must be a non-empty array"
        )
    else:
        sources = tuple(str(source) for source in raw_sources)
        if len(sources) != len(set(sources)):
            errors.append(
                "config.approved_distribution_sources: entries must be unique"
            )
        unsupported = sorted(set(sources) - PRODUCTION_SOURCES)
        if unsupported:
            errors.append(
                "config.approved_distribution_sources: unsupported or uniform "
                f"sources {unsupported}"
            )

    if errors:
        return None, digest
    return (
        {
            "approved_behavior_model_sha256": approved_sha,
            "approved_behavior_model_type": approved_type,
            "approved_distribution_sources": frozenset(sources),
        },
        digest,
    )


def _range_metadata_from_stratum(
    name: str, item: Any, errors: list[str]
) -> tuple[Mapping[str, Any] | None, str | None, int | None]:
    if not isinstance(item, Mapping):
        errors.append(f"strata.{name}: must be an object")
        return None, None, None

    actor = item.get("actor")
    joker_count = item.get("visible_joker_count")
    metadata: Any
    if "metadata" in item:
        metadata = item.get("metadata")
    elif "range_metadata" in item:
        metadata = item.get("range_metadata")
    elif "content_manifest" in item and "build_manifest" in item:
        metadata = item
    else:
        errors.append(
            f"strata.{name}: requires metadata/range_metadata or direct manifests"
        )
        return None, None, None

    if not isinstance(actor, str):
        errors.append(f"strata.{name}.actor: required string")
        actor = None
    if not _is_int(joker_count):
        errors.append(f"strata.{name}.visible_joker_count: required integer")
        joker_count = None
    if not isinstance(metadata, Mapping):
        errors.append(f"strata.{name}.metadata: must be an object")
        metadata = None
    return metadata, actor, joker_count


def _require_int_equal(
    mapping: Mapping[str, Any], key: str, expected: int, label: str, errors: list[str]
) -> None:
    actual = mapping.get(key)
    if not _is_int(actual) or actual != expected:
        errors.append(f"{label}.{key}: must equal raw-derived integer {expected}")


def _validate_query_audit(
    build: Mapping[str, Any],
    *,
    approved_sources: frozenset[str],
    label: str,
    errors: list[str],
) -> dict[str, Any]:
    raw_audit = build.get("behavior_query_audit")
    if not isinstance(raw_audit, list) or not raw_audit:
        errors.append(f"{label}.behavior_query_audit: must be a non-empty array")
        raw_audit = []

    seen: set[str] = set()
    source_counts: Counter[str] = Counter()
    fallback_count = 0
    fallback_unique_count = 0
    approved_hit_count = 0

    for index, row in enumerate(raw_audit):
        row_label = f"{label}.behavior_query_audit[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{row_label}: must be an object")
            continue
        digest = row.get("information_digest")
        distribution_sha = row.get("distribution_sha256")
        source = row.get("source")
        used_fallback = row.get("used_fallback")
        evaluation_count = row.get("evaluation_count")

        if not _valid_sha256(digest):
            errors.append(f"{row_label}.information_digest: invalid SHA256")
        elif digest in seen:
            errors.append(f"{row_label}.information_digest: duplicate query")
        else:
            seen.add(digest)
        if not _valid_sha256(distribution_sha):
            errors.append(f"{row_label}.distribution_sha256: invalid SHA256")
        if not isinstance(source, str) or not source:
            errors.append(f"{row_label}.source: must be a non-empty string")
            source = ""
        if not isinstance(used_fallback, bool):
            errors.append(f"{row_label}.used_fallback: must be boolean")
            used_fallback = False
        if not _is_int(evaluation_count) or evaluation_count <= 0:
            errors.append(f"{row_label}.evaluation_count: must be positive integer")
            evaluation_count = 0

        if source == "uniform_model":
            errors.append(f"{row_label}.source: uniform_model is forbidden")
        if source not in approved_sources:
            errors.append(
                f"{row_label}.source: {source!r} is not approved by the gate config"
            )
        if used_fallback != (source == "uniform_fallback"):
            errors.append(
                f"{row_label}: used_fallback must be true exactly for uniform_fallback"
            )

        source_counts[source] += evaluation_count
        if used_fallback:
            fallback_count += evaluation_count
            fallback_unique_count += 1
        if source in approved_sources and not used_fallback:
            approved_hit_count += evaluation_count

    query_count = sum(source_counts.values())
    unique_count = len(raw_audit)
    if query_count <= 0:
        errors.append(f"{label}.behavior_query_audit: derived query count must be positive")
        hit_rate = Fraction(0, 1)
    else:
        hit_rate = Fraction(query_count - fallback_count, query_count)
    approved_hit_rate = (
        Fraction(approved_hit_count, query_count) if query_count > 0 else Fraction(0, 1)
    )

    _require_int_equal(build, "behavior_query_count", query_count, label, errors)
    _require_int_equal(build, "behavior_unique_query_count", unique_count, label, errors)
    _require_int_equal(
        build, "behavior_model_evaluation_count", unique_count, label, errors
    )
    _require_int_equal(
        build, "behavior_uniform_fallback_count", fallback_count, label, errors
    )
    _require_int_equal(
        build,
        "behavior_uniform_fallback_unique_count",
        fallback_unique_count,
        label,
        errors,
    )

    published_sources = build.get("behavior_distribution_source_counts")
    published_sources_valid = isinstance(published_sources, Mapping) and all(
        isinstance(key, str) and _is_int(value) and value >= 0
        for key, value in published_sources.items()
    )
    if not published_sources_valid or dict(published_sources) != dict(
        sorted(source_counts.items())
    ):
        errors.append(
            f"{label}.behavior_distribution_source_counts: not equal to raw-derived counts"
        )

    expected_exact = f"{hit_rate.numerator}/{hit_rate.denominator}"
    if build.get("behavior_model_hit_rate_exact") != expected_exact:
        errors.append(
            f"{label}.behavior_model_hit_rate_exact: must equal {expected_exact!r}"
        )
    published_rate = build.get("behavior_model_hit_rate")
    if not _is_finite_number(published_rate) or float(published_rate) != float(hit_rate):
        errors.append(
            f"{label}.behavior_model_hit_rate: not equal to raw-derived hit rate"
        )

    validation_failures = build.get("behavior_distribution_validation_failures")
    if not _is_int(validation_failures) or validation_failures != 0:
        errors.append(
            f"{label}.behavior_distribution_validation_failures: must equal 0"
        )
    if fallback_count != 0 or fallback_unique_count != 0:
        errors.append(f"{label}: behavior fallback count must be zero")
    if hit_rate != 1 or approved_hit_rate != 1:
        errors.append(f"{label}: behavior model hit rate must equal one")

    return {
        "behavior_query_count": query_count,
        "behavior_unique_query_count": unique_count,
        "behavior_distribution_source_counts": dict(sorted(source_counts.items())),
        "behavior_uniform_fallback_count": fallback_count,
        "behavior_uniform_fallback_unique_count": fallback_unique_count,
        "behavior_model_hit_rate_exact": expected_exact,
        "approved_source_hit_rate_exact": (
            f"{approved_hit_rate.numerator}/{approved_hit_rate.denominator}"
        ),
        "behavior_distribution_validation_failures": (
            validation_failures if _is_int(validation_failures) else None
        ),
    }


def _validate_range_metadata(
    metadata: Any,
    *,
    config: Mapping[str, Any],
    label: str,
    expected_actor: str | None,
    expected_visible_joker_count: int | None,
) -> dict[str, Any]:
    errors: list[str] = []
    metrics: dict[str, Any] = {}
    if not isinstance(metadata, Mapping):
        return {"passed": False, "failures": [f"{label}: metadata must be an object"]}

    content = metadata.get("content_manifest")
    build = metadata.get("build_manifest")
    model = metadata.get("behavior_model_manifest")
    if not isinstance(content, Mapping):
        errors.append(f"{label}.content_manifest: required object")
    if not isinstance(build, Mapping):
        errors.append(f"{label}.build_manifest: required object")
    if not isinstance(model, Mapping):
        errors.append(f"{label}.behavior_model_manifest: required object")
    if not all(isinstance(item, Mapping) for item in (content, build, model)):
        return {"passed": False, "failures": errors}

    assert isinstance(content, Mapping)
    assert isinstance(build, Mapping)
    assert isinstance(model, Mapping)
    try:
        computed_content_sha = canonical_sha256(content)
        computed_build_sha = canonical_sha256(build)
        computed_model_sha = canonical_sha256(model)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{label}: manifests must be finite canonical JSON")
        return {"passed": False, "failures": errors}

    if content.get("schema") != CONTENT_SCHEMA:
        errors.append(f"{label}.content_manifest.schema: unsupported schema")
    if build.get("schema") != BUILD_SCHEMA:
        errors.append(f"{label}.build_manifest.schema: unsupported schema")
    if model.get("schema") != MODEL_SCHEMA:
        errors.append(f"{label}.behavior_model_manifest.schema: unsupported schema")
    if content.get("position_contract_version") != POSITION_CONTRACT_VERSION:
        errors.append(f"{label}.content_manifest: requires bb_first_v1")
    if model.get("position_contract_version") != POSITION_CONTRACT_VERSION:
        errors.append(f"{label}.behavior_model_manifest: requires bb_first_v1")
    if content.get("physical_joker_ids") != ["X1", "X2"]:
        errors.append(f"{label}.content_manifest.physical_joker_ids: must be X1/X2")
    if content.get("range_model") != build.get("range_model"):
        errors.append(f"{label}: content/build range_model mismatch")

    claimed_content_sha = metadata.get("range_content_sha256")
    claimed_build_sha = metadata.get("range_build_sha256")
    claimed_range_sha = metadata.get("range_sha256")
    if claimed_content_sha != computed_content_sha:
        errors.append(f"{label}.range_content_sha256: content hash mismatch")
    if claimed_range_sha != computed_content_sha:
        errors.append(f"{label}.range_sha256: must equal content hash")
    if claimed_build_sha != computed_build_sha:
        errors.append(f"{label}.range_build_sha256: build hash mismatch")
    if build.get("range_content_sha256") != computed_content_sha:
        errors.append(f"{label}.build_manifest: not bound to content manifest")

    approved_sha = config["approved_behavior_model_sha256"]
    approved_type = config["approved_behavior_model_type"]
    if computed_model_sha != approved_sha:
        errors.append(f"{label}.behavior_model_manifest: model SHA is not approved")
    if model.get("model_type") != approved_type:
        errors.append(f"{label}.behavior_model_manifest.model_type: not approved")
    if model.get("model_type") == "uniform_legal":
        errors.append(f"{label}.behavior_model_manifest: uniform model is forbidden")
    if model.get("promotion_eligible") is not True:
        errors.append(
            f"{label}.behavior_model_manifest.promotion_eligible: must be true"
        )
    if content.get("behavior_model_sha256") != computed_model_sha:
        errors.append(f"{label}.content_manifest: behavior model hash mismatch")
    if metadata.get("behavior_model_sha256") != computed_model_sha:
        errors.append(f"{label}.behavior_model_sha256: model hash mismatch")
    model_id = model.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        errors.append(f"{label}.behavior_model_manifest.model_id: required")
    if content.get("behavior_model_id") != model_id:
        errors.append(f"{label}.content_manifest: behavior model_id mismatch")
    if metadata.get("behavior_model_id") != model_id:
        errors.append(f"{label}.behavior_model_id: model_id mismatch")

    actor = content.get("actor")
    if actor not in ("bb", "btn"):
        errors.append(f"{label}.content_manifest.actor: must be bb or btn")
    if expected_actor is not None and actor != expected_actor:
        errors.append(
            f"{label}.content_manifest.actor: expected {expected_actor!r}, got {actor!r}"
        )
    if not _valid_sha256(content.get("observation_digest")):
        errors.append(f"{label}.content_manifest.observation_digest: invalid SHA256")
    content_joker_count = content.get("visible_joker_count")
    if not _is_int(content_joker_count) or content_joker_count not in (0, 1, 2):
        errors.append(
            f"{label}.content_manifest.visible_joker_count: must be 0, 1, or 2"
        )
    if (
        expected_visible_joker_count is not None
        and content_joker_count != expected_visible_joker_count
    ):
        errors.append(
            f"{label}.content_manifest.visible_joker_count: expected "
            f"{expected_visible_joker_count}, got {content_joker_count!r}"
        )

    metrics = _validate_query_audit(
        build,
        approved_sources=config["approved_distribution_sources"],
        label=f"{label}.build_manifest",
        errors=errors,
    )
    metrics.update(
        {
            "actor": actor,
            "content_visible_joker_count": content_joker_count,
            "behavior_model_sha256": computed_model_sha,
            "behavior_model_type": model.get("model_type"),
            "range_content_sha256": computed_content_sha,
            "range_build_sha256": computed_build_sha,
        }
    )
    return {"passed": not errors, "metrics": metrics, "failures": errors}


def validate_full_card_range_metadata(
    metadata: Any,
    *,
    config: Any,
    expected_actor: str | None = None,
    expected_visible_joker_count: int | None = None,
) -> dict[str, Any]:
    """Validate one FullCardRange metadata mapping without requiring six strata."""

    errors: list[str] = []
    normalized, config_digest = _config_snapshot(config, errors)
    if normalized is None:
        result = {
            "schema": RESULT_SCHEMA,
            "gate_id": GATE_ID,
            "passed": False,
            "status": FAIL_STATUS,
            "gate_config_sha256": config_digest,
            "failures": errors,
        }
        result["result_sha256"] = canonical_sha256(result)
        return result
    report = _validate_range_metadata(
        metadata,
        config=normalized,
        label="range",
        expected_actor=expected_actor,
        expected_visible_joker_count=expected_visible_joker_count,
    )
    result = {
        "schema": RESULT_SCHEMA,
        "gate_id": GATE_ID,
        "passed": report["passed"],
        "status": PASS_STATUS if report["passed"] else FAIL_STATUS,
        "gate_config_sha256": config_digest,
        "metrics": report.get("metrics", {}),
        "failures": report["failures"],
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def validate_promotion_evidence_m3_range(
    evidence: Any, *, config: Any
) -> dict[str, Any]:
    """Validate six BB/BTN x visible-joker M3 range strata fail-closed."""

    failures: list[str] = []
    normalized, config_digest = _config_snapshot(config, failures)
    evidence_digest: str | None = None
    try:
        evidence_digest = canonical_sha256(evidence)
    except (TypeError, ValueError, OverflowError):
        failures.append("evidence: must be finite canonical JSON")

    if not isinstance(evidence, Mapping):
        failures.append("evidence: must be an object")
        container: Mapping[str, Any] = {}
    else:
        possible_artifact = evidence.get("artifact")
        container = possible_artifact if isinstance(possible_artifact, Mapping) else evidence

    if container.get("schema") != EVIDENCE_SCHEMA:
        failures.append(f"evidence.schema: must equal {EVIDENCE_SCHEMA!r}")
    if container.get("gate_id") != GATE_ID:
        failures.append(f"evidence.gate_id: must equal {GATE_ID!r}")

    claimed_artifact_sha = container.get("artifact_sha256")
    if claimed_artifact_sha is not None:
        artifact_without_hash = dict(container)
        artifact_without_hash.pop("artifact_sha256", None)
        try:
            computed_artifact_sha = canonical_sha256(artifact_without_hash)
        except (TypeError, ValueError, OverflowError):
            computed_artifact_sha = None
            failures.append("evidence.artifact: must be finite canonical JSON")
        if claimed_artifact_sha != computed_artifact_sha:
            failures.append("evidence.artifact_sha256: canonical self-hash mismatch")

    raw_strata = container.get("strata")
    required = set(REQUIRED_STRATA)
    if not isinstance(raw_strata, Mapping):
        failures.append("evidence.strata: required object")
        raw_strata = {}
    non_string_keys = [key for key in raw_strata if not isinstance(key, str)]
    if non_string_keys:
        failures.append("evidence.strata: all keys must be strings")
    usable_strata = {
        key: value for key, value in raw_strata.items() if isinstance(key, str)
    }
    actual = set(usable_strata)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        failures.append(
            f"evidence.strata: exact six-stratum set required; missing={missing}, extra={extra}"
        )

    stratum_results: dict[str, Any] = {}
    content_hashes: dict[str, str] = {}
    if normalized is not None:
        for name in sorted(required & actual):
            item_errors: list[str] = []
            metadata, actor, joker_count = _range_metadata_from_stratum(
                name, usable_strata[name], item_errors
            )
            expected_actor, joker_text = name.split("_joker")
            expected_joker_count = int(joker_text)
            if actor != expected_actor:
                item_errors.append(
                    f"strata.{name}.actor: expected {expected_actor!r}, got {actor!r}"
                )
            if joker_count != expected_joker_count:
                item_errors.append(
                    "strata."
                    f"{name}.visible_joker_count: expected {expected_joker_count}, "
                    f"got {joker_count!r}"
                )
            if metadata is None:
                report = {"passed": False, "metrics": {}, "failures": item_errors}
            else:
                report = _validate_range_metadata(
                    metadata,
                    config=normalized,
                    label=f"strata.{name}.metadata",
                    expected_actor=expected_actor,
                    expected_visible_joker_count=expected_joker_count,
                )
                report["failures"] = item_errors + report["failures"]
                report["passed"] = not report["failures"]
            metrics = dict(report.get("metrics", {}))
            metrics.update(
                {"actor": actor, "visible_joker_count": joker_count}
            )
            stratum_results[name] = {
                "passed": report["passed"],
                "metrics": metrics,
                "failures": report["failures"],
            }
            failures.extend(report["failures"])
            content_hash = metrics.get("range_content_sha256")
            if isinstance(content_hash, str):
                prior = content_hashes.get(content_hash)
                if prior is not None:
                    failures.append(
                        f"strata.{name}: reuses range content from {prior}; pooled evidence is forbidden"
                    )
                else:
                    content_hashes[content_hash] = name

    passed = normalized is not None and not failures
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "gate_id": GATE_ID,
        "passed": passed,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "m3_range_promoted": passed,
        "gate_config_sha256": config_digest,
        "evidence_sha256": evidence_digest,
        "required_strata": list(REQUIRED_STRATA),
        "strata": stratum_results,
        "failures": failures,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


validate_m3_range_evidence = validate_promotion_evidence_m3_range


__all__ = [
    "CONFIG_SCHEMA",
    "EVIDENCE_SCHEMA",
    "GATE_ID",
    "REQUIRED_STRATA",
    "RESULT_SCHEMA",
    "canonical_sha256",
    "validate_full_card_range_metadata",
    "validate_m3_range_evidence",
    "validate_promotion_evidence_m3_range",
]
