from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from ai.tutor.promotion_gate_m3_range import (
    CONFIG_SCHEMA,
    EVIDENCE_SCHEMA,
    GATE_ID,
    REQUIRED_STRATA,
    canonical_sha256,
    validate_full_card_range_metadata,
    validate_promotion_evidence_m3_range,
)


MODEL_MANIFEST = {
    "schema": "ofc_frozen_behavior_model/v1",
    "model_id": "t1_t2_policy_behavior_v1",
    "model_type": "torch_policy_behavior_v1",
    "position_contract_version": "bb_first_v1",
    "checkpoint_sha256": "c" * 64,
    "supported_turns": [1, 2],
    "supported_actors": ["bb", "btn"],
    "probability_transform": "masked_softmax_q32_v1",
    "promotion_eligible": True,
}
MODEL_SHA256 = canonical_sha256(MODEL_MANIFEST)
CONFIG = {
    "schema": CONFIG_SCHEMA,
    "gate_id": GATE_ID,
    "approved_behavior_model_sha256": MODEL_SHA256,
    "approved_behavior_model_type": "torch_policy_behavior_v1",
    "approved_distribution_sources": ["model"],
}


def _range_metadata(
    actor: str,
    joker_count: int,
    *,
    source: str = "model",
    used_fallback: bool = False,
    evaluation_count: int = 3,
    validation_failures: int = 0,
    model_manifest: dict | None = None,
) -> dict:
    behavior_model_manifest = copy.deepcopy(model_manifest or MODEL_MANIFEST)
    model_sha256 = canonical_sha256(behavior_model_manifest)
    observation_digest = canonical_sha256(
        {"actor": actor, "visible_joker_count": joker_count, "fixture": "m3"}
    )
    content_manifest = {
        "schema": "ofc_full_card_range_content/v1",
        "range_model": "history_weighted_full_card_discard_particles_v1",
        "position_contract_version": "bb_first_v1",
        "deck_size": 54,
        "physical_joker_ids": ["X1", "X2"],
        "observation_digest": observation_digest,
        "actor": actor,
        "visible_joker_count": joker_count,
        "turn": 3,
        "phase": "t3_first" if actor == "bb" else "t3_second",
        "behavior_model_id": behavior_model_manifest["model_id"],
        "behavior_model_sha256": model_sha256,
        "epsilon": "1/1000",
        "particle_count": 2,
        "expected_undealt_card_count": 29 if actor == "bb" else 26,
        "posterior_probability_mass_exact": "1/1",
        "particles": [
            {"commitment": "a" * 64, "weight": "1/2"},
            {"commitment": "b" * 64, "weight": "1/2"},
        ],
    }
    range_content_sha256 = canonical_sha256(content_manifest)
    fallback_count = evaluation_count if used_fallback else 0
    hit_rate = Fraction(evaluation_count - fallback_count, evaluation_count)
    query_digest = canonical_sha256(
        {"observation": observation_digest, "query": 0}
    )
    audit = [
        {
            "information_digest": query_digest,
            "distribution_sha256": canonical_sha256(
                {"query": query_digest, "source": source}
            ),
            "source": source,
            "used_fallback": used_fallback,
            "evaluation_count": evaluation_count,
        }
    ]
    build_manifest = {
        "schema": "ofc_full_card_range_build/v1",
        "range_model": content_manifest["range_model"],
        "range_content_sha256": range_content_sha256,
        "sampler": "deterministic_uniform_without_replacement_hash_priority_v1",
        "seed": 17,
        "behavior_query_count": evaluation_count,
        "behavior_unique_query_count": 1,
        "behavior_model_evaluation_count": 1,
        "behavior_distribution_source_counts": {source: evaluation_count},
        "behavior_uniform_fallback_count": fallback_count,
        "behavior_uniform_fallback_unique_count": int(used_fallback),
        "behavior_model_hit_rate_exact": (
            f"{hit_rate.numerator}/{hit_rate.denominator}"
        ),
        "behavior_model_hit_rate": float(hit_rate),
        "behavior_distribution_validation_failures": validation_failures,
        "behavior_query_audit": audit,
    }
    range_build_sha256 = canonical_sha256(build_manifest)
    return {
        "behavior_model_id": behavior_model_manifest["model_id"],
        "behavior_model_sha256": model_sha256,
        "range_sha256": range_content_sha256,
        "range_content_sha256": range_content_sha256,
        "range_build_sha256": range_build_sha256,
        "content_manifest": content_manifest,
        "build_manifest": build_manifest,
        "behavior_model_manifest": behavior_model_manifest,
    }


def _evidence() -> dict:
    strata = {}
    for name in REQUIRED_STRATA:
        actor, joker_text = name.split("_joker")
        joker_count = int(joker_text)
        strata[name] = {
            "actor": actor,
            "visible_joker_count": joker_count,
            "metadata": _range_metadata(actor, joker_count),
        }
    return {
        "schema": EVIDENCE_SCHEMA,
        "gate_id": GATE_ID,
        "strata": strata,
    }


def _rehash_build(metadata: dict) -> None:
    metadata["range_build_sha256"] = canonical_sha256(
        metadata["build_manifest"]
    )


def _rehash_content_and_build(metadata: dict) -> None:
    content_sha256 = canonical_sha256(metadata["content_manifest"])
    metadata["range_sha256"] = content_sha256
    metadata["range_content_sha256"] = content_sha256
    metadata["build_manifest"]["range_content_sha256"] = content_sha256
    _rehash_build(metadata)


def _failure_text(result: dict) -> str:
    return "\n".join(result.get("failures", []))


def test_valid_six_strata_evidence_passes_and_result_is_content_addressed():
    result = validate_promotion_evidence_m3_range(_evidence(), config=CONFIG)

    assert result["passed"] is True
    assert result["m3_range_promoted"] is True
    assert result["status"] == "m3_range_ready"
    assert set(result["strata"]) == set(REQUIRED_STRATA)
    assert all(item["passed"] for item in result["strata"].values())
    assert all(
        item["metrics"]["behavior_model_hit_rate_exact"] == "1/1"
        for item in result["strata"].values()
    )
    claimed = result["result_sha256"]
    without_hash = dict(result)
    without_hash.pop("result_sha256")
    assert claimed == canonical_sha256(without_hash)


def test_one_metadata_mapping_can_be_validated_without_six_strata_wrapper():
    result = validate_full_card_range_metadata(
        _range_metadata("bb", 1),
        config=CONFIG,
        expected_actor="bb",
        expected_visible_joker_count=1,
    )

    assert result["passed"] is True
    assert result["metrics"]["behavior_query_count"] == 3
    assert result["metrics"]["behavior_distribution_source_counts"] == {
        "model": 3
    }


def test_artifact_wrapper_and_optional_self_hash_are_supported():
    artifact = _evidence()
    artifact["artifact_sha256"] = canonical_sha256(artifact)

    result = validate_promotion_evidence_m3_range(
        {"artifact": artifact}, config=CONFIG
    )

    assert result["passed"] is True


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_strata_set_must_be_exact(mutation):
    evidence = _evidence()
    if mutation == "missing":
        evidence["strata"].pop("btn_joker2")
    else:
        evidence["strata"]["pooled"] = copy.deepcopy(
            evidence["strata"]["bb_joker0"]
        )

    result = validate_promotion_evidence_m3_range(evidence, config=CONFIG)

    assert result["passed"] is False
    assert "exact six-stratum set required" in _failure_text(result)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("actor", "btn", "actor: expected"),
        ("visible_joker_count", 2, "visible_joker_count: expected"),
    ],
)
def test_stratum_identity_must_match_its_key(field, value, message):
    evidence = _evidence()
    evidence["strata"]["bb_joker0"][field] = value

    result = validate_promotion_evidence_m3_range(evidence, config=CONFIG)

    assert result["passed"] is False
    assert message in _failure_text(result)


def test_content_manifest_joker_count_is_independently_checked_against_stratum():
    evidence = _evidence()
    metadata = evidence["strata"]["bb_joker0"]["metadata"]
    metadata["content_manifest"]["visible_joker_count"] = 2
    _rehash_content_and_build(metadata)

    result = validate_promotion_evidence_m3_range(evidence, config=CONFIG)

    assert result["passed"] is False
    assert (
        "content_manifest.visible_joker_count: expected 0, got 2"
        in _failure_text(result)
    )


@pytest.mark.parametrize(
    ("manifest", "field", "value", "message"),
    [
        ("content_manifest", "particle_count", 999, "content hash mismatch"),
        ("build_manifest", "seed", 999, "build hash mismatch"),
        (
            "behavior_model_manifest",
            "checkpoint_sha256",
            "d" * 64,
            "model SHA is not approved",
        ),
    ],
)
def test_content_build_and_model_manifest_hashes_are_recomputed(
    manifest, field, value, message
):
    evidence = _evidence()
    evidence["strata"]["bb_joker0"]["metadata"][manifest][field] = value

    result = validate_promotion_evidence_m3_range(evidence, config=CONFIG)

    assert result["passed"] is False
    assert message in _failure_text(result)


@pytest.mark.parametrize(
    ("config_field", "value", "message"),
    [
        ("approved_behavior_model_sha256", "0" * 64, "model SHA is not approved"),
        ("approved_behavior_model_type", "other_type", "model_type: not approved"),
        ("approved_distribution_sources", ["table"], "not approved by the gate config"),
    ],
)
def test_model_hash_type_and_distribution_source_are_pinned_by_config(
    config_field, value, message
):
    config = copy.deepcopy(CONFIG)
    config[config_field] = value

    result = validate_promotion_evidence_m3_range(_evidence(), config=config)

    assert result["passed"] is False
    assert message in _failure_text(result)


def test_uniform_fallback_fails_even_when_published_aggregates_are_consistent():
    evidence = _evidence()
    for name, item in evidence["strata"].items():
        actor, joker_text = name.split("_joker")
        item["metadata"] = _range_metadata(
            actor,
            int(joker_text),
            source="uniform_fallback",
            used_fallback=True,
        )

    result = validate_promotion_evidence_m3_range(evidence, config=CONFIG)

    assert result["passed"] is False
    text = _failure_text(result)
    assert "behavior fallback count must be zero" in text
    assert "not approved by the gate config" in text


def test_uniform_model_is_forbidden_even_though_it_reports_no_fallback():
    evidence = _evidence()
    evidence["strata"]["bb_joker0"]["metadata"] = _range_metadata(
        "bb", 0, source="uniform_model", used_fallback=False
    )

    result = validate_promotion_evidence_m3_range(evidence, config=CONFIG)

    assert result["passed"] is False
    assert "uniform_model is forbidden" in _failure_text(result)


def test_explicitly_non_promoted_legacy_model_cannot_be_pinned_into_promotion():
    legacy_manifest = copy.deepcopy(MODEL_MANIFEST)
    legacy_manifest["promotion_eligible"] = False
    metadata = _range_metadata("bb", 0, model_manifest=legacy_manifest)
    config = copy.deepcopy(CONFIG)
    config["approved_behavior_model_sha256"] = canonical_sha256(legacy_manifest)

    result = validate_full_card_range_metadata(
        metadata,
        config=config,
        expected_actor="bb",
    )

    assert result["passed"] is False
    assert "promotion_eligible: must be true" in _failure_text(result)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("behavior_query_count", 4, "behavior_query_count"),
        ("behavior_unique_query_count", 2, "behavior_unique_query_count"),
        ("behavior_model_evaluation_count", 2, "behavior_model_evaluation_count"),
        (
            "behavior_distribution_source_counts",
            {"model": 2},
            "not equal to raw-derived counts",
        ),
        ("behavior_model_hit_rate_exact", "1/2", "must equal '1/1'"),
        ("behavior_model_hit_rate", 0.5, "not equal to raw-derived hit rate"),
        (
            "behavior_distribution_validation_failures",
            1,
            "behavior_distribution_validation_failures: must equal 0",
        ),
    ],
)
def test_published_behavior_aggregates_cannot_override_raw_audit(
    field, value, message
):
    evidence = _evidence()
    metadata = evidence["strata"]["bb_joker0"]["metadata"]
    metadata["build_manifest"][field] = value
    _rehash_build(metadata)

    result = validate_promotion_evidence_m3_range(evidence, config=CONFIG)

    assert result["passed"] is False
    assert message in _failure_text(result)


def test_duplicate_query_digest_is_rejected_even_with_rehashed_build_manifest():
    evidence = _evidence()
    metadata = evidence["strata"]["bb_joker0"]["metadata"]
    build = metadata["build_manifest"]
    build["behavior_query_audit"].append(
        copy.deepcopy(build["behavior_query_audit"][0])
    )
    build["behavior_query_count"] = 6
    build["behavior_unique_query_count"] = 2
    build["behavior_model_evaluation_count"] = 2
    build["behavior_distribution_source_counts"] = {"model": 6}
    _rehash_build(metadata)

    result = validate_promotion_evidence_m3_range(evidence, config=CONFIG)

    assert result["passed"] is False
    assert "duplicate query" in _failure_text(result)


def test_reusing_one_range_content_for_multiple_strata_is_rejected_as_pooling():
    evidence = _evidence()
    evidence["strata"]["bb_joker1"]["metadata"] = copy.deepcopy(
        evidence["strata"]["bb_joker0"]["metadata"]
    )

    result = validate_promotion_evidence_m3_range(evidence, config=CONFIG)

    assert result["passed"] is False
    assert "pooled evidence is forbidden" in _failure_text(result)


def test_bad_optional_artifact_hash_fails_closed():
    artifact = _evidence()
    artifact["artifact_sha256"] = "0" * 64

    result = validate_promotion_evidence_m3_range(
        {"artifact": artifact}, config=CONFIG
    )

    assert result["passed"] is False
    assert "canonical self-hash mismatch" in _failure_text(result)


def test_non_finite_evidence_and_uniform_approvals_fail_without_raising():
    evidence = _evidence()
    evidence["not_json"] = float("nan")
    uniform_config = copy.deepcopy(CONFIG)
    uniform_config["approved_behavior_model_type"] = "uniform_legal"
    uniform_config["approved_distribution_sources"] = ["uniform_model"]

    result = validate_promotion_evidence_m3_range(
        evidence, config=uniform_config
    )

    assert result["passed"] is False
    text = _failure_text(result)
    assert "finite canonical JSON" in text
    assert "uniform_legal is forbidden" in text
    assert "unsupported or uniform sources" in text


def test_non_finite_self_hashed_artifact_fails_closed_without_raising():
    artifact = _evidence()
    artifact["artifact_sha256"] = "0" * 64
    artifact["not_json"] = float("nan")

    result = validate_promotion_evidence_m3_range(
        {"artifact": artifact}, config=CONFIG
    )

    assert result["passed"] is False
    text = _failure_text(result)
    assert "finite canonical JSON" in text
    assert "canonical self-hash mismatch" in text


def test_non_string_stratum_key_fails_closed_without_sorting_error():
    evidence = _evidence()
    evidence["strata"][7] = evidence["strata"].pop("btn_joker2")

    result = validate_promotion_evidence_m3_range(evidence, config=CONFIG)

    assert result["passed"] is False
    assert "all keys must be strings" in _failure_text(result)


def test_boolean_source_count_cannot_equal_integer_count_by_python_coercion():
    evidence = _evidence()
    metadata = evidence["strata"]["bb_joker0"]["metadata"]
    metadata["build_manifest"]["behavior_distribution_source_counts"] = {
        "model": True
    }
    _rehash_build(metadata)

    result = validate_promotion_evidence_m3_range(evidence, config=CONFIG)

    assert result["passed"] is False
    assert "not equal to raw-derived counts" in _failure_text(result)
