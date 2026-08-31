import copy
import hashlib
import json
import math
from pathlib import Path

from ai.tutor.promotion_gate_v1 import (
    EVIDENCE_SCHEMA,
    FAIL_STATUS,
    PASS_STATUS,
    PROMOTION_GATE_V1_CONFIG_SHA256,
    run_promotion_gate,
    validate_promotion_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "ai" / "config" / "promotion_gate_v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _passing_metric_value(key: str, expected):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if key.endswith("_gt"):
            return expected + 1
        return expected
    return copy.deepcopy(expected)


def _gate_metrics(config: dict, gate_name: str) -> dict:
    ignored = {
        "required",
        "basis",
        "provenance",
        "required_fields",
        "sha256_fields",
        "required_values",
        "forbidden_policy_inputs",
        "visible_joker_counts",
    }
    metrics = {
        key: _passing_metric_value(key, expected)
        for key, expected in config["gates"][gate_name].items()
        if key not in ignored and not isinstance(expected, (dict, list))
    }
    if gate_name == "information_leakage":
        metrics["forbidden_policy_inputs_checked"] = copy.deepcopy(
            config["gates"][gate_name]["forbidden_policy_inputs"]
        )
        metrics["forbidden_policy_inputs_found"] = []
    return metrics


def _artifact(config: dict) -> dict:
    digest = "a" * 64
    artifact = {
        "schema": config["schema"],
        "gate_id": config["gate_id"],
        "scope": config["scope"]["name"],
        "status": FAIL_STATUS,
        "created_at_utc": "2026-07-13T00:00:00Z",
        "position_contract_version": config["contracts"]["position_contract_version"],
        "rules_version": config["contracts"]["rules_version"],
        "fl_ev_sha256": config["contracts"]["fl_ev_sha256"],
        "action_contract_sha256": digest,
        "infoset_contract_sha256": digest,
        "reduced_fixture_sha256": digest,
        "chance_model_sha256": digest,
        "range_model": "exact_reduced_physical_particles_v1",
        "range_model_sha256": digest,
        "range_particle_count": 6,
        "range_effective_sample_size": 6.0,
        "solver_config_sha256": digest,
        "solver_code_sha256": digest,
        "source_tree_manifest_sha256": digest,
        "artifact_sha256": digest,
        "method": "synthetic_validator_test_only",
        "information_model": "public_infoset_test_only",
        "strategy_fusion": False,
        "equilibrium_approx": True,
        "reduced_tree_enumeration_exact": True,
        "hu_exact": False,
        "iterations": 20_000,
        "seeds": [1, 2, 3],
        "exploitability_method": "exact_infoset_aware_best_response",
        "runtime_environment": {
            "machine": "synthetic",
            "cpu": "synthetic",
            "python": "synthetic",
            "git_commit": "synthetic",
            "git_dirty": True,
        },
        "metrics": {"note": "synthetic validator test only"},
        "tests": ["synthetic validator test only"],
    }
    assert set(config["gates"]["artifact_metadata"]["required_fields"]) <= set(artifact)
    return artifact


def _passing_evidence() -> dict:
    config = _config()
    strata = {}
    for name in config["strata"]["required"]:
        actor, joker_text = name.split("_joker")
        strata[name] = {
            "actor": actor,
            "visible_joker_count": int(joker_text),
            "canonical_reduced_fixture_count": config["strata"][
                "minimum_canonical_reduced_fixtures_per_stratum"
            ],
            "gates": {
                gate_name: _gate_metrics(config, gate_name)
                for gate_name, gate in config["gates"].items()
                if gate["required"] is True
            },
        }
    return {
        "schema": EVIDENCE_SCHEMA,
        "gate_id": config["gate_id"],
        "gate_config_sha256": PROMOTION_GATE_V1_CONFIG_SHA256,
        "artifact": _artifact(config),
        "strata": strata,
    }


def test_complete_synthetic_evidence_passes_only_the_reduced_m2_status():
    result = validate_promotion_evidence(_passing_evidence())

    assert result["passed"] is True
    assert result["status"] == PASS_STATUS
    assert result["promotion_result"] == PASS_STATUS
    assert result["scope"] == "reduced_t3_t4_public_tree"
    assert result["full_card_policy_promoted"] is False
    assert "full_card" not in result["status"]
    assert result["failures"] == []


def test_eq_lte_and_gte_style_threshold_boundaries_are_deterministic():
    evidence = _passing_evidence()
    bb0 = evidence["strata"]["bb_joker0"]["gates"]

    # Equality is accepted on eq, lte, and min/gte-style boundaries.
    bb0["correctness"]["legal_action_rate_eq"] = 1.0
    bb0["exploitability"]["exploitability_score_lte"] = 0.5
    bb0["strategy_drift"]["independent_runs_min"] = 3
    assert validate_promotion_evidence(evidence)["passed"] is True

    # _gt is strictly greater, so its exact boundary fails.
    evidence["strata"]["bb_joker0"]["gates"]["artifact_metadata"][
        "range_effective_sample_size_gt"
    ] = 0
    result = validate_promotion_evidence(evidence)
    assert result["passed"] is False
    assert any("range_effective_sample_size_gt" in failure for failure in result["failures"])

    evidence = _passing_evidence()
    evidence["strata"]["bb_joker0"]["gates"]["correctness"][
        "legal_action_rate_eq"
    ] = 0.999
    result = validate_promotion_evidence(evidence)
    assert result["passed"] is False
    assert any("legal_action_rate_eq" in failure for failure in result["failures"])


def test_missing_stratum_fails_even_when_pooled_metrics_claim_a_pass():
    evidence = _passing_evidence()
    evidence["pooled_metrics"] = {"passed": True, "exploitability_score": 0.0}
    del evidence["strata"]["btn_joker2"]

    result = validate_promotion_evidence(evidence)

    assert result["passed"] is False
    assert result["status"] == FAIL_STATUS
    assert result["promotion_result"] is None
    assert any("btn_joker2" in failure for failure in result["failures"])


def test_pooled_only_evidence_is_rejected():
    evidence = _passing_evidence()
    del evidence["strata"]
    evidence["pooled_metrics"] = {"all_required_gates_passed": True}

    result = validate_promotion_evidence(evidence)

    assert result["passed"] is False
    assert any("pooled evidence is insufficient" in failure for failure in result["failures"])


def test_missing_metric_or_artifact_field_fails_closed():
    evidence = _passing_evidence()
    del evidence["strata"]["btn_joker1"]["gates"]["runtime"]["measured_runs_min"]
    del evidence["artifact"]["solver_code_sha256"]

    result = validate_promotion_evidence(evidence)

    assert result["passed"] is False
    assert any("measured_runs_min" in failure for failure in result["failures"])
    assert any("solver_code_sha256" in failure for failure in result["failures"])


def test_nonfinite_metric_fails_closed_and_result_remains_json_safe():
    evidence = _passing_evidence()
    evidence["strata"]["bb_joker2"]["gates"]["runtime"][
        "canonical_recursive_fixture_wall_ms_p95_lte"
    ] = math.nan

    result = validate_promotion_evidence(evidence)

    assert result["passed"] is False
    assert result["evidence_sha256"] is None
    assert any("finite" in failure for failure in result["failures"])
    json.dumps(result, allow_nan=False)


def test_sha256_shape_required_values_and_full_card_claims_are_rejected():
    evidence = _passing_evidence()
    evidence["artifact"]["action_contract_sha256"] = "not-a-sha256"
    evidence["artifact"]["hu_exact"] = True
    evidence["artifact"]["status"] = "full_card_policy_ready"

    result = validate_promotion_evidence(evidence)

    assert result["passed"] is False
    assert result["status"] == FAIL_STATUS
    assert result["full_card_policy_promoted"] is False
    assert any("action_contract_sha256" in failure for failure in result["failures"])
    assert any("hu_exact" in failure for failure in result["failures"])
    assert any("full-card claims are forbidden" in failure for failure in result["failures"])


def test_config_hash_is_byte_locked_and_evidence_must_commit_to_it():
    actual = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert actual == PROMOTION_GATE_V1_CONFIG_SHA256

    evidence = _passing_evidence()
    evidence["gate_config_sha256"] = "0" * 64
    result = validate_promotion_evidence(evidence)

    assert result["passed"] is False
    assert result["gate_config_sha256"] == PROMOTION_GATE_V1_CONFIG_SHA256
    assert any("gate_config_sha256" in failure for failure in result["failures"])


def test_runner_writes_the_same_fail_closed_result(tmp_path: Path):
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "result.json"
    evidence = _passing_evidence()
    del evidence["strata"]["bb_joker0"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    result = run_promotion_gate(evidence_path, output_path)

    assert result["passed"] is False
    assert result["status"] == FAIL_STATUS
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
