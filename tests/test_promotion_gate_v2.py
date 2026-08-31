import copy
import hashlib
import json
from pathlib import Path

from ai.tutor.promotion_gate_v2 import (
    EVIDENCE_SCHEMA,
    PASS_STATUS,
    PROMOTION_GATE_V2_CONFIG_SHA256,
    canonical_sha256,
    derive_global_metrics,
    derive_stratum_metrics,
    finalize_artifact_hash,
    validate_promotion_evidence_v2,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "ai" / "config" / "promotion_gate_v2.json"


def _run(run_id: str) -> dict:
    strategy = {
        "infoset-a": {
            "key_json": '{"actor":"bb"}',
            "actor": "bb",
            "actions": {"a": float(0.25).hex(), "b": float(0.75).hex()},
        }
    }
    return {
        "run_id": run_id,
        "reverse_world_order": run_id == "r1",
        "rng_used": False,
        "fixture_manifest_sha256": "f" * 64,
        "strategy_sha256": canonical_sha256(strategy),
        "strategy": strategy,
        "infoset_reach": {"infoset-a": 1.0},
        "metrics": {
            "value_bb": 0.0,
            "bb_best_response": 0.1,
            "btn_best_response": -0.1,
            "nash_conv": 0.2,
            "exploitability": 0.1,
        },
        "exploitability_trace": [[1, 0.2], [200, 0.1]],
        "solve_wall_ms": 10.0,
        "solution_snapshot_round_trip_mismatches": 0,
    }


def _raw_stratum(actor: str, joker: int) -> dict:
    leaf = {
        "physical_state_commitment": "c" * 64,
        "action_key": "a",
        "contains_joker": bool(joker),
        "rust": {
            "score": 1.0,
            "raw_score": 1.0,
            "royalty": 0.0,
            "bust_rate": 0.0,
            "fl_rate": 0.0,
        },
        "python": {
            "score": 1.0,
            "raw_score": 1.0,
            "royalty": 0.0,
            "bust_rate": 0.0,
            "fl_rate": 0.0,
        },
    }
    return {
        "fixture_id": f"{actor}_joker{joker}_test",
        "fixture_manifest_sha256": "f" * 64,
        "actor": actor,
        "visible_joker_count": joker,
        "physical_hidden_world_count": 2,
        "root_infoset_digest": "d" * 64,
        "physical_state_commitments": ["c" * 64],
        "audit": {
            "legal_action_checks": 1,
            "legal_action_matches": 1,
            "candidate_actions_expected": 1,
            "candidate_actions_observed": 1,
            "transition_checks": 1,
            "transition_matches": 1,
            "chance_mass_errors": [0.0],
            "python_rust_leaf_metrics": [leaf],
            "physical_card_failures": 0,
            "preselection_failures": 0,
            "hidden_only_mutation_pairs": 1,
            "hidden_only_mutation_policy_tvs": [0.0],
            "forbidden_policy_input_failures": 0,
            "best_response_numerical_residual": 0.0,
            "x1_x2_identity_failures": 0,
            "visible_joker_stratum_match": True,
            "position_contract_mismatches": 0,
            "actor_sequence_mismatches": 0,
            "decision_board_shape_mismatches": 0,
            "public_history_order_mismatches": 0,
        },
        "runs": [_run("r0"), _run("r1"), _run("r2")],
    }


def _evidence() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw_strata = {
        name: _raw_stratum(name.split("_joker")[0], int(name[-1]))
        for name in config["strata"]["required"]
    }
    raw = {
        "strata": raw_strata,
        "pimc_counterexample_goldens_by_actor": {"bb": 1, "btn": 1},
        "solution_snapshot_round_trip_mismatches": 0,
        "runtime": {
            "calibration_measured_ms": [100.0] * 5,
            "recursive_warmup_ms_by_stratum": {
                name: [10.0] * 3 for name in raw_strata
            },
            "recursive_measured_ms_by_stratum": {
                name: [10.0] * 5 for name in raw_strata
            },
            "complete_gate_wall_s": 1.0,
            "peak_rss_mb": 100.0,
        },
    }
    manifests = {name: {"name": name} for name in config["provenance"]["required_hash_manifests"]}
    provenance = {
        "schema": "ofc_m2_provenance/v2",
        "hash_manifests": manifests,
        "rust_executable_sha256": "e" * 64,
    }
    artifact = {
        "schema": "ofc_m2_reduced_artifact/v2",
        "gate_id": config["gate_id"],
        "scope": config["scope"]["name"],
        "status": PASS_STATUS,
        "created_at_utc": "2026-07-13T00:00:00Z",
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
        "iterations": 200,
        "replay_ids": ["r0", "r1", "r2"],
        "runtime_environment": {"machine": "test"},
        "metrics": {"test": True},
        "tests": ["test"],
        "raw_measurements_sha256": canonical_sha256(raw),
    }
    field_by_manifest = {
        "action_contract": "action_contract_sha256",
        "infoset_contract": "infoset_contract_sha256",
        "reduced_fixtures": "reduced_fixture_sha256",
        "chance_model": "chance_model_sha256",
        "range_model": "range_model_sha256",
        "solver_config": "solver_config_sha256",
        "solver_code": "solver_code_sha256",
        "source_tree": "source_tree_manifest_sha256",
    }
    for manifest, field in field_by_manifest.items():
        artifact[field] = canonical_sha256(manifests[manifest])
    artifact = finalize_artifact_hash(artifact)
    strata = {
        name: {
            "actor": item["actor"],
            "visible_joker_count": item["visible_joker_count"],
            "canonical_reduced_fixture_count": 1,
            "metrics": derive_stratum_metrics(item),
        }
        for name, item in raw_strata.items()
    }
    return {
        "schema": EVIDENCE_SCHEMA,
        "gate_id": config["gate_id"],
        "gate_config_sha256": PROMOTION_GATE_V2_CONFIG_SHA256,
        "artifact": artifact,
        "strata": strata,
        "global_metrics": derive_global_metrics(raw, True),
        "raw_measurements": raw,
        "provenance": provenance,
    }


def test_v2_config_is_byte_locked():
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == PROMOTION_GATE_V2_CONFIG_SHA256


def test_content_bound_raw_derived_evidence_passes_only_m2_reduced_scope():
    result = validate_promotion_evidence_v2(_evidence())

    assert result["passed"] is True
    assert result["status"] == PASS_STATUS
    assert result["full_card_policy_promoted"] is False
    assert result["failures"] == []


def test_leaf_measurement_tamper_fails_derived_metrics_and_raw_hash_binding():
    evidence = _evidence()
    evidence["raw_measurements"]["strata"]["bb_joker0"]["audit"][
        "python_rust_leaf_metrics"
    ][0]["python"]["score"] = 9.0

    result = validate_promotion_evidence_v2(evidence)

    assert result["passed"] is False
    assert any("raw measurements hash mismatch" in item for item in result["failures"])
    assert any("not equal to raw-derived metrics" in item for item in result["failures"])


def test_manifest_tamper_and_status_mismatch_fail_closed():
    evidence = _evidence()
    evidence["provenance"]["hash_manifests"]["solver_code"]["tampered"] = True
    evidence["artifact"]["status"] = "full_card_ready"

    result = validate_promotion_evidence_v2(evidence)

    assert result["passed"] is False
    assert any("solver_code_sha256" in item for item in result["failures"])
    assert any("artifact.status" in item for item in result["failures"])
