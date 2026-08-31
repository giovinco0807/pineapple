import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "ai" / "config" / "promotion_gate_v1.json"


def _load_gate() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_promotion_gate_v1_is_scoped_and_pins_live_contracts():
    gate = _load_gate()

    assert gate["schema"] == "ofc_promotion_gate/v1"
    assert gate["gate_id"] == "promotion_gate_v1"
    assert gate["milestone"] == "M2"
    assert gate["scope"]["name"] == "reduced_t3_t4_public_tree"
    assert gate["scope"]["promotion_result"] == "m2_reduced_reference_ready"
    assert gate["scope"]["required_false_claims"] == {
        "hu_exact": False,
        "full_card_policy_promoted": False,
    }
    assert "serving_policy" in gate["scope"]["does_not_promote"]
    assert gate["decision_rule"]["full_card_root_disjoint_accuracy_gate"].startswith(
        "deferred_to_M3"
    )

    contracts = gate["contracts"]
    assert contracts["position_contract_version"] == "bb_first_v1"
    assert contracts["rules_version"] == "canonical_joker_bottom_middle_top_20260711"
    fl_ev = ROOT / contracts["fl_ev_path"]
    assert hashlib.sha256(fl_ev.read_bytes()).hexdigest() == contracts["fl_ev_sha256"]


def test_promotion_gate_v1_covers_order_roles_and_joker_strata_without_pooling():
    gate = _load_gate()
    sequence = gate["contracts"]["actor_sequence"]

    assert [step["phase"] for step in sequence] == [
        "t3_first",
        "t3_second",
        "t4_first",
        "t4_second",
        "terminal",
    ]
    assert [step["actor"] for step in sequence] == ["bb", "btn", "bb", "btn", None]
    assert [
        (step["bb_board_cards"], step["btn_board_cards"]) for step in sequence
    ] == [(9, 9), (11, 9), (11, 11), (13, 11), (13, 13)]

    strata = gate["strata"]
    assert set(strata["required"]) == {
        f"{actor}_joker{joker}"
        for actor in ("bb", "btn")
        for joker in (0, 1, 2)
    }
    assert strata["minimum_canonical_reduced_fixtures_per_stratum"] >= 1
    assert strata["no_pooled_pass"] is True


def test_promotion_gate_v1_thresholds_are_finite_consistent_and_fail_closed():
    gate = _load_gate()
    gates = gate["gates"]
    required_names = {
        "correctness",
        "information_leakage",
        "exploitability",
        "strategy_drift",
        "joker",
        "ordering",
        "runtime",
        "artifact_metadata",
    }
    assert set(gates) == required_names
    assert all(gates[name]["required"] is True for name in required_names)

    exploitability = gates["exploitability"]
    assert exploitability["nash_conv_score_lte"] == 2 * exploitability[
        "exploitability_score_lte"
    ]
    assert exploitability["max_unilateral_improvement_score_lte"] <= gate[
        "threshold_basis"
    ]["decision_epsilon_score"]

    drift = gates["strategy_drift"]
    assert 0 <= drift["reach_weighted_mean_total_variation_lte"] <= drift[
        "reach_weighted_p95_total_variation_lte"
    ] <= drift["max_total_variation_lte"] <= 1
    assert drift["independent_runs_min"] >= 3
    assert drift["pairwise_comparisons_min"] >= 3

    runtime = gates["runtime"]
    assert runtime["warmup_runs_min"] >= 1
    assert runtime["measured_runs_min"] >= 5
    assert runtime["reduced_bluff_20000_iterations_wall_ms_max_lte"] > gate[
        "evidence_baselines"
    ]["reduced_bluff_reference"]["observed_wall_ms_max"]

    def finite_numbers(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from finite_numbers(child)
        elif isinstance(value, list):
            for child in value:
                yield from finite_numbers(child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield value

    assert all(math.isfinite(value) for value in finite_numbers(gate))
    decision = gate["decision_rule"]
    assert decision["all_required_gates_must_pass"] is True
    assert decision["all_required_strata_must_pass"] is True
    assert decision["missing_or_non_finite_metric_is_failure"] is True
    assert decision["pooled_metric_cannot_override_stratum_failure"] is True


def test_promotion_gate_v1_artifacts_are_hashable_replayable_and_honestly_labeled():
    metadata = _load_gate()["gates"]["artifact_metadata"]
    required = metadata["required_fields"]
    sha_fields = metadata["sha256_fields"]

    assert len(required) == len(set(required))
    assert len(sha_fields) == len(set(sha_fields))
    assert set(sha_fields) <= set(required)
    assert {
        "position_contract_version",
        "rules_version",
        "fl_ev_sha256",
        "action_contract_sha256",
        "infoset_contract_sha256",
        "reduced_fixture_sha256",
        "chance_model_sha256",
        "range_model",
        "range_model_sha256",
        "range_particle_count",
        "range_effective_sample_size",
        "solver_config_sha256",
        "solver_code_sha256",
        "source_tree_manifest_sha256",
        "artifact_sha256",
        "information_model",
        "strategy_fusion",
        "equilibrium_approx",
        "reduced_tree_enumeration_exact",
        "hu_exact",
        "exploitability_method",
        "runtime_environment",
        "tests",
    } <= set(required)
    assert metadata["required_values"] == {
        "strategy_fusion": False,
        "equilibrium_approx": True,
        "reduced_tree_enumeration_exact": True,
        "hu_exact": False,
        "range_model": "exact_reduced_physical_particles_v1",
        "position_contract_version": "bb_first_v1",
        "rules_version": "canonical_joker_bottom_middle_top_20260711",
    }
    assert metadata["checkpoint_round_trip_metric_mismatches_eq"] == 0
    assert metadata["checkpoint_round_trip_strategy_hash_mismatches_eq"] == 0
    assert metadata["range_particle_count_min"] >= 1
    assert metadata["range_effective_sample_size_gt"] == 0
    assert metadata["range_effective_sample_size_must_not_exceed_particle_count"] is True
